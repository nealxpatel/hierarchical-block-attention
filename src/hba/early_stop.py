"""Pre-registered early-stopping rules for a healing run's capability panel
(docs/training-recipe.md, "Monitoring: the capability panel is a first-class
training signal").

This module is deliberately TORCH-FREE (numpy/math only) so the decision logic
can run anywhere and replay a completed run's `probe_log.jsonl` offline, with no
GPU and no model (see replay_probes.py) -- the rules operate purely on the
machine-readable panel log that probes.run_panel produces, one firing at a time.

Why pre-registered, pooled rules instead of a single-firing threshold: each
probe's per-firing accuracy is an average of 16 Bernoulli trials. At p=0.25 that
is SE = sqrt(0.25*0.75/16) ~= 0.11 -- docs/evals.md notes exactly this ("Gate
probes are small-sample by design ... treat differences smaller than ~0.1 as
noise"). An SE of that size is comparable to the effect sizes a stopping rule
cares about, so no rule here keys off a single firing: ES-1 requires a trailing
WINDOW of firings to agree, and ES-2 pools trials ACROSS a trailing window of
firings into one binomial estimate before testing it.

Firing schema (what heal.py appends to probe_log.jsonl, one JSON object per
line, and what every function below consumes as a list of these):

    {
      "step":     int,             training step this firing ran at
      "tokens":   float,           training tokens consumed by this step
      "accs":     {probe_name: float},   accuracy-kind probe results (NaN dropped)
      "n_trials": {probe_name: int},     Bernoulli trial count backing each acc
      "val_loss": float,           probes.val_loss_fixed's result
    }

Judgment calls made explicit (no spec pinned these numbers precisely):
  - EMA decay `ema_alpha=0.5`: a half-life of one firing, so the smoothed val-loss
    signal is dominated by the same trailing few firings the rules themselves
    inspect, while still damping single-firing noise. Tune via replay_probes.py
    against a real log if a run's noise profile warrants a different constant.
  - ES-1(b)'s "1 SE" is the pooled-Bernoulli SE (sqrt(p(1-p)/N), p and N pooled
    across the panel's acc-probes AT THE CURRENT FIRING) -- the same SE quantity
    docs/evals.md's ~0.11-at-p=0.25 note is about, applied to the panel mean
    rather than one probe.
  - ES-1(c)'s slope test needs >=3 points to fit a residual-based SE; with fewer
    than 3 firings containing a probe, that probe's criterion is treated as
    UNMET (conservative: insufficient evidence that it has stopped rising blocks
    a stop, it does not enable one).
  - ES-2's Wilson bound uses the standard two-sided 95% z (1.959963985984...).
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

Z95 = 1.959963984540054  # two-sided 95% normal quantile

_TOMBSTONE_NAME = "ES2_TOMBSTONE.json"


# ------------------------------------------------------------- pure building blocks
def wilson_ucb(hits, n, z=Z95):
    """Upper bound of the Wilson score confidence interval for a binomial
    proportion (hits successes out of n trials). More reliable than a normal
    (Wald) interval at small n / extreme p, which is exactly the regime a
    16-trial probe (or even a pooled 48-trial one) lives in. n=0 -> 1.0
    (maximally uninformative upper bound; a probe engine never has n=0 for a
    running_max>=0.25 probe by construction, but this keeps the function total).
    """
    if n <= 0:
        return 1.0
    p = hits / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center + margin) / denom


def ema_series(values, alpha=0.5):
    """Exponential moving average over `values` (in order), returned as a list
    the same length as `values`. `alpha` is the weight on the newest point.

    NaN-guarded: a NaN entry (e.g. probes.val_loss_fixed returning NaN because
    data/val_books.bin is missing) does NOT update the running EMA state --
    the update is skipped and the PREVIOUS ema value is carried forward into
    that slot instead (or NaN, if no real value has been seen yet). Without
    this, a single NaN input would poison `m` forever afterward (NaN propagates
    through arithmetic unconditionally), silently disabling ES-1's cond_a for
    the rest of the run."""
    out = []
    m = None
    for v in values:
        if isinstance(v, float) and math.isnan(v):
            out.append(m if m is not None else float("nan"))
            continue
        m = v if m is None else alpha * v + (1 - alpha) * m
        out.append(m)
    return out


def lstsq_slope(xs, ys):
    """Ordinary-least-squares slope and its standard error for y ~ a + b*x.
    Returns (slope, se_slope). With < 3 points, or degenerate x (all equal), the
    SE cannot be estimated from residuals; returns (slope_or_0, inf) so callers
    that compare `slope - z*se > 0` correctly treat it as "not significant"."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0, float("inf")
    xbar, ybar = x.mean(), y.mean()
    sxx = float(((x - xbar) ** 2).sum())
    if sxx == 0.0:
        return 0.0, float("inf")
    sxy = float(((x - xbar) * (y - ybar)).sum())
    slope = sxy / sxx
    if n < 3:
        return slope, float("inf")
    resid = y - (ybar + slope * (x - xbar))
    s2 = float((resid ** 2).sum()) / (n - 2)
    se = math.sqrt(max(0.0, s2) / sxx)
    return slope, se


def panel_mean(firing):
    """Mean of a firing's accuracy-kind probe results (NaN entries -- e.g. a
    disabled-shard P4 -- excluded). None if the firing has no usable acc probes
    at all (distinct from 0.0, a genuinely measured floor)."""
    vals = [v for v in firing.get("accs", {}).values() if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    return float(np.mean(vals))


def es_floor_tokens(warmup_end_tokens, phase_budget_tokens, min_post_warmup_tokens=100e6):
    """ES-FLOOR: no plateau stop (ES-1) before max(100M tokens after LR warmup
    end, 40% of the phase token budget). Both quantities are absolute tokens-
    into-the-phase (tokens=0 at phase start), so the floor is well-defined
    regardless of how long warmup itself took: max(warmup_end_tokens +
    min_post_warmup_tokens, 0.4 * phase_budget_tokens). The forgetting abort
    (ES-2) is exempt -- see EarlyStopEngine.evaluate.

    Rationale (docs/training-recipe.md's stage table is cosine-decayed): runs
    routinely look flat mid-schedule and improve into the decay tail, so a stop
    signal that fires during warmup, or even early in the main schedule, is
    presumed noise, not signal, until this floor is cleared.
    """
    return max(warmup_end_tokens + min_post_warmup_tokens, 0.4 * phase_budget_tokens)


def es3_ceiling(candidate_stop_tokens, phase_budget_tokens):
    """ES-3: the pre-registered phase token budget is a CEILING the other rules
    may only shorten, never extend. Clamp any proposed stop point down to the
    budget. (heal.py's own step-count loop already never trains past the budget
    by construction; this helper exists for offline/analysis call sites -- e.g.
    a replay tool computing "when would we have stopped" -- that must respect
    the same invariant.)"""
    return min(candidate_stop_tokens, phase_budget_tokens)


def _pooled_trailing(history, probe, k):
    """Pooled (hits, n, firings_used) for `probe` over the last `k` firings of
    `history` (searched newest-first) that contain a non-NaN value for it."""
    hits = n = used = 0
    for f in reversed(history):
        accs = f.get("accs", {})
        nts = f.get("n_trials", {})
        if probe not in accs or probe not in nts:
            continue
        a = accs[probe]
        if isinstance(a, float) and math.isnan(a):
            continue
        nt = nts[probe]
        hits += a * nt
        n += nt
        used += 1
        if used == k:
            break
    return hits, n, used


def _running_max(history, probe):
    vals = [
        f["accs"][probe]
        for f in history
        if probe in f.get("accs", {}) and not (isinstance(f["accs"][probe], float) and math.isnan(f["accs"][probe]))
    ]
    return max(vals) if vals else 0.0


# ---------------------------------------------------------------------- engine --
@dataclass
class Verdict:
    rule_fired: Optional[str]   # None | "ES-1" | "ES-2"
    details: dict = field(default_factory=dict)


@dataclass
class EarlyStopEngine:
    """Pure decision engine over a probe-log history. `evaluate(history)` takes
    the FULL firing history up to and including the current (latest) firing, in
    ascending step order, and returns a `Verdict`. Stateless across calls other
    than the constructor's fixed thresholds -- callers (heal.py, replay_probes.py)
    own the history list.
    """
    phase_budget_tokens: float
    warmup_end_tokens: float
    min_post_warmup_tokens: float = 100e6
    ema_alpha: float = 0.5
    val_loss_rel_eps: float = 0.002       # ES-1(a): "< 0.2% relative"
    slope_z: float = 2.0                  # ES-1(c): "within 2 SE"
    forming_floor: float = 0.25           # ES-2 eligibility: running_max >= this
    collapse_frac: float = 0.5            # ES-2 threshold: 0.5 * running_max
    collapse_abs_floor: float = 0.15      # ES-2 threshold floor
    trailing: int = 3                     # ES-1(a)/(b) and ES-2's pooling window
    slope_window: int = 5                 # ES-1(c)'s trailing window (uses fewer if
                                           # fewer are available; needs >= 3 to fire)
    # Loud-once flag for the NaN-val_loss warning in _check_es1 -- not part of
    # the engine's public constructor surface (init=False), not compared/repr'd
    # (this is process-local warning state, not decision-relevant config).
    _warned_nan_val_loss: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self):
        self.floor_tokens = es_floor_tokens(
            self.warmup_end_tokens, self.phase_budget_tokens, self.min_post_warmup_tokens
        )

    def evaluate(self, history):
        if not history:
            return Verdict(None, {"reason": "no history"})
        # ES-2 (safety) is checked first and is EXEMPT from ES-FLOOR: the floor
        # exists to keep plateau stops from firing on warmup/mid-schedule noise,
        # but a forgetting collapse is a rollback trigger, not a budget cut, and
        # ES-2 carries its own noise guards (the capability must have formed to
        # running_max >= forming_floor, and the pooled Wilson bound must clear).
        # A collapse caught at 20% of budget saves the doomed remainder; waiting
        # for the floor would burn it.
        v2 = self._check_es2(history)
        if v2.rule_fired:
            return v2
        cur = history[-1]
        if cur["tokens"] < self.floor_tokens:
            return Verdict(None, {"reason": "before ES-FLOOR", "floor_tokens": self.floor_tokens,
                                  "tokens": cur["tokens"]})
        return self._check_es1(history)

    # ---- ES-2: forgetting abort ------------------------------------------------
    def _check_es2(self, history):
        """Fires when a probe that DID form (running_max >= forming_floor -- "a
        probe that never formed cannot collapse") has its pooled trailing-3-
        firings trials' 95% Wilson upper bound fall below
        max(0.5 * running_max, collapse_abs_floor). Checks probes in a fixed
        (sorted) order and returns the first that fires; if several fire in the
        same firing that just reflects a single shared-cause collapse (e.g. an LR
        spike), not several independent ones."""
        probes = sorted({name for f in history for name in f.get("accs", {})})
        for name in probes:
            rmax = _running_max(history, name)
            if rmax < self.forming_floor:
                continue
            hits, n, used = _pooled_trailing(history, name, self.trailing)
            if used < self.trailing:
                continue  # not enough firings yet to trust the pooled estimate
            ucb = wilson_ucb(hits, n)
            threshold = max(self.collapse_frac * rmax, self.collapse_abs_floor)
            if ucb < threshold:
                cur = history[-1]
                return Verdict("ES-2", {
                    "probe": name, "running_max": rmax, "pooled_hits": hits, "pooled_n": n,
                    "wilson_ucb": ucb, "threshold": threshold,
                    "step": cur["step"], "tokens": cur["tokens"],
                })
        return Verdict(None, {"reason": "no probe crossed the ES-2 collapse bound"})

    # ---- ES-1: plateau stop -----------------------------------------------------
    def _check_es1(self, history):
        """Fires when, over the current firing and its trailing window, ALL of:
        (a) the val-loss EMA has improved < val_loss_rel_eps relative over the
            last `trailing` firings;
        (b) the panel mean is within one pooled-Bernoulli SE of its running max;
        (c) for every acc-probe, the OLS slope over its trailing `slope_window`
            (or fewer, down to a minimum of 3) firings is not significantly
            positive (slope - slope_z*SE <= 0) -- "no stop while anything is
            still rising", checked probe by probe so one still-forming capability
            blocks the stop even if the panel MEAN has plateaued.
        Needs at least `trailing + 1` firings of history (to compare "now" against
        "trailing firings ago"); returns None with a reason otherwise.
        """
        if len(history) < self.trailing + 1:
            return Verdict(None, {"reason": f"insufficient history for ES-1 (<{self.trailing + 1} firings)"})
        cur = history[-1]

        # (a) val-loss EMA relative improvement. ema_series is itself NaN-
        # guarded (a NaN input skips the running-state update rather than
        # poisoning it forever), but the DECISION here must not treat "no
        # val_loss signal at all" as "flat val loss" -- a NaN at the current
        # firing (or an EMA that has never seen a real value) makes cond_a
        # FAIL outright, with a loud once-per-engine warning, rather than
        # silently satisfying it via a 0/0-shaped comparison.
        vl = [f["val_loss"] for f in history]
        ema = ema_series(vl, self.ema_alpha)
        cur_val_loss = vl[-1]
        ema_now, ema_back = ema[-1], ema[-1 - self.trailing]
        nan_signal = (
            (isinstance(cur_val_loss, float) and math.isnan(cur_val_loss))
            or (isinstance(ema_now, float) and math.isnan(ema_now))
            or (isinstance(ema_back, float) and math.isnan(ema_back))
        )
        if nan_signal:
            if not self._warned_nan_val_loss:
                print(f"[early_stop] WARNING: val_loss is NaN at step {cur['step']} "
                      "(or no real val_loss has been observed yet) -- ES-1's cond_a "
                      "(flat val loss) is being treated as FAILED, not silently "
                      "satisfied, until real val_loss data is available", flush=True)
                self._warned_nan_val_loss = True
            cond_a = False
            rel_impr = float("nan")
        else:
            rel_impr = (ema_back - ema_now) / abs(ema_back) if ema_back != 0.0 else 0.0
            cond_a = rel_impr < self.val_loss_rel_eps

        # (b) panel mean within 1 pooled-Bernoulli SE of its running max
        means = [panel_mean(f) for f in history]
        if any(m is None for m in means):
            return Verdict(None, {"reason": "a firing in history has no usable acc probes"})
        running_max_mean = max(means)   # max panel_mean over the FULL history, not just trailing
        cur_mean = means[-1]
        hits = n = 0
        for name, a in cur.get("accs", {}).items():
            if isinstance(a, float) and math.isnan(a):
                continue
            hits += a * cur["n_trials"][name]
            n += cur["n_trials"][name]
        p_pool = hits / n if n else 0.0
        se = math.sqrt(p_pool * (1 - p_pool) / n) if n else float("inf")
        cond_b = (running_max_mean - cur_mean) <= se

        # (c) per-probe slope, trailing min(slope_window, len(history)) firings
        window = history[-min(self.slope_window, len(history)):]
        cond_c = True
        slopes = {}
        for name in cur.get("accs", {}):
            xs, ys = [], []
            for f in window:
                a = f.get("accs", {}).get(name)
                if a is not None and not (isinstance(a, float) and math.isnan(a)):
                    xs.append(f["step"])
                    ys.append(a)
            if len(xs) < 3:
                # Fewer than 3 non-NaN points in the trailing window is
                # normally treated as "not enough evidence this probe has
                # stopped rising" -> blocks the stop (conservative). EXCEPTION:
                # a probe whose ENTIRE history (not just the trailing window)
                # is NaN -- e.g. an enabled probe whose data shard is missing
                # (probes.needle_mini's docstring) -- can never accumulate 3
                # points and would otherwise block ES-1 FOREVER, silently. Such
                # a truly-dead probe carries no information and is skipped
                # instead of vetoing.
                has_any_real_history = any(
                    not (isinstance(f["accs"][name], float) and math.isnan(f["accs"][name]))
                    for f in history if name in f.get("accs", {})
                )
                slopes[name] = None
                if has_any_real_history:
                    cond_c = False
                continue
            slope, se_slope = lstsq_slope(xs, ys)
            slopes[name] = {"slope": slope, "se": se_slope, "n": len(xs)}
            if slope - self.slope_z * se_slope > 0:
                cond_c = False

        fired = cond_a and cond_b and cond_c
        details = {
            "step": cur["step"], "tokens": cur["tokens"],
            "val_loss_rel_improvement": rel_impr, "cond_a_flat_val_loss": cond_a,
            "panel_mean": cur_mean, "running_max_mean": running_max_mean,
            "panel_mean_se": se, "cond_b_panel_plateau": cond_b,
            "slopes": slopes, "cond_c_no_probe_rising": cond_c,
        }
        return Verdict("ES-1" if fired else None, details)


# ------------------------------------------------------------ log I/O helpers --
def load_probe_log(path, max_step=None):
    """Load a probe_log*.jsonl (heal.py's per-firing panel log) into a list of
    firing dicts, in ASCENDING step order, deduped by `step` (keeping the LAST
    occurrence of each step). `max_step`: keep only firings with step <=
    max_step (heal.py uses this on --resume to discard any firings that raced
    ahead of the checkpoint actually resumed from).

    Dedup rationale: a rollback->resume cycle (ES-2) or a killed-mid-append run
    can leave duplicate lines for the same step in the file (e.g. heal.py
    re-firing the panel at a step it already logged before a crash). Pooling
    both copies would double-weight that step's trials in ES-1/ES-2's pooled
    estimates; keeping only the last (most recent) write for a given step is
    the correct de-dup direction since it is what actually happened last on
    disk. A torn (unparseable) LAST line is tolerated -- see the try/except
    below; a torn line anywhere else in the file still raises, since that
    indicates real corruption rather than a kill mid-append."""
    hist = []
    if not os.path.exists(path):
        return hist
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    by_step = {}
    order = []
    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                # torn trailing line (process killed mid-append) -- skip it
                continue
            raise
        if max_step is not None and rec["step"] > max_step:
            continue
        if rec["step"] not in by_step:
            order.append(rec["step"])
        by_step[rec["step"]] = rec   # last occurrence wins
    for step in order:
        hist.append(by_step[step])
    return hist


def truncate_probe_log(path, max_step):
    """Load `path` filtered to firings with step <= max_step (deduped by step,
    last occurrence wins -- see load_probe_log's docstring), then ATOMICALLY
    rewrite the file (tmp + os.replace) to contain EXACTLY that filtered/
    deduped history. Returns the filtered history list (so a caller doesn't
    have to load_probe_log again after calling this).

    Used by heal.py on --resume. load_probe_log alone only filters IN MEMORY --
    stale lines past `max_step` (e.g. future-step firings a rolled-back ES-2
    collapse wrote before an operator deliberately cleared the tombstone), or
    duplicate lines for an already-logged step, would otherwise remain on disk
    untouched. Left there, a LATER resume's load_probe_log call would read
    them right back in and pool them into ES-2's trailing window, which can
    spuriously re-fire a forgetting-abort on an otherwise-healthy continuation
    (or double-weight a step in ES-1's estimates, via the duplicate-line
    case). Truncating the file on disk, not just filtering in memory, is what
    actually prevents that. No-op if `path` does not exist (returns [])."""
    history = load_probe_log(path, max_step=max_step)
    if os.path.exists(path):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for rec in history:
                f.write(json.dumps(rec) + "\n")
        os.replace(tmp, path)
    return history


def append_probe_log(path, firing):
    with open(path, "a") as f:
        f.write(json.dumps(firing) + "\n")


# ---------------------------------------------------------------- tombstone ----
def tombstone_path(results_dir):
    return os.path.join(results_dir, _TOMBSTONE_NAME)


def write_tombstone(results_dir, probe, step, tokens, rolled_back_to, phase=None):
    """Write the ES-2 tombstone. heal.py refuses to start or resume ANY phase
    while this file exists (docs/training-recipe.md: "Treat a collapsing probe
    as an abort signal ... halt, roll back ... diagnose before continuing") --
    the training data stream is deterministic, so an unattended restart would
    replay the identical collapse (a rollback -> collapse loop); the tombstone
    forces a human diagnosis step before anyone deletes it."""
    obj = {"probe": probe, "step": step, "tokens": tokens, "rolled_back_to": rolled_back_to}
    if phase is not None:
        obj["phase"] = phase
    tmp = tombstone_path(results_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, tombstone_path(results_dir))


def check_tombstone(results_dir):
    """Returns the tombstone dict if results/ES2_TOMBSTONE.json exists, else
    None. Callers (heal.py) should refuse to start/resume when this is not
    None."""
    p = tombstone_path(results_dir)
    if os.path.exists(p):
        return json.load(open(p))
    return None
