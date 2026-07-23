"""The in-training capability probe panel (docs/training-recipe.md, "Monitoring:
the capability panel is a first-class training signal"; docs/evals.md, "Capability
gates").

Perplexity tells you the model is a fluent language model; it tells you nothing
about whether it can retrieve, and a capability can be destroyed mid-run with no
perplexity signature (docs/training-recipe.md, "Capability rehearsal": a
full-parameter heal took a copy probe from 0.25 to 0.06 while perplexity stayed
flat and the stage bought zero perplexity over the previous one). These probes
are cheap enough (seconds of compute) to run on a fixed cadence throughout every
healing stage and feed a collapsing result to early_stop.py as an abort signal,
not a curiosity.

Every probe in `PANEL` is deliberately FIXED-length (P1/P2 at n=2048, P3 at
n=4096), independent of whatever training/curriculum context the calling phase
happens to be using. That is a different concern from the length-curriculum's own
per-length dose-response tracking (docs/training-recipe.md, "Length curriculum"),
which logs induction accuracy at the curriculum's own lengths separately -- this
panel exists so that firings are directly comparable across an entire run
(including across phases), which is what early_stop.py's cross-firing pooling,
EMA, and slope rules need.

Interface: every probe is a function `probe(model, tokenizer, cfg, rng) ->
{"probe_name": value}` -- a single-key dict, eval-mode/no-grad, taking a shared
numpy Generator so a caller that reseeds `rng` identically every firing (see
heal.py: `PANEL_SEED`) gets the IDENTICAL synthetic items every time. That
matters: docs/evals.md notes gate probes are small-sample by design (16 trials
per cell, SE ~0.11 at p=0.25) and to "treat differences smaller than ~0.1 as
noise" -- resampling different items every firing would add sampling noise on
top of that, making it impossible to tell a real accuracy change from a new
random draw. `run_panel` runs every enabled probe (in `PANEL` order) and merges
their single-key results into one dict.

Leakage note: these probes intentionally use the SAME surface form as the eval
suite (evals.induction_probe / evals.make_needle_batch) -- monitoring the exact
thing the eval suite will later measure is the point. The leakage rule that
matters (docs/training-recipe.md, "Capability rehearsal": "rehearsal formats
must be measurably distinct from every evaluation probe's surface form") is
about fam_data.py's TRAINING-data generator, not this module; see fam_data.py's
header for the format table showing its planted pairs are multi-token,
delimited, and inline, in contrast to both this probe's single-token adjacent
bigram and the needle probe's disjoint-pool appended-tail query.
"""

import math
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from .attention import rope_tables
from .chunked_ce import chunked_cross_entropy
from .config import DATA

# Fixed seed re-issued fresh by the caller every firing (see heal.py) so every
# firing of the panel sees identical synthetic items; do not vary this per-run.
PANEL_SEED = 20260716


# --------------------------------------------------------- induction (P1-P3) ---
def _induction_example(rng, V, n, window, placement, reps=3):
    """Build one (ids, target) induction example: a random-token haystack with a
    random (key, val) bigram planted `reps` times, then the key alone at the
    final position (the model must argmax the completion to `val`).

    placement:
      'std'  -- reps spread across the first 90% of the sequence (evals.
                induction_probe's own construction; P1).
      'near' -- every rep confined to the last `window` tokens, i.e. inside the
                RoPE local-attention path only (P2).
      'far'  -- every rep strictly before the last-window region, i.e. reachable
                ONLY via routed block selection (P3).
    """
    ids = rng.integers(100, V, size=n).astype(np.int64)
    key = int(rng.integers(100, V))
    val = int(rng.integers(100, V))
    if placement == "std":
        positions = [int(s * (n - 4)) for s in np.linspace(0.05, 0.9, reps)]
    elif placement == "near":
        lo = max(0, n - window)
        span = max(1, window - 4)
        positions = [lo + int(s * span) for s in np.linspace(0.05, 0.9, reps)]
    elif placement == "far":
        hi = max(1, n - window)
        span = max(1, hi - 4)
        positions = [int(s * span) for s in np.linspace(0.05, 0.9, reps)]
    else:
        raise ValueError(f"unknown placement {placement!r}")
    for p in positions:
        ids[p] = key
        ids[p + 1] = val
    ids[n - 1] = key
    return ids, val


@torch.no_grad()
def _run_induction(model, cfg, rng, n, window, placement, trials, reps=3):
    dev = next(model.parameters()).device
    cos, sin = rope_tables(n, cfg.head_dim, cfg.rope_theta, dev)
    V = cfg.vocab_size
    hit = 0
    for _ in range(trials):
        ids, val = _induction_example(rng, V, n, window, placement, reps)
        x = torch.from_numpy(ids)[None].to(dev)
        logits = model(x, cos, sin, cfg.mem_elem_cap, mode="eval", tail=2)
        hit += int(logits[0, -1].argmax()) == val
    return hit / trials


def induction_std(model, tokenizer, cfg, rng, n=2048, trials=16):
    """P1: the standard copy/induction probe (evals.induction_probe's
    construction) -- a single-token (key, val) bigram planted 3x anywhere in the
    first 90% of a random-token haystack, queried at the last position."""
    return {"induction_std": _run_induction(model, cfg, rng, n, cfg.window, "std", trials)}


def induction_near(model, tokenizer, cfg, rng, n=2048, trials=16):
    """P2: same construction as P1, but every plant occurrence is confined to the
    last `cfg.window` tokens -- inside the RoPE local-attention window, so this
    exercises the local path only, never routed block selection."""
    return {"induction_near": _run_induction(model, cfg, rng, n, cfg.window, "near", trials)}


def induction_far(model, tokenizer, cfg, rng, n=4096, trials=16):
    """P3: same construction as P1, but every plant occurrence is strictly before
    the local window, so the answer is reachable ONLY through routed block
    selection.

    Must run at n=4096, not the base heal length (2048): candidate blocks at a
    given n is (n - window) // block (the block count that lies fully before the
    window boundary; cf. evals.hier_fidelity's identical `cand` computation). At
    n=2048 with window=1024, block=64 that is (2048-1024)//64 = 16 -- exactly
    cfg.k_blocks's default (16), so top-k selects EVERY candidate and the probe
    degenerates into "attention works", not "routing works" (docs/training-
    recipe.md, "One probe-design trap"). At n=4096 there are (4096-1024)//64 = 48
    candidates with only 16 routed, so selection is genuinely exercised. The
    assert below derives this from the model's actual config rather than
    hardcoding it, so a config change that breaks the inequality fails loudly
    instead of silently degenerating.
    """
    candidate_blocks = (n - cfg.window) // cfg.block
    assert candidate_blocks > cfg.k_blocks, (
        f"induction_far requires candidate_blocks (={candidate_blocks}, from "
        f"(n={n} - window={cfg.window}) // block={cfg.block}) to exceed k_blocks "
        f"(={cfg.k_blocks}) -- otherwise top-k selects every candidate block and the "
        "probe no longer exercises routing (see this function's docstring); raise n."
    )
    return {"induction_far": _run_induction(model, cfg, rng, n, cfg.window, "far", trials)}


# ----------------------------------------------------------------- P4: needle --
def needle_mini(model, tokenizer, cfg, rng, n=2048, trials=4):
    """P4: a `trials`-example miniature of the needle-in-text eval format
    (evals.make_needle_batch / evals.needle_accuracy) -- reuses the eval suite's
    own batch constructor (disjoint key/val vocab pools, occurrences planted in
    real held-out text, SEP-delimited appended query tail) at a small batch size
    instead of duplicating that logic. Registered but DISABLED by default (see
    `PANEL`): it needs data/needle_books.bin (docs/evals.md's needle haystack),
    which not every phase's data directory will have staged; flip the panel
    entry's `.enabled` to True once it is available for the run. Returns NaN
    (rather than raising) if the shard is missing, so an accidentally-enabled
    panel does not crash a heal run over a missing optional shard. early_stop.py
    drops NaN accs from its pooled estimates firing-by-firing (ES-2), AND -- if
    the shard stays missing for the WHOLE run, so every firing is NaN -- treats
    this as a truly-dead probe and excludes it from ES-1's per-probe
    still-rising check too (`early_stop._check_es1`'s cond_c), rather than
    letting an always-NaN probe permanently block ES-1 from ever firing. A probe
    with SOME real history but fewer than 3 non-NaN points in the current
    trailing window is not "dead" by this rule and still blocks the stop, as
    the conservative default.
    """
    from .evals import NEEDLE_PAIRS, make_needle_batch  # local: avoid pulling evals.py's

    # (dataset/benchmark) imports into every panel firing when P4 stays disabled.
    stream_path = os.path.join(DATA, "needle_books.bin")
    if not os.path.exists(stream_path):
        return {"needle_mini": float("nan")}
    stream = np.memmap(stream_path, dtype=np.uint32, mode="r")
    dev = next(model.parameters()).device
    cos, sin = rope_tables(n, cfg.head_dim, cfg.rope_theta, dev)
    seed = int(rng.integers(0, 2**31 - 1))
    ids, mask = make_needle_batch(cfg, n, trials, seed, stream)
    ids = ids.to(dev)
    tail = 2 * NEEDLE_PAIRS + 4
    with torch.no_grad():
        logits = model(ids, cos, sin, cfg.mem_elem_cap, mode="eval", tail=tail)
    pred = logits[:, :-1].argmax(-1)
    tgt = ids[:, n - tail + 1:]
    m = mask[:, n - tail + 1:].to(dev)
    correct = int(((pred == tgt) & m).sum())
    total = int(m.sum())
    return {"needle_mini": correct / max(1, total)}


# ------------------------------------------------------------- P5: val loss ----
def val_loss_fixed(model, tokenizer, cfg, rng, n_windows=8):
    """P5: full next-token CE on a FIXED held-out batch (the first `n_windows`
    non-overlapping windows of data/val_books.bin -- the same held-out split
    evals.perplexity uses, at the same capped context evals.perplexity uses:
    min(cfg.heal_ctx, 2048)) -- a smooth LM-loss signal, since train loss is too
    noisy to key a stopping rule on (docs/training-recipe.md, panel probe (d)).
    Deterministic and independent of `rng`/`n_windows` selection by design: the
    panel needs the SAME batch every firing so loss deltas reflect the model, not
    which windows got sampled (see early_stop.py's val-loss-EMA rule). Returns a
    scalar loss (not an accuracy) -- callers must not average this into a panel
    accuracy mean; early_stop.panel_mean excludes it by construction (it is
    reported through the `val_loss` key, not `accs`).

    Runs the `n_windows` windows ONE AT A TIME (not as one batch), each via
    `model(..., return_hidden=True)` + `hba.chunked_ce.chunked_cross_entropy`,
    rather than a single batched forward through the full lm_head. A single
    [n_windows, ctx] forward in eval mode would materialize fp32 logits
    [n_windows, ctx, V] -- at V~152k and the default n_windows=8/ctx=2048 that
    is ~10 GB in one spike, an OOM hazard on every firing (this panel runs every
    `probe_every` steps for the whole run). The per-window chunked path bounds
    the peak to one chunk's logits (see chunked_ce.py's module docstring),
    while returning the exact same scalar semantics: the mean CE over every
    position of every window (a simple average of the `nw` per-window means,
    since every window contributes the same position count and this corpus
    never emits ignore_index labels)."""
    path = os.path.join(DATA, "val_books.bin")
    if not os.path.exists(path):
        return {"val_loss_fixed": float("nan")}
    ctx = min(cfg.heal_ctx, 2048)
    data = np.memmap(path, dtype=np.uint32, mode="r")
    W = (len(data) - 1) // ctx
    nw = min(n_windows, W)
    if nw <= 0:
        return {"val_loss_fixed": float("nan")}
    dev = next(model.parameters()).device
    cos, sin = rope_tables(ctx, cfg.head_dim, cfg.rope_theta, dev)
    total_nll = 0.0
    total_count = 0
    with torch.no_grad():
        for i in range(nw):
            ids = torch.from_numpy(
                data[i * ctx: i * ctx + ctx + 1].astype(np.int64)
            )[None].to(dev)
            hidden = model(ids[:, :-1], cos, sin, cfg.mem_elem_cap, mode="eval",
                          return_hidden=True)
            tgt = ids[:, 1:]
            window_loss = chunked_cross_entropy(hidden, model.lm_head.weight, tgt,
                                                bias=model.lm_head.bias, chunk_size=1024)
            n_pos = tgt.numel()
            total_nll += float(window_loss) * n_pos
            total_count += n_pos
    return {"val_loss_fixed": total_nll / total_count}


# ------------------------------------------------------------------- registry --
@dataclass
class ProbeSpec:
    name: str
    fn: Callable
    trials: Optional[int]   # Bernoulli-trial count (None for loss-kind probes)
    kind: str                # "acc" | "loss"
    enabled: bool = True


PANEL = [
    ProbeSpec("induction_std", induction_std, trials=16, kind="acc", enabled=True),
    ProbeSpec("induction_near", induction_near, trials=16, kind="acc", enabled=True),
    ProbeSpec("induction_far", induction_far, trials=16, kind="acc", enabled=True),
    ProbeSpec("needle_mini", needle_mini, trials=4, kind="acc", enabled=False),
    ProbeSpec("val_loss_fixed", val_loss_fixed, trials=None, kind="loss", enabled=True),
]
PANEL_BY_NAME = {p.name: p for p in PANEL}


@torch.no_grad()
def run_panel(model, tokenizer, cfg, rng, which=None):
    """Run every enabled probe in `PANEL` (or, if `which` is given, exactly the
    named probes regardless of their `.enabled` flag) in registry order, in
    eval-mode/no-grad, and merge their single-key results into one dict. The
    model's `.training` flag is restored on return.

    Budget: this is meant to run every ~200 steps mid-heal at negligible cost
    next to what it protects (docs/training-recipe.md, "Monitoring"). With the
    default enabled set (P1/P2 at n=2048 x16 trials, P3 at n=4096 x16 trials, P5
    an 8-window forward), the marginal cost is P3's n=4096 forward; P4 is
    disabled by default and adds nothing unless enabled.
    """
    was_training = model.training
    model.eval()
    out = {}
    try:
        for spec in PANEL:
            if which is not None:
                if spec.name not in which:
                    continue
            elif not spec.enabled:
                continue
            out.update(spec.fn(model, tokenizer, cfg, rng))
    finally:
        if was_training:
            model.train()
    return out
