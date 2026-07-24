"""Capability-rehearsal data mixer (docs/training-recipe.md, "Capability rehearsal").

WHY THIS EXISTS
----------------
Full-parameter healing on plain web/books/code text catastrophically forgets the
donor's induction/copying circuit while perplexity stays flat: nothing in ordinary
text rewards content-agnostic copying, so gradient descent trades the circuit away
for a loss improvement too small to see (docs/training-recipe.md, "The failure
this prevents"). The fix woven into stage 2 (and, at longer distances, stage 3) is
a lower LR plus this rehearsal mix: a small fraction of tokens are overwritten
with planted copy pairs, so the induction/copy capability is *rehearsed* during
full-parameter healing instead of being trained away.

WHAT IT DOES
------------
`FamMixer` wraps a token-window stream (see heal.WindowStream) and is a drop-in
for it: same `.batch(m)` contract, same deterministic-given-seed behaviour, same
(B, ctx+1) int64 tensor out. The underlying real-text distribution is UNCHANGED
except that a deterministic subset of positions per window is overwritten with
planted (key -> value) copy pairs. Each pair is planted `reps` times; predicting
the value tokens at the 2nd..reps-th occurrence is only possible by content-based
lookback to an earlier occurrence (the key/value tokens are random, so there is no
local shortcut) -- i.e. exactly the induction/copy circuit that plain full-
parameter healing destroys. The plain next-token CE already in heal.py does the
rest; no extra loss term is needed. Mixing is logged (fam_tok vs real_tok) and
reproducible.

DESIGN CONSTRAINTS SATISFIED
-----------------------------
(a) Format is varied AGGRESSIVELY per pair: key length 1-3 tokens, value length
    1-3 tokens, separator length 0-2 tokens, repetitions 2-4, and 1-8 distinct
    pairs per window. Distances between consecutive occurrences span BOTH inside
    the RoPE window ("near" mode, all occurrences confined to a random 1024-token
    span) AND far beyond it up to the full training context ("far" mode,
    occurrences spread across the whole window) -- so the capability must
    generalize across distances rather than memorize one template (docs/training-
    recipe.md, "Format variation is required" / "Distance coverage").
(b) Deterministic: every planting decision is a pure function of (base_seed, g),
    where g is the GLOBAL window index (see dist_util.global_window_index) --
    NOT (m, b) separately, and not rank/world. No running state feeds the RNG,
    so a resume at any step reproduces identical batches, AND a given window
    plants identically regardless of which rank owns it or how many ranks the
    world has (the multi-GPU shard-partition gate checks exactly this: same g
    -> same plant on every rank). This IS a stream-identity change from a
    single-process (m, b)-keyed RNG (see heal._sig's `stream_version`) -- it
    has to be, since a per-rank key cannot be made world-size-invariant.
(c) Reproducible + logged: FamMixer tracks fam_tok / real_tok / windows / fam_windows.

LEAKAGE AVOIDANCE vs the eval probes (evals.induction_probe, evals.make_needle_batch).
The generator's format is *measurably* different from BOTH probes so training
cannot memorize the eval template (docs/training-recipe.md, "Leakage rule"):

  eval induction_probe                     |  FamMixer
  -----------------------------------------+------------------------------------------------------
  key & value are SINGLE tokens (1-token   |  keys AND values are MULTI-token spans (1-3 tokens
    bigram key,val)                        |    each), lengths randomized per pair
  key immediately adjacent to value, NO    |  an explicit separator span (0-2 tokens, drawn from a
    separator                              |    fixed delimiter pool that excludes the needle
                                           |    probe's SEP=2)
  haystack is PURE RANDOM tokens           |  background is REAL text (a token-window stream);
                                           |    pairs are planted INTO real text
  query is the key placed at the LAST       |  there is NO appended/tail query -- the "query" is
    position (n-1); answer read at seq end |    simply the in-context 2nd..N-th occurrence of the
                                           |    pair, mid-sequence; the answer position is wherever
                                           |    that occurrence lands (never pinned to n-1)
  fixed reps=3 at linspace(0.05,0.9,3)     |  reps 2-4 at randomized, distance-varied positions
  exactly ONE pair per sequence            |  1-8 distinct pairs per window

  eval make_needle_batch                   |  FamMixer
  -----------------------------------------+------------------------------------------------------
  keys/vals drawn from DISJOINT vocab      |  keys/vals drawn from the SAME range [100, V); no
    pools (lower half keys, upper half     |    disjoint-pool structure
    vals)                                  |
  pairs planted in body, then queried in a |  no separate query region; second occurrences are
    STRUCTURED appended tail [SEP,k,v,...] |    inline, and SEP=2 is deliberately NOT used
    with SEP=2                             |
  each pair queried exactly ONCE           |  each pair recurs 2-4x, all inline

So the trained capability is "look back to a prior occurrence of an arbitrary
multi-token key and copy its multi-token value across an arbitrary distance",
which the eval probes measure but whose exact surface form the generator never
reproduces.
"""

import numpy as np
import torch

from . import dist_util

# separator token pool: a few fixed, valid (< vocab) ids used BETWEEN key and value.
# Chosen to EXCLUDE the needle probe's SEP=2 (leakage avoidance) and to be
# structurally distinct from the induction probe (which uses no separator at all).
_DELIMS = np.array([11, 13, 25, 220, 271, 353], dtype=np.int64)


class FamMixer:
    """Drop-in wrapper over a token-window stream: mixes synthetic copy/induction
    pairs into the real-text stream at ~`frac` of tokens. Deterministic given the
    underlying stream's seed + `seed`."""

    # per-window fam-token target fraction is drawn U(lo, hi); mean = frac. The
    # 1-8 pairs cap and the length/rep variety mean high-target windows saturate at
    # 8 pairs, so the realized mean is a touch under (lo+hi)/2 -- measured & logged
    # at smoke time.
    def __init__(self, stream, cfg, seed, frac=0.03):
        self.stream = stream
        self.cfg = cfg
        self.seed = int(seed)
        self.frac = float(frac)
        self.B = stream.B
        # per-window target frac ~ U(_lo, _hi). The 8-pair cap truncates high-target
        # windows, so the intended mean is set ~1.25x `frac` to make the REALIZED
        # ratio land near `frac` (measured: this yields ~3.0% for frac=0.03). Wide
        # band -> 1-8 pairs/window variety.
        self._lo = 0.25 * frac
        self._hi = 2.0 * (1.3 * frac) - self._lo
        # counters (logged; not fed back into the RNG -> resume-safe determinism)
        self.fam_tok = 0
        self.real_tok = 0
        self.windows = 0
        self.fam_windows = 0
        self.pairs_planted = 0

    def _rng(self, g):
        """Pure function of the GLOBAL window index g alone (see module
        docstring, point (b)) -- no rank, no world, no local (m, b) split. This
        is what makes a plant world-size-invariant: whichever rank happens to
        own window g in a given (world, micro_B, accum) decomposition, it
        derives the identical RNG stream and therefore the identical plant."""
        s = (self.seed * 2654435761 + g * 1000003 + 12345) & 0xFFFFFFFF
        return np.random.default_rng(s)

    def _place_pair(self, row, occupied, rng, L, V):
        """Plant one (key -> value) pair `reps` times into `row` (in place). Returns
        tokens planted (0 if fewer than 2 occurrences could be placed -> not an
        induction pair)."""
        klen = int(rng.integers(1, 4))     # 1..3
        vlen = int(rng.integers(1, 4))     # 1..3
        seplen = int(rng.integers(0, 3))   # 0..2
        reps = int(rng.integers(2, 5))     # 2..4
        key = rng.integers(100, V, size=klen)
        val = rng.integers(100, V, size=vlen)
        if seplen > 0:
            unit = np.concatenate([key, rng.choice(_DELIMS, size=seplen), val])
        else:
            unit = np.concatenate([key, val])
        ulen = len(unit)
        # distance mode: 'near' confines all occurrences to a random 1024-span
        # (in-window distances); 'far' spreads them across the whole window
        # (beyond-window distances).
        if rng.random() < 0.4:
            lo = int(rng.integers(0, max(1, L - 1024)))
            hi = min(L, lo + 1024)
        else:
            lo, hi = 0, L
        placed = []
        tries = 0
        hi_start = max(lo + 1, hi - ulen)
        while len(placed) < reps and tries < 40:
            tries += 1
            st = int(rng.integers(lo, hi_start))
            if st + ulen > L:
                continue
            if occupied[st:st + ulen].any():
                continue
            placed.append(st)
        if len(placed) < 2:
            return 0
        for st in placed:
            row[st:st + ulen] = unit
            occupied[st:st + ulen] = True
        return ulen * len(placed)

    def _plant(self, row, rng):
        L = len(row)
        V = self.cfg.vocab_size
        target = int(rng.uniform(self._lo, self._hi) * L)
        # Cap pairs/window PROPORTIONALLY to window length so the realized
        # rehearsal DENSITY is uniform across curriculum lengths. `target` already
        # scales with L (frac*L), but a FIXED cap makes density fall ~1/L: at the
        # 4096 base length 8 pairs ~= `frac`, but at 32K the same 8 pairs is only
        # ~frac/8 (measured: 2.5% at 4K -> 0.36% at 32K). Scaling the cap (and the
        # placement-attempt budget) by L/4096 keeps every length near `frac`.
        # <= 4096 is unchanged (max(8,...)), so stage-1/2 behavior is identical.
        max_pairs = max(8, round(8 * L / 4096))
        max_attempts = max(64, round(64 * L / 4096))
        occupied = np.zeros(L, dtype=bool)
        planted = 0
        npairs = 0
        attempts = 0
        while planted < target and npairs < max_pairs and attempts < max_attempts:
            attempts += 1
            got = self._place_pair(row, occupied, rng, L, V)
            if got > 0:
                planted += got
                npairs += 1
        return planted, npairs

    def batch(self, m, rank=0, world=1):
        # world-size-invariant global window index -- calls
        # dist_util.global_window_index directly (step=0, same call pattern as
        # heal.WindowStream.batch) rather than re-deriving the g formula here a
        # second time. At rank=0, world=1 this is g = m*B + b, bit-identical to
        # the single-rank formula.
        ids = self.stream.batch(m, rank=rank, world=world)   # torch (B, ctx+1) int64
        arr = ids.numpy().copy()
        for b in range(self.B):
            g = dist_util.global_window_index(step=0, micro=m, b=b, rank=rank,
                                              world=world, micro_B=self.B, accum=1)
            rng = self._rng(g)
            planted, npairs = self._plant(arr[b], rng)
            self.fam_tok += planted
            self.real_tok += arr.shape[1] - planted
            self.pairs_planted += npairs
            self.windows += 1
            if planted > 0:
                self.fam_windows += 1
        return torch.from_numpy(arr)

    def ratio(self):
        tot = self.fam_tok + self.real_tok
        return (self.fam_tok / tot) if tot else 0.0

    def stats(self):
        return dict(fam_tok=self.fam_tok, real_tok=self.real_tok, ratio=self.ratio(),
                    windows=self.windows, fam_windows=self.fam_windows,
                    pairs_planted=self.pairs_planted)
