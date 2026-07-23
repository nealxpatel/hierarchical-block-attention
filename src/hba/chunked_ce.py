"""Chunked cross-entropy with recompute-in-backward.

WHY THIS EXISTS
----------------
The training micro-batch ceiling is not attention -- it is the unchunked fp32 CE
cast materializing the full `[B*n, V]` logits tensor. With vocab V ~ 150k, at
micro-batch 4 x n=4096 that is `4*4096 x 150_000 x 4 B ~ 10 GB` in one spike, on
top of everything else already live at that point in the forward.

THE TRAP: chunking alone does not fix the peak. A naive fix -- loop over
~1024-position slices, run lm_head + CE per slice, sum the losses -- looks like
it bounds memory to one chunk, but it does not: ordinary autograd retains every
chunk's fp32 logits tensor because each one is needed for ITS OWN backward pass,
and nothing frees a chunk's logits until the whole graph's backward runs. By the
end of the forward loop, every chunk's logits are simultaneously alive -- the
spike is time-shifted (accumulated across the loop instead of allocated in one
shot), not removed. Peak memory is unchanged (or worse, since now there is also
loop overhead).

THE FIX: recompute-in-backward. Each chunk's lm_head-matmul-then-CE is wrapped in
`torch.utils.checkpoint.checkpoint` (`use_reentrant=False`). During the forward
pass, the chunk's logits are computed, immediately consumed to produce a scalar
partial loss, and then discarded (checkpoint does not save them) -- only the
chunk's small inputs (the hidden-state slice and the label slice) are kept.
During backward, checkpoint reruns the chunk's forward (recomputing the same
logits from the saved small inputs) and immediately backpropagates through it,
then discards the recomputed logits before moving to the next chunk. At no point
-- forward OR backward -- are two chunks' logits alive at once, so peak memory
equals ONE chunk's logits: `4 x 1024 x 150_000 x 4 B ~ 2.5 GB` at micro-batch 4
and chunk_size=1024, a ~4x reduction that scales with `n / chunk_size`. The cost
is one extra lm_head GEMM per chunk in backward -- small next to a 24-layer
transformer backward.

The full `[B, n, V]` fp32 logits tensor must never be materialized by this
module, at any point, for any reason. Both properties -- loss equality and
recompute-in-backward's actual memory behavior -- are enforced by
`gates.gate_chunked_ce` (repo convention, docs/training-recipe.md, "Correctness
gates": every optimized path keeps a naive oracle, and training scripts refuse to
launch unless the gates are green).

WHAT THIS MODULE PROVIDES
--------------------------
`reference_cross_entropy` -- the transparent, unchunked oracle: one lm_head
matmul over the full sequence, one `F.cross_entropy` call. Kept permanently, not
as a fallback but as the ground truth every other path is gated against.

`chunked_cross_entropy` -- the memory-safe path described above. Numerically
exact (not merely close) to the oracle in fp32: both compute the identical sum
of per-position fp32 negative log-likelihoods, divided by the identical
non-ignored-label count. The only difference is *when* logits are allocated and
freed, not what is computed.

CROSS-ENTROPY SEMANTICS (matched to the training loop's usage in heal.py -- see
that module's `mode="train"` forward for the call site this replaces)
-----------------------------------------------------------------------
- No label shifting inside either function. The caller shifts: hidden state at
  position t already corresponds to the position-t input predicting token t+1,
  so `hidden[:, t]` lines up 1:1 with `labels[:, t]`. heal.py forms this pairing
  once via `inp, tgt = ids[:, :-1], ids[:, 1:]`.
- fp32 logits. The lm_head matmul is cast to fp32 before the softmax/NLL for
  numerical stability, matching the CE call already in heal.py/model.py
  (`logits.float()...`) -- this holds under bf16 autocast (CUDA) and under plain
  fp32 (MPS/CPU) alike, since the cast is explicit and not autocast-dependent.
- Mean reduction over non-ignored positions, computed as
  `sum(per-position NLL) / count(labels != ignore_index)` -- global normalization
  by the TOTAL non-ignored count across every chunk, not a per-chunk mean of
  means (chunks can have different counts of non-ignored labels, especially the
  last partial chunk; averaging per-chunk means would silently mis-weight them).
  This is mathematically what `F.cross_entropy(..., reduction="mean")` computes
  in one shot too -- but `reference_cross_entropy` computes it via the same
  explicit sum-then-divide shape as `chunked_cross_entropy` (reduction="sum" /
  count) rather than delegating to the fused 'mean' kernel; see that function's
  own docstring for why (a measured ULP-level difference between the two that
  otherwise eats into the tight bit-honesty tolerance this module is gated at).
- No z-loss, no label smoothing, no aux loss of any kind -- heal.py's LM loss is
  plain next-token CE; the aux-KL for the summarizers is a wholly separate term
  computed elsewhere (see model.py's `_last_aux`) and is untouched by this
  module.
- `ignore_index` defaults to -100 (the `F.cross_entropy` default) for API
  generality; the current training corpus never emits it (every position in a
  WindowStream/FamMixer batch is a real token), so on today's call sites
  `chunked_cross_entropy` and `reference_cross_entropy` are simply the mean over
  every position -- but both are written to be correct if a future data path
  introduces padding.
- Tied embeddings: `lm_head_weight` is passed BY REFERENCE (as HBAModel does:
  `self.lm_head = donor.lm_head`, tied to `core.embed_tokens.weight`). Neither
  function copies it; gradients accumulate into the same tensor object whether
  it is also used as an embedding table elsewhere in the same backward pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def reference_cross_entropy(hidden, lm_head_weight, labels, *, bias=None, ignore_index=-100):
    """Transparent, unchunked correctness oracle. Materializes the full
    `[B, n, V]` fp32 logits tensor -- this is exactly the memory spike
    `chunked_cross_entropy` exists to avoid, so this function is for
    correctness-gating and small-scale use only, never the training hot path at
    real context length.

    Deliberately computed as `reduction="sum" / non-ignored count` rather than
    handing `reduction="mean"` to `F.cross_entropy` directly: both are the same
    mathematical quantity, but PyTorch's fused 'mean' kernel and a manual
    sum-then-divide round differently at the ULP level, which is a real
    (measured) fraction of the tight bit-honesty tolerance `gates.gate_chunked_ce`
    checks against. Using the same sum-then-divide shape here as
    `chunked_cross_entropy` uses (single unchunked sum vs several chunk sums)
    isolates that comparison to the one difference this module actually cares
    about -- chunking -- instead of also absorbing an unrelated kernel-choice
    difference."""
    logits = F.linear(hidden, lm_head_weight, bias).float()
    total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                            ignore_index=ignore_index, reduction="sum")
    total_valid = (labels != ignore_index).sum()
    return total / total_valid.to(total.dtype)


def _chunk_ce_sum(hidden_chunk, weight, bias, label_chunk, ignore_index):
    """One chunk's lm_head matmul + CE, reduction='sum'. This is the function
    that gets checkpointed: its fp32 logits tensor lives only for the duration of
    this call (forward: computed then immediately reduced to a scalar and
    discarded; backward: recomputed from the saved small inputs, used, and
    discarded again)."""
    logits = F.linear(hidden_chunk, weight, bias).float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), label_chunk.reshape(-1),
                           ignore_index=ignore_index, reduction="sum")


def chunked_cross_entropy(hidden, lm_head_weight, labels, *, chunk_size=1024, bias=None,
                          ignore_index=-100):
    """Memory-safe token-mean cross-entropy from PRE-lm_head hidden states.

    hidden: [B, n, d] (not yet projected through lm_head)
    lm_head_weight: [V, d] (passed by reference; may be tied to an embedding
        table -- never copied)
    labels: [B, n] int64, already aligned 1:1 with `hidden` (no internal shift)
    chunk_size: sequence-length slice per chunk; the last chunk is partial when
        `n` doesn't divide evenly -- handled by slicing to `min(n, start+chunk_size)`
    bias: optional lm_head bias (None for a bias-free head, the common case for a
        tied donor head)
    ignore_index: label value excluded from both the numerator and the
        normalizing denominator (default -100, matching `F.cross_entropy`)

    Returns a scalar loss numerically exact (fp32) to
    `reference_cross_entropy(hidden, lm_head_weight, labels, bias=bias,
    ignore_index=ignore_index)`, with peak memory bounded to one chunk's logits
    (see module docstring for why chunking alone does not achieve this without
    the recompute-in-backward wrapping done here).
    """
    B, n, d = hidden.shape
    assert labels.shape == (B, n), f"labels shape {tuple(labels.shape)} != hidden's (B,n)={(B, n)}"
    assert chunk_size > 0

    valid = labels != ignore_index
    total_valid = valid.sum()
    if int(total_valid) == 0:
        # No non-ignored labels anywhere: the reference (F.cross_entropy mean
        # reduction over zero elements) returns nan; match that rather than
        # dividing by zero silently differently.
        return hidden.sum() * float("nan")

    total = None
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        hidden_chunk = hidden[:, start:end]
        label_chunk = labels[:, start:end]
        chunk_sum = checkpoint(_chunk_ce_sum, hidden_chunk, lm_head_weight, bias, label_chunk,
                               ignore_index, use_reentrant=False)
        total = chunk_sum if total is None else total + chunk_sum
    return total / total_valid.to(total.dtype)
