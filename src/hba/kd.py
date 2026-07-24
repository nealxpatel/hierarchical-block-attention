"""Donor knowledge-distillation (KD): a full-logit KL to the frozen ORIGINAL
donor, added on top of stage-2 healing (docs/training-recipe.md's "Capability
rehearsal" section covers the OTHER stage-2 protection this is additive to).

WHAT THIS IS, AND WHY IT IS NOT THE SUMMARIZER AUX-KL
-------------------------------------------------------
This module distills the LM PATH (Q/K/V/O/MLP/embed/lm_head) onto the frozen
donor's full next-token distribution. It is unrelated to, and must never be
confused with, the summarizer auxiliary KL computed elsewhere (`cfg.aux_w *
aux` in heal.py's loss, gradient-isolated onto the summarizer probes/proj
only) -- that other KL distills ROUTING (which blocks a query attends to) from
the donor-path attention mass. This module's KL distills the OUTPUT
DISTRIBUTION (which token comes next). The two are separate loss terms, on
separate parameter groups, computed from separate teachers.

The conversion goal healing serves is "same behavior, new attention": the
healed model should reproduce the frozen donor's full output distribution,
not just the handful of capabilities a probe panel happens to exercise.
Capability rehearsal (fam_data.FamMixer, mixed into the stage-2 data) protects
specific ENUMERATED capabilities by re-exposing their circuits to gradient
pressure. A full-logit KL to the frozen donor is a complementary, broader
protection: at every training position it pulls the ENTIRE next-token
distribution toward the donor's, which covers capabilities the panel never
happened to probe. This module is that KL; it is ADDITIVE to rehearsal, not a
replacement for it (see heal.py's stage-2 integration).

DIRECTION AND TEMPERATURE (standard Hinton-style KD)
-------------------------------------------------------
The loss is the temperature-scaled FORWARD KL from teacher to student:

    L_KD = T^2 * KL( softmax(z_teacher / T) || softmax(z_student / T) )
         = T^2 * sum_v p_teacher(v) * [ log p_teacher(v) - log q_student(v) ]

Direction matters. This is `KL(teacher || student)`, i.e. the teacher's
distribution is the fixed target `p` and the student's is the moving `q`, with
`p` OUTSIDE the log-ratio. That direction is "mass-covering" (mean-seeking):
wherever the teacher places probability mass, the student is penalized
without bound as its own probability there goes to zero. A student that instead
minimized the REVERSE KL, `KL(student || teacher)`, could satisfy the
objective by mode-collapsing onto a subset of the teacher's support -- cheap to
optimize, but exactly the failure mode this loss exists to prevent (silently
narrowing the distribution while ordinary loss/perplexity stays flat, the same
blind spot that motivates capability rehearsal in the first place). Forward KL
forces the student to cover the teacher's whole distribution instead.

`T^2` is the standard Hinton-distillation correction: softening the
distributions with temperature `T > 1` shrinks the KD loss's gradient
magnitude by roughly `1/T^2` (softmax's Jacobian scales with `1/T`), so
multiplying the loss by `T^2` keeps the KD gradient's scale comparable across
temperature choices -- without it, retuning `--kd-temp` would silently retune
the effective KD learning rate too, confounding the two knobs. At `T=1` this
is a no-op (`T^2 = 1`).

NORMALIZATION AND IGNORE_INDEX
-------------------------------------------------------
Mean-reduced over non-ignored positions: `sum(per-position KL) /
count(labels != ignore_index)`, matching `chunked_ce.py`'s CE convention
exactly (sum-then-divide, not a per-chunk mean of means) so CE and KD agree
on which positions count and normalize identically. `labels` is required here
even though the KL itself doesn't consume label values (only their
ignore_index status) -- reusing the SAME `labels` tensor the caller already
computed for CE is what keeps the two losses' position sets identical; passing
label VALUES rather than a bare boolean mask matches the calling convention of
`chunked_ce.chunked_cross_entropy` this module is meant to sit next to.

MEMORY: RECOMPUTE-IN-BACKWARD, MIRRORING chunked_ce.py
-------------------------------------------------------
The student's `[B, n, V]` logits must never be materialized in one shot, for
the same reason `chunked_ce.py` avoids it for cross-entropy (see that module's
docstring for the full argument: chunking alone does not bound peak memory --
ordinary autograd keeps every chunk's logits alive for its own backward unless
each chunk's forward is wrapped in `torch.utils.checkpoint.checkpoint`, so
recomputation happens explicitly). `chunked_kd_kl` mirrors that pattern
exactly: each chunk's `lm_head` matmul + softened KL is checkpointed
(`use_reentrant=False`), computed and reduced to a scalar in the forward pass,
then discarded; backward recomputes the chunk's logits from the small saved
inputs and immediately frees them again. Peak memory is bounded to one
chunk's student logits, exactly as in `chunked_ce.chunked_cross_entropy`.

TEACHER LOGITS: FULL TENSOR (THIS MODULE) VS. PER-CHUNK (LARGER DONORS)
-------------------------------------------------------------------------
`teacher_logits` is expected as a precomputed, already-detached `[B, n, V]`
tensor (the caller runs the frozen teacher's dense forward under
`torch.no_grad()` -- see heal.py's stage-2 integration). At the 0.5B pilot
scale this is an acceptable memory cost (no grad, no autograd graph pinned to
it -- it is a plain leaf tensor the chunking loop below slices, not something
this module needs to protect against a memory spike). For a substantially
larger donor where materializing the full teacher-logits tensor is itself the
memory concern, the natural extension is to accept a callable
`teacher_logits(start, end) -> Tensor[B, end-start, V]` in place of the dense
tensor, so the teacher's own forward can be chunked (or even fused with its
own per-chunk lm_head projection) the same way the student's is here -- that
extension is NOT implemented in this module; only the full-tensor path is.

WHAT THIS MODULE PROVIDES
-------------------------------------------------------
`reference_kd_kl` -- the transparent, unchunked oracle: one lm_head matmul
over the full sequence, one softened-KL computation. Kept permanently as the
ground truth `chunked_kd_kl` is gated against (repo convention, docs/
training-recipe.md, "Correctness gates": every optimized path keeps a naive
oracle; see gates.gate_kd).

`chunked_kd_kl` -- the memory-safe path described above. Numerically exact
(fp32, up to ordinary chunked-matmul reassociation noise -- see
`chunked_ce.py`'s docstring for why that noise is real but bounded, not a
correctness bug) to `reference_kd_kl` on the same inputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _kd_kl_masked_sum(student_logits, teacher_logits, label_chunk, temperature, ignore_index):
    """Shared math for one chunk (or the whole sequence, for the reference):
    softened forward-KL KL(teacher || student), summed over vocab, masked to
    non-ignored positions, summed over positions. Returns a scalar (no T^2, no
    normalization -- both are applied once by the caller after every chunk's
    sum has been accumulated, so chunked and unchunked callers apply them in
    the same place / same order)."""
    s_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t_logp = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    t_p = t_logp.exp()
    kl = (t_p * (t_logp - s_logp)).sum(-1)                    # [*, positions]
    valid = (label_chunk != ignore_index)
    kl = torch.where(valid, kl, torch.zeros_like(kl))
    return kl.sum()


def reference_kd_kl(student_hidden, teacher_logits, lm_head_weight, labels, *, bias=None,
                    temperature=1.0, ignore_index=-100):
    """Transparent, unchunked correctness oracle. Materializes the full
    `[B, n, V]` fp32 student logits tensor -- exactly the memory spike
    `chunked_kd_kl` exists to avoid -- so this is for correctness-gating and
    small-scale use only, never the training hot path at real context length.

    student_hidden: [B, n, d] (PRE-lm_head hidden states, as returned by
        HBAModel(..., return_hidden=True))
    teacher_logits: [B, n, V] (precomputed by the caller under no_grad from
        the frozen teacher's dense forward; detached, no grad flows through it)
    lm_head_weight: [V, d] (passed by reference; may be tied to an embedding
        table -- never copied)
    labels: [B, n] int64, used ONLY for ignore_index masking (see module
        docstring) -- aligned 1:1 with student_hidden/teacher_logits exactly
        as chunked_ce.py's `labels` is aligned with `hidden`
    """
    student_logits = F.linear(student_hidden, lm_head_weight, bias)
    total = _kd_kl_masked_sum(student_logits, teacher_logits, labels, temperature, ignore_index)
    total_valid = (labels != ignore_index).sum()
    return (total / total_valid.to(total.dtype)) * (temperature ** 2)


def _kd_chunk_sum(hidden_chunk, weight, bias, teacher_logits_chunk, label_chunk, temperature,
                  ignore_index):
    """One chunk's lm_head matmul + softened KL, returned as a masked SUM (not
    yet normalized or T^2-scaled -- see `_kd_kl_masked_sum`). This is the
    function `chunked_kd_kl` checkpoints: its fp32 student-logits tensor lives
    only for the duration of this call (forward: computed, immediately reduced
    to a scalar, discarded; backward: recomputed from the saved small inputs,
    used, discarded again) -- mirrors chunked_ce.py's `_chunk_ce_sum` exactly."""
    student_logits = F.linear(hidden_chunk, weight, bias)
    return _kd_kl_masked_sum(student_logits, teacher_logits_chunk, label_chunk, temperature,
                             ignore_index)


def chunked_kd_kl(student_hidden, teacher_logits, lm_head_weight, labels, *, bias=None,
                  temperature=1.0, chunk_size=1024, ignore_index=-100):
    """Memory-safe donor-KD loss from PRE-lm_head student hidden states.

    student_hidden: [B, n, d] (not yet projected through lm_head)
    teacher_logits: [B, n, V], precomputed by the caller under no_grad from the
        frozen teacher's dense forward (detached; see module docstring for the
        per-chunk-callable extension a larger donor would need instead)
    lm_head_weight: [V, d] (passed by reference; may be tied to an embedding
        table -- never copied)
    labels: [B, n] int64, used only for ignore_index masking, aligned 1:1 with
        student_hidden/teacher_logits (no internal shift -- same convention as
        chunked_ce.chunked_cross_entropy)
    temperature: softmax temperature T (default 1.0, a no-op); see module
        docstring for the T^2 rationale
    chunk_size: sequence-length slice per chunk; the last chunk is partial when
        `n` doesn't divide evenly, handled the same way chunked_ce.py handles it
    bias: optional lm_head bias (None for a bias-free head, the common case for
        a tied donor head)
    ignore_index: label value excluded from both the KL numerator and the
        normalizing denominator (default -100, matching F.cross_entropy)

    Returns a scalar loss numerically exact (fp32, up to ordinary chunked-
    matmul reassociation noise) to `reference_kd_kl` on the same inputs, with
    peak memory bounded to one chunk's student logits (see module docstring).
    """
    B, n, d = student_hidden.shape
    assert labels.shape == (B, n), f"labels shape {tuple(labels.shape)} != hidden's (B,n)={(B, n)}"
    assert teacher_logits.shape[:2] == (B, n), (
        f"teacher_logits shape {tuple(teacher_logits.shape)} does not match student_hidden's "
        f"(B,n)={(B, n)} in its first two dims")
    assert chunk_size > 0

    valid = labels != ignore_index
    total_valid = valid.sum()
    if int(total_valid) == 0:
        # No non-ignored labels anywhere: match chunked_cross_entropy's
        # zero-denominator convention (nan, not a silent divide-by-zero).
        return student_hidden.sum() * float("nan")

    total = None
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        hidden_chunk = student_hidden[:, start:end]
        teacher_chunk = teacher_logits[:, start:end]
        label_chunk = labels[:, start:end]
        chunk_sum = checkpoint(_kd_chunk_sum, hidden_chunk, lm_head_weight, bias, teacher_chunk,
                               label_chunk, temperature, ignore_index, use_reentrant=False)
        total = chunk_sum if total is None else total + chunk_sum
    return (total / total_valid.to(total.dtype)) * (temperature ** 2)
