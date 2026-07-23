"""CPU correctness tests for hba.chunked_ce (chunked cross-entropy with
recompute-in-backward -- see that module's docstring for the memory-spike
problem this solves and why a naive chunk loop does NOT solve it).

Every test compares `chunked_cross_entropy` against the transparent
`reference_cross_entropy` oracle on tiny, deliberately awkward dims (a
chunk_size that does not evenly divide the sequence length, so a partial last
chunk is always exercised). Dims/seeds below are fixed and were verified (see
gates.gate_chunked_ce) to land the fp32 chunked-vs-reference loss difference at
essentially 0 ULPs for seed=0 -- fp32 matmul reassociation between a chunked and
an unchunked lm_head matmul is a real, bounded source of ULP-level noise (a
handful of 1e-7-scale ULPs at these magnitudes), not a bug; picking a verified
seed keeps this suite deterministic rather than occasionally flaky on an
unlucky draw.
"""

import gc
import os
import sys
import weakref

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from hba.chunked_ce import chunked_cross_entropy, reference_cross_entropy  # noqa: E402
import hba.chunked_ce as chunked_ce_mod  # noqa: E402

TOL = 1e-6


def _make_batch(seed, B, n, d, V, ignore_frac=0.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn(B, n, d, generator=g)
    weight = torch.randn(V, d, generator=g)
    bias = torch.randn(V, generator=g)
    labels = torch.randint(0, V, (B, n), generator=g)
    if ignore_frac > 0:
        mask = torch.rand(B, n, generator=g) < ignore_frac
        labels = labels.masked_fill(mask, -100)
    return hidden, weight, bias, labels


def _leaf(t):
    return t.clone().requires_grad_(True)


# ------------------------------------------------------------------ loss ------
def test_loss_matches_reference_partial_chunk():
    """chunk_size=13 does not divide n=50 (n % chunk_size == 11 != 0): the last
    chunk is a genuine partial slice."""
    B, n, d, V, chunk = 2, 50, 16, 24, 13
    hidden, weight, bias, labels = _make_batch(0, B, n, d, V, ignore_frac=0.15)
    assert n % chunk != 0, "test dims must exercise a partial last chunk"

    loss_ref = reference_cross_entropy(hidden, weight, labels, bias=bias)
    loss_ch = chunked_cross_entropy(hidden, weight, labels, bias=bias, chunk_size=chunk)
    assert abs(loss_ref.item() - loss_ch.item()) <= TOL


def test_loss_matches_reference_no_ignored_labels():
    """Same check with every label real (ignore_frac=0): the common case on this
    repo's current data paths, where no target is ever -100."""
    B, n, d, V, chunk = 2, 48, 12, 20, 20
    hidden, weight, bias, labels = _make_batch(1, B, n, d, V, ignore_frac=0.0)

    loss_ref = reference_cross_entropy(hidden, weight, labels, bias=bias)
    loss_ch = chunked_cross_entropy(hidden, weight, labels, bias=bias, chunk_size=chunk)
    assert abs(loss_ref.item() - loss_ch.item()) <= TOL


def test_bias_none_matches_reference():
    """The real integration (heal.py -> HBAModel.lm_head) typically has NO bias
    (a tied Qwen-style LM head is bias-free): both functions must accept
    bias=None cleanly and still match exactly."""
    B, n, d, V, chunk = 2, 50, 16, 24, 13
    hidden, weight, _bias, labels = _make_batch(0, B, n, d, V)

    h_ref, w_ref = _leaf(hidden), _leaf(weight)
    h_ch, w_ch = _leaf(hidden), _leaf(weight)

    reference_cross_entropy(h_ref, w_ref, labels, bias=None).backward()
    chunked_cross_entropy(h_ch, w_ch, labels, bias=None, chunk_size=chunk).backward()

    assert h_ref.grad is not None and h_ch.grad is not None
    assert (h_ref.grad - h_ch.grad).abs().max().item() <= TOL
    assert (w_ref.grad - w_ch.grad).abs().max().item() <= TOL


def test_single_chunk_when_n_less_than_chunk_size():
    """chunk_size larger than n: exactly one chunk, degenerate but must still
    match exactly (sanity check on the loop's range())."""
    B, n, d, V = 2, 10, 8, 15
    hidden, weight, bias, labels = _make_batch(2, B, n, d, V)

    loss_ref = reference_cross_entropy(hidden, weight, labels, bias=bias)
    loss_ch = chunked_cross_entropy(hidden, weight, labels, bias=bias, chunk_size=1024)
    assert abs(loss_ref.item() - loss_ch.item()) <= TOL


# -------------------------------------------------------------- gradients -----
def test_grad_matches_reference_all_tensors():
    """Recompute-in-backward must be bit-honest, not just loss-honest: hidden,
    weight, AND bias gradients must all agree with the reference."""
    B, n, d, V, chunk = 2, 50, 16, 24, 13
    hidden, weight, bias, labels = _make_batch(0, B, n, d, V, ignore_frac=0.15)

    h_ref, w_ref, b_ref = _leaf(hidden), _leaf(weight), _leaf(bias)
    h_ch, w_ch, b_ch = _leaf(hidden), _leaf(weight), _leaf(bias)

    reference_cross_entropy(h_ref, w_ref, labels, bias=b_ref).backward()
    chunked_cross_entropy(h_ch, w_ch, labels, bias=b_ch, chunk_size=chunk).backward()

    assert (h_ref.grad - h_ch.grad).abs().max().item() <= TOL
    assert (w_ref.grad - w_ch.grad).abs().max().item() <= TOL
    assert (b_ref.grad - b_ch.grad).abs().max().item() <= TOL


# ------------------------------------------------------------- ignore_index ---
def test_ignore_index_excluded_from_loss_and_grad():
    """Positions with label == ignore_index must contribute exactly 0 to both the
    loss numerator and the normalizing denominator -- verified two ways: (1)
    against the reference oracle (which shares the same F.cross_entropy
    ignore_index semantics), and (2) directly, by checking that an all-ignored
    chunk contributes nothing."""
    B, n, d, V, chunk = 2, 40, 8, 17, 9
    hidden, weight, bias, labels = _make_batch(3, B, n, d, V, ignore_frac=0.0)
    labels = labels.clone()
    labels[:, -9:] = -100   # the whole last (partial) chunk is entirely ignored

    loss_ref = reference_cross_entropy(hidden, weight, labels, bias=bias)
    loss_ch = chunked_cross_entropy(hidden, weight, labels, bias=bias, chunk_size=chunk)
    assert abs(loss_ref.item() - loss_ch.item()) <= TOL

    # Normalization sanity: dropping the last chunk's positions from `labels`
    # entirely (shrinking n instead of masking) must give the SAME loss as
    # masking them with ignore_index, proving the denominator is the
    # non-ignored count, not the raw position count.
    hidden_trunc = hidden[:, :-9]
    labels_trunc = labels[:, :-9]
    loss_trunc = chunked_cross_entropy(hidden_trunc, weight, labels_trunc, bias=bias,
                                       chunk_size=chunk)
    assert abs(loss_trunc.item() - loss_ch.item()) <= TOL


def test_ignore_index_partial_within_chunk():
    """Ignored labels scattered inside (not aligned to) chunk boundaries."""
    B, n, d, V, chunk = 2, 33, 10, 19, 7
    hidden, weight, bias, labels = _make_batch(4, B, n, d, V, ignore_frac=0.25)
    assert bool((labels == -100).any()), "test setup should actually produce some ignored labels"

    loss_ref = reference_cross_entropy(hidden, weight, labels, bias=bias)
    loss_ch = chunked_cross_entropy(hidden, weight, labels, bias=bias, chunk_size=chunk)
    assert abs(loss_ref.item() - loss_ch.item()) <= TOL


# ------------------------------------------------------------- tied weight ----
def test_tied_embedding_weight_not_copied_and_accumulates_grad():
    """lm_head_weight may be the SAME tensor object as an embedding table used
    elsewhere in the graph (HBAModel ties donor.lm_head to
    core.embed_tokens.weight). chunked_cross_entropy must not copy it, and
    gradients from both usages must accumulate into the one tensor's .grad."""
    B, n, d, V, chunk = 2, 30, 8, 14, 9
    hidden, weight, bias, labels = _make_batch(5, B, n, d, V)
    tied = weight.clone().requires_grad_(True)

    # a second, unrelated consumer of the SAME weight tensor (standing in for an
    # embedding lookup elsewhere in a real model's forward)
    other_ids = torch.randint(0, V, (B, n))
    embed_out = F.embedding(other_ids, tied)
    embed_loss = embed_out.sum()

    ce_loss = chunked_cross_entropy(hidden, tied, labels, bias=bias, chunk_size=chunk)
    total = ce_loss + embed_loss
    total.backward()

    assert tied.grad is not None
    # cross-check: the CE-only contribution to weight.grad must match a fresh
    # (untied) chunked call's weight grad exactly (no interference/corruption
    # from being used twice, and no accidental copy: this is the SAME
    # `chunked_cross_entropy` call, just wrapped in a bigger graph).
    w_isolated = weight.clone().requires_grad_(True)
    chunked_cross_entropy(hidden.clone().requires_grad_(True), w_isolated, labels, bias=bias,
                          chunk_size=chunk).backward()
    embed_only = weight.clone().requires_grad_(True)
    F.embedding(other_ids, embed_only).sum().backward()
    expected = w_isolated.grad + embed_only.grad
    assert (tied.grad - expected).abs().max().item() <= 1e-5


# ------------------------------------------------------------ memory proxy ----
def test_forward_frees_chunk_logits_cpu_proxy(monkeypatch):
    """GPU-only peak-memory verification lives in gates.gate_chunked_ce (it needs
    torch.cuda.max_memory_allocated, unavailable on CPU/MPS). This is the
    documented CPU-side proxy for the same property: instrument
    `_chunk_ce_sum` to record a weakref to each chunk's logits tensor, then
    assert every one of them is already garbage after the forward pass
    returns -- i.e. no chunk's logits are still reachable/retained once the
    next chunk starts, which is exactly what the naive (non-checkpointed) chunk
    loop described in chunked_ce.py's module docstring would violate (it keeps
    every chunk's logits alive for ITS OWN backward, so none of them would be
    freed until the whole loss's backward finishes).

    The weakref-is-garbage check ALONE does not actually distinguish the
    checkpointed path from a naive non-checkpointed chunk loop: autograd
    retains the log-softmax/CE OUTPUT needed for backward, not the `logits`
    tensor itself, so a naive loop's per-chunk `logits` would also become
    unreachable (and its weakref would also read None) once the forward
    function returns, despite every chunk's graph (and the memory it pins)
    staying alive until the very end. The property that actually proves
    recompute-in-backward is happening is INVOCATION COUNT: `_chunk_ce_sum`
    (the checkpointed function) must be called exactly twice per chunk -- once
    during the forward pass, once again during backward when checkpoint
    recomputes it -- whereas a naive (non-checkpointed) loop would call its
    per-chunk function exactly once per chunk, period. That 2x-vs-1x count is
    asserted below alongside the weakref check."""
    captured = []
    call_count = [0]

    def spy(hidden_chunk, weight, bias, label_chunk, ignore_index):
        call_count[0] += 1
        logits = F.linear(hidden_chunk, weight, bias).float()
        captured.append(weakref.ref(logits))
        out = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), label_chunk.reshape(-1),
                              ignore_index=ignore_index, reduction="sum")
        del logits
        return out

    monkeypatch.setattr(chunked_ce_mod, "_chunk_ce_sum", spy)

    B, n, d, V, chunk = 2, 50, 16, 24, 13
    hidden, weight, bias, labels = _make_batch(0, B, n, d, V)
    hidden = hidden.requires_grad_(True)
    weight = weight.requires_grad_(True)

    n_chunks = 4   # ceil(50/13)
    loss = chunked_ce_mod.chunked_cross_entropy(hidden, weight, labels, bias=bias,
                                                chunk_size=chunk)
    gc.collect()
    assert len(captured) == n_chunks
    assert call_count[0] == n_chunks, \
        f"expected exactly one _chunk_ce_sum call per chunk in forward, got {call_count[0]}"
    assert all(w() is None for w in captured), \
        "every chunk's logits must be freed by the time the forward loop returns"

    loss.backward()
    gc.collect()
    assert call_count[0] == 2 * n_chunks, (
        f"expected _chunk_ce_sum to be invoked exactly 2x per chunk (once forward, once "
        f"recomputed in backward) -- got {call_count[0]} calls for {n_chunks} chunks. A "
        "count stuck at 1x per chunk here would mean checkpoint's recompute-in-backward "
        "isn't actually happening (i.e. this would no longer distinguish the checkpointed "
        "path from a naive non-checkpointed chunk loop, which the weakref check alone "
        "cannot do -- see this test's docstring)."
    )
    assert all(w() is None for w in captured), \
        "recomputed logits must also be freed again after their chunk's backward"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
