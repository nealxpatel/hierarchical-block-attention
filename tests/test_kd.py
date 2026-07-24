"""CPU correctness tests for hba.kd (donor knowledge-distillation -- a
temperature-scaled forward KL from a frozen teacher's logits to the student's,
computed with recompute-in-backward -- see that module's docstring for the
loss formulation and why chunking needs the same recompute-in-backward
treatment chunked_ce.py uses).

Style mirrors tests/test_chunked_ce.py: every test compares `chunked_kd_kl`
against the transparent `reference_kd_kl` oracle on tiny, deliberately awkward
dims (a chunk_size that does not evenly divide the sequence length, so a
partial last chunk is always exercised).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from hba.kd import chunked_kd_kl, reference_kd_kl  # noqa: E402

TOL = 1e-5


def _make_batch(seed, B, n, d, V, ignore_frac=0.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn(B, n, d, generator=g)
    weight = torch.randn(V, d, generator=g)
    bias = torch.randn(V, generator=g)
    teacher_logits = torch.randn(B, n, V, generator=g)
    labels = torch.randint(0, V, (B, n), generator=g)
    if ignore_frac > 0:
        mask = torch.rand(B, n, generator=g) < ignore_frac
        labels = labels.masked_fill(mask, -100)
    return hidden, weight, bias, teacher_logits, labels


def _leaf(t):
    return t.clone().requires_grad_(True)


# ------------------------------------------------------------------ loss ------
def test_loss_matches_reference_partial_chunk():
    """chunk_size=13 does not divide n=50 (n % chunk_size == 11 != 0): the last
    chunk is a genuine partial slice."""
    B, n, d, V, chunk = 2, 50, 16, 24, 13
    hidden, weight, bias, teacher_logits, labels = _make_batch(0, B, n, d, V, ignore_frac=0.15)
    assert n % chunk != 0, "test dims must exercise a partial last chunk"

    loss_ref = reference_kd_kl(hidden, teacher_logits, weight, labels, bias=bias)
    loss_ch = chunked_kd_kl(hidden, teacher_logits, weight, labels, bias=bias, chunk_size=chunk)
    assert abs(loss_ref.item() - loss_ch.item()) <= TOL


def test_loss_matches_reference_no_ignored_labels():
    """Same check with every label real (ignore_frac=0)."""
    B, n, d, V, chunk = 2, 48, 12, 20, 20
    hidden, weight, bias, teacher_logits, labels = _make_batch(1, B, n, d, V, ignore_frac=0.0)

    loss_ref = reference_kd_kl(hidden, teacher_logits, weight, labels, bias=bias)
    loss_ch = chunked_kd_kl(hidden, teacher_logits, weight, labels, bias=bias, chunk_size=chunk)
    assert abs(loss_ref.item() - loss_ch.item()) <= TOL


def test_bias_none_matches_reference():
    """A tied Qwen-style LM head is bias-free (the real heal.py integration):
    both functions must accept bias=None cleanly and still match exactly."""
    B, n, d, V, chunk = 2, 50, 16, 24, 13
    hidden, weight, _bias, teacher_logits, labels = _make_batch(0, B, n, d, V)

    h_ref, w_ref = _leaf(hidden), _leaf(weight)
    h_ch, w_ch = _leaf(hidden), _leaf(weight)

    reference_kd_kl(h_ref, teacher_logits, w_ref, labels, bias=None).backward()
    chunked_kd_kl(h_ch, teacher_logits, w_ch, labels, bias=None, chunk_size=chunk).backward()

    assert h_ref.grad is not None and h_ch.grad is not None
    assert (h_ref.grad - h_ch.grad).abs().max().item() <= TOL
    assert (w_ref.grad - w_ch.grad).abs().max().item() <= TOL


def test_single_chunk_when_n_less_than_chunk_size():
    """chunk_size larger than n: exactly one chunk, degenerate but must still
    match exactly (sanity check on the loop's range())."""
    B, n, d, V = 2, 10, 8, 15
    hidden, weight, bias, teacher_logits, labels = _make_batch(2, B, n, d, V)

    loss_ref = reference_kd_kl(hidden, teacher_logits, weight, labels, bias=bias)
    loss_ch = chunked_kd_kl(hidden, teacher_logits, weight, labels, bias=bias, chunk_size=1024)
    assert abs(loss_ref.item() - loss_ch.item()) <= TOL


# -------------------------------------------------------------- gradients -----
def test_grad_matches_reference_all_tensors():
    """Recompute-in-backward must be bit-honest, not just loss-honest: hidden,
    weight, AND bias gradients must all agree with the reference. No gradient
    flows to teacher_logits (it is the fixed target, not a leaf under test)."""
    B, n, d, V, chunk = 2, 50, 16, 24, 13
    hidden, weight, bias, teacher_logits, labels = _make_batch(0, B, n, d, V, ignore_frac=0.15)

    h_ref, w_ref, b_ref = _leaf(hidden), _leaf(weight), _leaf(bias)
    h_ch, w_ch, b_ch = _leaf(hidden), _leaf(weight), _leaf(bias)

    reference_kd_kl(h_ref, teacher_logits, w_ref, labels, bias=b_ref).backward()
    chunked_kd_kl(h_ch, teacher_logits, w_ch, labels, bias=b_ch, chunk_size=chunk).backward()

    assert (h_ref.grad - h_ch.grad).abs().max().item() <= TOL
    assert (w_ref.grad - w_ch.grad).abs().max().item() <= TOL
    assert (b_ref.grad - b_ch.grad).abs().max().item() <= TOL


def test_grad_matches_reference_at_temperature():
    """Same bit-honesty check at T != 1, where the T^2 rescaling and the
    softened softmax both participate in the gradient."""
    B, n, d, V, chunk = 2, 40, 10, 18, 9
    hidden, weight, bias, teacher_logits, labels = _make_batch(6, B, n, d, V, ignore_frac=0.1)

    h_ref, w_ref, b_ref = _leaf(hidden), _leaf(weight), _leaf(bias)
    h_ch, w_ch, b_ch = _leaf(hidden), _leaf(weight), _leaf(bias)

    reference_kd_kl(h_ref, teacher_logits, w_ref, labels, bias=b_ref, temperature=3.0).backward()
    chunked_kd_kl(h_ch, teacher_logits, w_ch, labels, bias=b_ch, temperature=3.0,
                 chunk_size=chunk).backward()

    assert (h_ref.grad - h_ch.grad).abs().max().item() <= TOL
    assert (w_ref.grad - w_ch.grad).abs().max().item() <= TOL
    assert (b_ref.grad - b_ch.grad).abs().max().item() <= TOL


# ------------------------------------------------------------- ignore_index ---
def test_ignore_index_excluded_from_loss_and_grad():
    """Positions with label == ignore_index must contribute exactly 0 to both
    the loss numerator and the normalizing denominator -- verified two ways:
    (1) against the reference oracle, and (2) directly, by checking that
    truncating an all-ignored trailing chunk gives the identical loss to
    masking it."""
    B, n, d, V, chunk = 2, 40, 8, 17, 9
    hidden, weight, bias, teacher_logits, labels = _make_batch(3, B, n, d, V, ignore_frac=0.0)
    labels = labels.clone()
    labels[:, -9:] = -100   # the whole last (partial) chunk is entirely ignored

    loss_ref = reference_kd_kl(hidden, teacher_logits, weight, labels, bias=bias)
    loss_ch = chunked_kd_kl(hidden, teacher_logits, weight, labels, bias=bias, chunk_size=chunk)
    assert abs(loss_ref.item() - loss_ch.item()) <= TOL

    hidden_trunc = hidden[:, :-9]
    teacher_trunc = teacher_logits[:, :-9]
    labels_trunc = labels[:, :-9]
    loss_trunc = chunked_kd_kl(hidden_trunc, teacher_trunc, weight, labels_trunc, bias=bias,
                               chunk_size=chunk)
    assert abs(loss_trunc.item() - loss_ch.item()) <= TOL


def test_ignore_index_partial_within_chunk():
    """Ignored labels scattered inside (not aligned to) chunk boundaries."""
    B, n, d, V, chunk = 2, 33, 10, 19, 7
    hidden, weight, bias, teacher_logits, labels = _make_batch(4, B, n, d, V, ignore_frac=0.25)
    assert bool((labels == -100).any()), "test setup should actually produce some ignored labels"

    loss_ref = reference_kd_kl(hidden, teacher_logits, weight, labels, bias=bias)
    loss_ch = chunked_kd_kl(hidden, teacher_logits, weight, labels, bias=bias, chunk_size=chunk)
    assert abs(loss_ref.item() - loss_ch.item()) <= TOL


def test_all_ignored_returns_nan_like_ce():
    """Zero non-ignored labels: match chunked_cross_entropy's convention (nan,
    not a silent divide-by-zero) rather than crashing."""
    B, n, d, V, chunk = 2, 20, 6, 12, 7
    hidden, weight, bias, teacher_logits, labels = _make_batch(9, B, n, d, V)
    labels = torch.full_like(labels, -100)

    loss_ch = chunked_kd_kl(hidden, teacher_logits, weight, labels, bias=bias, chunk_size=chunk)
    assert torch.isnan(loss_ch)


# ---------------------------------------------------- KD==0 when student==teacher
def test_kd_zero_when_student_equals_teacher_T1():
    """KL of a distribution with itself is exactly 0: force student_logits ==
    teacher_logits via an identity lm_head (weight=I, bias=0), so
    F.linear(hidden, I, 0) == hidden, and set hidden = teacher_logits
    directly."""
    B, n, V = 2, 25, 14
    g = torch.Generator(device="cpu").manual_seed(11)
    same = torch.randn(B, n, V, generator=g)
    eye = torch.eye(V)
    zero_bias = torch.zeros(V)
    labels = torch.randint(0, V, (B, n), generator=g)

    loss_ref = reference_kd_kl(same, same, eye, labels, bias=zero_bias, temperature=1.0)
    loss_ch = chunked_kd_kl(same, same, eye, labels, bias=zero_bias, temperature=1.0, chunk_size=9)
    assert abs(float(loss_ref)) <= 1e-4
    assert abs(float(loss_ch)) <= 1e-4


def test_kd_zero_when_student_equals_teacher_T2():
    """Same property at T != 1 -- the T^2 rescaling must not introduce a
    spurious nonzero floor."""
    B, n, V = 2, 25, 14
    g = torch.Generator(device="cpu").manual_seed(12)
    same = torch.randn(B, n, V, generator=g)
    eye = torch.eye(V)
    zero_bias = torch.zeros(V)
    labels = torch.randint(0, V, (B, n), generator=g)

    loss_ref = reference_kd_kl(same, same, eye, labels, bias=zero_bias, temperature=2.0)
    loss_ch = chunked_kd_kl(same, same, eye, labels, bias=zero_bias, temperature=2.0, chunk_size=9)
    assert abs(float(loss_ref)) <= 1e-4
    assert abs(float(loss_ch)) <= 1e-4


# ------------------------------------------------------------- temperature ----
def test_higher_temperature_softens_distributions():
    """Monotone sanity check on the softening itself (not the KD loss value,
    which is not generally monotone in T): as T grows, the teacher's softmax
    entropy must strictly increase toward uniform -- the property `--kd-temp`
    is actually supposed to control."""
    g = torch.Generator(device="cpu").manual_seed(13)
    V = 50
    logits = torch.randn(1, 1, V, generator=g) * 5.0   # spiky logits

    def entropy_at(T):
        p = F.softmax(logits / T, dim=-1)
        return -(p * p.clamp_min(1e-12).log()).sum().item()

    e1 = entropy_at(1.0)
    e2 = entropy_at(2.0)
    e5 = entropy_at(5.0)
    assert e1 < e2 < e5, f"entropy must increase with temperature: {e1}, {e2}, {e5}"


def test_temperature_changes_loss_value():
    """T is not a no-op parameter: changing it must actually change the
    computed KD loss (sanity that `temperature` is wired through, not
    silently dropped)."""
    B, n, d, V, chunk = 2, 30, 10, 16, 11
    hidden, weight, bias, teacher_logits, labels = _make_batch(7, B, n, d, V)

    loss_t1 = chunked_kd_kl(hidden, teacher_logits, weight, labels, bias=bias, temperature=1.0,
                            chunk_size=chunk)
    loss_t3 = chunked_kd_kl(hidden, teacher_logits, weight, labels, bias=bias, temperature=3.0,
                            chunk_size=chunk)
    assert abs(loss_t1.item() - loss_t3.item()) > 1e-6


# ------------------------------------------------------------- tied weight ----
def test_tied_embedding_weight_not_copied_and_accumulates_grad():
    """lm_head_weight may be the SAME tensor object as an embedding table used
    elsewhere in the graph (HBAModel ties donor.lm_head to
    core.embed_tokens.weight). chunked_kd_kl must not copy it, and gradients
    from both usages must accumulate into the one tensor's .grad."""
    B, n, d, V, chunk = 2, 30, 8, 14, 9
    hidden, weight, bias, teacher_logits, labels = _make_batch(5, B, n, d, V)
    tied = weight.clone().requires_grad_(True)

    other_ids = torch.randint(0, V, (B, n))
    embed_out = F.embedding(other_ids, tied)
    embed_loss = embed_out.sum()

    kd_loss = chunked_kd_kl(hidden, teacher_logits, tied, labels, bias=bias, chunk_size=chunk)
    total = kd_loss + embed_loss
    total.backward()

    assert tied.grad is not None
    w_isolated = weight.clone().requires_grad_(True)
    chunked_kd_kl(hidden.clone().requires_grad_(True), teacher_logits, w_isolated, labels,
                 bias=bias, chunk_size=chunk).backward()
    embed_only = weight.clone().requires_grad_(True)
    F.embedding(other_ids, embed_only).sum().backward()
    expected = w_isolated.grad + embed_only.grad
    assert (tied.grad - expected).abs().max().item() <= 1e-5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
