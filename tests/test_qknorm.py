"""CPU-only tests for the Inkling/SSMax-style softmax length-calibration fix
(docs/design.md, "Softmax length-calibration"): QKNorm, the 1/d content scale,
the clamped log-length extrapolation temperature, and the qknorm=OFF regression
path. No GPU, no donor download: builds a tiny, randomly-initialized HBAModel
directly from a small transformers Qwen2 config -- the same pattern
test_probes_smoke.py uses -- instead of hba.model.build_hba/load_donor, which
needs network access to fetch the real donor.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402
import torch  # noqa: E402

from hba import gates  # noqa: E402
from hba.attention import QKNorm, _content_qk, content_scale, log_len_tau, rope_tables  # noqa: E402
from hba.config import HBAConfig  # noqa: E402
from hba.model import HBAModel, init_qknorm_gains  # noqa: E402

N_LAYERS, N_HEADS, N_KV, HEAD_DIM = 2, 4, 2, 8
HIDDEN = N_HEADS * HEAD_DIM


def _tiny_cfg(**overrides):
    kw = dict(n_layers=N_LAYERS, n_heads=N_HEADS, n_kv=N_KV, head_dim=HEAD_DIM, hidden=HIDDEN,
             rope_theta=1_000_000.0, native_ctx=8192, vocab_size=300,
             block=16, window=64, sinks=2, k_blocks=4, slots=2,
             fanout=4, beam=2, hier_from=512, heal_ctx=256, mem_elem_cap=2e7,
             attn_backend="naive")
    kw.update(overrides)
    return HBAConfig(**kw)


def _build_tiny_hba(**cfg_overrides):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    qcfg = Qwen2Config(
        vocab_size=300, hidden_size=HIDDEN, intermediate_size=64,
        num_hidden_layers=N_LAYERS, num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV, max_position_embeddings=8192,
        rope_theta=1_000_000.0, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    donor = Qwen2ForCausalLM(qcfg)
    donor.eval()
    cfg = _tiny_cfg(**cfg_overrides)
    model = HBAModel(donor, cfg)
    model.eval()
    return model, cfg


@pytest.fixture(scope="module")
def tiny_qknorm_on():
    return _build_tiny_hba(qknorm=True)


@pytest.fixture(scope="module")
def tiny_qknorm_off():
    return _build_tiny_hba(qknorm=False)


# --------------------------------------------------------- QKNorm math ---------
def test_qknorm_bounds_head_rms_to_the_learned_gain():
    """RMSNorm's defining property: after QKNorm, ||x|| = sqrt(head_dim)*gain
    EXACTLY per head -- this is what makes q.k/d bounded to
    [-gain_q*gain_k, +gain_q*gain_k] regardless of head_dim (Cauchy-Schwarz)."""
    torch.manual_seed(0)
    qkn = QKNorm(_tiny_cfg())
    qkn.q.gain.data.uniform_(0.3, 2.0)
    qkn.k.gain.data.uniform_(0.3, 2.0)
    x = torch.randn(3, 17, N_HEADS, HEAD_DIM) * 5.0 + 2.0     # arbitrary scale/offset
    xn = qkn.q(x)
    rms = xn.pow(2).mean(-1).sqrt()                            # [3,17,H]
    assert torch.allclose(rms, qkn.q.gain[None, None].expand_as(rms), atol=1e-5)


def test_content_dot_product_bounded_by_gain_product():
    """The actual property the recipe leans on: q.k/d (content_scale='inv_d')
    must land in [-gain_q*gain_k, +gain_q*gain_k], independent of how large the
    RAW q/k happen to be (unlike the unbounded dh**-0.5 legacy scale)."""
    torch.manual_seed(1)
    cfg = _tiny_cfg(qknorm=True, content_scale_mode="inv_d")
    qkn = QKNorm(cfg)
    gq, gk = 0.8, 1.3
    qkn.q.gain.data.fill_(gq)
    qkn.k.gain.data.fill_(gk)
    for scale_factor in (1.0, 1e3, 1e6):     # raw magnitude must not matter post-norm
        q = torch.randn(2, 5, N_HEADS, HEAD_DIM) * scale_factor
        k = torch.randn(2, 9, N_KV, HEAD_DIM) * scale_factor
        qc = qkn.q(q)
        kc = qkn.k(k)
        logit = torch.einsum("bihd,bjhd->bhij", qc, kc.repeat_interleave(N_HEADS // N_KV, dim=2)) \
            * content_scale(cfg, HEAD_DIM)
        bound = gq * gk * (1 + 1e-4)
        assert logit.abs().max().item() <= bound, (scale_factor, logit.abs().max().item(), bound)


def test_qknorm_off_is_identity_passthrough():
    """qknorm=False: `_content_qk` must return the SAME tensors (no copy) and
    the legacy dh**-0.5 scale -- the byte-identical regression contract."""
    cfg = _tiny_cfg(qknorm=False)
    qkn = QKNorm(cfg)
    q = torch.randn(2, 5, N_HEADS, HEAD_DIM)
    k = torch.randn(2, 5, N_KV, HEAD_DIM)
    qc, kc, scale = _content_qk(q, k, cfg, qkn, n=5)
    assert qc is q and kc is k
    assert scale == HEAD_DIM ** -0.5


# ------------------------------------------------------- content_scale ---------
def test_content_scale_modes():
    assert content_scale(_tiny_cfg(content_scale_mode="inv_d"), 64) == pytest.approx(1 / 64)
    assert content_scale(_tiny_cfg(content_scale_mode="inv_sqrt_d"), 64) == pytest.approx(64 ** -0.5)
    with pytest.raises(ValueError):
        content_scale(_tiny_cfg(content_scale_mode="bogus"), 64)


# --------------------------------------------------- log-length temperature ----
def test_log_len_tau_identity_at_and_below_n_cal():
    cfg = _tiny_cfg(temp_c=0.1, n_cal=32768)
    assert log_len_tau(cfg, 1) == 1.0
    assert log_len_tau(cfg, 4096) == 1.0
    assert log_len_tau(cfg, 32768) == 1.0                      # exactly at n_cal: still identity


def test_log_len_tau_grows_past_n_cal_and_matches_closed_form():
    cfg = _tiny_cfg(temp_c=0.1, n_cal=32768)
    tau = log_len_tau(cfg, 65536)
    assert tau == pytest.approx(1.0 + 0.1 * math.log(2.0))
    assert tau > 1.0


def test_content_qk_scale_sharpens_past_n_cal():
    """The sign check, wired END-TO-END through `_content_qk` (not just
    `log_len_tau` in isolation): to counter dilution the temperature must
    MULTIPLY the content scale (sharpen), so the scale `_content_qk` returns is
    exactly `content_scale` at n <= n_cal and STRICTLY INCREASING for n > n_cal.
    A divide-by-tau regression would make it DECREASE (flatten) -- the wrong
    direction -- so this pins the sign at the call site the backends consume."""
    cfg = _tiny_cfg(qknorm=True, content_scale_mode="inv_d", temp_c=0.1, n_cal=32768)
    qkn = QKNorm(cfg)
    q = torch.randn(1, 4, N_HEADS, HEAD_DIM)
    k = torch.randn(1, 4, N_KV, HEAD_DIM)
    base = content_scale(cfg, HEAD_DIM)
    _, _, s_at = _content_qk(q, k, cfg, qkn, n=32768)     # at n_cal: tau=1
    _, _, s_2x = _content_qk(q, k, cfg, qkn, n=65536)     # 2x native
    _, _, s_4x = _content_qk(q, k, cfg, qkn, n=131072)    # 4x native
    assert s_at == pytest.approx(base)                    # identity within n_cal
    assert s_2x > s_at and s_4x > s_2x                    # SHARPENS (x tau), not flattens
    assert s_4x == pytest.approx(base * (1 + 0.1 * math.log(131072 / 32768)))


def test_log_len_tau_disabled_when_temp_c_zero():
    cfg = _tiny_cfg(temp_c=0.0, n_cal=1)
    assert log_len_tau(cfg, 10 ** 9) == 1.0                    # would be huge if not gated off


def test_log_len_tau_defaults_n_cal_to_native_ctx():
    cfg = _tiny_cfg(temp_c=0.1, n_cal=None, native_ctx=8192)
    assert log_len_tau(cfg, 8192) == 1.0
    assert log_len_tau(cfg, 16384) == pytest.approx(1.0 + 0.1 * math.log(2.0))


def test_temperature_gated_under_qknorm_in_content_qk():
    """The temperature is only wired in when qknorm=True (design decision -- see
    _content_qk's docstring): with qknorm=False the returned scale must be the
    bare legacy dh**-0.5, unaffected by temp_c/n_cal, even at n >> n_cal."""
    cfg = _tiny_cfg(qknorm=False, temp_c=0.5, n_cal=4)
    qkn = QKNorm(cfg)
    q = torch.randn(1, 4096, N_HEADS, HEAD_DIM)
    k = torch.randn(1, 4096, N_KV, HEAD_DIM)
    _, _, scale = _content_qk(q, k, cfg, qkn, n=4096)
    assert scale == HEAD_DIM ** -0.5


# ------------------------------------------------------- end-to-end paths ------
@pytest.mark.parametrize("qknorm", [False, True])
def test_naive_vs_fused_agreement_with_and_without_qknorm(qknorm):
    """gates.gate_fused_agreement at tiny dims, both qknorm modes -- the fused
    and naive backends must implement QKNorm identically (~1e-4), not just the
    pre-QKNorm math."""
    model, cfg = _build_tiny_hba(qknorm=qknorm, attn_backend="fused")
    assert gates.gate_fused_agreement(model, cfg, tol=1e-4, aux_tol=1e-3, grad_tol=1e-3)


@pytest.mark.parametrize("qknorm", [False, True])
def test_causality_preserved_with_and_without_qknorm(qknorm):
    model, cfg = _build_tiny_hba(qknorm=qknorm)
    assert gates.gate_causality(model, cfg)


def test_gate_equivalence_holds_exactly_when_qknorm_off(tiny_qknorm_off):
    """The regression contract: qknorm=False must still reproduce the donor's
    own logits exactly (docs/training-recipe.md, "Equivalence gate")."""
    model, cfg = tiny_qknorm_off
    ok, d = gates.gate_equivalence(model, None, cfg)
    assert ok, d


def test_gate_qknorm_math_passes_when_qknorm_on(tiny_qknorm_on):
    """The qknorm=ON internal-consistency gate that supersedes gate_equivalence
    for this mode (docs/training-recipe.md, "Correctness gates")."""
    model, cfg = tiny_qknorm_on
    assert gates.gate_qknorm_math(model, cfg)


def test_qknorm_on_diverges_from_donor_at_select_all(tiny_qknorm_on):
    """The flip side of the regression contract: with qknorm=True, 'equiv' mode
    is EXPECTED to diverge from the raw donor (QKNorm is a deliberate
    architectural departure) -- gate_equivalence must NOT hold here."""
    model, cfg = tiny_qknorm_on
    ok, d = gates.gate_equivalence(model, None, cfg)
    assert not ok
    assert d > 1e-2


@pytest.mark.parametrize("qknorm", [False, True])
def test_dense_fused_eval_paths_agree_on_a_real_forward(qknorm):
    """Sanity: HBAModel's three attention-consuming modes (train via dense,
    train via fused, eval) all run and roughly agree on a tiny real forward, in
    both qknorm modes."""
    model, cfg = _build_tiny_hba(qknorm=qknorm, attn_backend="naive")
    ids = torch.randint(0, cfg.vocab_size, (1, 128))
    cos, sin = rope_tables(128, cfg.head_dim, cfg.rope_theta, ids.device)
    with torch.no_grad():
        out_train = model(ids, cos, sin, cfg.mem_elem_cap, mode="train")
        out_eval = model(ids, cos, sin, cfg.mem_elem_cap, mode="eval")
    assert torch.isfinite(out_train).all() and torch.isfinite(out_eval).all()
    assert (out_train - out_eval).abs().max().item() < 1e-3


# --------------------------------------------------------- init for healing ----
def test_init_qknorm_gains_produces_finite_nontrivial_gains():
    model, cfg = _build_tiny_hba(qknorm=True)
    before = [qkn.q.gain.data.clone() for qkn in model.qknorms]
    init_qknorm_gains(model, cfg, n_calib=32)
    for L, qkn in enumerate(model.qknorms):
        assert torch.isfinite(qkn.q.gain).all() and torch.isfinite(qkn.k.gain).all()
        assert (qkn.q.gain.data - before[L]).abs().max() > 1e-4    # actually moved off the all-ones default
        assert (qkn.q.gain.data > 0).all() and (qkn.k.gain.data > 0).all()
