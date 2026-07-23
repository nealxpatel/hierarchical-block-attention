"""CPU-only smoke test for hba.probes (the capability panel). No GPU, no
donor download: builds a tiny, randomly-initialized HBAModel directly from a
small transformers Qwen2 config (a few layers, tiny hidden size, small vocab)
instead of hba.model.build_hba/load_donor, which needs network access to fetch
the real donor.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from hba import probes  # noqa: E402
from hba.config import HBAConfig  # noqa: E402
from hba.model import HBAModel  # noqa: E402


def _build_tiny_hba():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    n_layers, n_heads, n_kv, head_dim = 2, 4, 2, 8
    hidden = n_heads * head_dim
    qcfg = Qwen2Config(
        vocab_size=300, hidden_size=hidden, intermediate_size=64,
        num_hidden_layers=n_layers, num_attention_heads=n_heads,
        num_key_value_heads=n_kv, max_position_embeddings=8192,
        rope_theta=1_000_000.0, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    donor = Qwen2ForCausalLM(qcfg)
    donor.eval()
    # HBA routing knobs deliberately small (fast CPU forward) but still exercise
    # real routing: block=16, window=64, k_blocks=4 -- candidate_blocks at the
    # default P3 length n=4096 is (4096-64)//16=252 >> 4, same inequality
    # induction_far's assert checks at full scale, just with smaller numbers.
    cfg = HBAConfig(
        n_layers=n_layers, n_heads=n_heads, n_kv=n_kv, head_dim=head_dim, hidden=hidden,
        rope_theta=1_000_000.0, native_ctx=8192, vocab_size=300,
        block=16, window=64, sinks=2, k_blocks=4, slots=2,
        fanout=4, beam=2, hier_from=512, heal_ctx=256, mem_elem_cap=2e7,
    )
    model = HBAModel(donor, cfg)
    model.eval()
    return model, cfg


@pytest.fixture(scope="module")
def tiny_hba():
    return _build_tiny_hba()


def test_panel_runs_and_returns_expected_keys(tiny_hba):
    model, cfg = tiny_hba
    rng = np.random.default_rng(0)
    out = probes.run_panel(model, None, cfg, rng)
    # needle_mini (P4) is disabled by default -> must not appear
    assert set(out) == {"induction_std", "induction_near", "induction_far", "val_loss_fixed"}
    for name in ("induction_std", "induction_near", "induction_far"):
        assert 0.0 <= out[name] <= 1.0
    # data/val_books.bin doesn't exist in a bare checkout -> NaN, not a crash
    assert isinstance(out["val_loss_fixed"], float)
    # model.training state must be restored (run_panel forces eval() internally)
    assert model.training is False


def test_run_panel_restores_training_mode(tiny_hba):
    model, cfg = tiny_hba
    model.train()
    rng = np.random.default_rng(0)
    probes.run_panel(model, None, cfg, rng)
    assert model.training is True
    model.eval()


def test_induction_far_candidate_count_assert_at_short_length(tiny_hba):
    model, cfg = tiny_hba
    rng = np.random.default_rng(0)
    # (n - window) // block = (96 - 64) // 16 = 2 <= k_blocks (4): must raise
    # BEFORE any forward pass (the assert is the first thing induction_far does).
    with pytest.raises(AssertionError, match="candidate_blocks"):
        probes.induction_far(model, None, cfg, rng, n=96)

    # sanity: the default n=4096 clears the same inequality with this cfg
    out = probes.induction_far(model, None, cfg, rng, trials=2)
    assert 0.0 <= out["induction_far"] <= 1.0


def test_needle_mini_disabled_by_default_but_runnable_via_which(tiny_hba):
    model, cfg = tiny_hba
    rng = np.random.default_rng(0)
    # data/needle_books.bin doesn't exist in a bare checkout -> NaN, not a crash,
    # even when explicitly requested via `which`.
    out = probes.run_panel(model, None, cfg, rng, which={"needle_mini"})
    assert set(out) == {"needle_mini"}
    assert out["needle_mini"] != out["needle_mini"] or 0.0 <= out["needle_mini"] <= 1.0  # NaN or valid
