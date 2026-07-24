"""Hierarchical Block Attention (HBA): near-constant per-query attention cost at
any context length, without positional extrapolation.

See docs/design.md for the architecture, docs/training-recipe.md for the staged
donor-conversion recipe, and docs/evals.md for the evaluation protocol.

Public API (the pieces most callers need):

  HBAConfig, smoke_config     -- architecture / conversion config
  HBAModel, build_hba         -- the donor-wrapped model and its constructor
  load_donor                  -- load the pretrained donor + tokenizer
  SlotSummarizer               -- the learned block-summary module
  hba_attention_dense          -- the naive reference oracle (train path)
  hba_attention_fused          -- the FlexAttention LSE-merge path (train path)
  hba_attention_eval           -- the chunked eval path (flat or hierarchical)
  rope_tables, yarn_theta      -- RoPE utilities
  QKNorm                       -- per-layer QKNorm gains (docs/design.md,
                                  "Softmax length-calibration")
  content_scale, log_len_tau   -- the shared union-softmax content scale and the
                                  clamped log-length extrapolation temperature
"""

from .attention import (QKNorm, content_scale, hba_attention_dense, hba_attention_eval,
                        hba_attention_fused, log_len_tau, rope_tables, yarn_theta)
from .config import DEVICE, HBAConfig, resolve_backend, smoke_config
from .model import HBAModel, build_hba, load_donor
from .summarizer import SlotSummarizer

__all__ = [
    "HBAConfig",
    "smoke_config",
    "HBAModel",
    "build_hba",
    "load_donor",
    "SlotSummarizer",
    "hba_attention_dense",
    "hba_attention_fused",
    "hba_attention_eval",
    "rope_tables",
    "yarn_theta",
    "QKNorm",
    "content_scale",
    "log_len_tau",
    "resolve_backend",
    "DEVICE",
]
