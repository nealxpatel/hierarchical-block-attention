"""The donor-wrapped HBA model.

`HBAModel` reuses a pretrained dense-attention donor's embeddings, Q/K/V/O
projections, RMSNorms, SwiGLU MLPs, final norm, and (tied) LM head BY REFERENCE,
and replaces only the attention computation with the HBA stack (docs/design.md).
A per-(layer, KV-head) `SlotSummarizer` is added per layer. See docs/training-
recipe.md for how the donor's weights and the new summarizers are trained across
the staged healing recipe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import hba_attention_dense, hba_attention_eval, hba_attention_fused
from .config import DEVICE, DONOR_NAME, log, resolve_backend
from .summarizer import SlotSummarizer


class HBAModel(nn.Module):
    """Donor-wrapped HBA model.

    modes: 'equiv' (dense uniform-RoPE == donor), 'train' (dense HBA + aux),
    'eval' (chunked flat selection), 'eval_hier' (chunked hierarchical selection).
    `self._last_aux` holds the mean aux-KL after a train forward.
    """

    def __init__(self, donor, cfg):
        super().__init__()
        self.cfg = cfg
        self.donor = donor                      # kept as submodule -> its params are trainable
        self.summarizers = nn.ModuleList(SlotSummarizer(cfg) for _ in range(cfg.n_layers))
        self._last_aux = None
        # locate the decoder stack (Qwen2ForCausalLM.model, or any HF causal-LM with
        # the same .model/.lm_head layout)
        self.core = donor.model
        self.lm_head = donor.lm_head
        self.attn_backend = resolve_backend(cfg)
        if self.attn_backend != getattr(cfg, "attn_backend", "naive"):
            log(f"attn_backend '{cfg.attn_backend}' unavailable on {DEVICE}; "
                f"using '{self.attn_backend}'")

    def set_trainable(self, groups):
        """groups: subset of {'summarizers','attn','norms','mlp','embed'}. Everything
        else is frozen."""
        for p in self.parameters():
            p.requires_grad_(False)
        for g in groups:
            if g == "summarizers":
                for p in self.summarizers.parameters():
                    p.requires_grad_(True)
            elif g == "attn":
                for lyr in self.core.layers:
                    for p in lyr.self_attn.parameters():
                        p.requires_grad_(True)
            elif g == "norms":
                for lyr in self.core.layers:
                    for p in lyr.input_layernorm.parameters():
                        p.requires_grad_(True)
                    for p in lyr.post_attention_layernorm.parameters():
                        p.requires_grad_(True)
                for p in self.core.norm.parameters():
                    p.requires_grad_(True)
            elif g == "mlp":
                for lyr in self.core.layers:
                    for p in lyr.mlp.parameters():
                        p.requires_grad_(True)
            elif g == "embed":
                self.core.embed_tokens.weight.requires_grad_(True)   # tied to lm_head
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.parameters())
        log(f"trainable groups={sorted(groups)}: {n_tr/1e6:.1f}M / {n_all/1e6:.1f}M params")

    def forward(self, ids, cos, sin, cap, mode="eval", tail=None, loss_tgt=None):
        """tail: if set, apply the (huge, vocab-way) lm_head only to the last `tail`
        positions and return logits [B, tail, V]. Needle/induction answers cluster
        at the sequence end, so this avoids materializing [B, n, V]. None = full.

        loss_tgt: if set (train path), return the SCALAR mean next-token CE against
        these targets [B, n] instead of logits, computed in fixed-size sequence
        chunks with per-chunk gradient checkpointing so the [B, n, V] logit tensor
        is never materialized (at long context the fp32 logits alone can run into
        the tens of GiB before backward). EXACT math: per-chunk fp32 CE with
        reduction='sum', divided by the total position count == the same mean CE as
        F.cross_entropy over full logits. Mutually exclusive with `tail`."""
        cfg = self.cfg
        B, n = ids.shape
        Hq, Hkv, dh = cfg.n_heads, cfg.n_kv, cfg.head_dim
        x = self.core.embed_tokens(ids)
        auxes = []
        use_ckpt = (mode == "train")
        for L, lyr in enumerate(self.core.layers):
            summ = self.summarizers[L]

            def block_fn(x, lyr=lyr, summ=summ):
                a = lyr.input_layernorm(x)
                q = lyr.self_attn.q_proj(a).view(B, n, Hq, dh)
                k = lyr.self_attn.k_proj(a).view(B, n, Hkv, dh)
                v = lyr.self_attn.v_proj(a).view(B, n, Hkv, dh)
                aux = None
                if mode == "train":
                    attn = (hba_attention_fused if self.attn_backend == "fused"
                            else hba_attention_dense)
                    o, aux = attn(q, k, v, cos, sin, cfg, summ)
                elif mode == "equiv":
                    # memory-capped chunked uniform-RoPE causal attention (== donor).
                    # The dense equiv branch would materialize [B,Hq,n,n] (tens of
                    # GiB at moderate context, absurd at long context), so the
                    # donor / donor+YaRN baselines at eval lengths MUST take this path.
                    o = hba_attention_eval(q, k, v, cos, sin, cfg, summ, cap, equiv=True)
                else:
                    o = hba_attention_eval(q, k, v, cos, sin, cfg, summ, cap,
                                           hier=(mode == "eval_hier"))
                o = lyr.self_attn.o_proj(o.reshape(B, n, Hq * dh))
                x = x + o
                x = x + lyr.mlp(lyr.post_attention_layernorm(x))
                return x, aux

            if use_ckpt and self.training:
                from torch.utils.checkpoint import checkpoint
                x, aux = checkpoint(block_fn, x, use_reentrant=False)
            else:
                x, aux = block_fn(x)
            if aux is not None:
                auxes.append(aux)
        x = self.core.norm(x)
        self._last_aux = (sum(auxes) / len(auxes)) if auxes else None
        if loss_tgt is not None:
            assert tail is None, "loss_tgt and tail are mutually exclusive"
            from torch.utils.checkpoint import checkpoint

            def _ce_sum(xc, tc):
                lg = self.lm_head(xc)
                return F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]),
                                       tc.reshape(-1), reduction="sum")

            CH = 2048
            tot = None
            for cs0 in range(0, x.shape[1], CH):
                xc = x[:, cs0:cs0 + CH]
                tc = loss_tgt[:, cs0:cs0 + CH]
                s_ = (checkpoint(_ce_sum, xc, tc, use_reentrant=False)
                      if self.training else _ce_sum(xc, tc))
                tot = s_ if tot is None else tot + s_
            return tot / loss_tgt.numel()
        if tail is not None:
            x = x[:, -tail:]
        logits = self.lm_head(x)
        return logits


def load_donor(dtype=None, device=None):
    """Load the donor model + tokenizer (offline-friendly; relies on the local HF
    cache once downloaded once)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .config import COMPUTE_DTYPE
    dtype = dtype or COMPUTE_DTYPE
    device = device or DEVICE
    tok = AutoTokenizer.from_pretrained(DONOR_NAME)
    donor = AutoModelForCausalLM.from_pretrained(DONOR_NAME, dtype=dtype,
                                                 attn_implementation="eager").to(device).eval()
    return donor, tok


def build_hba(cfg=None, dtype=None, device=None):
    from .config import HBAConfig
    cfg = cfg or HBAConfig()
    donor, tok = load_donor(dtype=dtype, device=device)
    model = HBAModel(donor, cfg).to(device or DEVICE)
    return model, tok, cfg
