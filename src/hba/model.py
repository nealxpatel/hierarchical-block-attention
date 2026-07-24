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

from .attention import QKNorm, hba_attention_dense, hba_attention_eval, hba_attention_fused
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
        # Per-layer QKNorm gains (docs/design.md, "Softmax length-calibration").
        # Always constructed (cheap: 2 small parameter vectors per layer) even
        # when cfg.qknorm=False, so a checkpoint/state_dict shape is stable
        # across the flag -- the flag only controls whether attention.py's
        # `_content_qk` actually CALLS these modules, not whether they exist.
        self.qknorms = nn.ModuleList(QKNorm(cfg) for _ in range(cfg.n_layers))
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
                # QKNorm gains are part of the attention geometry (they act
                # directly on Q/K before the content dot product), so they train
                # alongside Q/K/V/O whenever "attn" is trainable -- a no-op when
                # cfg.qknorm=False (the gains exist but attention.py never calls
                # them, so their gradient is always exactly 0 either way).
                for qkn in self.qknorms:
                    for p in qkn.parameters():
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

    def forward(self, ids, cos, sin, cap, mode="eval", tail=None, loss_tgt=None,
               return_hidden=False):
        """tail: if set, apply the (huge, vocab-way) lm_head only to the last `tail`
        positions and return logits [B, tail, V]. Needle/induction answers cluster
        at the sequence end, so this avoids materializing [B, n, V]. None = full.

        loss_tgt: if set (train path), return the SCALAR mean next-token CE against
        these targets [B, n] instead of logits, computed in fixed-size sequence
        chunks with per-chunk gradient checkpointing so the [B, n, V] logit tensor
        is never materialized (at long context the fp32 logits alone can run into
        the tens of GiB before backward). EXACT math: per-chunk fp32 CE with
        reduction='sum', divided by the total position count == the same mean CE as
        F.cross_entropy over full logits. Mutually exclusive with `tail`.

        return_hidden: if set, skip the lm_head entirely and return the POST-NORM,
        PRE-lm_head hidden states [B, n, hidden] instead of logits or a loss. For
        callers that want to run their own memory-safe CE (see
        `hba.chunked_ce.chunked_cross_entropy`) against `self.lm_head.weight` /
        `self.lm_head.bias` outside this function. Mutually exclusive with `tail`
        and `loss_tgt`. Default False, so no existing call site's behavior
        changes."""
        cfg = self.cfg
        B, n = ids.shape
        Hq, Hkv, dh = cfg.n_heads, cfg.n_kv, cfg.head_dim
        x = self.core.embed_tokens(ids)
        auxes = []
        use_ckpt = (mode == "train")
        for L, lyr in enumerate(self.core.layers):
            summ = self.summarizers[L]
            qkn = self.qknorms[L]

            def block_fn(x, lyr=lyr, summ=summ, qkn=qkn):
                a = lyr.input_layernorm(x)
                q = lyr.self_attn.q_proj(a).view(B, n, Hq, dh)
                k = lyr.self_attn.k_proj(a).view(B, n, Hkv, dh)
                v = lyr.self_attn.v_proj(a).view(B, n, Hkv, dh)
                aux = None
                if mode == "train":
                    attn = (hba_attention_fused if self.attn_backend == "fused"
                            else hba_attention_dense)
                    o, aux = attn(q, k, v, cos, sin, cfg, summ, qkn)
                elif mode == "equiv":
                    # memory-capped chunked uniform-RoPE causal attention (== donor
                    # when cfg.qknorm=False; == this model's own QKNorm'd full-
                    # attention limit when cfg.qknorm=True -- see hba_attention_eval's
                    # docstring). The dense equiv branch would materialize
                    # [B,Hq,n,n] (tens of GiB at moderate context, absurd at long
                    # context), so the donor / donor+YaRN baselines at eval
                    # lengths MUST take this path.
                    o = hba_attention_eval(q, k, v, cos, sin, cfg, summ, qkn, cap, equiv=True)
                else:
                    o = hba_attention_eval(q, k, v, cos, sin, cfg, summ, qkn, cap,
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
        if return_hidden:
            assert tail is None and loss_tgt is None, \
                "return_hidden is mutually exclusive with tail and loss_tgt"
            return x
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


@torch.no_grad()
def init_qknorm_gains(model, cfg, n_calib=256, seed=20260716):
    """Data-dependent QKNorm gain init (docs/design.md, "Softmax length-
    calibration"; docs/training-recipe.md's healing-absorption init note).
    QKNorm changes Q/K statistics fundamentally -- the converted model can no
    longer reproduce the donor exactly at init, by design (gates.gate_equivalence
    only holds in qknorm=OFF mode; see gates.gate_qknorm_math for the qknorm=ON
    internal-consistency check instead). This function's job is narrower: pick a
    starting gain so the FIRST healing step begins from a sane attention
    temperature, not an arbitrary one, so healing absorbs the architecture change
    from a good starting point rather than an adversarial one.

    Derivation. Raw (pre-QKNorm) content logit: q.k = ||q|| ||k|| cos(theta) =
    dh * rq * rk * cos(theta), where rq/rk are the donor's own per-(layer,head)
    RMS of q/k (||x|| = sqrt(dh) * RMS(x)). The donor's own 1/sqrt(dh) scale
    turns that into  sqrt(dh) * rq * rk * cos(theta) -- the donor's native logit
    magnitude. Our post-QKNorm vectors have ||qc|| = sqrt(dh) * gain_q (RMSNorm
    pins RMS to 1, times the gain), so qc.kc = dh * gain_q * gain_k * cos(theta);
    the 1/dh content scale (content_scale_mode='inv_d') then gives a post-QKNorm
    content logit of  (gain_q * gain_k) * cos(theta). Matching the two magnitudes
    requires gain_q * gain_k = dh**0.5 * rq * rk, solved (one of many valid splits;
    only the PRODUCT matters for the content logit's scale) by
        gain_q[head] = dh**0.25 * rq[head]      gain_k[head] = dh**0.25 * rk[head]
    rq/rk are measured directly from the FROZEN donor's own pre-RoPE q_proj/
    k_proj outputs on a short random-token calibration batch (no corpus/network
    dependency beyond the donor itself, which is already loaded) -- calibration,
    not training: no gradient, seconds of compute, run once at model construction
    (see build_hba)."""
    dev = next(model.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, cfg.vocab_size, (1, n_calib), generator=g).to(dev)
    Q, K, handles = {}, {}, []

    def mk(store, i):
        def hook(mod, inp, out):
            store[i] = out.detach().float()
        return hook
    for i, lyr in enumerate(model.core.layers):
        handles.append(lyr.self_attn.q_proj.register_forward_hook(mk(Q, i)))
        handles.append(lyr.self_attn.k_proj.register_forward_hook(mk(K, i)))
    was = model.training
    model.eval()
    model.donor(ids)
    model.train(was)
    for h in handles:
        h.remove()
    Hq, Hkv, dh = cfg.n_heads, cfg.n_kv, cfg.head_dim
    for L in range(cfg.n_layers):
        qf = Q[L].view(-1, Hq, dh)                                 # [n_calib, Hq, dh]
        kf = K[L].view(-1, Hkv, dh)                                # [n_calib, Hkv, dh]
        rq = qf.pow(2).mean(-1).sqrt().mean(0)                     # [Hq]  per-head RMS
        rk = kf.pow(2).mean(-1).sqrt().mean(0)                     # [Hkv]
        qkn = model.qknorms[L]
        qkn.q.gain.data.copy_((dh ** 0.25) * rq.to(qkn.q.gain.dtype))
        qkn.k.gain.data.copy_((dh ** 0.25) * rk.to(qkn.k.gain.dtype))
    log(f"[qknorm-init] calibrated {cfg.n_layers} layers' QKNorm gains from the frozen donor's "
        f"own pre-RoPE Q/K RMS on {n_calib} random-token calibration positions")


def build_hba(cfg=None, dtype=None, device=None):
    from .config import HBAConfig
    cfg = cfg or HBAConfig()
    donor, tok = load_donor(dtype=dtype, device=device)
    model = HBAModel(donor, cfg).to(device or DEVICE)
    if getattr(cfg, "qknorm", False):
        init_qknorm_gains(model, cfg)
    return model, tok, cfg
