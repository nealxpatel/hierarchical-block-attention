"""Learned block summarizers (docs/design.md, "Slot summarizers").

Mean-pooling a block's keys into one vector is a weak router (README principle 3).
`SlotSummarizer` instead gives each (layer, KV head) `m` learned probe vectors that
cross-attend into a block's pre-RoPE (NoPE) keys, producing `m` slot summaries per
block; a block's routing score for a query is the max over its slots. Position-free
by construction (it never sees RoPE) -- this is the mechanism the length-
generalization thesis rests on.
"""

import torch
import torch.nn as nn


class SlotSummarizer(nn.Module):
    """Per attention layer, per KV head: `m` learned probe vectors attention-pool
    each block's pre-RoPE (NoPE) keys into `m` slot summaries; a block's routing
    score for query `q` is max_over_slots(q_nope . slot). Selection is per KV head
    (GQA-grouped), shared by that head's G query heads (docs/design.md)."""

    def __init__(self, cfg):
        super().__init__()
        Hkv, dh, m = cfg.n_kv, cfg.head_dim, cfg.slots
        self.m = m
        self.probes = nn.Parameter(torch.randn(Hkv, m, dh) * dh ** -0.5)          # [Hkv,m,dh]
        # proj init = per-head IDENTITY + tiny noise. Load-bearing: a small-std proj
        # shrinks slot summaries toward zero, flattens block scores, and starves the
        # aux gradient. Identity makes the slot summary a magnitude-preserving pooled
        # key at init, so routing starts near mean-pool quality and the aux loss then
        # differentiates the slots (docs/design.md, "Slot summarizers").
        self.proj = nn.Parameter(torch.eye(dh).unsqueeze(0).repeat(Hkv, 1, 1)
                                 + torch.randn(Hkv, dh, dh) * 0.02)                # [Hkv,dh,dh]

    def summarize(self, k, Bk):
        """k: [B,n,Hkv,dh] pre-RoPE keys -> block summaries S: [B,Hkv,nb,m,dh]."""
        B, n, Hkv, dh = k.shape
        nb = n // Bk
        kb = k.view(B, nb, Bk, Hkv, dh)
        att = torch.einsum("hmd,bckhd->bhcmk", self.probes, kb) * dh ** -0.5       # [B,Hkv,nb,m,Bk]
        att = att.softmax(-1)
        S = torch.einsum("bhcmk,bckhd->bhcmd", att, kb)                            # [B,Hkv,nb,m,dh]
        return torch.einsum("bhcmd,hde->bhcme", S, self.proj)


def slot_block_scores(qn, S, scale):
    """qn: [B,Hkv,n,dh] NoPE grouped query, S: [B,Hkv,nb,m,dh] -> block scores [B,Hkv,n,nb]."""
    return torch.einsum("bhid,bhcmd->bhicm", qn, S).amax(-1) * scale


def grouped_query(q, cfg):
    """Sum the G query heads within each KV group -> one routing query per KV head.
    q: [B,n,Hq,dh] -> [B,Hkv,n,dh]."""
    B, n, Hq, dh = q.shape
    return q.view(B, n, cfg.n_kv, cfg.G, dh).sum(3).transpose(1, 2)
