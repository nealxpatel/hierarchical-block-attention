"""HBA attention: RoPE utilities, hierarchical selection, and the three attention
backends (naive oracle, fused FlexAttention LSE-merge, chunked eval) described in
docs/design.md.

Every layer computes one softmax over the disjoint union of three key/value
regions -- anchor/sink tokens (position-free), a local RoPE window, and top-k
routed content blocks (position-free, selected by the learned SlotSummarizer) --
as specified in docs/design.md. `hba_attention_dense` is the training-time
reference: it materializes full [B,H,n,n] scores and is the permanent correctness
oracle every optimized path is gated against (docs/training-recipe.md,
"Correctness gates"). `hba_attention_fused` computes the identical math as a
log-sum-exp merge of two FlexAttention calls, with no [n,n] tensor ever
materialized -- the throughput training path. `hba_attention_eval` is the
memory-capped chunked path used at arbitrary (including very long) context, with
an optional two-level hierarchical selection.
"""

import os

import torch

from .config import DEVICE, throttle_mps
from .summarizer import grouped_query, slot_block_scores

# ---------------------------------------------------------------- RoPE ---------
def rope_tables(n, head_dim, theta, device, dtype=torch.float32):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(n, device=device).float()
    fr = torch.outer(pos, inv)
    emb = torch.cat((fr, fr), dim=-1)          # [n, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rot_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rope(x, cos, sin):
    # x: [B, n, H, dh]; cos/sin: [n, dh]  (GPT-NeoX / Qwen rotation convention)
    c = cos[None, :, None, :].to(x.dtype)
    s = sin[None, :, None, :].to(x.dtype)
    return x * c + rot_half(x) * s


def yarn_theta(cfg, n):
    """NTK-aware base rescaling for the donor+YaRN comparison baseline at eval time."""
    if n <= cfg.native_ctx:
        return cfg.rope_theta
    s = n / cfg.native_ctx
    D = cfg.head_dim
    return cfg.rope_theta * (s ** (D / (D - 2)))


# ------------------------------------------------------ hierarchical selection -
def build_super(S0, fanout):
    """Mean-pool learned block summaries S0 [B,nb,m,dh] into super-summaries
    [B,ns,m,dh] (per-slot mean over `fanout` children); docs/design.md, "Hierarchy"."""
    B, nb, m, dh = S0.shape
    pad = (-nb) % fanout
    if pad:
        S0 = torch.cat([S0, S0.new_zeros(B, pad, m, dh)], dim=1)
    ns = S0.shape[1] // fanout
    S1 = S0.view(B, ns, fanout, m, dh).sum(2)
    cnt = torch.full((ns,), fanout, dtype=S1.dtype, device=S1.device)
    if pad:
        cnt[-1] = fanout - pad
    return S1 / cnt[None, :, None, None], ns


def hier_select(qn, S0, S1, fanout, beam, kk, cand, scale):
    """Coarse-to-fine two-level beam selection (docs/design.md, "Hierarchy"; causal
    by construction via `cand`). Returns (top_idx[B,c,kk], top_val[B,c,kk],
    n_comparisons). A pick is valid iff top_val is finite; callers mask -inf picks.
    qn:[B,c,dh]  S0:[B,nb,m,dh]  S1:[B,ns,m,dh]  cand:[c,nb] bool."""
    B, c, dh = qn.shape
    nb = S0.shape[1]
    ns = S1.shape[1]
    dev = qn.device
    pad = ns * fanout - nb
    cm = torch.cat([cand, cand.new_zeros(c, pad)], dim=1) if pad else cand
    cand_sup = cm.view(c, ns, fanout).any(-1)                                    # [c,ns]
    ssc = torch.einsum("bcd,bnmd->bcnm", qn, S1).amax(-1) * scale
    ssc = ssc.masked_fill(~cand_sup[None], float("-inf"))
    b_eff = min(beam, ns)
    parents = ssc.topk(b_eff, dim=-1).indices                                    # [B,c,beam]
    ar = torch.arange(fanout, device=dev)
    child = (parents[..., None] * fanout + ar).reshape(B, c, b_eff * fanout)
    in_range = child < nb
    child = child.clamp(0, nb - 1)
    Sg = torch.gather(S0, 1, child.reshape(B, -1)[..., None, None]
                      .expand(B, c * b_eff * fanout, S0.shape[2], dh)
                      ).reshape(B, c, b_eff * fanout, S0.shape[2], dh)
    csc = torch.einsum("bcd,bcjmd->bcjm", qn, Sg).amax(-1) * scale
    cflat = torch.gather(cand[None].expand(B, c, nb), 2, child) & in_range
    csc = csc.masked_fill(~cflat, float("-inf"))
    kk_eff = min(kk, b_eff * fanout)
    top_val, sub = csc.topk(kk_eff, dim=-1)
    top_idx = torch.gather(child, 2, sub)
    return top_idx, top_val, ns + b_eff * fanout


# ------------------------------------------- eval-only NoPE softmax temperature -
# The union softmax dilutes as the candidate crowd grows with context length
# (docs/design.md, "Softmax calibration across candidate counts"). This is the
# log-N temperature "assist, not fix" mitigation described there: EVAL PATH ONLY,
# default 0.0 == strictly OFF (the guarded branch below is never entered; bit-
# identical outputs). When c > 0, the NoPE-region logits (sinks + routed) entering
# the union softmax are scaled by 1/tau with
#     tau = 1 + c * log(max(N_nope / N_cal, 1))
# where N_nope is the per-query count of NoPE keys ELIGIBLE to that query (visible
# sinks + routed-candidate keys, from the same candidate arithmetic as routing) and
# N_cal is the same count for the last query at heal_ctx. Identity for any query
# whose candidate crowd is <= the heal-length crowd (i.e. for all n <= heal_ctx).
# Never used by the train paths; no checkpoint or config dependence.
NOPE_TEMP_C = float(os.environ.get("HBA_NOPE_TEMP_C", "0.0"))


def set_nope_temp_c(c):
    global NOPE_TEMP_C
    NOPE_TEMP_C = float(c)


# ------------------------------------------------------ HBA attention (train) --
def _route_candidates(n, W, nb, Bk, dev):
    """cand[n, nb]: content blocks fully BEFORE the window (excluding sink block 0)
    -- the routed candidates of the disjoint-union attention. Shared by the naive,
    fused, and eval paths so all three route identically given identical inputs."""
    i = torch.arange(n, device=dev)[:, None]
    barr = torch.arange(nb, device=dev)
    return (barr[None, :] >= 1) & ((barr[None, :] + 1) * Bk <= (i - W + 1))       # [n, nb]


def _causal_bucket_masks(n, S, W, nb, Bk, dev):
    """Return (m_sink[n,n], m_win[n,n], cand[n,nb]) for the disjoint-union
    attention. sink: first S keys (NoPE), causal. window: last W keys (RoPE),
    causal. routed candidates: content blocks fully BEFORE the window (NoPE) --
    disjoint from sinks and window."""
    i = torch.arange(n, device=dev)[:, None]
    j = torch.arange(n, device=dev)[None, :]
    m_sink = (j < S) & (j <= i)
    m_win = (j >= S) & (j <= i) & (j > i - W)
    cand = _route_candidates(n, W, nb, Bk, dev)
    return m_sink, m_win, cand


def route_topk(q, k, cfg, summ, cand):
    """Learned top-k routing, shared VERBATIM by the naive and fused train paths so
    both backends make the identical selection given identical inputs (same ops in
    the same order -> bitwise-equal scores -> equal top-k; this is what makes the
    fused/naive agreement gate's strict tolerance meaningful). Routing runs on
    DETACHED q/k -- the gradient paths are disjoint: the LM loss reaches only
    q/k/v/o, the aux-KL loss reaches only the summarizer's probes/proj (docs/
    training-recipe.md, "gradient-isolation rule").
    q:[B,n,Hq,dh]  k:[B,n,Hkv,dh]  cand:[n,nb]  ->  (sel[B,Hkv,n,nb] bool, bsc[B,Hkv,n,nb])."""
    B, n, Hq, dh = q.shape
    Hkv = cfg.n_kv
    Bk = cfg.block
    nb = n // Bk
    kk = min(cfg.k_blocks, nb)
    scale = dh ** -0.5
    qg = grouped_query(q, cfg)                                                    # [B,Hkv,n,dh]
    Sblk = summ.summarize(k.detach(), Bk)                                         # [B,Hkv,nb,m,dh]
    bsc = slot_block_scores(qg.detach(), Sblk, scale)                             # [B,Hkv,n,nb]
    bsc = bsc.masked_fill(~cand[None, None], float("-inf"))
    top = bsc.topk(kk, dim=-1).indices                                            # [B,Hkv,n,kk]
    sel = torch.zeros(B, Hkv, n, nb, dtype=torch.bool, device=q.device)
    sel.scatter_(-1, top, True)
    sel = sel & cand[None, None]
    return sel, bsc


def hba_attention_dense(q, k, v, cos, sin, cfg, summ, equiv=False):
    """TRAINING / dense path (GQA). q:[B,n,Hq,dh]  k,v:[B,n,Hkv,dh]. One softmax
    over the disjoint union {NoPE sinks, RoPE window, NoPE routed top-k blocks};
    block scores from the learned per-KV-head SlotSummarizer; selection is per KV
    head (grouped query), shared by the group. Returns (out[B,n,Hq,dh], aux_kl).

    equiv=True short-circuits to dense UNIFORM-RoPE causal attention (route-
    everything + RoPE-everywhere limit) -- reproduces the donor's own attention
    exactly (docs/training-recipe.md, "Equivalence gate"). aux is 0 there."""
    B, n, Hq, dh = q.shape
    Hkv, G = cfg.n_kv, cfg.G
    scale = dh ** -0.5
    dev = q.device
    qr = apply_rope(q, cos, sin).transpose(1, 2)                                  # [B,Hq,n,dh]
    kr = apply_rope(k, cos, sin).transpose(1, 2).repeat_interleave(G, dim=1)      # [B,Hq,n,dh]
    if equiv:
        i = torch.arange(n, device=dev)[:, None]
        j = torch.arange(n, device=dev)[None, :]
        sc = torch.matmul(qr, kr.transpose(-1, -2)) * scale
        sc = sc.masked_fill((j > i)[None, None], float("-inf"))
        out = torch.matmul(sc.softmax(-1), v.transpose(1, 2).repeat_interleave(G, dim=1))
        return out.transpose(1, 2), q.new_zeros(())

    Bk, S, W, kb = cfg.block, cfg.sinks, cfg.window, cfg.k_blocks
    assert n % Bk == 0, (n, Bk)
    nb = n // Bk
    kk = min(kb, nb)
    qn = q.transpose(1, 2)                                                        # [B,Hq,n,dh]
    kn_kv = k.transpose(1, 2)                                                     # [B,Hkv,n,dh]
    kn = kn_kv.repeat_interleave(G, dim=1)                                        # [B,Hq,n,dh]
    vv = v.transpose(1, 2).repeat_interleave(G, dim=1)
    # ---- routing on DETACHED q/k (grad paths disjoint: LM->qkv/o, aux->probes/proj ONLY) ----
    m_sink, m_win, cand = _causal_bucket_masks(n, S, W, nb, Bk, dev)
    sel, bsc = route_topk(q, k, cfg, summ, cand)                                  # [B,Hkv,n,nb]
    sel_q = sel.repeat_interleave(G, dim=1)                                       # [B,Hq,n,nb]
    m_rout = sel_q.repeat_interleave(Bk, dim=-1)                                  # [B,Hq,n,n]
    s_nope = torch.matmul(qn, kn.transpose(-1, -2)) * scale
    s_rope = torch.matmul(qr, kr.transpose(-1, -2)) * scale
    sc = torch.where(m_win[None, None], s_rope, s_nope)
    sc = sc.masked_fill(~(m_rout | (m_sink | m_win)[None, None]), float("-inf"))
    w = sc.softmax(-1)
    out = torch.matmul(w, vv).transpose(1, 2)                                     # [B,n,Hq,dh]

    # ---- auxiliary KL: distil per-KV-head slot scores toward the true content-block mass ----
    # teacher = group-averaged NoPE content-mass over candidate keys (docs/
    # training-recipe.md, "The summarizer auxiliary loss"). aux_w == 0.0 (stage 3,
    # summarizers frozen): skip the O(n^2) teacher ENTIRELY -- at longer curriculum
    # lengths the full-score teacher is ruinous, and with the summarizers frozen its
    # gradient has nowhere to go anyway. Both backends honor this identically.
    if getattr(cfg, "aux_w", 1.0) == 0.0:
        return out, out.new_zeros(())
    with torch.no_grad():
        cand_key = cand.repeat_interleave(Bk, dim=-1)                            # [n,n]
        t = s_nope.masked_fill(~cand_key[None, None], float("-inf"))
        wt = t.softmax(-1)                                                        # [B,Hq,n,n]
        pmass = wt.view(B, Hq, n, nb, Bk).sum(-1)                                 # [B,Hq,n,nb]
        pstar = pmass.view(B, Hkv, G, n, nb).mean(2)                             # [B,Hkv,n,nb]
        pstar = pstar / pstar.sum(-1, keepdim=True).clamp_min(1e-9)
    vq = cand.any(-1)                                                             # [n] queries w/ cands
    if vq.any():
        bv = bsc[:, :, vq]
        pv = pstar[:, :, vq]
        logp = torch.log_softmax(bv, dim=-1)
        term = torch.where(pv > 0, pv * logp, torch.zeros_like(logp))
        aux = -term.sum(-1).mean()
    else:
        aux = out.new_zeros(())
    return out, aux


# ----------------------------------------- fused train path (FlexAttention) ----
_FLEX_FN = None


def _flex_fn():
    """flex_attention, torch.compile'd on CUDA (uncompiled flex falls back to an
    unfused implementation that materializes the full score matrix -- the exact
    thing this path exists to avoid). Eager on CPU: the fallback is fine there."""
    global _FLEX_FN
    if _FLEX_FN is None:
        from torch.nn.attention.flex_attention import flex_attention
        if DEVICE == "cuda":
            # each distinct (shape, mask_mod, dtype/autocast) combo is a dynamo
            # recompile; the correctness gates alone exercise several shapes and
            # both masks. Past the default cache_size_limit dynamo SILENTLY runs
            # the frame eagerly -- and eager flex is the score-materializing math
            # fallback, exactly what this path must never do. Raise the limit well
            # clear of any realistic gate/training mix.
            torch._dynamo.config.cache_size_limit = max(
                64, torch._dynamo.config.cache_size_limit)
            _FLEX_FN = torch.compile(flex_attention, dynamic=False)
        else:
            _FLEX_FN = flex_attention
    return _FLEX_FN


_WIN_BM_CACHE = {}


def _window_blockmask(n, S, W, dev):
    """BlockMask for region B: the RoPE sliding window (kv>=S, causal, within W).
    Data-independent -> built once per (n, S, W, device) and cached across layers
    and steps."""
    key = (n, S, W, str(dev))
    bm = _WIN_BM_CACHE.get(key)
    if bm is None:
        from torch.nn.attention.flex_attention import create_block_mask

        def wmask(b, h, q_idx, kv_idx):
            return (kv_idx >= S) & (kv_idx <= q_idx) & (kv_idx > q_idx - W)

        bm = create_block_mask(wmask, None, None, n, n, device=dev)
        _WIN_BM_CACHE[key] = bm
    return bm


def _routed_blockmask(sel, cfg, n, dev):
    """BlockMask for region A: NoPE sinks + the per-query selected routed blocks
    (data-dependent -> rebuilt every call). Built DIRECTLY from `sel` at flex's
    128-token block granularity (no [n,n] materialization, no create_block_mask
    grid pass): a kernel block (128q x 128kv) is live iff any of its queries
    selected either of its two 64-token routing blocks; exactness inside live
    blocks comes from the mask_mod, which indexes the captured `sel` tensor (the
    documented document-masking pattern). Selection is per KV head; flex's
    enable_gqa passes the QUERY head index h to mask_mod, so it maps h -> its KV
    group via h // G."""
    from torch.nn.attention.flex_attention import BlockMask
    B, Hkv, n_, nb = sel.shape
    S, Bk, G = cfg.sinks, cfg.block, cfg.G
    QB = KB = 128
    assert KB % Bk == 0 and S <= KB
    nQ = (n + QB - 1) // QB
    nK = (n + KB - 1) // KB
    occ = sel
    padq = nQ * QB - n
    if padq:
        occ = torch.cat([occ, occ.new_zeros(B, Hkv, padq, nb)], dim=2)
    occ = occ.view(B, Hkv, nQ, QB, nb).any(3)                    # [B,Hkv,nQ,nb]
    r = KB // Bk
    padb = nK * r - nb
    if padb:
        occ = torch.cat([occ, occ.new_zeros(B, Hkv, nQ, padb)], dim=3)
    occ = occ.view(B, Hkv, nQ, nK, r).any(-1)                    # [B,Hkv,nQ,nK]
    occ[..., 0] = True                                           # sinks live for every q block
    occ = occ.repeat_interleave(G, dim=1)                        # KV-head selection -> Hq
    kv_num = occ.sum(-1, dtype=torch.int32)
    kv_idx = occ.to(torch.uint8).argsort(dim=-1, descending=True, stable=True).to(torch.int32)

    def amask(b, h, q_idx, kv_idx_):
        sink = (kv_idx_ < S) & (kv_idx_ <= q_idx)
        return sink | sel[b, h // G, q_idx, kv_idx_ // Bk]

    return BlockMask.from_kv_blocks(kv_num, kv_idx, None, None, BLOCK_SIZE=(QB, KB),
                                    mask_mod=amask, seq_lengths=(n, n))


def _teacher_chunk(qc, kt, cand_c, scale, Bk):
    """One query-chunk of the aux teacher: scores -> streaming block-LSE -> masked
    softmax over blocks -> group mean -> renormalize. qc:[B,Hkv,G,c,dh]
    kt:[B,Hkv,1,dh,n]  cand_c:[c,nb]. Kept as a single function so torch.compile
    can fuse the whole post-GEMM chain into a couple of passes over the [.., n]
    score tensor (eager evaluation does several passes and is markedly slower)."""
    B, Hkv, G, c, dh = qc.shape
    n = kt.shape[-1]
    s = (qc @ kt) * scale                                        # [B,Hkv,G,c,n]
    s = s.view(B, Hkv, G, c, n // Bk, Bk)
    m_ = s.amax(-1)
    e = (s - m_.unsqueeze(-1)).exp().sum(-1, dtype=torch.float32)
    blk = m_.float() + e.log()                                   # block-LSE == logsumexp
    blk = blk.masked_fill(~cand_c[None, None, None], float("-inf"))
    p = blk.softmax(-1).mean(2)                                  # [B,Hkv,c,nb]
    return p / p.sum(-1, keepdim=True).clamp_min(1e-9)


_TEACHER_FN = None


def _teacher_fn():
    global _TEACHER_FN
    if _TEACHER_FN is None:
        _TEACHER_FN = torch.compile(_teacher_chunk, dynamic=False) if DEVICE == "cuda" \
            else _teacher_chunk
    return _TEACHER_FN


def _aux_kl_chunked(q, k, bsc, cand, cfg):
    """The SAME aux-KL as the naive path (teacher = group-averaged NoPE content-
    mass over candidate blocks; softmax-over-keys-then-block-sum == softmax over
    blockwise LSEs, identical math), but computed in query chunks under no_grad so
    no [B,Hq,n,n] tensor is ever materialized. Teacher math in fp32 (autocast-
    safe); grads reach ONLY bsc (the summarizer)."""
    B, n, Hq, dh = q.shape
    Hkv, G = cfg.n_kv, cfg.G
    Bk = cfg.block
    nb = n // Bk
    scale = dh ** -0.5
    with torch.no_grad():
        # bmm-friendly layout: [B,Hkv,G,c,dh] @ [B,Hkv,1,dh,n] broadcasts the KV
        # head over its G query heads via stride-0 batched GEMM (no repeat_interleave copy).
        qn = q.view(B, n, Hkv, G, dh).permute(0, 2, 3, 1, 4)     # heads are group-contiguous
        kt = k.permute(0, 2, 3, 1).unsqueeze(2).contiguous()     # [B,Hkv,1,dh,n]
        chunk = max(Bk, int(cfg.mem_elem_cap / max(1, B * Hq * n)))
        chunk = (chunk // Bk) * Bk        # keep chunk shapes stable (compiled-teacher variants)
        pstar = torch.empty(B, Hkv, n, nb, device=q.device, dtype=torch.float32)
        tfn = _teacher_fn()
        for cs in range(0, n, chunk):
            ce = min(n, cs + chunk)
            pstar[:, :, cs:ce] = tfn(qn[:, :, :, cs:ce].contiguous(), kt,
                                     cand[cs:ce], scale, Bk)
    vq = cand.any(-1)                                            # [n] queries with candidates
    if vq.any():
        bv = bsc[:, :, vq]
        pv = pstar[:, :, vq].to(bsc.dtype)
        logp = torch.log_softmax(bv, dim=-1)
        term = torch.where(pv > 0, pv * logp, torch.zeros_like(logp))
        return -term.sum(-1).mean()
    return q.new_zeros(())


def hba_attention_fused(q, k, v, cos, sin, cfg, summ):
    """TRAINING / FUSED path (GQA, FlexAttention). Same math as
    hba_attention_dense -- ONE softmax over the disjoint union {NoPE sinks, NoPE
    routed top-k blocks, RoPE window} -- obtained as the log-sum-exp merge of two
    fused attentions over the disjoint regions (docs/design.md, "The three
    components"):

        A (NoPE q/k):  sinks + selected routed blocks, causal     -> (out_A, lse_A)
        B (RoPE q/k):  sliding window                             -> (out_B, lse_B)
        out = (e^{lse_A} out_A + e^{lse_B} out_B) / (e^{lse_A} + e^{lse_B})

    which is exactly the disjoint-union softmax. Autograd flows through both flex
    calls and the merge. No [n,n] score tensor is materialized -- the naive path's
    memory-bandwidth bottleneck. Merge + lse math in fp32 regardless of autocast;
    output cast to q.dtype. Region A is never empty (sink 0 is causally visible to
    every query) so lse_A is finite; region B IS empty for queries i<S (window
    excludes kv<S) -> flex returns out_B=0 rows with lse_B=-inf, and
    exp(-inf - m)=0 removes them from the merge cleanly.
    Returns (out[B,n,Hq,dh], aux_kl) -- same contract as the naive path."""
    from torch.nn.attention.flex_attention import AuxRequest
    B, n, Hq, dh = q.shape
    Bk, S, W = cfg.block, cfg.sinks, cfg.window
    scale = dh ** -0.5
    dev = q.device
    assert n % Bk == 0, (n, Bk)
    nb = n // Bk
    cand = _route_candidates(n, W, nb, Bk, dev)
    sel, bsc = route_topk(q, k, cfg, summ, cand)                 # identical selection to naive
    flex = _flex_fn()
    bmA = _routed_blockmask(sel, cfg, n, dev)                    # rebuilt each call (data-dep)
    bmB = _window_blockmask(n, S, W, dev)                        # cached (data-indep)
    qA = q.transpose(1, 2)                                       # [B,Hq,n,dh]  NoPE
    kA = k.transpose(1, 2)                                       # [B,Hkv,n,dh] (GQA: no expand)
    vA = v.transpose(1, 2)
    outA, auxA = flex(qA, kA, vA, block_mask=bmA, scale=scale, enable_gqa=True,
                      return_aux=AuxRequest(lse=True))
    qB = apply_rope(q, cos, sin).transpose(1, 2)                 # RoPE
    kB = apply_rope(k, cos, sin).transpose(1, 2)
    outB, auxB = flex(qB, kB, vA, block_mask=bmB, scale=scale, enable_gqa=True,
                      return_aux=AuxRequest(lse=True))
    lseA, lseB = auxA.lse.float(), auxB.lse.float()              # [B,Hq,n] fp32
    m = torch.maximum(lseA, lseB).detach()   # cancels analytically; detach = exact cancellation
    wA = (lseA - m).exp()[..., None]
    wB = (lseB - m).exp()[..., None]
    out = (wA * outA.float() + wB * outB.float()) / (wA + wB)
    out = out.to(q.dtype).transpose(1, 2)                        # [B,n,Hq,dh]
    # aux_w == 0.0 (stage 3): skip the chunked O(n^2) teacher (see hba_attention_dense note).
    aux = (q.new_zeros(()) if getattr(cfg, "aux_w", 1.0) == 0.0
           else _aux_kl_chunked(q, k, bsc, cand, cfg))
    return out, aux


def hba_attention_eval(q, k, v, cos, sin, cfg, summ, cap, hier=False, equiv=False):
    """EVAL path (GQA), memory-capped chunked gather. Selection per KV head
    (grouped query), shared by the group's G query heads. hier=True uses the
    two-level hierarchy (docs/design.md, "Hierarchy") for the routed selection.
    Mathematically equal to hba_attention_dense wherever the selections agree
    (path-equivalence gated -- see gates.py)."""
    B, n, Hq, dh = q.shape
    Hkv, G = cfg.n_kv, cfg.G
    Bk, S, W, kb = cfg.block, cfg.sinks, cfg.window, cfg.k_blocks
    scale = dh ** -0.5
    dev = q.device
    q_rope = apply_rope(q, cos, sin)
    k_rope = apply_rope(k, cos, sin)
    if equiv:
        out = torch.empty_like(q)
        qr = q_rope.transpose(1, 2)
        kr = k_rope.transpose(1, 2).repeat_interleave(G, dim=1)
        vv = v.transpose(1, 2).repeat_interleave(G, dim=1)
        chunk = max(64, int(cap / max(1, B * Hq * n)))
        for it, cs in enumerate(range(0, n, chunk)):
            ce = min(n, cs + chunk)
            sc = torch.matmul(qr[:, :, cs:ce], kr[:, :, :ce].transpose(-1, -2)) * scale
            qp = torch.arange(cs, ce, device=dev)[:, None]
            kp = torch.arange(ce, device=dev)[None, :]
            sc = sc.masked_fill((kp > qp)[None, None], float("-inf"))
            oc = torch.matmul(sc.softmax(-1), vv[:, :, :ce]).transpose(1, 2)      # [B,c,Hq,dh]
            out[:, cs:ce] = oc
            throttle_mps(it)
        return out
    assert n % Bk == 0, (n, Bk)
    nb = n // Bk
    kk = min(kb, nb)
    out = torch.empty_like(q)
    M = S + W + kk * Bk
    chunk = max(Bk, int(cap / max(1, B * M * dh)))
    barr = torch.arange(nb, device=dev)
    woff = torch.arange(-W + 1, 1, device=dev)
    bidx = torch.arange(B, device=dev)[:, None, None]
    Sblk = summ.summarize(k, Bk)                                                  # [B,Hkv,nb,m,dh]
    kblk = k.view(B, nb, Bk, Hkv, dh)
    vblk = v.view(B, nb, Bk, Hkv, dh)
    temp_c = NOPE_TEMP_C
    if temp_c != 0.0:
        # calibration count from the code's OWN candidate arithmetic at heal_ctx:
        # NoPE keys eligible to the LAST heal-length query = sinks + candidate
        # blocks * Bk (docs/design.md, "Softmax calibration").
        _hc = cfg.heal_ctx
        _cal_blocks = int(_route_candidates(_hc, W, _hc // Bk, Bk, dev)[-1].sum())
        N_cal = float(S + _cal_blocks * Bk)
    it = 0
    for g in range(Hkv):                            # selection is per KV head (shared by group)
        Sh = Sblk[:, g]                             # [B,nb,m,dh]
        S1h, ns = build_super(Sh, cfg.fanout) if hier else (None, None)
        kn_g = k[:, :, g]; kr_g = k_rope[:, :, g]; vv_g = v[:, :, g]             # [B,n,dh]
        knb = kblk[:, :, :, g]; vnb = vblk[:, :, :, g]                            # [B,nb,Bk,dh]
        sink_k = kn_g[:, :S]; sink_v = vv_g[:, :S]
        for hh in range(G):                         # each query head in the group
            h = g * G + hh
            qn = q[:, :, h]; qr = q_rope[:, :, h]
            qg = grouped_query(q, cfg)[:, g]        # [B,n,dh] grouped routing query (per KV head)
            for cs in range(0, n, chunk):
                ce = min(n, cs + chunk); c = ce - cs
                i = torch.arange(cs, ce, device=dev)
                # sinks (NoPE)
                ssc = torch.einsum("bcd,bsd->bcs", qn[:, cs:ce], sink_k) * scale
                ssc = ssc.masked_fill((torch.arange(S, device=dev)[None, :] > i[:, None])[None],
                                      float("-inf"))
                # window (RoPE)
                widx = i[:, None] + woff[None, :]
                valid_w = (widx >= S) & (widx <= i[:, None])
                widx_c = widx.clamp(0, n - 1).reshape(-1)
                wk = kr_g[:, widx_c].reshape(B, c, W, dh)
                wv = vv_g[:, widx_c].reshape(B, c, W, dh)
                wsc = torch.einsum("bcd,bcwd->bcw", qr[:, cs:ce], wk) * scale
                wsc = wsc.masked_fill(~valid_w[None], float("-inf"))
                # routed (NoPE) via learned slot scores on the GROUPED query
                cand = (barr[None, :] >= 1) & ((barr[None, :] + 1) * Bk <= (i[:, None] - W + 1))
                if hier:
                    idx, vals, _ = hier_select(qg[:, cs:ce], Sh, S1h, cfg.fanout, cfg.beam, kk,
                                               cand, scale)
                    bad = torch.isinf(vals)
                else:
                    bsc = torch.einsum("bcd,bnmd->bcnm", qg[:, cs:ce], Sh).amax(-1) * scale
                    bsc = bsc.masked_fill(~cand[None], float("-inf"))
                    tk = bsc.topk(kk, dim=-1)
                    idx, bad = tk.indices, torch.isinf(tk.values)
                rk = knb[bidx, idx].reshape(B, c, kk * Bk, dh)
                rv = vnb[bidx, idx].reshape(B, c, kk * Bk, dh)
                rsc = torch.einsum("bcd,bcmd->bcm", qn[:, cs:ce], rk) * scale
                rvalid = (~bad)[..., None].expand(B, c, kk, Bk).reshape(B, c, kk * Bk)
                rsc = rsc.masked_fill(~rvalid, float("-inf"))
                if temp_c != 0.0:
                    # log-length NoPE temperature: sharpen sinks+routed logits when
                    # this query's eligible NoPE crowd exceeds the heal-length
                    # calibration crowd. tau = 1 exactly for all queries at
                    # n <= heal_ctx (ratio clamped to 1); window (RoPE) logits
                    # untouched; -inf stays -inf under the positive scale.
                    n_nope = torch.minimum(i + 1, i.new_tensor(S)) + cand.sum(-1) * Bk    # [c]
                    tau = 1.0 + temp_c * (n_nope.to(torch.float32) / N_cal).clamp(min=1.0).log()
                    inv_tau = (1.0 / tau).to(ssc.dtype)[None, :, None]
                    ssc = ssc * inv_tau
                    rsc = rsc * inv_tau
                allsc = torch.cat([ssc, wsc, rsc], dim=-1)
                allsc = allsc - allsc.max(-1, keepdim=True).values.detach()
                wgt = allsc.softmax(-1)
                ws, ww, wr = wgt.split([S, W, kk * Bk], dim=-1)
                oc = (torch.einsum("bcs,bsd->bcd", ws, sink_v)
                      + torch.einsum("bcw,bcwd->bcd", ww, wv)
                      + torch.einsum("bcm,bcmd->bcd", wr, rv))
                out[:, cs:ce, h] = oc
                throttle_mps(it); it += 1
    return out
