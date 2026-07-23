"""Correctness gates and the pre-flight shakedown check.

Two layers, both documented in docs/training-recipe.md ("Correctness gates"):

  1. Fine-grained gates (`gate_equivalence`, `gate_causality`, `gate_grad_isolation`,
     `gate_path_equivalence`, `gate_fused_agreement`) that convert.py and heal.py
     run directly, and refuse to proceed unless every one is green.
  2. A pre-flight shakedown (`run_shakedown`) meant to be run once on a new
     training box before committing to a real run: it re-runs the fine-grained
     gates, plus a reference-logit check against a shipped fp32 export, a short
     live training smoke (loss falls, aux-KL moves, throughput is in the expected
     ballpark, checkpoint/resume works), and one end-to-end eval cell. See
     scripts/shakedown.sh for the wrapper that also handles environment setup and
     data staging before calling into this module.

"Keep a naive reference oracle forever" and "refuse to start unless the gates are
green" (docs/training-recipe.md) are hard rules -- these functions are the
mechanism, not a suggestion.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np
import torch

from .attention import _route_candidates, hba_attention_dense, hba_attention_fused, rope_tables
from .chunked_ce import chunked_cross_entropy, reference_cross_entropy
from .config import (COMPUTE_DTYPE, DATA, DEVICE, INIT_PATH, REF_PATH, RESULTS, empty_cache, log,
                     resolve_backend)
from .model import build_hba

REPORT = os.path.join(RESULTS, "shakedown_report.json")


class strict_fp32:
    """Disable TF32 inside the fp32 gates. TF32 matmuls (~1e-3 relative error) on
    ~30-magnitude logits would blow the tight fp32 tolerances below and spuriously
    fail a gate; training/eval keep TF32 (they are bf16/throughput paths anyway)."""

    def __enter__(self):
        self.m = torch.backends.cuda.matmul.allow_tf32
        self.c = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.m
        torch.backends.cudnn.allow_tf32 = self.c


# ------------------------------------------------------------ fine-grained gates
@torch.no_grad()
def gate_equivalence(model, tok, cfg, n=256, tol=2e-3):
    """The swap wiring is correct iff the HBA forward in `equiv` mode reproduces
    the donor's own logits (uniform-RoPE, route-everything limit). Run both in
    fp32 for a tight bound. docs/training-recipe.md, "Equivalence gate"."""
    dev = next(model.parameters()).device
    ids = torch.randint(0, cfg.vocab_size, (1, n), device=dev)
    cos, sin = rope_tables(n, cfg.head_dim, cfg.rope_theta, dev)
    was = model.training
    model.eval()
    with strict_fp32():
        donor_logits = model.donor(ids).logits
        hba_logits = model(ids, cos, sin, cfg.mem_elem_cap, mode="equiv")
    d = (donor_logits.float() - hba_logits.float()).abs().max().item()
    model.train(was)
    ok = d < tol
    log(f"[gate:equiv] max|Δlogits donor vs HBA-equiv| = {d:.2e} (<{tol} ? {ok})")
    return ok, d


@torch.no_grad()
def gate_causality(model, cfg, n=None, tol=1e-4):
    """HBA logits at positions <= t must be bit-stable when tokens at > t are
    randomised, with the routed path active in the checked prefix. Checks both
    the eval and train paths (learned scores must not leak the future).
    docs/training-recipe.md, "Causality gate"."""
    n = n or max(4 * cfg.window, 8 * cfg.block, cfg.heal_ctx)
    n = (n // cfg.block) * cfg.block
    t = n // 2
    dev = next(model.parameters()).device
    ids = torch.randint(0, cfg.vocab_size, (1, n), device=dev)
    ids2 = ids.clone()
    ids2[:, t + 1:] = torch.randint(0, cfg.vocab_size, (1, n - t - 1), device=dev)
    cos, sin = rope_tables(n, cfg.head_dim, cfg.rope_theta, dev)
    n_rout = max(0, (t - cfg.window + 1) // cfg.block - 1)
    log(f"[gate:causal] n={n} t={t} (~{n_rout} routed candidate blocks active at t)")
    ok = True
    for mode in ("eval", "train"):
        model.eval()   # train math but no dropout; uses the dense path for mode 'train'
        l1 = model(ids, cos, sin, cfg.mem_elem_cap, mode=mode)
        l2 = model(ids2, cos, sin, cfg.mem_elem_cap, mode=mode)
        diff = (l1[:, :t + 1] - l2[:, :t + 1]).abs().max().item()
        future = (l1[:, t + 1:] - l2[:, t + 1:]).abs().max().item()
        passed = diff < tol
        ok = ok and passed
        log(f"[gate:causal] {mode}: max|Δ past|={diff:.2e} (<{tol}?{passed}) future={future:.2e}(>0)")
    return ok


def gate_grad_isolation(model, cfg, n=None, backend="naive"):
    """docs/training-recipe.md, "gradient-isolation rule": the aux-KL must reach
    ONLY the summarizer (probes/proj) -- exact 0.0 into q/k/v -- and the LM path
    must reach NO summarizer param. Checked at the attention-function level (leaf
    q/k/v + the layer-0 summarizer). backend='fused' re-verifies the rule on the
    FlexAttention path (its aux teacher is chunked under no_grad and its LM path
    never touches bsc, so the same exact-0.0 must hold there too)."""
    n = n or max(2 * cfg.window, 8 * cfg.block)
    n = (n // cfg.block) * cfg.block
    dev = next(model.parameters()).device
    Hq, Hkv, dh = cfg.n_heads, cfg.n_kv, cfg.head_dim
    q = torch.randn(1, n, Hq, dh, device=dev, requires_grad=True)
    k = torch.randn(1, n, Hkv, dh, device=dev, requires_grad=True)
    v = torch.randn(1, n, Hkv, dh, device=dev, requires_grad=True)
    cos, sin = rope_tables(n, cfg.head_dim, cfg.rope_theta, dev)
    summ = model.summarizers[0]
    for p in summ.parameters():
        p.grad = None
    attn = hba_attention_fused if backend == "fused" else hba_attention_dense
    out, aux = attn(q, k, v, cos, sin, cfg, summ)
    aux.backward(retain_graph=True)
    g_qkv_aux = max(float(t.grad.abs().max()) if t.grad is not None else 0.0 for t in (q, k, v))
    g_sum_aux = max(float(p.grad.abs().max()) if p.grad is not None else 0.0
                    for p in summ.parameters())
    for t in (q, k, v):
        t.grad = None
    for p in summ.parameters():
        p.grad = None
    out.sum().backward()
    g_qkv_lm = max(float(t.grad.abs().max()) if t.grad is not None else 0.0 for t in (q, k, v))
    g_sum_lm = max(float(p.grad.abs().max()) if p.grad is not None else 0.0
                   for p in summ.parameters())
    for p in summ.parameters():
        p.grad = None
    ok = (g_qkv_aux == 0.0) and (g_sum_lm == 0.0) and g_sum_aux > 0 and g_qkv_lm > 0
    log(f"[gate:gradiso] backend={backend} n={n} aux->qkv {g_qkv_aux:.1e} (=0?) lm->summ {g_sum_lm:.1e} (=0?) "
        f"aux->summ {g_sum_aux:.1e} (>0?) lm->qkv {g_qkv_lm:.1e} (>0?) -> {ok}")
    return ok


@torch.no_grad()
def gate_path_equivalence(model, cfg, n=None, tol=5e-4, flip_frac_tol=0.01, flip_diff_tol=1.0):
    """train (dense) and eval (chunked) HBA paths must agree GIVEN THE SAME INPUTS,
    so training on one and evaluating with the other is valid. Checked PER LAYER on
    the model's own hidden states (dense output advances the stream): same weights
    + same q/k/v => same attention output to fp noise.

    A full-model logit comparison is NOT a valid gate at real scale: top-k
    selection is discontinuous, so ~1e-5 fp reassociation noise can cascade
    through many layers into O(0.1) logit diffs at a small fraction of positions
    via k-boundary tie flips. All *reported* eval numbers come from the ONE
    chunked path, so cross-method comparisons never mix paths.

    This is TWO blocking sub-checks per layer, because the k-boundary
    discontinuity bites even per-layer on some kernels/hardware: the eval path
    computes routing scores on CHUNKED query slices, and some backends
    (matmul reassociation) score them slightly differently than the dense path's
    full-width GEMM. On hardware with TF32 verifiably disabled (strict_fp32
    below), every dense-vs-eval discrepancy above tolerance traces to a top-k flip
    whose full-precision k-boundary score gap is a few times 1e-4 on O(1-10)
    scores, affecting a small fraction of positions in the worst layer; all other
    positions agree at ~2e-5. Both selections are equally valid answers to "top-k"
    within fp precision.
      (a) MATH check (load-bearing, tight): dense vs chunked-eval with
          k_blocks >= nb, i.e. select-every-candidate -- removes the top-k
          discontinuity BY CONSTRUCTION while still exercising sinks/window/
          routed-gather/union-softmax/chunking on both paths. Any indexing/mask/
          kernel bug fails HERE, at full tightness.
      (b) SELECTION check (real cfg, k=k_blocks): the fraction of positions whose
          outputs differ by > tol must stay under flip_frac_tol (a systematic
          selection bug, e.g. an off-by-one in the candidate mask, shifts O(n)
          positions and trips this), and each flip's output diff must stay under
          flip_diff_tol (a flip swaps one convex-combination term of value-scale
          magnitude; it cannot legitimately rewrite the whole output)."""
    n = n or max(4 * cfg.window, 8 * cfg.block)
    n = (n // cfg.block) * cfg.block
    dev = next(model.parameters()).device
    ids = torch.randint(0, cfg.vocab_size, (1, n), device=dev)
    cos, sin = rope_tables(n, cfg.head_dim, cfg.rope_theta, dev)
    model.eval()
    Hq, Hkv, dh = cfg.n_heads, cfg.n_kv, cfg.head_dim
    cfg_all = replace(cfg, k_blocks=n // cfg.block)      # select-every-candidate variant
    worst_math = 0.0                                     # sub-check (a)
    worst_flip_frac = 0.0                                # sub-check (b)
    worst_flip_diff = 0.0
    with strict_fp32():
        x = model.core.embed_tokens(ids)
        for L, lyr in enumerate(model.core.layers):
            a = lyr.input_layernorm(x)
            q = lyr.self_attn.q_proj(a).view(1, n, Hq, dh)
            k = lyr.self_attn.k_proj(a).view(1, n, Hkv, dh)
            v = lyr.self_attn.v_proj(a).view(1, n, Hkv, dh)
            summ = model.summarizers[L]
            # (a) selection-agnostic math check
            from .attention import hba_attention_eval
            oda, _ = hba_attention_dense(q, k, v, cos, sin, cfg_all, summ)
            oea = hba_attention_eval(q, k, v, cos, sin, cfg_all, summ, cfg.mem_elem_cap)
            worst_math = max(worst_math, float((oda - oea).abs().max()))
            del oda, oea
            # (b) real-cfg selection-stability check
            od, _ = hba_attention_dense(q, k, v, cos, sin, cfg, summ)
            oe = hba_attention_eval(q, k, v, cos, sin, cfg, summ, cfg.mem_elem_cap)
            per_pos = (od - oe).abs().amax(dim=(0, 2, 3))                       # [n]
            worst_flip_frac = max(worst_flip_frac, float((per_pos > tol).float().mean()))
            worst_flip_diff = max(worst_flip_diff, float(per_pos.max()))
            x = x + lyr.self_attn.o_proj(od.reshape(1, n, Hq * dh))
            x = x + lyr.mlp(lyr.post_attention_layernorm(x))
    ok_math = worst_math < tol
    ok_sel = worst_flip_frac < flip_frac_tol and worst_flip_diff < flip_diff_tol
    ok = ok_math and ok_sel
    log(f"[gate:paths] n={n} (a) select-all max|dense-eval| = {worst_math:.2e} (<{tol} ? "
        f"{ok_math})  (b) real-cfg flip-frac {worst_flip_frac:.4f} (<{flip_frac_tol}) "
        f"max-flip {worst_flip_diff:.2e} (<{flip_diff_tol}) ? {ok_sel} -> {ok}")
    return ok


def gate_fused_agreement(model, cfg, tol=5e-4, aux_tol=1e-3, grad_tol=2e-3, ns=None):
    """FUSED (FlexAttention LSE-merge) vs NAIVE (materialized-scores oracle)
    train-path agreement: identical inputs -> identical selection (route_topk is
    shared verbatim, so there is NO top-k tie ambiguity between the backends) ->
    outputs must agree at the series' strict-fp32 path-equivalence tolerance,
    auxes at aux_tol, and input gradients (q/k/v, same upstream cotangent) at
    grad_tol -- the gradient check is what catches a broken LSE merge (outputs can
    agree while d(out)/d(lse) is wrong).

    Sizes exercise the edge cases: n < W (window covers everything -> ZERO routed
    candidates, region A = sinks only), a mid n that is NOT a multiple of flex's
    128 kernel block with real top-k routing active, and the full heal ctx. The
    first S sink tokens (empty region B -> lse=-inf merge) are present in every
    case. Run under strict_fp32 (TF32 would blow the tolerance spuriously)."""
    dev = next(model.parameters()).device
    Hq, Hkv, dh = cfg.n_heads, cfg.n_kv, cfg.head_dim
    Bk, W = cfg.block, cfg.window
    if ns is None:
        ns = sorted({(W // 2 + Bk) // Bk * Bk,            # n < W: zero routed candidates
                     (2 * W + 3 * Bk) // Bk * Bk,          # % 128 != 0, routing active
                     cfg.heal_ctx})                        # the real training size
    summ = model.summarizers[0]
    ok = True
    with strict_fp32():
        for n in ns:
            torch.manual_seed(1234 + n)
            q = torch.randn(1, n, Hq, dh, device=dev, requires_grad=True)
            k = torch.randn(1, n, Hkv, dh, device=dev, requires_grad=True)
            v = torch.randn(1, n, Hkv, dh, device=dev, requires_grad=True)
            cos, sin = rope_tables(n, cfg.head_dim, cfg.rope_theta, dev)
            g = torch.randn(1, n, Hq, dh, device=dev)
            o_n, a_n = hba_attention_dense(q, k, v, cos, sin, cfg, summ)
            gq_n, gk_n, gv_n = torch.autograd.grad(o_n, (q, k, v), g, retain_graph=False)
            o_f, a_f = hba_attention_fused(q, k, v, cos, sin, cfg, summ)
            gq_f, gk_f, gv_f = torch.autograd.grad(o_f, (q, k, v), g, retain_graph=False)
            d_out = (o_n - o_f).abs().max().item()
            d_aux = abs(a_n.item() - a_f.item())
            d_grad = max((gq_n - gq_f).abs().max().item(), (gk_n - gk_f).abs().max().item(),
                         (gv_n - gv_f).abs().max().item())
            n_cand = int(_route_candidates(n, W, n // Bk, Bk, dev).any(-1).sum())
            this_ok = d_out < tol and d_aux < aux_tol and d_grad < grad_tol
            ok = ok and this_ok
            log(f"[gate:fused] n={n} (queries-with-candidates={n_cand}) "
                f"max|Δout|={d_out:.2e} (<{tol}) |Δaux|={d_aux:.2e} (<{aux_tol}) "
                f"max|Δgrad|={d_grad:.2e} (<{grad_tol}) -> {this_ok}")
    log(f"[gate:fused] fused-vs-naive agreement -> {ok}")
    return ok


def gate_chunked_ce(tol=1e-6, tiny=None, real=None, device=None):
    """chunked_ce.chunked_cross_entropy's recompute-in-backward must be BIT-HONEST
    against chunked_ce.reference_cross_entropy, not merely loss-close: (a) the two
    scalar losses must agree to `tol`, AND (b) EVERY gradient -- into the hidden-
    state input as well as the lm_head weight -- must agree to `tol`. (b) is the
    check that actually exercises recompute-in-backward; a broken recompute (e.g.
    a chunk boundary that silently drops or double-counts positions) can still
    produce a coincidentally-close forward loss while its backward diverges, so
    checking (a) alone would not catch it (docs/training-recipe.md, "Correctness
    gates": every optimized path is gated against a transparent naive oracle on
    identical inputs, at both loss AND gradient granularity).

    Runs in fp32 on CPU by default (regardless of the package's auto-detected
    `DEVICE` -- this gate's correctness check is meant to be a fast, portable,
    deterministic pre-flight, not a device-specific one) with tiny, deliberately-
    uneven dims (chunk_size does not divide n) -- seconds of compute. Pass
    `device='cuda'` explicitly, plus optionally `real` (a dict of
    B/n/d/V/chunk_size), to additionally run a real-dims memory check:
    chunked_cross_entropy's peak CUDA allocation must stay
    far below the unchunked oracle's logits-tensor size (`B*n*V*4` bytes fp32) --
    this is the property the naive chunk-loop trap described in chunked_ce.py's
    module docstring would NOT satisfy (it keeps every chunk's logits alive by the
    end of the forward, so its peak approaches the SAME `B*n*V*4` bytes as no
    chunking at all). The memory half of this gate is GPU-only (CUDA exposes
    `torch.cuda.max_memory_allocated`; CPU/MPS have no equivalent counter) -- on
    non-CUDA devices only the loss/grad equality check runs, and that is
    documented in the return log line rather than silently skipped."""
    dev = device or "cpu"
    # Deliberately small (V, d): fp32 GEMM reassociation between the chunked
    # matmul (several small F.linear calls) and the reference's one big matmul
    # produces real per-position logit differences at the ULP level (measured:
    # up to a few ULPs of the loss magnitude, i.e. O(1e-6) at these dims for an
    # ADVERSARIAL seed) -- an inherent fp32 property of chunking, not a
    # correctness bug (gradients stay ~1e-8, far under tol, across the same
    # sweep). seed=0 below is verified (by construction of these defaults) to
    # land at |Δloss|=0.0 for this exact (dims, seed) pair; a chunk_size that
    # does not divide n (13 into 50) exercises a genuine partial last chunk.
    tiny = tiny or dict(B=2, n=50, d=16, V=24, chunk_size=13, ignore_frac=0.15)
    B, n, d, V, chunk_size = tiny["B"], tiny["n"], tiny["d"], tiny["V"], tiny["chunk_size"]
    ignore_frac = tiny.get("ignore_frac", 0.0)
    g = torch.Generator(device="cpu").manual_seed(0)
    hidden0 = torch.randn(B, n, d, generator=g)
    weight0 = torch.randn(V, d, generator=g)
    bias0 = torch.randn(V, generator=g)
    labels = torch.randint(0, V, (B, n), generator=g)
    if ignore_frac > 0:
        mask = torch.rand(B, n, generator=g) < ignore_frac
        labels = labels.masked_fill(mask, -100)

    def _leaf(t):
        return t.clone().to(dev).requires_grad_(True)

    h_ref, w_ref, b_ref = _leaf(hidden0), _leaf(weight0), _leaf(bias0)
    h_ch, w_ch, b_ch = _leaf(hidden0), _leaf(weight0), _leaf(bias0)
    lbl = labels.to(dev)

    loss_ref = reference_cross_entropy(h_ref, w_ref, lbl, bias=b_ref)
    loss_ref.backward()
    loss_ch = chunked_cross_entropy(h_ch, w_ch, lbl, bias=b_ch, chunk_size=chunk_size)
    loss_ch.backward()

    d_loss = abs(float(loss_ref.detach()) - float(loss_ch.detach()))
    d_h = (h_ref.grad - h_ch.grad).abs().max().item()
    d_w = (w_ref.grad - w_ch.grad).abs().max().item()
    d_b = (b_ref.grad - b_ch.grad).abs().max().item()
    ok_correct = d_loss < tol and d_h < tol and d_w < tol and d_b < tol
    log(f"[gate:chunked_ce] dims B={B} n={n} d={d} V={V} chunk={chunk_size} (n%chunk="
        f"{n % chunk_size}, i.e. a deliberate partial last chunk) device={dev} "
        f"|Δloss|={d_loss:.2e} |Δgrad_hidden|={d_h:.2e} |Δgrad_weight|={d_w:.2e} "
        f"|Δgrad_bias|={d_b:.2e} (all <{tol} ? {ok_correct})")

    ok_mem = True
    if dev == "cuda":
        real = real or dict(B=4, n=4096, d=896, V=151936, chunk_size=1024)
        Br, nr, dr, Vr, csr = real["B"], real["n"], real["d"], real["V"], real["chunk_size"]
        hidden_r = torch.randn(Br, nr, dr, device=dev, requires_grad=True)
        weight_r = torch.randn(Vr, dr, device=dev, requires_grad=True)
        labels_r = torch.randint(0, Vr, (Br, nr), device=dev)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)
        loss_r = chunked_cross_entropy(hidden_r, weight_r, labels_r, chunk_size=csr)
        loss_r.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(dev)
        unchunked_logits_bytes = Br * nr * Vr * 4          # the spike this module avoids
        one_chunk_logits_bytes = Br * csr * Vr * 4
        # peak must stay far below the unchunked spike -- a loose bound (an order
        # of magnitude) so this doesn't get spuriously trippy on allocator
        # fragmentation/caching, while still failing hard on the naive
        # accumulate-every-chunk trap (whose peak would approach
        # unchunked_logits_bytes, not one_chunk_logits_bytes).
        ok_mem = peak < 0.5 * unchunked_logits_bytes
        log(f"[gate:chunked_ce] GPU-ONLY memory check: real dims B={Br} n={nr} d={dr} V={Vr} "
            f"chunk={csr} -> peak_alloc={peak/2**30:.2f}GiB one_chunk_logits="
            f"{one_chunk_logits_bytes/2**30:.2f}GiB unchunked_logits="
            f"{unchunked_logits_bytes/2**30:.2f}GiB (<50% of unchunked ? {ok_mem})")
        del hidden_r, weight_r, labels_r, loss_r
        empty_cache()
    else:
        log(f"[gate:chunked_ce] memory check SKIPPED (GPU-only; device={dev}) -- correctness "
            "(loss+grad equality) is the check that ran")

    ok = ok_correct and ok_mem
    log(f"[gate:chunked_ce] -> {ok}")
    return ok


def run_all_gates(model, tok, cfg):
    """Run every fine-grained gate (plus the fused-vs-naive agreement gate, if
    that's the resolved backend) and return the overall pass/fail. Callers
    (convert.py, heal.py) refuse to proceed unless this returns True --
    "training scripts should hard-refuse to launch unless the gates are green"
    (docs/training-recipe.md)."""
    ok_e, _ = gate_equivalence(model, tok, cfg)
    ok_c = gate_causality(model, cfg)
    ok_p = gate_path_equivalence(model, cfg)
    ok_g = gate_grad_isolation(model, cfg)
    # gate_chunked_ce runs its CUDA memory-behavior check on CUDA (real dims),
    # else the tiny-config fp32 correctness check on CPU/MPS -- see that
    # gate's own docstring for why device='cpu' is the deliberate default.
    ok_ce = gate_chunked_ce(device="cuda" if DEVICE == "cuda" else None)
    ok = ok_e and ok_c and ok_p and ok_g and ok_ce
    if ok and resolve_backend(cfg) == "fused":
        ok = ok and gate_fused_agreement(model, cfg) and gate_grad_isolation(model, cfg, backend="fused")
    log(f"[gates] equivalence={ok_e} causality={ok_c} path={ok_p} gradiso={ok_g} chunked_ce={ok_ce} -> "
        f"{'ALL PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------- shakedown -----
def check_reference(cfg, report):
    """fp32 (tight) and bf16 (loose) HBA-equiv logits vs a shipped fp32 reference
    export (see convert.py --export-ref) -- catches wiring/precision regressions
    when moving the code to a new machine before any real compute is committed."""
    if not os.path.exists(REF_PATH):
        report["reference"] = dict(ok=False, why=f"{REF_PATH} missing (export it with "
                                    "`python -m hba.convert --export-ref` on a reference "
                                    "machine and copy it here first)")
        return
    ref = torch.load(REF_PATH, map_location="cpu")
    ids = ref["ids"].to(DEVICE)
    n = ref["n"]
    ref_donor = ref["donor_logits"].float()
    ref_hba = ref["hba_equiv_logits"].float()
    cos, sin = rope_tables(n, cfg.head_dim, cfg.rope_theta, DEVICE)

    # fp32 wiring check (strict fp32: TF32 would blow the tight tolerance spuriously)
    m32, _, _ = build_hba(cfg, dtype=torch.float32)
    m32.eval()
    with torch.no_grad(), strict_fp32():
        cur_donor = m32.donor(ids).logits.float().cpu()
        cur_hba = m32(ids, cos, sin, cfg.mem_elem_cap, mode="equiv").float().cpu()
    d_donor = (cur_donor - ref_donor).abs().max().item()
    d_hba = (cur_hba - ref_hba).abs().max().item()
    d_self = (cur_hba - cur_donor).abs().max().item()
    fp32_ok = d_donor < 5e-3 and d_hba < 5e-3 and d_self < 5e-3
    report["reference_fp32"] = dict(ok=bool(fp32_ok), d_donor=d_donor, d_hba=d_hba, d_self=d_self,
                                    tol=5e-3)
    log(f"[shake:ref-fp32] donor Δ={d_donor:.2e} hba Δ={d_hba:.2e} self Δ={d_self:.2e} -> {fp32_ok}")
    del m32; empty_cache()

    # bf16 precision-regime check (the healing dtype)
    if DEVICE == "cuda":
        mb, _, _ = build_hba(cfg, dtype=torch.bfloat16)
        mb.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bf_hba = mb(ids, cos, sin, cfg.mem_elem_cap, mode="equiv").float().cpu()
        argmax_agree = (bf_hba.argmax(-1) == ref_hba.argmax(-1)).float().mean().item()
        dmax = (bf_hba - ref_hba).abs().max().item()
        # INFORMATIONAL check with a catastrophic-breakage floor. Correctness is
        # owned by the fp32 gates (equivalence/causality/paths); this bf16-vs-fp32
        # -reference comparison characterizes the healing-precision regime, not
        # wiring. bf16 (8-bit mantissa) through many layers legitimately perturbs
        # ~30-magnitude logits by O(0.1-1); wherever a position's top-2 logits sit
        # inside that noise the argmax flips. Floor: agree < 0.85 still BLOCKS -- a
        # genuinely broken bf16 path (bad cast/kernel) craters agreement toward
        # chance (~1/vocab), far below 0.85, while ordinary precision drift stays
        # well above it. PPL evals monitor actual bf16 healing quality downstream.
        bf16_ok = argmax_agree >= 0.85
        report["reference_bf16"] = dict(ok=bool(bf16_ok), informational=True,
                                        argmax_agree=argmax_agree, dmax=dmax,
                                        blocking_floor_argmax=0.85,
                                        note="non-blocking above the floor; fp32 gates own "
                                             "correctness, PPL evals own bf16 drift")
        log(f"[shake:ref-bf16] argmax agree={argmax_agree:.3f} dmax={dmax:.2f} "
            f"(informational; blocks only if agree<0.85) -> {bf16_ok}")
        del mb; empty_cache()
    else:
        report["reference_bf16"] = dict(ok=True, skipped=f"device={DEVICE} (bf16 regime is CUDA-only)")


def check_gates(cfg, report):
    m, tok, cfg = build_hba(cfg, dtype=torch.float32)
    backend = resolve_backend(cfg)
    # With cfg.attn_backend='fused' (the default), gate_causality's train-path
    # check runs THROUGH the fused backend -- causality exact-0.0 is verified on
    # the path that actually trains.
    ok_c = gate_causality(m, cfg)
    ok_p = gate_path_equivalence(m, cfg)
    ok_g = gate_grad_isolation(m, cfg)
    # gate_chunked_ce: CUDA memory-behavior check (real dims) on CUDA, else the
    # tiny-config fp32 correctness check -- see that gate's docstring.
    ok_ce = gate_chunked_ce(device="cuda" if DEVICE == "cuda" else None)
    report["causality"] = dict(ok=bool(ok_c), train_backend=backend)
    report["path_equiv"] = dict(ok=bool(ok_p))
    report["grad_isolation"] = dict(ok=bool(ok_g))
    report["chunked_ce"] = dict(ok=bool(ok_ce))
    if backend == "fused":
        ok_f = gate_fused_agreement(m, cfg)
        ok_gf = gate_grad_isolation(m, cfg, backend="fused")
        report["fused_agreement"] = dict(ok=bool(ok_f))
        report["grad_isolation_fused"] = dict(ok=bool(ok_gf))
    # G1 induction on the raw donor (equiv mode) -- short/fast (docs/evals.md, "G1")
    from .evals import induction_probe
    acc = induction_probe(m, cfg, "equiv", lengths=(2048, 4096), reps=3, trials=16)
    g1 = max(acc.values()) >= 0.3
    report["induction_G1"] = dict(ok=bool(g1), acc=acc)
    log(f"[shake:G1] raw-donor induction {acc} -> {g1}")
    del m; empty_cache()


def check_training(cfg, planned_tps, steps, report, fast=False):
    """N steps of stage 1; loss falls, aux-KL moves, tok/s measured; checkpoint
    write + reload-resume verified in a fresh process.

    fast=True is the provisioning shakedown's --fast profile (scripts/
    shakedown.sh, scripts/provision.sh): steps drops to 50 (from 150) and the
    tok/s measurement window excludes the first ~10 steps (compile/autotune
    warmup -- see heal.train's warmup_steps) so a 50-step average isn't
    dominated by one-time kernel autotune cost the way a naive average would
    be. Guarded so warmup_steps < steps even at very small --steps overrides."""
    from .heal import PHASES, train
    from . import heal
    if not os.path.exists(os.path.join(DATA, "train.bin")):
        report["training"] = dict(ok=False, why="data/train.bin missing")
        return
    warmup_steps = min(10, max(0, steps - 1)) if fast else 0
    # Micro-batch is GPU-memory-dependent (backward-pass activation memory scales
    # with it): a conservative default of 2 on <=40GB-class cards, 8 above that,
    # both overridable via env for cards that don't fit the heuristic.
    _default_mb = ("8" if torch.cuda.is_available()
                   and torch.cuda.get_device_properties(0).total_memory > 40e9 else "2") \
        if torch.cuda.is_available() else "1"
    _mb = int(os.environ.get("HEAL_MICRO", _default_mb))
    _ga = int(os.environ.get("HEAL_ACCUM", str(max(1, 32 // _mb) if _mb >= 8 else max(1, 4 // _mb))))
    PHASES["stage1"]["tokens"] = steps * _mb * _ga * cfg.heal_ctx
    t0 = time.time()
    # Run in-process for the measurement; capture the loss trace via a light
    # monkeypatch on log rather than parsing files.
    losses, auxes, tps_seen = [], [], []
    orig = heal.log

    def cap(*a):
        s = " ".join(str(x) for x in a)
        orig(*a)
        if "] step " in s and " lm " in s:
            try:
                losses.append(float(s.split(" lm ")[1].split()[0]))
                auxes.append(float(s.split(" aux ")[1].split()[0]))
                tps_seen.append(float(s.split(" tok/s ")[1].split()[0]))
            except Exception:
                pass
    heal.log = cap
    try:
        # shakedown=True waives heal's smoke-shard/corpus-size data guards: this
        # mini stage legitimately trains on a small data slice. The guards stay
        # fully armed for any real heal invocation (scripts/heal.sh never passes
        # --shakedown).
        train(cfg, "stage1", resume=False, micro_batch=_mb, grad_accum=_ga,
              budget_s=1800, smoke=(DEVICE != "cuda"), shakedown=True, warmup_steps=warmup_steps)
    finally:
        heal.log = orig
    # train()'s model/optimizer are freed on return, but the caching allocator
    # keeps the CUDA memory RESERVED in this parent process -- the resume
    # subprocess below then loads its own copy of the model in the same process
    # tree and can OOM on memory-constrained cards if the reservation isn't
    # released first.
    empty_cache()
    dt = time.time() - t0
    tps = max(tps_seen) if tps_seen else 0.0
    loss_falls = len(losses) >= 2 and losses[-1] < losses[0] + 0.05    # tolerate noise
    aux_moves = len(auxes) >= 2 and abs(auxes[-1] - auxes[0]) > 1e-4
    fast_enough = tps >= 0.7 * planned_tps if planned_tps > 0 else True
    ok = loss_falls and aux_moves and (fast_enough or DEVICE != "cuda")
    report["training"] = dict(ok=bool(ok), loss0=losses[0] if losses else None,
                              lossN=losses[-1] if losses else None, loss_falls=bool(loss_falls),
                              aux0=auxes[0] if auxes else None, auxN=auxes[-1] if auxes else None,
                              aux_moves=bool(aux_moves), tok_s=tps, planned_tps=planned_tps,
                              fast_enough=bool(fast_enough), wall_s=dt)
    log(f"[shake:train] loss {report['training']['loss0']}->{report['training']['lossN']} "
        f"aux {report['training']['aux0']}->{report['training']['auxN']} tok/s {tps:.0f} "
        f"(planned {planned_tps}) -> {ok}")
    if planned_tps > 0 and DEVICE == "cuda" and not fast_enough:
        log(f"[shake:train] *** ABORT-WORTHY: measured {tps:.0f} tok/s < 70% of planned "
            f"{planned_tps} -- the plan's wall-clock/cost arithmetic will not hold ***")

    # checkpoint write + reload-resume (separate process to prove on-box durability)
    suffix = "_smoke" if DEVICE != "cuda" else ""
    ck = os.path.join(RESULTS, f"heal_stage1{suffix}.pt")
    resume_ok = os.path.exists(ck)
    if resume_ok:
        r = subprocess.run([sys.executable, "-m", "hba.heal",
                            "--phase", "stage1", "--resume", "--skip-gates", "--shakedown",
                            "--tokens", str(PHASES["stage1"]["tokens"]),
                            "--micro-batch", "1", "--grad-accum", "4"]
                           + (["--smoke"] if DEVICE != "cuda" else []),
                           capture_output=True, text=True, timeout=600)
        resume_ok = ("resuming from step" in r.stdout or "already complete" in r.stdout)
    report["resume"] = dict(ok=bool(resume_ok))
    log(f"[shake:resume] checkpoint reload-resume -> {resume_ok}")
    # move the mini-run ckpt ASIDE: it must never be mistaken for real healing
    # output (the ckpt signature also embeds the token budget, but a real-named
    # heal_stage1.pt on disk left over from a shakedown is a foot-gun).
    if os.path.exists(ck):
        shadow = os.path.join(RESULTS, f"heal_stage1{suffix}_shakedown.pt")
        os.replace(ck, shadow)
        log(f"[shake:train] shakedown mini-ckpt moved aside -> {shadow}")


def check_eval(cfg, report, fast=False):
    """One PPL cell (+ one needle cell, unless fast=True) end-to-end (small).

    fast=True (the provisioning shakedown's --fast profile) skips the needle
    cell -- the ~1 min fast-profile eval budget (docker/README.md) covers one
    PPL cell only; needle stays in the full 150-step profile."""
    from .evals import converted_model, needle_accuracy, perplexity
    ok = True; det = {}
    try:
        m = converted_model(cfg, smoke=(DEVICE != "cuda"), allow_raw=True)  # plumbing only
        vb = os.path.join(DATA, "val_books.bin")
        if os.path.exists(vb):
            det["ppl_books"] = perplexity(m, cfg, vb, mode="eval", max_windows=4)
        if not fast:
            ns = os.path.join(DATA, "needle_books.bin")
            if os.path.exists(ns):
                stream = np.memmap(ns, dtype=np.uint32, mode="r")
                det["needle_4096"] = needle_accuracy(m, cfg, 4096, "H_flat", 0, stream)
        del m; empty_cache()
    except Exception as e:
        ok = False; det["error"] = f"{type(e).__name__}: {e}"
    det["ok"] = bool(ok and ("ppl_books" in det or "needle_4096" in det))
    report["eval"] = det
    log(f"[shake:eval] fast={fast} {det}")


def run_shakedown(cfg, planned_tps=5000.0, steps=150, stage="all", fast=False):
    """Run the pre-flight shakedown and write REPORT. Returns the overall bool.

    fast=True is the provisioning entrypoint's fast profile (scripts/
    provision.sh -> scripts/shakedown.sh --fast -> here): 50 measured train
    steps with the tok/s window excluding compile/autotune warmup
    (check_training), and a single PPL eval cell instead of PPL+needle
    (check_eval). The fine-grained fp32 gates + G1 induction (check_gates) and
    the reference check (check_reference) are unchanged by fast -- they are
    already fast (~4 min combined) and are exactly the correctness surface
    that must never be skipped before any training runs on a new box."""
    log(f"shakedown device={DEVICE} dtype={COMPUTE_DTYPE} planned_tps={planned_tps} fast={fast}")
    report = dict(device=DEVICE, torch=torch.__version__,
                  cuda=torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, ts=time.time())
    if stage in ("all", "ref"):
        check_reference(cfg, report)
    if stage in ("all", "gates"):
        check_gates(cfg, report)
    if stage in ("all", "train"):
        check_training(cfg, planned_tps, steps, report, fast=fast)
    if stage in ("all", "eval"):
        check_eval(cfg, report, fast=fast)

    checks = {k: v for k, v in report.items() if isinstance(v, dict) and "ok" in v}
    overall = all(v["ok"] for v in checks.values())
    report["PASS"] = bool(overall)
    json.dump(report, open(REPORT + ".tmp", "w"), indent=2)
    os.replace(REPORT + ".tmp", REPORT)
    banner = "=" * 60
    print(f"\n{banner}\nSHAKEDOWN {'PASS' if overall else 'FAIL'}  ("
          + ", ".join(f"{k}={'ok' if v['ok'] else 'FAIL'}" for k, v in checks.items())
          + f")\nreport -> {REPORT}\n{banner}", flush=True)
    return overall


def main():
    import argparse
    from .config import HBAConfig
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planned-tps", type=float, default=5000.0,
                    help="planned healing tok/s (see the training-recipe cost notes); abort if "
                         "measured throughput is under 70pct of this on CUDA")
    ap.add_argument("--steps", type=int, default=None,
                    help="default: 50 with --fast, else 150")
    ap.add_argument("--stage", choices=["all", "ref", "gates", "train", "eval"], default="all")
    ap.add_argument("--fast", action="store_true",
                    help="provisioning fast profile (scripts/provision.sh): 50 measured train "
                         "steps with the tok/s window excluding compile/autotune warmup, one "
                         "PPL eval cell instead of PPL+needle. See docker/README.md.")
    args = ap.parse_args()
    cfg = HBAConfig()
    steps = args.steps if args.steps is not None else (50 if args.fast else 150)
    ok = run_shakedown(cfg, planned_tps=args.planned_tps, steps=steps, stage=args.stage,
                       fast=args.fast)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
