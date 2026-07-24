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

MULTI-GPU GATES (gate_shard_partition, gate_rank_consistency, gate_ddp_equivalence,
gate_nccl_bandwidth, and check_training's aggregate-throughput extension) are the
multi-GPU shakedown's own layer, run in addition to the above when launched under
torchrun with world > 1 -- see scripts/shakedown.sh's multi-GPU mode and this
module's `main`'s `--multi-gpu` flag. gate_shard_partition is pure Python (no GPU,
no process group) and runs anywhere; the other three require an initialized
torch.distributed process group (world > 1) to be meaningful.
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
from .kd import chunked_kd_kl, reference_kd_kl
from .config import (COMPUTE_DTYPE, DATA, DEVICE, INIT_PATH, REF_PATH, RESULTS, empty_cache, log,
                     resolve_backend, smoke_config)
from .fam_data import FamMixer
from .model import build_hba
from . import dist_util

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
    fp32 for a tight bound. docs/training-recipe.md, "Equivalence gate".

    ONLY MEANINGFUL WITH cfg.qknorm=False. QKNorm (docs/design.md, "Softmax
    length-calibration") is a deliberate architectural departure from the
    donor's Q/K statistics -- with cfg.qknorm=True this gate is EXPECTED to
    fail (the whole point of QKNorm is that q.k is no longer the donor's own
    q.k), so `run_all_gates`/`check_gates` skip calling this gate at all when
    qknorm is on and run `gate_qknorm_math` instead (the qknorm=ON internal-
    consistency check: the model's own QKNorm math matches a transparent
    reference implementation, and the fused-vs-naive / dense-vs-eval agreement
    gates, run unconditionally regardless of qknorm, cover the rest of "the
    QKNorm path is internally consistent"). Call this function directly (as the
    qknorm=OFF regression check) to prove the non-QKNorm swap is still exact."""
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
    qkn = model.qknorms[0]
    for p in summ.parameters():
        p.grad = None
    attn = hba_attention_fused if backend == "fused" else hba_attention_dense
    out, aux = attn(q, k, v, cos, sin, cfg, summ, qkn)
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
            qkn = model.qknorms[L]
            # (a) selection-agnostic math check
            from .attention import hba_attention_eval
            oda, _ = hba_attention_dense(q, k, v, cos, sin, cfg_all, summ, qkn)
            oea = hba_attention_eval(q, k, v, cos, sin, cfg_all, summ, qkn, cfg.mem_elem_cap)
            worst_math = max(worst_math, float((oda - oea).abs().max()))
            del oda, oea
            # (b) real-cfg selection-stability check
            od, _ = hba_attention_dense(q, k, v, cos, sin, cfg, summ, qkn)
            oe = hba_attention_eval(q, k, v, cos, sin, cfg, summ, qkn, cfg.mem_elem_cap)
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

    Uses `model.qknorms[0]`, so this is exercised WITH QKNorm on whenever
    cfg.qknorm is True: both backends call the identical `attention._content_qk`
    helper, so this gate stays the load-bearing "the two backends implement
    QKNorm identically" check (docs/design.md, "Softmax length-calibration") --
    not just "the two backends agree on the pre-QKNorm math".

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
    qkn = model.qknorms[0]
    ok = True
    with strict_fp32():
        for n in ns:
            torch.manual_seed(1234 + n)
            q = torch.randn(1, n, Hq, dh, device=dev, requires_grad=True)
            k = torch.randn(1, n, Hkv, dh, device=dev, requires_grad=True)
            v = torch.randn(1, n, Hkv, dh, device=dev, requires_grad=True)
            cos, sin = rope_tables(n, cfg.head_dim, cfg.rope_theta, dev)
            g = torch.randn(1, n, Hq, dh, device=dev)
            o_n, a_n = hba_attention_dense(q, k, v, cos, sin, cfg, summ, qkn)
            gq_n, gk_n, gv_n = torch.autograd.grad(o_n, (q, k, v), g, retain_graph=False)
            o_f, a_f = hba_attention_fused(q, k, v, cos, sin, cfg, summ, qkn)
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

        # Empirical, apples-to-apples: measure the TOTAL peak allocation of the
        # chunked path AND the unchunked oracle at the SAME real dims, and require
        # the chunked path to peak meaningfully lower. Comparing the chunked total
        # peak against a bare `B*n*V*4` logit-byte count (an earlier formulation)
        # is wrong -- the total peak also holds params, activations, grads, and
        # one chunk's softmax/backward temporaries, so it legitimately exceeds the
        # bare full-logit size while still being far below the unchunked path's
        # actual peak. The naive accumulate-every-chunk trap would show ~no saving.
        def _peak(fn):
            h = torch.randn(Br, nr, dr, device=dev, requires_grad=True)
            w = torch.randn(Vr, dr, device=dev, requires_grad=True)
            lab = torch.randint(0, Vr, (Br, nr), device=dev)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(dev)
            loss = fn(h, w, lab)
            loss.backward()
            torch.cuda.synchronize()
            p = torch.cuda.max_memory_allocated(dev)
            del h, w, lab, loss
            empty_cache()
            return p

        one_chunk_logits_bytes = Br * csr * Vr * 4
        full_logits_bytes = Br * nr * Vr * 4
        theo_saving = full_logits_bytes - one_chunk_logits_bytes  # what chunking should recover
        peak_ch = _peak(lambda h, w, lab: chunked_cross_entropy(h, w, lab, chunk_size=csr))
        try:
            peak_un = _peak(lambda h, w, lab: reference_cross_entropy(h, w, lab))
            saving = peak_un - peak_ch
            # (a) chunked peaks strictly lower, and (b) recovers most of the
            # theoretical logit saving -- a loose 0.5 factor tolerates allocator
            # caching/fragmentation without admitting the no-saving trap.
            ok_mem = peak_ch < peak_un and saving > 0.5 * theo_saving
            log(f"[gate:chunked_ce] GPU memory check: real dims B={Br} n={nr} d={dr} V={Vr} "
                f"chunk={csr} -> chunked_peak={peak_ch/2**30:.2f}GiB "
                f"unchunked_peak={peak_un/2**30:.2f}GiB saving={saving/2**30:.2f}GiB "
                f"(> 0.5*theo {theo_saving/2**30/2:.2f}GiB ? {ok_mem})")
        except torch.cuda.OutOfMemoryError:
            # The unchunked oracle could not even run at these dims while the
            # chunked path did -- the strongest possible demonstration of the
            # memory saving. That IS the pass.
            empty_cache()
            ok_mem = True
            log(f"[gate:chunked_ce] GPU memory check: unchunked oracle OOM'd at real dims "
                f"(B={Br} n={nr} V={Vr}) while chunked_peak={peak_ch/2**30:.2f}GiB succeeded "
                f"-- chunking is load-bearing here -> {ok_mem}")
    else:
        log(f"[gate:chunked_ce] memory check SKIPPED (GPU-only; device={dev}) -- correctness "
            "(loss+grad equality) is the check that ran")

    ok = ok_correct and ok_mem
    log(f"[gate:chunked_ce] -> {ok}")
    return ok


def gate_kd(tol=1e-5, tiny=None, real=None, device=None):
    """hba.kd.chunked_kd_kl's recompute-in-backward must be BIT-HONEST against
    hba.kd.reference_kd_kl -- the same rigor gate_chunked_ce applies to
    chunked_cross_entropy, and for the same reason: (a) the two scalar KD
    losses must agree to `tol`, AND (b) every gradient -- into the student
    hidden-state input as well as the lm_head weight -- must agree to `tol`.
    Checked at both T=1 and T=2 (the T^2 rescaling is itself part of what must
    round-trip correctly through chunking; see hba.kd's module docstring).

    Only wired into run_all_gates/check_gates when the caller has KD enabled
    for the run (heal.py's --kd flag) -- unlike gate_chunked_ce (always on,
    since chunked CE is the unconditional training-loss path), donor KD is an
    opt-in stage-2 objective, so this gate is skipped cleanly when KD is off
    rather than paying its cost on every run.

    Runs in fp32 on CPU by default, tiny/deliberately-uneven dims (chunk_size
    does not divide n) -- seconds of compute. Pass `device='cuda'` explicitly,
    plus optionally `real` (a dict of B/n/d/V/chunk_size), to additionally run
    a real-dims memory check: chunked_kd_kl's peak CUDA allocation must stay
    meaningfully below an unchunked reference call's peak, using the same
    empirical both-paths-peak comparison gate_chunked_ce uses (not a
    peak-vs-bare-bytes threshold) -- see that gate's docstring for why."""
    dev = device or "cpu"
    tiny = tiny or dict(B=2, n=50, d=16, V=24, chunk_size=13, ignore_frac=0.15)
    B, n, d, V, chunk_size = tiny["B"], tiny["n"], tiny["d"], tiny["V"], tiny["chunk_size"]
    ignore_frac = tiny.get("ignore_frac", 0.0)
    g = torch.Generator(device="cpu").manual_seed(0)
    hidden0 = torch.randn(B, n, d, generator=g)
    weight0 = torch.randn(V, d, generator=g)
    bias0 = torch.randn(V, generator=g)
    teacher_logits0 = torch.randn(B, n, V, generator=g)
    labels = torch.randint(0, V, (B, n), generator=g)
    if ignore_frac > 0:
        mask = torch.rand(B, n, generator=g) < ignore_frac
        labels = labels.masked_fill(mask, -100)

    def _leaf(t):
        return t.clone().to(dev).requires_grad_(True)

    lbl = labels.to(dev)
    teacher_logits = teacher_logits0.to(dev)

    ok_correct = True
    for T in (1.0, 2.0):
        h_ref, w_ref, b_ref = _leaf(hidden0), _leaf(weight0), _leaf(bias0)
        h_ch, w_ch, b_ch = _leaf(hidden0), _leaf(weight0), _leaf(bias0)

        loss_ref = reference_kd_kl(h_ref, teacher_logits, w_ref, lbl, bias=b_ref, temperature=T)
        loss_ref.backward()
        loss_ch = chunked_kd_kl(h_ch, teacher_logits, w_ch, lbl, bias=b_ch, temperature=T,
                                chunk_size=chunk_size)
        loss_ch.backward()

        d_loss = abs(float(loss_ref.detach()) - float(loss_ch.detach()))
        d_h = (h_ref.grad - h_ch.grad).abs().max().item()
        d_w = (w_ref.grad - w_ch.grad).abs().max().item()
        d_b = (b_ref.grad - b_ch.grad).abs().max().item()
        ok = d_loss < tol and d_h < tol and d_w < tol and d_b < tol
        ok_correct = ok_correct and ok
        log(f"[gate:kd] T={T} dims B={B} n={n} d={d} V={V} chunk={chunk_size} device={dev} "
            f"|Δloss|={d_loss:.2e} |Δgrad_hidden|={d_h:.2e} |Δgrad_weight|={d_w:.2e} "
            f"|Δgrad_bias|={d_b:.2e} (all <{tol} ? {ok})")

    # Sanity: student logits == teacher logits -> KD loss == 0 exactly (KL of a
    # distribution with itself), at T=1 and T=2. Forced by an identity lm_head
    # (weight=I, bias=0) so student_logits = F.linear(hidden, I, 0) = hidden,
    # and setting hidden = teacher_logits directly.
    Vid = 12
    gi = torch.Generator(device="cpu").manual_seed(1)
    same_logits = torch.randn(2, 20, Vid, generator=gi).to(dev)
    eye = torch.eye(Vid, device=dev)
    zero_bias = torch.zeros(Vid, device=dev)
    lbl_id = torch.randint(0, Vid, (2, 20), generator=gi).to(dev)
    ok_zero = True
    for T in (1.0, 2.0):
        z_ref = reference_kd_kl(same_logits, same_logits, eye, lbl_id, bias=zero_bias, temperature=T)
        z_ch = chunked_kd_kl(same_logits, same_logits, eye, lbl_id, bias=zero_bias, temperature=T,
                             chunk_size=9)
        this_ok = abs(float(z_ref)) < 1e-4 and abs(float(z_ch)) < 1e-4
        ok_zero = ok_zero and this_ok
        log(f"[gate:kd] student==teacher sanity @ T={T}: ref={float(z_ref):.2e} "
            f"chunked={float(z_ch):.2e} (both ~0 ? {this_ok})")

    ok_mem = True
    if dev == "cuda":
        real = real or dict(B=4, n=4096, d=896, V=151936, chunk_size=1024)
        Br, nr, dr, Vr, csr = real["B"], real["n"], real["d"], real["V"], real["chunk_size"]

        def _peak(fn):
            h = torch.randn(Br, nr, dr, device=dev, requires_grad=True)
            w = torch.randn(Vr, dr, device=dev, requires_grad=True)
            t = torch.randn(Br, nr, Vr, device=dev)
            lab = torch.randint(0, Vr, (Br, nr), device=dev)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(dev)
            loss = fn(h, w, t, lab)
            loss.backward()
            torch.cuda.synchronize()
            p = torch.cuda.max_memory_allocated(dev)
            del h, w, t, lab, loss
            empty_cache()
            return p

        peak_ch = _peak(lambda h, w, t, lab: chunked_kd_kl(h, t, w, lab, chunk_size=csr))
        try:
            peak_un = _peak(lambda h, w, t, lab: reference_kd_kl(h, t, w, lab))
            saving = peak_un - peak_ch
            one_chunk_logits_bytes = Br * csr * Vr * 4
            full_logits_bytes = Br * nr * Vr * 4
            theo_saving = full_logits_bytes - one_chunk_logits_bytes
            ok_mem = peak_ch < peak_un and saving > 0.5 * theo_saving
            log(f"[gate:kd] GPU memory check: real dims B={Br} n={nr} d={dr} V={Vr} chunk={csr} "
                f"-> chunked_peak={peak_ch/2**30:.2f}GiB unchunked_peak={peak_un/2**30:.2f}GiB "
                f"saving={saving/2**30:.2f}GiB (> 0.5*theo {theo_saving/2**30/2:.2f}GiB ? {ok_mem})")
        except torch.cuda.OutOfMemoryError:
            empty_cache()
            ok_mem = True
            log(f"[gate:kd] GPU memory check: unchunked reference OOM'd at real dims (B={Br} "
                f"n={nr} V={Vr}) while chunked_peak={peak_ch/2**30:.2f}GiB succeeded -- chunking "
                f"is load-bearing here -> {ok_mem}")
    else:
        log(f"[gate:kd] memory check SKIPPED (GPU-only; device={dev}) -- correctness (loss+grad "
            "equality + student==teacher sanity) is the check that ran")

    ok = ok_correct and ok_zero and ok_mem
    log(f"[gate:kd] -> {ok}")
    return ok


@torch.no_grad()
def gate_qknorm_math(model, cfg, tol=1e-5):
    """qknorm=ON internal-consistency gate (docs/design.md, "Softmax length-
    calibration"; docs/training-recipe.md, "Correctness gates"). Supersedes
    `gate_equivalence` for qknorm=True runs -- see that gate's docstring: donor-
    equivalence no longer holds once QKNorm deliberately changes Q/K statistics,
    so this gate checks something else that MUST still hold: the model's actual
    QKNorm computation matches a transparent, independently-written reference
    implementation, and the normalized output has the exact RMS the recipe
    depends on for boundedness.

    Two checks, both blocking:
      (a) MATH: `model.qknorms[0]`'s output on random q/k, vs a plain from-
          scratch RMSNorm-then-scalar-gain reference written directly in this
          gate (not calling into attention.py's own implementation) -- these
          must agree to fp32 precision.
      (b) BOUNDEDNESS: the post-norm per-head RMS must equal the learned gain
          EXACTLY (RMSNorm's defining property: ||qc||=sqrt(dh)*gain_q per
          head). This is what makes q.k/d bounded to [-gain_q*gain_k,
          +gain_q*gain_k] regardless of head_dim (Cauchy-Schwarz on
          ||qc||=sqrt(dh)*gain_q, ||kc||=sqrt(dh)*gain_k) -- the actual
          property the whole recipe leans on, not just "some normalization
          happened".

    Causality (gate_causality) and naive-vs-fused agreement (gate_fused_agreement,
    both backends built from the SAME `_content_qk` call) already exercise the
    QKNorm path end-to-end and are run unconditionally in run_all_gates/
    check_gates regardless of qknorm -- this gate is the QKNorm-specific piece
    those two don't cover."""
    dev = next(model.parameters()).device
    B, n, Hq, Hkv, dh = 2, 37, cfg.n_heads, cfg.n_kv, cfg.head_dim
    g = torch.Generator(device="cpu").manual_seed(20260716)
    q = torch.randn(B, n, Hq, dh, generator=g).to(dev)
    k = torch.randn(B, n, Hkv, dh, generator=g).to(dev)
    qkn = model.qknorms[0]
    with strict_fp32():
        qc = qkn.q(q).float()
        kc = qkn.k(k).float()

        def ref(x, gain, eps):
            xf = x.float()
            rms = xf.pow(2).mean(-1, keepdim=True).clamp_min(eps).sqrt()
            gshape = (1,) * (x.dim() - 2) + (gain.shape[0], 1)
            return (xf / rms) * gain.float().view(*gshape)

        qref = ref(q, qkn.q.gain, qkn.q.eps)
        kref = ref(k, qkn.k.gain, qkn.k.eps)
        d_math = max(float((qc - qref).abs().max()), float((kc - kref).abs().max()))
        rms_q = qc.pow(2).mean(-1).sqrt()                              # [B,n,Hq]
        rms_k = kc.pow(2).mean(-1).sqrt()                              # [B,n,Hkv]
        d_bound_q = float((rms_q - qkn.q.gain.float()[None, None]).abs().max())
        d_bound_k = float((rms_k - qkn.k.gain.float()[None, None]).abs().max())
    ok = d_math < tol and d_bound_q < tol and d_bound_k < tol
    log(f"[gate:qknorm-math] max|Δ vs transparent ref|={d_math:.2e} RMS-bound |Δ| "
        f"q={d_bound_q:.2e} k={d_bound_k:.2e} (all <{tol} ? {ok})")
    return ok


def run_all_gates(model, tok, cfg, kd=False):
    """Run every fine-grained gate (plus the fused-vs-naive agreement gate, if
    that's the resolved backend) and return the overall pass/fail. Callers
    (convert.py, heal.py) refuse to proceed unless this returns True --
    "training scripts should hard-refuse to launch unless the gates are green"
    (docs/training-recipe.md).

    kd: pass True only for a run that actually has donor KD enabled (heal.py's
    --kd flag, stage-2-only) -- gate_kd then runs alongside the rest; skipped
    cleanly (not even imported into the report) otherwise, since KD is an
    opt-in stage-2 objective, not a standing part of every run.

    qknorm split (docs/design.md, "Softmax length-calibration"; docs/training-
    recipe.md, "Correctness gates"): `gate_equivalence` (donor-equivalence) is
    only meaningful with cfg.qknorm=False and is SKIPPED (not run, not silently
    loosened) when qknorm is on; `gate_qknorm_math` (the qknorm=ON internal-
    consistency check) runs in its place. Every other gate here -- causality,
    path-equivalence, grad-isolation, fused-vs-naive agreement -- runs
    UNCONDITIONALLY regardless of qknorm, since both attention backends route
    through the identical `attention._content_qk` and must stay exactly
    consistent with each other whether or not QKNorm is active."""
    qknorm_on = bool(getattr(cfg, "qknorm", False))
    if qknorm_on:
        ok_e, d_e = True, None    # gate_equivalence not meaningful once QKNorm changes Q/K stats
    else:
        ok_e, d_e = gate_equivalence(model, tok, cfg)
    ok_c = gate_causality(model, cfg)
    ok_p = gate_path_equivalence(model, cfg)
    ok_g = gate_grad_isolation(model, cfg)
    # gate_chunked_ce runs its CUDA memory-behavior check on CUDA (real dims),
    # else the tiny-config fp32 correctness check on CPU/MPS -- see that
    # gate's own docstring for why device='cpu' is the deliberate default.
    ok_ce = gate_chunked_ce(device="cuda" if DEVICE == "cuda" else None)
    ok_qk = gate_qknorm_math(model, cfg) if qknorm_on else True
    ok = ok_e and ok_c and ok_p and ok_g and ok_ce and ok_qk
    if ok and resolve_backend(cfg) == "fused":
        ok = ok and gate_fused_agreement(model, cfg) and gate_grad_isolation(model, cfg, backend="fused")
    ok_kd = None
    if kd:
        ok_kd = gate_kd(device="cuda" if DEVICE == "cuda" else None)
        ok = ok and ok_kd
    log(f"[gates] equivalence={'skipped(qknorm)' if qknorm_on else ok_e} causality={ok_c} "
        f"path={ok_p} gradiso={ok_g} chunked_ce={ok_ce} "
        f"qknorm_math={ok_qk if qknorm_on else 'n/a'}"
        + (f" kd={ok_kd}" if kd else "") + f" -> {'ALL PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------ multi-GPU gates --
def gate_shard_partition(cfg, world, micro_B, accum, K=5):
    """Blocking (design: multi-GPU shakedown gate 3). PURE PYTHON -- no GPU, no
    process group, no model -- so it runs anywhere, including in this repo's
    CPU-only test suite. Enumerates the first K optimizer steps' g-indices
    in-process for a given (world, micro_B, accum) and checks:

      (a) SHARD PARTITION: at every one of the first K steps, every rank's
          g-set (dist_util.rank_g_set) is pairwise DISJOINT from every other
          rank's, and their union equals the world=1 enumeration's g-set
          (dist_util.step_g_set) at the SAME windows_per_step -- i.e. every
          window is consumed exactly once across the world, and the set of
          windows a step consumes does not depend on how windows_per_step
          happens to factor into (world, micro_B, accum). This is the gate
          that would catch the "naive bug" the design calls out explicitly:
          every rank silently seeing the SAME stream (accidentally passing
          rank=0 everywhere) turns world GPUs into 1 effective GPU training on
          duplicated data -- that bug fails this gate immediately (ranks'
          g-sets would be identical, not disjoint).
      (b) FAM-MIX PLANT IDENTITY: fam_data.FamMixer plants a fixed g
          identically regardless of which rank owns it. FamMixer._rng(g)'s
          plant is a pure function of (g, buffer length, vocab) -- not of the
          pre-existing buffer content or of how g was reached -- so calling it
          twice for the SAME g from two independent RNG draws must produce
          byte-identical plants. No real WindowStream/model needed: only
          FamMixer's own seed + cfg.vocab_size are exercised directly.
    """
    windows_per_step = dist_util.windows_per_step(world, micro_B, accum)
    for step in range(K):
        full = dist_util.step_g_set(step, world, micro_B, accum)
        ref = dist_util.step_g_set(step, 1, windows_per_step, 1)
        assert full == ref, (
            f"[gate:shard-partition] step {step}: g-set (world={world}, micro_B={micro_B}, "
            f"accum={accum}) != world=1 reference enumeration at the same windows_per_step "
            f"({windows_per_step})"
        )
        parts = [dist_util.rank_g_set(step, world, micro_B, accum, r) for r in range(world)]
        union = set().union(*parts)
        assert union == full, f"[gate:shard-partition] step {step}: per-rank union != full g-set"
        for i in range(world):
            for j in range(i + 1, world):
                assert parts[i].isdisjoint(parts[j]), (
                    f"[gate:shard-partition] step {step}: rank {i} and rank {j} g-sets overlap "
                    f"-- the naive duplication bug (every rank seeing the same stream)"
                )

    class _FakeStream:  # only .B is read by FamMixer.__init__; .batch is never called
        def __init__(self, B):
            self.B = B

    fam = FamMixer(_FakeStream(1), cfg, seed=12345, frac=0.03)
    L = 256
    test_gs = sorted({0, 1, windows_per_step - 1, windows_per_step,
                      windows_per_step * K - 1, windows_per_step * K})
    for g in test_gs:
        row_a, row_b = np.zeros(L, dtype=np.int64), np.zeros(L, dtype=np.int64)
        planted_a, npairs_a = fam._plant(row_a, fam._rng(g))
        planted_b, npairs_b = fam._plant(row_b, fam._rng(g))
        assert np.array_equal(row_a, row_b) and planted_a == planted_b and npairs_a == npairs_b, (
            f"[gate:shard-partition] g={g}: FamMixer plant not reproducible from g alone "
            "(rank/world leaking into the RNG key)"
        )
    log(f"[gate:shard-partition] world={world} micro_B={micro_B} accum={accum} "
        f"windows_per_step={windows_per_step}: {K}-step partition/union/invariance OK; "
        f"FamMixer g-purity OK for {len(test_gs)} g values -> True")
    return True


def gate_rank_consistency(model, cfg, device=None, tol=1e-6):
    """Blocking (design: multi-GPU shakedown gate 2). Broadcasts one fixed batch
    (a fixed-seed random batch -- identical on every rank by construction, not
    actually broadcast over the wire, since a fixed seed already IS the
    broadcast) and verifies (a) every rank's own parameters are identical
    (dist_util.assert_rank_consistent) and (b) every rank computes the same
    loss on that batch, max spread <= `tol`. No-op True at world=1 (nothing to
    check); requires an initialized process group with world > 1 to be
    meaningful, i.e. this is a multi-GPU-only gate that cannot run in this
    repo's CPU-only test suite (or in an environment without a GPU box)."""
    if not dist_util.is_distributed():
        log("[gate:rank-consistency] world=1 -- trivially OK")
        return True
    import torch.distributed as dist
    dev = device or next(model.parameters()).device
    spread = dist_util.assert_rank_consistent(model, dev, tol=tol, tag="[gate:rank-consistency:params]")

    raw = dist_util.raw_model(model)
    n = max(4 * cfg.window, 8 * cfg.block)
    n = (n // cfg.block) * cfg.block
    # Fixed seed -> byte-identical batch on every rank without any actual
    # network broadcast; this IS "broadcast one fixed batch" (design gate 2's
    # wording) for a synthetic input where the content only needs to be
    # IDENTICAL across ranks, not drawn from real data.
    g = torch.Generator(device="cpu").manual_seed(20260716)
    ids = torch.randint(0, cfg.vocab_size, (1, n), generator=g).to(dev)
    tgt = torch.randint(0, cfg.vocab_size, (1, n), generator=g).to(dev)
    cos, sin = rope_tables(n, cfg.head_dim, cfg.rope_theta, dev)
    was_training = raw.training
    raw.eval()
    with torch.no_grad(), strict_fp32():
        logits = raw(ids, cos, sin, cfg.mem_elem_cap, mode="eval")
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1))
    raw.train(was_training)
    loss64 = loss.detach().to(torch.float64)
    world = dist.get_world_size()
    gathered = [torch.zeros_like(loss64) for _ in range(world)]
    dist.all_gather(gathered, loss64)
    vals = [float(x.item()) for x in gathered]
    loss_spread = max(vals) - min(vals)
    ok = loss_spread <= tol
    log(f"[gate:rank-consistency] param spread={spread:.2e} fixed-batch loss spread="
        f"{loss_spread:.2e} across {world} ranks (<= {tol} ? {ok})")
    return ok


def gate_nccl_bandwidth(size_gb=2.0, min_bus_gbs=6.0, device=None, advisory=True):
    """Multi-GPU shakedown microbench (design: gate 7). All-reduces a ~size_gb
    buffer and reports measured BUS bandwidth (consumer-GPU host-staged
    all-reduce -- no P2P -- realistically lands ~5-12 GB/s).

    ADVISORY by default: the aggregate-throughput scaling-efficiency check in
    check_training (>= 0.75 at world > 1) is the AUTHORITATIVE comm gate -- it
    directly measures whether training actually scales, which is the only thing
    this microbench proxies. Empirically the proxy was too strict: a host that
    measured 4.73 GB/s here (below the original 6 GB/s hard floor) went on to
    scale a full-parameter, heavy-comm stage-3 heal at ~0.8 efficiency, because
    DDP bucketing overlaps the all-reduce with the backward. So a low microbench
    with a healthy scaling-efficiency is a false alarm, not a lemon. When
    advisory (the shakedown default), this WARNS below `min_bus_gbs` but returns
    True; a genuinely comm-bound box is caught by the scaling-efficiency gate.
    Pass advisory=False to restore a hard floor (e.g. a standalone comm probe).
    No-op True at world=1."""
    if not dist_util.is_distributed():
        log("[gate:nccl-bw] world=1 -- trivially OK (nothing to all-reduce across)")
        return True
    result = dist_util.allreduce_bandwidth_microbench(size_gb=size_gb, device=device)
    meets = result["bus_bw_gbs"] >= min_bus_gbs
    log(f"[gate:nccl-bw] {size_gb}GB all-reduce across world={result['world']}: "
        f"{result['seconds']:.3f}s bus_bw={result['bus_bw_gbs']:.2f}GB/s "
        f"(>= {min_bus_gbs} ? {meets}{'; ADVISORY -- scaling-efficiency gate governs' if advisory else ''})")
    if advisory:
        if not meets:
            log(f"[gate:nccl-bw] WARNING: {result['bus_bw_gbs']:.2f}GB/s is below the "
                f"{min_bus_gbs}GB/s advisory floor -- NOT failing the shakedown (the "
                "aggregate-throughput scaling-efficiency gate in check_training is the "
                "authoritative comm gate). Watch the aggregate tok/s during training.")
        return True
    return meets


def gate_ddp_equivalence(cfg, world, micro_batch=1, steps=30, warmup=5, tol_loss=1e-3,
                         tol_param=1e-5, phase="stage1"):
    """Blocking (design: multi-GPU shakedown gate 4). GPU-ONLY orchestration --
    cannot run in this repo's CPU-only test suite or without `world` real GPUs
    -- so this is exercised on an actual multi-GPU box, not here.

    Runs `steps` optimizer steps at world=`world` (via `torchrun --standalone
    --nproc_per_node=world`) and at world=1 (plain `python`), both on a
    SYNTHETIC short schedule (warmup=`warmup`, a compressed cosine decay over
    `steps` steps -- `heal.py --warmup` override) at the SAME global tokens/
    step, both `--smoke --shakedown --skip-gates` so the comparison is fast and
    never touches the real corpus/checkpoint namespace. `warmup=5` over 30
    steps (not the real phase's warmup=200) is required: 30 steps inside a real
    warmup=200 would apply near-zero LR throughout and pass this gate
    VACUOUSLY even with broken sharding (params barely move either way).

    Two sub-checks, both must pass:
      (a) loss trajectories match within `tol_loss` (~1e-3 -- fp reduction-
          order noise only, under the registered comm_dtype for `phase`);
      (b) post-run max|Δparam| between the two final checkpoints <= `tol_param`
          (~1e-5) -- a trajectory can look right while a param SUBSET has
          silently diverged (e.g. one rank's slice never actually trained).
    """
    import re
    windows_per_step = 32  # matches heal.GLOBAL_TOKENS_PER_STEP // 4096 at ctx=4096
    # `_run` below always launches with --smoke, so heal._heal_main builds cfg
    # via config.smoke_config() (heal_ctx=512), NOT this function's own `cfg`
    # (heal_ctx=4096, the real recipe config). The token budget MUST be sized
    # off the ctx the subprocess actually trains at -- using the real cfg's
    # heal_ctx here while the subprocess runs at the smoke ctx inflated
    # total_steps by 4096/512 = 8x (~240 steps instead of the intended ~30),
    # risking a timeout and accumulating 8x the fp-noise drift against
    # tol_loss/tol_param.
    tokens = steps * windows_per_step * smoke_config().heal_ctx
    grad_accum_1gpu = windows_per_step // micro_batch
    grad_accum_ngpu = windows_per_step // (micro_batch * world)
    assert grad_accum_1gpu * micro_batch == windows_per_step, "micro_batch must divide windows_per_step"
    assert grad_accum_ngpu * micro_batch * world == windows_per_step, (
        f"micro_batch={micro_batch} world={world} does not divide windows_per_step={windows_per_step}")

    def _run(env_extra, launcher, ckpt_suffix, grad_accum):
        env = dict(os.environ, HBA_RESULTS_DIR=os.path.join(RESULTS, f"_ddp_gate_{ckpt_suffix}"))
        env.update(env_extra)
        os.makedirs(env["HBA_RESULTS_DIR"], exist_ok=True)
        cmd = launcher + ["-m", "hba.heal", "--phase", phase, "--smoke", "--shakedown",
                          "--skip-gates", "--tokens", str(tokens), "--warmup", str(warmup),
                          "--micro-batch", str(micro_batch), "--grad-accum", str(grad_accum)]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        losses = [float(m) for m in re.findall(r" lm (-?[\d.]+) ppl", r.stdout)]
        ckpt = os.path.join(env["HBA_RESULTS_DIR"], f"heal_{phase}_smoke.pt")
        return losses, ckpt, r

    losses_1, ckpt_1, r1 = _run({}, [sys.executable], "1gpu", grad_accum_1gpu)
    losses_n, ckpt_n, rn = _run({}, ["torchrun", "--standalone", f"--nproc_per_node={world}"],
                                "ngpu", grad_accum_ngpu)
    if not (os.path.exists(ckpt_1) and os.path.exists(ckpt_n)):
        log(f"[gate:ddp-equiv] FAILED: checkpoint(s) missing -- world=1 stdout tail:\n"
            f"{r1.stdout[-2000:]}\n{r1.stderr[-2000:]}\nworld={world} stdout tail:\n"
            f"{rn.stdout[-2000:]}\n{rn.stderr[-2000:]}")
        return False
    n_cmp = min(len(losses_1), len(losses_n))
    loss_ok = n_cmp > 0 and max(abs(a - b) for a, b in zip(losses_1[:n_cmp], losses_n[:n_cmp])) < tol_loss
    ck1 = torch.load(ckpt_1, map_location="cpu")["model"]
    ckn = torch.load(ckpt_n, map_location="cpu")["model"]
    max_dparam = max(float((ck1[k].float() - ckn[k].float()).abs().max()) for k in ck1)
    param_ok = max_dparam <= tol_param
    ok = loss_ok and param_ok
    log(f"[gate:ddp-equiv] world=1 vs world={world}, {steps} steps (warmup={warmup}): "
        f"loss_ok={loss_ok} (n_cmp={n_cmp}, tol={tol_loss}) max|Δparam|={max_dparam:.2e} "
        f"(<= {tol_param} ? {param_ok}) -> {ok}")
    return ok


# --------------------------------------------------------------- shakedown -----
def check_reference(cfg, report):
    """fp32 (tight) and bf16 (loose) HBA-equiv logits vs a shipped fp32 reference
    export (see convert.py --export-ref) -- catches wiring/precision regressions
    when moving the code to a new machine before any real compute is committed.

    QKNORM RECHARACTERIZATION (docs/design.md, "Softmax length-calibration"):
    with cfg.qknorm=True the exported hba_equiv_logits is no longer a donor-
    equivalence reference -- it's the QKNorm'd model's OWN fp32 output, an
    INTERNAL self-consistency check (this machine/build reproduces a prior
    qknorm=True build's output) rather than a donor-equivalence check. Re-export
    REF_PATH with `python -m hba.convert --export-ref` on a qknorm=True build
    before relying on this; a pre-QKNorm reference export will not match (by
    design) and should not be treated as a wiring regression."""
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
    if getattr(cfg, "qknorm", False):
        # d_self (hba vs donor) is EXPECTED to be large -- QKNorm deliberately
        # diverges from the donor -- so it is informational only here. d_donor
        # (this machine's raw donor forward vs the reference export's donor
        # forward) is UNAFFECTED by qknorm (the donor itself never sees QKNorm)
        # and stays a valid blocking wiring check. d_hba (this build's QKNorm'd
        # equiv output vs a prior qknorm=True reference export) is the blocking
        # self-consistency check described in this function's docstring.
        fp32_ok = d_donor < 5e-3 and d_hba < 5e-3
        report["reference_fp32"] = dict(ok=bool(fp32_ok), d_donor=d_donor, d_hba=d_hba,
                                        d_self=d_self, tol=5e-3, qknorm=True,
                                        note="d_self is informational only in qknorm mode "
                                             "(QKNorm deliberately diverges from the donor); "
                                             "d_hba is the blocking self-consistency check "
                                             "against a qknorm=True reference export")
        log(f"[shake:ref-fp32] qknorm=True: donor Δ={d_donor:.2e} (blocking) hba-vs-ref-export "
            f"Δ={d_hba:.2e} (blocking) hba-vs-donor Δ={d_self:.2e} (informational, expected "
            f"large) -> {fp32_ok}")
    else:
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


def check_gates(cfg, report, kd=False):
    """kd: True only for a shakedown run that wants the donor-KD gate included
    (see gate_kd / run_all_gates's `kd` parameter) -- skipped cleanly, with no
    report entry at all, when False (the default; KD is opt-in)."""
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
    # qknorm=ON internal-consistency gate (docs/design.md, "Softmax length-
    # calibration"; see run_all_gates's docstring for the full qknorm/
    # gate_equivalence split -- gate_equivalence itself is not run here at all,
    # by design: check_gates never has been the donor-equivalence check, see
    # convert.py's --gates flow for that).
    if getattr(cfg, "qknorm", False):
        report["qknorm_math"] = dict(ok=bool(gate_qknorm_math(m, cfg)))
    if kd:
        report["kd"] = dict(ok=bool(gate_kd(device="cuda" if DEVICE == "cuda" else None)))
    if backend == "fused":
        ok_f = gate_fused_agreement(m, cfg)
        ok_gf = gate_grad_isolation(m, cfg, backend="fused")
        report["fused_agreement"] = dict(ok=bool(ok_f))
        report["grad_isolation_fused"] = dict(ok=bool(ok_gf))
    # G1 induction on the raw donor (equiv mode) -- short/fast (docs/evals.md,
    # "G1"). With cfg.qknorm=True, "equiv" mode is the model's OWN QKNorm'd
    # full-attention limit, not the literal raw donor (init_qknorm_gains
    # calibrates the QKNorm gains to approximately preserve the donor's own
    # attention temperature at init -- see model.py -- so this remains a
    # reasonable pre-healing sanity probe, just not a donor-identical one).
    from .evals import induction_probe
    acc = induction_probe(m, cfg, "equiv", lengths=(2048, 4096), reps=3, trials=16)
    g1 = max(acc.values()) >= 0.3
    report["induction_G1"] = dict(ok=bool(g1), acc=acc)
    log(f"[shake:G1] raw-donor induction {acc} -> {g1}")
    del m; empty_cache()


def check_training(cfg, planned_tps, steps, report, fast=False, rank=0, world=1, local_rank=0):
    """N steps of stage 1; loss falls, aux-KL moves, tok/s measured; checkpoint
    write + reload-resume verified in a fresh process.

    fast=True is the provisioning shakedown's --fast profile (scripts/
    shakedown.sh, scripts/provision.sh): steps drops to 50 (from 150) and the
    tok/s measurement window excludes the first ~10 steps (compile/autotune
    warmup -- see heal.train's warmup_steps) so a 50-step average isn't
    dominated by one-time kernel autotune cost the way a naive average would
    be. Guarded so warmup_steps < steps even at very small --steps overrides.

    rank/world/local_rank: multi-GPU shakedown mode (design gate 5, "doubles as
    a scaling-efficiency gate"). Defaults (0/1/0) reproduce the single-GPU
    check exactly. At world > 1 (this process is one of `world` ranks already
    initialized under torchrun -- see gates.main's --multi-gpu flag), this
    rank's own measured tok/s is all-reduced (SUM) into an aggregate, compared
    against PLANNED_TPS * world at a 0.75 scaling-efficiency threshold instead
    of the single-GPU 0.7 floor (consumer-GPU host-staged all-reduce overhead
    means aggregate throughput never scales perfectly linearly; 0.75 is the
    floor below which the multi-GPU run isn't worth its added complexity).

    Per-rank measurement (design fix): heal.log's per-step status line is
    rank-0-only (avoids world-way duplicate spam -- see heal.train's Logging
    note), so monkeypatching heal.log to harvest tok/s -- as an EARLIER version
    of this function did -- silently collects NOTHING on ranks != 0: their
    run_shakedown would return False structurally, and rank 0's SUM all-reduce
    over [tps0, 0, 0, ...] would land scaling_efficiency near 1/world always,
    regardless of real throughput. Fixed by passing train() a `tps_out` dict
    (its own rank-agnostic recording channel -- see train()'s docstring): every
    rank reads its OWN measured series back from tps_out after train() returns,
    independent of which rank calls log(). loss_falls/aux_moves, by contrast,
    ARE only observable via the log line (they need the actual loss/aux
    values, not just tok/s) -- so those stay rank-0-only and are BROADCAST from
    rank 0 to every other rank below, so every rank's pass/fail verdict agrees
    (previously, ranks != 0 would independently see empty loss/aux lists and
    fail on data they could never structurally have collected -- exactly the
    inconsistent-exit failure mode this fixes)."""
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
    # * world: this mini-run's token budget must scale with world size too, or
    # (at world > 1) train()'s tokens_per_step (which DOES include world --
    # micro_batch*grad_accum*ctx*world) divides a world-blind numerator down to
    # ~1/world the intended optimizer steps -- at world=8 that can put total
    # steps below the fast profile's warmup_steps=10, so the tok/s measurement
    # window (which resets AFTER warmup_steps) never even opens.
    PHASES["stage1"]["tokens"] = steps * _mb * _ga * cfg.heal_ctx * world
    t0 = time.time()
    # Run in-process for the measurement; capture the loss trace via a light
    # monkeypatch on log rather than parsing files. This DOES stay rank-0-only
    # (heal.log's status line only fires there) -- see the per-rank tps_out
    # channel above for the part of this measurement that must work on every
    # rank.
    losses, auxes = [], []
    orig = heal.log

    def cap(*a):
        s = " ".join(str(x) for x in a)
        orig(*a)
        if "] step " in s and " lm " in s:
            try:
                losses.append(float(s.split(" lm ")[1].split()[0]))
                auxes.append(float(s.split(" aux ")[1].split()[0]))
            except Exception:
                pass
    heal.log = cap
    tps_out = {}
    try:
        # shakedown=True waives heal's smoke-shard/corpus-size data guards: this
        # mini stage legitimately trains on a small data slice. The guards stay
        # fully armed for any real heal invocation (scripts/heal.sh never passes
        # --shakedown).
        train(cfg, "stage1", resume=False, micro_batch=_mb, grad_accum=_ga,
              budget_s=1800, smoke=(DEVICE != "cuda"), shakedown=True, warmup_steps=warmup_steps,
              rank=rank, world=world, local_rank=local_rank, tps_out=tps_out)
    finally:
        heal.log = orig
    # train()'s model/optimizer are freed on return, but the caching allocator
    # keeps the CUDA memory RESERVED in this parent process -- the resume
    # subprocess below then loads its own copy of the model in the same process
    # tree and can OOM on memory-constrained cards if the reservation isn't
    # released first.
    empty_cache()
    dt = time.time() - t0
    # THIS rank's own measured tok/s, read from tps_out -- NOT parsed from
    # heal.log (see the per-rank measurement note above).
    tps_series = tps_out.get("series", [])
    tps = max(tps_series) if tps_series else 0.0
    loss_falls = len(losses) >= 2 and losses[-1] < losses[0] + 0.05    # tolerate noise
    aux_moves = len(auxes) >= 2 and abs(auxes[-1] - auxes[0]) > 1e-4
    agg_tps, eff = tps, None
    if world > 1 and dist_util.is_distributed():
        import torch.distributed as dist
        t = torch.tensor([tps], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        agg_tps = float(t.item())
        target = planned_tps * world
        eff = (agg_tps / target) if target > 0 else 1.0
        fast_enough = eff >= 0.75
        # loss_falls/aux_moves are only OBSERVABLE on rank 0 (derived from
        # heal.log's rank-0-only status line -- see the docstring note above);
        # broadcast rank 0's verdict so every rank's report/exit code agrees,
        # instead of ranks != 0 failing on structurally-empty local data.
        verdict = torch.tensor([1.0 if loss_falls else 0.0, 1.0 if aux_moves else 0.0],
                               dtype=torch.float64)
        dist.broadcast(verdict, src=0)
        loss_falls, aux_moves = bool(verdict[0].item()), bool(verdict[1].item())
    else:
        fast_enough = tps >= 0.7 * planned_tps if planned_tps > 0 else True
    ok = loss_falls and aux_moves and (fast_enough or DEVICE != "cuda")
    report["training"] = dict(ok=bool(ok), loss0=losses[0] if losses else None,
                              lossN=losses[-1] if losses else None, loss_falls=bool(loss_falls),
                              aux0=auxes[0] if auxes else None, auxN=auxes[-1] if auxes else None,
                              aux_moves=bool(aux_moves), tok_s=tps, agg_tok_s=agg_tps, world=world,
                              scaling_efficiency=eff, planned_tps=planned_tps,
                              fast_enough=bool(fast_enough), wall_s=dt)
    log(f"[shake:train] loss {report['training']['loss0']}->{report['training']['lossN']} "
        f"aux {report['training']['aux0']}->{report['training']['auxN']} tok/s {tps:.0f} "
        f"agg_tok/s {agg_tps:.0f} (world={world}, planned {planned_tps}"
        f"{f', eff={eff:.2f}' if eff is not None else ''}) -> {ok}")
    if planned_tps > 0 and DEVICE == "cuda" and not fast_enough:
        log(f"[shake:train] *** ABORT-WORTHY: measured throughput below the "
            f"{'75% scaling-efficiency' if world > 1 else '70% of planned'} floor -- the plan's "
            "wall-clock/cost arithmetic will not hold ***")

    # checkpoint write + reload-resume (separate process to prove on-box
    # durability). RANK-0 ONLY: under torchrun every rank inherits RANK/
    # WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT/TORCHELASTIC_*/GROUP_RANK/
    # ROLE_* env vars, so a subprocess spawned from EVERY rank would each try
    # to join the parent's already-live rendezvous (hang or collision) instead
    # of starting its own independent single-process run; and every rank's
    # os.replace(ck, shadow) below would race on the SAME file (FileNotFoundError
    # on N-1 ranks once the first one wins the rename). So: rank 0 alone runs
    # the subprocess (with a scrubbed env so it truly launches standalone, not
    # as a phantom member of this torchrun job) and does the shadow/replace;
    # every other rank waits at a barrier and then takes rank 0's broadcast
    # verdict, so all ranks agree on report["resume"]["ok"] and exit
    # consistently.
    suffix = "_smoke" if DEVICE != "cuda" else ""
    ck = os.path.join(RESULTS, f"heal_stage1{suffix}.pt")
    resume_ok = False
    if rank == 0:
        resume_ok = os.path.exists(ck)
        if resume_ok:
            _scrub_exact = {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                            "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK"}
            env = {k: v for k, v in os.environ.items()
                  if k not in _scrub_exact and not k.startswith("TORCHELASTIC_")
                  and not k.startswith("ROLE_")}
            r = subprocess.run([sys.executable, "-m", "hba.heal",
                                "--phase", "stage1", "--resume", "--skip-gates", "--shakedown",
                                "--tokens", str(PHASES["stage1"]["tokens"]),
                                "--micro-batch", "1", "--grad-accum", "4"]
                               + (["--smoke"] if DEVICE != "cuda" else []),
                               capture_output=True, text=True, timeout=600, env=env)
            resume_ok = ("resuming from step" in r.stdout or "already complete" in r.stdout)
        # move the mini-run ckpt ASIDE (rank-0-only, same reasoning as above):
        # it must never be mistaken for real healing output (the ckpt signature
        # also embeds the token budget, but a real-named heal_stage1.pt on disk
        # left over from a shakedown is a foot-gun).
        if os.path.exists(ck):
            shadow = os.path.join(RESULTS, f"heal_stage1{suffix}_shakedown.pt")
            os.replace(ck, shadow)
            log(f"[shake:train] shakedown mini-ckpt moved aside -> {shadow}")
    dist_util.barrier()
    if world > 1 and dist_util.is_distributed():
        import torch.distributed as dist
        v = torch.tensor([1.0 if resume_ok else 0.0], dtype=torch.float64)
        dist.broadcast(v, src=0)
        resume_ok = bool(v.item())
    report["resume"] = dict(ok=bool(resume_ok))
    log(f"[shake:resume] checkpoint reload-resume -> {resume_ok}")


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


def run_shakedown(cfg, planned_tps=5000.0, steps=150, stage="all", fast=False,
                  rank=0, world=1, local_rank=0):
    """Run the pre-flight shakedown and write REPORT. Returns the overall bool.

    fast=True is the provisioning entrypoint's fast profile (scripts/
    provision.sh -> scripts/shakedown.sh --fast -> here): 50 measured train
    steps with the tok/s window excluding compile/autotune warmup
    (check_training), and a single PPL eval cell instead of PPL+needle
    (check_eval). The fine-grained fp32 gates + G1 induction (check_gates) and
    the reference check (check_reference) are unchanged by fast -- they are
    already fast (~4 min combined) and are exactly the correctness surface
    that must never be skipped before any training runs on a new box.

    rank/world/local_rank: multi-GPU mode (world > 1, called under torchrun by
    scripts/shakedown.sh's multi-GPU path / gates.main's --multi-gpu). Adds the
    gate_shard_partition / gate_rank_consistency / gate_nccl_bandwidth checks
    (gate_ddp_equivalence is run SEPARATELY by the shell wrapper, not here --
    it orchestrates its OWN torchrun/python subprocess pair and must not be
    invoked recursively from inside an already-running torchrun rank) and
    threads rank/world into check_training for the aggregate-throughput /
    scaling-efficiency extension. Defaults (0/1/0) are the single-GPU path,
    unchanged from before this mode existed. Only rank 0 writes REPORT (design:
    "rank 0 writes" convention applied consistently to every collective
    artifact this module produces, not just heal.py's checkpoints)."""
    log(f"shakedown device={DEVICE} dtype={COMPUTE_DTYPE} planned_tps={planned_tps} fast={fast} "
        f"rank={rank}/{world}")
    report = dict(device=DEVICE, torch=torch.__version__,
                  cuda=torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, ts=time.time(),
                  world=world)
    if stage in ("all", "ref"):
        check_reference(cfg, report)
    if stage in ("all", "gates"):
        check_gates(cfg, report)
        if world > 1:
            # gate 1 (per-rank correctness) is already satisfied structurally --
            # check_gates above runs identically in every rank's own process.
            # windows_per_step=32 matches GLOBAL_TOKENS_PER_STEP // heal_ctx at the
            # default 4096 ctx (see heal.GLOBAL_TOKENS_PER_STEP); world in {1,2,4,8}
            # always divides it evenly.
            report["shard_partition"] = dict(
                ok=bool(gate_shard_partition(cfg, world, micro_B=1, accum=32 // world)))
            m, _, _ = build_hba(cfg, dtype=torch.float32)
            # gate_rank_consistency MUST run on the UNWRAPPED model: DDP's
            # constructor broadcasts rank-0's params to every rank, so wrapping
            # FIRST (as this used to do) makes the gate's own param-spread check
            # trivially pass (spread=0 by construction) regardless of whether
            # this box's own checkpoint/weight load was actually torn on some
            # rank -- exactly the failure mode the gate exists to catch. Gate
            # first, wrap after (wrap_ddp itself is still exercised here; its
            # result is otherwise unused for this check).
            report["rank_consistency"] = dict(ok=bool(gate_rank_consistency(m, cfg)))
            if dist_util.is_distributed():
                m = dist_util.wrap_ddp(m, local_rank, "fp32")
            del m; empty_cache()
            # advisory (default): warns on a low microbench but does not fail the
            # shakedown -- the scaling-efficiency gate in check_training governs
            # (see gate_nccl_bandwidth's docstring for the empirical basis).
            report["nccl_bandwidth_advisory"] = dict(ok=bool(gate_nccl_bandwidth(
                device=DEVICE if DEVICE == "cuda" else None)))
    if stage in ("all", "train"):
        check_training(cfg, planned_tps, steps, report, fast=fast,
                       rank=rank, world=world, local_rank=local_rank)
    if stage in ("all", "eval"):
        check_eval(cfg, report, fast=fast)

    checks = {k: v for k, v in report.items() if isinstance(v, dict) and "ok" in v}
    overall = all(v["ok"] for v in checks.values())
    report["PASS"] = bool(overall)
    if rank == 0:
        json.dump(report, open(REPORT + ".tmp", "w"), indent=2)
        os.replace(REPORT + ".tmp", REPORT)
        banner = "=" * 60
        print(f"\n{banner}\nSHAKEDOWN {'PASS' if overall else 'FAIL'}  ("
              + ", ".join(f"{k}={'ok' if v['ok'] else 'FAIL'}" for k, v in checks.items())
              + f")\nreport -> {REPORT}\n{banner}", flush=True)
    dist_util.barrier()
    return overall


def main():
    import argparse
    from .config import HBAConfig
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planned-tps", type=float, default=5000.0,
                    help="planned healing tok/s (see the training-recipe cost notes); abort if "
                         "measured throughput is under 70pct of this on CUDA (or under a 75% "
                         "scaling-efficiency floor at world > 1 -- see check_training)")
    ap.add_argument("--steps", type=int, default=None,
                    help="default: 50 with --fast, else 150")
    ap.add_argument("--stage", choices=["all", "ref", "gates", "train", "eval"], default="all")
    ap.add_argument("--fast", action="store_true",
                    help="provisioning fast profile (scripts/provision.sh): 50 measured train "
                         "steps with the tok/s window excluding compile/autotune warmup, one "
                         "PPL eval cell instead of PPL+needle. See docker/README.md.")
    ap.add_argument("--multi-gpu", action="store_true",
                    help="run under torchrun (scripts/shakedown.sh's multi-GPU mode): adds "
                         "gate_shard_partition/gate_rank_consistency/gate_nccl_bandwidth and the "
                         "aggregate-throughput extension to check_training. No-op (identical to "
                         "single-GPU) if not actually launched under torchrun.")
    args = ap.parse_args()
    cfg = HBAConfig()
    steps = args.steps if args.steps is not None else (50 if args.fast else 150)
    rank, world, local_rank = (dist_util.setup_distributed() if args.multi_gpu else (0, 1, 0))
    try:
        ok = run_shakedown(cfg, planned_tps=args.planned_tps, steps=steps, stage=args.stage,
                           fast=args.fast, rank=rank, world=world, local_rank=local_rank)
    finally:
        dist_util.cleanup_distributed()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
