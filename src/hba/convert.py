"""Donor conversion entrypoint: stage 0 (distill-init) and the pre-training gates
+ healing-seed checkpoint (docs/training-recipe.md, stage table).

Two independent flows live in this module:

  --distill-init   Stage 0. Initialize each per-(layer, KV-head) SlotSummarizer by
                    distilling against the FROZEN donor's own attention mass (KL
                    to the group-averaged NoPE content-mass teacher -- exactly the
                    teacher healing's aux-KL uses). This is an INIT, not a result:
                    docs/training-recipe.md is explicit that position-free routing
                    "cannot be distilled onto frozen weights" and recovers almost
                    none of the routing-recall gap -- it puts the summarizers near
                    mean-pool quality with differentiated slots so the aux loss
                    has a sane starting point once healing (stage 1+) co-trains
                    Q/K. Cheap: hours on CPU/laptop-class hardware, no GPU needed.

  --gates / --save-init / --export-ref
                    Load the donor, swap in the HBA stack, optionally load the
                    stage-0 distilled summarizers, and run the correctness gates
                    (gates.run_all_gates): equivalence, causality, path-
                    equivalence, gradient isolation (+ fused-vs-naive agreement if
                    the fused backend is resolved). A wiring bug or gradient leak
                    makes every downstream retrieval number fake, so healing
                    refuses to start unless all gates pass (docs/training-
                    recipe.md, "Refuse to start"). --save-init writes the healing
                    seed checkpoint (summarizer state + config).

Usage:
  python -m hba.convert --distill-init [--smoke]
  python -m hba.convert --gates [--smoke]
  python -m hba.convert --save-init [--from-distill] [--smoke]
  python -m hba.convert --export-ref            # fp32 reference logits, for gates.check_reference
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from .config import (CORPUS_DIR, DATA, DEVICE, DISTILL_PATH, INIT_PATH, REF_PATH, HBAConfig, log,
                     save_ckpt_atomic, smoke_config)
from .model import build_hba, load_donor
from .attention import rope_tables
from .summarizer import SlotSummarizer

REF_N = 128           # short fixed-input reference; kept small so it's cheap to ship/compare
REF_SEED = 70721

DISTILL_NQ = 512                      # query band = last NQ positions of each window
DISTILL_WIN_FRACS = [0.2, 0.5, 0.8]


# ============================================================ gates / save-init
@torch.no_grad()
def export_reference(model, cfg):
    """Save fp32 donor + HBA-equiv logits on a FIXED input so gates.check_reference
    can verify a swap on a new machine reproduces this one (fp32 tight; bf16
    loose). Run in fp32 on the reference machine.

    With cfg.qknorm=True, hba_equiv_logits is no longer a donor-equivalence
    reference (QKNorm deliberately changes Q/K statistics away from the donor's
    own -- docs/design.md, "Softmax length-calibration") -- it becomes the
    QKNorm'd model's OWN fp32 output, an INTERNAL consistency check that a new
    machine/build reproduces THIS build, not that HBA reproduces the donor. See
    gates.check_reference's docstring for exactly which of its comparisons stay
    blocking vs become informational under qknorm=True. Re-export whenever
    cfg.qknorm flips (a qknorm=OFF export cannot validate a qknorm=ON build, or
    vice versa)."""
    dev = next(model.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(REF_SEED)
    ids = torch.randint(0, cfg.vocab_size, (1, REF_N), generator=g)
    cos, sin = rope_tables(REF_N, cfg.head_dim, cfg.rope_theta, dev)
    donor = model.donor(ids.to(dev)).logits.float().cpu()
    hba = model(ids.to(dev), cos, sin, cfg.mem_elem_cap, mode="equiv").float().cpu()
    save_ckpt_atomic({"ids": ids, "donor_logits": donor, "hba_equiv_logits": hba,
                      "n": REF_N, "seed": REF_SEED, "cfg": cfg.__dict__}, REF_PATH)
    log(f"[convert] exported reference logits (n={REF_N}) -> {REF_PATH}")


def summarizer_state(model):
    return {k: v.cpu() for k, v in model.summarizers.state_dict().items()}


def load_summarizers(model, path):
    sd = torch.load(path, map_location="cpu")
    model.summarizers.load_state_dict(sd["summarizers"] if "summarizers" in sd else sd)
    log(f"loaded distilled summarizers from {path}")


def run_gates(model, tok, cfg):
    from .gates import run_all_gates
    return run_all_gates(model, tok, cfg)


# ================================================= stage 0: distill-init =======
def _clean_gutenberg(t):
    """Strip Project Gutenberg boilerplate from a raw .txt if present; a no-op on
    plain prose that doesn't have the markers."""
    s = t.find("*** START OF")
    if s != -1:
        t = t[t.find("\n", s) + 1:]
    e = t.find("*** END OF")
    if e != -1:
        t = t[:e]
    return t.strip()


def _distill_windows(tok, n_ctx, per_book, corpus_dir, exclude=()):
    paths = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
    paths = [p for p in paths if not any(x in os.path.basename(p).lower() for x in exclude)]
    outs = []
    for p in paths:
        text = _clean_gutenberg(open(p, encoding="utf-8", errors="ignore").read())
        for frac in DISTILL_WIN_FRACS[:per_book]:
            start = int(frac * max(len(text) - 300000, 1))
            ids = tok(text[start:start + 400000], return_tensors="pt",
                      truncation=True, max_length=n_ctx).input_ids
            if ids.shape[1] >= n_ctx:
                outs.append(ids[:, :n_ctx])
    return outs


@torch.no_grad()
def _harvest(donor, cfg, ids, n_ctx):
    """One donor forward with pre-RoPE q/k hooks -> per (layer, KV head): grouped
    NoPE query qg [NL,Hkv,NQ,dh], per-block NoPE keys kblk [NL,Hkv,nb,B,dh], teacher
    pstar [NL,Hkv,NQ,nb], keep mask [NL,Hkv,NQ]. Matches the healing teacher
    exactly (attention.hba_attention_dense's pstar computation)."""
    Hq, Hkv, G, dh = cfg.n_heads, cfg.n_kv, cfg.G, cfg.head_dim
    Bk, W, S = cfg.block, cfg.window, cfg.sinks
    NL = cfg.n_layers
    nb = n_ctx // Bk
    dev = next(donor.parameters()).device
    Q, K, handles = {}, {}, []

    def mk(store, i):
        def hook(mod, inp, out):
            store[i] = out.detach().float()
        return hook
    for i, lyr in enumerate(donor.model.layers):
        handles.append(lyr.self_attn.q_proj.register_forward_hook(mk(Q, i)))
        handles.append(lyr.self_attn.k_proj.register_forward_hook(mk(K, i)))
    donor(ids.to(dev))
    for h in handles:
        h.remove()

    band_s = n_ctx - DISTILL_NQ
    scale = dh ** -0.5
    qg_o = torch.zeros(NL, Hkv, DISTILL_NQ, dh)
    kb_o = torch.zeros(NL, Hkv, nb, Bk, dh)
    ps_o = torch.zeros(NL, Hkv, DISTILL_NQ, nb)
    keep_o = torch.zeros(NL, Hkv, DISTILL_NQ, dtype=torch.bool)
    # candidate blocks per query in the band (blocks fully before the window)
    i = torch.arange(band_s, n_ctx, device=dev)[:, None]
    barr = torch.arange(nb, device=dev)
    cand = (barr[None, :] >= 1) & ((barr[None, :] + 1) * Bk <= (i - W + 1))       # [NQ, nb]
    for L in range(NL):
        q = Q[L][0].view(n_ctx, Hq, dh)
        k = K[L][0].view(n_ctx, Hkv, dh)
        qg = q.view(n_ctx, Hkv, G, dh).sum(2)[band_s:]                            # [NQ,Hkv,dh]
        # per-query-head NoPE content mass -> group average -> per KV head
        kexp = k.repeat_interleave(G, dim=1)                                     # [n,Hq,dh]
        qb = q[band_s:]                                                          # [NQ,Hq,dh]
        s = torch.einsum("qhd,khd->hqk", qb, kexp) * scale                       # [Hq,NQ,n]
        # restrict to candidate keys (blocks before window); causal is implied by cand
        cand_key = cand.repeat_interleave(Bk, dim=-1)                            # [NQ,n]
        s = s.masked_fill(~cand_key[None], float("-inf"))
        w = s.softmax(-1)
        pmass = w.view(Hq, DISTILL_NQ, nb, Bk).sum(-1)                           # [Hq,NQ,nb]
        pstar = pmass.view(Hkv, G, DISTILL_NQ, nb).mean(1)                       # [Hkv,NQ,nb]
        rmass = pstar.sum(-1)                                                    # routable mass
        pstar = pstar / pstar.sum(-1, keepdim=True).clamp_min(1e-9)
        qg_o[L] = qg.transpose(0, 1).cpu()
        kb_o[L] = k.view(nb, Bk, Hkv, dh).permute(2, 0, 1, 3).cpu()
        ps_o[L] = pstar.cpu()
        keep_o[L] = (rmass >= 0.05).cpu()               # [Hkv, NQ]
        del Q[L], K[L]
    return dict(qg=qg_o, kb=kb_o, pstar=ps_o, keep=keep_o)


def _kl_loss(scores, pstar):
    logp = torch.log_softmax(scores, dim=-1)
    term = torch.where(pstar > 0, pstar * logp, torch.zeros_like(logp))
    return -term.sum(-1).mean()


def _train_summarizers(cfg, cache, steps, lr, batch, dev):
    """Train one SlotSummarizer per layer (all KV heads jointly) on the harvested cache."""
    NL, Hkv, dh, Bk = cfg.n_layers, cfg.n_kv, cfg.head_dim, cfg.block
    scale = dh ** -0.5
    summs = nn.ModuleList(SlotSummarizer(cfg) for _ in range(NL)).to(dev)
    qg = torch.stack([c["qg"] for c in cache]).to(dev)
    kb = torch.stack([c["kb"] for c in cache]).to(dev)
    ps = torch.stack([c["pstar"] for c in cache]).to(dev)
    keep = torch.stack([c["keep"] for c in cache]).to(dev)
    NW = qg.shape[0]
    for L in range(NL):
        summ = summs[L]
        opt = torch.optim.Adam(summ.parameters(), lr=lr)
        first = last = float("nan")
        for step in range(steps):
            wi = int(torch.randint(NW, (1,)).item())
            # summarize this window's block keys -> [1,Hkv,nb,m,dh]
            kw = kb[wi, L].permute(1, 0, 2, 3).reshape(1, -1, Hkv, dh)           # [1,n,Hkv,dh]
            S = summ.summarize(kw, Bk)                                           # [1,Hkv,nb,m,dh]
            loss = 0.0
            for g in range(Hkv):
                kept = torch.where(keep[wi, L, g])[0]
                if len(kept) < 10:
                    continue
                bidx = kept[torch.randint(len(kept), (min(batch, len(kept)),), device=dev)]
                qsel = qg[wi, L, g][bidx][None]                                  # [1,nq,dh]
                sc = torch.einsum("bqd,bcmd->bqcm", qsel, S[:, g]).amax(-1) * scale  # [1,nq,nb]
                loss = loss + _kl_loss(sc[0], ps[wi, L, g][bidx])
            if not torch.is_tensor(loss):
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            if step == 0:
                first = float(loss)
            last = float(loss)
        log(f"[distill] layer {L:2d}: KL {first:.3f} -> {last:.3f}")
    return summs


def run_distill_init(cfg, n_ctx, per_book, steps, batch, corpus_dir,
                     needle_book=None):
    donor, tok = load_donor(dtype=torch.float32)
    wins = _distill_windows(tok, n_ctx, per_book, corpus_dir, exclude=(needle_book,) if needle_book else ())
    if not wins:
        log(f"no windows harvested ({corpus_dir}/*.txt missing?)"); sys.exit(1)
    log(f"harvesting {len(wins)} windows @ ctx {n_ctx} ...")
    cache = []
    for w, ids in enumerate(wins):
        t0 = time.time()
        cache.append(_harvest(donor, cfg, ids, n_ctx))
        log(f"  window {w+1}/{len(wins)} harvested in {time.time()-t0:.0f}s")
    del donor
    if DEVICE == "mps":
        torch.mps.empty_cache()

    summs = _train_summarizers(cfg, cache, steps, lr=1e-3, batch=batch, dev=DEVICE)
    save_ckpt_atomic({"summarizers": summs.state_dict(), "cfg": cfg.__dict__,
                      "n_windows": len(wins), "ctx": n_ctx}, DISTILL_PATH)
    log(f"[distill] wrote {DISTILL_PATH}")


# ============================================================== CLI ============
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--save-init", action="store_true")
    ap.add_argument("--from-distill", action="store_true",
                    help="load distilled Stage-0 summarizers into the init checkpoint")
    ap.add_argument("--export-ref", action="store_true",
                    help="export fp32 reference logits (real cfg) for gates.check_reference")
    ap.add_argument("--distill-init", action="store_true",
                    help="run Stage 0 (frozen-donor summarizer distillation) instead of the "
                         "gates/save-init flow")
    ap.add_argument("--needle-book", default=os.environ.get("HBA_NEEDLE_BOOK"),
                    help="basename (substring) of the held-out needle-eval book to EXCLUDE from distill-init harvesting")
    ap.add_argument("--corpus-dir", default=CORPUS_DIR,
                    help="directory of *.txt documents for --distill-init (default: "
                         "$HBA_CORPUS_DIR or <data>/corpus)")
    ap.add_argument("--ctx", type=int, default=None, help="--distill-init: harvest context length")
    ap.add_argument("--per-book", type=int, default=None, help="--distill-init: windows per document")
    ap.add_argument("--steps", type=int, default=None, help="--distill-init: Adam steps per layer")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = smoke_config() if args.smoke else HBAConfig()

    if args.distill_init:
        n_ctx = args.ctx or (512 if args.smoke else cfg.heal_ctx)
        per_book = args.per_book or (1 if args.smoke else 3)
        steps = args.steps or (30 if args.smoke else 1500)
        batch = 64 if args.smoke else 256
        log(f"distill-init device={DEVICE} smoke={args.smoke} ctx={n_ctx} per_book={per_book} "
            f"steps={steps} corpus={args.corpus_dir}")
        run_distill_init(cfg, n_ctx, per_book, steps, batch, args.corpus_dir,
                         needle_book=args.needle_book)
        return

    # gates want fp32 for a tight equivalence bound; healing uses bf16 on CUDA.
    dtype = torch.float32
    log(f"convert device={DEVICE} smoke={args.smoke} dtype={dtype}")
    model, tok, cfg = build_hba(cfg, dtype=dtype)

    if args.from_distill and os.path.exists(DISTILL_PATH):
        load_summarizers(model, DISTILL_PATH)

    ok = True
    if args.gates or not (args.save_init or args.export_ref):
        ok = run_gates(model, tok, cfg)

    if args.export_ref:
        export_reference(model, cfg)

    if args.save_init:
        save_ckpt_atomic({"summarizers": summarizer_state(model), "cfg": cfg.__dict__,
                          "from_distill": args.from_distill and os.path.exists(DISTILL_PATH)},
                         INIT_PATH)
        log(f"[convert] wrote healing seed -> {INIT_PATH}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
