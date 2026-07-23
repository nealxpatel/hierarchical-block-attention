"""Evaluation suite (docs/evals.md): capability gates before any expensive sweep.
Per-cell atomic + resumable; each stage runnable in its own process.

Gates (cheap, ~minutes; a FAIL here stops the sweep):
  G1  induction probe on the RAW DONOR at 2-8K (pair repeated 3x then queried
      once). Must pass -- if the donor can't do it, the comparison downstream is
      unmeasurable, not failed.
  G2  the same probe on the CONVERTED model (after any healing stage). If
      conversion destroyed induction, STOP and diagnose -- do not sweep.

Then, only if the gates pass:
  PPL       held-out web/books/code, converted vs raw donor (bar: within 5-10%).
  NEEDLE    key->value retrieval in held-out real text at {4K,16K,32K,64K,128K}
            for donor / donor+YaRN / converted(flat) / converted(hier); 3 seeds
            -> mean +/- SE.
  BENCH     HellaSwag / ARC-e / PIQA loglikelihood, donor vs converted
            (self-contained scorer; skips cleanly if the datasets aren't
            downloadable).
  HIER      hierarchy fidelity + comparisons/query at 64K/128K.

Baselines all share ONE code path (model.HBAModel): raw-donor weights in `equiv`
mode with native RoPE = donor; with yarn_theta(n) cos/sin = donor+YaRN; healed
weights in `eval`/`eval_hier` = converted. See docs/evals.md for the full
protocol and validation-scale results.

Usage:
  python -m hba.evals --stage gate --which donor       # G1
  python -m hba.evals --stage gate --which converted   # G2 (needs a heal checkpoint)
  python -m hba.evals --stage ppl
  python -m hba.evals --stage needle [--length 65536] [--method H_hier]
  python -m hba.evals --stage bench
  python -m hba.evals --stage hier
  python -m hba.evals --stage all
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .attention import build_super, hier_select, rope_tables, yarn_theta
from .config import DATA, DEVICE, RESULTS, HBAConfig, empty_cache, log, smoke_config
from .model import build_hba, load_donor
from .summarizer import grouped_query

RES = os.path.join(RESULTS, "hba_results.json")    # smoke runs redirect to _smoke (see main):
                                                    # smoke cells must NEVER pre-fill ("skip
                                                    # cached") the real sweep
EVAL_LENGTHS = (4096, 16384, 32768, 65536, 131072)
EX_PER_LEN = {4096: 48, 16384: 32, 32768: 24, 65536: 12, 131072: 8}
FWD_BATCH = {4096: 4, 16384: 2, 32768: 1, 65536: 1, 131072: 1}
NEEDLE_PAIRS = 6
SEEDS = (0, 1, 2)
# Ordered stage2 seed the eval suite prefers, worst-to-best; converted_model()
# falls back down this chain if the requested/latest stage isn't checkpointed.
STAGE_CHAIN = ("stage1", "stage2", "stage3")


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def save_json(o, p):
    tmp = p + ".tmp"
    json.dump(o, open(tmp, "w"), indent=2)
    os.replace(tmp, p)


# ------------------------------------------------------------- model loaders ---
def donor_model(cfg):
    """Raw-donor HBAModel (equiv mode) -> donor / donor+YaRN baselines."""
    m, tok, cfg = build_hba(cfg, dtype=torch.float32)
    m.eval()
    return m


def converted_model(cfg, phase="stage3", smoke=False, allow_raw=None):
    """Healed HBAModel from a heal checkpoint. Preference order stage3 -> stage2
    -> stage1 (logged loudly). HARD-FAILS if no heal checkpoint exists, unless
    allow_raw (plumbing/smoke only) -- evaluating a raw donor as 'converted' would
    silently produce a plausible-looking fake verdict."""
    if allow_raw is None:
        allow_raw = smoke
    m, tok, cfg = build_hba(cfg, dtype=torch.float32)
    sfx = "_smoke" if smoke else ""
    # an explicitly requested stage is pinned exactly; the default walks the chain newest-first
    for ph in ([phase] if phase != STAGE_CHAIN[-1] else list(reversed(STAGE_CHAIN))):
        ckpt = os.path.join(RESULTS, f"heal_{ph}{sfx}.pt")
        if os.path.exists(ckpt):
            ck = torch.load(ckpt, map_location=DEVICE)
            m.load_state_dict(ck["model"])
            note = "" if ck.get("done") else "  *** PARTIAL (done=False) ***"
            if ph != phase:
                note += f"  *** FELL BACK TO {ph} (no later-stage checkpoint) ***"
            log(f"[eval] loaded converted weights <- {ckpt}{note}")
            m.eval()
            return m
    if not allow_raw:
        raise RuntimeError(f"no heal checkpoint (heal_stage3{sfx}.pt / heal_stage2{sfx}.pt / "
                           f"heal_stage1{sfx}.pt) -- refusing to evaluate a raw donor as "
                           "'converted' (fake numbers); run healing first (plumbing tests: --smoke)")
    log("[eval] WARNING: no heal ckpt -- using distilled/raw model (PLUMBING ONLY)")
    from .config import INIT_PATH
    if os.path.exists(INIT_PATH):
        m.summarizers.load_state_dict(torch.load(INIT_PATH, map_location="cpu")["summarizers"])
    m.eval()
    return m


# ------------------------------------------------------------- G1/G2 induction -
@torch.no_grad()
def induction_probe(model, cfg, mode, lengths=(2048, 4096, 8192), reps=3, trials=32):
    """Plant a random (key,val) token bigram `reps` times in a random-token
    haystack, then present the key once more at the end; PASS if the model's
    argmax at that position is val. Chance ~ 1/V.
    mode: 'equiv' (raw donor via native RoPE) or 'eval' (converted)."""
    dev = next(model.parameters()).device
    V = cfg.vocab_size
    out = {}
    for n in lengths:
        cos, sin = rope_tables(n, cfg.head_dim, cfg.rope_theta, dev)
        hit = 0
        for t in range(trials):
            g = np.random.default_rng(1234 + t + n)
            ids = g.integers(100, V, size=n).astype(np.int64)
            key = int(g.integers(100, V)); val = int(g.integers(100, V))
            # plant reps copies of (key,val) spread through the first 90%, then key at the end
            slots = np.linspace(0.05, 0.9, reps)
            for s in slots:
                p = int(s * (n - 4))
                ids[p] = key; ids[p + 1] = val
            ids[n - 1] = key
            x = torch.from_numpy(ids)[None].to(dev)
            logits = model(x, cos, sin, cfg.mem_elem_cap, mode=mode, tail=2)
            pred = int(logits[0, -1].argmax())
            hit += (pred == val)
        acc = hit / trials
        out[n] = acc
        log(f"[induction:{mode}] n={n} reps={reps}: acc={acc:.3f} (chance~{1/V:.1e})")
    return out


# ------------------------------------------------------------- needle in text --
def _pools(cfg):
    v = cfg.vocab_size
    lo = max(100, v // 4)
    keys = np.arange(lo, lo + (v - lo) // 2)
    vals = np.arange(lo + (v - lo) // 2, v)
    return keys, vals


def make_needle_batch(cfg, n, bsz, seed, stream):
    keys_pool, vals_pool = _pools(cfg)
    rng = np.random.default_rng(seed)
    L = len(stream)
    Q = 1 + 2 * NEEDLE_PAIRS
    body_len = n - Q
    SEP = 2
    ids = np.empty((bsz, n), dtype=np.int64)
    mask = np.zeros((bsz, n), dtype=bool)
    win = cfg.window
    for b in range(bsz):
        off = int(rng.integers(0, max(1, L - body_len - 1)))
        body = stream[off: off + body_len].astype(np.int64).copy()
        keys = rng.choice(keys_pool, size=NEEDLE_PAIRS, replace=False)
        vals = rng.choice(vals_pool, size=NEEDLE_PAIRS, replace=False)
        max_depth = body_len - win - 4
        depths = np.linspace(0.05, 0.85, NEEDLE_PAIRS); rng.shuffle(depths)
        used = set()
        for j in range(NEEDLE_PAIRS):
            p = int(depths[j] * max_depth)
            while p in used or p + 1 in used:
                p += 2
            used.add(p); used.add(p + 1)
            body[p] = keys[j]; body[p + 1] = vals[j]
        query = [SEP]; vpos = []
        for j in range(NEEDLE_PAIRS):
            query.append(int(keys[j])); vpos.append(body_len + len(query)); query.append(int(vals[j]))
        row = np.concatenate([body, np.asarray(query, dtype=np.int64)])
        ids[b] = row[:n]
        for vp in vpos:
            if vp < n:
                mask[b, vp] = True
    return torch.from_numpy(ids), torch.from_numpy(mask)


@torch.no_grad()
def needle_accuracy(model, cfg, n, method, seed, stream):
    theta = yarn_theta(cfg, n) if method == "D_yarn" else cfg.rope_theta
    mode = {"D": "equiv", "D_yarn": "equiv", "H_flat": "eval", "H_hier": "eval_hier"}[method]
    cos, sin = rope_tables(n, cfg.head_dim, theta, DEVICE)
    bsz, n_ex = FWD_BATCH[n], EX_PER_LEN[n]
    tail = 2 * NEEDLE_PAIRS + 4          # all value-answer positions sit in the appended query tail
    correct = total = done = 0
    while done < n_ex:
        b = min(bsz, n_ex - done)
        ids, mask = make_needle_batch(cfg, n, b, seed * 10_000 + n + done, stream)
        ids = ids.to(DEVICE)
        logits = model(ids, cos, sin, cfg.mem_elem_cap, mode=mode, tail=tail)   # [b, tail, V]
        pred = logits[:, :-1].argmax(-1)                                        # predicts abs n-tail+i+1
        tgt = ids[:, n - tail + 1:]
        m = mask[:, n - tail + 1:].to(DEVICE)
        correct += int(((pred == tgt) & m).sum()); total += int(m.sum()); done += b
        empty_cache()
    return correct / max(1, total)


# ------------------------------------------------------------- perplexity ------
@torch.no_grad()
def perplexity(model, cfg, shard, mode="eval", max_windows=100, ctx=None):
    # cap ctx/bsz so full-vocab logits stay a manageable size; PPL at 2048 is the
    # quality anchor even though healing runs at heal_ctx.
    ctx = min(ctx or cfg.heal_ctx, 2048)
    data = np.memmap(shard, dtype=np.uint32, mode="r")
    W = min((len(data) - 1) // ctx, max_windows)
    cos, sin = rope_tables(ctx, cfg.head_dim, cfg.rope_theta, DEVICE)
    tot_nll = tot = 0
    i, bsz = 0, 2
    while i < W:
        b = min(bsz, W - i)
        ids = torch.from_numpy(np.stack([data[(i + k) * ctx:(i + k) * ctx + ctx + 1].astype(np.int64)
                                         for k in range(b)])).to(DEVICE)
        logits = model(ids[:, :-1], cos, sin, cfg.mem_elem_cap, mode=mode)
        nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                              ids[:, 1:].reshape(-1), reduction="sum")
        tot_nll += float(nll); tot += ids[:, 1:].numel(); i += b
        empty_cache()
    return math.exp(tot_nll / max(1, tot))


# ------------------------------------------------------------- benchmark battery
@torch.no_grad()
def mc_benchmark(model, tok, cfg, name, mode="eval", limit=500):
    """Multiple-choice loglikelihood scorer (HellaSwag/ARC-e/PIQA). Returns
    accuracy or None if the dataset can't be loaded offline."""
    try:
        from datasets import load_dataset
        if name == "hellaswag":
            ds = load_dataset("hellaswag", split="validation")
            get = lambda r: (r["ctx"], r["endings"], int(r["label"]))
        elif name == "arc_easy":
            ds = load_dataset("ai2_arc", "ARC-Easy", split="validation")
            get = lambda r: (r["question"], r["choices"]["text"],
                             r["choices"]["label"].index(r["answerKey"]))
        elif name == "piqa":
            ds = load_dataset("piqa", split="validation")
            get = lambda r: (r["goal"], [r["sol1"], r["sol2"]], int(r["label"]))
        else:
            return None
    except Exception as e:
        log(f"[bench] {name} unavailable ({type(e).__name__}: {e}) -- skipping")
        return None
    correct = tot = 0
    for r in ds.select(range(min(limit, len(ds)))):
        try:
            ctx, choices, gold = get(r)
        except Exception:
            continue
        scores = []
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
        for ch in choices:
            ci = tok(ctx, add_special_tokens=False).input_ids
            ai = tok(" " + ch, add_special_tokens=False).input_ids
            seq = ci + ai
            n = len(seq)
            # the HBA eval path requires n % block == 0; right-pad (causal
            # attention -> logits at positions < n are unaffected by the pad) and
            # score only the real span.
            n_pad = n if mode == "equiv" else ((n + cfg.block - 1) // cfg.block) * cfg.block
            ids = torch.tensor(seq + [pad_id] * (n_pad - n), device=DEVICE)[None]
            cos, sin = rope_tables(n_pad, cfg.head_dim, cfg.rope_theta, DEVICE)
            logits = model(ids, cos, sin, cfg.mem_elem_cap, mode=mode)
            lp = torch.log_softmax(logits[0, :n - 1].float(), -1)
            tgt = ids[0, 1:n]
            span = lp[torch.arange(n - 1, device=DEVICE), tgt][len(ci) - 1:]
            scores.append(float(span.mean()))          # length-normalized loglik
        correct += (int(np.argmax(scores)) == gold); tot += 1
    return correct / max(1, tot)


# ------------------------------------------------------------- hierarchy -------
@torch.no_grad()
def hier_fidelity(model, cfg, n, seed, stream):
    ids, _ = make_needle_batch(cfg, n, FWD_BATCH[n], seed * 7 + n, stream)
    ids = ids.to(DEVICE)
    scale = cfg.head_dim ** -0.5
    nb = n // cfg.block
    kk = min(cfg.k_blocks, nb)
    lyr = model.core.layers[0]
    B = ids.shape[0]
    x = model.core.embed_tokens(ids)
    a = lyr.input_layernorm(x)
    q = lyr.self_attn.q_proj(a).view(B, n, cfg.n_heads, cfg.head_dim)
    k = lyr.self_attn.k_proj(a).view(B, n, cfg.n_kv, cfg.head_dim)
    qg = grouped_query(q, cfg)                                # [B,Hkv,n,dh]
    Sblk = model.summarizers[0].summarize(k, cfg.block)       # [B,Hkv,nb,m,dh]
    barr = torch.arange(nb, device=DEVICE)
    i = torch.tensor([n - 1], device=DEVICE)[:, None]
    cand = (barr[None, :] >= 1) & ((barr[None, :] + 1) * cfg.block <= (i - cfg.window + 1))
    n_cand = int(cand.sum())
    agrees = []
    for g in range(cfg.n_kv):
        Sh = Sblk[:, g]
        q1 = qg[:, g, n - 1][:, None, :]
        bsc = torch.einsum("bcd,bnmd->bcnm", q1, Sh).amax(-1) * scale
        bsc = bsc.masked_fill(~cand[None], float("-inf"))
        flat = bsc.topk(min(kk, n_cand), -1).indices[:, 0]
        S1h, ns = build_super(Sh, cfg.fanout)
        idx, _v, _c = hier_select(q1, Sh, S1h, cfg.fanout, cfg.beam, kk, cand, scale)
        hh = idx[:, 0]
        for bb in range(B):
            fs, hs = set(flat[bb].tolist()), set(hh[bb].tolist())
            agrees.append(len(fs & hs) / max(1, len(fs)))
    ns = math.ceil(nb / cfg.fanout)
    comps_hier = ns + min(cfg.beam, ns) * cfg.fanout
    return dict(agreement=float(np.mean(agrees)), comps_flat=n_cand, comps_hier=comps_hier,
                speedup=n_cand / max(1, comps_hier))


# ------------------------------------------------------------- driver ----------
def cell(res, key, fn):
    if key in res:
        log(f"[eval] skip cached {key} = {res[key]}"); return
    # per-cell compute cost is a first-class result: the "context extension is
    # affordable" claim needs seconds + peak-GiB alongside accuracy at every length
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        v = fn()
    except Exception as e:
        log(f"[eval] *** CELL {key} FAILED: {type(e).__name__}: {e} -- skip, retry on rerun ***")
        empty_cache(); return
    if v is None:
        return
    cost = dict(wall_s=round(time.time() - t0, 1),
                peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if torch.cuda.is_available() else None)
    res[key] = v; res[f"{key}|cost"] = cost
    save_json(res, RES); log(f"[eval] {key} = {v}  cost={cost}"); empty_cache()


def _agg(res, prefix):
    vals = [res[f"{prefix}|s{s}"] for s in SEEDS if f"{prefix}|s{s}" in res]
    if not vals:
        return None
    a = np.array(vals, float)
    return dict(mean=float(a.mean()),
                se=float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0, n=len(a))


def verdict(cfg, res):
    pd, ph = res.get("ppl_books|D"), res.get("ppl_books|H")
    if pd is None or ph is None:
        return "INCOMPLETE (ppl cells missing)"
    ppl_ok = ph <= 1.10 * pd
    ind_ok = max(res.get(f"induction|converted|{n}", 0) for n in (2048,4096,8192)) >= 0.3  # G2 preserved (chance ~1e-5)
    # converted(hier) >= donor+YaRN beyond native ctx (64K, 128K)
    beat = []
    for n in (65536, 131072):
        hh = _agg(res, f"needle|H_hier|{n}"); dy = _agg(res, f"needle|D_yarn|{n}")
        if hh and dy:
            beat.append(hh["mean"] >= dy["mean"])
    beat_ok = bool(beat) and all(beat)
    bench_ok = True
    for bmk in ("hellaswag", "arc_easy", "piqa"):
        d, h = res.get(f"bench_{bmk}|D"), res.get(f"bench_{bmk}|H")
        if d is not None and h is not None and h < d - 0.03:
            bench_ok = False
    if ppl_ok and ind_ok and beat_ok and bench_ok:
        return "PASS"
    # PARTIAL: quality + induction preserved and a monotone-improving needle gap with length
    gaps = []
    for n in EVAL_LENGTHS:
        h = _agg(res, f"needle|H_hier|{n}") or _agg(res, f"needle|H_flat|{n}")
        d = _agg(res, f"needle|D_yarn|{n}") or _agg(res, f"needle|D|{n}")
        if h and d:
            gaps.append(h["mean"] - d["mean"])
    mono = len(gaps) >= 3 and all(gaps[i + 1] >= gaps[i] - 0.02 for i in range(len(gaps) - 1))
    if ppl_ok and ind_ok and (beat_ok or mono):
        return "PARTIAL"
    if ppl_ok and ind_ok:
        return "WEAK (quality+induction preserved; retrieval below bar)"
    if not ind_ok:
        return "FAIL (induction not preserved through conversion)"
    return "FAIL (quality not preserved)"


def write_summary(cfg, res):
    v = verdict(cfg, res)
    L = ["# HBA donor conversion -- results\n",
         f"Donor -> HBA (window {cfg.window} + {cfg.sinks} sinks + top-{cfg.k_blocks} learned "
         f"routing, m={cfg.slots}; hierarchy fanout {cfg.fanout} beam {cfg.beam}).\n",
         "## Induction gates\n", "| model | 2048 | 4096 | 8192 |", "|---|---|---|---|"]
    for who in ("donor", "converted"):
        row = [f"{res.get(f'induction|{who}|{n}', '-')}" for n in (2048, 4096, 8192)]
        L.append(f"| {who} | " + " | ".join(row) + " |")
    L += ["\n## Perplexity (held-out)\n", "| domain | donor | converted | H/D |", "|---|---|---|---|"]
    for dom in ("books", "code", "web"):
        d, h = res.get(f"ppl_{dom}|D"), res.get(f"ppl_{dom}|H")
        L.append(f"| {dom} | {d if d else '-'} | {h if h else '-'} | "
                 f"{f'{h/d:.3f}' if (d and h) else '-'} |")
    L += ["\n## Needle retrieval (mean±SE, 3 seeds)\n",
          "| length | xnative | donor | donor+YaRN | conv(flat) | conv(hier) |",
          "|---|---|---|---|---|---|"]

    def cs(method, n):
        a = _agg(res, f"needle|{method}|{n}")
        return f"{a['mean']:.3f}±{a['se']:.3f}" if a else "-"
    for n in EVAL_LENGTHS:
        L.append(f"| {n} | {n/cfg.native_ctx:.2f}x | {cs('D',n)} | {cs('D_yarn',n)} | "
                 f"{cs('H_flat',n)} | {cs('H_hier',n)} |")
    L += ["\n## Benchmarks (loglik acc)\n", "| task | donor | converted |", "|---|---|---|"]
    for bmk in ("hellaswag", "arc_easy", "piqa"):
        L.append(f"| {bmk} | {res.get(f'bench_{bmk}|D','-')} | {res.get(f'bench_{bmk}|H','-')} |")
    L += ["\n## Hierarchy fidelity\n", "| length | agreement | comps flat | comps hier | speedup |",
          "|---|---|---|---|---|"]
    for n in EVAL_LENGTHS:
        h = res.get(f"hier|{n}|s0")
        if h:
            L.append(f"| {n} | {h['agreement']:.3f} | {h['comps_flat']} | {h['comps_hier']} | "
                     f"{h['speedup']:.1f}x |")
    L.append(f"\n## Verdict: **{v}**\n")
    sfx = "_smoke" if RES.endswith("_smoke.json") else ""
    open(os.path.join(RESULTS, f"summary{sfx}.md"), "w").write("\n".join(L))
    log(f"[eval] VERDICT: {v}")


def run(cfg, args):
    res = load_json(RES)
    stream = None
    ns = os.path.join(DATA, "needle_books.bin")
    if os.path.exists(ns):
        stream = np.memmap(ns, dtype=np.uint32, mode="r")

    if args.stage in ("gate", "all"):
        which = args.which or "donor"
        m = donor_model(cfg) if which == "donor" else converted_model(cfg, smoke=args.smoke)
        mode = "equiv" if which == "donor" else "eval"
        acc = induction_probe(m, cfg, mode)
        for n, a in acc.items():
            res[f"induction|{which}|{n}"] = a
        save_json(res, RES)
        del m; empty_cache()
        # G1/G2: gates ABORT the process (exit 1) so `--stage all` cannot sweep
        # past a failed gate. Threshold 0.3 (chance ~1e-5) -- the same bar
        # verdict() uses.
        if which == "donor" and max(acc.values()) < 0.3:
            log("*** G1 FAILED: raw donor shows no induction -- harness/probe broken, STOP ***")
            raise SystemExit(1)
        if which == "converted" and max(acc.values()) < 0.3:
            log("*** G2 FAILED: conversion destroyed induction -- STOP and diagnose, do not sweep ***")
            raise SystemExit(1)

    if args.stage in ("ppl", "all"):
        for who, mode in (("D", "equiv"), ("H", "eval")):
            m = donor_model(cfg) if who == "D" else converted_model(cfg, smoke=args.smoke)
            for dom, fn in (("books", "val_books.bin"), ("code", "val_code.bin"), ("web", "val_web.bin")):
                sp = os.path.join(DATA, fn)
                if os.path.exists(sp):
                    cell(res, f"ppl_{dom}|{who}", lambda m=m, sp=sp, mode=mode: perplexity(m, cfg, sp, mode))
            del m; empty_cache()

    if args.stage in ("needle", "all") and stream is not None:
        methods = {"D": donor_model, "D_yarn": donor_model,
                   "H_flat": converted_model, "H_hier": converted_model}
        lengths = [args.length] if args.length else list(EVAL_LENGTHS)
        for method in methods:
            if args.method and method != args.method:
                continue
            m = None
            for n in lengths:
                if method == "H_hier" and n < cfg.hier_from:
                    continue
                if method == "D_yarn" and n <= cfg.native_ctx:
                    continue
                if all(f"needle|{method}|{n}|s{s}" in res for s in SEEDS):
                    continue
                if m is None:
                    m = methods[method](cfg) if method.startswith("D") else methods[method](cfg, smoke=args.smoke)
                for s in SEEDS:
                    cell(res, f"needle|{method}|{n}|s{s}",
                         lambda m=m, n=n, method=method, s=s: needle_accuracy(m, cfg, n, method, s, stream))
            del m; empty_cache()

    if args.stage in ("bench", "all"):
        _d, tok = load_donor(dtype=torch.float32)
        del _d; empty_cache()
        for who, mode in (("D", "equiv"), ("H", "eval")):
            m = donor_model(cfg) if who == "D" else converted_model(cfg, smoke=args.smoke)
            for bmk in ("hellaswag", "arc_easy", "piqa"):
                cell(res, f"bench_{bmk}|{who}", lambda m=m, bmk=bmk, mode=mode: mc_benchmark(m, tok, cfg, bmk, mode))
            del m; empty_cache()

    if args.stage in ("hier", "all") and stream is not None:
        m = converted_model(cfg, smoke=args.smoke)
        for n in (args.length,) if args.length else EVAL_LENGTHS:
            if n < cfg.hier_from:
                continue
            for s in SEEDS:
                cell(res, f"hier|{n}|s{s}", lambda n=n, s=s: hier_fidelity(m, cfg, n, s, stream))
        del m; empty_cache()

    write_summary(cfg, res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["gate", "ppl", "needle", "bench", "hier", "summary", "all"],
                    default="all")
    ap.add_argument("--which", choices=["donor", "converted"], default=None)
    ap.add_argument("--length", type=int, default=None)
    ap.add_argument("--method", choices=["D", "D_yarn", "H_flat", "H_hier"], default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = smoke_config() if args.smoke else HBAConfig()
    if args.smoke:  # smoke cells must never land in (or read from) the real results cache
        global RES
        RES = os.path.join(RESULTS, "hba_results_smoke.json")
    log(f"hba eval device={DEVICE} stage={args.stage} smoke={args.smoke} res={RES}")
    if args.stage == "summary":
        write_summary(cfg, load_json(RES)); return
    run(cfg, args)
    log("hba eval complete")


if __name__ == "__main__":
    main()
