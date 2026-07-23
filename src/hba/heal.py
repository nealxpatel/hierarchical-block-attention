"""Staged healing: continued pretraining that makes the donor's Q/K
self-consistent with content-based NoPE routing (docs/training-recipe.md, stage
table). Position-free routing does not exist in a dense model's query/key
geometry and cannot be distilled onto frozen weights (convert.py's stage 0 is an
init, not a result) -- the geometry must adapt under continued pretraining. This
module implements stages 1-3; stage 0 (distill-init) lives in convert.py.

  stage1  attention + summarizer params ONLY (embed/MLP/norms frozen), a modest
          token budget. Re-aligns Q/K to the sparse routing budget and lets the
          aux-KL differentiate the slots, without disturbing the donor's
          knowledge (which lives largely in the MLPs).
  stage2  full-parameter, with capability rehearsal (fam_data.FamMixer) mixed
          into the data and a low LR. Plain full-parameter healing on generic
          text catastrophically forgets the donor's induction/copy circuit while
          perplexity stays flat (docs/training-recipe.md, "Capability
          rehearsal") -- rehearsal + a 4x-lower LR than a naive full-param LR is
          what prevents that, and IS this stage's recipe; there is no
          unrehearsed full-parameter variant in this codebase.
  stage3  length-curriculum extension: continues from the finished stage2
          checkpoint at a deterministic per-step mixed-context cycle (short:
          medium:long steps), summarizers FROZEN and the aux-KL teacher OFF
          (cfg.aux_w=0.0 -- the teacher is O(n^2) and ruinous at longer context;
          the summarizers are already trained). This repairs the Q/K union-
          softmax calibration across candidate-count regimes (docs/design.md,
          "Softmax calibration"), not the routing itself. Rehearsal continues at
          every curriculum length (its 'far' placement mode spans the full
          window length, so pair distances reach each stream's context
          automatically). A multi-length capability panel (probes.run_panel)
          runs on a fixed cadence -- the success signal is longer-length
          induction accuracy climbing toward the same bar as the base length.

Loss: full next-token CE through the hard-top-k selected-block exact attention (no
straight-through tricks) + aux_w * aux-KL for the summarizers, gradient-isolated
(LM -> q/k/v/o/mlp/embed, aux -> probes/proj only; attention.hba_attention_dense
and hba_attention_fused enforce the disjoint paths via detach -- see
gates.gate_grad_isolation). bf16 autocast + fp32 softmax on CUDA (fp32 everywhere
on MPS/CPU, the exact-comparability anchor there). Cosine LR per stage with
warmup, grad clip, no weight decay on norms/biases/embed/summarizer parameters.
Atomic checkpoints, fully resumable, wall-clock guard. Refuses to start unless the
correctness gates pass.

Usage (on a training box):
  python -m hba.heal --phase stage1 --resume
  python -m hba.heal --phase stage2 --resume
  python -m hba.heal --phase stage3 --resume   # length-curriculum heal (mixed ctx, aux OFF)
  python -m hba.heal --phase stage1 --smoke    # tiny, CPU/MPS plumbing
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .attention import rope_tables
from .config import (COMPUTE_DTYPE, DATA, DEVICE, INIT_PATH, RESULTS, HBAConfig, empty_cache,
                     log, resolve_backend, save_ckpt_atomic, smoke_config)
from .fam_data import FamMixer
from .gates import gate_causality, gate_fused_agreement, gate_grad_isolation, gate_path_equivalence
from .model import build_hba
from . import probes as capability_probes

# Per-stage token budgets and LRs (continued pretraining on a pretrained donor ->
# small LRs, no from-scratch warmup blowups). Ratios match docs/training-
# recipe.md's stage table (given there as tokens/param; instantiated here at
# validation scale, 0.5B params). Note there is no unrehearsed full-parameter
# stage in this budget table -- see the module docstring.
PHASES = {
    "stage1": dict(trainable={"attn", "summarizers"}, tokens=7.5e7, lr=2e-4, warmup=200),
    # Full-parameter heal WITH capability rehearsal: a lower LR (vs a naive full-
    # param LR) plus a ~3% needle-familiarization data mix (fam_data.FamMixer) so
    # the induction/copy circuit is rehearsed during full-param healing instead of
    # trained away. The low LR is load-bearing on its own too (docs/training-
    # recipe.md: "4x higher destroyed retrieval capability at validation scale").
    "stage2": dict(trainable={"attn", "summarizers", "mlp", "norms", "embed"},
                   tokens=2.5e8, lr=2.5e-5, warmup=200, fam_frac=0.03, probe_every=200),
    # Length-curriculum heal (docs/design.md, "Softmax calibration"; docs/training-
    # recipe.md, "Length curriculum"). Continues from the FINISHED stage2
    # checkpoint at a deterministic per-step mixed-context cycle short:medium:long
    # = 1:1:2 steps (the longest length is where the union-softmax calibration
    # must form), with per-length (micro, accum) keeping tokens/step invariant at
    # every length. AUX-KL TEACHER OFF (aux_off -> cfg.aux_w=0.0) and summarizers
    # FROZEN (not in trainable): the teacher is O(n^2) and ruinous at the longer
    # curriculum lengths; the summarizers are already trained; stage 3 repairs the
    # Q/K union-softmax calibration only. Rehearsal continues at ~2.6% per stream;
    # its 'far' placement mode spans the full window length, so pair distances
    # reach each stream's context automatically. Multi-length probe PANEL every
    # 200 steps at {4K, 8K, 16K} (+32K every 400 steps) -- the success signal is
    # 8K/16K induction accuracy climbing toward the same bar stage 2 reached.
    "stage3": dict(trainable={"attn", "mlp", "norms", "embed"},
                   tokens=1.0e8, lr=2e-5, warmup=100, fam_frac=0.026, probe_every=200,
                   aux_off=True,
                   ctx_cycle=(4096, 16384, 8192, 16384),
                   ctx_micro={4096: (2, 16), 8192: (1, 16), 16384: (1, 8)},
                   panel_ns=(4096, 8192, 16384), panel_extra_n=32768, panel_extra_every=400),
}

# stage -> the prior stage it seeds its weights from ("stage1" seeds from
# convert.py's stage-0 distilled-summarizer checkpoint instead; see train() below).
_SEED_FROM = {"stage2": "stage1", "stage3": "stage2"}


def _steps_with_ctx(step, cycle, c):
    """# of steps with context c among steps [0, step) of the deterministic ctx
    cycle. Gives a resume-safe micro index for that ctx's stream as a pure
    function of the global step."""
    full, rem = divmod(step, len(cycle))
    return full * cycle.count(c) + sum(1 for x in cycle[:rem] if x == c)


class WindowStream:
    """Deterministic, resumable stream of (B, ctx+1) token windows from a uint32 .bin."""
    def __init__(self, bin_path, ctx, batch, seed):
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")   # vocab > uint16 range
        self.l, self.B, self.seed = ctx, batch, seed
        self.W = (len(self.data) - 1) // ctx
        assert self.W >= batch, f"corpus too small: {self.W} windows < batch {batch}"
        self._epoch, self._perm = -1, None

    def _perm_for(self, epoch):
        if epoch != self._epoch:
            rng = np.random.default_rng(self.seed * 1_000_003 + epoch)
            self._perm = rng.permutation(self.W)
            self._epoch = epoch
        return self._perm

    def batch(self, m):
        ids = np.empty((self.B, self.l + 1), dtype=np.int64)
        for b in range(self.B):
            g = m * self.B + b
            epoch, pos = divmod(g, self.W)
            w = int(self._perm_for(epoch)[pos])
            ids[b] = self.data[w * self.l: w * self.l + self.l + 1].astype(np.int64)
        return torch.from_numpy(ids)


def lr_at(step, warmup, total, lr, min_frac=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def make_opt(model, lr):
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and "embed" not in n and "summ" not in n and "summarizer" not in n:
            decay.append(p)
        else:
            nodecay.append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                              {"params": nodecay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))


def _sig(cfg, phase_tokens=None, ctx_schedule=None):
    s = {k: getattr(cfg, k) for k in ("n_layers", "n_heads", "n_kv", "head_dim", "block",
                                      "window", "sinks", "k_blocks", "slots", "heal_ctx")}
    # The RESOLVED attention backend is part of the run identity (fused and naive
    # are gate-verified equivalent, but a benchmark/ablation checkpoint from one
    # backend must never silently resume as the other's run).
    s["attn_backend"] = resolve_backend(cfg)
    # The token budget is part of the identity: a short shakedown/--tokens run must
    # NEVER be resumable as (or mark itself done for) the full run -- a done=True
    # mini-checkpoint would make the real `--resume` skip training entirely.
    if phase_tokens is not None:
        s["phase_tokens"] = float(phase_tokens)
    # Mixed-length phases: the ctx cycle + per-length (micro, accum) are part of
    # the run identity (a checkpoint from one schedule must never silently resume
    # as another's).
    if ctx_schedule is not None:
        s["ctx_schedule"] = ctx_schedule
    return s


def _data_guards(phase, tokens, smoke, allow_small, shakedown=False):
    # shakedown=True: the shakedown's OWN mini training stage (threaded explicitly
    # from gates.check_training / the --shakedown CLI flag) legitimately runs its
    # ~150 steps on a small data slice -- that is the point of the shakedown. Its
    # checkpoint cannot leak into the real run (_sig embeds the token budget;
    # gates.check_training also moves the checkpoint aside), so exempting it is
    # safe. A REAL heal (scripts/heal.sh never passes --shakedown) keeps both
    # guards fully intact.
    if shakedown:
        log(f"[{phase}] shakedown mode: smoke-shard/corpus-size guards waived for the "
            "shakedown's mini training stage (real heal runs keep them)")
        return
    meta_p = os.path.join(DATA, "meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    if not smoke and meta.get("smoke", False):
        log("ABORT: data shards are SMOKE shards but this is a real heal run -- run "
            "`python -m hba.data_prep` at full size first (rm data/ if needed)")
        sys.exit(1)
    n_corpus = os.path.getsize(os.path.join(DATA, "train.bin")) // 4   # uint32
    if not smoke and n_corpus < 0.5 * tokens and not allow_small:
        log(f"ABORT: train corpus {n_corpus/1e6:.1f}M tok < half the {phase} budget "
            f"{tokens/1e6:.0f}M -- a shakedown's small data slice is not the full corpus; "
            "run `python -m hba.data_prep --train-tokens 4e8` (or pass "
            "--allow-small-corpus to override)")
        sys.exit(1)


def train(cfg, phase, resume, micro_batch, grad_accum, budget_s, smoke, allow_small=False,
          shakedown=False, probe_every=None):
    """Returns 'complete' | 'done' | 'guard' (wall-clock guard hit; NOT finished)."""
    p = PHASES[phase]
    mixed = "ctx_cycle" in p
    ckpt_path = os.path.join(RESULTS, f"heal_{phase}{'_smoke' if smoke else ''}.pt")
    if mixed:
        cycle = tuple(p["ctx_cycle"])
        ctx_micro = dict(p["ctx_micro"])
        tps_set = {c * mb * ga for c, (mb, ga) in ctx_micro.items()}
        assert len(tps_set) == 1, f"tokens/step not invariant across lengths: {ctx_micro}"
        tokens_per_step = tps_set.pop()
        ctx = max(ctx_micro)                      # longest ctx (rope tables per-length below)
        sig_sched = f"cycle={cycle} micro={sorted(ctx_micro.items())}"
        log(f"[{phase}] MIXED-LENGTH schedule: per-step ctx cycle {cycle} "
            f"(4K:8K:16K = {cycle.count(4096)}:{cycle.count(8192)}:{cycle.count(16384)} per "
            f"cycle), per-ctx (micro,accum)={ctx_micro} -> tokens/step={tokens_per_step} "
            "invariant at every length")
        log(f"[{phase}] NOTE: CLI --micro-batch/--grad-accum are IGNORED for mixed-length "
            "phases (the schedule fixes them per length)")
    else:
        cycle = ctx_micro = None
        sig_sched = None
        ctx = cfg.heal_ctx
        tokens_per_step = micro_batch * grad_accum * ctx
    total_steps = math.ceil(p["tokens"] / tokens_per_step)
    log(f"[{phase}] budget {p['tokens']/1e6:.0f}M tok  tps={tokens_per_step} -> {total_steps} steps  "
        f"lr={p['lr']} guard={budget_s/3600:.1f}h ctx={ctx}")
    if p.get("aux_off"):
        cfg.aux_w = 0.0
        log(f"[{phase}] *** AUX-KL TEACHER OFF (aux_w=0.0): the O(n^2) teacher is skipped "
            "entirely in both attention backends (ruinous at long context); summarizers are "
            f"FROZEN (trainable groups = {sorted(p['trainable'])}); {phase} repairs the Q/K "
            "union-softmax calibration only ***")
    _data_guards(phase, p["tokens"], smoke, allow_small, shakedown=shakedown)

    # fp32 MASTER weights + bf16 autocast compute on CUDA (autocast keeps softmax
    # fp32 by policy). Bare-bf16 weights would also make the AdamW states bf16 and
    # silently lose small updates at these LRs -- the memory budget already
    # assumes fp32 optimizer states.
    model, tok, cfg = build_hba(cfg, dtype=torch.float32)
    # seed weights: stage1 from the distilled stage-0 init; stage2/stage3 from the
    # finished prior stage's checkpoint.
    if not (resume and os.path.exists(ckpt_path)):
        if phase == "stage1" and os.path.exists(INIT_PATH):
            sd = torch.load(INIT_PATH, map_location="cpu")
            model.summarizers.load_state_dict(sd["summarizers"])
            log(f"[stage1] seeded summarizers from {INIT_PATH}")
        elif phase in _SEED_FROM:
            seed_phase = _SEED_FROM[phase]
            prev = os.path.join(RESULTS, f"heal_{seed_phase}{'_smoke' if smoke else ''}.pt")
            if os.path.exists(prev):
                pk = torch.load(prev, map_location="cpu")
                if not pk.get("done") and not smoke:
                    log(f"ABORT: {prev} exists but is NOT complete (done=False) -- finish "
                        f"{seed_phase} first (`python -m hba.heal --phase {seed_phase} "
                        f"--resume`); healing {phase} from a partial {seed_phase} would "
                        "silently change the pre-registered recipe")
                    sys.exit(1)
                model.load_state_dict(pk["model"])
                log(f"[{phase}] seeded full model from {prev} (done={pk.get('done')})")
            elif smoke:
                log(f"[{phase}] SMOKE: no {prev} -- healing from the raw distilled donor")
                if os.path.exists(INIT_PATH):
                    model.summarizers.load_state_dict(
                        torch.load(INIT_PATH, map_location="cpu")["summarizers"])
            else:
                log(f"ABORT: no {prev} -- run phase {seed_phase} first ({phase} seeds from the "
                    f"finished {seed_phase})")
                sys.exit(1)

    model.set_trainable(p["trainable"])
    opt = make_opt(model, p["lr"])

    step0 = 0
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE)
        if ck.get("cfg_sig") == _sig(cfg, p["tokens"], sig_sched):
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]
            if ck.get("done"):
                log(f"[{phase}] already complete at step {step0}"); return "complete"
            log(f"[{phase}] resuming from step {step0}/{total_steps}")
        else:
            # A mismatched checkpoint cannot fall through to "fresh": the seed
            # block above already decided we were resuming, so a fresh start here
            # would train from RANDOM summarizers (stage1) or the raw donor
            # (stage2/3) with the seeded init silently dropped.
            log(f"[{phase}] ABORT: {ckpt_path} exists but its cfg_sig mismatches this run "
                "(cfg/backend/token-budget changed). Move the stale checkpoint aside "
                "explicitly, or rerun without --resume after verifying which run it belongs to.")
            sys.exit(1)

    base_seed = cfg.seed if hasattr(cfg, "seed") else 0
    stream = fam = None
    streams = fams = tabs = None
    if mixed:
        # one WindowStream + FamMixer PER LENGTH, over the same train.bin with
        # distinct seeds. Token overlap across the per-length streams is
        # acceptable (a documented design decision: the corpus is generic
        # web/books/code and stage 3 is calibration repair, not knowledge
        # acquisition -- seeing a window at two lengths is rehearsal, not
        # leakage). FamMixer's 'far' placement mode spans the full window
        # length, so pair distances reach each stream's ctx automatically -- no
        # extension needed.
        streams, fams, tabs = {}, {}, {}
        for i, c in enumerate(sorted(ctx_micro)):
            mb_c = ctx_micro[c][0]
            s = WindowStream(os.path.join(DATA, "train.bin"), c, mb_c, base_seed + i)
            f = None
            if p.get("fam_frac"):
                f = FamMixer(s, cfg, seed=base_seed + 101 + i, frac=p["fam_frac"])
                s = f
            streams[c], fams[c] = s, f
            tabs[c] = rope_tables(c, cfg.head_dim, cfg.rope_theta, DEVICE)
        if p.get("fam_frac"):
            log(f"[{phase}] FAM-MIX enabled per length: target frac={p['fam_frac']:.3f} for "
                f"each of ctx {sorted(ctx_micro)} (distances span up to each stream's ctx)")
        cos = sin = None
    else:
        stream = WindowStream(os.path.join(DATA, "train.bin"), ctx, micro_batch, base_seed)
        # stage2: weave the ~3% needle-familiarization mix over the (otherwise
        # unchanged) real-text stream. Deterministic given base_seed; counts
        # logged for auditability.
        if p.get("fam_frac"):
            fam = FamMixer(stream, cfg, seed=base_seed + 1, frac=p["fam_frac"])
            stream = fam
            log(f"[{phase}] FAM-MIX enabled: target frac={p['fam_frac']:.3f} "
                f"(fam_data.FamMixer over WindowStream)")
        cos, sin = rope_tables(ctx, cfg.head_dim, cfg.rope_theta, DEVICE)
    # in-training capability gate: probe cadence from arg override, else phase config (0 = off)
    pe = probe_every if probe_every is not None else p.get("probe_every", 0)
    panel_ns = panel_tabs = None
    extra_n = extra_every = None
    probe_cos = probe_sin = None
    if pe and mixed:
        # MULTI-LENGTH probe panel (docs/training-recipe.md: a length axis is
        # mandatory -- a single short-length probe can show a perfect plateau
        # while the model cliffs at longer lengths). Every `pe` steps: induction
        # std at panel_ns; every `panel_extra_every` steps additionally the
        # extrapolation cell.
        panel_ns = tuple(p.get("panel_ns", (4096, 8192, 16384)))
        extra_n = p.get("panel_extra_n", 32768)
        extra_every = p.get("panel_extra_every", 2 * pe)
        panel_tabs = {}
        for pn in panel_ns + (extra_n,):
            panel_tabs[pn] = tabs[pn] if pn in tabs else \
                rope_tables(pn, cfg.head_dim, cfg.rope_theta, DEVICE)
        log(f"[{phase}] PANEL every {pe} steps: induction std at n={panel_ns} (16 trials each); "
            f"every {extra_every} steps additionally n={extra_n} (extrapolation). SUCCESS "
            "SIGNAL: longer-length accuracy climbing toward the base-length bar")
    elif pe:
        probe_cos, probe_sin = rope_tables(2048, cfg.head_dim, cfg.rope_theta, DEVICE)
        log(f"[{phase}] mini induction-probe gate every {pe} steps (std placement, n=2048, "
            "16 trials, eval mode; target acc>=0.3 sustained)")
    cap = cfg.mem_elem_cap
    t_start = t_ckpt = time.time()
    tok_seen = step0 * tokens_per_step
    autocast = (DEVICE == "cuda")

    def save(step, done):
        save_ckpt_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                          "done": done, "cfg_sig": _sig(cfg, p["tokens"], sig_sched),
                          "phase": phase, "tok_seen": tok_seen}, ckpt_path)

    per_ctx = ({c: dict(tok=0, sec=0.0, steps=0, peak=0.0) for c in ctx_micro} if mixed else None)
    for step in range(step0, total_steps):
        lr = lr_at(step, p["warmup"], total_steps, p["lr"])
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_v = aux_v = 0.0
        if mixed:
            c_s = cycle[step % len(cycle)]
            mb_s, ga_s = ctx_micro[c_s]
            stream_s = streams[c_s]
            cos_s, sin_s = tabs[c_s]
            # resume-safe: micro index for this ctx's stream is a pure function of `step`
            m0 = _steps_with_ctx(step, cycle, c_s) * ga_s
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
        else:
            c_s, ga_s, stream_s, cos_s, sin_s, m0 = ctx, grad_accum, stream, cos, sin, \
                step * grad_accum
        t_step = time.time()
        for micro in range(ga_s):
            ids = stream_s.batch(m0 + micro).to(DEVICE)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            ctx_ = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast \
                else torch.autocast(device_type="cpu", enabled=False)
            with ctx_:
                if mixed:
                    # chunked-CE forward: the [B,n,V] logit tensor is never
                    # materialized (a mixed-length OOM hazard at longer context).
                    # Exact same mean CE.
                    loss_lm = model(inp, cos_s, sin_s, cap, mode="train", loss_tgt=tgt)
                    aux = model._last_aux
                    aux = aux if aux is not None else loss_lm.new_zeros(())
                else:
                    logits = model(inp, cos_s, sin_s, cap, mode="train")
                    aux = model._last_aux
                    aux = aux if aux is not None else logits.new_zeros(())
                    loss_lm = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                              tgt.reshape(-1))
                loss = loss_lm + cfg.aux_w * aux
            (loss / ga_s).backward()
            loss_v += float(loss_lm.detach()) / ga_s
            aux_v += float(aux.detach()) / ga_s
            tok_seen += inp.numel()
        torch.nn.utils.clip_grad_norm_([pp for pp in model.parameters() if pp.requires_grad], 1.0)
        opt.step()
        if mixed:
            st = per_ctx[c_s]
            st["tok"] += mb_s * ga_s * c_s
            st["sec"] += time.time() - t_step
            st["steps"] += 1
            if DEVICE == "cuda":
                st["peak"] = max(st["peak"], torch.cuda.max_memory_allocated() / 2**30)

        # mixed: log one FULL cycle every 20 steps (20 % len(cycle) == 0, so a bare
        # %20 gate would only ever show cycle position 0; the bracket accumulators
        # cover all lengths either way, but per-length loss visibility needs the
        # whole cycle logged).
        if (step % 20 < len(cycle)) if mixed else (step % 20 == 0):
            el = time.time() - t_start
            tps = (tok_seen - step0 * tokens_per_step) / max(1e-6, el)
            fam_s = f" fam {fam.ratio()*100:.2f}%" if fam is not None else ""
            if mixed:
                parts = []
                for c in sorted(per_ctx):
                    st = per_ctx[c]
                    if st["steps"]:
                        fr = f",fam{fams[c].ratio()*100:.2f}%" if fams[c] is not None else ""
                        parts.append(f"{c//1024}K:{st['tok']/max(1e-6, st['sec']):.0f}t/s,"
                                     f"{st['peak']:.1f}G{fr}")
                fam_s = "  [" + " ".join(parts) + "]"
            ctx_s_ = f" ctx {c_s}" if mixed else ""
            log(f"[{phase}] step {step:6d}/{total_steps}{ctx_s_} lm {loss_v:.4f} ppl "
                f"{math.exp(min(20, loss_v)):.2f} aux {aux_v:.4f} tok/s {tps:6.0f} lr {lr:.2e} "
                f"tok {tok_seen/1e6:.0f}M elapsed {el/3600:.2f}h{fam_s}")
        if pe and step % pe == 0:
            if mixed:
                ns_list = list(panel_ns) + ([extra_n] if step % extra_every == 0 else [])
                t_p = time.time()
                accs = capability_probes.run_panel(model, cfg, panel_tabs, ns_list, trials=16)
                msg = " ".join(f"n={pn}:{a:.3f}" for pn, a in accs)
                log(f"[{phase}] PANEL @ step {step}: induction-std {msg} "
                    f"({time.time()-t_p:.0f}s) [signal: longer lengths -> base-length bar] "
                    f"lm_ppl_proxy {math.exp(min(20, loss_v)):.2f}")
            else:
                acc = capability_probes.induction_probe_quick(model, cfg, probe_cos, probe_sin,
                                                               n=2048, trials=16)
                gate = "OK" if acc >= 0.3 else "below-0.3"
                log(f"[{phase}] MINI-PROBE @ step {step}: induction std n=2048 acc={acc:.3f} "
                    f"[target>=0.3 {gate}] lm_ppl_proxy {math.exp(min(20, loss_v)):.2f}")
        if time.time() - t_ckpt > 1200:
            save(step + 1, False); t_ckpt = time.time(); log(f"[{phase}] checkpoint @ {step+1}")
        if time.time() - t_start > budget_s:
            log(f"[{phase}] *** WALL-CLOCK GUARD {budget_s/3600:.1f}h HIT @ {step}; saving ***")
            save(step + 1, False); return "guard"
    save(total_steps, True)
    log(f"[{phase}] COMPLETE @ {total_steps} ({tok_seen/1e6:.0f}M tok)")
    return "done"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["stage1", "stage2", "stage3"], required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--probe-every", type=int, default=None,
                    help="override the in-training mini-probe cadence (steps; 0 disables). "
                         "Default: phase config (stage2/stage3 = 200).")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--budget-s", type=float, default=None)
    ap.add_argument("--tokens", type=float, default=None, help="override phase token budget")
    ap.add_argument("--skip-gates", action="store_true")
    ap.add_argument("--allow-small-corpus", action="store_true",
                    help="override the corpus-vs-budget size guard (multi-epoch healing)")
    ap.add_argument("--shakedown", action="store_true",
                    help="INTERNAL (threaded from gates.check_training only): this is the "
                         "shakedown's mini training stage -- waive the smoke-shard/corpus-size "
                         "data guards. Never pass this for a real heal (scripts/heal.sh does not).")
    ap.add_argument("--attn-backend", choices=["naive", "fused"], default=None,
                    help="train-path attention backend (default: cfg.attn_backend = fused). "
                         "naive = the materialized-scores correctness oracle (markedly slower).")
    args = ap.parse_args()

    cfg = smoke_config() if args.smoke else HBAConfig()
    if args.attn_backend is not None:
        cfg.attn_backend = args.attn_backend
    if not hasattr(cfg, "seed"):
        cfg.seed = 0
    if args.tokens is not None:
        PHASES[args.phase]["tokens"] = args.tokens
    # CUDA default micro_batch=1, grad_accum=32 (tokens/step 131072): backward-pass
    # activation memory at micro_batch=2 and heal_ctx=4096 can exceed a 24GB-class
    # card even with grad checkpointing + bf16 autocast; raise --micro-batch on
    # cards with more headroom.
    micro_batch = args.micro_batch or 1
    grad_accum = args.grad_accum or (1 if args.smoke else 32)
    budget_s = args.budget_s or (600 if args.smoke else 20 * 3600)
    log(f"heal device={DEVICE} phase={args.phase} smoke={args.smoke} dtype={COMPUTE_DTYPE} "
        f"attn_backend={resolve_backend(cfg)}")

    # data sanity
    if not os.path.exists(os.path.join(DATA, "train.bin")):
        log("ABORT: data/train.bin missing -- run `python -m hba.data_prep` first"); sys.exit(1)

    if not args.skip_gates:
        from .gates import gate_equivalence
        gmodel, gtok, gcfg = build_hba(cfg, dtype=torch.float32)
        backend = resolve_backend(gcfg)
        ok = (gate_equivalence(gmodel, gtok, gcfg)[0] and gate_causality(gmodel, gcfg)
              and gate_path_equivalence(gmodel, gcfg) and gate_grad_isolation(gmodel, gcfg))
        if ok and backend == "fused":
            # the fused backend must agree with the naive oracle + keep gradient isolation
            ok = (gate_fused_agreement(gmodel, gcfg)
                  and gate_grad_isolation(gmodel, gcfg, backend="fused"))
        del gmodel
        empty_cache()
        if not ok:
            log("CONVERSION GATES FAILED -- aborting heal (fake numbers otherwise)"); sys.exit(1)

    status = train(cfg, args.phase, args.resume, micro_batch, grad_accum, budget_s, args.smoke,
                   allow_small=args.allow_small_corpus, shakedown=args.shakedown,
                   probe_every=args.probe_every)
    if status == "guard":
        log(f"[{args.phase}] NOT finished (wall-clock guard) -- rerun with --resume")
        sys.exit(3)


if __name__ == "__main__":
    main()
