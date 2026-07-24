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

          Optionally, stage2 can also add donor KD (`--kd`): a full-logit KL
          between the student's next-token distribution and a FROZEN, separately
          -loaded copy of the original donor (see hba.kd's module docstring for
          the loss formulation, and hba.gates.gate_kd for its correctness gate).
          KD is ADDITIVE to rehearsal, not a replacement for it: rehearsal
          protects enumerated capabilities the fam-data mix exercises; KD
          protects the donor's WHOLE next-token distribution at every position,
          whether or not a probe happens to exercise it. Off by default; see
          docs/training-recipe.md's stage-2 section for the full rationale.
  stage3  length-curriculum extension: continues from the finished stage2
          checkpoint at a deterministic per-step mixed-context cycle (short:
          medium:long steps), summarizers FROZEN and the aux-KL teacher OFF
          (cfg.aux_w=0.0 -- the teacher is O(n^2) and ruinous at longer context;
          the summarizers are already trained). This repairs the Q/K union-
          softmax calibration across candidate-count regimes (docs/design.md,
          "Softmax calibration"), not the routing itself. Rehearsal continues at
          every curriculum length (its 'far' placement mode spans the full
          window length, so pair distances reach each stream's context
          automatically).

Capability monitoring (docs/training-recipe.md, "Monitoring: the capability
panel is a first-class training signal"): every stage runs probes.run_panel on a
fixed cadence (`probe_every`, default 200 steps) and appends one JSON line per
firing to results/probe_log_<phase>.jsonl. early_stop.py's EarlyStopEngine
consumes that log every firing and can (a) call a clean ES-1 plateau stop, or (b)
call an ES-2 forgetting abort -- halt, roll back to the best-panel checkpoint,
and write results/ES2_TOMBSTONE.json, which blocks any future start/resume until
a human diagnoses and deletes it deliberately. Checkpoints kept per phase:
heal_<phase>.pt (the resumable running checkpoint), heal_<phase>_best.pt (highest
panel mean seen), and heal_<phase>_milestone{25,50,75}.pt (fixed token-budget
fractions) -- so cross-run comparisons always have a common-token-count
checkpoint even when a run stops early.

Loss: full next-token CE through the hard-top-k selected-block exact attention (no
straight-through tricks) + aux_w * aux-KL for the summarizers, gradient-isolated
(LM -> q/k/v/o/mlp/embed, aux -> probes/proj only; attention.hba_attention_dense
and hba_attention_fused enforce the disjoint paths via detach -- see
gates.gate_grad_isolation). bf16 autocast + fp32 softmax on CUDA (fp32 everywhere
on MPS/CPU, the exact-comparability anchor there). Cosine LR per stage with
warmup, grad clip, no weight decay on norms/biases/embed/summarizer parameters.
Atomic checkpoints, fully resumable, wall-clock guard. Refuses to start unless the
correctness gates pass (or a live ES-2 tombstone is present -- see above).

Usage (on a training box):
  python -m hba.heal --phase stage1 --resume
  python -m hba.heal --phase stage2 --resume
  python -m hba.heal --phase stage2 --resume --kd --kd-weight 1.0 --kd-temp 1.0  # + donor KD
  python -m hba.heal --phase stage3 --resume   # length-curriculum heal (mixed ctx, aux OFF)
  python -m hba.heal --phase stage1 --smoke    # tiny, CPU/MPS plumbing
"""

import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

from .attention import rope_tables
from .chunked_ce import chunked_cross_entropy
from .kd import chunked_kd_kl
from .config import (COMPUTE_DTYPE, DATA, DEVICE, DONOR_NAME, INIT_PATH, RESULTS, HBAConfig,
                     empty_cache, log, resolve_backend, save_ckpt_atomic, smoke_config)
from .fam_data import FamMixer
from .gates import (gate_causality, gate_fused_agreement, gate_grad_isolation, gate_kd,
                    gate_path_equivalence)
from .model import build_hba, load_donor
from . import dist_util
from . import early_stop
from . import probes as capability_probes

# Global tokens/optimizer-step target the multi-GPU stream-sharding contract holds
# CONSTANT across every (world, micro_batch, grad_accum) decomposition (dist_util's
# module docstring): 32 windows * 4096 ctx at this repo's single-GPU defaults
# (micro_batch=1, grad_accum=32 -- see main()'s non-smoke defaults below). Every
# phase's per-length tokens/step (including stage3's mixed-length ctx_micro table)
# is pre-registered to this same value; see dist_util.assert_valid_world_config for
# the divisibility check a (world, micro_batch) pair must satisfy.
GLOBAL_TOKENS_PER_STEP = 131072

# Bumped whenever the realized data-stream enumeration (WindowStream/FamMixer g
# indexing) or the FamMixer RNG-keying scheme changes for a FIXED (seed, cfg) --
# i.e. whenever two runs at identical seed/cfg would now see different windows or
# different rehearsal plants than before. 2 = the multi-GPU change: FamMixer keys
# its RNG on the global window index g (dist_util.global_window_index) instead of
# the single-process (m, b) pair (see fam_data.FamMixer._rng's docstring) --
# required to make plants world-size-invariant, but it silently changes what a
# pre-existing checkpoint would resume onto, hence the bump (see _sig below).
STREAM_VERSION = 2

# Per-stage token budgets and LRs (continued pretraining on a pretrained donor ->
# small LRs, no from-scratch warmup blowups). Ratios match docs/training-
# recipe.md's stage table (given there as tokens/param; instantiated here at
# validation scale, 0.5B params). Note there is no unrehearsed full-parameter
# stage in this budget table -- see the module docstring.
# comm_dtype (multi-GPU DDP only; see dist_util.register_comm_hook): the
# gradient-reduction dtype for that stage's all-reduce, pre-registered per stage
# (not a runtime knob -- changing it is a recipe change, recorded in _sig).
# Attention-only stage1 has small grads -> plain fp32 bucket all-reduce (DDP's
# default, no comm hook needed). Full-parameter stage2/stage3 reduce the whole
# model -> the bf16 gradient-compression hook halves communicated bytes, which
# matters on hardware without GPU-to-GPU P2P (all-reduce stages through host
# memory).
PHASES = {
    "stage1": dict(trainable={"attn", "summarizers"}, tokens=7.5e7, lr=2e-4, warmup=200,
                   comm_dtype="fp32"),
    # Full-parameter heal WITH capability rehearsal: a lower LR (vs a naive full-
    # param LR) plus a ~3% needle-familiarization data mix (fam_data.FamMixer) so
    # the induction/copy circuit is rehearsed during full-param healing instead of
    # trained away. The low LR is load-bearing on its own too (docs/training-
    # recipe.md: "4x higher destroyed retrieval capability at validation scale").
    "stage2": dict(trainable={"attn", "summarizers", "mlp", "norms", "embed"},
                   tokens=2.5e8, lr=2.5e-5, warmup=200, fam_frac=0.03, probe_every=200,
                   comm_dtype="bf16"),
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
    # reach each stream's context automatically. The capability PANEL (probes.py)
    # runs every 200 steps at ITS OWN fixed lengths (2048/2048/4096), independent
    # of this stage's ctx_cycle -- see probes.py's module docstring for why that
    # independence is deliberate (early_stop.py's rules need firings that are
    # comparable across the whole run). NOTE: this means the panel does NOT, by
    # itself, track induction accuracy AT the curriculum lengths (8K/16K) the way
    # an earlier version of this cadence did -- only the per-step LM loss at each
    # curriculum length is visible in the training log above. Per-curriculum-
    # length induction dose-response (docs/training-recipe.md, "Length
    # curriculum") is an evals.py / offline concern for this stage, not this
    # panel's job.
    "stage3": dict(trainable={"attn", "mlp", "norms", "embed"},
                   tokens=1.0e8, lr=2e-5, warmup=100, fam_frac=0.026, probe_every=200,
                   aux_off=True,
                   ctx_cycle=(4096, 16384, 8192, 16384),
                   ctx_micro={4096: (2, 16), 8192: (1, 16), 16384: (1, 8)},
                   comm_dtype="bf16"),
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

    def batch(self, m, rank=0, world=1):
        """`self.B` is the PER-RANK micro-batch. g is the world-size-invariant
        global window index, computed by calling dist_util.global_window_index
        directly (step=0 so its step term vanishes identically, leaving
        exactly g = m*(world*B) + rank*B + b -- see that function's docstring
        for the general formula) rather than re-deriving the same arithmetic a
        second time here. At rank=0, world=1 (the single-GPU default) this is
        exactly g = m*B + b -- bit-identical to the pre-multi-GPU formula, so
        single-process behavior is unchanged."""
        ids = np.empty((self.B, self.l + 1), dtype=np.int64)
        for b in range(self.B):
            g = dist_util.global_window_index(step=0, micro=m, b=b, rank=rank,
                                              world=world, micro_B=self.B, accum=1)
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


def _sig(cfg, phase_tokens=None, ctx_schedule=None, comm_dtype=None, kd=None):
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
    # --- multi-GPU identity fields (design: "checkpoint identity") -----------
    # stream_version: bumped whenever the realized data-stream enumeration or the
    # FamMixer RNG-keying scheme changes for a fixed (seed, cfg) -- see this
    # module's STREAM_VERSION docstring above. A pre-bump checkpoint resuming on
    # the post-bump stream would silently continue training on DIFFERENT data
    # than the run it thinks it's continuing; refusing that resume (via a _sig
    # mismatch) is the whole point of this field.
    s["stream_version"] = STREAM_VERSION
    # global_tokens_per_step: binds the world-size-invariance CONTRACT itself (see
    # dist_util's module docstring) -- a run at a different global tokens/step is
    # a different recipe, not a portable resharding of the same one, and must not
    # resume across that boundary either.
    s["global_tokens_per_step"] = GLOBAL_TOKENS_PER_STEP
    # comm_dtype: the DDP gradient-reduction dtype (dist_util.register_comm_hook)
    # is part of the training recipe (fp32 vs bf16-compressed all-reduce changes
    # the realized gradient, if only at fp noise level) -- pre-registered per
    # phase (PHASES[phase]["comm_dtype"]), not a free runtime knob, so it belongs
    # in the identity like attn_backend above. Deliberately world-size-agnostic:
    # world size and per-rank micro_batch/grad_accum are NOT part of _sig (that
    # omission IS the portability -- a checkpoint must resume unchanged whether
    # relaunched at world=1, 4, or 8, as long as global_tokens_per_step matches).
    if comm_dtype is not None:
        s["comm_dtype"] = comm_dtype
    # kd: donor KD (this module's --kd/--kd-weight/--kd-temp) changes the
    # training OBJECTIVE, not just a runtime knob -- a stage-2 checkpoint
    # trained with the extra donor-KD loss term and one trained without it are
    # different recipes and must never silently resume into each other (the
    # same reasoning as comm_dtype/attn_backend above). Included only for
    # phases where KD is even applicable (train() passes kd=None for stage1/
    # stage3), so those phases' signatures are byte-identical to before this
    # field existed. For stage2, this is ALWAYS present (even when KD is off,
    # as {"on": False}) so a KD run and a non-KD run of stage2 differ in
    # signature -- not just a KD-on run differing from a pre-KD-feature
    # checkpoint.
    if kd is not None:
        s["kd"] = kd
    return s


def _write_json_atomic(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


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
          shakedown=False, probe_every=None, warmup_steps=0, rank=0, world=1, local_rank=0,
          tps_out=None, kd=False, kd_weight=1.0, kd_temp=1.0):
    """Returns 'complete' | 'done' | 'guard' (wall-clock guard hit; NOT finished).

    warmup_steps: INTERNAL (threaded from gates.check_training's --fast profile
    only; a plain heal.py run leaves this 0). FlexAttention kernel autotune (and
    torch.compile, if enabled) burns real wall-clock in the first few steps
    without being representative of steady-state throughput. The shakedown's
    fast profile measures only 50 steps total, so that warmup tax is no longer
    diluted away the way it is at a real phase's step count -- once
    `warmup_steps` steps have completed, the tok/s baseline resets so the
    reported throughput reflects steady state only (see the tps computation
    below).

    tps_out: OPTIONAL mutable dict (a rank-agnostic recording channel -- e.g.
    `{}` passed in by the caller and read back after this returns). If given,
    THIS rank's own measured tok/s series is appended to tps_out["series"] at
    the same cadence as the status log line below, REGARDLESS of `rank` (the
    log line itself stays rank-0-only to avoid spam -- see the Logging note
    below). This is how gates.check_training measures per-rank throughput
    directly at world > 1: every rank calls train() in-process and reads its
    OWN tps_out, instead of monkeypatching heal.log and parsing the rank-0-only
    status line, which structurally never fires on ranks != 0 (design fix for
    the multi-GPU throughput gate -- see check_training's docstring).

    rank/world/local_rank: multi-GPU DDP coordinates (dist_util.setup_distributed's
    return values; defaults 0/1/0 reproduce single-GPU behavior exactly -- every
    distributed call in this function is behind an `is_distributed()` guard, so a
    plain `python -m hba.heal` invocation never touches torch.distributed at all).
    `micro_batch` and `grad_accum` are PER-RANK (see dist_util's module docstring
    for the world-size-invariant g = step*(world*micro_B*accum) + micro*(world*
    micro_B) + rank*micro_B + b sharding formula); global tokens/step = world *
    micro_batch * grad_accum * ctx, held constant at GLOBAL_TOKENS_PER_STEP across
    every (world, micro_batch, grad_accum) decomposition main() validates before
    calling in.

    Logging: high-frequency per-step/per-firing log lines (tok/s status, panel
    firings, checkpoint/milestone/guard notices) are rank-0-only below, to avoid
    `world`-way duplicate spam under torchrun -- one-time setup lines above stay
    unrestricted (small volume; seeing them from every rank is a useful sanity
    check that all ranks agree on the schedule).

    kd/kd_weight/kd_temp: donor knowledge-distillation (see hba.kd's module
    docstring for the loss itself). STAGE-2 ONLY: stage1 barely moves the
    donor's weights (attention-only) so there is little to distill against
    yet, and stage3's dense-teacher forward is O(n^2) at the length-curriculum
    context lengths -- the exact reason `cfg.aux_w` is forced to 0 there too
    (see PHASES["stage3"]'s comment). A caller passing kd=True for any phase
    other than stage2 gets a hard refusal below, not a silent ignore -- "KD
    changes the objective" is exactly the kind of silent-drift this module
    refuses by default elsewhere (see _sig's docstring comment on kd)."""
    def rlog(*a):
        if rank == 0:
            log(*a)
    # ES-2 tombstone: a prior run's forgetting-abort halted, rolled back, and
    # left this marker (early_stop.write_tombstone). The data stream is
    # deterministic, so restarting unattended would just replay the identical
    # collapse; refuse until a human diagnoses it and deletes the file
    # deliberately (see early_stop.check_tombstone's docstring).
    tomb = early_stop.check_tombstone(RESULTS)
    if tomb is not None:
        log(f"ABORT: {early_stop.tombstone_path(RESULTS)} exists (ES-2 forgetting-abort fired "
            f"at step {tomb['step']} / {tomb['tokens']/1e6:.0f}M tok, probe={tomb['probe']}, "
            f"rolled back to step {tomb['rolled_back_to']}) -- refusing to start or resume ANY "
            "phase. Diagnose the collapse first (docs/evals.md's discriminator ladder localizes "
            "retrieval failures in minutes), then delete the tombstone deliberately once addressed.")
        sys.exit(1)
    # Donor KD is stage-2-only (see this function's docstring) -- refuse rather
    # than silently ignore --kd on stage1/stage3 (main() already refuses this
    # earlier at the CLI level for a plain `python -m hba.heal` invocation; this
    # is the belt-and-suspenders copy for any other caller of train() directly,
    # e.g. gates.check_training).
    kd_on = bool(kd)
    if kd_on and phase != "stage2":
        log(f"ABORT: --kd was requested for phase={phase}, but donor KD is STAGE-2-ONLY "
            "(stage1 barely moves the donor's weights; stage3's dense-teacher forward is "
            "O(n^2)-unaffordable at the length-curriculum context lengths, the same reason "
            "cfg.aux_w is forced to 0 there). Rerun with --phase stage2, or drop --kd.")
        sys.exit(1)
    p = PHASES[phase]
    comm_dtype = p.get("comm_dtype", "fp32")
    mixed = "ctx_cycle" in p
    assert not (kd_on and mixed), f"[{phase}] donor KD requested on a mixed-length phase -- " \
        "should have been refused above"
    # kd_sig: recorded in _sig (below) ONLY when KD is actually on (stage2), so a
    # rehearsal-only stage2 checkpoint and a rehearsal+KD stage2 checkpoint never
    # silently resume into each other. When KD is OFF, kd_sig is None -- a KD-off
    # stage2 run is the SAME recipe as a pre-KD-feature stage2 run, so its
    # signature stays byte-identical to before this feature existed (otherwise an
    # in-flight rehearsal-only stage2 checkpoint would refuse to resume the moment
    # this feature merged). None for stage1/stage3 too (KD isn't applicable there).
    kd_sig = (dict(on=True, weight=round(kd_weight, 6), temp=round(kd_temp, 6))
              if (kd_on and phase == "stage2") else None)
    ckpt_path = os.path.join(RESULTS, f"heal_{phase}{'_smoke' if smoke else ''}.pt")
    if mixed:
        cycle = tuple(p["ctx_cycle"])
        ctx_micro = dict(p["ctx_micro"])
        # World-size reshard (dist_util's module docstring): the PHASES table's
        # per-length (micro_B, accum) is a world=1 recipe. At world>1, holding
        # GLOBAL tokens/step (== windows_per_step_1gpu = micro_B*accum at world=1)
        # constant means shrinking accum by a factor of `world` while micro_B (a
        # GPU-memory-bound per-rank quantity) stays as configured; the divisibility
        # assert below is the launcher-level guard from the design (section 1) that
        # a bad (world, micro_B) combination fails loudly instead of silently
        # drifting global tokens/step.
        if world > 1:
            resharded = {}
            for c, (mb, ga) in ctx_micro.items():
                try:
                    resharded[c] = (mb, dist_util.assert_valid_world_config(mb * ga, mb, world))
                except AssertionError as e:
                    # dist_util.assert_valid_world_config's own message identifies
                    # the (windows_per_step_target, micro_B, world) mismatch but
                    # not WHICH curriculum length it was resharding for -- add
                    # that here so a stage-3 launch failure names the exact ctx.
                    raise AssertionError(
                        f"[{phase}] stage-3 world-size reshard failed for ctx={c} "
                        f"(micro_B={mb}, accum={ga}, world={world}): {e}"
                    ) from e
            ctx_micro = resharded
        tps_set = {c * mb * ga * world for c, (mb, ga) in ctx_micro.items()}
        assert len(tps_set) == 1, f"tokens/step not invariant across lengths: {ctx_micro}"
        tokens_per_step = tps_set.pop()
        ctx = max(ctx_micro)                      # longest ctx (rope tables per-length below)
        sig_sched = f"cycle={cycle} micro={sorted(ctx_micro.items())}"
        log(f"[{phase}] MIXED-LENGTH schedule: per-step ctx cycle {cycle} "
            f"(4K:8K:16K = {cycle.count(4096)}:{cycle.count(8192)}:{cycle.count(16384)} per "
            f"cycle), per-rank-ctx (micro,accum)={ctx_micro} world={world} -> "
            f"global tokens/step={tokens_per_step} invariant at every length")
        log(f"[{phase}] NOTE: CLI --micro-batch/--grad-accum are IGNORED for mixed-length "
            "phases (the schedule fixes them per length)")
    else:
        cycle = ctx_micro = None
        sig_sched = None
        ctx = cfg.heal_ctx
        tokens_per_step = micro_batch * grad_accum * ctx * world
        # Belt-and-suspenders: main() already picks/validates grad_accum against
        # GLOBAL_TOKENS_PER_STEP for a real (non-smoke, non-shakedown) run, but
        # train() re-checks here too (repo convention: refuse rather than silently
        # drift -- see _data_guards/ES-2 tombstone's identical "belt only, not the
        # only place it's enforced" pattern above).
        if not smoke and not shakedown:
            assert tokens_per_step == GLOBAL_TOKENS_PER_STEP, (
                f"[{phase}] tokens/step={tokens_per_step} (micro_batch={micro_batch} x "
                f"grad_accum={grad_accum} x ctx={ctx} x world={world}) != "
                f"GLOBAL_TOKENS_PER_STEP={GLOBAL_TOKENS_PER_STEP} -- the multi-GPU sharding "
                "contract requires holding global tokens/step constant across world sizes; "
                "see dist_util.assert_valid_world_config"
            )
    total_steps = math.ceil(p["tokens"] / tokens_per_step)
    log(f"[{phase}] budget {p['tokens']/1e6:.0f}M tok  tps={tokens_per_step} -> {total_steps} steps  "
        f"lr={p['lr']} guard={budget_s/3600:.1f}h ctx={ctx} world={world} rank={rank} "
        f"comm_dtype={comm_dtype}")
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
        # ALL ranks independently torch.load + load_state_dict here (not just
        # rank 0 + DDP's constructor-time broadcast): the design's point is a
        # DIRECT check that every rank's own read of the checkpoint landed on
        # identical parameters, which a broadcast-from-rank-0 would silently
        # paper over (it would just overwrite a bad rank's params with rank 0's,
        # masking exactly the torn-read failure mode this guards against).
        ck = torch.load(ckpt_path, map_location=DEVICE)
        if ck.get("cfg_sig") == _sig(cfg, p["tokens"], sig_sched, comm_dtype, kd_sig):
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
                "(cfg/backend/token-budget changed -- OR, for an otherwise-unchanged older "
                f"checkpoint, STREAM_VERSION was bumped since it was written: currently "
                f"{STREAM_VERSION}, see this module's STREAM_VERSION docstring for what that "
                "bump means and why it isn't resumable). Move the stale checkpoint aside "
                "explicitly, or rerun without --resume after verifying which run it belongs to.")
            sys.exit(1)
        # Blocking rank-consistency check (design: "a rank-consistency check
        # verifies identical parameters before any step -- guards a torn read"):
        # every rank's own independent load above must have landed on the exact
        # same parameters. No-op at world=1.
        dist_util.assert_rank_consistent(model, DEVICE, tag=f"[{phase}] post-resume-load")

    # DDP construction happens AFTER set_trainable (the trainable set changes per
    # stage -- DDP's bucket/hook registration must see the final requires_grad
    # state) and after the resume-load block above (so DDP's own constructor-time
    # broadcast, if it ran, would be redundant with -- not a substitute for -- the
    # independent-load-then-verify sequence just above). No-op at world=1: `model`
    # stays the plain HBAModel, and `raw is model`, so every attribute access below
    # (model.lm_head, model._last_aux, model.state_dict(), ...) is unchanged from
    # the pre-multi-GPU code path.
    if dist_util.is_distributed():
        model = dist_util.wrap_ddp(model, local_rank, comm_dtype)
    raw = dist_util.raw_model(model)

    # Donor-KD teacher: a SEPARATE, frozen, freshly-loaded copy of the original
    # donor -- NOT `raw.donor` (that submodule is reused BY REFERENCE inside the
    # student and is one of the trainable={..., "embed"} groups in stage2, so it
    # drifts under training; using it as the teacher would make the target move
    # along with the student, which isn't distillation, just a no-op consistency
    # loss). Loaded fresh from the same DONOR_NAME the rest of the codebase uses
    # (see model.load_donor), eval() + requires_grad_(False), at COMPUTE_DTYPE
    # (bf16 on CUDA, matching the student's autocast compute dtype; fp32 on
    # MPS/CPU) -- at 0.5B this is ~1GB with no grad/optimizer state, co-resident
    # on the same device as the student for the pilot (see hba.kd's module
    # docstring for why a larger donor would need a different placement).
    teacher = None
    if kd_on:
        teacher, _ = load_donor(dtype=COMPUTE_DTYPE, device=DEVICE)
        teacher.eval()
        teacher.requires_grad_(False)
        rlog(f"[{phase}] donor KD ENABLED: frozen teacher ({DONOR_NAME}) loaded fresh at "
            f"dtype={COMPUTE_DTYPE}, kd_weight={kd_weight} kd_temp={kd_temp} -- additive to "
            "rehearsal, not a replacement (see this module's docstring)")

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
    cap = cfg.mem_elem_cap
    t_start = t_ckpt = time.time()
    tok_seen = step0 * tokens_per_step
    autocast = (DEVICE == "cuda")
    # tok/s measurement baseline (see warmup_steps docstring above): reset once
    # after `warmup_steps` steps complete, so logged tps thereafter excludes
    # the warmup wall-clock instead of averaging it in for the rest of the run.
    warm_t0 = warm_tok0 = None

    def save(step, done):
        # Collective: rank 0 writes (the atomic tmp+rename pattern, unchanged from
        # single-GPU), bracketed by barriers so no rank proceeds into the next
        # step's collective forward/backward while rank 0 is still mid-write, and
        # no rank races ahead reading a file rank 0 hasn't finished renaming into
        # place yet. `raw.state_dict()` -- the UNWRAPPED model -- so single-GPU
        # tools (evals.py, gates.py's reference check) load the checkpoint
        # unchanged regardless of what world size produced it. Optimizer state is
        # replicated identically across ranks by construction (every rank runs the
        # same AdamW update on the same post-all-reduce gradient), so rank 0's opt
        # state IS the whole truth -- no need to gather it from other ranks.
        dist_util.barrier()
        if rank == 0:
            save_ckpt_atomic({"model": raw.state_dict(), "opt": opt.state_dict(), "step": step,
                              "done": done, "cfg_sig": _sig(cfg, p["tokens"], sig_sched, comm_dtype, kd_sig),
                              "phase": phase, "tok_seen": tok_seen}, ckpt_path)
        dist_util.barrier()

    # in-training capability PANEL (probes.run_panel; docs/training-recipe.md,
    # "Monitoring") + the pre-registered early-stopping rules that consume its
    # log (early_stop.py). Cadence from arg override, else phase config (0 =
    # off). suffix/sfx matches the ckpt_path naming convention above.
    pe = probe_every if probe_every is not None else p.get("probe_every", 0)
    sfx = "_smoke" if smoke else ""
    probe_log_path = os.path.join(RESULTS, f"probe_log_{phase}{sfx}.jsonl")
    best_ckpt_path = os.path.join(RESULTS, f"heal_{phase}{sfx}_best.pt")
    # Milestone checkpoints (25/50/75% of the phase token budget) exist for
    # cross-run comparability at common token counts -- a concern independent
    # of the capability panel/probing, so this is initialized unconditionally
    # (a `--probe-every 0` run must still get milestone checkpoints).
    milestones_hit = {pct for pct in (25, 50, 75) if tok_seen >= pct / 100 * p["tokens"]}
    engine = probe_history = best_panel_mean = None
    if pe:
        if step0 == 0 and os.path.exists(probe_log_path):
            stale = probe_log_path + f".stale-{int(time.time())}"
            os.replace(probe_log_path, stale)
            log(f"[{phase}] fresh start: moved stale probe log aside -> {stale}")
        if step0 > 0:
            # truncate_probe_log filters to step<=step0-1 (deduped, last write
            # wins) AND atomically rewrites the file on disk to match -- not
            # just an in-memory filter -- so stale future-step/duplicate lines
            # (e.g. from a rolled-back ES-2 collapse an operator deliberately
            # cleared the tombstone for) can never get pooled back in on a
            # LATER resume. See early_stop.truncate_probe_log's docstring.
            probe_history = early_stop.truncate_probe_log(probe_log_path, max_step=step0 - 1)
            log(f"[{phase}] resume: probe log truncated to {len(probe_history)} firing(s) "
                f"at/before step {step0 - 1} -> {probe_log_path}")
        else:
            probe_history = []
        best_panel_mean = max([early_stop.panel_mean(f) or -1.0 for f in probe_history], default=-1.0)
        engine = early_stop.EarlyStopEngine(phase_budget_tokens=p["tokens"],
                                            warmup_end_tokens=p["warmup"] * tokens_per_step)
        enabled = [s.name for s in capability_probes.PANEL if s.enabled]
        log(f"[{phase}] PANEL every {pe} steps ({', '.join(enabled)}); ES-FLOOR at "
            f"{engine.floor_tokens/1e6:.0f}M tok; log -> {probe_log_path}")

    per_ctx = ({c: dict(tok=0, sec=0.0, steps=0, peak=0.0) for c in ctx_micro} if mixed else None)
    for step in range(step0, total_steps):
        lr = lr_at(step, p["warmup"], total_steps, p["lr"])
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_v = aux_v = kd_v = 0.0
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
            # world-size-invariant sharding: each rank pulls its OWN micro_B-sized
            # slice of this micro-step's global window range (dist_util's module
            # docstring / WindowStream.batch's docstring) -- a partition, not a
            # duplication, of the data every optimizer step consumes.
            ids = stream_s.batch(m0 + micro, rank=rank, world=world).to(DEVICE)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            ctx_ = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast \
                else torch.autocast(device_type="cpu", enabled=False)
            # DDP gradient sync: reduce ONCE per optimizer step, not once per
            # micro-step. no_sync() on every micro-step except the last accumulates
            # grads locally (each rank's .backward() below still runs, just without
            # triggering DDP's all-reduce hooks); the final micro-step's backward
            # runs OUTSIDE no_sync so its all-reduce also captures every earlier
            # micro-step's already-accumulated local gradient in the same bucket
            # flush. No-op context at world=1 (nullcontext).
            last_micro = (micro == ga_s - 1)
            sync_ctx = (model.no_sync() if (dist_util.is_distributed() and not last_micro)
                       else contextlib.nullcontext())
            with sync_ctx:
                with ctx_:
                    if mixed:
                        # chunked-CE forward: the [B,n,V] logit tensor is never
                        # materialized (a mixed-length OOM hazard at longer context).
                        # Exact same mean CE.
                        loss_lm = model(inp, cos_s, sin_s, cap, mode="train", loss_tgt=tgt)
                        aux = raw._last_aux
                        aux = aux if aux is not None else loss_lm.new_zeros(())
                    else:
                        # chunked CE here too: recompute-in-backward keeps the peak
                        # at one chunk's fp32 logits instead of the full [B*n, V]
                        # tensor (the micro-batch ceiling; see chunked_ce.py). This
                        # ONE student forward (return_hidden=True) feeds BOTH the
                        # CE below AND, when KD is on, the KD loss right after --
                        # the student is never run twice for the two losses.
                        hidden = model(inp, cos_s, sin_s, cap, mode="train",
                                       return_hidden=True)
                        aux = raw._last_aux
                        aux = aux if aux is not None else hidden.new_zeros(())
                        loss_lm = chunked_cross_entropy(hidden, raw.lm_head.weight,
                                                        tgt, bias=raw.lm_head.bias,
                                                        chunk_size=1024)
                        kd = hidden.new_zeros(())
                        if kd_on:
                            # Teacher forward: the frozen, separately-loaded donor's
                            # OWN dense attention, under no_grad -- the extra cost
                            # this stage-2 option adds (one dense donor forward per
                            # micro-step; see hba.kd's module docstring for why the
                            # teacher must be this frozen copy, not raw.donor).
                            with torch.no_grad():
                                teacher_logits = teacher(inp).logits
                            kd = chunked_kd_kl(hidden, teacher_logits, raw.lm_head.weight, tgt,
                                              bias=raw.lm_head.bias, temperature=kd_temp,
                                              chunk_size=1024)
                    loss = loss_lm + cfg.aux_w * aux + (kd_weight * kd if kd_on else 0.0)
                (loss / ga_s).backward()
            loss_v += float(loss_lm.detach()) / ga_s
            aux_v += float(aux.detach()) / ga_s
            if kd_on:
                kd_v += float(kd.detach()) / ga_s
            tok_seen += inp.numel()
        # Grad clip AFTER the final synced backward (the summarizer aux-loss
        # gradient-isolation property -- gates.gate_grad_isolation -- is per-rank
        # and untouched by DDP's all-reduce, which only sums the LM-path grads the
        # isolation rule already routes to q/k/v/o/mlp/embed).
        torch.nn.utils.clip_grad_norm_([pp for pp in model.parameters() if pp.requires_grad], 1.0)
        opt.step()
        if warmup_steps and warm_t0 is None and (step - step0 + 1) >= warmup_steps:
            warm_t0, warm_tok0 = time.time(), tok_seen
            rlog(f"[{phase}] warmup ({warmup_steps} steps) excluded from tok/s window -- "
                "resetting throughput baseline")
        if mixed:
            st = per_ctx[c_s]
            st["tok"] += mb_s * ga_s * c_s
            st["sec"] += time.time() - t_step
            st["steps"] += 1
            if DEVICE == "cuda":
                st["peak"] = max(st["peak"], torch.cuda.max_memory_allocated() / 2**30)

        # milestone checkpoints at 25/50/75% of the phase token budget (100% is
        # just the COMPLETE checkpoint `save(total_steps, True)` writes below) --
        # kept ALONGSIDE the best-panel checkpoint (below) so cross-run
        # comparisons always have a common-token-count checkpoint to compare at,
        # even if a run stops early on ES-1/ES-2: early stopping must never make
        # two runs incomparable. tok_seen advances IDENTICALLY on every rank (same
        # per-rank micro_batch/grad_accum/ctx everywhere), so every rank crosses a
        # milestone at the identical step with no broadcast needed -- only the
        # FILE WRITE itself needs to be collective (rank 0 + barriers).
        if milestones_hit is not None:
            for pct in (25, 50, 75):
                if pct not in milestones_hit and tok_seen >= pct / 100 * p["tokens"]:
                    milestones_hit.add(pct)
                    mpath = os.path.join(RESULTS, f"heal_{phase}{sfx}_milestone{pct}.pt")
                    dist_util.barrier()
                    if rank == 0:
                        save_ckpt_atomic({"model": raw.state_dict(), "opt": opt.state_dict(),
                                          "step": step + 1, "done": False,
                                          "cfg_sig": _sig(cfg, p["tokens"], sig_sched, comm_dtype, kd_sig),
                                          "phase": phase, "tok_seen": tok_seen,
                                          "milestone_pct": pct}, mpath)
                        log(f"[{phase}] milestone checkpoint {pct}% ({tok_seen/1e6:.0f}M tok) -> {mpath}")
                    dist_util.barrier()

        # mixed: log one FULL cycle every 20 steps (20 % len(cycle) == 0, so a bare
        # %20 gate would only ever show cycle position 0; the bracket accumulators
        # cover all lengths either way, but per-length loss visibility needs the
        # whole cycle logged). The tps MEASUREMENT below runs on EVERY rank
        # (needed for tps_out -- gates.check_training's per-rank throughput
        # channel, see train()'s docstring); only the actual log() call is
        # rank-0-only, to avoid world-way duplicate spam under torchrun (every
        # rank's loss/tok-s numbers are near-identical by construction anyway).
        if (step % 20 < len(cycle)) if mixed else (step % 20 == 0):
            el = time.time() - t_start
            if warm_t0 is not None:
                tps = (tok_seen - warm_tok0) / max(1e-6, time.time() - warm_t0)
            else:
                tps = (tok_seen - step0 * tokens_per_step) / max(1e-6, el)
            if tps_out is not None:
                tps_out.setdefault("series", []).append(tps)
            if rank == 0:
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
                wtag = f" world={world}" if world > 1 else ""
                kd_s = f" kd {kd_v:.4f}" if kd_on else ""
                log(f"[{phase}] step {step:6d}/{total_steps}{ctx_s_} lm {loss_v:.4f} ppl "
                    f"{math.exp(min(20, loss_v)):.2f} aux {aux_v:.4f}{kd_s} tok/s {tps:6.0f}{wtag} "
                    f"lr {lr:.2e} tok {tok_seen/1e6:.0f}M elapsed {el/3600:.2f}h{fam_s}")

        # ---- collective control flow (design: "ALL control flow is step-based or
        # rank-0-broadcast -- NEVER rank-local wall-clock") ----------------------
        # Rank-local time.time() reads drift: if every rank independently decided
        # "time to checkpoint" or "wall-clock guard hit" or consumed its OWN copy
        # of the early-stop verdict, ranks could take DIFFERENT branches at the
        # same step (e.g. rank 0 enters a collective save() while another rank
        # enters the next step's collective forward/backward) -- an NCCL deadlock,
        # not a crash. So: rank 0 ALONE evaluates the panel/early-stop engine (a
        # GPU-cost concern, not just a correctness one -- running the panel on
        # every rank would be redundant compute) and the two wall-clock checks,
        # packs the result into a 3-int control tensor, and broadcasts it; every
        # rank then branches on the IDENTICAL received values. No-op broadcast at
        # world=1 (dist_util.broadcast_ctrl returns the local tensor unchanged),
        # so this reduces to exactly the original single-process control flow.
        rank0_do = dict(save_ckpt=False, guard=False, es_action=0, es_verdict=None)
        if rank == 0:
            if pe and step % pe == 0:
                t_p = time.time()
                # Fixed seed reissued fresh every firing (see probes.PANEL_SEED's
                # docstring): the panel must present IDENTICAL synthetic items
                # every time it fires so accuracy deltas reflect the model, not
                # resampling. Runs on rank 0's `raw` (unwrapped) model only --
                # see module docstring: redundant on every rank, and DDP doesn't
                # proxy the attribute access (model.lm_head etc.) probes.py needs.
                rng = np.random.default_rng(capability_probes.PANEL_SEED)
                panel_out = capability_probes.run_panel(raw, tok, cfg, rng)
                meta = capability_probes.PANEL_BY_NAME
                accs = {k: v for k, v in panel_out.items() if meta[k].kind == "acc"}
                n_trials = {k: meta[k].trials for k in accs}
                val_loss = panel_out.get("val_loss_fixed", float("nan"))
                firing = dict(step=step, tokens=tok_seen, accs=accs, n_trials=n_trials,
                             val_loss=val_loss, wall_s=round(time.time() - t_p, 2))
                # kd: the most recent step's training-loss KD value, logged
                # alongside the panel firing purely for visibility (early_stop.py's
                # rules never key off it -- see that module's firing-schema
                # docstring, unchanged here). Omitted entirely when KD is off, so
                # a non-KD run's probe log is byte-identical to before this
                # feature existed.
                if kd_on:
                    firing["kd"] = kd_v
                early_stop.append_probe_log(probe_log_path, firing)
                probe_history.append(firing)
                msg = " ".join(f"{k}={v:.3f}" for k, v in accs.items())
                log(f"[{phase}] PANEL @ step {step}: {msg} val_loss={val_loss:.4f} "
                    f"({firing['wall_s']:.0f}s) tok={tok_seen/1e6:.0f}M")

                pm = early_stop.panel_mean(firing)
                if pm is not None and pm > best_panel_mean:
                    best_panel_mean = pm
                    save_ckpt_atomic({"model": raw.state_dict(), "opt": opt.state_dict(),
                                      "step": step + 1, "done": False,
                                      "cfg_sig": _sig(cfg, p["tokens"], sig_sched, comm_dtype, kd_sig),
                                      "phase": phase,
                                      "tok_seen": tok_seen, "panel_mean": pm}, best_ckpt_path)
                    log(f"[{phase}] new best-panel checkpoint (mean={pm:.3f}) -> {best_ckpt_path}")

                verdict = engine.evaluate(probe_history)
                if verdict.rule_fired == "ES-2":
                    log(f"[{phase}] *** ES-2 FORGETTING ABORT: {verdict.details} ***")
                    rank0_do["es_action"] = dist_util.ES_ACTION_ES2
                    rank0_do["es_verdict"] = verdict
                elif verdict.rule_fired == "ES-1":
                    log(f"[{phase}] *** ES-1 PLATEAU STOP: {verdict.details} ***")
                    rank0_do["es_action"] = dist_util.ES_ACTION_ES1
                    rank0_do["es_verdict"] = verdict
            if rank0_do["es_action"] == 0:
                # Only evaluated when ES didn't already decide to act this step --
                # matches the original code's control flow exactly (there, these
                # checks sit OUTSIDE the `if pe...` block and are simply never
                # reached once that block has returned).
                if time.time() - t_ckpt > 1200:
                    rank0_do["save_ckpt"] = True
                if time.time() - t_start > budget_s:
                    rank0_do["guard"] = True

        ctrl = dist_util.broadcast_ctrl(
            dist_util.make_ctrl(rank0_do["save_ckpt"], rank0_do["guard"], rank0_do["es_action"]), rank)
        do_save, do_guard, es_action = (int(x) for x in ctrl.tolist())

        if es_action == dist_util.ES_ACTION_ES2:  # collective forgetting-abort rollback
            # "rank-0 decides -> broadcast -> all ranks load the rollback
            # checkpoint -> barrier -> resume" (design section 4). Every rank
            # (not just rank 0) loads best_ckpt_path into its OWN model/opt here,
            # even though this process is about to exit right after -- this is
            # what makes the in-memory state consistent with what gets promoted
            # to ckpt_path below, and is the collective op a future code path
            # that continues past ES-2 (rather than halting) would rely on.
            rolled_back_to = None
            best = None
            if os.path.exists(best_ckpt_path):
                best = torch.load(best_ckpt_path, map_location=DEVICE)
                raw.load_state_dict(best["model"]); opt.load_state_dict(best["opt"])
                rolled_back_to = best["step"]
            dist_util.barrier()
            if rank == 0:
                verdict = rank0_do["es_verdict"]
                if best is not None:
                    # Promote the best-panel checkpoint's EXACT saved (model, opt,
                    # step) triple to be the phase checkpoint -- not a fresh save
                    # of the current (collapsed) step with rolled-back weights
                    # spliced in, which would desynchronize the optimizer state
                    # and step count from the weights. A later --resume (once the
                    # tombstone is deliberately deleted) then genuinely resumes
                    # training FROM the best-panel point, deterministically
                    # replaying the data stream from there -- the human diagnosing
                    # the collapse is exactly the one who should decide that's
                    # what they want (docs/training-recipe.md: "halt, roll back
                    # ... diagnose before continuing").
                    save_ckpt_atomic(best, ckpt_path)
                    log(f"[{phase}] rolled back to best-panel checkpoint @ step {rolled_back_to} "
                        f"(panel_mean={best.get('panel_mean')})")
                else:
                    log(f"[{phase}] WARNING: no best-panel checkpoint exists yet -- rollback "
                        "is a no-op (this should not happen: ES-2 requires a probe that once "
                        "reached running_max>=0.25, which implies at least one strong firing)")
                early_stop.write_tombstone(RESULTS, probe=verdict.details.get("probe"),
                                           step=step, tokens=tok_seen,
                                           rolled_back_to=rolled_back_to, phase=phase)
                log(f"[{phase}] tombstone written -> {early_stop.tombstone_path(RESULTS)}; "
                    "heal.py will refuse to start/resume ANY phase until it is diagnosed and "
                    "deleted deliberately")
            dist_util.barrier()
            return "es2_halt"
        if es_action == dist_util.ES_ACTION_ES1:  # clean plateau stop
            save(step + 1, True)
            if rank == 0:
                verdict = rank0_do["es_verdict"]
                _write_json_atomic(
                    dict(budget=p["tokens"], stopped_at_tokens=tok_seen, rule_fired="ES-1",
                        details=verdict.details),
                    os.path.join(RESULTS, f"{phase}{sfx}_early_stop.json"))
                log(f"[{phase}] clean stop @ step {step} ({tok_seen/1e6:.0f}M / "
                    f"{p['tokens']/1e6:.0f}M tok)")
            return "early_stop"
        if do_save:
            save(step + 1, False); t_ckpt = time.time(); rlog(f"[{phase}] checkpoint @ {step+1}")
        if do_guard:
            rlog(f"[{phase}] *** WALL-CLOCK GUARD {budget_s/3600:.1f}h HIT @ {step}; saving ***")
            save(step + 1, False); return "guard"
    save(total_steps, True)
    rlog(f"[{phase}] COMPLETE @ {total_steps} ({tok_seen/1e6:.0f}M tok)")
    return "done"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["stage1", "stage2", "stage3"], required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--probe-every", type=int, default=None,
                    help="override the in-training capability-panel cadence (steps; 0 "
                         "disables the panel AND early stopping). Default: phase config "
                         "(stage2/stage3 = 200).")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--budget-s", type=float, default=None)
    ap.add_argument("--tokens", type=float, default=None, help="override phase token budget")
    ap.add_argument("--warmup", type=int, default=None,
                    help="override phase LR-warmup steps (INTERNAL: used by gates."
                         "gate_ddp_equivalence to compress the DDP-vs-single-GPU equivalence "
                         "gate's 30-step comparison into a synthetic short schedule -- a real "
                         "warmup=200 would apply near-zero LR across all 30 steps and pass the "
                         "gate vacuously even with broken sharding).")
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
    ap.add_argument("--kd", action="store_true",
                    help="enable donor knowledge-distillation (STAGE-2 ONLY): a full-logit KL "
                         "between the student's next-token distribution and a frozen, "
                         "separately-loaded copy of the original donor, ADDITIVE to capability "
                         "rehearsal (not a replacement) -- see hba.kd's module docstring and "
                         "docs/training-recipe.md's stage-2 section. Refused (not ignored) on "
                         "stage1/stage3. Off by default.")
    ap.add_argument("--kd-weight", type=float, default=1.0,
                    help="weight on the donor-KD loss term (loss = loss_lm + cfg.aux_w * aux + "
                         "kd_weight * kd_loss when --kd is set). Default 1.0.")
    ap.add_argument("--kd-temp", type=float, default=1.0,
                    help="softmax temperature T for donor KD (see hba.kd's module docstring for "
                         "the T^2 rescaling this implies). Default 1.0 (no softening).")
    args = ap.parse_args()

    # Multi-GPU setup FIRST, before any CUDA allocation: torch.cuda.set_device
    # (inside setup_distributed) must run before build_hba's donor .to(device)
    # call, else every rank would allocate on GPU 0. No-op (rank=0, world=1,
    # local_rank=0) when not launched under torchrun -- every distributed call
    # anywhere below this point is behind an is_distributed() guard, so a plain
    # `python -m hba.heal` invocation is completely unaffected.
    rank, world, local_rank = dist_util.setup_distributed()
    try:
        _heal_main(args, rank, world, local_rank)
    finally:
        dist_util.cleanup_distributed()


def _heal_main(args, rank, world, local_rank):
    cfg = smoke_config() if args.smoke else HBAConfig()
    if args.attn_backend is not None:
        cfg.attn_backend = args.attn_backend
    if not hasattr(cfg, "seed"):
        cfg.seed = 0
    if args.tokens is not None:
        PHASES[args.phase]["tokens"] = args.tokens
    if args.warmup is not None:
        PHASES[args.phase]["warmup"] = args.warmup
    # CUDA default micro_batch=1 (PER RANK), grad_accum computed to hold GLOBAL
    # tokens/step == GLOBAL_TOKENS_PER_STEP constant across world size (dist_util's
    # module docstring): backward-pass activation memory at micro_batch=2 and
    # heal_ctx=4096 can exceed a 24GB-class card even with grad checkpointing +
    # bf16 autocast; raise --micro-batch on cards with more headroom, or add more
    # ranks (world) instead of raising grad_accum, to use extra GPUs.
    micro_batch = args.micro_batch or 1
    if args.smoke:
        grad_accum = args.grad_accum or 1
    elif "ctx_cycle" in PHASES[args.phase]:
        # mixed-length phases (stage3) ignore CLI micro/grad_accum entirely (see
        # train()'s own NOTE log) -- train() reshards PHASES[...]["ctx_micro"]'s
        # per-length accum by `world` itself. This value is unused; kept only so
        # the function signature below stays uniform across phases.
        grad_accum = args.grad_accum or 32
    else:
        # windows_per_step_target = GLOBAL_TOKENS_PER_STEP / ctx must be an
        # integer (true for every ctx this recipe uses: 4096/8192/16384 all
        # divide 131072), then (micro_batch, world) must divide THAT evenly --
        # dist_util.assert_valid_world_config raises with the exact mismatch
        # otherwise ("the launcher asserts... before launch", design section 6).
        assert GLOBAL_TOKENS_PER_STEP % cfg.heal_ctx == 0, (
            f"GLOBAL_TOKENS_PER_STEP={GLOBAL_TOKENS_PER_STEP} not divisible by "
            f"cfg.heal_ctx={cfg.heal_ctx}")
        windows_target = GLOBAL_TOKENS_PER_STEP // cfg.heal_ctx
        expected_accum = dist_util.assert_valid_world_config(windows_target, micro_batch, world)
        if args.grad_accum is not None:
            grad_accum = args.grad_accum
            # shakedown's mini training stage legitimately uses small custom
            # micro/accum values that do NOT hold GLOBAL_TOKENS_PER_STEP (see
            # gates.check_training) -- exempted the same way _data_guards exempts
            # it. A real heal (scripts/heal.sh) never passes --shakedown.
            if not args.shakedown and grad_accum != expected_accum:
                log(f"ABORT: --grad-accum {grad_accum} breaks the tokens/step invariance "
                    f"contract (micro_batch={micro_batch} world={world} ctx={cfg.heal_ctx} -> "
                    f"expected grad_accum={expected_accum} for global tokens/step="
                    f"{GLOBAL_TOKENS_PER_STEP}); omit --grad-accum to use the computed value, "
                    "or pass --shakedown if this really is a shakedown mini-run")
                sys.exit(1)
        else:
            grad_accum = expected_accum
    budget_s = args.budget_s or (600 if args.smoke else 20 * 3600)
    log(f"heal device={DEVICE} phase={args.phase} smoke={args.smoke} dtype={COMPUTE_DTYPE} "
        f"attn_backend={resolve_backend(cfg)} rank={rank}/{world} local_rank={local_rank}")

    # Donor KD is stage-2-only -- fail fast here (before gates/donor loading),
    # not just inside train() (train() re-checks this too for any other caller;
    # see its docstring -- same "belt only, not the only place it's enforced"
    # pattern as the ES-2/data-sanity checks right below).
    if args.kd and args.phase != "stage2":
        log(f"ABORT: --kd was passed with --phase {args.phase}, but donor KD is STAGE-2-ONLY "
            "(see heal.train's docstring for why). Rerun with --phase stage2, or drop --kd.")
        sys.exit(1)

    # ES-2 tombstone: fail fast, before spending any time on gates/donor loading
    # (train() re-checks this too -- see its docstring -- so this is belt only,
    # not the only place it's enforced).
    tomb = early_stop.check_tombstone(RESULTS)
    if tomb is not None:
        log(f"ABORT: {early_stop.tombstone_path(RESULTS)} exists (ES-2 forgetting-abort fired "
            f"at step {tomb['step']} / {tomb['tokens']/1e6:.0f}M tok, probe={tomb['probe']}) -- "
            "refusing to start or resume. Diagnose the collapse, then delete the tombstone "
            "deliberately once addressed.")
        sys.exit(1)

    # data sanity
    if not os.path.exists(os.path.join(DATA, "train.bin")):
        log("ABORT: data/train.bin missing -- run `python -m hba.data_prep` first"); sys.exit(1)

    if not args.skip_gates:
        # Per-rank correctness gates (design section 5, gate 1): torchrun launches
        # one independent process per rank, so simply running these gates in every
        # process -- no special multi-GPU code needed here -- already satisfies
        # "run identically on EVERY rank" and catches a single bad GPU (a rank
        # whose kernels/driver silently produce wrong numbers fails its OWN gate
        # run, right here, before that rank ever joins the DDP process group's
        # first collective).
        from .gates import gate_equivalence
        gmodel, gtok, gcfg = build_hba(cfg, dtype=torch.float32)
        backend = resolve_backend(gcfg)
        ok = (gate_equivalence(gmodel, gtok, gcfg)[0] and gate_causality(gmodel, gcfg)
              and gate_path_equivalence(gmodel, gcfg) and gate_grad_isolation(gmodel, gcfg))
        if ok and backend == "fused":
            # the fused backend must agree with the naive oracle + keep gradient isolation
            ok = (gate_fused_agreement(gmodel, gcfg)
                  and gate_grad_isolation(gmodel, gcfg, backend="fused"))
        if ok and args.kd:
            # Donor-KD's own correctness gate (chunked_kd_kl vs the reference_kd_kl
            # oracle) -- run ONLY when --kd is set (opt-in objective; see
            # gate_kd's docstring), so a plain rehearsal-only run pays nothing
            # extra here.
            ok = gate_kd(device="cuda" if DEVICE == "cuda" else None)
        del gmodel
        empty_cache()
        if not ok:
            # No barrier/broadcast here: this rank exits ALONE. Any rank whose
            # gates passed continues past this block toward train()'s
            # dist_util.wrap_ddp call, a collective (DDP's constructor
            # broadcasts rank-0 params) -- with this rank gone, that collective
            # has no matching peer and the surviving ranks hang until
            # torchrun's health-check/timeout tears down the whole job. That
            # teardown (not a clean collective abort here) is what actually
            # ends the run.
            log(f"CONVERSION GATES FAILED on rank {rank} -- aborting heal (fake numbers "
                "otherwise)")
            sys.exit(1)

    status = train(cfg, args.phase, args.resume, micro_batch, grad_accum, budget_s, args.smoke,
                   allow_small=args.allow_small_corpus, shakedown=args.shakedown,
                   probe_every=args.probe_every, rank=rank, world=world, local_rank=local_rank,
                   kd=args.kd, kd_weight=args.kd_weight, kd_temp=args.kd_temp)
    if status == "guard":
        log(f"[{args.phase}] NOT finished (wall-clock guard) -- rerun with --resume")
        sys.exit(3)
    if status == "es2_halt":
        log(f"[{args.phase}] HALTED by ES-2 (forgetting abort) -- see "
            f"{early_stop.tombstone_path(RESULTS)}; will not restart until diagnosed")
        sys.exit(4)
    if status == "early_stop":
        log(f"[{args.phase}] stopped early by ES-1 (plateau) -- see results/{args.phase}"
            f"{'_smoke' if args.smoke else ''}_early_stop.json")


if __name__ == "__main__":
    main()
