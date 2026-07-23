"""Single-node multi-GPU data-parallel primitives for heal.py (torchrun DDP,
1-8 GPUs). Everything here composes with the existing single-GPU path: at
world_size=1 (no torchrun launcher), `is_distributed()` is False and every
function in this module that heal.py calls becomes a no-op or an identity
pass-through, so single-GPU behavior is unchanged.

WHY A SEPARATE MODULE
----------------------
The distributed mechanics (process-group setup, the world-size-invariant
stream-sharding formula, the collective control-flow broadcast, the comm-hook
registration, the rank-consistency check) are orthogonal to heal.py's staged
healing recipe and to fam_data.py's rehearsal-mix generator. Keeping them here
lets both import a tested, pure-where-possible surface instead of duplicating
distributed boilerplate, and lets the g-enumeration formula (the load-bearing
piece: get it wrong and 4 GPUs silently train on duplicated data, quietly
turning a 4x run into a 1x run) be unit-tested without a GPU or a process
group at all -- see tests/test_distributed.py.

WORLD-SIZE-INVARIANT STREAM SHARDING
--------------------------------------
The single-GPU stream was a pure function of a "global micro index" `m` (see
heal.WindowStream, pre-multi-GPU: `g = m*B + b` for b in the per-step batch).
Generalizing to `world` ranks each pulling their own `micro_B`-sized slice:

    g = m * (world * micro_B) + rank * micro_B + b
      = step*(world*micro_B*accum) + micro*(world*micro_B) + rank*micro_B + b

(`m = step*accum + micro`, unchanged from the single-GPU definition.) For a
fixed step, as (rank, micro, b) range over their full domains, `rank*micro_B +
b` sweeps [0, world*micro_B) exactly once for each `micro`, and `micro *
(world*micro_B)` tiles that block `accum` times -- so the g-set for a step is
exactly the contiguous range `[step*P, step*P + P)` where `P = world *
micro_B * accum` (windows_per_step), REGARDLESS of how P factors into
(world, micro_B, accum). Two decompositions of the same P (e.g. resuming a
4-GPU run's checkpoint on 1 or 8 GPUs, with accum adjusted so `world*micro_B*
accum` is unchanged) enumerate the IDENTICAL per-step g-set -- partition, not
duplication: for fixed step, ranks' g-ranges are disjoint and their union is
that contiguous range, so every window is consumed exactly once across the
world. This is what world_g_set/gate_shard_partition below verify mechanically
rather than merely asserting by construction.

At world=1, rank=0, `micro_B` == the old single-rank batch B: g reduces to
`m*B + b`, bit-identical to the pre-multi-GPU formula.
"""

import math
import os
import time

import numpy as np
import torch

try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover -- torch always ships torch.distributed
    dist = None


# --------------------------------------------------------- g-enumeration (pure) --
def global_window_index(step, micro, b, rank, world, micro_B, accum):
    """The single source of truth for the world-size-invariant stream-sharding
    formula (see module docstring). Pure arithmetic -- no torch, no process
    group -- so it is unit-testable on CPU with no GPU/torchrun involved.

    step:    optimizer step (0-based)
    micro:   grad-accumulation micro-step within this optimizer step, [0, accum)
    b:       position within this rank's own micro-batch, [0, micro_B)
    rank:    this process's rank, [0, world)
    world:   number of ranks (data-parallel world size)
    micro_B: PER-RANK micro-batch size
    accum:   grad-accumulation steps per optimizer step
    """
    return (step * (world * micro_B * accum)
            + micro * (world * micro_B)
            + rank * micro_B
            + b)


def windows_per_step(world, micro_B, accum):
    """P = world*micro_B*accum: the number of distinct windows (across the whole
    world) consumed by one optimizer step. tokens_per_step = P * ctx."""
    return world * micro_B * accum


def step_g_set(step, world, micro_B, accum):
    """Every g consumed at `step` across the WHOLE world, for shard-partition
    gate / test use. O(P) -- only meant for small K-step checks, not training."""
    return {
        global_window_index(step, micro, b, rank, world, micro_B, accum)
        for rank in range(world)
        for micro in range(accum)
        for b in range(micro_B)
    }


def rank_g_set(step, world, micro_B, accum, rank):
    """The g-set a single rank consumes at `step` (its slice of step_g_set)."""
    return {
        global_window_index(step, micro, b, rank, world, micro_B, accum)
        for micro in range(accum)
        for b in range(micro_B)
    }


def assert_valid_world_config(windows_per_step_target, micro_B, world):
    """The launcher-side divisibility assert (design doc section 1): given a
    FIXED windows_per_step target (global_tokens_per_step / ctx), (micro_B,
    world) must divide it evenly so `accum` is a positive integer. Returns the
    resulting `accum`. Raises AssertionError with a message identifying the
    exact mismatch (not just a bare assert) -- this runs at launch, before any
    compute, specifically so a misconfigured (world, micro_B) combination fails
    loudly instead of silently drifting global tokens/step away from the
    pre-registered recipe."""
    denom = micro_B * world
    assert windows_per_step_target % denom == 0, (
        f"windows_per_step ({windows_per_step_target}) not divisible by "
        f"micro_B*world ({micro_B}*{world}={denom}) -- grad_accum would not be a "
        "positive integer; pick a micro_B/world combination that divides the "
        "target windows_per_step evenly (world in {1,2,4,8})"
    )
    accum = windows_per_step_target // denom
    assert accum >= 1, f"computed grad_accum={accum} < 1 for micro_B={micro_B} world={world}"
    return accum


# ------------------------------------------------------------- process group ----
def is_distributed():
    """True iff a torch.distributed process group is initialized (i.e. this
    process was launched under torchrun with world_size > 1). False -- and
    every other function below either no-ops or is simply never called -- for
    a plain `python -m hba.heal` single-GPU/CPU/MPS run."""
    return dist is not None and dist.is_available() and dist.is_initialized()


def _pick_backend(cuda_available=None):
    """Pure decision (no torch.distributed calls) so the B1 fix -- the hybrid
    "cpu:gloo,cuda:nccl" backend string on CUDA, plain "gloo" otherwise -- is
    unit-testable on CPU without a GPU or a process group; see
    setup_distributed's "DEVICE RULE FOR COLLECTIVES" docstring for why."""
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    return "cpu:gloo,cuda:nccl" if cuda_available else "gloo"


def setup_distributed():
    """Initialize the process group from torchrun's env vars (RANK, WORLD_SIZE,
    LOCAL_RANK) if present; a plain `python -m hba.heal` invocation (no
    torchrun) has none of these set and this is a no-op that returns (0, 1, 0)
    -- world=1 behavior is therefore bit-identical to before this module
    existed (no distributed calls are ever made on that path; see
    is_distributed() guards throughout heal.py).

    NCCL on CUDA, gloo otherwise (CPU/MPS smoke/dev runs under torchrun, and
    the CPU 2-process test in tests/test_distributed.py). `torch.cuda.
    set_device(local_rank)` is called BEFORE process-group init so every
    subsequent bare `.to("cuda")` / `hba.config.DEVICE`-targeted allocation in
    this process lands on the correct GPU without threading local_rank through
    the rest of the codebase.

    DEVICE RULE FOR COLLECTIVES (why the CUDA backend string below is
    "cpu:gloo,cuda:nccl", not a bare "nccl"): NCCL only operates on CUDA
    tensors. Several collectives in this module and in gates.py/heal.py
    deliberately build small CPU tensors (make_ctrl's 3-int control tensor,
    broadcast every training step; gates.check_training's tok/s all-reduce and
    its resume-verdict broadcast) because they are cheap scalars/flags that
    never need to live on the GPU. A bare "nccl" backend would crash the first
    time any of those hit a collective (NCCL cannot see a CPU tensor at all).
    The HYBRID backend string registers gloo for CPU-tensor collectives and
    NCCL for CUDA-tensor collectives on the SAME process group -- each
    collective call dispatches on its tensor's own device automatically, so
    every call site below (and any future one) is correct BY CONSTRUCTION
    without having to remember to .to(device) every small control/scalar
    tensor by hand. GPU-tensor collectives (params_fingerprint's fp64 buffer,
    all_gather rank-consistency, the bandwidth microbench) are already built on
    `device` by their callers and ride NCCL as before -- this changes nothing
    for them."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    # See "DEVICE RULE FOR COLLECTIVES" above: hybrid backend on CUDA so CPU
    # control/scalar tensors ride gloo and CUDA tensors ride NCCL in the same
    # process group; gloo-only (unchanged) off CUDA, where nothing is NCCL.
    backend = _pick_backend()
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return rank, world, local_rank


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def barrier():
    if is_distributed():
        dist.barrier()


# ------------------------------------------------------ DDP construction/comm ---
def wrap_ddp(model, local_rank, comm_dtype="fp32"):
    """Construct DistributedDataParallel over `model` (called AFTER
    model.set_trainable(...) -- the trainable set changes per healing stage,
    and DDP's bucket/hook registration must see the final requires_grad state).
    find_unused_parameters=False: a silently-unused trainable param hangs the
    all-reduce instead of erroring, so this is verified once per stage by the
    shakedown's rank-consistency + DDP-vs-single equivalence gates, not
    discovered mid-run.

    comm_dtype: 'fp32' (DDP's default bucket reduction -- the attention-only
    stage's small grads) or 'bf16' (registers the gradient-compression comm
    hook -- full-parameter stages, where halving all-reduce bytes matters on
    hardware without GPU-to-GPU P2P, i.e. reductions stage through host
    memory). This choice is recorded in heal._sig's `comm_dtype` field: it is
    a recipe decision (docs/training-recipe.md conventions -- every optimized
    numerical path is gated and its identity is part of the checkpoint
    signature), not a free runtime knob."""
    from torch.nn.parallel import DistributedDataParallel as DDP
    # Cheap DDP-safety assert (see attention.py's "no candidate blocks" aux
    # fallback in hba_attention_dense and _aux_kl_chunked): if the summarizers
    # are trainable, cfg.heal_ctx must exceed cfg.window, or a query's
    # candidate set can come back empty on every position, in which case that
    # fallback returns a graph-free zero aux -- the summarizer gets NO gradient
    # at all that step. With find_unused_parameters=False below, that is a
    # hang (DDP expects every trainable param's grad every step), not a
    # crash. Only checked when cfg/window/heal_ctx are cleanly available on
    # `model` (the plain pre-wrap HBAModel this is always called with); no
    # config in this repo currently sets ctx <= window, so this is a guard
    # against a future config, not a live bug.
    cfg = getattr(model, "cfg", None)
    if cfg is not None and hasattr(model, "summarizers"):
        summ_trainable = any(p.requires_grad for p in model.summarizers.parameters())
        ctx, window = getattr(cfg, "heal_ctx", None), getattr(cfg, "window", None)
        if summ_trainable and ctx is not None and window is not None:
            assert ctx > window, (
                f"wrap_ddp: summarizers are trainable but cfg.heal_ctx={ctx} <= "
                f"cfg.window={window} -- every query's candidate set can be empty, "
                "which starves the summarizer of gradient and hangs DDP's "
                "find_unused_parameters=False all-reduce (see attention.py's "
                "'no candidate blocks' aux fallback)"
            )
    device_ids = [local_rank] if torch.cuda.is_available() else None
    ddp = DDP(model, device_ids=device_ids, find_unused_parameters=False)
    register_comm_hook(ddp, comm_dtype)
    return ddp


def register_comm_hook(ddp_model, comm_dtype):
    """fp32 = no hook (DDP's built-in bucket all-reduce is already fp32).
    bf16 = torch.distributed.algorithms.ddp_comm_hooks.default_hooks.
    bf16_compress_hook: casts each gradient bucket to bf16 before the
    all-reduce and back to fp32 after, halving communicated bytes. Registered
    once, right after DDP construction (a comm hook can only be registered
    before the first backward)."""
    if comm_dtype == "fp32":
        return
    if comm_dtype == "bf16":
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        ddp_model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)
        return
    raise ValueError(f"unknown comm_dtype {comm_dtype!r} (expected 'fp32' or 'bf16')")


def raw_model(model):
    """Unwrap a DDP-wrapped model back to the underlying HBAModel (DDP does NOT
    proxy arbitrary attribute access to `.module` -- callers that need
    model.lm_head, model.cfg, model.core, etc. -- e.g. probes.run_panel,
    checkpoint state_dict save/load -- must go through the raw module, not the
    DDP wrapper, which only proxies __call__/forward)."""
    from torch.nn.parallel import DistributedDataParallel as DDP
    return model.module if isinstance(model, DDP) else model


# ---------------------------------------------------- collective control flow ---
# Control-tensor layout, broadcast from rank 0 every step: [do_save, do_guard,
# es_action]. Indices (position within the tensor, NOT values):
CTRL_IDX_SAVE, CTRL_IDX_GUARD, CTRL_IDX_ES = 0, 1, 2
CTRL_LEN = 3
# es_action VALUES (the int at CTRL_IDX_ES): 0 = none/continue, 1 = ES-1 plateau
# stop, 2 = ES-2 forgetting-abort halt+rollback. All three control-tensor fields
# are evaluated ONLY on rank 0 (wall clock reads, and the probe panel +
# early-stop engine, which only rank 0 runs) and broadcast so every rank takes
# the identical branch at the identical step -- see module docstring in
# heal.py's train() for why rank-local wall-clock decisions are a deadlock
# hazard under DDP (rank 0 enters a collective save while another rank enters
# the next step's collective forward/backward).
ES_ACTION_NONE, ES_ACTION_ES1, ES_ACTION_ES2 = 0, 1, 2


def make_ctrl(do_save=False, do_guard=False, es_action=0):
    # Deliberately a CPU tensor (no device= kwarg): see setup_distributed's
    # "DEVICE RULE FOR COLLECTIVES" docstring -- the hybrid "cpu:gloo,cuda:nccl"
    # backend routes this over gloo automatically, so it does not need to live
    # on the compute device just to participate in broadcast_ctrl's collective.
    return torch.tensor([int(do_save), int(do_guard), int(es_action)], dtype=torch.int64)


def broadcast_ctrl(ctrl, rank):
    """Broadcast the [do_save, do_guard, es_action] control tensor from rank 0.
    No-op (returns `ctrl` unchanged) when not distributed -- rank 0 IS the only
    rank, so its own locally-evaluated decision already applies directly."""
    if not is_distributed():
        return ctrl
    ctrl = ctrl.clone()
    dist.broadcast(ctrl, src=0)
    return ctrl


# --------------------------------------------------------- rank consistency -----
def params_fingerprint(model, device):
    """A single fp64 scalar summarizing every parameter tensor's values (sum of
    the elementwise sum over every parameter, in fp64 to limit cancellation).
    Not cryptographic -- it is a cheap CHECK, not a proof of exact equality --
    but a torn read, a rank that silently failed to load the checkpoint, or a
    rank training on a divergent stream all move this value by far more than
    fp reduction-order noise, which is all the tolerance below needs to catch."""
    total = torch.zeros((), dtype=torch.float64, device=device)
    for p in model.parameters():
        total += p.detach().to(torch.float64).sum()
    return total


def assert_rank_consistent(model, device, tol=1e-6, tag=""):
    """Blocking rank-consistency check (design doc gate #2 / the post-resume
    guard in heal.py): all ranks' parameters must be identical before any
    step runs. Gathers each rank's params_fingerprint and asserts the max
    pairwise spread is within `tol`. No-op (trivially consistent) at world=1.
    Returns the measured spread (0.0 if not distributed)."""
    if not is_distributed():
        return 0.0
    fp = params_fingerprint(model, device)
    world = dist.get_world_size()
    gathered = [torch.zeros_like(fp) for _ in range(world)]
    dist.all_gather(gathered, fp)
    vals = [float(g.item()) for g in gathered]
    spread = max(vals) - min(vals)
    assert spread <= tol, (
        f"[rank-consistency{(' ' + tag) if tag else ''}] FAILED: param fingerprint "
        f"spread {spread:.3e} across {world} ranks exceeds tol={tol:.1e} "
        f"(values={vals}) -- guards a torn checkpoint read; do not proceed"
    )
    return spread


# ------------------------------------------------------------- NCCL microbench --
def allreduce_bandwidth_microbench(size_gb=2.0, device=None, dtype=torch.bfloat16):
    """All-reduce a ~size_gb buffer and report measured BUS bandwidth in GB/s
    (design doc gate #7). Bus bandwidth (not algorithmic bandwidth): the
    standard ring-all-reduce correction factor 2*(world-1)/world converts
    measured throughput to the per-link bandwidth that is comparable across
    world sizes and is what actually characterizes "the box's comm reality"
    (host-staged all-reduce on consumer GPUs realistically lands 6-12 GB/s;
    below ~6 GB/s the gate aborts). GPU/NCCL-only in any meaningful sense --
    on CPU/gloo this still runs (useful for the CPU broadcast smoke test) but
    the reported number characterizes loopback/gloo, not the real interconnect,
    so callers should treat a passing number as informational off-GPU."""
    if not is_distributed():
        return None
    world = dist.get_world_size()
    n_elem = int(size_gb * (1024 ** 3) / torch.tensor([], dtype=dtype).element_size())
    buf = torch.ones(n_elem, dtype=dtype, device=device)
    barrier()
    t0 = time.time()
    dist.all_reduce(buf)
    if device is not None and "cuda" in str(device):
        torch.cuda.synchronize()
    dt = time.time() - t0
    bytes_moved = buf.numel() * buf.element_size()
    algo_bw = bytes_moved / dt  # bytes/s, one rank's view of the all-reduce
    bus_bw = algo_bw * 2 * (world - 1) / world if world > 1 else algo_bw
    return dict(seconds=dt, world=world, size_gb=size_gb, algo_bw_gbs=algo_bw / 1e9,
               bus_bw_gbs=bus_bw / 1e9)
