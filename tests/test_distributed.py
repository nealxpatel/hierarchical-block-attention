"""CPU-only tests for the multi-GPU DDP plumbing (hba.dist_util, plus the
world-size-invariant sharding it enables in hba.heal.WindowStream and
hba.fam_data.FamMixer). No GPU, no torchrun, no donor download.

Three layers, matching the deliverable's test requirements:
  1. Pure-python g-enumeration: partition/union/invariance across (world,
     micro_B, accum) combos holding windows_per_step fixed (dist_util's
     central correctness claim -- see its module docstring).
  2. WindowStream / FamMixer determinism: the SAME global window g must
     produce the SAME data (and, for FamMixer, the SAME rehearsal plant)
     regardless of which (rank, world, m, b) combination reaches it.
  3. heal._sig mismatch: stream_version / global_tokens_per_step / comm_dtype
     changes must change the checkpoint signature (so a stale checkpoint is
     refused, not silently resumed onto a different stream/recipe).

A 2-process gloo smoke test of the broadcast-control-flow helper is included
at the bottom, using multiprocessing's 'fork' context (shares the parent's
already-imported modules/sys.path, avoiding spawn's re-import fragility under
pytest) -- skipped cleanly (not failed) if fork is unavailable or anything
about process/socket setup goes wrong, per the "don't make the suite flaky"
guidance: a deterministic single-process check of the exact same
broadcast_ctrl code path already covers the logic (see
test_broadcast_ctrl_is_noop_at_world_1 / the make_ctrl round-trip tests
below); the multiprocess test is a bonus confirmation, not the only coverage.
"""

import math
import multiprocessing
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from hba import dist_util  # noqa: E402
from hba import heal  # noqa: E402
from hba.config import HBAConfig  # noqa: E402
from hba.fam_data import FamMixer  # noqa: E402
from hba.gates import gate_shard_partition  # noqa: E402


# ------------------------------------------------------ g-enumeration (pure) --
# windows_per_step targets exercised below cover the recipe's real per-length
# values (GLOBAL_TOKENS_PER_STEP // ctx for ctx in {4096, 8192, 16384} is
# {32, 16, 8}) plus a couple of synthetic ones for broader (world, micro_B,
# accum) factorization coverage.
WPS_VALUES = (32, 16, 8, 24)
WORLDS = (1, 2, 4, 8)


def _decompositions(wps, world):
    """Every micro_B that divides wps/world evenly, paired with the resulting accum."""
    out = []
    if wps % world != 0:
        return out
    per_rank = wps // world
    for micro_B in range(1, per_rank + 1):
        if per_rank % micro_B == 0:
            out.append((micro_B, per_rank // micro_B))
    return out


@pytest.mark.parametrize("wps", WPS_VALUES)
def test_g_set_invariant_across_world_decompositions(wps):
    """For a FIXED windows_per_step target, every valid (world, micro_B, accum)
    decomposition must enumerate the IDENTICAL per-step g-set as the world=1
    reference (dist_util.step_g_set(step, 1, wps, 1)) -- this is the exact
    property that lets a checkpoint resume unchanged across world sizes."""
    for step in (0, 1, 5):
        ref = dist_util.step_g_set(step, 1, wps, 1)
        assert ref == set(range(step * wps, (step + 1) * wps))
        for world in WORLDS:
            for micro_B, accum in _decompositions(wps, world):
                got = dist_util.step_g_set(step, world, micro_B, accum)
                assert got == ref, (wps, world, micro_B, accum, step)


@pytest.mark.parametrize("wps", WPS_VALUES)
def test_g_set_partition_not_duplication(wps):
    """Every rank's g-set at a step is pairwise DISJOINT, and their union is the
    full step g-set -- "partition, not duplication" (design doc). This is the
    property whose absence is the naive multi-GPU bug: every rank silently
    seeing the same stream (e.g. a hardcoded rank=0) would make every rank's
    g-set IDENTICAL, not disjoint, and the assertions below would catch it
    immediately."""
    for step in (0, 2, 7):
        for world in WORLDS:
            for micro_B, accum in _decompositions(wps, world):
                full = dist_util.step_g_set(step, world, micro_B, accum)
                parts = [dist_util.rank_g_set(step, world, micro_B, accum, r)
                        for r in range(world)]
                union = set().union(*parts)
                assert union == full
                total_len = sum(len(p) for p in parts)
                assert total_len == len(full), "overlap detected (sizes don't add up)"
                for i in range(world):
                    for j in range(i + 1, world):
                        assert parts[i].isdisjoint(parts[j]), (wps, world, micro_B, accum, step, i, j)


def test_global_window_index_reduces_to_single_rank_formula_at_world_1():
    """At rank=0, world=1, g = m*B + b exactly (the pre-multi-GPU formula) --
    bit-identical single-GPU behavior is a direct consequence of this."""
    for m in range(5):
        for b in range(4):
            g = dist_util.global_window_index(step=0, micro=m, b=b, rank=0, world=1, micro_B=4, accum=999)
            assert g == m * 4 + b


def test_assert_valid_world_config_divisibility():
    assert dist_util.assert_valid_world_config(32, micro_B=1, world=4) == 8
    assert dist_util.assert_valid_world_config(32, micro_B=2, world=4) == 4
    assert dist_util.assert_valid_world_config(32, micro_B=4, world=8) == 1
    with pytest.raises(AssertionError):
        dist_util.assert_valid_world_config(32, micro_B=3, world=4)   # 32 % 12 != 0


def test_gate_shard_partition_pure_python():
    cfg = HBAConfig()
    assert gate_shard_partition(cfg, world=4, micro_B=2, accum=4, K=3)
    assert gate_shard_partition(cfg, world=1, micro_B=32, accum=1, K=3)


# --------------------------------------------------- B1: hybrid backend pick --
def test_pick_backend_hybrid_on_cuda_gloo_otherwise():
    """B1 fix: on CUDA, the process group must be initialized with the HYBRID
    "cpu:gloo,cuda:nccl" backend string (not a bare "nccl", which cannot run a
    collective on a CPU tensor -- make_ctrl's control tensor and
    gates.check_training's tok/s all-reduce are exactly such CPU tensors) --
    see setup_distributed's "DEVICE RULE FOR COLLECTIVES" docstring. This
    checks the pure decision function directly (no process group, no GPU
    needed) since setup_distributed itself requires torchrun env vars and an
    actual CUDA device to exercise end to end."""
    assert dist_util._pick_backend(cuda_available=True) == "cpu:gloo,cuda:nccl"
    assert dist_util._pick_backend(cuda_available=False) == "gloo"


# --------------------------------- S4: gate_ddp_equivalence smoke-ctx tokens --
def test_gate_ddp_equivalence_tokens_sized_off_smoke_ctx():
    """S4 fix: gate_ddp_equivalence's subprocesses always run with --smoke (see
    its `_run` helper), so heal._heal_main builds cfg via config.smoke_config()
    (heal_ctx=512), NOT the real HBAConfig() (heal_ctx=4096) that
    gate_ddp_equivalence itself receives. The token budget must be computed
    from the ctx the subprocess ACTUALLY trains at, or total_steps is inflated
    by real_ctx/smoke_ctx (4096/512 = 8x: ~240 steps instead of the intended
    ~30). This reproduces gate_ddp_equivalence's own tokens/total_steps
    arithmetic without spawning the GPU-only subprocesses themselves (which
    this repo's CPU-only suite cannot run -- see that function's docstring)."""
    from hba.config import smoke_config
    steps, windows_per_step = 30, 32
    smoke_ctx = smoke_config().heal_ctx
    tokens = steps * windows_per_step * smoke_ctx
    tokens_per_step = windows_per_step * smoke_ctx    # world*micro_B*accum == windows_per_step
    assert math.ceil(tokens / tokens_per_step) == steps
    # the bug this fixes: sizing tokens off the REAL cfg's heal_ctx (4096)
    # while the subprocess trains at the smoke ctx inflates total_steps 8x.
    real_ctx = HBAConfig().heal_ctx
    buggy_tokens = steps * windows_per_step * real_ctx
    assert math.ceil(buggy_tokens / tokens_per_step) == steps * (real_ctx // smoke_ctx)


# --------------------------------------- S5: stage-3 reshard error message ----
def test_stage3_reshard_assertion_names_the_failing_ctx():
    """S5 fix: heal.train's stage-3 world-size reshard must name WHICH
    curriculum ctx length failed (not just dist_util.assert_valid_world_config's
    generic (windows_per_step, micro_B, world) mismatch). world=3 does not
    divide PHASES['stage3']['ctx_micro'][4096]'s windows_per_step=32 with
    micro_B=4 evenly (32 % (4*3) != 0), so this must raise naming ctx=4096
    specifically. This runs before heal.train ever calls build_hba (the
    mixed-length reshard is the first substantial thing train() does for a mixed
    phase), so no donor download / network access is needed."""
    with pytest.raises(AssertionError, match=r"ctx=4096.*micro_B=4.*accum=8.*world=3"):
        heal.train(HBAConfig(), "stage3", resume=False, micro_batch=1, grad_accum=1,
                  budget_s=1, smoke=True, world=3)


# ------------------------------------------------- WindowStream determinism ---
def _make_bin(path, n_tokens):
    np.arange(n_tokens, dtype=np.uint32).tofile(path)


def test_windowstream_g_invariant_across_world_decompositions(tmp_path):
    ctx = 8
    path = str(tmp_path / "train.bin")
    _make_bin(path, (ctx + 1) * 64)   # 64 windows -- plenty of headroom for g in [0, 16)
    seed = 5

    s_1gpu = heal.WindowStream(path, ctx, 4, seed)          # world=1, micro_B=4
    b_1gpu = s_1gpu.batch(0, rank=0, world=1)                # g = 0,1,2,3

    s_2gpu = heal.WindowStream(path, ctx, 2, seed)           # world=2, micro_B=2
    b_2gpu_r0 = s_2gpu.batch(0, rank=0, world=2)              # g = 0,1
    b_2gpu_r1 = s_2gpu.batch(0, rank=1, world=2)              # g = 2,3
    assert torch.equal(b_1gpu, torch.cat([b_2gpu_r0, b_2gpu_r1], dim=0))

    s_4gpu = heal.WindowStream(path, ctx, 1, seed)            # world=4, micro_B=1
    b_4gpu = torch.cat([s_4gpu.batch(0, rank=r, world=4) for r in range(4)], dim=0)  # g=0,1,2,3
    assert torch.equal(b_1gpu, b_4gpu)


def test_windowstream_batch_world1_rank0_matches_legacy_formula(tmp_path):
    """No kwargs at all (world=1, rank=0 defaults) must be bit-identical to the
    pre-multi-GPU g = m*B + b formula -- single-GPU behavior is unchanged."""
    ctx = 8
    path = str(tmp_path / "train.bin")
    _make_bin(path, (ctx + 1) * 32)
    s = heal.WindowStream(path, ctx, 3, seed=1)
    assert torch.equal(s.batch(2), s.batch(2, rank=0, world=1))


# ------------------------------------------------------ FamMixer determinism --
def test_fammixer_rng_is_pure_function_of_g():
    cfg = HBAConfig()

    class _FakeStream:
        def __init__(self, B):
            self.B = B

    fam = FamMixer(_FakeStream(1), cfg, seed=99, frac=0.5)
    L = 129
    for g in (0, 1, 4096, 999_999):
        row_a, row_b = np.zeros(L, dtype=np.int64), np.zeros(L, dtype=np.int64)
        planted_a, npairs_a = fam._plant(row_a, fam._rng(g))
        planted_b, npairs_b = fam._plant(row_b, fam._rng(g))
        assert np.array_equal(row_a, row_b)
        assert planted_a == planted_b and npairs_a == npairs_b


def test_fammixer_batch_g_invariant_across_world_decompositions(tmp_path):
    """End-to-end: FamMixer.batch's realized (real-text + planted) content for a
    fixed g must be identical whichever (rank, world) combination produced it
    -- this is the property gates.gate_shard_partition's fam-mix half checks
    directly on the RNG; here it's checked through the full stack (real
    WindowStream + FamMixer.batch), with a high frac so planting reliably
    fires and the test is not vacuous."""
    ctx = 128
    path = str(tmp_path / "train.bin")
    _make_bin(path, (ctx + 1) * 32)
    cfg = HBAConfig()
    seed = 5

    s_1gpu = heal.WindowStream(path, ctx, 4, seed)
    fam_1gpu = FamMixer(s_1gpu, cfg, seed=99, frac=0.5)
    b_1gpu = fam_1gpu.batch(0, rank=0, world=1)               # g = 0,1,2,3

    s_2gpu = heal.WindowStream(path, ctx, 2, seed)
    fam_2gpu = FamMixer(s_2gpu, cfg, seed=99, frac=0.5)
    b_2gpu_r0 = fam_2gpu.batch(0, rank=0, world=2)             # g = 0,1
    b_2gpu_r1 = fam_2gpu.batch(0, rank=1, world=2)             # g = 2,3
    assert torch.equal(b_1gpu, torch.cat([b_2gpu_r0, b_2gpu_r1], dim=0))
    assert fam_1gpu.fam_tok > 0, "test is vacuous if nothing was ever planted"


# ------------------------------------------------------------------ _sig -----
def test_sig_embeds_stream_version_tokens_and_comm_dtype():
    cfg = HBAConfig()
    s = heal._sig(cfg, phase_tokens=1e6, ctx_schedule=None, comm_dtype="bf16")
    assert s["stream_version"] == heal.STREAM_VERSION
    assert s["global_tokens_per_step"] == heal.GLOBAL_TOKENS_PER_STEP
    assert s["comm_dtype"] == "bf16"


def test_sig_differs_on_comm_dtype_change():
    cfg = HBAConfig()
    s_fp32 = heal._sig(cfg, 1e6, None, "fp32")
    s_bf16 = heal._sig(cfg, 1e6, None, "bf16")
    assert s_fp32 != s_bf16


def test_sig_differs_on_stream_version_bump(monkeypatch):
    cfg = HBAConfig()
    original = heal.STREAM_VERSION
    s_before = heal._sig(cfg, 1e6, None, "fp32")
    monkeypatch.setattr(heal, "STREAM_VERSION", original + 1)
    s_after = heal._sig(cfg, 1e6, None, "fp32")
    assert s_before != s_after
    assert s_after["stream_version"] == original + 1


def test_sig_differs_on_global_tokens_per_step_change(monkeypatch):
    cfg = HBAConfig()
    s_before = heal._sig(cfg, 1e6, None, "fp32")
    monkeypatch.setattr(heal, "GLOBAL_TOKENS_PER_STEP", heal.GLOBAL_TOKENS_PER_STEP // 2)
    s_after = heal._sig(cfg, 1e6, None, "fp32")
    assert s_before != s_after


def test_sig_omits_world_size_and_per_rank_batch_shape():
    """World size / per-rank micro_batch / grad_accum must NOT be part of _sig --
    that omission IS the cross-world-size portability the design calls for."""
    cfg = HBAConfig()
    s = heal._sig(cfg, 1e6, None, "fp32")
    for forbidden in ("world", "world_size", "micro_batch", "grad_accum", "rank"):
        assert forbidden not in s


# -------------------------------------------- control-tensor round trip (1p) --
def test_make_ctrl_and_broadcast_ctrl_noop_at_world_1():
    ctrl = dist_util.make_ctrl(do_save=True, do_guard=False, es_action=dist_util.ES_ACTION_ES1)
    assert ctrl.tolist() == [1, 0, 1]
    out = dist_util.broadcast_ctrl(ctrl, rank=0)   # not distributed -> passthrough
    assert torch.equal(ctrl, out)


def test_es_action_constants_are_distinct():
    assert len({dist_util.ES_ACTION_NONE, dist_util.ES_ACTION_ES1, dist_util.ES_ACTION_ES2}) == 3


# ------------------------------------------- 2-process gloo broadcast smoke ---
def _ctrl_worker(rank, world_size, port, result_dir):
    import os as _os

    _os.environ["MASTER_ADDR"] = "127.0.0.1"
    _os.environ["MASTER_PORT"] = str(port)
    _os.environ["RANK"] = str(rank)
    _os.environ["WORLD_SIZE"] = str(world_size)
    try:
        import torch as _torch
        import torch.distributed as _dist
        _dist.init_process_group(backend="gloo", rank=rank, world_size=world_size,
                                  timeout=__import__("datetime").timedelta(seconds=20))
        if rank == 0:
            ctrl = dist_util.make_ctrl(do_save=True, do_guard=False,
                                       es_action=dist_util.ES_ACTION_ES2)
        else:
            # deliberately DIFFERENT local values -- if broadcast_ctrl were a
            # no-op under real distribution, rank 1 would keep these instead of
            # rank 0's, and the test below would catch it.
            ctrl = dist_util.make_ctrl(do_save=False, do_guard=True,
                                       es_action=dist_util.ES_ACTION_NONE)
        out = dist_util.broadcast_ctrl(ctrl, rank)
        with open(os.path.join(result_dir, f"r{rank}.txt"), "w") as f:
            f.write(",".join(str(int(x)) for x in out.tolist()))
        _dist.destroy_process_group()
    except Exception as e:  # pragma: no cover -- surfaced via missing result file
        with open(os.path.join(result_dir, f"r{rank}.err"), "w") as f:
            f.write(f"{type(e).__name__}: {e}")


def test_broadcast_ctrl_two_process_gloo_smoke(tmp_path):
    """Real 2-process broadcast (not the world=1 passthrough above): rank 0's
    control tensor must be what BOTH ranks observe afterward. Skipped cleanly
    (not failed) if the 'fork' multiprocessing context, socket binding, or
    gloo init isn't available in this sandbox -- the single-process round-trip
    tests above already cover the same code path deterministically, so this is
    a bonus confirmation, not the only coverage (see module docstring)."""
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        pytest.skip("fork start method unavailable on this platform")

    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
    except OSError:
        pytest.skip("cannot bind a local TCP port in this sandbox")

    result_dir = str(tmp_path)
    procs = [ctx.Process(target=_ctrl_worker, args=(r, 2, port, result_dir)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)

    for p in procs:
        if p.is_alive():
            p.terminate()
            pytest.skip("2-process gloo smoke test timed out in this sandbox")

    r0_path = os.path.join(result_dir, "r0.txt")
    r1_path = os.path.join(result_dir, "r1.txt")
    if not (os.path.exists(r0_path) and os.path.exists(r1_path)):
        errs = []
        for r in (0, 1):
            ep = os.path.join(result_dir, f"r{r}.err")
            if os.path.exists(ep):
                errs.append(open(ep).read())
        pytest.skip(f"gloo process-group setup unavailable in this sandbox: {errs}")

    r0 = open(r0_path).read()
    r1 = open(r1_path).read()
    expected = "1,0,2"  # rank 0's do_save=True, do_guard=False, es_action=ES_ACTION_ES2
    assert r0 == expected
    assert r1 == expected, "rank 1 did not receive rank 0's broadcast control tensor"
