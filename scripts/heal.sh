#!/usr/bin/env bash
set -uo pipefail
# Full healing run: stage 1 (attention-only) -> stage 2 (full-parameter, with
# capability rehearsal) -> stage 3 (length-curriculum extension); see
# docs/training-recipe.md for what each stage trains and why. Every stage is
# resumable; a stage MUST NOT start unless the previous one actually finished
# (hba.heal exits 3 on a wall-clock-guard stop, nonzero on any abort -- do not
# chain past a failure). Refuses to start unless the shakedown PASSED on this
# machine, unless --skip-shakedown is passed. Run under tmux/nohup for
# multi-hour training.
#
# Multi-GPU (single node, 1-8 GPUs): set WORLD to the GPU count. WORLD=1 (the
# default) is the plain `python3` path, bit-identical to before multi-GPU
# support existed. WORLD>1 launches under `torchrun --standalone
# --nproc_per_node=$WORLD`, one process per GPU; hba.heal reads torchrun's
# RANK/WORLD_SIZE/LOCAL_RANK env vars itself (hba.dist_util.setup_distributed)
# -- nothing else changes about this script's structure. Global tokens/step
# (windows_per_step * heal_ctx) is held CONSTANT across WORLD via a smaller
# per-rank grad_accum (dist_util.assert_valid_world_config); the divisibility
# check below runs BEFORE torchrun launches (fails in seconds, not after
# spending minutes loading a donor model on every rank).
cd "$(dirname "$0")/.."
WORLD="${WORLD:-1}"
if [ "$WORLD" -gt 1 ]; then
  python3 - "$WORLD" <<'PY' || { echo "REFUSING to start: WORLD=$WORLD does not divide the recipe's per-length windows_per_step evenly (see hba.dist_util.assert_valid_world_config) -- pick WORLD in {1,2,4,8} matching a micro_batch that divides windows_per_step/WORLD."; exit 1; }
import sys
sys.path.insert(0, "src")
from hba.heal import GLOBAL_TOKENS_PER_STEP
from hba.dist_util import assert_valid_world_config
world = int(sys.argv[1])
for ctx in (4096, 8192, 16384):        # every ctx this recipe's phases use
    assert GLOBAL_TOKENS_PER_STEP % ctx == 0
    wps = GLOBAL_TOKENS_PER_STEP // ctx
    # micro_batch=1 is the loosest constraint (largest set of valid worlds);
    # heal.py's own main() re-validates against the ACTUAL --micro-batch at
    # launch time -- this is the fast early sanity check, not the only gate.
    assert_valid_world_config(wps, micro_B=1, world=world)
print(f"WORLD={world}: tokens/step divisibility OK for ctx in (4096, 8192, 16384)")
PY
  LAUNCH=(torchrun --standalone --nproc_per_node="$WORLD")
  echo "[heal] multi-GPU: WORLD=$WORLD (torchrun --standalone --nproc_per_node=$WORLD)"
else
  LAUNCH=(python3)
fi
SKIP=0; ARGS=()
for a in "$@"; do if [ "$a" = "--skip-shakedown" ]; then SKIP=1; else ARGS+=("$a"); fi; done

if [ "$SKIP" -eq 0 ]; then
python3 - <<'PY' || { echo "REFUSING to start: shakedown not green on this machine. Run scripts/shakedown.sh (or pass --skip-shakedown)."; exit 1; }
import json, os, sys, torch
p = "results/shakedown_report.json"
if not os.path.exists(p): sys.exit(1)
r = json.load(open(p))
if not r.get("PASS"): sys.exit(1)
# same-machine check: GPU name must match the one the shakedown ran on
now = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
if r.get("cuda") != now:
    print(f"shakedown ran on {r.get('cuda')} but this machine is {now} -- re-run scripts/shakedown.sh here"); sys.exit(1)
print("shakedown PASS confirmed on this machine")
PY
fi

# GPU-memory-aware micro-batch for stage1/stage2 (stage3 fixes its own per-length
# micro-batch/accum via a mixed-context schedule -- see hba/heal.py, PHASES["stage3"];
# hba.heal reshards that table's per-length accum by WORLD internally, so nothing
# here needs to special-case stage3). Conservative default: micro=2 on <=40GB-class
# cards, micro=8 above that -- PER RANK (each GPU still holds its own micro_batch-
# sized activations; WORLD only changes how many accumulation steps are needed to
# reach the SAME global tokens/step, not the per-GPU memory footprint). Stage 2
# trains the full parameter count (more activation memory than stage 1's attn-only
# set), so it drops one notch on constrained cards for unattended safety.
BIG=$(python3 -c "import torch; print(1 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 40e9 else 0)")
if [ "$BIG" = "1" ]; then M1=8; M2=8; else M1=2; M2=1; fi
M1=${HEAL_MICRO:-$M1}; M2=${HEAL_MICRO:-$M2}
# Pre-flight: validate the ACTUAL (M1, WORLD) / (M2, WORLD) combos before
# computing A1/A2 below via bash's `$(( ... ))` integer division, which
# silently TRUNCATES TO 0 on a non-dividing combo (e.g. M1=8, WORLD=8 ->
# 32/(8*8) == 0 in bash, no error) instead of failing -- that 0 would reach
# hba.heal as `--grad-accum 0` and every one of the WORLD ranks would die in a
# python AssertionError mid-launch instead of refusing here, fast, with a
# message naming the exact bad combo. Only meaningful when A1/A2 are actually
# auto-computed below (HEAL_ACCUM unset) -- an explicit HEAL_ACCUM bypasses
# this division entirely and the caller owns its correctness.
if [ -z "${HEAL_ACCUM:-}" ]; then
python3 - "$WORLD" "$M1" "$M2" <<'PY' || { echo "REFUSING to start: micro-batch does not divide evenly for WORLD=$WORLD (see message above) -- set HEAL_MICRO to a smaller per-rank micro-batch, set HEAL_ACCUM explicitly, or reduce WORLD."; exit 1; }
import sys
sys.path.insert(0, "src")
from hba.heal import GLOBAL_TOKENS_PER_STEP
from hba.dist_util import assert_valid_world_config
world, m1, m2 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
wps = GLOBAL_TOKENS_PER_STEP // 4096   # stage1/stage2 both heal at ctx=4096 -> wps=32
for name, m in (("stage1 micro_batch M1", m1), ("stage2 micro_batch M2", m2)):
    accum = assert_valid_world_config(wps, micro_B=m, world=world)
    assert accum >= 1, f"{name}={m} at WORLD={world}: computed grad_accum={accum} < 1"
print(f"WORLD={world}: M1={m1} M2={m2} both divide windows_per_step={wps} evenly "
      "(grad_accum >= 1)")
PY
fi
# grad_accum holds GLOBAL tokens/step (32 windows @ ctx=4096) constant across
# WORLD -- omitting --grad-accum below would make hba.heal compute this same
# value itself (main()'s dist_util.assert_valid_world_config call), but it's
# computed here too so the echo line below is informative before any Python
# starts.
A1=${HEAL_ACCUM:-$(( 32 / (M1 * WORLD) ))}; A2=${HEAL_ACCUM:-$(( 32 / (M2 * WORLD) ))}
echo "[heal] world=$WORLD stage1 micro=$M1 accum=$A1 ; stage2 micro=$M2 accum=$A2"

"${LAUNCH[@]}" -m hba.heal --phase stage1 --resume --micro-batch "$M1" --grad-accum "$A1" ${ARGS[@]+"${ARGS[@]}"} \
  || { echo "stage1 did not complete (guard/interrupt/abort) -- rerun scripts/heal.sh to resume"; exit 3; }
"${LAUNCH[@]}" -m hba.heal --phase stage2 --resume --micro-batch "$M2" --grad-accum "$A2" ${ARGS[@]+"${ARGS[@]}"} \
  || { echo "stage2 did not complete (guard/interrupt/abort) -- rerun scripts/heal.sh to resume"; exit 3; }
"${LAUNCH[@]}" -m hba.heal --phase stage3 --resume ${ARGS[@]+"${ARGS[@]}"} \
  || { echo "stage3 did not complete (guard/interrupt/abort) -- rerun scripts/heal.sh to resume"; exit 3; }
echo "healing complete -- checkpoints in results/heal_stage1.pt, results/heal_stage2.pt, results/heal_stage3.pt"
