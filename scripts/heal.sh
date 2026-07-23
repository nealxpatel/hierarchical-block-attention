#!/usr/bin/env bash
# Full healing run: stage 1 (attention-only) -> stage 2 (full-parameter, with
# capability rehearsal) -> stage 3 (length-curriculum extension); see
# docs/training-recipe.md for what each stage trains and why. Every stage is
# resumable; a stage MUST NOT start unless the previous one actually finished
# (hba.heal exits 3 on a wall-clock-guard stop, nonzero on any abort -- do not
# chain past a failure). Refuses to start unless the shakedown PASSED on this
# machine, unless --skip-shakedown is passed. Run under tmux/nohup for
# multi-hour training.
set -uo pipefail
cd "$(dirname "$0")/.."
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
# micro-batch/accum via a mixed-context schedule -- see hba/heal.py, PHASES["stage3"]).
# Conservative default: micro=2 on <=40GB-class cards, micro=8 above that. Stage 2
# trains the full parameter count (more activation memory than stage 1's attn-only
# set), so it drops one notch on constrained cards for unattended safety.
BIG=$(python3 -c "import torch; print(1 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 40e9 else 0)")
if [ "$BIG" = "1" ]; then M1=8; M2=8; else M1=2; M2=1; fi
M1=${HEAL_MICRO:-$M1}; M2=${HEAL_MICRO:-$M2}
A1=${HEAL_ACCUM:-$(( 32 / M1 ))}; A2=${HEAL_ACCUM:-$(( 32 / M2 ))}
echo "[heal] stage1 micro=$M1 accum=$A1 ; stage2 micro=$M2 accum=$A2"

python3 -m hba.heal --phase stage1 --resume --micro-batch "$M1" --grad-accum "$A1" ${ARGS[@]+"${ARGS[@]}"} \
  || { echo "stage1 did not complete (guard/interrupt/abort) -- rerun scripts/heal.sh to resume"; exit 3; }
python3 -m hba.heal --phase stage2 --resume --micro-batch "$M2" --grad-accum "$A2" ${ARGS[@]+"${ARGS[@]}"} \
  || { echo "stage2 did not complete (guard/interrupt/abort) -- rerun scripts/heal.sh to resume"; exit 3; }
python3 -m hba.heal --phase stage3 --resume ${ARGS[@]+"${ARGS[@]}"} \
  || { echo "stage3 did not complete (guard/interrupt/abort) -- rerun scripts/heal.sh to resume"; exit 3; }
echo "healing complete -- checkpoints in results/heal_stage1.pt, results/heal_stage2.pt, results/heal_stage3.pt"
