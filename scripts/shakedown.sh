#!/usr/bin/env bash
# Pre-flight shakedown: run once on a new training machine before committing to a
# real healing run (docs/training-recipe.md, "Refuse to start"). Fails loudly and
# early; writes results/shakedown_report.json with a per-check PASS/FAIL and one
# overall verdict. scripts/heal.sh and scripts/evals.sh refuse to start unless
# this report shows PASS (or --skip-shakedown).
#
# --fast: the provisioning profile (scripts/provision.sh, docker/README.md) for
# a pre-baked image where pins are already verified and data is already
# fetched+verified by the caller -- this script then skips the pip-install and
# data-download work below (both would otherwise cost real minutes and, worse,
# a pip install would be a silent network dependency the whole point of the
# baked image is to avoid) and hands off to `hba.gates --fast` for a reduced
# 50-step training measurement + single PPL eval cell. Full 150-step profile
# (the default, no flag) is unchanged and remains what's required before any
# multi-hour unattended run.
#
# Multi-GPU (WORLD > 1, single node): stages 1-2 below (env/pip/data) are
# UNCHANGED -- they run once, single-process, regardless of WORLD. Stage 3
# (model gates / training / eval) launches under `torchrun --standalone
# --nproc_per_node=$WORLD -m hba.gates --multi-gpu`, which adds gate
# gate_shard_partition (pure Python -- runs on every rank, trivially), plus
# gate_rank_consistency and gate_nccl_bandwidth (both require the initialized
# process group torchrun provides), and turns check_training's throughput
# check into an aggregate-tok/s + scaling-efficiency gate (design doc gate 5).
# Gate 1 (per-rank correctness) needs no special wiring: torchrun already runs
# hba.gates.check_gates identically in every rank's own process. gate 4
# (DDP-vs-single equivalence) is run as a SEPARATE step below, plain `python3`
# (not torchrun) -- it orchestrates its OWN world=1 and world=WORLD subprocess
# pair internally (gates.gate_ddp_equivalence) and must not be invoked
# recursively from inside an already-running torchrun rank. Gate 6 (kill one
# rank mid-run -> torchrun tears down -> relaunch --resume -> loss continues on
# trajectory) is a MANUAL drill, documented (not automated) below -- it is
# cheap to describe, not cheap to script unattended (it requires killing a
# live training process partway through a real multi-hour run, not a
# shakedown-scale smoke).
set -uo pipefail
cd "$(dirname "$0")/.."
PLANNED_TPS="${PLANNED_TPS:-5000}"     # override to match the card; see docs/training-recipe.md
WORLD="${WORLD:-1}"
fail() { echo "SHAKEDOWN ABORT: $*" >&2; exit 1; }

FAST=0
for a in "$@"; do
  case "$a" in
    --fast) FAST=1 ;;
  esac
done
if [ "$FAST" -eq 1 ]; then
  STEPS="${STEPS:-50}"
  echo "[shakedown] --fast profile: 50 measured train steps (warmup-excluded tok/s), no pip "
  echo "  installs, no data download, one PPL cell (docker/README.md)"
else
  STEPS="${STEPS:-150}"
fi

echo "======== STAGE 1: environment ========"
python3 - <<'PY' || fail "torch/CUDA not usable"
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")
else:
    print("WARNING: no CUDA -- shakedown will run in MPS/CPU smoke mode")
PY
if [ "$FAST" -eq 1 ]; then
  echo "-- skipping pip install: pre-baked image, pins already verified by provision.sh's env "
  echo "   pin assert (a network pip install here would be a FAIL, not a fallback -- stale "
  echo "   image means rebake, see docker/README.md)"
else
  echo "-- pip installs (pinned) --"
  pip install -q -r requirements.txt || fail "pip install failed"
  pip install -q -e . || fail "package install failed"
fi
if [ ! -f results/reference_logits.pt ]; then
  fail "results/reference_logits.pt missing -- export it on a reference machine with \`python -m hba.convert --export-ref\` and copy it here first"
fi

echo "======== STAGE 2: data (small slice) ========"
# HBA_DATA_DIR (src/hba/config.py's env-override) governs where data actually
# lives -- honored here too (not just by the python side) so
# scripts/provision.sh's data_gpu/ fetch target (exported as HBA_DATA_DIR) is
# the SAME directory both this script's own file-presence checks and
# hba.data_prep/hba.gates look at. Defaults to ./data, matching config.py.
DATADIR="${HBA_DATA_DIR:-data}"
if [ "$FAST" -eq 1 ]; then
  echo "-- fast profile: data assumed already fetched+verified by scripts/provision.sh -- "
  echo "   checking presence only (no download) in ${DATADIR}/"
  for f in train.bin val_books.bin needle_books.bin meta.json; do
    [ -f "${DATADIR}/$f" ] || fail "${DATADIR}/$f missing -- scripts/provision.sh's data-fetch step should have populated ${DATADIR}/ before --fast shakedown runs"
  done
else
  if [ ! -f "${DATADIR}/train.bin" ]; then
    python3 -m hba.data_prep --train-tokens 2e7 || fail "data prep failed"
  fi
fi
python3 - <<'PY' || fail "data sanity failed"
import json, os, numpy as np
from hba.config import DATA
m = json.load(open(os.path.join(DATA, "meta.json"))); print("meta", m, "DATA=", DATA)
for f in ("train.bin","val_books.bin","needle_books.bin"):
    p = os.path.join(DATA, f); assert os.path.exists(p), f"missing {f}"
    n = np.memmap(p, dtype=np.uint32, mode="r").shape[0]; print(f, n, "tokens")
    assert n > 1000, f"{f} suspiciously small"
assert m["vocab_size"] > 0
PY

echo "======== STAGES 3-5: model gates / training / eval ========"
FASTFLAG=""
[ "$FAST" -eq 1 ] && FASTFLAG="--fast"
if [ "$WORLD" -gt 1 ]; then
  echo "-- multi-GPU mode: WORLD=$WORLD (torchrun --standalone --nproc_per_node=$WORLD) --"
  torchrun --standalone --nproc_per_node="$WORLD" -m hba.gates \
    --planned-tps "$PLANNED_TPS" --steps "$STEPS" $FASTFLAG --multi-gpu
else
  python3 -m hba.gates --planned-tps "$PLANNED_TPS" --steps "$STEPS" $FASTFLAG
fi
RC=$?
if [ $RC -ne 0 ]; then echo "SHAKEDOWN FAILED (see results/shakedown_report.json)"; exit $RC; fi

if [ "$WORLD" -gt 1 ]; then
  echo "======== STAGE 6: DDP-vs-single-GPU equivalence gate (design doc gate 4) ========"
  # Plain python3, NOT torchrun -- this gate orchestrates its OWN world=1 and
  # world=WORLD subprocess pair internally (30 steps each, synthetic warmup=5
  # compressed-cosine schedule -- see gates.gate_ddp_equivalence's docstring
  # for why a real warmup=200 would pass this vacuously) and must not itself
  # run inside an already-running torchrun rank.
  python3 - "$WORLD" <<'PY' || fail "DDP-vs-single-GPU equivalence gate failed"
import sys
from hba.config import HBAConfig
from hba.gates import gate_ddp_equivalence
world = int(sys.argv[1])
ok = gate_ddp_equivalence(HBAConfig(), world=world)
sys.exit(0 if ok else 1)
PY
  echo ""
  echo "======== GATE 6 (manual drill, not automated): kill+resume at WORLD=$WORLD ========"
  echo "Not run by this script -- requires killing a LIVE multi-hour training process"
  echo "partway through a real run, not a shakedown-scale smoke. To verify manually once a"
  echo "real WORLD=$WORLD heal is underway (scripts/heal.sh with WORLD=$WORLD):"
  echo "  1. Let it run past the first checkpoint write (~20 min, see heal.py's 1200s cadence)."
  echo "  2. Kill ONE rank's process (e.g. \`kill\` the PID for local_rank=1) -- do NOT kill the"
  echo "     whole torchrun process group; the point is verifying a single-rank failure tears"
  echo "     the WHOLE run down (torchrun's default behavior) rather than hanging."
  echo "  3. Confirm torchrun exits nonzero and every other rank's process also exits (NCCL"
  echo "     detects the dropped peer -- if it hangs instead, that is a FAIL: report it)."
  echo "  4. Relaunch: \`WORLD=$WORLD scripts/heal.sh --resume\`."
  echo "  5. Confirm the loss log continues smoothly from the last checkpoint's step (no lm-loss"
  echo "     discontinuity beyond ordinary step-to-step noise) -- a discontinuity here means the"
  echo "     resumed stream/optimizer state diverged from what was actually checkpointed."
fi

echo "SHAKEDOWN PASS -- safe to launch the full run on this machine."
