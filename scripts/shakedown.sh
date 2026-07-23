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
set -uo pipefail
cd "$(dirname "$0")/.."
PLANNED_TPS="${PLANNED_TPS:-5000}"     # override to match the card; see docs/training-recipe.md
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
python3 -m hba.gates --planned-tps "$PLANNED_TPS" --steps "$STEPS" $FASTFLAG
RC=$?
if [ $RC -ne 0 ]; then echo "SHAKEDOWN FAILED (see results/shakedown_report.json)"; exit $RC; fi
echo "SHAKEDOWN PASS -- safe to launch the full run on this machine."
