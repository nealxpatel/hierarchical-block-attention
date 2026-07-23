#!/usr/bin/env bash
# Pre-flight shakedown: run once on a new training machine before committing to a
# real healing run (docs/training-recipe.md, "Refuse to start"). Fails loudly and
# early; writes results/shakedown_report.json with a per-check PASS/FAIL and one
# overall verdict. scripts/heal.sh and scripts/evals.sh refuse to start unless
# this report shows PASS (or --skip-shakedown).
set -uo pipefail
cd "$(dirname "$0")/.."
PLANNED_TPS="${PLANNED_TPS:-5000}"     # override to match the card; see docs/training-recipe.md
fail() { echo "SHAKEDOWN ABORT: $*" >&2; exit 1; }

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
echo "-- pip installs (pinned) --"
pip install -q -r requirements.txt || fail "pip install failed"
pip install -q -e . || fail "package install failed"
if [ ! -f results/reference_logits.pt ]; then
  fail "results/reference_logits.pt missing -- export it on a reference machine with \`python -m hba.convert --export-ref\` and copy it here first"
fi

echo "======== STAGE 2: data (small slice) ========"
if [ ! -f data/train.bin ]; then
  python3 -m hba.data_prep --train-tokens 2e7 || fail "data prep failed"
fi
python3 - <<'PY' || fail "data sanity failed"
import json, os, numpy as np
m = json.load(open("data/meta.json")); print("meta", m)
for f in ("train.bin","val_books.bin","needle_books.bin"):
    p = os.path.join("data", f); assert os.path.exists(p), f"missing {f}"
    n = np.memmap(p, dtype=np.uint32, mode="r").shape[0]; print(f, n, "tokens")
    assert n > 1000, f"{f} suspiciously small"
assert m["vocab_size"] > 0
PY

echo "======== STAGES 3-5: model gates / training / eval ========"
python3 -m hba.gates --planned-tps "$PLANNED_TPS" --steps "${STEPS:-150}"
RC=$?
if [ $RC -ne 0 ]; then echo "SHAKEDOWN FAILED (see results/shakedown_report.json)"; exit $RC; fi
echo "SHAKEDOWN PASS -- safe to launch the full run on this machine."
