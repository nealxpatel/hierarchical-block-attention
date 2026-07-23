#!/usr/bin/env bash
# Full eval sweep (docs/evals.md): capability gates FIRST, then only sweep if they
# pass. Each stage / long cell is its own process (per-cell resumable). Refuses to
# start unless the shakedown PASSED on this machine (or --skip-shakedown).
set -uo pipefail
cd "$(dirname "$0")/.."
SKIP=0; for a in "$@"; do [ "$a" = "--skip-shakedown" ] && SKIP=1; done
if [ "$SKIP" -eq 0 ]; then
python3 - <<'PY' || { echo "REFUSING: shakedown not green on this machine (run scripts/shakedown.sh or --skip-shakedown)."; exit 1; }
import json, os, sys, torch
r = json.load(open("results/shakedown_report.json")) if os.path.exists("results/shakedown_report.json") else {}
if not r.get("PASS"): sys.exit(1)
now = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
sys.exit(0 if r.get("cuda") == now else 1)
PY
fi

echo "== G1: raw-donor induction (must pass) =="
python3 -m hba.evals --stage gate --which donor \
  || { echo "G1 FAILED -- raw donor shows no induction; harness/probe broken. STOP."; exit 1; }
echo "== G2: converted induction (post-heal; STOP if destroyed) =="
python3 -m hba.evals --stage gate --which converted \
  || { echo "G2 FAILED -- conversion destroyed induction (or no heal ckpt). STOP, do NOT sweep."; exit 1; }
python3 - <<'PY' || { echo "G2 FAILED -- conversion destroyed induction. STOP and diagnose (do NOT sweep)."; exit 1; }
import json, sys
r = json.load(open("results/hba_results.json"))
# threshold 0.3 == evals.verdict()'s ind_ok bar (chance ~1e-5)
sys.exit(0 if max(r.get(f"induction|converted|{n}", 0) for n in (2048, 4096, 8192)) >= 0.3 else 1)
PY

echo "== PPL + benchmarks =="
python3 -m hba.evals --stage ppl
python3 -m hba.evals --stage bench
echo "== needle sweep (short->long; 64K/128K last, own process each; H before O(n^2) D) =="
for L in 4096 16384 32768; do python3 -m hba.evals --stage needle --length $L; done
python3 -m hba.evals --stage hier --length 32768
for L in 65536 131072; do
  for M in H_flat H_hier D_yarn D; do python3 -m hba.evals --stage needle --length $L --method $M; done
  python3 -m hba.evals --stage hier --length $L
done
python3 -m hba.evals --stage summary
echo "eval sweep complete -- results/summary.md"
