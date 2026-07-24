#!/usr/bin/env bash
# Rent-to-ready entrypoint (docker/Dockerfile ENTRYPOINT; see docker/README.md
# for the full flow and every env var below). Runs, in order:
#   a. expected-manifest assert   b. env pin assert
#   c. data verify (launcher rsyncs data in beforehand; bucket fetch is an
#      optional fallback if rsync'd data isn't present)   d. fast shakedown
# Every step prints a PASS/FAIL line; ANY FAIL exits nonzero immediately. A
# final GREEN/RED banner reports total elapsed time against the <=15 min
# rent-to-shakedown-green target.
set -uo pipefail
cd "$(dirname "$0")/.."

PROV_T0=$(date +%s)

pass() { echo "PASS: $*"; }
fail() {
  echo "FAIL: $*" >&2
  echo
  echo "############################################################"
  echo "PROVISION RED  ($(( $(date +%s) - PROV_T0 ))s elapsed)"
  echo "############################################################"
  exit 1
}

echo "======== STEP a: expected-manifest assert ========"
# Rationale (docker/README.md, "code-overlay update flow"): an on-box-only
# manifest check is self-consistent even when a code-overlay push was
# skipped -- stale code verifies against its own stale manifest. Anchoring to
# a hash the LAUNCHER computed (on its own machine, from the code it intended
# to ship) makes a skipped push a loud FAIL instead of a silent wrong-code
# run.
: "${EXPECTED_MANIFEST_SHA:?EXPECTED_MANIFEST_SHA not set -- the launcher must compute this itself at launch time (scripts/make_manifest.sh against the code it intends to ship) and pass it in. Refusing to fall back to an on-box-only check (see the rationale above).}"
ACTUAL_MANIFEST_SHA="$(bash scripts/make_manifest.sh)" || fail "scripts/make_manifest.sh failed to run on-box"
if [ "$ACTUAL_MANIFEST_SHA" != "$EXPECTED_MANIFEST_SHA" ]; then
  fail "manifest mismatch: on-box=${ACTUAL_MANIFEST_SHA} launcher-expected=${EXPECTED_MANIFEST_SHA} -- code on this box does not match what the launcher intended to ship (a tampered file, a stale overlay, or a skipped overlay push all look like this). Re-run the launcher's code-overlay step; if this is a dependency/entrypoint change, rebake instead (docker/README.md, \"rebake threshold\")."
fi
pass "manifest verify: on-box=${ACTUAL_MANIFEST_SHA} matches launcher-provided EXPECTED_MANIFEST_SHA"

echo "======== STEP b: env pin assert ========"
# A network pip install here would be a FAIL, not a fallback -- it means the
# image is stale (rebake; docker/README.md). This step only ever VERIFIES.
ACTUAL_PINS="$(python3 - <<'PY'
import torch, transformers, tokenizers
print(f"{torch.__version__}|{transformers.__version__}|{tokenizers.__version__}")
PY
)" || fail "could not import torch/transformers/tokenizers to check pins -- image is broken, rebake"
A_TORCH="${ACTUAL_PINS%%|*}"; _REST="${ACTUAL_PINS#*|}"
A_TRANSFORMERS="${_REST%%|*}"; A_TOKENIZERS="${_REST#*|}"

: "${HBA_PINNED_TORCH:?HBA_PINNED_TORCH not set -- this image was not built from docker/Dockerfile (stale/foreign image)}"
EXP_TORCH="$HBA_PINNED_TORCH"
EXP_TRANSFORMERS="$(grep -E '^transformers==' docker/requirements.txt | cut -d= -f3)"
EXP_TOKENIZERS="$(grep -E '^tokenizers==' docker/requirements.txt | cut -d= -f3)"
[ -n "$EXP_TRANSFORMERS" ] || fail "could not read the transformers pin out of docker/requirements.txt"
[ -n "$EXP_TOKENIZERS" ] || fail "could not read the tokenizers pin out of docker/requirements.txt"

PIN_MISMATCH=0
check_pin() {  # name actual expected
  if [ "$2" != "$3" ]; then
    echo "  FAIL pin $1: actual=$2 expected=$3"
    PIN_MISMATCH=1
  else
    echo "  ok   pin $1: $2"
  fi
}
# torch carries a local-version build suffix (e.g. 2.10.0+cu128) that the base
# image tag fixes -- the CUDA build is pinned by the FROM line, so the version
# pin asserts the BASE version. Match the base (strip +local) unless the
# operator deliberately pinned the exact build in HBA_PINNED_TORCH.
A_TORCH_BASE="${A_TORCH%%+*}"
if [ "$A_TORCH" = "$EXP_TORCH" ] || [ "$A_TORCH_BASE" = "$EXP_TORCH" ]; then
  echo "  ok   pin torch: $A_TORCH"
else
  echo "  FAIL pin torch: actual=$A_TORCH (base $A_TORCH_BASE) expected=$EXP_TORCH"
  PIN_MISMATCH=1
fi
check_pin transformers "$A_TRANSFORMERS" "$EXP_TRANSFORMERS"
check_pin tokenizers "$A_TOKENIZERS" "$EXP_TOKENIZERS"
[ "$PIN_MISMATCH" -eq 0 ] || fail "env pin mismatch -- image is stale. Rebake (docker/README.md); a network pip install during provisioning is NOT a valid fallback here."
pass "env pin assert: torch=${A_TORCH} transformers=${A_TRANSFORMERS} tokenizers=${A_TOKENIZERS}"

echo "======== STEP c: data verify (rsync-primary; bucket fetch is the optional fallback) ========"
# PRIMARY path: the operator's launcher rsyncs data_gpu/ (shards +
# data_manifest.sha256) into place BEFORE invoking this script -- see the
# private launch tooling's "data rsync" step, which runs over the same
# rsync/ssh conventions as its code-overlay step. This step is therefore
# VERIFY-ONLY by default: check data_gpu/ exists and verifies against
# data_gpu/data_manifest.sha256; if clean, PASS immediately, no network I/O.
#
# OPTIONAL fallback: an S3-compatible bucket, configured ONLY via env (never
# hardcoded) -- either
#   DATA_REMOTE               a preconfigured rclone remote name, or
#   DATA_ENDPOINT + DATA_BUCKET + DATA_ACCESS_KEY_ID + DATA_SECRET_ACCESS_KEY
# This path only runs if data_gpu/ is missing/dirty AND one of the above is
# configured -- it exists for operators who provision a bucket instead of (or
# in addition to) rsyncing from their own machine (docker/README.md, "Data
# provisioning"). If neither the rsync'd data nor a bucket is available, this
# step FAILs with instructions rather than silently proceeding.
#
# data_gpu/ (not data/ -- HBA_DATA_DIR is exported to it below, per
# src/hba/config.py's env-override convention). The manifest is ALWAYS
# verified before proceeding, whether the data arrived via rsync or a bucket
# fetch, so a corrupt transfer or a tampered bucket object never trains
# silently.
mkdir -p data_gpu
verify_data_manifest() {
  [ -f data_gpu/data_manifest.sha256 ] || return 1
  ( cd data_gpu && sha256sum -c data_manifest.sha256 --status )
}

if [ -n "$(ls -A data_gpu 2>/dev/null)" ] && verify_data_manifest; then
  pass "data: data_gpu/ already present and verifies against data_manifest.sha256 (rsynced by the launcher) -- no fetch needed"
else
  if [ -n "${DATA_REMOTE:-}" ]; then
    echo "  data_gpu/ missing or dirty -- DATA_REMOTE is set, falling back to the optional bucket fetch"
    command -v rclone >/dev/null 2>&1 || fail "DATA_REMOTE=${DATA_REMOTE} set but rclone is not on PATH (image is stale, rebake)"
    echo "  fetching via rclone remote '${DATA_REMOTE}'"
    rclone sync "${DATA_REMOTE}:" data_gpu/ || fail "rclone sync from remote '${DATA_REMOTE}' failed"
    verify_data_manifest || fail "data_gpu/ does not verify against data_manifest.sha256 after the bucket fetch (corrupt transfer or a tampered bucket object) -- refusing to train on unverified data"
    pass "data: bucket fetch complete and verifies against data_manifest.sha256"
  elif [ -n "${DATA_ENDPOINT:-}" ] || [ -n "${DATA_BUCKET:-}" ]; then
    echo "  data_gpu/ missing or dirty -- DATA_ENDPOINT/DATA_BUCKET is set, falling back to the optional bucket fetch"
    : "${DATA_ENDPOINT:?DATA_ENDPOINT must be set alongside DATA_BUCKET for the S3-compatible fetch path}"
    : "${DATA_BUCKET:?DATA_BUCKET must be set alongside DATA_ENDPOINT}"
    : "${DATA_ACCESS_KEY_ID:?DATA_ACCESS_KEY_ID must be set for the S3-compatible fetch path}"
    : "${DATA_SECRET_ACCESS_KEY:?DATA_SECRET_ACCESS_KEY must be set for the S3-compatible fetch path}"
    echo "  fetching via S3-compatible endpoint '${DATA_ENDPOINT}' bucket '${DATA_BUCKET}'"
    DATA_ENDPOINT="$DATA_ENDPOINT" DATA_BUCKET="$DATA_BUCKET" \
    DATA_ACCESS_KEY_ID="$DATA_ACCESS_KEY_ID" DATA_SECRET_ACCESS_KEY="$DATA_SECRET_ACCESS_KEY" \
    python3 - <<'PY' || fail "S3-compatible data fetch failed"
import os
import boto3

endpoint = os.environ["DATA_ENDPOINT"]
bucket = os.environ["DATA_BUCKET"]
s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.environ["DATA_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["DATA_SECRET_ACCESS_KEY"],
)
paginator = s3.get_paginator("list_objects_v2")
n = 0
for page in paginator.paginate(Bucket=bucket):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        dest = os.path.join("data_gpu", key)
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        s3.download_file(bucket, key, dest)
        n += 1
print(f"[data] fetched {n} objects from s3://{bucket} via {endpoint}")
if n == 0:
    raise SystemExit(f"no objects found under s3://{bucket} -- check DATA_BUCKET/prefix")
PY
    verify_data_manifest || fail "data_gpu/ does not verify against data_manifest.sha256 after the bucket fetch (corrupt transfer or a tampered bucket object) -- refusing to train on unverified data"
    pass "data: bucket fetch complete and verifies against data_manifest.sha256"
  else
    fail "data_gpu/ is missing or does not verify against data_manifest.sha256, and no bucket fallback is configured (DATA_REMOTE, or DATA_ENDPOINT+DATA_BUCKET+DATA_ACCESS_KEY_ID+DATA_SECRET_ACCESS_KEY). The launcher must rsync data_gpu/ (shards + data_manifest.sha256) to this box BEFORE invoking provision.sh -- re-run the launcher's data-rsync step (docker/README.md, \"Data provisioning\"). If you intend to use a bucket instead, set the DATA_* env and re-run."
  fi
fi

export HBA_DATA_DIR="$(pwd)/data_gpu"

echo "======== STEP d: fast shakedown ========"
D_T0=$(date +%s)
if bash scripts/shakedown.sh --fast; then
  D_ELAPSED=$(( $(date +%s) - D_T0 ))
  TOTAL_ELAPSED=$(( $(date +%s) - PROV_T0 ))
  pass "fast shakedown (${D_ELAPSED}s)"
  echo
  echo "############################################################"
  echo "PROVISION GREEN -- rent-to-shakedown-green in ${TOTAL_ELAPSED}s (target <= 900s / 15 min)"
  echo "############################################################"
  exit 0
else
  RC=$?
  D_ELAPSED=$(( $(date +%s) - D_T0 ))
  fail "fast shakedown FAILED after ${D_ELAPSED}s (see scripts/shakedown.sh output / results/shakedown_report.json above); exit code $RC"
fi
