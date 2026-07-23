#!/usr/bin/env bash
# Deterministic sha256 manifest of "the code" -- the file set whose content
# determines training behavior. scripts/provision.sh's expected-manifest
# assert (docker/README.md) is only meaningful if this script produces
# byte-identical output for byte-identical trees on any machine: the
# operator's Mac (launcher, computes EXPECTED_MANIFEST_SHA) and the rented
# Linux GPU box (provisioning, recomputes it on-box) alike.
#
# Included: src/**/*.py (the reference implementation), scripts/*.sh (the
# entrypoints themselves -- an entrypoint change IS a code change), docker/*
# (Dockerfile/requirements.txt -- changing how the image is built is a code
# change too). Excluded: everything else (docs/, results/, data/,
# checkpoints, __pycache__, .git) -- those either don't affect training
# behavior or are covered by a separate check (training/eval data has its
# own data_manifest.sha256; see scripts/provision.sh).
#
# Output:
#   1. manifest.sha256 at the repo root: sorted "<hash>  <path>" lines.
#   2. stdout: the sha256 of THAT manifest file (the "manifest hash") -- the
#      single value scripts/provision.sh compares against
#      EXPECTED_MANIFEST_SHA.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v sha256sum >/dev/null 2>&1; then
  hash_files() { sha256sum "$@"; }
elif command -v shasum >/dev/null 2>&1; then
  hash_files() { shasum -a 256 "$@"; }
else
  echo "make_manifest: need sha256sum (Linux) or shasum (macOS) on PATH" >&2
  exit 1
fi

# LC_ALL=C: sort collation otherwise depends on host locale, which would
# silently change manifest.sha256's byte content across machines even when
# the file set is identical -- exactly the failure mode this script exists
# to prevent.
# -not -name '.*' on every clause: a dotfile under any of these directories
# (e.g. a future, gitignored docker/.DS_Store) must never enter the manifest
# -- gitignored means it's never part of the code overlay either, so if it
# silently got hashed here, every subsequent launch's on-box recompute would
# come up RED the moment such a file appeared/changed locally.
FILES=$(
  { find src -type f -name '*.py' -not -name '.*' 2>/dev/null || true
    find scripts -maxdepth 1 -type f -name '*.sh' -not -name '.*' 2>/dev/null || true
    find docker -maxdepth 1 -type f -not -name '.*' 2>/dev/null || true
  } | LC_ALL=C sort
)

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
: > "$TMP"
while IFS= read -r f; do
  [ -n "$f" ] || continue
  hash_files "$f" >> "$TMP"
done <<< "$FILES"

mv "$TMP" manifest.sha256
trap - EXIT

hash_files manifest.sha256 | awk '{print $1}'
