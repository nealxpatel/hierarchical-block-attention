#!/usr/bin/env bash
# Deterministic sha256 manifest of a data-shard directory -- mirrors
# scripts/make_manifest.sh's pattern (LC_ALL=C sort, sha256sum/shasum
# cross-platform fallback) but hashes shard files instead of code, and writes
# the result INTO the target directory as data_manifest.sha256 rather than at
# the repo root, because scripts/provision.sh's verify_data_manifest reads
# `data_gpu/data_manifest.sha256` on-box via `sha256sum -c --status`.
#
# Usage: bash scripts/make_data_manifest.sh [DATA_DIR]
#   DATA_DIR defaults to ./data_gpu (relative to the repo root, matching
#   provision.sh's fetch/verify target). Pass an absolute path to hash any
#   other shard directory -- e.g. the private launch tooling calls this
#   against its local shard source dir (HBA_DATA_SRC) before rsyncing it to
#   the instance's data_gpu/.
#
# Only top-level files in DATA_DIR are hashed (shards are flat, not nested);
# data_manifest.sha256 itself and dotfiles are excluded from the listing, or
# a re-run would hash its own previous output.
#
# Output:
#   1. DATA_DIR/data_manifest.sha256: sorted "<hash>  <path>" lines, paths
#      relative to DATA_DIR (so `cd data_gpu && sha256sum -c
#      data_manifest.sha256` -- provision.sh's exact verify command -- works
#      unchanged on-box regardless of where DATA_DIR lived when this ran).
#   2. stdout: the sha256 of THAT manifest file, for callers that want a
#      single value to log/compare (not consumed by provision.sh, which reads
#      the manifest file directly via sha256sum -c).
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${1:-data_gpu}"
[ -d "$DATA_DIR" ] || { echo "make_data_manifest: directory '$DATA_DIR' not found" >&2; exit 1; }

if command -v sha256sum >/dev/null 2>&1; then
  hash_files() { sha256sum "$@"; }
elif command -v shasum >/dev/null 2>&1; then
  hash_files() { shasum -a 256 "$@"; }
else
  echo "make_data_manifest: need sha256sum (Linux) or shasum (macOS) on PATH" >&2
  exit 1
fi

cd "$DATA_DIR"

# -not -name '.*': same rationale as make_manifest.sh -- a stray dotfile
# (.DS_Store etc.) must never enter the manifest. data_manifest.sha256 itself
# is excluded so a re-run hashes only shard content, not its own last output.
FILES=$(
  find . -maxdepth 1 -type f -not -name '.*' -not -name 'data_manifest.sha256' \
    | sed 's|^\./||' | LC_ALL=C sort
)
[ -n "$FILES" ] || { echo "make_data_manifest: no shard files found in '$DATA_DIR'" >&2; exit 1; }

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
: > "$TMP"
while IFS= read -r f; do
  [ -n "$f" ] || continue
  hash_files "$f" >> "$TMP"
done <<< "$FILES"

mv "$TMP" data_manifest.sha256
trap - EXIT

hash_files data_manifest.sha256 | awk '{print $1}'
