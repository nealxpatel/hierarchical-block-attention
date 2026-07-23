#!/usr/bin/env bash
# Build and push the pre-baked training image (docker/README.md). Run from the
# repo root: the build context must be the repo root (not docker/) because
# docker/Dockerfile COPYs src/, scripts/, docs/, docker/, pyproject.toml,
# README.md from there.
#
# Assumes `docker login ghcr.io` was already done by the operator -- this
# script does not log in or touch credentials. Registry: ghcr.io, not Docker
# Hub -- anonymous Docker Hub pulls from datacenter IPs hit rate limits, a
# provisioning-time lemon of our own making (docker/README.md).
#
# Rebake threshold (docker/README.md): any dependency change, an overlay of
# > ~20 files / > 1 MB, or any entrypoint change -- if that's why you're here,
# this is the right tool (not a code overlay).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${HBA_IMAGE:-ghcr.io/nealxpatel/hba-train}"
DATE_TAG="$(date -u +%Y%m%d)"
SHA_TAG="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
TAG="${HBA_IMAGE_TAG:-${DATE_TAG}-${SHA_TAG}}"

echo "[build_push] context: $(pwd)"
echo "[build_push] building ${IMAGE}:${TAG} (also tagging :latest)"
docker build -f docker/Dockerfile -t "${IMAGE}:${TAG}" -t "${IMAGE}:latest" .

echo "[build_push] pushing ${IMAGE}:${TAG}"
docker push "${IMAGE}:${TAG}"
echo "[build_push] pushing ${IMAGE}:latest"
docker push "${IMAGE}:latest"

echo "[build_push] done -- ${IMAGE}:${TAG} (also :latest)"
echo "[build_push] launchers should pin the dated tag (${TAG}), not :latest, so"
echo "  a mid-series rebake never silently changes a run already in flight."
