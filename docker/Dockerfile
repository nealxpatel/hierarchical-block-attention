# Pre-baked training image for Hierarchical Block Attention (see
# docker/README.md for the full rationale and the code-overlay update flow).
# Goal: a rented GPU host goes from "rented" to "shakedown-green" in <=15 min,
# with ZERO network pip installs during provisioning -- everything below is
# baked at BUILD time (this file); scripts/provision.sh only verifies pins at
# provisioning time, it never installs.
#
# Build from the repo root (build context = repo root, not docker/):
#   docker build -f docker/Dockerfile .
# (see docker/build_push.sh for the tag/push flow).
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

# ---- pin bookkeeping ----
# torch is NOT reinstalled below -- this base image tag IS the torch pin.
# Recorded as an env var so scripts/provision.sh's env-pin assert can check it
# without re-parsing a base-image tag string. transformers/tokenizers pins
# live in docker/requirements.txt (provision.sh parses that file directly, so
# there is exactly one place to bump a pin).
ENV HBA_PINNED_TORCH=2.10.0

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/workspace/hf \
    HF_HUB_OFFLINE=1 \
    PLANNED_TPS=13600 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace/hba

# gcc: triton/flex_attention JIT-compiles a kernel at RUNTIME (not just at
# image build time) and needs a C compiler present in the container.
# rclone: one of the two data-fetch backends scripts/provision.sh supports
# (the other, boto3, is a pip dep below) -- baked so no network install is
# needed for it at provisioning time either.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc rclone \
    && rm -rf /var/lib/apt/lists/*

# torchvision/torchaudio ship with the base image but HBA never imports
# either -- drop them to save space (they drag in their own CUDA-linked
# wheels).
RUN pip uninstall -y torchvision torchaudio || true

COPY docker/requirements.txt /workspace/hba/docker/requirements.txt
RUN pip install --no-cache-dir -r /workspace/hba/docker/requirements.txt

# ---- bake the donor model + tokenizer into the HF cache ----
# HF_HUB_OFFLINE=1 (set above) is what makes provisioning-time runs skip HF
# network pings entirely -- it must NOT be in effect for this download step,
# hence the explicit override on this RUN line only.
RUN HF_HUB_OFFLINE=0 python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct')"

# ---- bake the repo code ----
# src/, scripts/, docs/, docker/: docker/ is included (not just referenced)
# because scripts/make_manifest.sh hashes docker/* as part of "the code"
# (changing how the image is built is a code change too); docs/ ships for
# reference but is not manifest-hashed (make_manifest.sh only covers
# src/**/*.py, scripts/*.sh, docker/* -- see that script's header).
COPY src/ /workspace/hba/src/
COPY scripts/ /workspace/hba/scripts/
COPY docs/ /workspace/hba/docs/
COPY docker/ /workspace/hba/docker/
COPY pyproject.toml /workspace/hba/pyproject.toml
COPY README.md /workspace/hba/README.md

RUN chmod +x /workspace/hba/scripts/*.sh \
    && pip install --no-cache-dir --no-deps -e /workspace/hba \
    && bash scripts/make_manifest.sh > /workspace/hba/BUILD_MANIFEST_SHA.txt \
    && cat /workspace/hba/BUILD_MANIFEST_SHA.txt

# scripts/provision.sh is the rent-to-ready entrypoint: manifest assert -> env
# pin assert -> data fetch -> fast shakedown -> GREEN/RED banner.
ENTRYPOINT ["scripts/provision.sh"]
