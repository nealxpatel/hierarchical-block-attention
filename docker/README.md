# Pre-baked training image

Goal: a rented GPU host goes from "rented" to **shakedown-green, ready to
train** in **≤ 15 minutes**, with **zero network pip installs** during
provisioning. This is achieved by baking everything slow/flaky (base image,
pinned deps, the donor model, the repo code) into a Docker image ahead of
time, and keeping the provisioning-time entrypoint (`scripts/provision.sh`) to
four fast, purely-verifying steps.

## Why baked

Installing pinned deps and downloading the donor model over a host's network
connection is the single biggest source of both wall-clock (an hour+ pip
install on a slow host is a real, observed failure mode) and flakiness
(reboots/network blips mid-install silently corrupt the environment). Baking
the image turns that hour into a one-time, off-the-clock build step, and turns
provisioning into a set of assertions that either pass fast or fail loud.

## Image contents (`docker/Dockerfile`)

| Layer | Contents | Why baked |
|---|---|---|
| Base | `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime` (**runtime**, not `devel`) | torch 2.10 / cu128 pin; smaller than the devel image |
| System deps | `gcc` (triton/`flex_attention` JIT-compiles a kernel at runtime and needs a C compiler in the container, not just at build time), `rclone` (one data-fetch backend) | avoids a runtime apt-get |
| Python deps | pinned `docker/requirements.txt` (`transformers==4.57.6`, `tokenizers==0.22.2`, `datasets==4.6.1`, `numpy`, `boto3` — the other data-fetch backend); `torchvision`/`torchaudio` uninstalled (unneeded, saves space) | eliminates the pip-install step entirely at provisioning time |
| Donor | HF cache (`HF_HOME=/workspace/hf`) with `Qwen/Qwen2.5-0.5B-Instruct` + tokenizer | removes a flaky, first-run HF download from every provisioning run |
| Code | `src/`, `scripts/`, `docs/`, `docker/`, `pyproject.toml`, `README.md` copied to `/workspace/hba`; `manifest.sha256` generated in-image via `scripts/make_manifest.sh` at build time; package installed with `pip install --no-deps -e .` | the code-overlay/manifest-verify flow below needs a baseline to diff and verify against |
| Entrypoint | `scripts/provision.sh` | one command, rent to verdict |

**Known gap, by design:** `results/` (checkpoints, `reference_logits.pt`,
prior shakedown reports) is excluded from the image — see `.dockerignore` —
because it's run-produced state, not code, and baking it would make the image
grow with every artifact ever produced. `results/reference_logits.pt`
specifically (the fp32 reference `scripts/shakedown.sh`'s `check_reference`
gate compares against — produced once via `python -m hba.convert
--export-ref` on a reference machine) is therefore **not present on a freshly
provisioned box** unless something places it there. The private launch
tooling's code-overlay step is the natural vehicle for this (it already
rsyncs into `/workspace/hba`); this repo does not prescribe how, since that's
provider/operator-specific orchestration.

## Build and push (`docker/build_push.sh`)

```bash
# from the repo root -- build context MUST be the repo root, not docker/,
# because the Dockerfile COPYs src/, scripts/, docs/, docker/ from there
docker build -f docker/Dockerfile .

# or use the wrapper (tags with today's date + short git sha, plus :latest,
# and pushes both) -- does NOT log in for you
docker login ghcr.io
docker/build_push.sh
```

**Registry: `ghcr.io`, not Docker Hub.** Anonymous Docker Hub pulls from
datacenter IPs (which is what every rented GPU host is) hit rate limits — a
provisioning-time lemon of our own making. `ghcr.io/nealxpatel/hba-train` is
the default image name (override with `HBA_IMAGE`); launchers should pin the
dated tag the build produced, not `:latest`, so a later rebake never silently
changes a run already in flight.

## The four provisioning steps (`scripts/provision.sh`)

Every step prints a `PASS:`/`FAIL:` line; **any FAIL exits nonzero
immediately** (no partial-credit runs). A final `PROVISION GREEN`/`PROVISION
RED` banner reports total elapsed time against the ≤ 15 min target.

**a. Expected-manifest assert.** The launcher computes a manifest hash on its
*own* machine, at launch time, from the code it intends to ship (see
`scripts/make_manifest.sh`), and passes it in as `EXPECTED_MANIFEST_SHA`.
`provision.sh` recomputes the on-box manifest hash the same way and asserts
equality. **Why anchor to the launcher's hash instead of just checking the
on-box manifest is internally consistent:** an on-box-only check can't
distinguish "code is correct" from "code is stale but self-consistent" — if a
code-overlay push silently failed or was skipped, the stale code on disk
verifies perfectly against its own stale `manifest.sha256`. Anchoring to a
hash computed *elsewhere*, from the code that was actually supposed to ship,
turns a skipped push into a loud FAIL instead of a silent wrong-code run.

**b. Env pin assert.** Checks the on-box `torch`/`transformers`/`tokenizers`
versions against the pins baked into the image (`HBA_PINNED_TORCH` env var +
`docker/requirements.txt`, parsed directly — one place to bump a pin). **A
network pip install during provisioning is a FAIL, not a fallback:** if the
pins don't match, the image is stale and the fix is to rebake (below), not to
patch a running box.

**c. Data fetch.** Configured only via environment (never hardcoded, never
committed):

- `DATA_REMOTE` — a preconfigured `rclone` remote name, **or**
- `DATA_ENDPOINT` + `DATA_BUCKET` + `DATA_ACCESS_KEY_ID` +
  `DATA_SECRET_ACCESS_KEY` — any S3-compatible bucket (Cloudflare R2, Backblaze
  B2, MinIO, actual S3), fetched via `boto3` against `endpoint_url`.

Skips the download if `data_gpu/` is already present **and** verifies against
`data_gpu/data_manifest.sha256`; the data manifest is **always** re-verified
before proceeding (whether or not a fetch just happened), so a corrupt
transfer or a tampered bucket object never trains silently. On success,
`HBA_DATA_DIR` is exported to `data_gpu/` (the env-override
`src/hba/config.py` and `scripts/shakedown.sh` already support) so the rest of
the pipeline picks the fetched data up automatically.

**d. Fast shakedown.** Runs `scripts/shakedown.sh --fast` (below) and reports
the final banner.

## Env vars `scripts/provision.sh` consumes

| Var | Required | Purpose |
|---|---|---|
| `EXPECTED_MANIFEST_SHA` | yes | launcher-computed code manifest hash (step a) |
| `HBA_PINNED_TORCH` | baked by the image | expected torch version (step b) |
| `DATA_REMOTE` | one of this or the four below | preconfigured rclone remote name (step c) |
| `DATA_ENDPOINT`, `DATA_BUCKET`, `DATA_ACCESS_KEY_ID`, `DATA_SECRET_ACCESS_KEY` | one of the four or `DATA_REMOTE` | S3-compatible bucket credentials (step c) |
| `PLANNED_TPS` | no (baked default `13600`) | passed through to `scripts/shakedown.sh` / `hba.gates` for the throughput abort check |
| `PIP_BREAK_SYSTEM_PACKAGES`, `PYTORCH_ALLOC_CONF`, `HF_HOME`, `HF_HUB_OFFLINE` | baked | see `docker/Dockerfile`; not operator-set |

## Fast shakedown profile (`scripts/shakedown.sh --fast`)

Trims the ~10-minute-plus full shakedown to fit inside the ≤ 15 min
provisioning budget, on the reasoning that a pre-baked, already-verified image
doesn't need to re-prove things `provision.sh`'s earlier steps already proved:

| Step | Full (default) | `--fast` |
|---|---|---|
| pip install | pinned deps + `-e .` | **skipped** (pins already verified by provision.sh step b; a pip install here would be a network dependency the whole point of the image is to avoid) |
| data | generates a small slice if missing | **no download** — presence-only check (provision.sh step c already fetched + verified it) |
| fp32 correctness gates + G1 induction | run | run, unchanged (~4 min) — this is exactly the correctness surface that must never be skipped before training runs on a new box |
| training measurement | 150 steps | **50 steps**, with the tok/s window **excluding the first ~10 steps** (compile/autotune warmup) — see below |
| eval | 1 PPL cell + 1 needle cell | **1 PPL cell only** (~1 min budget) |

`--fast` is wired through `scripts/shakedown.sh` → `python -m hba.gates
--fast` → `hba.gates.run_shakedown(fast=True)` → `check_training(...,
fast=True)` / `check_eval(..., fast=True)`.

**Warmup-excluded tok/s measurement.** At 150 steps, one-time FlexAttention
kernel autotune in the first few steps is diluted into noise by the other
140+. At 50 steps it wouldn't be — a naive average over all 50 steps would
understate a perfectly healthy box's steady-state throughput and could
spuriously trip the `PLANNED_TPS` abort. `check_training` passes
`warmup_steps=min(10, steps-1)` down into `hba.heal.train()`, which resets its
tok/s baseline (elapsed time and tokens-seen zero point) once that many steps
have completed, so every throughput number logged afterward reflects steady
state only. This lives in `hba/heal.py`'s training loop — the one place the
per-step wall-clock is actually measured — rather than being reconstructed
after the fact from log lines.

The full 150-step profile remains the default (no flag) and is what's
required before committing to any multi-hour unattended run
(`scripts/heal.sh` / `scripts/evals.sh` gate on a shakedown PASS either way).

## Code-update flow after image bake (thin overlay, launcher-anchored)

The image carries a code **baseline** + `manifest.sha256`. Small post-bake
changes don't need a rebake:

1. `rsync` only the changed files into `/workspace/hba` on the box (preferred
   over `git pull` of a private repo — no token needs to live on a disposable
   host).
2. Regenerate the manifest on-box: `bash scripts/make_manifest.sh` (writes
   `manifest.sha256`, prints the manifest hash).
3. The launcher independently computes the **same** hash locally, from the
   code it intended to ship, and passes it as `EXPECTED_MANIFEST_SHA` to
   `scripts/provision.sh`, which asserts the two match (step a above) — this
   is what turns "the overlay silently didn't land" into a loud FAIL instead
   of a run on stale code that happens to verify against itself.

**Rebake threshold (pre-registered — don't improvise around it):**

- any dependency change (`docker/requirements.txt`, the base image tag), **or**
- an overlay of more than ~20 files or ~1 MB, **or**
- any change to `scripts/provision.sh` (the entrypoint) or `docker/Dockerfile`

→ rebake (`docker/build_push.sh`, ~30 min build + push, done from the
operator's machine, **off the GPU clock** — never rebake on a paid box).
Below that threshold, overlay + on-box manifest regeneration + the launcher's
`EXPECTED_MANIFEST_SHA` assert is the flow.

## The ≤ 15 min target

Steps a–c (manifest assert, pin assert, data fetch-or-verify) are each on the
order of a minute; step d (`--fast` shakedown) is the remaining ~9 minutes
(4 min correctness gates + 3 min training measurement + 1 min PPL + change).
`scripts/provision.sh` prints the total elapsed time in its final banner so
this is measured, not assumed, on every box.
