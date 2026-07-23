"""Device/dtype abstraction, path configuration, and the HBA architecture config.

Everything that is a *policy* decision (which device, which compute dtype, where
checkpoints/data/logs live) is centralized here so the rest of the package only
ever imports `DEVICE`, `COMPUTE_DTYPE`, or the path constants below rather than
re-deriving them.
"""

import os
import time
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------- paths --------
# All three directories are override-able via environment variables so the package
# works the same whether it's run from a repo checkout, installed into site-packages,
# or pointed at a scratch volume on a training box. Defaults are relative to the
# current working directory (NOT the package install location) so `pip install -e .`
# and running from an arbitrary CWD both do the sane thing.
RESULTS = os.environ.get("HBA_RESULTS_DIR", os.path.join(os.getcwd(), "results"))
DATA = os.environ.get("HBA_DATA_DIR", os.path.join(os.getcwd(), "data"))
LOGS = os.environ.get("HBA_LOGS_DIR", os.path.join(os.getcwd(), "logs"))
for _d in (RESULTS, DATA, LOGS):
    os.makedirs(_d, exist_ok=True)

DONOR_NAME = os.environ.get("HBA_DONOR", "Qwen/Qwen2.5-0.5B-Instruct")

# Well-known checkpoint/artifact filenames under RESULTS, shared across convert.py
# (which produces them) and gates.py (which consumes them for the shakedown
# reference check) -- defined here, not in either module, so neither has to
# import the other at module scope.
INIT_PATH = os.path.join(RESULTS, "hba_init.pt")
DISTILL_PATH = os.path.join(RESULTS, "summarizers_distilled.pt")
REF_PATH = os.path.join(RESULTS, "reference_logits.pt")

# Plain-text corpus directory shared by convert.py (stage-0 distillation harvest)
# and data_prep.py (the needle-holdout / book-domain val+train text). Any
# directory of *.txt documents works; public-domain novels (e.g. Project
# Gutenberg) are a good source. One held-out book excluded from training text is
# required as the needle-eval haystack -- see data_prep.py.
CORPUS_DIR = os.environ.get("HBA_CORPUS_DIR", os.path.join(DATA, "corpus"))


# ---------------------------------------------------------------- device -------
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
# bf16 compute on CUDA (fp32 softmax is kept inside the attention math by policy);
# fp32 everywhere on MPS/CPU, which doubles as the exact-comparability anchor for
# smoke runs on non-CUDA hardware.
COMPUTE_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # expandable_segments avoids allocator-fragmentation OOMs at small micro-batch /
    # long-context combinations on memory-constrained (~24GB-class) cards: without it
    # we observed several GB "reserved but unallocated" survive to the loss-cast step
    # and tip an otherwise-fitting step into OOM. Must be set before the first CUDA
    # allocation, so this module should be imported before any entry script touches
    # a tensor. setdefault: an explicit user PYTORCH_ALLOC_CONF wins.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def save_ckpt_atomic(ckpt, path):
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def throttle_mps(it, every=8):
    """Cap in-flight MPS command-buffer temporaries in the chunked eval loops: on MPS
    the CPU can enqueue chunks faster than the GPU drains them, pinning Metal
    temporaries outside the torch allocator pool until OOM. No-op on CUDA/CPU."""
    if DEVICE == "mps" and it % every == every - 1:
        torch.mps.synchronize()


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def empty_cache():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


# ---------------------------------------------------------------- config -------
@dataclass
class HBAConfig:
    """Architecture + conversion config. Defaults are the validated 0.5B reference
    configuration (donor: Qwen2.5-0.5B-Instruct, 24 layers, 14 query / 2 KV heads,
    head_dim 64, rope_theta 1e6, tied embeddings, native ctx 32768); see
    docs/design.md for the notation and docs/training-recipe.md for the stage
    budgets these knobs feed into. Larger donors scale the geometry fields
    (n_layers/n_heads/n_kv/hidden/rope_theta/native_ctx/vocab_size); the HBA knobs
    below (block/window/sinks/k_blocks/slots/fanout/beam) are geometry-agnostic
    starting points, not fixed constants.
    """
    # ---- donor geometry ----
    n_layers: int = 24
    n_heads: int = 14          # query heads
    n_kv: int = 2              # KV heads (GQA); G = n_heads / n_kv
    head_dim: int = 64
    hidden: int = 896
    rope_theta: float = 1_000_000.0
    native_ctx: int = 32768
    vocab_size: int = 151936
    # ---- HBA stack ----
    block: int = 64            # routing block size (B)
    window: int = 1024         # RoPE local window (W)
    sinks: int = 4             # NoPE attention sinks (S)
    k_blocks: int = 16         # routed budget (top-k)
    slots: int = 4             # SlotSummarizer probes per (layer, KV head) (m)
    aux_w: float = 1.0         # aux-KL weight (the summarizer's ONLY gradient source)
    # ---- eval-time hierarchy (long-context selection speedup; not trained) ----
    fanout: int = 16
    beam: int = 4
    hier_from: int = 32768
    # ---- healing ----
    heal_ctx: int = 4096       # base heal context (window 1024 -> candidate blocks @ heal_ctx)
    mem_elem_cap: float = 1.0e8
    # ---- train-path attention backend ----
    # 'fused' = FlexAttention LSE-merge (no n^2 tensor materialized; the throughput
    # path on CUDA/CPU). 'naive' = the materialized-scores path, kept FOREVER as the
    # correctness oracle every optimized path is gated against, and as the MPS path
    # (no FlexAttention backend on MPS).
    attn_backend: str = "fused"

    @property
    def G(self):
        return self.n_heads // self.n_kv


def resolve_backend(cfg):
    """'fused' needs FlexAttention (CUDA fused kernels, or the CPU eager fallback);
    MPS has no flex backend, so on MPS a 'fused' request silently takes the naive
    path (which IS the MPS comparability anchor -- see COMPUTE_DTYPE above)."""
    b = getattr(cfg, "attn_backend", "naive")
    if b == "fused" and DEVICE == "mps":
        return "naive"
    return b


def smoke_config():
    """Tiny HBA config for fast CPU/MPS plumbing smoke tests (uses the REAL donor
    geometry -- only the HBA routing knobs shrink, so routing is still exercised at
    short context)."""
    return HBAConfig(block=16, window=64, sinks=2, k_blocks=4, slots=2,
                     fanout=4, beam=2, hier_from=512, heal_ctx=512, mem_elem_cap=2e7)
