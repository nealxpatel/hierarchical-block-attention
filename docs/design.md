# HBA architecture

This document specifies Hierarchical Block Attention precisely enough to reimplement it. Parameter values given are the validated 0.5B reference configuration; the architecture is geometry-agnostic and larger models scale the knobs (noted where relevant).

## Notation

| Symbol | Meaning | Reference value |
|---|---|---|
| `n` | context length | 4K–128K |
| `S` | number of anchor/sink tokens (prefix) | 4 |
| `W` | local RoPE window size | 1024 |
| `B` | key/value block size | 64 |
| `k` | routed blocks per query group | 16 |
| `m` | summary slots per block per (layer, KV head) | 4 |
| `G` | query heads per KV head (GQA group size) | model-dependent (7 in the 0.5B validation donor) |
| `N` | candidate block count = `⌈(n − W − S)/B⌉` | grows with `n` |
| `F`, `b` | hierarchy fanout, beam width | 16, 4 |

## The three components

Every attention layer computes one softmax over the **disjoint union** of three key/value regions:

1. **Anchor/sink tokens** — the first `S` tokens, attended by every query, **position-free** (no RoPE, "NoPE"). Pretrained models park a large fraction of attention mass (~25% measured) on early tokens; serving them unconditionally preserves that mechanism at negligible cost.
2. **Local RoPE window** — the last `W` tokens before the query, with standard rotary embeddings. This is the only place position enters the computation. Because `W` is fixed, relative positions inside the window never exceed what training saw — **the model never extrapolates positionally**, at any context length.
3. **Routed content blocks** — all tokens strictly before the window are grouped into blocks of `B`. Each block is advertised by learned summary vectors; each query scores the summaries **without RoPE** and attends exactly to the tokens of its top-`k` blocks. Long-range attention is thus purely content-addressed.

The regions are disjoint by construction (window tokens are excluded from blocks), so the union softmax is exact:

```
attn(q) = softmax_over( {q·k_i : i ∈ sinks} ∪ {q·k_i^RoPE : i ∈ window} ∪ {q·k_i : i ∈ selected blocks} ) · V
```

In a fused implementation this is computed as a **log-sum-exp merge** of two attention calls — one NoPE region (sinks + routed blocks) and one RoPE sliding-window region — which avoids materializing any `n×n` tensor and yielded a 5.3× end-to-end training speedup over the naive path in our implementation (agreement with the naive reference ~1e-6).

### Design note: why the window keeps RoPE

HBA removes positional encoding from **long range** — where extrapolation beyond training
offsets is what breaks length generalization — not from the model. Inside the window it is
kept, deliberately:

- **Local order carries meaning.** Syntax, negation, and adjacency are genuinely
  position-dependent; measured attention in pretrained models is position-structured
  locally and content-addressed at distance. The decomposition mirrors that split.
- **Confined RoPE cannot extrapolate.** With relative offsets capped at `W`, the rotary
  computation is permanently in-distribution at any sequence length. The failure mode is
  amputated; the mechanism is not.
- **Conversion economics.** A pretrained donor's local attention behavior lives in its
  RoPE'd Q/K geometry. Keeping window RoPE means that behavior transfers intact, which is
  a large part of why the attention-only healing stage is cheap. Replacing it would mean
  relearning local attention — the expensive part — from scratch.

Concurrent from-scratch work (described in the [Inkling essay](https://idlemachines.co.uk/essays/inkling)) makes the other choice: a
learnt, content-dependent additive distance bias over the same-sized local window, with no
rotary embeddings anywhere. That is a defensible design when pretraining from scratch and
plausibly more expressive, at a higher kernel cost (the bias must be materialized inside
the attention tile loop). For converting existing checkpoints, window-RoPE is the
pragmatic optimum. The accurate summary of both designs is the same: **no positional
encoding beyond the local window.**

## Slot summarizers

Mean-pooling blocks into single vectors is a demonstrably weak router (see README principle 3). Instead, each (layer, KV head) owns a **SlotSummarizer**: `m` learned probe vectors that cross-attend into the block's keys, producing `m` slot vectors per block. A block's routing score for query `q` is:

```
score(q, block) = max over m slots of ( q_nope · slot )
```

Max-over-slots lets a block advertise several distinct things it contains (a name, a number, a code identifier) rather than their average, which is what lifts routing recall 1.8–5.1× over mean-pooling. The summarizer's projection is **identity-initialized** — a small-random init shrinks the slot vectors toward zero, flattening block scores and killing the summarizer's training signal.

## GQA-grouped selection

Selection is per **KV head**, shared by its `G` query heads, using the **grouped query** (sum of the group's query heads) to score blocks. One block set per group means the KV cache's sharing structure survives conversion: routed keys/values are gathered once per KV head and broadcast to the group for exact attention. The summarizer's training target is likewise the group-averaged content mass (see [training-recipe.md](training-recipe.md)).

## Hierarchy

Flat selection compares each query group against all `N` block summaries — `O(N)` per query, fine at 32K, wasteful beyond. HBA adds a second level: block summaries are pooled into **super-summaries** of `F` blocks each. Selection descends with a beam: score all super-summaries, keep the top `b`, score only their `F·b` children, take the final top-`k`.

- **Fidelity:** two-level beam descent recovers the flat top-`k` selection nearly exactly at these depths (measured agreement on selected sets is near-perfect; all reported quality numbers are unchanged between flat and hierarchical selection at the lengths where both run).
- **Measured speedup** (summary comparisons per query, vs flat): **5.2× at 32K, 7.9× at 64K, 10.6× at 128K.**
- Depth generalizes: additional levels give `O(log N)` selection; two levels suffice through 128K.

## Softmax length-calibration (read this before scaling context)

The union softmax has one subtlety that does not exist in dense attention: **the routed region's aggregate mass depends on how many candidate blocks compete**. Train at context `n₀` and the model calibrates its NoPE logit scale against `N₀` candidates. Serve at `2n₀` and roughly `2N₀` candidates contribute — even if each irrelevant block is only *slightly* warm, their summed exponentials dilute the softmax and can bury a still-sharp correct-answer logit.

This is measurable and severe. Our discriminator at 2× the trained context, same weights throughout (32-trial probe; read accuracies coarsely):

| Configuration | Retrieval acc | Median rank of answer token |
|---|---|---|
| Dense mode (full attention, same weights) | **0.531** | **1** |
| HBA path, routing off (window + sinks only) | 0.156 | ~2,300 |
| HBA path, routing on | 0.031 | ~56,000 |

The weights are perfect (dense-mode rank 1); block *selection* is also fine (oracle selection matches learned selection). The failure is purely the diluted union softmax. **More training dose does not fix this by itself** — a length-curriculum stage that dosed a longer regime moved that regime's own retrieval but left it well below the dense-mode ceiling; the miscalibration is structural, not (only) a data-coverage gap. That motivates treating it architecturally.

### The fix: QKNorm + a bounded, shared union scale + a clamped extrapolation temperature

The recipe below follows the [Inkling essay](https://idlemachines.co.uk/essays/inkling) and the Scalable-Softmax paper ([arXiv:2501.19399](https://arxiv.org/abs/2501.19399)), adapted to HBA's specific union-softmax structure. Three interlocking parts, implemented in `hba/attention.py` (`QKNorm`, `content_scale`, `log_len_tau`, `_content_qk`) and config-driven via `HBAConfig.qknorm` / `.content_scale_mode` / `.temp_c` / `.n_cal`:

1. **QKNorm.** RMS-normalize queries and keys per head, then apply a learned **per-head SCALAR gain** (not a per-dimension elementwise affine — see `QKNorm`'s docstring for why the scalar form is load-bearing). After normalization, `||q|| = ||k|| = sqrt(dh)` exactly (times the gain), so `q·k = gain_q·gain_k·dh·cos(θ)` — a quantity bounded by the gains, independent of head_dim or however large the raw Q/K statistics happen to be.
2. **1/d content scale (not 1/√d).** Dividing the QKNorm'd `q·k` by `dh` (not `sqrt(dh)`) leaves the content logit at `gain_q·gain_k·cos(θ) ∈ [-gain_q·gain_k, +gain_q·gain_k]` — bounded regardless of context length or candidate count, which is the actual property that fixes dilution: a bounded per-candidate logit means the union softmax's calibration doesn't have to relearn its scale every time the candidate crowd changes size.
3. **A scale shared across the whole union, by construction, not by training.** The union softmax mixes three logit sources per query — the RoPE local window, the NoPE sinks, and the NoPE routed blocks — and *all three* must share one scale, or one branch dominates regardless of content relevance. HBA's design here is deliberately simple: QKNorm normalizes Q/K **once per layer**, before either branch consumes them. The NoPE branches (sinks, routed) use the normalized tensors directly; the RoPE window branch applies rotary embeddings to the *same* normalized tensors (rotation is norm-preserving, so it does not change the `[-gain_q·gain_k, +gain_q·gain_k]` bound). Both branches therefore enter the union softmax on a **provably** identical scale — the same normalized vectors, only rotated differently for the window — rather than two independently-scaled branches that training merely learns to match. The routing/summarizer scoring branch (which block a query selects, not the attention weight it gets once selected) shares the same QKNorm'd Q/K for the identical reason: an unbounded routing score would drift with candidate count the same way the old union softmax did, even though hard top-k selection is itself scale-invariant, the summarizer's own softmax-normalized aux-KL objective is not.
4. **A clamped log-length temperature for extrapolation beyond the trained/native length.** `τ = 1 + c·log(max(n/n_cal, 1))`, `c ≈ 0.1`, multiplying the content scale (equivalently, scaling the query). This is **identity** (`τ=1` exactly, since `log(1)=0`) for any served length `n ≤ n_cal` — QKNorm + the bounded 1/d scale is what calibrates the softmax *within* the trained/native length; this temperature only grows past `n_cal`, sharpening logits to counter dilution when genuinely extrapolating beyond what the model was healed at. `n_cal` defaults to `native_ctx` (32,768 for the 0.5B validation donor and the primary 32K-native release target), so within-32K serving is architecture-calibrated (τ=1) and this knob only activates for longer-than-native extension.

Ablation flags (`HBAConfig`): `qknorm` (default on; off reproduces the pre-QKNorm attention byte-for-byte — see `training-recipe.md`, "Correctness gates"), `content_scale_mode` (`'inv_d'` vs the legacy `'inv_sqrt_d'`, to isolate which half of the fix matters), and `temp_c` / `n_cal` for the extrapolation temperature (gated under `qknorm`; inert when `qknorm=False`).

A related trained-length-clamped temperature appears in the Inkling essay's own recipe, which we take as convergent evidence for both the phenomenon and the boundary choice — their position-aware local window is also 1,024 tokens. Unlike Inkling's from-scratch design (a learnt additive distance bias inside the window, no rotary embeddings anywhere), HBA keeps window-RoPE for conversion economics (see "Design note: why the window keeps RoPE" above) — QKNorm + the shared 1/d scale is the piece of the recipe that transfers to a RoPE-window design unchanged.

**Length curriculum remains necessary, not optional.** The architectural fix bounds the logits so the union softmax's calibration problem is well-posed at every length; it does not, by itself, teach the model to route and weight correctly at lengths it never trained at. Stage 3's length curriculum (`training-recipe.md`) is still how the model learns to *use* that bounded scale well beyond a single training length — the architecture and the curriculum are complementary, not substitutes.

## Complexity

Per-query attended tokens are constant by construction: `S + W + k·B` (= 4 + 1024 + 1024 = 2,052, ~**2.1K keys** in the reference config), independent of `n`. Selection cost is `O(N)` flat or `O(b·F + N/F)` hierarchical. Memory is dominated by the KV cache (linear in `n`, same as dense — HBA saves *bandwidth and compute* per query, and the block structure is amenable to offloading cold blocks, since only summaries must stay resident for selection).

Measured end-to-end evaluation footprint (0.5B model, bf16, single GPU):

| Context | Per-query keys read | Peak memory | Hierarchical selection speedup |
|---|---|---|---|
| 32K | ~2.1K | 4.3 GiB | 5.2× |
| 64K | ~2.1K | 6.8 GiB | 7.9× |
| 128K | ~2.1K | 11.8 GiB (KV-cache dominated) | 10.6× |

## Attention decomposition (diagram)

```mermaid
flowchart TB
    Q[Query token q] --> SEL

    subgraph CTX[Context]
        direction LR
        SINK["Anchor/sink tokens (S=4), NoPE — always visible"]
        BLOCKS["Content blocks (B=64 tokens each) — everything before the window"]
        WIN["Local window (W=1024), RoPE — always visible"]
    end

    BLOCKS --> SUMM["SlotSummarizer per (layer, KV head): m=4 learned slots per block"]
    SUMM --> HIER["Hierarchy: super-summaries (fanout F=16) → beam b=4 descent"]
    HIER --> SEL["Top-k selection (k=16), grouped query per KV head, position-free scoring"]
    SEL --> GATHER["Gather k×B routed tokens (NoPE)"]

    SINK --> UNION
    WIN --> UNION
    GATHER --> UNION["One softmax over the disjoint union (LSE merge of NoPE and RoPE regions)"]
    UNION --> OUT["Attention output — ~2.1K keys read per query at any context length"]
```
