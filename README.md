# Hierarchical Block Attention (HBA)

**Near-constant per-query attention cost at any context length, without positional extrapolation.**

Validated end-to-end at 0.5B scale by converting a pretrained dense-attention model; a release-scale conversion of a current-generation Qwen donor is upcoming.

## The problem with dense attention

- **Quadratic cost.** Every token attends to every other token. Doubling the context quadruples the attention work — at 128K context, each query reads 128K keys.
- **Almost all of that work is wasted.** Measured on a pretrained 0.5B model: ~95% of attention mass falls on fewer than 9% of the keys at 8K context — and the concentration *increases* with length. A ~2K-key budget recovers ≥90% of the mass.
- **Position encodings don't extrapolate.** RoPE-style embeddings break beyond the offsets they trained on: dense retrieval collapses from 1.00 to 0.03 by 8× training length. Interpolation tricks (YaRN) delay, but don't remove, the decay.

## The idea

Read long context the way a person reads a reference book: keep the current page in full detail, remember a few anchor points, and for everything else consult an **index** — then open only the pages that matter.

- **Local window** — the last 1,024 tokens, with standard RoPE. Word order matters nearby (syntax, negation); a fixed-size window means relative positions never exceed training, so there is nothing to extrapolate.
- **Anchor (sink) tokens** — a handful of always-visible tokens that soak up the large default attention mass pretrained models park there.
- **Everything else: routed content blocks** — the rest of the context, grouped into 64-token blocks. Each block is advertised by **learned summary vectors**; each query scores the summaries — *by content, with no positional encoding* — and attends only inside its top-k blocks.

Long-range retrieval becomes a question of *what was said*, not *how far back it was said* — which is why there is no positional cliff for it to fall off.

### At scale: search the index as a tree

When contexts get long enough that even scanning summaries costs too much, summaries get summaries. A query descends the tier: score the super-summaries, expand the best few, score their children, select top-k — beam search over a shallow tree instead of a linear scan. Measured selection speedups: **5.2× at 32K, 7.9× at 64K, 10.6× at 128K**, with near-perfect agreement with the flat scan.

### One attention operation

The three regions are disjoint by construction and share a **single softmax over their union**, so the model trains end-to-end as one attention op. A fused implementation computes it as a log-sum-exp merge of two attention calls (5.3× training speedup over the naive path, agreement ~1e-6; the naive path is kept forever as a correctness oracle).

**Net effect:** per-query reads stay at ~2K keys whether the context is 4K or 128K.

## Key insights from validation

Each design decision is a claim we tested. The numbers below are from the 0.5B validation program ([docs/evals.md](docs/evals.md)).

1. **Sparsity is real and structured.** Attention mass decomposes into sinks (~25%), local window (~39%), and a routable long-range remainder (~20%) — mapping exactly onto HBA's three components.
2. **Summaries must be learned — and co-trained.** Mean-pooling is a weak router everywhere; learned multi-slot summarizers lift routing recall **1.8–5.1×** at every length. Distillation onto a *frozen* model recovers almost none of the gap (0.20 vs a 0.55 target): the query/key geometry itself must adapt. The healing phase is load-bearing.
3. **Content routing has no length cliff.** Position-free routing degrades smoothly where dense RoPE collapses; its gap to a YaRN'd dense baseline shrinks monotonically and crosses zero at ~32× training length (single-seed at the crossover point; the monotone trend is the robust finding).
4. **Quality survives conversion.** From scratch: +3–5% perplexity vs an identical dense baseline. Converted: the healed model **matches or beats its donor** on held-out perplexity and stays within noise on benchmarks (parity-or-better is the defensible claim — the converted model received continued training the donor did not).
5. **Perplexity is blind to capability loss.** Full-parameter healing on generic text silently destroyed retrieval (0.25 → 0.06 on a copy probe) while perplexity stayed flat. **Capability rehearsal** — a few percent of format-varied retrieval-exercising data, a 4× lower learning rate, and a live capability probe during training — revived it within 50 steps and held it above donor level.
6. **Softmax calibration is length-specific.** At 2× the healing context, the same weights retrieve perfectly in dense mode (answer rank 1) while the routed softmax buries the answer (rank ~56,000) — many slightly-warm routed logits drown a sharp signal. An analytic log-temperature helps ~2× but plateaus; **training at the served lengths** (the length-curriculum stage) is the fix, and repair tracks the training dose at each length.

## Converting a pretrained model

Keep every weight; replace only the attention computation; heal in stages:

| Stage | What trains | Purpose |
|---|---|---|
| 0 | summarizers only (distill-init) | cheap initialization against the frozen donor's own attention mass |
| 1 | attention + summarizers | re-align Q/K geometry to the sparse budget (everything else frozen) |
| 2 | full model, **with capability rehearsal** | settle the network into the sparse regime without trading capabilities away |
| 3 | full model, **mixed context lengths** | calibrate the softmax for the candidate counts it will serve |

Full procedure, hyperparameters, and monitoring requirements: [docs/training-recipe.md](docs/training-recipe.md).

## Status

- **Validated (0.5B):** quality parity with donor, retrieval at healing length above donor, efficiency curve confirmed to 128K context.
- **Upcoming (release scale):** target donor **Qwen3.5-9B** (primary) or **Qwen3.5-4B** (alternative) — both Apache-2.0; selection pending verification of attention layout and context configuration.
- **Model:** coming to Hugging Face — link will appear here at release.

| Benchmark | HBA (converted, release scale) | Donor baseline |
|---|---|---|
| MMLU | pending release run | pending |
| HellaSwag | pending release run | pending |
| ARC | pending release run | pending |
| GSM8K | pending release run | pending |
| HumanEval | pending release run | pending |
| Needle-in-haystack 4K / 32K / 128K | pending release run | pending |
| RULER | pending release run | pending |
| Held-out PPL (web / books / code) | pending release run | pending |

0.5B validation-scale numbers: [docs/evals.md](docs/evals.md).

## Reproducing

The conversion is specified as a reproducible recipe — stage table, hyperparameters, rehearsal data spec, length curriculum, monitoring, correctness gates — in [docs/training-recipe.md](docs/training-recipe.md), with the evaluation protocol in [docs/evals.md](docs/evals.md).

## Citation

```bibtex
@misc{patel2026hba,
  author       = {Patel, Neal},
  title        = {Hierarchical Block Attention},
  year         = {2026},
  howpublished = {\url{https://github.com/nealxpatel/hierarchical-block-attention}}
}
```

## Repo map

| Path | Contents |
|---|---|
| [src/hba/](src/hba/) | Reference implementation: attention (naive oracle + fused), summarizers, conversion, staged healing, capability probes, evals |
| [docs/design.md](docs/design.md) | Full architecture: notation, the three components and their union softmax, GQA-grouped selection, slot summarizers, hierarchy, softmax calibration, complexity analysis |
| [docs/training-recipe.md](docs/training-recipe.md) | The staged conversion recipe as a reproducible procedure |
| [docs/evals.md](docs/evals.md) | Evaluation protocol and validation-scale results |
| LICENSE | Apache-2.0 |
