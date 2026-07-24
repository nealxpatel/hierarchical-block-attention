# Hierarchical Block Attention (HBA)

**Match a dense donor's attention quality at its native context length, at a near-constant per-query cost — with no positional cliff if you push past it.**

Fundamentals and efficiency validated end-to-end at 0.5B by converting a pretrained dense-attention model. A release-scale conversion of a current-generation Qwen donor, targeting the primary objective below, is in progress.

## Two objectives, in order

1. **Primary — efficient attention at the donor's native context length (e.g. 32K).** At native length a dense donor already gets full quality; it just pays for it quadratically, reading all 32K keys per query. HBA's job is to match that quality while reading a near-constant **~2.1K keys per query** instead. That alone — same reach, a fraction of the cost, no extrapolation involved — is a real inference-efficiency result.
2. **Secondary — extending beyond native length.** Content-addressed routing has no positional cliff, so going past what the donor ever saw is the natural follow-on. It's also the harder, second-order goal, and it only matters once (1) is solid.

Everything below — the architecture, the validation results, the open problem — is organized around that ordering.

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

### One softmax, calibrated across lengths

The three regions are disjoint by construction and share a **single softmax over their union**, so the model trains end-to-end as one attention op (a fused log-sum-exp merge of two attention calls gives a 5.3× training speedup over the naive path; the naive path is kept forever as a correctness oracle).

The union softmax has one subtlety dense attention doesn't: as the candidate-block count grows with context length, the routed region's aggregate mass shifts even when no individual candidate's score does — enough slightly-warm distractors can dilute a sharp correct answer's logit (see below). The design bounds this architecturally rather than leaving it to be learned: **QKNorm + 1/d scaling** on the content branch, so routed logits are cosine similarities in `[-1, 1]`; **one shared scale across every branch of the union**; and a **log-length temperature that is identity within the trained length range and only activates beyond it**. This follows the public [Inkling essay](https://idlemachines.co.uk/essays/inkling)'s length-calibration recipe and [Scalable-Softmax](https://arxiv.org/abs/2501.19399)'s approach to bounding softmax scale growth with context length.

**Net effect:** per-query reads stay at ~2K keys whether the context is 4K or 128K.

## Key insights from validation

Each design decision is a claim we tested. The numbers below are from the 0.5B validation program ([docs/evals.md](docs/evals.md)).

1. **Sparsity is real and structured.** Attention mass decomposes into sinks (~25%), local window (~39%), and a routable long-range remainder (~20%) — mapping exactly onto HBA's three components.
2. **Summaries must be learned — and co-trained.** Mean-pooling is a weak router everywhere; learned multi-slot summarizers lift routing recall **1.8–5.1×** at every length. Distillation onto a *frozen* model recovers almost none of the gap (0.20 vs a 0.55 target): the query/key geometry itself must adapt.
3. **Block selection is correct; content routing has no length cliff.** Position-free routing degrades smoothly where dense RoPE collapses, and oracle vs. learned selection agree closely. Its gap to a YaRN'd dense baseline shrinks monotonically and crosses zero at ~32× training length (single-seed at the crossover; the monotone trend is the robust finding).
4. **Quality holds at the healed length.** The converted model matches or beats its donor on held-out perplexity and stays within noise on benchmarks at the length it was healed at (parity-or-better is the defensible claim — the converted model received continued training the donor did not).
5. **The efficiency curve holds to 128K.** Per-query keys stay ~constant; hierarchy speedups measured at **5.2× / 7.9× / 10.6×** at 32K / 64K / 128K.
6. **The open problem: union-softmax length calibration.** Push past the healed length and the same weights that retrieve perfectly in dense mode, with correct block selection, get buried by the routed softmax as the candidate count grows — measured retrieval falls from parity at the heal length toward zero by 4×. Notably, **more training dose stops helping past a point**: the length curriculum recovers the first doubling of context but stalls at ~4× the heal length, and a dedicated extra pass at that length left retrieval near zero — this is a scale/calibration problem, not a knowledge gap. The architectural fix above — bounded logits, one shared scale, an identity-until-extrapolation temperature — is the current answer being validated.

## Converting a pretrained model

Keep every weight; replace only the attention computation; heal in stages:

| Stage | What trains | Purpose |
|---|---|---|
| 0 | summarizers only (distill-init) | cheap initialization against the frozen donor's own attention mass |
| 1 | attention + summarizers | re-align Q/K geometry to the sparse budget (everything else frozen) |
| 2 | full model, **with capability rehearsal** | settle the network into the sparse regime without trading capabilities away |
| 3 | full model, **mixed context lengths, calibrated softmax** | exercise the routed path across candidate-count regimes with the architectural calibration in place |

Full procedure, hyperparameters, and monitoring requirements: [docs/training-recipe.md](docs/training-recipe.md).

## Status

- **Validated (0.5B):** the fundamentals above and the efficiency curve — quality parity with the donor and retrieval above donor level at the healed length, per-query cost held to ~2.1K keys, hierarchy speedups confirmed to 128K.
- **In progress — the primary objective:** native-length (e.g. 32K) retrieval parity for a release-scale conversion, using the architecturally-calibrated softmax described above. The calibration failure has been localized (weights and block selection are fine; the union softmax's scale behavior is the crux), and the corrected conversion run is underway. **This is not yet a claimed result** — the benchmark table below is pending it.
- **Upcoming (release scale):** target donor **Qwen3.5-9B** (primary) or **Qwen3.5-4B** (alternative) — both Apache-2.0; selection pending verification of attention layout and context configuration.
- **Model:** coming to Hugging Face — link will appear here at release.

| Benchmark | HBA (converted, release scale) | Donor baseline |
|---|---|---|
| MMLU | pending primary-objective run | pending |
| HellaSwag | pending primary-objective run | pending |
| ARC | pending primary-objective run | pending |
| GSM8K | pending primary-objective run | pending |
| HumanEval | pending primary-objective run | pending |
| Needle-in-haystack 4K / 32K / 128K | pending primary-objective run | pending |
| RULER | pending primary-objective run | pending |
| Held-out PPL (web / books / code) | pending primary-objective run | pending |

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
