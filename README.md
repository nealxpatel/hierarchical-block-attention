# Hierarchical Block Attention (HBA)

**Near-constant per-query attention cost at any context length, without positional extrapolation.**

HBA decomposes attention into three parts: a fixed-size **local window** with RoPE (position-aware, never leaves its training distribution), a handful of always-visible **anchor/sink tokens**, and **position-free content routing** — each query scores learned summaries of fixed-size key/value blocks and attends to only the top-k, selected hierarchically at scale. All three regions share a single softmax over their disjoint union, so the model is trained end-to-end as one attention operation. The result is a per-query read budget that stays roughly constant (~2K keys in our reference config) whether the context is 4K or 128K.

This repo documents the architecture, the empirical evidence behind each design decision, and a staged recipe for **converting an existing pretrained dense-attention model** to HBA — validated end-to-end at 0.5B scale, with a release-scale conversion of a current-generation Qwen donor upcoming.

## Why it works

Each design choice is a claim about attention that we tested directly:

1. **Attention is empirically sparse — and gets sparser with length.** Measured on a pretrained 0.5B transformer: ~95% of attention mass falls in under 9% of keys at 8K context, and the concentration *grows* as context grows. The mass decomposes cleanly into sink tokens (~25%), the local window (~39%), and a routable long-range remainder (~20%) — which maps exactly onto HBA's three components. A ~2K-token read budget recovers ≥90% of the mass.

2. **Long-range attention is content-addressed, not position-addressed.** Position matters locally; HBA's RoPE window is a fixed size, so it never extrapolates beyond its training distribution — there is no positional cliff to fall off. In controlled comparisons, dense RoPE retrieval collapses from 1.00 to 0.03 by 8× training length, while position-free content routing degrades smoothly with no cliff, and its gap to a position-interpolated (YaRN) dense baseline shrinks monotonically with length and crosses zero at ~32× training length. (Honest caveat: the crossover point itself is a single-seed measurement within noise; the monotone trend across five lengths is the robust finding.)

3. **Block summaries must be learned, and co-trained.** Mean-pooled block summaries are a weak router in every regime we tested. Learned multi-slot summarizers lift routing recall **1.8–5.1×** at every evaluation length. And routing cannot be bolted on: distilling summarizers against a *frozen* pretrained model recovers almost none of the gap (0.20 recall vs a 0.55 target) — the query/key geometry itself must adapt, which is why the conversion recipe's continued-pretraining ("healing") phase is load-bearing, not cosmetic.

4. **Hierarchy makes selection logarithmic with near-perfect fidelity.** Two-level beam search over super-summaries of summaries recovers flat top-k selection almost exactly at moderate depth, with measured selection speedups of **5.2× at 32K, 7.9× at 64K, and 10.6× at 128K** context.

5. **Quality is preserved.** From scratch, the full HBA stack costs only **+3–5% perplexity** versus an identical dense baseline while reading ~97% of context through top-16 routing. Converted from the validation-scale donor (Qwen2.5-0.5B-Instruct), the healed HBA model **matches or beats the donor** on held-out perplexity across web/books/code and stays within noise on benchmarks (parity-or-better is the defensible claim — the converted model received continued training the donor did not; see [docs/evals.md](docs/evals.md)). Per-query compute stays near-constant with length: ~2.1K keys read per query and 11.8 GiB peak evaluation memory (KV-cache dominated) at 128K.

## Converting a pretrained model

Take a dense-attention donor, keep every weight, replace only the attention computation, and heal in stages: **(0)** distill-initialize the block summarizers against the frozen donor's own attention mass (a cheap init, not a result); **(1)** attention-only healing (Q/K/V/O + summarizers train, everything else frozen), re-aligning the query/key geometry to the sparse budget; **(2)** full-parameter healing **with capability rehearsal** mixed into the data; **(3)** length-curriculum extension at mixed context lengths. Two lessons from validation are stated plainly because they will bite anyone who skips them.

- **Capability rehearsal is not optional.** Full-parameter healing on generic text silently destroyed the model's retrieval/induction capability (0.25 → 0.06 on a copy probe) *while perplexity stayed flat* — nothing in ordinary text rewards content-agnostic copying, so gradient descent trades the circuit away. A few percent of format-varied retrieval-exercising data, a 4× lower LR, and a live capability probe revived the capability within 50 steps and held it above the donor's level. Perplexity cannot see this failure; a capability probe can.
- **Train at the lengths you serve.** The union softmax's calibration is specific to the candidate-count regime it was trained in: at 2× the healing context, the same weights retrieve perfectly in dense mode (answer rank 1) while the routed path's diluted softmax buries the answer (rank ~56,000) — many slightly-warm routed logits drown a still-sharp signal. An analytic log-candidate-count temperature helps (~2× accuracy, ~30× answer mass) but plateaus far below dense; the model must see the longer regime during training. Stage 3 exists because of this.

Details and evidence: [docs/training-recipe.md](docs/training-recipe.md).

## Status

**Current state:** architecture and conversion recipe validated end-to-end at validation scale (0.5B donor: Qwen2.5-0.5B-Instruct — quality parity with donor, retrieval at healing length above donor, efficiency curve confirmed to 128K). A full release-scale conversion run is upcoming. **Target donor: Qwen3.5-9B (primary candidate) or Qwen3.5-4B (alternative)** — both Apache-2.0; final selection pending verification of attention layout and context configuration (criteria in [docs/training-recipe.md](docs/training-recipe.md)). The length-curriculum stage — motivated by the calibration finding above — is part of that run, so all release-scale numbers below are pending.

**Model:** coming to Hugging Face — link will appear here at release.

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

0.5B validation-scale numbers are available now in [docs/evals.md](docs/evals.md).

## Reproducing

The conversion procedure — stage table, hyperparameters, rehearsal data spec, length curriculum, monitoring, and correctness gates — is specified as a reproducible recipe in [docs/training-recipe.md](docs/training-recipe.md), with the evaluation protocol in [docs/evals.md](docs/evals.md).

## Citation

```bibtex
@misc{patel2026hba,
  author       = {Patel, Neal},
  title        = {Hierarchical Block Attention},
  year         = {2026},
  howpublished = {\url{https://github.com/nealpatel/hierarchical-block-attention}}
}
```

## Repo map

| Path | Contents |
|---|---|
| [docs/design.md](docs/design.md) | Full architecture: notation, the three attention components and their union softmax, GQA-grouped selection, slot summarizers, hierarchy, softmax calibration, complexity analysis |
| [docs/training-recipe.md](docs/training-recipe.md) | The staged conversion recipe as a reproducible procedure: stage table, rehearsal data spec, length curriculum, monitoring, correctness gates |
| [docs/evals.md](docs/evals.md) | Evaluation protocol and validation-scale results |
| LICENSE | Apache-2.0 |
