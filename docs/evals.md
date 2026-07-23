# Evaluation protocol

The protocol is ordered deliberately: **capability gates run first**, because expensive sweeps that measure a capability the model doesn't have produce hours of meaningless zeros. Everything downstream is conditioned on the gates.

Throughout: "validation scale" = the completed conversion of a 0.5B donor (Qwen2.5-0.5B-Instruct); "release scale" = the upcoming full conversion run — target donor Qwen3.5-9B (primary candidate) or Qwen3.5-4B (alternative), selection pending final verification — marked *pending*.

## 1. Capability gates

Cheap probes (~minutes) that a model must pass before any sweep runs:

- **G1 — donor induction:** plant a random key/value bigram several times in a long haystack, present the key once; pass if the model's argmax completes it (chance ~1e-5). Run on the *donor* first — if the donor can't do it, the comparison is unmeasurable, not failed.
- **G2 — converted induction:** the same probe on the converted model after each healing stage, across lengths (1×/2×/4× heal context). A G2 failure stops everything and triggers diagnosis — do not sweep.
- **Diagnosis tool — the discriminator ladder.** Three cells that localize any retrieval failure in minutes: (a) dense mode (same weights, full attention) — isolates the weights; (b) routing off (window + sinks only) — isolates the local path; (c) routing on — the full mechanism. An in-window failure is not a routing failure; always test with routing off.

**Validation-scale gate results** (induction accuracy):

| Probe | Converted (healed) | Donor |
|---|---|---|
| Induction @2K | **0.875** | 0.531 |
| Induction @4K | **0.719** | 0.500 |

Gate probes are small-sample by design (16–32 trials per cell — they exist to be cheap); treat differences smaller than ~0.1 as noise. The converted-above-donor direction is consistent across lengths and across the in-run probe history.

## 2. Quality

- **Held-out perplexity** vs the donor on disjoint web / books / code sets, identical tokenization, identical precision regime for every model in one comparison. Bar: within 5–10% of donor.
- **Standard benchmarks** (log-likelihood scoring): bar is parity within a few points.

**Validation-scale results:**

| Metric | HBA (converted 0.5B) | Donor | Notes |
|---|---|---|---|
| PPL books | **20.11** | 20.3 | |
| PPL web | **14.37** | 15.26 | |
| PPL code | **2.59** | 2.71 | |
| HellaSwag | 0.450 | 0.454 | within noise |

Caveat, stated plainly: the converted model received 575M tokens of continued training on this data mix and the donor did not; "parity or better" is the defensible claim, not "strictly beats".

| Metric | HBA (release scale) | Donor |
|---|---|---|
| MMLU / HellaSwag / ARC / GSM8K / HumanEval | pending release run | pending |
| Held-out PPL (web/books/code) | pending release run | pending |

## 3. Retrieval

- **Needle-in-real-text** at 4K → 128K, needle planted in a held-out book corpus, **≥3 seeds, mean ± SE**. Baselines: raw donor and donor+YaRN (the honest long-context baseline — raw RoPE is a strawman beyond native context), all sharing one evaluation code path and precision regime.
- **RULER** (or an equivalent public long-context suite) at release scale.
- Flat and hierarchical selection both evaluated where both run (fidelity check doubles as a retrieval cell).

**Validation-scale results** (needle accuracy, mean of 3 seeds):

| Method | 4K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| HBA converted (healed at 4K, **pre length-curriculum**) | **0.493** | 0.0 | 0.0 | 0.0 | 0.0 |
| HBA converted, **+ length-curriculum stage** (4K/8K/16K mix, 100M tok) | **0.499** | 0.021 | 0.0 | 0.0 | not rerun † |
| Donor (native 32K) | 0.458 | 0.356 | 0.238 | 0.037 | 0.0 |
| Donor + YaRN | — | — | — | 0.097 | **0.0** |

† bounded by the measured zeros at 16-64K.

The curriculum stage's clearest effect is on the **induction gate**, the substrate probe:
at 2× the original healing context (8K), accuracy went **0.031 → 0.469** (dense-mode
ceiling: 0.531) after 100M mixed-length tokens — while quality stayed at donor parity
(held-out PPL within ~3% of the pre-curriculum checkpoint; HellaSwag/ARC-e unchanged
or better). Repair tracked the training dose at each length: 8K (well-dosed) recovered
fully; 16K (lightly dosed, retrieval mostly through routed blocks) moved only trace
amounts; 32K (untrained) stayed at zero. Validation-scale dosing was deliberately small
(~$1 of compute); the release-scale run budgets its curriculum stage by per-length dose
accordingly.

Read this table in both directions. At its healed length, HBA beats the donor. Beyond it, this pre-curriculum checkpoint collapses — the softmax-dilution calibration failure analyzed in [design.md](design.md), with the mechanism fully localized (the same weights retrieve at rank 1 in dense mode; selection is correct; only the union softmax's calibration fails). The length-curriculum stage of the recipe exists because of this row, and the release-scale run includes it. Note also the last column: **nothing retrieves at 128K** — the YaRN'd donor is at 0.097 by 64K and 0.0 at 128K — while HBA's per-query cost at 128K is unchanged from 4K. The needle/RULER rows below are the ones to watch.

| Method | 4K | 32K | 64K | 128K | RULER |
|---|---|---|---|---|---|
| HBA release scale (with length curriculum) | pending | pending | pending | pending | pending |
| Donor / donor+YaRN | pending | pending | pending | pending | pending |

Supporting validation-scale evidence that content routing is the right long-range bet: in matched from-scratch comparisons, the routed model's retrieval degrades with no cliff while dense RoPE collapses (1.00 → 0.03 by 8× training length), and the gap to a YaRN'd dense control shrinks monotonically with length (+0.28 → +0.24 → +0.15 → +0.04 → −0.02), crossing zero at ~32× training length (single seed; the monotone trend is the robust finding, the crossover point is within noise).

## 4. Efficiency

- **Per-query attended keys** vs context length (analytic + verified in the implementation): constant, ~2.1K in the reference config.
- **Peak evaluation memory** vs context length, measured end-to-end.
- **Hierarchical selection speedup** (summary comparisons per query vs flat) and **fidelity** (selected-set agreement vs flat).

**Validation-scale results** (0.5B, bf16):

| Context | Peak memory | Hierarchy speedup | Per-query keys |
|---|---|---|---|
| 32K | 4.3 GiB | 5.2× | ~2.1K |
| 64K | 6.8 GiB | 7.9× | ~2.1K |
| 128K | 11.8 GiB (KV-cache dominated) | 10.6× | ~2.1K |

Release-scale efficiency table: pending release run.

## Protocol notes

- All head-to-head numbers within a sweep share one precision regime and one code path; never compare across bf16/fp32 or across evaluator implementations.
- Healing runs at validation scale used a single training seed (evaluation seeds were varied); conclusions are about the recipe's feasibility, not run-to-run training variance.
- Correctness gates ([training-recipe.md](training-recipe.md)) precede every reported number; every optimized evaluation path is checked against a naive oracle on identical inputs.
