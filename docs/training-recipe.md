# Conversion recipe: dense donor → HBA

Converting a pretrained dense-attention model to HBA is a *healing* problem, not a distillation problem. Position-free content routing does not exist in a dense model's query/key geometry, and it **cannot be distilled onto frozen weights** — our attempt recovered almost none of the routing-recall gap (0.20 vs a 0.55 target). The geometry must adapt under continued pretraining. Everything below is organized around making that healing safe: the donor's knowledge must survive, its capabilities must survive, and both must be *verified during training*, not discovered afterward.

The conversion reuses **every donor weight by reference** — embeddings, Q/K/V/O projections, norms, MLPs, LM head — and replaces only the attention computation.

## Donor selection

The recipe is donor-agnostic within these criteria:

- **GQA attention layout.** The conversion grafts one SlotSummarizer per (layer, KV head) and selects blocks per KV-head group; a grouped-query attention layout is what the machinery is built for.
- **License permitting derivative weight release.** The validation donor (Qwen2.5-0.5B-Instruct) and both release candidates are Apache-2.0.
- **Native context length.** A longer-context donor gives the length-curriculum stage more headroom before any extension beyond what the donor ever saw.
- **Headroom for the length curriculum** in the training budget: stage 3 must cover the candidate-count regimes you will serve (below).

Release-scale candidates: **Qwen3.5-9B (primary)** and **Qwen3.5-4B (alternative)**, both Apache-2.0; final selection pending verification of attention layout and context configuration.

## Stages

Token budgets are given as ratios of parameter count (validated at 0.5B; the release-scale run uses the same ratios — at a 9B donor, stages 1–3 instantiate to roughly 1B / 4–5B / 1–3B tokens: continued-pretraining scale, not pretraining scale). LRs are cosine-decayed with short warmup; AdamW (β 0.9/0.95), grad clip 1.0, no weight decay on embeddings, norms, biases, or summarizers.

| Stage | Trains | Frozen | Token budget | LR | Purpose |
|---|---|---|---|---|---|
| **0. Distill-init** | summarizers only | entire donor | — (hours on CPU/laptop-class hardware) | — | Initialize each SlotSummarizer by distilling against the frozen donor's own attention mass (KL, see below). This is an **init, not a result** — it starts the router near mean-pool quality with differentiated slots so the aux loss has a sane starting point. |
| **1. Attention-only heal** | Q/K/V/O + summarizers | embeddings, MLPs, norms | ~0.1–0.15 tokens/param | 2e-4 | Re-align query/key geometry to the sparse budget without touching the donor's knowledge (which lives largely in the MLPs). |
| **2. Full-parameter heal** | everything | — | ~0.4–0.6 tokens/param | **2.5e-5** | Let the whole network settle. **Must include capability rehearsal** (below). The low LR is load-bearing: 4× higher destroyed retrieval capability at validation scale. |
| **3. Length-curriculum extension** | everything | — | ~0.1–0.3 tokens/param | ≤ stage-2 LR | Calibrate the union softmax across the candidate-count regimes you will serve (below). Repair-scale, not retraining-scale. |

### The summarizer auxiliary loss, and the gradient-isolation rule

The summarizer trains against a **KL to the donor-path attention mass**: the teacher is the group-averaged, position-free content mass over candidate blocks (what the exact attention actually gives each block), and the loss is

```
L_aux = KL( p_teacher(blocks) ‖ softmax(slot scores) )
```

**Gradient isolation is a hard rule:** the summarizer receives gradients *only* from `L_aux`, computed on **detached** queries/keys; the LM loss trains Q/K/V/O/MLP/embeddings and *never* the summarizer. Without isolation the summarizer can leak degenerate solutions into the routing path (and the LM loss can deform summaries away from their advertising role). Enforce it by construction (detach) and verify it with an exact gate: `∂L_aux/∂(q,k,v) = 0` and `∂L_LM/∂(summarizer) = 0`, exactly, before any run.

Total loss: full next-token CE through the hard top-k selected exact attention (no straight-through tricks), plus `w_aux · L_aux`.

## Capability rehearsal (do not skip)

**The failure this prevents:** full-parameter healing on generic text destroyed the model's induction/retrieval capability — 0.25 → 0.06 on a copy probe — **while perplexity stayed flat and the stage bought zero perplexity over the attention-only stage**. Nothing in ordinary text rewards content-agnostic copying; gradient descent trades the circuit away for a loss improvement too small to see. Perplexity is structurally blind to this. The fix (rehearsal + 4× lower LR + live probes) revived the capability within 50 training steps and held it above the donor's level for the rest of the run.

Rehearsal data spec:

- **Quantity:** a few percent of stage-2/3 tokens (~2.6% validated) of retrieval-exercising data — key/value pairs planted in otherwise-real text, where predicting the continuation requires copying from earlier context.
- **Format variation is required.** Vary the surface form along several independent axes (delimiters, key/value token types, casing, spacing, phrasing, plant count). A single fixed format teaches a format detector, not a capability.
- **Leakage rule:** rehearsal formats must be *measurably distinct* from every evaluation probe's surface form. You are rehearsing the capability, not training on the test.
- **Distance coverage:** plant pairs at all distances the architecture serves — inside the local window, just beyond it, and deep in routed territory. A small number of pairs per training window suffices; coverage across distances matters more than density.
- In stage 3, plant across **all curriculum lengths**, so the routed path is exercised at every candidate-count regime.

## Length curriculum (do not skip either)

**The failure this prevents:** the union softmax's calibration is specific to the candidate-count regime seen in training. At 2× the healed context, the same weights retrieved perfectly in dense mode (answer rank 1) while the routed path buried the answer (rank ~56,000) — dilution by the doubled candidate crowd, not a weight or selection failure. An analytic log-N temperature recovered only a fraction of the gap: **the model must see the longer regime in training.**

Spec:

- **Mixed-context scheduling:** interleave context lengths spanning the serving range (e.g. 1×/2×/4× the base heal context, extending upward for a long-context release) rather than a monotone ramp — the short-context calibration must not be forgotten while the long one forms.
- **Tokens/step invariance:** hold global tokens per optimizer step constant across context lengths (adjust batch composition, not step size), so the gradient scale and LR schedule mean the same thing at every length.
- **Rehearsal at every length**, per the distance-coverage rule above.
- **Per-length capability panel:** during any heal, run capability probes at **1×, 2×, and 4× the current training context** on a fixed cadence (see Monitoring). Our validation run probed only at one short length and showed a perfect plateau while the model was cliffing at 2× — the cliff would have been visible within 200 steps with a length axis.
- One probe-design trap: a "far retrieval" probe only exercises routing if the candidate block count exceeds `k` at the probe's context length — otherwise top-k selects every block and the probe degenerates into "attention works". Size probe contexts accordingly.

**Dose-response (measured):** repair at each length tracks the gradient spent there, and
only there. In our validation curriculum (4K/8K/16K mixed 1:1:2, 100M tokens), the 2×
length recovered fully (induction 0.031 → 0.469, ≈ the dense-mode ceiling), the 4× length
— which received light rehearsal density and depends mostly on routed-path retrieval —
moved only marginally, and untrained lengths stayed at floor. Budget the stage by
**per-length dose** (rehearsal density × steps at that length), not total tokens; expect
mid-run oscillation across lengths while the LR is high (calibrations compete) and judge
only the annealed endpoint. If serving 128K, the curriculum must reach lengths close to the serving
regime — measured repair did not extrapolate meaningfully beyond the trained lengths.

## Monitoring: the capability panel is a first-class training signal

Perplexity tells you the model is a fluent language model. It tells you nothing about whether the model can retrieve — a model can be fluent and a complete non-retriever, and a capability can be destroyed mid-run with no perplexity signature (both observed directly). Therefore:

- Run a **capability panel** every ~200 steps: (a) a standard copy/induction probe; (b) a near-field probe (answer inside the local window — exercises the RoPE path); (c) a far-field probe (answer reachable only via routing); (d) a fixed held-out LM-loss batch as the smooth quality signal. Each probe is seconds of compute; the panel's cost is negligible next to what it protects.
- Probes run at multiple lengths (1×/2×/4× training context) per the length-curriculum spec.
- Treat a collapsing probe as an **abort signal**, not a curiosity: halt, roll back to the last good checkpoint, diagnose (see the discriminator ladder in [evals.md](evals.md)) before continuing. A doomed stage completed is budget burned.
- Log the panel machine-readably; it is the evidence base for every "capability preserved" claim you will later make.

## Correctness gates (before any run)

The routed attention path is exotic enough that silent wrongness is the default failure mode. The discipline that kept validation honest:

- **Keep a naive reference oracle forever.** Every optimized path — the fused kernel, the chunked long-context evaluator — is permanently gated against a transparent naive implementation on identical inputs (our fused-vs-naive agreement: ~1e-6). The oracle is never deleted for being slow.
- **Equivalence gate:** with routing configured to select everything, the converted model must reproduce the donor's logits (measured: max |Δ| ≈ 8e-5). This proves the swap is exact before healing changes anything.
- **Causality gate:** perturbing future tokens must change past logits by exactly 0.0, on both train and eval paths.
- **Gradient-isolation gate:** the exact-zero checks from the aux-loss section.
- **Top-k discontinuity honesty:** top-k selection is discontinuous, so float-level noise can flip near-tie block picks between two correct implementations. Gate the discontinuity-free property (select-all mode) at tight tolerance, and separately bound the flip *fraction* and per-flip magnitude at the real config — don't loosen the main gate to paper over ties.
- **Refuse to start:** training scripts should hard-refuse to launch unless the gates are green. Gates that can be skipped will be.
