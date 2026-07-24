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

## Softmax length-calibration (QKNorm) and healing

The architectural fix for union-softmax dilution (`docs/design.md`, "Softmax length-calibration": QKNorm + a bounded 1/d content scale + a shared union scale + a clamped log-length extrapolation temperature) is a **Q/K geometry change**, so it interacts with healing in ways worth stating explicitly:

- **A fresh conversion is required.** QKNorm changes Q/K statistics fundamentally — the converted model can no longer reproduce the donor's own attention exactly at init, by design (that departure is the whole point). This means stage 0→1→2→3 must run fresh under `cfg.qknorm=True`; you **cannot** seed a QKNorm run from an existing pre-QKNorm checkpoint (the summarizer distill-init, the Q/K healing trajectory, and the correctness-gate reference export are all specific to one `qknorm` setting — see "Correctness gates" below).
- **Init for healing absorption.** `hba.model.init_qknorm_gains` (called automatically by `build_hba` whenever `cfg.qknorm=True`) calibrates each layer's QKNorm gains from the **frozen donor's own** pre-RoPE Q/K RMS on a short random-token batch, so the post-QKNorm content logit starts at roughly the donor's own attention temperature (see that function's docstring for the closed-form derivation) — healing then absorbs the architecture change from a sane starting point instead of an arbitrary one, the same "don't start healing from a broken place" discipline the rest of this recipe applies elsewhere.
- **`n_cal` is the calibration boundary, not a training-length knob.** It defaults to `native_ctx` — the donor's own native/trained context (32,768 for the 0.5B validation donor; the release target is likewise a 32K-native donor, `docs/training-recipe.md`'s Donor selection section). Within `n_cal`, QKNorm + the bounded content scale is what calibrates the union softmax — architecturally, not learned — so the log-length temperature (`HBAConfig.temp_c`) is identity there by construction. The temperature is the **extrapolation** knob: it only grows past `n_cal`, for serving beyond the model's native/trained length. This is a different job from the length curriculum below, which is still what teaches the model to *use* the routed path well at every trained length, including lengths at or under `n_cal`.
- **CLI to run a fresh qknorm=True conversion:**

  ```
  python -m hba.convert --distill-init [--smoke]     # stage 0 (unaffected by qknorm; frozen-donor distill)
  python -m hba.convert --gates [--smoke]             # runs gate_qknorm_math in place of gate_equivalence
  python -m hba.convert --save-init --from-distill [--smoke]
  python -m hba.convert --export-ref                  # re-export: qknorm=True reference is NOT donor-equivalence, see below
  python -m hba.heal --phase stage1 --resume
  python -m hba.heal --phase stage2 --resume
  python -m hba.heal --phase stage3 --resume
  ```

  `HBAConfig.qknorm` defaults to `True`; pass `qknorm=False` (e.g. via a config override in a driver script) to run the ablation/regression path instead — byte-identical to the pre-QKNorm attention, with `gate_equivalence` (not `gate_qknorm_math`) as the correctness gate.

## Capability rehearsal (do not skip)

**The failure this prevents:** full-parameter healing on generic text destroyed the model's induction/retrieval capability — 0.25 → 0.06 on a copy probe — **while perplexity stayed flat and the stage bought zero perplexity over the attention-only stage**. Nothing in ordinary text rewards content-agnostic copying; gradient descent trades the circuit away for a loss improvement too small to see. Perplexity is structurally blind to this. The fix (rehearsal + 4× lower LR + live probes) revived the capability within 50 training steps and held it above the donor's level for the rest of the run.

Rehearsal data spec:

- **Quantity:** a few percent of stage-2/3 tokens (~2.6% validated) of retrieval-exercising data — key/value pairs planted in otherwise-real text, where predicting the continuation requires copying from earlier context.
- **Format variation is required.** Vary the surface form along several independent axes (delimiters, key/value token types, casing, spacing, phrasing, plant count). A single fixed format teaches a format detector, not a capability.
- **Leakage rule:** rehearsal formats must be *measurably distinct* from every evaluation probe's surface form. You are rehearsing the capability, not training on the test.
- **Distance coverage:** plant pairs at all distances the architecture serves — inside the local window, just beyond it, and deep in routed territory. A small number of pairs per training window suffices; coverage across distances matters more than density.
- In stage 3, plant across **all curriculum lengths**, so the routed path is exercised at every candidate-count regime.

### Donor knowledge-distillation (optional, additive to rehearsal)

Stage 2 can optionally add a second protection alongside capability rehearsal: a full-logit KL between the student's next-token distribution and the **frozen original donor's**. This is a distinct mechanism from the summarizer's own auxiliary KL (above) — that one distills *routing* (which blocks a query attends to) onto the summarizer, gradient-isolated from the LM path; donor KD distills the *output distribution* (which token comes next) onto Q/K/V/O/MLP/embeddings/LM-head, the same parameter groups the LM loss already trains. The two losses never touch the same gradient path and must not be confused with each other.

**Why:** capability rehearsal protects *enumerated* capabilities — whatever the fam-data mix and the probe panel happen to exercise. A full-logit KL to the frozen donor is a broader, complementary protection: at every training position it pulls the student's entire next-token distribution toward the donor's, covering capabilities no probe ever happened to test. It is **additive**, not a substitute — the rehearsal data mix stays in place exactly as before.

**Loss:** the standard temperature-scaled forward KL, `T² · KL(softmax(z_teacher/T) ‖ softmax(z_student/T))`. The direction matters: teacher-as-target forward KL is mass-covering (the student is penalized for placing near-zero probability wherever the teacher places mass), which forces the student to cover the donor's whole distribution rather than mode-collapsing onto a subset of it — the same silent-narrowing failure mode rehearsal exists to catch elsewhere. `T²` is the standard Hinton-distillation correction that keeps the KD gradient's scale roughly temperature-invariant. See `hba/kd.py`'s module docstring for the full derivation, the ignore_index/normalization convention (matched to the CE loss so the two agree on which positions count), and the chunked recompute-in-backward implementation (mirrors `chunked_ce.py`'s pattern so the student's full `[B, n, V]` logits are never materialized).

**Teacher placement:** the teacher must be a *separate*, frozen copy of the original donor — never the student's own `.donor` submodule, which is being trained in stage 2 and would drift, turning the KL into a no-op consistency loss instead of a distillation target. At the validated 0.5B scale, the frozen teacher (~1 GB, no gradients, no optimizer state) co-resides on the same device as the student for the pilot; a substantially larger donor would instead want the teacher's logits precomputed offline or placed on a second device, since holding two full copies of a large donor (one training, one frozen) on one device stops being trivial. `hba/kd.py`'s module docstring notes the extension point (a per-chunk teacher callable) for that case.

**Cost:** one extra dense donor forward per training micro-step, under `torch.no_grad()`. The student forward is unchanged — one student forward already feeds both the CE and (when enabled) the KD loss, from the same hidden states.

**Stage-2 only:** KD is refused (not silently ignored) outside stage 2. Stage 1 barely moves the donor's weights (attention-only), so there is little yet to distill against; stage 3's dense teacher forward would be O(n²) at the length-curriculum context lengths — the identical affordability argument that turns the summarizer's aux-KL teacher off in stage 3 (above).

**Correctness gate:** `hba/gates.py`'s `gate_kd` follows the same discipline as `gate_chunked_ce` — the chunked KD path is checked bit-honest (loss and every gradient) against a transparent unchunked oracle, plus a sanity property (KD loss is exactly 0 when the student's distribution equals the teacher's, at multiple temperatures). Wired into the gate suite only when a run actually has KD enabled.

## Length curriculum (do not skip either)

**The failure this prevents:** the union softmax's calibration is specific to the candidate-count regime seen in training. At 2× the healed context, the same weights retrieved perfectly in dense mode (answer rank 1) while the routed path buried the answer (rank ~56,000) — dilution by the doubled candidate crowd, not a weight or selection failure. An analytic log-N temperature recovered only a fraction of the gap: **the model must see the longer regime in training.** `docs/design.md`'s "Softmax length-calibration" section (QKNorm + a bounded, shared union scale) makes the calibration *within* a trained length architectural rather than learned, and adds a clamped extrapolation temperature for lengths beyond `n_cal` — but it does not replace this section: the architecture bounds the logits so calibration is well-posed at every length, it does not by itself teach the model to route and weight well at candidate-count regimes it never trained at. The length curriculum is still how that gets learned.

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

- **Keep a naive reference oracle forever.** Every optimized path — the fused kernel, the chunked long-context evaluator — is permanently gated against a transparent naive implementation on identical inputs (our fused-vs-naive agreement: ~1e-6, and still ~1e-4-tight with QKNorm on — see below). The oracle is never deleted for being slow.
- **Equivalence gate (qknorm=OFF only):** with routing configured to select everything, the converted model must reproduce the donor's logits (measured: max |Δ| ≈ 8e-5). This proves the swap is exact before healing changes anything — but only in `qknorm=False` mode. With `qknorm=True`, QKNorm is a *deliberate* departure from the donor's Q/K statistics, so this equivalence no longer holds by design, and asserting it would be wrong, not merely loose. `hba.gates.run_all_gates`/`check_gates` skip `gate_equivalence` entirely when `cfg.qknorm=True` (they do not silently relax its tolerance) and run `gate_qknorm_math` in its place: an internal-consistency check that the model's own QKNorm computation matches a transparent independent reference implementation, and that the resulting per-head RMS exactly equals the learned gain (the property that makes the content logit bounded — `docs/design.md`, "Softmax length-calibration"). Causality, path-equivalence (dense vs. eval), and grad-isolation all run **unconditionally**, regardless of `qknorm` — both attention backends share one `_content_qk` call site, so these remain the load-bearing "the QKNorm path is internally consistent" checks alongside `gate_qknorm_math`.
- **Causality gate:** perturbing future tokens must change past logits by exactly 0.0, on both train and eval paths.
- **Gradient-isolation gate:** the exact-zero checks from the aux-loss section.
- **Top-k discontinuity honesty:** top-k selection is discontinuous, so float-level noise can flip near-tie block picks between two correct implementations. Gate the discontinuity-free property (select-all mode) at tight tolerance, and separately bound the flip *fraction* and per-flip magnitude at the real config — don't loosen the main gate to paper over ties.
- **Reference export is qknorm-mode-specific:** `hba.convert --export-ref` exports fp32 donor + HBA-equiv logits on a fixed input for `hba.gates.check_reference` to compare against on a new machine. With `qknorm=True`, the HBA-equiv half of that export is no longer a donor-equivalence reference — it is the QKNorm'd model's *own* fp32 output, an internal self-consistency check (does a new machine/build reproduce this one) rather than a donor-equivalence check. Re-export whenever `qknorm` flips; a `qknorm=False` export cannot validate a `qknorm=True` build or vice versa.
- **Refuse to start:** training scripts should hard-refuse to launch unless the gates are green. Gates that can be skipped will be.
