"""The in-training capability panel (docs/training-recipe.md, "Monitoring: the
capability panel is a first-class training signal").

Perplexity tells you the model is a fluent language model; it tells you nothing
about whether it can retrieve, and a capability can be destroyed mid-run with no
perplexity signature. These probes are cheap enough (seconds of compute) to run on
a fixed cadence throughout every healing stage and treat a collapsing result as an
abort signal, not a curiosity.

`induction_probe_quick` is the single-length probe heal.py runs during
non-length-curriculum stages. `run_panel` runs it across several context lengths
in one call -- the length-curriculum stage (stage 3) needs a probe at 1x/2x/4x the
current training context, because a single short-length probe can show a perfect
plateau while the model is silently cliffing at longer lengths (exactly the
failure the length curriculum exists to prevent; see docs/training-recipe.md,
"Length curriculum").
"""

import numpy as np
import torch


@torch.no_grad()
def induction_probe_quick(model, cfg, cos, sin, n=2048, trials=16):
    """Fast in-training capability gate: the same std-placement induction probe as
    evals.induction_probe, at one fixed length (monitor the thing the eval suite
    will later measure). Plants one random (key,val) bigram 3x in a random-token
    haystack, presents the key once more at the end, PASS if argmax==val. Returns
    accuracy over `trials`. Runs in eval mode, no grad; the caller's train() state
    is restored before returning."""
    was_training = model.training
    model.eval()
    V = cfg.vocab_size
    dev = next(model.parameters()).device
    hit = 0
    for t in range(trials):
        g = np.random.default_rng(1234 + t + n)          # identical construction to evals.py
        ids = g.integers(100, V, size=n).astype(np.int64)
        key = int(g.integers(100, V)); val = int(g.integers(100, V))
        for s in np.linspace(0.05, 0.9, 3):
            p = int(s * (n - 4))
            ids[p] = key; ids[p + 1] = val
        ids[n - 1] = key
        x = torch.from_numpy(ids)[None].to(dev)
        logits = model(x, cos, sin, cfg.mem_elem_cap, mode="eval", tail=2)
        hit += (int(logits[0, -1].argmax()) == val)
    if was_training:
        model.train()
    return hit / trials


def run_panel(model, cfg, panel_tabs, ns, trials=16):
    """Run `induction_probe_quick` at every length in `ns`. panel_tabs: {n: (cos,
    sin)} precomputed RoPE tables for each length the caller intends to probe.
    Returns [(n, acc), ...] in the order of `ns`.

    One probe-design trap (docs/training-recipe.md): a "far retrieval" probe only
    exercises routing if the candidate block count at that length exceeds
    cfg.k_blocks -- otherwise top-k selects every block and the probe degenerates
    into "attention works". Callers should size `ns` accordingly (the stage-3
    default panel spans 4096/8192/16384/32768, well past that threshold)."""
    out = []
    for n in ns:
        cos, sin = panel_tabs[n]
        out.append((n, induction_probe_quick(model, cfg, cos, sin, n=n, trials=trials)))
    return out
