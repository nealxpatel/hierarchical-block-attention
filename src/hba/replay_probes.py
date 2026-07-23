"""Offline replay: feed a probe_log*.jsonl (heal.py's capability-panel log; see
probes.py and early_stop.py) through EarlyStopEngine and print, firing by
firing, which rule (if any) would have fired and why.

This is pure CPU and needs neither torch nor a model: early_stop.py's decision
logic is torch-free by design, so a completed (or in-progress) run's monitoring
history can be re-examined anywhere after the fact -- e.g. to sanity-check a
proposed change to the rule constants against a real run's log before touching
heal.py, or to double-check exactly why a run stopped.

Usage:
  python -m hba.replay_probes results/probe_log_stage2.jsonl \\
      --budget-tokens 2.5e8 --warmup-tokens 5e6

  python -m hba.replay_probes results/probe_log_stage2.jsonl --phase stage2 \\
      --warmup-tokens 5e6   # --phase looks up the token budget from heal.PHASES;
                            # warmup-tokens still needs an explicit value since it
                            # depends on the run's realized tokens/step, which the
                            # log alone doesn't record
"""

import argparse

from .early_stop import EarlyStopEngine, load_probe_log, panel_mean


def replay(log_path, phase=None, budget_tokens=None, warmup_tokens=None):
    history = load_probe_log(log_path)
    if not history:
        print(f"no firings found in {log_path}")
        return None

    if budget_tokens is None and phase is not None:
        from .heal import PHASES  # lazy: keep this module torch-free unless --phase is used
        budget_tokens = PHASES[phase]["tokens"]

    if budget_tokens is None or warmup_tokens is None:
        raise SystemExit(
            "--budget-tokens and --warmup-tokens are required (or pass --phase for the "
            "token budget; --warmup-tokens still needs an explicit value -- it depends on "
            "the run's realized tokens/step, which the probe log alone doesn't record)"
        )

    engine = EarlyStopEngine(phase_budget_tokens=budget_tokens, warmup_end_tokens=warmup_tokens)
    print(f"replaying {len(history)} firings from {log_path}  "
          f"(ES-FLOOR = {engine.floor_tokens / 1e6:.1f}M tokens)")
    verdict = None
    for i in range(len(history)):
        verdict = engine.evaluate(history[: i + 1])
        f = history[i]
        pm = panel_mean(f)
        pm_s = f"{pm:.3f}" if pm is not None else "-"
        tag = f"*** {verdict.rule_fired} ***" if verdict.rule_fired else "-"
        print(f"[{i:3d}] step={f['step']:<8} tok={f['tokens'] / 1e6:9.1f}M "
              f"panel_mean={pm_s} val_loss={f.get('val_loss')} -> {tag}")
        if verdict.rule_fired:
            print(f"      details: {verdict.details}")
    return verdict


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("log", help="path to a probe_log*.jsonl file")
    ap.add_argument("--phase", choices=["stage1", "stage2", "stage3"], default=None,
                    help="look up the token budget from heal.PHASES (still need --warmup-tokens)")
    ap.add_argument("--budget-tokens", type=float, default=None)
    ap.add_argument("--warmup-tokens", type=float, default=None)
    args = ap.parse_args()
    replay(args.log, phase=args.phase, budget_tokens=args.budget_tokens,
          warmup_tokens=args.warmup_tokens)


if __name__ == "__main__":
    main()
