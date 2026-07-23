"""CPU-only tests for hba.early_stop (no GPU, no model, no downloads -- the
engine is pure numpy/math over synthesized firing dicts, exactly the "replay a
probe_log.jsonl anywhere" property the module is built for).

Every synthetic trace below sets `warmup_end_tokens=0` and picks
`phase_budget_tokens` so `0.4 * phase_budget_tokens` is at or below the first
firing's token count -- i.e. ES-FLOOR is already cleared at firing 0, so
"floor + K firings" in a test's comment/assert means "the Kth firing of the
trace", which is what the deliverable's test descriptions ("fires exactly at
floor + 3 firings") are stated in terms of.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hba.early_stop import (  # noqa: E402
    EarlyStopEngine,
    append_probe_log,
    check_tombstone,
    es_floor_tokens,
    load_probe_log,
    tombstone_path,
    truncate_probe_log,
    write_tombstone,
)


def _firing(i, step_stride, tok_stride, accs, n_trials, val_loss):
    return dict(step=i * step_stride, tokens=(i + 1) * tok_stride, accs=dict(accs),
               n_trials=dict(n_trials), val_loss=val_loss)


# --------------------------------------------------------------- ES-FLOOR ------
def test_es_floor_tokens_is_the_larger_of_the_two_quantities():
    # warmup ends at 10M tok; 100M-after-warmup = 110M; 40% of a 500M budget = 200M
    assert es_floor_tokens(10e6, 500e6) == 200e6
    # warmup ends at 300M tok; 100M-after-warmup = 400M; 40% of a 500M budget = 200M
    assert es_floor_tokens(300e6, 500e6) == 400e6


# ----------------------------------------------------------------- ES-2 --------
def test_es2_forgetting_abort_fires_within_3_firings_of_collapse():
    """A probe that forms (running_max >= 0.25) then collapses: pooled
    trailing-3-firings Wilson UCB should cross below max(0.5*running_max, 0.15)
    once collapsed firings dominate the trailing-3 pool, and ES-2 should fire at
    that point (not before, since ES-2 requires 3 pooled firings, and not much
    after -- see the per-firing UCB values computed in the deliverable's design
    notes: firing 3 is the first fully-collapsed firing, firing 5 is where the
    engine actually fires -- within the required 3-firing window)."""
    engine = EarlyStopEngine(phase_budget_tokens=1e6, warmup_end_tokens=0,
                             min_post_warmup_tokens=0)
    accs = [0.9, 0.9, 0.9, 0.05, 0.05, 0.05]   # forms, then collapses at index 3
    history = []
    fired_at = None
    for i, a in enumerate(accs):
        history.append(_firing(i, 200, 1e6, {"p": a}, {"p": 16}, val_loss=2.0))
        v = engine.evaluate(history)
        if v.rule_fired == "ES-2" and fired_at is None:
            fired_at = i
            assert v.details["probe"] == "p"
    assert fired_at is not None, "ES-2 should have fired on the collapse"
    assert 3 <= fired_at <= 5, f"expected the fire within 3 firings of the collapse start (index 3), got {fired_at}"


def test_es2_is_exempt_from_es_floor_but_es1_is_not():
    """ES-FLOOR guards the plateau stop (ES-1) against warmup/mid-schedule
    noise; the forgetting abort (ES-2) is a rollback trigger with its own noise
    guards (formed-probe requirement + pooled Wilson bound) and must fire even
    before the floor -- a collapse caught at 20% of budget saves the doomed
    remainder. Here the floor sits at 40e6 tokens and the whole trace stays
    below it: ES-2 must still fire; ES-1 must not (flat-plateau shape aside)."""
    engine = EarlyStopEngine(phase_budget_tokens=100e6, warmup_end_tokens=0,
                             min_post_warmup_tokens=100e6)  # floor = 100e6
    accs = [0.9, 0.9, 0.9, 0.05, 0.05, 0.05]
    history = []
    fired_at = None
    for i, a in enumerate(accs):
        history.append(_firing(i, 200, 1e6, {"p": a}, {"p": 16}, val_loss=2.0))
        assert history[-1]["tokens"] < engine.floor_tokens
        v = engine.evaluate(history)
        assert v.rule_fired != "ES-1"
        if v.rule_fired == "ES-2" and fired_at is None:
            fired_at = i
    assert fired_at is not None, "ES-2 must fire below ES-FLOOR"
    assert 3 <= fired_at <= 5


def test_es2_ignores_a_probe_that_never_formed():
    """running_max < 0.25 the whole trace -- 'a probe that never formed cannot
    collapse' -- so ES-2 must never fire on it even though its accuracy trends
    toward chance."""
    engine = EarlyStopEngine(phase_budget_tokens=1e6, warmup_end_tokens=0,
                             min_post_warmup_tokens=0)
    accs = [0.2, 0.15, 0.1, 0.05, 0.02, 0.01]
    history = []
    for i, a in enumerate(accs):
        history.append(_firing(i, 200, 1e6, {"p": a}, {"p": 16}, val_loss=2.0))
        v = engine.evaluate(history)
        assert v.rule_fired != "ES-2"


# ----------------------------------------------------------------- ES-1 --------
def test_es1_plateau_stop_fires_exactly_at_floor_plus_3_firings():
    """Flat val loss + flat panel + all per-probe slopes <= 0: ES-1 needs
    trailing+1 = 4 total firings before it can even be evaluated (comparing 'now'
    against 3 firings back), so the earliest possible fire is the 4th firing --
    i.e. floor + 3 firings past the one where the floor was first cleared."""
    engine = EarlyStopEngine(phase_budget_tokens=1e6, warmup_end_tokens=0,
                             min_post_warmup_tokens=0)
    history = []
    fired = []
    for i in range(6):
        history.append(_firing(i, 200, 1e6, {"a": 0.5, "b": 0.6}, {"a": 16, "b": 16}, val_loss=2.0))
        v = engine.evaluate(history)
        if v.rule_fired:
            fired.append((i, v.rule_fired))
    assert fired, "expected ES-1 to fire on a perfectly flat trace"
    first_i, first_rule = fired[0]
    assert first_rule == "ES-1"
    assert first_i == 3, f"expected the first fire at firing index 3 (floor + 3 firings), got {first_i}"


def test_es1_does_not_fire_while_one_probe_is_still_rising():
    """Same flat trace as above, except probe 'b' has a clean, unmistakable
    upward trend -- ES-1's per-probe slope check must block the stop even though
    val loss and the panel MEAN look identical to the plateau case."""
    engine = EarlyStopEngine(phase_budget_tokens=1e6, warmup_end_tokens=0,
                             min_post_warmup_tokens=0)
    b_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    history = []
    for i in range(6):
        history.append(_firing(i, 200, 1e6, {"a": 0.5, "b": b_vals[i]},
                               {"a": 16, "b": 16}, val_loss=2.0))
        v = engine.evaluate(history)
        assert v.rule_fired != "ES-1", f"ES-1 fired at firing {i} despite probe b still rising"


def test_healthy_improving_trace_nothing_fires_before_floor():
    """A steadily improving val loss + steadily improving (still-rising) panel:
    nothing should fire before ES-FLOOR, and ES-1 in particular should never
    fire (a genuinely improving run should never be told it has plateaued)."""
    engine = EarlyStopEngine(phase_budget_tokens=10e6, warmup_end_tokens=0,
                             min_post_warmup_tokens=2e6)
    assert engine.floor_tokens == 4e6   # max(0 + 2e6, 0.4 * 10e6)
    history = []
    for i in range(20):
        tokens = (i + 1) * 0.5e6
        val_loss = 3.0 - 0.05 * i
        acc = min(0.9, 0.2 + 0.03 * i)
        history.append(dict(step=i * 200, tokens=tokens, accs={"p": acc}, n_trials={"p": 16},
                            val_loss=val_loss))
        v = engine.evaluate(history)
        if tokens < engine.floor_tokens:
            assert v.rule_fired is None, f"firing {i} (tok={tokens}) fired before ES-FLOOR"
        assert v.rule_fired != "ES-1", f"ES-1 fired at firing {i} on an improving trace"


# -------------------------------------------------------------- tombstone ------
def test_tombstone_write_and_check_blocks_a_start(tmp_path):
    results_dir = str(tmp_path)
    assert check_tombstone(results_dir) is None, "no tombstone should exist yet"

    write_tombstone(results_dir, probe="induction_far", step=1000, tokens=123_456_789.0,
                    rolled_back_to=800, phase="stage2")

    tomb = check_tombstone(results_dir)
    assert tomb is not None, "heal.py's start/resume guard checks exactly this return value"
    assert tomb["probe"] == "induction_far"
    assert tomb["step"] == 1000
    assert tomb["tokens"] == 123_456_789.0
    assert tomb["rolled_back_to"] == 800
    assert tomb["phase"] == "stage2"
    assert os.path.exists(tombstone_path(results_dir))


# --------------------------------------------------- probe-log poisoning (B3) --
def test_load_probe_log_dedupes_by_step_keeping_last_occurrence(tmp_path):
    """A duplicate line for a step already in the file (e.g. a killed-mid-write
    re-fire) must not double-count: load_probe_log must keep only the LAST
    write for that step, not both (which would double-weight that step's
    trials in ES-1/ES-2's pooled estimates)."""
    path = str(tmp_path / "probe_log.jsonl")
    append_probe_log(path, _firing(0, 200, 1e6, {"p": 0.5}, {"p": 16}, val_loss=2.0))
    append_probe_log(path, _firing(1, 200, 1e6, {"p": 0.6}, {"p": 16}, val_loss=1.9))
    # a second, DIFFERENT-valued write for step 200 (the same step as firing 1)
    with open(path, "a") as f:
        f.write(json.dumps(dict(step=200, tokens=99e6, accs={"p": 0.99}, n_trials={"p": 16},
                                val_loss=0.1)) + "\n")
    hist = load_probe_log(path)
    assert [f["step"] for f in hist] == [0, 200], "must dedupe, not append a third entry"
    assert hist[1]["accs"]["p"] == 0.99, "the LAST write for a duplicated step must win"


def test_load_probe_log_tolerates_torn_trailing_line(tmp_path):
    """A process killed mid-append can leave a truncated (unparseable) LAST
    line; that must be skipped, not crash the load. A torn line anywhere else
    in the file is real corruption and must still raise."""
    path = str(tmp_path / "probe_log.jsonl")
    append_probe_log(path, _firing(0, 200, 1e6, {"p": 0.5}, {"p": 16}, val_loss=2.0))
    with open(path, "a") as f:
        f.write('{"step": 200, "tokens": 2000000.0, "accs": {"p": 0.6}, "n_trial')  # torn, no newline
    hist = load_probe_log(path)
    assert [f["step"] for f in hist] == [0], "the torn trailing line must be skipped, not crash"

    # a torn line NOT at the end must still raise
    path2 = str(tmp_path / "probe_log2.jsonl")
    with open(path2, "w") as f:
        f.write('{"step": 0, "tokens": 1000000.0, "accs": {"p": 0.5}, "n_trial\n')  # torn, mid-file
        f.write('{"step": 200, "tokens": 2000000.0, "accs": {"p": 0.6}, "n_trials": {"p": 16}, '
                '"val_loss": 1.9}\n')
    with pytest.raises(json.JSONDecodeError):
        load_probe_log(path2)


def test_truncate_probe_log_resume_then_reappend_prevents_es2_repoisoning(tmp_path):
    """Simulate the ES-2 rollback -> operator-clears-tombstone -> resume flow
    end to end at the log-file level: (1) a probe forms then collapses (ES-2
    would fire and roll back to the last healthy checkpoint); (2) resume from
    that rolled-back step calls truncate_probe_log (what heal.py now does),
    which must both return the filtered pre-collapse history AND rewrite the
    file on disk to match EXACTLY -- the stale collapsed firings must be gone,
    not just filtered in memory; (3) training continues and re-appends a
    HEALTHY continuation at step numbers that do NOT coincide with the old
    (discarded) collapsed firings' step numbers (e.g. a different
    --probe-every cadence post-resume) -- the case step-based dedup alone
    cannot fix, since the stale and new firings never collide on the same
    step; only the on-disk truncation removes them. Assert a LATER full reload
    of the file (as another resume, or a replay, would do) never lets ES-2
    spuriously re-fire on the healthy continuation."""
    path = str(tmp_path / "probe_log_stage2.jsonl")

    # phase 1: forms (0.9) then collapses (0.05) -- mirrors
    # test_es2_forgetting_abort_fires_within_3_firings_of_collapse
    collapse_accs = [0.9, 0.9, 0.9, 0.05, 0.05, 0.05]
    for i, a in enumerate(collapse_accs):
        append_probe_log(path, _firing(i, 200, 1e6, {"p": a}, {"p": 16}, val_loss=2.0))

    # rollback point: right after the last healthy firing (index 2, step 400)
    step0 = 200 * 2 + 1
    truncated = truncate_probe_log(path, max_step=step0 - 1)
    assert [f["step"] for f in truncated] == [0, 200, 400]
    assert all(f["accs"]["p"] == 0.9 for f in truncated), "only pre-collapse firings should survive"

    # the file on disk must now match EXACTLY -- steps 600/800/1000 (collapsed)
    # must be GONE, not merely filtered out of some in-memory list
    on_disk = load_probe_log(path)
    assert [f["step"] for f in on_disk] == [0, 200, 400], \
        "stale collapsed firings must be truncated from disk, not just filtered in memory"

    # phase 2: healthy continuation, re-appended at step numbers that do NOT
    # collide with the purged 600/800/1000 firings (a different post-resume cadence)
    for j, a in enumerate([0.85, 0.88, 0.90, 0.91, 0.92]):
        step = step0 + j * 150
        append_probe_log(path, dict(step=step, tokens=float((step + 1) * 1e6),
                                    accs={"p": a}, n_trials={"p": 16}, val_loss=1.5))

    # a LATER resume (or an offline replay) loads the WHOLE file from scratch
    # -- exactly the read path that would repoison ES-2 without the truncation fix
    full_history = load_probe_log(path)
    engine = EarlyStopEngine(phase_budget_tokens=1e6, warmup_end_tokens=0, min_post_warmup_tokens=0)
    for i in range(len(full_history)):
        v = engine.evaluate(full_history[: i + 1])
        assert v.rule_fired != "ES-2", (
            f"ES-2 spuriously re-fired at firing {i} on a healthy continuation "
            "after the stale collapsed firings were truncated from disk"
        )


# ------------------------------------------------- ES-1 dead-probe / NaN (S2) --
def test_es1_all_nan_probe_does_not_veto_plateau_stop():
    """A probe that is enabled but has no data (e.g. its shard is missing) so
    it returns NaN on EVERY firing must not permanently block ES-1. Before the
    fix, cond_c's <3-non-NaN-points rule vetoed the stop forever (silently,
    every firing) for such a probe; after the fix, a probe whose ENTIRE
    history is NaN is recognized as dead and excluded from cond_c, so a flat,
    plateaued trace with one real probe can still stop on schedule (same
    firing index as the flat-trace baseline in
    test_es1_plateau_stop_fires_exactly_at_floor_plus_3_firings)."""
    engine = EarlyStopEngine(phase_budget_tokens=1e6, warmup_end_tokens=0, min_post_warmup_tokens=0)
    history = []
    fired = []
    for i in range(6):
        history.append(_firing(i, 200, 1e6, {"a": 0.5, "dead": float("nan")},
                               {"a": 16, "dead": 16}, val_loss=2.0))
        v = engine.evaluate(history)
        if v.rule_fired:
            fired.append((i, v.rule_fired))
    assert fired, "ES-1 should still fire despite the always-NaN 'dead' probe"
    assert fired[0] == (3, "ES-1")


def test_es1_partial_nan_probe_still_blocks_when_too_few_recent_points():
    """Contrast with the all-NaN case above: a probe with REAL history (some
    non-NaN values exist) but fewer than 3 non-NaN points in the CURRENT
    trailing window is not 'dead' and must still conservatively block ES-1 --
    only a probe with NO non-NaN values anywhere is treated as dead."""
    engine = EarlyStopEngine(phase_budget_tokens=1e6, warmup_end_tokens=0, min_post_warmup_tokens=0,
                             slope_window=2)   # trailing window of 2 < the 3 points slope needs
    history = []
    for i in range(6):
        history.append(_firing(i, 200, 1e6, {"a": 0.5, "sparse": 0.5},
                               {"a": 16, "sparse": 16}, val_loss=2.0))
        v = engine.evaluate(history)
    # 'sparse' has REAL history but slope_window=2 means the trailing window
    # never has 3 points -> cond_c must stay blocked (not skipped as dead)
    assert v.rule_fired != "ES-1"


def test_ema_series_nan_input_does_not_poison_running_state():
    from hba.early_stop import ema_series
    vals = [2.0, 2.0, float("nan"), 2.0, 2.0]
    out = ema_series(vals, alpha=0.5)
    assert out[2] == 2.0, "a NaN input must not become the new EMA state -- forward-fill instead"
    assert out[-1] == 2.0, "the EMA must recover cleanly once real values resume"
    assert not any(isinstance(v, float) and v != v for v in out), \
        "no NaN should ever leak into the returned series once a real value has been seen"


def test_es1_nan_val_loss_does_not_poison_ema_and_fails_cond_a_only_while_nan():
    """A NaN val_loss firing (e.g. data/val_books.bin briefly unavailable) must
    (a) FAIL cond_a outright while it is the current firing -- not silently
    pass via a 0/0-shaped comparison -- and (b) not permanently poison the EMA:
    once real val_loss data resumes, ES-1 must be able to fire again."""
    engine = EarlyStopEngine(phase_budget_tokens=1e6, warmup_end_tokens=0, min_post_warmup_tokens=0)
    history = []
    for i in range(4):
        history.append(_firing(i, 200, 1e6, {"a": 0.5}, {"a": 16}, val_loss=2.0))
    history.append(_firing(4, 200, 1e6, {"a": 0.5}, {"a": 16}, val_loss=float("nan")))
    v = engine.evaluate(history)
    assert v.rule_fired != "ES-1", "a NaN val_loss firing must not let ES-1 fire via cond_a"
    assert v.details.get("cond_a_flat_val_loss") is False

    fired_after_recovery = False
    for i in range(5, 9):
        history.append(_firing(i, 200, 1e6, {"a": 0.5}, {"a": 16}, val_loss=2.0))
        v = engine.evaluate(history)
        if v.rule_fired == "ES-1":
            fired_after_recovery = True
    assert fired_after_recovery, "ES-1 must be able to fire again once val_loss data recovers"
