"""Exporter schema: every row's t is an epoch near now (no double-added epoch), gpu rows <= 120 s apart.
python3 -m pytest -q test_train_export.py"""
import json
import time

import train_export as te


def test_rows_epoch_sane_and_gpu_every_minute(tmp_path, monkeypatch):
    now = time.time()
    run = "sft-heal-round-x"
    d = tmp_path / run
    d.mkdir()
    start = now - 1800
    with open(d / "gpu.jsonl", "w") as f:
        for i in range(180):  # every 10 s, the round's sampler
            f.write(json.dumps({"t": i * 10, "epoch": start + i * 10, "util": [90, 91, 10, 0], "mem_gib": [200, 200, 130, 130]}) + "\n")
    with open(d / "teacher-stream.log", "w") as f:
        f.write(json.dumps({"t": 230.4, "event": "stream_piece", "tok_per_s": 900.0}) + "\n")  # relative t
        f.write(json.dumps({"t": start + 900, "event": "stream_teacher", "pending_tokens": 776, "open_chains": 1}) + "\n")  # epoch t
    monkeypatch.setattr(te, "MON", tmp_path)
    monkeypatch.setattr(te, "OUT", tmp_path / "out")
    (tmp_path / "out").mkdir()
    rows = te.lines(run)
    assert rows and all(1.7e9 < r["t"] < now + 60 for r in rows), [r["t"] for r in rows if not 1.7e9 < r["t"] < now + 60]
    gpu = sorted(r["t"] for r in rows if any(k.startswith("infra.gpu_util") for k in r["m"]))
    assert len(gpu) >= 25 and max(b - a for a, b in zip(gpu, gpu[1:])) <= 120


def test_checkpoint_resume_keeps_its_own_step_numbers(tmp_path, monkeypatch):
    # c7g (24 Oct): resumed c7c ckpt-6 with STEP0=6, its events say steps 7..; c7c itself continues a parent (+100)
    monkeypatch.setattr(te, "MON", tmp_path)
    monkeypatch.setattr(te, "OUT", tmp_path / "out")
    (tmp_path / "out").mkdir()
    parents = {"child": "par", "fresh": "par"}
    monkeypatch.setattr(te, "parent_of", lambda r: parents.get(r))
    for run, steps in (("par", range(1, 9)), ("child", range(7, 10)), ("fresh", range(1, 3))):
        (tmp_path / run).mkdir()
        (tmp_path / run / "train_events.jsonl").write_text("".join(json.dumps({"event": "step", "step": s}) + "\n" for s in steps))
    (tmp_path / "out/par.jsonl").write_text("".join(json.dumps({"t": 1.79e9 + s, "m": {"train.step": s + 100}}) + "\n" for s in range(1, 9)))
    assert te._lineage("child")[1] == 100  # parent's own offset only: child step 7 -> 107
    assert te._lineage("fresh")[1] == 108  # a fresh start (step 1) counts on from the parent's last step


def test_step_heldout_curve_exported(tmp_path, monkeypatch):
    monkeypatch.setattr(te, "MON", tmp_path)
    monkeypatch.setattr(te, "OUT", tmp_path / "out")
    monkeypatch.setattr(te, "parent_of", lambda r: None)
    (tmp_path / "out").mkdir()
    (tmp_path / "r").mkdir()
    now = time.time()
    (tmp_path / "r" / "train_events.jsonl").write_text(json.dumps(
        {"t": now, "event": "step", "step": 1, "heldout_kd": 0.18, "heldout_kd_rank": [0.17, 0.2, 0.17]}) + "\n")
    m = {k: v for r in te.lines("r") for k, v in r["m"].items()}
    assert m["train.kd_heldout_step"] == 0.18 and m["eval.step_heldout_kd_r1"] == 0.2


def test_preflight_exporter_keys_check(tmp_path, monkeypatch):
    """heal_preflight._exporter_keys_check on round-A-style step events: every REQUIRED key, or the missing ones."""
    import ast
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "scripts" / "posttrain" / "heal_preflight.py"
    fn = [f for f in ast.parse(src.read_text()).body if isinstance(f, ast.FunctionDef) and f.name == "_exporter_keys_check"]
    ns = {"json": json}
    exec(compile(ast.Module(fn, []), "heal_preflight", "exec"), ns)
    sys.modules.setdefault("train_export", te)
    monkeypatch.setattr(te, "MON", te.MON)  # the check repoints it
    ev0 = tmp_path / "events-dead.jsonl"  # c7m: the held-out key present but 0.0 on every rank fails the check
    with open(ev0, "w") as f:
        for s in (1, 2, 3):
            f.write(json.dumps({"t": 100.0 * s, "event": "step", "step": s, "kd": 0.5, "kd_reply": 0.6, "grad_norm": 1.0,
                                "heldout_kd": 0.0, "heldout_kd_rank": [0.0, 0.0]}) + "\n")
    ok, info = ns["_exporter_keys_check"](ev0, tmp_path / "mon-dead", 2)
    assert not ok and info["dead"] == ["train.kd_heldout_step"] and info["missing"] == [], info
    for held, missing in ((True, []), (False, ["train.kd_heldout_step"])):
        ev = tmp_path / f"events-{held}.jsonl"
        with open(ev, "w") as f:
            for s in (1, 2, 3):
                f.write(json.dumps({"t": 100.0 * s, "event": "step", "step": s, "kd": 0.5, "kd_reply": 0.6, "grad_norm": 1.0,
                                    **({"heldout_kd": 0.7, "heldout_kd_rank": [0.7, 0.7]} if held else {})}) + "\n")
        ok, info = ns["_exporter_keys_check"](ev, tmp_path / f"mon-{held}", 2)
        assert ok == (not missing) and info["missing"] == missing, info


def test_dead_keys_and_first_step_watch_rule():
    """c7m (24 Oct): a required training metric exactly 0 (or NaN) on every row/rank is dead, not merely present."""
    rows = [{"m": {"train.kd": 0.5, "train.kd_heldout_step": 0.0, "train.grad_norm": float("nan")}},
            {"m": {"train.kd": 0.4, "train.kd_heldout_step": 0.0, "train.grad_norm": float("nan")}}]
    assert te.dead_keys(rows) == ["train.kd_heldout_step", "train.grad_norm"]
    assert te.dead_keys([{"m": {"train.kd_heldout_step": 0.0}}, {"m": {"train.kd_heldout_step": 0.21}}]) == []
    import opd_watch
    c7m = {"event": "step", "step": 1, "kd": 0.353, "kd_reply": 0.285, "grad_norm": 2.41, "heldout_kd": 0.0,
           "heldout_kd_rank": [0.0, 0.0, 0.0]}  # the c7m step-1 event: fires
    assert "heldout_kd" in opd_watch.dead_metrics([c7m]) and "heldout_kd_rank" in opd_watch.dead_metrics([c7m])
    healthy = {**c7m, "heldout_kd": 0.231, "heldout_kd_rank": [0.22, 0.24, 0.23]}
    assert opd_watch.dead_metrics([healthy]) is None and opd_watch.dead_metrics([]) is None
    c7h = {"event": "step", "step": 10, "kd": 0.30, "kd_reply": 0.26, "grad_norm": 2.16}  # no held-out keys: quiet
    assert opd_watch.dead_metrics([c7h]) is None


def test_dead_metric_flags_unmasked_think_rows():
    import opd_watch

    e = {"step": 1, "kd": .3, "kd_reply": .3, "grad_norm": 2.0, "heldout_kd": .25}
    assert opd_watch.dead_metrics([e], {}) is None
    assert "think_eos_masked" in opd_watch.dead_metrics([e], {"THINK_MASK": "1"})
    assert opd_watch.dead_metrics([dict(e, think_eos_masked=812)], {"THINK_MASK": "1"}) is None


def test_held_rule_needs_three_steps_and_every_rank():
    """c7n: held-out mean > 0.2582 on 3 consecutive steps with every rank over its baseline."""
    import opd_watch

    env = {"HELD_BAR": "0.2582", "HELD_RANK_BASE": "0.28099,0.21084,0.2459"}
    st = lambda i, m, r: {"step": i, "heldout_kd": m, "heldout_kd_rank": r}  # noqa: E731
    c7n = [st(1, .2456, [.2819, .2098, .2451]), st(2, .2436, [.2793, .2077, .2437]), st(3, .2485, [.2817, .2150, .2489]),
           st(4, .2612, [.2957, .2281, .2599])]
    assert opd_watch.held_rule(c7n, env) is None and opd_watch.held_rule(c7n, {}) is None
    assert "steps [4, 5, 6]" in opd_watch.held_rule(c7n + [st(5, .262, [.29, .22, .26]), st(6, .27, [.30, .23, .27])], env)
    assert opd_watch.held_rule(c7n + [st(5, .262, [.29, .20, .26]), st(6, .27, [.30, .23, .27])], env) is None
    auto = {"HELD_BAR": "auto"}  # c7n's steps 1-3 give its registered bar 0.2582 and fire on the same steps
    assert opd_watch.held_rule(c7n, auto) is None
    why = opd_watch.held_rule(c7n + [st(5, .262, [.29, .22, .26]), st(6, .27, [.30, .23, .27])], auto)
    assert "> 0.2582" in why and "steps [4, 5, 6]" in why


def test_teacher_vram_alert_fires_on_growth_only():
    """c7o: the teacher card past 235 GiB and +10 GiB in 10 min (after 20 min) alerts; a plateau stays quiet."""
    import opd_watch

    t0 = 1000.0
    grow = [{"epoch": t0 + 60 * i, "mem_gib": [200, 200, 200, 140 + 4 * i]} for i in range(32)]
    flat = [{"epoch": t0 + 60 * i, "mem_gib": [200, 200, 200, min(140 + 10 * i, 246.5)]} for i in range(45)]
    assert opd_watch.teacher_vram(grow, {}) and opd_watch.teacher_vram(flat, {}) is None
    assert opd_watch.teacher_vram(grow[:10], {}) is None  # warm-up


def test_data_starve_signals():
    """c7q: (b) rank 0 idle 5 min with no step alerts; (a) a 30% wait share over 15 min alerts; a healthy stretch
    (short waits, rank 0 busy) stays quiet."""
    import opd_watch

    t0 = 10000.0
    steps = [{"step": i, "t": t0 + 220 * i, "stream_wait_s": 2.0 * i} for i in range(1, 6)]
    busy = [{"epoch": t0 + 10 * k, "util": [95, 100, 100, 90]} for k in range(130)]
    assert opd_watch.data_starve(steps, busy, t0 + 1100, {}) is None
    idle = busy[:111] + [{"epoch": t0 + 1110 + 10 * k, "util": [0, 100, 100, 90]} for k in range(40)]
    assert "stream wait" in opd_watch.data_starve(steps, idle, t0 + 1110 + 390, {})
    slow = [{"step": i, "t": t0 + 300 * i, "stream_wait_s": 90.0 * i} for i in range(1, 5)]
    assert "data wait" in opd_watch.data_starve(slow, busy, t0 + 1200, {})


def test_play_rule_fires_on_two_degraded_windows_only():
    import opd_watch
    t0 = 1_000_000.0

    def window(k, n, tool, loop):  # n replies in window k: the first tool*n call a tool, the first loop*n loop
        return [(t0 + 600 * k + i * 500 / n, i < tool * n, i < loop * n) for i in range(n)]
    d = window(0, 200, .97, .002) + window(1, 200, .92, .004)  # v32's D windows: quiet
    assert opd_watch.play_rule(opd_watch.play_windows(d, t0 + 1200)) is None
    one = d + window(2, 200, .57, .01)  # one bad window: quiet
    assert opd_watch.play_rule(opd_watch.play_windows(one, t0 + 1800)) is None
    two = one + window(3, 200, .62, .003)  # v32's c7n v4-v7 windows: fires
    assert "tool 0.57" in opd_watch.play_rule(opd_watch.play_windows(two, t0 + 2400))
    assert opd_watch.play_rule(opd_watch.play_windows(two, t0 + 2300)) is None  # window 3 not complete yet
    loops = d + window(2, 200, .95, .05) + window(3, 200, .95, .04)  # loop branch
    assert "loop 5.0%" in opd_watch.play_rule(opd_watch.play_windows(loops, t0 + 2400))
    thin = d + window(2, 50, .5, 0) + window(3, 50, .5, 0)  # < 100 replies: windows skipped
    assert opd_watch.play_rule(opd_watch.play_windows(thin, t0 + 2400)) is None
    rec = {"kind": "model", "purpose": "gameplay", "time": 5.0, "message": {"reasoning": "a b " * 400, "tool_calls": []}}
    assert opd_watch.reply_row(rec) == (5.0, False, True, False) and opd_watch.reply_row({"kind": "observation"}) is None
    cut = dict(rec, finish_reason="length", message={"reasoning": "x y z", "content": ""})
    assert opd_watch.reply_row(cut)[3] and not opd_watch.reply_row(dict(cut, finish_reason="stop"))[3]
    cap = lambda k, share: [(t0 + 600 * k + i * 2.5, True, False, i < share * 200) for i in range(200)]
    assert opd_watch.play_rule(opd_watch.play_windows(cap(0, .006) + cap(1, .012), t0 + 1200)) is None  # D's windows
    assert "cap-in-think 4.0%" in opd_watch.play_rule(opd_watch.play_windows(cap(0, .04) + cap(1, .04), t0 + 1200))
