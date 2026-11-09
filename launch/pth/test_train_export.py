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
    for held, missing in ((True, []), (False, ["train.kd_heldout_step"])):
        ev = tmp_path / f"events-{held}.jsonl"
        with open(ev, "w") as f:
            for s in (1, 2, 3):
                f.write(json.dumps({"t": 100.0 * s, "event": "step", "step": s, "kd": 0.5, "kd_reply": 0.6, "grad_norm": 1.0,
                                    **({"heldout_kd": 0.7, "heldout_kd_rank": [0.7, 0.7]} if held else {})}) + "\n")
        ok, info = ns["_exporter_keys_check"](ev, tmp_path / f"mon-{held}", 2)
        assert ok == (not missing) and info["missing"] == missing, info
