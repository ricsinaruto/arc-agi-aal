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
