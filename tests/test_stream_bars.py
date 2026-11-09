"""heal_stream.stream_bars on c6's real events (23 Oct): "seconds" and stream_wait_s are cumulative; c6 was stopped at
step 3 by a bar that read the elapsed 1,370 s as step compute. Per-step: step 2 = 354 s - 6.4 s wait - 82.4 s checkpoint
blocking + 6.2 s gather = 271 s; step 3 = 101.5 - 0.2 + 6.6 = 108 s."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import sys as _sys  # trunk 24 Oct: PTH's trainer copies (scripts/posttrain/pth) shadow RL's for this module
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "pth"))
for _m in ("rl1_trainer", "flash_qlora_job", "rl1_sparse"):
    _sys.modules.pop(_m, None)
import heal_stream  # noqa: E402

C6 = [{"t": 1792787809.2, "event": "step", "step": 1, "seconds": 908.3, "stream_wait_s": 157.3}, {"t": 1792787816.9, "event": "delta_export", "step": 1, "gather_seconds": 7.75}, {"t": 1792788084.7, "event": "checkpoint", "step": 1, "seconds": 267.8, "blocking_seconds": 82.4}, {"t": 1792788163.1, "event": "step", "step": 2, "seconds": 1262.3, "stream_wait_s": 163.7}, {"t": 1792788169.3, "event": "delta_export", "step": 2, "gather_seconds": 6.18}, {"t": 1792788264.6, "event": "step", "step": 3, "seconds": 1363.8, "stream_wait_s": 163.9}, {"t": 1792788271.2, "event": "delta_export", "step": 3, "gather_seconds": 6.6}]
BARS = {"step_seconds": 300.0, "teacher_tok_per_s": 2960.0, "teacher_after": 1000000}


def write(tmp, rows, tok=10974.5):
    (tmp / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp / "status.json").write_text(json.dumps({"tok_per_s": tok, "tokens": 10202982}))
    return tmp / "events.jsonl", tmp / "status.json"


def test_c6_passes_on_per_step_compute(tmp_path):
    verdict, read = heal_stream.stream_bars(*write(tmp_path, C6), BARS)
    assert verdict == "pass", (verdict, read)
    assert 265 < read["steps"][1] < 275 and 100 < read["steps"][2] < 115


def test_a_slow_step_still_fails(tmp_path):
    rows = [dict(r) for r in C6]
    for r in rows:
        if r["event"] == "step" and r["step"] == 3:
            r["seconds"] += 400  # step 3 really 500 s
    verdict, _ = heal_stream.stream_bars(*write(tmp_path, rows), BARS)
    assert verdict.startswith("fail: step compute")
