import ast
import json
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts/posttrain/modal_rl1.py"


def load():
    fn = next(n for n in ast.parse(SRC.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "preempt_retry")
    ns = {"time": time, "json": json}
    exec(compile(ast.Module([fn], []), str(SRC), "exec"), ns)
    return ns["preempt_retry"]


def test_preempt_retry(tmp_path):
    pr, d = load(), tmp_path / "run"
    assert pr(d) == {}
    (d / "train").mkdir()
    (d / "train" / "routing.pt").write_text("r")
    for step, done in ((6, True), (12, True), (14, False)):
        (d / "train" / f"ckpt-{step}").mkdir()
        if done:
            (d / "train" / f"ckpt-{step}" / "progress.json").write_text(json.dumps({"student": "/base"}))
    r = pr(d)
    assert r["student"] == "/base" and r["step0"] == 12 and r["routing"].endswith("train/routing.pt") and "teacher_dir" not in r
    time.sleep(1)
    (d / "train").mkdir()  # a second attempt preempted before its first checkpoint
    assert pr(d)["resume"] == r["resume"]
    time.sleep(1)
    (d / "teacher").mkdir()
    (d / "teacher" / "index.json").write_text("{}")
    assert pr(d)["teacher_dir"].endswith("/teacher")
