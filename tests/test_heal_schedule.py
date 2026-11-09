import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "scripts/posttrain/heal_kd.py"
NAMES = {"warmup_steps", "wsd_lr", "cosine_lr", "lr_at"}


def load():
    fns = [n for n in ast.parse(SRC.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    ns = {"math": math, "os": os}
    exec(compile(ast.Module(fns, []), str(SRC), "exec"), ns)
    return ns["lr_at"]


def test_wsd_branch_and_rewarm(monkeypatch):
    lr_at = load()
    args = SimpleNamespace(schedule="wsd", total_steps=40, lr=1e-5, warmup=0.05, step0=0)
    monkeypatch.setenv("HEAL_SCHEDULE_STEPS", "400")
    monkeypatch.setenv("HEAL_WSD_DECAY", "0")
    assert lr_at(0, args) < lr_at(19, args) < lr_at(20, args) == 1e-5  # 20 warm-up steps over the whole run, then stable
    assert lr_at(399, args) == 1e-5  # no main-line decay
    monkeypatch.setenv("HEAL_DECAY_FROM", "90")
    monkeypatch.setenv("HEAL_DECAY_STEPS", "10")
    assert lr_at(89, args) == 1e-5 and lr_at(90, args) == 1e-5 and abs(lr_at(99, args) - 1e-6) < 1e-15
    assert lr_at(100, args) == 0.0
    monkeypatch.delenv("HEAL_DECAY_FROM")
    monkeypatch.delenv("HEAL_DECAY_STEPS")
    monkeypatch.setenv("HEAL_REWARM", "2")
    args.step0 = 50
    assert lr_at(49, args) == 1e-5  # the resume check's lr(step0 - 1) is untouched
    assert abs(lr_at(50, args) - 1e-5 / 3) < 1e-15 and abs(lr_at(51, args) - 2e-5 / 3) < 1e-15
    assert lr_at(52, args) == 1e-5


def test_defaults_unchanged(monkeypatch):
    lr_at = load()
    for k in ("HEAL_SCHEDULE_STEPS", "HEAL_WSD_DECAY", "HEAL_DECAY_FROM", "HEAL_DECAY_STEPS", "HEAL_REWARM"):
        monkeypatch.delenv(k, raising=False)
    args = SimpleNamespace(schedule="wsd", total_steps=20, lr=1e-5, warmup=0.1, step0=0)
    assert lr_at(19, args) == 1e-5 * 1 / 4 and lr_at(16, args) == 1e-5  # decay over the last 4 of 20 (0.2)
    args.schedule = "cosine"
    assert abs(lr_at(19, args) - 1e-6 - 9e-6 * 0.5 * (1 + math.cos(math.pi * 17 / 18))) < 1e-15
