"""Stream rounds (c3, 23 Oct): auto --total-steps gave 1 step; the round needs an explicit count (warm-up then a
fixed lr, no decay). Stream chains > 1 are disabled (MultiForward wrong on GPU)."""
import ast
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")
SRC = Path(__file__).resolve().parents[1] / "scripts" / "posttrain"
sys.path.insert(0, str(SRC))
import sys as _sys  # trunk 24 Oct: PTH's trainer copies (scripts/posttrain/pth) shadow RL's for this module
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "pth"))
for _m in ("rl1_trainer", "flash_qlora_job", "rl1_sparse"):
    _sys.modules.pop(_m, None)
import heal_kd  # noqa: E402


def test_c3_settings_refused_c4_fixed_lr_passes(monkeypatch):
    c3 = {"HEAL_SCHEDULE_STEPS": "500", "HEAL_WSD_DECAY": "0", "HEAL_REWARM": "2"}
    assert "explicit --total-steps" in heal_kd.stream_schedule_problem(0, "wsd", c3)
    assert heal_kd.stream_schedule_problem(500, "wsd", c3) is None  # warm-up then fixed lr, no decay
    assert "decay must complete" in heal_kd.stream_schedule_problem(60, "wsd", {**c3, "HEAL_DECAY_FROM": "45",
                                                                              "HEAL_DECAY_STEPS": "11"})
    assert "would not complete" in heal_kd.stream_schedule_problem(56, "wsd", {"HEAL_SCHEDULE_STEPS": "500"})
    for k, v in c3.items():
        monkeypatch.setenv(k, v)
    a = types.SimpleNamespace(schedule="wsd", lr=3.5e-6, warmup=0.0, total_steps=500, step0=0)
    lrs = [heal_kd.lr_at(s, a) for s in range(500)]
    assert lrs[0] < lrs[1] < lrs[2] == lrs[499] == 3.5e-6  # warm-up, then fixed to the last step


def test_stream_round_checks_the_schedule_and_preflight_checks_the_launch():
    src = (SRC / "heal_kd.py").read_text()
    i = src.index("files, args.allow_unfinished_schedule = [], True")
    assert "stream_schedule_problem(args.total_steps, args.schedule)" in src[i:i + 300]
    pf = (SRC / "heal_preflight.py").read_text()
    assert 'check("schedule", why is None' in pf
    assert 'senv.update(HEAL_SCHEDULE_STEPS="3")' in pf  # the tiny rounds' own count
    assert '("HEAL_START_READS", "start-reads.json"), ("HEAL_HELDOUT2_BEH", "heldout2-beh.pt")' in pf  # not the run's


def test_stream_chains_above_one_refused():
    tree = ast.parse((SRC / "modal_rl1.py").read_text())
    body = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "stream_teacher_env")
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "CHAINS_OFF" for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), "modal_rl1.py", "exec"), ns)
    with pytest.raises(ValueError, match="disabled"):
        ns["stream_teacher_env"](1, "RL1_EXPERTS_FP8=1,RL1_STREAM_CHAINS=2")
    assert ns["stream_teacher_env"](1, "RL1_EXPERTS_FP8=1,RL1_STREAM_CHAINS=1") == {"RL1_EXPERTS_FP8": "1",
                                                                                    "RL1_STREAM_CHAINS": "1"}
    src = (SRC / "rl1_trainer.py").read_text()
    i = src.index('chains = int(cmd.get("chains") or os.environ.get("RL1_STREAM_CHAINS") or 1)')
    assert "raise RuntimeError" in src[i:i + 300]


def test_plan_dry_run_of_a_stream_round(monkeypatch):
    """c4 (23 Oct): the CPU plan applied the staged-fit rule to a stream round and refused --total-steps 500."""
    monkeypatch.setenv("HEAL_SCHEDULE_STEPS", "500")
    monkeypatch.setenv("HEAL_WSD_DECAY", "0")
    a = types.SimpleNamespace(total_steps=500, step0=0, allow_unfinished_schedule=False, schedule="wsd")
    with pytest.raises(SystemExit, match="would not complete"):
        heal_kd.schedule_steps(a, {}, 16)  # the c4 refusal
    a.allow_unfinished_schedule = True
    assert heal_kd.schedule_steps(a, {}, 16) == 500
    src = (SRC / "heal_kd.py").read_text()
    i = src.index("def run_plan(")
    assert "stream_schedule_problem(args.total_steps, args.schedule)" in src[i:src.index("\ndef ", i + 10)]
    m = (SRC / "modal_rl1.py").read_text()
    j = m.index("def plan_round") if "def plan_round" in m else m.index("The mandatory CPU dry run")
    assert 'allow_unfinished=kw.get("allow_unfinished") or bool(kw.get("stream_play"))' in m[j:j + 2500]


def test_mid_read_general_probe():
    src = (SRC / "heal_kd.py").read_text()
    i = src.index("    def mid_read(at):")
    body = src[i:src.index("event(log, event=\"mid_read\"", i)]
    assert 'os.environ.get("HEAL_MID_GENERAL") == "1"' in body and 'got["g_kd"] = probe_kl(' in body


def test_inert_check_never_on_a_resumed_round():
    src = (SRC / "heal_kd.py").read_text()
    i = src.index('inert_check = os.environ.get("HEAL_INERT_CHECK") == "1"')
    assert "if inert_check and args.step0:" in src[i:i + 300] and "inert_check = False" in src[i:i + 400]
