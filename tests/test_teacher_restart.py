"""modal_rl1.teacher_with_restarts (coloc-v5 IMA, 23 Oct): a dead stream teacher restarts with skip_existing and its
error + log tail recorded; no restart once the round's stop file exists; the limit holds. The helper is lifted from
the source (modal_rl1 builds Modal images at import)."""
import ast
import asyncio
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "modal_rl1.py"


def helper():
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "teacher_with_restarts")
    ns = {"Path": Path, "time": time, "TEACHER_RESTARTS": 3}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SRC), "exec"), ns)
    return ns["teacher_with_restarts"]


class Fake:
    def __init__(self, log):
        self.log, self.closed, self.exited = log, False, False

    async def call(self, op, timeout=0):
        self.exited = op == "exit"

    def close(self):
        self.closed = True


def run(outcomes, stop_file, limit=3, stop_after=None, tail="... torch.AcceleratorError: illegal memory access"):
    made, calls, restarts, report = [], [], [], {}

    def start(attempt):
        made.append(Fake(f"log {attempt}"))
        return made[-1], go

    async def go(trainer, attempt):
        calls.append(attempt)
        if stop_after is not None and attempt == stop_after:
            Path(stop_file).write_text("stop")
        out = outcomes[attempt]
        if isinstance(out, Exception):
            raise out
        return out

    got = asyncio.run(helper()(start, stop_file, restarts, report, limit=limit,
                               log_tail=lambda a: f"{tail} (attempt {a})"))
    return got, calls, restarts, made


def test_restarts_until_a_clean_end(tmp_path):
    ima = RuntimeError("trainer exited during stream_topk (1)")
    got, calls, restarts, made = run([ima, ima, {"tokens": 5}], tmp_path / "stop")
    assert got == {"tokens": 5} and calls == [0, 1, 2]
    assert [r["attempt"] for r in restarts] == [0, 1] and "illegal memory access (attempt 0)" in restarts[0]["log_tail"]
    assert all(t.closed and t.exited for t in made)


def test_no_restart_after_the_stop_file_or_past_the_limit(tmp_path):
    ima = RuntimeError("trainer exited during stream_topk (1)")
    got, calls, restarts, _ = run([ima, {"tokens": 1}], tmp_path / "stop", stop_after=0)
    assert "error" in got and calls == [0] and restarts == []
    got, calls, restarts, _ = run([ima, ima, ima], tmp_path / "stop2", limit=2)
    assert "error" in got and calls == [0, 1, 2] and len(restarts) == 2


def test_a_verify_failure_is_never_restarted(tmp_path):
    exit_ = RuntimeError("trainer exited during stream_topk (1)")
    got, calls, restarts, made = run([exit_, {"tokens": 1}], tmp_path / "stop",
                                     tail="stream teacher verify failed: {'top1_agree': 0.93}")
    assert got.get("verify_failed") and calls == [0] and restarts == [] and made[0].closed


def test_restarts_keep_the_verify_on():
    src = (SRC.parent / "modal_rl1.py").read_text()
    assert "verify_after=verify_after if not attempt else 0" not in src and "verify_after=verify_after," in src


def test_a_verify_failure_stops_the_round():
    src = (SRC.parent / "modal_rl1.py").read_text()
    i = src.index('get("verify_failed") and not report.get("teacher_verify_failed")')
    assert "os.killpg(proc.pid, signal.SIGTERM)" in src[i:i + 300]


def test_teacher_env_accepts_the_pin_and_lock_flags():
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stream_teacher_env")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SRC), "exec"), ns)
    env = ns["stream_teacher_env"](1, "RL1_EXPERTS_FP8=1,RL1_STREAM_CHAINS=1,RL1_GDN_PIN_O=0,RL1_CHAIN_LOCK=1")
    assert env == {"RL1_EXPERTS_FP8": "1", "RL1_STREAM_CHAINS": "1", "RL1_GDN_PIN_O": "0", "RL1_CHAIN_LOCK": "1"}
    import pytest
    with pytest.raises(ValueError, match="not a teacher flag"):
        ns["stream_teacher_env"](1, "RL1_EXPERTS_FP8=1,FOO=1")
