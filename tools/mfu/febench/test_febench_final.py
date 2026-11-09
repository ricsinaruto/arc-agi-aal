import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import febench  # noqa: E402
from arc_infra.runops.watch import END_MARKER  # noqa: E402


def test_final_written_once_on_done_only(tmp_path, capsys):
    assert not febench.write_final(tmp_path, {"phase": "W2-sleep", "run": "x"})
    assert not (tmp_path / "final.json").exists()
    assert not febench.write_final(tmp_path, {"phase": "error", "run": "x"})
    (tmp_path / "final.json").write_text("{}")  # left by an earlier run of the same name (pf-mfu2-sleepkv4c)
    status = {"phase": "done", "run": "x", "t": 5, "total_s": 9}
    assert febench.write_final(tmp_path, status)
    assert json.loads((tmp_path / "final.json").read_text())["state"] == "ok"
    assert not febench.write_final(tmp_path, status)
    out = capsys.readouterr().out
    assert out.count("RUNOPS END ok") == 1 and re.search(END_MARKER, out)


def test_status_keeps_its_methods():  # write_final once split the class (pf-mfu2-sleepkv4)
    assert all(callable(getattr(febench.Status, m, None)) for m in ("dump", "gpu_sampler"))


def test_sizing_covers_the_measured_runs():
    import sizing
    # mfu2-aot2: R-0,J1,J2,J0,J3 took ~1,620 s of boots + setup + decode; W3 died at the hand-set 2,400 s
    need = sizing.need_s("R-0,J1,J2,J0,J3,W3", "tput")
    assert need > 2400 * 1.2 and need < 7200
    assert sizing.need_s("R-0,W3", "tput") >= 1.3 * (180 + 360 + 480 + 240)
    assert sizing.need_s("R-0,L5-70", "tput", warm=60, measure=120) > sizing.need_s("R-0,L5-70", "tput", warm=0, measure=0)
    assert sizing.main(["R-0,J1", "--cpu"]) is None
