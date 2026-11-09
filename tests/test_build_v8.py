"""docs/validation-20261007/build_v8.py: every v8 item is on in the built study and each ``--off`` switch works."""
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "docs/validation-20261007/build_v8.py"
MODULE = runpy.run_path(str(BUILDER), run_name="build_v8")
V8, ITEMS, WEIGHTS, plan = MODULE["V8"], MODULE["ITEMS"], MODULE["WEIGHTS"], MODULE["plan"]


def built(tmp_path, *arguments):
    """The study of a build without wheels (the bundle check at the end of build_v71.py refuses it, after the study
    is written) and the items ``plan`` switched on."""
    out = tmp_path / "v8-test"
    arguments = [str(out), "--wall=7200", *arguments]
    run = subprocess.run([sys.executable, str(BUILDER), *arguments], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert "Bundle lacks offline dependencies" in run.stderr, run.stderr[-2000:]
    return json.loads((out / "configs/comparison.json").read_text()), plan(arguments)[1]


@pytest.fixture
def head(tmp_path):
    folder = tmp_path / "head"
    folder.mkdir()
    (folder / "overlay.safetensors").write_bytes(b"weights")
    (folder / "overlay-manifest.json").write_text("{}")
    return f"--draft-head-overlay={folder}"


BASE = ("vision", "guards", "actions", "level_prompt", "frame_diff", "animation")  # v7.1p's dfr_harness flags
OPT_IN = ("transient", "reset_offer", "border_moves", "sweep")
V8_FLAGS = {flag: flag != "reset_offer" for flag in BASE + OPT_IN}  # no reset offer in v8


def shown(study):
    return {name: check(study) for name, (_, check) in ITEMS.items() if check is not None}


def test_v8_build_has_every_item_on(tmp_path, head):
    study, on = built(tmp_path, head)
    assert on == list(V8) and shown(study) == {name: name in V8 for name in shown(study)}, shown(study)
    assert study["server"]["expert_prune"]["mask"].endswith(Path(WEIGHTS["all25"][0]).name)
    assert study["server"]["expert_overlay"]["dataset"] == WEIGHTS["all25"][1].split(":")[0]
    assert study["dfr_harness"] == V8_FLAGS
    assert study["b200"]["version"] == "v8" and study["wall_seconds"] == 7200 and study["active_limit"] == 44
    # v7.1p's harness (--restore-v6d) and nothing rejected for v8
    assert study["hx_harness"]["toolbox"] and "stuck_notice" in study["harness_fixes"]["fixes"]
    assert not {"compaction_keep_fraction", "compaction_full_prefix", "first_trigger_stagger"} & set(study)
    assert "fresh_attempt" not in study and "collapse_log" not in study["cache_experiment"]["prefix_cache"]


@pytest.mark.parametrize("item", [name for name in V8 if name != "weights"])
def test_each_off_switch_changes_only_its_item(tmp_path, head, item):
    study, on = built(tmp_path, f"--off={item}", *([] if item == "draft-head" else [head]))
    assert shown(study) == {name: name in V8 and name != item for name in shown(study)}
    assert item not in on
    flag = item.replace("-", "_")
    assert study["dfr_harness"] == {**V8_FLAGS, **({flag: False} if flag in V8_FLAGS else {})}


def test_weights_off_serves_the_v71p_set_without_a_default_head(tmp_path):
    study, on = built(tmp_path, "--off=weights")
    assert study["server"]["expert_prune"]["mask"].endswith("prune-t37-int3-requant-excl5-v1.json")
    assert study["server"]["expert_overlay"]["manifest_sha256"] == WEIGHTS["excl5"][1].split(":")[1]
    assert "draft-head" not in on and "draft_head" not in study["server"]


def test_everything_off_is_v71p(tmp_path):
    study, on = built(tmp_path, "--off=" + ",".join(V8))
    assert on == [] and not any(shown(study).values())
    assert study["priority"]["floor"] is True
    assert study["dfr_harness"] == {flag: flag in BASE for flag in BASE + OPT_IN}


def test_reset_offer_is_opt_in(tmp_path):
    study, on = built(tmp_path, "--on=reset-offer")
    assert "reset-offer" in on and study["dfr_harness"] == {flag: True for flag in BASE + OPT_IN}


def test_collapse_log_is_an_extra_switch(tmp_path):
    study, on = built(tmp_path, "--collapse-log")
    assert study["cache_experiment"]["prefix_cache"]["collapse_log"] is True
    assert all(on_ for name, on_ in shown(study).items() if name not in ("draft-head", "reset-offer"))


@pytest.mark.parametrize("arguments", [["--compact"], ["--compact", "--compact-full-prefix"], ["--stagger=0.5"],
                                       ["--streams=48"], ["--off=nothing"], ["--off=reset-offer"], ["--on=sweep"], ["--collapse-log", "--kaggle"], ["--weights=unknown"],
                                       ["--off=weights", "--mask=a.json", "--overlay=a/b:c"]])
def test_refusals(arguments):
    with pytest.raises(SystemExit):
        plan(["out", *arguments])
