"""Golden (g): the study each checked-in run config builds (launch/rl/build.inline) stays identical through rl-min.
GOLDEN_RECORD=1 rewrites tests/golden/data/study.json; only an intentional, reviewed change may."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "launch/rl"))
import build as B  # noqa: E402

DATA = Path(__file__).with_name("data") / "study.json"


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def digests(derived=None):
    """Per config: the built study, and for rl-loop studies the settings the loop derives from it (code defaults
    included: rl_loop.settings). ``derived``: the configs whose
    settings are derived (those pinned on this tree; a config built from an older source keeps only its study)."""
    from arena import rl_loop

    out = {}
    for p in sorted((ROOT / "launch/rl/runs").glob("*.json")):
        study = B.inline(json.loads(p.read_text())["study"])
        out[p.stem] = sha(study)
        if "rl_loop" in study and "mg" in study["rl_loop"] and (derived is None or p.stem in derived):
            cfg = rl_loop.settings(study["rl_loop"])
            out[p.stem + ":settings"] = sha(cfg)
        if p.stem in LIVE:  # the redesign loop's own settings (rl_seq.settings over mg.redesign)
            from arena import rl_seq

            out[p.stem + ":seq"] = sha(rl_seq.settings(study["rl_loop"]["mg"]))
    return out


LIVE = {"rl1-loop-seq-real-v8", "rl1-loop-seq-glong-v5"}  # rl-min C (28 Oct): pinned through the refactor


def test_built_study_values_are_unchanged():
    want = json.loads(DATA.read_text()) if DATA.is_file() else {}
    now = digests(LIVE)  # settings only for configs whose source is this tree's recipe (the hread reads keep theirs)
    if os.environ.get("GOLDEN_RECORD") == "1":
        DATA.parent.mkdir(exist_ok=True)
        DATA.write_text(json.dumps(now, indent=1) + "\n")
        want = now
    assert {k: now.get(k) for k in want} == want
