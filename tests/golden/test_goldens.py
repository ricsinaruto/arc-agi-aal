"""Goldens: loss, masking and export must reproduce data/<name>.json (pinned from sft-heal 5f7b222de)."""
import json
import math
from pathlib import Path

import goldens

DATA = Path(__file__).resolve().parent / "data"


def same(got, want, path="", rel=1e-5):
    if isinstance(want, dict):
        assert set(got) == set(want), f"{path}: keys {sorted(set(got) ^ set(want))[:5]}"
        for k in want:
            same(got[k], want[k], f"{path}.{k}", rel)
    elif isinstance(want, list):
        assert len(got) == len(want), f"{path}: length {len(got)} vs {len(want)}"
        for i, (a, b) in enumerate(zip(got, want)):
            same(a, b, f"{path}[{i}]", rel)
    elif isinstance(want, float):
        assert math.isclose(got, want, rel_tol=rel, abs_tol=1e-7), f"{path}: {got} vs {want}"
    else:
        assert got == want, f"{path}: {got} vs {want}"


def load(name):
    return json.loads(json.dumps(json.loads((DATA / f"{name}.json").read_text())))


def test_loss():
    got, want = json.loads(json.dumps(goldens.loss())), load("loss")
    same(got.pop("grad_norms"), want.pop("grad_norms"), "grad_norms", rel=2e-3)  # CPU reduction order varies run to run
    same(got, want)


def test_masking(tmp_path):
    same(json.loads(json.dumps(goldens.masking(tmp_path))), load("masking"))


def test_export(tmp_path, monkeypatch):
    same(json.loads(json.dumps(goldens.export(tmp_path, monkeypatch))), load("export"))


def test_mtp():
    same(json.loads(json.dumps(goldens.mtp())), load("mtp"), rel=1e-4)
