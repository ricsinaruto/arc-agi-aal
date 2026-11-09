"""heal_reload_check.compare: a world-2 sharded checkpoint against the export heal_delta builds from the same weights."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import heal_delta  # noqa: E402
import heal_reload_check  # noqa: E402


def _setup(tmp_path):
    g = torch.Generator().manual_seed(0)
    params = {"model.model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(8, 128, generator=g),
              "model.model.language_model.layers.1.mlp.gate.weight": torch.randn(4, 128, generator=g),
              "model.model.language_model.layers.1.mlp.experts.gate_up_proj": torch.randn(4, 64, 128, generator=g),
              "model.model.language_model.layers.1.mlp.experts.down_proj": torch.randn(4, 128, 32, generator=g)}
    cut = heal_delta.Cut({"student_cut": {"source_layers": [0, 3]}})
    export = {}
    for key, full in params.items():
        name = key[len("model."):]
        rows = [1, 3] if heal_delta.FUSED.match(name) else None
        export.update(dict(heal_delta.entries(name, full, cut, rows=rows, search=20)))
    ck = tmp_path / "ckpt-4"
    ck.mkdir()
    for r in range(2):
        torch.save({k: v.chunk(2, 0)[r].clone() for k, v in params.items()}, ck / f"params-rank{r}.pt")
        torch.save({"adam": {"state": {0: {"step": torch.tensor(4.0), "exp_avg": torch.zeros(3)}}},
                    "rows": {"t": 4, "rows": [], "m": [], "v": []}}, ck / f"optim-rank{r}.pt")
    (ck / "progress.json").write_text(json.dumps({"world": 2, "student": str(tmp_path)}))
    return ck, export, cut, params


def test_reload_check_matches(tmp_path):
    ck, export, cut, _ = _setup(tmp_path)
    v = heal_reload_check.compare(ck, export, cut, experts=8)
    assert v["ok"], v
    assert v["dense"]["checked"] == 2 and v["experts"]["present_rows"] == 4 and v["experts"]["checked"] == 4


def test_reload_check_catches_swapped_shards(tmp_path):
    ck, export, cut, params = _setup(tmp_path)
    a, b = (torch.load(ck / f"params-rank{r}.pt") for r in range(2))
    torch.save(b, ck / "params-rank0.pt")
    torch.save(a, ck / "params-rank1.pt")
    assert not heal_reload_check.compare(ck, export, cut, experts=8)["ok"]


def test_reload_check_catches_stale_weights(tmp_path):
    ck, export, cut, params = _setup(tmp_path)
    s = torch.load(ck / "params-rank0.pt")
    s["model.model.language_model.layers.0.self_attn.q_proj.weight"] += 1e-3
    torch.save(s, ck / "params-rank0.pt")
    assert not heal_reload_check.compare(ck, export, cut, experts=8)["ok"]
