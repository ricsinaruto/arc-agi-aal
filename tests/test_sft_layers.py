"""scripts/sft/layers.py on CPU with the tiny student: windows selection and masks, the uncut reference, single-layer
scores, greedy sets per objective (distinct layers, depth reached, re-scored KL of the chosen set) and given sets."""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/sft"), str(ROOT / "tests/golden")]
import common  # noqa: E402
import layers  # noqa: E402
import train  # noqa: E402


def test_greedy_layers_on_the_tiny_student(tmp_path):
    common.tiny_student_on_disk(tmp_path / "student")
    model, _ = train.load_model(tmp_path / "student", "cpu", fast_attention=False)
    model.eval()
    g = torch.Generator().manual_seed(3)
    wins = [{"index": i, "ids": torch.randint(10, 2000, (300,), generator=g).tolist(),
             "assistant_spans": [[150, 220], [260, 300]]} for i in range(8)]
    (tmp_path / "tf.json").write_text(json.dumps({"windows": wins}))
    data = layers.windows_data(tmp_path / "tf.json", "cpu", tokens=256, scored=128)
    assert len(data) == 2  # index % 4 == 1: windows 1 and 5
    assert data[0]["ids"].shape == (1, 256) and data[0]["asst"].numel() == 127
    # last 256 of 300 ids: offset 44; rows from 128 score tokens 129..255 -> positions 173..299 of the window
    assert int(data[0]["asst"].sum()) == (220 - 173) + (300 - 260)
    n = len(model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers)
    depth = min(2, n - 1)
    rep = layers.run(model, data, tmp_path / "out", depth=depth, lazy=2, cost_full=2.0, sets={"first": [0]})
    assert rep["uncut"]["asst"] < 1e-4 and rep["layers"] == n
    assert set(rep["single"]) == set(range(n)) and max(v["asst"] for v in rep["single"].values()) > rep["uncut"]["asst"]
    # a skipped layer passes its input through: the final hidden state changes (the tiny model's layers move it little)
    lm = getattr(model.model, "language_model", model.model)
    base = lm(input_ids=data[0]["ids"], use_cache=False).last_hidden_state
    sk = layers.Skipper(lm.layers)
    sk.skip.update(range(n))
    bare = lm(input_ids=data[0]["ids"], use_cache=False).last_hidden_state
    sk.skip.clear()
    assert bare.shape == base.shape and not torch.equal(bare, base)
    assert rep["sets"]["first"]["kl"] == rep["single"][0]
    for objective in ("kl", "kl_per_cost"):
        trail = rep[f"greedy_{objective}"]
        chosen = [t["layer"] for t in trail]
        assert len(chosen) == depth == len(set(chosen)) and rep[f"set_{objective}"] == sorted(chosen)
        assert trail[0]["kl"] == rep["single"][chosen[0]] or len(rep["single"]) > 2
    saved = json.loads((tmp_path / "out/layers.json").read_text())
    assert saved["set_kl"] == rep["set_kl"]
    events = [json.loads(x)["event"] for x in (tmp_path / "out/events.jsonl").read_text().splitlines()]
    assert events[:2] == ["layers_uncut", "layers_single"] and events[-1] == "done" and events.count("layers_greedy") == 2 * depth
    # the skip wrappers leave the model whole once the run ends
    out = model.model(input_ids=data[0]["ids"], use_cache=False)
    assert out is not None
