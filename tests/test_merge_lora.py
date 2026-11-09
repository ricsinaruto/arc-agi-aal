"""Merge read (19 Oct): the expert LoRA delta in PEFT ParamWrapper layout, the prune mask, and the pre-registered
read verdicts."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import merge_lora  # noqa: E402


def test_expert_delta_matches_per_expert_products(tmp_path):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    shapes = {"E": 3, "H": 4, "INTER": 2, "R": 2}
    E, H, I, R = shapes["E"], shapes["H"], shapes["INTER"], shapes["R"]
    g = torch.Generator().manual_seed(0)
    base = "base_model.model.model.language_model.layers.5.mlp.experts."
    t = {}
    for stem, (o, i) in ((base + "base_layer.", (2 * I, H)), (base, (H, I))):
        t[stem + "lora_A.weight"] = torch.randn(R * E, i, generator=g)  # expert-major rows
        t[stem + "lora_B.weight"] = torch.randn(o, R * E, generator=g)  # expert-minor columns
    save_file(t, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 32, "r": R}))
    d13, d2, scale = merge_lora.expert_delta(str(tmp_path), 5, torch, shapes)
    assert scale == 16.0 and d13.shape == (E, 2 * I, H) and d2.shape == (E, H, I)
    for stem, d in ((base + "base_layer.", d13), (base, d2)):
        a, b = t[stem + "lora_A.weight"], t[stem + "lora_B.weight"]
        for e in range(E):
            want = scale * b[:, [r * E + e for r in range(R)]] @ a[e * R:(e + 1) * R]
            assert torch.allclose(d[e], want, atol=1e-5)
    with pytest.raises(RuntimeError):
        merge_lora.expert_delta(str(tmp_path), 5, torch, {**shapes, "E": 4})


def test_kept_of_mask():
    assert merge_lora.kept_of({"pruned": {"0": [1, 3], "2": []}}, experts=5) == {0: [0, 2, 4], 2: [0, 1, 2, 3, 4]}


def row(outcome, lead=1000, ttc=None, spent=0, actions=10):
    return {"outcome": outcome, "cleared": outcome == "cleared", "lead_tokens": lead, "tokens_to_clear": ttc,
            "completion_tokens": spent, "actions": actions}


def test_read_verdicts():
    assert merge_lora.l2_try(row("lead_out")) == (False, False, 0)
    assert merge_lora.l2_try(row("budget", spent=32000)) == (True, False, 32000)
    assert merge_lora.l2_try(row("cleared", ttc=5000)) == (True, True, 5000)
    merged = [row("cleared", ttc=4000)] * 12
    zero = [row("cleared", ttc=20000)] * 8 + [row("budget", spent=32000)] * 4
    shipped = [row("cleared", ttc=18000)] * 9 + [row("budget", spent=32000)] * 3
    out = merge_lora.verdicts({"merged": merged, "zero": zero, "shipped": shipped})
    assert out["merged"]["L2_tokens_per_clear"] == 4000 and out["zero"]["L2_tokens_per_clear"] == 36000
    assert out["zero_over_merged"][0] == 9.0 and out["path_works"]
    assert out["chain_ok"] and out["zero_over_shipped"][1][0] <= 1.0 <= out["zero_over_shipped"][1][1]
    lead_out = merge_lora.verdicts({"zero": zero + [row("lead_out", lead=None)], "merged": merged})
    assert lead_out["zero"]["entered_L2"] == 12 and lead_out["zero"]["tries"] == 13
