"""v17 a7 (30 Oct): U20's accumulate trained all 299 windows and died in its result block (NameError served_before)
because no test ran an update with the served router on. Here: the tiny Flash-shaped trainer of test_rl1_dist, R3 on
(op_route served), every window resolved through a real served-route store on disk (rl1_encode.served_table ->
Trainer.served_account), one accumulate + step through the result block; the call's coverage row is checked."""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
pytest.importorskip("peft")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))

import test_rl1_dist as base  # noqa: E402
from arena import served_route  # noqa: E402

sys.modules.setdefault("served_route", served_route)  # the bundle copies it beside rl1_encode (rl1/)

LAYERS = [0, 1, 2, 3]  # every decoder layer of the tiny model routes (MoE)
K = 2  # num_experts_per_tok


def write_record(folder, rid, ids, experts=8):
    """One engine request file <rid>-xxxxxxxx.rt: a record from position 0 over the window's tokens, its picks."""
    import numpy as np

    t = np.asarray(ids, dtype=np.int32)
    picks = np.stack([(np.arange(len(t))[:, None] + np.arange(K)[None, :] + layer) % experts
                      for layer in range(len(LAYERS))], axis=1).astype(np.int16)  # [T, L, K], distinct per row
    head = np.array([0, len(t), len(LAYERS), K], dtype=np.int32).tobytes()
    (folder / f"{rid}-abcdefgh.rt").write_bytes(head + t.tobytes() + picks.tobytes())
    (folder / f"{rid}-abcdefgh.mm.json").write_text(json.dumps({"lora": "policy"}))


@base.isolate
def test_served_accumulate_runs_its_result_block(tmp_path):
    import rl1_trainer as rt

    route = tmp_path / "served-route"
    route.mkdir()
    (route / "layers.json").write_text(json.dumps({"layers": LAYERS}))
    saved = (rt.CARD, dict(rt.ROUTE), dict(rt.SERVED_AUDIT), dict(rt.SERVED))
    try:
        rt.CARD = "cpu"
        t = base.trainer(tmp_path / "t")
        assert t.op_route({"router": "served"})["router"] == "served"
        made_rows = {}
        for name in base.WINDOWS:
            made = base.window(name)
            rid = f"chatcmpl-{name}"
            write_record(route, rid, made["inputs"]["input_ids"][0].tolist())
            made_rows[name] = [{"n": 0, "rid": rid, "keep": 0, "messages": [{}]}]

        def feed(specs_, cmd, **kw):
            todo = list(specs_)

            def nxt():
                spec = todo.pop(0)
                made = base.window(spec["name"])
                made["replies"] = [{"request": 0}]
                return t.finish_window({**spec, "served_route": str(route)}, made, made_rows[spec["name"]])
            return nxt, (lambda: None), {"waits": 0.0}

        t.window_feed = feed
        acc = t.op_accumulate({"attempts": base.specs(), "lr": base.LR, "first": True})
        assert acc["ok"], acc
        assert acc["served"].get("windows") == len(base.WINDOWS) and not acc["served"].get("no_record"), acc["served"]
        assert int(rt.ROUTE["used"]) > 0  # the routers took the served picks
        step = t.op_step({"lr": base.LR, "ref": "keep"})
        assert step["ok"], step
    finally:
        rt.CARD = saved[0]
        rt.ROUTE.clear(), rt.ROUTE.update(saved[1])
        rt.SERVED_AUDIT.clear(), rt.SERVED_AUDIT.update(saved[2])
        rt.SERVED.clear(), rt.SERVED.update(saved[3])
