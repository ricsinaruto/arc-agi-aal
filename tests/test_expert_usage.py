"""Expert-usage measurement: probe runtime logic, comparisons and the trial flow on a mock server."""

import json
import math
import random
import time

import httpx
import numpy as np
import pytest

from arena import expert_usage as eu
from arena.kernel_gate import kl_coarse


def test_compare_matches_the_kernel_gate_kl():
    rng = np.random.default_rng(0)
    n = 40
    a_ids, b_ids = np.zeros((n, 20), np.int32), np.zeros((n, 20), np.int32)
    a_lp, b_lp = np.zeros((n, 20), np.float32), np.zeros((n, 20), np.float32)
    for i in range(n):
        la = rng.normal(0, 2, 60)
        lb = la + rng.normal(0, 0.5, 60)
        la -= np.log(np.exp(la).sum())
        lb -= np.log(np.exp(lb).sum())
        ia, ib = np.argsort(-la)[:20], np.argsort(-lb)[:20]
        a_ids[i], a_lp[i], b_ids[i], b_lp[i] = ia, la[ia], ib, lb[ib]
    kl, agree, outside = eu.compare((a_ids, a_lp), (b_ids, b_lp))
    expected = [kl_coarse({int(t): float(v) for t, v in zip(a_ids[i], a_lp[i])},
                          {int(t): float(v) for t, v in zip(b_ids[i], b_lp[i])}) for i in range(n)]
    assert np.allclose(kl, expected, atol=1e-9)
    same, _, _ = eu.compare((a_ids, a_lp), (a_ids, a_lp))
    assert np.allclose(same, 0.0)
    assert agree.dtype == bool and outside.shape == (n,)


def test_masks_prune_the_least_weighted_experts_per_layer():
    weight = np.tile(np.arange(eu.EXPERTS, dtype=float), (2, 1))
    weight[1] = weight[1][::-1]
    count = np.ones_like(weight)
    pruned = eu.masks(weight, count, [0.25], [3, 7])["0.25"]
    assert pruned["3"] == list(range(128))
    assert pruned["7"] == list(range(eu.EXPERTS - 128, eu.EXPERTS))
    cover = eu.coverage(np.array([[1.0] * 10 + [0.0] * (eu.EXPERTS - 10)]))
    assert cover["0.9"] == [9] and cover["0.999"] == [10]


def test_assistant_mask_marks_positions_predicting_assistant_tokens():
    mask = eu.assistant_mask(9, [[3, 6]])
    assert mask.tolist() == [False, False, True, True, True, False, False, False, False]


def test_parse_positions_keeps_the_top_k_and_the_recorded_token():
    ids = [5, 1, 2]
    entries = [None,
               {"1": {"logprob": -0.1}, "4": {"logprob": -3.0}},
               {"4": {"logprob": -0.2}, "1": {"logprob": -2.5}, "2": {"logprob": -9.0}}]
    actual, top_ids, top_lps = eu.parse_positions(entries, ids, 2)
    assert actual.tolist() == pytest.approx([-0.1, -9.0])
    assert top_ids.tolist() == [[1, 4], [4, 1]]
    with pytest.raises(RuntimeError):
        eu.parse_positions(entries[:2], ids, 2)


def _mock_server():
    state = {"record": False, "mask": {}, "count": np.zeros((4, eu.EXPERTS)), "weight": np.zeros((4, eu.EXPERTS)),
             "prob": np.zeros((4, eu.EXPERTS)), "tokens": np.zeros(4)}

    def logprobs(ids, k, noise):
        out, rng = [None], random.Random(len(ids) * 7 + int(noise * 100))
        for t in ids[1:]:
            logits = [rng.gauss(0, 2) + (3 if tok == 0 else 0) + rng.gauss(0, noise or 1e-9) * (noise > 0)
                      for tok in range(30)]
            z = math.log(sum(math.exp(v) for v in logits) + 50)
            ranked = sorted(((tok, v - z) for tok, v in enumerate(logits)), key=lambda kv: -kv[1])
            entry = {str(tok): {"logprob": v} for tok, v in ranked[:k]}
            entry.setdefault(str(t), {"logprob": dict(ranked).get(t, -12.0)})
            out.append(entry)
        return out

    def handler(request):
        body = json.loads(request.content)
        if request.url.path.endswith("/collective_rpc"):
            assert body["method"] == "arc_expert_probe"
            cmd = json.loads(body["args"][0])
            if cmd["op"] == "reset":
                for key in ("count", "weight", "prob", "tokens"):
                    state[key][:] = 0
            elif cmd["op"] == "record":
                state["record"] = cmd["on"]
            elif cmd["op"] == "mask":
                state["mask"] = cmd["pruned"]
            reply = {"ok": True, "op": cmd["op"], "layers": [0, 1, 2, 3], "calls": 8}
            if cmd["op"] == "dump":
                reply.update(dump_layers=[0, 1, 2, 3], **{k: state[k].tolist()
                                                          for k in ("count", "weight", "prob", "tokens")})
            return httpx.Response(200, json={"results": [reply]})
        ids = body["prompt"]
        assert body["max_tokens"] == 1 and body["prompt_logprobs"] in (0, 20)
        if state["record"]:
            experts = np.random.default_rng(len(ids)).zipf(1.5, size=len(ids) * 10) % eu.EXPERTS
            for layer in range(4):
                np.add.at(state["count"][layer], experts, 1)
                np.add.at(state["weight"][layer], experts, 0.1)
            state["tokens"] += len(ids)
        noise = 0.3 + 0.001 * sum(len(v) for v in state["mask"].values()) if state["mask"] else 0.0
        return httpx.Response(200, json={"choices": [{"prompt_logprobs": logprobs(ids, body["prompt_logprobs"],
                                                                                    noise)}]})

    return httpx.MockTransport(handler)


async def test_trial_runs_every_pass_against_a_mock_server(tmp_path, monkeypatch):
    async def idle(*args, **kwargs):
        return {}

    monkeypatch.setattr(eu, "isolate_trial", idle)
    rng = random.Random(1)
    folder = tmp_path / "src" / eu.INPUTS
    folder.mkdir(parents=True)
    for name, count, length in (("usage.json", 5, 200), ("heldout.json", 3, 300)):
        rows = [{"index": i, "ids": [rng.randrange(30) for _ in range(length)], "assistant_spans": [[5, 150]]}
                for i in range(count)]
        (folder / name).write_text(json.dumps(rows))
    result = await eu.run_expert_usage({"server": {}, "served_model": "m"}, {"name": "expert-usage"},
                                       source=tmp_path / "src", output=tmp_path / "out", base_url="http://x/v1",
                                       deadline_epoch=time.time() + 600, transport=_mock_server(),
                                       log=lambda *a, **k: None)
    assert result["status"] == "complete" and not result["skipped"], result["errors"]
    assert set(result["passes"]) == {"reference", "repeat", *(f"mask-{f}" for f in eu.SETTINGS["fractions"])}
    assert result["passes"]["repeat"]["summary"]["all"]["mean_kl"] == pytest.approx(0.0)
    assert result["passes"]["mask-0.5"]["summary"]["assistant"]["mean_kl"] > 0
    assert len(result["masks"]["0.125"]["0"]) == 64
    assert json.loads((tmp_path / "out/expert-usage.json").read_text())["status"] == "complete"


def test_lab_overrides_parse_the_diagnostic_switches():
    from arena import serving

    lab = serving.lab_overrides({"ARENA_EXPERT_PROBE": "0", "ARENA_ENFORCE_EAGER": "1",
                                 "ARENA_READY_GUARD": "180,720"})
    assert lab == {"expert_probe": False, "enforce_eager": True, "ready_guard": (180, 720)}
    with pytest.raises(ValueError):
        serving.lab_overrides({"ARENA_READY_GUARD": "30,720"})
    with pytest.raises(ValueError):
        serving.lab_overrides({"ARENA_ENFORCE_EAGER": "yes"})


def test_ready_guard_gives_up_after_graph_capture(monkeypatch):
    import time as time_module

    from arena import serving

    clock = {"t": 0.0}

    class Recipe:
        BASE_URL, SERVED_MODEL_NAME = "http://x/v1", "m"

        @staticmethod
        def alive(pid, ticks):
            return True

        @staticmethod
        def request_json(url, timeout):
            raise RuntimeError("not ready")

        @staticmethod
        def tail_log(lines):
            return "Graph capturing finished in 14 secs" if clock["t"] >= 50 else "loading"

    monkeypatch.setattr(time_module, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(time_module, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    wait = serving.ready_guard(Recipe, 180, 720)
    with pytest.raises(TimeoutError, match="after CUDA-graph capture"):
        wait({"pid": 1, "start_ticks": 1}, 1e9)
    assert 230 <= clock["t"] <= 240


async def test_server_check_records_a_started_server(tmp_path):
    def handler(request):
        if request.url.path.endswith("/collective_rpc"):
            return httpx.Response(200, json={"results": [{"ok": True, "layers": [0, 1], "calls": 4}]})
        body = json.loads(request.content)
        if body.get("prompt_logprobs") is not None:
            return httpx.Response(200, json={"choices": [{"prompt_logprobs": [None] * len(body["prompt"])}]})
        return httpx.Response(200, json={"choices": [{"text": " Paris."}]})

    result = await eu.server_check({"served_model": "m"}, {"name": "diag", "server_env": {"X": "1"}},
                                   base_url="http://x/v1", output=tmp_path, transport=httpx.MockTransport(handler),
                                   log=lambda *a, **k: None)
    assert result["status"] == "complete" and result["probe"] == {"layers": 2, "calls": 4}
    assert result["decode"]["text"] == " Paris." and result["prefill"]["positions"] == 600
