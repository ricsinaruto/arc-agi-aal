import json

from arc2 import tasks, ttt_data
from arc2.loop import augment_back, augment_view

TASK = {"train": [{"input": [[1, 2], [3, 4]], "output": [[1, 3], [2, 4]]},
                  {"input": [[5, 6]], "output": [[5], [6]]},
                  {"input": [[7], [8]], "output": [[7, 8]]}],
        "test": [{"input": [[7, 8, 9]], "output": [[7], [8], [9]]}]}


def test_leave_one_out_samples():
    out = ttt_data.samples(TASK, "t1", augments=4)
    assert len(out) == 3 * 4 and {(s["view"], s["held_out"]) for s in out} == {(k, j) for k in range(4) for j in range(3)}
    for s in out:
        view = augment_view(TASK, "t1", s["view"])
        held = view["train"][s["held_out"]]
        user = s["messages"][1]["content"]
        assert s["target"] == "```\n" + tasks.text(held["output"]) + "\n```"
        assert user.count("Training pair") == 2 and "Test input 1" in user and tasks.text(held["input"]) in user
        others = [p for i, p in enumerate(view["train"]) if i != s["held_out"]]
        assert all(tasks.text(p["output"]) in user for p in others)
        assert s["messages"][0]["content"] == tasks.DIRECT_SYSTEM and s["tokens"] > 0
    assert ttt_data.samples({"train": TASK["train"][:1], "test": TASK["test"]}, "t1") == []
    assert ttt_data.samples(TASK, "t1", augments=1, max_tokens=5) == []


def test_queries_target_in_view_frame():
    for view in range(4):
        q = ttt_data.queries(TASK, "t1", view)[0]
        grid = [[int(c) for c in row.split()] for row in q["target"].strip("`\n").split("\n")]
        assert augment_back(grid, "t1", view) == TASK["test"][0]["output"]
        assert q["messages"][1]["content"].count("Training pair") == 3
    assert "target" not in ttt_data.queries({"train": TASK["train"], "test": [{"input": [[1]]}]}, "t1")[0]


def test_nll_parity_against_fake_server(tmp_path, monkeypatch):
    import asyncio

    import httpx

    from arc2 import nll_parity

    def render(messages, gen):
        text = "".join(m["role"] + ":" + m["content"] + "|" for m in messages) + ("assistant:" if gen else "")
        return [ord(c) for c in text]

    def handler(request):
        body = json.loads(request.content or b"{}")
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        if request.url.path == "/tokenize":
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            return httpx.Response(200, json={"tokens": render(body["messages"], body["add_generation_prompt"])})
        assert request.url.path == "/v1/completions" and body["prompt_logprobs"] == 1
        return httpx.Response(200, json={"choices": [{"prompt_logprobs": [None] + [
            {str(t): {"logprob": -0.5}} for t in body["prompt"][1:]]}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    data = tmp_path / "data/evaluation"
    data.mkdir(parents=True)
    (data / "t1.json").write_text(json.dumps(TASK))
    report = asyncio.run(nll_parity.run_nll_parity({"arc2": {"data": "data"}}, {"tasks": ["t1"], "split": "evaluation"},
                                                   source=tmp_path, output=tmp_path / "out",
                                                   base_url="http://x/v1"))
    assert report["status"] == "complete" and report["nll"] == {"t1": 0.5}
    row = json.loads((tmp_path / "out/parity.json").read_text())["rows"][0]
    target = ttt_data.queries(TASK, "t1")[0]["target"]
    assert row["full_ids"][len(row["prompt_ids"]):] == [ord(c) for c in target + "|"]


def test_ttt_screen_load_dispatch(monkeypatch):
    """r3: the Flash mode ran parity on the tiny model (a dangling else). Each mode loads exactly its own model."""
    import importlib.util
    import pathlib

    import pytest

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts/arc2_ttt/ttt_screen.py"
    spec = importlib.util.spec_from_file_location("ttt_screen_under_test", path)
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    Flash = type("Qwen4ExpForConditionalGeneration", (), {})
    Tiny = type("Qwen3ForCausalLM", (), {})
    calls = []
    monkeypatch.setattr(ts, "load_flash", lambda cfg, out: calls.append("flash") or (Flash(), "tok", {}))
    monkeypatch.setattr(ts, "load_tiny", lambda cfg: calls.append("tiny") or (Tiny(), "tok", {}))
    model, _, loaded = ts.load({"mode": "flash"}, "out")
    assert calls == ["flash"] and isinstance(model, Flash) and loaded["model_class"].startswith("Qwen4Exp")
    calls.clear()
    model, _, _ = ts.load({"mode": "tiny"}, "out")
    assert calls == ["tiny"] and isinstance(model, Tiny)
    monkeypatch.setattr(ts, "load_flash", lambda cfg, out: (Tiny(), "tok", {}))
    with pytest.raises(RuntimeError):
        ts.load({"mode": "flash"}, "out")
    with pytest.raises(ValueError):
        ts.load({"mode": "other"}, "out")
