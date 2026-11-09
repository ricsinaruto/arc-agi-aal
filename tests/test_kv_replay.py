import asyncio
from types import SimpleNamespace

from arena import kv_replay


def test_parse_metrics_sums_labels_and_keeps_vllm_families():
    text = ('# HELP x\nvllm:kv_cache_usage_perc{engine="0"} 0.5\nvllm:prompt_tokens_total{engine="0"} 100\n'
            'vllm:prompt_tokens_total{engine="1"} 50\nvllm:prefix_cache_hits_total 80\nvllm:other 3\n')
    m = kv_replay.parse_metrics(text)
    assert m == {"vllm:kv_cache_usage_perc": 0.5, "vllm:prompt_tokens_total": 150.0, "vllm:prefix_cache_hits_total": 80.0}


def test_replay_extends_each_try_and_admits_under_the_kv_target(monkeypatch):
    """Each request extends the try's own stream (prefix of the next), tries of a group share the first prompt,
    exact completion lengths, and no new try starts while KV is at or above kv_admit."""
    sent = []
    state = {"kv": 0.1, "prompt": 0.0, "gen": 0.0}

    class Server:
        async def post(self, path, body, timeout=None):
            sent.append(body)
            state["prompt"] += len(body["prompt"])
            state["gen"] += body["max_tokens"]
            state["kv"] = min(0.99, state["kv"] + 0.2)
            await asyncio.sleep(0.01)
            return {}

    async def metrics(server):
        return {"vllm:kv_cache_usage_perc": state["kv"], "vllm:prompt_tokens_total": state["prompt"],
                "vllm:generation_tokens_total": state["gen"], "vllm:prefix_cache_queries_total": state["prompt"],
                "vllm:prefix_cache_hits_total": state["prompt"] * 0.5}

    monkeypatch.setattr(kv_replay, "metrics", metrics)
    trace = {"tries": [{"try": "a", "group": "g1", "reqs": [[100, 5, 0], [130, 7, 100]]},
                       {"try": "b", "group": "g1", "reqs": [[100, 3, 0], [120, 2, 100]]}]}
    cfg = {**kv_replay.DEFAULT, "seconds": 1.0, "gap_s": 0.0, "stagger_s": 0.0, "sample_s": 0.05, "kv_admit": 0.5}
    replay = kv_replay.Replay(Server(), "m", trace, cfg)
    out = asyncio.run(replay.run())
    assert all(b["min_tokens"] == b["max_tokens"] and b["ignore_eos"] for b in sent)
    assert len(sent[0]["prompt"]) == 100
    firsts = [b["prompt"] for b in sent if len(b["prompt"]) == 100]
    assert all(f == firsts[0] for f in firsts)  # one group: one shared first prompt
    seconds = [b["prompt"] for b in sent if len(b["prompt"]) == 130]
    assert seconds and seconds[0][:100] == firsts[0]
    assert out["started"] <= 3  # KV passed 0.5 after the first requests: admission stopped
    assert kv_replay.summary(replay.samples, 0.0)["gpu_hit"] == 0.5
