import json

from arena import rl_single


def make_try(root, label, n):
    att = root / "attempts" / label
    att.mkdir(parents=True)
    with open(att / "log.jsonl", "w") as f:
        for step in range(1, n + 1):
            f.write(json.dumps({"kind": "observation", "step": step, "levels": 0, "frames": [[f"{step % 3}1", "00"]]},
                               separators=(",", ":")) + "\n")
        f.write(json.dumps({"kind": "level_bench_clear", "step": n}, separators=(",", ":")) + "\n")
    with open(att / "rollout.jsonl", "w") as f:
        for k in range(n):
            f.write(json.dumps({"n": k, "step": k + 1, "usage": {"completion_tokens": 10 + k}, "finish_reason": "length" if k == 2 else "stop",
                                "tokens": [1, 2, 3]}) + "\n")
    return att


def test_pool_matches_inline_parse(tmp_path):
    atts = [make_try(tmp_path, f"a{i:03d}", 4 + i % 3) for i in range(10)]
    rf = {"capped": True}
    rl_single.TURN_CACHE.clear()
    assert rl_single.turn_prefetch(atts, rf) == 10
    key = json.dumps(rf, sort_keys=True)
    for a in atts:
        assert rl_single.TURN_CACHE[(str(a), key)] == rl_single.turn_try(str(a), rf)
    replies, end, flagged = rl_single.turn_try(str(atts[0]), rf)
    assert end == 4 and len(replies) == 4 and flagged[2]["kind"] == "capped"
    assert rl_single.turn_prefetch(atts, rf) == 0  # cached
