import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from arena import replay_stub as rs


def rollout(folder, name, rows):
    path = folder / "rec" / "attempts" / name
    path.mkdir(parents=True)
    with open(path / "rollout.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def call(i):
    return {"role": "assistant", "content": None, "tool_calls": [{"id": f"t{i}", "type": "function",
                                                                   "function": {"name": "act", "arguments": f'{{"a": {i}}}'}}]}


SYS, OBS = {"role": "system", "content": "s"}, {"role": "user", "content": "board 0"}


def game(folder, name, k):
    """two requests: [sys, obs] -> call(k); [sys, obs, call(k), tool(k)] -> call(k + 10)"""
    tool = {"role": "tool", "tool_call_id": f"t{k}", "content": f"board {k}"}
    rollout(folder, name, [
        {"n": 0, "keep": 0, "gameplay": True, "messages": [SYS, OBS], "reply": call(k), "finish_reason": "tool_calls",
         "tokens": [1, 2, 3], "logprobs": [-0.1, -0.2, -0.3], "top": [[[1], [-0.1]], [[2], [-0.2]], [[3], [-0.3]]]},
        {"n": 1, "keep": 2, "gameplay": True, "messages": [call(k), tool], "reply": call(k + 10),
         "finish_reason": "tool_calls", "tokens": [4, 5], "logprobs": [-0.4, -0.5], "top": []}])
    return tool


def test_stub_replays_recorded_paths_round_robin_and_falls_back(tmp_path):
    t1, t2 = game(tmp_path, "a1", 1), game(tmp_path, "a2", 2)
    state = rs.State(rs.Corpus([tmp_path]), speed=0)
    tools = [{"type": "function", "function": {"name": "act"}}]
    first = [rs.chat_reply(state, {"messages": [SYS, OBS], "tools": tools})[1] for _ in range(2)]
    assert {f["choices"][0]["message"]["tool_calls"][0]["id"] for f in first} == {"t1", "t2"}  # each recording once
    code, out = rs.chat_reply(state, {"messages": [SYS, OBS, call(2), t2], "tools": tools, "logprobs": True,
                                      "return_token_ids": True})
    assert code == 200 and out["stub_match"] == "exact" and out["choices"][0]["message"]["tool_calls"][0]["id"] == "t12"
    assert out["choices"][0]["token_ids"] == [4, 5] and out["choices"][0]["logprobs"]["content"][0]["token"] == "token_id:4"
    assert out["usage"]["prompt_tokens"] == len(out["prompt_token_ids"])
    # the same observation after another history: the reply recorded for it
    _, out = rs.chat_reply(state, {"messages": [SYS, {"role": "user", "content": "x"}, t1], "tools": tools})
    assert out["stub_match"] == "last" and out["choices"][0]["message"]["tool_calls"][0]["id"] == "t11"
    _, out = rs.chat_reply(state, {"messages": [SYS, {"role": "user", "content": "new"}], "tools": tools, "max_tokens": 2})
    assert out["stub_match"] == "any" and out["choices"][0]["finish_reason"] == "length"
    assert out["usage"]["completion_tokens"] == 2 and not out["choices"][0]["message"]["tool_calls"]
    _, out = rs.chat_reply(state, {"messages": [{"role": "user", "content": "probe"}], "max_tokens": 32})
    assert out["choices"][0]["message"]["content"] == "ready"
    assert state.corpus.stats["errors"] == 0


def test_stub_http_tokenize_matches_usage_and_adapter_checks(tmp_path):
    game(tmp_path, "a1", 1)
    state = rs.State(rs.Corpus([tmp_path]), speed=0)
    server = ThreadingHTTPServer(("127.0.0.1", 0), rs.handler_for(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    root = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path, body):
        req = urllib.request.Request(root + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    probe = {"messages": [{"role": "user", "content": "probe"}], "max_tokens": 32}
    count = json.loads(post("/tokenize", {"messages": probe["messages"]})[1])["count"]
    assert json.loads(post("/v1/chat/completions", probe)[1])["usage"]["prompt_tokens"] == count
    assert post("/v1/load_lora_adapter", {"lora_name": "a", "lora_path": str(tmp_path / "none")})[0] == 400
    (tmp_path / "ad").mkdir()
    (tmp_path / "ad" / "adapter_config.json").write_text("{}")
    assert post("/v1/load_lora_adapter", {"lora_name": "a", "lora_path": str(tmp_path / "ad")})[0] == 200
    with urllib.request.urlopen(root + "/v1/models") as r:
        assert [m["id"] for m in json.loads(r.read())["data"]] == [rs.SERVED, "a"]
    post("/sleep", {})
    assert post("/v1/chat/completions", probe)[0] == 503
    post("/wake_up", {})
    assert json.loads(post("/reset_prefix_cache", {})[1])["success"] is True
    server.shutdown()
