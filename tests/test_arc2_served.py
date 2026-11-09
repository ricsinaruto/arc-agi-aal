import json

from arc2 import served


def test_served_log_rebuilds_every_payload_byte_exact_across_resets_images_and_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(served, "CHUNK_BYTES", 600)
    log = served.ServedLog(tmp_path)
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 200}}
    head = [{"role": "system", "content": "sys"}, {"role": "user", "content": [{"type": "text", "text": "task"}, img]}]
    sent = []
    msgs = list(head)
    for k in range(6):
        if k == 3:  # a new conversation (reopen, drain): shorter, different tail
            msgs = list(head) + [{"role": "user", "content": "fresh"}]
        payload = {"model": "m", "messages": [dict(m) for m in msgs], "max_tokens": 100 + k,
                   "chat_template_kwargs": {"enable_thinking": True}}
        sent.append(json.dumps(payload))
        log.write("t#1", payload, {"choices": [{"message": {"role": "assistant", "content": f"r{k}"},
                                                 "finish_reason": "stop"}], "usage": {"prompt_tokens": k}}, 0, 1)
        msgs = msgs + [{"role": "assistant", "content": f"r{k}"}, {"role": "tool", "content": f"out{k}"}]
    log.write("t#2", {"model": "m", "messages": head, "max_tokens": 1}, {"choices": [{"message": {}}]}, 0, 1)
    rows = list(served.rebuild(tmp_path))
    assert [json.dumps(p) for r, p in rows if r["task"] == "t#1"] == sent
    assert len(list(tmp_path.glob("served-[0-9]*.jsonl"))) > 1
    assert len((tmp_path / "served-images.jsonl").read_text().splitlines()) == 1
    assert [r["keep"] for r, _ in rows if r["task"] == "t#1"] == [0, 2, 4, 2, 3, 5]
    assert rows[-1][0]["task"] == "t#2" and rows[-1][0]["keep"] == 0


def test_served_log_prompt_ids_delta_chain_and_reply_link(tmp_path):
    log = served.ServedLog(tmp_path)
    ids = [list(range(10)), list(range(14)), [0, 1, 2, 99]]
    for k, p in enumerate(ids):
        log.write("t", {"model": "m", "messages": [{"role": "user", "content": str(k)}], "max_tokens": 5},
                  {"choices": [{"message": {"content": "x"}, "token_ids": [7, 8]}], "prompt_token_ids": p}, 0, 1,
                  reply=k + 1)
    rows = [r for r, _ in served.rebuild(tmp_path)]
    assert [r["prompt_token_ids"] for r in rows] == ids and [r["reply"] for r in rows] == [1, 2, 3]
    assert [r["prompt_ids"]["keep"] for r in rows] == [0, 10, 3] and rows[0]["token_ids"] == [7, 8]
    chunk = next(tmp_path.glob("served-0*.jsonl"))
    lines = chunk.read_text().splitlines()
    bad = json.loads(lines[1]); bad["prompt_ids"]["append"][0] = 5
    chunk.write_text("\n".join([lines[0], json.dumps(bad), lines[2]]) + "\n")
    try:
        list(served.rebuild(tmp_path))
        raise AssertionError("a broken chain must raise")
    except ValueError:
        pass
