"""Closed-loop replay driver (one process of several): each stream replays recorded attempts' live requests in order
(payload rebuilt from the recorder's keep/messages deltas; tools, template and sampling as recorded; the recorder's
logprob and token-id fields; max_tokens = min_tokens = the recorded completion length), one request in flight per
stream, until the stop file appears. One record per finished request.

    python driver.py IDX NPROC STREAMS MODEL ROOT DATA OUT STOP
"""
import asyncio
import glob
import json
import sys
import time

import httpx

idx, nproc, streams = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
model, root, data, out, stop = sys.argv[4:9]
MODELS = model.split(",")  # several adapters: each replayed attempt keeps one, round-robin over attempts
NO_MIN = __import__("os").environ.get("FE_NO_MIN") == "1"  # natural reply ends (speculative-decoding arms)


def load(path):
    payload, tools, reqs = [], None, []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            payload = payload[:r.get("keep", len(payload))] + r["messages"]
            if "tools" in r:
                tools = r["tools"]
            if not r.get("scripted"):
                s = r.get("sampling") or {}
                body = {"payload": payload, "tools": tools, "template": r.get("template") or {},
                        "sampling": {k: s[k] for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                                                       "frequency_penalty", "repetition_penalty") if s.get(k) is not None},
                        "ct": max(1, int(r["usage"]["completion_tokens"])), "purpose": r.get("purpose")}
                reqs.append(body)
            del r
    return reqs


def body_of(q, model=MODELS[0]):
    b = {"model": model, "messages": q["payload"], "chat_template_kwargs": q["template"], **q["sampling"],
         "max_tokens": q["ct"], "min_tokens": 0 if NO_MIN else q["ct"], "logprobs": True, "top_logprobs": 20,
         "return_tokens_as_token_ids": True, "return_token_ids": True}
    if q["tools"]:
        b["tools"] = q["tools"]
        if q["purpose"] == "summary":
            b["tool_choice"] = "none"
    return b


async def main():
    pat = __import__("os").environ.get("FE_DATA_GLOB", "r12-*")
    files = sorted(glob.glob(f"{data}/{pat}/*.jsonl"))[idx::nproc]
    attempts = [load(f) for f in files]
    attempts = [a for a in attempts if a]
    print(json.dumps({"driver": idx, "attempts": len(attempts), "requests": sum(len(a) for a in attempts)}),
          flush=True)
    order = list(range(len(attempts)))
    nxt = [0]
    sink = open(out, "a")
    done = asyncio.Event()

    async def stream(s, client):
        while not done.is_set():
            a = order[nxt[0] % len(order)]
            nxt[0] += 1
            reqs = attempts[a]
            # MFU-1 mm-cache A/B: resumed tries start mid-game (their first request carries every earlier frame)
            reqs = reqs[int(len(reqs) * float(__import__("os").environ.get("FE_START_FRAC", "0"))):]
            m = MODELS[(idx + a * nproc) % len(MODELS)]
            for q in reqs:
                if done.is_set():
                    return
                t0 = time.time()
                rec = {"s": s, "a": a, "m": m, "t0": round(t0, 3), "ct_want": q["ct"]}
                try:
                    r = await client.post(root + "/v1/chat/completions", json=body_of(q, m))
                    rec["status"] = r.status_code
                    if r.status_code == 200:
                        u = r.json().get("usage") or {}
                        rec.update(pt=u.get("prompt_tokens"), ct=u.get("completion_tokens"),
                                   cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"))
                    else:
                        rec["err"] = r.text[:300]
                except Exception as exc:  # noqa: BLE001
                    rec["status"], rec["err"] = -1, f"{type(exc).__name__}: {exc}"[:300]
                rec["t1"] = round(time.time(), 3)
                sink.write(json.dumps(rec) + "\n")
                sink.flush()
                if rec["status"] != 200:
                    await asyncio.sleep(1)

    async def stopper():
        import os
        while not os.path.exists(stop):
            await asyncio.sleep(1)
        done.set()

    limits = httpx.Limits(max_connections=streams + 4, max_keepalive_connections=streams + 4,
                          keepalive_expiry=2.0)  # under the server's 5 s keep-alive, as the loop (fe2/fe3: ReadErrors)
    async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(3600, connect=30), limits=limits) as client:
        tasks = [asyncio.create_task(stream(s, client)) for s in range(streams)]
        st = asyncio.create_task(stopper())
        await done.wait()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        st.cancel()


asyncio.run(main())
