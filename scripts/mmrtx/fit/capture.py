"""mmrtx fit-data capture: every successful upstream request's messages + tools + the reply, latest per conversation,
gzip-dumped (atomic) every ``every`` seconds and at the end. Conversation key = system + first user turn + first
assistant turn (a drain that drops the first assistant turn starts a new conversation; single-request prefixes are
kept and dropped offline when a longer conversation extends them)."""
import gzip
import hashlib
import json
import os
import threading
import time


def conv_key(messages):
    head = [m for m in messages if m.get("role") in ("system", "user")][:2]
    first = next((m for m in messages if m.get("role") == "assistant"), None)
    part = [[m.get("role"), m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))[:20000]]
            for m in head]
    if first is not None:
        part.append([str(first.get("reasoning") or "")[:4000], str(first.get("content") or "")[:4000],
                     json.dumps(first.get("tool_calls"), default=str)[:4000]])
    else:
        part.append(len(messages))
    return hashlib.sha1(json.dumps(part, default=str).encode()).hexdigest()[:20]


class Recorder:
    def __init__(self, path, every=120, on_flush=None):
        self.path, self.every, self.on_flush = str(path), every, on_flush
        self.convs, self.lock, self.errors, self.n = {}, threading.Lock(), 0, 0
        self.stop = threading.Event()

    def record(self, sent, data):
        msg = ((data.get("choices") or [{}])[0]).get("message") or {}
        reply = {k: msg.get(k) for k in ("content", "reasoning", "reasoning_content", "tool_calls") if msg.get(k)}
        row = {"t": time.time(), "messages": sent.get("messages"), "tools": sent.get("tools"),
               "reasoning": sent.get("reasoning"), "reply": reply, "usage": data.get("usage"),
               "finish": ((data.get("choices") or [{}])[0]).get("finish_reason")}
        k = conv_key(sent.get("messages") or [])
        with self.lock:
            self.convs[k] = row
            self.n += 1

    def flush(self):
        with self.lock:
            rows = [{"key": k, **v} for k, v in self.convs.items()]
            meta = {"requests": self.n, "errors": self.errors, "convs": len(rows), "t": time.time()}
        tmp = self.path + ".tmp"
        with gzip.open(tmp, "wt") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        os.replace(tmp, self.path)
        with open(self.path + ".meta.json", "w") as f:
            json.dump(meta, f)
        if self.on_flush:
            self.on_flush()

    def start(self):
        def loop():
            while not self.stop.wait(self.every):
                try:
                    self.flush()
                except Exception as exc:  # noqa: BLE001 - next pass
                    print(f"capture flush failed: {exc}", flush=True)
        threading.Thread(target=loop, daemon=True).start()
        return self


def install(api_proxy, recorder):
    """Wrap api_proxy.forward (looked up as a module global by serve_forever) to record 200 replies."""
    orig = api_proxy.forward

    def forward(body, target, key, *args, **kwargs):
        code, data = orig(body, target, key, *args, **kwargs)
        if code == 200:
            try:
                recorder.record(api_proxy.upstream_body(body, target), data)
            except Exception:  # noqa: BLE001 - counted
                recorder.errors += 1
        return code, data

    api_proxy.forward = forward
