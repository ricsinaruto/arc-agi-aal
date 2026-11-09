"""Metering proxy between every session and the served model (fixed; runs in the root controller).

Sessions reach the model only here (their own network is closed to everything but loopback, ``rsi_loop.sandbox``).
Each session gets its own key; every request is logged with its tokens. A request naming any model other than the
served one is refused and counted (``non_flash``): with the egress counters that is the frontier-call audit of
docs/rsi.md section 7. One translation: an OpenAI ``reasoning_effort`` moves into the chat template's kwargs, where
Flash reads it (``enable_thinking`` and ``preserve_thinking`` on unless the request sets them).
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

COUNTERS = ("requests", "prompt_tokens", "completion_tokens", "cached_tokens", "errors", "non_flash", "unknown_key",
            "refused_path", "capped")
# The model endpoints only: the server's other routes (cache resets, sleep, collective RPC in vLLM's dev mode) stay
# with the root controller.
ALLOWED = {("POST", "/v1/chat/completions"), ("POST", "/v1/completions"), ("POST", "/tokenize"),
           ("POST", "/detokenize"), ("GET", "/v1/models"), ("GET", "/health"), ("GET", "/version")}


class Gate:
    """Opt-in KV admission gate (a candidate's ``serving/study.json`` ``rsi_gate``): a model request waits, first come
    first served, until its estimated tokens fit in ``capacity`` next to the requests in flight; a request alone is
    always admitted, so nothing waits forever."""

    def __init__(self, capacity):
        self.capacity, self.used, self.inflight, self.queue = int(capacity), 0, 0, deque()
        self.waited_s = 0.0

    def _fits(self, cost):
        return self.inflight == 0 or self.used + cost <= self.capacity

    async def acquire(self, cost):
        if not self.queue and self._fits(cost):
            self.used, self.inflight = self.used + cost, self.inflight + 1
            return
        started, future = time.time(), asyncio.get_running_loop().create_future()
        self.queue.append((cost, future))
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self.release(cost)  # admitted just as the caller went away
            else:
                self.queue = deque(item for item in self.queue if item[1] is not future)
            raise
        finally:
            self.waited_s += time.time() - started

    def release(self, cost):
        self.used, self.inflight = max(0, self.used - cost), max(0, self.inflight - 1)
        while self.queue and self._fits(self.queue[0][0]):
            cost, future = self.queue.popleft()
            if future.done():
                continue
            self.used, self.inflight = self.used + cost, self.inflight + 1
            future.set_result(True)

    def view(self):
        return {"capacity_tokens": self.capacity, "reserved_tokens": self.used, "in_flight": self.inflight,
                "waiting": len(self.queue), "waited_hours": round(self.waited_s / 3600, 3)}


class Proxy:
    def __init__(self, upstream, served_model, log_dir, port, unix_path=None):
        self.upstream, self.served, self.port = upstream.rstrip("/"), served_model, port
        self.unix_path = unix_path  # sessions in their own network namespaces reach the proxy here
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.keys = {}  # key -> {"session": id, "meter": "loop" | "heldout", ...}
        self.totals = defaultdict(lambda: dict.fromkeys(COUNTERS, 0))
        self.inflight = defaultdict(int)
        self.runner = self.client = None
        self.gate = None  # Gate when the live stack opts in
        self.last_tokens = {}  # session -> prompt + completion tokens of its last request (the gate's estimate)
        self.last_activity = {}  # session -> when its last model request started or ended

    def register(self, key, session, meter, **info):
        self.keys[key] = {"session": session, "meter": meter, **info}

    def forget(self, key):
        self.keys.pop(key, None)

    def session_totals(self, session):
        return dict(self.totals[session])

    async def start(self):
        self.client = ClientSession(timeout=ClientTimeout(total=None, sock_read=3600))
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", self.port).start()
        if self.unix_path:
            import os
            path = Path(self.unix_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(path.parent, 0o755)
            path.unlink(missing_ok=True)
            await web.UnixSite(self.runner, str(path)).start()
            os.chmod(path, 0o666)  # the per-session key decides who it is

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()
        if self.client:
            await self.client.close()

    def _log(self, session, row):
        with (self.log_dir / f"{session}.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    async def handle(self, request):
        auth = request.headers.get("Authorization", "")
        key = auth.removeprefix("Bearer ").strip()
        owner = self.keys.get(key)
        if owner is None:
            self.totals["_unknown"]["unknown_key"] += 1
            return web.json_response({"error": {"message": "unknown key"}}, status=401)
        session = owner["session"]
        path = "/" + request.match_info["tail"].lstrip("/")
        if (request.method, path) not in ALLOWED:
            self.totals[session]["refused_path"] += 1
            self._log(session, {"t": time.time(), "refused_path": f"{request.method} {path}"[:200]})
            return web.json_response({"error": {"message": f"{request.method} {path} is not served here"}}, status=403)
        body = await request.read()
        payload = None
        if body and request.content_type == "application/json":
            payload = json.loads(body)
            model = payload.get("model")
            if model is not None and model != self.served:
                self.totals[session]["non_flash"] += 1
                self._log(session, {"t": time.time(), "refused_model": model})
                return web.json_response({"error": {"message": f"only {self.served} is served"}}, status=404)
            effort = payload.pop("reasoning_effort", None)
            if effort is not None:
                kwargs = payload.setdefault("chat_template_kwargs", {})
                kwargs.setdefault("reasoning_effort", effort)
                kwargs.setdefault("enable_thinking", effort not in ("none", "off"))
                kwargs.setdefault("preserve_thinking", True)
            if payload.get("stream"):
                payload.setdefault("stream_options", {})["include_usage"] = True
            body = json.dumps(payload).encode()
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length", "authorization", "transfer-encoding")}
        url = self.upstream + "/" + request.match_info["tail"]
        if request.query_string:
            url += "?" + request.query_string
        cap = owner.get("token_cap")
        if cap and payload is not None and path in ("/v1/chat/completions", "/v1/completions"):
            left = int(cap) - self.totals[session]["completion_tokens"]
            if left <= 0:
                self.totals[session]["capped"] += 1
                return web.json_response({"error": {"message": "this game's token cap is reached"}}, status=429)
            for name in ("max_tokens", "max_completion_tokens"):
                if name in payload or name == "max_tokens":
                    payload[name] = min(int(payload.get(name) or 16384), left)
            body = json.dumps(payload).encode()
        gate, cost = self.gate, 0
        started = time.time()
        self.last_activity[session] = started
        self.inflight[session] += 1  # from arrival: a request waiting at the KV gate is in flight (idle rule)
        usage, finish, status = None, None, None
        acquired = False
        try:
            if gate is not None and payload is not None and path in ("/v1/chat/completions", "/v1/completions"):
                reply = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 16384)
                cost = min(gate.capacity, max(self.last_tokens.get(session, 0), len(body) // 4) + reply)
                await gate.acquire(cost)
                acquired = True
            else:
                gate = None
            async with self.client.request(request.method, url, data=body, headers=headers) as upstream:
                status = upstream.status
                response = web.StreamResponse(status=status, headers={
                    k: v for k, v in upstream.headers.items()
                    if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")})
                await response.prepare(request)
                tail, whole = b"", "event-stream" not in upstream.headers.get("content-type", "")
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                    tail = tail + chunk if whole else (tail + chunk)[-65536:]
                await response.write_eof()
            usage, finish = _usage(tail)
            return response
        except Exception as exc:  # noqa: BLE001 - logged and re-raised as a 502 to the session
            self.totals[session]["errors"] += 1
            self._log(session, {"t": started, "error": f"{type(exc).__name__}: {exc}"[:300]})
            return web.json_response({"error": {"message": "upstream failure"}}, status=502)
        finally:
            if gate is not None and acquired:
                gate.release(cost)
            self.inflight[session] -= 1
            self.last_activity[session] = time.time()
            totals = self.totals[session]
            totals["requests"] += 1
            if status and status >= 400:
                totals["errors"] += 1
            if usage:
                self.last_tokens[session] = int(usage.get("prompt_tokens") or 0) + int(
                    usage.get("completion_tokens") or 0)
                totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                totals["cached_tokens"] += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            if payload is not None:
                self._log(session, {"t": started, "s": round(time.time() - started, 3), "path": request.path,
                                    "status": status, "usage": usage, "finish": finish})


def _usage(tail):
    """Usage and finish reason from the end of a JSON body or an SSE stream."""
    text = tail.decode("utf-8", "replace")
    usage = finish = None
    candidates = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")] or [text]
    for item in candidates:
        if not item or item == "[DONE]":
            continue
        try:
            data = json.loads(item)
        except ValueError:
            continue
        usage = data.get("usage") or usage
        for choice in data.get("choices") or []:
            finish = choice.get("finish_reason") or finish
    return usage, finish
