"""A hosted model behind the endpoints run_arc2 reads from vLLM (model screens): /v1/models,
/tokenize (an estimate: text characters / 3.2 + 300 an image; the gate and the context guard only),
/metrics (vllm:generation_tokens_total from the replies' usage) and /v1/chat/completions, forwarded to an
OpenAI-compatible API (OpenRouter) with the study's effort as ``reasoning.effort`` (xhigh sent as high).
Earlier reasoning goes back as each assistant message's ``reasoning`` (a provider that refuses it gets the
request again without it). 429 / 5xx / transport errors are retried with backoff (at most ``RETRIES``)."""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("ARC2_API_BASE", "https://openrouter.ai/api/v1")
RETRIES = int(os.environ.get("ARC2_API_RETRIES", "8"))
EFFORTS = {"xhigh": "high", "high": "high", "medium": "medium", "low": "low"}


def estimate(messages):
    """Prompt tokens, roughly: text characters / 3.2 plus 300 an image part."""
    chars, images = 0, 0
    for message in messages or []:
        content = message.get("content")
        parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "image_url":
                images += 1
            else:
                chars += len(part.get("text") or "" if isinstance(part, dict) else str(part))
        chars += len(message.get("reasoning") or "") + len(json.dumps(message.get("tool_calls") or ""))
    return int(chars / 3.2) + 300 * images + 8 * len(messages or [])


def upstream_body(body, target, keep_reasoning=True):
    """The request as the hosted API takes it."""
    effort = (body.get("chat_template_kwargs") or {}).get("reasoning_effort") or "high"
    messages = []
    for message in body.get("messages") or []:
        message = {k: v for k, v in message.items() if k != "reasoning_content"}
        if not keep_reasoning:
            message.pop("reasoning", None)
        messages.append(message)
    out = {"model": target, "messages": messages, "max_tokens": body.get("max_tokens"),
           "temperature": body.get("temperature"), "top_p": body.get("top_p"),
           "reasoning": {"effort": EFFORTS.get(effort, "high")}, "usage": {"include": True}}
    if body.get("tools"):
        out.update(tools=body["tools"], tool_choice=body.get("tool_choice", "auto"))
    return {k: v for k, v in out.items() if v is not None}


def reply_body(data):
    """The upstream reply as run_arc2 reads a vLLM one: reasoning as ``reasoning_content`` too."""
    for choice in data.get("choices") or []:
        message = choice.get("message") or {}
        if message.get("reasoning") and not message.get("reasoning_content"):
            message["reasoning_content"] = message["reasoning"]
    return data


def post(body, key, timeout):
    request = urllib.request.Request(UPSTREAM + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def forward(body, target, key, timeout=1800, log=None):
    """(status, reply) after retries; the reasoning in the history is dropped once if the provider refuses it."""
    keep, delay, last = True, 4.0, None
    for attempt in range(RETRIES + 1):
        try:
            status, data = post(upstream_body(body, target, keep), key, timeout)
            if data.get("error"):  # some providers answer 200 with an error object
                raise urllib.error.HTTPError(UPSTREAM, int(data["error"].get("code") or 502), json.dumps(data["error"]),
                                             None, None)
            if not (data.get("choices") or [{}])[0].get("message"):
                raise urllib.error.HTTPError(UPSTREAM, 502, "no message in the reply", None, None)
            return 200, reply_body(data)
        except urllib.error.HTTPError as error:
            text = error.read().decode(errors="replace")[:2000] if error.fp else str(error.msg)
            last = (error.code, text)
            if error.code == 400 and keep and "reasoning" in text.lower():
                keep = False
                continue
            # an upstream provider's own 400 ("Provider returned error") is transient on shared/free routes: retried
            if error.code in (400, 401, 403, 404) and "context" not in text.lower() and "provider returned" not in text.lower():
                break
            if error.code == 400 and "context" in text.lower():
                return 400, {"error": text}
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as error:
            last = (599, f"{type(error).__name__}: {error}"[:2000])
        if log:
            log({"event": "api_retry", "attempt": attempt, "status": last[0], "body": last[1][:300]})
        time.sleep(delay)
        delay = min(delay * 2, 120.0)
    return (502 if last[0] in (599, 429) else last[0]), {"error": last[1]}


def serve_forever(port, target, key, served="flash", log_path=None):
    state = {"generated": 0, "requests": 0, "errors": 0}
    lock = threading.Lock()

    def log(row):
        if log_path:
            with lock, open(log_path, "a") as sink:
                sink.write(json.dumps({"t": round(time.time(), 1), **row}) + "\n")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body, code=200, kind="application/json"):
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/metrics"):
                return self.reply(f"vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
                                  f"vllm:kv_cache_usage_perc 0.1\nvllm:generation_tokens_total {state['generated']}\n",
                                  kind="text/plain")
            if self.path.startswith("/v1/models"):
                return self.reply({"data": [{"id": served}]})
            return self.reply({"ok": True})

        def do_POST(self):
            size = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(size) or b"{}")
            path = self.path.split("?")[0]
            if path == "/reset_prefix_cache":
                return self.reply({"success": True})
            if path == "/tokenize":
                count = estimate(body.get("messages"))
                return self.reply({"count": count, "tokens": [], "max_model_len": 262144})
            if path == "/v1/chat/completions":
                began = time.time()
                code, data = forward(body, target, key, log=log)
                usage = data.get("usage") or {}
                with lock:
                    state["requests"] += 1
                    state["generated"] += int(usage.get("completion_tokens") or 0)
                    state["errors"] += code != 200
                log({"event": "reply", "status": code, "seconds": round(time.time() - began, 1),
                     "prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens"),
                     "reasoning": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                     "cost": usage.get("cost"), "finish": ((data.get("choices") or [{}])[0]).get("finish_reason"),
                     **({"error": str(data.get("error"))[:300]} if code != 200 else {})})
                return self.reply(data, code)
            return self.reply({"error": f"unknown path {path}"}, 404)

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
