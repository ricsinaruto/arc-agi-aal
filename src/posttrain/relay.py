"""Relay policy: the real arena Duck harness plays a game while an outside policy writes the replies.

Data source (b) of the post-training arm (Claude generates the trajectories
directly). ``RelayTransport`` stands in for the vLLM HTTP client of an ``ArenaTransferDuckAgent``:

- a ``/tokenize`` call is answered locally (characters / 3.2, a slight over-count that keeps the
  harness's compaction threshold conservative);
- a chat completion writes ``turn-NNNN.request.json`` (the exact payload: messages, tools, sampling)
  and ``turn-NNNN.view.txt`` (the newest user/tool messages, for the policy to read), then waits for
  ``turn-NNNN.response.json``: ``{"reasoning": str, "content": str, "code": str}`` (a python tool
  call) or ``{"reasoning": str, "content": str}`` (a plain reply).

The reply is returned in the vLLM shape (``reasoning`` beside ``content`` and ``tool_calls``), so
the harness logs it exactly as it logs Flash's replies, and the game folder (``log.jsonl``,
``duck-transcript.txt``, ``prompts/``) has the arena layout. Every request payload is kept, so
SFT records can use the exact messages the policy saw.
"""

import asyncio
import json
import time
from pathlib import Path

import httpx

TOKENS_PER_CHAR = 1 / 3.2


def estimate_tokens(payload):
    return int(len(json.dumps(payload.get("messages", payload), ensure_ascii=False)) * TOKENS_PER_CHAR) + 1


def view_text(payload, turn):
    """The messages since the policy's last reply (the new observation and tool results)."""
    messages = payload.get("messages", [])
    last = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    tail = messages[last + 1:] if last >= 0 else messages
    out = [f"=== turn {turn} | {len(messages)} messages | ~{estimate_tokens(payload)} prompt tokens ==="]
    for message in tail:
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(part.get("text", "[image]") for part in content if isinstance(part, dict))
        out.append(f"--- {message.get('role')} ---\n{content}")
    return "\n".join(out) + "\n"


def response_json(answer, turn, prompt_tokens):
    message = {"role": "assistant", "content": answer.get("content") or None, "reasoning": answer.get("reasoning") or None}
    finish = "stop"
    if answer.get("code") is not None:
        message["tool_calls"] = [{"id": f"relay-{turn}", "type": "function",
                                  "function": {"name": "python", "arguments": json.dumps({"code": answer["code"]})}}]
        finish = "tool_calls"
    completion = int(len((answer.get("reasoning") or "") + (answer.get("content") or "") + (answer.get("code") or ""))
                     * TOKENS_PER_CHAR) + 1
    return {"choices": [{"finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion}}


class RelayTransport:
    base_url = httpx.URL("http://relay/v1/")

    def __init__(self, folder, *, poll_seconds=0.5, timeout_seconds=24 * 3600, tokens_per_second=110.0,
                 request_overhead_seconds=1.65):
        self.folder = Path(folder)
        # Virtual time: the harness clocks see the policy's waiting time replaced by the time the
        # served model would take (overhead + completion tokens at the measured phase-2 stream rate).
        self.waited = 0.0
        self.modeled = 0.0
        self.tokens_per_second = tokens_per_second
        self.request_overhead_seconds = request_overhead_seconds
        self.folder.mkdir(parents=True, exist_ok=True)
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.turn = len(list(self.folder.glob("turn-*.request.json")))

    async def post(self, url, **kwargs):
        payload = kwargs.get("json") or {}
        request = httpx.Request("POST", str(url))
        if str(url).endswith("/tokenize"):
            return httpx.Response(200, request=request, json={"count": estimate_tokens(payload), "max_model_len": 262144})
        self.turn += 1
        stem = self.folder / f"turn-{self.turn:04d}"
        stem.with_suffix(".request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        stem.with_suffix(".view.txt").write_text(view_text(payload, self.turn))
        (self.folder / "PENDING").write_text(f"{self.turn}\n")
        answer_path = stem.with_suffix(".response.json")
        started = time.monotonic()
        while not answer_path.exists():
            if time.monotonic() - started > self.timeout_seconds:
                raise TimeoutError(f"no relay answer for turn {self.turn}")
            await asyncio.sleep(self.poll_seconds)
        self.waited += time.monotonic() - started
        answer = json.loads(answer_path.read_text())
        (self.folder / "PENDING").unlink(missing_ok=True)
        body = response_json(answer, self.turn, estimate_tokens(payload))
        self.modeled += self.request_overhead_seconds + body["usage"]["completion_tokens"] / self.tokens_per_second
        return httpx.Response(200, request=request, json=body)

    def now(self):
        """The virtual monotonic clock for the harness (see ``__init__``)."""
        return time.monotonic() - self.waited + self.modeled
