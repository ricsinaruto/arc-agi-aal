"""Scripted OpenAI-compatible server for the prime2 tests and preflight: the vLLM routes arc2.prime uses (/v1/models,
/tokenize, /metrics, streamed /v1/chat/completions). A top-level session loads its task, starts an rlm() child, submits
the test inputs as its answer (reporting a near-full context once, so Pi compacts), resubmits after a continuation,
then stops each turn; the child answers at once; summaries and refinement get short replies.

``conv=True`` (v4, conv1's workflow): the session checks the kernel's globals, runs one cell past the call limit,
prints twice to fill the context (one near-full report, so Pi compacts), stops once with no program (the harness's
"neither" reply), sends a program (the harness checks it: "Program 1"), calls ``program_1`` and submits. The step is
read from the STEP markers in the conversation (the summary carries the last one through a compaction)."""
import json
import re
import sys
import threading
import time

from aiohttp import web

CONTINUATION_START = "No human input is available in autonomous mode."
CHILD_TASK = "Reply with the word ok."
STEPS = [
    "import arc_task\nt = arc_task.load()\nprint(len(t['train']), len(t['test']))",
    "print('x' * 45000)",  # history long enough for Pi's cut (keeps 20,000 estimated tokens)
    "print('y' * 45000)",
    f"res = await rlm({CHILD_TASK!r})\nprint('CHILD', res.answer)",
    "import arc_task\nt = arc_task.load()\nprint(arc_task.submit([x['input'] for x in t['test']]))",
]
AFTER_CONTINUE = ("import arc_task\nt = arc_task.load()\n"
                  "print(arc_task.submit([[x['input'], [[0]]] for x in t['test']]))")
COMPLETION_TOKENS = 20
CONV_STEPS = [
    ("tool", "print('STEP0', len(train), len(test_inputs), np.__name__)"),
    ("tool", "import time\nprint('STEP1')\ntime.sleep(6)\nprint('not reached')"),
    ("tool", "print('STEP2' + 'x' * 45000)"),
    ("tool", "print('STEP3' + 'y' * 45000)"),
    ("text", "STEP4 I have not found the rule yet."),
    ("text", "STEP5 The rule:\n```python\ndef transform(grid):\n    return [row[::-1] for row in grid]\n```"),
    ("tool", "print('STEP6', program_1(test_inputs[0]))"),
    ("tool", "print('STEP7', submit(test_inputs[0] if len(test_inputs) == 1 else test_inputs, None))"),
    ("text", "STEP8 done"),
]
# ``cut=True`` (v5): after STEP0 one reply is cut at the cap with reasoning only; the next steps run only if that
# reasoning (its STEP1 marker) is back in the conversation
CONV_STEPS_CUT = [CONV_STEPS[0], ("cut", "STEP1 thinking about the grids at length")] + [
    (what, body.replace(f"STEP{k}", f"STEP{k + 1}")) for k, (what, body) in enumerate(CONV_STEPS) if k >= 1]
MARK = re.compile(r"STEP(\d+)")


class Fake:
    def __init__(self, big_prompt=120_000, log=None, fail_every=0, fail_kind="500", first_delay=0.0, down_at=None,
                 conv=False, cut=False):
        """Faults: every ``fail_every``-th chat request fails (``fail_kind`` "500": vLLM's EngineDead body; "drop": the
        connection closes before any byte); ``first_delay`` s before the first streamed byte (a prefill); from the
        ``down_at``-th chat request on, every request is refused (a dead server)."""
        self.big_prompt, self.log, self.conv = big_prompt, log, conv
        self.steps = CONV_STEPS_CUT if cut else CONV_STEPS
        self.cuts = 0
        self.first_requests = []  # conv: (system prompt, image parts in the first user message) per first request
        self.fail_every, self.fail_kind, self.first_delay, self.down_at = fail_every, fail_kind, first_delay, down_at
        self.generated = 0
        self.requests = []
        self.calls = self.failed = 0
        self.prefill_now = self.prefill_max = 0

    def text_of(self, message):
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        return str(content or "")

    def reply(self, body):
        msgs = body["messages"]
        system = self.text_of(msgs[0]) if msgs and msgs[0].get("role") in ("system", "developer") else ""
        users = [self.text_of(m) for m in msgs if m.get("role") == "user"]
        kind = "gameplay"
        if system.startswith("You are a context summarization assistant."):
            kind = "summary"
            seen = max((int(x) for x in MARK.findall(json.dumps(msgs))), default=-1) if self.conv else -1
            return kind, {"content": "## Goal\nsolve\n## Constraints & Preferences\n-\n## Progress\n-\n"
                                     "## Key Decisions\n-\n## Next Steps\n-\n## Critical Context\n-"
                                     + (f"\nlast STEP{seen}" if seen >= 0 else "")}, 0
        if system.startswith("You are Prime Agent's"):
            kind = "refine"
            return kind, {"content": json.dumps({"shouldRefine": False, "reason": "nothing to change",
                                                 "summary": "no edits", "edits": []})}, 0
        if any(CHILD_TASK in u for u in users[:1]):
            return "child", {"content": "ok"}, 0
        if self.conv:
            seen = max((int(x) for x in MARK.findall(json.dumps(msgs))), default=-1)
            if seen < 0:
                first = next(m for m in msgs if m.get("role") == "user")
                parts = first.get("content") if isinstance(first.get("content"), list) else []
                self.first_requests.append((system, sum(p.get("type") == "image_url" for p in parts)))
            step = min(seen + 1, len(self.steps) - 1)
            what, body = self.steps[step]
            big = 5 if self.steps is CONV_STEPS_CUT else 4
            prompt = self.big_prompt if step == big and not any("## Goal" in u for u in users) else 0
            if what == "cut":
                self.cuts += 1
                return kind, {"cut": body}, prompt
            return kind, ({"tool": body} if what == "tool" else {"content": body}), prompt
        tools = sum(1 for m in msgs if m.get("role") == "tool")
        continued = sum(CONTINUATION_START in u for u in users)
        if tools < len(STEPS):
            prompt = self.big_prompt if tools == 4 and not any("## Goal" in u for u in users) else 0
            return kind, {"tool": STEPS[tools]}, prompt
        if continued == 1 and tools == len(STEPS):
            return kind, {"tool": AFTER_CONTINUE}, 0
        return kind, {"content": "done"}, 0

    async def chat(self, request):
        body = await request.json()
        self.calls += 1
        if self.down_at is not None and self.calls >= self.down_at:
            self.failed += 1
            raise web.HTTPServiceUnavailable(text="down")
        if self.fail_every and self.calls % self.fail_every == 0:
            self.failed += 1
            if self.fail_kind == "drop":
                request.transport.close()
                raise ConnectionResetError("dropped")
            return web.json_response({"error": {"message": "EngineCore encountered an issue. See stack trace (above) "
                                                           "for the root cause."}}, status=500)
        self.prefill_now += 1
        self.prefill_max = max(self.prefill_max, self.prefill_now)
        try:
            await __import__("asyncio").sleep(self.first_delay)
        finally:
            self.prefill_now -= 1
        kind, out, prompt_override = self.reply(body)
        prompt = prompt_override or len(json.dumps(body["messages"])) // 4
        self.requests.append({"t": time.time(), "kind": kind, "kwargs": body.get("chat_template_kwargs"),
                              "temperature": body.get("temperature"), "top_p": body.get("top_p"),
                              "top_k": body.get("top_k"), "max_tokens": body.get("max_tokens"),
                              "model": body.get("model")})
        if self.log:
            self.log.write(json.dumps(self.requests[-1]) + "\n")
            self.log.flush()
        resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await resp.prepare(request)
        base = {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": body.get("model")}

        async def ev(d):
            await resp.write(("data: " + json.dumps(d) + "\n\n").encode())
        await ev({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "thinking"},
                                       "finish_reason": None}]})
        if "cut" in out:  # the cap: reasoning only, finish "length"
            await ev({**base, "choices": [{"index": 0, "delta": {"reasoning_content": " " + out["cut"]},
                                           "finish_reason": None}]})
            await ev({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
        elif "tool" in out:
            n = len(self.requests)
            call = {"index": 0, "id": f"c{n}", "type": "function",
                    "function": {"name": "ipython", "arguments": json.dumps({"code": out["tool"]})}}
            await ev({**base, "choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}]})
            await ev({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
        else:
            await ev({**base, "choices": [{"index": 0, "delta": {"content": out["content"]}, "finish_reason": None}]})
            await ev({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        self.generated += COMPLETION_TOKENS
        await ev({**base, "choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": COMPLETION_TOKENS,
                                                   "total_tokens": prompt + COMPLETION_TOKENS}})
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def tokenize(self, request):
        body = await request.json()
        return web.json_response({"count": len(json.dumps(body.get("messages"))) // 4})

    async def models(self, request):
        return web.json_response({"data": [{"id": "flash-served"}]})

    async def metrics(self, request):
        return web.Response(text=f"vllm:generation_tokens_total{{model=\"x\"}} {float(self.generated)}\n")

    async def health(self, request):
        return web.Response(text="ok")

    def app(self):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_post("/tokenize", self.tokenize)
        app.router.add_get("/v1/models", self.models)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_get("/health", self.health)
        return app


def serve_in_thread(port, **kw):
    """Start the scripted server on 127.0.0.1:port in a daemon thread; returns the Fake (its request log)."""
    import asyncio

    fake = Fake(**kw)
    ready = threading.Event()

    def main():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(fake.app(), access_log=None, handler_cancellation=True)  # a dropped client aborts its
        # request, as vLLM does (prefill_now counts live requests only)
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", port).start())
        ready.set()
        loop.run_forever()

    threading.Thread(target=main, daemon=True).start()
    ready.wait(10)
    return fake


if __name__ == "__main__":
    web.run_app(Fake(log=open(sys.argv[2], "a")).app(), host="127.0.0.1", port=int(sys.argv[1]), access_log=None)
