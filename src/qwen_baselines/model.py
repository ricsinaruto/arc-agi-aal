"""Shared exact accounting, admission and deadline-aware requests."""

import asyncio
import base64
import copy
import json
import time
from contextlib import nullcontext

import httpx

from arc_infra.model_templates import request_messages, template_kwargs


class RequestLayer:
    def __init__(self, config, clock, *, client=None):
        self.config, self.clock = config, clock
        self.client = client or httpx.AsyncClient(
            base_url=config.base_url.rstrip("/") + "/", timeout=None, trust_env=False
        )
        self.slots = asyncio.Semaphore(config.generation_slots)
        self.active = 0
        self.tokenizer_retries = 0

    def template_settings(self, *, summary=False):
        return template_kwargs(self.config.model, {
            "enable_thinking": not summary,
            "preserve_thinking": self.config.preserve_thinking,
            "reasoning_effort": self.config.reasoning_effort,
        })

    def payload(self, messages, tools=None, *, summary=False):
        body = {
            "model": self.config.model,
            "messages": copy.deepcopy(request_messages(self.config.model, messages)),
            "temperature": self.config.temperature,
            "top_p": 0.95,
            "top_k": 20,
            "chat_template_kwargs": self.template_settings(summary=summary),
        }
        if tools:
            body["tools"] = tools
        return body

    def fresh(self, payload):
        payload = copy.deepcopy(payload)
        # Separate user feedback avoids changing the stable system prefix or any tool pair.
        # The tool_response wrapper keeps the pinned template's "last query" at the last
        # real observation, so in-turn reasoning survives when preserve_thinking is off.
        payload["messages"].append(
            {
                "role": "user",
                "content": "<tool_response>\nActual gameplay clock: "
                + json.dumps(self.clock.feedback())
                + "\n</tool_response>",
            }
        )
        return payload

    async def count(self, payload):
        body = {
            k: payload[k]
            for k in ("model", "messages", "tools", "chat_template_kwargs")
            if k in payload
        }
        body["add_generation_prompt"] = True
        remaining = self.clock.remaining()
        if remaining <= 0:
            raise TimeoutError("Gameplay deadline reached before tokenization")
        # Tokenization has no model generation or environment side effect. A
        # dropped connection can safely retry this exact body, within the same
        # deadline. Never apply this recovery to chat/completions or tool actions.
        async with asyncio.timeout(remaining):
            for attempt in range(3):
                if attempt:
                    self.tokenizer_retries += 1
                try:
                    response = await self.client.post(
                        self.client.base_url.copy_with(path="/tokenize"), json=body
                    )
                    break
                except httpx.TransportError:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(.05 * (attempt + 1))
        if response.is_error:
            # Surface vLLM's validation detail; a bare status code hides the cause.
            raise RuntimeError(f"/tokenize returned {response.status_code}: {response.text[:2000]}")
        data = response.json()
        self.validate_tokenizer_context(data)
        return int(data["count"])

    def validate_tokenizer_context(self, data):
        if data.get("max_model_len") != self.config.context_window:
            raise RuntimeError(
                f"Native context preflight failed: tokenizer reports {data.get('max_model_len')}; requested {self.config.context_window}. No downgrade allowed."
            )

    async def preflight(self):
        return await self.count(
            self.payload([{"role": "user", "content": "Native context admission check."}])
        )

    def validate_admission(self, count, allowance):
        """Optional adapter check after refreshing timing and counting the sent payload."""

    def inference_scope(self, summary=False):
        return nullcontext()

    async def complete_payload(self, payload, archive, *, summary=False, stream=False, max_tokens=None):
        waiting = time.monotonic()
        async with asyncio.timeout(self.clock.remaining()):
            async with self.slots:
                waited = time.monotonic() - waiting
                self.active += 1
                try:
                    payload = self.fresh(payload)
                    count = await self.count(payload)
                    allowance = self.config.context_window - count
                    if allowance <= 0:
                        raise ValueError("Request exceeds native context after timing refresh")
                    if max_tokens:
                        # Retrodict v5: a per-request output cap (8192) below the allowance.
                        allowance = min(allowance, max_tokens)
                    payload.update(
                        # Pi's compaction summary must not hit the cap: a 'length' stop makes Pi abort the
                        # session (C/ar25 v3 died at 746 s under a 2,048 cap).
                        max_tokens=min(8192, allowance) if summary else allowance,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    if summary:
                        payload["chat_template_kwargs"] = self.template_settings(summary=True)
                    self.validate_admission(count, payload["max_tokens"])
                    archive.event(
                        "request",
                        seconds=self.clock.elapsed(),
                        waiting_seconds=waited,
                        input_tokens=count,
                        summary=summary,
                        payload=payload,
                    )
                    started = time.monotonic()
                    chunks = []
                    try:
                        with self.inference_scope(summary):
                            async with self.client.stream(
                                "POST", "chat/completions", json=payload
                            ) as response:
                                response.raise_for_status()
                                async for chunk in response.aiter_bytes():
                                    chunks.append(chunk)
                                    archive.event(
                                        "generation_chunk",
                                        seconds=self.clock.elapsed(),
                                        chunk_base64=base64.b64encode(chunk).decode(),
                                        summary=summary,
                                    )
                    except BaseException:
                        archive.event(
                            "generation_interrupted",
                            seconds=self.clock.elapsed(),
                            inference_seconds=time.monotonic() - started,
                            summary=summary,
                            completion_usage_known=False,
                        )
                        raise
                    content = b"".join(chunks)
                    response = httpx.Response(
                        200,
                        content=content if stream else json.dumps(assemble(content)).encode(),
                        request=response.request,
                    )
                    archive.event(
                        "response",
                        seconds=self.clock.elapsed(),
                        inference_seconds=time.monotonic() - started,
                        summary=summary,
                        response=response.text if stream else response.json(),
                    )
                    return response
                finally:
                    self.active -= 1

    async def complete(self, messages, tools, archive):
        response = await self.complete_payload(self.payload(messages, tools), archive)
        return response.json()["choices"][0]["message"]

    async def complete_full(self, messages, tools, archive, *, max_tokens=None):
        """The assembled completion: choices (message, finish_reason) and usage."""
        response = await self.complete_payload(
            self.payload(messages, tools), archive, max_tokens=max_tokens
        )
        return response.json()

    async def close(self):
        await self.client.aclose()


def assemble(content):
    """Reconstruct native reasoning and tool arguments from a streamed completion."""
    text = content.decode()
    if not text.startswith("data:"):
        return json.loads(text)  # Deterministic non-streaming mock endpoints.
    message = {"role": "assistant", "content": ""}
    tool_calls, usage, finish = {}, {}, None
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta", {})
            for key in ("content", "reasoning_content", "reasoning"):
                if delta.get(key):
                    message[key] = message.get(key, "") + delta[key]
            for call in delta.get("tool_calls", []):
                target = tool_calls.setdefault(
                    call["index"], {"type": "function", "function": {"name": "", "arguments": ""}}
                )
                if call.get("id"):
                    target["id"] = call["id"]
                for key, value in call.get("function", {}).items():
                    if value:
                        target["function"][key] = target["function"].get(key, "") + value
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}


class Conversation:
    """A/B trim complete exchanges; latest observation always survives."""

    def __init__(self, system, observation, archive):
        self.system = {"role": "system", "content": system}
        self.groups = [[observation]]
        self.latest = observation
        self.archive = archive

    def append(self, exchange, observation=None):
        self.archive.event("exchange", messages=exchange, observation=observation)
        self.groups.append(exchange)
        if observation is not None:
            # Observation stays attached to the exchange it follows for chronology.
            self.groups[-1].append(observation)
            self.latest = observation

    def messages(self):
        messages = [self.system]
        for group in self.groups:
            messages.extend(group)
        if not any(message is self.latest for message in messages):
            messages.append(self.latest)
        return messages

    async def maintain(self, layer, tools):
        limit = layer.config.context_window - layer.config.reserve_tokens
        history = getattr(layer.config, "history_tokens", None)
        if history:
            limit = min(limit, history)
        while await layer.count(layer.fresh(layer.payload(self.messages(), tools))) > limit:
            if len(self.groups) > 1:
                removed = self.groups.pop(0)
                self.archive.event("context_trim", messages=removed)
            elif self.groups and await self.reduce_output(layer, tools, limit):
                continue
            elif self.groups:
                removed = self.groups.pop(0)
                self.archive.event("context_trim", messages=removed)
            else:
                raise ValueError(
                    "Current observation and tool schema exceed the native input budget"
                )
        return self.messages()

    async def reduce_output(self, layer, tools, limit):
        """Fit newest Python output using actual tokens; full output stays archived."""
        for message in reversed(self.groups[-1]):
            if message.get("role") != "tool":
                continue
            try:
                result = json.loads(message["content"])
            except (TypeError, ValueError):
                continue
            output = result.get("output")
            if not output or "call_id" not in result:
                continue
            low, high = 0, len(output)
            result["output_truncated"] = True
            result["retrieval"] = f"transcript[{result['call_id']}]['output'] (full output)"

            def assign(length):
                result["output"] = output[:length]
                message["content"] = json.dumps(result)

            assign(0)
            if await layer.count(layer.fresh(layer.payload(self.messages(), tools))) > limit:
                self.archive.event("output_truncated", call_id=result["call_id"], kept_characters=0)
                return True
            while low < high:
                mid = (low + high + 1) // 2
                assign(mid)
                if await layer.count(layer.fresh(layer.payload(self.messages(), tools))) <= limit:
                    low = mid
                else:
                    high = mid - 1
            assign(low)
            self.archive.event("output_truncated", call_id=result["call_id"], kept_characters=low)
            return True
        return False
