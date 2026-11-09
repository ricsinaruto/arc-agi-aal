"""The request layer of the native-context preflight: chat payloads and deadline-bound /tokenize counts."""

import asyncio
import copy

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

    async def close(self):
        await self.client.aclose()
