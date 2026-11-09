"""One asynchronous local model client shared by an entire game fleet."""

import asyncio
import time

import httpx


class ModelClient:
    def __init__(self, config, tool_schema=None):
        self.config = config
        self.tool_schema = tool_schema
        self.client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/") + "/",
            limits=httpx.Limits(max_connections=64),
            trust_env=False,
        )
        self.slots = asyncio.Semaphore(config.parallel_games)

    async def close(self):
        await self.client.aclose()

    def template_kwargs(self, no_thinking=False):
        return {
            "enable_thinking": not no_thinking,
            "preserve_thinking": self.config.preserve_reasoning,
            "reasoning_effort": self.config.reasoning_effort,
        }

    async def count_tokens(
        self, messages, deadline, *, no_thinking=False, allow_tools=True, tools=None,
        template_kwargs=None, continue_final_message=False,
    ):
        """Use the serving tokenizer, including the actual tools and chat template.

        ``tools`` (Duck v5b) counts with an explicit tool schema instead of ``self.tool_schema``.
        ``template_kwargs`` (Duck v5d) counts with the request's own ``chat_template_kwargs``:
        the template renders every earlier reasoning block unless ``preserve_thinking`` is
        false, so a count made with different kwargs misses thousands of tokens.
        ``continue_final_message`` counts a prefilled request (the arena's compaction summary)
        as it is served: the last assistant message continued, no generation prompt.
        """
        payload = {
            "model": self.config.model,
            "messages": messages,
            "chat_template_kwargs": (
                dict(template_kwargs) if template_kwargs is not None
                else self.template_kwargs(no_thinking)
            ),
        }
        if continue_final_message:
            payload.update(add_generation_prompt=False, continue_final_message=True)
        if tools is not None:
            payload["tools"] = tools
        elif allow_tools and self.tool_schema is not None:
            payload["tools"] = self.tool_schema
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Gameplay deadline reached before token counting")
        if getattr(self.config, "serving_backend", "vllm") == "llamacpp":
            # llama-server renders the same template, tools and kwargs and counts the prompt; its
            # per-slot window is validated once at startup (arena.llamacpp.SlotCapacity).
            url = self.client.base_url.copy_with(path="/v1/chat/completions/input_tokens")
            response = await self.client.post(url, json=payload, timeout=remaining)
            response.raise_for_status()
            return int(response.json()["input_tokens"])
        url = self.client.base_url.copy_with(path="/tokenize")
        try:
            response = await self.client.post(url, json=payload, timeout=remaining)
        except httpx.TimeoutException:
            if time.monotonic() >= deadline - 1:
                raise TimeoutError("Gameplay deadline reached during token counting") from None
            raise
        response.raise_for_status()
        data = response.json()
        if data["max_model_len"] < self.config.max_model_len:
            raise ValueError("Serving context is smaller than the configured context")
        return data["count"]
