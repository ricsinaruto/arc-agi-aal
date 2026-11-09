"""One asynchronous local model client shared by an entire game fleet."""

import asyncio
import base64
import struct
import time

import httpx
from pydantic import BaseModel

# ``text_images`` counting: the chat template renders an image part as this text, and the serving processor expands
# its one image_pad token to the image's token count.
IMAGE_TEXT = "<|vision_start|><|image_pad|><|vision_end|>"


class _TokenCount(BaseModel):
    # Skip the potentially 130K token IDs without allocating a Python list of integers.
    count: int
    max_model_len: int


def png_size(url):
    """(width, height) of a base64 PNG data URL from its IHDR header (the first 24 bytes), else None."""
    if not isinstance(url, str) or not url.startswith("data:image/png;base64,"):
        return None
    head = base64.b64decode(url[22:22 + 32])
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", head[16:24])


def text_image_messages(messages):
    """``messages`` with every image part replaced by its rendered text, and the images' (width, height) in order;
    None when an image is not a PNG data URL."""
    sizes, out = [], []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                size = png_size((part.get("image_url") or {}).get("url"))
                if size is None:
                    return None
                sizes.append(size)
                part = {"type": "text", "text": IMAGE_TEXT}
            parts.append(part)
        out.append({**message, "content": parts})
    return out, sizes


class ModelClient:
    image_tokens = None  # (width, height) -> the serving processor's token count (text_images), per client
    text_images_failed = None
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
        template_kwargs=None, continue_final_message=False, text_images=False,
    ):
        """Use the serving tokenizer, including the actual tools and chat template.

        ``tools`` (Duck v5b) counts with an explicit tool schema instead of ``self.tool_schema``.
        ``template_kwargs`` (Duck v5d) counts with the request's own ``chat_template_kwargs``:
        the template renders every earlier reasoning block unless ``preserve_thinking`` is
        false, so a count made with different kwargs misses thousands of tokens.
        ``continue_final_message`` counts a prefilled request (the arena's compaction summary)
        as it is served: the last assistant message continued, no generation prompt.
        ``text_images`` (B200, 5 October): the same count without the server's multimodal processing, which one API
        server runs on a single thread for every request: image parts are sent as the text the template renders for
        them, plus each image's token count less the one image_pad token (measured once per image size by the
        server itself).
        """
        if text_images and not self.text_images_failed:
            converted = text_image_messages(messages)
            if converted is not None and converted[1]:
                text_messages, sizes = converted
                try:
                    extra = 0
                    for size in sizes:
                        extra += await self.image_token_count(size, messages, deadline) - 1
                    count = await self.count_tokens(
                        text_messages, deadline, no_thinking=no_thinking, allow_tools=allow_tools, tools=tools,
                        template_kwargs=template_kwargs, continue_final_message=continue_final_message)
                    return count + extra
                except httpx.HTTPStatusError as exc:  # the server refused the text form: count as before from now on
                    self.text_images_failed = f"{exc.response.status_code}: {exc.response.text[:300]}"
                    print(f"ARC_TEXT_COUNT_DISABLED {self.text_images_failed}", flush=True)
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
        url = self.client.base_url.copy_with(path="/tokenize")
        try:
            response = await self.client.post(url, json=payload, timeout=remaining)
        except httpx.TimeoutException:
            if time.monotonic() >= deadline - 1:
                raise TimeoutError("Gameplay deadline reached during token counting") from None
            raise
        response.raise_for_status()
        data = _TokenCount.model_validate_json(response.content, strict=True)
        if data.max_model_len < self.config.max_model_len:
            raise ValueError("Serving context is smaller than the configured context")
        return data.count

    async def image_token_count(self, size, messages, deadline):
        """The serving processor's token count for an image of ``size``: a one-image message counted with the image
        and with its rendered text (a cached, exact difference plus the one image_pad token)."""
        if self.image_tokens is None:
            self.image_tokens = {}
        if size not in self.image_tokens:
            url = next(p["image_url"]["url"] for m in messages if isinstance(m.get("content"), list)
                       for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url"
                       and png_size(p["image_url"].get("url")) == size)
            kwargs = {"allow_tools": False, "template_kwargs": {}}
            image = await self.count_tokens([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}}]}], deadline, **kwargs)
            text = await self.count_tokens([{"role": "user", "content": [
                {"type": "text", "text": IMAGE_TEXT}]}], deadline, **kwargs)
            if image - text + 1 < 2:
                raise RuntimeError(f"image token count for {size}: {image} with the image, {text} as text")
            self.image_tokens[size] = image - text + 1
        return self.image_tokens[size]
