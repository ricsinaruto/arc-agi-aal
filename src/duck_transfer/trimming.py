"""Context trimming that never discards the latest observation.

The public ``ToolAgent`` trims history from the oldest block until the request fits its
context budget, then drops everything before the first remaining user message. During a
long turn without actions (many Python-tool exchanges after one observation) the only
user message is the oldest block, so it is dropped and the whole history collapses to the
system prompt. This mixin keeps the newest observation and frees space behind and after
it instead.
"""

import base64
import binascii
import json
import math
import struct


def _role(message):
    return str(message.get("role", "")).strip()


def _text_tokens(text, chars_per_token=3.0):
    return int(len(text) / chars_per_token) + 1


# Qwen3-VL vision tokens (the served Qwen3.6/3.8 ``preprocessor_config.json``, sha256 27225450...: patch 16, merge 2,
# shortest_edge 65,536 px, longest_edge 16,777,216 px): the processor's ``smart_resize`` rounds each side to a multiple
# of 32 px (and rescales below the minimum or above the maximum pixel count), then one token per 32x32 pixels, plus
# the vision start/end markers. The v5b run confirmed 64 for the loop's 256x256 board images (64x64 cells at
# MULTIMODAL_UPSCALE 4); a 640x640 board (the arena's dfr_harness, upscale 10) is 400.
VISION_PIXELS_PER_TOKEN = 32
VISION_MARKER_TOKENS = 2
VISION_MIN_PIXELS = 65536
VISION_MAX_PIXELS = 16777216


def smart_resize(height, width, factor=VISION_PIXELS_PER_TOKEN, min_pixels=VISION_MIN_PIXELS,
                 max_pixels=VISION_MAX_PIXELS):
    """``(height, width)`` after the Qwen2-VL image processor's resize (transformers ``smart_resize``)."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def vision_tokens(width, height):
    """Vision tokens of a ``width`` x ``height`` image as the served processor counts them, plus the markers."""
    h_bar, w_bar = smart_resize(height, width)
    return (h_bar // VISION_PIXELS_PER_TOKEN) * (w_bar // VISION_PIXELS_PER_TOKEN) + VISION_MARKER_TOKENS


def png_dimensions(data_url):
    """``(width, height)`` from a PNG data URL's IHDR chunk, or ``None``."""
    if not isinstance(data_url, str):
        return None
    head, sep, payload = data_url.partition(",")
    if not sep or "image/png" not in head:
        return None
    try:
        raw = base64.b64decode(payload[:64] + "=" * (-len(payload[:64]) % 4), validate=False)
    except (binascii.Error, ValueError):
        return None
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", raw[16:24])
    return (width, height) if width and height else None


def image_tokens_for(part, fallback):
    """Vision tokens of one ``image_url`` part from its PNG geometry; ``fallback`` otherwise."""
    url = (part.get("image_url") or {}).get("url") if isinstance(part, dict) else None
    dims = png_dimensions(url)
    if dims is None:
        return int(fallback)
    width, height = dims
    return vision_tokens(width, height)


class ObservationPreservingTrim:
    TRUNCATION_MARK = "\n... [tool output truncated to fit the context budget]"
    # Vision tokens per attached board image (the public estimator scores a 256 px data URL
    # at about 3,300 tokens); overridden from Config.image_tokens_estimate by the bridge and
    # calibrated from the server's prompt_tokens (v5b).
    IMAGE_TOKENS = 400
    # Characters per text token; the v5 flat 3.0 under-counted grid-heavy prompts (lp85 ran at
    # 2.5), so the bridge calibrates it from the server tokenizer's count (v5b).
    CHARS_PER_TOKEN = 3.0

    def _ratio(self):
        return getattr(self, "_chars_per_token", self.CHARS_PER_TOKEN)

    def _message_tokens(self, message):
        content = message.get("content")
        ratio = self._ratio()
        tokens = 0
        if isinstance(content, str):
            tokens += _text_tokens(content, ratio)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    tokens += image_tokens_for(part, getattr(self, "_image_tokens", self.IMAGE_TOKENS))
                elif isinstance(part, dict):
                    tokens += _text_tokens(str(part.get("text", "")), ratio)
                else:
                    tokens += _text_tokens(str(part), ratio)
        rest = {key: value for key, value in message.items() if key != "content"}
        if rest:
            tokens += _text_tokens(
                json.dumps(rest, ensure_ascii=True, sort_keys=True, default=str), ratio
            )
        return tokens

    def _estimate_request_input_tokens(self, messages, *, tools=None):
        """Linear, memoised estimate: no whole-payload serialisation in the trim loop.

        The memo is keyed by message identity and the current calibration, so a ratio or
        image-token change re-estimates without clearing.
        """
        cache = self.__dict__.setdefault("_token_cache", {})
        stamp = (self._ratio(), getattr(self, "_image_tokens", self.IMAGE_TOKENS))
        total = 0
        for message in messages:
            key = id(message)
            hit = cache.get(key)
            if hit is None or hit[0] is not message or hit[2] != stamp:
                hit = cache[key] = (message, self._message_tokens(message), stamp)
            total += hit[1]
        if tools:
            key = id(tools)
            hit = cache.get(key)
            if hit is None or hit[0] is not tools or hit[2] != stamp:
                hit = cache[key] = (
                    tools,
                    _text_tokens(
                        json.dumps(tools, ensure_ascii=True, sort_keys=True, default=str), stamp[0]
                    ),
                    stamp,
                )
            total += hit[1]
        return total

    def _split_at_latest_observation(self, history):
        for index in range(len(history) - 1, -1, -1):
            if _role(history[index]) == "user":
                return history[:index], history[index:]
        return list(history), []

    def _reduce_once(self, prefix, tail):
        """Free one block: oldest history, then the oldest exchange after the observation,
        then the longest tool output; ``False`` when nothing more can go."""
        if prefix and self._drop_oldest_history_block(prefix, preserve_recent=0):
            return True
        exchanges = [i for i in range(1, len(tail)) if _role(tail[i]) == "assistant"]
        if len(exchanges) > 1:
            del tail[exchanges[0] : exchanges[1]]
            return True
        limit = max(256, self._tool_output_chars // 4)
        longest, length = None, limit
        for index, message in enumerate(tail):
            content = message.get("content")
            if _role(message) == "tool" and isinstance(content, str) and len(content) > length:
                longest, length = index, len(content)
        if longest is None:
            return False
        content = tail[longest]["content"]
        tail[longest] = {**tail[longest], "content": content[:limit] + self.TRUNCATION_MARK}
        return True

    def _trim_messages_for_context(
        self, messages, *, tools=None, preserve_recent=1, extra_safety_tokens=0
    ):
        if not messages:
            return []
        system_message = messages[0]
        prefix, tail = self._split_at_latest_observation(list(messages[1:]))
        budget_tokens = max(1, self._context_budget_tokens - max(0, extra_safety_tokens))
        while (
            self._estimate_request_input_tokens([system_message, *prefix, *tail], tools=tools)
            > budget_tokens
        ):
            if not self._reduce_once(prefix, tail):
                break
        return [system_message, *self._drop_until_first_user_message(prefix), *tail]

    def _force_reduce_messages(self, messages, *, preserve_recent=1):
        if not messages:
            return []
        system_message = messages[0]
        prefix, tail = self._split_at_latest_observation(list(messages[1:]))
        if not self._reduce_once(prefix, tail):
            return list(messages)
        return [system_message, *self._drop_until_first_user_message(prefix), *tail]
