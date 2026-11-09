"""Independent game sessions sharing one model server. Sessions never share game state, scores or KV cache."""

import asyncio
import time

import httpx

from .model import ModelClient


# the latest finished generation request in this process (RL stall watchdog: replies still arriving = not stalled)
ACTIVITY = {"last_reply": 0.0, "replies": 0}


class SessionHTTP:
    """Keep frozen request construction; add only a replica cache salt and telemetry."""

    # ``post(..., record_extra={...})`` adds fields to the request's telemetry record (the arena's parallel samples tag
    # their ping, primary and siblings); never sent to the server.
    records_extra = True

    def __init__(self, shared, replica, instance_id, requests):
        self.shared = shared
        self.replica, self.instance_id, self.requests = replica, instance_id, requests
        self.base_url = shared.base_url
        self.tokenizer_retries = 0
        self.generation_retries = 0
        # pauses before each replay of a generation request after a transport failure; (1.0,): one replay (default).
        # RL benches (19 Oct): (1, 2, 4), after 12 of 60 base-reference recordings ended on ReadError.
        self.backoff = (1.0,)

    async def _post(self, url, **kwargs):
        response = await self.shared.post(url, **kwargs)
        if response.is_closed and response.is_stream_consumed:
            # HTTPX's bound stream references its Response. Break that cycle after the non-streaming read:
            # gc.freeze can otherwise retain completed requests and their large bodies for the whole run.
            response.stream = httpx.ByteStream(response.content)
        return response

    async def post(self, url, **kwargs):
        extra = kwargs.pop("record_extra", None)
        if str(url).rstrip("/").endswith("chat/completions"):
            # Different games within one replica still share their common instructions.
            # Copies of the SAME game cannot hit each other's KV cache. Stable per turn.
            payload = dict(kwargs["json"])
            payload["cache_salt"] = f"arc-public-replica-{self.replica}"
            kwargs["json"] = payload
            record = {"instance_id": self.instance_id, "started_monotonic": time.monotonic()}
            if extra:
                record.update(extra)
            record["image_count"] = sum(
                part.get("type") == "image_url"
                for message in payload.get("messages", [])
                if isinstance(message.get("content"), list)
                for part in message["content"]
                if isinstance(part, dict)
            )
            self.requests.append(record)
            try:
                try:
                    response = await self._post(url, **kwargs)
                except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as first:
                    # Replays (24 September): a transport failure on a generation request
                    # used to end the game for the whole trial (lp85 at 5 s, effort-low v1).
                    # A duplicate generation costs GPU time at worst; it issues no action.
                    exc, response = first, None
                    for pause in self.backoff:
                        remaining = kwargs["timeout"] - (time.monotonic() - record["started_monotonic"])
                        if remaining <= pause:
                            raise exc
                        record["transport_retry"] = type(exc).__name__
                        record["transport_retries"] = record.get("transport_retries", 0) + 1
                        self.generation_retries += 1
                        await asyncio.sleep(pause)
                        try:
                            response = await self._post(url, **{**kwargs, "timeout": remaining - pause})
                            break
                        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as again:
                            exc = again
                    if response is None:
                        raise exc
                record["status"] = response.status_code
                if not response.is_success:
                    record["response_error"] = response.text[:4000]
                if response.is_success:
                    data = response.json()
                    record["usage"] = data.get("usage", {})
                    record["finish_reason"] = data["choices"][0]["finish_reason"]
                return response
            except BaseException as exc:
                record["error"] = type(exc).__name__
                raise
            finally:
                record["seconds"] = time.monotonic() - record["started_monotonic"]
                ACTIVITY["last_reply"], ACTIVITY["replies"] = time.time(), ACTIVITY["replies"] + 1
        if str(url).rstrip("/").endswith("/tokenize"):
            # Read-only requests are safe to replay after transport failure. Never
            # replay an ambiguous generation: it may still be using the GPU.
            end = time.monotonic() + kwargs["timeout"]
            for attempt in range(3):
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Gameplay deadline reached during tokenizer recovery")
                try:
                    return await self._post(url, **{**kwargs, "timeout": remaining})
                except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError):
                    if attempt == 2:
                        raise
                    self.tokenizer_retries += 1
                    await asyncio.sleep(min(0.1 * (attempt + 1), max(0, end - time.monotonic())))
        return await self._post(url, **kwargs)


class SessionClient(ModelClient):
    def __init__(self, config, shared, slots, entry, requests, tool_schema=None):
        # No per-session transport or semaphore: every session shares these resources.
        self.config, self.tool_schema = config, tool_schema
        self.slots = slots
        self.client = SessionHTTP(shared, entry["replica"], entry["instance_id"], requests)

