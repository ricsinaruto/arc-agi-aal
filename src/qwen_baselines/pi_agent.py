"""Actual Pi session with a local, shared-accounting model gateway."""

import asyncio
import base64
import json
import os
import secrets
import shutil
import traceback
from pathlib import Path

from aiohttp import web

from arc_infra.model_templates import text_only

from .evidence import png
from .executor import Executor
from .prompts import memory_block, system
from .sandbox import kill_group


class PiAgent:
    TOOLS = ("read", "write", "edit", "bash")

    def __init__(self, bridge, layer, *, pi_dir, node="node"):
        self.bridge, self.layer = bridge, layer
        self.pi_dir, self.node = Path(pi_dir).resolve(), node
        self.executor = self.make_executor()
        self.process = None
        self.web_runner = None
        self.pending = set()
        self.last_model_error = None
        # Gameplay requests since the current prompt began; the PRO-LONG agent caps them.
        self.turns = 0
        # Assistant text of the current prompt and the MEMORY: block carried between turns.
        self.text = ""
        self.memory = None

    def make_executor(self):
        return Executor(
            self.bridge,
            self.bridge.archive.root / "workspace",
            suppress_dumps=self.layer.config.suppress_grid_dumps,
        )

    def system_prompt(self):
        cfg = self.layer.config
        return system(
            "C",
            cfg.gameplay_seconds,
            actions_per_turn=cfg.actions_per_turn,
            object_view=cfg.object_view,
        )

    def object_view(self):
        cfg = self.layer.config
        if not cfg.object_view:
            return None
        return self.bridge.archive.object_view(limit=cfg.object_view_limit)

    def compaction_threshold(self):
        cfg = self.layer.config
        return cfg.compaction_tokens or cfg.context_window - cfg.reserve_tokens

    async def admit(self):
        """Gameplay-request hook; subclasses may refuse further tool turns."""
        self.turns += 1

    async def begin_turn(self):
        """Per-request boundary handling; the bridge auto-RESETs after GAME_OVER."""
        await self.bridge.begin_turn()

    async def gateway(self, request):
        try:
            body = await request.json()
            summary = not bool(body.get("tools"))
            payload = self.layer.payload(body["messages"], body.get("tools"), summary=summary)
            if not summary:
                await self.admit()
                before = self.bridge.archive.count()
                await self.begin_turn()
                if self.bridge.archive.count() > before:
                    payload["messages"].append(
                        {
                            "role": "user",
                            "content": "Runner reset receipt: "
                            + json.dumps(self.bridge.archive.metadata()),
                        }
                    )
                count = await self.layer.count(self.layer.fresh(payload))
                if count > self.compaction_threshold():
                    self.bridge.archive.event("pi_context_overflow", input_tokens=count)
                    # Pi's own overflow recovery invokes its native compaction machinery.
                    return web.json_response(
                        {
                            "error": {
                                "type": "invalid_request_error",
                                "code": "context_length_exceeded",
                                "message": "maximum context length exceeded: input crossed "
                                f"{self.compaction_threshold()}-token compaction threshold",
                            }
                        },
                        status=400,
                    )
            result = await self.layer.complete_payload(
                payload, self.bridge.archive, summary=summary, stream=bool(body.get("stream"))
            )
            return web.Response(
                body=result.content,
                content_type="text/event-stream" if body.get("stream") else "application/json",
            )
        except (TimeoutError, asyncio.CancelledError):
            return web.json_response(
                {"error": {"message": "Gameplay deadline reached"}}, status=408
            )
        except Exception as exc:
            # repr keeps the exception type when its message is empty (kernel v5: '500 {"message":""}').
            detail = f"{type(exc).__name__}: {exc}"
            self.bridge.archive.event(
                "pi_gateway_error", error=detail, traceback=traceback.format_exc()[-4000:]
            )
            print(f"[{self.bridge.label}] pi gateway error: {detail}", flush=True)
            return web.json_response({"error": {"message": detail}}, status=500)

    async def send(self, packet):
        self.process.stdin.write((json.dumps(packet) + "\n").encode())
        await self.process.stdin.drain()

    def gateway_runner(self, token):
        app = web.Application(client_max_size=128 * 1024**2)
        app.router.add_post(f"/{token}/v1/chat/completions", self.gateway)
        # A timed-out/aborted Pi request must not remain queued and later consume
        # inference for a disconnected client. RequestLayer's cancellation path
        # retains partial generation evidence and releases its admission slot.
        return web.AppRunner(app, handler_cancellation=True)

    async def start(self):
        await self.executor.prepare()
        token = secrets.token_hex(16)
        self.web_runner = self.gateway_runner(token)
        await self.web_runner.setup()
        site = web.TCPSite(self.web_runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        root = self.bridge.archive.root
        agent_dir = root / "pi-config"
        agent_dir.mkdir(exist_ok=True)
        node = shutil.which(self.node) or self.node
        self.process = await asyncio.create_subprocess_exec(
            node,
            str(self.pi_dir / "agent.mjs"),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=(root / "pi-stderr.log").open("wb"),
            start_new_session=True,
            env={
                "PATH": os.defpath,
                "HOME": str(agent_dir),
                "TMPDIR": str(self.executor.workspace),
                "PI_CODING_AGENT_DIR": str(agent_dir),
                "PI_SKIP_VERSION_CHECK": "1",
                "NO_COLOR": "1",
            },
            limit=16 * 1024**2,
        )
        await self.send(
            {
                "type": "init",
                "workspace": str(self.executor.workspace),
                "agentDir": str(agent_dir),
                "sessionDir": str(root / "pi-sessions"),
                "baseUrl": f"http://127.0.0.1:{port}/{token}/v1",
                "model": self.layer.config.model,
                "textOnly": text_only(self.layer.config.model),
                "contextWindow": self.layer.config.context_window,
                "system": self.system_prompt(),
                "tools": list(self.TOOLS),
            }
        )
        async with asyncio.timeout(60):
            await self.receive_until("ready")

    async def handle_rpc(self, message):
        try:
            if message["operation"] == "metadata":
                result = self.bridge.archive.metadata()
                view = self.object_view()
                if view:
                    result["object_view"] = view
                if "since" in message:
                    result["receipts"] = [
                        self.bridge.archive.metadata(i)
                        for i in range(message["since"] + 1, self.bridge.archive.count())
                    ]
            else:
                args = {k: v for k, v in message.items() if k not in {"type", "id", "operation"}}
                operation = message['operation']
                if operation == 'read_grid':
                    if not text_only(self.layer.config.model):
                        raise ValueError('read_grid is only available to text-only models')
                    from arena.text_observations import recorded_grid_path

                    args['path'] = recorded_grid_path(self.bridge.archive, self.executor.workspace, args['path'])
                    operation = 'read'
                result = await self.executor.execute(operation, **args)
            await self.send({"type": "reply", "id": message["id"], "result": result})
        except Exception as exc:
            await self.send({"type": "reply", "id": message["id"], "error": str(exc)})

    def observe_event(self, event):
        if (
            event.get("type") == "message_end"
            and event.get("message", {}).get("role") == "assistant"
        ):
            assistant = event["message"]
            self.last_model_error = (
                assistant.get("errorMessage", "Model request failed")
                if assistant.get("stopReason") == "error"
                else None
            )
            for block in assistant.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    self.text += block.get("text", "")

    def remember(self):
        """Carry the turn's MEMORY: block; a missing block keeps the previous one."""
        memory = memory_block(self.text)
        if memory:
            self.memory = memory
            self.bridge.archive.event("memory", chars=len(memory))
        else:
            self.bridge.archive.event("memory_missing", carried=self.memory is not None)

    def continuation(self):
        text = "Continue when ready. The game remains available."
        if self.memory:
            text += "\nMemory from your previous turn:\n" + self.memory
        return text

    async def receive_until(self, kind, *, raise_model_errors=True):
        while True:
            line = await self.process.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"Pi exited; inspect {self.bridge.archive.root / 'pi-stderr.log'}"
                )
            message = json.loads(line)
            if message["type"] == kind:
                if kind == "done" and self.last_model_error and raise_model_errors:
                    raise RuntimeError(self.last_model_error)
                return message
            if message["type"] == "fatal":
                raise RuntimeError(message["error"])
            if message["type"] == "rpc":
                task = asyncio.create_task(self.handle_rpc(message))
                self.pending.add(task)
                task.add_done_callback(self.pending.discard)
            elif message["type"] == "event":
                self.bridge.archive.event("pi_event", event=message["event"])
                self.observe_event(message["event"])

    async def run(self):
        initial = True
        errors = 0
        try:
            async with asyncio.timeout(self.bridge.clock.remaining()):
                while (
                    self.bridge.clock.remaining() > 0
                    and self.bridge.archive.metadata()["status"] != "WIN"
                ):
                    text = (
                        "Initial observation: "
                        + json.dumps(self.bridge.archive.metadata())
                        + ("\n\nObjects (settled frame):\n" + view if (view := self.object_view()) else "")
                        if initial
                        else self.continuation()
                    )
                    packet = {"type": "prompt", "text": text}
                    self.turns, self.text = 0, ""
                    if initial:
                        # Pi's ImageContent schema: mimeType/data at the top level. Its
                        # OpenAI converter builds the data URL from exactly these fields.
                        packet["images"] = [
                            {
                                "type": "image",
                                "mimeType": "image/png",
                                "data": base64.b64encode(png(self.bridge.archive.frame())).decode(),
                            }
                        ]
                    await self.send(packet)
                    await self.receive_until("done", raise_model_errors=False)
                    if self.last_model_error:
                        # A failed turn (e.g. Pi's compaction summary error) ends the prompt,
                        # not the game: the next prompt lets Pi recover. Three in a row stop.
                        errors += 1
                        self.bridge.archive.event(
                            "pi_turn_error", error=self.last_model_error, consecutive=errors
                        )
                        if errors >= 3:
                            raise RuntimeError(self.last_model_error)
                    else:
                        errors = 0
                    self.remember()
                    if not initial:
                        self.bridge.archive.event("neutral_continuation")
                    initial = False
        except TimeoutError:
            self.bridge.archive.event("deadline", seconds=self.bridge.clock.elapsed())
        finally:
            await self.close()

    async def close(self):
        for task in list(self.pending):
            task.cancel()
        await asyncio.gather(*self.pending, return_exceptions=True)
        kill_group(self.executor.process)
        kill_group(self.process)
        if self.process:
            await self.process.wait()
            self.process = None
        if self.web_runner:
            await self.web_runner.cleanup()
            self.web_runner = None
