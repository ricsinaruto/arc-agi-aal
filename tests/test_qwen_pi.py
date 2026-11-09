import asyncio
import json
import os
import sys

import httpx
import pytest
from test_qwen_core import bridge_at

from qwen_baselines.config import Config
from qwen_baselines.evidence import Archive
from qwen_baselines.executor import Executor
from qwen_baselines.model import RequestLayer
from qwen_baselines.pi_agent import PiAgent

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Requires pinned Node and Linux sandbox"
)


async def exported_bridge(tmp_path):
    bridge = await bridge_at(tmp_path)
    bridge.archive.close()
    bridge.archive = Archive(tmp_path, export=True)
    # Fresh export fixture, including initial immutable image paths.
    bridge.archive.close()
    for name in ("evidence.sqlite", "evidence.sqlite-wal", "evidence.sqlite-shm"):
        (tmp_path / name).unlink(missing_ok=True)
    bridge.archive = Archive(tmp_path, export=True)
    await bridge.initialize()
    return bridge


async def test_shell_live_actions_partial_failure_and_isolation(tmp_path):
    bridge = await exported_bridge(tmp_path / "game")
    executor = Executor(bridge, tmp_path / "game/workspace")
    result = await executor.execute(
        "bash",
        command="python - <<'PY'\nimport numpy as np\nfrom arc_api import observe, step\nb=np.load(observe()['settled']['grid_npy'])\na=np.load(step('ACTION7')['settled']['grid_npy'])\nprint(int((a!=b).sum()))\nraise ValueError('after action')\nPY",
    )
    assert result["exitCode"] != 0 and "4096" in result["output"]
    assert bridge.actions == 1
    # The v4 helpers: the transition summary through arc_api, and grid dumps suppressed.
    bridge.objects = True
    filtering = Executor(bridge, tmp_path / "game/workspace", suppress_dumps=True)
    result = await filtering.execute(
        "bash",
        command="python - <<'PY'\nfrom arc_api import step, transition\n"
        "out = step('ACTION7')\nprint(out['transition']['changed_cells'], transition()['change_bbox'])\n"
        "print('\\n'.join('0' * 64 for _ in range(64)))\nprint('done')\nPY",
    )
    assert result["exitCode"] == 0 and "4096 [0, 0, 64, 64]" in result["output"]
    assert "[grid dump suppressed: 64 rows" in result["output"] and result["output"].endswith("done\n")
    assert result["output_suppressed"] == {"suppressed_rows": 64}
    assert bridge.actions == 2
    secret = tmp_path / "other-game.txt"
    secret.write_text("cross game secret")
    for command in (
        f"cat {secret}",
        "cat /proc/1/environ",
        "python -c 'import socket; socket.socket()'",
        "python -c 'import os; os.setsid()'",
        "cat /app/src/arc_infra/session.py",
        "python -c 'import os; os.chroot(\"/\")'",
        "python -c 'import pathlib,os; pathlib.Path(os.__file__).write_text(\"bad\")'",
        "printf bad > evidence/obs-000000/frame-000.png",
    ):
        result = await executor.execute("bash", command=command)
        assert result["exitCode"] != 0, result
        assert "cross game secret" not in result["output"]
    with pytest.raises(PermissionError):
        await executor.execute("read", path=str(secret))
    with pytest.raises(PermissionError):
        await executor.execute(
            "write", path=str(executor.workspace / "evidence/index.jsonl"), content="bad"
        )
    caps = await executor.execute(
        "bash",
        command="python - <<'PY'\nimport ctypes\nc=ctypes.CDLL(None)\nh=(ctypes.c_uint32*2)(0x20080522,0)\nd=(ctypes.c_uint32*6)()\nassert c.capget(h,d)==0 and not any(d)\nassert all(c.prctl(23,i,0,0,0)==0 for i in range(40))\nprint('no capabilities')\nPY",
    )
    assert caps["exitCode"] == 0 and "no capabilities" in caps["output"]
    sink = await executor.execute("bash", command="echo ok >/dev/null; test ! -s /dev/null")
    assert sink["exitCode"] == 0
    # Background children must not survive a tool result, even if the shell exits.
    marker = executor.workspace / "late"
    await executor.execute("bash", command="(sleep 1; touch late) >/dev/null 2>&1 &")
    await asyncio.sleep(1.1)
    assert not marker.exists()


def sse(message):
    delta = {k: v for k, v in message.items() if k != "tool_calls"}
    if "tool_calls" in message:
        delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]
    chunks = [
        dict(
            id="mock",
            object="chat.completion.chunk",
            model="mock",
            created=1,
            choices=[{"index": 0, "delta": delta, "finish_reason": None}],
        ),
        dict(
            id="mock",
            object="chat.completion.chunk",
            model="mock",
            created=1,
            choices=[
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                }
            ],
            usage={"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
        ),
    ]
    return "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"


@pytest.mark.parametrize("text_model", [False, True])
async def test_actual_pi_sdk_native_tools_images_and_compaction(tmp_path, text_model):
    bridge = await exported_bridge(tmp_path)
    requests = []
    actions = [
        ("read", {"path": "ARC_API.md"}),
        (
            "write",
            {
                "path": "probe.py",
                "content": "from arc_api import step\nprint(step('ACTION7')['observation_id'])\n",
            },
        ),
        ("edit", {"path": "probe.py", "edits": [{"oldText": "ACTION7", "newText": "ACTION4"}]}),
        ("bash", {"command": "python probe.py"}),
        ("read", {"path": "evidence/obs-000001/frame-000.png"}),
        (
            "bash",
            {
                "command": "python -c \"import numpy as np; from arc_api import observe; from segmentation import segment; g=np.load(observe()['settled']['grid_npy']); print('segments', len(segment(g)['nodes']))\""
            },
        ),
    ]

    def endpoint(request):
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 500, "max_model_len": cfg.context_window})
        requests.append(body)
        if not body.get("tools"):
            message = {
                "role": "assistant",
                "content": "## Goal\nPlay the game.\n## Progress\nExecuted ACTION4, observation obs-000001.\n## Next\nInspect result.",
            }
        elif actions:
            name, args = actions.pop(0)
            message = {
                "role": "assistant",
                "reasoning_content": "Use the requested interface.",
                "tool_calls": [
                    {
                        "id": f"call{len(actions)}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
        else:
            message = {
                "role": "assistant",
                "content": "Evidence reviewed. " * 15000 + "\nMEMORY:\nACTION4 changed nothing.",
            }
        return httpx.Response(200, text=sse(message), headers={"content-type": "text/event-stream"})

    from arc_infra.model_templates import MINICPM_MODEL

    cfg = Config(model=MINICPM_MODEL, reasoning_effort="native", context_window=131072) if text_model else Config()
    layer = RequestLayer(
        cfg,
        bridge.clock,
        client=httpx.AsyncClient(
            base_url="http://mock/v1/", transport=httpx.MockTransport(endpoint)
        ),
    )
    if text_model:
        from types import SimpleNamespace

        from arena.pi import ArenaPiAgent
        from arena.text_observations import pi_initial

        class TextPi(ArenaPiAgent):
            # This SDK test exercises tools/compaction; scheduler parking is tested separately.
            admit = PiAgent.admit

        agent = TextPi(bridge, layer, pi_dir=os.environ.get("ARC_PI_ARENA_DIR", "/app/pi-arena"),
                       game=SimpleNamespace())
        packet = {"type": "prompt", "text": "Play this game." + pi_initial(bridge.archive)}
    else:
        agent = PiAgent(bridge, layer, pi_dir=os.environ.get("ARC_PI_DIR", "/app/pi"))
        packet = {"type": "prompt", "text": "Play this game.", "images": [{
            "type": "image", "mimeType": "image/png", "data": __import__("base64").b64encode(
                (tmp_path / "evidence/obs-000000/frame-000.png").read_bytes()).decode()}]}
    try:
        await agent.start()
        await agent.send(packet)
        async with asyncio.timeout(60):
            await agent.receive_until("done")
        assert bridge.actions == 1
        assert (agent.executor.workspace / "probe.py").read_text().find("ACTION4") >= 0
        assert len(requests) == 7
        segments = "".join(
            bridge.archive.transcript(i).get("output", "")
            for i in range(bridge.archive.count("calls"))
        )
        assert "segments 1" in segments
        agent.remember()
        assert agent.memory == "ACTION4 changed nothing."
        assert agent.continuation().endswith(
            "Memory from your previous turn:\nACTION4 changed nothing."
        )

        def image_urls(request):
            return [
                part["image_url"]["url"]
                for m in request["messages"]
                for part in (m["content"] if isinstance(m["content"], list) else [])
                if part.get("type") == "image_url"
            ]

        # Pi must build well-formed data URLs from the prompt and tool-result images;
        # a schema mismatch previously produced data:undefined;base64,undefined.
        for request in (requests[0], requests[-1]):
            urls = image_urls(request)
            if text_model:
                assert not urls and all(isinstance(m.get('content', ''), (str, type(None))) for m in request['messages'])
                assert 'No images can be viewed' in json.dumps(request)
            else:
                assert urls, "expected an image part"
            for url in urls:
                assert url.startswith("data:image/png;base64,"), url[:60]
                assert (
                    __import__("base64").b64decode(url.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"
                )
        await agent.send({"type": "compact"})
        async with asyncio.timeout(30):
            await agent.receive_until("compacted")
        summary = [r for r in requests if not r.get("tools")]
        assert summary and summary[-1]["max_tokens"] == 8192
        assert not summary[-1]["chat_template_kwargs"]["enable_thinking"]
        if text_model:
            assert all(set(r['chat_template_kwargs']) == {'enable_thinking'} for r in requests)
        else:
            assert all(r["chat_template_kwargs"]["preserve_thinking"] is False for r in requests)
        assert all(r["temperature"] == 0.7 and r["top_k"] == 20 for r in requests)
    finally:
        await agent.close()
        await layer.close()


async def test_tool_output_excludes_host_interpreter_stderr_noise(tmp_path, monkeypatch):
    """Kaggle's Python prints a sitecustomize error at startup; Pi decodes reads as base64."""
    import base64
    import stat

    real = sys.executable
    wrapper = tmp_path / "noisy-python"
    wrapper.write_text(
        "#!/bin/sh\necho 'Error in sitecustomize; set PYTHONVERBOSE for traceback:' >&2\n"
        f'exec "{real}" "$@"\n'
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(sys, "executable", str(wrapper))
    bridge = await exported_bridge(tmp_path / "game")
    executor = Executor(bridge, tmp_path / "game/workspace")
    png = tmp_path / "game/evidence/obs-000000/frame-000.png"
    read = await executor.execute("read", path=str(png))
    assert read["exitCode"] == 0
    assert read["output"].strip() == base64.b64encode(png.read_bytes()).decode()
    access = await executor.execute("access", path=str(png))
    assert access["output"] == ""
    shell = await executor.execute("bash", command="echo out; echo err >&2")
    assert shell["exitCode"] == 0 and shell["output"] == "out\nerr\n"
    failed = await executor.execute("bash", command="python -c 'import sys; sys.exit(3)'")
    assert failed["exitCode"] == 3 and "sitecustomize" not in failed["output"]
    with pytest.raises(RuntimeError, match="No such file"):
        await executor.execute("read", path=str(executor.workspace / "missing.txt"))
    kinds = [k for (k,) in bridge.archive.db.execute("select kind from events")]
    assert "tool_stderr" in kinds
