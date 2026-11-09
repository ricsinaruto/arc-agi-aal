"""Tests of the malformed tool-name repair (tool_repair.py) on a patched v8.5p source tree. Run from a copy of the
v84-sub checkout whose src was patched with tool_repair.apply: ``pytest tests/test_tool_repair.py`` (copy this file
into its tests/). Replies are the l45 150K read's recorded first calls (k006, k018)."""

import json

import httpx
from test_context_drain import drain_agent
from test_harness_fixes import SequenceTransport, events_of, text_of

from arena import context_drain as cd

TOOLS = [{"type": "function", "function": {"name": "python", "parameters": {
    "type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
K006 = {"id": "k006", "type": "function", "function": {"name": "python</parameter>\n<parameter=code", "arguments": "{}"}}
K018 = {"id": "k018", "type": "function", "function": {"name": 'python", "code": "', "arguments": "{}"}}
GOOD = {"id": "ok", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": "print(1)"})}}


def test_recorded_l45_calls_are_flagged_and_a_good_call_is_not():
    assert cd.unknown_tool_calls({"tool_calls": [GOOD]}, TOOLS) == []
    assert cd.unknown_tool_calls({"tool_calls": [K006]}, TOOLS)[0].startswith("unknown tool name 'python</parameter>")
    assert cd.unknown_tool_calls({"tool_calls": [GOOD, K018]}, TOOLS)[0].startswith("unknown tool name")
    empty = {"id": "e", "type": "function", "function": {"name": "python", "arguments": "{}"}}
    assert cd.unknown_tool_calls({"tool_calls": [empty]}, TOOLS) == ["python: missing required arguments ['code']"]
    assert cd.unknown_tool_calls({"tool_calls": [K006]}, []) != []  # no tool offered: any call is unknown
    assert cd.unknown_tool_calls({"content": "x"}, TOOLS) == []


def test_the_key_is_optional_and_checked():
    assert "drop_unknown_tools" not in cd.settings(True)  # absent: earlier builds record the same block
    assert cd.settings({"drop_unknown_tools": True})["drop_unknown_tools"] is True
    try:
        cd.settings({"drop_unknown_tools": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("a non-bool drop_unknown_tools was accepted")


class BadNameFirst(SequenceTransport):
    """The first gameplay reply is l45's k006 call (finish tool_calls, valid JSON {}); then the script."""

    async def post(self, url, **kwargs):
        if self.payloads:
            return await super().post(url, **kwargs)
        self.payloads.append(json.loads(json.dumps(kwargs["json"])))
        return httpx.Response(200, request=httpx.Request("POST", "http://local/v1/chat/completions"), json={
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "", "reasoning_content": "Let me get the segmentation first.",
                "tool_calls": [K006]}}],
            "usage": {"completion_tokens": 207, "prompt_tokens": 2000}})


async def test_on_a_malformed_name_never_reaches_history(tmp_path):
    transport = BadNameFirst(["act"])
    spec = {**cd.settings(True), "drop_unknown_tools": True}
    agent, _, _, _, _, _ = drain_agent(tmp_path, transport, rule=lambda m: 3_000, actions=1, drain=spec)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1 and len(transport.payloads) == 2
    second = transport.payloads[1]["messages"]
    assert not any(c.get("id") == "k006" for m in second for c in m.get("tool_calls") or [])
    assert second[-2]["role"] == "assistant" and "tool_calls" not in second[-2]
    assert text_of(second[-1]).startswith(cd.NAME_CALL_FOLLOWUP)
    assert result["context_drain"]["malformed_replies"] == 1
    assert events_of(agent, "malformed_tool_calls_dropped")[0]["errors"][0].startswith("unknown tool name")
    # the game log keeps the raw reply, so the broken-call share still counts it
    assert events_of(agent, "model")[0]["message"]["tool_calls"][0]["id"] == "k006"


async def test_off_the_malformed_name_reaches_history_as_before(tmp_path):
    transport = BadNameFirst(["act"])
    agent, _, _, _, _, _ = drain_agent(tmp_path, transport, rule=lambda m: 3_000, actions=1)
    result = await agent.run()
    second = transport.payloads[1]["messages"]
    assert any(c.get("id") == "k006" for m in second for c in m.get("tool_calls") or [])
    assert result["context_drain"]["malformed_replies"] == 0
