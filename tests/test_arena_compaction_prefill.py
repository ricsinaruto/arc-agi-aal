"""Compaction summary modes (study ``compaction_thinking``): the pre-filled empty think block after the gameplay
prefix (default for new builds, 29 September), thinking at the gameplay effort (28 September) and tool-less off."""

import copy
import json

import httpx
import pytest
from conftest import ROOT, tool_reply
from test_arena_duck import CAP, arena_agent

from arena.compaction import SUMMARY_PREFILL, prefilled_summary_text
from arena.duck import ArenaTransferDuckAgent

TEMPLATE = ROOT / "tests/fixtures/qwen27-chat-template.jinja"  # the pinned Qwen3.8 template (sha256 c3cf9e34...)
GENERATION_PROMPT = "<|im_start|>assistant\n<think>\n"  # thinking on
EMPTY_THINK = "<think>\n\n</think>\n\n"  # the template's non-thinking generation prompt
CONTINUE_TAG = "CONTINUE_FINAL_MESSAGE_TAG "  # transformers 5.x continue_final_message


def reply(message, usage=None):
    return httpx.Response(
        200,
        request=httpx.Request("POST", "http://local/v1/chat/completions"),
        json={"choices": [{"finish_reason": "stop", "message": {"role": "assistant", **message}}],
              "usage": usage or {"completion_tokens": 300, "prompt_tokens": 5000}},
    )


class Transport:
    """Gameplay: one python round, then an action; a summary request gets ``summary`` as its message."""

    def __init__(self, summary):
        self.payloads, self.summary = [], summary

    async def post(self, url, **kwargs):
        payload = copy.deepcopy(kwargs["json"])
        self.payloads.append(payload)
        if len(self.payloads) == 1:
            return tool_reply("saved = current_frame.ascii\nprint('probe')")
        if payload.get("tool_choice") == "none" or "tools" not in payload:
            return reply(self.summary)
        return tool_reply("action([{'action':'UP'}])", call_id="act")


def events(agent):
    return [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]


@pytest.mark.parametrize("field", ["content", "reasoning"])
async def test_prefilled_summary_continues_an_empty_think_block_after_the_gameplay_prefix(tmp_path, field):
    # vLLM's reasoning parser keeps the gameplay kwargs (thinking on), so the continuation may come back as reasoning.
    transport = Transport({field: " tiles are walls.\nPlan: press UP.", **({"content": None} if field != "content" else {})})
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, window=20000)
    agent.compaction_thinking = "prefilled-empty"
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1 and game.compactions == 1
    first, summary, second = transport.payloads
    # Byte-identical to the gameplay request up to its last message: template settings, tools, messages.
    assert summary["chat_template_kwargs"] == first["chat_template_kwargs"] == {"enable_thinking": True,
                                                                                "reasoning_effort": "xhigh"}
    assert summary["tools"] == first["tools"] and summary["tool_choice"] == "none"
    assert summary["messages"][:len(first["messages"])] == first["messages"]
    assert [m["role"] for m in summary["messages"]][-4:] == ["assistant", "tool", "user", "assistant"]
    assert summary["messages"][-2]["content"].startswith("Your context is full.")
    assert summary["messages"][-1] == {"role": "assistant", "content": SUMMARY_PREFILL}
    assert summary["add_generation_prompt"] is False and summary["continue_final_message"] is True
    assert 2048 <= summary["max_tokens"] <= CAP
    for key in ("add_generation_prompt", "continue_final_message"):
        assert key not in first and key not in second  # gameplay requests unchanged
    assert "World model: tiles are walls.\nPlan: press UP." in second["messages"][1]["content"]
    compaction = [e for e in events(agent) if e["kind"] == "compaction"]
    assert compaction[0]["thinking"] == "prefilled-empty" and compaction[0]["mechanical_fallback"] is False


async def test_an_empty_prefilled_continuation_is_retried(tmp_path):
    transport = Transport({"content": "  "})
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, window=20000)
    agent.compaction_thinking = "prefilled-empty"
    result = await agent.run()
    assert result["error"] is None
    summaries = [p for p in transport.payloads if p.get("tool_choice") == "none"]
    assert len(summaries) == 2 and all(p["messages"][-1]["content"] == SUMMARY_PREFILL for p in summaries)
    assert [e["attempt"] for e in events(agent) if e["kind"] == "compaction_empty"] == [1, 2]


@pytest.mark.parametrize("mode,model", [("gameplay", None), ("off", None), ("prefilled-empty", "nex")])
async def test_older_modes_and_other_templates_send_the_earlier_requests(tmp_path, mode, model):
    transport = Transport({"content": "World model: tiles. Plan: press UP."})
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, window=20000)
    agent.compaction_thinking = mode
    if model == "nex":
        from arc_infra.model_templates import NEX_MODEL, NEX_REVISION

        agent.config.model, agent.config.model_revision = NEX_MODEL, NEX_REVISION
        agent.config.reasoning_effort = "high"
    result = await agent.run()
    assert result["error"] is None and game.compactions == 1
    first, summary, second = transport.payloads
    assert "continue_final_message" not in summary and "add_generation_prompt" not in summary
    assert summary["messages"][-1]["content"].startswith("Your context is full.")
    if mode == "off":
        assert "tools" not in summary and summary["chat_template_kwargs"] == {"enable_thinking": False}
    else:  # a1e350d's request: the gameplay prefix, thinking at the gameplay effort
        assert summary["tools"] == first["tools"] and summary["chat_template_kwargs"] == first["chat_template_kwargs"]
    compaction = [e for e in events(agent) if e["kind"] == "compaction"]
    assert compaction[0].get("thinking") == ("off" if mode == "off" else None)


async def test_unknown_mode_is_refused(tmp_path):
    agent, env, game, gate = arena_agent(tmp_path, Transport({}), ArenaTransferDuckAgent)
    agent.compaction_thinking = "prefilled"
    with pytest.raises(ValueError, match="compaction_thinking"):
        agent.resolved_compaction_thinking()
    agent.executor.shutdown()


async def test_exact_admission_counts_the_prefilled_request_as_served(tmp_path):
    from arena.cache import BlockBudget

    counts = []
    transport = Transport({"content": "World model: tiles."})
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, window=20000)
    agent.compaction_thinking = "prefilled-empty"
    gate.budget = BlockBudget({'schema': 1, 'num_blocks': 100, 'max_model_len': 262144, 'lookahead': 3,
                               'costs': [{'tokens': 262144, 'blocks': 20, 'groups': [20]}]})
    gate.capacity = 90

    async def count(messages, deadline, **kwargs):
        counts.append((messages[-1]["role"], kwargs.get("continue_final_message", False)))
        return 3000

    agent.client.count_tokens = count
    result = await agent.run()
    assert result["error"] is None and game.compactions == 1
    assert counts == [("user", False), ("assistant", True), ("user", False)]


def test_prefilled_summary_text():
    assert prefilled_summary_text({"content": " walls.\nPlan: UP."}, SUMMARY_PREFILL) == "World model: walls.\nPlan: UP."
    assert prefilled_summary_text({"content": None, "reasoning": "walls."}, "World model:") == "World model: walls."
    same = {"reasoning": "walls.", "reasoning_content": "walls.", "content": None}
    assert prefilled_summary_text(same, "World model:") == "World model: walls."
    assert prefilled_summary_text({"content": "\n", "reasoning": None}, "World model:") == ""


# --- the pinned template: the summary request's prompt extends the gameplay prompt ------------------------------------


def render(payload, *, messages=None, generation_prompt=None):
    """The prompt vLLM renders for ``payload`` (transformers 5.x ``continue_final_message``)."""
    from posttrain.dataset import ChatTemplate, template_message

    template = ChatTemplate(TEMPLATE).template
    messages = [template_message(m) for m in (messages if messages is not None else payload["messages"])]
    # vLLM hands the assistant's ``reasoning`` field to the template as ``reasoning_content``.
    messages = [{**{k: v for k, v in m.items() if k != "reasoning"}, "reasoning_content": m["reasoning"]}
                if "reasoning" in m else m for m in messages]
    continued = payload.get("continue_final_message", False) and generation_prompt is None
    if continued:
        messages[-1] = {**messages[-1], "content": messages[-1]["content"] + CONTINUE_TAG}
    text = template.render(messages=messages, tools=payload.get("tools"), add_vision_id=False,
                           add_generation_prompt=(payload.get("add_generation_prompt", True)
                                                  if generation_prompt is None else generation_prompt),
                           **(payload.get("chat_template_kwargs") or {}))
    if continued:
        at = text.rindex(CONTINUE_TAG.strip())
        text = text[:at] if text[at:at + len(CONTINUE_TAG)] == CONTINUE_TAG else text[:at].rstrip()
    return text


@pytest.mark.parametrize("mode", ["prefilled-empty", "gameplay", "off"])
async def test_the_summary_prompt_extends_the_gameplay_prompt_under_the_pinned_template(tmp_path, mode):
    transport = Transport({"content": "World model: tiles."})
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, window=20000)
    agent.compaction_thinking = mode
    await agent.run()
    first, summary, _ = transport.payloads
    history = summary["messages"][:-2] if mode == "prefilled-empty" else summary["messages"][:-1]
    prompt = next(m["content"] for m in reversed(summary["messages"]) if m["role"] == "user")
    # The gameplay request that would have gone out on the same history, and the one before it.
    gameplay = render(first, messages=history, generation_prompt=True)
    before = render(first, generation_prompt=True)
    served = render(summary)
    assert gameplay.endswith(GENERATION_PROMPT) and before.endswith(GENERATION_PROMPT)
    shared = gameplay[:-len(GENERATION_PROMPT)]  # ends with the tool result's <|im_end|>\n
    assert shared.endswith("</tool_response><|im_end|>\n") and before[:-len(GENERATION_PROMPT)] in shared
    if mode == "off":
        # The effort line and the tool list leave the system block: nothing past the first tokens is shared.
        assert not served.startswith(shared[:200])
        assert "Reasoning effort is set to xhigh" not in served and "# Tools" not in served
        return
    assert served.startswith(shared)
    tail = served[len(shared):]
    head = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    assert tail.startswith(head)
    if mode == "prefilled-empty":
        assert tail[len(head):] == EMPTY_THINK + SUMMARY_PREFILL  # the template's own empty think block
    else:
        assert tail[len(head):] == "<think>\n"
