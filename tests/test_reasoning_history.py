"""Reasoning history round trip (29 September, user: "yes to the round-trip fix"; prefix-cache-v2.md).

A reply the model generated must reappear in the next prompt exactly as generated, so the prefix cache can resume from
the states written while the reply was decoded. The public Duck loop re-sends a normalized reasoning string (every
blank line removed), which breaks that at the first paragraph break; ``reasoning_history: "verbatim"`` re-sends the
server's string with only ``<think>`` tags removed.

The template is the Qwen3.8 chat template copy the served Flash-Next template matches (``runs/serving-research``);
the token-level check uses the served tokenizer when it and ``tokenizers`` are available.
"""

import importlib.util
import json
from pathlib import Path

import jinja2
import pytest

from arc_infra.config import Config
from duck_transfer.agent import verbatim_history_text

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "tests/fixtures/qwen38-flash-chat-template.jinja").read_text()
TOKENIZER = Path("./runs/tokenizer-cache/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/"
                 "7b719225242aacd3dbd3f9407468c2ee9a9d2594/tokenizer.json")
TOOLS = [{"type": "function", "function": {"name": "python", "description": "Run Python.",
                                           "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}}]
REASONING = ("The timer bar shrank by one cell.\n\nSo each action costs one step; the block moved right.\n\n"
             "Plan: probe the marker, then\n\n\nmove left twice.")
CODE = "print(current_frame.step)\nresult = action(['LEFT'])"


def render(messages, generation_prompt):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
    # vLLM maps a message's ``reasoning`` to ``reasoning_content`` for the template (chat_utils.py:1869).
    messages = [{**m, "reasoning_content": m["reasoning"]} if m.get("reasoning") is not None else m for m in messages]
    return env.from_string(TEMPLATE).render(messages=messages, tools=TOOLS, add_generation_prompt=generation_prompt,
                                            reasoning_effort="xhigh")


def turn(history_reasoning):
    """(previous prompt + generated reply, next prompt) for one tool-call round with the given re-sent reasoning."""
    messages = [{"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Observation:\nboard at step 3"}]
    prompt = render(messages, True)
    # What the model emitted after the generation prompt's "<think>\n" (qwen3 reasoning + qwen3_coder tool parsers).
    generated = (REASONING + "\n</think>\n\n<tool_call>\n<function=python>\n<parameter=code>\n" + CODE
                 + "\n</parameter>\n</function>\n</tool_call><|im_end|>")
    server_message = {"role": "assistant", "content": None, "reasoning": REASONING,
                      "tool_calls": [{"id": "c1", "type": "function",
                                      "function": {"name": "python", "arguments": json.dumps({"code": CODE})}}]}
    assistant = {"role": "assistant", "content": None, "reasoning": history_reasoning(server_message),
                 "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "python",
                                                                             "arguments": {"code": CODE}}}]}
    nxt = render(messages + [assistant, {"role": "tool", "tool_call_id": "c1", "content": "3"}], True)
    return prompt + generated, nxt


def normalized(message):
    """The public loop's history text (tool_agent._normalize_message_content: blank lines removed, ends stripped)."""
    spec = importlib.util.find_spec("inference.agent.tool_agent")
    assert spec is not None
    from inference.agent import tool_agent
    return tool_agent._extract_reasoning_text(message)


def verbatim(message):
    return verbatim_history_text("reasoning", message, normalized(message))


def test_verbatim_keeps_paragraph_breaks_and_drops_only_think_tags():
    message = {"reasoning": "<think>a\n\nb\n\n\nc</think>", "content": "x\n\ny"}
    assert verbatim_history_text("reasoning", message, "a\nb\nc") == "a\n\nb\n\n\nc"
    assert verbatim_history_text("content", message, "x\ny") == "x\n\ny"
    assert verbatim_history_text("reasoning", {"reasoning_content": "r\n\ns"}, "r\ns") == "r\n\ns"
    # A field the public loop drops stays dropped; a non-string falls back to the public text.
    assert verbatim_history_text("reasoning", {"reasoning": "   "}, "") == ""
    assert verbatim_history_text("content", {"content": [{"type": "text", "text": "t"}]}, "t") == "t"


def test_verbatim_reasoning_round_trips_as_an_exact_prefix_continuation():
    before, nxt = turn(verbatim)
    assert nxt.startswith(before), "the re-rendered reply differs from the generated one"


def test_the_public_normalization_breaks_the_round_trip_at_the_first_paragraph_break():
    before, nxt = turn(normalized)
    assert not nxt.startswith(before)
    common = next(i for i, (a, b) in enumerate(zip(before, nxt)) if a != b)
    assert before[common - 1:common + 1] == "\n\n" and before.index(REASONING) + REASONING.index("\n\n") + 1 == common


def test_round_trip_is_a_token_prefix_with_the_served_tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    if not TOKENIZER.is_file():
        pytest.skip("served tokenizer not in runs/tokenizer-cache")
    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER))
    before, nxt = turn(verbatim)
    a, b = tok.encode(before, add_special_tokens=False).ids, tok.encode(nxt, add_special_tokens=False).ids
    assert b[:len(a)] == a
    before, nxt = turn(normalized)
    a, b = tok.encode(before, add_special_tokens=False).ids, tok.encode(nxt, add_special_tokens=False).ids
    assert b[:len(a)] != a


def test_config_mode_is_validated_and_defaults_to_the_public_normalization():
    assert Config().reasoning_history == "normalized"
    with pytest.raises(ValueError, match="reasoning_history"):
        Config(reasoning_history="raw")


def test_public_loop_routes_every_history_text_through_the_hook():
    source = (ROOT / "src/inference/agent/tool_agent.py").read_text()
    assert source.count("self._history_text(") == 3  # reasoning, content without and with tool calls
    assert 'assistant_message["reasoning"] = reasoning\n' not in source
    from inference.agent.tool_agent import ToolAgent
    assert ToolAgent._history_text(object(), "reasoning", {"reasoning": "a\n\nb"}, "a\nb", False) == "a\nb"


def test_study_key_reaches_the_duck_config_and_old_studies_stay_normalized():
    from arena.trial import duck_config

    study = json.loads((ROOT / "configs/arena_30m.json").read_text())
    kwargs = dict(window=65536, cap=16384, wall=900, base_url="http://model/v1")
    assert duck_config(study, ROOT, **kwargs).reasoning_history == "normalized"
    assert duck_config({**study, "reasoning_history": "verbatim"}, ROOT, **kwargs).reasoning_history == "verbatim"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_arena_rh", ROOT / "scripts/build_arena.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_builds_default_to_verbatim_and_normalized_rebuilds_carry_no_key():
    builder = load_builder()
    duck = {"trials": [{"kind": "duck-transfer"}]}
    assert builder.with_reasoning_history(dict(duck))["reasoning_history"] == "verbatim"
    assert "reasoning_history" not in builder.with_reasoning_history(dict(duck), "normalized")
    assert "reasoning_history" not in builder.with_reasoning_history({**duck, "reasoning_history": "normalized"})
    with pytest.raises(ValueError, match="contradicts"):
        builder.with_reasoning_history({**duck, "reasoning_history": "verbatim"}, "normalized")
    pi = {"trials": [{"kind": "pi"}]}
    assert builder.with_reasoning_history(dict(pi)) == pi
    with pytest.raises(ValueError, match="Duck"):
        builder.with_reasoning_history(dict(pi), "normalized")
