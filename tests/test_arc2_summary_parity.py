"""arc2's compaction summary request renders byte for byte as v8.5p's "prefilled-empty" one (arena.compaction) on the
same context under the pinned Qwen3.8 template, and extends the play request's prompt (prefix cache reuse)."""

from test_arena_compaction_prefill import EMPTY_THINK, GENERATION_PROMPT, render

from arc2 import loop, tasks
from arc2.trial import CONTINUE
from arena.compaction import summary_payload
from inference.utils.openai_compat import build_chat_payload

PREFILL = "World model:"  # summary_prefill when on (off by default until the maintainer approves)
SETTINGS = {**loop.DEFAULTS, "conversation": True, "tool": True}
CALL = {"id": "call_0", "type": "function", "function": {"name": "python", "arguments": '{"code": "print(1)"}'}}
CONTEXT = [  # a recorded-shape arc2 conversation: system, task, one python round, a reasoning reply
    {"role": "system", "content": tasks.conv_system(SETTINGS)},
    {"role": "user", "content": "Task 0934a4d8: train pairs ... test input ..."},
    {"role": "assistant", "content": None, "reasoning": "Try the transpose.", "tool_calls": [CALL]},
    {"role": "tool", "tool_call_id": "call_0", "content": "1"},
]


def arc2_payload(messages, summary):
    """What arc2.trial.Transport.complete sends (play request; ``summary``: conv.compact's continued request)."""
    payload = tasks.chat_request("m", messages, 4096, temperature=0.7, effort="xhigh",
                                 tools=[tasks.tool_spec(SETTINGS)])
    if summary:
        payload.update(CONTINUE)
        payload["tool_choice"] = "none"
    return payload


def test_prefill_is_v85p_s():
    from arena.compaction import SUMMARY_PREFILL
    assert PREFILL == SUMMARY_PREFILL


def test_summary_request_renders_as_v85p_prefilled_empty_and_extends_the_play_prompt():
    asked = CONTEXT + [{"role": "user", "content": tasks.CONV_COMPACT}, {"role": "assistant", "content": PREFILL}]
    ours = render(arc2_payload(asked, True))
    play = arc2_payload(CONTEXT, False)
    theirs = render(summary_payload(build_chat_payload=build_chat_payload, model="m", messages=CONTEXT,
                                    prompt=tasks.CONV_COMPACT, cap=4096, temperature=0.7, top_p=0.95, top_k=20,
                                    thinking=True, tools=play["tools"], template_kwargs=play["chat_template_kwargs"],
                                    prefill=PREFILL))
    assert ours == theirs
    shared = render(play, generation_prompt=True)
    assert shared.endswith(GENERATION_PROMPT)
    shared = shared[:-len(GENERATION_PROMPT)]
    assert ours.startswith(shared) and "# Tools" in shared and "xhigh" in shared
    head = f"<|im_start|>user\n{tasks.CONV_COMPACT}<|im_end|>\n<|im_start|>assistant\n"
    assert ours[len(shared):] == head + EMPTY_THINK + PREFILL
