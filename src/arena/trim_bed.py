"""hx7 (9 October): trim variants and the trim bed's record / replay of an attempt's model replies.

Study key ``trim`` (``settings``; None on every earlier build):

- ``budget_tokens``: the trim triggers at this prompt size instead of ``window - cap - margin`` (never above it), so
  arms served at one window can trim at another build's trigger;
- ``summary``: "default" (``DUCK_SUMMARY_PROMPT``), "long": fixed sections that are all required, an experiment
  log, a length target (``summary_target_tokens``) and an explicit merge of an earlier handoff; or "playbook": no
  summary request, the model's own playbook (hx_harness ``memo_style`` 1) is what the trim carries.

The bed (``arena.level_bench``): a collector attempt plays a fresh game with ``record`` on and stops at the first
request whose prompt estimate reaches ``record["stop_tokens"]``; its gameplay replies are the point's ``script``. A
bed attempt answers its first requests from the script (charged as recorded, tool calls executed as usual, so the
interpreter, the history and the harness state are rebuilt), checks the board against the recording and then plays
live for the token budget. Every arm therefore reaches its first trim with the same context.
"""

import json

SUMMARIES = ("default", "long", "playbook")
DEFAULTS = {"budget_tokens": None, "summary": "default", "summary_target_tokens": 5000}

LONG_SUMMARY_PROMPT = (
    "Your context is full. The harness now replaces the earlier conversation with what you write here; anything "
    "you leave out is lost. Write the handoff for yourself as a structured record, not a short summary: aim for "
    "about {target} tokens, and prefer a long exact record to a short one. Do not call tools. Use exactly these "
    "sections, in this order, and keep every heading even when a section is empty. Give coordinates, colours and "
    "step numbers.\n"
    "## Rules (verified)\n"
    "One line per rule: what each action does, what the objects are, what completes a level; with the level or "
    "step where you saw it.\n"
    "## Hypotheses (not verified)\n"
    "What you believe but have not tested, each with the test that would decide it.\n"
    "## Experiment log\n"
    "One line per distinct experiment: what you did, what you expected, what happened. Include the failures; "
    "merge repeats into one line with a count.\n"
    "## Ruled out\n"
    "Approaches and action sequences that did not work on the current level, so that you do not repeat them.\n"
    "## Levels so far\n"
    "Per completed level: how it was solved, and what was different on the next level.\n"
    "## Current level and state\n"
    "The level number, the board as it is now, counters or timers, the last actions and their results.\n"
    "## Python session\n"
    "The functions and variables you defined that still exist (name, arguments, what it returns, known faults) "
    "and what you saved with note().\n"
    "## Plan\n"
    "The next actions and what you expect to see.\n"
    "If an earlier handoff is in the context, carry forward everything in it that still holds and correct what "
    "later evidence contradicted (the later evidence wins); what you do not carry forward is lost."
)


def settings(value):
    """Validated ``study["trim"]`` over the defaults, or None."""
    if not value:
        return None
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown trim keys {sorted(unknown)}")
    merged = {**DEFAULTS, **value}
    if merged["summary"] not in SUMMARIES:
        raise ValueError(f"trim.summary is one of {SUMMARIES}")
    for key in ("budget_tokens", "summary_target_tokens"):
        if merged[key] is not None and (type(merged[key]) is not int or merged[key] < 1024):
            raise ValueError(f"trim.{key} is an integer of at least 1024")
    return merged


def summary_prompt(trim, default):
    if trim is None or trim["summary"] == "default":
        return default
    return LONG_SUMMARY_PROMPT.format(target=trim["summary_target_tokens"])


def record_settings(value):
    """Validated arm ``study["trim_record"]``: ``{"stop_tokens"}``."""
    if not value:
        return None
    if set(value) != {"stop_tokens"} or type(value["stop_tokens"]) is not int or value["stop_tokens"] < 1024:
        raise ValueError("trim_record is {stop_tokens: int}")
    return dict(value)


class ScriptedResponse:
    """A recorded gameplay reply in the shape of the server's response."""

    status_code = 200

    def __init__(self, reply):
        self.data = {"choices": [{"index": 0, "message": reply["message"], "finish_reason": reply["finish_reason"]}],
                     "usage": reply["usage"]}
        self.text = ""

    def json(self):
        return json.loads(json.dumps(self.data))  # a fresh copy: the caller owns the message

    def raise_for_status(self):
        return None


def reply_record(data):
    """The part of a served gameplay response the replay needs."""
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return {"message": choice["message"], "finish_reason": choice.get("finish_reason"),
            "usage": {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens") if key in usage}}
