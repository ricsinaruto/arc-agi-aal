"""Restore a recorded play-through attempt at a request: the attempt's first ``request``
replies are answered from its ``rollout.jsonl`` through the real agent loop (every tool call runs again, every action
is replayed on a fresh game), each request checked against the recorded one; the attempt then plays live.

``load(attempt_dir, request)`` reads the recording: the replies to script, the recorded request rows to compare, and
the state at the first live request (steps, levels, board hash, the action list, the level's first step).
``compare(row, payload, marks, previous)`` is the per-request check ``RolloutHTTP`` runs.

OPD: ``clear_hint`` writes a hint from a recorded clear (its action list and its own reasoning
before the clear); ``hint_messages`` puts it at the end of the system message of every live request of a hinted try.
"""

import hashlib
import json
import re
from pathlib import Path


def log_rows(path, kinds):
    out = []
    with open(path) as file:
        for line in file:
            if any(line.startswith('{"kind":"%s"' % kind) for kind in kinds):
                out.append(json.loads(line))
    return out


def action_text(action):
    """The level bench's ``bench_actions`` form of a logged action."""
    if not action or action == "None":
        return None
    if isinstance(action, str):
        return action
    name = action.get("action") or action.get("name")
    return name + (f" {action['x']} {action['y']}" if name == "ACTION6" else "")


def frame_hash(frame):
    import numpy as np

    from arc_infra.perception import board_hash

    return board_hash(np.array([[int(c, 16) for c in row] for row in frame], dtype=np.uint8))


# Events after which a recorded context cannot be replayed exactly: a model-written summary (its request is recorded
# as a gameplay row in summary configs), a playbook hand-off, a python interrupt (wall-clock dependent) and a drain
# forced by a context-length rejection (the rejected request is not recorded). A drain_half token or turn-cap drain is
# computed by the agent from the exact token count and replays exactly (19 Oct).
UNSAFE = ("compaction", "fraction_compaction", "playbook_handoff", "duck_python_interrupt", "context_drain",
          "duck_python_clock")
# Tool code reading the clock (duck_python_clock) never replays. Tool code drawing from ``random`` replays only in a
# recording whose sandbox seeded every call (its ``duck_python_rng`` event, 25 Oct); in an older recording the first
# such call is unsafe, as is its first clock read (both found in the rollout's replies).
RNG_SEEDED = "duck_python_rng"
RANDOM_CODE = re.compile(r"\brandom\b")


# Envs multi-game (21 Oct): a summary compaction replays from the recording too: the summary request is a recorded row
# (rollout_record answers it from the script and compares it like any request) and the split is computed from exact
# token counts. Allowed with ARC_RL1_TRAIN_SUMMARY=1 (the summary-compaction runs); a start whose replay differs is
# still dropped by the restore check.
SUMMARY_EVENTS = ("compaction", "fraction_compaction")


def unsafe_from(attempt_dir, allow=None):
    """The first step from which a level start of this recording cannot be restored exactly (None: every start can),
    and the steps of the replayable drains. ``allow``: event kinds treated as replayable (default: the summary
    compactions when ``ARC_RL1_TRAIN_SUMMARY=1``)."""
    import os

    if allow is None:
        allow = SUMMARY_EVENTS if os.environ.get("ARC_RL1_TRAIN_SUMMARY") == "1" else ()
    events = log_rows(Path(attempt_dir) / "log.jsonl", UNSAFE + (RNG_SEEDED,))
    rows = [e for e in events if e["kind"] in UNSAFE and e["kind"] not in allow]
    drains = [int(e.get("step") or 0) for e in rows if e["kind"] == "context_drain" and e.get("trigger") != "forced"]
    stops = [int(e.get("step") or 0) for e in rows if e["kind"] != "context_drain" or e.get("trigger") == "forced"]
    if not any(e["kind"] == RNG_SEEDED for e in events):
        unseeded = unseeded_step(attempt_dir)
        if unseeded is not None:
            stops.append(unseeded)
    return (min(stops) if stops else None), drains


def unseeded_step(attempt_dir):
    """An unseeded recording's first step whose tool code draws from ``random`` or reads the clock (None: none)."""
    from duck_transfer.sandbox import CLOCK_CODE

    try:
        file = open(Path(attempt_dir) / "rollout.jsonl")
    except OSError:
        return None
    with file:
        for line in file:
            if "random" not in line and not CLOCK_CODE.search(line):  # most rows: no parse
                continue
            row = json.loads(line)
            for call in (row.get("reply") or {}).get("tool_calls") or []:
                try:
                    code = str(json.loads((call.get("function") or {}).get("arguments") or "{}").get("code") or "")
                except (ValueError, AttributeError):
                    continue
                if RANDOM_CODE.search(code) or CLOCK_CODE.search(code):
                    return int(row.get("step") or 0)
    return None


def level_starts(attempt_dir, rows=None):
    """``[(request, levels done, the level's first step)]``: the first request at or after each level's first board
    (level 1: request 0). ``rows``: the rollout's parsed rows when the caller has them (attempt_scan)."""
    folder = Path(attempt_dir)
    if rows is None:
        rows = [json.loads(line) for line in open(folder / "rollout.jsonl")]
    marks = [(0, 0)] + [(int(e["level"]), int(e["steps"])) for e in log_rows(folder / "log.jsonl", ("level_bench_level",))]
    out = []
    for done, step in marks:
        request = next((r["n"] for r in rows if r["gameplay"] and int(r["step"]) >= step), None)
        if request is not None and not any(o[0] == request for o in out):
            out.append((request, done, step))
    return out


def attempt_scan(attempt_dir):
    """One recording's ``(first unsafe step, prompt tokens per request, level_starts)``, the rollout parsed once: the
    values ``mg_attempt_start`` reads per level (RL start-up, ARC_RL1_SEED_SCAN)."""
    folder = Path(attempt_dir)
    rows = [json.loads(line) for line in open(folder / "rollout.jsonl")]
    prompts = [int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in rows]
    first, _ = unsafe_from(folder)
    return first, prompts, level_starts(folder, rows=rows)


def load(attempt_dir, request):
    folder = Path(attempt_dir)
    rows = [json.loads(line) for line in open(folder / "rollout.jsonl")]
    if any(r["n"] != i for i, r in enumerate(rows)):
        raise ValueError(f"{folder}: rollout rows are not numbered 0..n-1")
    if not 0 <= request < len(rows):
        raise ValueError(f"{folder}: request {request} of {len(rows)}")
    observations = {int(o["step"]): o for o in log_rows(folder / "log.jsonl", ("observation",))}
    step = int(rows[request]["step"])
    seen = observations[step]
    actions = [action_text(observations[s].get("action")) for s in range(1, step + 1)]
    levels = int(seen["levels"])
    marks = [(int(e["level"]), int(e["steps"])) for e in log_rows(folder / "log.jsonl", ("level_bench_level",))]
    starts = [s for done, s in marks if done == levels]
    if not starts:  # a level entered by replay or restore: the attempt's own record of its first step
        starts = [int(e["level_start_step"]) for e in log_rows(folder / "log.jsonl", ("level_bench_level_start",))
                  if int(e["levels"]) == levels and int(e["level_start_step"]) <= step][-1:]
    start = max(starts or [0])
    tools, scripted = None, []
    for row in rows[:request + 1]:
        tools = row.get("tools", tools) if "tools" in row else tools
        scripted.append({**row, "tools_now": tools})
    return {"source": str(folder), "request": int(request), "rows": scripted,
            "tokens": sum(int((r["usage"] or {}).get("completion_tokens") or 0) for r in rows[:request]),
            "stop": {"steps": step, "levels": levels, "board_hash": frame_hash(seen["frames"][-1])},
            "actions": [a for a in actions if a is not None], "level_start_step": start, "level": levels + 1}


def marks_of(messages):
    return [hashlib.sha1(json.dumps(m, sort_keys=True).encode()).digest() for m in messages]


def compare(row, payload, keep, new):
    """Differences between a request and the recorded one: the shared prefix, the new messages, the template and
    the tool list."""
    diffs = []
    if keep != row["keep"]:
        diffs.append(f"keep {keep} != {row['keep']}")
    elif new != row["messages"]:
        first = next((i for i, (a, b) in enumerate(zip(new, row["messages"])) if a != b), min(len(new), len(row["messages"])))
        diffs.append(f"message {row['keep'] + first} differs ({len(new)} vs {len(row['messages'])} new)")
        if first < min(len(new), len(row["messages"])):  # the first differing span of both texts (22 Oct: a timing-
            a, b = json.dumps(new[first], sort_keys=True), json.dumps(row["messages"][first], sort_keys=True)  # dependent tool result)
            at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            diffs.append(f"at char {at}: live {a[max(0, at - 80):at + 160]!r} recorded {b[max(0, at - 80):at + 160]!r}")
    if (payload.get("chat_template_kwargs") or {}) != (row.get("template") or {}):
        diffs.append("template differs")
    if payload.get("tools") != row.get("tools_now"):
        diffs.append("tools differ")
    return diffs


def response(row):
    """The recorded reply in the shape of the server's chat response (logprob fields already stripped)."""
    usage = {k: v for k, v in (row["usage"] or {}).items()}
    return {"choices": [{"index": 0, "message": row["reply"], "finish_reason": row["finish_reason"]}], "usage": usage}


HINT = ("Your notes from an earlier play of this game, written when you cleared level {level} (the level's first board "
        "is the same every time):\n- Your reasoning just before that clear: {note}\n- The actions that cleared it, in "
        "order: {actions}")


def hint_messages(messages, hint):
    """``messages`` with ``hint`` appended to the system message (the same function as
    ``posttrain.rl_rollouts.hint_messages``, which the trainer bundle ships)."""
    if not hint:
        return messages
    first = messages[0] if messages else None
    if first and first.get("role") == "system" and isinstance(first.get("content"), str):
        return [{**first, "content": first["content"] + "\n\n" + hint}, *messages[1:]]
    return [{"role": "system", "content": hint}, *messages]


def model_action(action):
    from inference.agent.action_names import to_model_action

    text = action_text(action)
    if text is None:
        return None
    name, *xy = text.split()
    return f"MOUSE(row={xy[1]}, col={xy[0]})" if name == "ACTION6" else to_model_action(name)


def clear_hint(attempt_dir, level, note_chars=800, note_floor=200):
    """``(hint, info)`` from an attempt's clear of ``level``: the actions from the level's first board to the clear
    and the tail of the reasoning of its last reply before the clear with at least ``note_floor`` characters; None
    when the attempt did not clear it."""
    folder = Path(attempt_dir)
    seen = {int(o["step"]): o for o in log_rows(folder / "log.jsonl", ("observation",))}
    end = min((s for s, o in seen.items() if int(o.get("levels") or 0) >= level), default=None)
    if end is None:
        return None, None
    start = min(s for s, o in seen.items() if int(o.get("levels") or 0) == level - 1 or s == end)
    seen = {s: o.get("action") for s, o in seen.items()}
    actions = [a for a in (model_action(seen.get(s)) for s in range(start + 1, end + 1)) if a]
    note, tokens = "", 0
    with open(folder / "rollout.jsonl") as file:
        for line in file:
            row = json.loads(line)
            if row.get("scripted") or not start <= int(row.get("step") or 0) < end:
                continue
            tokens += int((row.get("usage") or {}).get("completion_tokens") or 0)
            reasoning = (row.get("reply") or {}).get("reasoning") or (row.get("reply") or {}).get("reasoning_content") or ""
            if len(reasoning) >= note_floor:
                note = reasoning
    note = " ".join(note.split())
    note = note if len(note) <= note_chars else "..." + note[-note_chars:]
    hint = HINT.format(level=level, note=note or "(none)", actions=" ".join(actions))
    return hint, {"source": str(folder), "level": level, "actions": len(actions), "tokens": tokens,
                  "hint_chars": len(hint)}
