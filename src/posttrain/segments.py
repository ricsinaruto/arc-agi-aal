"""Verified level-clear segments from recorded Duck-transfer arena runs (post-training source 1).

A *segment* is the stretch of one recorded game on one level: every analyzer turn from the first
turn played on level L to the turn whose python tool call executed the clearing action. Each
segment becomes one SFT record holding the level's environment actions, its efficiency against the
game's ``baseline_actions`` and the chat transcript of those turns in OpenAI format. Only runs of
the teacher (``Qwen/Qwen3.8-Flash-Next-NVFP4`` at ``reasoning_effort=xhigh``) on DEV games
(``posttrain.split``) are used, and a record reaches the main output only after the local engine
reproduced it (``verify_game``).

Log schema (``games/<game>/log.jsonl``, one JSON object per line, ``kind`` names the record):

* ``observation``: ``step``, ``action`` (``None`` at step 0, else ``{"action": "ACTION6", "x", "y"}``),
  ``frames`` (every animation frame as 64 hex rows), ``levels`` (completed), ``total_levels``,
  ``state``, ``available`` (engine action names), ``full_reset``.
* ``model_request``: ``step`` and a ``runtime`` block of clocks only; the request messages are
  NOT logged.
* ``model``: ``step``, ``purpose`` (``gameplay``; ``compaction``, ``memory`` and ``notes`` for the
  tool-less aside requests), ``message`` (the raw assistant message: ``content``, ``reasoning`` =
  the full reasoning trace, ``tool_calls``), ``finish_reason``, ``usage`` (``prompt_tokens``,
  ``completion_tokens``, ``completion_tokens_details.reasoning_tokens``), ``seconds``,
  ``reasoning_effort``, ``max_output_tokens``.
* ``duck_python``: ``code`` and ``result``, the python tool's reply; the tool message the model
  saw is exactly ``json.dumps(result, indent=2)``.
* ``compaction`` (summary size, usage, ``mechanical_fallback``), ``stuck_notice`` (the notice text
  appended to that turn's user prompt), ``auto_reset`` (GAME_OVER then an automatic RESET),
  ``knowledge_retained``, ``transfer_notes``, ``action_timing``, ``token_count``,
  ``exact_admission_count``, ``action_skipped``, ``finished`` and variant events (``notes_*``,
  ``memory_rewrite``, ``game_snapshot``, ``explorer``).

``duck-transcript.txt`` (written by the public ``ToolAgent.analyze``; present for the Modal runs and
some Kaggle runs, about 40% of the teacher games) holds, per analyzer turn, the exact system
prompt and user prompt texts, the reasoning, tool calls and rendered tool results.
``prompts/prompt.log`` keeps only the latest request.

What is reconstructed, and how faithfully:

* Assistant turns are exact: reasoning (``reasoning_content``, the raw trace as generated), content
  and tool calls come from the ``model`` records. When the tool call was only written as markup in
  the text, it is recovered with the public loop's own parser (as the harness did) and the markup
  is removed from the text.
* Tool results are exact (``duck_python``). Tool calls the loop never dispatched (a second call
  after one that executed actions) are dropped from the assistant message (``caveats``).
* User turns: the verbatim transcript text where a transcript exists and aligns with the log
  (``user_source: "transcript"``); otherwise the text is re-derived by running the harness's own
  prompt builder (public ``_build_user_prompt`` + duck_transfer's ``PromptMixin`` and transfer notes)
  over the logged replies, action receipts, stuck notices and automatic resets
  (``user_source: "simulated"``). The simulation is checked against every transcript turn
  (``stats.json`` ``prompt_simulation``). The per-turn board image (PNG, 4x upscale, the frame at
  the turn's start step) is re-rendered from the observation with the harness's renderer and
  labelled as the harness labelled it.
* The system prompt is the transcript's, or the current code's for log-only games (validated
  against the transcripts). The ``tools`` list is the current code's python tool.
* Full context. In the arena the Duck-transfer context is append-only until a compaction, so the
  first request of level L also saw the earlier levels' turns (or a compaction summary). The whole
  game is rebuilt once (``game_transcript``) and a segment's ``messages`` hold the system prompt,
  the context its first request saw (earlier levels' turns from the latest compaction on) and then
  the segment's own turns; ``target_start`` is the index of the segment's first message, and only
  assistant messages from there on are targets (earlier ones are context without loss).
  ``request_meta`` gives, per target, the served ``prompt_tokens``, ``finish_reason``, python error
  and action count from the log, so the rebuilt prompt can be checked against what the server
  counted. Each user turn is kept whole (text, carried world model and image), as the arena sent it.
* A compaction replaced every earlier message with the model's own summary; it is inserted as the
  user message the harness sent (``SUMMARY_PREAMBLE`` + summary) before the latest user message
  (the ``compaction_keep_observations``-th latest in the keep-3 variants), and
  ``compaction_indices`` marks where a trainer must cut the visible context: later requests saw
  only the system prompt and the messages from that index on. Each segment is cut from a snapshot
  taken at its last turn, so a later compaction never changes an earlier segment.
  ``starts_after_compaction``: the context at the segment's first request starts at a summary.
* The public loop's request cap: runs whose bundle imported ``inference.agent`` before the
  analyzer environment was applied (30m-r23, 60m-r23 and probe runs) ran with the public default
  of 12 model requests per analyzer turn instead of none; ``request_cap`` records which rule split
  the turns (checked against every transcript).
* Messages carried by variant runs (cross-game notes, memory block, game snapshot) sat after the
  system prompt in the outbound request only; they are omitted and named in ``caveats``. The run's
  study keys that inject context (``CONTEXT_STUDY_KEYS``: those three and the stuck procedure's
  prompt block) are recorded as ``context_injections``; the dataset filters drop such segments.
  Explorer games (engine actions outside the log) are excluded.
* Environment actions that the model did not issue (the automatic RESET after GAME_OVER) are part
  of ``actions`` (they count for the level's score) but not of the transcript.
"""

import copy
import glob
import json
import logging
import math
import os
import re
import statistics
import tempfile
from pathlib import Path
from types import SimpleNamespace

from .split import (
    check_eval_games,
    check_training_games,
    is_dev,
    is_generated,
    is_generated_holdout,
)

TEACHER = "Qwen3.8-Flash-Next-NVFP4 xhigh"
TEACHER_MODEL = "Qwen/Qwen3.8-Flash-Next-NVFP4"
TEACHER_EFFORT = "xhigh"
RUN_PATTERNS = (
    "runs/modal-validation/arena-duck-*/arena/trials/duck-transfer",
    "runs/kaggle-*/**/trials/duck-transfer",
)
# Teacher runs on the generated games (``extract_segments.py --source teacher``). Adding a run is one
# line here, then re-extract and rebuild the dataset.
TEACHER_RUNS = (
    "runs/kaggle-arena-duck-teacher-gen-v2/**/trials/duck-transfer",
    "runs/kaggle-arena-duck-teacher-gen-v3/**/trials/duck-transfer",  # the 20 skins
    "runs/kaggle-arena-duck-teacher-gen-v4/**/trials/duck-transfer",  # the 20 mechanics again
)
# Study keys whose runs sent context the logs do not carry, or a prompt block the base harness lacks.
CONTEXT_STUDY_KEYS = ("cross_game_notes", "memory_block", "game_snapshot", "stuck_procedure")
IMAGE_UPSCALE = 4  # the analyzer's MULTIMODAL_UPSCALE
HEX = "0123456789abcdef"
PUBLIC_TOOL_STEPS = 12  # tool_agent's LOCAL_ANALYZER_TOOL_STEPS default
TAIL_MARGIN = math.log(1.25)  # a compaction tail estimate this far from the served size is re-chosen
# Study settings whose prompt surfaces the logs do not carry (harness prompt.md/playbook/skills,
# CRITICAL notes, a second model, harness v2 receipts and notices): segments from such runs are excluded.
UNRECONSTRUCTED_STUDY_KEYS = ("duck_harness", "critical_errors", "model_escalation", "harness_v2")
VARIANT_EVENTS = {
    "notes_injected": "cross_game_notes",
    "memory_rewrite": "memory_block",
    "game_snapshot": "game_snapshot",
    "harness_preload": "harness_playbook",
}

# The public loop's follow-up after a reply without a tool call (tool_agent.ToolAgent.analyze).
FOLLOWUP_PREFIX = "You have not acted yet. Investigate first. "
FOLLOWUP_MARKUP_PREFIX = (
    "You did not call a tool. We detected `<tool_call>` markup inside your reasoning or assistant "
    "text, so no parsed tool call was executed. On this retry, do not add a note or explanation "
    "first. Emit exactly one `python` tool call directly as your next response. Do not place "
    "`<tool_call>` markup inside reasoning, explanation, or notes. "
)
FOLLOWUP_BODY = (
    "Then investigate and revise your working world model of what the level contains, what actions "
    "appear to do, what the current goal seems to be, and what plan looks best. If helpful, include "
    "short world-model update lines such as `World model:`, `Goal model:`, `Action model:`, "
    "`Recent findings:`, `Open questions:`, `Plan:`, or `Cross-level notes:`. Call the `python` tool "
    "with code that inspects `current_frame`, `previous_frame`, `last_transition`, `history`, or "
    "`valid_actions` -- use `current_frame.segmentation` as the primary view, and `.ascii` only for a "
    "small specific region -- compare `previous_frame` to `current_frame` for the most recent "
    "change, derives a compact board summary, programs a small search or scorer over candidate "
    "actions or short sequences, then call `action(actions)` inside Python with the best valid "
    "action or ordered batch that your code selected. "
)


def level_score(baseline, actions):
    return min(115.0, 100.0 * (baseline / actions) ** 2) if actions else 0.0


# --- harness code (the public Duck loop and duck_transfer), imported lazily -----------------------

_MODULE = None


def duck_module():
    """The public ``tool_agent`` module under the analyzer environment the arena ran it with."""
    global _MODULE
    if _MODULE is None:
        from duck_transfer.loader import load

        _MODULE = load()
    return _MODULE


def followup_prompt(markup):
    from inference.agent.prompts import TOOL_CALL_FORMAT_GUIDANCE

    return (FOLLOWUP_MARKUP_PREFIX if markup else FOLLOWUP_PREFIX) + FOLLOWUP_BODY + (
        TOOL_CALL_FORMAT_GUIDANCE
    )


_TOOLS = None


def tools_schema():
    """The tool list the Duck-transfer analyzer sent (the public python tool, described as a
    persistent interpreter)."""
    global _TOOLS
    if _TOOLS is None:
        _TOOLS = PromptSimulator(1).analyzer._tools(Path("segments") / "tool_runtime_state.json")
    return copy.deepcopy(_TOOLS)


def reply_parts(message):
    """How the loop read one assistant reply: the tool calls it dispatched, the (normalised, markup
    free) text it kept in history, and the raw reasoning/content as generated."""
    module = duck_module()
    reasoning = module._extract_reasoning_text(message)
    content = module._normalize_message_content(message.get("content", ""))
    tool_calls = copy.deepcopy(message.get("tool_calls") or [])
    markup = module._contains_tool_call_markup(reasoning, content)
    recovered = False
    if not tool_calls and markup:
        tool_calls = module._recover_tool_calls_from_markup(reasoning, content)
        recovered = bool(tool_calls)
    raw_reasoning = message.get("reasoning")
    if raw_reasoning in (None, ""):
        raw_reasoning = message.get("reasoning_content") or ""
    raw_content = message.get("content")
    raw_content = raw_content if isinstance(raw_content, str) else module._normalize_message_content(
        raw_content or "")
    if markup:
        raw_reasoning = module._TOOL_CALL_BLOCK_RE.sub("", raw_reasoning)
        raw_content = module._TOOL_CALL_BLOCK_RE.sub("", raw_content)
    return {
        "tool_calls": [
            {"id": call.get("id", ""), "type": call.get("type", "function"),
             "function": {"name": (call.get("function") or {}).get("name", ""),
                          "arguments": (call.get("function") or {}).get("arguments", "{}")}}
            for call in tool_calls if isinstance(call, dict)
        ],
        "markup": markup,
        "recovered": recovered,
        "history_reasoning": module._strip_tool_call_markup(reasoning) if markup else reasoning,
        "history_content": module._strip_tool_call_markup(content) if markup else content,
        "reasoning": raw_reasoning.strip(),
        "content": raw_content.strip(),
    }


# --- discovery ------------------------------------------------------------------------------------


def find_trials(root=".", patterns=RUN_PATTERNS):
    found = set()
    for pattern in patterns:
        for path in glob.glob(str(Path(root) / pattern), recursive=True):
            trial = Path(path)
            if "-lost-" not in path and (trial / "games").is_dir():
                found.add(trial)
    return sorted(found)


def run_name(trial):
    parts = Path(trial).parts
    start = len(parts) - 1 - parts[::-1].index("runs") if "runs" in parts else 0
    return "/".join(parts[start + 1:-3])


def trial_info(trial):
    """Run name, model, effort, context window and the public-roster id mapping of one trial."""
    trial = Path(trial)
    info = {"trial": str(trial), "run": run_name(trial), "model": None, "effort": None,
            "max_model_len": None, "gameplay_seconds": None, "roster": {}, "config_source": None}
    arena = trial / "arena.json"
    if arena.is_file():
        config = (json.loads(arena.read_text()).get("config") or {})
        info.update(model=config.get("model"), effort=config.get("reasoning_effort"),
                    max_model_len=config.get("max_model_len"),
                    gameplay_seconds=config.get("gameplay_seconds"), config_source="arena.json")
    else:  # a lost run: the served model from the server identity, the effort from the replies
        identity = trial.parents[1] / "serving" / "vllm-server-identity.json"
        if identity.is_file():
            argv = json.loads(identity.read_text()).get("argv") or []
            for flag, key in (("--served-model-name", "model"), ("--max-model-len", "max_model_len")):
                if flag in argv and argv.index(flag) + 1 < len(argv):
                    value = argv[argv.index(flag) + 1]
                    info[key] = int(value) if key == "max_model_len" else value
            info["config_source"] = "serving-identity"
    info["request_cap"], info["request_cap_source"] = request_cap(info["run"], root_of(trial))
    study_path = trial.parent / "study.json"
    study = json.loads(study_path.read_text()).get("study", {}) if study_path.is_file() else {}
    variant = study.get("arena_variant")
    info["variant"] = variant.get("name") if isinstance(variant, dict) else variant
    info["keep_observations"] = study.get("compaction_keep_observations") or 1
    info["unreconstructed"] = sorted(k for k in UNRECONSTRUCTED_STUDY_KEYS if study.get(k))
    info["context_injections"] = sorted(k for k in CONTEXT_STUDY_KEYS if study.get(k))
    roster = trial / "public-roster.json"
    if roster.is_file():
        info["roster"] = {row["game_id"]: row["source_game_id"] for row in json.loads(roster.read_text())}
    return info


def root_of(trial):
    parts = Path(trial).resolve().parts
    return Path(*parts[:len(parts) - 1 - parts[::-1].index("runs")]) if "runs" in parts else Path(".")


def request_cap(run, root="."):
    """``(cap, source)``: the gameplay requests one analyzer turn could make.

    The Duck-transfer loader sets ``LOCAL_ANALYZER_TOOL_STEPS=0`` (no cap) before importing the
    public ``tool_agent``, but a bundle whose arena modules import ``inference.agent`` at module
    level (``arena/snapshot.py`` from 25 September: the 30m-r23, 60m-r23 and probe runs) imported
    it first, with the public default of 12 requests per turn. Read from the run's ``dist/``
    bundle; ``None`` when there is none (the transcript decides, see ``reconstruct_game``)."""
    name = re.sub(r"-modal-\d{8}$", "", run.split("/")[-1])
    name = name[len("kaggle-"):] if name.startswith("kaggle-") else name
    arena = Path(root) / "dist" / name / "src" / "arena"
    if not arena.is_dir():
        return None, "no_bundle"
    early = any(re.search(r"^(from|import) inference\.agent", path.read_text(errors="replace"), re.M)
                for path in arena.glob("*.py"))
    return (PUBLIC_TOOL_STEPS if early else None), "bundle"


def read_log(path):
    records = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "kind" in record:
                records.append(record)
    return records


# --- transcript -----------------------------------------------------------------------------------

HEADER_RE = re.compile(r"^--- (?:analysis_step=(\d+) \| )?action=(\d+) \| [0-9:]+ \| tool-agent ---$")
LABEL_RE = re.compile(
    r"^\[(SYSTEM PROMPT|USER PROMPT|MODEL RESPONSE META|THINKING|ASSISTANT|ANALYZER STATUS"
    r"|TOOL CALL: [^\]]+|TOOL RESULT: [^\]]+)\]$"
)
TOOL_COUNT_RE = re.compile(r"^tool_call_count: (\d+)$", re.M)


def parse_transcript(text):
    """Analyzer turns of a ``duck-transcript.txt``: ``{"system", "users": [...], "metas": [...]}``;
    ``metas`` holds each model reply's tool-call count in order."""
    turns, turn, label, lines = [], None, None, []

    def flush():
        if turn is None or label is None:
            return
        content = "\n".join(lines).strip()
        if label == "SYSTEM PROMPT":
            turn["system"] = content
        elif label == "USER PROMPT":
            turn["users"].append(content)
            turn["order"].append("user")
        elif label == "MODEL RESPONSE META":
            match = TOOL_COUNT_RE.search(content)
            turn["metas"].append(int(match.group(1)) if match else None)
            turn["order"].append("meta")

    previous = ""
    for line in text.split("\n"):
        header = HEADER_RE.match(line) if previous == "" else None
        if header:
            flush()
            turn = {"analysis_step": int(header.group(1)) if header.group(1) else None,
                    "action": int(header.group(2)), "system": None, "users": [], "metas": [],
                    "order": []}
            turns.append(turn)
            label, lines = None, []
        elif turn is not None and previous in ("", None) and LABEL_RE.match(line):
            flush()
            label, lines = LABEL_RE.match(line).group(1), []
        elif label is not None:
            lines.append(line)
        previous = "" if header else line
    flush()
    return turns


# --- turns from the log ---------------------------------------------------------------------------


def grid(rows):
    return tuple(tuple(int(c, 16) for c in row) for row in rows)


def display_action(action):
    from inference.agent.action_names import to_model_action

    if action["action"] == "ACTION6":
        return f"MOUSE(row={action.get('y')}, col={action.get('x')})"
    return to_model_action(action["action"])


def game_turns(records, max_requests=None):
    """``(turns, observations, extras)``: the analyzer turns of one game log.

    A turn is one ``analyze`` call: gameplay requests until a python call executes actions (or,
    under a request cap, ``max_requests`` requests without one). Each request carries its reply,
    its python results, the actions it executed and any compaction summary sent before it.
    Observations outside a python call are ``extras["external"]``."""
    from arena.compaction import mechanical_summary, summary_text

    observations, turns, external = [], [], []
    auto_resets, variants, explorer = {}, set(), False
    turn = request = pending_notice = None
    pending_summary = None
    for record in records:
        kind = record["kind"]
        if kind == "observation":
            if record.get("action") is not None and request is not None and request["pending"] > 0:
                request["actions"].append(record["step"])
                request["executing_call"] = len(request["python"])
            elif record.get("action") is not None:
                prior = observations[-1] if observations else {}
                external.append({"step": record["step"], "action": record["action"]["action"],
                                 "auto_reset": record["action"]["action"] == "RESET"
                                 and prior.get("state") == "GAME_OVER"})
            observations.append(record)
        elif kind == "model":
            purpose = record.get("purpose", "gameplay")
            if purpose == "compaction":
                text = summary_text(record.get("message") or {})
                pending_summary = {"text": text or (pending_summary or {}).get("text", ""),
                                   "tokens": int((record.get("usage") or {}).get("completion_tokens")
                                                 or 0) + (pending_summary or {}).get("tokens", 0)}
                continue
            if purpose != "gameplay":
                continue
            if turn is not None and max_requests and len(turn["requests"]) >= max_requests:
                turn["closed"] = True
            if turn is None or turn["closed"]:
                last = observations[-1] if observations else {}
                turn = {"index": len(turns), "start_step": record.get("step", 0),
                        "levels": int(last.get("levels", 0)), "requests": [], "closed": False,
                        "notice": pending_notice}
                pending_notice = None
                turns.append(turn)
            parts = reply_parts(record.get("message") or {})
            request = {"record": record, **parts, "python": [], "actions": [],
                       "executing_call": None, "summary": pending_summary,
                       "pending": sum(c["function"]["name"] == "python" for c in parts["tool_calls"])}
            pending_summary = None
            turn["requests"].append(request)
        elif kind == "compaction":
            # The summary request(s) are logged first; a double empty reply falls back to the
            # runner's mechanical handoff.
            if record.get("mechanical_fallback") or not (pending_summary or {}).get("text"):
                last = observations[-1] if observations else {}
                pending_summary = {"text": mechanical_summary(
                    steps=record.get("step"), levels=last.get("levels", 0),
                    total_levels=last.get("total_levels")),
                    "tokens": (pending_summary or {}).get("tokens", 0)}
        elif kind == "duck_python":
            if request is None or request["pending"] <= 0:
                continue
            request["python"].append(record)
            request["pending"] -= 1
            if request["actions"]:
                request["pending"] = 0
                turn["closed"] = True
        elif kind == "stuck_notice":  # built into the next turn's user prompt
            pending_notice = record.get("text")
        elif kind == "auto_reset":
            auto_resets[record.get("step")] = record
        elif kind in VARIANT_EVENTS:
            variants.add(VARIANT_EVENTS[kind])
        elif kind == "explorer":
            explorer = True
    extras = {"external": external, "auto_resets": auto_resets,
              "variants": sorted(variants), "explorer": explorer}
    return turns, observations, extras


def level_clears(observations):
    """``[{"level", "start_step", "clear_step"}]`` for each level the game cleared."""
    clears = []
    if not observations:
        return clears
    levels, start = int(observations[0].get("levels", 0)), int(observations[0]["step"])
    for record in observations[1:]:
        new = int(record.get("levels", levels))
        if new > levels:
            clears.append({"level": levels + 1, "start_step": start, "clear_step": int(record["step"]),
                           "jump": new - levels})
            start = int(record["step"])
        levels = max(levels, new)
    return clears


# --- user prompt simulation -----------------------------------------------------------------------


class PromptSimulator:
    """The Duck-transfer analyzer's prompt state, driven by a game log instead of a live game.

    Builds a real ``TransferDuckAgent`` analyzer (public ``ToolAgent`` + ``PromptMixin`` + transfer
    notes) without an environment or server, then feeds it what the live loop fed it: the reply
    texts (world-model updates), the executed action receipts, stuck notices and automatic resets.
    """

    def __init__(self, total_levels):
        from duck_transfer.transfer import TransferDuckAgent

        duck_module()
        owner = TransferDuckAgent.__new__(TransferDuckAgent)
        owner.module = duck_module()
        owner.config = SimpleNamespace(model=TEACHER_MODEL, base_url="http://127.0.0.1:1234/v1",
                                       max_output_tokens=16384, max_model_len=131072)
        owner.image_tokens, owner.chars_per_token = 400, 3.0
        owner.events = []
        owner.workspace = SimpleNamespace(append=lambda kind, **data: owner.events.append((kind, data)))
        owner.steps, owner.auto_reset, owner.no_ops = 0, None, {}
        owner.obs = SimpleNamespace(state="NOT_FINISHED", levels=0, total_levels=total_levels,
                                    available=[])
        self.notice = None
        owner.stuck = SimpleNamespace(notice=lambda level, tried: self.notice)
        owner.tried_no_effect_here = lambda: []
        self.owner = owner
        self.analyzer = owner.analyzer(None)
        self.total_levels = total_levels
        self.max_level = 1

    @property
    def system_prompt(self):
        return self.analyzer._system_prompt

    def user_prompt(self, turn, observation, notice=None, auto_reset=None):
        from inference.agent.action_names import to_model_actions
        from inference.agent.runtime_state import Frame

        owner = self.owner
        levels = int(observation.get("levels", 0))
        total = int(observation.get("total_levels") or self.total_levels)
        level = max(1, min(total, levels + 1))
        self.max_level = max(self.max_level, level)
        owner.steps = turn["start_step"]
        owner.obs = SimpleNamespace(state=observation.get("state"), levels=levels, total_levels=total,
                                    available=observation.get("available") or [])
        owner.auto_reset = auto_reset
        self.notice = notice
        frame = Frame(grid=grid(observation["frames"][-1]), step=turn["start_step"], level=level)
        history = [SimpleNamespace(frame=SimpleNamespace(level=self.max_level))]
        return self.analyzer._build_user_prompt(
            turn["start_step"], valid_actions=to_model_actions(owner.obs.available),
            current_frame=frame, history_entries=history,
            previous_step_summary=self.analyzer._last_step_summary,
        )

    def reply(self, request):
        """World-model update from a reply's text (before its tool calls run)."""
        if request["history_content"]:
            self.analyzer._update_summarized_knowledge_from_assistant(request["history_content"])

    def executed(self, observations):
        """The step summary of one executing python call, from its observations."""
        if not observations:
            return
        last = observations[-1]
        total = int(last.get("total_levels") or self.total_levels)
        payload = {
            "executed": True,
            "executed_count": len(observations),
            "executed_actions": [display_action(o["action"]) for o in observations],
            "action_num": int(last["step"]),
            "level": max(1, min(total, int(last.get("levels", 0)) + 1)),
            "level_completed": any(o.get("_level_up") for o in observations),
            "game_over": any(o.get("state") == "GAME_OVER" for o in observations),
            "run_complete": last.get("state") == "WIN",
            "done": last.get("state") == "WIN",
            "board_changed": True,
        }
        analyzer = self.analyzer
        analyzer._last_step_summary = analyzer._summarize_step_sequence([payload])
        analyzer._update_summarized_knowledge_from_step_summary()


def image_part(rows, *, inline=True):
    if not inline:
        return None
    from inference.agent.runtime_state import Frame
    from inference.agent.vision_context import frame_to_png_data_url

    url = frame_to_png_data_url(Frame(grid=grid(rows), step=0, level=1), upscale=IMAGE_UPSCALE)
    return {"type": "image_url", "image_url": {"url": url}}


def user_message(text, step, rows, *, images=True):
    from duck_transfer.prompting import IMAGE_LABEL

    label = f"{text}\n\n{IMAGE_LABEL.format(step=step)}"
    part = image_part(rows, inline=images)
    if part is None:
        return {"role": "user", "content": [{"type": "text", "text": label}]}
    return {"role": "user", "content": [{"type": "text", "text": label}, part]}


# --- one game -------------------------------------------------------------------------------------


def align_transcript(transcript_turns, turns):
    """True when the transcript's turns and replies line up one to one with the log's."""
    if len(transcript_turns) < len(turns):
        return False
    for logged, written in zip(turns, transcript_turns, strict=False):
        metas = written["metas"]
        if len(metas) != len(logged["requests"]) or not written["users"]:
            return False
        for request, count in zip(logged["requests"], metas, strict=True):
            if count is not None and count != len(request["tool_calls"]):
                return False
        followups = sum(1 for r in logged["requests"] if not r["tool_calls"])
        if len(written["users"]) != 1 + followups:
            return False
    return True


def reconstruct_game(game_dir, info, *, images=True):
    """Every cleared level of one recorded game as an unverified segment record (with its
    messages), plus the game's full action list for verification and prompt-check counts."""
    game_dir = Path(game_dir)
    records = read_log(game_dir / "log.jsonl")
    cap = info.get("request_cap")
    turns, observations, extras = game_turns(records, cap)
    if not observations:
        return {"segments": [], "actions": [], "observations": [], "prompt_check": {}}
    by_step = {int(o["step"]): o for o in observations}
    previous_levels = int(observations[0].get("levels", 0))
    for o in observations:
        o["_level_up"] = int(o.get("levels", 0)) > previous_levels
        previous_levels = max(previous_levels, int(o.get("levels", 0)))
    total_levels = int(observations[0].get("total_levels") or 0)

    transcript_turns, aligned = [], False
    transcript = game_dir / "duck-transcript.txt"
    if transcript.is_file():
        transcript_turns = parse_transcript(transcript.read_text(encoding="utf-8", errors="replace"))
        aligned = align_transcript(transcript_turns, turns)
        if not aligned:  # the transcript shows the turn structure: try the other request cap
            other = None if cap else PUBLIC_TOOL_STEPS
            other_turns, _, _ = game_turns(records, other)
            if align_transcript(transcript_turns, other_turns):
                turns, cap, aligned = other_turns, other, True

    # Walk every turn once: simulated prompts (validated against the transcript) and the followups.
    simulator = PromptSimulator(total_levels)
    check = {"turns": 0, "user_exact": 0, "system_exact": 0, "followups": 0, "followup_exact": 0}
    mismatches = []
    # The runner's ``auto_reset`` note stays set from the automatic RESET until the next action.
    auto_steps = {e["step"] for e in extras["external"] if e["auto_reset"]}
    for turn in turns:
        start = turn["start_step"]
        observation = by_step.get(start) or observations[-1]
        auto = None
        if start in auto_steps:
            record = extras["auto_resets"].get(start) or {}
            auto = {"step": record.get("game_over_step"), "actions": record.get("actions") or [],
                    "reset_step": start}
        text = simulator.user_prompt(turn, observation, turn["notice"], auto)
        turn["simulated_user"] = text
        turn["simulated_followups"] = []
        for request in turn["requests"]:
            simulator.reply(request)
            if not request["tool_calls"]:
                turn["simulated_followups"].append(followup_prompt(request["markup"]))
            elif request["actions"]:
                simulator.executed([by_step[s] for s in request["actions"] if s in by_step])
        if aligned:
            written = transcript_turns[turn["index"]]
            check["turns"] += 1
            check["user_exact"] += written["users"][0] == text.strip()
            if written["users"][0] != text.strip():
                mismatches.append({"game": game_dir.name, "turn": turn["index"],
                                   "step": turn["start_step"],
                                   "diff": _first_difference(written["users"][0], text.strip())})
            check["system_exact"] += written["system"] == simulator.system_prompt.strip()
            for mine, theirs in zip(turn["simulated_followups"], written["users"][1:], strict=False):
                check["followups"] += 1
                check["followup_exact"] += mine.strip() == theirs
    system = simulator.system_prompt
    if aligned and transcript_turns and transcript_turns[0]["system"]:
        system = transcript_turns[0]["system"]

    clears = level_clears(observations)
    seg_turns_of = {c["level"]: segment_turns(turns, c) for c in clears}
    snapshots = game_transcript(turns, observations, by_step, info, transcript_turns if aligned else None,
                                system, images=images, tools=tools_schema(),
                                stop_after={t[-1]["index"] for t in seg_turns_of.values() if t})
    segments = []
    actions_all = [o["action"] for o in observations if o.get("action") is not None]
    for clear in clears:
        seg_turns = seg_turns_of[clear["level"]]
        snapshot = snapshots.get(seg_turns[-1]["index"]) if seg_turns else None
        segments.append(build_segment(game_dir, info, clear, seg_turns, observations, by_step, extras,
                                      snapshot, system))
    for segment in segments:
        segment["request_cap"] = cap
    return {"segments": segments, "actions": actions_all, "observations": observations,
            "prompt_check": check, "aligned": aligned, "has_transcript": bool(transcript_turns),
            "turns": len(turns), "request_cap": cap, "explorer": extras["explorer"],
            "prompt_mismatches": mismatches}


def _first_difference(expected, actual, context=80):
    index = next((i for i, (a, b) in enumerate(zip(expected, actual, strict=False)) if a != b),
                 min(len(expected), len(actual)))
    return {"at": index, "transcript": expected[max(0, index - 20):index + context],
            "simulated": actual[max(0, index - 20):index + context]}


def segment_turns(turns, clear):
    """The analyzer turns played on ``clear``'s level up to its clearing action."""
    level, start, end = clear["level"], clear["start_step"], clear["clear_step"]
    return [t for t in turns if t["levels"] == level - 1 and start <= t["start_step"] < end]


def turn_texts(turn, transcript_turns):
    """``(user text, follow-up texts, source)`` of one turn: the transcript's verbatim texts where
    a transcript aligns (the simulated, unstripped text where it reproduces them), else the
    simulation."""
    if transcript_turns is None:
        return turn["simulated_user"], list(turn["simulated_followups"]), "simulated"
    # The transcript stores each prompt stripped; where the simulation reproduces it, the
    # simulated (unstripped) text is the one that was sent.
    written = transcript_turns[turn["index"]]
    simulated = [turn["simulated_user"], *turn["simulated_followups"]]
    texts = [mine if mine.strip() == theirs else theirs
             for mine, theirs in zip(simulated, written["users"], strict=False)]
    texts += written["users"][len(texts):]
    same = all(mine.strip() == theirs for mine, theirs in zip(simulated, written["users"], strict=False))
    return texts[0], texts[1:], "transcript_verified" if same else "transcript"


def python_error(results):
    """The first python error of a request's tool results (a short text), or None."""
    for entry in results:
        result = entry.get("result") or {}
        if result.get("error") or result.get("returncode") not in (0, None):
            return str(result.get("error") or result.get("stderr") or result.get("stdout") or "error")[:160]
    return None


def _tail_candidates(history, roles, keep):
    """The harness's compaction tails (``CompactionMixin._trim_messages_for_context``), in its order
    of preference: from the ``keep``-th latest user message; from the latest (keep > 1 that did not
    fit); only the latest observation (the tail did not fit). ``history`` are message ids after the
    system prompt, ``roles`` their roles."""

    def split(keep):
        users = [i for i, sid in enumerate(history) if roles[sid] == "user"]
        if not users:
            return []
        return history[users[-keep] if len(users) >= keep else users[0]:]

    candidates = [("keep", split(keep))]
    if keep > 1:
        candidates.append(("latest", split(1)))
    candidates.append(("observation_only", candidates[-1][1][:1]))
    unique, seen = [], set()
    for name, tail in candidates:
        if tuple(tail) not in seen:
            seen.add(tuple(tail))
            unique.append((name, tail))
    return unique


def _estimate(messages, tools):
    """Prompt tokens of ``messages`` (characters / 4.1 + 64 per board image; a user turn built without
    its image part still counts one)."""
    from .sft_records import IMAGE_TOKENS, count_images, prompt_tokens_estimate

    missing = sum(1 for m in messages if m.get("role") == "user" and isinstance(m.get("content"), list)
                  and not count_images([m]))
    return prompt_tokens_estimate(messages, tools) + IMAGE_TOKENS * missing


def game_transcript(turns, observations, by_step, info, transcript_turns, system, *, images=True,
                    stop_after=(), tools=None):
    """The whole game's chat as the harness sent it.

    The context is append-only until a compaction replaces it with ``[system, summary, *tail]``
    (``arena.compaction``). Messages live in one linear list (a summary is inserted right before its
    tail) and every request records its exact context as the list of message ids it saw, so a
    request made before a compaction never sees the summary, and messages the compaction dropped
    stay out of later contexts. Where the harness had a choice of tail (a tail that did not fit the
    budget), the candidate whose estimated size is closest to the request's logged
    ``prompt_tokens`` is taken (``compaction_tail`` in the request's meta).

    Returns ``{turn index: snapshot}`` for the turns in ``stop_after``; a snapshot is the state right
    after that turn: ``messages``, per message its ``ids``, ``owners`` (``(turn, request)``) and
    ``kinds``, ``meta`` (per assistant message: served usage, finish reason, python error, actions,
    ``context`` = message ids of its request), and per-request ``caveats``."""
    tools = tools if tools is not None else []
    messages, owners, kinds, meta, ids = [{"role": "system", "content": system}], [None], ["system"], [None], [0]
    store, roles = {0: messages[0]}, {0: "system"}
    visible = [0]
    caveats, snapshots = {}, {}

    def append(message, owner, kind, info_=None):
        sid = len(store)
        store[sid], roles[sid] = message, message["role"]
        for column, value in ((messages, message), (owners, owner), (kinds, kind), (meta, info_), (ids, sid)):
            column.append(value)
        visible.append(sid)
        return sid

    for turn in turns:
        user_text, followups, source = turn_texts(turn, transcript_turns)
        turn["user_source"] = source
        observation = by_step.get(turn["start_step"]) or observations[0]
        for index, request in enumerate(turn["requests"]):
            owner = (turn["index"], index)
            notes = caveats.setdefault(owner, [])
            if index == 0:
                append(user_message(user_text, turn["start_step"], observation["frames"][-1], images=images),
                       owner, "user")
            record = request["record"]
            usage = record.get("usage") or {}
            tail_name = None
            if request["summary"] is not None:
                keep = int(info.get("keep_observations") or 1)
                summary = {"role": "user", "content": _summary_preamble() + request["summary"]["text"].strip()}
                candidates = _tail_candidates(visible[1:], roles, keep)
                served = usage.get("prompt_tokens")
                tail_name, tail = candidates[0]
                if served and len(candidates) > 1:
                    def distance(candidate):
                        estimate = _estimate([messages[0], summary, *(store[s] for s in candidate[1])], tools)
                        return abs(math.log(max(estimate, 1) / served))
                    # The harness's first choice unless the served size clearly contradicts it.
                    if distance(candidates[0]) > TAIL_MARGIN:
                        tail_name, tail = min(candidates, key=distance)
                if keep > 1:
                    notes.append(f"compaction_keeps_{keep}_observations")
                if tail_name != "keep":
                    notes.append(f"compaction_tail_{tail_name}")
                sid = len(store)
                store[sid], roles[sid] = summary, "user"
                position = ids.index(tail[0]) if tail else len(messages)
                for column, value in ((messages, summary), (owners, owner), (kinds, "summary"), (meta, None),
                                      (ids, sid)):
                    column.insert(position, value)
                visible[:] = [0, sid, *tail]
            calls = request["tool_calls"]
            dispatched = dispatch(request)
            if len(dispatched) < len(calls):
                notes.append("undispatched_tool_calls_dropped")
            if request["recovered"]:
                notes.append("tool_call_recovered_from_markup")
            if dispatched or request["content"] or request["reasoning"]:
                message = {"role": "assistant", "content": request["content"] or None,
                           "reasoning_content": request["reasoning"]}
                if dispatched:
                    message["tool_calls"] = [call for call, _ in dispatched]
                context = list(visible)
                append(message, owner, "assistant", {
                    "step": record.get("step"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "finish_reason": record.get("finish_reason"),
                    "python_error": python_error(request["python"]),
                    "actions": len(request["actions"]),
                    "compaction_tail": tail_name,
                    "context": context,
                })
            for call, content in dispatched:
                append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content}, owner, "tool")
            if not calls:
                text = followups.pop(0) if followups else followup_prompt(request["markup"])
                append({"role": "user", "content": text}, owner, "followup")
        if turn["index"] in stop_after:
            snapshots[turn["index"]] = {"messages": list(messages), "ids": list(ids), "owners": list(owners),
                                        "kinds": list(kinds), "meta": list(meta), "caveats": caveats}
    return snapshots


def _ranges(positions):
    """``[[start, end), ...]`` runs of sorted positions."""
    out = []
    for p in positions:
        if out and out[-1][1] == p:
            out[-1][1] = p + 1
        else:
            out.append([p, p + 1])
    return out


def build_segment(game_dir, info, clear, seg_turns, observations, by_step, extras, snapshot, system):
    level, start, end = clear["level"], clear["start_step"], clear["clear_step"]
    game_dir = Path(game_dir)
    game_id = info["roster"].get(game_dir.name, game_dir.name)
    actions = [by_step[s]["action"] for s in range(start + 1, end + 1) if s in by_step]
    caveats = [f"carried_context_omitted:{name}" for name in extras["variants"]]
    record = {
        "id": f"{info['run']}:{game_dir.name}:L{level}",
        "run": info["run"],
        "game": game_id.split("-")[0],
        "game_id": game_id,
        "level": level,
        "total_levels": int(observations[0].get("total_levels") or 0),
        "start_step": start,
        "clear_step": end,
        "actions": actions,
        "action_count": len(actions),
        "teacher": TEACHER,
        "variant": info.get("variant"),
        "context_injections": list(info.get("context_injections") or []),
        "max_model_len": info.get("max_model_len"),
    }
    reasons = []
    if len(actions) != end - start:
        reasons.append("missing_observations")
    if clear["jump"] != 1:
        reasons.append("multi_level_jump")
    if extras["explorer"]:
        reasons.append("explorer_game")
    reasons += [f"unreconstructed_variant:{key}" for key in info.get("unreconstructed") or []]
    external = [e for e in extras["external"] if start < e["step"] <= end]
    if any(not e["auto_reset"] for e in external):
        reasons.append("external_actions")
    record["auto_resets"] = sum(e["auto_reset"] for e in external)
    if not seg_turns or not seg_turns[-1]["requests"] or \
            end not in seg_turns[-1]["requests"][-1]["actions"]:
        reasons.append("clear_not_in_turns")
    model_steps = {s for t in seg_turns for r in t["requests"] for s in r["actions"]}
    if any(s not in model_steps for s in range(start + 1, end + 1)
           if not any(e["step"] == s and e["auto_reset"] for e in external)):
        reasons.append("unattributed_actions")

    for turn in seg_turns:
        if record.get("user_source") in (None, "transcript_verified"):
            record["user_source"] = turn.get("user_source")
    messages, compaction_indices, request_meta, target_start = [{"role": "system", "content": system}], [], [], 1
    starts_after_compaction = False
    if snapshot is not None:
        mine = {t["index"] for t in seg_turns}
        kinds = snapshot["kinds"]
        first = next(i for i, (o, k) in enumerate(zip(snapshot["owners"], kinds, strict=True))
                     if o is not None and o[0] in mine and k != "summary")
        position = {sid: i for i, sid in enumerate(snapshot["ids"])}
        targets = [i for i in range(first, len(kinds)) if kinds[i] == "assistant"]
        contexts = [[position[sid] for sid in snapshot["meta"][i]["context"]] for i in targets]
        begin = min([p for ctx in contexts for p in ctx if p > 0] + [first])
        messages = [snapshot["messages"][0], *snapshot["messages"][begin:]]

        def shift(p):
            return 0 if p == 0 else p - begin + 1

        compaction_indices = [shift(i) for i in range(begin, len(kinds)) if kinds[i] == "summary"]
        target_start = shift(first)
        for i, ctx in zip(targets, contexts, strict=True):
            row = {k: v for k, v in snapshot["meta"][i].items() if k != "context"}
            row["context"] = _ranges([shift(p) for p in ctx])
            request_meta.append(row)
        starts_after_compaction = bool(contexts) and len(contexts[0]) > 1 and kinds[contexts[0][1]] == "summary"
        for owner, notes in snapshot["caveats"].items():
            if owner[0] in mine:
                caveats += notes
    requests = [r for t in seg_turns for r in t["requests"]]
    usages = [r["record"].get("usage") or {} for r in requests]
    record.update(
        turns=len(seg_turns),
        requests=len(requests),
        completion_tokens=sum(int(u.get("completion_tokens") or 0) for u in usages),
        reasoning_tokens=sum(int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
                             for u in usages),
        compaction_completion_tokens=sum((r["summary"] or {}).get("tokens", 0) for r in requests),
        first_prompt_tokens=usages[0].get("prompt_tokens") if usages else None,
        compaction_indices=compaction_indices,
        starts_after_compaction=starts_after_compaction,
        target_start=target_start,
        request_meta=request_meta,
        caveats=sorted(set(caveats)),
        messages=messages,
        tools=None,  # filled once per output file (identical for every record)
        exclusion=reasons,
    )
    return record


def dispatch(request):
    """``[(tool call, tool message content)]`` in the loop's order: each python call gets its
    ``duck_python`` result, another name the loop's unknown-tool error; the loop stopped after
    the call that executed actions (and at the end of the game)."""
    dispatched, results = [], request["python"]
    position = 0
    for call in request["tool_calls"]:
        name = call["function"]["name"]
        if name != "python":
            dispatched.append((call, json.dumps({"error": f"Unknown tool: {name}"}, indent=2)))
            continue
        if position >= len(results):
            break
        dispatched.append((call, json.dumps(results[position]["result"], indent=2)))
        position += 1
        if request["actions"] and position - 1 == request["executing_call"]:
            break
    return dispatched


def _summary_preamble():
    from arena.compaction import SUMMARY_PREAMBLE

    return SUMMARY_PREAMBLE


# --- verification ---------------------------------------------------------------------------------


GENERATED_GAMES_DIR = Path("data/posttrain/games/environment_files")


def environment_dir(root="."):
    return Path(root) / "environment_files"


def game_environments_dir(game_id, environments_dir):
    """Generated games (teacher runs, ``qNNN``) replay from the generated-games store, whose files
    are the ones the run's bundle shipped (same content hash, same game id)."""
    if is_generated(game_id):
        from .games.episodes import environments_dir_for

        # Rejection-sampling episodes (q200+) live in their own batch folders.
        episode = environments_dir_for(game_id, [Path(environments_dir).parent / "data/posttrain/episodes"])
        return episode or Path(environments_dir).parent / GENERATED_GAMES_DIR
    return environments_dir


def replay(game_id, actions, environments_dir):
    """Play ``actions`` from ``make`` in the offline arcade (level-only resets, as the arena ran);
    yields ``(frames as hex rows, levels, state)`` for the start and after every action."""
    import numpy as np
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    os.environ["ONLY_RESET_LEVELS"] = "true"
    logger = logging.getLogger("arc.segments")
    logger.setLevel(logging.WARNING)
    for name in ("arc_agi", "arc_agi.scorecard", "arcengine"):
        logging.getLogger(name).setLevel(logging.WARNING)

    def encode(raw):
        return [["".join(HEX[int(c)] for c in row) for row in np.asarray(frame)] for frame in raw.frame]

    with tempfile.TemporaryDirectory() as recordings:
        arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(Path(environments_dir)
                                                                                   .resolve()),
                        recordings_dir=recordings, logger=logger)
        env = arcade.make(game_id)
        if env is None:
            raise RuntimeError(f"arcade could not load {game_id}")
        raw = env.observation_space or env.reset()
        yield encode(raw), int(raw.levels_completed), getattr(raw.state, "name", str(raw.state))
        for action in actions:
            if action["action"] == "RESET":
                raw = env.reset()
            else:
                data = {"x": action.get("x"), "y": action.get("y")} if action["action"] == "ACTION6" else {}
                raw = env.step(GameAction[action["action"]], data)
            frames = encode(raw) if raw is not None and raw.frame else None
            yield frames, int(raw.levels_completed), getattr(raw.state, "name", str(raw.state))


def verify_game(game_id, observations, clears, environments_dir, replay_fn=replay):
    """``{level: (verified, reason)}``: replay the game's recorded actions from reset up to its
    last clear, comparing every recorded observation (all animation frames, levels, state); a
    level is verified when everything up to its clear reproduced and the clear happened on the
    same action."""
    if not clears:
        return {}
    version = game_id.split("-", 1)[1] if "-" in game_id else ""
    if not (Path(environments_dir) / game_id.split("-")[0] / version).is_dir():
        return {c["level"]: (False, "game_files_missing") for c in clears}
    last = max(c["clear_step"] for c in clears)
    recorded = [o for o in observations if int(o["step"]) <= last]
    if [int(o["step"]) for o in recorded] != list(range(int(recorded[0]["step"]),
                                                          int(recorded[0]["step"]) + len(recorded))):
        return {c["level"]: (False, "observation_gap") for c in clears}
    actions = [o["action"] for o in recorded[1:]]
    first_bad, animation = None, []
    try:
        for index, (frames, levels, state) in enumerate(replay_fn(game_id, actions, environments_dir)):
            want = recorded[index]
            if int(want.get("levels", 0)) != levels:
                first_bad = (int(want["step"]), "level_mismatch")
                break
            if frames is not None and frames[-1] != want["frames"][-1]:
                first_bad = (int(want["step"]), "frame_mismatch" if index else "initial_frame_mismatch")
                break
            if frames is not None and frames != want["frames"]:
                # Only the animation frames before the settled board differ (lf52 animates a
                # random sparkle: two local replays differ too); the board the agent acted on
                # is the last frame.
                animation.append(int(want["step"]))
            if want.get("state") and want["state"] != state:
                first_bad = (int(want["step"]), "state_mismatch")
                break
    except Exception as exc:  # noqa: BLE001 - reported per level
        return {c["level"]: (False, f"engine_error:{type(exc).__name__}") for c in clears}
    result = {}
    for clear in clears:
        if first_bad is None or first_bad[0] > clear["clear_step"]:
            differ = any(step <= clear["clear_step"] for step in animation)
            result[clear["level"]] = (True, "reproduced_final_frames" if differ else "reproduced")
        else:
            result[clear["level"]] = (False, f"{first_bad[1]}_at_step_{first_bad[0]}")
    return result


# --- a game end to end ----------------------------------------------------------------------------


def process_game(game_dir, info, *, environments_dir, images=True, replay_fn=replay):
    """Reconstruct and verify every cleared level of one game; records carry ``verified``,
    ``verify_reason`` and, from the metadata, the baseline, efficiency and level score."""
    game_dir = Path(game_dir)
    game_id = info["roster"].get(game_dir.name, game_dir.name)
    environments_dir = game_environments_dir(game_id, environments_dir)
    result = reconstruct_game(game_dir, info, images=images)
    clears = level_clears(result["observations"])
    if result.get("explorer"):  # the explorer stepped the engine without observation records
        verdicts = {c["level"]: (False, "explorer_game") for c in clears}
    else:
        verdicts = verify_game(game_id, result["observations"], clears, environments_dir, replay_fn)
    baselines = game_baselines(game_id, environments_dir)
    for record in result["segments"]:
        verified, reason = verdicts.get(record["level"], (False, "not_replayed"))
        baseline = baselines[record["level"] - 1] if record["level"] <= len(baselines) else None
        record["baseline_actions"] = baseline
        count = record["action_count"]
        record["efficiency"] = round(baseline / count, 4) if baseline and count else None
        record["level_score"] = round(level_score(baseline, count), 3) if baseline else None
        record["verified"] = verified
        record["verify_reason"] = reason
    return {"segments": result["segments"], "prompt_check": result["prompt_check"],
            "prompt_mismatches": result.get("prompt_mismatches", []),
            "aligned": result.get("aligned", False), "has_transcript": result.get("has_transcript"),
            "game_id": game_id, "turns": result.get("turns", 0)}


def game_baselines(game_id, environments_dir):
    prefix, _, version = game_id.partition("-")
    path = Path(environments_dir) / prefix / version / "metadata.json"
    if not path.is_file():
        return []
    return list(json.loads(path.read_text()).get("baseline_actions") or [])


GAME_SETS = {
    "dev": is_dev,
    "generated": is_generated,  # training mechanics and the held-out ones (the eval split)
    "all": lambda game_id: is_dev(game_id) or is_generated(game_id),
}


def teacher_games(root=".", patterns=RUN_PATTERNS, games="all"):
    """``(game_dir, info)`` for every recorded game of the teacher on a DEV or generated game
    (``games``: a ``GAME_SETS`` name), and a tally of what was skipped."""
    wanted = GAME_SETS[games]
    found, skipped = [], {"non_teacher_trials": [], "holdout_or_other_games": 0, "other_set_games": 0}
    for trial in find_trials(root, patterns):
        info = trial_info(trial)
        dirs = sorted(p for p in (trial / "games").iterdir() if (p / "log.jsonl").is_file())
        if info["effort"] is None and dirs:  # lost run: the effort the replies were made with
            for record in read_log(dirs[0] / "log.jsonl"):
                if record["kind"] == "model" and record.get("purpose") == "gameplay":
                    info["effort"] = record.get("reasoning_effort")
                    break
        if info["model"] != TEACHER_MODEL or info["effort"] != TEACHER_EFFORT:
            skipped["non_teacher_trials"].append(f"{info['run']} ({info['model']}, {info['effort']})")
            continue
        for game_dir in dirs:
            game_id = info["roster"].get(game_dir.name, game_dir.name)
            if wanted(game_id):
                found.append((game_dir, info))
            elif is_dev(game_id) or is_generated(game_id):
                skipped["other_set_games"] += 1
            else:
                skipped["holdout_or_other_games"] += 1
    return found, skipped


def deduplicate(records):
    """Keep the first record of each (game, level, action sequence); the kept one counts its
    copies (``sequence_count``) and names the others (``duplicate_ids``)."""
    kept, index = [], {}
    for record in records:
        key = (record["game_id"], record["level"], json.dumps(record["actions"], sort_keys=True))
        if key in index:
            first = index[key]
            first["sequence_count"] += 1
            first["duplicate_ids"].append(record["id"])
            continue
        record["sequence_count"], record["duplicate_ids"] = 1, []
        index[key] = record
        kept.append(record)
    return kept


def _summary(values):
    if not values:
        return None
    return {"n": len(values), "min": round(min(values), 3), "median": round(statistics.median(values), 3),
            "mean": round(statistics.fmean(values), 3), "max": round(max(values), 3)}


def build_stats(all_records, kept, extra):
    """Per game and level: segments, verified, unique, failures, actions vs baseline, scores and
    tokens; plus totals and the prompt-simulation check."""
    per = {}
    for record in all_records:
        key = f"{record['game']}:L{record['level']}"
        row = per.setdefault(key, {"game": record["game"], "level": record["level"], "segments": 0,
                                   "verified": 0, "unique_verified": 0, "failed": {}, "excluded": {},
                                   "baseline_actions": record.get("baseline_actions"),
                                   "_actions": [], "_scores": [], "_tokens": []})
        row["segments"] += 1
        if not record["verified"]:
            row["failed"][record["verify_reason"]] = row["failed"].get(record["verify_reason"], 0) + 1
        elif record["exclusion"]:
            for reason in record["exclusion"]:
                row["excluded"][reason] = row["excluded"].get(reason, 0) + 1
        else:
            row["verified"] += 1
            row["_actions"].append(record["action_count"])
            if record.get("level_score") is not None:
                row["_scores"].append(record["level_score"])
            row["_tokens"].append(record["completion_tokens"])
    for record in kept:
        per[f"{record['game']}:L{record['level']}"]["unique_verified"] += 1
    rows = []
    for key in sorted(per, key=lambda k: (per[k]["game"], per[k]["level"])):
        row = per[key]
        actions, scores, tokens = row.pop("_actions"), row.pop("_scores"), row.pop("_tokens")
        row["median_actions"] = statistics.median(actions) if actions else None
        row["min_actions"] = min(actions) if actions else None
        row["score"] = _summary(scores)
        row["completion_tokens"] = {"sum": sum(tokens), "median": statistics.median(tokens)} if tokens else None
        rows.append(row)
    usable = [r for r in all_records if r["verified"] and not r["exclusion"]]
    failed, excluded = {}, {}
    for record in all_records:
        if not record["verified"]:
            failed[record["verify_reason"]] = failed.get(record["verify_reason"], 0) + 1
        elif record["exclusion"]:
            for reason in record["exclusion"]:
                excluded[reason] = excluded.get(reason, 0) + 1
    caveats = {}
    for record in kept:
        for caveat in record["caveats"]:
            caveats[caveat] = caveats.get(caveat, 0) + 1
    scores = [r["level_score"] for r in kept if r.get("level_score") is not None]
    totals = {
        "segments": len(all_records),
        "verified": sum(r["verified"] for r in all_records),
        "usable_verified": len(usable),
        "unique_verified": len(kept),
        "failed_verification": failed,
        "excluded_after_verification": excluded,
        "completion_tokens_usable": sum(r["completion_tokens"] for r in usable),
        "completion_tokens_unique": sum(r["completion_tokens"] for r in kept),
        "reasoning_tokens_unique": sum(r["reasoning_tokens"] for r in kept),
        "user_source_unique": {s: sum(r.get("user_source") == s for r in kept)
                               for s in ("transcript_verified", "transcript", "simulated")},
        "starts_after_compaction_unique": sum(r["starts_after_compaction"] for r in kept),
        "with_compaction_inside_unique": sum(
            bool(r["compaction_indices"]) and not r["starts_after_compaction"] for r in kept),
        "caveats_unique": caveats,
        "level_score_unique": _summary(scores),
        "score_histogram_unique": _histogram(scores),
    }
    return {"teacher": TEACHER, "totals": totals, **extra, "per_game_level": rows}


def _histogram(scores):
    edges = [0, 10, 25, 50, 75, 100, 115.01]
    counts = {}
    for low, high in zip(edges, edges[1:], strict=False):
        counts[f"{low}-{min(high, 115):g}"] = sum(low <= s < high for s in scores)
    return counts


def finalize(results, skipped, tools, *, with_eval=False):
    """Split processed games into the main output (verified, unexcluded, deduplicated), the
    rejected list (without messages) and the stats. Segments of the held-out generated mechanics
    (``split.GENERATED_HOLDOUT``) never reach the main output: with ``with_eval`` they are returned
    as a fourth value (the evaluation split), otherwise ``check_training_games`` rejects them."""
    all_records = [r for result in results for r in result["segments"]]
    held = [r for r in all_records if is_generated_holdout(r["game_id"])] if with_eval else []
    all_records = [r for r in all_records if not (with_eval and is_generated_holdout(r["game_id"]))]
    check_training_games(r["game_id"] for r in all_records)
    usable = [r for r in all_records if r["verified"] and not r["exclusion"]]
    usable.sort(key=lambda r: (r["game_id"], r["level"], r["run"], r["id"]))
    kept = deduplicate(usable)
    for record in kept:
        record["tools"] = tools
    eval_kept = deduplicate(sorted((r for r in held if r["verified"] and not r["exclusion"]),
                                   key=lambda r: (r["game_id"], r["level"], r["run"], r["id"])))
    check_eval_games(r["game_id"] for r in eval_kept)
    for record in eval_kept:
        record["tools"] = tools
    rejected = [{k: v for k, v in r.items() if k not in ("messages", "tools")}
                for r in all_records if not (r["verified"] and not r["exclusion"])]
    check = {"turns": 0, "user_exact": 0, "system_exact": 0, "followups": 0, "followup_exact": 0}
    for result in results:
        for key in check:
            check[key] += result["prompt_check"].get(key, 0)
    extra = {
        "games_processed": len(results),
        "games_with_transcript": sum(bool(r.get("has_transcript")) for r in results),
        "games_transcript_aligned": sum(bool(r.get("aligned")) for r in results),
        "prompt_simulation": {**check, "mismatch_examples": [
            m for result in results for m in result.get("prompt_mismatches", [])][:20]},
        "skipped": skipped,
    }
    stats = build_stats(all_records, kept, extra)
    if not with_eval:
        return kept, rejected, stats
    stats["eval_split"] = {"segments": len(held), "unique_verified": len(eval_kept),
                           "games": sorted({r["game_id"] for r in eval_kept}),
                           "completion_tokens": sum(r["completion_tokens"] for r in eval_kept)}
    return kept, rejected, stats, eval_kept
