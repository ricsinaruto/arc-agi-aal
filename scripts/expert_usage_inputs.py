#!/usr/bin/env python3
"""Inputs of the expert-usage measurement (28 September; docs/validation-20260927/expert-usage.md).

Teacher-forced sequences rebuilt from recorded Flash xhigh one-hour arena runs (the primary corpus of
``scripts/specialization_vocab.py``). Each game's ``prompts/prompt.log`` holds its latest model request
(system prompt, user turns, every earlier assistant turn with its reasoning and tool call, tool results);
it is parsed back into chat messages, rendered with the served chat template (tools, xhigh effort) and
tokenized with the served tokenizer, so a sequence is what the server prefilled for that request up to
its generation prompt. The assistant turns in it are the model's own outputs.

Split by game family (the 4-letter source game): a usage set (~1.5M tokens, ~100 games) and a held-out
set of other families (~0.4M tokens). Sequences are capped at ``--cap`` tokens (a prefix).

Usage: tokvenv/bin/python scripts/expert_usage_inputs.py --out runs/expert-usage/inputs-v1
Needs ``tokenizers`` and ``jinja2`` (the scratchpad tokvenv); nothing touches a GPU.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from specialization_vocab import RUNS, primary_runs  # noqa: E402

TOKENIZER_DIR = f"{RUNS}/serving-research/qwen-tokenizer"
IM_START, IM_END = 248045, 248046
ASSISTANT = 74455  # "assistant" after <|im_start|>
HEADER = re.compile(r"^\[([A-Z][A-Z _:a-z0-9\-]*)\]\s*$")
PARAMETER_DESCRIPTION = "Python code to run in this game's persistent session; definitions persist across tool calls."
TOOL_CALL = re.compile(r"<function=([^>\n]+)>\n(.*)</function>\s*</tool_call>\s*$", re.S)
PARAM = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>\n", re.S)


def family(run_dir: str, game_dir: str) -> str:
    name = os.path.basename(game_dir)
    roster = os.path.join(os.path.dirname(os.path.dirname(game_dir)), "public-roster.json")
    if re.fullmatch(r"k\d{3}", name) and os.path.exists(roster):
        for row in json.load(open(roster)):
            if row["game_id"] == name:
                return row["source_game_id"][:4]
        raise ValueError(f"{game_dir}: clone missing from the roster")
    return name[:4]


def parse_prompt_log(path: str):
    """(tool description, messages) of the logged request; None when the log is not a full snapshot."""
    lines = open(path, errors="replace").read().split("\n")
    if "[MODEL INPUT]" not in lines or "[TURN TRANSCRIPT SO FAR]" not in lines:
        return None
    tools_at = lines.index("[AVAILABLE TOOLS]") if "[AVAILABLE TOOLS]" in lines else None
    description = None
    if tools_at is not None:
        text = "\n".join(lines[tools_at + 1:lines.index("[MODEL INPUT]")]).strip()
        if text.startswith("- python: "):
            description = text[len("- python: "):]
    start, end = lines.index("[MODEL INPUT]") + 1, lines.index("[TURN TRANSCRIPT SO FAR]")
    sections, current, buf = [], None, []
    for line in lines[start:end]:
        m = HEADER.match(line)
        if m:
            if current is not None:
                sections.append((current, "\n".join(buf)))
            current, buf = m.group(1), []
        else:
            buf.append(line)
    if current is not None:
        sections.append((current, "\n".join(buf)))
    messages = []
    for head, body in sections:
        if head == "SYSTEM":
            messages.append({"role": "system", "content": body.strip("\n")})
        elif head == "USER":
            messages.append({"role": "user", "content": body.strip("\n")})
        elif head == "ASSISTANT":
            messages.append({"role": "assistant", "content": body.strip("\n"), "reasoning_content": "",
                             "tool_calls": []})
        elif head == "REASONING":
            if not messages or messages[-1]["role"] != "assistant":
                return None
            messages[-1]["reasoning_content"] = body.strip("\n")
        elif head.startswith("ASSISTANT TOOL CALL"):
            if not messages or messages[-1]["role"] != "assistant":
                return None
            text = body.strip("\n")
            call_id = ""
            if text.startswith("id: "):
                call_id, _, text = text.partition("\n")
                call_id = call_id[4:].strip()
            m = TOOL_CALL.search(text)
            if not m:
                return None
            arguments = {k: v for k, v in PARAM.findall(m.group(2) + "\n")}
            messages[-1]["tool_calls"].append({"id": call_id, "type": "function",
                                               "function": {"name": m.group(1), "arguments": arguments}})
        elif head.startswith("TOOL RESULT"):
            messages.append({"role": "tool", "content": body.strip("\n"),
                             "tool_call_id": head.partition(":")[2].strip()})
        else:
            return None
    if not messages or messages[0]["role"] != "system":
        return None
    return description, messages


def normalize(text) -> str:
    """The prompt log's rendering of a message text (tool_agent._normalize_message_content)."""
    text = re.sub(r"</?think>", "", str(text or ""), flags=re.IGNORECASE)
    return "\n".join(line for line in text.splitlines() if line.strip()).strip()


def tool_display(result: dict) -> str | None:
    """tool_agent._render_tool_result_display for the shapes it prints verbatim (stdout, or error only)."""
    stdout = str(result.get("stdout", "") or "").rstrip("\n")
    error = str(result.get("error", "") or "").rstrip("\n")
    has_result = result.get("result") not in (None, "", [], {})
    if stdout and not error and not has_result:
        return stdout
    if error and not stdout and not has_result:
        return error
    return None


def exact_sources(game_dir: str) -> dict:
    """Exact texts the prompt log shows normalized: model replies and tool payloads (log.jsonl) and the
    system prompts (duck-transcript.txt). Keyed by their prompt-log rendering."""
    replies, tools, systems = {}, {}, {}
    for line in open(os.path.join(game_dir, "log.jsonl"), errors="replace"):
        if '"model"' not in line and '"duck_python"' not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "model" and isinstance(record.get("message"), dict):
            message = record["message"]
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            if reasoning:
                replies[normalize(reasoning)] = message
        elif record.get("kind") == "duck_python" and isinstance(record.get("result"), dict):
            display = tool_display(record["result"])
            if display is not None:
                tools[display] = json.dumps(record["result"], indent=2)
    transcript = os.path.join(game_dir, "duck-transcript.txt")
    if os.path.exists(transcript):
        current, buf = None, []
        for line in open(transcript, errors="replace").read().split("\n") + ["[END]"]:
            m = HEADER.match(line)
            if m or line.startswith("--- analysis_step="):
                if current == "SYSTEM PROMPT":
                    text = "\n".join(buf).strip()
                    systems[normalize(text)] = text
                current, buf = (m.group(1) if m else None), []
            else:
                buf.append(line)
    return {"replies": replies, "tools": tools, "systems": systems}


def exactify(messages: list[dict], sources: dict, stats: dict) -> list[dict]:
    """Replace the prompt log's normalized texts with the exact request texts where they are recorded."""
    out = []
    for message in messages:
        message = dict(message)
        role = message["role"]
        if role == "system":
            exact = sources["systems"].get(message["content"])
            stats["system"][exact is not None] += 1
            if exact is not None:
                message["content"] = exact
        elif role == "assistant" and message.get("reasoning_content"):
            exact = sources["replies"].get(message["reasoning_content"])
            stats["assistant"][exact is not None] += 1
            if exact is not None:
                message["reasoning_content"] = (exact.get("reasoning") or exact.get("reasoning_content")).strip()
                calls = []
                for call in exact.get("tool_calls") or []:
                    function = call.get("function", call)
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = None
                    if not isinstance(arguments, dict):
                        calls = None
                        break
                    calls.append({"id": call.get("id", ""), "type": "function",
                                  "function": {"name": function.get("name"), "arguments": arguments}})
                if calls and len(calls) == len(message.get("tool_calls") or []):
                    message["tool_calls"] = calls
                if exact.get("content") and message.get("content") == normalize(exact["content"]):
                    message["content"] = exact["content"]
        elif role == "tool":
            exact = sources["tools"].get(message["content"])
            stats["tool"][exact is not None] += 1
            # Unmatched: the payload shape the agent sends (json.dumps(payload, indent=2)) around the text.
            message["content"] = exact if exact is not None else json.dumps(
                {"tool": "python", "returncode": 0, "stdout": message["content"] + "\n"}, indent=2)
        out.append(message)
    return out


def renderer():
    import jinja2

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def tojson(value, indent=None, separators=None, sort_keys=False):
        # transformers' chat-template tojson (no HTML escaping, key order kept).
        return json.dumps(value, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(open(os.path.join(TOKENIZER_DIR, "chat_template.jinja")).read())


def assistant_spans(ids):
    """[start, end) token spans of assistant turns: after '<|im_start|>assistant', through '<|im_end|>'."""
    spans, i, n = [], 0, len(ids)
    while i < n - 1:
        if ids[i] == IM_START and ids[i + 1] == ASSISTANT:
            j = i + 2
            while j < n and ids[j] != IM_END:
                j += 1
            spans.append([i + 2, min(j + 1, n)])
            i = j
        i += 1
    return spans


def main():
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--usage-tokens", type=int, default=1_500_000)
    parser.add_argument("--heldout-tokens", type=int, default=400_000)
    parser.add_argument("--cap", type=int, default=65_000)
    parser.add_argument("--heldout-families", default="sb26,s5i5,re86,g50t,ls20,tr87",
                        help="Held-out game families (the harness-evolution held-out roster, minus su15/dc22/cd82)")
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    held = set(args.heldout_families.split(","))
    tok = Tokenizer.from_file(os.path.join(TOKENIZER_DIR, "tokenizer.json"))
    template = renderer()
    games = []
    for run in primary_runs():
        for log in sorted(glob.glob(os.path.join(RUNS, run, "**/games/*/prompts/prompt.log"), recursive=True)):
            gdir = os.path.dirname(os.path.dirname(log))
            games.append((run, gdir, log))
    rng = random.Random(args.seed)
    rng.shuffle(games)
    sets = {"usage": [], "heldout": []}
    totals = {"usage": 0, "heldout": 0}
    targets = {"usage": args.usage_tokens, "heldout": args.heldout_tokens}
    skipped = {"unparsed": 0, "short": 0}
    per_family = {"usage": {}, "heldout": {}}
    match = {"system": [0, 0], "assistant": [0, 0], "tool": [0, 0]}  # [unmatched, matched]
    for run, gdir, log in games:
        fam = family(os.path.join(RUNS, run), gdir)
        split = "heldout" if fam in held else "usage"
        if totals[split] >= targets[split]:
            if all(totals[k] >= targets[k] for k in totals):
                break
            continue
        # Spread each set over families: at most ~1/6 of a set's tokens from one family.
        if per_family[split].get(fam, 0) > targets[split] / (6 if split == "heldout" else 12):
            continue
        parsed = parse_prompt_log(log)
        if parsed is None:
            skipped["unparsed"] += 1
            continue
        description, messages = parsed
        messages = exactify(messages, exact_sources(gdir), match)
        tools = [{"type": "function", "function": {
            "name": "python", "description": description,
            "parameters": {"type": "object", "properties": {"code": {"type": "string",
                                                                      "description": PARAMETER_DESCRIPTION}},
                           "required": ["code"]}}}]
        text = template.render(messages=messages, tools=tools, add_generation_prompt=True,
                               enable_thinking=True, reasoning_effort="xhigh")
        ids = tok.encode(text, add_special_tokens=False).ids
        full = len(ids)
        ids = ids[:args.cap]
        spans = assistant_spans(ids)
        if len(ids) < 4000 or not spans:
            skipped["short"] += 1
            continue
        # The server's own count of the game's last admitted request (normally this snapshot; it includes the
        # newest turn's 64-token grid image, which token ids cannot carry).
        admitted = None
        for line in open(os.path.join(gdir, "log.jsonl"), errors="replace"):
            if '"exact_admission_count"' in line:
                admitted = int(json.loads(line).get("prompt_tokens") or 0) or None
        sets[split].append({"index": len(sets[split]), "run": run, "game": os.path.basename(gdir), "family": fam,
                            "tokens": len(ids), "full_tokens": full, "admitted_tokens": admitted,
                            "assistant_spans": spans,
                            "assistant_tokens": sum(b - a for a, b in spans), "ids": ids})
        totals[split] += len(ids)
        per_family[split][fam] = per_family[split].get(fam, 0) + len(ids)
    os.makedirs(args.out, exist_ok=True)
    print("exact-text matches [unmatched, matched]:", match, flush=True)
    summary = {"seed": args.seed, "cap": args.cap, "heldout_families": sorted(held), "skipped": skipped,
               "exact_text_matches": {k: {"unmatched": v[0], "matched": v[1]} for k, v in match.items()},
               "tokenizer": TOKENIZER_DIR, "chat_template_sha256": hashlib.sha256(
                   open(os.path.join(TOKENIZER_DIR, "chat_template.jinja"), "rb").read()).hexdigest()}
    for split, rows in sets.items():
        path = os.path.join(args.out, f"{split}.json")
        with open(path, "w") as f:
            json.dump(rows, f, separators=(",", ":"))
        summary[split] = {
            "sequences": len(rows), "tokens": totals[split],
            "assistant_tokens": sum(r["assistant_tokens"] for r in rows),
            "truncated": sum(r["full_tokens"] > r["tokens"] for r in rows),
            "families": dict(sorted(per_family[split].items())),
            "runs": len({r["run"] for r in rows}),
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "rows": [{k: r[k] for k in ("index", "run", "game", "family", "tokens", "full_tokens", "admitted_tokens",
                                        "assistant_tokens")} for r in rows],
        }
        print(split, summary[split]["sequences"], summary[split]["tokens"], summary[split]["assistant_tokens"],
              summary[split]["families"], flush=True)
    with open(os.path.join(args.out, "selection.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
