#!/usr/bin/env python3
"""Training sequences for the MTP draft-head fine-tune (28 September; docs/validation-20260927/draft-head-finetune.md).

CPU only. One sequence per recorded game: the conversation of the game's last model call (its
``prompts/prompt.log`` snapshot) rendered with the served chat template, every assistant turn replaced
by the lossless ``log.jsonl`` record of that call (reasoning, content, tool calls; matched by tool-call
id). The loss mask marks the generated tokens (everything the model emitted after
``<|im_start|>assistant\\n<think>\\n``, through ``<|im_end|>``); everything else is context.

- Split by game: the 9 posttrain holdout games (``posttrain.split.HOLDOUT``) are the held-out evaluation
  set and never enter training; the other roster games train.
- The snapshot's system/user/tool text is the prompt log's rendering (blank lines inside a message are
  dropped there), so context tokens are close to, not byte-identical with, the served request. The
  check column compares the rendered prefix of each assistant turn with that call's recorded
  ``usage.prompt_tokens`` (exact only while the harness had not yet trimmed history).
- All generated text is Flash's own output (no Claude trajectories).

Usage: tokvenv/bin/python scripts/draft_head_data.py --out DIR [--max-tokens 32768] [--train-games N]
       [--eval-games N] [--runs GLOB ...]
Writes ``train.npz`` / ``eval.npz`` (tokens int32, loss uint8, offsets int64) and ``manifest.json``.
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
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from posttrain.split import EXCLUDED, HOLDOUT, prefix  # noqa: E402

# Throughput arm (6 October): the 10 dev / 15 held-out split; ls20 and re86 are the dev evaluation families (as the
# expert-quant tuning inputs); held-out games never enter either split.
DEV10 = ("cd82", "cn04", "ft09", "lp85", "ls20", "re86", "sb26", "sk48", "vc33", "wa30")
DEV10_EVAL = ("ls20", "re86")

RUNS = Path(os.environ.get("ARC_RUNS", "./runs"))
TOKENIZER_DIR = RUNS / "serving-research/qwen-tokenizer"
# The specialization corpus (docs/validation-20260927/specialization.md): Flash xhigh one-hour runs.
PRIMARY_GLOBS = ["modal-validation/arena-duck-60m-c110-*", "modal-validation/arena-duck-60m-r23-*",
                 "kaggle-arena-duck-60m-r23-*"]
PRIMARY_EXCLUDE = ("effort-medium", "no-think", "ornith", "adaptive-think")
HEADER = re.compile(r"^\[(SYSTEM|USER|ASSISTANT|TOOL RESULT: [^\]]+)\]$")
GEN_PREFIX = "<|im_start|>assistant\n<think>\n"
END = "<|im_end|>"
PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": None,  # the snapshot's [AVAILABLE TOOLS] line
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": None}},
                       "required": ["code"]},
    },
}
PERSISTENT_PARAM = "Python code to run in this game's persistent session; definitions persist across tool calls."
EPHEMERAL_PARAM = "Python code to run. The snippet is ephemeral and is not saved across tool calls."


def primary_runs():
    out = []
    for pattern in PRIMARY_GLOBS:
        for run in sorted(glob.glob(str(RUNS / pattern))):
            name = os.path.relpath(run, RUNS)
            if not any(x in name for x in PRIMARY_EXCLUDE):
                out.append(name)
    return out


def game_dirs(run):
    for log in sorted(glob.glob(str(RUNS / run / "**/games/*/log.jsonl"), recursive=True)):
        gdir = Path(log).parent
        if (gdir / "prompts/prompt.log").is_file():
            yield gdir


def game_id(gdir):
    """The roster game (``ar25``) of a game directory; clone runs map ``k000`` through public-roster.json."""
    name = gdir.name
    roster = gdir.parent.parent / "public-roster.json"
    if roster.is_file():
        for row in json.loads(roster.read_text()):
            if row.get("game_id") == name:
                return prefix(row["source_game_id"])
    return prefix(name)


def parse_snapshot(text):
    """``(tool_description, [(role, body_lines, tool_call_ids)])`` from a prompt-log snapshot."""
    tools = text.split("[AVAILABLE TOOLS]\n", 1)[1].split("\n\n[MODEL INPUT]", 1)[0]
    description = None
    for line in tools.splitlines():
        if line.startswith("- python: "):
            description = line[len("- python: "):]
    body = text.split("[MODEL INPUT]\n", 1)[1].split("\n\n[TURN TRANSCRIPT SO FAR]", 1)[0]
    lines = body.split("\n")
    blocks, current = [], None
    for index, line in enumerate(lines):
        match = HEADER.match(line)
        if match and (index == 0 or lines[index - 1] == ""):
            if current:
                blocks.append(current)
            current = [match.group(1), [], []]
            continue
        if current is None:
            continue
        if current[0] == "ASSISTANT" and line.startswith("id: chatcmpl-tool-"):
            current[2].append(line[4:].strip())
        current[1].append(line)
    if current:
        blocks.append(current)
    for block in blocks:  # the join between messages is one blank line
        while block[1] and block[1][-1] == "":
            block[1].pop()
    return description, blocks


def model_records(gdir):
    records = []
    for line in open(gdir / "log.jsonl", errors="replace"):
        if '"model"' not in line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("kind") == "model" and isinstance(d.get("message"), dict):
            records.append(d)
    return records


def assistant_message(msg):
    reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "")
    calls = []
    for call in msg.get("tool_calls") or []:
        fn = call.get("function", call)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        calls.append({"type": "function", "function": {"name": fn.get("name"), "arguments": args}})
    out = {"role": "assistant", "content": msg.get("content") or "", "reasoning_content": reasoning}
    if calls:
        out["tool_calls"] = calls
    return out


def build_messages(gdir):
    description, blocks = parse_snapshot((gdir / "prompts/prompt.log").read_text(errors="replace"))
    records = model_records(gdir)
    by_id = {}
    for index, rec in enumerate(records):
        for call in rec["message"].get("tool_calls") or []:
            if call.get("id"):
                by_id[call["id"]] = index
    messages, used, unmatched = [], [], 0
    for role, lines, ids in blocks:
        text = "\n".join(lines).strip()
        if role == "SYSTEM":
            messages.append({"role": "system", "content": text})
        elif role == "USER":
            messages.append({"role": "user", "content": text})
        elif role.startswith("TOOL RESULT"):
            messages.append({"role": "tool", "content": text})
        else:
            index = next((by_id[i] for i in ids if i in by_id), None)
            if index is None:
                unmatched += 1
                messages.append(None)  # unmatched assistant turn: the sequence is cut before it
                continue
            used.append(index)
            messages.append(assistant_message(records[index]["message"]))
    if None in messages:
        messages = messages[:messages.index(None)]
        while messages and messages[-1]["role"] != "tool":
            messages.pop()
    tool = json.loads(json.dumps(PYTHON_TOOL))
    tool["function"]["description"] = description or ""
    persistent = "persistent interpreter" in (description or "")
    tool["function"]["parameters"]["properties"]["code"]["description"] = (
        PERSISTENT_PARAM if persistent else EPHEMERAL_PARAM)
    return messages, [tool], [records[i] for i in used], unmatched


def template_env():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                          sort_keys=sort_keys)

    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string((TOKENIZER_DIR / "chat_template.jinja").read_text())


def render(template, messages, tools):
    return template.render(messages=messages, tools=tools, add_generation_prompt=False,
                           reasoning_effort="xhigh")


def encode(tokenizer, text, spans):
    """Token ids of ``text`` and a 0/1 mask of tokens whose characters lie in ``spans``."""
    enc = tokenizer.encode(text, add_special_tokens=False)
    ids = np.asarray(enc.ids, dtype=np.int32)
    starts = np.asarray([o[0] for o in enc.offsets], dtype=np.int64)
    mask = np.zeros(len(ids), dtype=np.uint8)
    for a, b in spans:
        mask[(starts >= a) & (starts < b)] = 1
    return ids, mask, starts


def sequence(gdir, template, tokenizer, max_tokens):
    messages, tools, records, unmatched = build_messages(gdir)
    if not any(m["role"] == "assistant" for m in messages):
        return None
    text = render(template, messages, tools)
    spans, check, cursor = [], [], 0
    for rec in records:
        start = text.index(GEN_PREFIX, cursor) + len(GEN_PREFIX)
        end = text.index(END, start) + len(END)
        spans.append((start, end))
        cursor = end
    ids, mask, starts = encode(tokenizer, text, spans)
    for rec, (start, _) in zip(records, spans):
        prefix_tokens = int(np.searchsorted(starts, start))
        recorded = int((rec.get("usage") or {}).get("prompt_tokens") or -1)
        check.append((prefix_tokens, recorded))
    if len(ids) > max_tokens:
        # Keep whole turns: cut after the last generation that ends inside the budget.
        ends = [int(np.searchsorted(starts, b)) for _, b in spans]
        keep = [e for e in ends if e <= max_tokens]
        if not keep:
            return None
        ids, mask = ids[:keep[-1]], mask[:keep[-1]]
        check = check[:len(keep)]
    return {"tokens": ids, "loss": mask, "check": check, "unmatched": unmatched,
            "calls": len(check), "generated": int(mask.sum())}


GEN_PREFIX_IDS = (198, 248068, 198)  # "\n<think>\n" after "<|im_start|>assistant": the generation prompt


def from_usage(args):
    """Splits from expert-usage inputs (``scripts/expert_usage_inputs.py``: the same snapshots rendered with the
    served template, exact replies, tool payloads and system prompts). Generated = each assistant span after its
    ``\n<think>\n`` generation prompt, through ``<|im_end|>`` (or the cut)."""
    option = lambda rows: dict(r.split("=", 1) for r in rows)  # noqa: E731
    sources, families, per_family = option(args.from_usage), option(args.usage_families), option(args.usage_per_family)
    key = lambda r: (r["run"], r["game"], r["full_tokens"])  # noqa: E731
    splits = {}
    for split, path in sources.items():
        rows = json.loads(Path(path).read_text())
        keep = set(families[split].split(",")) if split in families else None
        cap, taken, out = int(per_family.get(split, 0)), {}, []
        for row in rows:
            fam = row["family"]
            if keep is not None and fam not in keep:
                continue
            ids, spans = row["ids"], row["assistant_spans"]
            if cap:
                room = cap - taken.get(fam, 0)
                if room <= 0:
                    continue
                ids = ids[:room]
                spans = [[a, min(b, len(ids))] for a, b in spans if a < len(ids)]
            taken[fam] = taken.get(fam, 0) + len(ids)
            mask = np.zeros(len(ids), dtype=np.uint8)
            for a, b in spans:
                if tuple(ids[a:a + 3]) != GEN_PREFIX_IDS and b - a >= 3:
                    raise ValueError(f"{split}: assistant span without the generation prompt")
                mask[a + 3:b] = 1
            out.append({"key": key(row), "family": fam, "run": row["run"], "game": row["game"],
                        "ids": np.asarray(ids, dtype=np.int32), "mask": mask})
        splits[split] = out
    other = {r["key"] for split, rows in splits.items() if split != args.usage_disjoint for r in rows}
    dropped = [r for r in splits.get(args.usage_disjoint, []) if r["key"] in other]
    if dropped:
        splits[args.usage_disjoint] = [r for r in splits[args.usage_disjoint] if r["key"] not in other]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "expert-usage inputs", "inputs": {
        k: {"path": v, "sha256": hashlib.sha256(Path(v).read_bytes()).hexdigest()} for k, v in sources.items()},
        "families": families, "per_family_tokens": per_family, "disjoint": args.usage_disjoint,
        "dropped_overlap": [{"family": r["family"], "run": r["run"], "game": r["game"], "tokens": len(r["ids"])}
                            for r in dropped], "splits": {}}
    for split, rows in splits.items():
        offsets = np.cumsum([0] + [len(r["ids"]) for r in rows]).astype(np.int64)
        np.savez(out_dir / f"{split}.npz", tokens=np.concatenate([r["ids"] for r in rows]),
                 loss=np.concatenate([r["mask"] for r in rows]), offsets=offsets)
        per = {}
        for r in rows:
            row = per.setdefault(r["family"], {"sequences": 0, "tokens": 0, "generated_tokens": 0})
            row["sequences"] += 1
            row["tokens"] += len(r["ids"])
            row["generated_tokens"] += int(r["mask"].sum())
        manifest["splits"][split] = {
            "sequences": len(rows), "games": sorted(per), "tokens": int(offsets[-1]),
            "generated_tokens": int(sum(v["generated_tokens"] for v in per.values())), "calls": None,
            "per_family": dict(sorted(per.items())),
            "sequences_meta": [{"family": r["family"], "run": r["run"], "game": r["game"], "tokens": len(r["ids"]),
                                "generated": int(r["mask"].sum())} for r in rows]}
        print(json.dumps({k: v for k, v in manifest["splits"][split].items() if k not in ("sequences_meta", "per_family")}))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--train-games", type=int, default=0, help="cap (0: all)")
    parser.add_argument("--eval-games", type=int, default=0, help="cap (0: all)")
    parser.add_argument("--runs", nargs="*", help="run globs relative to the runs root (default: primary corpus)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("posttrain", "dev10"), default="posttrain")
    parser.add_argument("--from-usage", nargs="+", metavar="SPLIT=JSON",
                        help="7 Oct: splits from expert-usage input files (rows with ids and assistant_spans) "
                             "instead of the recorded runs")
    parser.add_argument("--usage-families", nargs="*", default=[], metavar="SPLIT=a,b",
                        help="keep only these families of a --from-usage split")
    parser.add_argument("--usage-per-family", nargs="*", default=[], metavar="SPLIT=N",
                        help="whole sequences per family up to N tokens (the last one cut)")
    parser.add_argument("--usage-disjoint", default="train",
                        help="rows of this split that also appear in another split are dropped")
    args = parser.parse_args(argv)
    if args.from_usage:
        return from_usage(args)
    holdout, excluded = HOLDOUT, EXCLUDED
    if args.split == "dev10":
        holdout = DEV10_EVAL
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))
    template = template_env()
    runs = primary_runs() if not args.runs else sorted(
        os.path.relpath(r, RUNS) for pattern in args.runs for r in glob.glob(str(RUNS / pattern)))
    candidates = {"train": [], "eval": []}
    for run in runs:
        for gdir in game_dirs(run):
            gid = game_id(gdir)
            if gid in excluded or (args.split == "dev10" and gid not in DEV10):
                continue
            candidates["eval" if gid in holdout else "train"].append((run, gdir, gid))
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"max_tokens": args.max_tokens, "runs": runs, "split": args.split, "holdout_games": list(holdout),
                "excluded_games": list(excluded), "splits": {}}
    for split, rows in candidates.items():
        rng.shuffle(rows)
        cap = args.train_games if split == "train" else args.eval_games
        tokens, loss, offsets, meta, deltas = [], [], [0], [], []
        for run, gdir, gid in rows:
            if cap and len(meta) >= cap:
                break
            try:
                seq = sequence(gdir, template, tokenizer, args.max_tokens)
            except (ValueError, KeyError, IndexError) as exc:
                print(f"skip {gdir}: {type(exc).__name__}: {exc}"[:200], file=sys.stderr)
                continue
            if seq is None:
                continue
            if split == "train" and (gid in holdout or (args.split == "dev10" and gid not in DEV10)):
                raise AssertionError("holdout game in training data")
            tokens.append(seq["tokens"])
            loss.append(seq["loss"])
            offsets.append(offsets[-1] + len(seq["tokens"]))
            deltas.extend(p - r for p, r in seq["check"] if r > 0)
            meta.append({"run": run, "game_dir": str(gdir.relative_to(RUNS)), "game": gid,
                         "tokens": len(seq["tokens"]), "generated": seq["generated"], "calls": seq["calls"],
                         "unmatched_turns": seq["unmatched"],
                         "first_turn_prompt_delta": (seq["check"][0][0] - seq["check"][0][1])
                         if seq["check"] and seq["check"][0][1] > 0 else None})
        np.savez(out / f"{split}.npz", tokens=np.concatenate(tokens) if tokens else np.zeros(0, np.int32),
                 loss=np.concatenate(loss) if loss else np.zeros(0, np.uint8),
                 offsets=np.asarray(offsets, dtype=np.int64))
        d = np.asarray(deltas, dtype=np.int64)
        first = np.asarray([m["first_turn_prompt_delta"] for m in meta if m["first_turn_prompt_delta"] is not None])
        manifest["splits"][split] = {
            "sequences": len(meta), "games": sorted({m["game"] for m in meta}),
            "tokens": int(offsets[-1]), "generated_tokens": int(sum(m["generated"] for m in meta)),
            "calls": int(sum(m["calls"] for m in meta)),
            "prompt_delta_first_turn": ({"median": float(np.median(first)), "mean_abs": float(np.abs(first).mean()),
                                         "n": int(len(first))} if len(first) else None),
            "prompt_delta_all_turns": ({"median": float(np.median(d)), "p10": float(np.percentile(d, 10)),
                                        "p90": float(np.percentile(d, 90)), "n": int(len(d))} if len(d) else None),
            "sequences_meta": meta,
        }
        print(json.dumps({k: v for k, v in manifest["splits"][split].items() if k != "sequences_meta"}))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
