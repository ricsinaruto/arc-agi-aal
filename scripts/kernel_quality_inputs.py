#!/usr/bin/env python3
"""Build the fixed kernel-quality inputs (data/kernel_quality) from a recorded arena run.

- ``prompts.json`` ``greedy``: 16 arena prompts, one per source game, from the Duck model-call snapshots
  (``games/<clone>/prompts/prompt.log``): the real system prompt, and the latest part of the rendered
  conversation (observations, tool results, earlier reasoning) cut to about 8K / 16K / 28K characters.
- ``prompts.json`` ``teacher_forced``: 40 passages (~1,800 characters, ~512 tokens) of the model's own
  recorded reasoning (``[THINKING]`` blocks of ``duck-transcript.txt``), two per source game.
- ``gsm8k.jsonl``: 200 GSM8K test problems at a fixed stride (index round(i * 1319 / 200)).

Deterministic for a given run folder and GSM8K file. Usage:
  python scripts/kernel_quality_inputs.py --trial <run>/arena/trials/duck-transfer --gsm8k test.jsonl
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/kernel_quality"
GREEDY_CHARS = (8000, 16000, 28000)
TF_CHARS = 1800
HEADER = re.compile(r"^\[[A-Z][A-Z _]*\]$")


def sections(text):
    """(header, body) pairs of a snapshot/transcript rendered with [HEADER] lines."""
    rows, header, body = [], None, []
    for line in text.splitlines():
        if HEADER.match(line.strip()):
            if header is not None:
                rows.append((header, "\n".join(body).strip()))
            header, body = line.strip(), []
        else:
            body.append(line)
    if header is not None:
        rows.append((header, "\n".join(body).strip()))
    return rows


def greedy_prompt(snapshot, chars):
    text = snapshot.read_text(errors="replace")
    model_input = text.split("[MODEL INPUT]", 1)[1].split("[TURN TRANSCRIPT SO FAR]", 1)[0]
    parts = sections(model_input)
    system = next(body for header, body in parts if header == "[SYSTEM]")
    history = "\n\n".join(f"{header}\n{body}" for header, body in parts if header != "[SYSTEM]")
    tail = history[-chars:]
    cut = tail.find("\n")
    tail = tail[cut + 1:] if 0 <= cut < 400 else tail
    user = ("Game log (latest part):\n" + tail + "\n\nContinue from the latest step: think about the mechanics, "
            "then write the next python tool call.")
    return system, user


def thinking_passages(transcript, count):
    blocks = [body for header, body in sections(transcript.read_text(errors="replace"))
              if header == "[THINKING]" and len(body) >= TF_CHARS]
    if not blocks:
        return []
    picks = [blocks[len(blocks) // 3], blocks[(2 * len(blocks)) // 3]][:count]
    return [block[:TF_CHARS].rsplit(" ", 1)[0] for block in picks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True, help="recorded duck-transfer trial folder (games/, public-roster.json)")
    parser.add_argument("--gsm8k", required=True, help="GSM8K test.jsonl (openai/grade-school-math)")
    args = parser.parse_args()
    trial = Path(args.trial)
    roster = json.loads((trial / "public-roster.json").read_text())
    by_source = {}
    for row in roster:
        by_source.setdefault(row["source_game_id"], []).append(row["game_id"])
    greedy, teacher = [], []
    for source in sorted(by_source):
        clones = sorted(by_source[source], key=lambda clone: -(trial / "games" / clone / "prompts/prompt.log").stat().st_size
                        if (trial / "games" / clone / "prompts/prompt.log").is_file() else 0)
        clone = clones[0]
        snapshot = trial / "games" / clone / "prompts/prompt.log"
        if len(greedy) < 16 and snapshot.is_file():
            system, user = greedy_prompt(snapshot, GREEDY_CHARS[len(greedy) % len(GREEDY_CHARS)])
            greedy.append({"index": len(greedy), "game": clone, "source_game": source, "system": system, "user": user})
        for passage in thinking_passages(trial / "games" / clone / "duck-transcript.txt", 2):
            teacher.append({"index": len(teacher), "game": clone, "source_game": source, "text": passage})
    teacher = teacher[:40]
    if len(greedy) != 16 or len(teacher) != 40:
        raise SystemExit(f"expected 16 greedy prompts and 40 passages, got {len(greedy)} and {len(teacher)}")
    lines = Path(args.gsm8k).read_text().splitlines()
    if len(lines) != 1319:
        raise SystemExit("expected the 1,319-problem GSM8K test split")
    problems = []
    for i in range(200):
        index = round(i * len(lines) / 200)
        row = json.loads(lines[index])
        problems.append({"index": index, "question": row["question"], "answer": row["answer"]})
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = {"source_trial": str(trial), "greedy": greedy, "teacher_forced": teacher}
    (OUT / "prompts.json").write_text(json.dumps(prompts, indent=1) + "\n")
    (OUT / "gsm8k.jsonl").write_text("".join(json.dumps(p) + "\n" for p in problems))
    print(json.dumps({
        "greedy": len(greedy), "greedy_chars": [len(p["system"]) + len(p["user"]) for p in greedy],
        "teacher_forced": len(teacher), "gsm8k": len(problems),
        "gsm8k_source_sha256": hashlib.sha256(Path(args.gsm8k).read_bytes()).hexdigest(),
    }))


if __name__ == "__main__":
    main()
