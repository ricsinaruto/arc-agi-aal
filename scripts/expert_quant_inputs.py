#!/usr/bin/env python3
"""Inputs of the ARC expert re-quantization (28 September; docs/validation-20260927/expert-quant.md).

Teacher-forced sequences rebuilt from our own Flash xhigh one-hour arena runs (the primary corpus of
``scripts/specialization_vocab.py``: Qwen3.8-Flash-Next-NVFP4 played every game; no Claude-played trajectory).
Each game's ``prompts/prompt.log`` holds its latest model request; ``scripts/expert_usage_inputs.py`` parses it
back into chat messages with the exact recorded texts, renders it with the served chat template (tools, xhigh
effort) and tokenizes it with the served tokenizer, so a sequence is what the server prefilled for that request.

Hard rules:
- the 9 holdout games of the evolve-v4 14/9 split (``posttrain.split.HOLDOUT``) are excluded everywhere;
- calibration and evaluation are split by game (source family): EVAL_FAMILIES are evaluation only.

Outputs (``--out``):
- ``stage1.json``: a small capture set for the stage-1 microbenchmark: ``calib`` rows (the activation scale is
  calibrated on them) and ``eval`` rows (errors are measured on them);
- ``calib.json``: the stage-2 calibration sequences (GPTQ Hessians, activation scales);
- ``eval.json``: the stage-2 held-out sequences (per-layer errors);
- ``quality/prompts.json``: the stage-2 end-to-end inputs in the kernel-quality format (``arena.kernel_gate``):
  ``teacher_forced`` items carry token ids (prompt_logprobs at every position) and ``greedy`` items the ids of a
  real request up to its generation prompt;
- ``selection.json``: what was chosen, with sha256 of every file.

Usage: <tokvenv>/bin/python scripts/expert_quant_inputs.py --out ./runs/expert-quant/inputs-v1
Needs ``tokenizers`` and ``jinja2``; nothing touches a GPU.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from expert_usage_inputs import (  # noqa: E402
    PARAMETER_DESCRIPTION,
    TOKENIZER_DIR,
    assistant_spans,
    exact_sources,
    exactify,
    family,
    parse_prompt_log,
    renderer,
)
from specialization_vocab import RUNS, primary_runs  # noqa: E402

from posttrain.split import DEV, HOLDOUT  # noqa: E402

EVAL_FAMILIES = ("cn04", "sk48", "wa30")
# Phase 5 split (throughput arm, 5 October): tune on these 10, test on the 15 held-out.
TUNING_FAMILIES = ("cd82", "cn04", "ft09", "lp85", "ls20", "re86", "sb26", "sk48", "vc33", "wa30")
HELDOUT_FAMILIES = ("ar25", "bp35", "dc22", "g50t", "ka59", "lf52", "m0r0", "r11l", "s5i5", "sc25", "sp80", "su15",
                    "tn36", "tr87", "tu93")
SEED = 20260928


def eligible(fam: str) -> bool:
    """Only dev games: never a holdout game, and nothing outside the 14 dev families."""
    return fam in DEV and fam not in HOLDOUT


ALL25 = False  # v7 refit (5 October): calibrate on all 25 families; eval = half of the ls20/re86 games (by hash)


def split_of(fam: str, key: str = "") -> str:
    if ALL25 and fam in EVAL_FAMILIES:
        return "eval" if hashlib.sha256(key.encode()).digest()[0] < 128 else "calib"
    return "eval" if fam in EVAL_FAMILIES else "calib"


def render_sequence(template, tok, gdir, log, cap):
    parsed = parse_prompt_log(log)
    if parsed is None:
        return None
    description, messages = parsed
    match = {"system": [0, 0], "assistant": [0, 0], "tool": [0, 0]}
    messages = exactify(messages, exact_sources(gdir), match)
    tools = [{"type": "function", "function": {
        "name": "python", "description": description,
        "parameters": {"type": "object", "properties": {"code": {"type": "string",
                                                                  "description": PARAMETER_DESCRIPTION}},
                       "required": ["code"]}}}]
    text = template.render(messages=messages, tools=tools, add_generation_prompt=True,
                           enable_thinking=True, reasoning_effort="xhigh")
    ids = tok.encode(text, add_special_tokens=False).ids
    return ids[:cap], len(ids), match


def select(games, targets, per_family_share, template, tok, cap, min_tokens=4000):
    """Greedy fill of each split to its token target, at most ``per_family_share`` of a split per family."""
    sets = {k: [] for k in targets}
    totals = {k: 0 for k in targets}
    per_family = {k: {} for k in targets}
    for run, gdir, log, fam in games:
        split = split_of(fam, f"{run}/{gdir}")
        if split not in targets or totals[split] >= targets[split]:
            if all(totals[k] >= targets[k] for k in targets):
                break
            continue
        if per_family[split].get(fam, 0) > targets[split] * per_family_share[split]:
            continue
        rendered = render_sequence(template, tok, gdir, log, cap)
        if rendered is None:
            continue
        ids, full, _ = rendered
        spans = assistant_spans(ids)
        if len(ids) < min_tokens or not spans:
            continue
        sets[split].append({"index": len(sets[split]), "run": run, "game": os.path.basename(gdir), "family": fam,
                            "tokens": len(ids), "full_tokens": full, "assistant_spans": spans,
                            "assistant_tokens": sum(b - a for a, b in spans), "ids": ids})
        totals[split] += len(ids)
        per_family[split][fam] = per_family[split].get(fam, 0) + len(ids)
    return sets, totals, per_family


def write_json(path, value):
    with open(path, "w") as f:
        json.dump(value, f, separators=(",", ":"))
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def summary(rows):
    return {"sequences": len(rows), "tokens": sum(r["tokens"] for r in rows),
            "assistant_tokens": sum(r["assistant_tokens"] for r in rows),
            "families": sorted({r["family"] for r in rows}),
            "games": sorted({(r["run"], r["game"]) for r in rows}).__len__(),
            "rows": [{k: r[k] for k in ("index", "run", "game", "family", "tokens", "full_tokens", "assistant_tokens")}
                     for r in rows]}


def main():
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--calib-tokens", type=int, default=700_000)
    parser.add_argument("--eval-tokens", type=int, default=160_000)
    parser.add_argument("--stage1-calib-tokens", type=int, default=40_000)
    parser.add_argument("--stage1-eval-tokens", type=int, default=60_000)
    parser.add_argument("--cap", type=int, default=65_000)
    parser.add_argument("--quality-cap", type=int, default=24_000)
    parser.add_argument("--quality-sequences", type=int, default=12)
    parser.add_argument("--tuning", action="store_true",
                        help="phase 5 (5 October): calibrate on the 10 tuning families only (eval: ls20, re86)")
    parser.add_argument("--reuse-eval", default=None,
                        help="an earlier inputs folder whose stage-1 and held-out sets are kept (new calibration only)")
    parser.add_argument("--all25", action="store_true",
                        help="v7 submission refit (5 October): calibrate on all 25 public families")
    parser.add_argument("--from-usage", default=None,
                        help="the balanced expert-usage inputs folder (equal tokens per family): calib = "
                             "its usage split, eval = its per-family screen split")
    args = parser.parse_args()
    if args.from_usage:
        return from_usage(args)
    global DEV, HOLDOUT, EVAL_FAMILIES, ALL25
    if args.all25:
        ALL25 = True
        DEV = TUNING_FAMILIES + HELDOUT_FAMILIES
        HOLDOUT = ()
        EVAL_FAMILIES = ("ls20", "re86")
    if args.tuning:
        DEV = TUNING_FAMILIES
        HOLDOUT = HELDOUT_FAMILIES
        EVAL_FAMILIES = ("ls20", "re86")
    tok = Tokenizer.from_file(os.path.join(TOKENIZER_DIR, "tokenizer.json"))
    template = renderer()
    games, skipped = [], {"holdout": 0, "not_dev": 0}
    for run in primary_runs():
        for log in sorted(glob.glob(os.path.join(RUNS, run, "**/games/*/prompts/prompt.log"), recursive=True)):
            gdir = os.path.dirname(os.path.dirname(log))
            fam = family(os.path.join(RUNS, run), gdir)
            if fam in HOLDOUT:
                skipped["holdout"] += 1
                continue
            if not eligible(fam):
                skipped["not_dev"] += 1
                continue
            games.append((run, gdir, log, fam))
    rng = random.Random(SEED)
    rng.shuffle(games)
    if ALL25:
        # Round-robin over families (shuffled order within each), so a family with few recorded games (vc33) is
        # not left behind once the token targets fill.
        by_family = {}
        for g in games:
            by_family.setdefault(g[3], []).append(g)
        queues = list(by_family.values())
        games = [q[i] for i in range(max(map(len, queues))) for q in queues if i < len(q)]
    # Stage 1: a small, diverse capture set (short sequences from several games).
    stage1, s1_totals, _ = select(games, {"calib": args.stage1_calib_tokens, "eval": args.stage1_eval_tokens},
                                  {"calib": 0.34, "eval": 0.34}, template, tok, cap=20_000)
    used = {(r["run"], r["game"]) for rows in stage1.values() for r in rows}
    reused = None
    if args.reuse_eval:
        # Phase 3: keep an earlier build's stage-1 and held-out sets (comparable held-out errors) and draw only a new,
        # larger calibration set from the other games (eval families stay excluded by family).
        prev = args.reuse_eval
        rows1 = json.load(open(os.path.join(prev, "stage1.json")))
        stage1 = {split: [r for r in rows1 if r["split"] == split] for split in ("calib", "eval")}
        reused = json.load(open(os.path.join(prev, "eval.json")))
        used = {(r["run"], r["game"]) for rows in (rows1, reused) for r in rows}
    # Stage 2: calibration and held-out sets from the remaining games (stage-1 games are not reused).
    rest = [g for g in games if (g[0], os.path.basename(g[1])) not in used]
    targets = {"calib": args.calib_tokens} if reused is not None else {"calib": args.calib_tokens,
                                                                         "eval": args.eval_tokens}
    stage2, s2_totals, s2_families = select(rest, targets, {"calib": 1 / 8, "eval": 0.4}, template, tok, cap=args.cap)
    if reused is not None:
        stage2["eval"] = reused
        s2_families["eval"] = {}
        for r in reused:
            s2_families["eval"][r["family"]] = s2_families["eval"].get(r["family"], 0) + r["tokens"]
    os.makedirs(os.path.join(args.out, "quality"), exist_ok=True)
    digests = {}
    stage1_rows = [{**r, "split": split} for split in ("calib", "eval") for r in stage1[split]]
    for i, r in enumerate(stage1_rows):
        r["index"] = i
    digests["stage1.json"] = write_json(os.path.join(args.out, "stage1.json"), stage1_rows)
    digests["calib.json"] = write_json(os.path.join(args.out, "calib.json"), stage2["calib"])
    digests["eval.json"] = write_json(os.path.join(args.out, "eval.json"), stage2["eval"])
    # End-to-end inputs: held-out sequences (a suffix of at most quality_cap tokens is not a valid context,
    # so a prefix is used) and their generation prompts for greedy continuations.
    quality_rows = sorted(stage2["eval"], key=lambda r: (r["family"], r["index"]))[:args.quality_sequences]
    teacher = [{"index": i, "text": None, "ids": r["ids"][:args.quality_cap], "source": r["index"],
                "family": r["family"], "assistant_spans": [s for s in assistant_spans(r["ids"][:args.quality_cap])]}
               for i, r in enumerate(quality_rows)]
    greedy = [{"index": i, "prompt_ids": r["ids"], "source": r["index"], "family": r["family"]}
              for i, r in enumerate(quality_rows) if r["tokens"] == r["full_tokens"]]
    digests["quality/prompts.json"] = write_json(os.path.join(args.out, "quality/prompts.json"),
                                                 {"teacher_forced": teacher, "greedy": greedy})
    selection = {
        "seed": SEED, "eval_families": list(EVAL_FAMILIES), "holdout_excluded": list(HOLDOUT), "dev": list(DEV),
        "skipped_games": skipped, "eligible_games": len(games), "tokenizer": TOKENIZER_DIR,
        "chat_template_sha256": hashlib.sha256(open(os.path.join(TOKENIZER_DIR, "chat_template.jinja"), "rb").read())
        .hexdigest(),
        "stage1": {"calib": summary(stage1["calib"]), "eval": summary(stage1["eval"])},
        "calib": {**summary(stage2["calib"]), "tokens_per_family": s2_families["calib"]},
        "eval": {**summary(stage2["eval"]), "tokens_per_family": s2_families["eval"]},
        "quality": {"teacher_forced": len(teacher), "teacher_tokens": sum(len(t["ids"]) for t in teacher),
                    "greedy": len(greedy)},
        "sha256": digests,
    }
    with open(os.path.join(args.out, "selection.json"), "w") as f:
        json.dump(selection, f, indent=1)
    for key in ("stage1", "calib", "eval", "quality"):
        value = selection[key]
        if key == "stage1":
            print(key, {k: (v["sequences"], v["tokens"], v["families"]) for k, v in value.items()})
        elif key == "quality":
            print(key, value)
        else:
            print(key, value["sequences"], value["tokens"], value["assistant_tokens"], value["tokens_per_family"])
    print("skipped", skipped, "eligible", len(games))


def from_usage(args):
    """Calibration inputs from the balanced usage inputs, so the mask ranking and the GPTQ Hessians see the same
    equal-per-family traffic; per-family token counts are recorded in selection.json."""
    src = args.from_usage
    usage = json.load(open(os.path.join(src, "usage.json")))
    screen = json.load(open(os.path.join(src, "heldout.json")))
    picked = json.load(open(os.path.join(src, "selection.json")))
    os.makedirs(os.path.join(args.out, "quality"), exist_ok=True)
    keep = ("index", "run", "game", "family", "tokens", "full_tokens", "assistant_spans", "assistant_tokens", "ids")
    calib = [{k: r[k] for k in keep} for r in usage]
    evals = [{k: r[k] for k in keep} for r in screen]
    digests = {"calib.json": write_json(os.path.join(args.out, "calib.json"), calib),
               "eval.json": write_json(os.path.join(args.out, "eval.json"), evals)}
    # stage 1 (microbenchmark only): one short calib and one eval sequence from each of the first six families
    stage1 = []
    for split, rows in (("calib", calib), ("eval", evals)):
        seen = set()
        for r in rows:
            if r["family"] not in seen and len(seen) < 6:
                seen.add(r["family"])
                stage1.append({**r, "ids": r["ids"][:8000], "tokens": min(r["tokens"], 8000), "split": split,
                               "index": len(stage1)})
    digests["stage1.json"] = write_json(os.path.join(args.out, "stage1.json"), stage1)
    rng = random.Random(SEED)
    pool = [r for r in evals if r["tokens"] >= 8000]
    quality_rows = sorted(rng.sample(pool, min(args.quality_sequences, len(pool))), key=lambda r: r["index"])
    teacher = [{"index": i, "text": None, "ids": r["ids"][:args.quality_cap], "source": r["index"],
                "family": r["family"], "assistant_spans": assistant_spans(r["ids"][:args.quality_cap])}
               for i, r in enumerate(quality_rows)]
    greedy = [{"index": i, "prompt_ids": r["ids"], "source": r["index"], "family": r["family"]}
              for i, r in enumerate(quality_rows) if r["tokens"] == r["full_tokens"]]
    digests["quality/prompts.json"] = write_json(os.path.join(args.out, "quality/prompts.json"),
                                                 {"teacher_forced": teacher, "greedy": greedy})

    def per_family(rows):
        out = {}
        for r in rows:
            row = out.setdefault(r["family"], {"tokens": 0, "assistant_tokens": 0, "sequences": 0, "runs": set()})
            row["tokens"] += r["tokens"]
            row["assistant_tokens"] += r["assistant_tokens"]
            row["sequences"] += 1
            row["runs"].add(r["run"])
        return {f: {**v, "runs": len(v["runs"])} for f, v in sorted(out.items())}

    selection = {
        "seed": SEED, "source": src, "balanced_per_family": picked.get("balanced_per_family"),
        "eval_families": "per-family screen split (every family, other games than calibration)",
        "holdout_excluded": [], "dev": list(picked["families"]), "tokenizer": TOKENIZER_DIR,
        "chat_template_sha256": picked["chat_template_sha256"],
        "calib": {**summary(calib), "per_family": per_family(calib)},
        "eval": {**summary(evals), "per_family": per_family(evals)},
        "quality": {"teacher_forced": len(teacher), "teacher_tokens": sum(len(t["ids"]) for t in teacher),
                    "greedy": len(greedy), "families": [r["family"] for r in quality_rows]},
        "sha256": digests,
    }
    with open(os.path.join(args.out, "selection.json"), "w") as f:
        json.dump(selection, f, indent=1)
    print("calib", selection["calib"]["sequences"], selection["calib"]["tokens"], "eval",
          selection["eval"]["sequences"], selection["eval"]["tokens"], "quality", selection["quality"])


if __name__ == "__main__":
    main()
