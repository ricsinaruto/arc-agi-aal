"""trunk/sft ARC-2 shards: arc2's replayed served logs (volume arc2-replay, /out/<run>/part-NNN) -> training examples on
arc-prolong-runs /sft-data/NAME/, in build.py's format. Never join these to an ARC-3 root.

    .venv/bin/python scripts/sft/build_arc2.py --runs runs.txt --name NAME [--sample N]

One Modal CPU job per part; a rerun skips finished parts (.done); --only held_out builds those tasks first (a held-out
root long before the whole list), a later --only train the rest of the same parts into the same root; SFT_BUILD_VOLUME + SFT_BUILD_PARENT put the root on its own
v2 volume (another run's commits stay untouched); --feed S rewrites the index every S seconds from the finished prefix of the
list, so scoring can start long before the end. Each request: arc2.served.rebuild's payload, encoded from
its text (data.encode_request, canonical: the reply's own text, not the served splice), images kept. Gate v3 (Gate) keeps
a row while its task's unexplained context deviation stays small; a row outside it is never a target. A part's kept
counts are compared with its regate.json. Examples: data.chain_examples per task, index tags run, trial (the part), k (the
task as logged), game (the 8-hex base id) and split (--split: held_out or train; a task in neither raises).
priority.json: held-out tasks, then the first era in --runs (a "#" line ends it), then the rest, each shuffled, so a
partial scoring covers each evenly."""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
WINDOW, CONTEXT = 65_536, 130_816
LOOP_WINDOW, LOOP_MAX = 64, 0.10  # build.py's degenerate-loop filter
SLACK, RATE, HI = 16, 0.01, 4  # splice band per appended reply: [-(SLACK + RATE * its completion tokens), +HI]
MIN_DEV, FRAC = 64, 0.005  # a row is kept while its unexplained context deviation <= max(MIN_DEV, FRAC * prompt)
PROCESSOR = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"  # served


class Gate:
    """Gate v3 (scratch sft24/gate_arc2.py, as regate.json ran it), one task's rows in order. diff = rendered - served
    prompt tokens; RTX serving splices each reply's own ids into the next prompt, so a turn appended since the previous
    request moves diff by the canonical split minus the model's: inside the band that is expected, past it the excess
    is context deviation, which accumulates until a fresh context."""

    def __init__(self):
        self.last = self.messages = self.completion = None
        self.dev = 0.0

    def row(self, keep, messages, rendered, served, completion):
        diff = rendered - served
        band = SLACK + RATE * (self.completion or 0)
        if keep == 0 or self.last is None:  # fresh context: no spliced replies yet
            change, lo, hi = diff, -SLACK, HI
            self.dev = 0.0
        elif keep < self.messages:  # a drain dropped earlier turns (and their splice diffs)
            change, lo, hi = diff, -(abs(self.last) + band), abs(self.last) + HI
        else:  # the turns appended since the previous request
            change, lo, hi = diff - self.last, -band, HI
        if not lo <= change <= hi:
            self.dev += abs(change - (lo if change < lo else hi))
        self.last, self.messages, self.completion = diff, messages, completion
        return self.dev <= max(MIN_DEV, FRAC * served)


def era_of(path):
    """Runs of a --runs file in eras: a line "#" ends the first."""
    eras, now = [], []
    for line in Path(path).read_text().splitlines():
        if line.strip() == "#":
            eras.append(now)
            now = []
        elif line.strip():
            now.append(line.strip())
    return eras + [now]


def order_of(files, eras, seed=0):
    """The scorer's list: held-out examples, then era by era, each shuffled."""
    out = []
    groups = [lambda e: e["split"] == "held_out"] + [lambda e, r=set(runs_): e["split"] != "held_out" and e["run"] in r for runs_ in eras]
    for group in groups:
        names = sorted(e["file"] for e in files if group(e))
        random.Random(seed).shuffle(names)
        out += names
    return out


if modal.is_local():
    sys.path.insert(0, str(HERE.parents[1] / "src"))
from arc_infra.runops.ciimage import image as ci_image

ARC2 = os.environ.get("ARC2_SRC", "./.claude/worktrees/arc2/src")  # arc2.served: branch arc2
VOLUME, PARENT = os.environ.get("SFT_BUILD_VOLUME", "arc-prolong-runs"), os.environ.get("SFT_BUILD_PARENT", "sft-data")
OWN = VOLUME != "arc-prolong-runs"
BASE = Path("/runs") / PARENT
image = ci_image().env({"SFT_BUILD_VOLUME": VOLUME, "SFT_BUILD_PARENT": PARENT}) if OWN else ci_image()
image = (image.add_local_python_source("arc_infra").add_local_file(str(HERE / "data.py"), "/root/sft_data.py")
         .add_local_file(f"{ARC2}/arc2/served.py", "/root/arc2_served.py"))
app = modal.App("sft-build-arc2")
runs = modal.Volume.from_name(VOLUME, **({"create_if_missing": True, "version": 2} if OWN else {}))
assets = modal.Volume.from_name("arc-prolong-assets")
replay = modal.Volume.from_name("arc2-replay")
_processor = []


@app.function(image=image, timeout=2 * 3600, retries=0, cpu=(0.125, 4), max_containers=int(os.environ.get("SFT_BUILD_MAX_CONTAINERS", 50)),
              volumes={"/runs": runs, "/replay": replay.read_only(), "/assets": assets.read_only()})
def part(job: dict, name: str, held: list = (), only: str = "") -> dict:
    import copy
    import importlib.util

    import torch

    def load(label, path):
        spec = importlib.util.spec_from_file_location(label, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    data, served = load("sft_data", "/root/sft_data.py"), load("arc2_served", "/root/arc2_served.py")
    tag = f"{job['run']}__{job['part']}"
    done = BASE / name / ".done" / f"{tag}.json"
    held = set(held)
    wanted = lambda task: not only or (task.split("@")[0].split("#")[0] in held) == (only == "held_out")  # noqa: E731
    if done.exists():  # a whole part: this pass's share of it
        whole = json.loads(done.read_text())
        return {**whole, "examples": [e for e in whole["examples"] if wanted(e["k"])]}
    done = done.with_name(f"{tag}__{only}.json") if only else done
    if done.exists():
        return json.loads(done.read_text())
    if not _processor:
        from transformers import AutoProcessor
        _processor.append(AutoProcessor.from_pretrained(PROCESSOR))
    processor = _processor[0]
    grids = {}

    class GridOnly:  # build.py's: first pass without pixels, kept examples re-encoded through the real processor
        def __getattr__(self, attr):
            return getattr(processor.image_processor, attr)

        def __call__(self, images, **kwargs):
            from transformers import BatchFeature

            rows = []
            for one in images:
                key = (one.size, one.mode)
                if key not in grids:
                    grids[key] = processor.image_processor([one], **kwargs)["image_grid_thw"]
                rows.append(grids[key])
            return BatchFeature({"pixel_values": torch.zeros(0), "image_grid_thw": torch.cat(rows)})

    first = copy.copy(processor)
    first.image_processor = GridOnly()
    folder = Path("/replay/out") / job["run"] / job["part"]
    stats, per, gates, chains = {}, {}, {}, {}

    def tally(reason, usage):
        s = stats.setdefault(reason, {"requests": 0, "reply": 0, "prompt": 0})
        s["requests"] += 1
        s["reply"] += usage.get("completion_tokens") or 0
        s["prompt"] += usage.get("prompt_tokens") or 0

    cut, spent = job.get("cut"), {}  # --cut: a try's requests up to its right point (its own completion tokens), nothing after
    for row, payload in served.rebuild(folder):
        task, usage = row["task"], row.get("usage") or {}
        if not wanted(task):
            continue
        if cut is not None:
            if task not in cut or spent.get(task, 0) >= cut[task]:
                continue
            spent[task] = spent.get(task, 0) + (usage.get("completion_tokens") or 0)
        per.setdefault(task, [0, 0])[0] += 1
        if usage.get("prompt_tokens") is None:
            tally("no_counts", usage)
            continue
        capture = {"request": row["seq"], "prompt_tokens": usage["prompt_tokens"], "usage": usage,
                   "reply": row["response"], "payload": {k: payload[k] for k in ("messages", "tools", "chat_template_kwargs")
                                                         if k in payload}}
        try:
            inputs, checks = data.encode_request(first, capture)
            rendered = checks["prompt_tokens"]
        except ValueError:  # a reply that does not render (bad call arguments): the prompt alone for the gate
            inputs, messages = None, data.served_messages(payload["messages"])
            text = first.tokenizer.apply_chat_template(messages[0], tools=payload.get("tools"), tokenize=False,
                                                       add_generation_prompt=True, **(payload.get("chat_template_kwargs") or {}))
            rendered = int(first(text=[text], images=messages[1] or None, return_tensors="pt")["input_ids"].shape[1])
        kept = gates.setdefault(task, Gate()).row(row["keep"], len(payload["messages"]), rendered, usage["prompt_tokens"],
                                                  usage.get("completion_tokens"))
        per[task][1] += kept
        chain = chains.setdefault(task, [])
        if not kept:
            tally("gate", usage)
        elif not usage.get("completion_tokens"):
            tally("no_counts", usage)
        elif inputs is None:
            tally("render_error", usage)
        elif checks["reply_tokens"] + 1 > WINDOW:
            tally("reply_over_window", usage)
        else:
            tally("kept", usage)
            for key in ("pixel_values", "attention_mask", "mm_token_type_ids"):
                inputs.pop(key, None)
            chain.append({"capture": capture, "encoded": (inputs, rendered)})
    regate = json.loads((folder / "regate.json").read_text())
    gate_diff = [] if cut is not None else sorted(t for t in per if per[t] != regate["per_task"].get(t))
    out_dir = BASE / name / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    examples, loops = [], 0
    for task, chain in chains.items():
        slug = re.sub(r"[^0-9A-Za-z-]+", "-", task)
        for j, (inputs, tail, targets) in enumerate(data.chain_examples(lambda c: c["encoded"], chain, window=WINDOW)):
            ids, labels = inputs["input_ids"], inputs["labels"]
            reply = int((labels != -100).sum())
            if ids.shape[1] > CONTEXT:
                tally("over_context", {"completion_tokens": reply, "prompt_tokens": int(ids.shape[1])})
                continue
            tokens = ids[0][labels[0] != -100].tolist()
            windows = [tuple(tokens[i * LOOP_WINDOW:(i + 1) * LOOP_WINDOW]) for i in range(len(tokens) // LOOP_WINDOW)]
            if len(windows) - len(set(windows)) > LOOP_MAX * max(1, len(windows)):
                loops += 1
                tally("reply_loop", {"completion_tokens": reply})
                continue
            example = {"input_ids": ids[0].to(torch.int32), "labels": labels[0].to(torch.int32), "tail": int(tail),
                       "targets": int(targets)}
            if inputs.get("image_grid_thw") is not None:
                last = next(c["capture"] for c in chain if c["encoded"][0] is inputs)
                again, _ = data.encode_request(processor, last)
                if not torch.equal(again["input_ids"], ids):
                    raise RuntimeError(f"{tag} {task}: re-encoding differs")
                example.update(pixel_values=again["pixel_values"], image_grid_thw=again["image_grid_thw"],
                               mm_token_type_ids=again["mm_token_type_ids"][0].to(torch.int32))
            f = out_dir / f"{tag}__{slug}__{j:03d}.pt"
            torch.save(example, f)
            examples.append({"file": f"train/{f.name}", "run": job["run"], "trial": job["part"], "k": task,
                             "game": task.split("@")[0].split("#")[0], "tokens": int(ids.shape[1]), "reply_tokens": reply,
                             "targets": int(targets), "tail": int(tail),
                             "images": int(inputs["image_grid_thw"].shape[0]) if inputs.get("image_grid_thw") is not None else 0})
    if cut is not None:  # a cut that the served log does not reach is reported, not built short
        short = sorted(t for t in cut if t in per and spent.get(t, 0) < cut[t])
        off = {t: spent[t] - cut[t] for t in cut if t in spent and spent[t] != cut[t] and t not in short}  # the right reply ends past it
        examples = [e for e in examples if e["k"] not in short]
    else:
        short, off = [], {}
    result = {"job": {k: v for k, v in job.items() if k != "cut"}, "short": short, "off": off, "examples": examples, "stats": stats, "per_task": per, "gate_diff": gate_diff,
              "folded": sum(len(c) for c in chains.values()) - len(examples)}
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(result))
    for attempt in range(6):  # as build.py's commit_runs: concurrent commits on the shared volume fail now and then
        try:
            runs.commit()
            break
        except Exception as error:  # noqa: BLE001 - DataLossError, "too many layers"
            if attempt == 5 or not (isinstance(error, modal.exception.DataLossError) or "too many layers" in str(error)):
                raise
            time.sleep(30 * (attempt + 1))
    return result


@app.function(image=image, timeout=1200, retries=0, cpu=(0.125, 2), volumes={"/replay": replay.read_only()})
def listing(names: list) -> list:
    return [{"run": r, "part": p.name} for r in names for p in sorted((Path("/replay/out") / r).iterdir())
            if (p / "regate.json").exists()]


@app.function(image=image, timeout=1200, retries=0, cpu=(0.125, 2), volumes={"/runs": runs})
def write_index(name: str, index: dict, order: list):
    root = BASE / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.json").write_text(json.dumps(index))
    (root / "priority.json").write_text(json.dumps(order))
    runs.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="run names, a line '#' ending the first era")
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", required=True, help="configs/arc2-split-100-20.json")
    ap.add_argument("--only", default="", choices=["", "held_out", "train"], help="this split's tasks; the index covers them")
    ap.add_argument("--sample", type=int, default=0, help="every run's first parts, N in all, spread over runs; no index")
    ap.add_argument("--cut", default="", help="JSON {run: {task as logged: tokens}}: only those tries, each up to its own "
                    "completion tokens at the right point (the request that reaches them is the last)")
    ap.add_argument("--feed", type=float, default=0, help="seconds between index rewrites from the finished prefix")
    a = ap.parse_args()
    eras = era_of(a.runs)
    split = json.loads(Path(a.split).read_text())
    held, allowed = set(split["held_out"]), set(split["train"])
    with modal.enable_output(), app.run():
        jobs = listing.remote([r for era in eras for r in era])
        if a.cut:
            cuts = json.loads(Path(a.cut).read_text())
            jobs = [{**j, "cut": cuts[j["run"]]} for j in jobs if j["run"] in cuts]
        if a.sample:
            random.Random(0).shuffle(jobs)
            jobs = jobs[: a.sample]

        def publish(results):
            """The index and scorer's list of ``results`` (the list's finished prefix), and the brief."""
            done = jobs[:len(results)]
            failed = [{"job": j, "error": repr(r)[:500]} for j, r in zip(done, results) if not isinstance(r, dict)]
            ok = [r for r in results if isinstance(r, dict)]
            files = [{**e, "split": "held_out" if e["game"] in held else "train"} for r in ok for e in r["examples"]]
            unknown = sorted({e["game"] for e in files} - held - allowed)
            if unknown:
                raise RuntimeError(f"tasks in neither split: {unknown[:8]}")
            total = {"examples": len(files), "tokens": sum(e["tokens"] for e in files),
                     "reply_tokens": sum(e["reply_tokens"] for e in files)}
            stats = {}
            for r in ok:
                for reason, s in r["stats"].items():
                    row = stats.setdefault(reason, {"requests": 0, "reply": 0, "prompt": 0})
                    for key, value in s.items():
                        row[key] += value
            if not a.sample:
                write_index.remote(a.name, {"name": a.name, "only": a.only, "manifest": a.runs, "window": WINDOW, "context": CONTEXT,
                                            "canonical": True, "train": {**total, "files": files}, "heldout": {"files": []},
                                            "failed": failed, "stats": stats, "parts": [len(ok), len(jobs)]}, order_of(files, eras))
            return {**total, "parts": len(ok), "stats": stats, "failed": failed,
                    "gate_diff": sum(len(r["gate_diff"]) for r in ok), "short": [[r["job"]["run"], t] for r in ok for t in r.get("short", [])][:50],
                    "off": sum(len(r.get("off", {})) for r in ok), "folded": sum(r["folded"] for r in ok),
                    "ratio": round(total["tokens"] / max(1, total["reply_tokens"]), 3),
                    "held_out": {"examples": sum(e["split"] == "held_out" for e in files),
                                 "tokens": sum(e["tokens"] for e in files if e["split"] == "held_out"),
                                 "reply_tokens": sum(e["reply_tokens"] for e in files if e["split"] == "held_out"),
                                 "tasks": len({e["game"] for e in files if e["split"] == "held_out"})}}

        results, fed = [], time.time()
        for r in part.starmap([(j, a.name, sorted(held), a.only) for j in jobs], return_exceptions=True):  # in list order
            results.append(r)
            if a.feed and time.time() - fed >= a.feed:
                fed = time.time()
                b = publish(results)
                print("FED " + json.dumps({k: b[k] for k in ("examples", "tokens", "reply_tokens", "parts")}), flush=True)
        brief = publish(results)
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
