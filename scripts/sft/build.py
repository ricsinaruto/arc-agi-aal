"""trunk/sft shards: pick.py's manifest -> token-exact training examples on arc-prolong-runs /sft-data/NAME/.

    .venv/bin/python scripts/sft/build.py --manifest manifest.json --name NAME [--canonical] [--cap 2] [--limit N]

One Modal CPU job per recorded session (games/<k>/rollout.jsonl of a kept run x game, or of the held-out sample):
requests as ``data.read_rollout`` (= heal_stream.SessionTail.read), encoded with ``data.encode_request`` and chained
with ``data.chain_examples`` at a 65,536-token gradient window over the full history (<= 130,816 tokens, no-grad
context). A request whose prompt does not encode to the served usage.prompt_tokens is dropped (token-exact); scripted
(restored) requests stay context only. Each example: {split}/{run}__{trial}__{k}__{j}.pt with input_ids, labels
(int32, -100 off the loss), tail, targets and, with images, pixel_values / image_grid_thw / mm_token_type_ids;
index.json: examples, tokens and reply tokens per game, and the removed share by reason and token type."""
import os
import sys

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
WINDOW, CONTEXT, HELDOUT_REPLY = 65_536, 130_816, 500_000
REBUILT = "/runs/sft-data/rebuilt"
BUCKET, ENDPOINT, REGION = "arc-runs-bucket", "https://storage.eu-south1.nebius.cloud", "eu-south1"
PROCESSORS = ("/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1",  # served
              "/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540")
REASONS = ("kept", "canonical", "non_gameplay", "no_counts", "scripted", "prompt_mismatch", "render_error", "reply_over_window",
           "over_context", "reply_loop", "other_level")
# Degenerate loops (28 Oct, main step 183: gn 9.7 vs median 0.55): an example whose labelled reply repeats more than
# LOOP_MAX of its 64-token windows exactly is dropped. s2-kl-v2 scan: 1,945 of 1,947 files <= 1.4%, the 2 loops 52% / 87%.
LOOP_WINDOW, LOOP_MAX = 64, 0.10


def loop_share(tokens, window=LOOP_WINDOW):
    """Share of the non-overlapping ``window``-token windows of ``tokens`` that exactly repeat an earlier window."""
    seen, rep, n = set(), 0, len(tokens) // window
    for i in range(n):
        key = tuple(tokens[i * window:(i + 1) * window])
        rep += key in seen
        seen.add(key)
    return rep / max(1, n)

app = modal.App("sft-build")
runs = modal.Volume.from_name("arc-prolong-runs")
assets = modal.Volume.from_name("arc-prolong-assets")
if modal.is_local():
    sys.path.insert(0, str(HERE.parents[1] / "src"))
from arc_infra.runops.ciimage import image as ci_image

image = (ci_image(apt=("rclone",)).add_local_python_source("arc_infra")
         .add_local_file(str(HERE / "data.py"), "/root/sft_data.py"))
_processor = []


def commit_runs(tries=6):
    """runs.commit() with retries: ~60 concurrent session commits raised DataLossError ("failed to publish commit") on
    5% of cur20-v1's sessions (28 Oct)."""
    import time
    for attempt in range(tries):
        try:
            return runs.commit()
        except Exception as error:  # noqa: BLE001 - DataLossError, or a refusal ("too many layers in volume", 30 Oct)
            if attempt == tries - 1 or not (isinstance(error, modal.exception.DataLossError) or "too many layers" in str(error)):
                raise
            time.sleep(30 * (attempt + 1))


def jobs_of(manifest):
    """(train jobs, held-out jobs): one per session; every held-out session is built, ``pick_heldout`` then keeps a
    balanced sample of its examples."""
    train = [{"split": "train", "run": r["bucket_run"], "trial": r["trial"], "k": k, "game": r["game"],
              "source": r.get("source", "recorded"), "priority": r.get("priority", 0),
              **({"levels": r["levels"][k]} if r.get("levels") else {})}
             for r in manifest["rows"] if r["kept"] for k, n in sorted(r["instance_reply"].items()) if n]
    held = [{"split": "heldout", "run": h["bucket_run"], "trial": h["trial"], "k": k, "game": h["game"],
             "source": h.get("source", "recorded")}
            for h in manifest["heldout"] for k, n in sorted(h["instance_reply"].items()) if n]
    return train, held


def pick_heldout(results):
    """Per held-out game, its built examples in a fixed hash order (sha1 of the file name) until HELDOUT_REPLY / games
    labelled reply tokens (whole examples; a game with less keeps all). Drops the rest from ``results``; returns
    (files to delete, per-game selection)."""
    held = [e for r in results if r["job"]["split"] == "heldout" for e in r["examples"]]
    games = sorted({e["game"] for e in held})
    quota = HELDOUT_REPLY / max(len(games), 1)
    keep, sel = set(), {}
    for g in games:
        got = 0
        for e in sorted((e for e in held if e["game"] == g), key=lambda e: hashlib.sha1(e["file"].encode()).hexdigest()):
            if got >= quota:
                break
            keep.add(e["file"])
            got += e["reply_tokens"]
        avail = sum(e["reply_tokens"] for e in held if e["game"] == g)
        sel[g] = {"available": avail, "picked": got, "all_taken": got >= avail}
    for r in results:
        if r["job"]["split"] == "heldout":
            r["examples"] = [e for e in r["examples"] if e["file"] in keep]
    return sorted(e["file"] for e in held if e["file"] not in keep), sel


def budget_train(results, budget):
    """Scoring order: ``priority`` 0 on about ``budget`` train tokens split equally across games (a game
    with less gives all it has, its remainder going evenly to the others), within a game round-robin over runs (each
    run's examples in sha1(file) order); every other train example gets priority 1. Returns (selection record, the
    selected files interleaved round-robin over games, each game's in its run rotation: priority.json)."""
    train = [e for r in results if r["job"]["split"] == "train" for e in r["examples"]]
    avail = {}
    for e in train:
        avail[e["game"]] = avail.get(e["game"], 0) + e["tokens"]
    quota, left = {}, float(budget)
    for i, g in enumerate(sorted(avail, key=avail.get)):
        quota[g] = min(avail[g], left / (len(avail) - i))
        left -= quota[g]
    record, picked = {}, {}
    for g in sorted(avail):
        runs_ = {}
        for e in sorted((e for e in train if e["game"] == g), key=lambda e: hashlib.sha1(e["file"].encode()).hexdigest()):
            e["priority"] = 1
            runs_.setdefault(e["run"], []).append(e)
        queues, got, reply = [list(v) for _, v in sorted(runs_.items())], 0, 0
        while got < quota[g] and any(queues):
            for q in queues:
                if q and got < quota[g]:
                    e = q.pop(0)
                    e["priority"] = 0
                    picked.setdefault(g, []).append(e["file"])
                    got, reply = got + e["tokens"], reply + e["reply_tokens"]
        record[g] = {"available": avail[g], "quota": round(quota[g]), "tokens": got, "reply_tokens": reply,
                     "runs": len(runs_)}
    lists = [picked[g] for g in sorted(picked)]
    order = [f for i in range(max(map(len, lists), default=0)) for f in (x[i] for x in lists if i < len(x))]
    return record, order


def cap_train(results, mult=2.0):
    """Per game, at most mult x the median game's labelled reply tokens: whole examples in sha1(file) order, an
    example that would pass the cap skipped. Drops the rest from ``results``; returns (files to delete, cap record)."""
    import statistics

    train = [e for r in results if r["job"]["split"] == "train" for e in r["examples"]]
    before = {}
    for e in train:
        before[e["game"]] = before.get(e["game"], 0) + e["reply_tokens"]
    if not before:
        return [], {}
    median = statistics.median(before.values())
    cap = mult * median
    keep, after = set(), {}
    for g in sorted(before):
        got = 0
        for e in sorted((e for e in train if e["game"] == g), key=lambda e: hashlib.sha1(e["file"].encode()).hexdigest()):
            if got + e["reply_tokens"] <= cap:
                keep.add(e["file"])
                got += e["reply_tokens"]
        after[g] = got
    for r in results:
        if r["job"]["split"] == "train":
            r["examples"] = [e for e in r["examples"] if e["file"] in keep]
    return (sorted(e["file"] for e in train if e["file"] not in keep),
            {"median": median, "cap": cap, "before": dict(sorted(before.items())), "after": after})


def level_points(rows):
    """[(cumulative completion tokens, levels completed)] of one instance log.jsonl (pick.curve's rule: ``kind ==
    "model"`` rows add their completion tokens, ``kind == "observation"`` rows raise the levels), levels monotone."""
    tokens, levels, points = 0, 0, [(0, 0)]
    for row in rows:
        if row.get("kind") == "model":
            tokens += (row.get("usage") or {}).get("completion_tokens") or 0
        elif row.get("kind") == "observation" and (row.get("levels") or 0) > levels:
            levels = row["levels"]
            points.append((tokens, levels))
    return points


def levels_of(points, caps):
    """The level each request was played on (levels completed before it + 1): its completion tokens so far (every
    gameplay request with counts, scripted ones included) against ``points``."""
    out, tokens = [], 0
    for c in caps:
        out.append(max(lv for t, lv in points if t <= tokens) + 1)
        tokens += (c["usage"] or {}).get("completion_tokens") or 0
    return out


def tally(stats, reason, capture=None, row=None):
    usage = ((capture or {}).get("usage") or (row or {}).get("usage")) or {}
    reply = usage.get("completion_tokens") or 0
    think = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    image_tokens = ((usage.get("prompt_tokens_details") or {}).get("multimodal_tokens") or {}).get("image") or 0
    s = stats.setdefault(reason, {"requests": 0, "reply": 0, "reply_think": 0, "reply_answer": 0, "prompt": 0,
                                  "prompt_image": 0})
    s["requests"] += 1
    s["reply"] += reply
    s["reply_think"] += think
    s["reply_answer"] += reply - think
    s["prompt"] += usage.get("prompt_tokens") or 0
    s["prompt_image"] += image_tokens


class GridOnly:
    """Image processor for a session's first pass, which uses image_grid_thw alone: one real call per image size (a grid
    depends on the size only), no pixels. A long session otherwise re-processes its whole image history on every
    request, for hours. Kept examples are re-encoded through the real processor and raise on any id difference."""

    def __init__(self, real):
        self.real, self.grids = real, {}

    def __getattr__(self, name):
        return getattr(self.real, name)

    def __call__(self, images, **kwargs):
        import torch
        from transformers import BatchFeature

        rows = []
        for one in images:
            key = (one.size, one.mode)
            if key not in self.grids:
                self.grids[key] = self.real([one], **kwargs)["image_grid_thw"]
            rows.append(self.grids[key])
        return BatchFeature({"pixel_values": torch.zeros(0), "image_grid_thw": torch.cat(rows)})


@app.function(image=image, timeout=int(os.environ.get("SFT_BUILD_TIMEOUT", 7200)), retries=0,
              nonpreemptible=bool(os.environ.get("SFT_BUILD_NONPREEMPTIBLE")),  # a session has no resume
              volumes={"/runs": runs, "/assets": assets.read_only()},
              secrets=[modal.Secret.from_name("nebius-s3")], max_containers=int(os.environ.get("SFT_BUILD_MAX_CONTAINERS", 8)))
def session(job: dict, name: str, canonical: bool = False, grid_only: bool = False) -> dict:
    import importlib.util
    import shutil

    import torch

    spec = importlib.util.spec_from_file_location("sft_data", "/root/sft_data.py")
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    if not _processor:
        from transformers import AutoProcessor
        for d in PROCESSORS:
            if Path(d, "tokenizer_config.json").exists():
                _processor.append(AutoProcessor.from_pretrained(d))
                break
    processor = _processor[0]
    first_pass = processor
    if grid_only:
        import copy
        first_pass = copy.copy(processor)
        first_pass.image_processor = GridOnly(processor.image_processor)
    os.environ.update(RCLONE_CONFIG_ARC_TYPE="s3", RCLONE_CONFIG_ARC_PROVIDER="Other", RCLONE_CONFIG_ARC_ENV_AUTH="true",
                      RCLONE_CONFIG_ARC_ENDPOINT=ENDPOINT, RCLONE_CONFIG_ARC_REGION=REGION,
                      RCLONE_CONFIG_ARC_NO_CHECK_BUCKET="true")
    tag = f"{job['run']}__{job['trial']}__{job['k']}"
    done = Path("/runs/sft-data") / name / ".done" / f"{tag}.json"  # a rerun after a dead client skips finished sessions
    if done.exists():
        return json.loads(done.read_text())
    local = Path("/tmp") / f"{tag}.jsonl"
    if job.get("source") == "rebuilt":  # scripts/sft/rebuild.py: the replayed requests of an unrecorded play
        shutil.copyfile(f"{REBUILT}/{job['run']}/{job['trial']}/{job['k']}/rollout.jsonl", local)
    else:
        subprocess.run(["rclone", "copyto", f"arc:{BUCKET}/runs/validation/{job['run']}/arena/working/trials/"
                        f"{job['trial']}/games/{job['k']}/rollout.jsonl", str(local), "--retries", "5"], check=True)
    stats = {}
    for line in open(local, "rb"):  # rows read_rollout leaves out
        row = json.loads(line) if line.strip() else None
        if row and not row.get("gameplay", True):
            tally(stats, "non_gameplay", row=row)
        elif row and not ((row["usage"] or {}).get("prompt_tokens") and (row["usage"] or {}).get("completion_tokens")):
            tally(stats, "no_counts", row=row)
    caps = data.read_rollout(local)
    level = [None] * len(caps)
    log = Path("/tmp") / f"{tag}.log.jsonl"
    got = subprocess.run(["rclone", "copyto", f"arc:{BUCKET}/runs/validation/{job['run']}/arena/working/trials/"
                          f"{job['trial']}/games/{job['k']}/log.jsonl", str(log), "--retries", "5"], check=False)
    if got.returncode == 0 and log.exists():
        level = levels_of(level_points(json.loads(x) for x in open(log, "rb") if x.strip()), caps)
        log.unlink()
    encoded, mismatched_reply, deltas, canon = {}, 0, [], set()
    for i, c in enumerate(caps):
        if job.get("levels") and level[i] not in job["levels"]:
            tally(stats, "other_level", c)  # a level-segment build: other levels are context only (never encoded)
            continue
        if c["scripted"]:
            tally(stats, "scripted", c)
            continue
        try:
            inputs, checks = data.encode_request(first_pass, c)
        except ValueError:
            tally(stats, "render_error", c)
            continue
        if checks["prompt_tokens"] != c["prompt_tokens"]:
            deltas.append([c["request"], checks["prompt_tokens"], c["prompt_tokens"]])
            if not canonical:
                tally(stats, "prompt_mismatch", c)
                continue
            canon.add(i)  # --canonical: kept as the text encodes (history off the served ids by the delta)
        if checks["reply_tokens"] + 1 > WINDOW:  # chain_examples leaves it out (never a target)
            tally(stats, "reply_over_window", c)
        mismatched_reply += checks["reply_tokens"] != checks["served_reply_tokens"]
        inputs.pop("pixel_values", None)  # re-made for the examples only (host RAM)
        encoded[i] = ({**inputs, "capture": i}, checks["prompt_tokens"])
    for i, c in enumerate(caps):
        c["index"] = i
    if job.get("levels"):  # level segments (curriculum builds): loss only on each selected level's replies, one chain each
        made = [(*m, lv) for lv in sorted(job["levels"]) for m in data.chain_examples(
            lambda c, lv=lv: encoded.get(c["index"]) if level[c["index"]] == lv else None, caps, window=WINDOW)]
        # the other levels' replies stay in the prompt exactly as served; only this level's are labelled
    else:
        made = [(*m, None) for m in data.chain_examples(lambda c: encoded.get(c["index"]), caps, window=WINDOW)]
    out_dir = Path("/runs/sft-data") / name / job["split"]
    out_dir.mkdir(parents=True, exist_ok=True)
    examples, targets_kept = [], set()
    chain = [i for i in range(len(caps)) if encoded.get(i)]  # chain_examples' rows: an example's targets are consecutive
    capped = [(encoded[i][1], encoded[i][0]["input_ids"].shape[1], encoded[i][0]["input_ids"][0])
              for i, c in enumerate(caps) if c["capped"] and i in encoded]
    capped_reply = 0
    for j, (inputs, tail, targets, lv) in enumerate(made):
        ids, labels, i = inputs["input_ids"], inputs["labels"], inputs["capture"]
        if ids.shape[1] > CONTEXT:
            tally(stats, "over_context", caps[i])
            continue
        capped_reply += data.mask_capped(ids[0], labels[0], capped)
        if capped and not bool((labels != -100).any()):
            tally(stats, "capped_empty", caps[i])
            continue
        if loop_share(ids[0][labels[0] != -100].tolist()) > LOOP_MAX:
            tally(stats, "reply_loop", caps[i])
            continue
        example = {"input_ids": ids[0].to(torch.int32), "labels": labels[0].to(torch.int32), "tail": int(tail),
                   "targets": int(targets)}
        if inputs.get("image_grid_thw") is not None:
            again, _ = data.encode_request(processor, caps[i])
            if not torch.equal(again["input_ids"], ids):
                raise RuntimeError(f"{tag} request {caps[i]['request']}: re-encoding differs")
            example.update(pixel_values=again["pixel_values"], image_grid_thw=again["image_grid_thw"],
                           mm_token_type_ids=again["mm_token_type_ids"][0].to(torch.int32))
        f = out_dir / (f"{tag}__L{lv:02d}__{j:03d}.pt" if lv is not None else f"{tag}__{j:03d}.pt")
        torch.save(example, f)
        targets_kept.add(i)
        examples.append({"file": f"{job['split']}/{f.name}", "run": job["run"], "trial": job["trial"], "k": job["k"],
                         "game": job["game"], "tokens": int(ids.shape[1]), "reply_tokens": int((labels != -100).sum()),
                         "targets": int(targets), "tail": int(tail), "images": int(inputs.get("image_grid_thw").shape[0])
                         if inputs.get("image_grid_thw") is not None else 0, "priority": job.get("priority", 0),
                         "level": level[i], "level_first": level[chain[chain.index(i) - int(targets) + 1]],
                         **({"seg_level": lv} if lv is not None else {})})
    labelled = {i for i in encoded if i not in targets_kept}  # targets folded into a later example
    for i in encoded:
        if encoded[i] and caps[i]["usage"]["completion_tokens"] + 1 <= WINDOW:
            tally(stats, "canonical" if i in canon else "kept", caps[i])
    result = {"job": job, "examples": examples, "stats": stats, "reply_mismatches": mismatched_reply, "prompt_deltas": deltas,
              "requests": len(caps), "folded": len(labelled), "capped_reply": capped_reply,
              **({"levels_found": sorted({e["seg_level"] for e in examples})} if job.get("levels") else {})}
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(result))
    commit_runs()
    local.unlink()
    return result


def select_built(results, curriculum, top=0.2, min_built=0.99):
    """Curriculum selection on what the build reproduced: a candidate segment counts when its examples hold
    >= ``min_built`` of its level reply tokens (``tokens``); per (game, level), the top ceil(``top`` x n) (min 1) of
    those n by ``rhae_per_tok`` are kept. Drops the rest from ``results``; returns (files to delete, kept curriculum in
    its order, record)."""
    import collections
    import math

    built = collections.Counter()
    for r in results:
        for e in r["examples"]:
            if e.get("seg_level") is not None:
                built[(e["run"], e["k"], e["seg_level"])] += e["reply_tokens"]
    whole = [c for c in curriculum if built[(c["run"], c["k"], c["level"])] >= min_built * c["tokens"]]
    cells = collections.defaultdict(list)
    for c in whole:
        cells[(c["game"], c["level"])].append(c)
    keep = set()
    for cs in cells.values():
        cs = sorted(cs, key=lambda c: -c["rhae_per_tok"])
        keep |= {(c["run"], c["k"], c["level"]) for c in cs[:max(1, math.ceil(top * len(cs)))]}
    drop = []
    for r in results:
        kept = [e for e in r["examples"] if (e["run"], e["k"], e.get("seg_level")) in keep]
        drop += [e["file"] for e in r["examples"] if e not in kept]
        r["examples"] = kept
    chosen = [c for c in curriculum if (c["run"], c["k"], c["level"]) in keep]
    cand_cells = {(c["game"], c["level"]) for c in curriculum}
    return drop, chosen, {"candidates": len(curriculum), "whole": len(whole), "selected": len(chosen),
                          "cells_candidates": len(cand_cells), "cells_selected": len(cells),
                          "candidate_tokens": sum(c["tokens"] for c in curriculum),
                          "whole_tokens": sum(c["tokens"] for c in whole),
                          "selected_tokens": sum(c["tokens"] for c in chosen),
                          "cells_without_whole": sorted(map(list, cand_cells - set(cells)))}


def curriculum_order(results, curriculum, drop=()):
    """Train example files in ``curriculum`` order ([{"run", "k", "level"}, ...], one entry per level segment); within
    a segment, chain order. Segments without examples are skipped."""
    rank = {(c["run"], c["k"], c["level"]): i for i, c in enumerate(curriculum)}
    ex = [e for r in results if r["job"]["split"] == "train" for e in r["examples"]
          if e.get("seg_level") is not None and e["file"] not in drop and (e["run"], e["k"], e["seg_level"]) in rank]
    return [e["file"] for e in sorted(ex, key=lambda e: (rank[(e["run"], e["k"], e["seg_level"])], e["file"]))]


@app.function(image=image, timeout=600, retries=0, volumes={"/runs": runs})
def write_index(name: str, index: dict, manifest: dict, drop: list, order: list | None = None):
    for f in drop:  # held-out examples outside the balanced sample, train examples over the cap
        Path("/runs/sft-data", name, f).unlink(missing_ok=True)
    Path("/runs/sft-data", name).mkdir(parents=True, exist_ok=True)
    if order is not None:  # the scorer's train selection, in order
        Path("/runs/sft-data", name, "priority.json").write_text(json.dumps(order))
    root = Path("/runs/sft-data", name)
    runs.reload()
    lost = [e["file"] for s in ("train", "heldout") for e in index[s]["files"] if not (root / e["file"]).is_file()]
    if lost:  # cur20-v2 (28 Oct): 6 example files missing after their sessions committed
        for s in ("train", "heldout"):
            gone = [e for e in index[s]["files"] if e["file"] in lost]
            index[s]["files"] = [e for e in index[s]["files"] if e["file"] not in lost]
            for key in ("tokens", "reply_tokens"):
                index[s][key] -= sum(e[key] for e in gone)
            index[s]["examples"] -= len(gone)
        index["lost"] = lost
        if order is not None:
            (root / "priority.json").write_text(json.dumps([f for f in order if f not in set(lost)]))
    Path("/runs/sft-data", name, "index.json").write_text(json.dumps(index, indent=1))
    Path("/runs/sft-data", name, "manifest.json").write_text(json.dumps(manifest, indent=1))
    commit_runs()


def summarize(results, split):
    out = {"examples": 0, "tokens": 0, "reply_tokens": 0, "sessions": 0, "per_game": {}, "removed": {},
           "reply_mismatches": 0, "prompt_deltas": [], "files": []}
    for r in results:
        if r["job"]["split"] != split:
            continue
        out["sessions"] += 1
        out["reply_mismatches"] += r["reply_mismatches"]
        out["prompt_deltas"] += [[f"{r['job']['run']}/{r['job']['k']}", *d] for d in r["prompt_deltas"]]
        for e in r["examples"]:
            g = out["per_game"].setdefault(e["game"], {"examples": 0, "tokens": 0, "reply_tokens": 0})
            for key in ("tokens", "reply_tokens"):
                g[key] += e[key]
                out[key] += e[key]
            g["examples"] += 1
            out["examples"] += 1
            p = out.setdefault("per_priority", {}).setdefault(str(e.get("priority", 0)),
                                                              {"examples": 0, "tokens": 0, "reply_tokens": 0})
            p["examples"] += 1
            p["tokens"] += e["tokens"]
            p["reply_tokens"] += e["reply_tokens"]
            out["files"].append(e)
        for reason, s in r["stats"].items():
            agg = out["removed"].setdefault(reason, dict.fromkeys(s, 0))
            for key, v in s.items():
                agg[key] += v
    total = {key: sum(s[key] for s in out["removed"].values()) for key in next(iter(out["removed"].values()), {})}
    share = {reason: {key: round(s[key] / total[key], 5) if total[key] else 0.0 for key in s}
             for reason, s in out["removed"].items()}
    out["removed_share"] = {r: v for r, v in share.items() if r not in ("kept", "canonical")}
    out["canonical_share"] = share.get("canonical", {})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N sessions of each split (smoke)")
    ap.add_argument("--canonical", action="store_true", help="keep requests whose prompt count differs from the served "
                    "one, encoded from their text (reason 'canonical'; the play server held sampled reply ids)")
    ap.add_argument("--cap", type=float, default=0, help="train: per game at most CAP x the median game's reply tokens")
    ap.add_argument("--heldout-only", action="store_true", help="build the held-out sessions only")
    ap.add_argument("--heldout-all", action="store_true", help="keep every held-out example (no balanced sample)")
    ap.add_argument("--budget", type=float, default=0, help="train: priority 0 on about BUDGET tokens, equal per game "
                    "(budget_train); the selection in order -> priority.json")
    a = ap.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    train, held = jobs_of(manifest)
    if a.heldout_only:
        train = []
    if a.limit:
        train, held = train[: a.limit], held[: a.limit]
    with modal.enable_output(), app.run():
        subprocess.run([sys.executable, str(HERE.parents[1] / "tools/ops/records.py"), "cpu-app", app.app_id], check=False)
        results = list(session.starmap([(j, a.name, a.canonical) for j in train + held], return_exceptions=True))
        failed = [{"job": j, "error": repr(r)[:500]} for j, r in zip(train + held, results) if not isinstance(r, dict)]
        ok = [r for r in results if isinstance(r, dict)]
        drop, selection = ([], {}) if a.heldout_all else pick_heldout(ok)
        capped, cap = cap_train(ok, a.cap) if a.cap else ([], {})
        drop += capped
        budget, order = budget_train(ok, a.budget) if a.budget else ({}, None)
        cur_select = None
        if manifest.get("select"):  # top share per (game, level) among segments built whole
            sel_drop, chosen, cur_select = select_built(ok, manifest["curriculum"], **manifest["select"])
            drop += sel_drop
            manifest = {**manifest, "curriculum": chosen}
        if manifest.get("curriculum"):  # level segments in the manifest's order (one pass) -> priority.json
            order = curriculum_order(ok, manifest["curriculum"], set(drop))
        index = {"name": a.name, "manifest": manifest["reference"], "window": WINDOW, "context": CONTEXT,
                 "canonical": a.canonical, "train": {**summarize(ok, "train"), "cap": cap, "budget": budget},
                 "heldout": {**summarize(ok, "heldout"), "selection": selection}, "failed": failed,
                 **({"select": cur_select} if cur_select else {})}
        write_index.remote(a.name, index, manifest, drop, order)
    brief = {s: {k: v for k, v in index[s].items() if k != "files"} for s in ("train", "heldout")}
    print(json.dumps({**brief, "failed": failed}, indent=1))


@app.local_entrypoint()
def one(manifest: str, name: str, run: str, k: str):
    """One long session for `modal run --detach build.py::one`: spawned, so it outlives its client (a .remote() call is
    cancelled when the client dies). A later build.py rerun skips it through its .done marker and writes the index."""
    train, held = jobs_of(json.loads(Path(manifest).read_text()))
    job = next(j for j in train + held if (j["run"], j["k"]) == (run, k))
    print(session.spawn(job, name, True, True).object_id)


if __name__ == "__main__":
    main()
