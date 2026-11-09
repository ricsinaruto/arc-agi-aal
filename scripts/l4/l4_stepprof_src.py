"""Step profile of the serving stack on four L4s, between the probe and the play (``stepprof`` in the options).

Torch-profiler traces (40 steps each) on the ready server: pure decode at 16K and 56K of context and 32 rows, at
100K and 16 rows, decode with prefill mixed in, and pure prefill. One rank's GPU kernels per phase plus the engine's
step counters; the mapping to dense / experts / all-reduce / draft / GDN is read from the kernel names offline.
"""

import argparse
import gzip
import json
import random
import shutil
import threading
import time
from pathlib import Path

from arena.throughput_bench import summarize_trace

from . import tasks
from .l4_depth import messages
from .l4_serving import cards, decode_steps, post

BASE = "http://127.0.0.1:1234/v1"
TOP = 80


def counters(root):
    reply = post(root + "/metrics", timeout=10)
    out = {}
    for line in reply["body"].splitlines() if reply["status"] == 200 else ():
        if line.startswith("vllm:") and ("spec_decode" in line or "generation_tokens_total" in line
                                         or "prompt_tokens_total" in line or "iteration_tokens_total" in line):
            name, value = line.split("{")[0], line.rsplit(" ", 1)[1]
            try:
                out[name] = out.get(name, 0.0) + float(value)
            except ValueError:
                pass
    return out


def collect(folder, phase, dest, done):
    """The trace files new since the last phase: every rank summarized, rank files kept for the first only."""
    new = sorted(p for p in Path(folder).rglob("*.json*") if p not in done)
    rows = []
    for k, path in enumerate(new):
        done.add(path)
        try:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt") as stream:
                summary = summarize_trace(json.load(stream), top=TOP)
            rows.append({"file": path.name, **summary})
            if k == 0:
                shutil.copyfile(path, Path(dest) / f"{phase}-{path.name}")
        except Exception as exc:  # noqa: BLE001 - a reading
            rows.append({"file": path.name, "error": repr(exc)[:300]})
    return rows


def phase_decode(base, model, fields, name, rows, lines, tokens, ctx):
    root = base.rstrip("/").rsplit("/v1", 1)[0]
    before = counters(root)
    row = decode_steps(base, model, fields, [rows], tokens=tokens, level_seconds=600, used=cards,
                       ask=(lambda i: messages(i, lines)) if lines else None, window=(0.5, 0.95), profile=[rows])
    after = counters(root)
    ctx["rows"].append({"phase": name, "decode": row, "counters": {k: after[k] - before.get(k, 0) for k in after}})


def warm(chat, model, fields, rows, lines):
    replies = [None] * rows

    def one(i):
        replies[i] = post(chat, tasks.chat_request(model, messages(i, lines), 1, **fields), timeout=1200)

    threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(rows)]
    t0 = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    good = [json.loads(r["body"]).get("usage") or {} for r in replies if r and r["status"] == 200]
    return {"answered": len(good), "seconds": round(time.monotonic() - t0, 1),
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in good),
            "cached_tokens": sum((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0 for u in good)}


def run(study, server_dir, dest, budget, base=BASE):
    begun, spec = time.monotonic(), study["arc2"]
    root, chat = base.rstrip("/").rsplit("/v1", 1)[0], base.rstrip("/") + "/chat/completions"
    model = json.loads(post(base + "/models", timeout=60)["body"])["data"][0]["id"]
    fields = {"temperature": (spec.get("settings") or {}).get("temperature") or float(study["temperature"]),
              "effort": (spec.get("settings") or {}).get("effort") or study["reasoning_effort"]}
    folder, done = Path(server_dir) / "torch-profile", set()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    ctx = {"rows": [], "phases": {}}
    body = tasks.chat_request(model, messages(0, 400), 1, **fields)
    fill = {key: body[key] for key in ("model", "messages", "chat_template_kwargs")}
    a_line = int(json.loads(post(root + "/tokenize", fill, timeout=60)["body"])["count"]) / 400
    rows = min(32, int(spec.get("l4x4", {}).get("slots") or study["generation_slots"]))
    left = lambda: budget - (time.monotonic() - begun)  # noqa: E731
    lines = {depth: int(depth / a_line) for depth in (16000, 56000, 100000)}

    def traced(name, call):
        call()
        time.sleep(3)
        ctx["phases"][name] = collect(folder, name, dest, done)
        Path(dest / "stepprof.json").write_text(json.dumps(ctx, indent=1))

    traced("decode-shallow", lambda: phase_decode(base, model, fields, "decode-shallow", rows, 0, 512, ctx))

    def prefill():  # pure prefill of the deepest of the 32-row prompts, traced in its first steps
        start = post(root + "/start_profile", method="POST", timeout=60)["status"]
        row = warm(chat, model, fields, rows, lines[56000])
        ctx["rows"].append({"phase": "prefill", "start": start, "warm": row,
                            "stop": post(root + "/stop_profile", method="POST", timeout=300)["status"]})

    traced("prefill-56k", prefill)
    traced("decode-56k", lambda: phase_decode(base, model, fields, "decode-56k", rows, lines[56000], 640, ctx))
    traced("decode-16k", lambda: phase_decode(base, model, fields, "decode-16k", rows, lines[16000], 640, ctx))

    def mixed():  # 28 rows decoding at 56K while cold 12K prompts arrive, as a play's turns do
        stop, sent = threading.Event(), []

        def inject():
            n, rng = 0, random.Random(7)
            while not stop.is_set():
                n += 1
                text = [{"role": "user", "content": "\n".join(str(rng.randrange(10**4, 10**6)) for _ in range(
                    int(12000 / a_line))) + "\nSay ok."}]
                threading.Thread(target=post, args=(chat, tasks.chat_request(model, text, 8, **fields)),
                                 kwargs={"timeout": 300}, daemon=True).start()
                sent.append(n)
                stop.wait(1.5)

        thread = threading.Thread(target=inject, daemon=True)
        thread.start()
        phase_decode(base, model, fields, "mixed-56k", min(28, rows), lines[56000], 640, ctx)
        stop.set()
        ctx["rows"][-1]["injected"] = len(sent)

    if left() > 500:
        traced("mixed-56k", mixed)
    if left() > 400:  # 16 rows at 100K: a 56K prefix is cached; more would not fit the pool
        ctx["rows"].append({"phase": "warm-100k", "warm": warm(chat, model, fields, 16, lines[100000])})
        traced("decode-100k", lambda: phase_decode(base, model, fields, "decode-100k", 16, lines[100000], 512, ctx))
    post(root + "/reset_prefix_cache", method="POST", timeout=60)
    ctx["seconds"] = round(time.monotonic() - begun, 1)
    Path(dest / "stepprof.json").write_text(json.dumps(ctx, indent=1))
    return ctx


def in_play(server_dir, dest, delay, base=BASE):
    """A trace of the play's own steps; never raises into the driver."""
    try:
        time.sleep(delay)
        root = base.rstrip("/").rsplit("/v1", 1)[0]
        folder, done = Path(server_dir) / "torch-profile", set(Path(server_dir, "torch-profile").rglob("*.json*"))
        before = counters(root)
        start = post(root + "/start_profile", method="POST", timeout=60)["status"]
        time.sleep(45)
        stop = post(root + "/stop_profile", method="POST", timeout=300)["status"]
        after = counters(root)
        time.sleep(3)
        Path(dest, "in-play.json").write_text(json.dumps(
            {"start": start, "stop": stop, "counters": {k: after[k] - before.get(k, 0) for k in after},
             "summary": collect(folder, "in-play", dest, done)}, indent=1))
    except Exception as exc:  # noqa: BLE001
        Path(dest, "in-play.json").write_text(json.dumps({"error": repr(exc)[:400]}))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--server-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-seconds", type=float, default=1500)
    args = parser.parse_args(argv)
    found = run(json.loads(Path(args.study).read_text()), args.server_dir, args.output, args.budget_seconds)
    print("ARC2_L4_STEPPROF " + json.dumps([{k: r.get(k) for k in ("phase",)} for r in found["rows"]]), flush=True)


if __name__ == "__main__":
    main()
