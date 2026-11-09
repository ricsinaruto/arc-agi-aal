"""E1/E2/E2+ replay: every stored candidate program of a c240 run, re-run on 16
extra views of its task (8 symmetries x {identity colours, one fixed colour permutation}) composed on top of the
copy's own view. Per candidate and view: training pairs passed and the test outputs mapped back to the task frame.
The selection rules are scored offline from these rows (``view_rules.py``); nothing here picks an answer.
usage: modal run --detach scripts/arc2_replay/modal_view_replay.py --runs v49r18,base18 --out DIR"""
import hashlib
import json
import re
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[min(1, len(HERE.parents) - 1)]
SCRATCH = Path("<scratch>/v49")
DATA = ROOT / "dist/arc2-rtx-c240-hx-base18-v3b/data/arc2/evaluation"
app = modal.App("arc2-view-replay")
image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.2.6", "scipy==1.15.3")
         .add_local_dir(str(ROOT / "src/arc2"), "/w/arc2", ignore=["__pycache__"]))


def sources(run_dir):
    """candidate hash -> program source, recovered from the run's replies (tool code and fenced blocks)."""
    want = set()
    for line in open(run_dir / "tasks.jsonl"):
        for c in json.loads(line)["candidates"]:
            want.add(c["code"])
    found = {}
    for line in open(run_dir / "events.jsonl"):
        d = json.loads(line)
        for src in [d.get("tool_code") or ""] + re.findall(r"```(?:python)?\n(.*?)```", d.get("content") or "", re.S):
            for s in {src, src.strip(), src.strip() + "\n"} if src else ():
                h = hashlib.sha256(s.encode()).hexdigest()[:16]
                if h in want:
                    found[h] = s
    return found


@app.function(image=image, cpu=(0.125, 2), timeout=1800, retries=0, max_containers=60)
def replay(item):
    import sys

    sys.path[:0] = ["/w"]
    from arc2 import loop, sandbox, tasks

    task, tid = item["task"], item["task_id"]
    perm = list(range(10))
    seed = int(hashlib.sha256(f"{tid}:replay".encode()).hexdigest()[:12], 16)
    for n in range(9, 1, -1):  # a fixed permutation of colours 1-9 (0 stays: background)
        seed, r = divmod(seed * 6364136223846793005 + 1442695040888963407 & (2**64 - 1), n)
        perm[n], perm[r + 1] = perm[r + 1], perm[n]
    inv = {v: i for i, v in enumerate(perm)}
    out = []
    for c in item["candidates"]:
        view = loop.augment_view(task, tid, c["copy"])
        rows = []
        for colour in (False, True):
            for s, (fwd, back) in enumerate(loop.SYMMETRIES):
                g = (lambda x: [[perm[v] for v in row] for row in fwd(x)]) if colour else (lambda x: fwd(x))
                ungo = (lambda y: back([[inv.get(v, v) for v in row] for row in y])) if colour else (lambda y: back(y))
                vt = {"train": [{"input": g(p["input"]), "output": g(p["output"])} for p in view["train"]],
                      "test": [{"input": g(t["input"])} for t in view["test"]]}
                res = sandbox.run(c["src"], [p["input"] for p in vt["train"]] + [t["input"] for t in vt["test"]],
                                  strict=False)
                verdict = tasks.judge(vt, res)
                tests = []
                for t in verdict["tests"]:
                    try:
                        tests.append(None if t is None else loop.augment_back(ungo(t), tid, c["copy"]))
                    except Exception:  # noqa: BLE001 - a ragged grid
                        tests.append(None)
                rows.append({"colour": colour, "sym": s, "passed": verdict["passed"], "pairs": verdict["pairs"],
                             "tests": tests})
        out.append({"key": c["key"], "views": rows})
    return {"row": item["row"], "candidates": out}


@app.local_entrypoint()
def main(runs: str, out: str):
    items = []
    for run in runs.split(","):
        found = sources(SCRATCH / run)
        for i, line in enumerate(open(SCRATCH / run / "tasks.jsonl")):
            r = json.loads(line)
            tid = r["task"].split("@")[0].split("#")[0]
            task = json.loads((DATA / f"{tid}.json").read_text())
            cands = [{"key": f"{run}:{i}:{j}", "src": found[c["code"]], "copy": c.get("copy", 0)}
                     for j, c in enumerate(r["candidates"]) if c["code"] in found]
            if cands:
                items.append({"row": f"{run}:{i}", "task": task, "task_id": tid, "candidates": cands})
    print(f"{len(items)} task copies, {sum(len(x['candidates']) for x in items)} candidates")
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "views.jsonl", "w") as fh:
        for res in replay.map(items, order_outputs=False, return_exceptions=True):
            fh.write(json.dumps(res if isinstance(res, dict) else {"error": str(res)[:300]}) + "\n")
    print("REPLAY_DONE", out)
