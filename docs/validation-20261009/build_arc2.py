"""ARC-AGI-2 arm bundle: a built v8.2 tick + cap bundle with the arc2 trials in place of the game trial.

    python docs/validation-20261009/build_arc2.py BASE OUTPUT DATA [--tune-wall=1200] [--eval-wall=6000]
                                                  [--tune-tasks=80] [--seed=0] [--plan=v1]
                                                  [--candidate=qwen38-27b-fp8] [--rates=130,180,270,330]

BASE    a bundle built from this branch's base commit by build_v82.py (worktree v82tc: dist/v82tc-w64-c110-24h-v1).
        Its server, weights, window, streams, gate and wheels are kept byte for byte; the source must equal this
        worktree's except for the arm's files.
DATA    the data/ folder of the public ARC-AGI-2 repository (github.com/arcprize/ARC-AGI-2, Apache-2.0). Every task
        is parsed, checked and re-written; nothing from it is executed.

Trials (test kind ``arc2-run``): ``tune`` on a seeded sample of the training set, half at each of two efforts, then
``eval`` on the 120 public evaluation tasks, which runs only if ``tune`` completed. Tune on ``tune`` alone.

--plan=v2 (9 October, after arc2-vr-v1: 76% of tokens went to replies cut at the cap; the v1 sample scored 87%):
``tune`` on larger training tasks (cells at least the evaluation set's lower quartile, none from the seed-0 sample of
v1), every task under each of four arms (effort xhigh / low x reply cap 16,384 / 8,192); ``eval`` plays the arm with
the best score at 30K tokens per task. Use --tune-wall=2700 --eval-wall=4400 --tune-tasks=40 --seed=1.

--plan=v3-tune / v3-eval (test as the competition's 240-task run is scored). One bundle per
trial. Roster trials: no cap per task, one total of completion tokens, the score read against it at the totals four
L4s would generate in 12 h at 180 / 270 / 330 / 550 tok/s (for 240 tasks; scaled to the roster's size).
  v3-tune  the v2 sample and arms, every task under every arm, each arm its own roster and total, to the 330 mark.
           --tune-tasks=48 --seed=1 --tune-wall=7200
  v3-eval  the 120 public evaluation tasks twice (240 independent entries) under --arm=<name>, the arm
           ``python -m arc2.trial winner <tune report.json>`` names; to the 550 mark. --eval-wall=14100

Hard training roster (10 October: the evaluation's verified programs are right on 62% of test outputs, the tune
roster's on 96%, so the tune roster cannot measure a change aimed at that). Training tasks only.
  v3-screen  the --tune-tasks=400 largest training tasks no earlier sample used, each under xhigh-8k with its own
             25,000 tokens (the loop as it is: a task ends after its confirming attempt). --tune-wall=7000
  v3-hard    --screen=<the screen's trial folder>: 48 of its tasks that ended with a verified program wrong on a
             test output or with none verified, --mix=29,19 of the two (the evaluation's misses: 60% / 40%), drawn
             with --seed; every task under a baseline arm, the review turn and the 16K reply cap, each arm its
             own roster to the 330 mark (59.4K tokens a task). --tune-wall=7100
             --arms=base,tool: those arms instead (tool: the persistent `python` tool, stage 3). --tune-wall=4800
Kaggle package (10 October; four L4s, ``arc2.kaggle``):
  kaggle     the v3-tune roster (the start probe's first turns) and the 120 evaluation tasks (a scored rehearsal
             off Kaggle), plus the notebook, its metadata and the dataset folder. --tune-tasks=48 --seed=1
             --weights=all25v71b
  v3-review  --screen=<the screen's trial folder>: the review turn alone on the screen's stored conversations
             (``arc2.trial.review_replay``). Items: every task's first verified program, once, and every
             verified program that is wrong on a test output, --samples=4 times. --tune-wall=2700
             --review-style=second: the message that asks for the other reading as the second answer.
  v3-review-eval  --screen=<the evaluation run's trial folder>: one shot (docs/arc2-arm.md). One review, in the
             frozen "second" message, of the first verified program of every entry of that run; then
             ``python -m arc2.trial rescore``. The conversations are read here by script; counts only come out.
"""
import json
import random
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path.cwd()
OWN = {"tune-wall": "1200", "eval-wall": "6000", "tune-tasks": "80", "seed": "0", "plan": "v1", "arm": "",
       "candidate": "", "rates": "", "screen": "", "mix": "29,19", "samples": "4", "review-style": "check",
       "arms": "", "weights": "",
       "kaggle-arm": "", "nested-mask": ""}
SCREEN_ARM, SCREEN_TOKENS = {"name": "xhigh-8k", "cap": 8192}, 25000
HARD_ARMS = [{"name": "base", "cap": 8192}, {"name": "review", "cap": 8192, "review": True}, {"name": "cap16"}]
TOOL_ARM = {"name": "tool", "cap": 8192, "tool": True}  # stage 3: arc2.tasks.PYTHON_TOOL
HARD_RATES = (130, 180, 270, 330)
REVIEW_SECOND_SHA256 = "5cd66010cb6d072fa89e0bbae01907dccd3ffc1d6430a05af5af51dadce43986"  # as in replay v2
CANDIDATE_STREAMS = 48  # a dense 27B checkpoint on the RTX PRO 6000: the gate's block budget decides below this
V2_ARMS = [{"name": "xhigh-16k"}, {"name": "low-16k", "effort": "low"}, {"name": "xhigh-8k", "cap": 8192},
           {"name": "low-8k", "effort": "low", "cap": 8192}]
RATES, RUN_SECONDS, ROSTER = (180, 270, 330, 550), 43200, 240  # tok/s on four L4s, the 12 h run, its tasks
POLICY = {"confirm_delay": 20000, "settled_delay": 100000}  # roster order: arc2.loop
# --nested-mask=<mask file>: a mask that prunes more inside the weights' own mask, served from the same dataset.
# --weights=<name>: the 3-bit expert set in place of the base bundle's (its mask from data/expert_prune). all25v71b:
# the set the submitted ARC-AGI-3 kernels serve, the only one on the Kaggle account (values as in their bundles).
WEIGHTS = {"all25v71b": {"level": "prune-t37-int3-requant-all25v71b-v1",
                         "dataset": "richardcsaky/arc-expert-int3-requant-t37-all25v71b-v1",
                         "mask_sha256": "43dd9b2f9d36d0537aa73dfa0d65cfcca172698a8ad7a4eba829646fae3a006b",
                         "manifest_sha256": "17161a50dabe4c93360079fb32a147d20e3a5a601be6e678804ae26bc5f2d099"}}
KAGGLE_COMPETITION = "arc-prize-2026-arc-agi-2"
# Kaggle's Python 3.12 image, as the ARC-AGI-3 kernels pin it.
KAGGLE_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
# The notebook's one cell. The hidden rerun (240 tasks) plays to 11 h 30 min after the cell starts; the saved
# version plays REHEARSAL_TASKS public tasks for REHEARSAL_SECONDS, scored when the file has their solutions.
KAGGLE_CELL = '''from pathlib import Path
import hashlib, json, os, subprocess, sys, tarfile, time, traceback
START = time.time()
SAVE_TASKS, SAVE_SECONDS, RUN_SECONDS = 11, 1800, 41400  # the save run: the competition's time per task, 30 minutes
OUT = Path('/kaggle/working/submission.json')
status = {'mode': 'full', 'error': None}


def grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) and 1 <= len(r) <= 30 for r in g)
            and all(len(r) == len(g[0]) for r in g) and all(type(c) is int and 0 <= c <= 9 for r in g for c in r))


def repaired(answers, tests):
    # the file the scorer takes: every task of the test file, per test input two grids; the input where one is missing
    out = {}
    for name, task in tests.items():
        rows = answers.get(name) if isinstance(answers, dict) else None
        rows = rows if isinstance(rows, list) and len(rows) == len(task['test']) else [None] * len(task['test'])
        out[name] = [{key: row[key] if isinstance(row, dict) and grid(row.get(key)) else t['input']
                      for key in ('attempt_1', 'attempt_2')} for row, t in zip(rows, task['test'])]
    return out


tests = None
try:
    data = next(p for p in (Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-2'),
                            Path('/kaggle/input/arc-prize-2026-arc-agi-2')) if p.is_dir())
    tests = json.loads((data / 'arc-agi_test_challenges.json').read_text())
    OUT.write_text(json.dumps(repaired({}, tests)))  # before anything that can fail
    assert sys.version_info[:2] == (3, 12), 'the Python 3.12 image (kernel-metadata docker_image) is required'
    roots = [Path('/kaggle/input/datasets/__DATASET__'), Path('/kaggle/input/' + '__DATASET__'.split('/')[1])]
    mount = next((p for p in roots if p.is_dir()), None)
    assert mount is not None, 'attach the dataset __DATASET__'
    blob = mount / '__BLOB__'
    with blob.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == '__DIGEST__', 'the runtime archive differs from the build'
    bundle = Path('/tmp/arena-runtime')  # and /tmp/arena to work in: where the ARC-AGI-3 kernels put theirs
    bundle.mkdir(exist_ok=False)
    with tarfile.open(blob, 'r:gz') as archive:
        archive.extractall(bundle, filter='data')
    # The short save run only when both hold: Kaggle does not mark a scoring rerun, and the test file is, byte for
    # byte, the public placeholder (240 training tasks, the competition's file of 10 March 2026). Anything else
    # is the full run: a wrong test costs a long save run, never a short scoring run.
    placeholder = hashlib.sha256((data / 'arc-agi_test_challenges.json').read_bytes()).hexdigest() == '__PLACEHOLDER__'
    short = placeholder and not os.environ.get('KAGGLE_IS_COMPETITION_RERUN')
    status.update(mode='save' if short else 'full', placeholder=placeholder, tasks=len(tests),
                  rerun=os.environ.get('KAGGLE_IS_COMPETITION_RERUN'))
    print(json.dumps(status), flush=True)  # the mode, in the log from the first minute
    command = [sys.executable, '-u', '-m', 'arc2.kaggle', '--bundle', str(bundle), '--working', '/tmp/arena',
               '--output', '/kaggle/working/arc2', '--input-root', '/kaggle/input',
               '--challenges', str(data / 'arc-agi_test_challenges.json'), '--submission', str(OUT),
               '--end-epoch', str(START + RUN_SECONDS)__OPTIONS__]
    if short:
        command += ['--tasks', str(SAVE_TASKS), '--play-seconds', str(SAVE_SECONDS)]
        solutions = data / 'arc-agi_training_solutions.json'  # the placeholder's tasks are training tasks
        if solutions.is_file() and set(tests) <= set(json.loads(solutions.read_text())):
            command += ['--solutions', str(solutions)]
    with open('/kaggle/working/arc2-driver.log', 'w') as log:
        result = subprocess.run(command, check=False, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, 'PYTHONPATH': str(bundle / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'})
    record = Path('/kaggle/working/arc2/run.json')
    record = json.loads(record.read_text()) if record.is_file() else {}
    status.update(exit=result.returncode, **{k: record.get(k) for k in (
        'stage', 'error', 'answered', 'tasks_in_file', 'submission_problems', 'gpus', 'attempts', 'probe', 'play')})
except BaseException as exc:  # the file below stands whatever broke: a notebook that raises scores nothing
    status['error'] = status.get('error') or (type(exc).__name__ + ': ' + str(exc))[:600]
    traceback.print_exc()
if tests is not None:
    try:
        answers = json.loads(OUT.read_text())
    except (OSError, ValueError):
        answers = {}
    final = repaired(answers, tests)
    if final != answers:
        OUT.write_text(json.dumps(final))
    status.update(tasks_written=len(final), outputs=sum(len(v) for v in final.values()),
                  answered=sum(row['attempt_1'] != t['input'] or row['attempt_2'] != t['input']
                               for name, rows in final.items() for row, t in zip(rows, tests[name]['test'])))
print(json.dumps({**status, 'elapsed': round(time.time() - START)}))
'''
# sha256 of the competition's public arc-agi_test_challenges.json (240 placeholder tasks, 259 outputs)
KAGGLE_PLACEHOLDER = "232264c58f825ee77327dcfc9f4e5cb2f83b8d997eb69032be1bf2205bbe1a83"
CURVE_STEP = 2500  # tokens per task between the points of the fine curve
SETUP_SECONDS = 3900  # the base bundle's allowance (int3 weights)
# The bench: the L4 probe only. kv4_runtime: the rotation guard, off unless ARC_KV4_GUARD=1 (the L4 profile sets it).
CHANGED = {"src/arena/run.py", "src/arena/expert_3bit_bench.py", "src/arc_infra/kv4_runtime.py"}
TUNE_TIERS, EVAL_TIERS = [10000, 30000], [10000, 30000, 60000, 100000, 300000]


def kaggle_files(out, arm=""):
    """The runtime archive in a dataset folder, the submission notebook and its metadata (four L4s, no internet,
    the Python 3.12 image: the runtime and the wheels are 3.12's, Kaggle's current image is 3.13). ``arm``
    (--kaggle-arm): the bundle's arm the notebook plays, e.g. ``tool``; none: ``arc2.kaggle``'s (the tool arm)."""
    import hashlib
    import tarfile

    study = json.loads((out / "configs/comparison.json").read_text())
    upload, dataset = out / "dataset-upload", f"richardcsaky/{out.name}-runtime"
    upload.mkdir()
    blob = upload / f"{out.name}.tar.gz.blob"
    with tarfile.open(blob, "w:gz") as archive:
        for path in sorted(out.iterdir()):
            if path.name != "dataset-upload":
                archive.add(path, arcname=path.name)
    digest = hashlib.sha256(blob.read_bytes()).hexdigest()
    (upload / "dataset-metadata.json").write_text(json.dumps({
        "id": dataset, "title": f"{out.name} runtime", "licenses": [{"name": "other"}],
        "description": f"Offline runtime of an ARC-AGI-2 notebook; third-party wheels keep their licences. "
                       f"{blob.name} sha256 {digest}."}, indent=2) + "\n")
    chosen = {"roster": {"arm": arm, "rates": [330]}} if arm else {}
    if arm and arm not in [a["name"] for a in study["trials"][0]["arms"]]:
        raise SystemExit(f"--kaggle-arm={arm}: not an arm of the bundle's roster trial")
    options = f", '--options', {json.dumps(chosen)!r}" if chosen else ""
    code = (KAGGLE_CELL.replace("__DATASET__", dataset).replace("__BLOB__", blob.name).replace("__DIGEST__", digest)
            .replace("__OPTIONS__", options).replace("__PLACEHOLDER__", KAGGLE_PLACEHOLDER))
    compile(code, "notebook", "exec")
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [
                 "# ARC-AGI-2: verify-and-refine on four L4s\n", "Qwen3.8-Flash-Next on the four cards; per task, "
                 "programs that must reproduce every training pair, two answers per test input. Offline."]},
             {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
              "source": code.splitlines(keepends=True)}]
    (out / "submission.ipynb").write_text(json.dumps({
        "cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                    "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5}, indent=1) + "\n")
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
    from build_arena import KAGGLE_INPUTS, MODEL_SOURCE

    metadata = {"id": f"richardcsaky/{out.name}", "title": out.name, "code_file": "submission.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
                "enable_internet": False, "machine_shape": "NvidiaL4", "docker_image": KAGGLE_IMAGE,
                "docker_image_pinning_type": "original",
                "dataset_sources": [dataset, *KAGGLE_INPUTS[1:], *study.get("model_datasets", [])],
                "competition_sources": [KAGGLE_COMPETITION], "model_sources": [MODEL_SOURCE], "kernel_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"kernel": metadata["id"], "dataset": dataset, "blob_mib": round(blob.stat().st_size / 2**20),
            "sha256": digest, "inputs": metadata["dataset_sources"] + metadata["model_sources"]}


def checked(path):
    sys.path.insert(0, str(ROOT / "src"))
    from arc2.tasks import valid

    task = json.loads(path.read_text())
    grids = [g for part in ("train", "test") for pair in task[part] for g in (pair["input"], pair["output"])]
    if set(task) != {"train", "test"} or not task["train"] or not task["test"] or not all(valid(g) for g in grids):
        raise SystemExit(f"not an ARC task: {path.name}")
    return {part: [{"input": p["input"], "output": p["output"]} for p in task[part]] for part in ("train", "test")}


if __name__ == "__main__":
    plain = [a for a in sys.argv[1:] if not a.startswith("--")]
    values = {**OWN, **{a[2:].split("=", 1)[0]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--")}}
    if len(plain) != 3 or set(values) != set(OWN):
        raise SystemExit(__doc__)
    base, out, data = (Path(p).resolve() for p in plain)
    for path in (base / "src").rglob("*.py"):
        rel = path.relative_to(base)
        if rel.as_posix() not in CHANGED and path.read_bytes() != (ROOT / rel).read_bytes():
            raise SystemExit(f"the base bundle's source differs from this worktree: {rel}")
    shutil.copytree(base, out, ignore=shutil.ignore_patterns("dataset-upload", "__pycache__", "*.ipynb", "source.zip",
                                                             "kernel-metadata.json", "build-manifest.json"))
    shutil.copytree(ROOT / "src/arc2", out / "src/arc2", ignore=shutil.ignore_patterns("__pycache__"))
    for rel in CHANGED:
        shutil.copyfile(ROOT / rel, out / rel)
    ids = {split: sorted(p.stem for p in (data / split).glob("*.json")) for split in ("training", "evaluation")}
    if len(ids["evaluation"]) != 120 or len(ids["training"]) != 1000:
        raise SystemExit(f"expected 1,000 training and 120 evaluation tasks: {({k: len(v) for k, v in ids.items()})}")
    plan, screened, replay = values["plan"], None, None
    if plan in ("v2", "v3-tune", "v3-screen", "v3-hard", "v3-review", "kaggle"):
        def cells(split, name):
            task = checked(data / split / f"{name}.json")
            return sum(len(g) * len(g[0]) for part in task.values() for pair in part for g in pair.values())

        floor = sorted(cells("evaluation", name) for name in ids["evaluation"])[len(ids["evaluation"]) // 4]
        used = set(random.Random(0).sample(ids["training"], 80))  # the v1 sample
        large = [name for name in ids["training"] if name not in used and cells("training", name) >= floor]
        if plan in ("v3-screen", "v3-hard", "v3-review"):  # no task of the tune roster (the v2 sample is inside it)
            used |= set(random.Random(1).sample(large, 48))
            rest = sorted((name for name in ids["training"] if name not in used),
                          key=lambda name: (-cells("training", name), name))
        else:
            ids["training"] = large
    elif plan not in ("v1", "v3-eval", "v3-review-eval"):  # "kaggle" is "v3-tune" above and in the trial
        raise SystemExit("--plan is v1, v2, v3-tune, v3-eval, v3-screen, v3-hard, v3-review or v3-review-eval")
    if plan == "v3-screen":
        ids["training"] = sorted(rest[:int(values["tune-tasks"])])
    elif plan == "v3-hard":
        sys.path.insert(0, str(ROOT / "src"))
        from arc2.loop import guesses

        pools, seen = {"verified_wrong": [], "unverified": [], "solved": []}, {"tasks": 0, "outputs": 0, "right": 0}
        for row in map(json.loads, (Path(values["screen"]) / "tasks.jsonl").read_text().splitlines()):
            if row["task"] not in rest or row["ended"] not in ("verified", "budget"):
                continue  # not a screened task, or one the wall or request errors cut
            task = checked(data / "training" / f"{row['task']}.json")
            picks = guesses(row["candidates"], [t["input"] for t in task["test"]])
            right = [any(g == t["output"] for g, _ in pick) for pick, t in zip(picks, task["test"])]
            verified = any(c["verified"] for c in row["candidates"])
            if verified:
                seen["tasks"] += 1
                seen["outputs"] += len(right)
                seen["right"] += sum(right)
            pools["solved" if all(right) else "verified_wrong" if verified else "unverified"].append(row["task"])
        want = dict(zip(("verified_wrong", "unverified"), (int(x) for x in values["mix"].split(","))))
        total, rng = sum(want.values()), random.Random(int(values["seed"]))
        short = {kind: n - min(n, len(pools[kind])) for kind, n in want.items()}  # a short kind: the other fills it
        take = {kind: min(len(pools[kind]), n + short[other])
                for (kind, n), other in zip(want.items(), ("unverified", "verified_wrong"))}
        if sum(take.values()) < total:
            raise SystemExit(f"the screen left {({k: len(v) for k, v in pools.items()})}: fewer than {total} hard tasks")
        chosen = {kind: sorted(rng.sample(sorted(pools[kind]), n)) for kind, n in take.items()}
        ids["training"] = sorted(chosen["verified_wrong"] + chosen["unverified"])
        screened = {"tasks": sum(map(len, pools.values())), **{k: len(v) for k, v in pools.items()},
                    "verified_tasks": seen["tasks"], "verified_outputs": seen["outputs"],
                    "verified_outputs_right": seen["right"],
                    "kept": chosen, "mix": want}
    elif plan in ("v3-review", "v3-review-eval"):
        import hashlib

        sys.path.insert(0, str(ROOT / "src"))
        from arc2 import tasks as texts

        frozen = plan == "v3-review-eval"  # the evaluation run's entries: one review each, the frozen message
        split, pool = ("evaluation", set(ids["evaluation"])) if frozen else ("training", set(rest))
        if frozen and (hashlib.sha256(texts.REVIEW_SECOND.encode()).hexdigest() != REVIEW_SECOND_SHA256
                       or values["review-style"] != "second"):
            raise SystemExit("v3-review-eval runs the frozen review message: --review-style=second, unchanged text")
        said = {}  # task -> attempt -> its replies
        for row in map(json.loads, (Path(values["screen"]) / "events.jsonl").read_text().splitlines()):
            if row.get("event") == "reply":
                said.setdefault(row["task"], {}).setdefault(row["attempt"], []).append(row)
        replay = []
        for row in map(json.loads, (Path(values["screen"]) / "tasks.jsonl").read_text().splitlines()):
            verified = sorted((c for c in row["candidates"] if c["verified"]), key=lambda c: c["tokens_at"])
            name = row["task"].split("#")[0]  # an evaluation entry is task#copy
            if name not in pool or not verified:
                continue
            task = checked(data / split / f"{name}.json")
            want, seen = [t["output"] for t in task["test"]], set()
            for c in verified[:1] if frozen else verified:
                wrong = c["tests"] != want
                if (c is not verified[0] and not wrong) or c["code"] in seen:
                    continue
                seen.add(c["code"])
                messages = [{"role": "system", "content": texts.SYSTEM},
                            {"role": "user", "content": texts.task_prompt(task)}]
                for reply in sorted(said[row["task"]][c["attempt"]], key=lambda x: x["reply"]):
                    messages.append({"role": "assistant", "content": reply["content"],
                                     **({"reasoning": reply["reasoning"]} if reply["reasoning"] else {})})
                    if reply["reply"] == c["reply"]:
                        if hashlib.sha256(reply["code"].encode()).hexdigest()[:16] != c["code"]:
                            raise SystemExit(f"the stored reply is not the candidate's program: {row['task']}")
                        break
                    messages.append({"role": "user", "content": {
                        "truncated": texts.TRUNCATED, "no_code": texts.NO_CODE}.get(reply["outcome"]) or reply["feedback"]})
                replay.append({"id": f"{row['task']}:{c['code']}", "task": name, "tests": c["tests"],
                               "kind": "wrong" if wrong else "right", "messages": messages,
                               "samples": int(values["samples"]) if wrong and not frozen else 1})
        ids[split] = sorted({item["task"] for item in replay})
    else:
        ids["training"] = sorted(random.Random(int(values["seed"])).sample(ids["training"], int(values["tune-tasks"])))
    if plan.startswith("v3"):  # one trial per bundle; only its split is shipped
        split = "evaluation" if plan in ("v3-eval", "v3-review-eval") else "training"
        ids = {split: ids[split]}
    for split, names in ids.items():
        (out / "data/arc2" / split).mkdir(parents=True)
        for name in names:
            (out / "data/arc2" / split / f"{name}.json").write_text(json.dumps(checked(data / split / f"{name}.json")))
    path = out / "configs/comparison.json"
    study = json.loads(path.read_text())
    if len(study["trials"]) != 1 or study["trials"][0]["kind"] != "duck-transfer":
        raise SystemExit("the base bundle is not a one-trial Duck study")
    env = study["trials"][0]["server_env"]  # unchanged between the trials: one server start
    walls = int(values["tune-wall"]), int(values["eval-wall"])
    common = {"kind": "arc2", "competition": False, "server_env": env}

    def marks(entries, rates):
        return {str(rate): rate * RUN_SECONDS * entries // ROSTER for rate in rates}

    if plan in ("v3-tune", "kaggle"):
        walls, n = walls[:1], len(ids["training"])
        study["trials"] = [{"name": "tune", **common, "wall_seconds": walls[0], "split": "training",
                            "tasks": ids["training"], "paired": True, "min_verified_tasks": 3,
                            # the package can play the tool arm (arc2.kaggle --options, its roster arm)
                            "arms": [*V2_ARMS, TOOL_ARM] if plan == "kaggle" else V2_ARMS,
                            "marks": marks(n, RATES[:3]), "curve_step": CURVE_STEP * n}]
    elif plan == "v3-screen":  # per-task tokens, the loop's own ending: no roster settings
        walls = walls[:1]
        study["trials"] = [{"name": "screen", **common, "wall_seconds": walls[0], "split": "training",
                            "tasks": ids["training"], "tiers": [SCREEN_TOKENS], "arms": [SCREEN_ARM],
                            "min_verified_tasks": 3}]
    elif plan == "v3-hard":
        walls, n = walls[:1], len(ids["training"])
        rates = tuple(int(x) for x in values["rates"].split(",")) if values["rates"] else HARD_RATES
        known = {a["name"]: a for a in [*HARD_ARMS, TOOL_ARM]}
        arms = [known[name] for name in values["arms"].split(",")] if values["arms"] else HARD_ARMS
        study["trials"] = [{"name": "hard", **common, "wall_seconds": walls[0], "split": "training",
                            "tasks": ids["training"], "arms": arms, "paired": True,
                            "marks": marks(n, rates), "curve_step": CURVE_STEP * n}]
    elif plan in ("v3-review", "v3-review-eval"):
        walls = walls[:1]
        (out / "data/arc2/replay").mkdir()
        (out / "data/arc2/replay/review.json").write_text(json.dumps(replay))
        study["trials"] = [{"name": "review", **common, "wall_seconds": walls[0], "split": split,
                            "tasks": ids[split], "replay": "data/arc2/replay/review.json",
                            "arms": [{**SCREEN_ARM, "review_style": values["review-style"]}]}]
    elif plan == "v3-eval":
        arm = next((a for a in [*V2_ARMS, TOOL_ARM] if a["name"] == values["arm"]), None)
        if arm is None:
            raise SystemExit("--arm is one of " + ", ".join(a["name"] for a in [*V2_ARMS, TOOL_ARM]))
        walls = walls[1:]
        # --rates=130,180,270,330: the marks of this run; the arm stops at the largest (10 October: nothing on
        # four L4s reaches 550 tok/s, and that mark was most of the run)
        rates = tuple(int(x) for x in values["rates"].split(",")) if values["rates"] else RATES
        study["trials"] = [{"name": "eval", **common, "wall_seconds": walls[0], "split": "evaluation",
                            "tasks": ids["evaluation"], "copies": 2, "arms": [arm],
                            "marks": marks(ROSTER, rates), "curve_step": CURVE_STEP * ROSTER}]
    else:
        study["trials"] = [
            {"name": "tune", **common, "wall_seconds": walls[0], "split": "training", "tasks": ids["training"],
             "tiers": TUNE_TIERS, "arms": [{"name": "xhigh"}, {"name": "low", "effort": "low"}],
             "min_verified_tasks": 3},
            {"name": "eval", **common, "requires": "tune", "wall_seconds": walls[1], "split": "evaluation",
             "tasks": ids["evaluation"], "tiers": EVAL_TIERS},
        ]
    if plan == "v2":
        study["trials"][0].update(arms=V2_ARMS, paired=True, tiers=[10000, 20000, 30000, 40000])
        study["trials"][1].update(arm_from={"trial": "tune", "budget": 30000}, tiers=[10000, 30000, 60000, 100000])
    if values["candidate"]:
        # The same trial on a reviewed dense checkpoint (arena.candidate_profiles; the bundle's bootstrap starts
        # arena.candidate_serving): its own quantization, draft tokens and effort, stock prefix caching, and none
        # of Flash's serving keys, kernels, overlays or server environment.
        sys.path.insert(0, str(ROOT / "src"))
        from arena.candidate_profiles import PROFILES

        profile = PROFILES[values["candidate"]]
        inventory = json.loads(profile.manifest.read_text())
        flash = study["server"]
        study["server"] = {"max_model_len": profile.native_context, "prefix_caching": True,
                           "mtp_tokens": profile.mtp_tokens, "max_num_seqs": CANDIDATE_STREAMS,
                           "gpu_memory_utilization": flash["gpu_memory_utilization"],
                           "max_num_batched_tokens": flash["max_num_batched_tokens"],
                           # k + 1 positions per stream per step with a draft of k (arena.hero_v2)
                           "max_cudagraph_capture_size": min(256, CANDIDATE_STREAMS * (profile.mtp_tokens + 1))}
        study.update(candidate_profile=profile.name, served_model=profile.model, model_revision=profile.revision,
                     model_datasets=[inventory["dataset"]], reasoning_effort=profile.effort,
                     generation_slots=CANDIDATE_STREAMS, active_limit=CANDIDATE_STREAMS,
                     startup_timeout_seconds=1800,
                     cache_experiment={**{k: v for k, v in study["cache_experiment"].items() if k != "prefix_cache"},
                                       "reclaim_skipped_blocks": False})
        study.pop("stall_dump_seconds", None)
        for trial in study["trials"]:
            trial["server_env"] = {k: v for k, v in trial["server_env"].items() if k == "ARENA_READY_GUARD"}
    study["name"] = out.name
    study["notebook_seconds"] = SETUP_SECONDS + sum(walls) + 600
    commit = subprocess.check_output(["git", "-C", str(data), "rev-parse", "HEAD"], text=True).strip()
    study["arc2"] = {"data": "data/arc2", "source": "github.com/arcprize/ARC-AGI-2", "source_commit": commit,
                     "base_bundle": base.name, "tune_seed": int(values["seed"]), "plan": plan,
                     "builder": "docs/validation-20261009/build_arc2.py",
                     **({"candidate": values["candidate"]} if values["candidate"] else {}),
                     **({"screen": screened} if screened else {}),
                     **({"replay": {kind: [sum(i["kind"] == kind for i in replay),
                                           sum(i["samples"] for i in replay if i["kind"] == kind)]
                                    for kind in ("right", "wrong")}} if replay else {}),
                     **({"settings": POLICY} if plan in ("v3-tune", "v3-eval", "v3-hard", "kaggle") else {})}
    if values["weights"]:
        import hashlib

        new, server = WEIGHTS[values["weights"]], study["server"]
        old = {"level": Path(server["expert_prune"]["mask"]).stem, "dataset": server["expert_overlay"]["dataset"],
               "mask_sha256": server["expert_prune"]["sha256"],
               "manifest_sha256": server["expert_overlay"]["manifest_sha256"]}
        text = json.dumps(study)
        for key, value in old.items():
            text = text.replace(value, new[key])
        study = json.loads(text)
        (out / "configs/expert-prune" / f"{old['level']}.json").unlink()
        mask = (ROOT / "data/expert_prune" / f"{new['level']}.json").read_bytes()
        if hashlib.sha256(mask).hexdigest() != new["mask_sha256"]:
            raise SystemExit("the mask in data/expert_prune is not the one named in WEIGHTS")
        (out / "configs/expert-prune" / f"{new['level']}.json").write_bytes(mask)
        study["arc2"]["weights"] = {"name": values["weights"], "replaced": old["dataset"]}
    if values["nested-mask"]:  # a deeper mask inside the weights' own: the same dataset (arc2.l4_serving.nest_overlay)
        import hashlib

        sys.path.insert(0, str(ROOT / "src"))
        from arc2.l4_serving import within
        from arc_infra.expert_prune_patch import load_mask

        server, deep = study["server"], Path(values["nested-mask"])
        held = load_mask(out / server["expert_prune"]["mask"], server["expert_prune"]["sha256"])
        extra = within(held["pruned"], load_mask(deep)["pruned"])
        if not extra:
            raise SystemExit("--nested-mask: not a deeper mask inside the bundle's")
        (out / "configs/expert-prune" / deep.name).write_bytes(deep.read_bytes())
        study["arc2"]["nested_mask"] = {**server["expert_prune"], "served": deep.name,
                                        "experts_pruned_beyond_the_overlay": extra}
        server["expert_prune"] = {"mask": f"configs/expert-prune/{deep.name}",
                                  "sha256": hashlib.sha256(deep.read_bytes()).hexdigest()}
    path.write_text(json.dumps(study, indent=2) + "\n")
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
    from build_arena import manifest  # noqa: E402

    identity = json.loads((out / "MANIFEST.json").read_text())
    identity.update(study=out.name, files=manifest(out), source="arc2 worktree on " + identity["git_commit"][:8],
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    if any("__pycache__" in name or name.endswith(".pyc") for name in identity["files"]):
        raise SystemExit("bytecode in the manifest")
    (out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    runpy.run_path(str(out / "bootstrap.py"))["verify"](out)
    if plan == "kaggle":
        kaggle = kaggle_files(out, values["kaggle-arm"])
    print(json.dumps({"bundle": str(out), **({"kaggle": kaggle} if plan == "kaggle" else {}), "files": len(identity["files"]), "notebook_seconds": study["notebook_seconds"],
                      "play_seconds": sum(walls), "tasks": {k: len(v) for k, v in ids.items()},
                      "marks": study["trials"][0].get("marks"), "window": study["windows"], "streams": study["active_limit"],
                      "effort": study["reasoning_effort"], "cap": study["completion_cap"],
                      **({"replay": study["arc2"]["replay"]} if replay else {}),
                      **({"screen": {k: v for k, v in screened.items() if k != "kept"},
                          "kept": {k: len(v) for k, v in screened["kept"].items()}} if screened else {})}))
