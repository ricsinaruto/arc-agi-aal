"""Bundle of the RSI loop (docs/rsi.md, "Loop as built (17 Oct)").

    python scripts/rsi2/build_loop.py OUT --gen0 dist/rsi-g0noise-n1 --pool runs/rsi2/noise/pool29.json
        --v27 ../rsi-loop/dist/rsi-loop-v27 [--notebook 85800] [--resume <pulled output/rsi2 dir>] [--loop '{json}']

Base: a gen-0 arena bundle (build_gen0_v85.py: v8.5 + fixed-10 refits). Added: src/rsi2_loop, two patches of the
fixed side (the level-bench trial runs the loop; arena.competition reaches the loop's gateway by RSI2_GATEWAY_URL),
and rsi/: the proposer's runtime and Prime Agent build (from the v27 loop bundle), the practice games (v27's vetted
set, disjoint from the selection games), the serving sources the proposer reads, gen 0's candidate tree and the
proposer setup. The study runs one level-bench trial (the loop) for the leg.
"""
import argparse
import hashlib
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOOP = ROOT / "src/rsi2_loop"
NOT_SERVING = {"arcade.py", "objects.py", "perception.py", "plans.py", "segment_worker.py", "session.py",
               "transition.py"}
PATCHES = {
    "src/arena/level_bench.py": (
        "async def run_level_bench(study, trial, **kwargs):\n",
        "async def run_level_bench(study, trial, **kwargs):\n"
        "    if study.get(\"rsi2_loop\"):  # the RSI loop runs inside the level-bench trial (src/rsi2_loop/loop.py)\n"
        "        from rsi2_loop.loop import run_loop\n\n"
        "        return await run_loop(study, trial, **kwargs)\n"
        "    if study.get(\"rsi2_worker\"):  # its evaluation worker VM (src/rsi2_loop/worker.py)\n"
        "        from rsi2_loop.worker import run_worker\n\n"
        "        return await run_worker(study, trial, **kwargs)\n"),
    "src/arena/competition.py": (
        "        else:\n            server = CompetitionArcadeServer(",
        "        elif os.environ.get('RSI2_GATEWAY_URL'):\n"
        "            # RSI loop evaluation: the loop's root gateway serves the games (rsi2_loop.gateway)\n"
        "            arcade = gateway_arcade(os.environ['RSI2_GATEWAY_URL'], output / 'recordings')\n"
        "        else:\n            server = CompetitionArcadeServer("),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serving_closure(src):
    """arena modules the server imports (from arena.serving, transitively) + helper files read by name."""
    import ast
    import re

    seen, todo = set(), ["arena/serving.py"]
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        package = Path(rel).parent
        for node in ast.walk(ast.parse((src / rel).read_text())):
            names = []
            if isinstance(node, ast.ImportFrom) and node.level:
                base = package
                for _ in range(node.level - 1):
                    base = base.parent
                names = [base / node.module.replace(".", "/")] if node.module else [base / a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "arena":
                names = [Path(node.module.replace(".", "/"))]
            for name in names:
                for candidate in (f"{name}.py", f"{name}/__init__.py"):
                    if (src / candidate).exists():
                        todo.append(candidate)
    for rel in list(seen):
        for name in re.findall(r"with_name\(['\"]([\w.]+)['\"]\)", (src / rel).read_text()):
            if (src / Path(rel).parent / name).is_file():
                seen.add(str(Path(rel).parent / name))
    for rel in list(seen):  # files read by path: Path(__file__).resolve().parents[k] / "a" / "b.py"
        for up, tail in re.findall(r"parents\[(\d)\]((?:\s*/\s*['\"][\w.]+['\"])+)", (src / rel).read_text()):
            parts = re.findall(r"['\"]([\w.]+)['\"]", tail)
            base = Path(rel).parents[int(up)] if int(up) < len(Path(rel).parents) else Path(".")
            target = base.joinpath(*parts)
            if (src / target).is_file():
                seen.add(str(target))
    for rel in list(seen):  # helpers run by path (module / "worker.py"): a sub-package travels whole
        folder = Path(rel).parent
        if len(folder.parts) >= 2:
            seen.update(str(p.relative_to(src)) for p in (src / folder).glob("*.py"))
    for rel in list(seen):
        for parent in Path(rel).parents:
            if str(parent) != "." and (src / parent / "__init__.py").is_file():
                seen.add(str(parent / "__init__.py"))
    return sorted(r for r in seen if not r.startswith("arc_infra/"))  # arc_infra/ is copied whole


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--gen0", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--v27", required=True)
    ap.add_argument("--notebook", type=int, default=85800)
    ap.add_argument("--resume")
    ap.add_argument("--loop", default="{}")
    ap.add_argument("--split-peer", help="split evaluation: the other VM's run id")
    ap.add_argument("--role", choices=("loop", "worker"), default="loop")
    ap.add_argument("--frontier", action="store_true", help="frontier-level evaluation first")
    ap.add_argument("--extras", action="store_true",
                    help="(1a) proposer-continue, (3a) confirmation runs, the continuous-signal decision rule")
    a = ap.parse_args()
    out, gen0, v27 = Path(a.out), Path(a.gen0), Path(a.v27) / "rsi"
    if out.exists():
        raise SystemExit(f"{out} exists")
    pool = json.loads(Path(a.pool).read_text())["pool"]
    shutil.copytree(gen0, out, ignore=shutil.ignore_patterns("__pycache__", "MANIFEST.json"))
    study = json.loads((out / "configs/comparison.json").read_text())
    games = out / study["generated_games"]["dir"]
    for folder in list(games.iterdir()):  # the pool only
        if not any(json.loads(m.read_text())["game_id"] in pool for m in folder.glob("*/metadata.json")):
            shutil.rmtree(folder)
    found = sorted(json.loads(m.read_text())["game_id"] for m in games.glob("*/*/metadata.json"))
    if found != sorted(pool):
        raise SystemExit(f"pool games not matched: {sorted(set(pool) ^ set(found))[:6]}")
    # gen 0's candidate tree (before the loop code and patches go into the bundle's src)
    tree = out / "rsi/gen0/tree"
    shutil.copytree(out / "src", tree / "player/src", ignore=shutil.ignore_patterns("__pycache__", "rsi_loop",
                                                                                    "rsi2_loop"))
    (tree / "player/configs").mkdir(parents=True)
    shutil.copy2(out / "configs/duck_controls.yaml", tree / "player/configs/duck_controls.yaml")
    from rsi2_loop.evaluate import FIXED_KEYS
    player = {k: v for k, v in study.items() if k not in FIXED_KEYS and k not in ("rsi2_loop",)}
    (tree / "player/study.json").write_text(json.dumps(player, indent=1) + "\n")
    server = json.loads(json.dumps(study["server"]))
    serving = tree / "serving"
    (serving / "expert-prune").mkdir(parents=True)
    mask = Path(server["expert_prune"]["mask"])
    shutil.copy2(out / mask, serving / "expert-prune" / mask.name)
    server["expert_prune"]["mask"] = f"expert-prune/{mask.name}"
    head = Path(server["draft_head"]["overlay"]["path"])
    shutil.copytree(out / head, serving / "draft-head/overlay")
    server["draft_head"]["overlay"]["path"] = "draft-head/overlay"
    (serving / "server.json").write_text(json.dumps(server, indent=1) + "\n")
    eval_trial = next(t for t in study["trials"] if t["name"] == "duck-transfer")
    (serving / "env.json").write_text(json.dumps(eval_trial.get("server_env") or {}, indent=1) + "\n")
    (tree / "notes.md").write_text(
        "Gen 0: our competition build v8.5 (Duck-transfer arena harness, 48 streams, int4 KV) on Flash, with every\n"
        "fit refitted on the 10 fixed public games: the t37 expert-prune mask, the INT3 requant overlay, the MTP draft\n"
        "head (serving/draft-head) and the 16K draft vocabulary (player/src/arc_infra/serving_draft_vocab.json).\n"
        "Serving: serving/server.json (paths relative to serving/), serving/env.json. Harness: player/.\n")
    # the fixed side's patches and the loop code
    for rel, (old, new) in PATCHES.items():
        text = (out / rel).read_text()
        if text.count(old) != 1:
            raise SystemExit(f"patch anchor not unique in {rel}")
        (out / rel).write_text(text.replace(old, new))
    for rel in ("arena/competition.py",):  # the candidate's fixed files are the bundle's (patched)
        shutil.copy2(out / "src" / rel, tree / "player/src" / rel)
    shutil.copytree(LOOP, out / "src/rsi2_loop", ignore=shutil.ignore_patterns("__pycache__", "gen0"))
    rsi = out / "rsi"
    shutil.copytree(v27 / "runtime", rsi / "runtime")
    (rsi / "proposer").mkdir()
    shutil.copy2(v27 / "gen0/prime-agent.tar.gz", rsi / "proposer/prime-agent.tar.gz")
    shutil.copytree(LOOP / "gen0/proposer", rsi / "proposer/meta", ignore=shutil.ignore_patterns("__pycache__"))
    extras = {}
    if a.extras:  # (1a) + (3a) + the continuous signal; no operator notes for the proposer
        extras = {"proposer_continue": True, "confirm_accepts": True, "continuous": True}
        agents = rsi / "proposer/meta/AGENTS.md"
        text = agents.read_text()
        old = text[text.index("  with gen 0's"):text.index("- You see per-game scores")]
        text = text.replace(old, (
            "  with gen 0's (`/workspace/archive/c0000/scores.json`). Two numbers per game: `score` (the scorecard) and\n"
            "  `cont`, the speed of the first two level clears (100 x mean over clears 1 and 2 of 1 - T / mark, 0 if not\n"
            "  cleared by the mark). They are combined into one z: z = (d_score / SE_score + d_cont / SE_cont) /\n"
            "  sqrt(2 + 2r), the SEs measured on repeated gen-0 runs. It stops at s2 if z <= 0 (s1 is shown, never\n"
            "  cuts); at s3 it passes if z >= 2 and both differences are positive. A pass is final only after a\n"
            "  confirmation: the candidate plays the pool again and the mean of its two runs must pass the same rule.\n"))
        text = text.replace("A pool evaluation holds the GPU for up to ~2.2 h,",
                            "A pool evaluation holds the GPU for up to ~2.2 h (a confirmation another 2.2 h),")
        agents.write_text(text)
    if a.frontier:
        extras["frontier"] = True
        agents = rsi / "proposer/meta/AGENTS.md"
        text = agents.read_text()
        anchor = "- You see per-game scores"
        text = text.replace(anchor, (
            "- Frontier levels first: a submitted candidate first plays the incumbent's frontier levels (levels the\n"
            "  incumbent clears in about 20-80% of attempts, `rsi.status()[\"frontier\"]`), each started from the board\n"
            "  where the incumbent reached it, several attempts per level at once (~75 min). Per attempt: cleared, and how\n"
            "  fast. It goes on to the full pool only if the combined z of the two differences against the incumbent's\n"
            "  reference attempts is >= 2 with both differences positive. The incumbent is gen 0 until a candidate passes\n"
            "  the full pool and its confirmation; then that candidate is the reference for later candidates.\n" + anchor), 1)
        agents.write_text(text)
    shutil.copytree(v27 / "games/practice", rsi / "games/practice")
    view = rsi / "serving-src"
    src = out / "src"
    for path in sorted((src / "arc_infra").glob("*")):
        if path.is_file() and path.name not in NOT_SERVING:
            (view / "arc_infra").mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, view / "arc_infra" / path.name)
    for rel in serving_closure(src):
        (view / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, view / rel)
    check = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
                            "from arena.serving_core import segmentation, serving, worker, sandbox; segmentation.source()", str(view)],
                           capture_output=True, text=True)
    if check.returncode:  # the unprivileged serving stack starts from this view (l8w: a missing helper killed it)
        raise SystemExit(f"serving-src view incomplete: {check.stderr[-800:]}")
    (view / "rsi_boot").mkdir()
    shutil.copy2(LOOP / "overlay_boot.py", view / "rsi_boot/sitecustomize.py")
    if a.resume:
        shutil.copytree(a.resume, out / "configs/rsi2_resume")
    # the study: one level-bench trial (the loop) for the leg; the evaluation trial is kept for compose()
    loop_trial = {"name": "rsi2-loop", "kind": "level-bench", "competition": False,
                  "server_env": eval_trial.get("server_env") or {}}
    study["trials"] = [loop_trial]
    study["rsi2_loop"] = {"version": 1, "pool": pool, "eval_trial": eval_trial, **extras, **json.loads(a.loop)}
    if a.split_peer:
        study["nebius_exchange_in"] = [f"{a.split_peer}/rsi-exchange/out"]
        if a.role == "worker":
            study["rsi2_worker"] = {"peer": a.split_peer, "pool": pool}
            study["rsi2_worker_of"] = study.pop("rsi2_loop")  # kept for compose() (eval_trial); no loop here
        else:
            study["rsi2_loop"]["split_peer"] = a.split_peer
    study["generated_games"]["games"] = pool
    study["games"] = pool
    study.update(name=out.name, status="prepared-not-launched", notebook_seconds=a.notebook,
                 wall_seconds=a.notebook - 3600)
    study["rsi2"] = {**study.get("rsi2", {}), "role": "loop", "games": len(pool)}
    (out / "configs/comparison.json").write_text(json.dumps(study, indent=2) + "\n")
    for cache in list(out.rglob("__pycache__")):
        shutil.rmtree(cache)
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
    from build_arena import manifest

    identity = json.loads((gen0 / "MANIFEST.json").read_text())
    identity["files"] = manifest(out)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    identity.update(study=out.name, git_commit=commit)
    (out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    runpy.run_path(str(out / "bootstrap.py"))["verify"](out)
    print(json.dumps({"bundle": str(out), "pool": len(pool), "practice": len(list((rsi / "games/practice").glob(
        "environment_files/*"))), "notebook": a.notebook, "manifest_sha256": sha(out / "MANIFEST.json")}))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
