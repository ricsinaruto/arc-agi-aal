"""Bundle of the RSI loop (docs/rsi.md as written, rebuild of 12 Oct).

    .venv/bin/python src/rsi_loop/build.py dist/<name> --games=<registry out> --draft-head-overlay=DIR \
        --prime-agent=<prime-agent-v0.3.3-linux-x64.tar.zst> --wheelhouse=<dir> --node=<node-v22.19.0-linux-x64.tar.xz> \
        --practice-vet=<practice_vet.json> --practice-src=<arc-interactive clone> --practice-exclude=wm01,wm02,wm03,wm04 \
        [--practice-vet2=<vgdl vet.json> --practice-src2=<vgdl port dir>] \
        --heldout-b=<game folder> --heldout-b-levels=<probe out.jsonl> [--wall=N] [--setup=N] [--resume=<pulled rsi dir>]
        [--loop=<rsi_loop JSON overrides>] [--exchange-in=<run id>,...]

Base: the Prime arm's bench build (``build_hx4.py --serving=v82``) for the runtime, the launcher's level-bench trial and
the bootstrap; then the serving block is replaced by gen 0's (``gen0/serving``: stock NVFP4 experts, stock MTP head,
no draft vocabulary; the v8.2 masks, int3 overlay, tuned draft head and arena16k vocabulary are all fitted on public
games and are removed from the bundle), and the loop's own folder ``rsi/`` is added:

    rsi/runtime/       node tarball, the kernel wheelhouse
    rsi/gen0/          gen 0's candidate (stock Prime Agent v0.3.3 build + the stock ARC runner files), proposer
                       setup (meta), proposer docs
    rsi/games/         pool, practice, heldout_a (public 25), heldout_b (sonpham arena 50)
    rsi/serving-src/   the serving sources the proposer reads (our serving patches; vLLM comes from the install)

The ``--draft-head-overlay`` only satisfies the v8.2 base builder; it is deleted from the bundle.
"""
import hashlib
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
PUBLIC_DIR = next(p / "environment_files" for p in (ROOT, *ROOT.parents) if (p / "environment_files").is_dir())
GEN0 = ROOT / "src/rsi_loop/gen0"
SERVED = "Qwen/Qwen3.8-Flash-Next-NVFP4"
NOT_SERVING = {"arcade.py", "objects.py", "perception.py", "plans.py", "segment_worker.py", "session.py",
               "transition.py", "serving_draft_vocab.json"}


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


def point(game, role, baselines, total_levels, source):
    return {"id": f"{game}-game", "game": game, "split": role, "level_index": 0, "level": 1, "through": True,
            "total_levels": total_levels, "baselines": baselines, "source": source}


def serving_closure():
    """arena modules the server imports (relative and absolute imports from arena.serving, transitively)."""
    import ast

    src, seen, todo = ROOT / "src", set(), ["arena/serving.py"]
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
    # helper files the patches read by name next to themselves (Path(__file__).with_name("x_runtime.py")): v14's
    # unprivileged stack died on arena/cache_export_runtime.py missing from serving-src (13 Oct)
    import re

    for rel in list(seen):
        for name in re.findall(r"with_name\(['\"]([\w.]+)['\"]\)", (src / rel).read_text()):
            if (src / Path(rel).parent / name).is_file():
                seen.add(str(Path(rel).parent / name))
    for rel in list(seen):  # regular packages: a namespace package loses to any other ``arena`` on sys.path
        for parent in Path(rel).parents:
            if str(parent) != "." and (src / parent / "__init__.py").is_file():
                seen.add(str(parent / "__init__.py"))
    return sorted(r for r in seen if r.startswith("arena/"))


def copy_game(folder, target):
    shutil.copytree(folder, target / folder.parent.name / folder.name, ignore=shutil.ignore_patterns("__pycache__"))


def main():
    out = Path(sys.argv[1])
    games = Path(arg("games"))
    registry = json.loads((games / "registry.json").read_text())
    sys.path[:0] = [str(ROOT / "src")]
    from rsi_loop.registry import roles

    # --pool-rule=level2 (needs --revet: the re-vet's random_levels merged into each entry's vet),
    # --witness=practice (every Witness game practice); level1 / split / drop stay as switches.
    if arg("revet"):
        fresh = {r["game_id"]: r for r in (json.loads(line) for line in Path(arg("revet")).read_text().splitlines()
                                             if line.strip())}
        for entry in registry["games"]:
            got = fresh.get(entry["game_id"]) or {}
            if "random_levels" in got:
                entry["vet"] = {**(entry.get("vet") or {}), **{k: got[k] for k in (
                    "random_levels", "random_level1_wins", "random_level2_wins")}}
    pool_rule, witness = arg("pool-rule", "level2"), arg("witness", "practice")
    pool_entries, practice_entries, dropped = roles(registry, pool_rule, witness)
    pool = [point(e["game_id"], "pool", e["reference_actions"], e["total_levels"], e["source"]) for e in pool_entries]
    practice = [point(e["game_id"], "practice", e["reference_actions"], e["total_levels"], e["source"])
                for e in practice_entries]
    scratch = out.parent / f".{out.name}-points.json"
    scratch.write_text(json.dumps({"schema": 1, "split": "pool", "points": [
        {**p, "baseline_actions": None, "prefix_actions": 0, "board_hash": e["board_hash"], "prefix": []}
        for p, e in zip(pool, pool_entries)]}))
    wall, setup = int(arg("wall", "84000")), int(arg("setup", "1400"))
    own = sys.argv[:]
    sys.argv = ["build_hx4.py", str(out), f"--points={scratch}", "--budget=1000000000", "--passes=1",
                "--concurrency=13", f"--wall={wall}", f"--setup={setup}", "--serving=v82", "--split=all",
                f"--draft-head-overlay={arg('draft-head-overlay')}", "--window=130816", "--streams=48",
                "--level-cap=300000", "--admission-tick=2.0"]
    try:
        runpy.run_path(str(ROOT / "docs/validation-20261007/build_hx4.py"), run_name="__main__")
    finally:
        sys.argv = own
        scratch.unlink()
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
    from build_arena import manifest

    # public-game fitted artifacts out of the bundle (and the base build's Kaggle upload, a tarball made before this
    # removal: it still holds them)
    removed = []
    for rel in ("draft-head", "configs/expert-prune", "src/arc_infra/serving_draft_vocab.json", "dataset-upload"):
        path = out / rel
        if path.exists():
            shutil.rmtree(path) if path.is_dir() else path.unlink()
            removed.append(rel)
    shutil.copytree(ROOT / "src/rsi_loop", out / "src/rsi_loop", ignore=shutil.ignore_patterns("__pycache__", "gen0"),
                    dirs_exist_ok=True)

    rsi = out / "rsi"
    (rsi / "runtime").mkdir(parents=True)
    shutil.copy2(arg("node"), rsi / "runtime" / Path(arg("node")).name)
    shutil.copytree(arg("wheelhouse"), rsi / "runtime/wheelhouse")
    # gen 0: the stock ARC runner files + Prime Agent v0.3.3 (re-packed as tar.gz: the VM's Python reads it)
    shutil.copytree(GEN0 / "player", rsi / "gen0/tree/player")
    shutil.copytree(GEN0 / "serving", rsi / "gen0/tree/serving")
    (rsi / "gen0/tree/notes.md").write_text(
        "Gen 0: stock Prime Agent v0.3.3 (PrimeIntellect-ai/prime-agent bc6894c8, MIT) with the stock ARC-AGI-3 runner\n"
        "files (PrimeIntellect-ai/arc-agi-3-prime-agent 398d4dd6, MIT: AGENTS.md, game prompt, broker client, guard),\n"
        "on Flash (stock NVFP4 checkpoint, stock MTP head). Thinking xhigh, as in the published scored runs; sub-agents\n"
        "off (the stock game prompt forbids them). Autonomous limits lifted (run.json --autonomous-max-*\n"
        "flags), so only the game's time slice or token cap ends it. Serving: serving/server.json.\n")
    staging = out.parent / f".{out.name}-prime"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    subprocess.run(["tar", "--zstd", "-xf", arg("prime-agent"), "-C", str(staging)], check=True)
    subprocess.run(["tar", "-czf", str(rsi / "gen0/prime-agent.tar.gz"), "-C", str(staging / "prime-agent"), "."],
                   check=True)
    shutil.rmtree(staging)
    shutil.copytree(GEN0 / "proposer", rsi / "gen0/meta", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "src/rsi_loop/policy.py", rsi / "gen0/meta/policy.py")
    shutil.copytree(GEN0 / "docs", rsi / "gen0/docs")
    # games
    target = {role: rsi / "games" / role / "environment_files" for role in ("pool", "practice", "heldout_a",
                                                                             "heldout_b")}
    for role, entries in (("pool", pool_entries), ("practice", practice_entries)):
        for e in entries:
            copy_game(games / e["dir"], target[role])
    practice_dropped = {"short": [], "unsolved": []}
    dropped_names = set(filter(None, arg("practice-exclude", "").split(",")))
    # copies of any public game (fixed pool or held-out A) never become practice: their traces would show them
    dropped_names |= {m.parent.parent.name[:4] for m in PUBLIC_DIR.glob("*/*/metadata.json")}
    known = {p["game"][:4] for p in pool + practice}
    # --practice-vet/--practice-src, then --practice-vet2/--practice-src2 (the VGDL port); a vet's "source" labels it
    for vet_path, src in ((arg("practice-vet"), arg("practice-src")), (arg("practice-vet2"), arg("practice-src2"))):
        if not vet_path:
            continue
        vet = json.loads(Path(vet_path).read_text())
        for g in vet["games"]:
            if not g.get("passed") or g["game_id"][:4] in known or g["game_id"].split("-", 1)[0] in dropped_names:
                continue
            solved = sorted(g["levels"], key=lambda r: r["level"])
            solved = [lv for i, lv in enumerate(solved) if lv["level"] == i + 1]  # a verified prefix of levels
            if not solved:
                practice_dropped["unsolved"].append(g["game_id"])
                continue
            # near-trivial: every verified level solved in at most 5 actions (random play did not clear level 1
            # in the vet, the pool's stricter rule, but such levels teach nothing)
            if max(lv["n"] for lv in solved) <= 5:
                practice_dropped["short"].append(g["game_id"])
                continue
            folder = Path(src) / g["dir"]
            copy_game(folder, target["practice"])
            row = point(g["game_id"], "practice", [lv["n"] for lv in solved], len(solved),
                        vet.get("source", "theredbluepill/arc-interactive"))
            if len(solved) < int(g["total_levels"]):  # capped at its verified levels (the broker ends it there)
                row["level_cap"] = len(solved)
                row["levels_in_game"] = int(g["total_levels"])
            practice.append(row)
            known.add(g["game_id"][:4])
    heldout_a = []
    # the 10 easiest public games (v8.2 128K 24 h run, score at matched tokens) are fixed pool games,
    # held-out A is the other 15
    fixed = set(filter(None, arg("fixed-pool", "").split(",")))
    for meta in sorted(PUBLIC_DIR.glob("*/*/metadata.json")):
        data = json.loads(meta.read_text())
        role = "pool" if data["game_id"][:4] in fixed else "heldout_a"
        copy_game(meta.parent, target[role])
        row = point(data["game_id"], role, data["baseline_actions"], len(data["baseline_actions"]),
                    "ARC-AGI-3 public" + (" (fixed pool game)" if role == "pool" else ""))
        if role == "pool":
            row["fixed"] = True
            pool.append(row)
        else:
            heldout_a.append(row)
    if fixed and sum(1 for p in pool if p.get("fixed")) != len(fixed):
        raise SystemExit(f"fixed pool games not all found among the public games: {sorted(fixed)}")
    levels_b = {json.loads(line)["id"]: json.loads(line)["win_levels"]
                for line in Path(arg("heldout-b-levels")).read_text().splitlines() if line.strip()}
    heldout_b = []
    for meta in sorted(Path(arg("heldout-b")).glob("*/*/metadata.json")):
        data = json.loads(meta.read_text())
        copy_game(meta.parent, target["heldout_b"])
        prefix = data["game_id"].split("-", 1)[0]
        heldout_b.append(point(data["game_id"], "heldout_b", None, int(levels_b[prefix]), "sonpham-org/arc-3 arena"))
    every = [p["game"] for p in pool + practice + heldout_a + heldout_b]
    if len(set(every)) != len(every) or len({g[:4] for g in every}) != len(every):
        raise SystemExit("game ids (or their 4-character prefixes) are not unique across the sets")
    # serving sources for the proposer (our serving patches; no harness or game code)
    view = rsi / "serving-src"
    for path in sorted((ROOT / "src/arc_infra").glob("*")):
        if path.is_file() and path.name not in NOT_SERVING:
            (view / "arc_infra").mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, view / "arc_infra" / path.name)
    for rel in serving_closure():
        (view / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "src" / rel, view / rel)
    import re

    missing = [f"{f.relative_to(view)} -> {name}" for f in view.rglob("*.py")
               for name in re.findall(r"Path\(__file__\)\.with_name\(['\"]([\w]+\.py)['\"]\)", f.read_text())
               if not (f.parent / name).is_file()]
    if missing:  # a stack served from serving-src would die reading them
        raise SystemExit(f"serving-src lacks helper files: {missing[:5]}")
    (view / "rsi_boot").mkdir(parents=True, exist_ok=True)  # per-module overlay of a proposer-built stack
    shutil.copy2(ROOT / "src/rsi_loop/overlay_boot.py", view / "rsi_boot/sitecustomize.py")
    configs = out / "configs/rsi"
    configs.mkdir(parents=True)
    for name, rows in (("pool", pool), ("practice", practice), ("heldout_a", heldout_a), ("heldout_b", heldout_b)):
        (configs / f"{name}_points.json").write_text(json.dumps({"schema": 1, "split": name, "points": rows}) + "\n")
    (configs / "registry.json").write_text(json.dumps({**registry, "dropped_trivial": dropped, "pool_rule": pool_rule,
                                                         "practice_dropped": practice_dropped}, indent=1))
    if arg("resume"):
        shutil.copytree(arg("resume"), out / "configs/rsi_resume")
    path = out / "configs/comparison.json"
    study = json.loads(path.read_text())
    study["server"] = json.loads((GEN0 / "serving/server.json").read_text())
    study.update(json.loads((GEN0 / "serving/study.json").read_text()))
    # 13 Oct: v10 died at server start (server.max_num_seqs 25 < generation_slots 52); same rule as serving.fit_slots
    study["generation_slots"] = min(study["generation_slots"], study["server"]["max_num_seqs"])
    # 13 Oct: v12 died on the base build's ARENA_STALL_DUMP (it rides on server.expert_prune, which gen 0 drops)
    for trial in study.get("trials", []):
        env = trial.get("server_env") or {}
        for key in ("ARENA_STALL_DUMP", "ARENA_EXPERT_PRUNE", "ARENA_EXPERT_OVERLAY"):
            if not study["server"].get("expert_prune"):
                env.pop(key, None)
    study["served_model"] = SERVED
    study.pop("generated_games", None)
    # the base build's records of its own (public-fitted) serving: metadata only, dropped so nothing fetches them
    study["model_datasets"] = []
    for key in ("arena_variant", "hero_v2"):
        study.pop(key, None)
    if (out / "generated_games").exists():
        shutil.rmtree(out / "generated_games")
        removed.append("generated_games")
    study["rsi_loop"] = {"version": 5, "pool_rule": pool_rule, "witness": witness, **json.loads(arg("loop", "{}"))}
    operator = out / "configs/rsi_operator.jsonl"  # docs/rsi.md 7: human actions on the run (id, at, by, what, human_seconds)
    operator.write_text(Path(arg("operator-log")).read_text() if arg("operator-log") else "")
    if arg("exchange-in"):  # the runner copies these runs' rsi-exchange/out into exchange-in/ (loop <-> scoring VMs)
        study["nebius_exchange_in"] = [f"{run}/rsi-exchange/out" for run in arg("exchange-in").split(",")]
    study["name"] = out.name
    path.write_text(json.dumps(study, indent=2) + "\n")
    kernel = out / "kernel-metadata.json"  # Kaggle inputs: no public-fitted overlay dataset
    if kernel.is_file():
        meta = json.loads(kernel.read_text())
        meta["dataset_sources"] = [d for d in meta.get("dataset_sources", [])
                                   if not any(k in d for k in ("excl5", "requant", "gptq", "draft", "prune", "hero-v2-dfr"))]
        kernel.write_text(json.dumps(meta, indent=2) + "\n")
    # the bundle's serving studies through arena.serving's own checks (no GPU): a refused study fails the build
    subprocess.run([sys.executable, "-I", "-B", str(ROOT / "src/rsi_loop/serving_check.py"), str(out)], check=True)
    subprocess.run([sys.executable, "-I", "-B", str(ROOT / "src/rsi_loop/serving_check.py"), str(out), "--serving-src"],
                   check=True)
    for cache in list(out.rglob("__pycache__")):  # 13 Oct: v11 died on a .pyc the check wrote, hashed, never shipped
        shutil.rmtree(cache)
    identity = json.loads((out / "MANIFEST.json").read_text())
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "scripts", "configs"],
                           capture_output=True, text=True).stdout.strip()
    identity.update(source="git commit" if not dirty else "git commit with uncommitted changes", git_commit=commit,
                    git_dirty=bool(dirty))
    identity["files"] = manifest(out)
    unshipped = [n for n in identity["files"] if "__pycache__" in Path(n).parts or n.endswith(".ipynb")
                 or Path(n).parts[0] in ("dataset-upload", "source.zip")]
    if unshipped:  # the launcher's pack_bundle drops these, so the VM's verify would fail on them
        raise SystemExit(f"MANIFEST lists files the bundle upload drops: {unshipped[:5]}")
    (out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    runpy.run_path(str(out / "bootstrap.py"))["verify"](out)
    digest = hashlib.sha256((out / "MANIFEST.json").read_bytes()).hexdigest()
    print(json.dumps({"output": str(out), "manifest_sha256": digest, "pool": len(pool), "practice": len(practice),
                      "heldout_a": len(heldout_a), "heldout_b": len(heldout_b), "dropped_trivial": len(dropped),
                      "practice_capped": sum(1 for p in practice if p.get("level_cap")),
                      "practice_dropped": {k: len(v) for k, v in practice_dropped.items()},
                      "removed": removed, "wall": wall, "notebook_seconds": study["notebook_seconds"],
                      "resume": bool(arg("resume"))}))


if __name__ == "__main__":
    main()
