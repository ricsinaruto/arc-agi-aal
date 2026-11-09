"""Game registry (docs/rsi.md section 2): the selection pool and the Witness practice remainder as native game folders,
deduplicated by full game_id, with reference action counts and the vet.

    python -I -m rsi_loop.registry build --felix=<clone> --witness=<clone> --out=<dir> [--jobs=8]

Each source clone is untrusted: it is only read here; every game module runs in ``vet.py`` under ``sandbox-exec``
(no network) with CPU and memory limits, cwd an empty temp dir. Output: ``<out>/environment_files/<id>/<ver>/`` (one
self-contained module per game: the NVIDIA adapter and Witness's grid helper and levels are inlined) and
``<out>/registry.json``. Reference counts: the recorded solve of each level (Felix561's per-segment solved attempt;
Witness's ``solution_actions``); the recordings are used for nothing else (no-frontier rule: never training data)."""

import argparse
import ast
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PUBLIC = ("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 "
          "tr87 tu93 vc33 wa30").split()
WITNESS_POOL_LEVELS = 8      # plan: solver-verified levels only, the first ~8 per game
WITNESS_PRACTICE_LEVELS = 8  # the next verified levels form one practice game per family (wpNN)
SANDBOX = "(version 1)(allow default)(deny network*)"
SRC = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def action_text(action):
    raw, params = int(action["provider_raw_action"]), action.get("parameters") or {}
    if raw == 6:
        return f"ACTION6@{int(params['column'])},{int(params['row'])}"
    if raw == 0:
        return "RESET"
    return f"ACTION{raw}"


def felix_traces(repo, part):
    """``{game_id: [[action text, ...] per level]}`` and ``{game_id: [reference count per level]}`` from the recorded
    segments: attempts in order, a RESET after a reset-interrupted one; the reference is the solved attempt's count."""
    segments = {}
    for line in gzip.open(repo / f"trajectories/{part}/trajectories.jsonl.gz", "rt"):
        row = json.loads(line)
        actions = row["actions"] if isinstance(row["actions"], list) else ast.literal_eval(row["actions"])
        key = (row["game_id"], int(row["level_id"]))
        attempt = int(str(row["trajectory_id"]).rsplit("attempt-", 1)[1])
        segments.setdefault(key, []).append((attempt, row["terminal_reason"], [action_text(a) for a in actions]))
    traces, refs = {}, {}
    for (game, level), items in sorted(segments.items()):
        flat, ref = [], None
        for _, reason, actions in sorted(items):
            flat += actions
            if reason == "native_level_reset":
                flat.append("RESET")
            else:
                ref = len([a for a in actions if a != "RESET"])
        traces.setdefault(game, []).append(flat)
        refs.setdefault(game, []).append(ref)
    return traces, refs


ADAPTER_IMPORT = re.compile(r"from arc_agi_3\.game_creator\.arcengine_adapter import \(\s*FunctionalArcGame as "
                            r"_FunctionalArcGame,\s*\)")


def inline_nvidia(text, adapter):
    adapter = adapter.replace("from __future__ import annotations\n", "")
    text, count = ADAPTER_IMPORT.subn(lambda _: adapter + "\n_FunctionalArcGame = FunctionalArcGame\n", text)
    if count != 1:
        raise ValueError("NVIDIA adapter import not found exactly once")
    return text


WITNESS_IMPORT = re.compile(r"from witness_grid import \([^)]*\)\n")
WITNESS_OPEN = re.compile(r"levels_path = os\.path\.join\(_code_dir, \"levels\", \"tw\d\d_levels\.json\"\)\n"
                          r"(\s+)if os\.path\.exists\(levels_path\):\n\s+with open\(levels_path\) as f:\n"
                          r"\s+data = json\.load\(f\)\n")


def witness_module(text, grid, entries, class_name, new_class):
    """A self-contained Witness module: grid helper inlined, ``entries`` embedded as the level list, class renamed."""
    a = 1  # the sys.path block stays: its directory holds no helper once the grid is inlined
    text, b = WITNESS_IMPORT.subn("", text)
    text, c = WITNESS_OPEN.subn(lambda m: f"if True:\n{m.group(1)}    data = _EMBEDDED_LEVELS\n", text)
    if (a, b, c) != (1, 1, 1):
        raise ValueError(f"Witness patch points: {(a, b, c)}")
    grid = "\n".join(line for line in grid.splitlines() if not line.startswith(("from typing", "import numpy")))
    head = f"import numpy as np\nfrom typing import List, Tuple, Optional, Set, Dict\n{grid}\n"
    embedded = f"\n_EMBEDDED_LEVELS = {json.dumps({'levels': entries})!r}\n_EMBEDDED_LEVELS = __import__('json').loads(_EMBEDDED_LEVELS)\n"
    body = text.replace(f"class {class_name}(", f"class {new_class}(")
    # module docstring first, then the inlined helper
    return head + embedded + body


def write_game(out, game_id, module_name, text, meta):
    prefix, version = game_id.split("-", 1)
    folder = out / "environment_files" / prefix / version
    folder.mkdir(parents=True, exist_ok=False)
    (folder / f"{module_name}.py").write_text(text)
    (folder / "metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    return folder


def build(felix, witness, out, jobs=8):
    felix, witness, out = Path(felix), Path(witness), Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "traces").mkdir(parents=True)
    entries, seen = [], set()
    catalog = {g["game_id"]: g for g in json.loads((felix / "catalog.json").read_text())["games"]}
    adapter = (felix / "third_party/nvidia/runtime_support/arc_agi_3/game_creator/arcengine_adapter.py").read_text()
    felix_commit = subprocess.run(["git", "-C", str(felix), "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip()
    witness_commit = subprocess.run(["git", "-C", str(witness), "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip()

    def add(entry, text, meta, module_name, trace):
        if entry["game_id"] in seen:
            entry["excluded"] = "duplicate game_id"
            entries.append(entry)
            return
        seen.add(entry["game_id"])
        folder = write_game(out, entry["game_id"], module_name, text, meta)
        entry["dir"] = str(folder.relative_to(out))
        entry["files_sha256"] = {p.name: sha(p) for p in sorted(folder.iterdir())}
        (out / "traces" / f"{entry['game_id']}.json").write_text(json.dumps(trace))
        entries.append(entry)

    for part, base, licence in (("studio", felix / "environment_files", "MIT"),
                                ("studio_v2", felix / "environment_files", "MIT"),
                                ("nvidia", felix / "third_party/nvidia/environment_files", "Apache-2.0")):
        traces, refs = felix_traces(felix, part)
        for game_id in sorted(traces):
            prefix, version = game_id.split("-", 1)
            folder = base / prefix / version
            meta = json.loads((folder / "metadata.json").read_text())
            module = next(p for p in folder.glob("*.py"))
            text = module.read_text()
            if part == "nvidia":
                text = inline_nvidia(text, adapter)
            if any(r is None for r in refs[game_id]):
                raise ValueError(f"{game_id}: a level without a solved segment")
            source_baseline = meta.get("baseline_actions")
            meta = {"game_id": game_id, "title": meta.get("title", prefix.upper()),
                    "default_fps": meta.get("default_fps", 8), "tags": meta.get("tags", []),
                    "baseline_actions": refs[game_id], "class_name": meta.get("class_name")}
            entry = {"game_id": game_id, "role": "pool", "source": f"Felix561/arc3-synthetic-games:{part}",
                     "commit": felix_commit, "licence": licence, "total_levels": len(refs[game_id]),
                     "reference_actions": refs[game_id], "source_baseline_actions": source_baseline,
                     "mechanics_text_withheld": bool(catalog.get(game_id, {}).get("mechanics"))}
            add(entry, text, meta, module.stem, traces[game_id])

    grid = (witness / "witness_grid.py").read_text()
    for folder in sorted((witness / "environment_files").glob("tw*")):
        prefix = folder.name
        meta = json.loads((folder / "metadata.json").read_text())
        levels = json.loads((witness / "levels" / f"{prefix}_levels.json").read_text())["levels"]
        verified = [lv for lv in levels if lv.get("validated") and lv.get("solution_actions")]
        text = (folder / f"{prefix}.py").read_text()
        class_name = meta["class_name"]
        for role, chosen, game_prefix in (
                ("pool", verified[:WITNESS_POOL_LEVELS], prefix),
                ("practice", verified[WITNESS_POOL_LEVELS:WITNESS_POOL_LEVELS + WITNESS_PRACTICE_LEVELS],
                 "wp" + prefix[2:])):
            if not chosen:
                entries.append({"game_id": game_prefix, "role": role, "source": "hoteizero/arc-witness-envs",
                                "excluded": "no solver-verified levels"})
                continue
            payload = [{"config": lv["config"], "validated": True} for lv in chosen]
            new_class = game_prefix[0].upper() + game_prefix[1:]
            module = witness_module(text, grid, payload, class_name, new_class)
            version = hashlib.sha256(module.encode()).hexdigest()[:16]
            game_id = f"{game_prefix}-{version}"
            refs = [len(lv["solution_actions"]) for lv in chosen]
            trace = [[f"ACTION{int(a)}" for a in lv["solution_actions"]] for lv in chosen]
            meta_out = {"game_id": game_id, "title": game_prefix.upper(), "default_fps": meta.get("default_fps", 8),
                        "tags": meta.get("tags", []), "baseline_actions": refs, "class_name": new_class}
            entry = {"game_id": game_id, "role": role, "source": "hoteizero/arc-witness-envs", "commit": witness_commit,
                     "licence": "MIT (README; no LICENSE file)", "total_levels": len(chosen),
                     "reference_actions": refs, "source_baseline_actions": [lv.get("baseline") for lv in chosen],
                     "witness_levels": [lv.get("level_index") for lv in chosen]}
            add(entry, module, meta_out, game_prefix, trace)

    prefixes = {}
    for e in entries:
        if "excluded" not in e:
            prefixes.setdefault(e["game_id"][:4], []).append(e["game_id"])
    clash = {k: v for k, v in prefixes.items() if len(v) > 1 or k in PUBLIC}
    if clash:
        raise ValueError(f"4-character prefixes clash (the arena keys some records by them): {clash}")

    def vet(entry):
        if "excluded" in entry:
            return entry
        env_dir = out / "environment_files" / entry["game_id"].split("-", 1)[0]
        with tempfile.TemporaryDirectory() as cwd:
            command = ["sandbox-exec", "-p", SANDBOX, sys.executable, "-I", str(SRC / "rsi_loop/vet.py"),
                       str(env_dir), entry["game_id"], str(out / "traces" / f"{entry['game_id']}.json")]
            limits = "import resource,os,sys;resource.setrlimit(resource.RLIMIT_CPU,(600,600));os.execvp(sys.argv[1],sys.argv[1:])"
            try:
                done = subprocess.run([sys.executable, "-I", "-c", limits, *command], cwd=cwd, capture_output=True,
                                      text=True, timeout=900,
                                      env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": cwd})
                lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
                result = json.loads(lines[-1]) if lines else {"ok": False, "error": done.stderr[-800:]}
            except subprocess.TimeoutExpired:
                result = {"ok": False, "error": "timeout"}
        entry["vet"] = result
        if result.get("ok"):
            measured = [lv["actions"] for lv in result["levels"]]
            entry["vet_ok"] = True
            entry["board_hash"] = result["board_hash"]
            if entry["role"] == "pool" and measured != [
                    sum(1 for a in t if a != "RESET") for t in json.loads((out / "traces" / f"{entry['game_id']}.json").read_text())]:
                entry["vet_ok"] = False
        else:
            entry["vet_ok"] = False
        return entry

    with ThreadPoolExecutor(jobs) as pool:
        entries = list(pool.map(vet, entries))
    registry = {"schema": 1, "sources": {"felix": felix_commit, "witness": witness_commit},
                "rules": {"witness_pool_levels": WITNESS_POOL_LEVELS, "witness_practice_levels": WITNESS_PRACTICE_LEVELS,
                          "reference": "recorded solved attempt per level (Felix561) / solution_actions (Witness)",
                          "excluded_everywhere": "anything derived from the public 25"},
                "games": entries}
    (out / "registry.json").write_text(json.dumps(registry, indent=1) + "\n")
    ok = [e for e in entries if e.get("vet_ok")]
    print(json.dumps({"games": len(entries), "vet_ok": len(ok),
                      "pool": sum(e["role"] == "pool" for e in ok), "practice": sum(e["role"] == "practice" for e in ok),
                      "failed": [(e["game_id"], (e.get("vet") or {}).get("problems") or (e.get("vet") or {}).get("error", "")[-200:]
                                  or e.get("excluded")) for e in entries if not e.get("vet_ok")],
                      "trivial": [e["game_id"] for e in ok if e["vet"].get("trivial")]}, indent=1))
    return registry


def points(registry, role, split=None):
    """Play-through start points (``arena.level_bench``) for the vetted games of ``role``."""
    out = []
    for e in registry["games"]:
        if e.get("vet_ok") and e["role"] == role:
            out.append({"id": f"{e['game_id']}-game", "game": e["game_id"], "split": split or role, "level_index": 0,
                        "level": 1, "through": True, "total_levels": e["total_levels"],
                        "baselines": e["reference_actions"], "baseline_actions": None, "prefix_actions": 0,
                        "board_hash": e["board_hash"], "prefix": []})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--felix", required=True)
    parser.add_argument("--witness", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    build(args.felix, args.witness, args.out, args.jobs)


if __name__ == "__main__":
    main()


POOL_RULES = ("level1", "level2")


def trivial(vet, rule="level1"):
    """A game is trivial when uniform random play (``vet.py``: 3 seeds x 300 actions) clears, in any seed, level 1
    (``level1``) or two levels (``level2``, the third audit's proposal: needs the re-vet's
    ``random_level2_wins``)."""
    vet = vet or {}
    if rule == "level1":
        return int(vet.get("random_level1_wins") or 0) >= 1
    if rule == "level2":
        if "random_level2_wins" not in vet:
            raise ValueError("pool rule level2 needs the re-vet (random_level2_wins)")
        return int(vet["random_level2_wins"]) >= 1
    raise ValueError(f"pool rule: {POOL_RULES}")


def trivial_level1(entry):
    return trivial(entry.get("vet"), "level1")


WITNESS_MODES = ("practice", "split", "drop")


def roles(registry, rule="level2", witness="practice"):
    """Pool and practice: games whose random play is trivial under ``rule`` leave the pool. Witness (one tw engine,
    ``twNN`` games and their later-level ``wpNN`` games): ``practice`` (every Witness game is practice,
    none in the pool), ``split`` (non-trivial ``twNN`` in the pool, a family's games practice only when ``twNN`` is out
    of it) or ``drop`` (split, minus Witness practice)."""
    if witness not in WITNESS_MODES:
        raise ValueError(f"witness: {WITNESS_MODES}")
    vetted = [e for e in registry["games"] if e.get("vet_ok")]
    witness_game = lambda e: e["game_id"][:2] in ("tw", "wp")  # noqa: E731
    pool = [e for e in vetted if e["role"] == "pool" and not trivial(e.get("vet"), rule)
            and not (witness == "practice" and witness_game(e))]
    in_pool = {e["game_id"][:4] for e in pool}
    practice = []
    for e in vetted:
        family = e["game_id"][:4]
        if e["role"] == "practice" and "tw" + family[2:] not in in_pool:
            practice.append(e)
        elif e["role"] == "pool" and family.startswith("tw") and family not in in_pool:
            practice.append(e)
    if witness == "drop":
        practice = [e for e in practice if not witness_game(e)]
    dropped = [e["game_id"] for e in vetted if e["role"] == "pool" and trivial(e.get("vet"), rule)]
    return pool, practice, dropped
