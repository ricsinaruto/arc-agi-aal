"""CPU gate for a bundled-games data run (30 Oct, v9b 24 h mixed roster): on a Modal CPU container (linux, py3.12), the
bundle's own agent venv (bootstrap.install_agent_environment: its locked wheels, mini-racer included), then the
harness's own game path: arena.competition.open_arcade(study) starts the clone server on the bundled games dir with
every slot of study.games, the one scorecard opens, and the first slot of every distinct game is made
(arc.make(clone, scorecard_id=card)) and stepped; one PuzzleScript game is stepped STEPS times plus a RESET.
Checks: every clone resolves to its roster game, every game returns frames, the PuzzleScript worker processes run.
Usage: SMOKE_BUNDLE=DIST_DIR modal run --detach games_smoke.py::main"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

app = modal.App("arc3sub-games-smoke", include_source=False)
BUNDLE = os.environ.get("SMOKE_BUNDLE")  # local only: the dist dir baked into the image
image = modal.Image.debian_slim(python_version="3.12")
if BUNDLE:
    image = image.add_local_dir(BUNDLE, f"/bundle/{Path(BUNDLE).name}", copy=True, ignore=["**/__pycache__/**"])
SMOKE = r'''
import json, os, sys, time, subprocess
from pathlib import Path
B = Path(sys.argv[1]); out = Path("/tmp/smoke-out"); out.mkdir(parents=True, exist_ok=True)
study = json.loads((B / "configs/comparison.json").read_text())
games_dir = str(B / study["generated_games"]["dir"])
from arena.competition import open_arcade
from arcengine import GameAction
slots = study["games"]; distinct = study["generated_games"]["distinct"]
res = {"slots": len(slots), "games": {}, "errors": []}
t0 = time.time()
with open_arcade(study, games_dir, out) as (resolved, arc):
    clones = resolved["games"]
    roster = json.loads((out / "public-roster.json").read_text())
    assert [r["source_game_id"] for r in roster] == slots, "clone -> roster game mapping differs"
    card = arc.open_scorecard(tags=["games-smoke"])
    first = {}
    for i, g in enumerate(slots):
        first.setdefault(g, i)
    ps_done = False
    for g, i in first.items():
        row = {"source": distinct[g], "clone": clones[i]}
        try:
            env = arc.make(clones[i], scorecard_id=card, save_recording=False)
            obs = env.observation_space
            row["levels"] = obs.win_levels; row["state"] = obs.state.name
            acts = [a for a in (obs.available_actions or []) if a != 0]
            row["actions"] = acts
            n = 25 if (distinct[g] == "puzzlescript" and not ps_done) else 3
            frames, ts = 0, time.time()
            for k in range(n):
                a = acts[k % len(acts)] if acts else 1
                ga = GameAction.from_id(a)
                data = {"x": 32, "y": 32} if ga.is_complex() else {}
                r = env.step(ga, data=data)
                frames += bool(r is not None and r.frame)
            if n == 25:
                r = env.step(GameAction.RESET)
                frames += bool(r is not None and r.frame)
                ps_done = True
                row["ps_workers"] = int(subprocess.run(["sh", "-c", "ps -eo args | grep -c '[p]y_mini_racer' || true"],
                                                       capture_output=True, text=True).stdout.strip() or 0)
            row["steps"] = n + (1 if n == 25 else 0); row["frames"] = frames
            row["ms_per_step"] = round(1000 * (time.time() - ts) / row["steps"], 1)
            if frames != row["steps"]:
                res["errors"].append(f"{g}: {frames}/{row['steps']} frames")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            res["errors"].append(f"{g}: {row['error']}")
        res["games"][g] = row
    arc.close_scorecard(card)
res["seconds"] = round(time.time() - t0, 1)
res["ok"] = not res["errors"] and len(res["games"]) == len(distinct)
print("GAMES_SMOKE " + json.dumps(res))
'''


@app.function(image=image, cpu=(0.125, 8), timeout=2400, retries=0, max_containers=1, serialized=True)
def smoke(bundle_name: str):
    B = Path(f"/bundle/{bundle_name}")
    import shutil
    work = Path("/tmp/b") / bundle_name
    shutil.copytree(B, work)
    sys.path.insert(0, str(work))
    from bootstrap import install_agent_environment, verify
    verify(work)
    t = time.time()
    python = install_agent_environment(work, Path("/tmp/venv"), dict(os.environ))
    print(json.dumps({"venv_seconds": round(time.time() - t, 1)}), flush=True)
    chk = subprocess.run([python, "-c", "import py_mini_racer, importlib.metadata as m; print(m.version('mini-racer'))"],
                         capture_output=True, text=True)
    print("MINI_RACER", chk.returncode, chk.stdout.strip(), chk.stderr[-300:], flush=True)
    (Path("/tmp") / "smoke.py").write_text(SMOKE)
    r = subprocess.run([python, "/tmp/smoke.py", str(work)], capture_output=True, text=True, timeout=1800,
                       env={**os.environ, "PYTHONPATH": str(work / "src")}, cwd="/tmp")
    print(r.stdout[-20000:], flush=True)
    print("STDERR", r.stderr[-6000:], flush=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("GAMES_SMOKE ")), None)
    return json.loads(line[len("GAMES_SMOKE "):]) if line else {"ok": False, "error": "no GAMES_SMOKE line",
                                                                   "returncode": r.returncode}


@app.local_entrypoint()
def main():
    b = Path(BUNDLE).resolve()
    call = smoke.spawn(b.name)
    print(json.dumps({"bundle": b.name, "function_call_id": call.object_id}), flush=True)
