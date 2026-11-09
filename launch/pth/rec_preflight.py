"""rec_preflight.py: CPU preflight of envs recording bundles (launch gate item 1; PTH Stage A D, 20 Oct).

    modal run ./.claude/worktrees/sft-heal/launch/pth/rec_preflight.py --bundles sft4x-d145-v1a,sft4x-d145-v1b

Per bundle, on Modal CPU (python 3.12, x86_64 like the RTX VM): the bundle's own install (bootstrap.py
install_agent_environment: --no-index --require-hashes from its wheels, then pip check), then in that venv the
arena's own roster path (arena.competition.open_arcade with the bundle's study and generated_games: the local
competition gateway, cyclic clones, one scorecard) and every rostered game: make, RESET, 30 random available
actions (RESET on GAME_OVER), frames 64x64 in colours 0-15. The model and the harness are the X bundles' (unchanged)."""
import json
import os
from pathlib import Path

import modal

DIST = Path("./.claude/worktrees/sft-prune/dist")
PATHS = [Path(n) if "/" in n else DIST / n
         for n in os.environ.get("REC_PF_BUNDLES", "sft4x-d145-v1a,sft4x-d145-v1b").split(",") if n]
NAMES = [p.name for p in PATHS]  # a full path for bundles outside sft-prune/dist
image = modal.Image.debian_slim(python_version="3.12")
for path in PATHS:
    image = image.add_local_dir(str(path), f"/bundles/{path.name}", copy=True)
app = modal.App("sft-rec-preflight", include_source=True)
assets = modal.Volume.from_name("arc-prolong-assets")  # public games: /assets/flash-input, as /kaggle/input on the VM

PLAY = r'''
import json, random, sys, time
from pathlib import Path
import numpy as np
bundle, out = Path(sys.argv[1]), Path(sys.argv[2])
study = json.loads((bundle / "configs/comparison.json").read_text())
from arena.competition import open_arcade
from arcengine import GameAction
if study.get("generated_games"):
    games_dir = bundle / study["generated_games"]["dir"]
else:  # the public games, found as the bootstrap finds them on the VM
    from arc_infra.arcade import find_games
    games_dir = find_games("/assets/flash-input")
res = {"study_games": len(study["games"]), "instances": study["competition"]["preview_instances"],
       "token_stop": study.get("token_stop"), "wall": study["wall_seconds"], "games": []}
rng = random.Random(0)
with open_arcade(study, games_dir, out) as (resolved, owner):
    roster = resolved["games"]
    res["roster"] = len(roster)
    res["mode"] = resolved["competition"].get("execution_mode")
    src = json.loads((out / "public-roster.json").read_text())
    from collections import Counter
    count = Counter(r["source_game_id"] for r in src)  # cyclic clones: every game, instances / games times each
    res["roster_sources_match"] = (set(count) == set(study["games"])
                                   and set(count.values()) == {len(src) // len(study["games"])})
    card = owner.open_scorecard()
    for gid in roster:
        row = {"game": gid}
        t0 = time.time()
        try:
            env = owner.make(gid, scorecard_id=card)
            raw = env.reset()
            row["levels"] = int(getattr(env.observation_space, "win_levels", 0) or 0)
            steps, overs, shapes, colours = 0, 0, set(), set()
            t1 = time.time()
            for _ in range(30):
                avail = [a for a in list(raw.available_actions) if int(a) != 0] or [1]
                a = GameAction.from_id(int(rng.choice(avail))) if hasattr(GameAction, "from_id") else GameAction(int(rng.choice(avail)))
                data = {"x": rng.randrange(64), "y": rng.randrange(64)} if a.name == "ACTION6" else {}
                raw = env.step(a, data)
                steps += 1
                if raw is None:
                    raise RuntimeError("no observation")
                f = np.array(raw.frame[-1]) if raw.frame else None
                if f is not None:
                    shapes.add(tuple(f.shape)); colours.update(int(x) for x in np.unique(f))
                if getattr(raw.state, "name", "") == "GAME_OVER":
                    overs += 1
                    raw = env.reset()
            row.update(ok=True, steps=steps, game_overs=overs, shapes=sorted(shapes),
                       colours_ok=all(0 <= c <= 15 for c in colours),
                       ms_per_step=round(1000 * (time.time() - t1) / max(1, steps), 2))
            row["frame_ok"] = row["shapes"] == [(64, 64)] and row["colours_ok"]
        except Exception as exc:
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
        row["seconds"] = round(time.time() - t0, 2)
        res["games"].append(row)
print("PREFLIGHT_JSON " + json.dumps(res))
'''


@app.function(image=image, timeout=1800, volumes={"/assets": assets})
def preflight(name: str):
    import subprocess
    import sys

    bundle = Path(f"/bundles/{name}")
    sys.path.insert(0, str(bundle))
    import bootstrap  # the bundle's own installer

    t0 = __import__("time").time()
    python = bootstrap.install_agent_environment(bundle, Path("/tmp/venv"), dict(os.environ))
    install_s = round(__import__("time").time() - t0, 1)
    Path("/tmp/play.py").write_text(PLAY)
    env = {**os.environ, "PYTHONPATH": str(bundle / "src"), "ONLY_RESET_LEVELS": "true"}
    p = subprocess.run([python, "/tmp/play.py", str(bundle), "/tmp/pf-out"], env=env, capture_output=True, text=True,
                       timeout=1500)
    line = next((x for x in p.stdout.splitlines() if x.startswith("PREFLIGHT_JSON ")), None)
    res = json.loads(line[len("PREFLIGHT_JSON "):]) if line else {"stdout": p.stdout[-1500:], "stderr": p.stderr[-3000:]}
    racer = subprocess.run([python, "-c", "import py_mini_racer; print(py_mini_racer.__file__)"], capture_output=True,
                           text=True)
    res.update(bundle=name, install_seconds=install_s, exit=p.returncode, mini_racer=racer.returncode == 0,
               needs_racer="mini-racer" in (bundle / "requirements.txt").read_text())
    return res


@app.local_entrypoint()
def main(bundles: str = ",".join(NAMES), out: str = ""):
    results = list(preflight.map(bundles.split(",")))
    summary = []
    for r in results:
        g = r.get("games", [])
        bad = [x for x in g if not x.get("ok") or not x.get("frame_ok")]
        summary.append({"bundle": r.get("bundle"), "pass": bool(g) and not bad and r.get("exit") == 0
                        and r.get("roster") == r.get("instances") and (r.get("mini_racer") or not r.get("needs_racer"))
                        and r.get("roster_sources_match"),
                        "roster": r.get("roster"), "instances": r.get("instances"), "mode": r.get("mode"),
                        "install_s": r.get("install_seconds"), "mini_racer": r.get("mini_racer"),
                        "games_ok": sum(1 for x in g if x.get("ok") and x.get("frame_ok")), "bad": bad[:5],
                        "slowest_ms": max((x.get("ms_per_step") or 0) for x in g) if g else None,
                        "token_stop": r.get("token_stop"), "wall": r.get("wall"),
                        "error": None if g else {k: r.get(k) for k in ("stdout", "stderr")}})
    print(json.dumps(summary, indent=1))
    if out:
        Path(out).write_text(json.dumps({"summary": summary, "results": results}, indent=1))
