"""build_play.py BASE NAME [--heal-reload] [--reload-bench OTHER] [--set KEY=JSON ...]: a play bundle derived from a
built one (sft-prune/dist/BASE) with this checkout's patch functions (build_v82_ablation.py) and study edits; MANIFEST
refreshed. --reload-bench OTHER: one heal-reload trial against overlay arc-heal/OTHER (PTH streaming OPD, 22 Oct)."""
import argparse, hashlib, json, shutil, sys
from pathlib import Path

WT = Path("./.claude/worktrees/sft-prune")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_gate  # noqa: E402
ci_gate.require(WT, "pth-play")  # a red CI blocks the build
sys.path.insert(0, str(WT / "scripts/dfr_exact"))
import build_v82_ablation as b  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("name")
ap.add_argument("--heal-reload", action="store_true")
ap.add_argument("--reload-bench", default="")
ap.add_argument("--reload-modes", default="layers", help="one heal-reload trial per mode (each on a fresh server)")
ap.add_argument("--set", action="append", default=[], help="study KEY=JSON (dotted keys for nested)")
ap.add_argument("--opd-base", default="", help='streaming OPD base play (22 Oct): JSON {"games": [prefixes], "clones": 7, '
                '"token_stop": 500000, "wall": 13500}: rollouts streamed out, no weights follow')
ap.add_argument("--opd", default="", help='streaming OPD play: JSON {"games": [prefixes], "clones": 7, "trainer_run": R, '
                '"play_run": P, "token_stop": 500000, "wall": 15000, "mode": "layers"} (heal reload on)')
ap.add_argument("--rename-play", default="", help="OLD=NEW: replace the play run name in every study path (a bundle "
                "derived from a built OPD play, e.g. g1-v17 -> a new play id)")
ap.add_argument("--live-knobs", default="", help="PLAY_RUN: port live knobs onto a built OPD play (done by --opd already)")
a = ap.parse_args()
src, out = WT / "dist" / a.base, WT / "dist" / a.name
if out.exists():
    raise SystemExit(f"{out} exists")
shutil.copytree(src, out, ignore=shutil.ignore_patterns("__pycache__"))
study_path = out / "configs/comparison.json"
study = json.loads(study_path.read_text())
if a.rename_play:
    old, new = a.rename_play.split("=", 1)
    study = json.loads(json.dumps(study).replace(f"/validation/{old}/", f"/validation/{new}/"))
    if old in json.dumps(study):
        raise SystemExit(f"--rename-play: {old} still appears outside /validation/ paths")
def port_live_knobs(play_run):
    """live knobs: active_limit and kv_margin from <run>/stream/knobs.json (sft-prune ee60ef423,
    runner nebius-opd c4ded35b7); set with play_knobs.py"""
    shutil.copyfile(WT / "src/arena/live_knobs.py", out / "src/arena/live_knobs.py")
    tr = (out / "src/arena/trial.py").read_text()
    if "LiveKnobs(" not in tr:
        a1 = "        last_progress = 0.0\n        try:\n            while wall.remaining() > 0 and any(not t.done() for t in tasks):\n"
        a2 = "                scheduler.tick()\n"
        if tr.count(a1) != 1 or tr.count(a2) != 1:
            raise SystemExit("--opd: live knobs port anchors missing")
        tr = tr.replace(a1, "        knobs = None\n        if study.get(\"live_knobs\"):\n"
                        "            from .live_knobs import LiveKnobs, settings as knob_settings\n\n"
                        "            if lane is not None:\n                raise ValueError(\"live_knobs is wired for the plain FIFO gate\")\n"
                        "            knobs = LiveKnobs(knob_settings(study[\"live_knobs\"]), scheduler, gate, log=self.gate_log,\n"
                        "                              request_tokens=window + self.cap)\n" + a1)
        tr = tr.replace(a2, "                if knobs is not None:\n                    knobs.poll(wall.elapsed())\n" + a2)
        (out / "src/arena/trial.py").write_text(tr)
    study["live_knobs"] = {"path": f"/runs/validation/{play_run}/exchange-in/knobs.json", "poll_seconds": 30}


if a.heal_reload or a.reload_bench:
    b.patch_heal_reload(out)
    study["research_checkpoint"]["heal_reload"] = True
if a.reload_bench:
    other = f"flash-input/models/arc-heal/{a.reload_bench}"
    study.setdefault("nebius_assets", []).append(other)
    env = study["trials"][0].get("server_env", {})
    modes = a.reload_modes.split(",")
    study["trials"] = [{"name": "heal-reload" if len(modes) == 1 else f"heal-reload-{m}", "kind": "heal-reload",
                        "mode": m, "other": f"/assets/{other}", "server_env": {**env, "ARC_HEAL_RELOAD_MODE": m},  # fresh server
                        "prompt_files": ["src/arena/trial.py", "src/arena/run.py", "src/arc_infra/fleet.py",
                                         "src/arena/research_checkpoint.py"],
                        "streams": 32, "stream_tokens": 3000, "pause_bar": 10.0, "wall_seconds": 3600, **({"load_experts_per_layer": 15} if m == "rows" else {})} for m in modes]
    study["wall_seconds"] = 3600 * len(modes)
if a.opd:  # streaming OPD play (22 Oct): rotation group x clones, weights follower, loop stop, rollouts streamed out
    o = json.loads(a.opd)
    b.patch_heal_reload(out)
    # the base bundle ships its own src: port the OPD play code (sft-prune 303c14bc) onto it
    import subprocess
    if "rollout_logprobs" not in (out / "src/arc_infra/fleet.py").read_text():
        b.patch_rollout_logprobs(out)
    subprocess.run(["patch", "-p1", "--no-backup-if-mismatch", "-r", "-", "-i", str(Path(__file__).parent / "opd_play.patch")],
                   cwd=out, capture_output=True)
    fl = (out / "src/arc_infra/fleet.py").read_text()
    anchor = "        self.rollout_path, self.rollout_previous, self.rollout_tools, self.rollout_n = None, None, None, 0\n"
    if "self.weights_version = None" not in fl:
        fl = fl.replace(anchor, anchor + "        self.loop_stop = None  # study loop_stop (OPD, 22 Oct)\n"
                        "        self.weights_version = None  # () -> the served weights version, OPD\n")
    fl = fl.replace("    def record_rollout(self, payload, data):", "    def record_rollout(self, payload, data, version=None):")
    (out / "src/arc_infra/fleet.py").write_text(fl)
    shutil.copyfile(WT / "src/arena/weights_follow.py", out / "src/arena/weights_follow.py")
    tr = (out / "src/arena/trial.py").read_text()
    for need, text in (("weights_version = None", fl), ("self.record_rollout(payload, data, version)", fl),
                       ("row[\"version\"], row[\"version_end\"]", fl), ("payload[\"repetition_detection\"]", fl),
                       ("WeightsFollower(study[\"weights_follow\"]", tr), ("client.client.weights_version = current", tr),
                       ("client.client.loop_stop", tr), ("following.cancel()", tr)):
        if text.count(need) != 1:
            raise SystemExit(f"--opd: OPD play port incomplete ({need!r})")
    port_live_knobs(o["play_run"])
    ids = json.loads((out / "configs/duck_public_notebook.json").read_text())["public_game_ids"]
    games = [next(g for g in ids if g.split("-")[0] == x) for x in o["games"]]
    n = len(games) * int(o["clones"])
    study.update(games=games, active_limit=n, rollout_record="logprobs", token_stop=int(o.get("token_stop", 500000)),
                 wall_seconds=int(o["wall"]), notebook_seconds=max(int(study.get("notebook_seconds") or 0), int(o["wall"]) + 3600),
                 loop_stop={"min_pattern_size": 32, "max_pattern_size": 1024, "min_count": 3},
                 weights_follow={"dir": f"/runs/validation/{o['play_run']}/exchange-in/{o['trainer_run']}/deltas",
                                 "poll_seconds": 5, "mode": o.get("mode", "layers"), "start_version": int(o.get("step0", 0))},
                 nebius_stream={"in": f"{o['trainer_run']}/deltas", "in_seconds": 10, "out_seconds": 15})
    study["competition"] = {**study["competition"], "preview_instances": n, "preview_seconds": int(o["wall"])}
    study["roster"] = (f"Streaming OPD play: student D plays {', '.join(o['games'])} x {o['clones']} copies to "
                       f"{study['token_stop'] // 1000}K tokens each (T 1.0), taking the trainer's weights in place")
    for t in study["trials"]:
        t["wall_seconds"] = int(o["wall"])
if a.opd_base:  # base-model play streamed to the trainer's teacher (forward-KD rows)
    o = json.loads(a.opd_base)
    ids = json.loads((out / "configs/duck_public_notebook.json").read_text())["public_game_ids"]
    games = [next(g for g in ids if g.split("-")[0] == x) for x in o["games"]]
    n = len(games) * int(o["clones"])
    if not study.get("rollout_record"):
        raise SystemExit("--opd-base: the base bundle records no rollouts")
    study.update(games=games, active_limit=n, token_stop=int(o.get("token_stop", 500000)), wall_seconds=int(o["wall"]),
                 notebook_seconds=max(int(study.get("notebook_seconds") or 0), int(o["wall"]) + 3600),
                 nebius_stream={"in_seconds": 10, "out_seconds": 15})
    study["competition"] = {**study["competition"], "preview_instances": n, "preview_seconds": int(o["wall"])}
    study["roster"] = (f"Streaming OPD base play: base Flash plays {', '.join(o['games'])} x {o['clones']} copies to "
                       f"{study['token_stop'] // 1000}K tokens each, rollouts streamed out for the teacher")
    for t in study["trials"]:
        t["wall_seconds"] = int(o["wall"])
if a.live_knobs:
    port_live_knobs(a.live_knobs)
for kv in a.set:
    key, value = kv.split("=", 1)
    node, parts = study, key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = json.loads(value)
study_path.write_text(json.dumps(study, indent=2) + "\n")
sha = lambda q: hashlib.sha256(q.read_bytes()).hexdigest()  # noqa: E731
ident = json.loads((out / "MANIFEST.json").read_text())
ident["files"] = {q.relative_to(out).as_posix(): sha(q) for q in sorted(out.rglob("*"))
                  if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(out / "MANIFEST.json").write_text(json.dumps(ident, indent=2) + "\n")
print(json.dumps({"bundle": str(out), "files": len(ident["files"]), "trials": [t["name"] for t in study["trials"]]}))
