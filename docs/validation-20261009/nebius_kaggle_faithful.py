"""A Nebius RTX bundle from a Kaggle competition build that plays the hidden rerun's path under Kaggle's host limits.

    python docs/validation-20261009/nebius_kaggle_faithful.py SRC OUT [--memory-max-gib 175] [--shm-size 87G]

SRC is a build_v82 Kaggle dist (e.g. dist/arena-flash-competition-v85l1b-sub). OUT keeps every file, adds
src/arena/kaggle_faithful.py and a three-line hook in bootstrap_arena.py, and sets in the study: the rerun branch with
the public rehearsal gateway (competition.rehearsal: the 32,400 s envelope less setup and cleanup, preview_instances
cyclic clones), the b200 block the Nebius launcher needs, notebook_seconds for its timeout, and ``kaggle_faithful``.
"""
import argparse
import difflib
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP = ("dataset-upload", "experiment.ipynb", "kernel-metadata.json", "__pycache__")
HOOK_AFTER = '    study = json.loads((bundle / "configs/comparison.json").read_text())\n'
# --live-score: the rehearsal peeks the local gateway's scorecard (main c25dbe183; monitoring only)
LIVE_OLD = """        if server is not None and (settings.get('research_clones')
                                   or (settings.get('preview_live_score') and not is_rerun())):"""
LIVE_NEW = """        if server is not None and (settings.get('research_clones') or (settings.get('preview_live_score') and (
                not is_rerun() or is_rehearsal(study)))):"""
# --dynamic-streams (throughput E1, 25 Oct): dynamic_streams.py + a trial-loop step + the scheduler's release
DYN_TRIAL = [("        last_progress = 0.0\n",
              "        dyn = None\n"
              "        if study.get(\"dynamic_streams\"):  # throughput E1 (25 Oct): active_limit follows live KV\n"
              "            from .dynamic_streams import DynamicStreams\n"
              "            dyn = DynamicStreams(study[\"dynamic_streams\"], log=self.gate_log)\n"
              "        last_progress = 0.0\n"),
             ("                scheduler.tick()\n",
              "                scheduler.tick()\n"
              "                if dyn is not None:\n"
              "                    dyn.step(wall.elapsed(), getattr(monitor, \"samples\", []), scheduler, gate)\n")]
DYN_SCHED = [("        if best is game:\n",
              "        if len(self.active()) > self.active_limit:  # dynamic_streams lowered the limit: release, no refill\n"
              "            game.priority_queued = self.priority_seq\n"
              "            game.priority_queued_score = value\n"
              "            self.park(game, \"handover\", extra={\"eviction\": kind, \"priority\": value, \"dynamic_release\": True})\n"
              "            self.priority_counts[\"dynamic_release\"] = self.priority_counts.get(\"dynamic_release\", 0) + 1\n"
              "            self._priority_note(\"dynamic_release\", game, eviction=kind, active=len(self.active()),\n"
              "                                limit=self.active_limit)\n"
              "            return\n"
              "        if best is game:\n")]
HOOK = ('    if study.get("kaggle_faithful"):  # Nebius test bundles only (nebius_kaggle_faithful.py)\n'
        '        sys.path.insert(0, str(bundle / "src"))\n'
        '        __import__("arena.kaggle_faithful").kaggle_faithful.apply(study["kaggle_faithful"], Path(args.output))\n')


# --cache-trace (E3 diagnostic, 25 Oct): the game id rides in vllm_xargs (SamplingParams.extra_args): no effect on
# prompt, hashing or sampling.
CT_AGENT_OLD = ('        extra = {"record_extra": record_extra} if record_extra else {}\n')
CT_AGENT_NEW = (CT_AGENT_OLD + '        payload = {**payload, "vllm_xargs": {**(payload.get("vllm_xargs") or {}), "arc_game": str(self.game_id)}}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--memory-max-gib", type=float, default=175)
    ap.add_argument("--shm-size", default="87G")
    ap.add_argument("--memtrace-seconds", type=int, default=60)
    ap.add_argument("--live-score", action="store_true", help="live score in progress.json on the rehearsal")
    ap.add_argument("--dynamic-streams", default="", metavar="LO,HI,UP_KV,DOWN_KV",
                    help="E1: active_limit in [LO, HI] from live KV (max_num_seqs HI + 4, decode graphs to match)")
    ap.add_argument("--timeout-seconds", type=int, default=0,
                    help="Nebius timeout (crash ceiling and spend bound) for a short read stopped gracefully before it; "
                         "the rehearsal envelope stays the Kaggle one")
    ap.add_argument("--cache-trace", action="store_true",
                    help="E3 diagnostic: cache_trace_runtime.py appended to arc_offload_runtime.py, requests tagged "
                         "with vllm_xargs.arc_game, ARC_CACHE_TRACE=1 + ARC_OFFLOAD_TRACE=1 in the server env")
    ap.add_argument("--offload-mamba-demote", default="", metavar="RUNTIME",
                    help="tier read (28 Oct): RUNTIME = arc_offload_runtime.py with ARC_OFFLOAD_MAMBA_DEMOTE (dfr-exact "
                         "tp-mamba-demote); only insertions over the bundle's runtime; the flag set in the server env")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    shutil.copytree(src, out, ignore=shutil.ignore_patterns(*SKIP))
    boot = out / "bootstrap_arena.py"
    text = boot.read_text()
    if text.count(HOOK_AFTER) != 1:
        raise SystemExit("bootstrap layout changed: hook anchor not found once")
    boot.write_text(text.replace(HOOK_AFTER, HOOK_AFTER + HOOK))
    shutil.copy(HERE / "kaggle_faithful.py", out / "src/arena/kaggle_faithful.py")
    changed = ["bootstrap_arena.py", "configs/comparison.json", "src/arena/kaggle_faithful.py"]
    if a.live_score:
        comp_py = out / "src/arena/competition.py"
        text = comp_py.read_text()
        if text.count(LIVE_OLD) != 1:
            raise SystemExit("--live-score: competition.py gating anchor not found once")
        comp_py.write_text(text.replace(LIVE_OLD, LIVE_NEW))
        changed.append("src/arena/competition.py")
    if a.dynamic_streams:
        for rel, edits in (("src/arena/trial.py", DYN_TRIAL), ("src/arena/scheduler.py", DYN_SCHED)):
            text = (out / rel).read_text()
            for old, new in edits:
                if text.count(old) != 1:
                    raise SystemExit(f"--dynamic-streams: anchor not found once in {rel}: {old!r}")
                text = text.replace(old, new)
            (out / rel).write_text(text)
            changed.append(rel)
        shutil.copy(HERE / "dynamic_streams.py", out / "src/arena/dynamic_streams.py")
        changed.append("src/arena/dynamic_streams.py")
    if a.cache_trace:
        rt = out / "src/arc_infra/arc_offload_runtime.py"
        if "class ArcOffloadingConnector" not in rt.read_text():
            raise SystemExit("--cache-trace: no offloading connector in the bundle (L1 stack only)")
        rt.write_text(rt.read_text() + "\n\n" + (HERE / "cache_trace_runtime.py").read_text())
        agent = out / "src/duck_transfer/agent.py"
        text = agent.read_text()
        if text.count(CT_AGENT_OLD) != 1:
            raise SystemExit("--cache-trace: agent request anchor not found once")
        agent.write_text(text.replace(CT_AGENT_OLD, CT_AGENT_NEW))
        changed += ["src/arc_infra/arc_offload_runtime.py", "src/duck_transfer/agent.py"]
    demote_sha = None
    if a.offload_mamba_demote:
        rt = out / "src/arc_infra/arc_offload_runtime.py"
        new = Path(a.offload_mamba_demote).read_text()
        ops = difflib.SequenceMatcher(None, rt.read_text().splitlines(), new.splitlines(), autojunk=False).get_opcodes()
        if "ARC_OFFLOAD_MAMBA_DEMOTE" not in new or any(op not in ("equal", "insert") for op, *_ in ops):
            raise SystemExit("--offload-mamba-demote: RUNTIME must be the bundle's runtime plus the flag block only")
        rt.write_text(new)
        demote_sha = hashlib.sha256(new.encode()).hexdigest()
        changed.append("src/arc_infra/arc_offload_runtime.py")
    study_path = out / "configs/comparison.json"
    study = json.loads(study_path.read_text())
    comp = study["competition"]
    envelope = int(comp["submission_seconds"])
    study["name"] = out.name
    comp["rehearsal"] = True
    if a.live_score:
        comp["preview_live_score"] = True
    comp["preview_seconds"] = envelope - int(comp["cleanup_seconds"])  # the launcher's play estimate only (rerun path)
    study["notebook_seconds"] = envelope + 1200  # the Nebius timeout (+300); the bootstrap uses submission_seconds
    if a.timeout_seconds:
        if not 1800 <= a.timeout_seconds <= envelope + 1200:
            raise SystemExit("--timeout-seconds: between 1,800 and the envelope + 1,200")
        study["notebook_seconds"] = a.timeout_seconds - 300  # launcher timeout = notebook_seconds + 300
    study["b200"] = {"test": "b200-run", "harness": "arena", "gpu": "RTX-PRO-6000",
                     "stop": {"score_target": 1000.0, "wall_seconds": envelope + 600, "usd_limit": None}}
    calib = (study.get("fp8_dense") or {}).get("calib")
    for trial in study["trials"]:  # the notebook copies the FP8 calibration to /tmp first (Kaggle FUSE); Nebius
        env = trial.get("server_env") or {}  # assets are local disk under /kaggle/input: read them in place
        if calib and env.get("ARC_FP8_CALIB", "").startswith("/tmp/arc-fp8-calib/"):
            if env["ARC_FP8_CALIB"].rsplit("/", 1)[1] != calib.split("/")[1]:
                raise SystemExit(f"ARC_FP8_CALIB {env['ARC_FP8_CALIB']} is not the study's calib {calib}")
            env["ARC_FP8_CALIB"] = f"/kaggle/input/datasets/{calib}"
    if a.dynamic_streams:
        lo, hi, up, down = a.dynamic_streams.split(",")
        lo, hi = int(lo), int(hi)
        srv = study["server"]
        if lo != study["active_limit"] or hi < lo:
            raise SystemExit(f"--dynamic-streams: LO must be the study's active_limit {study['active_limit']}")
        step = 1 + int(srv.get("mtp_tokens") or 0)
        srv["max_num_seqs"] = study["generation_slots"] = hi + 4
        need = srv["max_num_seqs"] * step
        sizes = srv["cudagraph_capture_sizes"]
        srv["cudagraph_capture_sizes"] = sorted(set(sizes) | set(range(sizes[-1] + 8, need + 8, 8)))
        srv["max_cudagraph_capture_size"] = srv["cudagraph_capture_sizes"][-1]
        if srv["max_cudagraph_capture_size"] < need or srv["max_cudagraph_capture_size"] > 256:
            raise SystemExit(f"--dynamic-streams: decode graphs to {srv['max_cudagraph_capture_size']} for {need}")
        study["dynamic_streams"] = {"lo": lo, "hi": hi, "up_kv": float(up), "down_kv": float(down), "every_seconds": 30}
    if a.cache_trace:
        for trial in study["trials"]:
            trial.setdefault("server_env", {}).update(ARC_CACHE_TRACE="1", ARC_OFFLOAD_TRACE="1")
        study["cache_trace"] = True
    if demote_sha:
        for trial in study["trials"]:
            trial.setdefault("server_env", {})["ARC_OFFLOAD_MAMBA_DEMOTE"] = "1"
        study["offload_mamba_demote"] = {"runtime_sha256": demote_sha}
    study["kaggle_faithful"] = {"rerun": True, "memory_max_gib": a.memory_max_gib, "shm_size": a.shm_size,
                                "memtrace_seconds": a.memtrace_seconds, "cgroup": "arc-kaggle"}
    study_path.write_text(json.dumps(study, indent=2) + "\n")
    manifest = json.loads((out / "MANIFEST.json").read_text())
    manifest["study"] = out.name
    for rel in changed:
        manifest["files"][rel] = hashlib.sha256((out / rel).read_bytes()).hexdigest()
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(out), "kaggle_faithful": study["kaggle_faithful"], "notebook_seconds":
                      study["notebook_seconds"], "games": len(study["games"]), "instances": comp["preview_instances"]}))


if __name__ == "__main__":
    main()
