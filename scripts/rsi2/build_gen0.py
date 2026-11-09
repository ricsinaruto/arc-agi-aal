"""RSI arm restart: gen 0 and its noise-check bundle.

    python scripts/rsi2/build_gen0.py OUT --v85b <arena-flash-competition-v85b-sub> --av2c <sabl-AV2C-c100-300k-v1>
        [--games data/rsi2/games-sel85] [--seed-tag s1] [--slice-fraction 1.0]

Gen 0 = v8.5's harness and serving with every public-game fit removed:
  - experts: v8.5's t37 mask and INT3 requant were fitted on all 25 public games; refitting the requant needs a new
    GPTQ calibration (heavy), so per the fallback the experts are Intel's AutoRound W4A16 checkpoint, all 512
    (AV2C's research_checkpoint; generic-text calibration, no mask);
  - draft: stock MTP head (v8.5's head was trained on all-25 traffic) and the full vocabulary (arena16k came from
    public-game runs); both lossless, throughput only;
  - everything else is v8.5b's study (harness settings, equal scheduler parked at 300K, BF16 dense, BF16 indexer keys,
    FP8 KV, 25 focus streams at gate 72.888, MTP 1); the tool-latency fix (history.py) is v8.5's source.
The bundle base is AV2C's (a Nebius-run vLLM bundle of the same harness with the AutoRound path and token_stop).

Roster: the 85 selection games (Felix561 50 Studio + 25 DreamTeam, plus the 10 fixed public games), one instance
each, all in one run on one GPU, fresh starts. Wall = 85 x the competition slice (30,000 s play / 110 games =
272.7 s of the GPU per game) x --slice-fraction; stage marks (50K/150K/380K of v8.5's ~380K slice) are recorded
in study.rsi2 and read from the live progress rows."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

SKIP = ("dataset-upload", "experiment.ipynb", "kernel-metadata.json", "MANIFEST.json", "__pycache__", "draft-head")
SLICE = 30000 / 110
STAGES = {"s1": 50 / 380, "s2": 150 / 380, "s3": 1.0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--v85b", required=True)
    ap.add_argument("--av2c", required=True)
    ap.add_argument("--games", default=str(Path(__file__).resolve().parents[2] / "data/rsi2/games-sel85"))
    ap.add_argument("--seed-tag", default="s1")
    ap.add_argument("--slice-fraction", type=float, default=1.0)
    a = ap.parse_args()
    out, v85b, av2c, games = Path(a.out), Path(a.v85b), Path(a.av2c), Path(a.games)
    if out.exists():
        raise SystemExit(f"{out} exists")
    shutil.copytree(av2c, out, ignore=shutil.ignore_patterns(*SKIP))
    # v8.5's harness source: AV2C's src is v8.2-tc + equal scheduler + token_stop; v8.5 adds the tool-latency fix.
    shutil.copyfile(v85b / "src/duck_transfer/history.py", out / "src/duck_transfer/history.py")
    for rel in ("src/arena/priority.py", "src/arena/scheduler.py", "src/arena/serving.py", "src/duck_transfer"):
        p = v85b / rel
        files = [p] if p.is_file() else sorted(x for x in p.rglob("*.py"))
        for f in files:
            if sha(f) != sha(out / f.relative_to(v85b)):
                raise SystemExit(f"harness source differs from v8.5b beyond history.py: {f.relative_to(v85b)}")
    study = json.loads((v85b / "configs/comparison.json").read_text())
    ref = json.loads((av2c / "configs/comparison.json").read_text())
    server = json.loads(json.dumps(ref["server"]))
    server.pop("draft_head", None)                    # stock MTP head
    server["serving_kernels"].pop("draft_vocab", None)  # full vocabulary
    server["mtp_tokens"] = study["server"]["mtp_tokens"]  # v8.5: 1
    server["max_num_seqs"] = max(server["max_num_seqs"], study["focus_oversubscription"]["focus_slots"])
    if server.get("expert_prune") or server.get("expert_overlay") or server["serving_kernels"]["moe_backend"] != "marlin":
        raise SystemExit("AV2C server block carries a mask or overlay")
    study["server"] = server
    for key in ("research_checkpoint", "nebius_assets", "b200", "cache_experiment", "trials"):
        study[key] = json.loads(json.dumps(ref[key]))
    study["cache_experiment"]["cpu_cache_gib"] = 0
    study["model_datasets"] = []
    study.pop("stall_dump_seconds", None)  # the stall dump rides on the prune patch (AV2C drops it too)
    for drop in ("arena_variant", "hero_v2", "serving_ablation", "games_excluded", "roster", "v84", "v85", "v85b"):
        study.pop(drop, None)
    shutil.copytree(games / "environment_files", out / "generated_games/environment_files")
    roster = json.loads((games / "roster.json").read_text())["games"]
    study["generated_games"] = {"dir": "generated_games/environment_files", "games": roster}
    study["games"] = roster
    play = round(len(roster) * SLICE * a.slice_fraction)
    marks = {k: round(len(roster) * SLICE * f) for k, f in STAGES.items() if f <= a.slice_fraction + 1e-9}
    study.update(name=out.name, status="prepared-not-launched", wall_seconds=play, token_stop=0,
                 notebook_seconds=play + 4800, generation_slots=server["max_num_seqs"])
    study["b200"]["stop"]["wall_seconds"] = play
    study["competition"].update(preview_instances=len(roster), preview_seconds=play, preview_live_score=True)
    study["live_progress"] = {"interval_seconds": 60, "score_seconds": 120}
    study["rsi2"] = {
        "role": "gen0 noise check", "seed_tag": a.seed_tag, "games": len(roster), "slice_seconds_per_game": SLICE,
        "marks_seconds": marks, "plan": "(docs/rsi.md, restart)",
        "gen0": "v8.5 harness + v8.5b serving; experts Intel AutoRound W4A16 all 512 (no mask, no requant); stock MTP "
                "head, full draft vocabulary (v8.5's all-25 fits removed)",
        "base_bundles": {"v85b": v85b.name, "av2c": av2c.name}}
    (out / "configs/comparison.json").write_text(json.dumps(study, indent=2) + "\n")
    identity = json.loads((av2c / "MANIFEST.json").read_text())
    identity["files"] = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
                         if p.is_file() and p.suffix != ".ipynb" and "__pycache__" not in p.parts
                         and p.name not in ("MANIFEST.json", "build-manifest.json")}
    identity.update(dependencies_prepared=True, study=out.name)
    (out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps({"bundle": str(out), "games": len(roster), "play_seconds": play, "marks": marks,
                      "files": len(identity["files"]), "manifest_sha256": sha(out / "MANIFEST.json")}))


if __name__ == "__main__":
    main()
