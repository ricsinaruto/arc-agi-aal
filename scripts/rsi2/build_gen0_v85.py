"""RSI gen 0 = the v8.5 stack with every public-25 fit refitted on the 10 fixed selection games.

    python scripts/rsi2/build_gen0_v85.py OUT --base <arena-flash-competition-v85-sub | -v85b-sub> --mask <mask.json>
        --overlay <owner/dataset> --overlay-manifest <sha256> --draft-head <overlay dir> --vocab <vocab.json>
        [--games data/rsi2/games-sel85] [--play-seconds N | --slice-fraction F] [--tag s1]

Everything is the base kernel bundle's (harness source, study, server block) except: the t37 mask (refit on fixed-10
routing), the INT3 requant overlay (mask-aware GPTQ W4 on fixed-10 activations, then the same g16 RTN-search requant),
the MTP draft head (trained on fixed-10 traffic over those weights) and the 16K draft vocabulary (fixed-10 generated
tokens; the record keeps the served label "arena16k", which the serving check pins). Roster: the 85 selection games,
one instance each, fresh starts, all at once on one GPU."""
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
    ap.add_argument("--base", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--overlay", required=True, help="owner/dataset of the INT3 requant overlay")
    ap.add_argument("--overlay-manifest", required=True)
    ap.add_argument("--draft-head", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--games", default=str(Path(__file__).resolve().parents[2] / "data/rsi2/games-sel85"))
    ap.add_argument("--play-seconds", type=int)
    ap.add_argument("--slice-fraction", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out, base = Path(a.out), Path(a.base)
    if out.exists():
        raise SystemExit(f"{out} exists")
    shutil.copytree(base, out, ignore=shutil.ignore_patterns(*SKIP))
    study = json.loads((base / "configs/comparison.json").read_text())
    server = study["server"]
    old_mask = out / server["expert_prune"]["mask"]
    old_mask.unlink()
    mask_rel = f"configs/expert-prune/{Path(a.mask).name}"
    shutil.copyfile(a.mask, out / mask_rel)
    m = json.loads((out / mask_rel).read_text())
    if m["pruned_per_layer"] != 192 or len(m["pruned"]) != 48:
        raise SystemExit("not a t37 uniform mask")
    server["expert_prune"] = {"mask": mask_rel, "sha256": sha(out / mask_rel)}
    old_dataset = server["expert_overlay"]["dataset"].split("/", 1)[1]
    server["expert_overlay"] = {"dataset": a.overlay, "manifest_sha256": a.overlay_manifest, "check": "size"}
    study["model_datasets"] = [a.overlay]
    target = out / "draft-head/overlay"
    target.mkdir(parents=True)
    for name in ("overlay.safetensors", "overlay-manifest.json"):
        shutil.copyfile(Path(a.draft_head) / name, target / name)
    server["draft_head"] = {"dump": False, "overlay": {"path": "draft-head/overlay",
                                                       "sha256": sha(target / "overlay-manifest.json")}}
    vocab = json.loads(Path(a.vocab).read_text())
    if len(vocab["ids"]) != 16384 or hashlib.sha256(json.dumps(vocab["ids"]).encode()).hexdigest() != vocab["ids_sha256"]:
        raise SystemExit("bad vocabulary record")
    vocab["fit"] = vocab.pop("name")  # the served label stays "arena16k" (serving_kernels_patch CHOICES)
    vocab["name"] = "arena16k"
    (out / "src/arc_infra/serving_draft_vocab.json").write_text(json.dumps(vocab, separators=(",", ":")) + "\n")
    games = Path(a.games)
    shutil.copytree(games / "environment_files", out / "generated_games/environment_files")
    roster = json.loads((games / "roster.json").read_text())["games"]
    study["generated_games"] = {"dir": "generated_games/environment_files", "games": roster}
    study["games"] = roster
    full = len(roster) * SLICE
    play = a.play_seconds or round(full * a.slice_fraction)
    marks = {k: round(full * f) for k, f in STAGES.items() if full * f <= play + 1}
    for drop in ("games_excluded", "roster"):
        study.pop(drop, None)
    study.update(name=out.name, status="prepared-not-launched", wall_seconds=play, notebook_seconds=play + 4800)
    study["competition"].update(preview_instances=len(roster), preview_seconds=play, preview_live_score=True)
    study["live_progress"] = {"interval_seconds": 60, "score_seconds": 120}
    study["rsi2"] = {
        "role": "gen 0", "tag": a.tag, "games": len(roster), "slice_seconds_per_game": SLICE, "marks_seconds": marks,
        "plan": "/ ~16:45 (docs/rsi.md, restart)", "base": base.name,
        "refits": {"mask": {"file": mask_rel, "source_run": m.get("source_run")},
                   "overlay": {"dataset": a.overlay, "manifest_sha256": a.overlay_manifest},
                   "draft_head": json.loads((target / "overlay-manifest.json").read_text()).get("source_run"),
                   "vocab": {k: vocab[k] for k in ("fit", "corpus", "corpus_coverage", "ids_sha256")}}}
    text = json.dumps(study, indent=2)  # descriptive records (arena_variant, hero_v2.components) name the new fits too
    for before, after in ((old_mask.stem, Path(mask_rel).stem), (old_dataset, a.overlay.split("/", 1)[1])):
        text = text.replace(before, after)
    (out / "configs/comparison.json").write_text(text + "\n")
    identity = json.loads((base / "MANIFEST.json").read_text())
    identity["files"] = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
                         if p.is_file() and p.suffix != ".ipynb" and "__pycache__" not in p.parts
                         and p.name not in ("MANIFEST.json", "build-manifest.json")}
    identity.update(dependencies_prepared=True, study=out.name)
    (out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps({"bundle": str(out), "games": len(roster), "play_seconds": play, "marks": marks,
                      "files": len(identity["files"]), "manifest_sha256": sha(out / "MANIFEST.json")}))


if __name__ == "__main__":
    main()
