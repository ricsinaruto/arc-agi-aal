"""Bundle for the cheapest-clear SFT runs on the rl1 stack (SFT arm, 16 Oct; ``arena.rl_sft``): a copy of an rl1
bundle (the v8.2 level bench on Flash NVFP4 + unmerged LoRA, one B300) with this branch's loop and trainer files, the
SFT corpus (``sft/train``, ``sft/probe``: attempt folders from the token-exact rebuild) and ``rl_loop.mode = "sft"``.

usage (this worktree): python scripts/posttrain/build_sft_rl1.py BASE_BUNDLE OUT CORPUS NOTEBOOK_S SFT_JSON [POINTS]
  CORPUS: a folder with train/ and probe/; SFT_JSON: the ``sft`` settings (arena.rl_sft.SFT_DEFAULTS keys)
Trainer attention: triton (rl1_sparse, rl0@72499355; the masked default ran 455 trained tok/s in sft-t2-v2).
Single card: the 32 GiB KV pool of the rl1 single-card loops (awake 148 GiB), the trainer resident beside it.
"""
import hashlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
from arena import rl_loop, rl_sft  # noqa: E402
from build_arena import manifest  # noqa: E402

ARENA_FILES = ("rl_loop.py", "rl_single.py", "rl_sft.py", "rl_ladder.py", "level_bench.py", "restore.py")
TRAINER_FILES = ("scripts/posttrain/rl1_trainer.py", "scripts/posttrain/rl1_sparse.py",
                 "scripts/posttrain/flash_qlora_job.py", "scripts/posttrain/rl_job.py", "src/posttrain/rl_rollouts.py",
                 "scripts/posttrain/heal_kd.py")
POINTS = "data/stuck_bench/easy10_through.json"  # fresh level-1 points (re86, m0r0, ... ) for the positive control

if __name__ == "__main__":
    base, out, corpus, notebook, sft = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), \
        json.loads(sys.argv[5])
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(base, out)
    for name in ARENA_FILES:
        shutil.copyfile(ROOT / "src/arena" / name, out / "src/arena" / name)
    for name in TRAINER_FILES:
        shutil.copyfile(ROOT / name, out / "rl1" / Path(name).name)
    shutil.rmtree(out / "sft", ignore_errors=True)  # a base that is itself an SFT bundle
    for part in ("train", "probe"):
        shutil.copytree(corpus / part, out / "sft" / part, ignore=shutil.ignore_patterns("index.jsonl"))
    shutil.copyfile(sys.argv[6] if len(sys.argv) > 6 else ROOT / POINTS, out / "configs/sft_points.json")
    path = out / "configs/comparison.json"
    study = json.loads(path.read_text())
    rl = dict(study["rl_loop"])
    for key in ("mg", "transfer", "paired", "resume_adapter", "resume_state", "resume_round", "equality_source",
                "round0_from", "round0_play_from", "lvl_resume", "l2_record"):
        rl[key] = None
    dual = os.environ.get("ARC_SFT_DUAL") == "1"  # two cards: the trainer on card 1, the server alone on card 0
    rl.update(mode="sft", sft=rl_sft.merged(sft), attention="triton", dual=dual, trainer_device="1", lvl=False, l2=False,
              f_small=False, opd=False,
              limits={**rl["limits"], "awake_gib": 148.0})
    study["rl_loop"] = rl_loop.settings(rl)
    study["level_bench"] = {**study["level_bench"], "start_points": "configs/sft_points.json", "games": ["re86"]}
    trial = dict(study["trials"][0])
    kv_gib = 96 if dual or os.environ.get("ARC_SFT_EVAL") == "1" else 32  # eval only: no trainer beside the server  # 96 GiB = the serving cap (128 failed sft-gate-re86-v2 at start); the 32 GiB single-card pool starved play (sft-gate-re86-v1: 27 preemptions at 16)
    trial["server_env"] = {**trial["server_env"], "ARENA_KV_CACHE_MEMORY_BYTES": str(kv_gib * 1024**3)}
    trial["wall_seconds"] = notebook
    study["trials"] = [trial]
    study["wall_seconds"] = study["notebook_seconds"] = notebook
    study["name"] = out.name
    study["sft_build"] = {"source": "scripts/posttrain/build_sft_rl1.py", "base_bundle": base.name,
                          "corpus": str(corpus), "train_segments": len(list((out / "sft/train").glob("*/meta.json"))),
                          "probe_segments": len(list((out / "sft/probe").glob("*/meta.json")))}
    path.write_text(json.dumps(study, indent=2) + "\n")
    identity = json.loads((out / "MANIFEST.json").read_text())
    identity["files"] = manifest(out)
    (out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    runpy.run_path(str(out / "bootstrap.py"))["verify"](out)
    print(json.dumps({"output": str(out), "manifest": hashlib.sha256((out / "MANIFEST.json").read_bytes()).hexdigest()[:8],
                      "notebook_seconds": notebook, "sft": study["rl_loop"]["sft"], **study["sft_build"]}))
