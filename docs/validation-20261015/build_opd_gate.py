"""OPD teacher gate on one Nebius RTX PRO 6000.

The rl1-loop-l2 bundle (rl0's build_rl1.py, B300 Modal) with the arena sources of this branch (OPD gate and the
other-game mismatched hint) and only these serving/run changes: mode opd_gate; no LoRA and no trainer (the gate plays
the base model only); RTX memory settings instead of the B300 pool env (gpu util 0.92, no fixed KV bytes); the
recorded inputs shipped in the bundle (rl1-loop-l1 round-0 base games for the L2 starts and the a008 hint; base
lp85 games of rl1-loop-v9c's eval-final for the mismatched L2 hint); started by the Nebius launcher's entry hook.
Serving stays rl1's reference profile: Flash NVFP4 (all 512 experts), BF16 dense, BF16 KV, no MTP.

python docs/validation-20261015/build_opd_gate.py OUT --base ../rl/dist/rl1-loop-l2 --l1 L1_RESUME_DIR --lp85 LP85_DIR
    [--held 8] [--tries 2] [--wall 5400]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ("src/arena/rl_single.py", "src/arena/rollout_record.py", "src/arena/rl_loop.py", "src/arena/level_bench.py",
       "src/arena/restore.py")
ASSETS = ["flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4",
          "flash-input/datasets/keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
          "flash-input/datasets/keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1", "games"]


def copy_games(src, dst, keep_game=None):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True)
    bench = json.loads((src / "level-bench.json").read_text())
    if keep_game:
        bench["attempts"] = [r for r in bench["attempts"] if str(r["game"]).startswith(keep_game)]
    (dst / "level-bench.json").write_text(json.dumps(bench))
    for row in bench["attempts"]:
        for name in ("log.jsonl", "rollout.jsonl"):
            f = src / "attempts" / row["label"] / name
            if f.is_file():
                (dst / "attempts" / row["label"]).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dst / "attempts" / row["label"] / name)
    return len(bench["attempts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--base", required=True)
    ap.add_argument("--l1", required=True, help="rl1-loop-l1/resume (games-00 inside)")
    ap.add_argument("--lp85", required=True, help="a bench folder with lp85 base attempts (level-bench.json, attempts/)")
    ap.add_argument("--held", type=int, default=8)
    ap.add_argument("--tries", type=int, default=2)
    ap.add_argument("--wall", type=int, default=5400)
    ap.add_argument("--shared", action="store_true", help="v3: one bench (one KV gate) for all arms, interleaved")
    ap.add_argument("--concurrency", type=int, default=8, help="--shared: attempts at once")
    ap.add_argument("--reserve", type=int, default=900, help="--shared: seconds the bench leaves before the trial deadline")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    shutil.copytree(a.base, out, ignore=shutil.ignore_patterns("__pycache__"))
    for rel in SRC:
        shutil.copyfile(ROOT / rel, out / rel)
    data = out / "rl1-data"
    n_l1 = copy_games(Path(a.l1) / "games-00", data / "l1resume/games-00")
    n_lp = copy_games(a.lp85, data / "lp85-v9c", keep_game="lp85")
    path = out / "configs/comparison.json"
    study = json.loads(path.read_text())
    rl = study["rl_loop"]
    rl.update(mode="opd_gate", dual=False, l2_held=a.held, opd_gate_tries=a.tries,
              lvl_resume={"folder": "/opt/arc-validation/rl1-data/l1resume", "state": None, "round": 0},
              opd_gate_mismatch={"folder": "rl1-data/lp85-v9c", "game": "lp85", "level": 2},
              concurrency=3 * a.held * a.tries)
    if a.shared:  # v2b: three concurrent benches oversubscribed the pool 3x and were cut at the wall (censored)
        rl.update(opd_gate_shared=True, opd_gate_concurrency=a.concurrency, opd_gate_reserve=a.reserve,
                  concurrency=a.concurrency)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
    from arena import rl_loop  # noqa: E402
    study["rl_loop"] = rl_loop.settings(rl)
    seqs = (a.concurrency if a.shared else 3 * a.held * a.tries) + 4
    top = -(-seqs // 8) * 8
    server = {k: v for k, v in study["server"].items() if k != "lora"}
    server.update(max_num_seqs=seqs, gpu_memory_utilization="0.90", max_cudagraph_capture_size=top,
                  cudagraph_capture_sizes=sorted({1, 2, 4, 8, top, *range(16, top + 1, 8)}))
    study["server"] = server
    study["generation_slots"] = seqs
    study["active_limit"] = max(study["active_limit"], seqs)
    # gate-v1c: unset utilization means 0.96 in the fork, and the cache probe's prefill (PLE short conv) OOMed there
    env = {k: v for k, v in study["trials"][0]["server_env"].items() if k != "ARENA_KV_CACHE_MEMORY_BYTES"}
    env["ARENA_GPU_MEMORY_UTILIZATION"] = "0.90"
    study["trials"] = [{**study["trials"][0], "server_env": env, "wall_seconds": a.wall}]
    study["wall_seconds"] = study["notebook_seconds"] = a.wall
    study["dfr_exact"] = {"entry": "bootstrap_arena.py", "timeout_seconds": a.wall + 300,
                          "play_backstop_seconds": a.wall - 2700, "assets": ASSETS}
    study["sft_opd_gate"] = {"source": "docs/validation-20261015/build_opd_gate.py (sft-opd)", "base": Path(a.base).name,
                             "l1_attempts": n_l1, "lp85_attempts": n_lp, "arms": ["none", "hint", "mismatched"],
                             "per_arm": a.held * a.tries, "server_removed": ["lora"],
                             "server_env_removed": ["ARENA_KV_CACHE_MEMORY_BYTES"], "gpu_memory_utilization": "0.90",
                             "shared_bench": a.shared, "concurrency": a.concurrency if a.shared else None}
    study["name"] = out.name
    path.write_text(json.dumps(study, indent=2) + "\n")
    from build_arena import manifest  # noqa: E402
    identity = json.loads((out / "MANIFEST.json").read_text())
    identity["files"] = manifest(out)
    (out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    import runpy
    runpy.run_path(str(out / "bootstrap.py"))["verify"](out)
    print(json.dumps({"bundle": str(out), "seqs": seqs, "l1_attempts": n_l1, "lp85_attempts": n_lp,
                      "rl_loop": {k: study["rl_loop"][k] for k in ("mode", "l2_held", "opd_gate_tries", "concurrency")}}))


if __name__ == "__main__":
    main()
