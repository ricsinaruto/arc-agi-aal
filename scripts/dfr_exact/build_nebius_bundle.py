"""Build the Nebius bundle of dfranzen's exact notebook: ``bootstrap_dfr.py`` + his notebook.

    python scripts/dfr_exact/build_nebius_bundle.py dist/<name> --b2d <b2d-24h-v2 bundle> [--stop-tokens 11000000]

The gateway script, its arc_agi 0.9.9 wheels and the roster come from the b2d-24h-v2 bundle (the gateway is held equal
to that run); the kernel's extra wheels (IPython 8.37, ipykernel 6.29.5, nbclient, pandas, pyarrow) are downloaded for
linux x86_64 / cp312 under the b2d wheel versions as constraints. The launcher (nebius branch) plans a study with
``backend: dfr-exact`` from its ``dfr_exact`` block: timeout, asset prefixes, play backstop.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "arc-agi-3-milestone-2-solution.ipynb"
NOTEBOOK_SHA256 = "7b76c194b478faa0309b01e5fba05840f1314da7f288e05e5f4b09972e4f9b4c"
ASSETS = ["games",  # b2d-24h-v2's games folder (the gateway's environments)
          "flash-input/datasets/dfranzen/pennyroyal-v253",
          "flash-input/datasets/dfranzen/taaf-kaggle-source-bundle-copy",
          "flash-input/competitions/arc-prize-2026-arc-agi-3",
          "flash-input/models/dfranzen/albucino-qwen3-8-flash-next-drafter",
          "flash-input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround"]
EXTRA = ["ipython==8.37.0", "ipykernel==6.29.5", "nbclient", "nbformat", "pandas", "pyarrow"]
HIS_WALL = 532 * 60  # cell 16, competition branch; his scheduler's tail fade starts at 80% of it
# His public version 3 (pulled 11 October 2026): 25 games x 4 passes, wall 532*60 * concurrency // 110 = 29,018 s.
PUBLIC_NOTEBOOK = HERE / "arc-agi-3-milestone-2-solution-v3.ipynb"
PUBLIC_SHA256 = "88a379f17c3a4ba8c98715fc23155346ccd6eb289bdafebadfc61a3178f9d9bd"
PUBLIC_EXCLUDED = ["ar25", "ft09", "lp85", "tu93", "vc33"]  # the 25 public games minus these = the 20 c110 families
PUBLIC_EDIT = {"cell": 16, "old": "demo_excluded_games = [] if TRUE_SUBMISSION else []",
               "new": f"demo_excluded_games = [] if TRUE_SUBMISSION else {PUBLIC_EXCLUDED!r}"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--b2d", required=True, help="the b2d-24h-v2 bundle (gateway, wheels, roster)")
    ap.add_argument("--stop-tokens", type=int, default=11_000_000)
    ap.add_argument("--play-backstop", type=int, default=23_400)
    ap.add_argument("--public", action="store_true", help="his public version 3 on 20 games x 4 passes, no gateway")
    ap.add_argument("--equal-24h", type=int, metavar="INSTANCES",
                    help="INSTANCES clones, 24 h, no token stop, the equal scheduler (dfr_equal.py)")
    ap.add_argument("--park-tokens", type=int, default=300_000)
    ap.add_argument("--wheels-from", help="a built dfrx bundle whose wheels/ to reuse instead of downloading")
    args = ap.parse_args()
    out, b2d = Path(args.output), Path(args.b2d)
    if out.exists():
        raise SystemExit(f"{out} exists")
    notebook, notebook_sha = (PUBLIC_NOTEBOOK, PUBLIC_SHA256) if args.public else (NOTEBOOK, NOTEBOOK_SHA256)
    if sha(notebook) != notebook_sha:
        raise SystemExit("notebook hash mismatch")
    if not (args.public or args.equal_24h) and args.play_backstop >= 0.8 * HIS_WALL - 1800:
        raise SystemExit("the play backstop must end well before his tail fade (80% of 532 min)")
    ref = json.loads((b2d / "configs/comparison.json").read_text())
    (out / "notebook").mkdir(parents=True)
    (out / "configs").mkdir()
    notebook_file = "notebook/" + notebook.name.replace(".ipynb", ".json")
    shutil.copyfile(notebook, out / notebook_file)
    for name in ("bootstrap_dfr.py", "run_notebook.py") + (("dfr_equal.py",) if args.equal_24h else ()):
        shutil.copyfile(HERE / name, out / name)
    if not args.public:
        shutil.copyfile(b2d / "b200_gateway.py", out / "gateway.py")
    shutil.copytree(Path(args.wheels_from or b2d) / "wheels", out / "wheels")
    pins = []
    for wheel in sorted((out / "wheels").glob("*.whl")):
        name, version = re.match(r"(.+?)-(\d[^-]*)-", wheel.name).groups()
        pins.append(f"{name.replace('_', '-')}=={version}")
    (out / "constraints.txt").write_text("\n".join(pins) + "\n")
    subprocess.run([] if args.wheels_from else [sys.executable, "-m", "pip", "download", "--quiet", "--only-binary=:all:", "--dest",
                    str(out / "wheels"), "--platform", "manylinux_2_28_x86_64", "--platform", "manylinux2014_x86_64",
                    "--platform", "manylinux_2_17_x86_64", "--python-version", "3.12", "--implementation", "cp",
                    "--abi", "cp312", "--abi", "abi3", "--abi", "none", "--constraint", str(out / "constraints.txt"),
                    *EXTRA, *(["imageio-ffmpeg"] if args.public else [])], check=True) if not args.wheels_from else None
    setup = 1500  # venvs, gateway, his wheelhouse install, weight load and graph capture (437 s on Modal)
    grace = 900
    study = {
        "name": out.name, "status": "prepared-not-launched", "backend": "dfr-exact", "games": ref["games"],
        "dfr_exact": {
            "notebook": "dfranzen/arc-agi-3-milestone-2-solution, pulled 1 October 2026", "notebook_sha256": NOTEBOOK_SHA256,
            "instances": ref["competition"]["preview_instances"], "stop_generated_tokens": args.stop_tokens,
            "marks": [args.stop_tokens // 2, args.stop_tokens], "play_backstop_seconds": args.play_backstop,
            "his_wall_seconds": HIS_WALL, "his_tail_fade_starts_seconds": int(0.8 * HIS_WALL),
            "stop_grace_seconds": grace, "timeout_seconds": setup + args.play_backstop + grace + 300,
            "assets": ASSETS, "gateway": {"script": "b2d-24h-v2 b200_gateway.py", "arc_agi": "0.9.9",
                                          "only_reset_levels": True, "roster_bundle": ref["name"]}},
    }
    if study["dfr_exact"]["instances"] != 110 or len(study["games"]) != 20:
        raise SystemExit("roster is not the 110 clones of 20 games")
    if args.equal_24h:
        timeout = 86_400
        backstop = timeout - setup - grace - 300
        study["dfr_exact"].update(
            mode="equal-24h", notebook_file=notebook_file, instances=args.equal_24h, stop_generated_tokens=None,
            marks=[], play_backstop_seconds=backstop, timeout_seconds=timeout, his_tail_fade_starts_seconds=None,
            equal={"park_tokens": args.park_tokens, "module": "dfr_equal.py",
                   "rule": "fewest generated tokens first; hand-over only at his drains and at game end; parked at "
                           "park_tokens until no unparked game waits"},
            edits=[{"cell": 16, "old": "        bm.solver.max_runtime_s_per_game = 532*60",
                    "new": f" bm.solver.max_runtime_s_per_game = {timeout}  # 24 h"}],
            blank_lines="his (dropped)", streams="10: his ARC3_MAX_ACTIVE_STREAMS and his server's --max-running-requests")
    if args.public:
        if {g[:4] for g in ref["games"]} & set(PUBLIC_EXCLUDED):
            raise SystemExit("an excluded game is in the 20 families")
        instances = len(ref["games"]) * 4
        wall = 532 * 60 * instances // 110  # his own formula in cell 16 with concurrency (25 - 5) * 4
        allowance = 1800  # his drain after the wall (438 s in his run), final saves, diagnostics and movies
        study["dfr_exact"] = {
            "mode": "public", "notebook": "dfranzen/arc-agi-3-milestone-2-solution version 3, pulled 11 October 2026",
            "notebook_file": notebook_file, "notebook_sha256": notebook_sha, "edits": [PUBLIC_EDIT],
            "instances": instances, "passes": 4, "stop_generated_tokens": None, "stop_at_backstop": False, "marks": [],
            "play_backstop_seconds": wall, "his_wall_seconds": wall, "his_tail_fade_starts_seconds": int(0.8 * wall),
            "his_public_wall_seconds": 532 * 60 * 100 // 110, "stop_grace_seconds": grace,
            "timeout_seconds": setup + wall + allowance + 300, "kernel_extra": ["imageio-ffmpeg"],
            "assets": ASSETS, "games": "his competition environment files, offline in his kernel (arc_agi 0.9.8)"}
    (out / "configs/comparison.json").write_text(json.dumps(study, indent=1) + "\n")
    files = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "MANIFEST.json").write_text(json.dumps({"study": out.name, "dependencies_prepared": True,
                                                   "files": files}, indent=1) + "\n")
    print(json.dumps({"bundle": str(out), "files": len(files), "wheels": len(list((out / "wheels").glob("*.whl"))),
                      "timeout_seconds": study["dfr_exact"]["timeout_seconds"],
                      "gateway_sha256": files.get("gateway.py"), "notebook_sha256": notebook_sha,
                      "manifest_sha256": sha(out / "MANIFEST.json")}))


if __name__ == "__main__":
    main()
