"""The full test suite in one command, on a Modal CPU job with CPU torch (never on this machine):

  .venv/bin/python tools/ops/test_modal.py [--tree PATH | --ref BRANCH] [--label L] [-n N] [-- PYTEST ARGS]

Packs the checkout's tracked files (docs/, resources/, plots/ left out; --tree: its working tree, default this one;
--ref: a committed branch), runs `pytest -q -rfEs -p no:cacheprovider` (plus PYTEST ARGS) in the container and prints
the counts and failures. Full log and junit: <scratch>/tests/<label>.log / .xml. The app is registered as a CPU app.
-n N: pytest-xdist workers (default 8, --dist loadfile: a file's tests stay in order on one worker; 0 = serial). The image is the shared CI/preflight image (runops.ciimage).
Exit status: pytest's.
"""
import argparse
import io
import pathlib
import subprocess
import sys
import tarfile
import time

import modal

_here = pathlib.Path(__file__).resolve()
if modal.is_local() and len(_here.parents) > 2:
    sys.path.insert(0, str(_here.parents[2] / "src"))
from arc_infra.runops.ciimage import image as ci_image  # noqa: E402  (shipped into the image below)

SKIP = ("docs/", "resources/", "plots/", "pv-t37v2/", "externals/", "working notes.md")
KEEP = (".py",)  # under a skipped dir, these still ship (tests load scripts kept under docs/validation-*)
W = "/w/a/b/repo"  # tests look two levels above the checkout too
app = modal.App("arc-tests")
image = ci_image().add_local_python_source("arc_infra")


@app.function(image=image, cpu=(0.125, 8), timeout=3 * 3600)  # timeout: a crash ceiling only
def pytest_run(bundle: bytes, args: list) -> dict:
    import os

    os.makedirs(W, exist_ok=True)
    tarfile.open(fileobj=io.BytesIO(bundle)).extractall(W)
    for cmd in (["init", "-q", "-b", "main"], ["add", "-A"],  # bundle builders record `git rev-parse HEAD`
                ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "bundle"]):
        subprocess.run(["git", "-C", W, *cmd], check=True, capture_output=True)
    env = {**os.environ, "PYTHONPATH": f"{W}/src", "ARC_RECORDS_ROOT": W, "MPLBACKEND": "Agg"}
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--junitxml=/w/j.xml",
                        *args], cwd=W, env=env, capture_output=True, text=True)
    junit = open("/w/j.xml").read() if os.path.exists("/w/j.xml") else ""
    return {"rc": p.returncode, "log": p.stdout + p.stderr, "junit": junit, "seconds": time.time() - t0}


def bundle(tree=None, ref=None):
    root = pathlib.Path(tree or pathlib.Path(__file__).resolve().parents[2])
    if ref:
        tar = subprocess.run(["git", "-C", str(root), "archive", ref], capture_output=True, check=True).stdout
        src = tarfile.open(fileobj=io.BytesIO(tar))
        out = io.BytesIO()
        with tarfile.open(fileobj=out, mode="w:gz") as dst:
            for m in src.getmembers():
                if not m.name.startswith(SKIP) or m.name.endswith(KEEP):
                    dst.addfile(m, src.extractfile(m) if m.isfile() else None)
            _games(dst)
        return out.getvalue()
    files = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as dst:
        for f in files.decode().split("\0"):
            if f and (not f.startswith(SKIP) or f.endswith(KEEP)) and (root / f).is_file():
                dst.add(root / f, arcname=f)
        _games(dst)
    return out.getvalue()


def _games(dst):
    """The local, untracked ARC games (environment_files) that tests play; an arm worktree uses the main checkout's."""
    here = pathlib.Path(__file__).resolve().parents[2]
    games = here / "environment_files"
    if not games.is_dir():
        common = subprocess.run(["git", "-C", str(here), "rev-parse", "--git-common-dir"], capture_output=True,
                                text=True).stdout.strip()
        games = (here / common).resolve().parent / "environment_files"
    if games.is_dir():
        dst.add(games, arcname="environment_files")
    generated = games.parent / "data/posttrain/games/environment_files"  # the envs arm's generated games (untracked)
    if generated.is_dir():
        dst.add(generated, arcname="data/posttrain/games/environment_files")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tree")
    ap.add_argument("--ref")
    ap.add_argument("--label")
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("pytest_args", nargs="*")
    a = ap.parse_args()
    label = a.label or (a.ref or pathlib.Path(a.tree or ".").resolve().name).replace("/", "-")
    data = bundle(a.tree, a.ref)
    print(f"bundle {len(data) / 1e6:.1f} MB; running pytest on Modal CPU ({label})", flush=True)
    with app.run():
        try:
            from arc_infra.runops import records

            records.register_cpu_app(app.app_id)
        except Exception as e:  # the ledger is best effort here; the CPU job itself is cents
            print(f"(cpu-app registration skipped: {e!r})")
        r = pytest_run.remote(data, (["-n", str(a.n), "--dist", "loadfile"] if a.n else []) + a.pytest_args)
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import paths  # local only (the container has no tools/ops)

    out = paths.SCRATCH / "tests"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{label}.log").write_text(r["log"])
    (out / f"{label}.xml").write_text(r["junit"])
    lines = r["log"].rstrip().splitlines()
    short = [x for x in lines if x.startswith(("FAILED", "ERROR"))]
    print("\n".join(short[:200]))
    print(f"{lines[-1] if lines else '(no output)'}  [{r['seconds'] / 60:.1f} min; log {out / (label + '.log')}]")
    return r["rc"]


if __name__ == "__main__":
    sys.exit(main())
