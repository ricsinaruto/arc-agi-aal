"""CI: the fast suite (incl. the gloo multi-rank tests) on every pushed head of the watched branches, on
Modal CPU with the shared image (runops.ciimage via test_modal.py); a red head blocks bundle builds.

  ci.py poll [--max 2]          dash loop, each pass: fetch; start a run for each watched head with no result yet
  ci.py run REF                 run the suite on REF now (blocking) and record the verdict
  ci.py require [REF] [--compile DIR ...]   bundle builders: exit 0 if REF (default HEAD of the cwd checkout) is green
                                and each DIR byte-compiles (python -m compileall); red -> exit 1 with the failures;
                                no result yet -> runs it now (blocking, ~5 min)
  ci.py status                  latest verdict per watched branch (exit 1 if any red)

Results: runs/ops/ci/<sha>.json {"sha", "ref", "verdict": green|red|error, "summary", "failed": [...], "t"} (main
checkout). Watched: BRANCHES below plus --branch.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

ROOT = paths.ROOT_PATH
OUT = ROOT / "runs/ops/ci"
PY = str(ROOT / ".venv/bin/python")
BRANCHES = ("main", "rl-async", "sft-heal", "nebius", "nebius-opd", "rl-launch", "pth-launch")


def git(*a, cwd=ROOT, check=False):
    return subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, text=True, check=check).stdout.strip()


def result(sha):
    p = OUT / f"{sha}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def run(ref, label=None):
    """Blocking: test_modal on the committed tree of ``ref``; writes and returns the result."""
    sha = git("rev-parse", ref) or ref
    OUT.mkdir(parents=True, exist_ok=True)
    label = label or f"ci-{sha[:10]}"
    (OUT / f"{sha}.running").write_text(str(os.getpid()))
    try:
        p = subprocess.run([PY, str(ROOT / "tools/ops/test_modal.py"), "--ref", sha, "--label", label],
                           capture_output=True, text=True, cwd=ROOT)
    finally:
        (OUT / f"{sha}.running").unlink(missing_ok=True)
    lines = (p.stdout + p.stderr).strip().splitlines()
    failed = [x.split(" - ")[0] for x in lines if x.startswith(("FAILED", "ERROR"))]
    log = paths.SCRATCH / "tests" / f"{label}.log"
    tail = log.read_text(errors="replace").splitlines()[-400:] if log.exists() else []  # stdout may end in teardown noise
    summary = next((x for x in reversed(lines + tail) if re.search(r"\d+ (passed|failed)", x)), lines[-1] if lines else "")
    verdict = "green" if p.returncode == 0 else "red" if failed or re.search(r"\d+ failed", summary) else "error"
    r = {"sha": sha, "ref": ref, "verdict": verdict, "summary": summary[:300], "failed": failed[:200], "t": time.time(),
         "log": str(paths.SCRATCH / "tests" / f"{label}.log"), "forks": forks(sha)}
    (OUT / f"{sha}.json").write_text(json.dumps(r, indent=1))
    return r


def forks(sha):
    """fork_report.py on the committed tree of ``sha``: shared functions whose bodies differ between RL's and PTH's
    copies of the forked trainer files (a report; never changes the verdict)."""
    import tempfile

    names = (git("ls-tree", "--name-only", sha, "scripts/posttrain/pth/") or "").split()
    if not names:
        return {}
    try:
        import fork_report
        with tempfile.TemporaryDirectory() as d:
            for n in names:
                for src in (n, "scripts/posttrain/" + Path(n).name):
                    text = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha}:{src}"], capture_output=True).stdout
                    (Path(d) / src).parent.mkdir(parents=True, exist_ok=True)
                    (Path(d) / src).write_bytes(text)
            return {k: {"differ": v["differ"], "same": v["same"]} for k, v in fork_report.report(d).items()}
    except Exception as e:  # noqa: BLE001 - a report only
        return {"error": repr(e)[:200]}


def heads(extra=()):
    git("fetch", "-q", "origin")
    out = {}
    for b in (*BRANCHES, *extra):
        sha = git("rev-parse", "--verify", "-q", f"origin/{b}")
        if sha:
            out[b] = sha
    return out


def poll(maximum=2, extra=()):
    """Start (detached) a run for each watched head without a result; at most ``maximum`` running at once."""
    OUT.mkdir(parents=True, exist_ok=True)
    running = [p for p in OUT.glob("*.running") if _alive(p)]
    started = []
    for b, sha in heads(extra).items():
        if len(running) + len(started) >= maximum:
            break
        if result(sha) or (OUT / f"{sha}.running").exists():
            continue
        log = open(OUT / f"{sha}.out", "w")
        subprocess.Popen([PY, __file__, "run", sha], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                         start_new_session=True)
        started.append(f"{b}@{sha[:10]}")
    return started


def _alive(p):
    try:
        os.kill(int(p.read_text()), 0)
        return True
    except (OSError, ValueError):
        p.unlink(missing_ok=True)
        return False


def compile_dirs(dirs):
    bad = []
    for d in dirs:
        r = subprocess.run([sys.executable, "-m", "compileall", "-q", "-x", r"/(\.venv|node_modules)/", str(d)],
                           capture_output=True, text=True)
        if r.returncode:
            bad.append(f"{d}: " + (r.stdout + r.stderr).strip()[-600:])
    return bad


def require(ref, dirs=()):
    sha = git("rev-parse", ref, cwd=Path.cwd()) or ref
    probs = compile_dirs(dirs)
    r = result(sha)
    if r is None:
        if git("cat-file", "-t", sha) != "commit":
            git("fetch", "-q", "origin")
        if git("cat-file", "-t", sha) != "commit":
            probs.append(f"{sha[:10]}: not pushed (CI tests pushed commits only: push first)")
        else:
            print(f"ci: no result for {sha[:10]} yet, running the suite now (~5 min)", flush=True)
            r = run(sha)
    if r and r["verdict"] != "green":
        probs.append(f"{sha[:10]} is {r['verdict']}: {r['summary']}\n  " + "\n  ".join(r["failed"][:30]))
    return probs


def status(extra=()):
    red = False
    for b, sha in heads(extra).items():
        r = result(sha)
        state = r["verdict"] if r else "running" if (OUT / f"{sha}.running").exists() else "pending"
        red |= state in ("red", "error")
        age = f"{(time.time() - r['t']) / 3600:.1f} h" if r else ""
        print(f"{b:14s} {sha[:10]} {state:8s} {age:>7s}  {(r or {}).get('summary', '')[:90]}")
    return red


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("poll")
    p.add_argument("--max", type=int, default=2)
    p.add_argument("--branch", action="append", default=[])
    r = sub.add_parser("run")
    r.add_argument("ref")
    q = sub.add_parser("require")
    q.add_argument("ref", nargs="?", default="HEAD")
    q.add_argument("--compile", nargs="*", default=[])
    s = sub.add_parser("status")
    s.add_argument("--branch", action="append", default=[])
    a = ap.parse_args()
    if a.cmd == "poll":
        started = poll(a.max, a.branch)
        print("ci: started " + (", ".join(started) if started else "nothing"))
    elif a.cmd == "run":
        res = run(a.ref)
        print(f"ci {res['verdict']} {res['sha'][:10]}: {res['summary']}")
        sys.exit(0 if res["verdict"] == "green" else 1)
    elif a.cmd == "require":
        probs = require(a.ref, a.compile)
        if probs:
            print("CI RED: bundle build blocked\n" + "\n".join(probs))
            sys.exit(1)
        print("ci green")
    elif a.cmd == "status":
        sys.exit(1 if status(a.branch) else 0)


if __name__ == "__main__":
    main()
