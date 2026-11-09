"""CI: the fast suite (incl. the gloo multi-rank tests) on every pushed head of the watched branches, on
Modal CPU with the shared image (runops.ciimage via test_modal.py); a red head blocks bundle builds.

  ci.py poll [--max 2]          dash loop, each pass: fetch; start a run for each watched head with no result yet
  ci.py run REF                 run the suite on REF now (blocking) and record the verdict
  ci.py require [REF] [--compile DIR ...]   bundle builders: exit 0 if REF (default HEAD of the cwd checkout) is green
                                and each DIR byte-compiles (python -m compileall); red -> exit 1 with the failures;
                                no result yet -> runs it now (blocking, ~5 min)
  ci.py gate --builder NAME [REF] [--compile DIR ...]   every launch/rl, launch/pth bundle builder, run in the source
                                checkout: blocks on uncommitted src/ scripts/ files, an unpushed or red REF (as require).
                                ARC_CI_SKIP="<reason, >= 12 chars>" overrides it; each override is a row in
                                runs/ops/ci/overrides.jsonl and shows on the dashboard for 6 h
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
OVERRIDES = OUT / "overrides.jsonl"
CODE_DIRS = ("src", "scripts")  # what the builders copy into bundles
PY = str(ROOT / ".venv/bin/python")
BRANCHES = ("main", "rl-async", "sft-heal", "nebius", "nebius-opd", "trunk/rl", "trunk/sft")  # trunk/rl: RL builds from it (v6j);
# rl-launch / pth-launch dropped 25 Oct: idle ~19 h, their launchers live on trunk/rl


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


def require(ref, dirs=(), cwd=None):
    sha = git("rev-parse", ref, cwd=cwd or Path.cwd()) or ref
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


def gate(builder, ref="HEAD", cwd=None, dirs=(), env=None):
    """Problems blocking a bundle build from the checkout ``cwd`` ([] = go). ARC_CI_SKIP overrides, logged."""
    env = os.environ if env is None else env
    cwd = Path(cwd or Path.cwd()).resolve()
    sha = git("rev-parse", ref, cwd=cwd) or ref
    reason = (env.get("ARC_CI_SKIP") or "").strip()
    if reason:
        if len(reason) < 12:
            return [f"ARC_CI_SKIP={reason!r}: give the reason in words (>= 12 chars), it is logged"]
        row = {"t": time.time(), "utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()), "builder": builder, "sha": sha,
               "branch": git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd), "cwd": str(cwd), "reason": reason}
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OVERRIDES, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"CI GATE OVERRIDDEN for {builder} @ {sha[:10]}: {reason} (logged to {OVERRIDES})", file=sys.stderr)
        return []
    dirty = [x.split(maxsplit=1)[1] for x in git("status", "--porcelain", "--", *CODE_DIRS, cwd=cwd).splitlines() if x.strip()]
    if dirty:  # CI tests the commit; the builder would copy the working tree
        return [f"{cwd}: uncommitted files under {', '.join(CODE_DIRS)} (commit and push first): " + ", ".join(dirty[:8])]
    return require(sha, dirs, cwd=cwd)


def overrides(since):
    rows = []
    for x in (OVERRIDES.read_text().splitlines() if OVERRIDES.is_file() else []):
        try:
            r = json.loads(x)
        except ValueError:
            continue
        if r.get("t", 0) >= since:
            rows.append(r)
    return rows


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
    g = sub.add_parser("gate")
    g.add_argument("ref", nargs="?", default="HEAD")
    g.add_argument("--builder", required=True)
    g.add_argument("--compile", nargs="*", default=[])
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
    elif a.cmd == "gate":
        probs = gate(a.builder, a.ref, dirs=a.compile)
        if probs:
            print(f"CI GATE: {a.builder} bundle build blocked\n" + "\n".join(probs) +
                  '\n(override, logged: ARC_CI_SKIP="<reason>")', file=sys.stderr)
            sys.exit(1)
        print("ci gate: go")
    elif a.cmd == "status":
        sys.exit(1 if status(a.branch) else 0)


if __name__ == "__main__":
    main()
