"""The <=5-min CPU launch gate that the RL bundle builder, pf_guard and every
GPU launcher run before `records.py reserve`. Steps run in parallel, each under its own time limit, all inside one
budget (default 300 s); any failure or timeout blocks the launch (exit 1). Receipt: runs/ops/gate/<run>.json.

  cpu_gate.py RUN --bundle DIR [--config CFG.json --config-class launch/rl/launch_hero.py:HeroConfig]
              [--ref SHA] [--schedsim "CMD"] [--fake-server "CMD"] [--patch-check "CMD"] [--step NAME="CMD" ...]
              [--budget 300]

Built-in steps:
  config     the typed launch config's dry run (runops.launchcfg: anchor, roster vs bundle, data, gate, open attempt)
  ci         the source commit is CI-green (fast suite incl. the gloo multi-rank tests) and the bundle byte-compiles
Arm steps (a command each; exit 0 = pass, its last output line is the reason on a failure):
  schedsim     the admission/scheduler simulation on the bundle's settings
  fake-server  stand-in server: start, one step, kill, resume from the checkpoint (CPU)
  patch-check  every runtime patch's hash guard and apply step against the exact serving image's files (no GPU):
               ARC_SERVING_PATCH_CHECK=1 boot of the bundle's serving setup on a CPU container of the same image
"""
import argparse
import concurrent.futures as cf
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

ROOT = paths.ROOT_PATH
sys.path.insert(0, str(ROOT / "src"))


def step_config(cfg_path, cls_spec, bundle_roster=None):
    from arc_infra.runops import launchcfg

    file, _, name = cls_spec.partition(":")
    spec = importlib.util.spec_from_file_location("gate_cfg_mod", file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, name)
    cfg = launchcfg.load(cls, cfg_path)
    roster = mod.bundle_roster(cfg) if hasattr(mod, "bundle_roster") else bundle_roster
    probs = launchcfg.dry_run(cfg, roster_ids=roster, modal_bin=str(ROOT / ".venv/bin/modal"))
    if hasattr(mod, "checks"):  # the arm's own config checks (e.g. launch/pth/heal_config.py)
        probs = list(mod.checks(cfg)) + probs
    return (not probs), "; ".join(probs) or f"config {cfg.run} OK (roster {len(cfg.roster.ids)}, anchor {cfg.anchor.run})"


def step_ci(ref, bundle, cwd):
    r = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "tools/ops/ci.py"), "require", ref, "--compile",
                        str(bundle)], capture_output=True, text=True, cwd=cwd)
    out = (r.stdout + r.stderr).strip().splitlines()
    return r.returncode == 0, " | ".join(out[-3:])[:400]


PYTEST_FAILED = re.compile(r"\b[1-9]\d* (failed|errors?)\b.* in \d")  # a pytest summary line with failures


def step_cmd(cmd, cwd):
    """bash -o pipefail: a failing command before `| tail -n` fails the step (v6m2 25 Oct: `| tail -1` hid pytest's
    exit code, "PASS 3 failed, 160 passed"; 25 Oct: `| tail -3` hid a CONFIG REFUSED). Exit 141 (a producer cut off by
    `| head`, SIGPIPE) passes. Also failed on a pytest summary line with failures."""
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True, cwd=cwd)
    out = (r.stdout + r.stderr).strip().splitlines()
    bad = [line for line in out if PYTEST_FAILED.search(line)]
    ok = r.returncode in (0, 141) and not bad
    why = bad[-1] if bad else out[-1] if out else f"exit {r.returncode}"
    return ok, why[:400]


def run_gate(run, steps, budget):
    """steps: {name: callable -> (ok, reason)}. All in parallel; a step still running at the budget fails."""
    t0, res = time.time(), {}
    ex = cf.ThreadPoolExecutor(max(1, len(steps)))
    futs = {ex.submit(fn): name for name, fn in steps.items()}
    done, pending = cf.wait(futs, timeout=budget)
    for f in done:
        try:
            ok, why = f.result()
        except Exception as e:  # noqa: BLE001 - a crashed step is a failed step
            ok, why = False, f"crashed: {e!r}"[:400]
        res[futs[f]] = {"ok": bool(ok), "why": why}
    for f in pending:
        res[futs[f]] = {"ok": False, "why": f"over the {budget:.0f} s gate budget"}
    ex.shutdown(wait=False, cancel_futures=True)
    ok = all(r["ok"] for r in res.values()) and bool(res)
    return {"run": run, "ok": ok, "seconds": round(time.time() - t0, 1), "t": time.time(), "steps": res}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--config")
    ap.add_argument("--config-class")
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--schedsim")
    ap.add_argument("--fake-server")
    ap.add_argument("--patch-check")
    ap.add_argument("--step", action="append", default=[], help='NAME="CMD"')
    ap.add_argument("--budget", type=float, default=300.0)
    a = ap.parse_args()
    cwd = str(Path(a.cwd).resolve())
    steps = {"ci": lambda: step_ci(a.ref, a.bundle, cwd)}
    if a.config:
        if not a.config_class:
            sys.exit("--config needs --config-class FILE:Class")
        steps["config"] = lambda: step_config(a.config, a.config_class)
    for name, cmd in (("schedsim", a.schedsim), ("fake-server", a.fake_server), ("patch-check", a.patch_check),
                      *(s.split("=", 1) for s in a.step)):
        if cmd:
            steps[name] = (lambda c=cmd: step_cmd(c, cwd))
    rec = run_gate(a.run, steps, a.budget)
    out = ROOT / "runs/ops/gate"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.run}.json").write_text(json.dumps(rec, indent=1))
    for name, r in sorted(rec["steps"].items()):
        print(f"{'PASS' if r['ok'] else 'FAIL'} {name:12s} {r['why']}")
    print(f"CPU GATE {'PASS' if rec['ok'] else 'FAIL'} {a.run} in {rec['seconds']} s (receipt {out / (a.run + '.json')})")
    sys.exit(0 if rec["ok"] else 1)


if __name__ == "__main__":
    main()
