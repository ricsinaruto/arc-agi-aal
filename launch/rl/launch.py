"""The one RL launcher: every run is launch/rl/runs/<run>.json, built by launch/rl/build.py.

  launch.py check CONFIG              config checks + typed dry run (anchor, roster vs bundle, data, caps gate) and the
                                      bundle built from this config's bundle part (MANIFEST spec_sha256)
  launch.py preflight CONFIG [TAG..]  the CPU preflight legs (config.preflight; all by default) of the exact bundle
  launch.py verdict CONFIG [TAG..]    their verdicts (modal_rl1 --verdict: docs/validation-rl/preflight-<bundle><tag>)
  launch.py launch CONFIG             check -> reserve -> config into the record -> knobs put -> modal_rl1 run (modal)
                                      or nlaunch.sh (nebius) -> app id / start into the record -> watcher started
  launch.py env CONFIG                the launch env (modal backend), for a look or a manual run

Logs: runs/ops/launch/<run>*.log in the main checkout. The watcher plan goes to runs/ops/watch-plans/<run>.json."""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402
import config as C  # noqa: E402

PY = str(C.MAIN / ".venv/bin/python")
MODAL = str(C.MAIN / ".venv/bin/modal")
REC = str(C.MAIN / "tools/ops/records.py")
LOGS = C.MAIN / "runs/ops/launch"


def bundle_dir(cfg):
    return B.DIST / cfg.bundle


def bundle_roster(cfg):
    """The ids the bundle plays: a level bench's start points (a read), else the study's games."""
    study = json.loads((bundle_dir(cfg) / "configs/comparison.json").read_text())
    trial = (study.get("trials") or [{}])[0]
    if trial.get("kind") == "level-bench" and (study.get("level_bench") or {}).get("start_points"):
        pts = json.loads((bundle_dir(cfg) / study["level_bench"]["start_points"]).read_text())["points"]
        return list(dict.fromkeys(p["game"].split("-")[0] for p in pts))
    return [g.split("-")[0] for g in study["games"]]


def check(cfg, cfg_path, gate=True):
    probs = C.checks(cfg)
    b = bundle_dir(cfg)
    if not (b / "MANIFEST.json").is_file():
        return probs + [f"bundle {b} not built (launch/rl/build.py {cfg_path})"]
    m = json.loads((b / "MANIFEST.json").read_text())
    if m.get("spec_sha256") != B.spec_sha256(cfg):
        probs.append(f"bundle {b} was not built from this config (rebuild: launch/rl/build.py {cfg_path})")
    want = cfg.source if len(cfg.source) == 40 else B.git("rev-parse", f"{cfg.source}^{{commit}}")
    if m.get("git_commit") != want:
        probs.append(f"bundle {b} source {str(m.get('git_commit'))[:10]} != config source {cfg.source[:10]}")
    try:
        roster = bundle_roster(cfg)
    except (OSError, KeyError, ValueError) as e:
        roster, probs = None, probs + [f"bundle roster unreadable: {e!r}"[:200]]
    return probs + C.launchcfg.dry_run(cfg, roster_ids=roster, modal_bin=MODAL, gate=gate)


def modal_env(cfg, extra=None):
    e = {"ARC_RL1_BUNDLE": str(bundle_dir(cfg)), "ARC_RL1_RESUME_FROM": ",".join(cfg.launch.resume_from),
         "ARC_RL1_AUTHORIZATION": cfg.authorization, **cfg.launch.env, **(extra or {})}  # authorization: the record
    return {k: v for k, v in e.items() if v != ""}


def sh(cmd, log=None, env=None, check=True):
    with open(log, "w") if log else open(os.devnull, "w") as f:
        r = subprocess.run(cmd, stdout=f if log else subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           env={**os.environ, **(env or {})}, cwd=str(C.CHECKOUT))
    if check and r.returncode:
        tail = Path(log).read_text()[-600:] if log else r.stdout[-600:]
        raise SystemExit(f"{cmd[0] if cmd[0] != PY else cmd[1]} failed:\n{tail}")
    return r


def preflight(cfg, tags=()):
    LOGS.mkdir(parents=True, exist_ok=True)
    out = []
    for tag in tags or list(cfg.preflight):
        if tag not in cfg.preflight:
            raise SystemExit(f"no preflight leg {tag!r}: {sorted(cfg.preflight)}")
        log = LOGS / f"{cfg.bundle}{tag}-preflight.log"
        env = modal_env(cfg, {**cfg.preflight[tag], "ARC_RL1_PREFLIGHT_TAG": tag})
        sh([MODAL, "run", "--detach", "scripts/posttrain/modal_rl1.py", "--run-id", cfg.bundle, "--preflight-run"], log, env)
        text = log.read_text()
        app = re.search(r"\bap-[A-Za-z0-9]{20,}", text)
        if not app or '"function_call_id"' not in text:
            raise SystemExit(f"preflight {tag} not submitted: {text[-600:]}")
        sh([PY, REC, "cpu-app", app.group(0)], check=False)
        out.append((tag, app.group(0)))
        print(f"preflight {cfg.bundle}{tag} {app.group(0)}")
    return out


def background(cmd, log):
    subprocess.Popen(cmd, stdout=open(log, "a"), stderr=subprocess.STDOUT, start_new_session=True, cwd=str(C.MAIN))


def verdict(cfg, tags=()):
    """Each leg's verdict: modal_rl1 --verdict (rl-loop legs), else its exit code (modal-identity.json): a patch-check
    leg (serving prep only) or another trial (e.g. a read's level bench, judged by its own gate step)."""
    import tempfile

    for tag in tags or list(cfg.preflight):
        leg = cfg.preflight[tag]
        if leg.get("ARC_RL1_PATCH_CHECK") == "1" or leg.get("ARC_RL1_PREFLIGHT_TRIAL", "rl-loop") != "rl-loop":
            with tempfile.TemporaryDirectory() as t:
                sh([MODAL, "volume", "get", "--force", "arc-prolong-runs",
                    f"validation/preflight/{cfg.bundle}{tag}/modal-identity.json", t], check=False)
                f = Path(t) / "modal-identity.json"
                ident = json.loads(f.read_text()) if f.is_file() else {}
            ok = ident.get("exit_code") == 0 and not ident.get("error")
            print(f"verdict {cfg.bundle}{tag}: {'pass' if ok else 'fail'} (exit {ident.get('exit_code')}; a patch check or "
                  "a non-rl-loop trial: its content is the arm's CPU gate step)")
            continue
        log = LOGS / f"{cfg.bundle}{tag}-verdict.log"
        r = sh([MODAL, "run", "scripts/posttrain/modal_rl1.py", "--run-id", cfg.bundle, "--verdict"], log,
               modal_env(cfg, {"ARC_RL1_PREFLIGHT_TAG": tag}), check=False)
        v = re.findall(r'"verdict": "(\w+)"', log.read_text())
        print(f"verdict {cfg.bundle}{tag}: {v[-1] if v else 'none'} (exit {r.returncode}, {log})")


def watch(cfg, app, epoch):
    """watch.py on the plan; watch.export {passes, base, continues}: the dashboard exporter (export_rl.py) beside it."""
    plan = {"run": cfg.run, "arm": cfg.arm, "platform": cfg.platform, "app_id": app, "launched": epoch,
            **C.launchcfg.to_dict(cfg.watch)}
    export = plan.pop("export", None)
    p = C.MAIN / "runs/ops/watch-plans" / f"{cfg.run}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan, indent=1) + "\n")
    background([PY, "-u", str(C.MAIN / "tools/ops/watch.py"), "run", str(p)], LOGS / f"{cfg.run}-watch.log")
    if export:
        background([PY, "-u", str(C.MAIN / "tools/ops/export_rl.py"), cfg.run,
                    *[f"--{k}={v}" for k, v in export.items()]], LOGS / f"{cfg.run}-export.log")
    return p


def launch(cfg, cfg_path):
    probs = check(cfg, cfg_path)
    if probs:
        raise SystemExit("LAUNCH REFUSED:\n  " + "\n  ".join(probs))
    LOGS.mkdir(parents=True, exist_ok=True)
    print(C.launchcfg.report(cfg).splitlines()[-1])
    if cfg.launch.backend == "modal":
        sh([PY, REC, "reserve", cfg.run, "--arm", cfg.arm, "--platform", cfg.platform, "--gpu",
            f"{cfg.gpu_count} x {cfg.gpu}", "--gpu-count", str(cfg.gpu_count), "--bound", str(cfg.bound_usd),
            "--max-seconds", str(cfg.max_seconds), "--kind", cfg.launch.kind, "--auth", cfg.authorization,
            "--what", cfg.what])
        C.launchcfg.remember(cfg)
        if cfg.launch.knobs is not None:
            k = LOGS / f"{cfg.run}-knobs.json"
            k.write_text(json.dumps(dict(cfg.launch.knobs), indent=1))
            sh([PY, str(HERE / "put_knobs.py"), "put", cfg.run, str(k)])
        log = LOGS / f"{cfg.run}-launch.log"
        r = sh([MODAL, "run", "--detach", "scripts/posttrain/modal_rl1.py", "--run-id", cfg.run], log, modal_env(cfg),
               check=False)
        text = log.read_text()
        if re.search(r"^launch gate:", text, re.M) or '"state": "submitted"' not in text or r.returncode:
            raise SystemExit(f"LAUNCH FAILED (exit {r.returncode}): {text[-800:]}")
        sh([PY, REC, "attach", cfg.run, str(log)])
        app = re.search(r"\bap-[A-Za-z0-9]{20,}", text).group(0)
    else:
        nb = dict(cfg.launch.nebius)
        log = LOGS / f"{cfg.run}-launch.log"
        sh([str(C.MAIN / "tools/ops/nlaunch.sh"), str(C.CHECKOUT / "scripts/modal_validation.py"), nb["test"],
            str(bundle_dir(cfg)), cfg.run, *(["--reason", nb["reason"]] if nb.get("reason") else []),
            *(["--approval", nb["approval"]] if nb.get("approval") else [])], log, {"ARC_WHAT": cfg.what})
        if " exit=0 " not in log.read_text():
            raise SystemExit(f"LAUNCH FAILED: {log.read_text()[-800:]}")
        C.launchcfg.remember(cfg)
        app = None
    epoch = int(time.time())
    sh([PY, REC, "queue", cfg.run, "--set", f"next={json.dumps(cfg.launch.queue_next)}",
        "--set", f"start={json.dumps(time.strftime('%H:%M', time.gmtime(epoch)))}"], check=False)
    plan = watch(cfg, app, epoch)
    print(f"launched {cfg.run} {app or '(nebius VM)'} {time.strftime('%H:%M:%S', time.gmtime(epoch))} UTC {epoch}; "
          f"watcher plan {plan}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("check", "preflight", "verdict", "launch", "env"))
    ap.add_argument("config")
    ap.add_argument("tags", nargs="*")
    a = ap.parse_args(argv)
    path = Path(a.config).resolve()
    cfg = C.load(path)
    if a.cmd == "check":
        probs = check(cfg, path)
        print(C.launchcfg.report(cfg).splitlines()[-1])
        if probs:
            raise SystemExit("CONFIG REFUSED:\n  " + "\n  ".join(probs))
        print(f"config {cfg.run} OK (bundle {bundle_dir(cfg)})")
    elif a.cmd == "env":
        print("\n".join(f"export {k}={json.dumps(v)}" for k, v in modal_env(cfg).items()))
    elif a.cmd == "preflight":
        preflight(cfg, a.tags)
    elif a.cmd == "verdict":
        verdict(cfg, a.tags)
    else:
        launch(cfg, path)


if __name__ == "__main__":
    main()
