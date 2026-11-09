"""The one RL launcher: every run is launch/rl/runs/<run>.json, built by launch/rl/build.py.

  launch.py check CONFIG              config checks + typed dry run (anchor, roster vs bundle, data, caps gate) and the
                                      bundle built from this config's bundle part (MANIFEST spec_sha256)
  launch.py preflight CONFIG [TAG..]  the CPU preflight legs (config.preflight; all by default) of the exact bundle
  launch.py verdict CONFIG [TAG..]    their verdicts (modal_rl1 --verdict: docs/validation-rl/preflight-<bundle><tag>)
                                      A kill/resume leg (``ARC_RL1_PREFLIGHT_KILL_S``: s, a standard leg)
                                      is cancelled at +s (containers terminated), resubmitted with an ARC_RESTART_ACK
                                      and passes only if the second attempt resumed (seq_resume) and passed the judge
  launch.py launch CONFIG             check -> reserve -> config into the record -> knobs put -> modal_rl1 run (modal)
                                      or nlaunch.sh (nebius) -> app id / start into the record -> watcher plan
                                      written (the agent starts watch.py run PLAN as its own background task)
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
SHARED_VOLUME = "arc-prolong-runs"


def run_volume(cfg):
    """The run's Modal volume: launch.env ARC_RL1_VOLUME "run" = its own v2 arc-run-<run> (one volume per
    run; its preflight legs and attempts share it), else the shared arc-prolong-runs."""
    if cfg.launch.env.get("ARC_RL1_VOLUME") != "run":
        return SHARED_VOLUME
    from arc_infra.runops import records
    return records.run_volume_name(cfg.run)


def input_volumes(cfg):
    """The volumes a per-run config reads at /in/<vol>/ (study, launch env, resume chain, preflight legs), read-only."""
    text = json.dumps([cfg.study, cfg.launch.env, cfg.launch.resume_from, cfg.preflight], default=dict)
    return sorted(set(re.findall(r"/in/([A-Za-z0-9._-]+)/", text)))


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


# v17 a5 (30 Oct): a window the planner cannot fit even fully checkpointed (n=0, its floor) runs anyway; U20's tails
# up to lvl_window 47104 predicted 253.8 GiB and OOMed at 258.4 allocated. n=0 peak = base + gap + a x tail/1K with a3's
# worst measured base 110.97, the gap prior 4.4 and a3's raised a 3.0011 (rl1_memory.ckpt_peak); the cap must keep it
# CKPT_ROOM GiB under the budget. R3 adds <= 0.36 GiB per window (3 held int16 [T, 48, 10] tables + router temporaries).
CKPT_FLOOR = {"base": 110.97, "gap": 4.4, "a": 3.0011, "r3": 0.36}
CKPT_ROOM = 5.0


def n0_peak(window_tokens):
    f = CKPT_FLOOR
    return f["base"] + f["gap"] + f["a"] * window_tokens / 1024.0 + f["r3"]


def ckpt_floor_problems(cfg):
    """The training windows' length cap (study.rl_loop.lvl_window) against the adaptive-ckpt budget."""
    loop = (cfg.study or {}).get("rl_loop") or {}
    window = loop.get("lvl_window")
    if not window or not loop.get("adaptive_ckpt", True):
        return []
    knobs = (cfg.launch.knobs or {}).get("seq") or {}
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from arena import rl_loop

    budget = float(knobs.get("ckpt_budget_gib") or rl_loop.FAST["ckpt_budget_gib"])
    peak = n0_peak(int(window))
    if peak > budget - CKPT_ROOM:
        return [f"lvl_window {window}: a fully checkpointed window peaks at {peak:.1f} GiB > budget {budget:g} - "
                f"{CKPT_ROOM:g} (cap it near {int((budget - CKPT_ROOM - n0_peak(0)) / CKPT_FLOOR['a'] * 1024)})"]
    return []


R3_TRAINING = "238b208ef"  # rl_seq trains on the served routing (mg.served_route) from this commit on


def r3_problems(cfg, roots=None, ancestor=None):
    """R3 gate (real-v9 28 Oct: its first update trained real-v7's route-less groups on the R3-off real-v6
    checkpoint): an rl_seq training config serves and trains on the served routing, and every run it resumes was
    built from a source with R3 in training (its checked-in config's source descends from R3_TRAINING)."""
    mg = (cfg.study.get("rl_loop") or {}).get("mg") or {}
    red = mg.get("redesign")
    if not isinstance(red, dict) or red.get("play_only"):
        return []
    probs = [] if mg.get("served_route") and (cfg.study.get("server") or {}).get("served_route") else [
        "R3: mg.served_route and server.served_route must be on for an rl_seq training run"]
    ancestor = ancestor or (lambda a, b: subprocess.run(["git", "-C", str(C.CHECKOUT), "merge-base", "--is-ancestor", a, b],
                                                        capture_output=True).returncode == 0)
    for root in cfg.launch.resume_from if roots is None else roots:
        run = re.sub(r"(-attempt\d+|-lost-\d+)+$", "", Path(root).name)  # an attempt / preempted folder: its run's config
        path = HERE / "runs" / f"{run}.json"
        if not path.is_file():
            probs.append(f"R3: resume {root}: no checked-in config to prove its routing (refused)")
            continue
        source = json.loads(path.read_text()).get("source") or ""
        if not ancestor(R3_TRAINING, source):
            probs.append(f"R3: resume {root}: built from {source[:10]}, before R3 in training ({R3_TRAINING}); its "
                         "checkpoint and pending groups carry no served routes")
    return probs


def check(cfg, cfg_path, gate=True):
    probs = C.checks(cfg) + [f"input pending: {p}" for p in C.pending(cfg)] + r3_problems(cfg) + ckpt_floor_problems(cfg)
    for tag, leg in cfg.preflight.items():  # a preflight leg's resume roots too (N11-style)
        roots = [r for r in (leg.get("ARC_RL1_RESUME_FROM") or "").split(",") if r]
        probs += [f"preflight {tag}: {p}" for p in r3_problems(cfg, roots) if roots and p.startswith("R3: resume")]
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
    if run_volume(cfg) != SHARED_VOLUME:
        e.update(ARC_RL1_VOLUME=run_volume(cfg), ARC_RL1_INPUT_VOLUMES=",".join(input_volumes(cfg)))
    return {k: v for k, v in e.items() if v != ""}


def sh(cmd, log=None, env=None, check=True):
    with open(log, "w") if log else open(os.devnull, "w") as f:
        r = subprocess.run(cmd, stdout=f if log else subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           env={**os.environ, **(env or {})}, cwd=str(C.CHECKOUT))
    if check and r.returncode:
        tail = Path(log).read_text()[-600:] if log else r.stdout[-600:]
        raise SystemExit(f"{cmd[0] if cmd[0] != PY else cmd[1]} failed:\n{tail}")
    return r


KILL = "ARC_RL1_PREFLIGHT_KILL_S"  # a launcher-side leg key (never sent to the container)


def preflight_env(cfg, tag):
    """A leg's env: the launcher's own, the leg's stub settings and the launch's knobs (real-v2 N1, 27 Oct: without
    the 200-try put the leg sat at mg.active_tries 120 = the back-off floor and could never cut)."""
    knobs = {} if cfg.launch.knobs is None else {"ARC_RL1_PREFLIGHT_KNOBS": json.dumps(dict(cfg.launch.knobs))}
    leg = {k: v for k, v in cfg.preflight[tag].items() if k != KILL}
    if KILL in cfg.preflight[tag]:  # the kill/resume leg plays on after its resume (4 steps; the kill lands at the
        # first checkpoint) so its restored attempts are judged: seeded, no seed-wait timeout, no digest-gate drop
        leg = {"ARC_RL1_PREFLIGHT_MAX_STEPS": "4", "ARC_RL1_PREFLIGHT_RESTORE_CHECK": "1", **leg}
    return modal_env(cfg, {**knobs, **leg, "ARC_RL1_PREFLIGHT_TAG": tag})


def leg_record(cfg, tag, suffix=""):
    return C.MAIN / "docs/validation-rl" / f"preflight-{cfg.bundle}{tag}{suffix}.json"


def submit(cfg, tag, extra=None, log_suffix=""):
    log = LOGS / f"{cfg.bundle}{tag}{log_suffix}-preflight.log"
    sh([MODAL, "run", "--detach", "scripts/posttrain/modal_rl1.py", "--run-id", cfg.bundle, "--preflight-run"], log,
       {**preflight_env(cfg, tag), **(extra or {})})
    text = log.read_text()
    app = re.search(r"\bap-[A-Za-z0-9]{20,}", text)
    if not app or '"function_call_id"' not in text:
        raise SystemExit(f"preflight {tag} not submitted: {text[-600:]}")
    sh([PY, REC, "cpu-app", app.group(0)], check=False)
    return app.group(0)


def preflight(cfg, tags=(), cfg_path=None):
    LOGS.mkdir(parents=True, exist_ok=True)
    out = []
    for tag in tags or list(cfg.preflight):
        if tag not in cfg.preflight:
            raise SystemExit(f"no preflight leg {tag!r}: {sorted(cfg.preflight)}")
        if KILL in cfg.preflight[tag]:  # the kill/resume driver runs detached (nk below); verdict waits for it
            log = LOGS / f"{cfg.bundle}{tag}-nk.log"
            background([PY, "-u", str(HERE / "launch.py"), "nk", str(cfg_path), tag], log)
            out.append((tag, None))
            print(f"preflight {cfg.bundle}{tag} kill/resume driver started ({log})")
            continue
        app = submit(cfg, tag)
        out.append((tag, app))
        print(f"preflight {cfg.bundle}{tag} {app}")
    return out


def checkpoint_step(cfg, tag):
    """The newest checkpoint_state step in the leg's live feed on the volume (mirrored and committed every 60 s), 0 if
    none yet or unreadable."""
    import modal

    path = f"validation/preflight/{cfg.bundle}{tag}/arena/trials/rl-loop/rl/feed.jsonl"
    try:
        text = b"".join(modal.Volume.from_name(run_volume(cfg)).read_file(path)).decode()
    except Exception:  # not written yet
        return 0
    steps = []
    for line in text.splitlines():
        if '"checkpoint_state"' in line:
            try:
                steps.append(int(json.loads(line).get("step") or 0))
            except (ValueError, TypeError, AttributeError):
                continue
    return max(steps or [0])


def nk(cfg, tag, sleep=time.sleep, cancel=None, step=None, poll_s=20.0, ended=None, identity=None):
    """The kill/resume leg (NK2 28 Oct, made standard): attempt 0 runs with ARC_RL1_PREFLIGHT_SELF_KILL=1 and kills
    itself at its first checkpoint past step 0 the way a preemption does (feed mirrored, volume committed, os._exit:
    no identity, no pack; a Modal cancel can run the function's finally, NKc's a0 then looked
    finished and a1 started fresh). The same tag is then resubmitted with an ARC_RESTART_ACK naming the planned kill;
    attempt 0's record is kept as ``-a0``. The second attempt must resume (RestartGuard: lost attempt) and pass the
    judge. KILL_S is the ceiling on attempt 0: past it the call is cancelled (no checkpoint: the leg fails). An attempt
    0 that wrote its identity (it ended, it was not killed) fails the leg without a second attempt."""
    kill_s = float(cfg.preflight[tag][KILL])
    step_fake = step is not None  # tests drive time through ``sleep``
    step = step or (lambda: checkpoint_step(cfg, tag))
    ended = ended or call_ended
    identity = identity or (lambda: leg_identity(cfg, tag))
    rec, a0 = leg_record(cfg, tag), leg_record(cfg, tag, "-a0")
    for p in (rec, a0):
        p.unlink(missing_ok=True)
    print(json.dumps({"event": "nk_a0", "app": submit(cfg, tag, {"ARC_RL1_PREFLIGHT_SELF_KILL": "1"}),
                      "kill_ceiling_s": kill_s}), flush=True)
    fc = json.loads(rec.read_text())["function_call_id"]
    waited, clock = 0.0, (lambda: waited) if step_fake else time.time
    began, gone = clock(), False
    while clock() - began < kill_s and not (gone := ended(fc)):  # wall clock (real-v10 NK: slow volume reads made a
        sleep(poll_s)                                            # sleep-only count take 76 min for a 1,200 s ceiling)
        waited += poll_s
    if not gone:
        if cancel is None:
            import modal

            cancel = lambda call: modal.FunctionCall.from_id(call).cancel(terminate_containers=True)  # noqa: E731
        cancel(fc)
    at = step()
    waited = round(clock() - began, 1) if not step_fake else waited
    print(json.dumps({"event": "nk_killed" if gone else "nk_cancelled", "call": fc, "checkpoint_step": at,
                      "waited_s": waited, "t": time.time()}), flush=True)
    if not at:
        print(json.dumps({"event": NK_NONE, "kill_ceiling_s": kill_s}), flush=True)
        return
    sleep(60)  # the dead container gone before the next attempt reads the attempts file
    if identity():
        print(json.dumps({"event": NK_SOFT, "detail": "attempt 0 wrote modal-identity.json: it ended, no kill"}),
              flush=True)
        return
    rec.rename(a0)
    ack = f"NK leg {tag}: attempt 0 killed itself after checkpoint {at} (launch.py nk, planned kill/resume test)"
    print(json.dumps({"event": "nk_a1", "app": submit(cfg, tag, {"ARC_RESTART_ACK": ack}, "-a1")}), flush=True)


def call_ended(fc):
    """The call has a result (returned or raised: a self-killed attempt fails its input)."""
    import modal

    try:
        modal.FunctionCall.from_id(fc).get(timeout=0)
    except TimeoutError:
        return False
    except Exception:  # noqa: BLE001 - the input failed (os._exit) or was cancelled
        return True
    return True


def leg_identity(cfg, tag):
    """The leg's modal-identity.json exists on the volume (the attempt ran its finally)."""
    return leg_file(cfg, tag, "modal-identity.json")


NK_SOFT = "nk_no_kill"


def leg_finished(cfg, tag):
    """The leg's function ran to its end: modal-identity.json and the in-container verdict are on the volume."""
    return leg_identity(cfg, tag) and leg_file(cfg, tag, "preflight-judged.json")


def leg_file(cfg, tag, name):
    import modal

    try:
        b"".join(modal.Volume.from_name(run_volume(cfg)).read_file(f"validation/preflight/{cfg.bundle}{tag}/{name}"))
        return True
    except Exception:  # noqa: BLE001 - not written
        return False
NK_NONE = "nk_no_checkpoint"


def background(cmd, log):
    subprocess.Popen(cmd, stdout=open(log, "a"), stderr=subprocess.STDOUT, start_new_session=True, cwd=str(C.MAIN))


def leg_call(cfg, tag):
    """("running", None) while the leg's preflight call has no result (a verdict then would judge a partial feed:
    real-v6 N8a, 27 Oct, judged fail mid-run), ("done", its returned identity), or ("unknown", None)."""
    try:
        rec = json.loads((C.MAIN / "docs/validation-rl" / f"preflight-{cfg.bundle}{tag}.json").read_text())
        import modal

        return "done", modal.FunctionCall.from_id(rec["function_call_id"]).get(timeout=0)
    except TimeoutError:
        return "running", None
    except Exception:  # noqa: BLE001 - the call raised, or no record: judged from its output
        return "unknown", None


def verdict(cfg, tags=()):
    """Each leg's verdict: modal_rl1 --verdict (rl-loop legs), else its exit code (modal-identity.json): a patch-check
    leg (serving prep only) or another trial (e.g. a read's level bench, judged by its own gate step)."""
    import tempfile

    for tag in tags or list(cfg.preflight):
        leg = cfg.preflight[tag]
        if KILL in leg and not leg_record(cfg, tag, "-a0").is_file():
            nklog = LOGS / f"{cfg.bundle}{tag}-nk.log"
            lines = (nklog.read_text() if nklog.is_file() else "").strip().splitlines()
            last = lines[-1] if lines else ""  # the log is appended across reruns: this run's outcome is its last row
            if NK_NONE in last:
                print(f"verdict {cfg.bundle}{tag}: fail (kill/resume: no checkpoint within the kill ceiling)")
            elif NK_SOFT in last:
                print(f"verdict {cfg.bundle}{tag}: fail (kill/resume: attempt 0 ended instead of dying, no kill to "
                      "resume from; rerun the leg under a new tag)")
            else:
                print(f"verdict {cfg.bundle}{tag}: running (kill/resume: attempt 1 not submitted yet)")
            continue
        state, ident = leg_call(cfg, tag)
        if state == "running" and leg_finished(cfg, tag):  # NKd 29 Oct: the function ended (identity + in-container
            state = "done"                                   # verdict on the volume) while its detached app lingered
        if state == "running":
            print(f"verdict {cfg.bundle}{tag}: running (judged once its call ends)")
            continue
        if leg.get("ARC_RL1_DRY_BOOT") == "1":  # the dry boot writes no output folder: its call's exit is the verdict
            ok = isinstance(ident, dict) and ident.get("exit_code") == 0 and not ident.get("error")
            print(f"verdict {cfg.bundle}{tag}: {'pass' if ok else 'fail'} (dry boot; call exit "
                  f"{(ident or {}).get('exit_code') if isinstance(ident, dict) else state})")
            continue
        if leg.get("ARC_RL1_PATCH_CHECK") == "1" or leg.get("ARC_RL1_PREFLIGHT_TRIAL", "rl-loop") != "rl-loop":
            with tempfile.TemporaryDirectory() as t:
                sh([MODAL, "volume", "get", "--force", run_volume(cfg),
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
        text = log.read_text()
        v = re.findall(r'"verdict": "(\w+)"', text)
        v = v[-1] if v else "none"
        if KILL in leg and not re.search(r'"resumed_from_checkpoint": true', text):  # NKc 29 Oct: a bare "fail"
            v = "fail (kill/resume: attempt 1 did not resume)"
        print(f"verdict {cfg.bundle}{tag}: {v} (exit {r.returncode}, {log})")


def out_folder(cfg):
    """The run's volume folder for its current attempt (records.out_id: <run>, or <run>-attempt<N> once re-reserved)."""
    from arc_infra.runops import records
    return records.out_id(cfg.run) if hasattr(records, "out_id") else cfg.run  # main's records (trunk src may lag)


def watch(cfg, app, epoch):
    """The watcher plan; watch.export {passes, base, continues}: the dashboard exporter (export_rl.py) beside it. The
    watcher itself runs as the launching agent's background task (printed command; its alerts wake
    the agent, so a detached one here would not)."""
    plan = {"run": cfg.run, "arm": cfg.arm, "platform": cfg.platform, "app_id": app, "launched": epoch,
            **C.launchcfg.to_dict(cfg.watch)}
    if cfg.platform == "modal":  # the run's own runops sampler + host log as infra.* rows (watch.py, infra 26 Oct)
        out = f"modal:{run_volume(cfg)}:validation/{out_folder(cfg)}"  # real-v12 28 Oct: attempt 2's feed sat in
        # <run>-attempt2 while the plan read <run>/, so the phase rules saw no train phase and no-play fired in one
        plan.setdefault("arena", {"sampler": f"{out}/live/sampler.jsonl", "cpulog": f"{out}/util/cpulog.txt",
                                  "feed": f"{out}/arena/trials/rl-loop/rl/feed.jsonl"})  # phase-aware rules (infra)
        plan.setdefault("follow_record", True)
    # every leg exports its dashboard rows (dl2-v3 ran without one); a config's own export wins
    export = plan.pop("export", None) or {"passes": str(C.MAIN / "runs/ops/export")}
    if export and cfg.launch.resume_from and "continues" not in export:  # v15 29 Oct: no "continues" left a resumed
        # leg without the target metric until its first step (dashboard check); the exporter carries the parent's
        parent = re.sub(r"(-attempt\d+|-lost-\d+)+$", "", Path(cfg.launch.resume_from[0]).name)
        if parent != cfg.run:  # an attempt of this same run: continuing itself hung the exporter (v17 a8/a9)
            export = {**export, "continues": parent}
    p = C.MAIN / "runs/ops/watch-plans" / f"{cfg.run}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan, indent=1) + "\n")
    if export:
        background([PY, "-u", str(C.MAIN / "tools/ops/export_rl.py"), cfg.run,
                    *[f"--{k}={v}" for k, v in export.items()]], LOGS / f"{cfg.run}-export.log")
    return p


def launch(cfg, cfg_path, checked=False):
    """``checked``: the caller ran its own checks (a held-out read: launch/rl/heldout/hread.py, a bundle derived from
    the config's, so not this config's spec hash)."""
    probs = [] if checked else check(cfg, cfg_path)
    if probs:
        raise SystemExit("LAUNCH REFUSED:\n  " + "\n  ".join(probs))
    LOGS.mkdir(parents=True, exist_ok=True)
    print(C.launchcfg.report(cfg).splitlines()[-1])
    if cfg.launch.backend == "modal":
        sh([PY, REC, "reserve", cfg.run, "--arm", cfg.arm, "--platform", cfg.platform, "--gpu",
            f"{cfg.gpu_count} x {cfg.gpu}", "--gpu-count", str(cfg.gpu_count), "--bound", str(cfg.bound_usd),
            "--max-seconds", str(cfg.max_seconds), "--kind", cfg.launch.kind, "--auth", cfg.authorization,
            "--what", cfg.what, *(["--volume", "run"] if run_volume(cfg) != SHARED_VOLUME else []),
            *(["--allowance-exempt", cfg.launch.allowance_exempt] if cfg.launch.allowance_exempt else [])])
        C.launchcfg.remember(cfg)
        if cfg.launch.knobs is not None:
            k = LOGS / f"{cfg.run}-knobs.json"
            import put_knobs as PK

            for root in cfg.launch.resume_from or ():  # a relaunch never drops a live knob
                lost = PK.dropped(PK.live_of(Path(root).name), dict(cfg.launch.knobs))
                if lost:
                    raise SystemExit(f"LAUNCH REFUSED: live knobs of {Path(root).name} missing from launch.knobs "
                                     f"(they would revert to the settings): {lost}; add them to the config")
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
          f"watcher plan {plan}\nSTART THE WATCHER as a background task now (exits at its first alert; restart it "
          f"after handling): {PY} -u {C.MAIN / 'tools/ops/watch.py'} run {plan}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("check", "preflight", "verdict", "launch", "env", "nk"))
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
        preflight(cfg, a.tags, path)
    elif a.cmd == "nk":
        nk(cfg, a.tags[0])
    elif a.cmd == "verdict":
        verdict(cfg, a.tags)
    else:
        launch(cfg, path)


if __name__ == "__main__":
    main()
