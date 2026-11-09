"""The redesign's held-out read (stage F), coordinator side. One config (launch/rl/runs/b2-rl-hread-rd-*.json) and one
base bundle (launch/rl/build.py); each read is that bundle plus a small study overlay (the adapter, its arm, its step),
launched as its own Nebius run <config run>-<train>-s<step>.

  hread.py follow CONFIG --train-run R     background task: reads R's feed (hread_due rows from rl_seq, mg.redesign.hread)
                                           and queues each read in order, one at a time (RL's RTX allowance), scoring
                                           each finished one against the base read; prints a line per action
  hread.py queue CONFIG --train-run R --step S   one read now: hand-off (S > 0) -> per-read bundle -> checks -> nlaunch
                                           -> queue row -> watcher plan (start `watch.py run PLAN` as a background task)
  hread.py handoff CONFIG --train-run R --step S  the adapter: volume -> Nebius bucket (Modal CPU job, sha-checked)
  hread.py bundle CONFIG --train-run R --step S   the per-read bundle only (a dry run of queue's local part)
  hread.py score CONFIG --train-run R --step S    the verdict of read S vs the base read (or --base/--policy files)

A read refuses an adapter trained under another expert mask than the read server's (rl_seq records the training mask's
sha256 in the hand-off manifest; checked here and again on the VM before the adapter loads). An adapter carrying stage D's
draft LoRA is served with it (the config's trial server_env sets ARC_DRAFT_LORA=1: merged into the draft at activation;
a config without it refuses such an adapter).

Long calls (handoff ~2-5 min, follow) run as background tasks. State:
runs/ops/hread/<train run>.json in the main checkout."""
import argparse
import copy
import dataclasses
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("hread_core", HERE.parents[2] / "src/arena/hread.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
sys.path.insert(0, str(HERE.parent))
import build as B  # noqa: E402
import config as C  # noqa: E402
import launch as L  # noqa: E402

STATE = C.MAIN / "runs/ops/hread"
VOLUME = "arc-prolong-runs"
ASSET_ROOT = "flash-input/models/arc-rl/hread"
REMOTE = "nebius-arc"  # the local rclone remote of the Nebius bucket (tools/ops/watch.py)
FEED = "{out}/arena/trials/rl-loop/rl/feed.jsonl"
RESULT = "validation/{run}/arena/trials/heldout-read/level-bench.json"
# the only study keys a read may change from the base bundle's (everything else: same bundle, same seeds)
OVERLAY = ("name", "load_adapters", "nebius_assets", "level_bench.arms", "hread.train_run", "hread.step",
           "hread.adapter", "hread.digest", "hread.draft", "hread.base")
ADAPTER_NAME = "policy"


# ---- names ---------------------------------------------------------------------------------------------------------

def short(train_run):
    return train_run.removeprefix("rl1-loop-")


def read_run(cfg, train_run, step):
    """A relaunch after a launch that never ran adds -a<n> (the Nebius ledger keeps a run id once reserved)."""
    st = load_state(train_run)
    if not int(step) and st.get("base_run"):  # a run continued from another (v12 -> v13) keeps its base read
        return st["base_run"]
    n = int((st.get("attempt") or {}).get(str(int(step))) or 1)
    return f"{cfg.run}-{short(train_run)}-s{int(step):04d}" + (f"-a{n}" if n > 1 else "")


def next_attempt(cfg, train_run, step):
    """Move past ended attempts (a VM never created, or a read stopped for a fix): a queue call relaunches the step."""
    sys.path.insert(0, str(C.MAIN / "src"))
    from arc_infra.runops import records

    while True:
        rec = records.get(read_run(cfg, train_run, step)) or {}
        if not rec or records.state_of(rec) != "ended":
            return
        st = load_state(train_run)
        att = st.setdefault("attempt", {})
        att[str(int(step))] = int(att.get(str(int(step))) or 1) + 1
        save_state(st)


def asset_prefix(train_run, step):
    return f"{ASSET_ROOT}/{train_run}/s{int(step):04d}"


def vm_path(prefix):
    return "/kaggle/input/" + prefix.removeprefix("flash-input/")


# ---- per-read study and bundle ---------------------------------------------------------------------------------------

def read_study(base, train_run, step, handoff=None, base_score=None):
    """The base bundle's study with this read's overlay. ``handoff``: the adapter's hand-off row (step > 0)."""
    if (base.get("hread") or {}).get("mask_check") is False:
        raise SystemExit("hread.mask_check false is a speed bench's setting, never a read's")
    s = copy.deepcopy(base)
    step = int(step)
    s["name"] = f"{base['name']}-{short(train_run)}-s{step:04d}"
    h = dict(s.get("hread") or {})
    h.update(train_run=train_run, step=step, base=base_score)
    if step == 0:
        s["load_adapters"], s["nebius_assets"] = [], []
        s["level_bench"]["arms"] = [{"name": "base"}]
        h.update(adapter=None, digest=None, draft=None)
    else:
        if not handoff or handoff.get("prefix") != asset_prefix(train_run, step):
            raise ValueError(f"read s{step:04d}: no hand-off of {asset_prefix(train_run, step)}")
        want = ((base.get("server") or {}).get("expert_mask") or {}).get("sha256")
        if handoff.get("expert_mask_sha256") != want:
            raise SystemExit(f"read s{step:04d}: the adapter was trained under expert mask "
                             f"{str(handoff.get('expert_mask_sha256'))[:12]}, the read serves {str(want)[:12]} "
                             "(a refit re-maps experts at the same shapes): refused")
        draft = handoff.get("draft")
        if draft not in (None, "lora"):
            raise SystemExit(f"read s{step:04d}: unknown draft kind {draft!r} in the hand-off")
        if draft and not draft_merge(base):
            raise SystemExit(f"read s{step:04d}: the adapter carries a stage-D draft LoRA and the read server does not "
                             "merge it (trial server_env ARC_DRAFT_LORA=1 missing): refused")
        s["load_adapters"] = [{"name": ADAPTER_NAME, "path": vm_path(handoff["prefix"])}]
        s["nebius_assets"] = [handoff["prefix"]]
        s["level_bench"]["arms"] = [{"name": f"s{step:04d}", "study": {"served_model": ADAPTER_NAME}}]
        h.update(adapter=handoff["prefix"], digest=handoff["digest"], draft=draft)
    s["hread"] = h
    return s


def draft_merge(study):
    """Every trial server of the study merges a stage-D draft LoRA at activation (ARC_DRAFT_LORA=1)."""
    trials = study.get("trials") or []
    return bool(trials) and all((t.get("server_env") or {}).get("ARC_DRAFT_LORA") == "1" for t in trials)


def study_diff(a, b, prefix=""):
    """Dotted paths where two studies differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += study_diff(a.get(k), b.get(k), f"{prefix}{k}.") if k in a and k in b else [f"{prefix}{k}"]
        return out
    return [] if a == b else [prefix.rstrip(".")]


def overlay_problems(base, read):
    return [p for p in study_diff(base, read) if not any(p == o or p.startswith(o + ".") for o in OVERLAY)]


def derive_bundle(base_dir, out_dir, study, meta, gate=True):
    """``base_dir`` + this study: every other file hard-linked, comparison.json and MANIFEST.json written new; the
    manifest keeps the base's identity and adds ``read`` (base manifest hash, overlay). Verified as build.py does.
    ``gate``: the base bundle's source commit is still CI-green (launch/bundle_gate.py), as for every build."""
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    if gate:
        sys.path.insert(0, str(C.CHECKOUT / "launch"))
        import bundle_gate

        bundle_gate.require(C.CHECKOUT, f"rl:{out_dir.name}",
                            ref=json.loads((base_dir / "MANIFEST.json").read_text())["git_commit"])
    base_study = json.loads((base_dir / "configs/comparison.json").read_text())
    bad = overlay_problems(base_study, study)
    if bad:
        raise SystemExit(f"read overlay changes keys outside {OVERLAY}: {bad[:6]}")
    tmp = out_dir.with_name(out_dir.name + ".building")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(base_dir, tmp, copy_function=os.link,
                    ignore=lambda d, names: [n for n in names if Path(d) == base_dir and n == "MANIFEST.json"])
    (tmp / "configs/comparison.json").unlink()
    (tmp / "configs/comparison.json").write_text(json.dumps(study, indent=2) + "\n")
    m = json.loads((base_dir / "MANIFEST.json").read_text())
    m["files"]["configs/comparison.json"] = B.sha256_file(tmp / "configs/comparison.json")
    m["study"] = study["name"]
    m["read"] = {**meta, "base_bundle": base_dir.name, "base_manifest_sha256": B.sha256_file(base_dir / "MANIFEST.json"),
                 "overlay": sorted(set(study_diff(base_study, study)))}
    (tmp / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
    import runpy

    runpy.run_path(str(tmp / "bootstrap.py"))["verify"](tmp)
    B.check_study(tmp, study)
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)
    return out_dir


def read_cfg(cfg, train_run, step, handoff=None):
    """The base config made this read's: run, bundle, anchor (the base read), data (the adapter on the volume), watch."""
    run = read_run(cfg, train_run, step)
    base_run = read_run(cfg, train_run, 0)
    sub = lambda v: (v.replace(cfg.run, run) if isinstance(v, str) else [sub(x) for x in v] if isinstance(v, list)
                     else {k: sub(x) for k, x in v.items()} if isinstance(v, dict) else v)
    data = tuple(cfg.data) + ((f"{train_volume(train_run)}:{handoff['src']}",) if step and handoff and handoff.get("src") else ())
    anchor = (dataclasses.replace(cfg.anchor, run=base_run, note=f"the base read of {train_run} (same bundle, seeds)")
              if step else cfg.anchor)
    what = f"{cfg.what} | read s{int(step):04d} of {train_run}" + (" (base, no adapter)" if not step else "")
    return dataclasses.replace(cfg, run=run, bundle=f"{cfg.bundle}-{short(train_run)}-s{int(step):04d}", what=what,
                               anchor=anchor, data=data, watch=sub(C.launchcfg.to_dict(cfg.watch)))


# ---- state ---------------------------------------------------------------------------------------------------------

def state_path(train_run):
    return STATE / f"{train_run}.json"


def load_state(train_run):
    p = state_path(train_run)
    return json.loads(p.read_text()) if p.is_file() else {"train_run": train_run, "due": {}, "handoff": {},
                                                          "launched": {}, "scores": {}}


def save_state(st):
    STATE.mkdir(parents=True, exist_ok=True)
    p = state_path(st["train_run"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1) + "\n")
    tmp.replace(p)


# ---- remote helpers ------------------------------------------------------------------------------------------------

def nebius_config():
    cfg = json.loads(Path(os.environ.get("ARC_NEBIUS_CONFIG") or C.MAIN / "runs/ops/nebius.json").read_text())
    region = cfg["region"]
    return {"bucket": cfg["bucket"], "region": region,
            "endpoint": cfg.get("endpoint") or f"https://storage.{region}.nebius.cloud"}


def sh(cmd, timeout=None, env=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(C.CHECKOUT), env=env)


def train_volume(train_run):
    """The training run's Modal volume (per-run volumes, 28 Oct: its record's; else arc-prolong-runs)."""
    sys.path.insert(0, str(C.MAIN / "src"))
    from arc_infra.runops import records

    try:
        return records.volume_of(train_run)
    except Exception:
        return VOLUME


def train_out(train_run):
    """The training run's folder on the volume (a re-reserved run writes validation/<run>-attempt<N>)."""
    sys.path.insert(0, str(C.MAIN / "src"))
    from arc_infra.runops import records

    try:
        return f"validation/{records.out_id(train_run)}"
    except Exception:
        return f"validation/{train_run}"


def feed_rows(train_run):
    r = sh([L.MODAL, "volume", "get", "--force", train_volume(train_run), FEED.format(out=train_out(train_run)), "-"], timeout=120)
    rows = []
    for line in (r.stdout or "").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def bucket_cat(key):
    r = sh(["rclone", "cat", f"{REMOTE}:{nebius_config()['bucket']}/{key}"], timeout=300)
    return r.stdout if r.returncode == 0 else None


# ---- commands ------------------------------------------------------------------------------------------------------

def handoff(cfg, train_run, step, due=None):
    """The adapter of step ``step`` -> the bucket; the bucket manifest is read back and its files compared with the
    trainer's digest. Returns the hand-off row (kept in the state)."""
    st = load_state(train_run)
    if str(step) in st["handoff"]:
        return st["handoff"][str(step)]
    due = due or st["due"].get(str(step))
    if not due or not due.get("path"):
        raise SystemExit(f"no hread_due row with a path for step {step} of {train_run} (feed not read yet?)")
    prefix, nb = asset_prefix(train_run, step), nebius_config()
    r = sh([L.MODAL, "run", "--detach", str(HERE / "handoff_modal.py"), "--src", due["path"], "--prefix", prefix,
            "--bucket", nb["bucket"], "--endpoint", nb["endpoint"], "--region", nb["region"]], timeout=3000,
           env={**os.environ, "ARC_HREAD_VOLUME": train_volume(train_run)})
    got = re.findall(r"^HREAD_HANDOFF (\{.*\})$", r.stdout or "", re.M)
    if r.returncode or not got:
        raise SystemExit(f"hand-off of {due['path']} failed (exit {r.returncode}): {(r.stdout + r.stderr)[-800:]}")
    row = json.loads(got[-1])
    man = json.loads(bucket_cat(f"assets-manifests/{H.asset_slug(prefix)}.json") or "{}")
    files = {k: v for k, v in (man.get("files") or {}).items() if k != H.MANIFEST}
    if H.digest(files) != due.get("digest") or row["digest"] != due.get("digest"):
        raise SystemExit(f"hand-off {prefix}: bucket manifest digest {H.digest(files)[:12]} != trainer "
                         f"{str(due.get('digest'))[:12]}")
    row.update(src=due["path"], checked_epoch=round(time.time()))
    st = load_state(train_run)
    st["handoff"][str(step)] = row
    save_state(st)
    return row


def bundle(cfg, train_run, step, handoff_row=None, gate=True):
    base_dir = L.bundle_dir(cfg)
    m = json.loads((base_dir / "MANIFEST.json").read_text()) if (base_dir / "MANIFEST.json").is_file() else {}
    if m.get("spec_sha256") != B.spec_sha256(cfg):
        raise SystemExit(f"base bundle {base_dir} not built from this config (launch/rl/build.py CONFIG)")
    base_score = (load_state(train_run)["scores"].get("0") or {}).get("base_row") if step else None
    study = read_study(json.loads((base_dir / "configs/comparison.json").read_text()), train_run, step, handoff_row,
                       base_score)
    rc = read_cfg(cfg, train_run, step, handoff_row)
    meta = {"train_run": train_run, "step": int(step), "digest": (handoff_row or {}).get("digest"),
            "base_spec_sha256": m["spec_sha256"]}
    return derive_bundle(base_dir, B.DIST / rc.bundle, study, meta, gate=gate), rc


def check_read(rc, cfg):
    probs = C.checks(rc) + [f"input pending: {p}" for p in C.pending(rc)]
    try:
        roster = L.bundle_roster(rc)
    except (OSError, KeyError, ValueError) as e:
        roster, probs = None, probs + [f"bundle roster unreadable: {e!r}"[:200]]
    return probs + C.launchcfg.dry_run(rc, roster_ids=roster, modal_bin=L.MODAL)


def queue(cfg, cfg_path, train_run, step):
    next_attempt(cfg, train_run, step)
    row = handoff(cfg, train_run, step) if step else None
    out, rc = bundle(cfg, train_run, step, row)
    probs = check_read(rc, cfg)
    if probs:
        raise SystemExit(f"READ {rc.run} REFUSED:\n  " + "\n  ".join(probs))
    L.launch(rc, cfg_path, checked=True)
    st = load_state(train_run)
    st["launched"][str(step)] = {"run": rc.run, "bundle": str(out), "epoch": round(time.time())}
    save_state(st)
    return rc.run


# v8.5p per (game, level); excl5 games from b2-sft4x-ex5ref-v1 (their v8.5p reference)
REF_HISTORY = ("launch/rl/data/v85p-levels.json")  # long v8.5p runs only (9 h / 300K-token
LONG_RUN = re.compile(r"9h|300k")  # c100/c110); short runs' censored levels read as fails; excl5 (ex5ref-v1, 30 min) uncovered
REF_POINTS = ("launch/rl/data/hread-rd-points-v1.json", "launch/rl/data/rl1_points.json")


def v85p_refs():
    rows = [r for f in REF_HISTORY for r in json.loads((C.CHECKOUT / f).read_text())["rows"] if LONG_RUN.search(r["run"])]
    base = {}
    for f in REF_POINTS:
        for x in json.loads((C.CHECKOUT / f).read_text())["points"]:
            if x.get("baselines"):
                base.setdefault(str(x["game"])[:4], x["baselines"])
    return H.history_refs(rows), base


def vs_ref(rows, policy_rows=None, budget=None):
    """``ref_ratios`` of the base read (and the policy read with policy / base per metric); ``budget`` the reads'
    whole-try token budget (censoring-matched)."""
    refs, base = v85p_refs()
    b = H.ref_ratios(rows, refs, base, budget=budget)
    if policy_rows is None:
        return {"base": b}
    p = H.ref_ratios(policy_rows, refs, base, budget=budget)
    keys = ("clear_rate_vs_ref", "fail_rate_vs_ref", "tokens_vs_ref", "rhae_tok_vs_ref", "rhae_tok_vs_ref_unmatched")
    q = {k: round(p["all"][k] / b["all"][k], 4) if p["all"].get(k) and b["all"].get(k) else None for k in keys}
    return {"base": b, "policy": p, "policy_over_base": q}


def rows_of(text):
    d = json.loads(text)
    return d, d["attempts"]


def score(cfg, train_run, step, base_file=None, policy_file=None, record=True):
    """Read ``step`` vs the base read (both reads' level-bench.json: local files or the bucket)."""
    def get(file, s):
        if file:
            return Path(file).read_text()
        text = bucket_cat("runs/" + RESULT.format(run=read_run(cfg, train_run, s)))
        if not text:
            raise SystemExit(f"read s{int(s):04d} of {train_run}: no level-bench.json in the bucket yet")
        return text

    bb, base_rows = rows_of(get(base_file, 0))
    budget = int((bb.get("settings") or {}).get("token_budget") or 0) or None
    st = load_state(train_run)
    if step == 0:
        vals = H.live_rhae(base_rows)
        base_row = {"rhae10k": vals[0], "se": vals[1], "run": read_run(cfg, train_run, 0)}
        out = {"step": 0, "base_row": base_row, "vs_ref": vs_ref(base_rows, budget=budget)}
        if record and vals[0] is not None:
            ref = {k: v for k, v in out["vs_ref"]["base"]["all"].items() if k.endswith("_vs_ref") and v is not None}
            sh([L.PY, L.REC, "result", base_row["run"], "--metric", f"rhae10k={vals[0]:.4f}",
                *(["--metric", f"se={vals[1]:.4f}"] if vals[1] is not None else []),
                *sum((["--metric", f"{k}={v}"] for k, v in ref.items()), []), "--note", "base read (no adapter)"])
    else:
        pb, pol_rows = rows_of(get(policy_file, step))
        diff = H.same_read(bb, pb)
        if diff:
            raise SystemExit(f"reads s0000 and s{int(step):04d} differ in {diff}: not the same bundle and seeds")
        reads = {m: H.score(base_rows, pol_rows, m, budget=budget) for m in ("drop", "partial")}
        final = all((r["aggregate"] or {}).get("passes_bar") for r in reads.values())
        out = {"step": int(step), "reads": reads, "final_passes_bar": final, "vs_ref": vs_ref(base_rows, pol_rows, budget=budget)}
        agg = reads["drop"]["aggregate"] or {}
        if record and agg:
            metrics = [f"rhae10k={agg['policy']:.4f}", f"rhae10k_base={agg['base']:.4f}", f"diff={agg['diff']:.4f}"]
            if agg.get("se_diff") is not None:
                metrics.append(f"se_diff={agg['se_diff']:.4f}")
            if agg.get("ratio"):
                metrics.append(f"ratio={agg['ratio'][0]}")
            metrics += [f"{k}={v}" for k, v in out["vs_ref"]["policy"]["all"].items() if k.endswith("_vs_ref") and v is not None]
            sh([L.PY, L.REC, "result", read_run(cfg, train_run, step), *sum((["--metric", x] for x in metrics), []),
                "--vs", read_run(cfg, train_run, 0), "--verdict", "pass" if final else "fail"])
    st["scores"][str(step)] = out
    save_state(st)
    if record:
        dashboard(train_run, out)
    return out


def dashboard(train_run, out):
    """The read on the training run's dashboard (<scratch>/train-metrics/<train run>.jsonl, export_rl's keys):
    held-out RHAE/10K per read with its base and the ratio (owner target 2x)."""
    sys.path.insert(0, str(C.MAIN / "src"))
    from arc_infra.runops import metrics

    vr = out.get("vs_ref") or {}
    ref = {f"eval.heldout_{k}": v for k, v in (vr.get("policy") or vr.get("base") or {}).get("all", {}).items()
           if k.endswith("_vs_ref")}
    ref.update({f"eval.heldout_{k}_base": v for k, v in (vr.get("base") or {}).get("all", {}).items() if k.endswith("_vs_ref")})
    if out["step"] == 0:
        b = out["base_row"]
        m = {"eval.heldout_rhae10k": b["rhae10k"], "eval.heldout_se": b["se"], "eval.heldout_rhae10k_base": b["rhae10k"]}
    else:
        a = out["reads"]["drop"]["aggregate"] or {}
        m = {"eval.heldout_rhae10k": a.get("policy"), "eval.heldout_se": a.get("se_policy"),
             "eval.heldout_rhae10k_base": a.get("base"), "eval.heldout_diff": a.get("diff"),
             "eval.heldout_diff_se": a.get("se_diff")}
        if a.get("ratio"):
            m["eval.heldout_ratio"], m["eval.heldout_ratio_lo"], m["eval.heldout_ratio_hi"] = a["ratio"]
            m["eval.heldout_ratio_target"] = 2.0
    metrics.writer(train_run).emit({**m, **ref, "eval.read_step": out["step"]}, note=f"held-out read s{out['step']:04d}")


def ended(run):
    sys.path.insert(0, str(C.MAIN / "src"))
    from arc_infra.runops import records

    rec = records.get(run)
    return bool(rec) and records.state_of(rec) == "ended"


def follow(cfg, cfg_path, train_run, interval=300, once=False, every=None):
    """Due reads in step order, one live at a time; each finished read is scored (the base first). ``every``: launch
    only steps that are multiples of it (reads every 16 steps while the run still exports every 8)."""
    while True:
        st = load_state(train_run)
        for r in feed_rows(train_run):
            if r.get("kind") == "hread_due" and str(r["step"]) not in st["due"]:
                st["due"][str(r["step"])] = r
                print(f"due: read s{int(r['step']):04d} of {train_run} ({r.get('bytes') or 0} bytes)", flush=True)
            elif r.get("kind") == "hread_error":
                print(f"ALERT hand-off failed in the run: step {r.get('step')}: {r.get('error')}", flush=True)
        save_state(st)
        live = [s for s, x in st["launched"].items() if not ended(x["run"])]
        for s, x in st["launched"].items():
            if s not in st["scores"] and s not in live and (s == "0" or "0" in st["scores"]):
                try:
                    res = score(cfg, train_run, int(s))
                    vr = res.get("vs_ref") or {}
                    print(f"scored s{int(s):04d}: {json.dumps(res.get('base_row') or res['reads']['drop']['aggregate'])} "
                          f"vs v8.5p: {json.dumps((vr.get('policy') or vr.get('base') or {}).get('all'))}",
                          flush=True)
                except SystemExit as e:
                    print(f"score s{int(s):04d}: {e}", flush=True)
        todo = sorted((int(s) for s in st["due"] if s not in st["launched"] and not (every and int(s) % every)), key=int)
        if todo and not live and (todo[0] == 0 or "0" in st["launched"]):
            try:
                print(f"launched read {queue(cfg, cfg_path, train_run, todo[0])}", flush=True)
            except SystemExit as e:
                print(f"queue s{todo[0]:04d} held: {e}", flush=True)
        if len(todo) > 1:
            print(f"behind: {len(todo)} reads due ({todo})", flush=True)
        if once:
            return
        time.sleep(interval)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("follow", "queue", "handoff", "bundle", "score"))
    ap.add_argument("config")
    ap.add_argument("--train-run")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--base")
    ap.add_argument("--policy")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=int, help="follow: launch only reads at multiples of this step")
    ap.add_argument("--base-from", help="follow: take the base read (s0000) of this earlier train run (continued run)")
    a = ap.parse_args(argv)
    path = Path(a.config).resolve()
    cfg = C.load(path)
    if a.cmd == "score" and a.base:
        return print(json.dumps(score(cfg, a.train_run or "local", a.step, a.base, a.policy, record=False), indent=1))
    if not a.train_run:
        raise SystemExit("--train-run is required")
    if a.cmd == "follow" and a.base_from:
        st, old = load_state(a.train_run), load_state(a.base_from)
        if "0" not in st["launched"]:
            st["base_run"] = read_run(cfg, a.base_from, 0)
            st["launched"]["0"] = old["launched"]["0"]
            if "0" in old["scores"]:
                st["scores"]["0"] = old["scores"]["0"]
            save_state(st)
    if a.cmd == "follow":
        return follow(cfg, path, a.train_run, a.interval, a.once, a.every)
    if a.cmd == "queue":
        return print(queue(cfg, path, a.train_run, a.step))
    if a.cmd == "handoff":
        return print(json.dumps(handoff(cfg, a.train_run, a.step)))
    if a.cmd == "bundle":
        row = load_state(a.train_run)["handoff"].get(str(a.step))
        out, rc = bundle(cfg, a.train_run, a.step, row)
        return print(f"{out} ({rc.run}): {check_read(rc, cfg) or 'checks pass'}")
    print(json.dumps(score(cfg, a.train_run, a.step), indent=1))


if __name__ == "__main__":
    main()
