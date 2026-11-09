"""trunk/sft events.jsonl -> <scratch>/train-metrics/<run>.jsonl rows {"t","m"} in the dashboard's names, plus <run>.meta.json.
Usage: export_sft.py RUN [--every 120] [--once] [--backfill-slice0 KL]"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B300_USD_S = 0.001972
sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402
from mfu import step_flops  # noqa: E402

TM = paths.SCRATCH / "train-metrics"

PEAK_FLOPS = 2.25e15  # B300 dense BF16

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--every", type=float, default=120)
ap.add_argument("--once", action="store_true")
ap.add_argument("--backfill-slice0", type=float, default=None)
ap.add_argument("--what", default="")
ap.add_argument("--app", default="", help="read events inside the app's live container (modal container exec)")
a = ap.parse_args()
out, meta_f, state_f = TM / f"{a.run}.extra.jsonl", TM / f"{a.run}.meta.json", TM / f".{a.run}.export-state.json"
st = json.loads(state_f.read_text()) if state_f.exists() else {"n": 0, "steps": 0, "clipped": 0, "t0": None, "slice0": False, "base": {}}


def meta(status):
    old = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    meta_f.write_text(json.dumps({**old, "run": a.run, "arm": "sft", "what": a.what or old.get("what", ""), "status": status,
                                  "cards": old.get("cards", 1)}))


def cards():
    return json.loads(meta_f.read_text()).get("cards", 1) if meta_f.exists() else 1


def record_running():
    """The run record's state (a relaunch = a new attempt of the same run): live while it is running."""
    try:
        return json.loads((ROOT / "docs/runs" / f"{a.run}.json").read_text()).get("state") == "running"
    except (OSError, ValueError):
        return False


def emit(t, m):
    with out.open("a") as f:
        f.write(json.dumps({"t": t, "m": {k: v for k, v in m.items() if isinstance(v, (int, float))}}) + "\n")


def rows(e):
    ev = e["event"]
    if ev == "start":
        st["t0"] = st["t0"] or e["t"]  # the lineage's first start: a resumed attempt keeps cost and time continuous
        return {"train.step": e.get("step") or 0}
    if ev == "step":
        st["steps"] += 1
        st["clipped"] += int((e.get("grad_norm") or 0) > 1.0)
        return {"train.step": e["step"], "train.loss": e.get("loss"), "train.kd": e.get("distill_kl"),
"train.mtp_kl": e.get("mtp_kl"), "train.accept1": e.get("accept1"),
                "train.grad_norm": e.get("grad_norm"), "train.lr": e.get("lr"), "train.tok_s": e.get("tok_s"),
                "time.step_s": e.get("seconds"), "train.util": e.get("util"), "train.vram_gib": e.get("vram_gib"),
                "train.clip_frac_steps": 100.0 * st["clipped"] / st["steps"], "infra.train_tok_s": e.get("tok_s"),
                "infra.peak_alloc_gib": e.get("peak_gib"),
                "train.mfu": 100 * step_flops(e) / e["seconds"] / PEAK_FLOPS if "prefix_tokens" in e and e.get("seconds") else None,
                "infra.cost_usd": B300_USD_S * cards() * (e["t"] - st["t0"]) if st["t0"] else None}
    if ev == "eval" and e.get("set") == "arc2":  # the ARC-2 held-out slice: base = the run's first read
        m = {"eval.arc2_heldout_kl": e.get("heldout_kl"), "eval.arc2_heldout_loss": e.get("heldout_loss")}
        for k, v in m.items():
            if v is not None:
                st["base"].setdefault(k, v)
        m.update({k + "_base": st["base"][k] for k in m if k in st["base"]})
        m["train.step"] = e["step"]
        return m
    if ev == "eval":
        sfx = "" if e.get("set") == "full" else "_slice"
        if sfx and e["step"] == 0:
            st["slice0"] = True
        m = {f"eval.heldout_reply_kl{sfx}": e.get("heldout_kl"), f"eval.heldout_loss{sfx}": e.get("heldout_loss"),
             f"eval.heldout_mtp_kl{sfx}": e.get("mtp_kl"), f"eval.accept1{sfx}": e.get("accept1"),
             f"eval.accept_len3{sfx}": e.get("accept_len3")}
        if e["step"] == 0:
            st["base"].update({k: v for k, v in m.items() if v is not None})
        m.update({k + "_base": st["base"][k] for k in m if k in st["base"]})
        m["train.step"] = e["step"]
        return m
    if ev == "boot" and e.get("step") == 0:
        return {"train.peak_gib": (e.get("warm") or {}).get("peak_gib")}
    return None


MODAL = str(ROOT / ".venv/bin/modal")


def _exec_cat(cid, path):
    """`cat` inside the container, complete or rc 1 (infra 28 Oct: a bare `modal container exec ... cat` stops at
    8,192 bytes; print the size, cat, pause, and reject a short stream)."""
    p = subprocess.run([MODAL, "container", "exec", cid, "--", "sh", "-c",
                        'n=$(wc -c < "") || exit 1; echo "$n"; cat ""; rc=$?; sleep 1; exit $rc', "_", path],
                       capture_output=True, text=True, timeout=120)
    size, _, body = p.stdout.partition("\n")
    ok = p.returncode == 0 and size.strip().isdigit() and len(body.encode()) >= int(size)
    return subprocess.CompletedProcess(p.args, 0 if ok else 1, body if ok else "", p.stderr)


def fetch():
    """events.jsonl text: inside the app's live container when --app is set (28 Oct: Modal volume reads hung for 30+
    min while the container's mount was current), else through the machine-wide volume mirror."""
    if a.app:
        cid = None
        try:
            ls = subprocess.run([MODAL, "container", "list", "--json"], capture_output=True, text=True, timeout=60)
            cid = next((c["container_id"] for c in json.loads(ls.stdout) if c.get("app_id") == a.app), None)
        except (subprocess.TimeoutExpired, ValueError, OSError) as error:
            print(time.strftime("%H:%M:%S"), "container list:", repr(error)[:200], flush=True)
        if cid:
            try:  # the watcher's feed (plan without metrics_remote): the run's metrics.jsonl, replaced whole
                m = _exec_cat(cid, f"/runs/validation/{a.run}/train/metrics.jsonl")
                if m.returncode == 0 and m.stdout.strip():
                    tmp = TM / f".{a.run}.jsonl.tmp"
                    tmp.write_text(m.stdout[:m.stdout.rfind("\n") + 1])
                    tmp.replace(TM / f"{a.run}.jsonl")
                else:
                    print(time.strftime("%H:%M:%S"), "metrics exec rc", m.returncode, m.stderr[-200:], flush=True)
            except (subprocess.TimeoutExpired, OSError) as error:
                print(time.strftime("%H:%M:%S"), "metrics exec:", repr(error)[:200], flush=True)
            try:
                r = _exec_cat(cid, f"/runs/validation/{a.run}/train/events.jsonl")
                if r.returncode == 0 and r.stdout.strip():
                    return r.stdout
                print(time.strftime("%H:%M:%S"), "events exec rc", r.returncode, r.stderr[-200:], flush=True)
            except (subprocess.TimeoutExpired, OSError) as error:
                print(time.strftime("%H:%M:%S"), "events exec:", repr(error)[:200], flush=True)
    # single-flight read through the machine-wide mirror (infra 28 Oct: no bare `modal volume get` beside watch/export)
    try:
        return subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "arc_infra.runops.volfetch",
                               "arc-prolong-runs", f"validation/{a.run}/train/events.jsonl", "60"],
                              capture_output=True, text=True, timeout=400).stdout
    except subprocess.TimeoutExpired:
        return ""


def parse(text):
    evs = []
    for line in text.splitlines():
        if line.startswith("{"):
            try:
                evs.append(json.loads(line))
            except ValueError:  # a line still being written
                break
    return evs


while True:
    evs = parse(fetch())
    for e in evs[st["n"]:]:
        m = rows(e)
        if m:
            emit(e["t"], m)
    # ended only when the newest event ends the run and its record is no longer running (27 Oct: attempt 1's "done"
    # left status "ended" through the relaunches)
    done = bool(evs) and evs[-1]["event"] in ("done", "memory_stop") and not record_running()
    st["n"] = max(st["n"], len(evs))
    if a.backfill_slice0 is not None and not st["slice0"] and st["t0"]:
        st["base"]["eval.heldout_reply_kl_slice"] = a.backfill_slice0
        emit(st["t0"] + 1, {"eval.heldout_reply_kl_slice": a.backfill_slice0, "eval.heldout_reply_kl_slice_base": a.backfill_slice0, "train.step": 0})
        st["slice0"] = True
    state_f.write_text(json.dumps(st))
    meta("ended" if done else "live")
    if done or a.once:
        break
    time.sleep(a.every)
