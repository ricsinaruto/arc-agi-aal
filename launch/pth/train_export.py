"""train_export.py: PTH training-run metrics for the live dashboard (tools/ops/train_dash.py reads them).
Rebuilt each watcher pass from the watcher's local copies (mon/<run>/: gpu.jsonl, train_events.jsonl or smoke-*,
teacher0.log), so a backfill is the same call: one line per step / checkpoint / probe / held-out read / teacher window,
plus one GPU line per 10-min bucket; alerts are kept in <run>.alerts.jsonl and merged in.
CLI (backfill): python3 train_export.py RUN [live|ended|stopped]"""
import json
import re
import sys
import time
from pathlib import Path

OUT = Path("<scratch>/train-metrics")
GPU_BUCKET = 60  # seconds per gpu row (the util rule needs 1-2 min rows; was 600)
MON = Path("~/arc-sft-data/mon")
QUEUE = Path("./docs/modal-queue.json")
CAP_GIB = 268.6  # B300 total per nvidia-smi
REC = Path("./docs/validation-rl")
# lineage (every plot spans the model's whole training): a leg continues the run whose train/ folder its
# record resumes (or loads its student) from, cut at that parent's last row; train.step counts across legs
NOT_TRAINING = ("sft-heal-round-tA-", "sft-score-")


def parent_of(run):
    if run.startswith(NOT_TRAINING):
        return None
    try:
        d = json.loads((REC / f"modal-{run}.json").read_text())
    except (OSError, ValueError):
        return None
    for key in ("resume", "student"):
        m = re.match(r"/runs/validation/([^/]+)/train(/|$)", d.get(key) or "")
        if m and m.group(1) != run:
            return m.group(1)
    return None


def round_of(run):
    m = re.search(r"opd-r(\d+)", run)
    return int(m.group(1)) if m else (0 if run == "sft-heal-round-sA-D-1e5" else None)


def stage_of(run):
    r = round_of(run)
    if r:
        return f"OPD round {r}"
    return {"sft-heal-round-sA-D-1e5": "D round", "sft-heal-round-cut1-v2": "warm start"}.get(run)


def _lineage(run):
    """(continues, step offset): the parent with its cut, and the parent's last cumulative train.step."""
    parent = parent_of(run)
    if not parent:
        return [], 0
    prow = _rows(OUT / f"{parent}.jsonl")
    if not prow and (MON / parent).is_dir():
        export(parent, "ended")
        prow = _rows(OUT / f"{parent}.jsonl")
    if not prow:
        return [], 0
    until = max(r["t"] for r in prow)
    steps = [r["m"]["train.step"] for r in prow if isinstance(r.get("m", {}).get("train.step"), (int, float))]
    cum = max(steps) if steps else 0
    first = _raw_steps(run)
    if first and min(first) > 1:  # a checkpoint resume (STEP0 > 0) numbers its own steps on: add only the parent's
        praw = _raw_steps(parent)  # own lineage offset, not its last step (c7g showed steps 14.. for 7..)
        return [{"run": parent, "until": until}], (cum - max(praw)) if praw else 0
    return [{"run": parent, "until": until}], cum


def _raw_steps(run):
    """The run's own step numbers from its train events."""
    return [e["step"] for e in _rows(MON / run / "train_events.jsonl")
            if e.get("event") == "step" and isinstance(e.get("step"), (int, float))]


def _rows(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _what(run):
    try:  # the first description seen is kept: finished rows leave the queue
        old = json.loads((OUT / f"{run}.meta.json").read_text()).get("what", "").split(" [held-out base")[0]
        if old and old != run:
            return old
    except (OSError, ValueError):
        pass
    try:
        for r in json.loads(QUEUE.read_text()).get("live", []):
            if r.get("run") == run:
                return (r.get("what") or "").split(". ")[0][:200]
    except (OSError, ValueError):
        pass
    return run


# the PTH dashboard keys every training run emits from its first steps
REQUIRED = ("train.kd", "train.kd_reply", "train.kd_heldout_step", "train.grad_norm", "time.step_s",
            "infra.gpu_util.card1", "infra.vram_gib.card1")


def missing_keys(rows, need=REQUIRED):
    """The REQUIRED keys no row carries a value for (heal_preflight's exporter_keys check, the watcher)."""
    seen = {k for r in rows for k, v in r.get("m", {}).items() if v is not None}
    return [k for k in need if k not in seen]


TRAIN_REQUIRED = tuple(k for k in REQUIRED if k.startswith("train."))


def dead_keys(rows, need=TRAIN_REQUIRED):
    """Required training keys present but exactly 0 or NaN in every row (c7m 24 Oct: the per-step held-out KD read 0.0
    on every rank and the missing-key check passed). GPU keys are left out: a CPU preflight reads 0 util."""
    out = []
    for k in need:
        vals = [r["m"][k] for r in rows if r.get("m", {}).get(k) is not None]
        if vals and all(v != v or v == 0 for v in vals):
            out.append(k)
    return out


BASE = MON / "heldout-base.json"  # {"reply", "all", "label"}: the teacher-H read of the starting weights (cut1-v2)


def _base():
    try:
        return json.loads(BASE.read_text())
    except (OSError, ValueError):
        return None


def lines(run, offset=0, rnd=None):
    D = MON / run
    out = []
    gpu = [g for g in _rows(D / "gpu.jsonl") if g.get("epoch")]
    buckets = {}
    for g in gpu:
        buckets.setdefault(int(g["epoch"] // GPU_BUCKET), []).append(g)
    for b, rows in sorted(buckets.items()):
        m = {"infra.vram_cap_gib": CAP_GIB}
        cards = len(rows[0].get("util") or [])
        for c in range(cards):
            util = [r["util"][c] for r in rows if len(r.get("util") or []) > c]
            mem = [r["mem_gib"][c] for r in rows if len(r.get("mem_gib") or []) > c]
            if util:
                m[f"infra.gpu_util.card{c + 1}"] = round(sum(util) / len(util), 1)
            if mem:
                m[f"infra.vram_gib.card{c + 1}"] = round(max(mem), 1)
        out.append({"t": round(rows[-1]["epoch"], 1), "m": m})
    ev = []
    for name in ("train_events.jsonl", "smoke-a_events.jsonl", "smoke-b_events.jsonl", "smoke-c_events.jsonl"):
        ev += _rows(D / name)
    waited = 0.0  # cumulative stream_wait_s at the last step (data-wait share)
    prev, ckpt = None, 0.0  # step time = event-time gap since the last step (or the opening probe), less checkpoints
    for e in ev:
        kind, m = e.get("event"), {}
        if kind == "step":
            m = {"train.kd": e.get("kd"), "train.kd_reply": e.get("kd_reply"), "train.kd_mtp": e.get("kd_mtp"),
                 "train.grad_norm": e.get("grad_norm"), "train.lr": e.get("lr"),
                 "train.step": (e["step"] + offset) if isinstance(e.get("step"), (int, float)) else None,
                 "train.round": rnd, "train.tokens": e.get("tokens"), "infra.peak_alloc_gib": e.get("peak_gib"),
                 # train.kd stays forward KD; the OPD objective has its own keys
                 "train.opd_objective": e.get("opd_objective"), "train.opd_sampled_rev": e.get("opd_sampled_rev"),
                 "train.opd_is_weight": e.get("opd_is_weight"),
                 # streaming OPD (22 Oct): updates between the weights that sampled each trained chunk and this step's
                 "train.age_median": e.get("age_median"), "train.age_max": e.get("age_max"),
                 "train.stream_dropped": e.get("stream_dropped"), "infra.host_avail_gib": e.get("host_avail_gib")}
            if e.get("heldout_kd") is not None:  # HEAL_STEP_HELDOUT: per-step dh10 slice KD, mean (the dashboard's
                m["train.kd_heldout_step"] = e["heldout_kd"]  # Loss plot takes train.kd*) and per rank
                for r, v in enumerate(e.get("heldout_kd_rank") or []):
                    m[f"eval.step_heldout_kd_r{r}"] = v
            if e.get("fresh_kd") is not None:  # HEAL_FRESH_HOLDOUT: unseen held-out student pieces, read once
                m["train.kd_fresh_heldout"] = e["fresh_kd"]
            if e.get("pinned_rank"):  # c7b 24 Oct: the pinned host pools over ranks, before / after the per-step empty
                m["infra.pinned_pool_gib"] = round(sum(r[0] for r in e["pinned_rank"]), 2)
                m["infra.pinned_after_empty_gib"] = round(sum(r[2] for r in e["pinned_rank"]), 2)
            if prev is not None and e["t"] - prev - ckpt > 0:
                dt = e["t"] - prev - ckpt
                m["time.step_s"] = round(dt, 1)
                if e.get("tokens"):
                    m["infra.train_tok_s"] = round(e["tokens"] / dt, 1)
                if isinstance(e.get("stream_wait_s"), (int, float)):
                    m["train.data_wait_share"] = round(max(0.0, e["stream_wait_s"] - waited) / dt, 4)
            if isinstance(e.get("stream_wait_s"), (int, float)):
                waited = e["stream_wait_s"]
            prev, ckpt = e["t"], 0.0
        elif kind == "checkpoint":
            m = {"time.checkpoint_s": e.get("seconds")}
            ckpt += e.get("seconds") or 0.0
        elif kind == "general_probe":
            m = {"train.kl_general": e.get("kd")}
            if e.get("when") == "before":
                prev, ckpt = e["t"], 0.0
        elif kind == "heldout":
            m = {"eval.heldout_reply_kl": e.get("kd_reply"), "eval.heldout_kl": e.get("kd"),
                 "eval.heldout_reverse": e.get("kd_reply_reverse")}
            base = _base()
            if base:
                m.update({"eval.heldout_reply_kl_base": base.get("reply"), "eval.heldout_kl_base": base.get("all")})
        elif kind == "heldout2":  # dh10: the student's own held-out-game play
            m = {"eval.dh10_reply_kl": e.get("kd_reply"), "eval.dh10_reverse": e.get("kd_reply_reverse"),
                 "eval.dh10_sampled_reverse": e.get("sampled_reverse"), "eval.dh10_tail32": e.get("student_out_top32_reply")}
        elif kind == "mid_read":  # in-round stop-rule reads (5 files each)
            m = {"eval.mid_teacherH_fwd": e.get("h_fwd"), "eval.mid_dh10_sampled": e.get("d_sampled"),
                 "eval.mid_dh10_tail32": e.get("d_tail32"), "eval.mid_dh10_fwd": e.get("d_fwd")}
        elif kind == "memory":
            m = {"infra.alloc_after_load_gib": e.get("allocated_gib")}
        elif kind == "host_mem":  # c7b 24 Oct: the shared watcher's host-ram rule reads this key
            m = {"infra.host_avail_gib": e.get("avail_gib")}
        elif kind == "ckpt_skip":
            m = {"infra.host_avail_gib": e.get("avail_gib"), "infra.ckpt_skipped": 1}
        elif kind == "ckpt_pin":
            m = {"infra.host_avail_gib": e.get("avail_gib"), "infra.next_alloc_gib": e.get("need_gib")}
        elif kind == "loaded":
            prev = None
        m = {k: v for k, v in m.items() if isinstance(v, (int, float))}
        if m and e.get("t"):
            out.append({"t": e["t"], "m": m})
    t0 = min((g["epoch"] - g.get("t", 0) for g in gpu), default=None)  # teacher0.log t is seconds since the start
    at = lambda t: round(t if t > 1e9 else t0 + t, 1)  # noqa: E731  teacher-stream ops carry epoch t already (22 Oct)
    for w in _rows(D / "teacher0.log"):
        if w.get("event") == "window" and w.get("seconds") and t0 is not None:
            m = {"infra.teacher_tok_s": round(w["tokens"] / w["seconds"], 1)}
            for c, p in enumerate(w.get("peak_gib") or []):
                m[f"infra.teacher_peak_gib.card{c + 1}"] = p
            out.append({"t": at(w["t"]), "m": m})
    for w in _rows(D / "teacher-stream.log"):  # streaming teacher pieces (heal_stream)
        if t0 is None or not w.get("t"):
            continue
        if w.get("event") == "stream_piece" and w.get("tok_per_s"):
            out.append({"t": at(w["t"]), "m": {"infra.teacher_tok_s": w["tok_per_s"], "infra.teacher_chains": w.get("batch", 1)}})
        elif w.get("event") == "stream_teacher":
            out.append({"t": at(w["t"]), "m": {k2: v for k2, v in (("infra.stream_pending_tokens", w.get("pending_tokens")),
                                                              ("infra.stream_open_chains", w.get("open_chains")),
                                                              ("infra.teacher_wall_tok_s", w.get("wall_tok_per_s")),
                                                              ("infra.teacher_forward_share", w.get("forward_share")))
                                           if isinstance(v, (int, float))}})
    out += _rows(OUT / f"{run}.alerts.jsonl")
    return sorted(out, key=lambda r: r["t"])


LABELS = {"sft-heal-round-opdS-g1": {
    "label": "OPD streaming g1: student D learns from its own play",
    "summary": "Student D plays 5 training games on one RTX while 2 B300 train it on those games as they stream in, "
               "scored by the BF16 teacher on 2 more B300, and its new weights go back into the playing server every "
               "2 steps. Judged at the end against D on held-out KL: dh10 sampled reverse <= 0.80x, forward and "
               "call-class <= 0.94x, guards teacher-H forward <= 1.10x, top-32 tail <= 2x."}}


def export(run, status="live", alert=None, cards=2):
    OUT.mkdir(parents=True, exist_ok=True)
    if alert:
        with open(OUT / f"{run}.alerts.jsonl", "a") as f:
            f.write(json.dumps({"t": round(time.time(), 1), "m": {}, "alert": alert}) + "\n")
    meta_path = OUT / f"{run}.meta.json"
    try:
        old = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        old = {}
    continues, offset = _lineage(run)
    continues = continues or old.get("continues") or []  # a hand-set lineage is kept when none is inferred
    rows = lines(run, offset, round_of(run))
    tmp = OUT / f"{run}.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.rename(OUT / f"{run}.jsonl")
    what, base = _what(run), _base()
    if base and base.get("label") and any("eval.heldout_reply_kl" in r.get("m", {}) for r in rows):
        what = f"{what} [held-out base: {base['label']}]"
    meta = {"run": run, "arm": "pth", "what": what, "status": status, "cards": cards}
    if continues:
        meta["continues"] = continues
    for k in ("label", "summary"):  # dashboard menu name and plain summary: hand-set or known
        v = old.get(k) or LABELS.get(run, {}).get(k)
        if v:
            meta[k] = v
    stage = old.get("stage") or stage_of(run)  # a hand-set stage is kept
    if stage:
        meta["stage"] = stage
    meta_path.write_text(json.dumps(meta))
    return len(rows)


if __name__ == "__main__":
    print(export(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "ended"))
