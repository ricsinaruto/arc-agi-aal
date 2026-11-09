"""mg.redesign.score_check (RL 30 Oct): do trained updates reach play? The same reply tokens of recorded tries, teacher
forced, on the served adapter and the trainer's (and on base on both), with the floors of the same weights scored
twice. {"rollouts": [rollout.jsonl of tries played on this checkpoint's version], "depths": [prompt tokens],
"per_depth": n}. The run resumes the checkpoint, scores, writes ``score-check/`` and stops; nothing plays or trains.
"""
import asyncio
import json
import math
from pathlib import Path

from . import kv_golden, rollout_record

UNCERTAIN = math.log(0.95)  # tokens either side holds below this carry the read; the rest sit near logp 0
REACH = {"r": 0.8, "slope": (0.8, 1.2), "gap_ratio": 1.25}
BUG = {"r": 0.5, "slope": (0.5, 1.5), "gap_ratio": 2.0, "reload_ratio": 2.0}
MOVE_FLOOR = 3.0  # the trainer's own move must clear this many times the larger floor, or the read says too small


def settings(cfg):
    """The validated spec, None when off."""
    spec = cfg.get("score_check")
    if not spec:
        return None
    if not isinstance(spec, dict) or not spec.get("rollouts") or not spec.get("depths"):
        raise ValueError("mg.redesign.score_check is {rollouts: [rollout.jsonl], depths: [prompt tokens], per_depth}")
    return spec


def probes_of(spec):
    probes = kv_golden.build(spec["rollouts"], spec["depths"], int(spec.get("per_depth") or 1))
    for i, p in enumerate(probes):
        p["label"] = f"{i:02d}-{p['label']}"
    return probes


def windows_of(loop, probes):
    """One trainer window per probe, ending at the probe's reply."""
    out = []
    for p in probes:
        path = Path(p["rollout"])
        spec = loop.windows(path.parent.parent.parent, [{"label": path.parent.name}])[0]
        out.append({**spec, "name": p["label"], "end_request": p["request"]})
    return out


async def seed(probes):
    """The server's reply-id cache holds each probe's earlier replies, so its prompt is the served one token for token."""
    files = []
    for p in probes:
        rows = [json.loads(line) for line in open(p["rollout"])]
        files.append(rollout_record.seed_reply_ids([r for r in rows if r["n"] <= p["request"]]))
    waits = await asyncio.gather(*(rollout_record.wait_seeds(f) for f in files if f is not None))
    return {"seeded": len(waits), "timeouts": sum(w is None for w in waits)}


def trainer_row(windows, probes):
    """A trainer score reply as a dump: {"probes": [{label, logprobs}]}, the reply of each probe's request."""
    rows = []
    for p in probes:
        got = next((r for r in (windows.get(p["label"]) or {}).get("replies") or [] if r["request"] == p["request"]), None)
        rows.append({"label": p["label"], "n": len(p["tokens"]), "start": None if got is None else got["start"],
                     "logprobs": [] if got is None else got["raw"]})
    return {"probes": rows}


def series(dump):
    return {p["label"]: p["logprobs"] for p in dump["probes"]}


def stats(pairs):
    """mean |a - b| and mean a - b of [(a, b)]."""
    n = len(pairs)
    return {"tokens": n, "mean_abs": sum(abs(a - b) for a, b in pairs) / n if n else None,
            "mean": sum(a - b for a, b in pairs) / n if n else None}


def fit(xs, ys):
    """Slope of y on x, and r."""
    n = len(xs)
    if n < 3:
        return None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if not sxx or not syy:
        return None, None
    return sxy / sxx, sxy / math.sqrt(sxx * syy)


def summary(dumps, engine=None):
    """``dumps``: {"s_policy", "s_policy2", "s_policy_together", "s_policy_reload", "s_base", "s_base2", "s_base_reload", "t_policy",
    "t_policy2", "t_base"}; ``engine``: {label: recorded live logprobs}. The read is on tokens every dump holds and that
    either policy holds below p 0.95; ``all`` keeps the near-certain tokens too."""
    keys = list(dumps)
    data = {k: series(v) for k, v in dumps.items()}
    rows, dropped = [], 0
    for label in data[keys[0]]:
        cols = [data[k].get(label) or [] for k in keys]
        size = min(len(c) for c in cols)
        dropped += max(len(c) for c in cols) - size
        for i in range(size):
            row = {k: c[i] for k, c in zip(keys, cols)}
            if engine and label in engine and i < len(engine[label]):
                row["engine"] = float(engine[label][i])
            if all(math.isfinite(v) for v in row.values()):
                rows.append(row)
            else:
                dropped += 1
    picked = [r for r in rows if min(r["s_policy"], r["t_policy"]) < UNCERTAIN]

    def part(group):
        out = {"tokens": len(group)}
        for name, a, b in (("floor_server", "s_policy", "s_policy2"), ("floor_together", "s_policy", "s_policy_together"),
                           ("floor_reload", "s_policy", "s_policy_reload"), ("floor_base", "s_base", "s_base2"),
                           ("floor_base_reload", "s_base", "s_base_reload"),
                           ("floor_trainer", "t_policy", "t_policy2"), ("gap_policy", "s_policy", "t_policy"),
                           ("gap_base", "s_base", "t_base"), ("move_server", "s_policy", "s_base"),
                           ("move_trainer", "t_policy", "t_base")):
            out[name] = stats([(r[a], r[b]) for r in group])
        with_engine = [r for r in group if "engine" in r]
        out["engine_vs_scored"] = stats([(r["engine"], r["s_policy"]) for r in with_engine])
        out["engine_vs_trainer"] = stats([(r["engine"], r["t_policy"]) for r in with_engine])
        out["slope"], out["r"] = fit([r["t_policy"] - r["t_base"] for r in group], [r["s_policy"] - r["s_base"] for r in group])
        return out

    return {"dropped": dropped, "all": part(rows), "uncertain": part(picked), "verdict": verdict(part(picked))}


def verdict(p):
    """The pre-registered read."""
    if not p["tokens"] or p["r"] is None:
        return "no_read"
    value = lambda name: p[name]["mean_abs"]
    floor = max(value("floor_server"), value("floor_trainer"))
    gap_ratio = value("gap_policy") / value("gap_base") if value("gap_base") else math.inf
    bug = []
    if p["r"] < BUG["r"] or not BUG["slope"][0] <= p["slope"] <= BUG["slope"][1]:
        bug.append("move")
    if gap_ratio >= BUG["gap_ratio"]:
        bug.append("gap")
    if value("floor_server") and value("floor_reload") >= BUG["reload_ratio"] * value("floor_server"):
        bug.append("reload")
    small = value("move_trainer") < MOVE_FLOOR * floor
    if "reload" in bug or "gap" in bug or ("move" in bug and not small):
        return "bug:" + ",".join(bug)
    if small:
        return "move_too_small"
    ok = p["r"] >= REACH["r"] and REACH["slope"][0] <= p["slope"] <= REACH["slope"][1] and gap_ratio <= REACH["gap_ratio"]
    return "reach" if ok else "inconclusive"


async def run(loop, ctl, server, spec, adapter, base):
    """The check, in play phase on entry; returns the stop why."""
    folder = Path(loop.output) / "score-check"
    folder.mkdir(parents=True, exist_ok=True)
    probes = probes_of(spec)
    (folder / "probes.json").write_text(json.dumps([{k: p[k] for k in ("label", "depth", "start", "request", "rollout", "tokens", "engine")}
                                                    for p in probes]))
    post = lambda body: server.post("/v1/chat/completions", body, timeout=900)
    dumps = {}

    def keep(name, got):
        dumps[name] = got
        (folder / f"{name}.json").write_text(json.dumps(got))

    seeded = [await seed(probes)]
    keep("s_policy", await kv_golden.dump(post, adapter["name"], probes))
    keep("s_policy2", await kv_golden.dump(post, adapter["name"], probes))
    keep("s_base", await kv_golden.dump(post, base, probes))
    keep("s_base2", await kv_golden.dump(post, base, probes))
    keep("s_policy_together", await kv_golden.dump(post, adapter["name"], probes, together=True))
    stop = await ctl.to_train()
    windows = windows_of(loop, probes)
    keep("t_policy", trainer_row((await loop.trainer.must("score", windows=windows, dump=True))["windows"], probes))
    keep("t_policy2", trainer_row((await loop.trainer.must("score", windows=windows, dump=True))["windows"], probes))
    await loop.trainer.must("reset")
    keep("t_base", trainer_row((await loop.trainer.must("score", windows=windows, dump=True))["windows"], probes))
    start = await ctl.to_play(adapter["path"])
    seeded.append(await seed(probes))
    keep("s_policy_reload", await kv_golden.dump(post, adapter["name"], probes))
    keep("s_base_reload", await kv_golden.dump(post, base, probes))
    engine = {p["label"]: p["engine"] for p in probes}
    out = {"probes": len(probes), "seeded": seeded, "switch": {"stop": stop, "start": start}, **summary(dumps, engine)}
    (folder / "summary.json").write_text(json.dumps(out))
    loop.feed.emit("score_check", **out)
    return "score_check"
