"""Branch groups (VinePPO-style credit; rl-plan "Branch-group credit"): after each round, some of its
training tries are cut at ``at`` (2/3) of their live replies in the level; next round ``group`` continuations are
played from each cut (the parent's conversation and game replayed through the restore path, then live), and trained
with a leave-one-out advantage inside the cut's group. Replayed replies are ``scripted`` (context, never trained), so
only the continuations' own replies carry the group's credit; the parent keeps its try-level advantage in its own set.

Offline check (21 Oct, pilot-1 tries, logeq reward): at 2/3 the reward variance after the cut is 0.59-0.68 of the
whole try's at 0.32-0.44 of its tokens (L1/L3), ~1.1-1.7x signal per token after the prefix-remainder correlation; at
1/3 ~1.0x. Hence 2/3 only, on L1/L3, gated: a set whose round clear rate is under ``clear_gate`` offers every try,
else only tries longer than the set's median.

The cut is snapshotted to ``$ARC_VALIDATION_OUTPUT/branch/<id>/`` (rollout + log through the cut's step) on the volume,
so a resume finds it; pool rows live in mg_state. The continuation's level tokens count from its first live request,
so its reward adds the parent's live level tokens before the cut (``prefix_tokens``): every try of a group is then
judged on the whole level, as fresh tries are.

``mg.branch``: {"levels": [1, 3], "at": 0.67, "group": 4, "parents": 6 (cuts per set per round), "clear_gate": 0.9,
"from_round": 1, "seed": 0, "score": "whole"}. ``score`` "remainder": a continuation is scored on
its actions and tokens after the cut against its cut group's medians (``score_rows``); the prefix only flattened the
whole-level ratios. Cuts made before it (no ``prefix_actions``) stay whole-level.
"""

import json
import random
import shutil
import statistics
from pathlib import Path

KEYS = {"levels": [1, 3], "at": 0.67, "group": 4, "parents": 6, "clear_gate": 0.9, "from_round": 1, "seed": 0,
        "score": "whole"}


def settings(mg):
    b = mg.get("branch")
    if not b:
        return None
    unknown = set(b) - set(KEYS)
    if unknown:
        raise ValueError(f"mg.branch: unknown keys {sorted(unknown)}")
    return {**KEYS, **b}


def root(loop):
    import os

    out = os.environ.get("ARC_VALIDATION_OUTPUT")
    return Path(out) if out else Path(loop.output)


def cut_point(attempt, at):
    """``(request, prefix_tokens, live)`` for a try: the live gameplay replies of its level (before a clear), the
    request at ``at`` of them, and the completion tokens of the live replies before it; None when too short."""
    attempt = Path(attempt)
    rows = [json.loads(line) for line in open(attempt / "rollout.jsonl") if line.strip()]
    clear_step = None
    for line in open(attempt / "log.jsonl"):
        if line.startswith('{"kind":"level_bench_clear"'):
            clear_step = int(json.loads(line).get("step") or 0)
            break
    live = [r for r in rows if not r.get("scripted") and r.get("gameplay")
            and (clear_step is None or int(r.get("step") or 0) < clear_step)]
    if len(live) < 3:
        return None
    k = max(1, min(len(live) - 1, round(at * len(live))))
    request = int(live[k]["n"])
    prefix = sum(int((r.get("usage") or {}).get("completion_tokens") or 0) for r in rows
                 if not r.get("scripted") and int(r["n"]) < request)
    return request, prefix, len(live)


def snapshot(attempt, dest, step):
    """rollout.jsonl and log.jsonl of ``attempt`` through ``step`` (the cut's observation) to ``dest`` atomically."""
    attempt, dest = Path(attempt), Path(dest)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copyfile(attempt / "rollout.jsonl", tmp / "rollout.jsonl")
    keep = []
    with open(attempt / "log.jsonl") as file:
        for line in file:
            if line.startswith('{"kind":"observation"'):
                if int(json.loads(line).get("step") or 0) > step:
                    break
            keep.append(line)
    (tmp / "log.jsonl").write_text("".join(keep))
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return dest


def collect(loop, number, plays, played_by):
    """Cuts from this round's training plays ``{set: (folder, rows, play)}``; appended to ``loop.mg_branch``.
    Returns the feed summary (None: branching off or before from_round)."""
    from .restore import load, unsafe_from
    from .rl_single import clear_tokens

    mg = loop.cfg["mg"]
    b = settings(mg)
    if b is None or number < int(b["from_round"]):
        return None
    sets = loop.cfg["mg_sets"]
    pool = loop.__dict__.setdefault("mg_branch", [])
    out, summary = root(loop), {"round": number, "by_set": {}, "dropped": {}}
    rng = random.Random(int(b["seed"]) * 1000 + number)

    def drop(reason):
        summary["dropped"][reason] = summary["dropped"].get(reason, 0) + 1

    for name, (folder, rows, _) in plays.items():
        spec = sets.get(name) or {}
        if spec.get("role") != "train" or spec.get("branch_of") or int(spec.get("level") or 1) not in [int(x) for x in b["levels"]]:
            continue
        if not rows:
            continue
        rate = sum(1 for r in rows if clear_tokens(r)) / len(rows)
        lengths = {r["label"]: int(r.get("completion_tokens") or 0) for r in rows}
        median = statistics.median(lengths.values())
        offered = [r for r in rows if rate < float(b["clear_gate"]) or lengths[r["label"]] > median]
        rng.shuffle(offered)
        got = 0
        for r in offered:
            if got >= int(b["parents"]):
                break
            rs = r.get("restore")
            if rs and not rs.get("ok"):
                drop("parent_restore_not_exact")
                continue
            attempt = Path(folder) / "attempts" / r["label"]
            sid = f"r{number:02d}-{name}-{r['label']}-b"
            try:
                cut = cut_point(attempt, float(b["at"]))
                if cut is None:
                    drop("too_short")
                    continue
                request, prefix, live = cut
                first, _ = unsafe_from(attempt)
                info = load(attempt, request)
                if first is not None and first <= int(info["stop"]["steps"]):
                    drop("unsafe_event")
                    continue
                if int(info["level"]) != int(spec.get("level") or 1):
                    drop("level_mismatch")
                    continue
                dest = snapshot(attempt, out / "branch" / sid, int(info["stop"]["steps"]))
                if load(dest, request)["stop"] != info["stop"]:
                    drop("snapshot_mismatch")
                    shutil.rmtree(dest, ignore_errors=True)
                    continue
            except (OSError, ValueError, KeyError, IndexError) as error:
                drop(f"cut_{type(error).__name__}")
                continue
            pool.append({"id": sid, "round": number, "set": name, "game": spec["game"], "level": int(spec.get("level") or 1),
                         "attempt": str(dest), "request": int(request), "prefix_tokens": int(prefix), "live": live,
                         "prefix_actions": int(info["stop"]["steps"]) - int(info["level_start_step"]),
                         "parent": str(attempt), "parent_cleared": bool(clear_tokens(r)),
                         "played_by": int(r.get("policy_version", played_by)),  # a look-ahead try's own adapter
                         "used": False})
            got += 1
        summary["by_set"][name] = {"clear_rate": round(rate, 3), "offered": len(offered), "cuts": got}
    return summary


def pick(loop, number, name):
    """The unused cuts of set ``name`` made before round ``number`` (each played once): ``[(attempt, request)]``."""
    b = settings(loop.cfg["mg"])
    if b is None:
        return []
    chosen = [r for r in getattr(loop, "mg_branch", None) or [] if r["set"] == name and not r["used"] and r["round"] < number]
    for r in chosen:
        r["used"] = number
    return [(r["attempt"], int(r["request"])) for r in chosen]


def cut_of(loop, row):
    """The pool row of a continuation's cut (None: not a branch try)."""
    rs = row.get("restore") or {}
    src = str(rs.get("source") or rs.get("attempt") or "")
    for r in getattr(loop, "mg_branch", None) or []:
        if r["attempt"] == src or Path(r["attempt"]).name == Path(src).name:
            return r
    return None


def prefix_of(loop, row):
    """The parent's live level tokens before the cut, for a continuation row (None: not a branch try)."""
    r = cut_of(loop, row)
    return int(r["prefix_tokens"]) if r is not None else None


def score_rows(loop, rows):
    """A branch set's rows for scoring: with ``score`` "remainder", each continuation of a cut that recorded its
    prefix actions gets ``branch_rem`` {pre, a (level actions after the cut), a_med, t_med (medians over the cut
    group's clears)}; every other row is whole-level (``with_prefix``). A copy of each row."""
    from .rl_single import clear_tokens

    b = settings(loop.cfg["mg"]) or {}
    if b.get("score", "whole") not in ("whole", "remainder"):
        raise ValueError(f"mg.branch.score: {b.get('score')!r}")
    if b.get("score") != "remainder":
        return with_prefix(loop, rows)
    groups, cuts = {}, []
    for r in rows:
        cut = cut_of(loop, r)
        cut = cut if cut is not None and cut.get("prefix_actions") is not None and r.get("level_rows") else None
        cuts.append(cut)
        if cut is not None:
            groups.setdefault(cut["attempt"], []).append(r)
    meds = {}
    for key, grp in groups.items():
        pre = int(next(c for c in cuts if c is not None and c["attempt"] == key)["prefix_actions"])
        clears = [r for r in grp if clear_tokens(r)]
        meds[key] = (pre, statistics.median(max(0, int(r["level_rows"][0].get("actions") or 0) - pre) for r in clears)
                     if clears else None, statistics.median(clear_tokens(r) for r in clears) if clears else None)
    out = []
    for r, cut in zip(rows, cuts):
        if cut is None:
            out += with_prefix(loop, [r])
            continue
        pre, a_med, t_med = meds[cut["attempt"]]
        a = max(0, int(r["level_rows"][0].get("actions") or 0) - pre)
        out.append({**r, "branch_rem": {"pre": pre, "a": a, "a_med": a_med, "t_med": t_med}})
    return out


def with_prefix(loop, rows):
    """Continuation rows with the parent's level tokens before the cut added to their level tokens (whole-level
    reward, as a fresh try's); a copy of each row."""
    out = []
    for r in rows:
        p = prefix_of(loop, r)
        if p and r.get("level_rows"):
            lr = [dict(x) for x in r["level_rows"]]
            lr[0]["tokens"] = int(lr[0].get("tokens") or 0) + p
            r = {**r, "level_rows": lr, "branch_prefix_tokens": p}
        out.append(r)
    return out


def state(loop):
    return {"branch": getattr(loop, "mg_branch", None) or []}


def restore_state(loop, data, root_=None):
    """A resumed run's cuts (unused ones stay playable when their snapshot is found)."""
    def found(path):
        p = Path(path)
        cands = [p] + ([Path(root_) / "branch" / p.name] if root_ else [])
        if str(p).startswith("/runs/"):
            cands.append(Path("/tmp") / p.relative_to("/runs"))
        return next((str(c) for c in cands if (c / "rollout.jsonl").is_file()), None)

    import os

    pool, from_tar = [], 0
    force = os.environ.get("ARC_BRANCH_FORCE_TAR") == "1"
    for r in data.get("branch") or []:
        at = None if force else found(r["attempt"])
        if at is None and root_:
            at = from_round_tar(Path(root_), r)
            from_tar += at is not None
        if at is not None:
            pool.append({**r, "attempt": at})
    loop.mg_branch = pool
    return {"pool": len(pool), "missing": len(data.get("branch") or []) - len(pool), "from_tar": from_tar}


def from_round_tar(run, r, scratch=Path("/tmp/branch-from-tar")):
    """A cut rebuilt from its parent's copy in the run's round archive (``rounds/rNN.tar``), when its own snapshot is
    gone: the parent's rollout and log extracted, cut at the same request, checked against the recorded stop."""
    import tarfile

    from .restore import load

    parent = Path(r["parent"])
    member = f"{parent.parents[1].name}/attempts/{parent.name}"
    archive = run / "rounds" / f"r{int(r['round']):02d}.tar"
    if not archive.is_file():
        return None
    src = scratch / r["id"] / "parent"
    try:
        with tarfile.open(archive) as tar:
            for name in ("rollout.jsonl", "log.jsonl"):
                data = tar.extractfile(f"{member}/{name}")
                if data is None:
                    return None
                src.mkdir(parents=True, exist_ok=True)
                (src / name).write_bytes(data.read())
        info = load(src, int(r["request"]))
        dest = snapshot(src, scratch / r["id"] / "cut", int(info["stop"]["steps"]))
        return str(dest)
    except (KeyError, OSError, ValueError, IndexError):
        return None
