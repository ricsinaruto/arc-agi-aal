"""In-run harvest of restored starts (built by the envs arm for RL's main runs and the multi-game loop).

A training try that clears its level (harvest point flag) sends one recorded 1-token request at the next level's
board: ``level_bench`` owes it after the clear, ``rollout_record`` writes it to ``<attempt>/harvest.jsonl`` (never
rollout.jsonl, so the trainer reads the try unchanged) and the log gets a ``level_bench_level`` mark at the clear and
a ``level_bench_harvest`` line. After each round ``collect`` turns those requests into starts one level deeper:

- snapshot: the try's rollout.jsonl plus the harvest row, and its log.jsonl cut through the ``level_bench_harvest``
  line, copied to ``$ARC_VALIDATION_OUTPUT/harvest/<id>/`` (the volume; the interim committer commits it), so a
  resume in a new container finds every start; pool rows point there;
- checks: a clean parent restore, the next level's first board (not a second clear or a game over), no unsafe event
  before it, the prompt ceiling, ``restore.load`` and ``mg_attempt_start`` agree on the request, and the CPU board
  check (``load``'s actions replayed on the offline game reach the recorded levels and board hash).

From ``switch_round`` on, the restored training set of that game and level draws ``train_starts`` starts x
``train_tries`` tries from the pool (fewest uses first, then the newest policy round), each start at most ``max_uses``
times, the rest from the set's bank rotation. Pool and use counts live in mg_state.json. Held-out reads (fixed starts,
tests) never cut starts and never draw from the pool.

``mg.harvest``: {"levels": the levels a start may be cut into (default [2, 3]: L1->L2 and L2->L3), "from_round" (1),
"switch_round" (required to draw), "since" (draw only rows cut in this round or later: a restart onto a changed
replay, e.g. the hash-seed fix), "train_starts" (6), "train_tries" (8), "max_prompt" (else the target set's),
"max_uses" (1), "fill" (top up from the bank, true), "board_check" (true)}.

Pool row: {"id", "round", "set", "target", "game", "level", "attempt", "request", "prompt_tokens", "played_by",
"parent", "parent_level", "clear_tokens", "source": "harvest", "family", "start": {"attempt", "request"}}; the
multi-game curriculum (``arena.curriculum.Curriculum.add_start``) reads the same rows.
"""

import json
import os
import shutil
from pathlib import Path

KEYS = {"levels": [2, 3], "from_round": 1, "switch_round": None, "train_starts": 6, "train_tries": 8,
        "max_prompt": None, "max_uses": 1, "fill": True, "board_check": True, "since": None, "stale": None}


def fresh_enough(h, r, number):
    """``stale`` (envs 22 Oct, explorer design): a start trains only while its adapter is at most this many updates
    older than round ``number``'s (round n plays on version n-1; rows cut in round m carry m-1, so ``stale`` k admits
    the last k rounds' cuts; >= 1); None: no limit."""
    return h.get("stale") is None or int(r.get("played_by") or 0) >= int(number) - 1 - int(h["stale"])


MARK = '{"kind":"level_bench_harvest"'


def settings(mg):
    h = mg.get("harvest")
    if not h:
        return None
    unknown = set(h) - set(KEYS)
    if unknown:
        raise ValueError(f"mg.harvest: unknown keys {sorted(unknown)}")
    if h.get("stale") is not None and int(h["stale"]) < 1:  # pick runs before the round's own cuts: 0 admits nothing
        raise ValueError("mg.harvest.stale must be >= 1 (or None)")
    return {**KEYS, **h}


def target(sets, spec):
    """The restored training set one level below ``spec`` on the same game (where its harvest goes), or None."""
    level = int(spec.get("level") or 1) + 1
    for name, s in sets.items():
        if (s.get("role") == "train" and not s.get("fresh") and s["game"] == spec["game"]
                and int(s.get("level") or 0) == level):
            return name
    return None


def wants(mg, sets, spec):
    """Do this set's tries cut starts? A training set whose next level is harvested and has a restored set."""
    h = settings(mg)
    if h is None or spec.get("role") != "train":
        return False
    return int(spec.get("level") or 1) + 1 in [int(x) for x in h["levels"]] and target(sets, spec) is not None


def root(loop):
    out = os.environ.get("ARC_VALIDATION_OUTPUT")
    return Path(out) if out else Path(loop.output)


def snapshot(attempt, dest):
    """Copy one harvested try as a bank-like recording: rollout.jsonl + the harvest row, log.jsonl through the
    ``level_bench_harvest`` line. Returns ``dest``."""
    attempt, dest = Path(attempt), Path(dest)
    lines = (attempt / "harvest.jsonl").read_text().splitlines()
    if len(lines) != 1:
        raise ValueError(f"{attempt}: {len(lines)} harvest rows")
    json.loads(lines[0])
    log, cut = [], False
    with open(attempt / "log.jsonl") as file:
        for line in file:
            log.append(line)
            if line.startswith(MARK):
                cut = True
                break
    if not cut:
        raise ValueError(f"{attempt}: no level_bench_harvest line")
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    data = (attempt / "rollout.jsonl").read_bytes()
    (tmp / "rollout.jsonl").write_bytes(data + (b"" if not data or data.endswith(b"\n") else b"\n")
                                        + lines[0].encode() + b"\n")
    (tmp / "log.jsonl").write_text("".join(log))
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)  # a reader (a resume) never sees half a snapshot
    return dest


def board_check(arc, card, game, snap, request):
    """``load``'s action list replayed on a fresh offline game: ``(ok, detail)`` against the recorded stop."""
    from .level_points import replay
    from .restore import load
    from arc_infra.perception import board_hash

    info = load(snap, request)
    prefix = []
    for a in info["actions"]:
        name, *xy = str(a).split()
        prefix.append(f"{name}@{xy[0]},{xy[1]}" if name == "ACTION6" else name)
    env = arc.make(game, scorecard_id=card, save_recording=False)
    raw, board = replay(env, prefix)
    stop = info["stop"]
    problems = []
    if int(raw.levels_completed) != int(stop["levels"]):
        problems.append(f"levels {int(raw.levels_completed)} != {stop['levels']}")
    if board is None or board_hash(board) != stop["board_hash"]:
        problems.append("board differs")
    return not problems, "; ".join(problems) or "ok"


def arcade(loop):
    """One offline arcade + scorecard for a round's board checks (None: no local games, the check is skipped)."""
    games = getattr(loop, "games_dir", None)
    if not games:
        return None
    from arc_infra.arcade import make_arcade

    arc = make_arcade("offline", games, Path(loop.output) / "harvest-recordings")
    return arc, arc.open_scorecard(tags=["harvest-check"])


def collect(loop, number, plays, played_by, checks=None):
    """The round's new starts from its training plays ``{set: (folder, rows, play)}``, snapshotted and checked;
    appended to ``loop.mg_harvest`` and ``harvest.jsonl``. Returns the feed summary (None: harvest off)."""
    from .restore import load, unsafe_from

    mg = loop.cfg["mg"]
    h = settings(mg)
    if h is None or number < int(h["from_round"]):
        return None
    sets = loop.cfg["mg_sets"]
    have = loop.__dict__.setdefault("mg_harvest", [])
    loop.__dict__.setdefault("mg_harvest_uses", {})
    seen = {r["id"] for r in have}
    new, why, offered = [], {}, 0
    out = root(loop)

    def drop(reason):
        why[reason] = why.get(reason, 0) + 1

    for name, (folder, rows, _) in plays.items():
        spec = sets.get(name) or {}
        if not wants(mg, sets, spec):
            continue
        tname = target(sets, spec)
        tspec = sets[tname]
        level = int(tspec["level"])
        ceiling = h["max_prompt"] or tspec.get("max_prompt")
        for r in rows:
            mark = r.get("harvest")
            if not mark:
                continue
            offered += 1
            rs = r.get("restore")
            if rs and not rs.get("ok"):
                drop("parent_restore_not_exact")
                continue
            if r.get("outcome") != "cleared":
                drop("not_cleared")
                continue
            if mark.get("levels") is not None and int(mark["levels"]) + 1 != level:
                drop("level_mismatch")
                continue
            if mark.get("state") not in (None, "NOT_FINISHED"):
                drop("game_state")
                continue
            if ceiling and int(mark.get("prompt_tokens") or 0) > int(ceiling):
                drop("prompt_over_ceiling")
                continue
            sid = f"r{number:02d}-{name}-{r['label']}"
            if sid in seen:
                continue
            attempt = Path(folder) / "attempts" / r["label"]
            dest = out / "harvest" / sid
            try:
                snapshot(attempt, dest)
                request = int(mark["request"])
                first, _ = unsafe_from(dest)
                if first is not None and first <= int(mark.get("step") or 0):
                    drop("unsafe_event")
                    shutil.rmtree(dest, ignore_errors=True)
                    continue
                if loop.mg_attempt_start(tspec, dest) != request or load(dest, request)["level"] != level:
                    drop("start_mismatch")
                    shutil.rmtree(dest, ignore_errors=True)
                    continue
                if h["board_check"] and checks is not None:
                    ok, detail = board_check(*checks, r.get("game") or spec["game"], dest, request)
                    if not ok:
                        drop("board_check")
                        loop.feed.emit("mg_harvest_board", round=number, id=sid, detail=detail)
                        shutil.rmtree(dest, ignore_errors=True)
                        continue
            except (OSError, ValueError, KeyError, IndexError) as error:
                drop(f"snapshot_{type(error).__name__}")
                shutil.rmtree(dest, ignore_errors=True)
                continue
            seen.add(sid)
            rec = {"id": sid, "round": number, "set": name, "target": tname, "game": tspec["game"], "level": level,
                   "attempt": str(dest), "request": request, "prompt_tokens": mark.get("prompt_tokens"),
                   "played_by": int(r.get("policy_version", played_by)),  # a look-ahead try's own adapter
                   "parent": str(attempt), "parent_level": int(spec.get("level") or 1),
                   "clear_tokens": r.get("tokens_to_clear"), "source": "harvest",
                   "family": tspec.get("family") or tspec["game"], "start": {"attempt": str(dest), "request": request}}
            new.append(rec)
    have.extend(new)
    for folder_ in {Path(loop.output) / "rl", out}:
        try:
            folder_.mkdir(parents=True, exist_ok=True)
            with open(folder_ / "harvest.jsonl", "a") as file:
                for rec in new:
                    file.write(json.dumps(rec) + "\n")
        except OSError:
            pass
    by = {}
    for rec in new:
        by[rec["target"]] = by.get(rec["target"], 0) + 1
    return {"round": number, "offered": offered, "new": len(new), "by_set": by, "dropped": why,
            "pool": pool_sizes(loop)}


def pool_sizes(loop):
    """Per target set: starts in the pool and starts still under max_uses."""
    h = settings(loop.cfg["mg"]) or KEYS
    uses = getattr(loop, "mg_harvest_uses", None) or {}
    out = {}
    for r in getattr(loop, "mg_harvest", None) or []:
        row = out.setdefault(r["target"], {"pool": 0, "fresh": 0})
        row["pool"] += 1
        row["fresh"] += uses.get(r["id"], 0) < int(h["max_uses"])
    return out


def pick(loop, number, name, spec, fallback, k=None, tries=None):
    """``(starts, tries)`` for restored training set ``name`` from the switch round on, else None. ``fallback`` is
    mg_pick's own (bank) answer; it fills a short pool when ``fill`` is on. ``k`` / ``tries``: the curriculum's
    starts and tries for the set this round (else mg.harvest's)."""
    mg = loop.cfg["mg"]
    h = settings(mg)
    if h is None or spec.get("fresh") or h["switch_round"] is None or number < int(h["switch_round"]):
        return None
    k, tries = int(k or h["train_starts"]), int(tries or h["train_tries"])
    uses = loop.__dict__.setdefault("mg_harvest_uses", {})
    pool = [r for r in getattr(loop, "mg_harvest", None) or []
            if r["target"] == name and uses.get(r["id"], 0) < int(h["max_uses"])
            and (h["since"] is None or int(r["round"]) >= int(h["since"])) and fresh_enough(h, r, number)]
    pool.sort(key=lambda r: (uses.get(r["id"], 0), -int(r["played_by"]), -int(r["round"]), r["id"]))
    chosen = pool[:k]
    for r in chosen:
        uses[r["id"]] = uses.get(r["id"], 0) + 1
    starts = [(r["attempt"], int(r["request"])) for r in chosen]
    filled = 0
    if len(starts) < k and h["fill"] and fallback:
        own = [tuple(x) for x in (fallback[0] or []) if tuple(x) not in starts][: k - len(starts)]
        starts, filled = starts + own, len(own)
    loop.feed.emit("mg_harvest_pick", round=number, name=name, harvested=len(chosen), filled=filled, tries=tries,
                   ids=[r["id"] for r in chosen])
    return (starts, tries) if starts else None


def sources(loop, plays):
    """Clear rate by start source per restored training set: {set: {"harvest": "c/n", "bank": "c/n"}}."""
    from .rl_single import clear_tokens

    mine = {r["attempt"] for r in getattr(loop, "mg_harvest", None) or []}
    if not mine:
        return None
    out = {}
    for name, (_, rows, _) in plays.items():
        for r in rows:
            rs = r.get("restore")
            if not rs:
                continue
            key = "harvest" if rs.get("source") in mine else "bank"
            c = out.setdefault(name, {}).setdefault(key, [0, 0])
            c[0] += bool(clear_tokens(r))
            c[1] += 1
    return {n: {k: f"{a}/{b}" for k, (a, b) in v.items()} for n, v in out.items() if "harvest" in v} or None


def state(loop):
    return {"harvest": getattr(loop, "mg_harvest", None) or [],
            "harvest_uses": dict(getattr(loop, "mg_harvest_uses", None) or {})}


def restore_state(loop, data, root=None):
    """A resumed run's pool. Snapshots stay on the volume where the source run wrote them; a packed preflight's are
    read from its unpacked copy (``root``: the resumed run dir; ``/tmp/<path under /runs>``). A start whose snapshot is
    gone leaves the pool."""
    def found(path):
        p = Path(path)
        cands = [p] + ([Path(root) / "harvest" / p.name] if root else [])
        if str(p).startswith("/runs/"):
            cands.append(Path("/tmp") / p.relative_to("/runs"))
        return next((str(c) for c in cands if (c / "rollout.jsonl").is_file()), None)

    pool = []
    for r in data.get("harvest") or []:
        at = found(r["attempt"])
        if at is not None:
            pool.append({**r, "attempt": at, "start": {"attempt": at, "request": int(r["request"])}})
    loop.mg_harvest = pool
    loop.mg_harvest_uses = {k: int(v) for k, v in (data.get("harvest_uses") or {}).items()}
    return {"pool": len(pool), "missing": len(data.get("harvest") or []) - len(pool)}
