"""Harvest of restored starts: a training try that clears its level (harvest point flag) sends one
recorded 1-token request at the next level's board: ``level_bench`` owes it after the clear, ``rollout_record`` writes
it to ``<attempt>/harvest.jsonl`` (never rollout.jsonl, so the trainer reads the try unchanged) and the log gets a
``level_bench_harvest`` line. ``arena.rl_seq`` turns those requests into starts one level deeper: ``snapshot`` copies
the try (rollout.jsonl plus the harvest row, log.jsonl cut through the ``level_bench_harvest`` line) under ``root``
(the volume), and ``board_check`` replays ``restore.load``'s actions on the offline game (``arcade``) against the
recorded levels and board hash.

``KEYS``: the ``mg.harvest`` keys the checked-in configs carry (validated by ``rl_loop.check_mg``).
"""

import json
import os
import shutil
from pathlib import Path

KEYS = {"levels": [2, 3], "from_round": 1, "switch_round": None, "train_starts": 6, "train_tries": 8,
        "max_prompt": None, "max_uses": 1, "fill": True, "board_check": True, "since": None, "stale": None}


MARK = '{"kind":"level_bench_harvest"'


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
