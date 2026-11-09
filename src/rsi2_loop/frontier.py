"""Frontier-level evaluation (): candidates play the incumbent's
frontier levels, started from saved states, many attempts at once, instead of whole pool games.

Fixed side only. Pieces:
- ``board_hash``: the hash of a frame (every layer, shape and values).
- the gateway logs every step of every instance (``--log-actions``) and hashes the board at each level-up; a later
  run replays a logged prefix into a fresh instance before the agent's first frame (``--prefixes``) and voids the
  instance if the replayed board's hash differs from the logged one.
- ``candidate_levels``: from a logged incumbent pool run, per pool game the first level it did not clear, with the
  steps that reached it and the board hash at its start.
- ``verify``: a CPU replay of each prefix in a fresh offline engine, twice; a level whose replay does not reproduce the
  logged hash (or level count) both times is dropped before any GPU time.
- ``attempt_stats`` / ``frontier_z``: per attempt, cleared (y) and speed (c = 1 - min(T, W) / W); the combined z of
  a candidate's attempts against the incumbent's reference attempts, level-paired.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def board_hash(frame) -> str:
    import numpy as np

    h = hashlib.sha256()
    for layer in frame or []:
        a = np.ascontiguousarray(np.asarray(layer, dtype=np.int16))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:32]


def read_log(path):
    """{exposed game: {"steps": [{"a": id, "d": data}, ...], "ups": {level: {"i": n_steps, "h": hash}}, "h0": ...}}"""
    out = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        g = out.setdefault(row["g"], {"steps": [], "ups": {}, "h0": None})
        if row.get("k") == "start":
            g["h0"] = row["h"]
        elif row.get("k") == "step":
            g["steps"].append({"a": row["a"], "d": row.get("d") or {}})
            if row.get("up"):
                g["ups"][int(row["lv"])] = {"i": len(g["steps"]), "h": row["h"]}
    return out


def candidate_levels(log, source_of, total_levels):
    """Per source game, the first level the incumbent did not clear: {source, level, steps, hash}. ``source_of`` maps
    exposed -> source game; ``total_levels`` source -> level count (a fully cleared game has no frontier level)."""
    out = []
    for exposed, g in sorted(log.items()):
        source = source_of.get(exposed, exposed)
        cleared = max(g["ups"], default=0)
        if cleared >= int(total_levels.get(source, 10 ** 6)):
            continue
        if cleared == 0:
            if g["h0"] is None:
                continue
            out.append({"source": source, "level": 1, "steps": [], "hash": g["h0"]})
        else:
            up = g["ups"][cleared]
            out.append({"source": source, "level": cleared + 1, "steps": g["steps"][:up["i"]], "hash": up["h"]})
    return out


def replay(arcade, spec):
    """Fresh instance of ``spec['source']`` with its prefix replayed: (env, hash, levels_completed)."""
    from arcengine import GameAction

    env = arcade.make(spec["source"])
    if env is None:
        return None, None, None
    resp = env.observation_space
    for step in spec["steps"]:
        resp = env.step(action=GameAction.from_id(int(step["a"])), data=step.get("d") or {})
        if resp is None:
            return env, None, None
    return env, board_hash(resp.frame), int(resp.levels_completed)


def verify(games_dir, specs, repeats=2):
    """(kept, dropped): a spec is kept only if every replay reproduces its hash and reaches level - 1."""
    import logging

    import arc_agi

    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(games_dir),
                            logger=logging.getLogger("rsi2.frontier"))
    kept, dropped = [], []
    for spec in specs:
        why = None
        for _ in range(repeats):
            try:
                _, h, lv = replay(arcade, spec)
            except Exception as exc:  # noqa: BLE001 - a broken replay drops the level
                why = f"{type(exc).__name__}: {exc}"[:200]
                break
            if h != spec["hash"] or lv != spec["level"] - 1:
                why = f"replay diverged (hash {h} vs {spec['hash']}, levels {lv} vs {spec['level'] - 1})"
                break
        (dropped if why else kept).append({**spec, **({"dropped": why} if why else {})})
    return kept, dropped


def attempt_stats(first_clear, wall):
    """y (cleared within the run) and c (100 x (1 - min(T, W) / W)) of one attempt."""
    t = first_clear if first_clear is not None else None
    return {"y": 1.0 if t is not None and t <= wall else 0.0,
            "c": 100.0 * (1 - min(t if t is not None else wall, wall) / wall)}


def _var(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def frontier_z(cand, ref, floor_y=0.05, floor_c=25.0):
    """Level-paired combined z of candidate attempts against reference attempts.

    ``cand`` / ``ref``: {level key: [{"y", "c"}, ...]}. Per level, the mean difference; the SE from the per-attempt
    variances on each side (floored: an all-0 or all-1 level still carries binomial noise), averaged over levels; the
    correlation of y and c pooled over the reference attempts (centred per level). z = (z_y + z_c) / sqrt(2 + 2r).
    """
    keys = [k for k in ref if cand.get(k)]
    if not keys:
        return None
    dy = dc = vy = vc = 0.0
    sxy = sxx = syy = 0.0
    for k in keys:
        a, b = cand[k], ref[k]
        ya, yb = [x["y"] for x in a], [x["y"] for x in b]
        ca, cb = [x["c"] for x in a], [x["c"] for x in b]
        dy += sum(ya) / len(ya) - sum(yb) / len(yb)
        dc += sum(ca) / len(ca) - sum(cb) / len(cb)
        vy += max(_var(ya), floor_y) / len(ya) + max(_var(yb), floor_y) / len(yb)
        vc += max(_var(ca), floor_c) / len(ca) + max(_var(cb), floor_c) / len(cb)
        my, mc = sum(yb) / len(yb), sum(cb) / len(cb)
        for x in b:
            sxy += (x["y"] - my) * (x["c"] - mc)
            sxx += (x["y"] - my) ** 2
            syy += (x["c"] - mc) ** 2
    n = len(keys)
    dy, dc = dy / n, dc / n
    se_y, se_c = vy ** .5 / n, vc ** .5 / n
    r = sxy / (sxx * syy) ** .5 if sxx > 0 and syy > 0 else 1.0
    r = max(-0.9, min(1.0, r))
    z = (dy / se_y + dc / se_c) / (2 + 2 * r) ** .5
    return {"levels": n, "diff_clear": round(dy, 4), "diff_speed": round(dc, 3), "se_clear": round(se_y, 4),
            "se_speed": round(se_c, 3), "r": round(r, 3), "z": round(z, 3)}


def pick_levels(rates, k, lo=0.2, hi=0.8):
    """Up to ``k`` level keys from the selection batch: clear rate inside [lo, hi] first, nearest to 0.5 first; then
    the nearest outside the band (never 0 or 1 unless nothing else is left). One level per game."""
    order = sorted(rates.items(), key=lambda kv: (not (lo <= kv[1] <= hi), kv[1] in (0.0, 1.0), abs(kv[1] - 0.5),
                                                 kv[0]))
    out, games = [], set()
    for key, _ in order:
        game = key.rsplit("@", 1)[0]
        if game in games:
            continue
        out.append(key)
        games.add(game)
        if len(out) == k:
            break
    return out
