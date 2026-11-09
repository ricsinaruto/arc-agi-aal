"""Blocks audit dry simulation (30 September): per-game token budgets of the real blocks scheduler over a full
competition wall, with per-source clear hazards measured on recorded c110 runs.

The real ``arena.scheduler.Scheduler(budget_policy="blocks")`` with the campaign arm's blocks settings (the
independent level allowance 90,000 x (cleared levels + 1), no stagger), ticked every second on fake clocks. Every
active game sends one gameplay reply of ``reply_tokens`` every ``reply_tokens / tok_s`` seconds (x U(0.5, 1.5)); a
parked game sends nothing. Each simulated game is a random source game; its cost of each level is drawn from an
exponential with the source's measured hazard (clears per gameplay token on level 0, and on levels >= 1, from the
recorded runs' time_to_level budget tokens, censored games included); a source that never cleared a level never clears
it. A game that clears its last level finishes.

Reported per arm: gameplay tokens per game by outcome (never cleared, progressed, finished), wall seconds active,
old-refill returns, idle slot-seconds, the boundary timeline, and the same invariant checks as ``blocks_sim.py``.

    PYTHONPATH=src python scripts/arena_analysis/blocks_audit_sim.py --model docs/validation-20260930/blocks-audit-model.json \
        --arm blocks3-40 --seeds 8 --out result.json
    PYTHONPATH=src python scripts/arena_analysis/blocks_audit_sim.py --fit traj.json --model-out model.json
"""

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from arena.clock import ActiveClock, GlobalClock  # noqa: E402
from arena.scheduler import GameRecord, Scheduler  # noqa: E402

# The campaign arms' blocks settings (arena.campaign.components): allowance 90K, stagger 0.
ARMS = {
    "blocks3-40": dict(blocks=3, slots=40, partition="streams"),  # the competition build (40 / 40 / 30)
    "blocks2-55": dict(blocks=2, slots=55),
    "blocks4-28": dict(blocks=4, slots=28),
}


class Now:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def fit(trajectories):
    """Per-source hazards: clears / exposure tokens on level 0 and on levels >= 1 (exposure = the costs of the cleared
    levels plus the censored remainder of the level a game was on)."""
    by = {}
    for t in trajectories:
        if not t.get("source") or not t.get("tokens"):
            continue
        row = by.setdefault(t["source"], {"total_levels": t["total_levels"], "c0": 0, "e0": 0.0, "c1": 0, "e1": 0.0,
                                          "games": 0})
        row["games"] += 1
        prev = 0.0
        for level, at in enumerate(t["clears"]):
            cost = max(0.0, float(at) - prev)
            prev = float(at)
            key = "0" if level == 0 else "1"
            row["c" + key] += 1
            row["e" + key] += cost
        if t["levels"] < (t["total_levels"] or 99):
            key = "0" if t["levels"] == 0 else "1"
            row["e" + key] += max(0.0, float(t["tokens"]) - prev)
    out = {}
    for source, r in sorted(by.items()):
        out[source] = {"total_levels": r["total_levels"], "games": r["games"],
                       "clears_level0": r["c0"], "exposure_level0": round(r["e0"]),
                       "clears_later": r["c1"], "exposure_later": round(r["e1"]),
                       "hazard_level0": r["c0"] / r["e0"] if r["e0"] else 0.0,
                       "hazard_later": r["c1"] / r["e1"] if r["e1"] else 0.0}
    return out


def simulate(seed, model, arm, *, wall=30600.0, games=110, tok_s=26.0, reply_tokens=1800, allowance=90000,
             check=True):
    rng = random.Random(seed)
    now = Now()
    clock = GlobalClock(wall, now=now)
    records = [GameRecord(f"k{i:03d}", i, ActiveClock(clock, now=now, budget_clock="tokens")) for i in range(games)]
    for g in records:
        g.clock.pause()
    blocks = dict(wall_seconds=wall, roster_games=games, stagger_seconds=0, level_token_allowance=allowance,
                  **ARMS[arm])
    sched = Scheduler(records, clock, budget_policy="blocks", backlog_policy="readmissions", blocks=blocks)
    cfg = sched.blocks
    for g in records:
        g.observer = sched
    hazards = model["hazards"] if "hazards" in model else model
    pool = model.get("trajectories")
    sources = sorted(hazards)
    source_of, costs, progress, next_reply = {}, {}, {}, {}

    def draw(rate):
        return rng.expovariate(rate) if rate > 0 else math.inf

    for g in records:
        if pool:
            # Resample a recorded game observed for >= 90K gameplay tokens (or finished): its own clear costs up to
            # its censoring point, then the source's later-level hazard (memoryless) beyond it; a level-0 game at
            # censoring never clears (it retired at 90K in the record).
            t = rng.choice(pool)
            s = t["source"]
            m = hazards[s]
            g.total_levels = int(t["total_levels"])
            prev, levels = 0.0, []
            for at in t["clears"]:
                levels.append(max(1.0, float(at) - prev))
                prev = float(at)
            if len(levels) < g.total_levels:
                rest = float(t["tokens"]) - prev
                levels.append(math.inf if not t["clears"] else rest + draw(m["hazard_later"]))
            while len(levels) < g.total_levels:
                levels.append(draw(m["hazard_later"]))
            costs[g.game_id] = levels[:g.total_levels]
        else:
            s = rng.choice(sources)
            m = hazards[s]
            g.total_levels = int(m["total_levels"])
            costs[g.game_id] = [draw(m["hazard_level0" if lv == 0 else "hazard_later"]) for lv in range(g.total_levels)]
        source_of[g.game_id] = s
        progress[g.game_id] = 0.0
    clock.release()
    for g in records:
        g.clock.release()
    sched.start_all()
    interval = reply_tokens / tok_s
    for g in records:
        next_reply[g.game_id] = rng.uniform(0, interval)
    violations, boundaries = [], []
    active_seconds = {g.game_id: 0.0 for g in records}
    old_tokens = {g.game_id: 0 for g in records}
    for t in range(1, int(wall) + 1):
        now.value = float(t)
        for g in records:
            if g.state != "active":
                continue
            active_seconds[g.game_id] += 1.0
            if g.role == "old":
                pass
            if t >= next_reply[g.game_id]:
                next_reply[g.game_id] = t + interval * rng.uniform(0.5, 1.5)
                g.clock.charge(8000, reply_tokens)
                if g.role == "old":
                    old_tokens[g.game_id] += reply_tokens
                progress[g.game_id] += reply_tokens
                need = costs[g.game_id][g.levels] if g.levels < g.total_levels else math.inf
                if progress[g.game_id] >= need:
                    progress[g.game_id] = 0.0
                    g.report_level(g.levels + 1, t, total_levels=g.total_levels)
                    if g.levels >= g.total_levels:
                        g.note_finished("won")
        # A parked game's next reply starts fresh when it returns (no in-flight reply is modelled).
        before = {g.game_id: (g.state, g.role) for g in records}
        block_before = sched.blocks_index
        sched.tick()
        for g in records:
            if before[g.game_id][0] != "active" and g.state == "active":
                next_reply[g.game_id] = t + interval * rng.uniform(0.5, 1.5)
        if sched.blocks_index != block_before:
            active = sched.active()
            p = sched.blocks_progress()
            boundaries.append({"wall_seconds": t, "block": sched.blocks_index + 1,
                               "parked_block_end": sum(1 for g in records if before[g.game_id][0] == "active"
                                                       and g.state == "parked"
                                                       and g.admissions[-1]["reason"] == "block_end"),
                               "admitted": {r: sum(1 for g in records if before[g.game_id][0] != "active"
                                                   and g.state == "active" and g.admissions[-1]["reason"] == r)
                                            for r in ("block_start", "filler", "old_refill")},
                               "active_by_role": {r: sum(g.role == r for g in active) for r in ("original", "filler", "old")},
                               "active_home_blocks": {str(b + 1): sum(g.home_block == b for g in active)
                                                      for b in range(cfg["blocks"])},
                               "started": p["started"], "unrun": p["unrun"], "parked_eligible": p["parked_eligible"],
                               "retired": p["retired"], "finished": p["finished"]})
        if not check:
            continue
        active = sched.active()
        eligible = len(sched.blocks_unrun()) + len(sched.blocks_old())
        if len(active) > cfg["slots"]:
            violations.append(f"{t}: {len(active)} active")
        if len(active) < cfg["slots"] and eligible:
            violations.append(f"{t}: idle slot with {eligible} eligible")
        for g in records:
            state, role = before[g.game_id]
            if state == "active" and g.state == "parked":
                reason = g.admissions[-1]["reason"]
                if reason == "level_retired" and g.spent_on_level() < allowance * (g.levels + 1):
                    violations.append(f"{t}: {g.game_id} retired under its allowance")
                if reason == "tenure_end" and (role != "filler" or
                                               abs(t - g.first_admit_wall - cfg["block_seconds"]) > 1.0):
                    violations.append(f"{t}: {g.game_id} tenure_end as {role} after {t - g.first_admit_wall}")
                if reason == "block_end" and (role != "original" or
                                              abs(t - g.first_admit_wall - cfg["block_seconds"]) > 1.0):
                    violations.append(f"{t}: {g.game_id} block_end as {role} after {t - g.first_admit_wall}")
                if role == "old" and reason in ("block_end", "tenure_end"):
                    violations.append(f"{t}: old refill {g.game_id} parked by {reason}")
    now.value = float(wall)
    sched.wall_end()
    summary = sched.blocks_summary()
    per_game = []
    for g in records:
        reasons = [e["reason"] for e in g.admissions]
        if g.state == "finished" and g.finished_reason == "won":
            outcome = "finished"
        elif g.first_admit_wall is None:
            outcome = "never_run"
        elif g.levels == 0:
            outcome = "never_cleared"
        else:
            outcome = "progressed"
        end = next((r for r in reversed(reasons) if r in ("block_end", "tenure_end", "level_retired", "finished",
                                                           "wall_end")), None)
        stops = [r for r in reasons if r in ("block_end", "tenure_end", "level_retired", "finished")]
        per_game.append({"game": g.game_id, "source": source_of[g.game_id], "home_block": g.home_block + 1,
                         "first_role": next((e["role"] for e in g.admissions if e["event"] == "admit"), None),
                         "first_admit_wall": g.first_admit_wall, "outcome": outcome, "levels": g.levels,
                         "tokens": g.clock.budget_tokens(), "active_seconds": active_seconds[g.game_id],
                         "old_refills": reasons.count("old_refill"), "old_tokens": old_tokens[g.game_id],
                         "stop": stops[0] if stops else end, "retired": g.retired})
    return {"seed": seed, "arm": arm, "settings": {k: cfg[k] for k in ("blocks", "slots", "block_seconds",
                                                                      "block_sizes", "wall_seconds",
                                                                      "level_token_allowance")},
            "counts": summary["counts"], "idle_slot_seconds": summary["idle_slot_seconds"],
            "idle_slot_seconds_with_eligible": summary["idle_slot_seconds_with_eligible"],
            "boundaries": boundaries, "violations": violations[:20], "violation_count": len(violations),
            "games": per_game}


def q(values, p):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def dist(values):
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": round(min(values)), "median": round(q(values, .5)), "p10": round(q(values, .1)),
            "p90": round(q(values, .9)), "max": round(max(values)), "mean": round(statistics.mean(values))}


def aggregate(rows):
    games = [g for r in rows for g in r["games"]]
    seeds = len(rows)
    out = {"seeds": seeds, "settings": rows[0]["settings"],
           "violations": sum(r["violation_count"] for r in rows),
           "idle_slot_seconds_mean": round(statistics.mean(r["idle_slot_seconds"] for r in rows)),
           "idle_slot_seconds_with_eligible_total": round(sum(r["idle_slot_seconds_with_eligible"] for r in rows), 3),
           "counts_mean": {k: round(statistics.mean(r["counts"][k] for r in rows), 1) for k in rows[0]["counts"]}}
    out["tokens_all_started"] = dist([g["tokens"] for g in games if g["outcome"] != "never_run"])
    out["by_outcome"] = {}
    for outcome in ("never_cleared", "progressed", "finished", "never_run"):
        sel = [g for g in games if g["outcome"] == outcome]
        out["by_outcome"][outcome] = {"games_per_seed": round(len(sel) / seeds, 1),
                                      "tokens": dist([g["tokens"] for g in sel]),
                                      "active_seconds": dist([g["active_seconds"] for g in sel])}
    prog = [g for g in games if g["outcome"] == "progressed"]
    out["progressed_by_stop"] = {}
    for stop in sorted({g["stop"] for g in prog}, key=str):
        sel = [g for g in prog if g["stop"] == stop]
        out["progressed_by_stop"][str(stop)] = {"games_per_seed": round(len(sel) / seeds, 1),
                                                "tokens": dist([g["tokens"] for g in sel])}
    # Games that were never stopped by a limit (not retired, not finished): the "similar budget" population.
    unlimited = [g for g in games if g["outcome"] != "never_run" and not g["retired"] and g["outcome"] != "finished"
                 and g["stop"] != "level_retired"]
    out["not_limited"] = {"games_per_seed": round(len(unlimited) / seeds, 1),
                          "tokens": dist([g["tokens"] for g in unlimited]),
                          "active_seconds": dist([g["active_seconds"] for g in unlimited]),
                          "without_old_refill": dist([g["tokens"] - g["old_tokens"] for g in unlimited])}
    out["not_limited_by_first_role"] = {}
    for role in ("original", "filler"):
        sel = [g for g in unlimited if g["first_role"] == role]
        out["not_limited_by_first_role"][role] = {
            "games_per_seed": round(len(sel) / seeds, 1),
            "first_tenure_tokens": dist([g["tokens"] - g["old_tokens"] for g in sel]),
            "first_tenure_active_seconds": dist([g["active_seconds"] for g in sel if g["old_refills"] == 0])}
    out["not_limited_by_home_block"] = {
        str(b): {"games_per_seed": round(sum(g["home_block"] == b for g in unlimited) / seeds, 1),
                 "tokens_excluding_old": dist([g["tokens"] - g["old_tokens"] for g in unlimited if g["home_block"] == b])}
        for b in range(1, rows[0]["settings"]["blocks"] + 1)}
    old = [g for g in games if g["old_refills"]]
    out["old_refills"] = {"games_per_seed": round(len(old) / seeds, 1),
                          "refills_per_seed": round(sum(g["old_refills"] for g in old) / seeds, 1),
                          "old_tokens": dist([g["old_tokens"] for g in old]),
                          "total_tokens_of_refilled_games": dist([g["tokens"] for g in old])}
    out["boundaries_seed0"] = rows[0]["boundaries"]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", help="trajectory JSON to fit the per-source model from")
    parser.add_argument("--model-out")
    parser.add_argument("--model")
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--wall", type=float, default=30600.0)
    parser.add_argument("--tok-s", type=float, default=26.0)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.fit:
        trajectories = json.loads(Path(args.fit).read_text())
        pool = [{k: t[k] for k in ("run", "game", "source", "total_levels", "levels", "tokens", "clears")}
                for t in trajectories if t.get("source") and t.get("tokens") and "blocks" in str(t.get("run"))
                and (float(t["tokens"]) >= 90000 or t["levels"] >= (t["total_levels"] or 99))]
        # Only the blocks run: every game there played on from its first admission (no opening/level-budget parks),
        # so being observed for >= 90K tokens does not depend on clearing level 1 (the fixed-2x runs park level-0 games
        # early and would bias the pool toward clearers).
        model = {"hazards": fit(trajectories), "trajectories": pool}
        Path(args.model_out).write_text(json.dumps(model, indent=1) + "\n")
        print(len(pool), "trajectories;", sum(not t["clears"] for t in pool), "never cleared level 1")
        return
    model = json.loads(Path(args.model).read_text())
    rows = [simulate(seed, model, args.arm, wall=args.wall, tok_s=args.tok_s)
            for seed in range(args.seed0, args.seed0 + args.seeds)]
    result = {"arm": args.arm, "tok_s": args.tok_s, "wall": args.wall, **aggregate(rows)}
    if args.out:
        Path(args.out).write_text(json.dumps({**result, "rows": rows}, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
