"""CPU dry simulation of the three-block scheduler over the full 30,600 s wall (hero-v2-blocks, 29 September).

The real ``arena.scheduler.Scheduler(budget_policy="blocks")`` on 110 games with fake clocks, ticked every second.
Synthetic progress only (no per-game data): every active game sends one reply of ``reply_tokens`` every
``reply_seconds`` (26 tok/s per stream: run A's 963 tok/s over 37 streams); each game draws a difficulty class
(easy / medium / hard with medians of 20K / 60K / 300K gameplay tokens per level, lognormal sigma 0.8) and 7 levels;
a game that clears its last level finishes. Checks, every tick after the scheduler's own tick:

- no slot is idle while a never-run or old eligible game exists;
- the active count never exceeds the slots;
- a game with a level is never retired; a retirement happens at >= 90K gameplay tokens on level 0;
- a filler's tenure ends one block after its first admission; old refills are never parked by a boundary or tenure;
- block boundaries park only the ending block's originals.

    PYTHONPATH=src uv run python scripts/arena_analysis/blocks_sim.py --seeds 10 --out result.json
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

CLASSES = (("easy", 0.2, 20_000), ("medium", 0.4, 60_000), ("hard", 0.4, 300_000))
# A harder mix, nearer the c110 runs (about 70-80 levels in 2 h over 110 clones).
MIXES = {"default": CLASSES,
         "hard": (("easy", 0.05, 25_000), ("medium", 0.25, 80_000), ("hard", 0.70, 300_000))}


class Now:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def simulate(seed, *, wall=30600, games=110, reply_tokens=1500, reply_seconds=58.0, levels=7, check=True,
             stagger=0, mix="default"):
    classes = MIXES[mix]
    rng = random.Random(seed)
    now = Now()
    clock = GlobalClock(wall, now=now)
    records = [GameRecord(f"k{i:03d}", i, ActiveClock(clock, now=now, budget_clock="tokens"), total_levels=levels)
               for i in range(games)]
    for g in records:
        g.clock.pause()
    sched = Scheduler(records, clock, budget_policy="blocks", backlog_policy="readmissions",
                      blocks={"stagger_seconds": stagger})
    for g in records:
        g.observer = sched
    clock.release()
    for g in records:
        g.clock.release()
    needs, kinds, next_reply, progress = {}, {}, {}, {}
    for g in records:
        r = rng.random()
        acc = 0.0
        for name, share, median in classes:
            acc += share
            if r <= acc:
                break
        kinds[g.game_id] = name
        needs[g.game_id] = [median * math.exp(rng.gauss(0, 0.8)) for _ in range(levels)]
        progress[g.game_id] = 0.0
        next_reply[g.game_id] = rng.uniform(0, reply_seconds)
    sched.start_all()
    violations = []
    idle_with_eligible = 0
    cfg = sched.blocks
    for t in range(1, int(wall) + 1):
        now.value = float(t)
        for g in records:
            if g.state != "active":
                continue
            if t >= next_reply[g.game_id]:
                next_reply[g.game_id] = t + reply_seconds * rng.uniform(0.5, 1.5)
                g.clock.charge(8000, reply_tokens)
                progress[g.game_id] += reply_tokens
                need = needs[g.game_id][g.levels] if g.levels < levels else math.inf
                if progress[g.game_id] >= need:
                    progress[g.game_id] = 0.0
                    g.report_level(g.levels + 1, t, total_levels=levels)
                    if g.levels >= levels:
                        g.note_finished("won")
        before = {g.game_id: (g.state, g.role) for g in records}
        sched.tick()
        if not check:
            continue
        active = sched.active()
        eligible = len(sched.blocks_unrun()) + len(sched.blocks_old())
        if len(active) > cfg["slots"]:
            violations.append(f"{t}: {len(active)} active")
        open_slots = sched.blocks_open_slots()
        if len(active) < open_slots and eligible:
            idle_with_eligible += open_slots - len(active)
            violations.append(f"{t}: idle slot with {eligible} eligible")
        for g in records:
            state, role = before[g.game_id]
            if state == "active" and g.state == "parked":
                reason = g.admissions[-1]["reason"]
                if reason == "level0_retired" and (g.levels or g.clock.budget_tokens() < cfg["level0_budget_tokens"]):
                    violations.append(f"{t}: {g.game_id} retired at level {g.levels}")
                if reason == "tenure_end" and (role != "filler" or
                                               abs(t - g.first_admit_wall - cfg["block_seconds"]) > 1.0):
                    violations.append(f"{t}: {g.game_id} tenure_end as {role} after {t - g.first_admit_wall}")
                if reason == "block_end" and (role != "original" or
                                              abs(t - g.first_admit_wall - cfg["block_seconds"]) > 1.0):
                    violations.append(f"{t}: {g.game_id} block_end as {role}")
                if role == "old" and reason in ("block_end", "tenure_end"):
                    violations.append(f"{t}: old refill {g.game_id} parked by {reason}")
    now.value = float(wall)
    sched.wall_end()
    summary = sched.blocks_summary()
    started = [g for g in records if g.first_admit_wall is not None]
    by_block = {}
    for block in range(cfg["blocks"]):
        home = [g for g in records if g.home_block == block]
        by_block[block + 1] = {"games": len(home), "started": sum(g.first_admit_wall is not None for g in home),
                               "started_as_filler": sum(any(e["reason"] == "filler" for e in g.admissions) for g in home),
                               "levels": sum(g.levels for g in home)}
    return {"seed": seed, "games_ever_run": len(started), "never_run": games - len(started),
            "counts": summary["counts"], "levels": sum(g.levels for g in records),
            "games_with_levels": sum(g.levels > 0 for g in records),
            "retired": sum(g.retired for g in records), "finished": sum(g.state == "finished" and g.finished_reason == "won"
                                                                      for g in records),
            "by_block": by_block, "by_class_retired": {c: sum(g.retired and kinds[g.game_id] == c for g in records)
                                                       for c, _, _ in classes},
            "idle_slot_seconds": summary["idle_slot_seconds"],
            "idle_slot_seconds_with_eligible": summary["idle_slot_seconds_with_eligible"],
            "idle_with_eligible_checked": idle_with_eligible, "violations": violations[:20],
            "violation_count": len(violations)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--stagger", type=float, default=0.0)
    parser.add_argument("--mix", choices=sorted(MIXES), default="default")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    rows = [simulate(seed, stagger=args.stagger, mix=args.mix) for seed in range(args.seeds)]
    keys = ("games_ever_run", "levels", "games_with_levels", "retired", "finished", "idle_slot_seconds")
    result = {"seeds": args.seeds, "stagger_seconds": args.stagger, "mix": args.mix,
              "classes": [list(c) for c in MIXES[args.mix]],
              "mean": {k: round(statistics.mean(r[k] for r in rows), 2) for k in keys},
              "mean_counts": {k: round(statistics.mean(r["counts"][k] for r in rows), 2) for k in rows[0]["counts"]},
              "violations": sum(r["violation_count"] for r in rows),
              "idle_slot_seconds_with_eligible": sum(r["idle_slot_seconds_with_eligible"] for r in rows),
              "rows": rows}
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(json.dumps({k: result[k] for k in ("mean", "mean_counts", "violations", "idle_slot_seconds_with_eligible")},
                     indent=1))


if __name__ == "__main__":
    main()
