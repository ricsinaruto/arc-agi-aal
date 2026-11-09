"""The fixed pool scorer: the competition's own per-game calculator (``arc_agi.scorecard.EnvironmentScoreCalculator``:
level score min(115, 100 (baseline / actions)^2), levels weighted by their index, the game score clipped to the
weighted share of levels with a score), applied to a play-through attempt's cleared levels."""

from arc_agi.scorecard import EnvironmentScoreCalculator


def rhae(baselines, level_actions):
    """Game score of an attempt that cleared ``len(level_actions)`` levels in order with those action counts."""
    calc = EnvironmentScoreCalculator()
    for index, baseline in enumerate(baselines):
        cleared = index < len(level_actions)
        calc.add_level(level_index=index + 1, completed=cleared,
                       actions_taken=int(level_actions[index]) if cleared else 0, baseline_actions=int(baseline))
    return float(calc.to_score().score)


def level_actions_of(row):
    """Per-level action counts of a level-bench play-through row (``level_rows``)."""
    return [int(r["actions"]) for r in sorted(row.get("level_rows") or [], key=lambda r: r["level"])]


def level_actions_from_record(rows):
    """Per-level action counts from a broker record (``rsi2_loop.broker``): every action, the counted RESETs
    included, between one level completion and the next."""
    out, mark, levels = [], 0, None
    for row in rows:
        if row.get("event") == "start":
            levels = row.get("levels_completed") or 0
            continue
        done = row.get("levels_completed")
        if done is not None and levels is not None and done > levels:
            out.append(row["n"] - mark)
            mark, levels = row["n"], done
    return out
