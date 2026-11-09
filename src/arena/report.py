"""The per-trial arena record and the study-level merge."""

import json
import time
from pathlib import Path

from arc_infra.fleet import fleet_summary

from .scheduler_audit import audit_scheduler


def game_row(game, base):
    clock = game.clock
    return {
        **base,
        "index": game.index,
        "state": game.state,
        "levels": max(base.get("levels", 0), game.levels),
        "active_seconds": round(clock.active_seconds(), 3),
        "budget_clock": clock.budget_clock,
        # On the token clock budget_seconds, t1_seconds, level0_grant_seconds and the admission budgets are
        # completion tokens (the keys keep their names for the audit and the analysis scripts).
        **({"budget_unit": clock.budget_unit} if clock.budget_clock == "tokens" else {}),
        "budget_seconds": round(clock.budget_seconds(), 3),
        "inference_seconds": round(clock.inference_seconds(), 3),
        "inference_intervals": list(clock.inference_intervals),
        "tool_seconds": round(clock.tool_seconds(), 3),
        "tool_intervals": list(clock.tool_intervals),
        "readmissions": game.readmissions,
        "normalized_seconds": round(clock.normalized_seconds(), 3),
        "charges": list(clock.charges),
        "pool_admissions": game.pool_admissions,
        "pool_clears": game.pool_clears,
        "stalls": game.stalls,
        "level0_grant_seconds": None if game.level0_grant is None else round(game.level0_grant, 3),
        "parked_seconds": round(clock.parked_seconds(), 3),
        "t1_seconds": None if game.t1 is None else round(game.t1, 3),
        "time_to_level": list(game.time_to_level),
        "admissions": list(game.admissions),
        "requests": len(game.requests),
        "gate_wait_seconds": round(sum(r["gate_wait_seconds"] for r in game.requests), 3),
        "admission_timing": {
            "attempts": list(game.admission_attempts),
            "eligible_wait_seconds": round(sum(r.get("eligible_wait_seconds", 0)
                                                for r in game.admission_attempts), 3),
            "parked_wait_seconds": round(sum(r.get("parked_wait_seconds", 0)
                                              for r in game.admission_attempts), 3),
            "context_seconds": round(sum(r.get("context_seconds", 0)
                                          for r in game.admission_attempts), 3),
        },
        "max_prompt_tokens": game.max_prompt_tokens,
        "compactions": game.compactions,
        "compaction_seconds": round(game.compaction_seconds, 3),
        "finished_reason": game.finished_reason,
    }


def trial_report(trial, *, rows, elapsed, choice, cap, config, scheduler, gate, engine, task_errors,
                 preemption_policy=None):
    """``preemption_policy`` (fixed-2x-commit, ``arena.commit.PreemptionAttribution``): preemptions of
    lane requests under the priority policy are expected; only those that could have hit a focus request
    count against ``preemption_free`` and the status."""
    summary = fleet_summary(rows, max(elapsed, 1e-6))
    preemptions = engine.get("preemptions") if engine.get("valid") else None
    if preemption_policy is not None and preemptions is not None:
        preemptions = preemption_policy.counted(preemptions)
    status = "complete"
    if task_errors or summary["failed_sessions"] or not engine.get("valid"):
        status = "error"
    elif preemptions:
        status = "complete_with_preemptions"
    result = {
        "trial": trial,
        "status": status,
        "window_choice": choice,
        "completion_cap": cap,
        "reasoning_effort": config.get("reasoning_effort"),
        "temperature": config.get("temperature"),
        "wall_seconds": round(elapsed, 3),
        **summary,
        "mean_score_equal_weight_games": summary["mean_score_equal_weight_25_games"],
        "score_denominator_games": len({row["game_id"] for row in rows}),
        "games": rows,
        "scheduler": scheduler,
        "gate": gate,
        "engine": engine,
        "preemption_free": None if preemptions is None else preemptions == 0,
        "task_errors": task_errors,
        "config": config,
    }
    if preemption_policy is not None:
        result["preemptions_lane_policy"] = preemption_policy.counts["lane_policy"]
        result["preemptions_counted"] = preemptions
        result["preemption_attribution"] = preemption_policy.summary()
    result['scheduler_audit'] = audit_scheduler(result)
    if not result['scheduler_audit']['valid']:
        result['status'] = 'error'
    return result


def load_record(output, study, *, hardware, api_validation, notebook_start):
    """The study record, extended when an earlier invocation (another trial) wrote one."""
    path = Path(output) / "study.json"
    if path.is_file():
        record = json.loads(path.read_text())
        record["api_validations"] = [*record.get("api_validations", []), api_validation]
        return record
    return {
        "study": study,
        "hardware": hardware,
        "setup_seconds": time.time() - notebook_start,
        "api_validation": api_validation,
        "trials": [],
    }


def merge_trial(record, execution):
    record["trials"] = [t for t in record["trials"] if t["trial"]["name"] != execution["trial"]["name"]]
    record["trials"].append(execution)
    record["comparison"] = [
        {
            "trial": t["trial"]["name"],
            "status": t.get("status"),
            "mean_score_equal_weight_25_games": (t.get("report") or {}).get("mean_score_equal_weight_25_games"),
            "levels": (t.get("report") or {}).get("levels"),
            "score_denominator_games": (t.get("report") or {}).get("score_denominator_games"),
            "actions": (t.get("report") or {}).get("actions"),
            "switch_wall_seconds": ((t.get("report") or {}).get("scheduler") or {}).get("switch_wall_seconds"),
            "preemptions": ((t.get("report") or {}).get("engine") or {}).get("preemptions"),
            "window": ((t.get("report") or {}).get("window_choice") or {}).get("window"),
        }
        for t in record["trials"]
    ]
    return record


def render(record):
    lines = ["| Trial | Status | Score (tested games) | Levels | Actions | Switch (s) | Preemptions | Window |", "|---|---|---|---|---|---|---|---|"]
    for row in record.get("comparison", []):
        score = row["mean_score_equal_weight_25_games"]
        lines.append(
            f"| {row['trial']} | {row['status']} | {'' if score is None else f'{score:.2f}%'} | {row['levels']} | "
            f"{row['actions']} | {row['switch_wall_seconds']} | {row['preemptions']} | {row['window']} |"
        )
    return "\n".join(lines)
