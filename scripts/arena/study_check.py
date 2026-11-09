"""Print the study check of a built arena bundle (hero-v2 follow-ups, 29 September).

    uv run python scripts/arena/study_check.py dist/<bundle> [--json]

Window, focus streams, max_num_seqs, KV mode, the budget and backlog policy, every hero-v2 component on or off with its
detail or reason, and the early stop: what the operator checks before a launch. Reads configs/comparison.json only.
"""

import argparse
import json
from pathlib import Path


def summarize(study):
    server = study["server"]
    kernels = server.get("serving_kernels") or {}
    hero = study.get("hero_v2") or {}
    layout = hero.get("layout") or {}
    kv = ("int4 (serving_kernels.kv_cache_4bit)" if kernels.get("kv_cache_4bit") == "int4"
          else f"{kernels.get('kv_cache_4bit')} (serving_kernels.kv_cache_4bit)" if kernels.get("kv_cache_4bit")
          else f"{kernels['kv_cache_8bit']} (serving_kernels.kv_cache_8bit)" if kernels.get("kv_cache_8bit")
          else "BF16")
    components = {}
    for name, row in sorted((hero.get("components") or {}).items()):
        detail = {k: v for k, v in row.items() if k not in ("on", "reason")}
        components[name] = {"on": row["on"], "detail": detail, "reason": row.get("reason")}
    return {
        "name": study["name"], "variant": (study.get("arena_variant") or {}).get("name"),
        "wall_seconds": study["wall_seconds"], "window": study["windows"][0],
        "focus_streams": study["active_limit"], "max_num_seqs": server.get("max_num_seqs"),
        "generation_slots": study["generation_slots"],
        "cuda_graph_capture": server.get("max_cudagraph_capture_size", 32),
        "kv_mode": kv, "block_tokens": layout.get("block_tokens"),
        "served_blocks": layout.get("blocks"), "gate_blocks": layout.get("gate_capacity"),
        "blocks_per_full_window": layout.get("blocks_per_window"), "full_windows_fit": layout.get("windows_fit"),
        "focus_fit": ("oversubscribed (exact KV gate): " + json.dumps(study["focus_oversubscription"])
                      if study.get("focus_oversubscription")
                      else f"required, fallback {study.get('focus_fit_fallback')}"
                      if study.get("require_focus_windows_fit") else "not required"),
        "budget_policy": study.get("level_budget_policy"), "budget_clock": study.get("budget_clock"),
        "backlog": {"key": study.get("backlog_key", "readmissions"), "level0_order": study.get("level0_order", "roster"),
                    "priority": study.get("backlog_priority")},
        "rank": study.get("rank"),
        "blocks": study.get("blocks"),
        "blocks_plan": ({k: v for k, v in study["blocks_plan"].items() if k != "block1"}
                        | {"block1_second_clones": study["blocks_plan"]["block1"]["second_clones"]}
                        if study.get("blocks_plan") else None),
        "harness_fixes": (study.get("harness_fixes") or {}).get("fixes"),
        "prefix_cache": (study.get("cache_experiment") or {}).get("prefix_cache"),
        "prompt_tokens_details": bool(server.get("prompt_tokens_details")),
        "prefix_match_unit": server.get("prefix_match_unit"),
        "compaction_thinking": study.get("compaction_thinking", "gameplay"),
        "reasoning_history": study.get("reasoning_history", "normalized"),
        "trials": [{"name": t["name"], "server_env": t.get("server_env")} for t in study["trials"]],
        "live_progress": study.get("live_progress"), "early_stop": study.get("early_stop"),
        "components": components,
    }


def render(summary):
    lines = [f"study {summary['name']} ({summary['variant']}), wall {summary['wall_seconds']} s"]
    for key in ("window", "focus_streams", "max_num_seqs", "generation_slots", "cuda_graph_capture", "kv_mode",
                "block_tokens", "served_blocks", "gate_blocks", "blocks_per_full_window", "full_windows_fit",
                "focus_fit", "budget_policy", "budget_clock", "backlog", "rank", "blocks", "blocks_plan",
                "harness_fixes", "prefix_cache",
                "prompt_tokens_details", "prefix_match_unit", "compaction_thinking", "reasoning_history", "trials",
                "live_progress", "early_stop"):
        lines.append(f"  {key}: {json.dumps(summary[key]) if isinstance(summary[key], (dict, list)) else summary[key]}")
    lines.append("  hero-v2 components:")
    for name, row in summary["components"].items():
        why = f" -- {row['reason']}" if row["reason"] else ""
        lines.append(f"    {'ON ' if row['on'] else 'off'} {name}: {json.dumps(row['detail'])}{why}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    study = json.loads((Path(args.bundle) / "configs/comparison.json").read_text())
    summary = summarize(study)
    print(json.dumps(summary, indent=1) if args.json else render(summary))


if __name__ == "__main__":
    main()
