"""check_v87b_stack.py BUNDLE [V87B]: CPU gate (): an RTX play matches the v87b
submission stack in what sets KV use and prefix hits: the KV gate, the stream count scaled to the pool by v87b's
formula, prefix caching. Model, MTP, indexer keys, graphs and kernels stay the research settings (not checked).
- src: the KV gate, block budget, capacity rule and prefix-cache code equal to v87b's (KV_PREFIX_FILES);
- study: the KV / prefix keys equal to v87b's; no dynamic cache_budget; the submission formula unchanged: gate_blocks_per_stream
  equal to v87b's, applied to the model's own served pool (was a token-converted density); streams consistent (active_limit = focus_slots <=
  generation_slots <= max_num_seqs); clones per game = streams / games, rounded either way (|instances - active_limit| < games)."""
import json
import sys
from pathlib import Path

B = Path(sys.argv[1])
V = Path(sys.argv[2] if len(sys.argv) > 2 else
         "./.claude/worktrees/v84sub/dist/arena-flash-competition-v87b-sub")
KV_PREFIX_FILES = ("src/arena/gate.py", "src/arena/cache.py", "src/arena/capacity.py", "src/arena/prefix_cache_patch.py",
                   "src/arena/prefix_tail_runtime.py")
KV_PREFIX_KEYS = ("kv_margin", "windows", "cache_admission", "cache_experiment", "isolate_trial_cache",
                  "server.prefix_caching", "server.prefix_match_unit", "server.max_num_batched_tokens",
                  "focus_oversubscription.min_full_windows", "focus_oversubscription.min_slots")


def get(d, key):
    for part in key.split("."):
        d = d.get(part) if isinstance(d, dict) else None
    return d


fail = []
for rel in KV_PREFIX_FILES:
    a, b = V / rel, B / rel
    if not b.is_file() or a.read_bytes() != b.read_bytes():
        fail.append(f"{rel} differs from v87b's")
vs = json.loads((V / "configs/comparison.json").read_text())
bs = json.loads((B / "configs/comparison.json").read_text())
for k in KV_PREFIX_KEYS:
    if get(vs, k) != get(bs, k):
        fail.append(f"study {k}: v87b {str(get(vs, k))[:80]} vs {str(get(bs, k))[:80]}")
if bs.get("cache_budget"):
    fail.append("study cache_budget: a dynamic stream count v87b does not have")
fo_v, fo_b = vs["focus_oversubscription"], bs.get("focus_oversubscription") or {}
if fo_b.get("gate_blocks_per_stream") != fo_v["gate_blocks_per_stream"]:
    fail.append(f"gate_blocks_per_stream {fo_b.get('gate_blocks_per_stream')} vs v87b {fo_v['gate_blocks_per_stream']} "
                "(the submission formula: streams = min(target, floor(served x kv_margin / it)))")
act, slots, seqs = bs["active_limit"], bs["generation_slots"], bs["server"]["max_num_seqs"]
if not (act == fo_b.get("focus_slots") and act <= slots <= seqs):
    fail.append(f"streams: active_limit {act} = focus_slots {fo_b.get('focus_slots')} <= generation_slots {slots} "
                f"<= max_num_seqs {seqs} does not hold")
inst, games = (bs.get("competition") or {}).get("preview_instances", 0), len(bs.get("games") or [])
if games and not abs(inst - act) < games:  # floor (score reads) or ceil (throughput reads: every stream busy)
    fail.append(f"clones: {inst} instances for {act} streams ({games} games): clones per game = streams / games")
for line in fail:
    print(line)
print(("STACK OK " if not fail else "STACK FAILED ") + B.name)
sys.exit(1 if fail else 0)
