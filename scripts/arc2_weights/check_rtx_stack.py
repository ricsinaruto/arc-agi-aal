"""check_rtx_stack.py BUNDLE [V87B] [--allow-no-tier]: CPU gate for arc2 RTX plays (match only what sets KV use and
prefix hits; arc2 keeps its own model stack). Fails unless, against arena-flash-competition-v87b-sub:
- prefix caching and cache admission keys equal (kv_margin, cache_admission, isolate_trial_cache,
  server.prefix_caching / prefix_match_unit / max_num_batched_tokens / gpu_memory_utilization, cache_experiment
  except collapse_log), and the trial's ARENA_GPU_MEMORY_UTILIZATION;
- the CPU prefix tier is v87b's: cpu_cache_gib equal, arc_offload_runtime.py byte-equal, the
  trial's ARC_OFFLOAD_* env equal, the fixed engine_id in serving.py (--allow-no-tier: the pre-tier designs);
- the gate is v87b's exact reserving KVGate (no arc2.admission in kv mode) with no L4-match caps (gate_match,
  prefill), so no request can be preempted;
- streams follow v87b's formula: slots = gate capacity / reserved blocks a stream (KV_BLOCKS x kv_margin /
  BLOCKS_PER_STREAM, within 5%), active_limit = generation_slots = slots, max_num_seqs = max(25, slots + 4), graphs
  to min(256, max_num_seqs x (1 + MTP));
- the PLE prefill bound is in (src/arc_infra/expert_prune_runtime.py byte-equal to v87b's);
- the stop is the 20M mark and the wall cannot end the run first at 500 tok/s (no time limits)."""
import json
import sys
from pathlib import Path

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
NO_TIER = "--allow-no-tier" in sys.argv
V9B = "--v9b" in sys.argv  # the v9b INT8 tree (build_p75_read.py): its own loader patches, 2,680 blocks
B = Path(argv[0])
V = Path(argv[1] if len(argv) > 1 else
         "./.claude/worktrees/v84sub/dist/arena-flash-competition-v87b-sub")
KV_BLOCKS, BLOCKS_PER_STREAM = 2180, 28.5  # v3b served layout; median reserved blocks a stream in 10 hx runs
STOP_TOKENS, MIN_RATE = 20_000_000, 500.0
KV_BLOCKS = 2680 if V9B else KV_BLOCKS
for a in sys.argv[1:]:  # another tree's boot pool (build_p75_read.py c647)
    if a.startswith("--kv-blocks="):
        KV_BLOCKS = int(a.split("=")[1])
for a in sys.argv[1:]:  # a coordinator-ordered short mechanism read (task subset) stops at its own token mark
    if a.startswith("--stop-tokens="):
        STOP_TOKENS = int(a.split("=", 1)[1])
vs = json.loads((V / "configs/comparison.json").read_text())
bs = json.loads((B / "configs/comparison.json").read_text())
fail = []
for key in ("kv_margin", "cache_admission", "isolate_trial_cache"):
    if vs.get(key) != bs.get(key):
        fail.append(f"{key}: {bs.get(key)} vs v87b {vs.get(key)}")
for key in ("prefix_caching", "prefix_match_unit", "max_num_batched_tokens", "gpu_memory_utilization"):
    if vs["server"].get(key) != bs["server"].get(key):
        fail.append(f"server.{key}: {bs['server'].get(key)} vs v87b {vs['server'].get(key)}")
skip = ("cpu_cache_gib",) if NO_TIER else ()
cv, cb = vs["cache_experiment"], bs["cache_experiment"]
for key in sorted(set(cv) | set(cb)):
    if key in skip:
        continue
    a, b = cv.get(key), cb.get(key)
    if key == "prefix_cache":
        a = {k: v for k, v in (a or {}).items() if k != "collapse_log"}
        b = {k: v for k, v in (b or {}).items() if k != "collapse_log"}
    if a != b:
        fail.append(f"cache_experiment.{key}: {str(b)[:80]} vs v87b {str(a)[:80]}")
env = "ARENA_GPU_MEMORY_UTILIZATION"
if not NO_TIER:
    off = "src/arc_infra/arc_offload_runtime.py"
    if (B / off).read_bytes() != (V / off).read_bytes():
        fail.append("CPU tier: arc_offload_runtime.py differs from v87b's")
    ve = {k: v for k, v in vs["trials"][0].get("server_env", {}).items() if k.startswith("ARC_OFFLOAD")}
    be = {k: v for k, v in bs["trials"][0].get("server_env", {}).items() if k.startswith("ARC_OFFLOAD")}
    if ve != be:
        fail.append(f"CPU tier env {be} vs v87b {ve}")
    if "'engine_id': 'arc-arena-offload'" not in (B / "src/arena/serving.py").read_text():
        fail.append("CPU tier: serving.py lacks the fixed engine_id (v87b's /dev/shm leak fix)")
if vs["trials"][0]["server_env"].get(env) != bs["trials"][0].get("server_env", {}).get(env):
    fail.append(f"trial {env} differs from v87b")
spec = bs.get("arc2") or {}
if (spec.get("admission") or {}).get("mode") == "kv":
    fail.append("arc2.admission mode kv: v87b's gate is the exact reserving KVGate")
for cap in ("gate_match", "prefill"):
    if cap in spec:
        fail.append(f"arc2.{cap}: an L4-match cap on an RTX play")
slots = int(bs["generation_slots"])
cap = max(int(a.get("cap") or 16384) for a in bs["trials"][0]["arms"])  # reply cap: reservation = prompt + cap
per = BLOCKS_PER_STREAM * (1.095 if cap == 24576 else 1.0)  # 24K: naive4 measured KV 0.686 at 58 slots -> 67 for 0.8
if spec.get("blocks_per_stream"):  # a window change: the builder's measured-rescaled reserve a stream (its reason in kv)
    per = float(spec["blocks_per_stream"])
assert cap in (16384, 24576), f"reply cap {cap}: measure its reserved blocks a stream first"
formula = KV_BLOCKS * float(bs["kv_margin"]) / per
srv = bs["server"]
seqs, mtp = int(srv["max_num_seqs"]), int(srv.get("mtp_tokens") or 0)
if abs(slots / formula - 1) > 0.05 or bs["active_limit"] != slots:
    fail.append(f"streams: slots {slots} / active_limit {bs['active_limit']} vs formula {formula:.1f}")
live = int((bs.get("arc2") or {}).get("live_slots_max") or slots)  # workers above the slots (live raise)
if seqs != int(spec.get("live_seqs") or max(25, slots + 4)) or not slots <= live <= seqs:
    fail.append(f"max_num_seqs {seqs} != max(25, slots + 4), or live slot ceiling {live} outside slots..max_num_seqs")
sizes = srv.get("cudagraph_capture_sizes")
if sizes and sizes[-1] != srv["max_cudagraph_capture_size"]:  # arena.serving.validate_graph_capture_sizes refuses it
    fail.append(f"cudagraph_capture_sizes ends at {sizes[-1]} != max_cudagraph_capture_size {srv['max_cudagraph_capture_size']}")
if srv["max_cudagraph_capture_size"] < min(256, seqs * (1 + mtp)):
    fail.append(f"graphs to {srv['max_cudagraph_capture_size']} < min(256, {seqs} x {1 + mtp})")
ple = "src/arc_infra/expert_prune_runtime.py"
if not V9B and (B / ple).read_bytes() != (V / ple).read_bytes():
    fail.append("PLE prefill bound: expert_prune_runtime.py differs from v87b's")
trial = bs["trials"][0]
if not (B / "src/arc2/served.py").is_file() or spec.get("served_log") is False:  # SFT-reusable runs
    fail.append("served-input log missing (src/arc2/served.py) or off: every run must be reusable for SFT")
if int(trial.get("copies") or 2) == 1 and not any(a.startswith("--stop-tokens=") for a in sys.argv[1:]):
    STOP_TOKENS = 10_000_000  # c120: each task once, reads end at 10M
if max((trial.get("marks") or {}).values(), default=0) != STOP_TOKENS:
    fail.append(f"stop: largest mark must be {STOP_TOKENS:,}")
if float(trial["wall_seconds"]) < STOP_TOKENS / MIN_RATE or bs["notebook_seconds"] < float(trial["wall_seconds"]) + 3600:
    fail.append(f"wall {trial['wall_seconds']} / notebook {bs['notebook_seconds']} could end the run before 20M")
if int(spec.get("timeout_cap_seconds") or 25200) < int(bs["notebook_seconds"]) + 300:
    fail.append("arc2.timeout_cap_seconds below notebook_seconds + 300: the launcher would cut the run before 20M")
if trial.get("fork"):  # fk-naive2/3/5 p2 (29 Oct) ran a stale fork.py with the parked bug: the bundle's must be HEAD's
    import subprocess
    head = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "show", "HEAD:src/arc2/fork.py"],
                          capture_output=True).stdout
    if not head or (B / "src/arc2/fork.py").read_bytes() != head:
        fail.append("fork bundle: src/arc2/fork.py differs from HEAD's (stale bundle: rebuild it)")
for line in fail:
    print(line)
print(("STACK OK " if not fail else "STACK FAILED ") + B.name)
sys.exit(1 if fail else 0)
