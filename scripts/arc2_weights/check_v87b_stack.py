"""check_v87b_stack.py BUNDLE [V87B]: CPU gate for arc2 RTX plays (rule b: every RTX play matches
the v87b submission serving stack). Fails on any difference from arena-flash-competition-v87b-sub outside the
exceptions declared in build_v87b_rtx.py:
- files: every v87b file (src, bootstrap, requirements, wheels, draft head, configs) byte-equal, except
  src/arena/engine_watchdog.py and src/arena/run.py (whose only change must be the arc2 trial-kind block); extra
  files only under src/arc2/, data/arc2/ and the arc2 k256 mask;
- study: every flattened key equal except the harness (arc2, trials, windows), weights (mask, overlay, datasets,
  dn4g8), stream and wall keys, which are checked by rule: trial env = v87b's less ARC_EXPERT_INT3_*; CacheGate
  admission with target <= 0.9 and a positive step; no L4-match caps (gate_match, prefill); ceiling >= 96 and
  active_limit <= generation_slots <= max_num_seqs; graphs to min(256, max_num_seqs x (1 + MTP)); the stop is the
  20M mark and the wall cannot end the run first at 500 tok/s (no time limits)."""
import json
import sys
from pathlib import Path

B = Path(sys.argv[1])
V = Path(sys.argv[2] if len(sys.argv) > 2 else
         "./.claude/worktrees/v84sub/dist/arena-flash-competition-v87b-sub")
DROPPED = ("dataset-upload/", "experiment.ipynb", "kernel-metadata.json", "build-manifest.json", "MANIFEST.json")
HARNESS_FILES = {"src/arena/engine_watchdog.py", "src/arena/run.py", "configs/comparison.json"}
EXTRA_OK = ("src/arc2/", "data/arc2/", "configs/expert-prune/prune-arc2-k256-v1.json", "MANIFEST.json")
ALLOWED = ("name", "arc2", "trials", "windows", "model_datasets", "server.expert_prune", "server.expert_overlay",
           "dn4g8", "wall_seconds", "notebook_seconds", "generation_slots", "active_limit", "server.max_num_seqs",
           "server.cudagraph_capture_sizes", "server.max_cudagraph_capture_size")
STOP_TOKENS, MIN_RATE = 20_000_000, 500.0


def flat(d, p=""):
    o = {}
    if isinstance(d, dict):
        for k, v in d.items():
            o.update(flat(v, f"{p}.{k}" if p else str(k)))
    elif isinstance(d, list) and any(isinstance(x, (dict, list)) for x in d):
        for i, v in enumerate(d):
            o.update(flat(v, f"{p}.{i}"))
    else:
        o[p] = d
    return o


def files(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


fail = []
fv, fb = files(V), files(B)
for rel in sorted(fv):
    if rel.startswith(DROPPED) or rel in HARNESS_FILES:
        continue
    if rel not in fb:
        fail.append(f"missing v87b file {rel}")
    elif (B / rel).read_bytes() != (V / rel).read_bytes():
        fail.append(f"differs from v87b: {rel}")
for rel in sorted(fb - fv):
    if not rel.startswith(EXTRA_OK):
        fail.append(f"extra file not in v87b: {rel}")
# run.py: v87b's text with exactly one inserted block, which dispatches the arc2 kind and nothing else
v_run, b_run = (V / "src/arena/run.py").read_text(), (B / "src/arena/run.py").read_text()
start, end = b_run.find('        if trial["kind"] == "arc2":\n'), b_run.find('        if trial["kind"] == "level-bench":\n')
block = b_run[start:end] if 0 <= start < end else ""
if not block or b_run[:start] + b_run[end:] != v_run or block.count('trial["kind"]') != 1 or "raise _Done" not in block:
    fail.append("run.py: the only change must be one inserted arc2 trial-kind block")

vs = json.loads((V / "configs/comparison.json").read_text())
bs = json.loads((B / "configs/comparison.json").read_text())
xv, xb = flat(vs), flat(bs)
for k in sorted(set(xv) | set(xb)):
    if xv.get(k) != xb.get(k) and not (k == "name" or k.startswith(tuple(a + "." for a in ALLOWED) + ALLOWED)):
        fail.append(f"study {k}: v87b {str(xv.get(k))[:80]} vs {str(xb.get(k))[:80]}")
trial = bs["trials"][0]
env_v = {k: v for k, v in vs["trials"][0]["server_env"].items() if not k.startswith("ARC_EXPERT_INT3_")}
if len(bs["trials"]) != 1 or trial.get("kind") != "arc2" or trial.get("server_env") != env_v:
    fail.append("trial: one arc2 trial with v87b's server_env less ARC_EXPERT_INT3_*")
spec = bs.get("arc2") or {}
adm = spec.get("admission") or {}
if adm.get("mode") != "kv" or not 0 < float(adm.get("target", 1)) <= 0.9 or float(adm.get("step", 0)) <= 0:
    fail.append(f"admission: CacheGate (mode kv, target <= 0.9, step > 0) required, got {adm}")
for cap in ("gate_match", "prefill"):
    if cap in spec:
        fail.append(f"arc2.{cap}: an L4-match cap on an RTX play")
srv = bs["server"]
seqs, mtp = int(srv["max_num_seqs"]), int(srv["mtp_tokens"])
if not (96 <= bs["active_limit"] <= bs["generation_slots"] <= seqs):
    fail.append(f"streams: need 96 <= active_limit {bs['active_limit']} <= generation_slots "
                f"{bs['generation_slots']} <= max_num_seqs {seqs}")
need = min(256, seqs * (1 + mtp))
if srv["max_cudagraph_capture_size"] < need or max(srv["cudagraph_capture_sizes"]) != srv["max_cudagraph_capture_size"]:
    fail.append(f"graphs to {srv['max_cudagraph_capture_size']} < min(256, max_num_seqs x (1 + MTP)) = {need}")
marks = trial.get("marks") or {}
if max(marks.values(), default=0) != STOP_TOKENS:
    fail.append(f"stop: largest mark must be {STOP_TOKENS:,}")
if float(trial["wall_seconds"]) < STOP_TOKENS / MIN_RATE or bs["notebook_seconds"] < float(trial["wall_seconds"]) + 3600:
    fail.append(f"wall {trial['wall_seconds']} / notebook {bs['notebook_seconds']} could end the run before 20M "
                f"(need wall >= {STOP_TOKENS / MIN_RATE:.0f}, notebook >= wall + 3600)")
for line in fail:
    print(line)
print(("STACK OK " if not fail else "STACK FAILED ") + B.name)
sys.exit(1 if fail else 0)
