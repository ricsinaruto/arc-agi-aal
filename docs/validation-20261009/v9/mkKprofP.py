"""kprofP: v9's PREFILL as served, profiled. The exact v9 v1 tree (chunked prefill, 4,096 batched
tokens) with one serving-bench trial running arena.kernel_profile on real v9-harness contexts (kprofP-contexts.json:
48 x ~64K, 48 x ~100K, 48 x ~128K; mkctxP.py), patched with two bench-only additions:
- subset "range:A:B" (contexts A..B-1);
- a "prefill" point: cold cache (isolate_trial), the chosen contexts sent at once with max_tokens 1, a torch trace
  (~40 steps) taken ``lead_seconds`` after the send, then every reply awaited; records the window's engine counters
  (ms/step, uncached tokens/step and /s) over the whole prefill.
Points: PF64x4 / PF100x4 / PF128x4 (pure prefill of drain-recompute-like fresh prompts), D64x48 (pure decode on
the ~64K set; warms it) and MX48: 48 streams decoding
on the ~64K set while an injector adds a fresh ~60K prompt every 20 s (a play-like drain recompute in mixed steps).
Usage: mkKprofP.py CONTEXTS [v87b]  (v87b: the v8.7b 9 h reference tree, the 3 prefill points only)"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

D = Path("./.claude/worktrees/v84sub/dist")
ARM = sys.argv[2] if len(sys.argv) > 2 else "v9"
V9 = {"v9": "b2-v9-p75c268-kf-ho10x11-9h-v1", "v87b": "b2-v9ref-v87b-kf-ho10x11-9h-v1"}[ARM]
contexts = Path(sys.argv[1])
out = D / f"b2-kprofP-{ARM}-v1"
if out.exists():
    shutil.rmtree(out)
shutil.copytree(D / V9, out, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "dataset-upload"))

kp = out / "src/arena/kernel_profile.py"
t = kp.read_text()
SUB_OLD = '''    if subset == "longest":
        return sorted(sorted(range(len(counts)), key=lambda i: -counts[i])[:streams])
    raise ValueError(f"unknown subset {subset!r}")'''
SUB_NEW = '''    if subset == "longest":
        return sorted(sorted(range(len(counts)), key=lambda i: -counts[i])[:streams])
    if subset.startswith("range:"):  # kprofP (30 Oct): contexts A..B-1
        a, b = (int(x) for x in subset.split(":")[1:3])
        if b - a != streams or not 0 <= a < b <= len(counts):
            raise ValueError(f"subset {subset!r} for {streams} streams over {len(counts)} prompts")
        return list(range(a, b))
    raise ValueError(f"unknown subset {subset!r}")'''
PF_ANCHOR = '''        tasks = [asyncio.create_task(one(i, b)) for i, b in enumerate(requests)]
        stop, injector, injected, inject_tasks = asyncio.Event(), None, [], []
'''
PF_NEW = '''        if point.get("prefill"):  # kprofP (30 Oct): cold-cache prefill of the chosen contexts, max_tokens 1
            record["cache_reset"] = await isolate_trial(bench.base_url)
            requests = [{**bodies[i], "max_tokens": 1} for i in chosen]
            first = await sample()
            started = time.monotonic()
            tasks = [asyncio.create_task(one(i, b)) for i, b in enumerate(requests)]
            await asyncio.sleep(float(point.get("lead_seconds", 3)))
            if point.get("profile"):
                try:
                    record["profile"] = await profile(point["label"], float(point.get("profile_seconds", 4)))
                except Exception as exc:  # noqa: BLE001 - a failed profile keeps the timing
                    record["profile_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                    result["errors"].append(f"profile {point['label']}: {record['profile_error']}")
            await asyncio.gather(*tasks)
            record["prefill_seconds"] = round(time.monotonic() - started, 3)
            last = await sample()
            record.update(window_stats(first, last, streams))
            record["statuses"] = sorted(set(statuses))
            record["prefill_tok_s"] = round(sum(sizes) / record["prefill_seconds"], 1)
            record["gpu_after"] = await smi(("clocks.sm", "power.draw", "temperature.gpu", "memory.free"))
            await wait_for_idle(sample)
            return record
''' + PF_ANCHOR
for old, new in ((SUB_OLD, SUB_NEW), (PF_ANCHOR, PF_NEW)):
    assert t.count(old) == 1, old[:60]
    t = t.replace(old, new)
compile(t, "kernel_profile.py", "exec")
kp.write_text(t)

p = out / "configs/comparison.json"
s = json.loads(p.read_text())
assert len(s["trials"]) == 1 and s["trials"][0]["kind"] == "duck-transfer" and s["server"]["max_num_batched_tokens"] == 4096
s["name"] = out.name
(out / "configs/bench").mkdir(parents=True, exist_ok=True)
shutil.copyfile(contexts, out / "configs/bench/kprof-contexts.json")
n_ctx = len(json.loads(contexts.read_text())["contexts"])
assert n_ctx == 144, n_ctx
env = {**s["trials"][0]["server_env"], "ARENA_TORCH_PROFILER": "1"}
pf = {"content": "real", "prefill": True, "profile": True, "lead_seconds": 3, "profile_seconds": 4}
points = [{**pf, "label": "PF64x4", "streams": 4, "subset": "range:0:4"},
          {**pf, "label": "PF100x4", "streams": 4, "subset": "range:48:52"},
          {**pf, "label": "PF128x4", "streams": 4, "subset": "range:96:100"},
          # D64x48 warms the ~64K set (pure decode required) so MX48's settle finds the prefixes cached
          {"label": "D64x48", "content": "real", "streams": 48, "subset": "range:0:48", "profile": True},
          {"label": "MX48", "content": "real", "streams": 48, "subset": "range:0:48", "profile": True,
           "inject": {"prefix_tokens": 256, "tail_tokens": 60000, "rate": 0.05}}]
if ARM == "v87b":  # prefill comparison: the PF points; v9's run ended silently after D64x48
    points = points[:3]
s["trials"] = [{"name": "kprof", "kind": "serving-bench", "competition": False, "wall_seconds": 5400, "server_env": env}]
s["throughput_bench"] = {"prompt_tokens": [], "concurrency": [], "shared_prefix": None, "output_tokens": 32768,
                         "kernel_profile": {"contexts_file": "configs/bench/kprof-contexts.json", "output_tokens": 32768,
                                            "settle_seconds": 1800, "window_seconds": 15, "profile_seconds": 4,
                                            "synthetic_streams": 48, "points": points}}
s["wall_seconds"] = 5400
s["chain"] = {"trials": ["kprof"]}
s["notebook_seconds"] = 9000  # setup (~2,450 s) + 3 prefill points (~5 min) + two settles (<= 1,800 s each) + margin
s["kaggle_faithful"]["rerun"] = False
s["b200"]["stop"]["wall_seconds"] = 9000
s["roster"] = "kernel profile v9 prefill: no games; cold-cache prefills and a mixed decode + drain-recompute point"
s.setdefault("evidence", {})["kprofP"] = f"from {V9}: serving-bench kernel_profile prefill points (bench-only patch)"
p.write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((out / "MANIFEST.json").read_text())
m["study"] = out.name
m["files"] = {q.relative_to(out).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(out.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(out / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(out)
