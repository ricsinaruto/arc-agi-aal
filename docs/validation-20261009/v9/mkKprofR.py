"""kprofR (30 Oct, owner yes): per-step log of the graph path on play-like turns, from a kprof tree. No serving change in
any live run. Usage: mkKprofR.py h3 (TURNS44, v9 p75 c268, as kprofH3) | mkKprofR.py cut (kprofC C100x50, 16 ms drained)."""
import hashlib, json, re, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve()
D = HERE.parents[3] / "dist"
arm = sys.argv[1]
SRC = D / {"h3": "b2-kprofH3-v9-v1", "cut": "b2-kprofC-cut-v3", "anchor": "b2-kprofC-anchor-v1", "fix": "b2-kprofC-anchor-v1",
           "cutfix": "b2-kprofC-cut-v3"}[arm]
# every FULL graph's size a multiple of the decode width (1 + MTP 2): the old list's 128, 160, 176 shadow 129, 162, 177
SIZES = [3, 6, 9, 18, 33, 48, 60, 66, 72, 81, 90, 96, 105, 114, 129, 144, 162, 177, 192]
OUT = D / f"b2-kprofR-{arm}-v1"
shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "dataset-upload"))
g = OUT / "src/arc_infra/gdn_defer_runtime.py"
t, A = g.read_text(), '    install_runner(GPUModelRunnerV2, "finish_requests")\n'
assert t.count(A) == 1 and "install_steplog" not in t
t = t.replace(A, A + "    install_steplog()\n") + "\n\n" + (HERE.parent / "steplog_runtime.py").read_text().split('"""', 2)[2].lstrip("\n")
compile(t, g.name, "exec")
g.write_text(t)
kp = next(p for p in OUT.rglob("kernel_profile.py"))
t = kp.read_text()
new = '''    async def spy_start(label, seconds):
        return label  # kprofR: no py-spy, the window copies the step log

    async def spy_end(spy, record):
        shutil.copyfile("/tmp/arc_steplog.jsonl", Path(bench.output) / f"steplog-{spy}.jsonl")
        record["steplog_end_epoch"] = time.time()

'''
if arm in ("h3",):
    t, n = re.subn(r"    async def spy_start\(label, seconds\):.*?(?=    async def measure\(point\):)", lambda m: new, t, flags=re.S)
    assert n == 1
else:  # no spy hooks in the kprofC trees
    A = '    result["host_end"] = {"gpu": await smi()}\n'
    assert t.count(A) == 1
    t = t.replace(A, '    shutil.copyfile("/tmp/arc_steplog.jsonl", Path(bench.output) / "steplog-window.jsonl")\n' + A)
compile(t, kp.name, "exec")
kp.write_text(t)
p = OUT / "configs/comparison.json"
s = json.loads(p.read_text().replace(SRC.name, OUT.name))
env = s["trials"][0]["server_env"]
env.update(ARC_STEPLOG="/tmp/arc_steplog.jsonl", ARC_STEPLOG_SYNC="64" if arm.endswith("fix") else "16")
if arm.endswith("fix"):
    assert s["server"]["max_cudagraph_capture_size"] == SIZES[-1] and s["server"]["mtp_tokens"] == 2
    s["server"]["cudagraph_capture_sizes"] = SIZES
kpf = s["throughput_bench"]["kernel_profile"]
assert len(kpf["points"]) == 1
pt = kpf["points"][0]
if not arm.endswith("fix"):
    env.pop("ARENA_TORCH_PROFILER", None)
    pt["pyspy"] = True
    pt["profile"] = False
if arm.endswith("fix"):  # one change from the kprofC arm: the list; the log has no drained steps
    env["ARC_STEPLOG_SYNC"] = "1000000000"
elif arm == "h3":
    assert pt["label"] == "TURNS44"
    pt["window_seconds"], pt["turns"]["warm_seconds"] = 300, 480
    s["notebook_seconds"] = 4500
    s["b200"]["stop"]["wall_seconds"] = 4500
else:
    pt["window_seconds"] = 300
    kpf["settle_seconds"] = 1800
    s["notebook_seconds"] = s["b200"]["stop"]["wall_seconds"] = 6000
s["roster"] = f"kernel profile, per-step graph path log, {arm}: no games"
s.setdefault("evidence", {})["kprofR"] = f"from {SRC.name}: steplog_runtime.py, ARC_STEPLOG, no py-spy, no torch profiler"
p.write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((OUT / "MANIFEST.json").read_text().replace(SRC.name, OUT.name))
m["study"] = OUT.name
m["files"] = {q.relative_to(OUT).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(OUT.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(OUT)
