"""kprofH3 (30 Oct): TURNS44 worker stacks by one-shot `py-spy dump` at 2 Hz for 240 s (arena.spy_dump), replacing
`py-spy record`, which hung on the worker under turns in kprofH and kprofH2 and lost the point in H2."""
import hashlib, json, re, shutil
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
SRC, OUT = ROOT / "dist/b2-kprofH2-v9-v1", ROOT / "dist/b2-kprofH3-v9-v1"
shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT, ignore=shutil.ignore_patterns("__pycache__"))
kp = next(p for p in OUT.rglob("kernel_profile.py"))
shutil.copyfile(ROOT / "src/arena/spy_dump.py", kp.parent / "spy_dump.py")
t = kp.read_text()
new = '''    async def spy_start(label, seconds):
        # kprofH3: one-shot dumps of the worker (arena.spy_dump); a hang costs one sample and nothing raises
        exe = Path(__file__).resolve().parents[2] / "tools/py-spy"
        os.chmod(exe, 0o755)
        pid = serving_processes()["worker"][0]
        return SpyDump(exe, pid, Path(bench.output) / f"pyspy-{label}" / f"worker-{pid}.jsonl", seconds).start()

    async def spy_end(spy, record):
        record["pyspy"] = await spy.finish()

'''
t, n = re.subn(r"    async def spy_start\(label, seconds\):.*?(?=    async def measure\(point\):)", lambda m: new, t, flags=re.S)
assert n == 1
a = "from .isolation import isolate_trial\n"
assert t.count(a) == 1
t = t.replace(a, a + "from .spy_dump import SpyDump\n")
compile(t, "kernel_profile.py", "exec")
kp.write_text(t)
p = OUT / "configs/comparison.json"
s = json.loads(p.read_text())
point = s["throughput_bench"]["kernel_profile"]["points"][0]
assert point["label"] == "TURNS44" and point["window_seconds"] == 60
point["window_seconds"] = 240
s["name"] = OUT.name
s["roster"] = "host-side profile v9: no games; TURNS44 with one-shot py-spy dumps of the worker"
s["evidence"]["kprofH3"] = "from b2-kprofH2-v9-v1: arena.spy_dump at 2 Hz, 240 s window"
p.write_text(json.dumps(s, indent=2).replace(SRC.name, OUT.name) + "\n")
m = json.loads((OUT / "MANIFEST.json").read_text())
m["study"] = OUT.name
m["files"] = {q.relative_to(OUT).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(OUT.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(OUT)
