"""kprofH2 (30 Oct): the TURNS44 worker stacks kprofH lost (py-spy timed out on the worker, no file). Same bundle, same
point; py-spy on the worker only, 67 Hz, 300 s wait, SIGINT on overrun so the file is still written."""
import hashlib, json, shutil
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[3] / "dist/b2-kprofH-v9-v1"
OUT = HERE.parents[3] / "dist/b2-kprofH2-v9-v1"
shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT, ignore=shutil.ignore_patterns("__pycache__"))
kp = next(p for p in OUT.rglob("kernel_profile.py"))
t = kp.read_text()
for old, new in (
    ('for role, pids in serving_processes().items():\n', 'for role, pids in serving_processes().items():\n            if role != "worker":\n                continue\n'),
    ('"--rate", "200"', '"--rate", "67"'),
    ('got, _ = await asyncio.wait_for(p.communicate(), 120)\n',
     'got, _ = await asyncio.wait_for(p.communicate(), 300)\n'),
    ('            except Exception as exc:  # noqa: BLE001 - diagnostics only\n                res[f"{role}-{pid}"] = {"error": repr(exc)[:200]}\n',
     '            except Exception as exc:  # noqa: BLE001 - diagnostics only\n                p.send_signal(2)\n                got, _ = await asyncio.wait_for(p.communicate(), 120)\n                res[f"{role}-{pid}"] = {"error": repr(exc)[:200], "after_sigint": got.decode(errors="replace")[-300:]}\n'),
):
    assert t.count(old) == 1, old
    t = t.replace(old, new)
compile(t, "kernel_profile.py", "exec")
kp.write_text(t)
p = OUT / "configs/comparison.json"
s = json.loads(p.read_text())
s["throughput_bench"]["kernel_profile"]["points"] = [q for q in s["throughput_bench"]["kernel_profile"]["points"] if q["label"] == "TURNS44"]
s["name"] = OUT.name
s["notebook_seconds"] = 3600
s["wall_seconds"] = 1800
s["trials"][0]["wall_seconds"] = 1800
s["b200"]["stop"]["wall_seconds"] = 3600
s["roster"] = "host-side profile v9: no games; TURNS44 with py-spy on the worker"
s.setdefault("evidence", {})["kprofH2"] = "from b2-kprofH-v9-v1: worker-only py-spy, 67 Hz, SIGINT fallback"
p.write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((OUT / "MANIFEST.json").read_text())
m["study"] = OUT.name
m["files"] = {q.relative_to(OUT).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(OUT.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(OUT)
