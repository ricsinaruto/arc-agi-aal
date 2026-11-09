"""Whole-game DEV check of the hx4 opt-in harness flags on a throughput-arm quality-protocol bundle.

usage: build_hx4_quality.py <baseline bundle> <output> --flags=<opt-in dfr_harness flags, comma separated>
                            [--base=652ba58d] [--fixes=HEAD]

The output is the baseline bundle (quality_eval version 2: DEV10 x 4 replicas, 200K generated tokens per game) with
``git diff <base> <fixes>`` of the harness files applied and the named flags switched on; nothing else changes, so
the run compares with the baseline's replicates. Launch: ARC_VALIDATION_TEST=hx4-quality scripts/modal_validation.py.
"""
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["src/arena/dfr_harness.py", "src/arena/duck.py", "src/arena/sim_check.py", "src/duck_transfer/sandbox.py"]
# Single hunks whose context the v7 baseline lacks (v8-only neighbours): (file, line the added lines follow, a line
# the hunk must add).
REPAIRS = [("src/arena/duck.py", "            await self.hx_stuck_probe()\n", "            self.sc_check()"),
           ("src/duck_transfer/sandbox.py", "        self.hx_state = None\n", "        self.sc_state = None")]


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


base, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
flags = [f for f in arg("flags", "").split(",") if f]
revs = [subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", arg(k, d)], text=True).strip()
        for k, d in (("base", "652ba58d"), ("fixes", "HEAD"))]
if out.exists():
    shutil.rmtree(out)
shutil.copytree(base, out, ignore=shutil.ignore_patterns("dataset-upload"))
patch = subprocess.check_output(["git", "-C", str(ROOT), "diff", *revs, "--", *FILES])
subprocess.run(["patch", "-p1", "--no-backup-if-mismatch", "-d", str(out)], input=patch)
rejects = sorted(str(p.relative_to(out))[:-4] for p in out.rglob("*.rej"))
for name, after, pinned in REPAIRS:
    if name in rejects:
        text = (out / name).read_text()
        reject = (out / (name + ".rej")).read_text()
        added = [line[1:] for line in reject.splitlines() if line.startswith("+") and not line.startswith("+++")]
        if text.count(after) != 1 or reject.count("@@ -") != 1 or pinned not in added:
            raise SystemExit(f"{name}: the rejected hunk is not the known one")
        (out / name).write_text(text.replace(after, after + "\n".join(added) + "\n"))
        (out / (name + ".rej")).unlink()
        rejects.remove(name)
if rejects:
    raise SystemExit(f"patch rejects left: {rejects}")
sys.path.insert(0, str(ROOT / "scripts"))
from build_arena import manifest  # noqa: E402

opt_in = runpy.run_path(str(out / "src/arena/dfr_harness.py"))["OPT_IN"]
if not flags or set(flags) - set(opt_in):
    raise SystemExit(f"--flags from {opt_in}")
path = out / "configs/comparison.json"
study = json.loads(path.read_text())
baseline = study["name"]
study["dfr_harness"] = {**study["dfr_harness"], **{f: True for f in flags}}
study["name"] = out.name
study["hx4"] = {"baseline": baseline, "flags": flags, "patch": {"base": revs[0], "fixes": revs[1], "files": FILES}}
path.write_text(json.dumps(study, indent=2) + "\n")
identity = json.loads((out / "MANIFEST.json").read_text())
identity.update(files=manifest(out), study=out.name)
(out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
runpy.run_path(str(out / "bootstrap.py"))["verify"](out)
print(json.dumps({"bundle": out.name, "baseline": baseline, "dfr_harness": study["dfr_harness"],
                  "quality_eval": study["quality_eval"], "workers": study["active_limit"],
                  "wall_seconds": study["wall_seconds"], "notebook_seconds": study["notebook_seconds"]}))
