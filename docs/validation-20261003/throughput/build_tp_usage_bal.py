"""Balanced all-25 refit, step 1: the routing probe on equal tokens per family.

expert-usage measurement (arena.expert_usage, Modal test expert-usage-measure-v2) on a balanced inputs folder
(scripts/expert_usage_inputs.py --balanced-per-family): routing dumped per family, the 0.375 mask ranked on their sum,
and on the same 25-family screen split the new mask plus the dev t37 and t37-all masks against the unpruned model.

argv: inputs output
"""
import json
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
from build_arena import build, manifest  # noqa: E402

inputs, out = Path(sys.argv[1]), Path(sys.argv[2])
EXTRA = {"t37dev": "data/expert_prune/prune-t37.json", "t37all": "data/expert_prune/prune-t37-all.json"}
output, study = build(out.name, output=out, dependencies=True, paired_30m=True, expert_usage=str(inputs),
                      expert_usage_measure_only=True, expert_usage_fractions=[0.375])
masks = {}
for name, src in EXTRA.items():
    rel = f"configs/expert-usage/masks/{Path(src).name}"
    (output / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / src, output / rel)
    masks[name] = rel
study["expert_usage"].update(per_family=True, extra_masks=masks)
study.update(notebook_seconds=2640, kernel_timeout=2700)
(output / "configs/comparison.json").write_text(json.dumps(study, indent=2) + "\n")
identity = json.loads((output / "MANIFEST.json").read_text())
identity["files"] = manifest(output)
(output / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
runpy.run_path(str(output / "bootstrap.py"))["verify"](output)
print(json.dumps({"output": str(output), "usage": study["expert_usage"]["inputs"]["usage"]["tokens"],
                  "screen": study["expert_usage"]["inputs"]["heldout"]["tokens"], "masks": masks}))
