"""Launch-time strict check of a dfr-exact bundle (build_v82_ablation.check_bundle): unread study / server keys and
the expert-overlay coverage check. Usage: check_bundle.py BUNDLE (exit 1 on a failure)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v82_ablation as b  # noqa: E402

out = Path(sys.argv[1]).resolve()
study = json.loads((out / "configs/comparison.json").read_text())
name = (study.get("serving_ablation") or {}).get("base_bundle")
base = Path(__file__).resolve().parents[2] / "dist/base" / name / "configs/comparison.json" if name else None
base_study = json.loads(base.read_text()) if base and base.exists() else {}
b.check_bundle(out, study, base_study)
print(f"bundle check ok: {out.name}")
