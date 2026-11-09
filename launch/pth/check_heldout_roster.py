"""check_heldout_roster.py BUNDLE_DIR POOL [POOL ...]: refuse a held-out play read whose roster shares a game with a
training pool (the qadv2 play test reused the sft4x-era h10 roster, 7 of whose games are in s2).
POOL: a pool's index.json (local path, or modal:<volume>:<path>); its "train" files' games are the trained games.
launch_rtx.sh runs it when HELDOUT_POOL is set (space-separated pools). Prints ROSTER OK / ROSTER OVERLAP; exit 1 on
overlap or an unreadable pool."""
import json
import subprocess
import sys
from pathlib import Path


def pool_games(spec):
    if spec.startswith("modal:"):
        _, vol, path = spec.split(":", 2)
        text = subprocess.run(["modal", "volume", "get", vol, path, "-"], capture_output=True, text=True,
                              check=True).stdout
        text = text[:text.rfind("}") + 1]
    else:
        text = Path(spec).read_text()
    files = json.loads(text)["train"]["files"]
    if not files:
        raise SystemExit(f"ROSTER CHECK FAILED: {spec} has no train files")
    return {e["game"] for e in files}


study = json.loads((Path(sys.argv[1]) / "configs/comparison.json").read_text())
roster = {g.split("-")[0] for g in study["games"]}
trained = set().union(*(pool_games(p) for p in sys.argv[2:]))
both = sorted(roster & trained)
if both:
    print(f"ROSTER OVERLAP {Path(sys.argv[1]).name}: {' '.join(both)} are in the training pool(s)")
    sys.exit(1)
print(f"ROSTER OK {Path(sys.argv[1]).name}: {len(roster)} games, none of {len(trained)} trained games")
