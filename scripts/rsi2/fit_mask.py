"""RSI gen 0: v8.5's t37 mask refit on the 10 fixed selection games. Same rule as arena.expert_usage.masks
(per layer, least usage-set gate-weight mass first, routing count breaks ties), uniform 192 of 512 per layer.
usage: fit_mask.py <expert-usage.json> <out mask.json> <source run id>"""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from arena.expert_usage import masks  # noqa: E402

res = json.loads(Path(sys.argv[1]).read_text())
u = res["usage"]
stats = u.get("stats", u)
layers = stats["dump_layers"]
pruned = masks(stats["weight"], stats["count"], [0.375], layers)["0.375"]
assert {str(k): sorted(v) for k, v in pruned.items()} == {str(k): sorted(v) for k, v in res["masks"]["0.375"].items()}, "differs from the run's own 0.375 mask"
assert len(pruned) == 48 and all(len(v) == 192 for v in pruned.values())
mask = {"format": "arc-expert-prune-v1", "experts": 512, "fraction": 0.375, "pruned_per_layer": 192, "rank_by": "weight",
        "source_run": sys.argv[3],
        "screen": "the 10 fixed selection games (sb26 tr87 r11l m0r0 sc25 re86 cd82 ft09 lp85 vc33), v7.1-harness traffic "
                  "(inputs-fixed10-v71-v1): 950,000 input tokens per game, 131K sequence cap, three level pools rotated "
                  "(3+, 1, 2); routing of the stock NVFP4 checkpoint (no fitted overlay)",
        "pruned": pruned}
Path(sys.argv[2]).write_text(json.dumps(mask))
print(hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest())
