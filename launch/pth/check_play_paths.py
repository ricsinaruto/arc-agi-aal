"""check_play_paths.py BUNDLE_DIR PLAY ROUND: the play bundle's weight paths match this launch (c7, 24 Oct: g1-v10's
nebius_stream.in still copied c4's deltas while its follower watched c7's; the play never followed). Exit 1 on any
mismatch."""
import json
import sys
from pathlib import Path

bundle, play, round_ = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
# DELTA_RUN (4th arg, 24 Oct): a resumed restart under a new run id publishes its deltas under this name (HEAL_DELTA_RUN)
delta_run = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else round_
c = json.loads((bundle / "configs" / "comparison.json").read_text())
want = f"{delta_run}/deltas"
bad = []
wf = (c.get("weights_follow") or {}).get("dir", "")
if wf != f"/runs/validation/{play}/exchange-in/{want}":
    bad.append(f"weights_follow.dir {wf!r} != /runs/validation/{play}/exchange-in/{want}")
si = (c.get("nebius_stream") or {}).get("in", "")
if si != want:
    bad.append(f"nebius_stream.in {si!r} != {want!r}")
for t in c.get("trials", []):
    for k, v in (t.get("server_env") or {}).items():
        if isinstance(v, str) and v.startswith("/runs/validation/") and not v.startswith(f"/runs/validation/{play}/"):
            bad.append(f"server_env {k} {v!r} is not under /runs/validation/{play}/")
        pre = f"/runs/validation/{play}/exchange-in/"  # a cache path whose entry is not copied in falls back silently
        if isinstance(v, str) and v.startswith(pre) and not any(
                v[len(pre):] == e or v[len(pre):].startswith(e.rstrip("/") + "/") for e in c.get("nebius_exchange_in") or []):
            bad.append(f"server_env {k} {v!r}: entry not in nebius_exchange_in")
if bad:
    print("REFUSED play bundle " + bundle.name + ": " + "; ".join(bad))
    sys.exit(1)
print(f"play bundle {bundle.name} OK for {play} / {round_}" + (f" (deltas under {delta_run})" if delta_run != round_ else ""))
