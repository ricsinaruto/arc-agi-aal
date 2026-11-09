"""Inodes and bytes on the shared runs volume (arc-prolong-runs), for the dashboard (ALERT at 80% of a cap). A Modal CPU
walk; the dash loop runs it hourly in the background. Output <scratch>/mon/volume.json:
{"t", "volume", "inodes", "bytes", "inode_cap", "byte_cap", "top": [[prefix, inodes, bytes], ...]}.

  volume_usage.py

Caps: Modal v1 volumes hold at most 500,000 files (INODE_CAP); BYTE_CAP (ARC_VOLUME_BYTE_CAP, bytes) is optional.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import modal
except ImportError:  # mon.py imports only alerts(); a uv sync without the modal extra can drop it
    modal = None

INODE_CAP = 500_000
VOLUME = "arc-prolong-runs"

if modal is not None:
    app = modal.App("arc-volume-usage")

    @app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=(0.125, 2), timeout=3600,
                  volumes={"/runs": modal.Volume.from_name(VOLUME)})
    def walk():
        """Every file and dir under /runs, grouped by top-level dir (validation/ by its run dirs)."""
        from collections import Counter

        n, size, per_n, per_b, new_h = 0, 0, Counter(), Counter(), Counter()
        hour = time.time() - 3600
        for root, dirs, files in os.walk("/runs"):
            parts = os.path.relpath(root, "/runs").split("/")
            key = "." if parts == ["."] else "/".join(parts[:2]) if parts[0] == "validation" else parts[0]
            key3 = "/".join(parts[:3] if parts[0] == "validation" else parts[:2])  # growth: one level deeper
            n += len(dirs) + len(files)
            per_n[key] += len(dirs) + len(files)
            for i, f in enumerate(files + dirs):
                try:
                    st = os.lstat(os.path.join(root, f))
                except OSError:
                    continue
                if st.st_mtime > hour:  # created or rewritten in the last hour: who is adding inodes
                    new_h[key3] += 1
                if i < len(files):
                    size += st.st_size
                    per_b[key] += st.st_size
        top = sorted(per_n, key=lambda k: -per_n[k])[:15]
        return {"inodes": n, "bytes": size, "top": [[k, per_n[k], per_b[k]] for k in top],
                "new_1h": sorted(new_h.items(), key=lambda x: -x[1])[:25]}


def alerts(v, frac=0.8):
    """Banner lines for a volume.json reading at or over ``frac`` of a cap."""
    out = []
    if v.get("inode_cap") and v["inodes"] >= frac * v["inode_cap"]:
        out.append(f"volume {v['volume']}: {v['inodes']:,} inodes = {v['inodes'] / v['inode_cap']:.0%} of "
                   f"{v['inode_cap']:,} (ALERT at {frac:.0%}): tar or delete old run dirs (largest: "
                   + ", ".join(f"{k} {i:,}" for k, i, _ in v.get("top", [])[:3]) + ")")
    if v.get("byte_cap") and v["bytes"] >= frac * v["byte_cap"]:
        out.append(f"volume {v['volume']}: {v['bytes'] / 2**40:.2f} TiB = {v['bytes'] / v['byte_cap']:.0%} of its "
                   f"byte cap (ALERT at {frac:.0%})")
    return out


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paths

    with app.run():
        try:
            sys.path.insert(0, str(paths.ROOT_PATH / "src"))
            from arc_infra.runops import records

            records.register_cpu_app(app.app_id, name="arc-volume-usage")
        except Exception as e:  # noqa: BLE001 - the ledger is best effort for a cents job
            print(f"(cpu-app registration skipped: {e!r})")
        r = walk.remote()
    cap = os.environ.get("ARC_VOLUME_BYTE_CAP")
    v = {"t": time.time(), "volume": VOLUME, **r, "inode_cap": INODE_CAP, "byte_cap": int(cap) if cap else None}
    out = paths.SCRATCH / "mon/volume.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(v, indent=1))
    print(f"{VOLUME}: {v['inodes']:,} inodes ({v['inodes'] / INODE_CAP:.0%} of {INODE_CAP:,}), "
          f"{v['bytes'] / 2**40:.2f} TiB")
    for a in alerts(v):
        print("ALERT " + a)


if __name__ == "__main__":
    main()
