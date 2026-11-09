"""Retire a run's own Modal volume (per-run volumes). Never scheduled: deleting a volume needs the maintainer's
yes, so it runs by hand on explicit approval only.

  retire.py RUN            check + plan only (what would be archived, nothing written)
  retire.py RUN --archive  the whole volume -> arc-archive:retired/<volume>/ (one tar.gz per top-level folder under
                           validation/, plus one for the rest), member counts verified, manifest.json; the volume stays
  retire.py RUN --delete   after a verified archive (manifest present, counts equal): `modal volume delete` and the
                           record's "retired" field

Refused: a record that is not ended, ended < 48 h ago, has "keep", or lives on the shared arc-prolong-runs."""
import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # in the archive container this file is /root/retire.py
sys.path.insert(0, str(REPO / "src"))
ARCHIVE_VOLUME, MIN_AGE_S = "arc-archive", 48 * 3600
MODAL = str(REPO / ".venv/bin/modal")


def refusal(run, rec, now=None):
    """Why ``run`` cannot be retired, or None."""
    from arc_infra.runops import records as R

    if not rec:
        return f"{run}: no record"
    vol = R.volume_of(run, rec=rec)
    if vol == R.SHARED_VOLUME:
        return f"{run}: on the shared {vol} (only per-run volumes retire)"
    if rec.get("keep"):
        return f"{run}: record has keep={rec['keep']!r}"
    if rec.get("retired"):
        return f"{run}: retired already ({rec['retired']})"
    if R.state_of(rec) != "ended":
        return f"{run}: state {R.state_of(rec)}, not ended"
    end = ((rec.get("ledger") or [{}])[-1].get("row") or {}).get("end_epoch") or rec.get("end_epoch")
    age = (now or time.time()) - float(end or 0)
    if not end or age < MIN_AGE_S:
        return f"{run}: ended {age / 3600:.1f} h ago (< {MIN_AGE_S // 3600} h)"
    return None


def groups(root):
    """{tar name: [files]}: one per validation/<folder>, one "rest" for everything else (relative paths)."""
    root = Path(root)
    out = {}
    for f in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        rel = f.relative_to(root)
        key = f"validation__{rel.parts[1]}" if rel.parts[0] == "validation" and len(rel.parts) > 2 else "rest"
        out.setdefault(key, []).append(str(rel))
    return out


def archive_tree(src, dst):
    """Tar ``src`` into ``dst`` (one tar.gz per group), each verified by member count; manifest.json written last."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    manifest = {"src": str(src), "t": time.time(), "tars": {}}
    for key, files in groups(src).items():
        tar = dst / f"{key}.tar.gz"
        with tarfile.open(str(tar) + ".part", "w:gz", compresslevel=1) as tf:
            for rel in files:
                tf.add(Path(src) / rel, arcname=rel, recursive=False)
        with tarfile.open(str(tar) + ".part", "r:gz") as tf:
            n = sum(1 for m in tf if m.isfile())
        if n != len(files):
            raise RuntimeError(f"{key}: {n} tar members != {len(files)} files")
        os.replace(str(tar) + ".part", tar)
        manifest["tars"][tar.name] = {"files": len(files), "bytes": tar.stat().st_size}
    manifest["files"] = sum(v["files"] for v in manifest["tars"].values())
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def modal_archive(vol):
    import modal

    app = modal.App("arc-retire")
    image = modal.Image.debian_slim(python_version="3.12").add_local_file(__file__, "/root/retire.py", copy=True)
    src, arch = modal.Volume.from_name(vol).read_only(), modal.Volume.from_name(ARCHIVE_VOLUME, create_if_missing=True,
                                                                                 version=2)

    @app.function(image=image, volumes={"/src": src, "/archive": arch}, timeout=4 * 3600, cpu=(0.125, 4), retries=0,
                  serialized=True)
    def run():
        sys.path.insert(0, "/root")
        import retire

        m = retire.archive_tree("/src", f"/archive/retired/{vol}")
        modal.Volume.from_name(ARCHIVE_VOLUME).commit()
        return m

    with modal.enable_output(), app.run():
        return run.remote()


def main(argv=None):
    from arc_infra.runops import records as R

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--archive", action="store_true")
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args(argv)
    rec = R.get(a.run)
    why = refusal(a.run, rec)
    if why:
        raise SystemExit(f"RETIRE REFUSED {why}")
    vol = R.volume_of(a.run, rec=rec)
    if a.archive:
        m = modal_archive(vol)
        print(json.dumps({"archived": vol, "to": f"{ARCHIVE_VOLUME}:retired/{vol}", "files": m["files"],
                          "tars": len(m["tars"])}))
    elif a.delete:
        got = subprocess.run([MODAL, "volume", "get", "--force", ARCHIVE_VOLUME, f"retired/{vol}/manifest.json", "-"],
                             capture_output=True, text=True, timeout=180)
        if got.returncode or not got.stdout.strip().startswith("{"):
            raise SystemExit(f"RETIRE REFUSED {a.run}: no verified archive (run --archive first)")
        m = json.loads(got.stdout)
        r = subprocess.run([MODAL, "volume", "delete", "-y", vol], capture_output=True, text=True, timeout=300)
        if r.returncode:
            raise SystemExit(f"volume delete {vol} failed: {(r.stderr or r.stdout)[-300:]}")
        R.update(a.run, lambda rec: rec.update(retired={"epoch": round(time.time()), "volume": vol,
                                                        "archive": f"{ARCHIVE_VOLUME}:retired/{vol}",
                                                        "files": m.get("files")}))
        print(f"retired {a.run}: {vol} deleted, archive {ARCHIVE_VOLUME}:retired/{vol} ({m.get('files')} files)")
    else:
        print(json.dumps({"run": a.run, "volume": vol, "would_archive_to": f"{ARCHIVE_VOLUME}:retired/{vol}"}))


if __name__ == "__main__":
    main()
