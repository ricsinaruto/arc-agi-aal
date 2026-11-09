"""Hourly backup of the machine-local run records (docs/runs + docs/budget.json) to the Nebius bucket, keep 48.

  records_backup.py backup                       tar.gz -> <remote>/records-backup/records-<UTC>.tar.gz, prune to 48
  records_backup.py list                         the backups, newest last
  records_backup.py restore [NAME] --to DIR      extract a backup (default the newest) into DIR (DIR/docs/runs, ...)
  records_backup.py restore [NAME] --live        into the main checkout: docs/runs is first moved to
                                                 docs/runs.pre-restore-<UTC>, then the backup is extracted

Remote: ARC_RECORDS_BACKUP_REMOTE (rclone path), else <rclone_remote>:<bucket> from the Nebius config.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

KEEP = 48
PREFIX = "records-backup"


def remote():
    r = os.environ.get("ARC_RECORDS_BACKUP_REMOTE")
    if r:
        return r.rstrip("/")
    cfg = json.loads(Path(os.environ.get("ARC_NEBIUS_CONFIG") or paths.SCRATCH / "nebius.json").read_text())
    return f"{cfg['rclone_remote']}:{cfg['bucket']}/{PREFIX}"


def rclone(*args):
    return subprocess.run(["rclone", *args], capture_output=True, text=True, check=True, timeout=600).stdout


def tar_bytes(root):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for p in sorted((root / "docs/runs").glob("*.json")):
            t.add(p, arcname=f"docs/runs/{p.name}")
        t.add(root / "docs/budget.json", arcname="docs/budget.json")
    return buf.getvalue()


def listing(rem):
    out = rclone("lsf", rem + "/")
    return sorted(x for x in out.split() if x.startswith("records-") and x.endswith(".tar.gz"))


def backup(root, rem, keep=KEEP):
    name = time.strftime("records-%Y%m%dT%H%M%SZ.tar.gz", time.gmtime())
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / name
        f.write_bytes(tar_bytes(root))
        rclone("copyto", str(f), f"{rem}/{name}")
        size = f.stat().st_size
    names = listing(rem)
    if name not in names:
        raise RuntimeError(f"backup {name} not listed after upload")
    for old in names[:-keep]:
        rclone("deletefile", f"{rem}/{old}")
    return name, size, min(len(names), keep)


def restore(rem, name, dest):
    name = name or listing(rem)[-1]
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / name
        rclone("copyto", f"{rem}/{name}", str(f))
        with tarfile.open(f) as t:
            members = [m for m in t.getmembers() if m.isfile() and (m.name == "docs/budget.json" or (
                m.name.startswith("docs/runs/") and "/" not in m.name[10:] and m.name.endswith(".json")))]
            t.extractall(dest, members=members, filter="data")
    return name, len(members)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("backup")
    sub.add_parser("list")
    r = sub.add_parser("restore")
    r.add_argument("name", nargs="?")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--to")
    g.add_argument("--live", action="store_true")
    a = ap.parse_args(argv)
    root, rem = paths.ROOT_PATH, remote()
    if a.cmd == "backup":
        name, size, n = backup(root, rem)
        print(f"backup {name} ({size / 1e6:.1f} MB) -> {rem}; {n} kept")
    elif a.cmd == "list":
        print("\n".join(listing(rem)))
    else:
        if a.live:
            runs = root / "docs/runs"
            if runs.exists():
                runs.rename(root / time.strftime("docs/runs.pre-restore-%Y%m%dT%H%M%SZ", time.gmtime()))
            dest = root
        else:
            dest = Path(a.to)
        name, n = restore(rem, a.name, dest)
        print(f"restored {name}: {n} files into {dest}")


if __name__ == "__main__":
    main()
