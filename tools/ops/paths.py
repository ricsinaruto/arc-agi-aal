"""Shared locations for the ops tools. Override with the environment variables named below; unset ones are read from the
KEY=VALUE file runs/ops/ops.env (a link to the scratch folder's ops.env), so a restarted loop keeps its scratch and Nebius config."""
import os, pathlib
ROOT_PATH = pathlib.Path(__file__).resolve().parents[2]
for _f in (ROOT_PATH / "runs/ops/ops.env", pathlib.Path(os.environ.get("ARC_OPS_SCRATCH", "/nonexistent")) / "ops.env"):
    if _f.is_file():
        for _x in _f.read_text().splitlines():
            _k, _, _v = _x.strip().removeprefix("export ").partition("=")
            if _k.startswith("ARC_") and _v: os.environ.setdefault(_k, _v.strip().strip('"'))
SCRATCH = pathlib.Path(os.environ.get("ARC_OPS_SCRATCH", ROOT_PATH / "runs/ops"))  # machine-local data: live feeds, launch logs, pulled results
NEBIUS_SCRIPTS = pathlib.Path(os.environ.get("ARC_NEBIUS_SCRIPTS", ROOT_PATH / ".claude/worktrees/nebius/scripts"))  # launcher lives on branch nebius
NEBIUS_CONFIG = pathlib.Path(os.environ.get("ARC_NEBIUS_CONFIG", SCRATCH / "nebius.json"))  # see nebius-config.example.json


def read_json(path, tries=10):
    """json.load that rides out another writer's truncate-then-write of the ledger or queue (a pass crashed on an empty
    spend.json, 23 Oct): retry for up to ~5 s, then fall back to the last copy that parsed (a hand edit left the queue
    invalid for minutes and every pass crashed, 23 Oct); raise only when there is none."""
    import json, sys, time
    p = pathlib.Path(path); good = SCRATCH / "lastgood" / p.name
    for i in range(tries):
        try:
            t = p.read_text(); d = json.loads(t)
            try: good.parent.mkdir(exist_ok=True); good.write_text(t)
            except OSError: pass
            return d
        except (json.JSONDecodeError, FileNotFoundError):
            if i == tries - 1:
                if not good.is_file(): raise
                print(f"WARNING {p.name} unreadable, using last good copy from {time.strftime('%H:%M', time.gmtime(good.stat().st_mtime))} UTC", file=sys.stderr)
                return json.loads(good.read_text())
            time.sleep(0.5)
