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


def modal_apps(modal_bin=None, timeout=60):
    """`modal app list --json` rows (keys lower_snake), or None when the listing is UNKNOWN: a non-zero exit, empty
    output, unparsable JSON or an empty list (the workspace always lists recent apps; an empty list is an API failure,
    24 Oct: api.modal.com unreachable read as every app gone). Callers must treat None as unknown, never as gone."""
    import json as _json
    import subprocess as _sp

    modal_bin = modal_bin or str(ROOT_PATH / ".venv/bin/modal")
    try:
        r = _sp.run([modal_bin, "app", "list", "--json"], capture_output=True, text=True, timeout=timeout)
        rows = _json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
    except (_sp.SubprocessError, OSError, ValueError):
        return None
    if not isinstance(rows, list) or not rows:
        return None
    return [{k.lower().replace(" ", "_"): v for k, v in a.items()} for a in rows]
