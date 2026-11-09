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
