"""Use the local Kaggle auth configuration without printing or packaging credentials."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command_status(arguments, result):
    """Kaggle can print a rejected launch/upload and still exit zero."""
    if result.returncode:
        return result.returncode
    error_prefix = {
        ("kernels", "push"): "Kernel push error:",
        ("datasets", "create"): "Dataset creation error:",
        ("datasets", "version"): "Dataset version creation error:",
    }.get(tuple(arguments[:2]))
    if error_prefix:
        output = (result.stdout + b"\n" + result.stderr).decode(errors="replace")
        if any(line.strip().startswith(error_prefix) for line in output.splitlines()):
            return 1
    return 0


def main():
    settings = {}
    for line in (ROOT / "kaggle/.env").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError("Invalid Kaggle environment setting")
        settings[key.strip()] = value.strip().strip("\"'")
    token_path = Path(settings["FACE_OF_AGI_KAGGLE_TOKEN_FILE"]).expanduser()
    if not token_path.is_absolute():
        token_path = ROOT / token_path
    token = token_path.read_text().strip()
    if not token:
        raise ValueError("Configured Kaggle authentication token is empty")
    with tempfile.TemporaryDirectory(prefix="arc-kaggle-config-") as empty_config:
        env = {
            **os.environ,
            "KAGGLE_API_TOKEN": token,
            "KAGGLE_USERNAME": settings["FACE_OF_AGI_KAGGLE_OWNER"],
            "KAGGLE_CONFIG_DIR": empty_config,
        }
        if sys.argv[1:2] == ["--python"]:
            command = [sys.executable, *sys.argv[2:]]
        else:
            command = [sys.executable, "-c", "from kaggle.cli import main; main()", *sys.argv[1:]]
        result = subprocess.run(command, env=env, capture_output=True)
        for data, stream in ((result.stdout, sys.stdout), (result.stderr, sys.stderr)):
            stream.write(data.decode(errors="replace").replace(token, "[REDACTED]"))
        raise SystemExit(command_status(sys.argv[1:], result))


if __name__ == "__main__":
    main()
