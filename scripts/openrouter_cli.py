"""Run a command with local OpenRouter auth; redact credentials from child output."""

import os
import subprocess
import sys
import threading
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    common = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"], text=True
    ).strip()
    token_path = Path(common).parent / ".openrouter/access_token"
    token = token_path.read_text().strip()
    if not token or len(token.split()) != 1:
        raise ValueError("OpenRouter token file must contain one nonempty token")
    arguments = sys.argv[1:]
    if arguments[:1] == ["--python"]:
        arguments = [sys.executable, *arguments[1:]]
    elif arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        raise SystemExit("Usage: openrouter_cli.py [--python] COMMAND [ARGS...]")
    process = subprocess.Popen(arguments, env={**os.environ, "OPENROUTER_API_KEY": token},
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")

    def forward(source, destination):
        for line in source:
            destination.write(line.replace(token, "[REDACTED]"))
            destination.flush()

    threads = [threading.Thread(target=forward, args=pair) for pair in
               ((process.stdout, sys.stdout), (process.stderr, sys.stderr))]
    for thread in threads:
        thread.start()
    code = process.wait()
    for thread in threads:
        thread.join()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
