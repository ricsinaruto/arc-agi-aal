"""Trusted bootstrap: install restrictions, then run exactly one tool operation."""

import base64
import importlib.util
import json
import os
import sys
from pathlib import Path


def main():
    spec = importlib.util.spec_from_file_location("sandbox", Path(__file__).with_name("sandbox.py"))
    sandbox = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sandbox)
    cfg = json.loads(sys.stdin.readline())
    os.chdir(cfg["workspace"])
    sandbox.restrict(
        read=[*sandbox.runtime_paths(), cfg["evidence"], cfg["api"]], write=[cfg["workspace"]]
    )
    operation = cfg["operation"]
    if operation == "bash":
        # The command's own stderr belongs in the model-visible output. Anything this
        # interpreter printed before now (host sitecustomize errors, locale warnings)
        # stayed on the separate stderr pipe the parent discards.
        sys.stderr.flush()
        os.dup2(sys.stdout.fileno(), sys.stderr.fileno())
        os.execve(
            "/bin/bash",
            ["bash", "--noprofile", "--norc", "-c", cfg["command"]],
            {
                "PATH": str(Path(sys.executable).parent) + ":/usr/local/bin:/usr/bin:/bin",
                "PYTHONPATH": cfg["api"],
                "PYTHONNOUSERSITE": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "ARC_RPC_FD": os.environ["ARC_RPC_FD"],
                "HOME": cfg["workspace"],
                "TMPDIR": cfg["workspace"],
                "LANG": "C.UTF-8",
            },
        )
    path = Path(cfg["path"])
    if operation == "read":
        print(base64.b64encode(path.read_bytes()).decode())
    elif operation == "write":
        path.write_text(cfg["content"])
    elif operation == "mkdir":
        path.mkdir(parents=True, exist_ok=True)
    elif operation == "access":
        with path.open("rb"):
            pass
    else:
        raise ValueError("Unknown tool operation")


if __name__ == "__main__":
    main()
