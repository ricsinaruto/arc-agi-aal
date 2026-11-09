"""The mmrtx serving image: vLLM's MiniMax-M3.1-Flash build (docker tag minimax-m31-flash, pinned digest) + mmrtx_hooks."""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
DIGEST = "sha256:f5ec69891fb7b855762f40598eea1aa5cad6e4f2f0c2a2f81d270e2bdcd93632"  # amd64, pushed 2026-10-23
IMAGE = (modal.Image.from_registry(f"vllm/vllm-openai@{DIGEST}",
                                   setup_dockerfile_commands=["RUN ln -sf $(command -v python3) /usr/local/bin/python || true"])
         .entrypoint([]))
if modal.is_local():
    IMAGE = (IMAGE.add_local_dir(str(HERE / "hooks"), "/opt/hooks", copy=True)
             .run_commands("python3 -m pip install --no-deps --no-build-isolation /opt/hooks")
             .add_local_file(__file__, "/root/image.py"))
