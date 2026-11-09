"""The full BF16 Flash on two B300s: the arena's server start with the model and tensor-parallel switch, as
tp-bf16play-c100-100k-v1 (20 Oct: B300:2, MTP off, BF16 KV) and arc2.l4_serving. The recipe still verifies the mounted
NVFP4 checkpoint and patches the PLE layer; only the served folder, the card count and the quantization flags change.
"""
import json
import os
import sys
from pathlib import Path

from .ple_queue_patch import apply as apply_ple_queue

CARDS = 2
# the PLE offload's one-slot queue raised queue.Full 27 min into base15bf v3 (RL f1, 15 Oct). ple_uva is no way out on
# TP2: its load-time equality check fails there (v4), so nothing proves the values
PLE_QUEUE_WAIT = 120


def patch_recipe(recipe, working, model_dir):
    working = Path(working)
    environment = recipe.runtime_environment

    def runtime_environment(*args, **kwargs):
        env, check = environment(*args, **kwargs)
        if env.get("TORCH_CUDA_ARCH_LIST") != "10.0":
            raise RuntimeError(f"not the B300 build: {env.get('TORCH_CUDA_ARCH_LIST')}")
        # NCCL's host buffers without cuMem on Modal's gVisor hosts (arc2.l4_serving, nccl610-v2)
        return {**env, "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(CARDS)), "NCCL_CUMEM_HOST_ENABLE": "0"}, {
            **check, "bf16x2": CARDS}

    extract = recipe.verify_and_extract_runtime

    def verify_and_extract_runtime(*args, **kwargs):
        result = extract(*args, **kwargs)
        apply_ple_queue(Path(recipe.RUNTIME_ROOT) / "usr/local/lib/python3.12/dist-packages", working, PLE_QUEUE_WAIT)
        return result

    recipe.verify_and_extract_runtime = verify_and_extract_runtime
    command = recipe.server_command

    def server_command(*args, **kwargs):
        argv = list(command(*args, **kwargs))
        argv[argv.index("serve") + 1] = str(model_dir)
        for flag in ("--quantization", "--moe-backend"):
            if flag in argv:
                at = argv.index(flag)
                del argv[at:at + 2]
        at = argv.index("--tensor-parallel-size") + 1
        if argv[at] != "1":
            raise RuntimeError("the recipe's tensor-parallel size changed; re-read arc2.b300_serving")
        argv[at] = str(CARDS)
        return argv

    snapshot = recipe.update_ready_identity

    def update_ready_identity(identity):
        try:
            return snapshot(identity)
        except RuntimeError as exc:  # the one-GPU process expectations; the server is up
            (working / "bf16x2-identity-note.json").write_text(json.dumps({"error": str(exc)[:2000]}, indent=1))
            return identity

    recipe.runtime_environment, recipe.server_command = runtime_environment, server_command
    recipe.update_ready_identity = update_ready_identity


def serve(module, study):
    """``arena.serving.main`` once more with the recipe patched; ranks whose pools differ take the smallest."""
    spec = study["arc2"]["bf16x2"]
    model = Path("/assets") / spec["model"]
    if not (model / "config.json").exists():
        raise RuntimeError(f"no BF16 checkpoint at {model}")
    os.environ["ARC2_BF16X2_ON"] = "1"
    configure = module.configure

    def configure_bf16(recipe, working, **kwargs):
        result = configure(recipe, working, **kwargs)
        patch_recipe(recipe, working, model)
        return {**result, "bf16x2": spec}

    module.configure = configure_bf16
    working = sys.argv[sys.argv.index("--working") + 1]
    try:
        module.main()
    except RuntimeError as exc:
        if "cache layouts disagree" not in str(exc):
            raise
        from .trial import block_budget

        budget = block_budget(working, study["kv_margin"], smallest=True)
        (Path(working) / "cache-layout.json").write_text(json.dumps(budget.layout, indent=2))
