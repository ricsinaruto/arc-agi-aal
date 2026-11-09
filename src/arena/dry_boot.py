"""CPU dry boot (RL 25 Oct, mtpcheck boot deaths): the bundle's real serving start, with the trial's server_env, on the
pinned runtime tree, up to the vLLM engine start, on a container without a GPU.

ARC_SERVING_DRY_BOOT=1 (bootstrap ARC_DRY_BOOT_ONLY): arena.serving.main -> configure -> recipe.main run unchanged
(runtime extraction, every boot patch, runtime_environment with our LoRA/PLE wrappers, server_command with every
argv wrapper) except two stand-ins: ``gpu_inventory`` (nvidia-smi) returns a skip row, and ``start_server`` is the
fake engine. The fake engine takes the exact command start_server would run, compiles every runtime file the boot
patches wrote (mtime after the extraction), imports each of them in the server's own environment and interpreter, then
prints ``ARC_DRY_BOOT engine start`` and exits 0 (3 on any failure). Two GPU-only lookups inside runtime_environment get
stand-ins too (v6 -db leg, 26 Oct): the real libcuda.so.1 (the runtime's own CUDA stub, copied out of its stubs path)
and nvidia-smi's GPU name for TORCH_CUDA_ARCH_LIST (ARC_DRY_BOOT_GPU, default the RTX PRO 6000). Unlike ARC_SERVING_PATCH_CHECK, which exits after
the patches, this also covers runtime_environment and the argv.
"""

import json
import os
import subprocess
import time
from pathlib import Path

ENV = "ARC_SERVING_DRY_BOOT"
GPU_ENV = "ARC_DRY_BOOT_GPU"
CUDA_STUB = "usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs/libcuda.so"
SITE = "usr/local/lib/python3.12/dist-packages"
MARK = "dry-boot-extracted.json"


def enabled():
    return os.environ.get(ENV) == "1"


def mark(working):
    """Right after the runtime tree is in place, before any boot patch: later mtimes are patch writes."""
    (Path(working) / MARK).write_text(json.dumps({"t": time.time()}) + "\n")


def changed(site, since):
    site = Path(site)
    return sorted(str(p.relative_to(site)) for p in site.rglob("*.py") if p.stat().st_mtime >= since)


def module_of(relative):
    parts = relative[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def entry_first(modules, site):
    """A model package's ``model`` module before its other patched modules, as the server's registry loads them:
    Flash's ``nvidia/qsa.py`` imports ``.model``, which imports ``.qsa``'s class, so qsa alone is a circular import
    (r3off -db 26 Oct: nvidia/model.py unpatched, qsa imported cold)."""
    out = []
    for name in modules:
        parts = name.split(".")
        if len(parts) >= 5 and parts[:2] == ["vllm", "models"] and parts[-1] != "model":
            entry = ".".join(parts[:4] + ["model"])
            if entry not in out and (Path(site) / (entry.replace(".", "/") + ".py")).is_file():
                out.append(entry)
        if name not in out:
            out.append(name)
    return out


IMPORT_PROGRAM = """
import importlib, json, sys, traceback
out = {}
for name in json.loads(sys.argv[1]):
    try:
        importlib.import_module(name)
        out[name] = "ok"
    except BaseException as e:
        out[name] = "".join(traceback.format_exception_only(type(e), e)).strip()[-400:]
print("ARC_DRY_BOOT_IMPORTS " + json.dumps(out))
"""


def fake_engine(command, env, site, working, emit=print, run=None):
    run = run or subprocess.run
    working = Path(working)
    since = json.loads((working / MARK).read_text())["t"] if (working / MARK).is_file() else 0.0
    files = changed(site, since)
    compiled = {}
    for rel in files:
        try:
            compile((Path(site) / rel).read_text(), rel, "exec")
            compiled[rel] = "ok"
        except SyntaxError as e:
            compiled[rel] = f"SyntaxError: {e}"
    modules = entry_first([module_of(r) for r in files if "/" in r or r.endswith(".py")], site)
    done = run([command[0], "-c", IMPORT_PROGRAM, json.dumps(modules)], env=env, capture_output=True, text=True,
               timeout=900)
    line = next((x for x in done.stdout.splitlines() if x.startswith("ARC_DRY_BOOT_IMPORTS ")), None)
    imports = json.loads(line.split(" ", 1)[1]) if line else {"<interpreter>": (done.stderr or "")[-800:]}
    bad = {k: v for d in (compiled, imports) for k, v in d.items() if v != "ok"}
    row = {"command": command, "patched_files": files, "imports": imports, "failed": bad,
           "env_keys": sorted(k for k in env if k.startswith(("ARC_", "ARENA_", "TAAF_", "VLLM_")))}
    (working / "dry-boot.json").write_text(json.dumps(row, indent=2) + "\n")
    if bad:
        emit("ARC_DRY_BOOT FAIL " + json.dumps(bad)[:4000], flush=True)
        raise SystemExit(3)
    emit(f"ARC_DRY_BOOT engine start ({len(files)} patched files compiled and imported) "
         + json.dumps(command), flush=True)
    raise SystemExit(0)


def install(recipe, working):
    """Before configure's own wrappers, so theirs run on top exactly as on the card."""
    original = recipe.gpu_inventory

    def gpu_inventory():
        try:
            return original()
        except Exception as e:  # noqa: BLE001 - no GPU on the dry-boot container
            return {"skipped": "dry boot: " + repr(e)[:120]}

    def start_server(model_dir, env, *, tuning=None, **kwargs):  # kwargs: the older fork's enable_chunked_prefill
        tuning = tuning or recipe.resolve_vllm_tuning(**kwargs)
        command = recipe.server_command(model_dir, tuning=tuning, **kwargs)
        print("VLLM_START_COMMAND", json.dumps(command), flush=True)
        fake_engine(command, env, recipe.RUNTIME_ROOT / SITE, working)

    find_driver = getattr(recipe, "_find_real_cuda_driver", None)

    def find_real_cuda_driver():
        try:
            return find_driver()
        except Exception:  # noqa: BLE001 - no driver on the dry-boot container: the runtime's stub, as a file
            stub = Path(working) / "dry-boot-libcuda" / "libcuda.so.1"
            stub.parent.mkdir(parents=True, exist_ok=True)
            source = Path(recipe.RUNTIME_ROOT) / CUDA_STUB
            stub.write_bytes(source.read_bytes() if source.is_file() else b"")
            return stub

    if getattr(recipe, "_CUDA_ARCH", "set") is None:
        try:
            recipe.cuda_arch()
        except Exception:  # noqa: BLE001 - no nvidia-smi
            recipe._CUDA_ARCH = recipe.SUPPORTED_GPUS[os.environ.get(GPU_ENV, "rtx pro 6000")]
    recipe.gpu_inventory = gpu_inventory
    recipe.start_server = start_server
    if find_driver is not None:
        recipe._find_real_cuda_driver = find_real_cuda_driver
