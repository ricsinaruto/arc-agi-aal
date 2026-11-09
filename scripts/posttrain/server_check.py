"""Pre-launch server check (mg5f / mg5g-chk boot failures, 17 Oct): run a bundle's server start path on this machine
with the GPU, sandbox and dataset steps stubbed, through every settings check up to the vLLM launch (arena.serving's
study checks, serving_core.start, load_recipe, the arena cache hook, the fork's resolve_vllm_tuning).

usage: python server_check.py BUNDLE   (prints the resolved tuning; exit 1 with the failing check)"""
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path


class Passed(Exception):
    pass


def main(bundle):
    sys.dont_write_bytecode = True  # bytecode inside the bundle breaks its manifest check at boot (rl1-screen-mg6-1)
    bundle = Path(bundle).resolve()
    sys.path.insert(0, str(bundle / "src"))
    study_path = bundle / "configs/comparison.json"
    study = json.loads(study_path.read_text())
    os.environ.update({k: str(v) for k, v in (study["trials"][0].get("server_env") or {}).items()})
    tmp = Path(tempfile.mkdtemp(prefix="server-check-"))

    import arc_infra.serving as infra
    from arena import serving as arena_serving
    from arena.serving_core import serving as core

    def stage(input_root, staging):  # the staged fork: its serving script and an identity naming it
        staging = Path(staging)
        staging.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(core.FORK_SETUP if hasattr(core, "FORK_SETUP") else bundle / "src/arc_infra/serving_setup.py",
                        staging / "serving_setup.py")
        return staging

    real_load = infra.load_recipe

    def load(working, input_root, **kw):
        import traceback

        infra.dataset_mount = lambda ref, root: tmp
        # load_recipe's settings checks, then its import of the fork; the stub bundle has no source identity, so only
        # a failure inside source_identity (after every settings check) is expected
        try:
            real_load(working, input_root, **kw)
        except Exception as exc:
            if "source_identity" not in [f.name for f in traceback.extract_tb(exc.__traceback__)]:
                raise
        import importlib.util

        spec = importlib.util.spec_from_file_location("arc_check_serving", Path(kw["bundle_dir"]) / "serving_setup.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def main_stub():
            print(json.dumps(module.resolve_vllm_tuning(), default=str))
            raise Passed()

        module.main = main_stub
        module.stop_owned_server = lambda: None
        return module

    core.verify_hardware = lambda: None
    core.needed = lambda: False
    core.worker_probe = lambda working, jail: types.SimpleNamespace(returncode=0, stdout='{"type": "ready"}\n', stderr="")
    core.stage_fork_bundle = stage
    core.load_recipe = load
    sys.argv = ["serving", "--study", str(study_path), "--working", str(tmp / "serving"), "--input-root", str(tmp)]
    try:
        arena_serving.main()
    except Passed:
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    raise SystemExit("server check: the start path returned without reaching the vLLM launch")


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except Passed:
        sys.exit(0)
