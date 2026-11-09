"""Golden (rl-min D3, 28 Oct): the boot-cache key inputs of real-v10's play server. arc_infra.boot_cache.key (k2) hashes
the server argv, the flag fingerprint and every *-patch.json / *-start.json in the serving working dir at server start
(name + keyed_bytes). This runs the real arena.serving.configure + serving_setup.main on the pinned vLLM files those
patches edit (data/boot_key_site.tar.xz: the 40 pristine files they read, ple_layer.py after the RadixArk PLE patch) with stubs
for hardware, mounts, extraction and the CUDA environment, stops where main asks for the boot cache, and records each
keyed file's sha and the key over serving_argv.json's real-v10 argv (ARC_BOOT_CACHE_KEY=2). The same inputs were
checked equal on trunk/rl and rl-min-d @be6c09cf6 (full pinned tree)."""
import hashlib
import importlib.util
import json
import os
import shutil
import tarfile

import pytest
from common import DATA, ROOT, B, check, config

RUN = "rl1-loop-seq-real-v10"
SITE_TAR = DATA / "boot_key_site.tar.xz"
# modal_rl1 sets these from the bundle's configs/boot.json (real-v10 seeds launch/rl/data/boot-rd.json)
BOOT_FLAGS = {"expert_mask_skip_read": "ARC_EXPERT_MASK_SKIP_READ", "expert_mask_compact": "ARC_EXPERT_MASK_COMPACT",
              "load_profile": "ARC_LOAD_PROFILE"}
TOKENIZERS = "transformers/tokenization_utils_tokenizers.py"  # token_segment's anchor (its marker is not keyed)


class Stop(Exception):
    pass


def server_environ(cfg, study):
    env = {str(k): str(v) for k, v in (study["trials"][0].get("server_env") or {}).items()}
    for k, v in study["rl_loop"]["mg"]["redesign"]["card_env"]["0"].items():
        env.pop(k, None) if v is None else env.__setitem__(k, str(v))
    boot = next(s["path"] for s in cfg["seeds"] if s["dest"] == "configs/boot.json")
    env.update({flag: "1" for key, flag in BOOT_FLAGS.items() if json.loads((ROOT / boot).read_text()).get(key)})
    return env


def test_boot_key_inputs(tmp_path):
    if not SITE_TAR.is_file():
        pytest.skip(f"{SITE_TAR} missing")
    cfg = config(RUN)
    study = B.inline(cfg["study"])
    server, settings = study["server"], study["cache_experiment"]
    env = server_environ(cfg, study)
    site_root, working, bundle = tmp_path / "rt", tmp_path / "serving", tmp_path / "bundle"
    site = site_root / "usr/local/lib/python3.12/dist-packages"
    working.mkdir()
    bundle.mkdir()
    saved = dict(os.environ)
    try:
        for k in [k for k in os.environ if k.startswith(("ARC_", "ARENA_", "TAAF_"))]:
            del os.environ[k]
        os.environ.update(env)
        from arc_infra import serving as SV

        os.environ.update(SV.SERVER_ENV)
        os.environ.update(  # load_recipe's settings for this study (arena.serving.start)
            TAAF_VLLM_MAX_NUM_BATCHED_TOKENS=str(server["max_num_batched_tokens"]), TAAF_VLLM_KV_CACHE_DTYPE="auto",
            TAAF_VLLM_MTP_TOKENS=str(server["mtp_tokens"]), TAAF_VLLM_MAX_NUM_SEQS=str(server["max_num_seqs"]),
            TAAF_VLLM_KV_CACHE_MEMORY_BYTES=env["ARENA_KV_CACHE_MEMORY_BYTES"], TAAF_VLLM_ENABLE_PREFIX_CACHING="1",
            TAAF_VLLM_GPU_MEMORY_UTILIZATION=env["ARENA_GPU_MEMORY_UTILIZATION"], TAAF_KAGGLE_BUNDLE_DIR=str(bundle),
            TAAF_KAGGLE_WORKING_DIR=str(working), TAAF_KAGGLE_SETUP_ENV=str(working / "serving-environment.json"))
        spec = importlib.util.spec_from_file_location("boot_key_recipe", ROOT / "src/arc_infra/serving_setup.py")
        r = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(r)
        r.RUNTIME_ROOT, r.CACHE_ROOT, r.COMPILE_CACHE_ROOT, r.TMP_ROOT = (site_root, tmp_path / "cache",
                                                                          tmp_path / "ccache", tmp_path / "tmp")

        def extract(runtime_dir, **kwargs):
            shutil.rmtree(site_root, ignore_errors=True)
            site.mkdir(parents=True)
            with tarfile.open(SITE_TAR) as tar:
                tar.extractall(site, filter="data")
            (site / TOKENIZERS).parent.mkdir(parents=True)
            (site / TOKENIZERS).write_text("def _encode_plus(self):\n        encodings = self._tokenizer.encode_batch(\n"
                                           "            1)\n")
            return {"mode": "stub"}

        r.source_identity = r.validate_runtime_storage = lambda: {}
        r.gpu_inventory = lambda: {"stub": True}
        r.resolve_runtime_dir, r.resolve_model_dir = (lambda: tmp_path / "runtime"), (lambda: tmp_path / "model")
        r.verify_and_extract_runtime, r.verify_model = extract, (lambda model_dir: {})
        r.patch_ple_layer = lambda: {"stub": "the archive's ple_layer.py is already PLE-patched"}
        r.runtime_environment = lambda *a, **k: ({**os.environ, "VLLM_PLE_CPU_OFFLOAD": "1"}, {})
        import arc_infra.boot_cache as BC
        from arena import serving as S

        seen = {}

        def restore(roots, cache_key, environ=None, emit=print):  # main's next step is the server start
            seen["key"] = cache_key
            raise Stop

        real_restore, BC.restore_from_environment = BC.restore_from_environment, restore
        try:
            lab = S.lab_overrides()
            mask = ROOT / cfg["files"][server["expert_mask"]["mask"]]
            S.configure(r, working, cpu_gib=settings["cpu_cache_gib"],
                        graph_capture_size=server.get("max_cudagraph_capture_size"),
                        graph_capture_sizes=server.get("cudagraph_capture_sizes"),
                        reclaim_skipped_blocks=settings.get("reclaim_skipped_blocks", False),
                        serving_kernels=server.get("serving_kernels"), served_route=server.get("served_route"),
                        serving_fast=server.get("serving_fast"), expert_mask=(str(mask), server["expert_mask"]["sha256"]),
                        prefix_cache=settings.get("prefix_cache"),
                        prompt_tokens_details=lab.get("prompt_tokens_details", server.get("prompt_tokens_details", False)),
                        prefix_match_unit=server.get("prefix_match_unit"),
                        renderer_num_workers=server.get("renderer_num_workers"),
                        mm_processor_cache_gb=server.get("mm_processor_cache_gb"), lora=server.get("lora"),
                        ready_guard_seconds=lab.get("ready_guard"), setup_timeouts=lab.get("setup_timeouts"))
            r.ANALYZER_CONTEXT = 262144
            with pytest.raises(Stop):
                r.main()
        finally:
            BC.restore_from_environment = real_restore
        assert seen["key"].startswith("k2-")
        files = sorted(working.glob("*-patch.json")) + sorted(working.glob("*-start.json"))
        argv = json.loads((DATA / "serving_argv.json").read_text())[RUN]["argv"]
        got = {"files": {p.name: hashlib.sha256(BC.keyed_bytes(p)).hexdigest() for p in files},
               "key": BC.key(argv, working, environ={**os.environ, "ARC_BOOT_CACHE_KEY": "2"})}
    finally:
        os.environ.clear()
        os.environ.update(saved)
    check("boot_key", got)
