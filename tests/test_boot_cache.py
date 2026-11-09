"""arc_infra.boot_cache: the archive key (v1 as before; v2 stable across ports / memory shares / run paths, split by
graph-relevant flags and overlays), the carry of AOT + autotune caches, the delayed re-save (stage H, 26 Oct)."""
import json
import tarfile

from arc_infra import boot_cache as B

ARGV = ["python", "-m", "vllm.entrypoints.cli.main", "serve", "/m", "--port", "1234", "--host", "127.0.0.1",
        "--kv-cache-memory-bytes", "151397597184", "--max-num-seqs", "152", "--served-model-name", "Q"]
ENV = {"ARC_PINNED_LOAD": "1", "ARC_MARLIN_BATCH": "1", "ARC_RL1_TRIAL": "rl-loop", "ARC_BOOT_CACHE_IN": "/runs/a",
       "ARC_VALIDATION_OUTPUT": "/runs/validation/x"}


def overlays(tmp_path, patched="aa"):
    d = tmp_path / "serving"
    d.mkdir(exist_ok=True)
    (d / "lora-graph-patch.json").write_text(json.dumps({"patched_sha256": patched}))
    (d / "serving-kernels-start.json").write_text(json.dumps({"settings": {"kv_cache_4bit": "int4"}}))
    (d / "pinned-load.json").write_text("{}")  # not an overlay of compiled code: never in the key
    return d


def with_port(argv, port):
    out = list(argv)
    out[out.index("--port") + 1] = str(port)
    return out


def test_v1_key_is_unchanged_by_default(tmp_path):
    d = overlays(tmp_path)
    k = B.key(ARGV, d, environ=ENV)
    assert len(k) == 16 and not k.startswith("k2-")
    assert B.key(with_port(ARGV, 1236), d, environ=ENV) != k  # v1: every argv token counts
    assert B.key(ARGV, d, environ={**ENV, "ARC_MOE_LORA_BM16": "1"}) == k  # v1 misses flags


def test_v2_key_ignores_placement_and_run_paths(tmp_path):
    d = overlays(tmp_path)
    env = {**ENV, B.KEY_ENV: "2"}
    k = B.key(ARGV, d, environ=env)
    assert k.startswith("k2-")
    assert B.key(with_port(ARGV, 1237), d, environ=env) == k  # a server per card, one archive
    kv = list(ARGV)
    kv[kv.index("--kv-cache-memory-bytes") + 1] = "120000000000"
    assert B.key(kv, d, environ=env) == k
    assert B.key(ARGV + ["--gpu-memory-utilization=0.5"], d, environ=env) == k
    assert B.key(ARGV, d, environ={**env, "ARC_RL1_TRIAL": "other", "ARC_VALIDATION_OUTPUT": "/y",
                                   "ARC_DRAFT_LOAD_FILTER": "1", "ARC_BOOT_CACHE_SAVE_AFTER": "600"}) == k
    (d / "pinned-load.json").write_text('{"x": 1}')
    assert B.key(ARGV, d, environ=env) == k


def test_v2_key_splits_on_graph_inputs(tmp_path):
    d = overlays(tmp_path)
    env = {**ENV, B.KEY_ENV: "2"}
    k = B.key(ARGV, d, environ=env)
    assert B.key(ARGV, d, environ={**env, "ARC_MOE_LORA_BM16": "1"}) != k  # a flag read inside traced code
    seqs = list(ARGV)
    seqs[seqs.index("--max-num-seqs") + 1] = "120"
    assert B.key(seqs, d, environ=env) != k
    assert B.key(ARGV, overlays(tmp_path, patched="bb"), environ=env) != k  # another patched file


def test_stable_argv_drops_flag_and_value_forms():
    assert B.stable_argv(["a", "--port", "1", "--host=h", "--x", "2"]) == ["a", "--x", "2"]


def make_archive(path, files):
    with tarfile.open(path, "w") as tar:
        for name, data in files.items():
            src = path.parent / "src.bin"
            src.write_bytes(data)
            tar.add(src, arcname=name.lstrip("/"))


def test_carry_takes_aot_and_autotune_of_the_env_root(tmp_path):
    cache, compile_ = tmp_path / "cache", tmp_path / "compile"
    fp = B.env_fingerprint({"ARC_PINNED_LOAD": "1"})
    env_root = cache / "vllm-env" / fp
    files = {
        f"{env_root}/torch_compile_cache/torch_aot_compile/h1/rank_0_0/model": b"graph",
        f"{env_root}/flashinfer_autotune_cache/0.6.17/103a/h1/autotune_configs.json": b"{}",
        f"{compile_}/triton/k/kernel.cubin": b"cubin",
        f"{compile_}/flashinfer/jit/x.so": b"so",
        f"{cache}/vllm/flashinfer_autotune_cache/0.6.17/103a/old/autotune_configs.json": b"{}",  # not under vllm-env
        f"{cache}/other/stale.bin": b"no",
    }
    make_archive(tmp_path / "boot-cache-k2-x.tar", files)
    rows = []
    row = B.carry(tmp_path, [cache, compile_], emit=lambda s, flush=False: rows.append(s), prefer=env_root)
    assert row["found"] and row["autotune"] == 1 and row["preferred"]
    assert (env_root / "flashinfer_autotune_cache/0.6.17/103a/h1/autotune_configs.json").is_file()
    assert (env_root / "torch_compile_cache/torch_aot_compile/h1/rank_0_0/model").read_bytes() == b"graph"
    assert (compile_ / "triton/k/kernel.cubin").is_file() and (compile_ / "flashinfer/jit/x.so").is_file()
    assert not (cache / "other/stale.bin").exists()
    assert not (cache / "vllm/flashinfer_autotune_cache").exists()


def test_save_twice_resaves_only_when_files_were_added(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a").write_text("1")
    slept = []

    def sleep(s):
        slept.append(s)
        (root / "b").write_text("2")  # a lazily compiled kernel during play

    first, second = B.save_twice(tmp_path / "out", [root], "k", "600", sleep=sleep, emit=lambda *a, **k: None)
    assert slept == [600.0] and first["files"] == 1 and second["files"] == 2
    with tarfile.open(tmp_path / "out" / "boot-cache-k.tar") as tar:
        assert len(tar.getmembers()) == 2
    first, second = B.save_twice(tmp_path / "out", [root], "k", "600", sleep=lambda s: None, emit=lambda *a, **k: None)
    assert second is None
    first, second = B.save_twice(tmp_path / "out", [root], "k", None, emit=lambda *a, **k: None)
    assert second is None
