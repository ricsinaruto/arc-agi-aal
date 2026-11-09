import json

from arc_infra.boot_cache import key, keyed_bytes


def _overlays(folder, cleared):
    (folder / "serving-kernels-patch.json").write_text(json.dumps({"files": {"a.py": "1"}}, indent=2))
    (folder / "serving-kernels-start.json").write_text(json.dumps(
        {"identity": {"files": {"a.py": "1"}}, "cleared_compile_cache": cleared}, indent=2))
    (folder / "expert-mask-start.json").write_text(json.dumps(
        {"identity": {"mask_sha256": "m"}, "cleared_compile_cache": cleared}, indent=2))


def test_cleared_compile_cache_does_not_move_the_key(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    _overlays(a, [])
    _overlays(b, ["/tmp/qwen38-flash-next-vllm-cache/vllm/torch_compile_cache"])
    argv = ["vllm", "serve", "/model", "--port", "1234"]
    assert key(argv, a) == key(argv, b)
    # a file with the field already empty keys byte-for-byte as before (existing archives keep their names)
    assert keyed_bytes(a / "serving-kernels-start.json") == (a / "serving-kernels-start.json").read_bytes()
    assert keyed_bytes(b / "serving-kernels-start.json") == (a / "serving-kernels-start.json").read_bytes()


def test_identity_change_still_moves_the_key(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    _overlays(a, [])
    _overlays(b, [])
    (b / "serving-kernels-start.json").write_text(json.dumps(
        {"identity": {"files": {"a.py": "2"}}, "cleared_compile_cache": []}, indent=2))
    assert key(["vllm"], a) != key(["vllm"], b)
