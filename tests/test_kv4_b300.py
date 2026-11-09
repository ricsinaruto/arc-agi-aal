"""E1 (RL redesign, 26 Oct): the competition's int4 KV cache on the B300 (SM100/SM103) RL play card. CPU only, no Triton
needed (the kernels load with a stand-in ``triton`` when it is missing; the kernel checks are tests/test_kv4_kernels.py,
run under the interpreter by tests/modal_kv4_triton.py): the device guard, the capture guard of the in-server sample,
the B300 run config, the serving flags and the dry boot's SM100 stand-in."""
import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from arc_infra import serving_kernels_patch as patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "launch/rl/runs/rl1-loop-seq-real-v8.json"  # the main run's int4 KV serving kernels


def stub_triton():
    triton = types.ModuleType("triton")
    language = types.ModuleType("triton.language")
    language.constexpr = object
    triton.jit = lambda fn: fn
    triton.language = language
    return {"triton": triton, "triton.language": language}


@pytest.fixture
def kv4(monkeypatch):
    if importlib.util.find_spec("triton") is None:
        for name, module in stub_triton().items():
            monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("arc_kv4_b300_under_test", ROOT / "src/arc_infra/kv4_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(module._STATE, "config", {"kv_cache_4bit": "int4"})
    return module


# ----------------------------------------------------------------------------- device guard
def test_device_guard_serves_sm100_sm103_and_sm120_with_hadacore_only(kv4, capsys):
    for cap in ((10, 0), (10, 3), (12, 0)):
        row = kv4.check_device(cap, True, "gpu")
        assert row["capability"] == f"{cap[0]}.{cap[1]}" and row["mode"] == "int4"
    assert kv4.check_device((10, 3), True, "NVIDIA B300")["arch"].startswith("SM100")
    assert "ARC_KV4_DEVICE " in capsys.readouterr().out
    for cap in ((9, 0), (8, 9), (11, 0)):
        with pytest.raises(NotImplementedError):
            kv4.check_device(cap, True)
    with pytest.raises(RuntimeError, match="Hadacore"):
        kv4.check_device((10, 3), False, "NVIDIA B300")


def test_warm_rht_checks_the_device_once(kv4, monkeypatch):
    calls = []
    ops = types.ModuleType("vllm.v1.attention.ops.int4_per_token_head")
    ops._hadacore_available = lambda: True
    ops._get_rht_signs = lambda d, r, device: calls.append(("signs", d, r))
    for name in ("vllm", "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.ops"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    sys.modules["vllm.v1.attention.ops"].int4_per_token_head = ops
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 3))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA B300 SXM6 AC")
    monkeypatch.setattr(kv4, "check_device", lambda *a: calls.append(("check", *a)) or {"capability": "10.3"})
    monkeypatch.setitem(kv4._STATE, "device", None)
    kv4._warm_rht()
    kv4._warm_rht()
    assert calls == [("check", (10, 3), True, "NVIDIA B300 SXM6 AC"), ("signs", 256, 0), ("signs", 256, 0)]


def test_sample_is_skipped_inside_a_cuda_graph_capture(kv4, monkeypatch):
    """The B300 card captures graphs up to 1,024 tokens: a >= 512-row step may be a capture, where the sample's host
    copies would invalidate it. Outside a capture the sample runs as before."""
    sampled, written = [], []
    monkeypatch.setattr(kv4, "_sample", lambda impl, k, v: sampled.append(k.shape[0]))
    monkeypatch.setattr(kv4, "cache_views", lambda cache, mode, holder=None: ())
    monkeypatch.setattr(kv4, "write", lambda k, v, *a: written.append(k.shape[0]))
    rows = torch.zeros((kv4.SAMPLE_MIN_ROWS, 2, 256))
    slots = torch.arange(kv4.SAMPLE_MIN_ROWS)
    monkeypatch.setattr(kv4, "_capturing", lambda tensor: True)
    kv4.cache_update(None, rows, rows, None, slots)
    assert sampled == [] and written == [kv4.SAMPLE_MIN_ROWS]
    monkeypatch.setattr(kv4, "_capturing", lambda tensor: False)
    kv4.cache_update(None, rows, rows, None, slots)
    assert sampled == [kv4.SAMPLE_MIN_ROWS]


def test_capturing_reads_the_current_stream_only_for_cuda_tensors(kv4, monkeypatch):
    assert kv4._capturing(torch.zeros(1)) is False  # CPU tensors never ask CUDA
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    fake = types.SimpleNamespace(is_cuda=True)
    assert kv4._capturing(fake) is True


def test_compare_names_the_rotation_or_the_writer(kv4):
    page = torch.zeros((1, 4, 2, 264), dtype=torch.uint8).transpose(1, 2)
    base = {"mode": "int4", "device": "a", "rotated_k": torch.zeros(4, 2, 256, dtype=torch.bfloat16),
            "rotated_v": torch.zeros(4, 2, 256, dtype=torch.bfloat16), "page": page, "attn_page": page.clone(),
            "attn_out": torch.ones(3), "attn_rel_error_vs_dequant": 0.004}
    other = {**base, "device": "b", "page": page.clone(), "rotated_v": base["rotated_v"].clone()}
    assert kv4.compare(base, other)["bitwise"]
    other["page"].transpose(1, 2)[0, 1, 0, 260] = 7  # a V scale byte
    other["rotated_v"][0, 0, 0] = 1.0
    r = kv4.compare(base, other)
    assert not r["bitwise"] and r["v_scale_equal"] < 1 and r["k_scale_equal"] == 1 and r["rotated_v_equal"] < 1


# ----------------------------------------------------------------------------- run config and serving flags
def test_kv4_with_a_draft_needs_the_bf16_state():
    spec = patch.validate({"kv_cache_4bit": "int4"})
    with pytest.raises(ValueError, match="bfloat16"):
        patch.server_flags(spec, 1)
    sub = patch.validate({"kv_cache_4bit": "int4", "mamba_ssm_cache_dtype": "bfloat16"})  # the submission's (MTP-1)
    assert patch.server_flags(sub, 1)[-4:] == ["--kv-cache-dtype", "int4_per_token_head", "--block-size", "3136"]


def test_retired_options_only_at_their_off_value():
    base = {"kv_cache_4bit": "int4", "mamba_ssm_cache_dtype": "bfloat16"}
    spec = patch.validate({**base, "moe_backend": None, "hc_fusion": False, "indexer_fp8": False})
    assert spec == patch.validate(base) and spec["moe_backend"] is None and spec["hc_fusion"] is False
    assert "indexer_fp8" not in spec
    for key, value in (("moe_backend", "humming"), ("hc_fusion", True), ("indexer_fp8", True)):
        with pytest.raises(ValueError, match="removed"):
            patch.validate({**base, key: value})
    with pytest.raises(ValueError):
        patch.validate({**base, "kv_cache_8bit": "int8"})


def test_fork_serves_b300_as_sm100():
    tree = ast.parse((ROOT / "src/arc_infra/serving_setup.py").read_text())
    gpus = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == "SUPPORTED_GPUS")
    assert gpus["b300"] == gpus["b200"] == "10.0" and gpus["rtx pro 6000"] == "12.0"


PINNED_SITE = Path(__import__("os").environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))


def test_kv4_qsa_patch_on_the_pinned_file_when_available():
    """The same hash-guarded qsa.py edits as the submission (the file and hash are QSA's on both cards)."""
    path = PINNED_SITE / patch.QSA["relative"]
    if not path.is_file():
        pytest.skip("Optional pinned runtime (ARC_PINNED_SITE)")
    patched = patch.patched_source(patch.QSA, path.read_bytes()).decode()
    tree = ast.parse(patched)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    impl = {n.name: n for n in classes["Qwen3_8FlashNextQSAFlashAttentionImpl"].body if isinstance(n, ast.FunctionDef)}
    forward = ast.unparse(impl["forward_qsa"])
    assert forward.index("output.zero_()") < forward.index("_arc_kv4_attention(") < forward.index(
        "qsa_sparse_paged_attention(")
    assert "do_kv_cache_update" in impl and "arc_kv8" not in patched


def test_runtime_helper_entry_points_match_the_qsa_patch():
    tree = ast.parse((ROOT / "src/arc_infra/kv4_runtime.py").read_text())
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    imported = set()
    for _, replacement, _ in patch.QSA["edits"]:
        for line in replacement.splitlines():
            if "from vllm.arc_kv4 import" in line:
                imported.add(line.split("import ")[1].split(" as ")[0])
    assert imported == {"customize_spec", "kv_cache_shape", "impl_init_args", "cache_update", "attention", "allowed"}
    assert imported <= names


# ----------------------------------------------------------------------------- dry boot on the SM100 stand-in
def test_dry_boot_b300_stand_in_boots_the_int4_command(tmp_path, monkeypatch, capsys):
    from test_dry_boot import base_patch, fake_recipe

    from arena import dry_boot
    from arena import serving as S

    monkeypatch.setenv(dry_boot.ENV, "1")
    monkeypatch.setenv(dry_boot.GPU_ENV, "b300")
    monkeypatch.setattr(S, "apply", base_patch)
    applied = []

    def apply_kernels(site, working, spec):  # the qsa.py patch on the pinned file: tests/test_kv4_kernels' site check
        (Path(site) / "vllm/arc_kv4.py").write_text("MODE = 'int4'\n")
        applied.append(spec)
        return {"server_flags": patch.server_flags(spec)}

    monkeypatch.setattr(S, "apply_serving_kernels", apply_kernels)
    monkeypatch.setattr(S, "apply_serving_kernels_ple", lambda site, working, spec: {})  # hero-v7's ple_uva
    monkeypatch.setattr(S, "serving_kernel_environment", lambda spec, env: env)
    recipe = fake_recipe(tmp_path / "rt", [])
    recipe.SUPPORTED_GPUS = {"rtx pro 6000": "12.0", "b200": "10.0", "b300": "10.0"}
    working = tmp_path / "w"
    working.mkdir()
    S.configure(recipe, working, serving_kernels=json.loads(CONFIG.read_text())["study"]["server"]["serving_kernels"])
    with pytest.raises(SystemExit) as e:
        recipe.main()
    out = capsys.readouterr().out
    assert e.value.code == 0, out
    assert recipe._CUDA_ARCH == "10.0" and applied[0]["kv_cache_4bit"] == "int4"
    command = json.loads(out.split("VLLM_START_COMMAND ", 1)[1].splitlines()[0])
    i = command.index("--kv-cache-dtype")
    assert command[i:i + 4] == ["--kv-cache-dtype", "int4_per_token_head", "--block-size", "3136"]
    row = json.loads((working / "dry-boot.json").read_text())
    assert "vllm/arc_kv4.py" in row["patched_files"] and row["imports"]["vllm.arc_kv4"] == "ok"
