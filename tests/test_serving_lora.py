"""arena.serving LoRA flags: MTP is refused beside an adapter unless lora.mtp opts in."""
import json

import pytest

from arena.serving import lora_flags, validate_lora


def test_lora_mtp_opt_in():
    spec = ["python", "--speculative-config", json.dumps({"method": "mtp"})]
    with pytest.raises(ValueError, match="without MTP"):
        lora_flags(spec, validate_lora({}))
    kept = lora_flags(spec, validate_lora({"mtp": True}))  # opt-in: the draft stays beside the adapter
    assert "--speculative-config" in kept and "--enable-lora" in kept
    with pytest.raises(ValueError, match="boolean"):
        validate_lora({"mtp": 1})


def test_nvfp4_draft_moe_turns_off_lora_before_backend_selection():
    """LoRA + MTP with the NvFp4 draft: the draft layer's config drops is_lora_enabled before b12x is chosen."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src/arc_infra/serving_kernels_runtime.py").read_text()
    body = src[src.index("class ArcNvfp4DraftMoEMethod"):src.index("def process_weights_after_loading")]
    assert body.index("self.moe.is_lora_enabled = False") < body.index("select_nvfp4_moe_backend(")


def test_draft_backend_by_arch():
    """E2 (26 Oct): the NvFp4 draft experts take b12x on SM120 and vLLM's own NvFp4 kernels on SM100."""
    import ast
    from pathlib import Path

    import pytest

    src = (Path(__file__).resolve().parents[1] / "src/arc_infra/serving_kernels_runtime.py").read_text()
    funcs = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
             and n.name in ("draft_moe_backend", "draft_backend_ok", "draft_check_skip")]  # the runtime imports torch at the top
    space = {}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "runtime", "exec"), space)
    draft_moe_backend, draft_backend_ok = space["draft_moe_backend"], space["draft_backend_ok"]
    draft_check_skip = space["draft_check_skip"]

    assert draft_moe_backend(12) == "flashinfer_b12x" and draft_moe_backend(10) == "auto"
    with pytest.raises(RuntimeError):
        draft_moe_backend(9)
    assert draft_backend_ok("flashinfer_b12x", "FLASHINFER_B12X") and not draft_backend_ok("flashinfer_b12x", "MARLIN")
    assert draft_backend_ok("auto", "FLASHINFER_TRTLLM") and draft_backend_ok("auto", "VLLM_CUTLASS")
    assert not draft_backend_ok("auto", "EMULATION") and not draft_backend_ok("auto", "MARLIN")

    # T3 (26 Oct): SM100 auto picked TrtLlmNvFp4ExpertsMonolithic, whose apply() has a router signature
    class Modular:
        is_monolithic = staticmethod(lambda: False)
        def apply(self, *a, **k): ...

    class TrtLlmNvFp4ExpertsMonolithic(Modular):
        is_monolithic = staticmethod(lambda: True)

    assert draft_check_skip(Modular()) is None
    assert "monolithic" in draft_check_skip(TrtLlmNvFp4ExpertsMonolithic())
    assert draft_check_skip(None) and draft_check_skip(object())


def _template(tmp_path):
    import struct

    t = tmp_path / "template"
    t.mkdir()
    head = json.dumps({"a.lora_A.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
    (t / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(head)) + head + b"\x01" * 8)
    (t / "adapter_config.json").write_text(json.dumps({"r": 16}))
    return t


def test_static_adapter_load_is_a_no_op_and_refuses_a_trained_one(tmp_path, monkeypatch):
    import asyncio

    from arena import rl_loop, rl_seq

    zero = tmp_path / "v0000"
    rl_seq.zero_adapter(_template(tmp_path), zero)
    server = rl_loop.Server("http://127.0.0.1:1", "m")
    server.static_lora = "policy"
    monkeypatch.setattr(rl_loop, "adapter_tensors", lambda path: None)  # file checks are not the subject here
    monkeypatch.setattr(rl_loop, "draft_files", lambda path: None)
    assert asyncio.run(server.load("policy", str(zero))) == 0.0 and server.loaded == "policy"
    with pytest.raises(RuntimeError, match="static adapter"):
        asyncio.run(server.load("policy", str(tmp_path / "template")))
    with pytest.raises(RuntimeError, match="static adapter"):
        asyncio.run(server.load("policy-r1", str(zero)))
