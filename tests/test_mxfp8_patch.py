"""CPU checks for the optional pinned-runtime backport and its protected scope."""

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from arc_infra import mxfp8_patch as patch
from arc_infra import mxfp8_runtime as runtime


@pytest.mark.parametrize('prefix', [
    'model.layers.0.self_attn.qkv_proj', 'language_model.model.layers.2.linear_attn.out_proj',
    'mtp.layers.48.self_attn.o_proj', 'model.layers.3.linear_attn.in_proj_qkvz',
])
def test_attention_eligibility(prefix):
    assert runtime.eligible(prefix, 4096)
    assert not runtime.eligible(prefix, 31)


@pytest.mark.parametrize('prefix', [
    'model.layers.0.linear_attn.conv1d', 'model.layers.0.linear_attn.in_proj_ba',
    'model.layers.0.mlp.gate', 'model.layers.0.mlp.experts', 'model.layers.0.ple.proj',
    'model.layers.0.attn_hyper_connection.input_mix_weight_up', 'lm_head',
    'visual.layers.0.self_attn.qkv_proj', 'other.model.layers.0.self_attn.o_proj',
])
def test_sensitive_layers_excluded(prefix):
    assert not runtime.eligible(prefix, 4096)


def test_dispatch_preserves_nvfp4_and_rejects_backend_fallback(monkeypatch):
    class BF16:
        pass

    class NVFP4:
        pass

    class Kernel:
        supported = True

        @classmethod
        def is_supported(cls):
            return cls.supported, 'missing native kernel'

        def __init__(self, config):
            self.config = config

    class Online:
        pass

    modules = {
        'vllm.model_executor.kernels.linear.mxfp8.flashinfer': SimpleNamespace(FlashInferCutlassMxfp8LinearKernel=Kernel),
        'vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel': SimpleNamespace(Mxfp8LinearLayerConfig=dict),
        'vllm.model_executor.layers.quantization.online.mxfp8': SimpleNamespace(Mxfp8OnlineLinearMethod=Online),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    layer = SimpleNamespace(input_size=4096, output_size=8192)
    prefix = 'model.layers.0.self_attn.qkv_proj'
    original = NVFP4()
    assert runtime.select(layer, prefix, original, BF16) is original
    original = BF16()
    assert runtime.select(layer, 'lm_head', original, BF16) is original
    selected = runtime.select(layer, prefix, original, BF16)
    assert isinstance(selected, Online) and isinstance(selected.kernel, Kernel)
    Kernel.supported = False
    with pytest.raises(RuntimeError, match='requires native FlashInfer CUTLASS'):
        runtime.select(layer, prefix, original, BF16)


def test_unknown_source_fails_closed():
    with pytest.raises(RuntimeError, match='unrecognized'):
        patch.patched_source(b'arbitrary upstream code')


def test_patch_idempotence_and_tamper_detection(tmp_path, monkeypatch):
    original = ('class Linear:\n    def __init__(self, return_bias, disable_tp):\n' + patch.ANCHOR).encode()
    monkeypatch.setattr(patch, 'ORIGINAL_SHA256', hashlib.sha256(original).hexdigest())
    target = tmp_path / patch.RELATIVE
    target.parent.mkdir(parents=True)
    target.write_bytes(original)
    identity = patch.apply(tmp_path, tmp_path / 'evidence')
    assert patch.apply(tmp_path, tmp_path / 'evidence') == identity
    assert target.read_text().count('arc_online_mxfp8 import select') == 1
    target.write_text(target.read_text() + '\n# tampered')
    with pytest.raises(RuntimeError, match='identity changed'):
        patch.apply(tmp_path, tmp_path / 'evidence')


def test_exact_pinned_constructor_dispatch_when_source_available(tmp_path):
    source = Path('/tmp/arc-pinned-vllm-source/model_executor/layers/linear.py')
    if not source.exists():
        pytest.skip('Optional exact runtime layer source; hash checked if supplied')
    patched = patch.patched_source(source.read_bytes())
    tree = ast.parse(patched)
    base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LinearBase')
    init = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    text = ast.unparse(init)
    # Hook runs after ModelOpt dispatch, before subclasses allocate/load weights.
    assert text.index('quant_config.get_quant_method') < text.index('self.quant_method = select(')
    assert text.index('self.quant_method = select(') < text.index('self.return_bias = return_bias')
    compile(patched, str(source), 'exec')
    (tmp_path / 'pinned-patch.json').write_text(json.dumps({
        'original': hashlib.sha256(source.read_bytes()).hexdigest(),
        'patched': hashlib.sha256(patched).hexdigest(),
    }))
