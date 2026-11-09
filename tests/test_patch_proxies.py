"""Every proxy our runtime patches install survives deepcopy and pickle (RL lesson, 25 Oct: v6n died at boot when vLLM's
renderer pool deep-copied the segtok proxy, and v6n2 served empty content when it reached Rust code).

Generic part: every class under src/arc_infra that defines ``__getattr__`` (found by AST, so a new proxy is caught
without editing this file) must (1) be registered below with a sample factory, and (2) answer attribute probes on a
bare ``cls.__new__(cls)`` instance with AttributeError only (copy/pickle probe ``__setstate__`` etc. on an
uninitialised object; a forwarding ``__getattr__`` that reads its own unset field recurses). Registered samples must
round-trip through deepcopy and pickle and still forward to the wrapped object.
"""
import ast
import copy
import importlib
import pickle
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _segtok():
    tokenizers = pytest.importorskip("tokenizers")
    from arc_infra.segment_token_runtime import SegmentTokenizer

    real = tokenizers.Tokenizer(tokenizers.models.WordLevel({"a": 0, "b": 1, "[UNK]": 2}, unk_token="[UNK]"))
    real.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    real.add_special_tokens(["<|im_end|>"])
    return SegmentTokenizer(real), "get_vocab_size", real.get_vocab_size


SAMPLES = {"arc_infra.segment_token_runtime.SegmentTokenizer": _segtok}
# classes with __getattr__ that never reach third-party copy/pickle paths (reason required)
EXEMPT = {}


def proxy_classes():
    found = []
    for path in sorted((ROOT / "src" / "arc_infra").rglob("*.py")):
        tree = ast.parse(path.read_text())
        module = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)
        for node in tree.body:  # module-level classes only (local classes cannot be installed as globals)
            if isinstance(node, ast.ClassDef) and any(isinstance(b, ast.FunctionDef) and b.name == "__getattr__"
                                                      for b in node.body):
                found.append(f"{module}.{node.name}")
    return found


def test_every_proxy_is_registered():
    missing = [c for c in proxy_classes() if c not in SAMPLES and c not in EXEMPT]
    assert not missing, f"register a sample (deepcopy/pickle check) or an exemption with a reason: {missing}"


@pytest.mark.parametrize("name", proxy_classes())
def test_bare_instance_probes_raise_attribute_error(name):
    module, _, cls_name = name.rpartition(".")
    try:
        cls = getattr(importlib.import_module(module), cls_name)
    except ImportError as error:  # a module needing the GPU runtime: its sample (if any) covers it where it imports
        pytest.skip(f"{module}: {error}")
    bare = cls.__new__(cls)
    for attr in ("__setstate__", "__getstate__", "__deepcopy__", "__reduce_ex__", "__getnewargs_ex__", "__copy__"):
        try:
            getattr(bare, attr)
        except AttributeError:
            pass  # the only acceptable failure (RecursionError, KeyError ... break copy and pickle)


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_sample_round_trips(name):
    obj, attr, value = SAMPLES[name]()
    for clone in (copy.deepcopy(obj), pickle.loads(pickle.dumps(obj))):
        assert type(clone) is type(obj) and getattr(clone, attr)() == value()
        assert clone.encode("a b <|im_end|> a").ids == obj.encode("a b <|im_end|> a").ids
