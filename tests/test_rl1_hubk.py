"""RL1_HUB_PREFETCH (rl1_hubk): the hub kernels a Flash forward resolves lazily, fetched before the gate; after it
rl1_firstcall's hub_fetch counts only later snapshot_download calls (D2 27 Oct: ~8 s of the first window's 13.6 s
cold prefix). Torch-free: fake modules, mapping and fetcher."""
import enum
import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))

import rl1_firstcall  # noqa: E402
import rl1_hubk  # noqa: E402

SOURCE = '''
from ...integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub_with_fallback

@use_kernel_forward_from_hub("RMSNormGated")
class Norm:
    pass

@use_kernel_func_from_hub_with_fallback("causal_conv1d_fn", "causal_conv1d")
def conv(): pass

@use_kernel_func_from_hub_with_fallback( "chunk_gated_delta_rule", "fla")
def gdr(): pass
'''


class Mode(enum.Enum):
    INFERENCE = 1
    TRAINING = 2


class Repo:
    def __init__(self, repo_id, version=None, revision=None):
        self.repo_id, self.version, self.revision = repo_id, version, revision


MAPPING = {
    "RMSNormGated": {"cuda": {Mode.TRAINING: Repo("kernels-community/mamba-ssm", 3),
                              Mode.INFERENCE: Repo("kernels-community/mamba-ssm", 3)}},
    "causal_conv1d_fn": {"cuda": {Mode.TRAINING: Repo("kernels-community/mamba-ssm", 3)}},
    "chunk_gated_delta_rule": {"cuda": {Mode.TRAINING: Repo("kernels-community/fla", 1)}, "xpu": {}},
    "MultiScaleDeformableAttention": {"cuda": Repo("kernels-community/deformable-detr", 1)},  # not declared
    "Pinned": {"cuda": Repo("kernels-community/pinned", revision="abc")},
}


def fake_model(tmp_path):
    path = tmp_path / "fake_modeling.py"
    path.write_text(SOURCE)
    spec = importlib.util.spec_from_file_location("transformers.models.fake.modeling_fake", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    layer = type("FakeLayer", (), {"__module__": spec.name})
    hub = type("Hub", (), {"__module__": "kernels.layer", "kernel_layer_name": "Pinned"})

    class Model:
        def modules(self):
            return iter([layer(), hub(), layer()])

    return Model()


def test_declared_names_map_to_cuda_training_repos(tmp_path):
    names = rl1_hubk.declared(fake_model(tmp_path))
    assert names == {"RMSNormGated", "causal_conv1d_fn", "chunk_gated_delta_rule", "Pinned"}
    assert rl1_hubk.repos(names, MAPPING) == [("kernels-community/pinned", None, "abc"),
                                               ("kernels-community/mamba-ssm", 3, None),
                                               ("kernels-community/fla", 1, None)]


def test_prefetch_fetches_each_repo_once_and_never_raises(tmp_path):
    got = []

    def get(repo, **kw):
        got.append((repo, kw))
        if repo.endswith("pinned"):
            raise FileNotFoundError("no build for this torch")

    row = rl1_hubk.prefetch(fake_model(tmp_path), get=get, mapping=MAPPING)
    assert got == [("kernels-community/pinned", {"revision": "abc"}), ("kernels-community/mamba-ssm", {"version": 3}),
                   ("kernels-community/fla", {"version": 1})]
    assert [r["repo"] for r in row["repos"]] == ["kernels-community/mamba-ssm", "kernels-community/fla"]
    assert len(row["errors"]) == 1 and "pinned" in row["errors"][0]
    assert rl1_hubk.prefetch(fake_model(tmp_path), get=get, mapping=None)["errors"]  # no transformers here: a row


def test_hub_fetch_counts_only_after_the_prefetch(tmp_path, monkeypatch):
    calls = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *a, **k: calls.append(a) or "/cache"
    kmod = types.ModuleType("kernels.utils")
    kmod.snapshot_download = hub.snapshot_download  # bound by name at import, as kernels does
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "kernels.utils", kmod)
    monkeypatch.setattr(rl1_firstcall, "INSTALLED", ["triton_compile"])
    monkeypatch.setattr(rl1_firstcall, "STATS", {})
    row = rl1_hubk.prefetch(fake_model(tmp_path), get=lambda repo, **kw: kmod.snapshot_download(repo), mapping=MAPPING)
    assert len(calls) == 3 and not row["errors"] and "hub_fetch" not in rl1_firstcall.STATS  # the prefetch's own: reset
    kmod.snapshot_download("kernels-community/fla")  # the first forward's lazy resolve
    assert rl1_firstcall.STATS["hub_fetch"][0] == 1 and rl1_firstcall.report()["hub_fetch"]["calls"] == 1
    rl1_firstcall.wrap_hub()  # idempotent: not wrapped twice
    kmod.snapshot_download("x")
    assert rl1_firstcall.STATS["hub_fetch"][0] == 2
