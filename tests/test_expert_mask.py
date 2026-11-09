"""Hero run (23 Oct): the pruned model served as a router-logit mask (arc_infra.expert_mask_patch / _runtime)."""
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

from arc_infra import expert_mask_patch as P

PINNED = os.environ.get("ARC_PINNED_MOE_RUNNER") or next(
    (str(p) for p in (Path(__file__).resolve().parents[2] / "throughput-arm/keep/vllm-full").glob(
        "vllm/model_executor/layers/fused_moe/runner/moe_runner.py")), None)


def mask_spec(tmp_path, experts=512, layers=48, per=192):
    pruned = {str(layer): list(range(layer % 7, layer % 7 + per)) for layer in range(layers)}
    raw = json.dumps({"format": "arc-expert-prune-v1", "experts": experts, "fraction": per / experts,
                      "pruned_per_layer": per, "pruned": pruned}).encode()
    path = tmp_path / "t37.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def fake_runner_source():
    anchor = P.RUNNER["edits"][0][0]
    return ("class MoERunner:\n    def _forward_impl(self, hidden_states, router_logits):\n        if self.gate is not None:\n"
            "            if self._fse_fuse_gate:\n                router_logits = F.linear(hidden_states, self.w)\n"
            + anchor + "            return router_logits\n").encode()


def test_patch_inserts_the_hook_once_and_is_hash_guarded(tmp_path, monkeypatch):
    src = fake_runner_source()
    monkeypatch.setitem(P.RUNNER, "sha256", hashlib.sha256(src).hexdigest())
    out = P.patched_source(src).decode()
    assert out.count("_arc_on_router_logits(self, router_logits)") == 1
    assert out.index("router_logits, _ = self.gate(hidden_states)") < out.index("_arc_on_router_logits")
    assert out.index("_arc_on_router_logits") < out.index("with self._sequence_parallel_context()")
    with pytest.raises(RuntimeError, match="unrecognized"):
        P.patched_source(src + b"\n")


def test_apply_writes_helper_settings_and_accepts_only_its_own_restart(tmp_path, monkeypatch):
    src = fake_runner_source()
    monkeypatch.setitem(P.RUNNER, "sha256", hashlib.sha256(src).hexdigest())
    site, out = tmp_path / "site", tmp_path / "out"
    target = site / P.RUNNER["relative"]
    target.parent.mkdir(parents=True)
    target.write_bytes(src)
    path, digest = mask_spec(tmp_path)
    ident = P.apply(site, out, path, digest)
    assert ident["pruned_total"] == 48 * 192 and (site / P.HELPER).is_file()
    settings = json.loads((site / P.SETTINGS).read_text())
    assert settings["experts"] == 512 and settings["label"] == "t37.json" and len(settings["pruned"]) == 48
    assert P.apply(site, out, path, digest) == ident  # restart in the same working dir
    with pytest.raises(RuntimeError, match="sha256"):
        P.apply(site, out, path, "0" * 64)
    (tmp_path / "x").mkdir()
    other, other_digest = mask_spec(tmp_path / "x", per=100)
    with pytest.raises(RuntimeError, match="modified or partially patched"):
        P.apply(site, out, other, other_digest)  # a different mask over an already patched runtime


@pytest.mark.skipif(PINNED is None, reason="no local copy of the pinned moe_runner.py")
def test_the_pinned_runner_patches():
    raw = Path(PINNED).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == P.RUNNER["sha256"]
    assert P.patched_source(raw).decode().count("arc_expert_mask") == 1


def test_validate():
    assert P.validate(None) is None
    with pytest.raises(ValueError):
        P.validate({"mask": "/abs.json", "sha256": "0" * 64})
    with pytest.raises(ValueError):
        P.validate({"mask": "m.json"})


def test_runtime_masks_each_layer_and_the_fused_gate_width(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    worker = types.ModuleType("vllm.v1.worker.worker_base")
    worker.WorkerWrapperBase = type("WorkerWrapperBase", (), {})
    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.worker_base", worker)
    from arc_infra import expert_mask_runtime as R

    settings = tmp_path / "arc_expert_mask.json"
    settings.write_text(json.dumps({"experts": 8, "pruned": {"0": [1, 2], "1": [], "2": [7]}, "label": "t"}))
    monkeypatch.setattr(R, "SETTINGS", str(settings))
    monkeypatch.setattr(R, "_STATE", None)
    runners = [types.SimpleNamespace(layer_id=i) for i in range(3)]
    x = torch.zeros(4, 8)
    assert torch.equal(R.on_router_logits(runners[0], x)[:, [1, 2]], torch.full((4, 2), R.MASK_VALUE))
    assert torch.count_nonzero(R.on_router_logits(runners[1], x)) == 0
    fused = R.on_router_logits(runners[2], torch.zeros(4, 9, dtype=torch.bfloat16))  # router + shared-expert gate
    assert fused.dtype == torch.bfloat16 and fused[:, 7].eq(torch.tensor(R.MASK_VALUE).to(torch.bfloat16)).all() and fused[:, 8].eq(0).all()
    with pytest.raises(RuntimeError, match="router logits"):
        R.on_router_logits(runners[0], torch.zeros(4, 10))
    with pytest.raises(RuntimeError, match="no row"):
        R.on_router_logits(types.SimpleNamespace(layer_id=5), x)
    status = worker.WorkerWrapperBase.arc_expert_mask(None, json.dumps({"op": "status"}))
    assert status == {"ok": True, "label": "t", "sha256": None, "pruned": 3, "masked_entries": 3, "layers": [0, 1, 2]}


def test_rl_servers_check_the_mask_after_start_and_wake(tmp_path, monkeypatch):
    import asyncio

    from arena import rl_loop

    replies = []

    async def post(self, path, payload=None, timeout=900):
        if path == "/collective_rpc":
            assert payload["method"] == "arc_expert_mask"
            return {"results": [replies.pop(0)]}
        return {}

    monkeypatch.setattr(rl_loop.Server, "post", post)
    monkeypatch.setattr(rl_loop.Server, "mask_expect", None)
    monkeypatch.setattr(rl_loop.Server, "mask_sha", None)
    s = rl_loop.Server("http://x/v1", "m")
    assert asyncio.run(s.mask_check("start")) is None  # no mask configured: no RPC
    path, _ = mask_spec(tmp_path, per=10)
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.study, loop.source, loop.server = {"server": {"expert_mask": {"mask": path.name, "sha256": "0" * 64}}}, tmp_path, s
    loop.feed = types.SimpleNamespace(emit=lambda kind, **kw: replies.append(("feed", kind, kw)))
    monkeypatch.delenv("ARC_PREFLIGHT_STUB", raising=False)
    replies.append({"ok": True, "label": "t37.json", "sha256": "1" * 64, "masked_entries": 480, "layers": list(range(48))})
    with pytest.raises(RuntimeError, match="not live after start"):  # another mask file (sha) on the server
        asyncio.run(loop.expert_mask_start())
    replies.append({"ok": True, "label": "t37.json", "sha256": "0" * 64, "masked_entries": 480, "layers": list(range(48))})
    asyncio.run(loop.expert_mask_start())
    row = replies.pop()
    assert rl_loop.Server.mask_expect == 480 and row[1] == "expert_mask" and row[2]["sha256"] == "0" * 64
    # the trainer: RL1_EXPERT_MASK applied with the bundle's file, by sha256 (hprobe-v4: silently ignored)
    monkeypatch.delenv("ARC_PREFLIGHT_FAKE_TRAINER", raising=False)
    assert loop.expert_mask_trainer({"expert_mask": {"sha256": "0" * 64, "layers": 48, "pruned": 480}})["layers"] == 48
    for loaded in ({}, {"expert_mask": {"sha256": "1" * 64}}):
        with pytest.raises(RuntimeError, match="trainer expert mask"):
            loop.expert_mask_trainer(loaded)
    monkeypatch.setenv("ARC_PREFLIGHT_FAKE_TRAINER", "1")
    assert loop.expert_mask_trainer({}) is None  # the CPU preflight's stand-in loads no model
    replies.append({"ok": True, "masked_entries": 0})  # a wake that lost the mask
    with pytest.raises(RuntimeError, match="not live after wake"):
        asyncio.run(s.wake())
    replies.append({"ok": False, "error": "mask not initialised"})
    with pytest.raises(RuntimeError, match="not live"):
        asyncio.run(s.mask_check("start"))


def test_skip_read_tail_only_when_switched_on(tmp_path, monkeypatch):
    src = fake_runner_source()
    monkeypatch.setitem(P.RUNNER, "sha256", hashlib.sha256(src).hexdigest())
    path, digest = mask_spec(tmp_path)
    runs = {}
    for on in (False, True):
        site, out = tmp_path / f"site{on}", tmp_path / f"out{on}"
        (site / P.RUNNER["relative"]).parent.mkdir(parents=True)
        (site / P.RUNNER["relative"]).write_bytes(src)
        monkeypatch.setenv(P.SKIP_ENV, "1" if on else "0")
        runs[on] = (P.apply(site, out, path, digest), (site / P.RUNNER["relative"]).read_text(),
                    json.loads((site / P.SETTINGS).read_text()))
    off, on = runs[False], runs[True]
    assert "install_skip_read" not in off[1] and "skip_read" not in off[2] and "skip_read" not in off[0]
    assert off[1] == P.patched_source(src).decode()  # off: the overlay as before the switch existed
    assert on[1].count("_arc_install_skip_read()") == 1 and on[2]["skip_read"] is True and on[0]["skip_read"] is True
    assert on[1].startswith(off[1].rstrip("\n"))  # the tail is the only difference
    monkeypatch.setenv(P.SKIP_ENV, "0")
    with pytest.raises(RuntimeError, match="modified or partially patched"):
        P.apply(tmp_path / "siteTrue", tmp_path / "outTrue", path, digest)  # a restart with the switch flipped


@pytest.mark.skipif(PINNED is None, reason="no local copy of the pinned moe_runner.py")
def test_the_pinned_runner_patches_with_skip_read():
    text = P.patched_source(Path(PINNED).read_bytes(), skip_read=True).decode()
    assert text.count("arc_expert_mask") == 2 and text.rstrip().endswith("_arc_install_skip_read()")


def _runtime(tmp_path, monkeypatch, spec):
    worker = types.ModuleType("vllm.v1.worker.worker_base")
    worker.WorkerWrapperBase = type("WorkerWrapperBase", (), {})
    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.worker_base", worker)
    from arc_infra import expert_mask_runtime as R

    settings = tmp_path / "arc_expert_mask.json"
    settings.write_text(json.dumps(spec))
    monkeypatch.setattr(R, "SETTINGS", str(settings))
    monkeypatch.setattr(R, "SKIP", {"installed": False, "skipped": 0})
    wu = types.SimpleNamespace(should_skip_weight=lambda name, ids: ids is not None and name.endswith("ep-other"))
    return R, wu


def test_skip_read_skips_only_masked_packed_weights(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    R, wu = _runtime(tmp_path, monkeypatch, {"experts": 8, "pruned": {"0": [1], "1": [2, 3]}, "label": "t",
                                             "skip_read": True})
    assert R.install_skip_read(wu) is True and R.install_skip_read(wu) is False
    pre = "model.language_model.layers"
    skip = [f"{pre}.0.mlp.experts.1.{p}_proj.weight" for p in ("gate", "up", "down")] + \
           [f"{pre}.1.mlp.experts.3.down_proj.weight"]
    keep = [f"{pre}.0.mlp.experts.1.gate_proj.{s}" for s in ("weight_scale", "weight_scale_2", "input_scale")] + \
           [f"{pre}.0.mlp.experts.2.gate_proj.weight", f"{pre}.1.mlp.experts.1.up_proj.weight",
            f"{pre}.0.mlp.gate.weight", f"{pre}.0.self_attn.q_proj.weight", "mtp.layers.0.mlp.experts.down_proj"]
    assert all(wu.should_skip_weight(n, None) for n in skip) and not any(wu.should_skip_weight(n, None) for n in keep)
    assert wu.should_skip_weight("x.ep-other", {1})  # the EP filter still applies
    assert R.SKIP["skipped"] == 4 and R.SKIP["expected"] == 9


def test_skip_read_off_installs_nothing(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    R, wu = _runtime(tmp_path, monkeypatch, {"experts": 8, "pruned": {"0": [1]}, "label": "t"})
    orig = wu.should_skip_weight
    assert R.install_skip_read(wu) is False and wu.should_skip_weight is orig


@pytest.mark.skipif(not os.environ.get("ARC_MASKSKIP_CHECK_DIR"), reason="needs the nvfp4 index + t37 mask copy")
def test_skip_read_on_the_real_checkpoint_names(tmp_path, monkeypatch):
    """ARC_MASKSKIP_CHECK_DIR: model.safetensors.index.json of the served nvfp4 checkpoint (nvfp4-index.json) and
    the hero's mask (mask.json). Skipped = exactly the 3 packed weights of every masked expert; no scale skipped."""
    pytest.importorskip("torch")
    d = Path(os.environ["ARC_MASKSKIP_CHECK_DIR"])
    mask = json.loads((d / "mask.json").read_text())
    R, wu = _runtime(tmp_path, monkeypatch, {"experts": mask["experts"], "pruned": mask["pruned"], "label": "t37",
                                             "skip_read": True})
    R.install_skip_read(wu)
    names = list(json.loads((d / "nvfp4-index.json").read_text())["weight_map"])
    skipped = [n for n in names if wu.should_skip_weight(n, None)]
    assert len(skipped) == R.SKIP["expected"] == 3 * sum(len(v) for v in mask["pruned"].values()) == 27648
    assert all(n.endswith("_proj.weight") and ".mlp.experts." in n for n in skipped)
    kept_scales = [n for n in names if n.startswith("model.language_model.") and ".mlp.experts." in n
                   and not n.endswith(".weight")]
    assert len(kept_scales) == 3 * 3 * 512 * 48  # every scale of every expert is still read


def test_load_profile_tail_and_settings_only_when_switched_on(tmp_path, monkeypatch):
    src = fake_runner_source()
    monkeypatch.setitem(P.RUNNER, "sha256", hashlib.sha256(src).hexdigest())
    path, digest = mask_spec(tmp_path)
    site, out = tmp_path / "site", tmp_path / "out"
    (site / P.RUNNER["relative"]).parent.mkdir(parents=True)
    (site / P.RUNNER["relative"]).write_bytes(src)
    monkeypatch.setenv(P.SKIP_ENV, "1")
    monkeypatch.setenv(P.PROFILE_ENV, "1")
    ident = P.apply(site, out, path, digest)
    text, settings = (site / P.RUNNER["relative"]).read_text(), json.loads((site / P.SETTINGS).read_text())
    assert text.count("_arc_install_load_profile()") == 1 and text.count("_arc_install_skip_read()") == 1
    assert settings["load_profile"] is True and ident["load_profile"] is True
    assert P.patched_source(src).decode() == P.patched_source(src, False, False).decode()  # off: unchanged


def test_load_profile_times_reads_and_expert_loader_and_passes_values_through(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    R, _ = _runtime(tmp_path, monkeypatch, {"experts": 8, "pruned": {"0": [1]}, "label": "t", "load_profile": True})
    monkeypatch.setattr(R, "PROFILE", {"installed": False, "loads": [], "expert": None})
    import time

    def safetensors_weights_iterator(files):
        for f in files:
            time.sleep(0.01)
            yield f, f.upper()

    wu = types.SimpleNamespace(safetensors_weights_iterator=safetensors_weights_iterator)
    loader_mod = types.SimpleNamespace(safetensors_weights_iterator=safetensors_weights_iterator)
    other = types.SimpleNamespace(safetensors_weights_iterator=lambda files: iter(()))
    got = []

    class FusedMoE:
        def weight_loader(self, param, loaded, name, shard_id=None, expert_id=None):
            time.sleep(0.005)
            got.append((name, loaded, expert_id))
            return True

    FusedMoE.weight_loader.supports_moe_loading = True  # as RoutedExperts (routed_experts.py)

    assert R.install_load_profile(wu, [loader_mod, other], FusedMoE) is True
    assert R.install_load_profile(wu, [loader_mod], FusedMoE) is False  # once
    assert getattr(FusedMoE().weight_loader, "supports_moe_loading", False) is True  # the flag survives the wrap
    assert loader_mod.safetensors_weights_iterator is wu.safetensors_weights_iterator  # the by-name import too
    assert other.safetensors_weights_iterator is not wu.safetensors_weights_iterator  # a different function: kept
    moe = FusedMoE()
    out = []
    for name, value in loader_mod.safetensors_weights_iterator(["a", "b", "c"]):
        out.append((name, value))
        if name != "b":
            assert moe.weight_loader(None, value, name, expert_id=1) is True
    assert out == [("a", "A"), ("b", "B"), ("c", "C")] and got == [("a", "A", 1), ("c", "C", 1)]
    row = R.PROFILE["loads"][0]
    assert row["tensors"] == 3 and row["expert_loader"]["n"] == 2
    assert row["read"]["wall"] >= 0.03 and row["expert_loader"]["wall"] >= 0.01
    assert row["seconds"] >= row["read"]["wall"] + row["expert_loader"]["wall"] - 0.02  # three 2-decimal roundings
    assert "ARC_LOAD_PROFILE " in capsys.readouterr().out


def test_load_cprofile_reports_top_functions(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    R, _ = _runtime(tmp_path, monkeypatch, {"experts": 8, "pruned": {"0": [1]}, "label": "t", "load_profile": True})
    assert R._cprofile_start() is None  # off without ARC_LOAD_CPROFILE
    monkeypatch.setenv("ARC_LOAD_CPROFILE", "1")

    def busy(files):
        for f in files:
            s = 0
            for i in range(300000):
                s += i
            yield f, s

    wrapped = R._profiled(busy, "busy")
    assert [v for _, v in wrapped(["a", "b"])] == [sum(range(300000))] * 2
    out = capsys.readouterr().out
    row = json.loads(out.split("ARC_LOAD_CPROFILE ")[1].splitlines()[0])
    assert row["iterator"] == "busy" and "busy" in row["top"][0][0]


def test_load_profile_off_installs_nothing(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    R, _ = _runtime(tmp_path, monkeypatch, {"experts": 8, "pruned": {"0": [1]}, "label": "t"})
    monkeypatch.setattr(R, "PROFILE", {"installed": False, "loads": [], "expert": None})
    wu = types.SimpleNamespace(safetensors_weights_iterator=lambda f: iter(f))
    assert R.install_load_profile(wu, [], type("F", (), {"weight_loader": lambda self: 1})) is False


def test_load_profile_install_failure_never_stops_the_load(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    R, _ = _runtime(tmp_path, monkeypatch, {"experts": 8, "pruned": {"0": [1]}, "label": "t", "load_profile": True})
    monkeypatch.setattr(R, "PROFILE", {"installed": False, "loads": [], "expert": None})
    wu = types.SimpleNamespace(safetensors_weights_iterator=lambda f: iter(f))
    assert R.install_load_profile(wu, [], object()) is False  # no weight_loader on the class
    assert "ARC_LOAD_PROFILE_ERROR" in capsys.readouterr().out


@pytest.mark.skipif(PINNED is None, reason="no local copy of the pinned runtime")
def test_load_profile_targets_exist_in_the_pinned_runtime():
    """The names install_load_profile wraps, in the pinned vLLM (0.1.dev20073): RoutedExperts.weight_loader (flagged
    supports_moe_loading) and the three weight iterators that default_loader imports by name."""
    vllm = Path(PINNED).parents[4]
    routed = (vllm / "model_executor/layers/fused_moe/routed_experts.py").read_text()
    utils = (vllm / "model_executor/model_loader/weight_utils.py").read_text()
    default = (vllm / "model_executor/model_loader/default_loader.py").read_text()
    assert "class RoutedExperts(" in routed and "RoutedExperts.weight_loader.supports_moe_loading = True" in routed
    for name in ("safetensors_weights_iterator", "multi_thread_safetensors_weights_iterator",
                 "fastsafetensors_weights_iterator"):
        assert f"\ndef {name}(" in utils and f"    {name},\n" in default


def test_postload_profile_times_quant_classes(monkeypatch, capsys):
    """ARC_LOAD_PROFILE: the post-load pass timed per quant-method class; values and calls pass through."""
    pytest.importorskip("torch")
    import sys
    import types

    from arc_infra import expert_mask_runtime as rt

    class QuantizeMethodBase:
        def process_weights_after_loading(self, layer):
            layer.append("base")

    class Marlin(QuantizeMethodBase):
        def process_weights_after_loading(self, layer):
            layer.append("marlin")

    class Child(Marlin):  # inherits Marlin's: timed once, under Marlin
        pass

    def pwal(model, *args):
        for m in model:
            m.process_weights_after_loading(model.log)
        return "done"

    model = type("M", (list,), {})([Marlin(), Child(), QuantizeMethodBase()])
    model.log = []
    utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    utils.process_weights_after_loading = pwal
    base = types.ModuleType("vllm.model_executor.layers.quantization.base_config")
    base.QuantizeMethodBase = QuantizeMethodBase
    loader = types.ModuleType("vllm.model_executor.model_loader.base_loader")
    loader.process_weights_after_loading = pwal
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.model_loader", "vllm.model_executor.layers",
                 "vllm.model_executor.layers.quantization"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    rt._install_postload_profile([loader])
    assert loader.process_weights_after_loading is utils.process_weights_after_loading
    assert utils.process_weights_after_loading(model) == "done"
    assert model.log == ["marlin", "marlin", "base"]
    line = [l for l in capsys.readouterr().out.splitlines() if l.startswith("ARC_POSTLOAD_PROFILE ")][0]
    row = __import__("json").loads(line.split(" ", 1)[1])
    assert row["classes"]["Marlin"]["n"] == 2 and "seconds" in row
