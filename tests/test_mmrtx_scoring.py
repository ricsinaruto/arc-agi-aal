"""mmrtx: NVFP4 codec (cand_build) and KL scoring (kl_score) on stubs; no modal, vllm or GPU."""
import gzip
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

D = Path(__file__).resolve().parents[1] / "scripts" / "mmrtx"


def _load(name, monkeypatch):
    monkeypatch.syspath_prepend(str(D))
    monkeypatch.setitem(sys.modules, "modal", None)  # import modal -> ImportError
    if name == "cand_build":
        fake = types.SimpleNamespace(
            App=lambda *a, **k: types.SimpleNamespace(function=lambda *a, **k: (lambda f: f),
                                                      local_entrypoint=lambda *a, **k: (lambda f: f), app_id=None),
            Volume=types.SimpleNamespace(from_name=lambda *a, **k: None), is_local=lambda: False)
        monkeypatch.setitem(sys.modules, "modal", fake)
        monkeypatch.setitem(sys.modules, "image", types.SimpleNamespace(IMAGE=None))
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_codec(monkeypatch):
    torch = pytest.importorskip("torch")
    C = _load("cand_build", monkeypatch)
    torch.manual_seed(0)
    O, I = 64, 256
    packed = torch.randint(0, 256, (O, I // 2), dtype=torch.uint8)
    gs = 1234.5
    scale = (torch.rand(O, I // 16) * 300 + 10).to(torch.float8_e4m3fn)
    assert torch.equal(C.pack(C.unpack(packed)), packed)  # incl. -0 codes
    w = C.dequant({"weight_packed": packed, "weight_scale": scale, "weight_global_scale": torch.tensor(gs)})
    assert torch.equal(C.codes_with_scale(w, scale, gs), C.unpack(packed))

    def err(p, s):
        return float(((C.unpack(p) * s.float().repeat_interleave(16, 1) / gs - w) ** 2).mean() / (w ** 2).mean())
    e4, e3, e2 = err(*C.quant_nvfp4(w, gs)), err(*C.quant_grid(w, gs, 3)), err(*C.quant_grid(w, gs, 2))
    assert e4 < 0.01 < e3 < e2
    for bits in (2, 3):  # grid values are exact E2M1 numbers and stay on the grid
        p, s = C.quant_grid(w, gs, bits)
        vals = set(C.unpack(p).abs().unique().tolist())
        assert vals <= {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0}


V = 50


def _dist(np, seed, p, shift):
    r = np.random.default_rng(seed * 1000 + p).normal(size=V) + shift * np.random.default_rng(7 + p).normal(size=V)
    return r - np.log(np.exp(r).sum())


def _stub_vllm(np, shift):
    class LP:
        def __init__(self, v):
            self.logprob = v

    class LLM:
        def __init__(self, **kw):
            self.kw = kw

        def generate(self, prompts, sp, use_tqdm=False):
            outs = []
            for pr in prompts:
                toks, pl = pr["prompt_token_ids"], [None]
                for p in range(1, len(toks)):
                    lp = _dist(np, toks[0], p, shift)
                    d = {int(t): LP(float(lp[t])) for t in np.argsort(-lp)[:20]}
                    d[toks[p]] = LP(float(lp[toks[p]]))
                    pl.append(d)
                outs.append(types.SimpleNamespace(prompt_logprobs=pl))
            return outs
    return types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **k: k)


def test_kl_score(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    ref = {}
    for name in ("held3", "held2"):
        rows = []
        for i in range(3):
            toks = [i + (10 if name == "held2" else 0)] + [int(t) for t in rng.integers(0, V, 40)]
            rid, spans = f"{name}-{i:04d}-x", [(5, 15), (25, 40)]
            rows.append({"id": rid, "tokens": toks, "spans": spans})
            pos = [p for s, e in spans for p in range(s, e)]
            ids, lps, tl = np.zeros((len(pos), 20), np.int32), np.zeros((len(pos), 20), np.float32), np.zeros(len(pos), np.float32)
            for j, p in enumerate(pos):
                lp = _dist(np, toks[0], p, 0)
                top = np.argsort(-lp)[:20]
                ids[j], lps[j], tl[j] = top, lp[top], lp[toks[p]]
            ref.update({f"{rid}|pos": np.array(pos, np.int32), f"{rid}|ids": ids, f"{rid}|lps": lps.astype(np.float16),
                        f"{rid}|tok_lp": tl})
        with gzip.open(tmp_path / f"{name}.jsonl.gz", "wt") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    np.savez(tmp_path / "kl.npz", **ref)
    K = _load("kl_score", monkeypatch)
    res = {}
    for shift in (0.0, 0.5):
        monkeypatch.setitem(sys.modules, "vllm", _stub_vllm(np, shift))
        out = tmp_path / f"s{shift}.json"
        K.score_one({"name": f"s{shift}", "model": "x", "kv": "fp8", "reverse": shift > 0}, str(tmp_path),
                    str(tmp_path / "kl.npz"), str(out))
        res[shift] = json.loads(out.read_text())
    a, b = res[0.0], res[0.5]
    assert abs(a["kl20"]) < 1e-3 and abs(a["dnll"]) < 1e-3 and a["top1"] == 1.0
    assert b["kl20"] > 0.01 and b["top1"] < 1 and b["n"] == a["n"] == 150
    lo, hi = K.bootstrap(b, a)
    assert 0 < lo <= b["kl20"] <= hi


def test_moe_a16_patch(tmp_path):
    """sitecustomize forces use_a16=True on the target class, positional or keyword, only when MMRTX_MOE_A16=1."""
    import subprocess
    pkg = tmp_path / "fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "m.py").write_text("class M:\n    def __init__(self, moe, layer_name=None, use_a16=False, key=1):\n"
                              "        self.use_a16 = use_a16\n")
    code = ("import fakepkg.m as m; print(m.M(0).use_a16, m.M(0, 'x', False).use_a16, m.M(0, use_a16=False).use_a16)")
    env = {"PYTHONPATH": f"{D / 'mmrtx_patch'}:{tmp_path}", "MMRTX_PATCH_TARGET": "fakepkg.m", "MMRTX_PATCH_CLASS": "M",
           "PATH": "/usr/bin:/bin"}
    on = subprocess.run([sys.executable, "-c", code], env={**env, "MMRTX_MOE_A16": "1"}, capture_output=True, text=True)
    off = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert on.stdout.split() == ["True", "True", "True"], on.stderr
    assert off.stdout.split() == ["False", "False", "False"], off.stderr


def test_moe_requant_patch(tmp_path):
    """MMRTX_MOE_REQUANT swaps only the NVFP4 routed-expert method for HummingMoEMethod with NVFP4 source + forced schema."""
    import subprocess
    files = {
        "fakepkg/__init__.py": "",
        "fakepkg/mo.py": ("class CompressedTensorsW4A4Nvfp4MoEMethod:\n    pass\n"
                          "class ModelOptMxFp8Config:\n    def get_quant_method(self, layer, prefix):\n"
                          "        return CompressedTensorsW4A4Nvfp4MoEMethod() if prefix.endswith('experts') else 'dense'\n"),
        "vllm/__init__.py": "", "vllm/model_executor/__init__.py": "", "vllm/model_executor/layers/__init__.py": "",
        "vllm/model_executor/layers/quantization/__init__.py": "",
        "vllm/model_executor/layers/quantization/humming.py": (
            "class HummingLayerQuantizationConfig:\n    def __init__(self, **kw):\n        self.kw = kw\n"
            "class HummingMoEMethod:\n    def __init__(self, qc, moe):\n        self.qc, self.moe = qc, moe\n"
            "    def prepare_weight_loader(self, layer, wl):\n        return wl\n"),
        "humming/__init__.py": "", "humming/schema/__init__.py": "",
        "humming/schema/base.py": "class BaseWeightSchema:\n    @staticmethod\n    def from_config(c):\n        return c\n",
        "humming/schema/humming.py": "class HummingInputSchema:\n    pass\n",
    }
    for rel, txt in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(txt)
    code = ("import types, fakepkg.mo as mo; c = mo.ModelOptMxFp8Config(); L = types.SimpleNamespace(moe_config='MC')\n"
            "m = c.get_quant_method(L, 'l.experts'); d = c.get_quant_method(L, 'l.qkv')\n"
            "print(type(m).__name__, d, m.moe, m.qc.kw['weight_schema']['format'], m.qc.kw['force_weight_schema']['b_dtype'],"
            " type(m.qc.kw['force_input_schema']).__name__)")
    env = {"PYTHONPATH": f"{D / 'mmrtx_patch'}:{tmp_path}", "MMRTX_MODELOPT_TARGET": "fakepkg.mo", "PATH": "/usr/bin:/bin"}
    force = '{"quant_method":"humming","b_dtype":"uint3","group_size":16,"has_zero_point":true}'
    on = subprocess.run([sys.executable, "-c", code], env={**env, "MMRTX_MOE_REQUANT": force}, capture_output=True, text=True)
    off = subprocess.run([sys.executable, "-c", code.replace("m.moe", "0").split("\nprint")[0] + "\nprint(type(m).__name__)"],
                         env=env, capture_output=True, text=True)
    assert on.stdout.split() == ["HummingMoEMethod", "dense", "MC", "nvfp4-pack-quantized", "uint3", "HummingInputSchema"], on.stderr
    assert off.stdout.split() == ["CompressedTensorsW4A4Nvfp4MoEMethod"], off.stderr
    nat = subprocess.run([sys.executable, "-c", code.split("\nprint")[0] + "\nprint(m.qc.kw['weight_schema']['b_dtype'], "
                          "'force_weight_schema' in m.qc.kw)"], env={**env, "MMRTX_MOE_HUMMING": force}, capture_output=True, text=True)
    assert nat.stdout.split() == ["uint3", "False"], nat.stderr


def test_scalar_global_scale_loader():
    """w13 global scale goes straight into the [E] param; mismatched w1/w3 raises; other params pass through."""
    import importlib.util
    import types
    import torch
    spec = importlib.util.spec_from_file_location("mm_sc", D / "mmrtx_patch/sitecustomize.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = []
    load = mod._scalar_global_scale(lambda layer, wl: lambda *a, **k: calls.append(k["shard_id"]) or "inner")(None, None)
    gs = types.SimpleNamespace(param_name="weight_global_scale", data=torch.zeros(4))
    load(gs, torch.tensor(3.0), "w", "w1", expert_id=2)
    assert load(gs, torch.tensor(3.0), "w", "w3", expert_id=2, return_success=True) is True
    assert gs.data.tolist() == [0, 0, 3.0, 0]
    with pytest.raises(ValueError):
        load(gs, torch.tensor(4.0), "w", "w3", expert_id=2)
    w2 = types.SimpleNamespace(param_name="weight_global_scale", data=torch.zeros(4))
    assert load(w2, torch.tensor(5.0), "w", "w2", expert_id=1) == "inner"
    packed = types.SimpleNamespace(param_name="weight_packed", data=torch.zeros(4, 2))
    assert load(packed, torch.zeros(2), "w", "w1", expert_id=1) == "inner" and calls == ["w2", "w1"]


def test_hum_build_files(tmp_path, monkeypatch):
    """hum_build main: expert tensors replaced by the schema's keys (quant faked; CUDA-only), rest symlinked, index whole."""
    import importlib.util
    import torch
    from safetensors.torch import save_file, load_file
    fake = types.ModuleType("humming"); fake.schema = types.ModuleType("humming.schema")
    base = types.ModuleType("humming.schema.base")
    base.BaseWeightSchema = types.SimpleNamespace(from_config=lambda c: types.SimpleNamespace(**c))
    for n, m in (("humming", fake), ("humming.schema", fake.schema), ("humming.schema.base", base)):
        monkeypatch.setitem(sys.modules, n, m)
    spec = importlib.util.spec_from_file_location("hb", D / "hum_build.py")
    hb = importlib.util.module_from_spec(spec); spec.loader.exec_module(hb)
    monkeypatch.setattr(hb, "quant", lambda sch, w, search=True: {"weight": torch.zeros(w.shape[0], 3, dtype=torch.int32),
                                                                 "weight_scale": torch.ones(w.shape[0], 2, dtype=torch.bfloat16)})
    monkeypatch.setattr(hb, "rel_mse", lambda sch, q, w: 0.03)
    monkeypatch.setattr(hb, "dequant", lambda t, dev: torch.ones(4, 32))
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    pre = "language_model.model.layers.3.block_sparse_moe."
    ex = {f"{pre}experts.{e}.{w}.{k}": torch.zeros(4, 16, dtype=torch.uint8) for e in (0, 1) for w in ("w1", "w2", "w3")
          for k in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")}
    rest = {f"{pre}gate.weight": torch.zeros(2, 4), "language_model.model.embed_tokens.weight": torch.zeros(3, 4)}
    save_file(ex, str(src / "model-layer03.safetensors")); save_file(rest, str(src / "model-rest.safetensors"))
    idx = {**{n: "model-layer03.safetensors" for n in ex}, **{n: "model-rest.safetensors" for n in rest}}
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": idx}))
    (src / "config.json").write_text("{}")
    monkeypatch.setattr(hb, "dequant", lambda t, dev: torch.ones(4, 32))
    info = hb.main(str(src), str(dst), '{"quant_method": "humming", "b_dtype": "uint3"}')
    wm = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(n for n in wm if ".experts." in n) == {f"{pre}experts.{e}.{w}.{k}" for e in (0, 1) for w in ("w1", "w2", "w3")
                                                      for k in ("weight", "weight_scale")}
    assert all(wm[n] == "model-rest.safetensors" for n in rest)
    assert (dst / "model-rest.safetensors").is_symlink() and (dst / "config.json").is_symlink()
    assert set(load_file(str(dst / "model-hum-layer03.safetensors"))) == {n for n in wm if ".experts." in n}
    assert info["rel_mse_mean"] == 0.03


def test_hum_variant(tmp_path):
    """variant: chosen layers' experts from the Humming build, others from the NVFP4 source; non-experts from source."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("hb2", D / "hum_build.py")
    hb = importlib.util.module_from_spec(spec); spec.loader.exec_module(hb)
    src, hum, dst = tmp_path / "src", tmp_path / "hum", tmp_path / "dst"
    src.mkdir(); hum.mkdir()
    ex = lambda L, k: f"language_model.model.layers.{L}.block_sparse_moe.experts.0.w1.{k}"  # noqa: E731
    si = {ex(3, "weight_packed"): "model-layer03.safetensors", ex(4, "weight_packed"): "model-layer04.safetensors",
          "lm_head.weight": "model-rest.safetensors"}
    hi = {ex(3, "weight"): "model-hum-layer03.safetensors", ex(4, "weight"): "model-hum-layer04.safetensors",
          "lm_head.weight": "model-rest.safetensors"}
    for f in set(si.values()):
        (src / f).write_text(f)
    for f in ("model-hum-layer03.safetensors", "model-hum-layer04.safetensors"):
        (hum / f).write_text(f)
    (hum / "model-rest.safetensors").symlink_to(src / "model-rest.safetensors")
    (src / "config.json").write_text("{}")
    for d, i in ((src, si), (hum, hi)):
        (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": i}))
    hb.variant(src, hum, dst, "3")
    wm = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    assert wm == {ex(3, "weight"): "model-hum-layer03.safetensors", ex(4, "weight_packed"): "model-layer04.safetensors",
                  "lm_head.weight": "model-rest.safetensors"}
    assert (dst / "model-hum-layer03.safetensors").read_text() == "model-hum-layer03.safetensors"
    assert (dst / "model-layer04.safetensors").read_text() == "model-layer04.safetensors"
    assert (dst / "config.json").exists() and not (dst / "model-hum-layer04.safetensors").exists()


def test_moe_humming_layers(tmp_path):
    """MMRTX_MOE_HUMMING_LAYERS: only listed layers' RoutedExperts get the Humming method."""
    import subprocess
    files = {
        "fakepkg/__init__.py": "",
        "fakepkg/mo.py": ("class CompressedTensorsW4A4Nvfp4MoEMethod:\n    pass\n"
                          "class RoutedExperts:\n    moe_config = 'MC'\n"
                          "class ModelOptMxFp8Config:\n    moe_quant_algo = 'NVFP4'\n"
                          "    def is_layer_excluded(self, prefix):\n        return False\n"
                          "    def get_quant_method(self, layer, prefix):\n        return CompressedTensorsW4A4Nvfp4MoEMethod()\n"),
        "vllm/__init__.py": "", "vllm/model_executor/__init__.py": "", "vllm/model_executor/layers/__init__.py": "",
        "vllm/model_executor/layers/quantization/__init__.py": "",
        "vllm/model_executor/layers/quantization/humming.py": (
            "class HummingLayerQuantizationConfig:\n    def __init__(self, **kw):\n        self.kw = kw\n"
            "class HummingMoEMethod:\n    def __init__(self, qc, moe):\n        self.qc, self.moe = qc, moe\n"
            "    def prepare_weight_loader(self, layer, wl):\n        return wl\n"),
        "humming/__init__.py": "", "humming/schema/__init__.py": "",
        "humming/schema/base.py": "class BaseWeightSchema:\n    @staticmethod\n    def from_config(c):\n        return c\n",
        "humming/schema/humming.py": "class HummingInputSchema:\n    pass\n",
    }
    for rel, txt in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(txt)
    code = ("import fakepkg.mo as mo; c = mo.ModelOptMxFp8Config(); L = mo.RoutedExperts()\n"
            "print(*[type(c.get_quant_method(L, f'language_model.model.layers.{i}.block_sparse_moe.experts')).__name__"
            " for i in (3, 4, 13)])")
    force = '{"quant_method":"humming","b_dtype":"uint3","group_size":32,"has_zero_point":false}'
    env = {"PYTHONPATH": f"{D / 'mmrtx_patch'}:{tmp_path}", "MMRTX_MODELOPT_TARGET": "fakepkg.mo", "PATH": "/usr/bin:/bin",
           "MMRTX_MOE_HUMMING": force}
    r = subprocess.run([sys.executable, "-c", code], env={**env, "MMRTX_MOE_HUMMING_LAYERS": "3,13"}, capture_output=True, text=True)
    assert r.stdout.split() == ["HummingMoEMethod", "CompressedTensorsW4A4Nvfp4MoEMethod", "HummingMoEMethod"], r.stderr
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.stdout.split() == ["HummingMoEMethod"] * 3, r.stderr


def test_select_layers(monkeypatch, tmp_path):
    """Pre-registered rule: cheapest groups per layer first until >= min_layers; predicted = anchor + sum of deltas."""
    K = _load("kl_score", monkeypatch)
    calls = []
    monkeypatch.setitem(sys.modules, "hum_build", types.SimpleNamespace(variant=lambda *a: calls.append(a)))
    groups = {"g1": [3, 4, 5], "g2": [6, 7], "g3": [8, 9, 10, 11]}
    done = {"base": {"kl20": 0.30}, "s1": {"kl20": 0.36}, "s2": {"kl20": 0.31}, "s3": {"kl20": 0.32}}
    job = {"name": "mixed", "env": {"A": "1"}, "select": {"base": "base", "groups": groups,
           "group_jobs": {"g1": "s1", "g2": "s2", "g3": "s3"}, "min_layers": 5, "anchor_kl20": 0.285,
           "src": "S", "hum": "H", "dst": "/tmp/v"}}
    j = K.select_layers(job, done, tmp_path)
    # per layer: g2 0.005, g3 0.005, g1 0.02 -> g2 (2 layers) then g3 (6 layers) reaches 5
    assert j["selection"]["chosen"][0] in ("g2", "g3") and set(j["selection"]["chosen"]) == {"g2", "g3"}
    assert j["env"]["MMRTX_MOE_HUMMING_LAYERS"] == "6,7,8,9,10,11" and j["env"]["A"] == "1" and j["model"] == "/tmp/v"
    assert abs(j["selection"]["predicted_kl20"] - (0.285 + 0.01 + 0.02)) < 1e-9 and calls == [("S", "H", "/tmp/v", "6,7,8,9,10,11")]
    assert "select" not in j and K.select_layers({**job}, {"base": {"kl20": 0.3}}, tmp_path) is None


def test_bootstrap_mixed_rows(monkeypatch):
    """Paired CI over different row sets (full set vs a stride subset) uses the shared rows only."""
    K = _load("kl_score", monkeypatch)
    a = {"per_seq": {"x": {"kl20": 2.0, "n": 10}, "y": {"kl20": 4.0, "n": 10}}}
    b = {"per_seq": {"x": {"kl20": 1.0, "n": 10}}}
    assert K.bootstrap(a, b) == (0.1, 0.1)
    assert K.bootstrap(a, {"per_seq": {"z": {"kl20": 1.0, "n": 1}}}) is None
