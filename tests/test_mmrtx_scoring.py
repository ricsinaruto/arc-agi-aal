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
