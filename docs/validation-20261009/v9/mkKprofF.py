"""Top-k 2 decode bench: kprofE-off's bundle, one VM, trials base, k2, baserep; k2 = text_overrides
num_experts_per_tok 2 through ARENA_RC_OVERRIDES, slim_base kept (a config field, no tensors). All arms log the live routed k:
one ARC_TOPK line per router (main layers and MTP) with router attributes and topk_ids' shape from its first select_experts."""
import hashlib, json, shutil
from pathlib import Path

D = Path(__file__).resolve().parents[3] / "dist"
SRC, OUT = D / "b2-kprofE-off-v1", D / "b2-kprofF-k2-v1"
shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "dataset-upload"))


def sub(path, a, b):
    t = path.read_text()
    assert t.count(a) == 1, a
    path.write_text(t.replace(a, b))


sub(OUT / "src/arena/research_checkpoint.py", '            if drop or overrides:\n                raise ValueError("slim_base with',
    '            if drop:\n                raise ValueError("slim_base with')
READ = '''

def install_topk_read():
    """One ARC_TOPK line per router: what it holds and the k its first selection returned. Never fails the worker."""
    import json as _json
    try:
        from vllm.model_executor.layers.fused_moe.runner import moe_runner as _m
        seen, pinned = set(), _m.MoERunner._forward_impl

        def wrapped(self, *args, **kwargs):
            router = getattr(self, "router", None)
            if router is not None and id(router) not in seen:
                seen.add(id(router))
                select = router.select_experts

                def once(*a, **k):
                    out = select(*a, **k)
                    router.select_experts = select
                    try:
                        cfg = getattr(self, "moe_config", None)
                        print("ARC_TOPK " + _json.dumps({
                            "n": len(seen), "router": type(router).__name__, "top_k": getattr(router, "top_k", None),
                            "config_k": getattr(cfg, "experts_per_token", None), "experts": getattr(cfg, "num_experts", None),
                            "topk_ids": list(out[1].shape)}), flush=True)
                    except Exception as exc:  # noqa: BLE001 - diagnostics only
                        print(f"ARC_TOPK read failed: {exc!r}", flush=True)
                    return out
                router.select_experts = once
            return pinned(self, *args, **kwargs)
        _m.MoERunner._forward_impl = wrapped
        print("ARC_TOPK installed", flush=True)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        print(f"ARC_TOPK install failed: {exc!r}", flush=True)
'''
g = OUT / "src/arc_infra/gdn_defer_runtime.py"
t = g.read_text()
A = "    install_steplog()\n"
assert t.count(A) == 1
t = t.replace(A, A + "    install_topk_read()\n") + READ
compile(t, g.name, "exec")
g.write_text(t)
p = OUT / "configs/comparison.json"
s = json.loads(p.read_text().replace(SRC.name, OUT.name))
env = s["trials"][0]["server_env"]
assert "ARENA_RC_OVERRIDES" not in env and s["kv_margin"] == 0.995 and s["throughput_bench"]["output_tokens"] == 14000
WALL = 1500


def arm(name, extra):
    return {"name": name, "kind": "serving-bench", "competition": False, "wall_seconds": WALL, "server_env": {**env, **extra}}


s["trials"] = [arm("base", {}), arm("k2", {"ARENA_RC_OVERRIDES": json.dumps({"text_overrides": {"num_experts_per_tok": 2}})}),
               arm("baserep", {"ARC_STEPLOG_REP": "1"})]
s["chain"] = {"trials": [t["name"] for t in s["trials"]], "server_restarts": 2}
s["notebook_seconds"] = s["b200"]["stop"]["wall_seconds"] = 2450 + 3 * WALL + 2 * 1130
s["nebius_assets"] = [a for a in s["nebius_assets"] if "p75-d" not in a]
s["roster"] = "top-k 2 bench: no games; 50 forced decodes on real ~100K contexts, base / k2 / base, one VM"
s["evidence"]["kprofF"] = f"from {SRC.name}: trials, slim guard on drop only, ARC_TOPK read"
p.write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((OUT / "MANIFEST.json").read_text().replace(SRC.name, OUT.name))
m["study"] = OUT.name
m["files"] = {q.relative_to(OUT).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(OUT.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(OUT)
