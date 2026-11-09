"""Offline check of a bundle's serving studies through arena.serving.main up to the GPU (python -I serving_check.py
BUNDLE; build.py runs it on every bundle): the bootstrap's configs/comparison.json, gen 0's stack study as
serving.Stacks.study_for composes it, and a candidate that lowers its sequences. 13 Oct: v10 died at server start
on a check this runs (max_num_seqs below generation_slots), v12 on the trial's ARENA_STALL_DUMP without a prune mask;
the e2e's fake stacks never start a server."""
import json, sys, tempfile, types
from pathlib import Path

bundle = Path(sys.argv[1]).resolve()
# --serving-src: import the server code from rsi/serving-src, as an unprivileged stack does (/serve/src)
SERVE = "--serving-src" in sys.argv
sys.path[:0] = [str(bundle / "rsi/serving-src")] if SERVE else [str(bundle / "src")]
import arena.serving as S  # noqa: E402

PREFIX_CACHE_OPT_IN = {"mamba_align_fix": True, "mamba_keep_state": True, "mamba_demote_superseded": True,
                       "request_end_tail": True, "spec_decode": True, "tail_keep_boundary": True,
                       "keep_reply_states": True, "tail_spec_acceptance": True, "reply_ids": True,
                       "tail_keepalive": True, "state_cotenant": True, "admission_tick": 2.0}


class Recipe:
    RUNTIME_ROOT = CACHE_ROOT = Path("/nonexistent")
    MAX_CUDAGRAPH_CAPTURE_SIZE_ENV = "X_CAPTURE"

    def __getattr__(self, name):
        return lambda *a, **k: None


def fake_start(working, input_root, *, recipe_hook=None, **kw):
    seen.update(kw)
    u = kw.get("gpu_memory_utilization")
    assert kw["mtp_tokens"] in (0, 1, 2, 3, 4), "mtp"
    assert u is None or (isinstance(u, str) and 0.5 <= float(u) <= 0.99), "utilization"
    recipe_hook(Recipe(), working)  # configure(): every server-block validation


S.start = fake_start
_configure = S.configure


def configure(recipe, working, **kw):
    seen["prefix_cache"] = kw.get("prefix_cache")
    return _configure(recipe, working, **kw)


S.configure = configure
S.BlockBudget = types.SimpleNamespace(load=lambda *a: types.SimpleNamespace(layout={}))


def check(study, label):
    global seen
    seen = {}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "configs/comparison.json"
        path.parent.mkdir()
        path.write_text(json.dumps(study))
        Path(tmp, "w").mkdir()
        sys.argv = ["serving", "--study", str(path), "--working", tmp + "/w", "--input-root", "/kaggle/input"]
        S.main()
    print(label, "OK", {"generation_slots": study["generation_slots"], "max_num_seqs": seen.get("generation_slots"),
                        "mtp": seen.get("mtp_tokens"), "batched": seen.get("max_num_batched_tokens")})


base = json.loads((bundle / "configs/comparison.json").read_text())
import os  # noqa: E402

kept = dict(os.environ)
os.environ.update({str(k): str(v) for k, v in ((base.get("trials") or [{}])[0].get("server_env") or {}).items()})
check(base, "bootstrap comparison.json + first trial's server_env")  # as bootstrap_arena.start_server runs it
os.environ.clear()
os.environ.update(kept)
sys.path.append(str(bundle / "src"))
from rsi_loop.serving import fit_slots  # noqa: E402
assert not SERVE or S.__file__.startswith(str(bundle / "rsi/serving-src")), S.__file__

gen0 = bundle / "rsi/gen0/tree/serving"
study = json.loads(json.dumps(base))
if (gen0 / "study.json").exists():
    study.update(json.loads((gen0 / "study.json").read_text()))
study["server"] = json.loads((gen0 / "server.json").read_text())
check(fit_slots(study), "gen 0 stack (study_for)")
low = json.loads(json.dumps(study))
low["server"]["max_num_seqs"] = 8
low["server"]["max_cudagraph_capture_size"] = 8
low["server"]["cudagraph_capture_sizes"] = [2, 4, 8]
check(fit_slots(low), "candidate lowering max_num_seqs to 8")
# the documented opt-in modules (loop facts): our prefix-cache patch set through env.json, and more sequences
os.environ["ARENA_PREFIX_CACHE"] = json.dumps(PREFIX_CACHE_OPT_IN)
wide = json.loads(json.dumps(study))
wide["server"]["max_num_seqs"] = 52
wide["server"]["max_cudagraph_capture_size"] = 104
wide["server"]["cudagraph_capture_sizes"] = [2, 4, 8, 16, 32, 48, 64, 80, 96, 104]
check(fit_slots(wide), "candidate opting in: prefix-cache patches + 52 sequences")
assert seen.get("prefix_cache") == PREFIX_CACHE_OPT_IN, seen.get("prefix_cache")
os.environ.pop("ARENA_PREFIX_CACHE")
