"""Golden (rl-min D1, 28 Oct): bytes the model or the server sees that the other goldens miss. The sandbox child code
(its line numbers reach the model through tracebacks), the serving argv + configure() result, and the prefix-cache
patch's output files and identity marker on synthetic pinned files, for each live config."""
import hashlib
import json
import tempfile
import types
from pathlib import Path

from common import B, check, config

LIVE = ("rl1-loop-seq-real-v9", "rl1-loop-seq-real-v10", "b2-rl-hread-rd-v1")


def h(text):
    return hashlib.sha256(text if isinstance(text, bytes) else text.encode()).hexdigest()


def test_sandbox_child_code():
    from arena import dfr_harness, hx_harness
    from inference.agent.python_tool_sandbox import _SANDBOX_BOOTSTRAP

    got = {}
    for run in LIVE:
        study = B.inline(config(run)["study"])
        text = hx_harness.with_toolbox(dfr_harness.with_dfr(_SANDBOX_BOOTSTRAP), hx_harness.settings(study["hx_harness"]))
        got[run] = {"sha": h(text), "lines": len(text.splitlines())}
    check("serving_sandbox", got)


def test_serving_argv_and_configure(monkeypatch):
    from arena import serving as S

    monkeypatch.setenv("ARENA_READY_GUARD", "180,3000")
    monkeypatch.setenv("ARENA_SETUP_TIMEOUTS", "2700,3600")
    monkeypatch.setenv("ARENA_PROMPT_TOKENS_DETAILS", "1")
    got = {}
    for run in LIVE:
        study = B.inline(config(run)["study"])
        server, ce = study["server"], study["cache_experiment"]
        spec = json.dumps({"num_speculative_tokens": server["mtp_tokens"]})
        recipe = types.SimpleNamespace(
            verify_and_extract_runtime=lambda *a, **k: None, MAX_CUDAGRAPH_CAPTURE_SIZE_ENV="X_CAP",
            SERVER_READY_TIMEOUT=1500, SETUP_TOTAL_TIMEOUT=1800, SERVED_MODEL_NAME="m",
            runtime_environment=lambda *a, **k: ({}, {}),
            server_command=lambda *a, _s=spec, **k: ["vllm", "serve", "m", "--speculative-config", _s, "--max-num-seqs", "288"])
        lab = S.lab_overrides()
        res = S.configure(
            recipe, "/tmp/x", cpu_gib=ce["cpu_cache_gib"], graph_capture_size=server.get("max_cudagraph_capture_size"),
            graph_capture_sizes=server.get("cudagraph_capture_sizes"),
            reclaim_skipped_blocks=ce.get("reclaim_skipped_blocks", False), serving_kernels=server.get("serving_kernels"),
            served_route=server.get("served_route"), serving_fast=server.get("serving_fast"),
            expert_mask=("/x/mask.json", "ab") if server.get("expert_mask") else None, prefix_cache=ce.get("prefix_cache"),
            prompt_tokens_details=lab.get("prompt_tokens_details", server.get("prompt_tokens_details", False)),
            prefix_match_unit=server.get("prefix_match_unit"), renderer_num_workers=server.get("renderer_num_workers"),
            mm_processor_cache_gb=server.get("mm_processor_cache_gb"), lora=server.get("lora"),
            ready_guard_seconds=lab.get("ready_guard"), setup_timeouts=lab.get("setup_timeouts"))
        got[run] = {"argv": recipe.server_command("m"), "configure": res}
    check("serving_argv", got)


FILES = {
    "SCHEDULER": ("vllm/v1/core/sched/scheduler.py", ["ALIGN_ANCHOR", "ACCEPT_ANCHOR", "DROP_ANCHOR", "TICK_INIT_ANCHOR",
                                                     "TICK_WAIT_ANCHOR", "TICK_END_ANCHOR"]),
    "MANAGER": ("vllm/v1/core/single_type_kv_cache_manager.py", ["CLASS_ANCHOR", "KEEP_INIT_ANCHOR", "KEEP_REMOVE_ANCHOR",
                                                                "KEEP_HELPER_ANCHOR", "KEEP_POP_ANCHOR"]),
    "POOL": ("vllm/v1/core/block_pool.py", ["DEMOTE_ANCHOR"]),
    "KV_MANAGER": ("vllm/v1/core/kv_cache_manager.py", ["TAIL_INIT_ANCHOR", "TAIL_LOOKUP_ANCHOR", "TAIL_FREE_ANCHOR",
                                                       "TAIL_POP_ANCHOR"]),
    "COORDINATOR": ("vllm/v1/core/kv_cache_coordinator.py", ["EAGLE_DROP_ANCHOR", "EAGLE_GROUP_ANCHOR"]),
    "CHAT_SERVING": ("vllm/entrypoints/openai/chat_completion/serving.py", ["REPLY_IMPORT_ANCHOR", "REPLY_RENDER_ANCHOR",
                                                                           "REPLY_FINISH_ANCHOR"]),
}


def test_prefix_cache_patch_output(monkeypatch):
    """The patch applied to synthetic files made of its own anchors (sha guards pointed at them, compile stubbed): every
    patched file, the identity marker (unchanged on a restart in the same dir) and validate()'s result."""
    from arena import prefix_cache_patch as P

    monkeypatch.setattr(P, "compile", lambda *a, **k: None, raising=False)
    texts = {}
    for key, (rel, anchors) in FILES.items():
        texts[key] = ("".join(f"# filler {i}\n" + getattr(P, a) for i, a in enumerate(anchors)) + "# end\n").encode()
        monkeypatch.setattr(P, f"{key}_SHA256", h(texts[key]))
    got = {}
    for run in LIVE:
        cfg = B.inline(config(run)["study"])["cache_experiment"]["prefix_cache"]
        site, work = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        for key, (rel, _) in FILES.items():
            (site / rel).parent.mkdir(parents=True, exist_ok=True)
            (site / rel).write_bytes(texts[key])
        ident = P.apply(site, work, dict(cfg))
        assert P.apply(site, work, dict(cfg)) == ident
        got[run] = {"files": {str(p.relative_to(site)): h(p.read_bytes()) for p in sorted(site.rglob("*.py"))},
                    "marker": h((work / P.MARKER).read_text()), "validate": P.validate(dict(cfg))}
    check("serving_prefix_cache", got)
