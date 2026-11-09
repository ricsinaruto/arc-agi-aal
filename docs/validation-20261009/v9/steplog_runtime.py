"""Per-step log for kprofR (bench bundles only; appended to the bundle's gdn_defer_runtime.py, so it runs in the worker).
ARC_STEPLOG=path: one JSON row per model step: which graph path it took, batch shape, CPU time in the path, graph and Triton
calls. Every ARC_STEPLOG_SYNC-th step drains the GPU around the forward (cpu_ms to the call's return, gpu_tail_ms after it)."""


def install_steplog():
    import json as _json
    import os as _os
    import time as _time

    path = _os.environ.get("ARC_STEPLOG")
    if not path or globals().get("_ARC_STEPLOG_ON"):
        return
    globals()["_ARC_STEPLOG_ON"] = True
    import torch as _torch

    from vllm.v1.worker.gpu import cudagraph_utils as cg
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as Runner

    every = int(_os.environ.get("ARC_STEPLOG_SYNC", "16"))
    state = {"n": 0, "row": None, "buffer": [], "end": None}
    clock = _time.perf_counter
    handle = open(path, "a", buffering=1)

    def timed(owner, name, key, count=None):
        pinned = getattr(owner, name)

        def wrapper(*args, **kwargs):
            start = clock()
            try:
                return pinned(*args, **kwargs)
            finally:
                row = state["row"]
                if row is not None:
                    row[key] = row.get(key, 0.0) + 1000 * (clock() - start)
                    if count:
                        row[count] = row.get(count, 0) + 1

        setattr(owner, name, wrapper)

    def forward(owner, name, label):
        pinned = getattr(owner, name)

        def wrapper(self, *args, **kwargs):
            row = state["row"]
            key = "main" if isinstance(self, cg.ModelCudaGraphManager) else "spec"
            drain = row is not None and row.get("drain") and key == "main" and label != "fullbase"
            if drain:
                t0 = clock()
                _torch.cuda.synchronize()
                row["pre_sync_ms"] = 1000 * (clock() - t0)
            start = clock()
            try:
                return pinned(self, *args, **kwargs)
            finally:
                if row is not None:
                    done = clock()
                    row[f"{key}_{label}_ms"] = row.get(f"{key}_{label}_ms", 0.0) + 1000 * (done - start)
                    if drain:
                        _torch.cuda.synchronize()
                        row["gpu_tail_ms"] = 1000 * (clock() - done)

        setattr(owner, name, wrapper)

    pinned_dispatch = cg.CudaGraphManager.dispatch

    def dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=None):
        desc = pinned_dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len)
        row = state["row"]
        if row is not None:
            key = "main" if isinstance(self, cg.ModelCudaGraphManager) else "spec"
            row[f"{key}_mode"] = desc.cg_mode.name
            row[f"{key}_desc_tokens"] = desc.num_tokens
            if key == "main":
                row.update(reqs=num_reqs, tokens=num_tokens, uniform=uniform_token_count, max_query=max_query_len)
        return desc

    cg.CudaGraphManager.dispatch = dispatch
    forward(cg.CudaGraphManager, "run_pw_graph", "pw")
    forward(cg.ModelCudaGraphManager, "run_fullgraph", "full")
    forward(cg.CudaGraphManager, "run_fullgraph", "fullbase")

    pinned_gather = Runner.gather_batch_req_state

    def gather(self, scheduler_output, dummy_run):
        out = pinned_gather(self, scheduler_output, dummy_run)
        row, batch = state["row"], out[0]
        if row is not None and batch is not None:
            mask = batch.is_prefilling_np
            row["prefill_reqs"] = int(mask.sum())
            row["prefill_tokens"] = int(batch.num_scheduled_tokens[mask].sum())
        return out

    Runner.gather_batch_req_state = gather
    pinned_execute = Runner.execute_model

    def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False, **kwargs):
        if dummy_run:
            return pinned_execute(self, scheduler_output, intermediate_tensors, True, **kwargs)
        state["n"] += 1
        start = clock()
        row = state["row"] = {"t": round(_time.time(), 3), "n": state["n"], "drain": state["n"] % every == 0}
        if state["end"] is not None:
            row["gap_ms"] = 1000 * (start - state["end"])
        try:
            return pinned_execute(self, scheduler_output, intermediate_tensors, False, **kwargs)
        finally:
            state["end"] = clock()
            row["exec_ms"] = 1000 * (state["end"] - start)
            state["row"] = None
            state["last"] = row
            state["buffer"].append(row)

    Runner.execute_model = execute_model
    pinned_sample = Runner.sample_tokens

    def sample_tokens(self, grammar_output):
        start = clock()
        try:
            return pinned_sample(self, grammar_output)
        finally:
            last = state.get("last")
            if last is not None:
                last["sample_ms"] = last.get("sample_ms", 0.0) + 1000 * (clock() - start)
            if len(state["buffer"]) >= 257:  # the newest row may still gain sample_ms
                keep = state["buffer"].pop()
                handle.write("".join(_json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()},
                                                 separators=(",", ":")) + "\n" for r in state["buffer"]))
                state["buffer"] = [keep]

    Runner.sample_tokens = sample_tokens
    try:
        import torch.cuda.graphs as graphs

        timed(graphs.CUDAGraph, "replay", "replay_ms", "replay_n")
    except Exception as exc:  # noqa: BLE001 - counts only
        print(f"ARC_STEPLOG no replay count: {exc!r}", flush=True)
    try:
        from triton.runtime import jit

        timed(jit.JITFunction, "run", "jit_ms", "jit_n")
    except Exception as exc:  # noqa: BLE001 - counts only
        print(f"ARC_STEPLOG no triton count: {exc!r}", flush=True)
    import hashlib

    shas = {m.__name__.rsplit(".", 1)[1]: hashlib.sha256(open(m.__file__, "rb").read()).hexdigest()[:16]
            for m in (cg, __import__("sys").modules[Runner.__module__])}
    print(f"ARC_STEPLOG installed path={path} sync_every={every} pid={_os.getpid()} sha={shas}", flush=True)
