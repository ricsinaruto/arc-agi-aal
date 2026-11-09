"""l4 debug (v49dbg2): appended to arc2/l4_logits.py in the bundle. ARC2_L4_FLUSH_EVERY=N: every N engine steps a
worker returns its free cached blocks to the driver (torch.cuda.empty_cache), as the allocator does on a failed
allocation. Draft runs died only right after such a mid-play release (A twice; C, B and the debug run had none), so a
forced release with synchronous launches should name the kernel that still uses freed memory."""


def _arc2_flush_hook():
    import os
    every = int(os.environ.get("ARC2_L4_FLUSH_EVERY") or 0)
    if every <= 0:
        return False
    import torch
    from vllm.v1.worker import gpu_worker

    cls = gpu_worker.Worker
    if getattr(cls, "_arc2_flush", None):
        return False
    original = cls.execute_model
    state = {"steps": 0, "flushes": 0}

    def execute_model(self, *args, **kwargs):
        try:
            out = original(self, *args, **kwargs)
        except BaseException:  # the full worker traceback, to a file the kernel returns (the logs keep only error lines)
            import traceback
            folder = os.environ.get("ARC2_L4_TRACE_DIR") or "/kaggle/working/arc2"
            try:
                os.makedirs(folder, exist_ok=True)
                with open(os.path.join(folder, f"l4-worker-trace-{os.getpid()}.txt"), "a") as f:
                    f.write(json.dumps(state) + "\n" + traceback.format_exc() + "\n")
            except OSError:
                pass
            raise
        state["steps"] += 1
        if state["steps"] % every == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            state["flushes"] += 1
            if state["flushes"] in (1, 10, 100, 1000):
                print("ARC2_L4_FLUSH " + json.dumps({**state, "pid": os.getpid()}), flush=True)
        return out

    cls.execute_model = execute_model
    cls._arc2_flush = every
    print("ARC2_L4_FLUSH " + json.dumps({"installed": every, "pid": os.getpid()}), flush=True)
    return True


def _arc2_draft_eager():
    """ARC2_L4_DRAFT_EAGER=1 (debug bisect): the draft's prefill and decode run without CUDA graphs."""
    import os
    if os.environ.get("ARC2_L4_DRAFT_EAGER") != "1":
        return False
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator

    cls = speculator.AutoRegressiveSpeculator
    if getattr(cls, "_arc2_eager", False):
        return False
    original = cls.init_cudagraph_manager

    def init_cudagraph_manager(self, cudagraph_mode):
        return original(self, CUDAGraphMode.NONE)

    cls.init_cudagraph_manager = init_cudagraph_manager
    cls._arc2_eager = True
    print("ARC2_L4_DRAFT_EAGER " + json.dumps({"installed": True, "pid": os.getpid()}), flush=True)
    return True


def _arc2_stale_scan():
    """ARC2_L4_STALE_SCAN=1 (debug): name the attention-metadata tensors a CUDA graph captured and the allocator has
    freed by the end of that manager's capture (the graph keeps their addresses). Writes l4-stale-<pid>.json."""
    import os
    if os.environ.get("ARC2_L4_STALE_SCAN") != "1":
        return False
    import dataclasses
    import torch
    from vllm.v1.worker.gpu import cudagraph_utils as cgu
    from vllm.v1.worker.gpu.spec_decode.autoregressive import cudagraph_utils as acgu

    if getattr(cgu.CudaGraphManager, "_arc2_scan", False):
        return False
    state = {"last": [], "captured": [], "reports": []}

    def walk(obj, path, out, depth=0):
        if depth > 6:
            return
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda and obj.numel():
                try:
                    from torch.multiprocessing.reductions import StorageWeakRef
                    ref = StorageWeakRef(obj.untyped_storage())
                except Exception:  # noqa: BLE001
                    ref = None
                out.append((path, obj.data_ptr(), obj.numel() * obj.element_size(), str(obj.dtype), list(obj.shape),
                            ref))
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            for f in dataclasses.fields(obj):
                walk(getattr(obj, f.name, None), f"{path}.{f.name}", out, depth + 1)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}[{k}]", out, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]", out, depth + 1)
        elif hasattr(obj, "__dict__") and type(obj).__module__.startswith("vllm"):
            for k, v in vars(obj).items():
                walk(v, f"{path}.{k}", out, depth + 1)

    def wrap_prepare(module):
        original = module.prepare_inputs_to_capture

        def prepare_inputs_to_capture(*args, **kwargs):
            result = original(*args, **kwargs)
            out = []
            walk(result, "meta", out)
            state["last"] = out
            return result

        module.prepare_inputs_to_capture = prepare_inputs_to_capture

    wrap_prepare(cgu)
    wrap_prepare(acgu)
    graph_cls = torch.cuda.graphs.CUDAGraph
    begin = graph_cls.capture_begin

    def capture_begin(self, *args, **kwargs):
        state["captured"].extend(state["last"])  # the metadata built just before this capture
        if state.get("draft_capture") and os.environ.get("ARC2_L4_GRAPH_DUMP") == "1":
            self.enable_debug_mode()  # kernel node params (pointers) for debug_dump
        return begin(self, *args, **kwargs)

    graph_cls.capture_begin = capture_begin

    def blocks():
        for seg in torch.cuda.memory_snapshot():
            addr = seg["address"]
            for b in seg["blocks"]:
                start = b.get("address", addr)
                yield start, b["size"], b["state"], seg.get("segment_pool_id")
                addr = start + b["size"]

    original_capture = cgu.CudaGraphManager.capture

    def freed_in_window(snap):
        """Blocks allocated and freed while this manager captured, grouped by allocation stack (vLLM/arc frames)."""
        groups = {}
        for trace in snap.get("device_traces", []):
            live = {}
            for e in trace:
                act, addr = e.get("action"), e.get("addr")
                if act == "alloc":
                    live[addr] = e
                elif act in ("free_requested", "free_completed") and addr in live:
                    a = live.pop(addr)
                    frames = [f"{f.get('filename', '').split('dist-packages/')[-1].split('/src/')[-1]}:{f.get('line')}"
                              f" {f.get('name')}" for f in a.get("frames", [])
                              if any(k in f.get("filename", "") for k in ("vllm", "arc2", "arc_infra", "arena"))][:8]
                    key = " | ".join(frames) or "<no python frames>"
                    g = groups.setdefault(key, {"count": 0, "bytes": 0, "max": 0})
                    g["count"] += 1
                    g["bytes"] += a.get("size", 0)
                    g["max"] = max(g["max"], a.get("size", 0))
        return sorted(({"stack": k, **v} for k, v in groups.items()), key=lambda r: -r["bytes"])

    def dump_graphs(manager, snap):
        """One draft graph's DOT file (kernel params carry its device pointers) and the capture window's allocations
        that ended freed inside a default-pool segment, with address, size and vLLM/arc frames. ~100 MB cap."""
        dist = getattr(torch, "distributed", None)
        if dist is not None and dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return  # rank 0 only
        folder = os.environ.get("ARC2_L4_TRACE_DIR") or "/kaggle/working/arc2"
        os.makedirs(folder, exist_ok=True)
        pid, cap = os.getpid(), 80 * 2**20
        graphs = list(getattr(manager, "graphs", {}).items())
        if graphs:
            dot = os.path.join(folder, f"l4-draft-graph-{pid}.dot")
            graphs[-1][1].debug_dump(dot)
            if os.path.getsize(dot) > cap:
                os.remove(dot)
        default = [(seg["address"], seg["address"] + seg["total_size"]) for seg in snap.get("segments", [])
                   if tuple(seg.get("segment_pool_id") or (0, 0)) == (0, 0)]
        rows, stacks = [], {}
        for trace in snap.get("device_traces", []):
            live = {}
            for e in trace:
                act, addr = e.get("action"), e.get("addr")
                if act == "alloc":
                    live[addr] = e
                elif act in ("free_requested", "free_completed") and addr in live:
                    a = live.pop(addr)
                    if not any(lo <= addr < hi for lo, hi in default):
                        continue
                    frames = " | ".join(f"{f.get('filename', '').split('dist-packages/')[-1]}:{f.get('line')} {f.get('name')}"
                                        for f in a.get("frames", [])
                                        if any(k in f.get("filename", "") for k in ("vllm", "arc2", "arc_infra")))[:600]
                    rows.append([addr, a.get("size", 0), stacks.setdefault(frames, len(stacks))])
        text = json.dumps({"graph": str(graphs[-1][0]) if graphs else None, "freed_default_pool": rows[:300000],
                           "stacks": {v: k for k, v in stacks.items()}})
        if len(text) <= cap:
            with open(os.path.join(folder, f"l4-draft-allocs-{pid}.json"), "w") as f:
                f.write(text)

    def capture(self, *args, **kwargs):
        state["captured"] = []
        draft = type(self).__name__ == "SpeculatorCudaGraphManager"
        if draft:
            torch.cuda.memory._record_memory_history(max_entries=200000, stacks="python")
            state["draft_capture"] = True
        try:
            result = original_capture(self, *args, **kwargs)
        finally:
            state["draft_capture"] = False
        torch.cuda.synchronize()
        if draft:
            try:
                snap = torch.cuda.memory._snapshot()
                state["freed"] = freed_in_window(snap)
                if os.environ.get("ARC2_L4_GRAPH_DUMP") == "1":
                    dump_graphs(self, snap)
            finally:
                torch.cuda.memory._record_memory_history(enabled=None)
        table = list(blocks())
        rows, seen = [], set()
        expired = {}
        for path, ptr, nbytes, dtype, shape, ref in state["captured"]:
            key = (path, ptr)
            if key in seen:
                continue
            seen.add(key)
            gone = ref is not None and ref.expired()
            if gone:  # the storage a captured graph points into has been freed (whatever reuses it now)
                field = path.split("]")[-1] or path
                layer = path.split("[")[2].split("]")[0] if path.count("[") >= 2 else ""
                kind = "mtp" if "mtp" in layer else ("full" if "self_attn" in layer or "attn" == layer[-4:] else layer.split(".")[-1])
                expired.setdefault(f"{kind}:{field}", [0, nbytes, dtype, shape])[0] += 1
            hit = next(((s, z, st, pool) for s, z, st, pool in table if s <= ptr < s + z), None)
            st = hit[2] if hit else "not_in_torch_pool"
            if st != "active_allocated":
                rows.append({"path": path, "ptr": ptr, "bytes": nbytes, "dtype": dtype, "shape": shape, "state": st})
        state["expired"] = expired
        keys = sorted({p.split("[")[2].split("]")[0] for p, *_ in state["captured"] if p.count("[") >= 2})
        report = {"manager": type(self).__name__, "captured_tensors": len(state["captured"]),
                  "expired_storage": state.pop("expired", {}), "layer_keys_mtp": [k for k in keys if "mtp" in k],
                  "layer_key_kinds": sorted({k.split(".")[-1] for k in keys}),
                  "stale": rows[:200], "stale_paths": sorted({r["path"] for r in rows})}
        if draft:
            report["freed_during_capture"] = state.pop("freed", [])[:300]
        state["reports"].append(report)
        print("ARC2_L4_STALE " + json.dumps({k: report[k] for k in ("manager", "captured_tensors", "expired_storage",
                                                                 "layer_keys_mtp")}), flush=True)
        folder = os.environ.get("ARC2_L4_TRACE_DIR") or "/kaggle/working/arc2"
        try:
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, f"l4-stale-{os.getpid()}.json"), "w") as f:
                json.dump(state["reports"], f, indent=1)
        except OSError:
            pass
        return result

    cgu.CudaGraphManager.capture = capture
    cgu.CudaGraphManager._arc2_scan = True
    print("ARC2_L4_STALE " + json.dumps({"installed": True, "pid": os.getpid()}), flush=True)
    return True


_arc2_register_topk = register


def register():  # noqa: F811  (the topk plugin, then the flush hook)
    _arc2_register_topk()
    for hook in (_arc2_flush_hook, _arc2_draft_eager, _arc2_stale_scan):
        try:
            hook()
        except Exception as exc:  # noqa: BLE001  (a debug hook never stops a start)
            print("ARC2_L4_DEBUG_HOOK " + json.dumps({"hook": hook.__name__, "reason": repr(exc)}), flush=True)
