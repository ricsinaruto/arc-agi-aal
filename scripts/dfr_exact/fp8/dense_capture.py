"""Dense-input capture for the FP8/FP4 dense calibration (throughput arm, 20 Oct; owner item 1).

The three LINEAR_FAMILIES (gdn_in_proj_qkvz, qsa_qkv, attn_out) keep their BF16 weights and run F.linear through an
opaque custom op that hands each call's input rows to vllm.arc_expert_capture.on_dense while a capture runs: the same
/collective_rpc start/stop, stride, format, queue and writer as the MoE capture, files D<layer><in|out>.x (+ .s for
fp8). Layer inputs: "in" = the input-norm output that feeds in_proj_qkvz / qkv_proj, "out" = the 6,144-wide input of
out_proj / o_proj. At most ARC_DENSE_CAPTURE_MAX_ROWS rows per key (default 65,536; a pass's max_rows also caps).
Server env ARC_DENSE_CAPTURE=1 switches it on; without it the bundle serves unchanged.
"""

DENSE_RUNTIME_CODE = r'''

# ----------------------------------------------------------------------------- dense-input capture (throughput arm, 20 Oct)
# ARC_DENSE_CAPTURE=1 (calibration-capture servers only; scripts/dfr_exact/fp8/dense_capture.py).
import os as _arc_cap_os
import re as _arc_cap_re

DENSE_CAPTURE = _arc_cap_os.environ.get("ARC_DENSE_CAPTURE") == "1"
_CAPTURE = {"keys": [], "method": None}


def _arc_capture_linear(x: torch.Tensor, weight: torch.Tensor, key: int) -> torch.Tensor:
    if not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
        from vllm.arc_expert_capture import on_dense

        on_dense(_CAPTURE["keys"][key], x.reshape(-1, x.shape[-1]))
    return F.linear(x, weight)


def _arc_capture_linear_fake(x: torch.Tensor, weight: torch.Tensor, key: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _register_capture_op():
    from vllm.utils.torch_utils import direct_register_custom_op

    if not hasattr(torch.ops.vllm, "arc_capture_linear"):
        direct_register_custom_op(op_name="arc_capture_linear", op_func=_arc_capture_linear,
                                  fake_impl=_arc_capture_linear_fake)


def capture_key(prefix, family):
    m = _arc_cap_re.search(r"layers\.(\d+)\.", prefix or "")
    if m is None:
        raise RuntimeError(f"ARC dense capture: no layer index in {prefix!r}")
    return f"D{int(m.group(1)):02d}{'in' if family in ('gdn_in_proj_qkvz', 'qsa_qkv') else 'out'}"


def capture_method(child, family):
    _CAPTURE["keys"].append(capture_key(getattr(child, "prefix", ""), family))
    if _CAPTURE["method"] is None:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        class CaptureLinearMethod(UnquantizedLinearMethod):
            def __init__(self, family, key_id):
                super().__init__()
                self.family, self.key_id = family, key_id

            def apply(self, layer, x, bias=None):
                y = torch.ops.vllm.arc_capture_linear(x, layer.weight, self.key_id)
                return y if bias is None else y + bias

        _CAPTURE["method"] = CaptureLinearMethod
    return _CAPTURE["method"](family, len(_CAPTURE["keys"]) - 1)
'''

DENSE_LOOP_OLD = """        family = family_of(child, is_head, shape)
"""
DENSE_LOOP_NEW = """        family = family_of(child, is_head, shape)
        if DENSE_CAPTURE and is_linear and family in LINEAR_FAMILIES:
            _register_capture_op()
            child.quant_method = capture_method(child, family)
            counts[f"capture:{family}"] = counts.get(f"capture:{family}", 0) + 1
            continue
"""

CAPTURE_RUNTIME_CODE = r'''

# ----------------------------------------------------------------------------- dense inputs (throughput arm, 20 Oct)
_DENSE_MAX = int(os.environ.get("ARC_DENSE_CAPTURE_MAX_ROWS", "65536"))
# Calibration (fp8) passes may sample dense rows sparser than the MoE capture, so the row cap lands near the end of
# the pass and every sequence contributes (v1 on Nebius filled the 270 GB disk at stride 32 x 65,536 rows).
_DENSE_CALIB_STRIDE = int(os.environ.get("ARC_DENSE_CAPTURE_CALIB_STRIDE", "0"))


def _dense_stride(state):
    return _DENSE_CALIB_STRIDE if (_DENSE_CALIB_STRIDE and state.format == "fp8") else state.stride


def _dense(state):
    return state.__dict__.setdefault("dense", {"counter": {}, "accepted": {}, "buffers": {}, "buffered": {},
                                               "written": {}, "shapes": {}})


def on_dense(key, x):
    """Dense-projection input rows: every stride-th row per key (its own running counter) while a capture runs."""
    state = _STATE
    if state is None or not state.active:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    if x.dim() != 2 or x.shape[0] == 0:
        return
    d = _dense(state)
    n = x.shape[0]
    start = d["counter"].get(key, 0)
    d["counter"][key] = start + n
    stride = _dense_stride(state)
    first = (-start) % stride
    if first >= n:
        return
    cap = _DENSE_MAX if state.max_rows is None else min(_DENSE_MAX, state.max_rows)
    room = cap - d["accepted"].get(key, 0)
    if room <= 0:
        return
    rows = x[first::stride][:room]
    if state.format == "fp8":
        scale = rows.float().abs().amax(dim=1).clamp(min=1e-12) / 448.0
        q = (rows.float() / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
        parts = {"x": q.view(torch.uint8).cpu(), "s": scale.cpu()}
    else:
        parts = {"x": rows.contiguous().view(torch.int16).cpu()}
    d["buffers"].setdefault(key, []).append(parts)
    d["buffered"][key] = d["buffered"].get(key, 0) + rows.shape[0]
    d["accepted"][key] = d["accepted"].get(key, 0) + rows.shape[0]
    d["shapes"].setdefault(key, {"hidden": int(rows.shape[1]), "x_dtype": str(x.dtype)})
    if d["buffered"][key] >= state.flush_rows:
        _dense_enqueue(state, key)


def _dense_enqueue(state, key):
    d = _dense(state)
    chunks = d["buffers"].pop(key, [])
    d["buffered"][key] = 0
    if not chunks:
        return
    size = sum(t.numel() * t.element_size() for c in chunks for t in c.values())
    with state.lock:
        if state.queued_bytes + size > state.max_queued or state.stopped:
            state.dropped_rows += sum(c["x"].shape[0] for c in chunks)
            return
        state.queued_bytes += size
    state.queue.put((("dense", key), chunks, size))
'''

CAPTURE_PATCHES = [
    ("""                for key in chunks[0]:
                    data = torch.cat([c[key] for c in chunks]).numpy().tobytes()
                    with open(os.path.join(state.dir, f"L{layer:02d}.{key}"), "ab") as handle:
                        handle.write(data)
                    state.bytes += len(data)
                state.written[layer] = state.written.get(layer, 0) + sum(c["ids"].shape[0] for c in chunks)
""",
     """                for key in chunks[0]:
                    data = torch.cat([c[key] for c in chunks]).numpy().tobytes()
                    name = f"{layer[1]}.{key}" if isinstance(layer, tuple) else f"L{layer:02d}.{key}"
                    with open(os.path.join(state.dir, name), "ab") as handle:
                        handle.write(data)
                    state.bytes += len(data)
                if isinstance(layer, tuple):  # dense inputs (on_dense)
                    written = _dense(state)["written"]
                    written[layer[1]] = written.get(layer[1], 0) + sum(c["x"].shape[0] for c in chunks)
                else:
                    state.written[layer] = state.written.get(layer, 0) + sum(c["ids"].shape[0] for c in chunks)
"""),
    ("""            for layer in list(state.buffers):
                _enqueue(state, layer)
""",
     """            for layer in list(state.buffers):
                _enqueue(state, layer)
            for key in list(_dense(state)["buffers"]):
                _dense_enqueue(state, key)
"""),
    ("""            meta = {**_summary(state), "shapes": {str(k): v for k, v in sorted(state.shapes.items())}}
""",
     """            meta = {**_summary(state), "shapes": {str(k): v for k, v in sorted(state.shapes.items())}}
            d = _dense(state)
            meta["dense"] = {"rows": dict(sorted(d["written"].items())), "shapes": dict(sorted(d["shapes"].items())),
                             "processed_tokens": dict(sorted(d["counter"].items())), "max_rows": _DENSE_MAX}
"""),
]


PERSIST_OLD = """            shutil.copytree(source, target, dirs_exist_ok=True)
            state.update(target=str(target), seconds=round(time.monotonic() - started, 1))"""
PERSIST_NEW = """            if not target.exists() and os.stat(source).st_dev == os.stat(root).st_dev:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.rename(source, target)  # one filesystem (Nebius): no second copy of the capture on disk
                state.update(moved=True)
            else:
                shutil.copytree(source, target, dirs_exist_ok=True)
            state.update(target=str(target), seconds=round(time.monotonic() - started, 1))"""


def patch_persist(expert_quant_path):
    """expert_quant.persist_async moves the capture when source and output share a filesystem."""
    text = expert_quant_path.read_text()
    if text.count(PERSIST_OLD) != 1:
        raise SystemExit("expert_quant persist anchor not found once")
    expert_quant_path.write_text(text.replace(PERSIST_OLD, PERSIST_NEW))


def apply(src):
    """Patch a bundle's src tree (pathlib.Path to <bundle>/src) in place; returns the patched file names."""
    dense = src / "arc_infra/dense_kernels_runtime.py"
    text = dense.read_text()
    if text.count(DENSE_LOOP_OLD) != 1 or "DENSE_CAPTURE" in text:
        raise SystemExit("dense runtime anchor not found once (or already patched)")
    dense.write_text(text.replace(DENSE_LOOP_OLD, DENSE_LOOP_NEW) + DENSE_RUNTIME_CODE)
    cap = src / "arc_infra/expert_capture_runtime.py"
    text = cap.read_text()
    for old, new in CAPTURE_PATCHES:
        if text.count(old) != 1:
            raise SystemExit(f"capture runtime anchor not found once: {old[:60]!r}")
        text = text.replace(old, new)
    cap.write_text(text + CAPTURE_RUNTIME_CODE)
    for path in (dense, cap):
        compile(path.read_text(), str(path), "exec")
    return [str(dense.relative_to(src)), str(cap.relative_to(src))]
