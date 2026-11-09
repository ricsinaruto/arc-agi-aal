"""FP8 W8A8 dense projections (throughput arm, 19 Oct): code appended to the bundle's dense_kernels_runtime.py and
the patches that wire it in (builder --fp8-dense)."""

RUNTIME_CODE = r'''

# ----------------------------------------------------------------------------- FP8 W8A8 (throughput arm, 19 Oct)
# ``fp8_w8a8``: FP8 E4M3 weights, one scale per output channel (amax / 448, from the checkpoint weights alone), and
# per-token dynamic FP8 activations (vLLM scaled_fp8_quant); cutlass_scaled_mm with BF16 output at every M. Nothing is
# fitted on data. ARC_FP8_TOGGLE=<path> (quality screens, eager only): the BF16 weights stay resident and the BF16
# path (F.linear) runs while <path> exists, so one server scores both arms.
import os as _arc_os

FP8_FAMILIES = tuple(LINEAR_FAMILIES)
FP8_MAX = 448.0
FP8_ERROR_LIMIT = 0.08
FP8_TOGGLE = _arc_os.environ.get("ARC_FP8_TOGGLE") or ""
# ARC_FP8_CALIB=<dir>: calibrated codes and per-row scales (fp8/fp8_calib.py) replace the
# amax/448 rounding; L<layer>-<in|out>.safetensors, rows concatenated in the file's "order" (vLLM's merged layout),
# checked row by row against the served BF16 weight before use.
FP8_CALIB = _arc_os.environ.get("ARC_FP8_CALIB") or ""
# The served MTP layer (48) is the retrained draft head, not the checkpoint the fit used: plain rounding there.
FP8_CALIB_SKIP = {int(v) for v in (_arc_os.environ.get("ARC_FP8_CALIB_SKIP") or "48").split(",") if v}
import re as _arc_re
FP8_CALIB_ROW_MAX, FP8_CALIB_ROW_MEAN = 0.25, 0.06
_FP8 = {"families": None, "method": None}
# ARC_FP8_W16=1 (A1, 23 Oct): weight-only FP8 (W8A16) through vLLM's Marlin FP8 GEMM; activations stay BF16, so the
# activation-side error of W8A8 (the measured loss) is gone; same codes and per-row scales as the W8A8 path.
FP8_W16 = _arc_os.environ.get("ARC_FP8_W16") == "1"
W16_ERROR_LIMIT = 0.02


def w16_prepare(layer, q, s):
    """Repack (codes [N, K], scales [N, 1]) into Marlin FP8 layout on ``layer``; returns (N, K)."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin

    n, k = q.shape
    if (getattr(layer, "output_size_per_partition", n), getattr(layer, "input_size_per_partition", k)) != (n, k):
        raise RuntimeError(f"ARC FP8 W16: partition sizes differ from the weight {(n, k)}")
    layer.output_size_per_partition, layer.input_size_per_partition = n, k
    layer.orig_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(q, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(s.reshape(n).contiguous(), requires_grad=False)
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    return n, k


def _w16_gemm(layer, x, bias=None):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear

    n, k = layer.arc_w16
    return apply_fp8_marlin_linear(input=x, weight=layer.weight, weight_scale=layer.weight_scale,
                                   workspace=layer.workspace, size_n=n, size_k=k, bias=bias)


def _w16_time(fn, reps=50):
    for _ in range(5):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return round(start.elapsed_time(end) * 1000 / reps, 1)


def check_w16(family, prefix, layer, q, s, weight):
    """Marlin W8A16 output vs FP32 x @ (codes * scale)^T on the layer's own weights (layout check), once per family;
    with GEMM times (us) of Marlin W8A16 vs the BF16 F.linear on the same weight."""
    key = ("w16", family)
    if key in _STATE["checked"]:
        return
    _STATE["checked"].add(key)
    deq = q.float() * s.float()
    generator = torch.Generator(device=q.device).manual_seed(0)
    result = {"family": family, "prefix": prefix, "shape": list(q.shape)}
    for m in (1, 16, 80, 192, 2048):
        x = torch.randn((m, q.shape[1]), generator=generator, device=q.device, dtype=torch.float32).to(torch.bfloat16)
        ref = x.float() @ deq.t()
        result[f"M{m}"] = _rel(_w16_gemm(layer, x), ref)
        result[f"us{m}"] = {"w16": _w16_time(lambda: _w16_gemm(layer, x)), "bf16": _w16_time(lambda: F.linear(x, weight))}
        if result[f"M{m}"] > W16_ERROR_LIMIT:
            log("CHECK", {**result, "failed": True})
            raise RuntimeError(f"ARC FP8 W16 self-check failed for {family}: {result}")
    log("CHECK", result)


def load_calibrated(prefix, family, weight):
    """(codes [N, K] float8, scales [N, 1] float32, check) of a layer from FP8_CALIB."""
    import re as _re
    from pathlib import Path as _Path

    from safetensors import safe_open

    m = _re.search(r"layers\.(\d+)\.", prefix or "")
    if m is None:
        raise RuntimeError(f"ARC FP8 calib: no layer index in {prefix!r}")
    side = "in" if family in ("gdn_in_proj_qkvz", "qsa_qkv") else "out"
    path = _Path(FP8_CALIB) / f"L{int(m.group(1)):02d}-{side}.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as f:
        order = json.loads(f.metadata()["order"])
        codes = torch.cat([f.get_tensor(n) for n in order]).to(weight.device)
        scale = torch.cat([f.get_tensor(n + "_scale") for n in order]).float().unsqueeze(1).to(weight.device)
    if tuple(codes.shape) != tuple(weight.shape) or codes.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"ARC FP8 calib {path.name}: {codes.dtype} {tuple(codes.shape)} for {tuple(weight.shape)}")
    w = weight.float()
    err = (codes.float() * scale - w).norm(dim=1) / w.norm(dim=1).clamp(min=1e-12)
    check = {"file": path.name, "order": order, "row_err_mean": float(err.mean()), "row_err_max": float(err.max())}
    if check["row_err_max"] > FP8_CALIB_ROW_MAX or check["row_err_mean"] > FP8_CALIB_ROW_MEAN:
        raise RuntimeError(f"ARC FP8 calib does not match the served weight (layout?): {check}")
    return codes.contiguous(), scale.contiguous(), check


def fp8_families():
    if _FP8["families"] is None:
        raw = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}
        fams = list(raw.get("fp8_w8a8") or [])
        unknown = sorted(set(fams) - set(FP8_FAMILIES))
        if unknown:
            raise RuntimeError(f"ARC dense kernels: unknown FP8 families {unknown}")
        _FP8["families"] = fams
    return _FP8["families"]


def quantize_fp8_rows(weight, chunk_rows=8192):
    n, k = weight.shape
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight.device)
    s = torch.empty((n, 1), dtype=torch.float32, device=weight.device)
    for start in range(0, n, chunk_rows):
        w = weight[start:start + chunk_rows].float()
        scale = (w.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
        q[start:start + chunk_rows] = (w / scale).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        s[start:start + chunk_rows] = scale
    return q, s


def _fp8_gemm(flat, weight, scale):
    from vllm import _custom_ops as ops

    xq, xs = ops.scaled_fp8_quant(flat, use_per_token_if_dynamic=True)
    return ops.cutlass_scaled_mm(xq, weight.t(), xs, scale, torch.bfloat16)


def _arc_fp8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, bf16: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    n = weight.shape[0]
    flat = x.reshape(-1, shape[-1])
    if flat.shape[0] == 0:
        return x.new_empty((*shape[:-1], n))
    if not flat.is_contiguous():
        flat = flat.contiguous()
    return _fp8_gemm(flat, weight, scale).reshape(*shape[:-1], n)


_TOG = {"stamp": None, "sel": None}
# ARC_FP8_BF16="<layer>:<family>,..." (21 Oct): these layers keep the BF16 path in served (non-toggle) mode.
FP8_BF16_KEYS = {k for k in (_arc_os.environ.get("ARC_FP8_BF16") or "").split(",") if k}


def fp8_keep_bf16(child, family):
    m = _arc_re.search(r"layers\.(\d+)\.", getattr(child, "prefix", "") or "")
    return bool(m) and f"{m.group(1)}:{family}" in FP8_BF16_KEYS


def _arc_toggle_bf16(key):
    """Toggle screens (21 Oct: per layer): no file -> FP8; file empty or 'all' -> BF16; JSON {"bf16": [keys]} -> BF16
    for those keys only; {"fp8": [keys]} -> FP8 for those keys only. A key is '<layer>:<family>' or '*:<family>'."""
    try:
        st = _arc_os.stat(FP8_TOGGLE)
    except FileNotFoundError:
        return False
    stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
    if _TOG["stamp"] != stamp:
        import json as _json
        with open(FP8_TOGGLE) as fh:
            text = fh.read().strip()
        _TOG.update(stamp=stamp, sel="all" if text in ("", "all") else _json.loads(text))
    sel = _TOG["sel"]
    if sel == "all":
        return True
    fam = key.split(":", 1)[1]
    if "bf16" in sel:
        return key in sel["bf16"] or f"*:{fam}" in sel["bf16"]
    return not (key in sel["fp8"] or f"*:{fam}" in sel["fp8"])


def _arc_fp8_linear_fake(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, bf16: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _register_fp8_op():
    from vllm.utils.torch_utils import direct_register_custom_op

    if not hasattr(torch.ops.vllm, "arc_fp8_linear"):
        direct_register_custom_op(op_name="arc_fp8_linear", op_func=_arc_fp8_linear, fake_impl=_arc_fp8_linear_fake)


def check_fp8(family, prefix, weight, q, s, rows=8192):
    """FP8 layer error on its real weights (first `rows` output channels), Gaussian activations, vs FP32."""
    key = ("fp8", family)
    if key in _STATE["checked"]:
        return
    _STATE["checked"].add(key)
    w, qs, ss = weight[:rows], q[:rows].contiguous(), s[:rows].contiguous()
    generator = torch.Generator(device=weight.device).manual_seed(0)
    result = {"family": family, "prefix": prefix, "shape": list(weight.shape), "rows_checked": int(w.shape[0])}
    for m in (1, 16, 192, 2048):
        x = torch.randn((m, w.shape[1]), generator=generator, device=weight.device, dtype=torch.float32)
        x = x.to(torch.bfloat16)
        ref = x.float() @ w.float().t()
        row = {"fp8_vs_fp32": _rel(_fp8_gemm(x, qs, ss), ref), "bf16_cublas_vs_fp32": _rel(F.linear(x, w), ref)}
        result[f"M{m}"] = row
        if row["fp8_vs_fp32"] > FP8_ERROR_LIMIT:
            log("CHECK", {**result, "failed": True})
            raise RuntimeError(f"ARC FP8 self-check failed for {family}: {row}")
    log("CHECK", result)


def fp8_method(family):
    if _FP8["method"] is None:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        class Fp8LinearMethod(UnquantizedLinearMethod):
            def __init__(self, family):
                super().__init__()
                self.family = family

            def process_weights_after_loading(self, layer):
                weight = layer.weight.data
                if weight.dtype != torch.bfloat16 or weight.dim() != 2:
                    raise RuntimeError(f"ARC FP8 expects a 2-D BF16 weight, got {weight.dtype} {tuple(weight.shape)}")
                prefix = getattr(layer, "prefix", "") or type(layer).__name__
                calib = None
                with torch.no_grad():
                    layer_idx = _arc_re.search(r"layers\.(\d+)\.", prefix or "")
                    if FP8_CALIB and not (layer_idx and int(layer_idx.group(1)) in FP8_CALIB_SKIP):
                        q, s, calib = load_calibrated(prefix, self.family, weight)
                    else:
                        q, s = quantize_fp8_rows(weight)
                    check_fp8(self.family, prefix, weight, q, s)
                keep = weight if FP8_TOGGLE else torch.empty(0, dtype=torch.bfloat16, device=weight.device)
                if FP8_W16:
                    layer.arc_w16 = w16_prepare(layer, q, s)
                    check_w16(self.family, prefix, layer, q, s, weight)
                    del q
                else:
                    layer.arc_w16 = None
                    layer.weight = torch.nn.Parameter(q, requires_grad=False)
                    layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)
                layer.arc_bf16 = torch.nn.Parameter(keep, requires_grad=False)
                layer.arc_key = f"{layer_idx.group(1) if layer_idx else '?'}:{self.family}"
                del weight
                stats = _STATE["quantized"].setdefault("fp8:" + self.family, {"layers": 0})
                stats["layers"] += 1
                log("FP8", {"prefix": prefix, "family": self.family, "shape": list(layer.arc_w16 or layer.weight.shape),
                            "totals": stats, "toggle": bool(FP8_TOGGLE), "calib": calib, "w16": FP8_W16})

            def apply(self, layer, x, bias=None):
                if FP8_TOGGLE and layer.arc_bf16.numel() and _arc_toggle_bf16(layer.arc_key):
                    y = F.linear(x, layer.arc_bf16)
                    return y if bias is None else y + bias
                if layer.arc_w16 is not None:
                    return _w16_gemm(layer, x, bias)
                y = torch.ops.vllm.arc_fp8_linear(x, layer.weight, layer.weight_scale, layer.arc_bf16)
                return y if bias is None else y + bias

        _FP8["method"] = Fp8LinearMethod
    return _FP8["method"](family)
'''

# (file, old, new) patches on the bundle's arc_infra files.
PATCHES = [
    ("dense_kernels_runtime.py",
     """        family = family_of(child, is_head, shape)
""",
     """        family = family_of(child, is_head, shape)
        if family is not None and is_linear and family in fp8_families() and not fp8_keep_bf16(child, family):
            _register_fp8_op()
            child.quant_method = fp8_method(family)
            counts[f"fp8:{family}"] = counts.get(f"fp8:{family}", 0) + 1
            continue
"""),
    ("dense_kernels_patch.py",
     """    if not isinstance(spec, dict) or set(spec) - {'skinny', 'int8_weights', 'mid_m', *PHASE6_KEYS, *WIDE_KEYS}:""",
     """    if not isinstance(spec, dict) or set(spec) - {'skinny', 'int8_weights', 'mid_m', 'fp8_w8a8', *PHASE6_KEYS,
                                                  *WIDE_KEYS}:"""),
    ("dense_kernels_patch.py",
     """    result.update(phase6)
    result.update(wide_keys)
    return result""",
     """    result.update(phase6)
    result.update(wide_keys)
    if spec.get('fp8_w8a8'):  # throughput arm, 19 Oct: FP8 W8A8 families (arc_infra dense runtime)
        result['fp8_w8a8'] = sorted(set(spec['fp8_w8a8']))
    return result"""),
]
