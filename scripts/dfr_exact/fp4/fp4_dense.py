"""NVFP4 W4A4 dense projections (throughput arm): code appended to the bundle's
dense_kernels_runtime.py and the patches that wire it in (builder --fp4-dense). The FP8 W8A8 path's sibling."""

RUNTIME_CODE = r'''

# ----------------------------------------------------------------------------- NVFP4 W4A4 (throughput arm, 20 Oct)
# ``fp4_w4a4``: NVFP4 weights (E2M1 values, FP8 E4M3 scale per 16 inputs, one FP32 global scale per layer from the
# checkpoint weight's amax: ModelOpt's rule), activations quantized per call to NVFP4 with a dynamic per-tensor global
# scale (amax on the device, graph-safe); vLLM's cutlass_scaled_fp4_mm with BF16 output at every M. Nothing is fitted.
FP4_FAMILIES = tuple(LINEAR_FAMILIES)
FP4_GS = 448.0 * 6.0
FP4_ERROR_LIMIT = 0.20
_FP4 = {"families": None, "method": None}


def fp4_families():
    if _FP4["families"] is None:
        raw = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}
        fams = list(raw.get("fp4_w4a4") or [])
        unknown = sorted(set(fams) - set(FP4_FAMILIES))
        if unknown:
            raise RuntimeError(f"ARC dense kernels: unknown FP4 families {unknown}")
        _FP4["families"] = fams
    return _FP4["families"]


def quantize_fp4_weight(weight):
    """(packed [N, K/2 (+pad)], swizzled block scales, weight global scale (1-elt fp32), padding cols)."""
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (pad_nvfp4_weight_for_cutlass,
                                                                           swizzle_blockscale)

    gs = (FP4_GS / weight.float().abs().amax().clamp(min=1e-8)).reshape(1).to(torch.float32)
    q, bs = scaled_fp4_quant(weight.contiguous(), gs, is_sf_swizzled_layout=False)
    q, pad = pad_nvfp4_weight_for_cutlass(q)
    return q, swizzle_blockscale(bs), gs, int(pad)


def _fp4_gemm(flat, weight, bscale, wgs, pad, n):
    from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import slice_nvfp4_output

    xgs = (FP4_GS / flat.abs().amax().float().clamp(min=1e-8)).reshape(1)
    xq, xs = scaled_fp4_quant(flat, xgs, is_sf_swizzled_layout=True, backend="cutlass",
                              padded_n=flat.shape[-1] + pad * 2)
    alpha = 1.0 / (xgs * wgs)
    out = cutlass_scaled_fp4_mm(xq, weight, xs, bscale, alpha, torch.bfloat16)
    return slice_nvfp4_output(out, n)


def _arc_fp4_linear(x: torch.Tensor, weight: torch.Tensor, bscale: torch.Tensor, wgs: torch.Tensor,
                    n: int, pad: int) -> torch.Tensor:
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    if flat.shape[0] == 0:
        return x.new_empty((*shape[:-1], n))
    if not flat.is_contiguous():
        flat = flat.contiguous()
    return _fp4_gemm(flat, weight, bscale, wgs, pad, n).reshape(*shape[:-1], n)


def _arc_fp4_linear_fake(x: torch.Tensor, weight: torch.Tensor, bscale: torch.Tensor, wgs: torch.Tensor,
                         n: int, pad: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], n))


def _register_fp4_op():
    from vllm.utils.torch_utils import direct_register_custom_op

    if not hasattr(torch.ops.vllm, "arc_fp4_linear"):
        direct_register_custom_op(op_name="arc_fp4_linear", op_func=_arc_fp4_linear, fake_impl=_arc_fp4_linear_fake)


def check_fp4(family, prefix, weight, packed, rows=8192):
    """FP4 layer error on its real weights (first `rows` output channels re-quantized), Gaussian activations, vs FP32."""
    key = ("fp4", family)
    if key in _STATE["checked"]:
        return
    _STATE["checked"].add(key)
    w = weight[:rows].contiguous()
    q, bs, gs, pad = quantize_fp4_weight(w)
    generator = torch.Generator(device=weight.device).manual_seed(0)
    result = {"family": family, "prefix": prefix, "shape": list(weight.shape), "rows_checked": int(w.shape[0]),
              "pad": pad}
    for m in (1, 16, 192, 2048):
        x = torch.randn((m, w.shape[1]), generator=generator, device=weight.device, dtype=torch.float32)
        x = x.to(torch.bfloat16)
        ref = x.float() @ w.float().t()
        row = {"fp4_vs_fp32": _rel(_fp4_gemm(x, q, bs, gs, pad, w.shape[0]), ref),
               "bf16_cublas_vs_fp32": _rel(F.linear(x, w), ref)}
        result[f"M{m}"] = row
        if row["fp4_vs_fp32"] > FP4_ERROR_LIMIT:
            log("CHECK", {**result, "failed": True})
            raise RuntimeError(f"ARC FP4 self-check failed for {family}: {row}")
    log("CHECK", result)


def fp4_method(family):
    if _FP4["method"] is None:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        class Fp4LinearMethod(UnquantizedLinearMethod):
            def __init__(self, family):
                super().__init__()
                self.family = family

            def process_weights_after_loading(self, layer):
                weight = layer.weight.data
                if weight.dtype != torch.bfloat16 or weight.dim() != 2:
                    raise RuntimeError(f"ARC FP4 expects a 2-D BF16 weight, got {weight.dtype} {tuple(weight.shape)}")
                prefix = getattr(layer, "prefix", "") or type(layer).__name__
                with torch.no_grad():
                    q, bs, gs, pad = quantize_fp4_weight(weight)
                    check_fp4(self.family, prefix, weight, q)
                n = int(weight.shape[0])
                layer.weight = torch.nn.Parameter(q, requires_grad=False)
                layer.arc_fp4_bscale = torch.nn.Parameter(bs, requires_grad=False)
                layer.arc_fp4_gs = torch.nn.Parameter(gs, requires_grad=False)
                layer.arc_fp4_n, layer.arc_fp4_pad = n, pad
                del weight
                stats = _STATE["quantized"].setdefault("fp4:" + self.family, {"layers": 0})
                stats["layers"] += 1
                log("FP4", {"prefix": prefix, "family": self.family, "shape": [n, int(q.shape[1]) * 2], "pad": pad,
                            "totals": stats})

            def apply(self, layer, x, bias=None):
                y = torch.ops.vllm.arc_fp4_linear(x, layer.weight, layer.arc_fp4_bscale, layer.arc_fp4_gs,
                                                  layer.arc_fp4_n, layer.arc_fp4_pad)
                return y if bias is None else y + bias

        _FP4["method"] = Fp4LinearMethod
    return _FP4["method"](family)
'''

PATCHES = [
    ("dense_kernels_runtime.py",
     """        family = family_of(child, is_head, shape)
""",
     """        family = family_of(child, is_head, shape)
        if family is not None and is_linear and family in fp4_families():
            _register_fp4_op()
            child.quant_method = fp4_method(family)
            counts[f"fp4:{family}"] = counts.get(f"fp4:{family}", 0) + 1
            continue
"""),
    ("dense_kernels_patch.py",
     """    if not isinstance(spec, dict) or set(spec) - {'skinny', 'int8_weights', 'mid_m', *PHASE6_KEYS, *WIDE_KEYS}:""",
     """    if not isinstance(spec, dict) or set(spec) - {'skinny', 'int8_weights', 'mid_m', 'fp4_w4a4', *PHASE6_KEYS,
                                                  *WIDE_KEYS}:"""),
    ("dense_kernels_patch.py",
     """    result.update(phase6)
    result.update(wide_keys)
    return result""",
     """    result.update(phase6)
    result.update(wide_keys)
    if spec.get('fp4_w4a4'):  # throughput arm, 20 Oct: NVFP4 W4A4 families (arc_infra dense runtime)
        result['fp4_w4a4'] = sorted(set(spec['fp4_w4a4']))
    return result"""),
]
