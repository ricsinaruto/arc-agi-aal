"""Optional, narrow online MXFP8 dispatch for the pinned Flash-Next runtime.

Copied into vLLM by mxfp8_patch; no arc agent dependency in the serving worker.
Only ordinary unquantized attention projections are eligible. NVFP4 experts,
routers, convolutions, PLE, vision, HyperConnections and lm_head retain their methods.
"""

import json

SUFFIXES = (
    '.self_attn.qkv_proj', '.self_attn.o_proj', '.self_attn.indexer.index_qk_proj',
    '.linear_attn.in_proj_qkvz', '.linear_attn.in_proj_qkv', '.linear_attn.in_proj_z',
    '.linear_attn.out_proj',
)


def eligible(prefix, input_size):
    return (prefix.startswith(('model.layers.', 'language_model.model.layers.', 'mtp.layers.'))
            and prefix.endswith(SUFFIXES) and input_size > 0 and input_size % 32 == 0)


def select(layer, prefix, original, unquantized_class):
    if type(original) is not unquantized_class or not eligible(prefix, layer.input_size):
        return original
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutlassMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
    from vllm.model_executor.layers.quantization.online.mxfp8 import Mxfp8OnlineLinearMethod

    supported, reason = FlashInferCutlassMxfp8LinearKernel.is_supported()
    if not supported:
        raise RuntimeError(f'ARC online MXFP8 requires native FlashInfer CUTLASS: {reason}')
    method = Mxfp8OnlineLinearMethod()
    # Pin the native SM120 path; never accept BF16 emulation or a different backend.
    method.kernel = FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    print('ARC_MXFP8_LAYER ' + json.dumps({
        'prefix': prefix, 'input_size': layer.input_size, 'output_size': layer.output_size,
        'method': type(method).__name__, 'kernel': type(method.kernel).__name__,
    }), flush=True)
    return method
