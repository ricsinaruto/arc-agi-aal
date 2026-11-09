"""The large dense projections in 8 bits, quantized from the BF16 checkpoint weights after the load (11 October).

A decode step on an L4 reads 2.9 GiB of BF16 projection weights a rank at the card's 220-290 GB/s (14-17 ms of a
51 ms step; ``arc2.l4_dense_bench``). The six projections of ``SUFFIXES`` are served through Humming's dense GEMM
(the 3-bit experts' library, what vLLM's ``HummingLinearMethod`` calls) on int8 weights with a BF16 scale per 64
inputs: 9.2 ms against 14.2 at 24 rows, 1.76 GiB against 2.87. The three small ones (GDN B/A, indexer Q/K, the
shared expert) are slower through Humming and stay as they are.

Loaded as a vLLM general plugin (``arc2.l4_serving`` writes the entry point for the study's ``dense_int8``); it
does nothing without ``ARC2_L4_DENSE`` ({"mode": ...}) in the worker's environment:

- ``serve``: after the worker's load, chunk by chunk: quantize on the card, check, park the result on the host, drop
  the BF16 weights, hand whole free segments back, then put the 8-bit tensors on the card. The runner's weight
  memory (what sizes the cache) is lowered by what the allocator gave back.
- ``label``: both stay on the card and a forward takes the 8-bit path while the expert probe's mask label contains
  ``dense8`` (the routing screen on one RTX PRO 6000: BF16 and 8-bit passes against one reference in one server);
  ``dense8-hc`` switches only the hyper-connection mixers, ``dense8-nohc`` only the rest.
  A scale covers 64 inputs of one output row, so the values are the four ranks' whatever the split.

``families`` in the setting keeps the swap to those of ``SUFFIXES``' labels (default: all six).

Every layer is checked on its real weight before it is swapped: the output on 24 random BF16 rows against the
float32 product. Above ``LIMIT`` (or on any error) the layer keeps its BF16 weight. One ``ARC2_DENSE8 {json}`` line
a worker: layers, bytes and the error by family, what was left alone, allocated and reserved before and after.
"""

import json
import os
import sys

from .l4_dense_bench import COMPUTE, schema_of

ENV = "ARC2_L4_DENSE"
SCHEMA = ("int8_g64", "int8", "bfloat16", 64, False)  # arc2-l4-dense-v2: 0.7% output error (FP8: 2.7% for 0.4 ms)
GROUP = SCHEMA[3]
SUFFIXES = ((".input_mix_weight_down_block_inject", "hc_down_inject"), (".input_mix_weight_up", "hc_up"),
            (".linear_attn.in_proj_qkvz", "gdn_qkvz"), (".linear_attn.out_proj", "mixer_out"),
            (".self_attn.o_proj", "mixer_out"), (".self_attn.qkv_proj", "attention_qkv_gate"),
            (".lm_head", "output_head"))
LEFT_OUT = ("mtp", "draft", "visual", "vision")
LIMIT = 0.05  # relative output error; the bench's wrong scale grouping gave 0.6-1.4
CHUNK_MIB = 512
ROWS = 24


def family(name):
    """The projection family of a module name, or None (not one of the six, or not the language model's)."""
    low = "." + name.lower()
    if any(word in low for word in LEFT_OUT):
        return None
    return next((label for suffix, label in SUFFIXES if low.endswith(suffix)), None)


def eligible(torch, module, label, head_float=False):
    """Why a module of a family is left alone, or None."""
    weight, method = getattr(module, "weight", None), getattr(module, "quant_method", None)
    if not torch.is_tensor(weight) or weight.dim() != 2 or weight.dtype != torch.bfloat16:
        return "weight is not a BF16 matrix"
    if not type(method).__name__.startswith("Unquantized"):
        return f"method {type(method).__name__}"
    if weight.shape[1] % GROUP:
        return f"{weight.shape[1]} inputs, not a multiple of {GROUP}"
    if label == "output_head" and head_float:
        return "the head is computed in another dtype"
    return None


def quantize(torch, weight):
    """Humming's layer config and tensors of one BF16 weight (as ``arc2.l4_dense_bench.humming_layer``)."""
    from humming.schema import HummingInputSchema
    from humming.transform import prepare_layer_config, transform_humming_tensors

    schema = schema_of(*SCHEMA)
    tensors = schema.quant_tensor(weight, schema, torch.bfloat16)
    config = prepare_layer_config(shape_n=weight.shape[0], shape_k=weight.shape[1], weight_schema=schema,
                                  input_schema=HummingInputSchema(), pad_n_to_multiple=256, pad_k_to_multiple=128,
                                  has_bias=False, torch_dtype=torch.bfloat16)
    return config, {k: v for k, v in transform_humming_tensors(config, tensors).items() if torch.is_tensor(v)}


def runner(torch, humming_forward, config, outputs, locks):
    def run(tensors, x):
        flat = x.reshape(-1, x.shape[-1])
        if flat.dtype != torch.bfloat16:
            flat = flat.to(torch.bfloat16)
        out = humming_forward(config, inputs=flat, weight=tensors["weight"], weight_scale=tensors.get("weight_scale"),
                              zero_point=tensors.get("zero_point"), bias=None,
                              weight_scale_2=tensors.get("weight_scale_2"), locks=locks, compute_config=COMPUTE)
        if out.shape[-1] != outputs:
            out = out[:, :outputs]
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        return out if x.dim() == 2 else out.reshape(*x.shape[:-1], outputs)

    return run


class Int8Method:
    """A layer's quant method on the 8-bit tensors; ``switch`` (label mode) chooses per forward."""

    def __init__(self, original, run, tensors, switch=None):
        self.original, self.run, self.tensors, self.switch = original, run, tensors, switch

    def apply(self, layer, x, bias=None):
        if self.switch is not None and not self.switch():
            return self.original.apply(layer, x, bias)
        out = self.run(self.tensors, x)
        return out if bias is None else out + bias

    def __getattr__(self, name):  # what else a caller asks the method (an embedding's lookup, tie_weights)
        return getattr(self.__dict__["original"], name)


def output_error(torch, weight, run, tensors, rows=ROWS):
    """Relative error of the 8-bit output, and of the BF16 one, against the float32 product on random rows."""
    generator = torch.Generator(device="cpu").manual_seed(weight.shape[0] * 131 + weight.shape[1])
    x = torch.randn((rows, weight.shape[1]), generator=generator).to(device=weight.device, dtype=torch.bfloat16)
    exact = x.float() @ weight.float().T
    scale = exact.norm().clamp_min(1e-30)
    got = float(((run(tensors, x).float() - exact).norm() / scale).item())
    served = float(((torch.nn.functional.linear(x, weight).float() - exact).norm() / scale).item())
    return got, served


def label_selects(text, label):
    """A mask label's ``dense8`` / ``dense8-hc`` / ``dense8-nohc`` covers the family ``label``."""
    _, found, tail = (text or "").partition("dense8")
    if not found:
        return False
    if tail.startswith("-hc"):
        return label.startswith("hc_")
    return not label.startswith("hc_") if tail.startswith("-nohc") else True


def probe_label(label):
    """Label mode's switch: the expert probe's current mask label covers the family."""
    state = getattr(sys.modules.get("vllm.arc_expert_probe"), "_STATE", None)
    return label_selects(getattr(state, "mask_label", None), label)


def nbytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors.values())


def convert(torch, model, mode="serve", chunk_mib=CHUNK_MIB, switch=probe_label, tools=None, progress=None,
            families=None):
    """Swap the eligible projections of ``model``; the report. ``tools``: (quantize, humming_forward) for tests;
    ``progress["dropped"]`` counts the BF16 weights already gone (an error after the first cannot be survived)."""
    import gc

    if mode not in ("serve", "label"):
        raise ValueError("mode is serve or label")
    progress = {} if progress is None else progress
    progress["dropped"] = 0
    if tools is None:
        from humming.forward import humming_forward

        tools = (quantize, humming_forward)
    make, humming_forward = tools
    head_float = any(getattr(m, "head_dtype", None) not in (None, torch.bfloat16) for m in model.modules())
    wanted, families, left, pending = families, {}, [], []
    card = next((p.device for p in model.parameters() if p.is_cuda), None)

    def memory():
        return [torch.cuda.memory_allocated(card), torch.cuda.memory_reserved(card)] if card is not None else [0, 0]

    def flush():
        if mode == "serve":
            gc.collect()
            if card is not None:
                torch.cuda.empty_cache()  # whole free segments back before the smaller tensors go up
        for module, config, tensors, outputs, device, label in pending:
            tensors = {k: v.to(device) for k, v in tensors.items()}
            locks = torch.zeros(1024, dtype=torch.int32, device=device)
            if mode == "serve":
                for key, tensor in tensors.items():
                    setattr(module, key, torch.nn.Parameter(tensor, requires_grad=False))
            module.quant_method = Int8Method(module.quant_method, runner(torch, humming_forward, config, outputs, locks),
                                             tensors, None if mode == "serve" else lambda label=label: switch(label))
        pending.clear()

    before, held = memory(), 0
    for name, module in list(model.named_modules()):
        label = family(name)
        if label is None or (wanted and label not in wanted):
            continue
        why = eligible(torch, module, label, head_float)
        if why is None:
            try:
                weight = module.weight.data
                config, tensors = make(torch, weight)
                locks = torch.zeros(1024, dtype=torch.int32, device=weight.device)
                errors = output_error(torch, weight, runner(torch, humming_forward, config, weight.shape[0], locks),
                                      tensors)
                if not errors[0] <= LIMIT:  # also a NaN
                    why = f"output error {errors[0]:.4g}"
            except Exception as exc:  # noqa: BLE001 - the layer keeps its BF16 weight
                why = f"{type(exc).__name__}: {exc}"[:200]
        if why is not None:
            left.append([name, label, why])
            continue
        shape, device, old = list(weight.shape), weight.device, weight.numel() * weight.element_size()
        del weight, locks
        if mode == "serve":
            tensors = {k: v.cpu() for k, v in tensors.items()}
            delattr(module, "weight")
            progress["dropped"] += 1
        pending.append((module, config, tensors, shape[0], device, label))
        row = families.setdefault(label, {"layers": 0, "shape": shape, "bf16_mib": 0.0, "int8_mib": 0.0,
                                          "error_max": 0.0, "error_sum": 0.0, "bf16_error_max": 0.0})
        row["layers"] += 1
        row["bf16_mib"] += old / 2**20
        row["int8_mib"] += nbytes(tensors) / 2**20
        row["error_max"], row["error_sum"] = max(row["error_max"], errors[0]), row["error_sum"] + errors[0]
        row["bf16_error_max"] = max(row["bf16_error_max"], errors[1])
        held += old
        if mode == "serve" and held >= chunk_mib * 2**20:
            flush()
            held = 0
    flush()
    gc.collect()
    if card is not None:
        torch.cuda.empty_cache()
    after = memory()
    for row in families.values():
        row["error_mean"] = round(row.pop("error_sum") / row["layers"], 5)
        for key in ("bf16_mib", "int8_mib"):
            row[key] = round(row[key], 1)
        for key in ("error_max", "bf16_error_max"):
            row[key] = round(row[key], 5)
    return {"mode": mode, "schema": SCHEMA[0], "layers": sum(r["layers"] for r in families.values()),
            "bf16_gib": round(sum(r["bf16_mib"] for r in families.values()) / 1024, 3),
            "int8_gib": round(sum(r["int8_mib"] for r in families.values()) / 1024, 3),
            "error_max": max((r["error_max"] for r in families.values()), default=None), "families": families,
            "left": left[:12], "left_count": len(left),
            "allocated_gib": [round(v / 2**30, 3) for v in (before[0], after[0])],
            "reserved_gib": [round(v / 2**30, 3) for v in (before[1], after[1])],
            "freed_bytes": before[0] - after[0]}


def register():
    """The plugin entry point: every vLLM process calls it; only a worker being initialised is touched."""
    try:
        setting = json.loads(os.environ.get(ENV) or "null")
        if not setting:
            return
        # A one-card server keeps its worker in the engine process, which loads the plugins once, before the
        # worker exists (``EngineCore.__init__``): label mode (the screen) takes that call too.
        core = os.path.join("v1", "engine", "core.py") if setting.get("mode") == "label" else None
        frame, inside = sys._getframe(1), False
        while frame is not None and not inside:
            inside = frame.f_code.co_name == "init_worker" or bool(core and frame.f_code.co_filename.endswith(core))
            frame = frame.f_back
        if not inside:
            return
        from vllm.v1.worker import gpu_worker

        original = gpu_worker.Worker.load_model
        if getattr(original, "arc2_dense", False):
            return
    except Exception as exc:  # noqa: BLE001
        print("ARC2_DENSE8_ERROR " + json.dumps(repr(exc)[:400]), flush=True)
        return

    def load_model(self, *args, **kwargs):
        import torch

        result = original(self, *args, **kwargs)
        mode, progress = setting.get("mode", "serve"), {}
        try:
            with torch.no_grad():
                report = convert(torch, self.model_runner.get_model(), mode, progress=progress,
                                 families=setting.get("families"))
            runner_ = self.model_runner
            if mode == "serve" and report["freed_bytes"] > 0 and hasattr(runner_, "model_memory_usage"):
                report["weights_gib"] = [round(runner_.model_memory_usage / 2**30, 3)]
                runner_.model_memory_usage -= report["freed_bytes"]  # what sizes the cache
                report["weights_gib"].append(round(runner_.model_memory_usage / 2**30, 3))
            report.update(pid=os.getpid(), rank=getattr(self, "rank", None))
            print("ARC2_DENSE8 " + json.dumps(report), flush=True)
        except Exception as exc:  # noqa: BLE001
            print("ARC2_DENSE8_ERROR " + json.dumps(repr(exc)[:400]), flush=True)
            if progress.get("dropped"):  # a layer without its weight: the start must fail
                raise
        return result

    load_model.arc2_dense = True
    gpu_worker.Worker.load_model = load_model
