"""Appended to the bundle's expert_prune_runtime.py (= vllm/arc_expert_prune.py in the worker) by
build_v82_ablation.py --ple-bound; prepare_config calls install_ple_prefill_bound first. Uses that module's os, json, resolve."""


PLE_LAYER = "vllm.models.qwen3_8_flash_next.nvidia.ple_layer.Qwen3_8FlashNextPLELayer"


def _prefill_groups(starts, cap):
    """Contiguous request groups [a, b) whose padded size (count x longest query) stays within ``cap`` tokens; a
    request longer than ``cap`` forms its own group."""
    groups, a, n = [], 0, len(starts) - 1
    while a < n:
        b, longest = a + 1, starts[a + 1] - starts[a]
        while b < n:
            wider = max(longest, starts[b + 1] - starts[b])
            if (b + 1 - a) * wider > cap:
                break
            longest, b = wider, b + 1
        groups.append((a, b, longest))
        a = b
    return groups


def install_ple_prefill_bound(cls=None, log=print, limit_mib=None):
    """Bound the PLE short-conv prefill's padded packing (17 Oct, throughput arm).

    ``_short_conv_dilated_prefill_batched`` packs every prefill request of a step into [prefills, longest query,
    hc hidden] and keeps about five such tensors live. The step's tokens are capped (4,096) but the padding is not:
    one long chunk with many short prefills asks for several GiB (CUDA OOM in 7 AV2C runs, 1.06-1.48 GiB single
    allocations; the engine then dies for good). Above ``limit_mib`` per tensor the step's prefills run in contiguous
    groups within it. Each request's conv and state update read only that request, so the result is the same math;
    the grouped path syncs the query starts once and never runs under graph capture. ARC_PLE_PREFILL_BOUND=0 turns it
    off."""
    if os.environ.get("ARC_PLE_PREFILL_BOUND", "1") == "0":
        return False
    try:
        cls = cls or resolve(PLE_LAYER)
        original = cls._short_conv_dilated_prefill_batched
    except (ImportError, AttributeError) as exc:  # not the Flash PLE model: nothing to bound
        log("ARC_PLE_PREFILL_BOUND " + json.dumps({"installed": None, "reason": repr(exc)}), flush=True)
        return False
    if getattr(original, "_arc_bounded", False):
        return False
    limit = int(float(limit_mib or os.environ.get("ARC_PLE_PREFILL_BOUND_MIB", "128")) * 2**20)
    import copy

    import torch

    stats = {"steps": 0, "groups": 0}
    selftest = int(os.environ.get("ARC_PLE_PREFILL_SELFTEST", "0"))  # compare the first N grouped steps

    def bounded(self, x_p, metadata, conv_state, conv_weights, state_indices_tensor_p, num_prefills,
                num_decode_tokens, num_prefill_tokens):
        per_token = x_p.shape[1] * x_p.element_size()
        longest = metadata.max_prefill_query_len
        if (num_prefills <= 1 or longest <= 0 or num_prefills * longest * per_token <= limit
                or torch.cuda.is_current_stream_capturing()):
            return original(self, x_p, metadata, conv_state, conv_weights, state_indices_tensor_p, num_prefills,
                            num_decode_tokens, num_prefill_tokens)
        loc = metadata.non_spec_query_start_loc
        starts = (loc[-num_prefills - 1:] - num_decode_tokens).tolist()
        groups = _prefill_groups(starts, max(1, limit // per_token))
        stats["steps"] += 1
        stats["groups"] += len(groups)
        if stats["steps"] in (1, 10, 100, 1000, 10000):
            log("ARC_PLE_PREFILL_BOUND " + json.dumps({"steps": stats["steps"], "groups": stats["groups"],
                                                       "prefills": num_prefills, "longest": longest,
                                                       "limit_mib": limit / 2**20}), flush=True)
        check = (selftest and stats.get("checks", 0) < selftest
                 and num_prefills * longest * per_token <= 2 * limit)  # the unbounded reference must fit too
        if check:  # the unbounded call on the same inputs; its state rows are put back before the grouped run
            rows = state_indices_tensor_p[:num_prefills].long().clamp(min=0)
            saved = conv_state.index_select(0, rows).clone()
            reference = original(self, x_p, metadata, conv_state, conv_weights, state_indices_tensor_p,
                                 num_prefills, num_decode_tokens, num_prefill_tokens)
            reference_rows = conv_state.index_select(0, rows).clone()
            conv_state.index_copy_(0, rows, saved)
        parts = []
        for a, b, wide in groups:
            sub = copy.copy(metadata)
            sub.non_spec_query_start_loc = loc.new_tensor(starts[a:b + 1]) - starts[a]
            sub.max_prefill_query_len = wide
            sub.has_initial_states_p = metadata.has_initial_states_p[a:b]
            parts.append(original(self, x_p[starts[a]:starts[b]], sub, conv_state, conv_weights,
                                  state_indices_tensor_p[a:b], b - a, 0, starts[b] - starts[a]))
        output = torch.cat(parts)
        if check:
            stats["checks"] = stats.get("checks", 0) + 1
            out_diff = (output.float() - reference.float()).abs().max().item()
            state_diff = (conv_state.index_select(0, rows).float() - reference_rows.float()).abs().max().item()
            log("ARC_PLE_PREFILL_SELFTEST " + json.dumps({
                "check": stats["checks"], "prefills": num_prefills, "longest": longest, "groups": len(groups),
                "output_equal": bool(torch.equal(output, reference)), "output_max_abs": out_diff,
                "state_equal": bool(torch.equal(conv_state.index_select(0, rows), reference_rows)),
                "state_max_abs": state_diff}), flush=True)
        return output

    bounded._arc_bounded = True
    cls._short_conv_dilated_prefill_batched = bounded
    log("ARC_PLE_PREFILL_BOUND " + json.dumps({"installed": f"{cls.__module__}.{cls.__qualname__}",
                                               "limit_mib": limit / 2**20, "pid": os.getpid()}), flush=True)
    return True
