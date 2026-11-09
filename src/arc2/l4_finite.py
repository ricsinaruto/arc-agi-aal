"""Where a non-finite value first appears in the routed experts: ``ARC2_FINITE {json}`` lines per worker in the
server log (the line's prefix names the rank).

Loaded as a vLLM general plugin, like ``arc2.l4_inventory`` (``arc2.l4_serving`` writes the entry point when the
study asks for ``finite_check``). ``HummingIndexedExperts.apply`` is wrapped; it only reads, and nothing here may
raise into the server. Only forwards of ``MIN_ROWS`` or more tokens are looked at: prefill, which runs as plain
Python (a captured decode graph never re-enters this code, and a capture is skipped).

Lines, by ``kind``:
- ``load``: once per layer, the per-expert tensors this rank holds (shape, experts that are all zero or hold a
  non-finite value): an expert that was never loaded shows here;
- ``moe``: the layer's input was finite and this rank's expert output is not. With the bad rows, the local experts
  those rows use (and how many sound rows use them), the pairs this rank computes against the estimate the kernel
  is tuned by, and two repeats of the same call: as it was, and with the scratch buffers zeroed first. The output
  is put back as it was, so the server behaves as without the check;
- ``input``: the first layer of a forward whose input is already non-finite on this rank (it came from another
  rank's experts or from the mixer before this layer);
- ``summary``: counts so far.

``ARC2_FINITE_CHECK=attention`` checks the 4-bit sparse attention instead (``checked_attention``): ``attention``
and ``query`` lines, and a summary with the block reuse and the rows that select an index beyond their own
position or a cache row with a non-finite scale (what decides whether the same path is open on another pool).
"""

import inspect
import json
import sys
import time

TARGET = "vllm.model_executor.layers.fused_moe.experts.fused_humming_moe"
MIN_ROWS = 64
EVENT_LINES = 40
RERUNS = 12
SUMMARY_EVERY = 2400
EXPERT_TENSORS = ("w1_scale", "w2_scale", "w1_zp", "w2_zp", "g1_alphas", "g2_alphas", "w1_bias", "w2_bias")


def bad_rows(torch, tensor):
    return ~torch.isfinite(tensor.reshape(tensor.shape[0], -1)).all(dim=1)


def suspects(torch, ids, local, bad, limit=6):
    """Local experts by the bad rows that use them: [expert, bad rows, sound rows]."""
    ids = ids.long()
    size = int(ids.max().item()) + 1

    def count(rows):
        return torch.bincount(ids[rows][local[rows]], minlength=size)

    in_bad, in_good = count(bad), count(~bad)
    order = torch.argsort(in_bad, descending=True)[:limit]
    return [[int(e), int(in_bad[e]), int(in_good[e])] for e in order if int(in_bad[e])]


def block_for(config, pairs):
    for low, high, entry in config or ():
        if low < pairs <= high:
            return [low, high, entry["block_shape"][0]]
    return None


def load_row(torch, experts, w1, w2):
    """The per-expert tensors of one layer on this rank: first dimension = local experts."""
    tensors, quant = {"w1": w1, "w2": w2}, getattr(experts, "quant_config", None)
    for name in EXPERT_TENSORS:
        value = getattr(quant, name, None)
        if torch.is_tensor(value):
            tensors[name] = value
    out = {}
    for name, tensor in tensors.items():
        row = {"shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", "")}
        try:
            flat = tensor.reshape(tensor.shape[0], -1) if tensor.dim() > 1 else tensor.reshape(-1, 1)
            row["zero"] = int((flat == 0).all(dim=1).sum())
            if tensor.is_floating_point():
                finite = torch.isfinite(flat)
                row["nonfinite"] = int((~finite).any(dim=1).sum())
                row["absmax"] = float(flat[finite].abs().max()) if bool(finite.any()) else None
        except Exception as exc:  # noqa: BLE001 - a dtype the checks do not take
            row["error"] = repr(exc)[:120]
        out[name] = row
    return out


def checked(original, torch, log=print, clock=time.time):
    signature = inspect.signature(original)
    state = {"layers": {}, "steps": 0, "calls": 0, "moe": 0, "input": 0, "output_bad_calls": 0,
             "input_bad_calls": 0, "lines": 0, "reruns": 0, "errors": 0, "ratio": [None, None], "step_input_bad": False}

    def say(row):
        log("ARC2_FINITE " + json.dumps(row), flush=True)

    def note(exc):
        state["errors"] += 1
        if state["errors"] <= 5:
            log("ARC2_FINITE_ERROR " + json.dumps(repr(exc)[:400]), flush=True)

    def summary():
        say({"kind": "summary", "t": round(clock(), 2), **{k: state[k] for k in (
            "steps", "calls", "moe", "input", "output_bad_calls", "input_bad_calls", "errors")},
             "layers": len(state["layers"]), "pair_ratio": state["ratio"]})

    def after(self, bound, layer, before):
        a = bound.arguments
        out, ids, expert_map = a["output"], a["topk_ids"], a.get("expert_map")
        bad = bad_rows(torch, out)
        count = int(bad.sum())
        local = torch.ones_like(ids, dtype=torch.bool) if expert_map is None else expert_map[ids.long()] >= 0
        pairs, estimate = int(local.sum()), None
        try:
            estimate = int(self.estimate_local_valid_shape_m(ids))
            ratio = pairs / max(1, estimate)
            low, high = state["ratio"]
            state["ratio"] = [ratio if low is None else round(min(low, ratio), 3),
                              ratio if high is None else round(max(high, ratio), 3)]
        except Exception as exc:  # noqa: BLE001
            note(exc)
        state["calls"] += 1
        state["output_bad_calls"] += bool(count)
        state["input_bad_calls"] += bool(before)
        if before and not state["step_input_bad"]:
            state["step_input_bad"] = True
            state["input"] += 1
            if state["lines"] < EVENT_LINES:
                state["lines"] += 1
                say({"kind": "input", "t": round(clock(), 2), "step": state["steps"], "layer": layer,
                     "rows": int(ids.shape[0]), "bad_rows": int(before)})
        if count and not before:
            state["moe"] += 1
            if state["lines"] < EVENT_LINES:
                state["lines"] += 1
                where = bad.nonzero().flatten()
                first = int(where[0])
                tuning = getattr(self, "w13_tuning_config", None)
                row = {"kind": "moe", "t": round(clock(), 2), "step": state["steps"], "layer": layer,
                       "rows": int(ids.shape[0]), "bad_rows": count, "first_bad": first, "last_bad": int(where[-1]),
                       "bad_values_first_row": int((~torch.isfinite(out[first])).sum()), "width": int(out.shape[-1]),
                       "pairs": pairs, "estimate": estimate, "local_experts": int(a["w1"].shape[0]),
                       "expert_map": expert_map is not None,
                       "block_estimate": block_for(tuning, estimate or 0), "block_pairs": block_for(tuning, pairs),
                       "weights_finite": bool(torch.isfinite(a["topk_weights"]).all()),
                       "suspects": suspects(torch, ids, local, bad)}
                if state["reruns"] < RERUNS:
                    state["reruns"] += 1
                    saved = out.clone()
                    try:
                        original(*bound.args, **bound.kwargs)
                        row["rerun_same"] = int(bad_rows(torch, out).sum())
                        for name in ("workspace13", "workspace2"):
                            if torch.is_tensor(a.get(name)):
                                a[name].zero_()
                        original(*bound.args, **bound.kwargs)
                        row["rerun_zeroed"] = int(bad_rows(torch, out).sum())
                    finally:
                        out.copy_(saved)
                say(row)
        if state["calls"] % SUMMARY_EVERY == 0:
            summary()

    def apply(self, *args, **kwargs):
        bound = layer = before = None
        try:
            capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
            bound = signature.bind(self, *args, **kwargs)
            a = bound.arguments
            if int(a["topk_ids"].shape[0]) < MIN_ROWS or capturing:
                bound = None
            else:
                new = id(self) not in state["layers"]
                layer = state["layers"].setdefault(id(self), len(state["layers"]))
                if layer == 0:
                    if state["steps"] in (1, 10, 100):  # the forwards so far, early: a short run still has a count
                        summary()
                    state["steps"] += 1
                    state["step_input_bad"] = False
                if new:
                    say({"kind": "load", "layer": layer, "tensors": load_row(torch, self, a["w1"], a["w2"])})
                hidden = a["hidden_states"]
                before = int(bad_rows(torch, hidden.reshape(-1, hidden.shape[-1])).sum())
        except Exception as exc:  # noqa: BLE001 - a reading, never the server
            note(exc)
            bound = None
        result = original(self, *args, **kwargs)
        if bound is not None:
            try:
                after(self, bound, layer, before)
            except Exception as exc:  # noqa: BLE001
                note(exc)
        return result

    apply.arc2_finite = state
    return apply


KV4 = "vllm.arc_kv4"
_DONE = {}
ATTENTION_EVENTS = 40
SAMPLE_ROWS = 32
ABSURD_SCALE = 1.0e6
SITE_ENV = "ARC2_FINITE_CHECK"  # "moe" (the routed experts, the default) or "attention" (the 4-bit sparse attention)


def scale_values(torch, raw):
    """Stored 4-bit scales (int32 bit patterns, the zero point in the low four bits) as floats, and which are
    non-finite or absurd (a row this sequence never wrote keeps the bytes of the block's last use)."""
    values = (raw & -16).contiguous().view(torch.float32)
    return values, ~torch.isfinite(values) | (values.abs() > ABSURD_SCALE)


def checked_attention(mod, torch, log=print, clock=time.time):
    """Wraps of ``arc_kv4.cache_update`` and ``arc_kv4.attention`` (prefill forwards only). The update keeps, per
    attention layer, when each cache slot was last written by a seen forward and counts page starts on blocks
    written before (block reuse). The attention then knows each row's own position (its slot in its request's
    page table) and reads, for a sample of rows and for every non-finite output row: indices beyond the row's own
    position, and the stored key and value scales of the selected rows. ``attention`` lines: a non-finite output
    on a finite query; ``query`` lines: the query was already non-finite."""
    update, attend = mod.cache_update, mod.attention
    state = {"sites": {}, "forwards": 0, "calls": 0, "rows": 0, "bad_out_calls": 0, "bad_query_calls": 0,
             "events": 0, "lines": 0, "errors": 0, "sampled_rows": 0, "entries": 0, "future_entries": 0,
             "rows_with_future": 0, "scale_bad_entries": 0, "rows_with_scale_bad": 0, "block_first": 0,
             "block_reuse": 0, "mismatch": 0, "splits": {}, "shape": None}

    def say(row):
        log("ARC2_FINITE " + json.dumps(row), flush=True)

    def note(exc):
        state["errors"] += 1
        if state["errors"] <= 5:
            log("ARC2_FINITE_ERROR " + json.dumps(repr(exc)[:400]), flush=True)

    def summary():
        guard = None
        try:  # the rotation guard's counts, when it is on (arc_infra.kv4_runtime.guarded_rht)
            seen = getattr(mod, "_STATE", {}).get("guard")
            if seen:
                guard = {"calls": seen["calls"], "padded": seen["padded"],
                         "zeroed": None if seen["zeroed"] is None else int(seen["zeroed"])}
        except Exception as exc:  # noqa: BLE001
            note(exc)
        say({"kind": "summary", "site": "attention", "t": round(clock(), 2),
             **{k: v for k, v in state.items() if k != "sites"}, "layers": len(state["sites"]), "guard": guard})

    def active(rows):
        return rows >= MIN_ROWS and not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())

    def cache_update(impl, key, value, kv_cache, slot_mapping):
        result = update(impl, key, value, kv_cache, slot_mapping)
        try:
            if active(int(slot_mapping.shape[0])):
                blocks, page = int(kv_cache.shape[0]), int(kv_cache.shape[2])
                site = state["sites"].get(id(impl))
                if site is None:
                    site = state["sites"][id(impl)] = {
                        "layer": len(state["sites"]),
                        "stamp": torch.zeros(blocks * page, dtype=torch.int32, device=kv_cache.device)}
                if site["layer"] == 0:
                    if state["forwards"] in (1, 10, 100) or (state["forwards"] and state["forwards"] % 50 == 0):
                        summary()
                    state["forwards"] += 1
                slots = slot_mapping.long()
                slots = slots[(slots >= 0) & (slots < blocks * page)]
                starts = slots[slots % page == 0]
                before = site["stamp"][starts]
                state["block_reuse"] += int((before > 0).sum())
                state["block_first"] += int((before == 0).sum())
                site["stamp"][slots] = state["forwards"]
                site["slots"], site["forward"] = slot_mapping, state["forwards"]
        except Exception as exc:  # noqa: BLE001 - a reading, never the server
            note(exc)
        return result

    def look(impl, query, kv_cache, indices, block_table, token_to_req, out):
        rows = int(query.shape[0])
        site = state["sites"].get(id(impl))
        state["calls"] += 1
        state["rows"] += rows
        bad = bad_rows(torch, out)
        query_bad = bad_rows(torch, query)
        count, count_query = int(bad.sum()), int(query_bad.sum())
        state["bad_out_calls"] += bool(count)
        state["bad_query_calls"] += bool(count_query)
        k_codes, _, k_scale, v_scale = mod.cache_views(kv_cache, mod.configured_mode(), impl)
        page, heads, width = int(k_codes.shape[1]), int(k_codes.shape[2]), int(indices.shape[1])
        if state["shape"] is None:
            state["shape"] = {"blocks": int(k_codes.shape[0]), "page": page, "kv_heads": heads,
                              "query_heads": int(query.shape[1]), "indices": width, "mode": mod.configured_mode()}
        try:
            splits = str(mod._kv8.launch_profile(rows, heads, int(query.shape[1]) // heads, width)["splits"])
            state["splits"][splits] = state["splits"].get(splits, 0) + 1
        except Exception as exc:  # noqa: BLE001
            note(exc)
        slots = None if site is None or site.get("forward") != state["forwards"] else site["slots"]
        if slots is None or int(slots.shape[0]) != rows:
            state["mismatch"] += 1
            return
        chosen = torch.arange(0, rows, max(1, rows // SAMPLE_ROWS), device=query.device)
        chosen = torch.unique(torch.cat([chosen, bad.nonzero().flatten()[:8]]))
        request = token_to_req[chosen].long()
        known = (request >= 0) & (request < block_table.shape[0]) & (slots[chosen] >= 0)
        chosen, request = chosen[known], request[known]
        if not int(chosen.numel()):
            return
        table = block_table[request].long()                                   # [n, table width]
        own = slots[chosen].long()
        match = table == (own // page)[:, None]
        found = match.any(dim=1)
        if not bool(found.all()):
            chosen, request, table, own, match = chosen[found], request[found], table[found], own[found], match[found]
            if not int(chosen.numel()):
                return
        column = match.int().argmax(dim=1)
        position = column * page + own % page
        idx = indices[chosen].long()
        valid = idx >= 0
        future = valid & (idx > position[:, None])
        physical = torch.gather(table, 1, (idx.clamp(min=0) // page).clamp(max=table.shape[1] - 1))
        valid &= (physical >= 0) & (physical < k_codes.shape[0]) & (idx // page < table.shape[1])
        physical, offset = physical.clamp(min=0), idx.clamp(min=0) % page
        k_values, k_bad = scale_values(torch, k_scale[physical, offset])     # [n, width, kv heads]
        v_values, v_bad = scale_values(torch, v_scale[physical, offset])
        entry_bad = (k_bad | v_bad).any(dim=-1) & valid
        state["sampled_rows"] += int(chosen.numel())
        state["entries"] += int(valid.sum())
        state["future_entries"] += int(future.sum())
        state["rows_with_future"] += int(future.any(dim=1).sum())
        state["scale_bad_entries"] += int(entry_bad.sum())
        state["rows_with_scale_bad"] += int(entry_bad.any(dim=1).sum())
        if not count:
            return
        if not int((bad & ~query_bad).sum()):  # the query was already non-finite: upstream of this kernel
            if state["lines"] < ATTENTION_EVENTS and not site.get("query_line") == state["forwards"]:
                site["query_line"] = state["forwards"]
                state["lines"] += 1
                say({"kind": "query", "t": round(clock(), 2), "forward": state["forwards"], "layer": site["layer"],
                     "rows": rows, "bad_rows": count, "bad_queries": count_query})
            return
        state["events"] += 1
        if state["lines"] >= ATTENTION_EVENTS:
            return
        state["lines"] += 1
        detail = []
        for n in (bad[chosen] & ~query_bad[chosen]).nonzero().flatten()[:3].tolist():
            hit = entry_bad[n].nonzero().flatten()[:6]
            slot = physical[n, hit] * page + offset[n, hit]
            stamp = site["stamp"][slot]
            detail.append({
                "row": int(chosen[n]), "request": int(request[n]), "position": int(position[n]),
                "pages": int((table[n] >= 0).sum()), "valid": int(valid[n].sum()), "future": int(future[n].sum()),
                "max_index": int(idx[n].max()), "scale_bad": int(entry_bad[n].sum()),
                "k_bad": int((k_bad[n].any(dim=-1) & valid[n]).sum()),
                "v_bad": int((v_bad[n].any(dim=-1) & valid[n]).sum()),
                "bad_index": idx[n, hit].tolist(), "bad_index_future": future[n, hit].tolist(),
                "bad_block": physical[n, hit].tolist(),
                "forwards_since_written": [None if int(x) == 0 else state["forwards"] - int(x) for x in stamp],
                "v_scale": [repr(float(x)) for x in v_values[n, hit].flatten()[:6]],
                "k_scale": [repr(float(x)) for x in k_values[n, hit].flatten()[:6]],
                "bad_heads": int((~torch.isfinite(out[chosen[n]])).any(dim=-1).sum())})
        say({"kind": "attention", "t": round(clock(), 2), "forward": state["forwards"], "layer": site["layer"],
             "rows": rows, "bad_rows": count, "bad_queries": count_query, "detail": detail})

    def attention(impl, query, kv_cache, logical_indices, block_table, token_to_req, out):
        result = attend(impl, query, kv_cache, logical_indices, block_table, token_to_req, out)
        try:
            if active(int(query.shape[0])):
                look(impl, query, kv_cache, logical_indices, block_table, token_to_req, out)
        except Exception as exc:  # noqa: BLE001
            note(exc)
        return result

    attention.arc2_finite = cache_update.arc2_finite = state
    return cache_update, attention


def file_hashes():
    """Hashes of the files this path runs through, to compare with another runtime's (the RTX's)."""
    import hashlib
    import importlib.util

    out = {}
    for name in ("vllm.arc_kv4", "vllm.arc_kv8", "vllm.models.qwen3_8_flash_next.nvidia.ops.qsa",
                 "vllm.models.qwen3_8_flash_next.nvidia.indexer_qsa", "vllm.models.qwen3_8_flash_next.nvidia.qsa"):
        try:
            origin = importlib.util.find_spec(name).origin
            out[name.rsplit(".", 2)[-2] + "." + name.rsplit(".", 1)[-1]] = hashlib.sha256(
                open(origin, "rb").read()).hexdigest()[:16]
        except Exception as exc:  # noqa: BLE001
            out[name] = repr(exc)[:80]
    return out


GUARD_ENV = "ARC_KV4_GUARD"
ROTATION_TILE = 16
GUARD_COUNT_MIN = 64


def guard_rotation(mod, torch):
    """``arc_infra.kv4_runtime.guarded_rht`` for an installed helper built before it had one (a bundle older than
    the guard): the helper's ``rht`` is replaced by the same thing. Whole tiles (zero rows appended and dropped)
    and no non-finite row left; counts in ``mod._STATE["guard"]``. False when the helper has its own."""
    if hasattr(mod, "guarded_rht") or getattr(mod.rht, "arc2_guard", False):
        return False
    plain = mod.rht

    def rht(x, inverse=False):
        lead, d = x.shape[:-1], x.shape[-1]
        flat = x.reshape(-1, d)
        count = flat.shape[0]
        pad = -count % ROTATION_TILE
        if pad:  # the one copy; the repair below is in place (arc_infra.kv4_runtime.guarded_rht)
            flat = torch.cat([flat, flat.new_zeros((pad, d))])
        out = plain(flat, inverse)[:count]
        bad = ~torch.isfinite(out.sum(dim=-1, keepdim=True))
        out.masked_fill_(bad, 0.0)
        if count >= GUARD_COUNT_MIN:
            seen = mod._STATE.setdefault("guard", {"calls": 0, "padded": 0, "zeroed": None})
            seen["calls"] += 1
            seen["padded"] += bool(pad)
            zeroed = bad.sum()
            seen["zeroed"] = zeroed if seen["zeroed"] is None else seen["zeroed"] + zeroed
        return out.reshape(*lead, d)

    rht.arc2_guard = True
    mod.rht = rht
    return True


def install(log=print, site=None):
    import importlib
    import os

    import torch

    if os.environ.get(GUARD_ENV) == "1" and not _DONE.get("guard"):
        mod = importlib.import_module(KV4)
        _DONE["guard"] = "helper" if hasattr(mod, "guarded_rht") else "plugin"
        guard_rotation(mod, torch)
        log("ARC2_FINITE " + json.dumps({"kind": "guard", "by": _DONE["guard"]}), flush=True)
    site = site or os.environ.get(SITE_ENV)
    if not site:
        return
    if site == "attention":
        mod = importlib.import_module(KV4)
        if getattr(mod.attention, "arc2_finite", None) is None:
            mod.cache_update, mod.attention = checked_attention(mod, torch, log=log)
            log("ARC2_FINITE " + json.dumps({"kind": "installed", "site": site, "min_rows": MIN_ROWS,
                                             "files": file_hashes()}), flush=True)
        return
    cls = importlib.import_module(TARGET).HummingIndexedExperts
    if getattr(cls.apply, "arc2_finite", None) is None:
        cls.apply = checked(cls.apply, torch, log=log)
        log("ARC2_FINITE " + json.dumps({"kind": "installed", "site": site, "min_rows": MIN_ROWS}), flush=True)


def register():
    """The plugin entry point: every vLLM process calls it; only a worker being initialised is touched. If the
    experts' module cannot be imported this early, the wrap waits for the worker's model load."""
    try:
        frame, inside = sys._getframe(1), False
        while frame is not None and not inside:
            inside = frame.f_code.co_name == "init_worker"
            frame = frame.f_back
        if not inside:
            return
        try:
            install()
        except Exception as exc:  # noqa: BLE001
            print("ARC2_FINITE_ERROR " + json.dumps("early: " + repr(exc)[:300]), flush=True)
            from vllm.v1.worker import gpu_worker

            original = gpu_worker.Worker.load_model

            def load_model(self, *args, **kwargs):
                result = original(self, *args, **kwargs)
                try:
                    install()
                except Exception as late:  # noqa: BLE001 - a reading, never the server
                    print("ARC2_FINITE_ERROR " + json.dumps("late: " + repr(late)[:300]), flush=True)
                return result

            gpu_worker.Worker.load_model = load_model
    except Exception as exc:  # noqa: BLE001
        print("ARC2_FINITE_ERROR " + json.dumps(repr(exc)[:400]), flush=True)
