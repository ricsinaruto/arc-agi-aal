"""Read-only allocation-failure diagnostics inside the pinned vLLM process.

Only failure paths call this helper. At most 100 snapshots are emitted per process;
later failures emit counters at powers of two. No prompts or generated text are logged.
"""
import json
import time

_failures = 0


def describe(manager, request, *, stage, required, available, reserved, watermark):
    groups = []
    for index, group in enumerate(manager.coordinator.single_type_managers):
        requests = {}
        unique = {}
        for request_id, blocks in group.req_to_blocks.items():
            held = {b.block_id: b for b in blocks if not b.is_null}
            if held:
                requests[request_id] = len(held)
                unique.update(held)
        groups.append({"index": index, "type": type(group).__name__,
                       "held_by_request": requests, "unique_blocks": len(unique),
                       "shared_blocks": sum(b.ref_cnt > 1 for b in unique.values())})
    return {"epoch": time.time(), "stage": stage, "request_id": request.request_id,
            "status": str(request.status), "sequence_tokens": request.num_tokens,
            "computed_tokens": request.num_computed_tokens,
            "required_blocks": required, "available_blocks": available,
            "free_blocks": manager.block_pool.get_num_free_blocks(),
            "reserved_blocks": reserved, "watermark_blocks": watermark, "groups": groups}


def emit_failure(manager, request, **fields):
    global _failures
    _failures += 1
    try:
        if _failures <= 100:
            record = describe(manager, request, **fields)
        elif _failures & (_failures - 1) == 0:
            record = {"epoch": time.time(), "snapshots_suppressed": _failures - 100}
        else:
            return
        record["failure_number"] = _failures
        print("ARC_CACHE_ALLOCATION_FAILURE " + json.dumps(record, separators=(",", ":")), flush=True)
    except Exception as exc:
        # Diagnostics must never change an allocation decision or crash serving.
        try:
            print("ARC_CACHE_DIAGNOSTIC_ERROR " + type(exc).__name__, flush=True)
        except Exception:
            pass
