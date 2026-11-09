"""QSA pick capture (throughput arm, 19 Oct; QSA query-union tiling, step 1). Appended to arc_infra/kv4_runtime.py by
the builder (--qsa-dump). Every EVERY-th eager QSA call with >= MIN_ROWS rows, after AFTER seconds from the first
such call, saves the call's picks and addressing (indices, token_to_req, block_table, shapes; no KV, no queries) to
ARC_QSA_DUMP/NNN.pt, at most MAX files. Output is unchanged; one sync and a ~20 MB copy per saved call."""

RUNTIME_CODE = '''

# ---- QSA pick capture (--qsa-dump; scripts/dfr_exact/qsa/qsa_dump.py)
_ARC_QD = {"t0": None, "n": 0, "saved": 0, "layers": {}}


def _arc_qsa_dump(impl, query, kv_cache, logical_indices, block_table, token_to_req):
    import time
    root = os.environ.get("ARC_QSA_DUMP")
    if not root or query.shape[0] < int(os.environ.get("ARC_QSA_DUMP_MIN_ROWS", "256")):
        return
    if torch.cuda.is_current_stream_capturing():
        return
    st = _ARC_QD
    layer = st["layers"].setdefault(id(impl), len(st["layers"]))
    now = time.time()
    if st["t0"] is None:
        st["t0"] = now
    if now - st["t0"] < float(os.environ.get("ARC_QSA_DUMP_AFTER", "480")):
        return
    st["n"] += 1
    if st["n"] % int(os.environ.get("ARC_QSA_DUMP_EVERY", "50")) or st["saved"] >= int(os.environ.get("ARC_QSA_DUMP_MAX", "24")):
        return
    try:
        os.makedirs(root, exist_ok=True)
        torch.save({"layer": layer, "t": now - st["t0"], "indices": logical_indices.cpu(),
                    "token_to_req": token_to_req.cpu(), "block_table": block_table.cpu(),
                    "q_shape": list(query.shape), "kv_shape": list(kv_cache.shape),
                    "kv_stride": list(kv_cache.stride()), "mode": configured_mode()},
                   os.path.join(root, f"{st['saved']:03d}.pt"))
        st["saved"] += 1
    except Exception as exc:  # never break serving
        print("ARC_QSA_DUMP error", exc, flush=True)
'''

PATCHES = [(
    "kv4_runtime.py",
    '''    mode = configured_mode()
    if not query.shape[0]:
        return out
    return attend(query, *cache_views(kv_cache, mode, impl), logical_indices, block_table, token_to_req, out, mode)''',
    '''    mode = configured_mode()
    if not query.shape[0]:
        return out
    _arc_qsa_dump(impl, query, kv_cache, logical_indices, block_table, token_to_req)
    return attend(query, *cache_views(kv_cache, mode, impl), logical_indices, block_table, token_to_req, out, mode)''',
)]
