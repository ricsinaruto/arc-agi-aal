"""Drain-wave cache trace (throughput E3 diagnostic, 25 Oct). Appended to the bundle's arc_offload_runtime.py by
nebius_kaggle_faithful.py --cache-trace; active with ARC_CACHE_TRACE=1 in the engine process. Monitoring only: it
records, never changes eviction or lookup. Lines (server log):
  ARC_CT_INIT   CPU tier blocks/policy
  ARC_CT_LOOKUP per new request: game, tokens, GPU-local and CPU-external hit tokens, the CPU key that missed
  ARC_CT_STORE  per CPU prepare_store: keys asked, stored, and the CPU keys evicted (short hashes)
  ARC_CT_EVICT  per second: GPU prefix-cache blocks evicted, by owner game (the game of the request that freed them)
The client tags requests with vllm_xargs.arc_game (SamplingParams.extra_args; no effect on hashing or outputs)."""
import time as _ct_time

if _os.environ.get("ARC_CACHE_TRACE") == "1":
    from vllm.v1.core import block_pool as _ct_bp

    _CT_OWNER = {}            # GPU block id -> game of the request that last freed it
    _CT_EVICT = {}            # game -> GPU blocks evicted since the last flush
    _CT_LAST = [0.0]
    _CT_MISS = {}             # group -> first CPU key that missed in the current lookup

    from vllm.v1.kv_offload.base import get_offload_block_hash as _ct_hash, get_offload_group_idx as _ct_group

    def _ct_key(key):
        return bytes(_ct_hash(key)).hex()[:16]

    def _ct_game(request):
        sp = getattr(request, "sampling_params", None)
        return ((getattr(sp, "extra_args", None) or {}).get("arc_game")) or "?"

    def _ct_flush(force=False):
        now = _ct_time.time()
        if _CT_EVICT and (force or now - _CT_LAST[0] >= 1.0):
            print("ARC_CT_EVICT " + _json.dumps({"t": round(now, 2), "by_game": dict(_CT_EVICT)}), flush=True)
            _CT_EVICT.clear()
            _CT_LAST[0] = now

    _ct_free = _kvm.KVCacheManager.free

    def _ct_free_wrapped(self, request):
        game = _ct_game(request)
        try:
            for group in self.coordinator.get_blocks(request.request_id):
                for b in group:
                    if not b.is_null:
                        _CT_OWNER[b.block_id] = game
        except Exception as exc:  # monitoring only
            print("ARC_CT_ERROR free " + repr(exc)[:200], flush=True)
        return _ct_free(self, request)

    _kvm.KVCacheManager.free = _ct_free_wrapped
    _ct_evict = _ct_bp.BlockPool._maybe_evict_cached_block

    def _ct_evict_wrapped(self, block):
        evicted = _ct_evict(self, block)
        if evicted:
            g = _CT_OWNER.get(block.block_id, "?")
            _CT_EVICT[g] = _CT_EVICT.get(g, 0) + 1
            _ct_flush()
        return evicted

    _ct_bp.BlockPool._maybe_evict_cached_block = _ct_evict_wrapped

    def _ct_wrap_manager(manager):
        if getattr(manager, "_ct_wrapped", False):
            return
        manager._ct_wrapped = True
        print("ARC_CT_INIT " + _json.dumps({"type": type(manager).__name__, **{k: v for k, v in vars(manager).items()
              if isinstance(v, (int, float, str, bool)) and not k.startswith("__")}}, default=str), flush=True)
        lookup, prepare_store = manager.lookup, manager.prepare_store

        def ct_lookup(key, req_context, *a, **k):
            out = lookup(key, req_context, *a, **k)
            if getattr(out, "name", "") == "MISS":
                _CT_MISS.setdefault(str(_ct_group(key)), _ct_key(key))  # the first miss per group
            return out

        def ct_prepare_store(keys, req_context, *a, **k):
            keys = list(keys)
            out = prepare_store(keys, req_context, *a, **k)
            if out is not None:
                print("ARC_CT_STORE " + _json.dumps({"t": round(_ct_time.time(), 2), "asked": len(keys),
                      "stored": len(out.keys_to_store), "evicted": [[_ct_group(e), _ct_key(e)] for e in out.evicted_keys]}), flush=True)
            return out

        manager.lookup, manager.prepare_store = ct_lookup, ct_prepare_store

    _ct_gnmt = ArcOffloadingConnector.get_num_new_matched_tokens

    def _ct_gnmt_wrapped(self, request, num_computed_tokens):
        sched = self.connector_scheduler
        if sched is not None:
            _ct_wrap_manager(sched.manager)
        _CT_MISS.clear()
        out = _ct_gnmt(self, request, num_computed_tokens)
        if sched is not None:
            print("ARC_CT_LOOKUP " + _json.dumps({"t": round(_ct_time.time(), 2), "req": request.request_id,
                  "game": _ct_game(request), "tokens": request.num_tokens, "local": num_computed_tokens,
                  "ext": out[0], "miss": dict(_CT_MISS)}), flush=True)
        _ct_flush()
        return out

    ArcOffloadingConnector.get_num_new_matched_tokens = _ct_gnmt_wrapped
