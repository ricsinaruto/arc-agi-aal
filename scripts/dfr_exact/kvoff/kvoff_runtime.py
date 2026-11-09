"""KV offload for the CSA+linear Flash runtime (throughput arm, 18 Oct; milestone 2a: placement, no cache).

Installed as ``vllm/arc_kvoff.py``; hooks appended to ``v1/core/kv_cache_utils.py`` (block count), ``v1/worker/
gpu_model_runner.py`` (allocation), ``v1/worker/block_table.py`` (GDN block-id translation) and ``v1/worker/
mamba_utils.py`` (the scalar state-copy path). Off unless ``ARC_KVOFF=1``.

Layout. The CSA+linear allocator gives every QSA layer one main tensor (num_blocks x page) shared by its main K/V
and one layer of each GDN group, plus small device tensors for the compressed keys / ring. Here:

- the main tensors live in page-locked host memory (one mapped region per tensor), read and written by the
  kernels through UVA: the scheduler's num_blocks becomes ``ARC_KVOFF_TOTAL_BLOCKS`` (host-sized);
- the GDN layers that shared a main tensor get a device tensor of ``ARC_KVOFF_DEV_BLOCKS`` pages; their block
  tables are translated on the worker (block id -> device slot) at every commit. A GDN block that is not in the
  batch may lose its slot: its state is copied to the block id's own (unused) host page and copied back when the
  block is next used. A block id handed to an attention group drops its slot without a copy.
- compressed keys and the ring stay on the device (num_blocks pages, small).

Output-exact by construction: the same bytes at different addresses.
"""

import json
import os

import numpy as np
import torch

ENABLED = os.environ.get("ARC_KVOFF") == "1"
_TOTAL = os.environ.get("ARC_KVOFF_TOTAL_BLOCKS", "") or ""  # absolute, or "x1.5": a factor on the profiled count
TOTAL = 0 if _TOTAL.startswith("x") else int(_TOTAL or 0)
FACTOR = float(_TOTAL[1:]) if _TOTAL.startswith("x") else 0.0
DEV = int(os.environ.get("ARC_KVOFF_DEV_BLOCKS", "0") or 0)
STAGING_RING = 8
STATE = {"translator": None, "logged": set()}


def log(kind, payload):
    print("ARC_KVOFF " + json.dumps({"kind": kind, **payload}), flush=True)


# ------------------------------------------------------------------------------------------------ block count
def install_kv_cache_utils(module):
    if not ENABLED or not (TOTAL or FACTOR) or getattr(module, "_arc_kvoff", False):
        return
    original = module.may_override_num_blocks

    def may_override_num_blocks(vllm_config, num_blocks):
        served = original(vllm_config, num_blocks)
        total = TOTAL or int(served * FACTOR)
        if "num_blocks" not in STATE["logged"]:
            STATE["logged"].add("num_blocks")
            log("num_blocks", {"profiled": int(num_blocks), "override": int(served), "served": total})
        return total

    module.may_override_num_blocks = may_override_num_blocks
    module._arc_kvoff = True


# ------------------------------------------------------------------------------------------------ translator
class Translator:
    """GDN block id -> device slot (slot 0 = the null block)."""

    def __init__(self, num_blocks, dev_blocks):
        if dev_blocks < 2:
            raise ValueError("ARC_KVOFF_DEV_BLOCKS must be >= 2")
        self.pmap = np.full(num_blocks, -1, dtype=np.int64)
        self.pmap[0] = 0
        self.owner = np.full(dev_blocks, -1, dtype=np.int64)
        self.owner[0] = 0
        self.last = np.zeros(dev_blocks, dtype=np.int64)
        self.free = list(range(dev_blocks - 1, 0, -1))
        self.spilled = np.zeros(num_blocks, dtype=bool)
        self.gdn = np.zeros(num_blocks, dtype=bool)  # last handed to a GDN group (stale rows map to the null slot)
        self.step = 1
        self.tensors = []  # (host CPU alias, device tensor, page bytes)
        self.stats = {"promote": 0, "spill": 0, "drop": 0, "steps": 0, "peak_used": 0}

    def add_tensor(self, host, dev, page):
        self.tensors.append((host, dev, page))

    def _copy(self, block, slot, to_device):
        for host, dev, page in self.tensors:
            if dev is None:
                continue
            h = host[block * page:(block + 1) * page]
            d = dev[slot * page:(slot + 1) * page]
            (d if to_device else h).copy_(h if to_device else d, non_blocking=True)

    def drop(self, ids):
        """Block ids now owned by an attention group: free their slots, no copy."""
        for b in ids:
            if b <= 0 or b >= len(self.pmap):
                continue
            slot = self.pmap[b]
            if slot > 0:
                self.owner[slot] = -1
                self.free.append(int(slot))
                self.stats["drop"] += 1
            self.pmap[b] = -1
            self.spilled[b] = False
            self.gdn[b] = False

    def own(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        self.gdn[ids[(ids > 0) & (ids < len(self.gdn))]] = True

    def _slot(self):
        if self.free:
            return self.free.pop()
        candidates = np.nonzero((self.owner > 0) & (self.last < self.step))[0]
        if not len(candidates):
            raise RuntimeError(f"ARC_KVOFF: all {len(self.owner) - 1} device GDN slots are in this step's batch; "
                               "raise ARC_KVOFF_DEV_BLOCKS or lower max_num_seqs")
        slot = int(candidates[np.argmin(self.last[candidates])])
        victim = int(self.owner[slot])
        self._copy(victim, slot, to_device=False)
        self.spilled[victim] = True
        self.pmap[victim] = -1
        self.owner[slot] = -1
        self.stats["spill"] += 1
        return slot

    def ensure(self, ids):
        """Map every id (array of block ids, 0 = null) to a device slot, stamping them as in this step."""
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        ids = ids[(ids > 0) & self.gdn[np.clip(ids, 0, len(self.gdn) - 1)]]
        mapped = self.pmap[ids] >= 0
        self.last[self.pmap[ids[mapped]]] = self.step  # before any eviction: this batch's slots stay
        for b in ids[~mapped]:
            slot = self._slot()
            if self.spilled[b]:
                self._copy(int(b), slot, to_device=True)
                self.spilled[b] = False
                self.stats["promote"] += 1
            self.pmap[b] = slot
            self.owner[slot] = b
            self.last[slot] = self.step
        return ids

    def begin_step(self):
        """A new batch: ids stamped from here on (block-table commit, then the mamba copies) stay resident."""
        self.step += 1
        self.stats["steps"] += 1
        used = int((self.owner > 0).sum())
        self.stats["peak_used"] = max(self.stats["peak_used"], used)
        if self.stats["steps"] in (1, 10, 100, 1000, 10000, 100000):
            log("translator", {**self.stats, "used": used, "slots": len(self.owner) - 1})


# ------------------------------------------------------------------------------------------------ allocation
def _host_region(nbytes, device, name):
    from vllm.arc_serving_kernels import host_table

    cpu, view = host_table((nbytes,), torch.int8, device, name)
    return cpu, view


def _allocate(kv_cache_config, device, skip=()):
    """The KV tensors (layer name -> raw tensor) with the main tensors in host RAM."""
    n = int(kv_cache_config.num_blocks)
    if TOTAL and n != TOTAL:
        raise RuntimeError(f"ARC_KVOFF: KV config has {n} blocks, expected {TOTAL}")
    tr = Translator(n, DEV)
    raw, first, packed, mains = {}, None, None, []
    host_bytes = dev_bytes = other_bytes = 0
    kinds = {}  # layer name -> spec class name (the CSA main KV owners are FullAttentionSpec)
    for group in kv_cache_config.kv_cache_groups:
        per = getattr(group.kv_cache_spec, "kv_cache_specs", None) or {}
        for name in group.layer_names:
            kinds[name] = type(per.get(name, group.kv_cache_spec)).__name__
    for i, t in enumerate(kv_cache_config.kv_cache_tensors):
        attn = [x for x in t.shared_by if kinds.get(x) == "FullAttentionSpec"]
        if attn and any(kinds.get(x) not in ("FullAttentionSpec", "MambaSpec") for x in t.shared_by):
            raise RuntimeError(f"ARC_KVOFF: main tensor {i} is shared by {t.shared_by} ({kinds})")
        if t.block_stride > 0:  # packed layout: one backing for all (as upstream), stays on the device
            if attn:
                raise RuntimeError("ARC_KVOFF: packed KV layout with attention main tensors is not supported")
            if packed is None:
                packed = torch.zeros(t.size, dtype=torch.int8, device=device)
                other_bytes += t.size
            for x in t.shared_by:
                raw[x] = packed
            continue
        if not attn:
            tensor = torch.zeros(t.size, dtype=torch.int8, device=device)
            other_bytes += t.size
            for x in t.shared_by:
                raw[x] = tensor
            continue
        if t.size % n:
            raise RuntimeError(f"ARC_KVOFF: main tensor {i} size {t.size} is not {n} pages")
        page = t.size // n
        cpu, view = _host_region(t.size, device, f"kvoff-main-{i}")
        if first is None:
            first = (cpu, view)
        mains.append((view.data_ptr(), t.size, page))
        for x in attn:
            raw[x] = view
        host_bytes += t.size
        gdn = [x for x in t.shared_by if x not in attn]
        if gdn:
            dev = torch.zeros(DEV * page, dtype=torch.int8, device=device)
            for x in gdn:
                raw[x] = dev
            tr.add_tensor(cpu, dev, page)
            dev_bytes += DEV * page
        else:
            tr.add_tensor(cpu, None, page)
    # self-check: a device write through the UVA view is visible in the host alias, and back
    if first is None:
        raise RuntimeError("ARC_KVOFF: no attention main tensor in the KV config")
    host, view = first
    probe = torch.arange(64, dtype=torch.int8, device=device)
    host[-64:].zero_()
    view[-64:].copy_(probe)
    torch.cuda.synchronize()
    ok = bool(torch.equal(host[-64:], probe.cpu()))
    view[-64:].zero_()
    torch.cuda.synchronize()
    if not ok:
        raise RuntimeError("ARC_KVOFF: the UVA view of the host main tensor does not alias it")
    STATE["translator"] = tr
    if CACHE_SLOTS:
        pages = {pg for _, _, pg in mains}
        if len(pages) != 1 or next(iter(pages)) % (2 * RECORD):
            raise RuntimeError(f"ARC_KVOFF: main pages {pages} are not 2 heads x {RECORD} B per token")
        gc = GroupCache([(ptr, size) for ptr, size, _ in mains], n, next(iter(pages)) // (2 * RECORD),
                        CACHE_SLOTS, device)
        STATE["cache"] = gc
        import importlib

        install_kv4(importlib.import_module("vllm.arc_kv4"))
        log("group_cache", {"layers": gc.layers, "slots": CACHE_SLOTS, "page_tokens": gc.gpb * GROUP_TOKENS,
                            "gib": round(gc.nbytes() / 2**30, 2)})
    STATE["gdn_groups"] = {i for i, g in enumerate(kv_cache_config.kv_cache_groups)
                           if type(getattr(g.kv_cache_spec, "first_spec", g.kv_cache_spec)).__name__ == "MambaSpec"}
    if not STATE["gdn_groups"]:
        raise RuntimeError("ARC_KVOFF: no Mamba (GDN) KV cache group")
    log("allocated", {"blocks": n, "dev_gdn_blocks": DEV, "main_tensors": len(tr.tensors),
                      "host_gib": round(host_bytes / 2**30, 2), "dev_gdn_gib": round(dev_bytes / 2**30, 2),
                      "dev_other_gib": round(other_bytes / 2**30, 2), "uva_check": ok})
    names = set()
    for group in kv_cache_config.kv_cache_groups:
        for name in group.layer_names:
            if name not in skip:
                names.add(name)
    if names != set(raw):
        raise RuntimeError("ARC_KVOFF: some layers are not allocated")
    return raw


def install_runner(cls):
    if not ENABLED or getattr(cls, "_arc_kvoff", False):
        return
    if not (TOTAL or FACTOR) or not DEV:
        raise RuntimeError("ARC_KVOFF needs ARC_KVOFF_TOTAL_BLOCKS and ARC_KVOFF_DEV_BLOCKS")

    def _allocate_kv_cache_tensors(self, kv_cache_config):
        return _allocate(kv_cache_config, self.device, self.runner_only_attn_layers)

    cls._allocate_kv_cache_tensors = _allocate_kv_cache_tensors
    cls._arc_kvoff = True
    log("installed", {"runner": cls.__name__, "total_blocks": TOTAL, "dev_blocks": DEV})


# ------------------------------------------------------------------------------------------------ block tables
def install_block_table(multi_cls):
    """GDN tables = the Mamba groups (the QSA compressor ring is SlotMappingMode.NONE too, but stays untranslated)."""
    if not ENABLED or getattr(multi_cls, "_arc_kvoff", False):
        return

    def _gdn(self):
        groups = STATE.get("gdn_groups") or set()
        return [i in groups for i in range(len(self.block_tables))]

    original_append = multi_cls.append_row
    original_add = multi_cls.add_row

    def _note(self, block_ids):
        tr = STATE["translator"]
        if tr is None:
            return
        for table, ids, gdn in zip(self.block_tables, block_ids, _gdn(self)):
            if not ids:
                continue
            if gdn:
                if table.blocks_per_kv_block != 1:
                    raise RuntimeError("ARC_KVOFF: a GDN block table maps one block id to several kernel blocks")
                tr.own(ids)
            else:
                tr.drop(ids)
                if STATE.get("cache") is not None:
                    STATE["cache"].invalid.extend(ids)

    def append_row(self, block_ids, row_idx):
        _note(self, block_ids)
        return original_append(self, block_ids, row_idx)

    def add_row(self, block_ids, row_idx):
        _note(self, block_ids)
        return original_add(self, block_ids, row_idx)

    def commit_block_table(self, num_reqs):
        tr = STATE["translator"]
        flags = _gdn(self)
        if tr is None or not any(flags):
            for table in self.block_tables:
                table.commit_block_table(num_reqs)
            return
        tr.begin_step()
        if STATE.get("cache") is not None:
            STATE["cache"].begin_step(tr.step)
        rows = [t.block_table.np[:num_reqs] for t, g in zip(self.block_tables, flags) if g]
        tr.ensure(np.concatenate([r.reshape(-1) for r in rows]))
        for table, g in zip(self.block_tables, flags):
            if not g:
                table.commit_block_table(num_reqs)
                continue
            ring = getattr(table, "_arc_kvoff_staging", None)
            if ring is None:
                shape = tuple(table.block_table.cpu.shape)
                ring = [torch.zeros(shape, dtype=torch.int32, pin_memory=torch.cuda.is_available())
                        for _ in range(STAGING_RING)]
                table._arc_kvoff_staging = ring
                table._arc_kvoff_next = 0
            staging = ring[table._arc_kvoff_next]
            table._arc_kvoff_next = (table._arc_kvoff_next + 1) % STAGING_RING
            src = table.block_table.np[:num_reqs]
            mapped = tr.pmap[src]
            staging.numpy()[:num_reqs] = np.where(mapped >= 0, mapped, 0)  # stale ids -> the null slot
            table.block_table.gpu[:num_reqs].copy_(staging[:num_reqs], non_blocking=True)

    multi_cls.append_row = append_row
    multi_cls.add_row = add_row
    multi_cls.commit_block_table = commit_block_table
    multi_cls._arc_kvoff = True


# ------------------------------------------------------------------------------------------------ mamba copies
class _TranslatedState:
    def __init__(self, req_state, block_ids):
        self._req_state = req_state
        self.block_ids = block_ids

    def __getattr__(self, name):
        return getattr(self._req_state, name)


def install_mamba_utils(module):
    if not ENABLED or getattr(module, "_arc_kvoff", False):
        return
    original = module.collect_mamba_copy_meta

    def collect_mamba_copy_meta(copy_bufs, kv_cache_config, funcs, mamba_group_ids, src_block_idx, dest_block_idx,
                                accept_token_bias, req_state, forward_context):
        tr = STATE["translator"]
        if tr is None:
            return original(copy_bufs, kv_cache_config, funcs, mamba_group_ids, src_block_idx, dest_block_idx,
                            accept_token_bias, req_state, forward_context)
        ids = list(req_state.block_ids)
        for g in mamba_group_ids:
            tr.ensure(ids[g])
            ids[g] = [max(int(tr.pmap[b]), 0) if b > 0 else 0 for b in ids[g]]
        return original(copy_bufs, kv_cache_config, funcs, mamba_group_ids, src_block_idx, dest_block_idx,
                        accept_token_bias, _TranslatedState(req_state, tuple(ids)), forward_context)

    module.collect_mamba_copy_meta = collect_mamba_copy_meta
    module._arc_kvoff = True


# ================================================================================ milestone 2b: GPU group cache
# Fill-before-attend. Each main tensor (one QSA layer) gets a GPU cache of CACHE_SLOTS 4-token groups keyed by
# physical location (block id, group in block): remap[layer, key] -> slot (-1 none, -2 being filled this step).
# Per layer and step: (1) lookup stamps the hits and enqueues each missing stable group once; (2) claim takes a
# slot per miss (CLOCK: skip slots used this step, second chance on the reference bit) and copies the group's 8
# (token, head) records from host RAM; (3) attention (the served kernel with a two-source address) reads a pick
# from its slot when remap has one and from the host page otherwise. A group is cached only when its 4 tokens are
# final: below the request's sequence length for a prefill (query longer than DECODE_QLEN), and below
# seq_len - query_len - SAFETY for a decode/verify row (drafts and optimistic async lengths stay on the host).
# A block id appended to any non-GDN block table drops its groups (new content). Physical block 0 (null) is never
# cached. Same loads, same order, same arithmetic as the served kernel: output-exact.
import triton
import triton.language as tl

CACHE_SLOTS = int(os.environ.get("ARC_KVOFF_CACHE_SLOTS", "0") or 0)
DECODE_QLEN = int(os.environ.get("ARC_KVOFF_DECODE_QLEN", "4") or 4)
SAFETY = int(os.environ.get("ARC_KVOFF_SAFETY", "16") or 16)
MAX_PROBES = 4096
CLAIM_PROGRAMS = 1024
GROUP_TOKENS = 4
RECORD = 264  # [K 128 | K scale 4 | V 128 | V scale 4] per (token, head), int4
LOOKUP_BLOCK = 256


@triton.jit
def _kvoff_invalidate_kernel(remap_ptr, ids_ptr, layer_stride, GPB: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    layer = tl.program_id(1)
    block = tl.load(ids_ptr + i).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    base = remap_ptr + layer.to(tl.int64) * layer_stride + block * GPB
    for start in range(0, GPB, BLOCK):
        tl.store(base + start + cols, tl.full((BLOCK,), -1, tl.int32), mask=start + cols < GPB)


@triton.jit
def _kvoff_lookup_kernel(indices_ptr, block_table_ptr, token_to_req_ptr, stable_ptr, remap_ptr, stamp_ptr, ref_ptr,
                         step_ptr, count_ptr, miss_ptr, dummy_ptr,
                         stride_indices_row, stride_table_req, num_requests, num_cache_blocks, max_misses,
                         TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr,
                         GPB: tl.constexpr, GROUP_TOKENS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    step = tl.load(step_ptr)
    request = tl.load(token_to_req_ptr + row)
    req_ok = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    stable_len = tl.load(stable_ptr + safe_request)
    logical = tl.load(indices_ptr + row * stride_indices_row + cols, mask=cols < TOPK, other=-1)
    safe = tl.maximum(logical, 0)
    page = safe // PAGE_SIZE
    valid = req_ok & (logical >= 0) & (page < PAGE_TABLE_WIDTH)
    physical = tl.load(block_table_ptr + safe_request * stride_table_req + tl.minimum(page, PAGE_TABLE_WIDTH - 1),
                       mask=valid, other=-1)
    group = (safe % PAGE_SIZE) // GROUP_TOKENS
    group_end = page * PAGE_SIZE + (group + 1) * GROUP_TOKENS
    stable = valid & (physical > 0) & (physical < num_cache_blocks) & (group_end <= stable_len)
    key = tl.maximum(physical, 0).to(tl.int64) * GPB + group
    slot = tl.load(remap_ptr + key, mask=stable, other=-3)
    hit = stable & (slot >= 0)
    tl.store(stamp_ptr + tl.maximum(slot, 0), tl.zeros_like(slot) + step, mask=hit)
    tl.store(ref_ptr + tl.maximum(slot, 0), tl.full(slot.shape, 1, tl.int32), mask=hit)
    miss = stable & (slot == -1)
    old = tl.atomic_cas(tl.where(miss, remap_ptr + key, dummy_ptr + tl.zeros_like(key)), tl.where(miss, -1, -7),
                        tl.full(slot.shape, -2, tl.int32))
    owner = miss & (old == -1)
    idx = tl.atomic_add(count_ptr + tl.zeros_like(slot), tl.where(owner, 1, 0), mask=owner)
    keep = owner & (idx < max_misses)
    tl.store(miss_ptr + tl.maximum(idx, 0), key, mask=keep)
    tl.store(remap_ptr + key, tl.full(slot.shape, -1, tl.int32), mask=owner & (idx >= max_misses))


@triton.jit
def _kvoff_claim_kernel(base_ptr, host_delta, remap_ptr, owner_ptr, stamp_ptr, ref_ptr, hand_ptr, step_ptr,
                        count_ptr, miss_ptr, max_misses, num_slots,
                        stride_block, stride_token, stride_head,
                        GPB: tl.constexpr, PAGE_SIZE: tl.constexpr, GROUP_TOKENS: tl.constexpr,
                        RECORD: tl.constexpr, MAX_PROBES: tl.constexpr, PROGRAMS: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.load(step_ptr)
    n = tl.minimum(tl.load(count_ptr), max_misses)
    rec = tl.arange(0, 8)  # (token, head) records of a group: token = rec // 2, head = rec % 2
    words = tl.arange(0, 128)
    for i in range(pid, n, PROGRAMS):
        key = tl.load(miss_ptr + i)
        got = tl.full([], -1, tl.int64)
        probes = 0
        while (got < 0) & (probes < MAX_PROBES):
            h = (tl.atomic_add(hand_ptr, 1) % num_slots).to(tl.int64)
            st = tl.load(stamp_ptr + h, volatile=True)
            if st != step:
                r = tl.atomic_xchg(ref_ptr + h, 0)
                if r == 0:
                    if tl.atomic_cas(stamp_ptr + h, st, step) == st:
                        got = h
            probes += 1
        if got >= 0:
            prev = tl.load(owner_ptr + got)
            if prev >= 0:
                tl.atomic_cas(remap_ptr + prev, got.to(tl.int32), tl.full([], -1, tl.int32))
            tl.store(owner_ptr + got, key)
            block = key // GPB
            token = (key % GPB) * GROUP_TOKENS
            src = host_delta + block * stride_block + (token + rec // 2).to(tl.int64) * stride_token \
                + (rec % 2).to(tl.int64) * stride_head
            dst = got * (GROUP_TOKENS * 2 * RECORD) + rec.to(tl.int64) * RECORD
            m = words[None, :] < RECORD // 4
            data = tl.load((base_ptr + src[:, None]).to(tl.pointer_type(tl.int32)) + words[None, :], mask=m)
            tl.store((base_ptr + dst[:, None]).to(tl.pointer_type(tl.int32)) + words[None, :], data, mask=m)
            tl.atomic_xchg(remap_ptr + key, got.to(tl.int32))
        else:
            tl.atomic_xchg(remap_ptr + key, tl.full([], -1, tl.int32))


@triton.jit
def _kvoff_attn_kernel(
    q_ptr, base_ptr, indices_ptr, block_table_ptr, token_to_req_ptr, stable_ptr, remap_ptr,
    partial_output_ptr, partial_lse_ptr, output_ptr,
    host_delta, stride_block, stride_token, stride_head,
    stride_q_row, stride_q_head, stride_indices_row, stride_table_req, stride_output_row, stride_output_head,
    num_rows, num_cache_blocks, num_requests,
    TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr, GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr, NUM_SPLITS: tl.constexpr, NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GPB: tl.constexpr, GROUP_TOKENS: tl.constexpr,
    RECORD: tl.constexpr, INTERPRET: tl.constexpr = False,
) -> None:
    # arc_kv4._qsa_kv4_splitk_kernel (KEY_BITS 4) with a two-source address: the GPU cache slot when remap has one
    # (and the group is stable), the host page through UVA otherwise. Loads, order and arithmetic unchanged.
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    stable_len = tl.load(stable_ptr + safe_request)

    head_offsets = tl.arange(0, BLOCK_M)
    half_offsets = tl.arange(0, HEAD_DIM // 2)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    head_mask = head_offsets[:, None] < GROUP_SIZE
    q_rows = q_ptr + row * stride_q_row + (first_head + head_offsets[:, None]) * stride_q_head
    q_even = tl.load(q_rows + 2 * half_offsets[None, :], mask=head_mask, other=0.0)
    q_odd = tl.load(q_rows + 2 * half_offsets[None, :] + 1, mask=head_mask, other=0.0)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_even = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)
    acc_odd = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(indices_ptr + row * stride_indices_row + columns, mask=columns < TOPK, other=-1)
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (request >= 0) & (request < num_requests) & (logical_token >= 0) & (logical_page < PAGE_TABLE_WIDTH)
        physical_page = tl.load(
            block_table_ptr + safe_request * stride_table_req + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid, other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        group = page_offset // GROUP_TOKENS
        cacheable = valid & (physical_page > 0) & (logical_page * PAGE_SIZE + (group + 1) * GROUP_TOKENS <= stable_len)
        slot = tl.load(remap_ptr + safe_page * GPB + group, mask=cacheable, other=-1)
        host_off = host_delta + safe_page * stride_block + page_offset.to(tl.int64) * stride_token
        cache_off = (tl.maximum(slot, 0).to(tl.int64) * GROUP_TOKENS + page_offset % GROUP_TOKENS) * (2 * RECORD)
        token_off = tl.where(slot >= 0, cache_off, host_off) + kv_head * tl.where(slot >= 0, RECORD, stride_head)
        key_scale_raw = tl.load((base_ptr + token_off + 128).to(tl.pointer_type(tl.int32)), mask=valid, other=0)
        packed = tl.load(base_ptr + token_off[None, :] + half_offsets[:, None], mask=valid[None, :], other=0).to(tl.int32)
        key_zero = key_scale_raw & 0xF
        key_scale = (key_scale_raw & -16).to(tl.float32, bitcast=True)
        k_even = ((packed & 0xF) - key_zero[None, :]).to(tl.float32)
        k_odd = (((packed >> 4) & 0xF) - key_zero[None, :]).to(tl.float32)
        if INTERPRET:
            scores = tl.dot(q_even.to(tl.float32), k_even) + tl.dot(q_odd.to(tl.float32), k_odd)
        else:
            scores = tl.dot(q_even, k_even.to(tl.bfloat16)) + tl.dot(q_odd, k_odd.to(tl.bfloat16))
        scores *= key_scale[None, :] * softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0)
        value_packed = tl.load(base_ptr + token_off[:, None] + 132 + half_offsets[None, :], mask=valid[:, None],
                               other=0).to(tl.int32)
        value_scale_raw = tl.load((base_ptr + token_off + 260).to(tl.pointer_type(tl.int32)), mask=valid, other=0)
        value_zero = value_scale_raw & 0xF
        value_scale = (value_scale_raw & -16).to(tl.float32, bitcast=True)
        v_even = ((value_packed & 0xF) - value_zero[:, None]).to(tl.float32)
        v_odd = (((value_packed >> 4) & 0xF) - value_zero[:, None]).to(tl.float32)
        weights = (probabilities * value_scale[None, :]).to(tl.bfloat16)
        if INTERPRET:
            acc_even = tl.dot(weights.to(tl.float32), v_even, acc=acc_even * alpha[:, None])
            acc_odd = tl.dot(weights.to(tl.float32), v_odd, acc=acc_odd * alpha[:, None])
        else:
            acc_even = tl.dot(weights, v_even.to(tl.bfloat16), acc=acc_even * alpha[:, None])
            acc_odd = tl.dot(weights, v_odd.to(tl.bfloat16), acc=acc_odd * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    denominator = tl.maximum(normalizer[:, None], 1.0e-20)
    out_even = tl.where(has_values[:, None], acc_even / denominator, 0.0)
    out_odd = tl.where(has_values[:, None], acc_odd / denominator, 0.0)
    even_columns = 2 * half_offsets[None, :]
    if NUM_SPLITS == 1:
        out_rows = output_ptr + row * stride_output_row + (first_head + head_offsets[:, None]) * stride_output_head
        tl.store(out_rows + even_columns, out_even, mask=head_mask)
        tl.store(out_rows + even_columns + 1, out_odd, mask=head_mask)
    else:
        partial_lse = tl.where(has_values, max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)), -float("inf"))
        partial_rows = partial_output_ptr + ((split_id * num_rows + row) * NUM_QUERY_HEADS + first_head
                                             + head_offsets[:, None]) * HEAD_DIM
        tl.store(partial_rows + even_columns, out_even, mask=head_mask)
        tl.store(partial_rows + even_columns + 1, out_odd, mask=head_mask)
        tl.store(partial_lse_ptr + (split_id * num_rows + row) * NUM_QUERY_HEADS + first_head + head_offsets,
                 partial_lse, mask=head_offsets < GROUP_SIZE)


class GroupCache:
    """The per-layer GPU group caches over the host main tensors (one per tensor), plus shared step state."""

    def __init__(self, regions, num_blocks, page_tokens, slots, device):
        self.regions = regions  # [(host view data_ptr, nbytes)]
        self.layers = len(regions)
        self.gpb = page_tokens // GROUP_TOKENS
        self.num_blocks, self.slots, self.device = num_blocks, slots, device
        self.remap = torch.full((self.layers, num_blocks * self.gpb), -1, dtype=torch.int32, device=device)
        self.owner = torch.full((self.layers, slots), -1, dtype=torch.int64, device=device)
        self.stamp = torch.zeros((self.layers, slots), dtype=torch.int32, device=device)
        self.ref = torch.zeros((self.layers, slots), dtype=torch.int32, device=device)
        self.hand = torch.zeros((self.layers, 1), dtype=torch.int64, device=device)
        self.count = torch.zeros((self.layers, 1), dtype=torch.int32, device=device)
        self.miss = torch.zeros((self.layers, slots), dtype=torch.int64, device=device)
        self.cache = torch.zeros((self.layers, slots * GROUP_TOKENS * 2 * RECORD), dtype=torch.uint8, device=device)
        self.step = torch.zeros(1, dtype=torch.int32, device=device)
        self.dummy = torch.zeros(1, dtype=torch.int32, device=device)
        self.invalid = []
        self.ids_staging = None

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.remap, self.owner, self.stamp, self.ref, self.miss,
                                                            self.cache))

    def layer_of(self, ptr):
        for i, (start, size) in enumerate(self.regions):
            if start <= ptr < start + size:
                return i
        return None

    def begin_step(self, step):
        self.step.fill_(step)
        if self.invalid:
            ids = np.unique(np.asarray(self.invalid, dtype=np.int32))
            ids = ids[(ids > 0) & (ids < self.num_blocks)]
            self.invalid = []
            if len(ids):
                staging = torch.from_numpy(ids).pin_memory() if torch.cuda.is_available() else torch.from_numpy(ids)
                dev = staging.to(self.device, non_blocking=True)
                self.ids_staging = (self.ids_staging or [])[-7:] + [(staging, dev)]  # alive until the copies ran
                _kvoff_invalidate_kernel[(len(ids), self.layers)](self.remap, dev, self.remap.stride(0),
                                                                   GPB=self.gpb, BLOCK=1024)


def _stable_lengths(block_table):
    """seq_len (prefill) or seq_len - query_len - SAFETY (decode/verify) per request, from the forward context's
    metadata that owns this block table; None if it is not found (every pick then reads the host)."""
    from vllm.forward_context import get_forward_context

    try:
        metadata = get_forward_context().attn_metadata
    except Exception:
        return None
    groups = metadata if isinstance(metadata, list) else [metadata]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for m in group.values():
            if getattr(m, "block_table", None) is block_table and getattr(m, "seq_lens", None) is not None:
                qsl = m.query_start_loc
                n = block_table.shape[0]
                seq = m.seq_lens[:n].to(torch.int32)
                qlen = (qsl[1:n + 1] - qsl[:n]).to(torch.int32)
                return torch.where(qlen > DECODE_QLEN, seq, seq - qlen - SAFETY)
    return None


def install_kv4(module):
    """Wraps arc_kv4.sparse_attention: picks over a host main tensor go through the group cache."""
    if not ENABLED or not CACHE_SLOTS or getattr(module, "_arc_kvoff", False):
        return
    original = module.sparse_attention
    kv8 = module._kv8

    def sparse_attention(q, k_codes, v_codes, k_scale, v_scale, logical_indices, block_table, token_to_req, out,
                         key_bits, profile=None):
        gc = STATE.get("cache")
        layer = gc.layer_of(k_codes.data_ptr()) if gc is not None else None
        if layer is None or key_bits != 4 or not q.shape[0]:
            return original(q, k_codes, v_codes, k_scale, v_scale, logical_indices, block_table, token_to_req, out,
                            key_bits, profile)
        stable = _stable_lengths(block_table)
        if stable is None:
            if "no_metadata" not in STATE["logged"]:
                STATE["logged"].add("no_metadata")
                log("no_metadata", {"layer": layer})
            stable = torch.full((block_table.shape[0],), -1, dtype=torch.int32, device=q.device)
        if k_codes.stride(3) != 1 or v_codes.data_ptr() != k_codes.data_ptr() + 132:
            raise RuntimeError("ARC_KVOFF: unexpected int4 page layout")
        rows, topk = logical_indices.shape
        num_requests, width = block_table.shape
        page = k_codes.shape[1]
        num_cache_blocks = k_codes.shape[0]
        gpb = page // GROUP_TOKENS  # groups per kernel block (the manager block may hold several)
        if page % GROUP_TOKENS or num_cache_blocks * gpb != gc.num_blocks * gc.gpb:
            raise RuntimeError(f"ARC_KVOFF: kernel pages {num_cache_blocks}x{page} do not tile the manager blocks")
        remap, cache = gc.remap[layer], gc.cache[layer]
        host_delta = k_codes.data_ptr() - cache.data_ptr()
        sb, st, sh = k_codes.stride(0), k_codes.stride(1), k_codes.stride(2)
        gc.count[layer].zero_()
        _kvoff_lookup_kernel[(rows, triton.cdiv(topk, LOOKUP_BLOCK))](
            logical_indices, block_table, token_to_req, stable, remap, gc.stamp[layer], gc.ref[layer], gc.step,
            gc.count[layer], gc.miss[layer], gc.dummy,
            logical_indices.stride(0), block_table.stride(0), num_requests, num_cache_blocks, gc.slots,
            TOPK=topk, PAGE_SIZE=page, PAGE_TABLE_WIDTH=width, GPB=gpb, GROUP_TOKENS=GROUP_TOKENS,
            BLOCK=LOOKUP_BLOCK)
        _kvoff_claim_kernel[(CLAIM_PROGRAMS,)](
            cache, host_delta, remap, gc.owner[layer], gc.stamp[layer], gc.ref[layer], gc.hand[layer], gc.step,
            gc.count[layer], gc.miss[layer], gc.slots, gc.slots, sb, st, sh,
            GPB=gpb, PAGE_SIZE=page, GROUP_TOKENS=GROUP_TOKENS, RECORD=RECORD, MAX_PROBES=MAX_PROBES,
            PROGRAMS=CLAIM_PROGRAMS)
        group_size = q.shape[1] // k_codes.shape[2]
        p = profile or kv8.launch_profile(rows, k_codes.shape[2], group_size, topk)
        if p["splits"] == 1:
            partial_output, partial_lse = out, out
        else:
            partial_output = torch.empty((p["splits"], *q.shape), dtype=torch.float32, device=q.device)
            partial_lse = torch.empty((p["splits"], rows, q.shape[1]), dtype=torch.float32, device=q.device)
        _kvoff_attn_kernel[(rows, k_codes.shape[2], p["splits"])](
            q, cache, logical_indices, block_table, token_to_req, stable, remap,
            partial_output, partial_lse, out,
            host_delta, sb, st, sh,
            q.stride(0), q.stride(1), logical_indices.stride(0), block_table.stride(0), out.stride(0), out.stride(1),
            rows, num_cache_blocks, num_requests,
            TOPK=topk, PAGE_SIZE=page, PAGE_TABLE_WIDTH=width, GROUP_SIZE=group_size, HEAD_DIM=module.HEAD_DIM,
            NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=p["splits"], NUM_TILES=p["tiles"], BLOCK_M=p["block_m"],
            BLOCK_N=p["block_n"], GPB=gpb, GROUP_TOKENS=GROUP_TOKENS, RECORD=RECORD,
            INTERPRET=module._interpreting(), num_warps=p["warps"], num_stages=2)
        if p["splits"] > 1:
            kv8._qsa_kv8_merge_kernel[(rows, q.shape[1])](
                partial_output, partial_lse, out, out.stride(0), out.stride(1), rows,
                HEAD_DIM=module.HEAD_DIM, NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=p["splits"],
                BLOCK_SPLITS=triton.next_power_of_2(p["splits"]), num_warps=2, num_stages=1)
        return out

    module.sparse_attention = sparse_attention
    module._arc_kvoff = True
    log("installed_kv4", {"cache_slots": CACHE_SLOTS, "decode_qlen": DECODE_QLEN, "safety": SAFETY})


# ================================================================================ V2 model runner (v1/worker/gpu)
# The served runtime uses the V2 runner: attn_utils._allocate_kv_cache allocates, BlockTables keeps per-request
# rows on the GPU (staged writes) and gathers the batch's rows into persistent input tables in prepare_attn. GDN
# rows are translated there, on the gathered tables (mamba pre/post-processing and the forward read those).
GDN_TAIL = int(os.environ.get("ARC_KVOFF_GDN_TAIL", "4") or 4)  # columns a step can read: prev, current, spec


def install_attn_utils_v2(module):
    if not ENABLED or getattr(module, "_arc_kvoff", False):
        return
    if not (TOTAL or FACTOR) or not DEV:
        raise RuntimeError("ARC_KVOFF needs ARC_KVOFF_TOTAL_BLOCKS and ARC_KVOFF_DEV_BLOCKS")

    def _allocate_kv_cache(kv_cache_config, shared_layers, device):
        return _allocate(kv_cache_config, device, set(shared_layers))

    module._allocate_kv_cache = _allocate_kv_cache
    module._arc_kvoff = True
    log("installed", {"runner": "v2", "total_blocks": TOTAL or f"x{FACTOR}", "dev_blocks": DEV})


def install_block_tables_v2(cls):
    if not ENABLED or getattr(cls, "_arc_kvoff", False):
        return
    original = cls.append_block_ids

    def append_block_ids(self, req_index, new_block_ids, overwrite):
        tr = STATE.get("translator")
        if tr is not None:
            groups = STATE.get("gdn_groups") or set()
            mirror = self.__dict__.setdefault("_arc_kvoff_gdn", {})
            row = mirror.get(req_index) if not overwrite else None
            row = row if row is not None else {}
            for i, ids in enumerate(new_block_ids):
                if not ids:
                    continue
                if i in groups:
                    if self.blocks_per_kv_block[i] != 1:
                        raise RuntimeError("ARC_KVOFF: a GDN block table maps one block id to several kernel blocks")
                    tr.own(ids)
                    row[i] = (row.get(i, []) + [int(b) for b in ids if b > 0])[-GDN_TAIL:]
                else:
                    tr.drop(ids)
                    if STATE.get("cache") is not None:
                        STATE["cache"].invalid.extend(ids)
            mirror[req_index] = row
        return original(self, req_index, new_block_ids, overwrite)

    cls.append_block_ids = append_block_ids
    cls._arc_kvoff = True


def install_runner_v2(cls):
    if not ENABLED or getattr(cls, "_arc_kvoff_v2", False):
        return
    original = cls.prepare_attn

    def prepare_attn(self, input_batch):
        tr = STATE.get("translator")
        if tr is None:
            return original(self, input_batch)
        tr.begin_step()
        if STATE.get("cache") is not None:
            STATE["cache"].begin_step(tr.step)
        mirror = self.block_tables.__dict__.get("_arc_kvoff_gdn", {})
        ids = [b for r in np.asarray(input_batch.idx_mapping_np).tolist()
               for tail in mirror.get(int(r), {}).values() for b in tail]
        tr.ensure(np.asarray(ids, dtype=np.int64))
        dev = STATE.get("pmap_dev")
        if dev is None:
            dev = STATE["pmap_dev"] = torch.zeros(len(tr.pmap), dtype=torch.int32, device=self.device)
            STATE["pmap_ring"] = [torch.zeros(len(tr.pmap), dtype=torch.int32, pin_memory=True)
                                  for _ in range(STAGING_RING)]
            STATE["pmap_next"] = 0
        staging = STATE["pmap_ring"][STATE["pmap_next"]]
        STATE["pmap_next"] = (STATE["pmap_next"] + 1) % STAGING_RING
        np.copyto(staging.numpy(), np.where(tr.pmap >= 0, tr.pmap, 0), casting="unsafe")
        dev.copy_(staging, non_blocking=True)
        block_tables, slot_mappings = original(self, input_batch)
        for g in STATE.get("gdn_groups") or ():
            bt = block_tables[g]
            bt.copy_(dev[bt.long().clamp_(0, len(tr.pmap) - 1)])
        return block_tables, slot_mappings

    cls.prepare_attn = prepare_attn
    cls._arc_kvoff_v2 = True
    module = __import__("sys").modules[cls.__module__]
    if hasattr(module, "copy_kv_cache_blocks_inplace"):
        module.copy_kv_cache_blocks_inplace = _wrap_block_copies(module.copy_kv_cache_blocks_inplace)
    if hasattr(module, "set_dummy_context"):
        original_dummy = module.set_dummy_context

        def set_dummy_context(input_batch, block_tables, *args, **kwargs):
            # Fabricated ids run up to num_blocks; the GDN pools hold DEV pages: point dummy GDN rows at the null page.
            out = original_dummy(input_batch, block_tables, *args, **kwargs)
            for g in STATE.get("gdn_groups") or ():
                block_tables.input_block_tables[g][:input_batch.num_reqs].zero_()
            return out

        module.set_dummy_context = set_dummy_context


def _wrap_block_copies(original):
    """Copy-on-write block copies (partial prefix hits): the host and device tensors as upstream; the GDN device
    pools (DEV pages, not num_blocks) through the translation."""

    def copy_kv_cache_blocks_inplace(kv_caches, num_blocks, kv_cache_block_copies):
        tr = STATE.get("translator")
        if tr is None or not kv_cache_block_copies:
            return original(kv_caches, num_blocks, kv_cache_block_copies)
        pools = {dev.untyped_storage().data_ptr() for _, dev, _ in tr.tensors if dev is not None}
        kept = []
        for entry in kv_caches:
            tensors = entry if isinstance(entry, (list, tuple)) else (entry,)
            rest = [t for t in tensors if t.untyped_storage().data_ptr() not in pools]
            if rest:
                kept.append(rest if isinstance(entry, (list, tuple)) else rest[0])
        original(kept, num_blocks, kv_cache_block_copies)
        gdn = [(int(a), int(b)) for a, b in kv_cache_block_copies if tr.gdn[int(a)] or tr.gdn[int(b)]]
        for src, dst in gdn:
            tr.ensure(np.array([src, dst], dtype=np.int64))
            a, b = int(tr.pmap[src]), int(tr.pmap[dst])
            if a > 0 and b > 0 and a != b:
                for _, dev, page in tr.tensors:
                    if dev is not None:
                        dev[b * page:(b + 1) * page].copy_(dev[a * page:(a + 1) * page])
        tr.stats["cow_gdn"] = tr.stats.get("cow_gdn", 0) + len(gdn)
        tr.stats["cow"] = tr.stats.get("cow", 0) + len(kv_cache_block_copies)

    return copy_kv_cache_blocks_inplace
