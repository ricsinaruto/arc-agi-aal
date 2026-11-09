"""Serving ablation on the v8.2 harness: which part of our serving costs score per token.

    python scripts/dfr_exact/build_v82_ablation.py OUT --v82 <v82tc-w128 bundle> --arm base|R|P|K [--rl <rl-pool bundle>]
        [--instances 100] [--stop-tokens 75000] [--wall SECONDS]

Each arm is the v82tc-w128 bundle with one serving change, played on c100 with the equal scheduler (A v2's
``priority.equal``, no parking, no level cap, no tail fade); an instance finishes at ``--stop-tokens`` generated tokens
and the trial ends when every instance has finished (``study.token_stop``; the wall is only a backstop).

    base  unchanged (t37 prune + int3 g16 overlay, Humming), 48 streams
    R     all 512 experts, stock NVFP4, Marlin: the RL arm's reference profile server block (rl-pool-v1-dev-p1), at most
          21 streams, GPU memory 0.96 as there
    P     t37 prune, stock NVFP4 kept experts (no int3 overlay), Marlin
    K     base + 8-bit FP8 KV (kv_cache_8bit fp8) + BF16 indexer keys
    Kb base + BF16 indexer keys only (int4 KV kept), 48 streams
    Kf    base + FP8 KV only (FP8 indexer keys kept), 27 streams with the gate density rescaled to the FP8 pool
          (72.888 gate blocks per stream: v8.2's 41.666 x the 1.711 block-cost ratio; K at 47 slots over-admitted)
    O     base with the long run's old priority scheduler (priority.equal off: dfranzen's priority value at hand-overs,
          level cap 300K as the long run; tail fade off: it starts at 80% of a 24 h wall in the long run, never within
          100K per instance)
    AR    R with Intel's AutoRound W4A16 checkpoint (his weights: all 512 routed experts W4A16 group 128, dense/attention
          /indexer/MTP BF16) through arena.research_checkpoint (port of the 5 Oct registry); the bucket's verified copy
          of dfranzen's Kaggle input (study nebius_assets); at most 21 streams as R
    D     base + BF16 dense projections (no int8 dense weights; the freed KV shrinks, the density rule lowers streams)
P and K take at most 48 streams and R 21; the served pool lowers it (arena.capacity.competition_focus_fit).
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import build_v82_his_serving as his

SKIP = ("dataset-upload", "experiment.ipynb", "kernel-metadata.json", "MANIFEST.json", "__pycache__")
INT3 = "richardcsaky/arc-expert-int3-requant-t37-excl5-v1"
# Pruning-level arms: the all-expert INT3 overlay fitted on excl5 (unmasked W4 GPTQ fit, then the same
# g16 RTN-search requant as t37's) that the t37 mask was screened on; masks from the same excl5 usage rows (nested).
INT3_ALL = {"dataset": "richardcsaky/arc-expert-int3-requant-all-excl5-v1",
            "manifest_sha256": "664e61d1cda3fbd343b38512f917823486edd1e64ba700cb1ee2afcfeaf1dde5", "check": "size"}
LEVELS = {"T25": ("prune-t25-int3-requant-excl5-v1.json", 384), "T12": ("prune-t12-int3-requant-excl5-v1.json", 448),
          "T50": ("prune-t50-excl5-v1.json", 256),
          "T62": ("prune-t62-excl5-v1.json", 192)}  # T62: 2K track (18 Oct), same excl5 usage rows  # T50: 2K track Gi3 (16 Oct), same excl5 usage rows
# Item 4 (21 Oct): x5's INT3 overlay with the 160 coldest kept experts per layer (excl5 capture row counts) on the
# INT2 grid, hot experts byte-identical (throughput-arm f43cb906 keep_hot; Modal tp-int2cold-keephot-x5-v1-20261021).
INT2COLD = {"dataset": "richardcsaky/arc-expert-int3-t37-int2cold-keephot-excl5-v1",
            "manifest_sha256": "5672c7d38546270e423bd338e7dc8eaa1bf5f1112be7bd91e5fd819341e77f18", "check": "size"}
# INT2-cold overlay per served INT3 overlay: x5 (requant excl5) and v85f (direct all25, 21 Oct; coldest 160 kept
# experts on the all25 capture, Modal tp-int2cold-keephot-v85f-v1-20261021).
INT2COLD_FOR = {INT3: INT2COLD,
                "richardcsaky/arc-expert-int3-direct-t37-all25v71b-v1": {
                    "dataset": "richardcsaky/arc-expert-int3-direct-t37-all25-int2cold-v1",
                    "manifest_sha256": "3721889420dd50b0929c124fc1c4b06b792a882b5af599b309def653add4f0dc", "check": "size"}}
COMBOS = {"DKbT12": ("DKb", "T12"), "CGT4": ("CG", "TICK4")}  # 300K throughput combos: parts applied in order
OFFLOAD_PATCHES = [
    ("    if type(cpu_gib) is not int or not 0 <= cpu_gib <= 32:\n        raise ValueError('Arena CPU cache must be 0..32 GiB')",
     "    cpu_gib = int(os.environ.get('ARENA_CPU_CACHE_GIB', cpu_gib))  # per-trial override (offload probes)\n"
     "    if type(cpu_gib) is not int or not 0 <= cpu_gib <= 128:\n        raise ValueError('Arena CPU cache must be 0..128 GiB')"),
    ("        result = extract(*args, **kwargs)\n        apply(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working,",
     "        result = extract(*args, **kwargs)\n"
     "        if cpu_gib:  # the hybrid-cache offloading connector (arc_infra.arc_offload_runtime, arc2 7ccf5cd0)\n"
     "            target = recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages/vllm/arc_offload.py'\n"
     "            target.parent.mkdir(parents=True, exist_ok=True)\n"
     "            target.write_bytes((Path(__file__).parents[1] / 'arc_infra/arc_offload_runtime.py').read_bytes())\n"
     "        apply(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working,"),
    ("                'kv_connector': 'OffloadingConnector', 'kv_role': 'kv_both',",
     "                'kv_connector': 'ArcOffloadingConnector', 'kv_connector_module_path': 'vllm.arc_offload',\n"
     "                'kv_role': 'kv_both',"),
]
# Expert pruning on the AutoRound research checkpoint: the loader overlay renames
# model.language_model.layers.L.mlp.experts.E.* (AutoRound qweight/qzeros/scales match) and slices mlp.gate.weight;
# the MTP draft builds from its own config. arena.serving refused the pair as a guard only.
PRUNE_GUARD_OLD = "    if research_checkpoint is not None and (expert_overlay is not None or expert_prune is not None):"
PRUNE_GUARD_NEW = "    if research_checkpoint is not None and expert_overlay is not None:"
# A pruned pool (~2-3M tokens) holds the whole startup probe (25 sessions to ~21K tokens), so nothing is evicted and
# nothing comes back from host RAM: the round-trip demand fails although the connector is healthy (vo24p25 v1).
# Recorded, not required, there (EQOFF v4/v5 cover the offload path).
PROBE_RT_OLD = """                if record['cpu_load_bytes'] <= 0:
                    raise RuntimeError('CPU cache configured but no round-trip load demonstrated')"""
PROBE_RT_NEW = """                if record['cpu_load_bytes'] <= 0:
                    record['cpu_load_note'] = 'no host load: the GPU pool held every probe prefix'"""
# Batch-aware expert routing (OEA, arXiv 2511.02237): vllm/arc_oea.py hooked into the pinned
# FusedTopKRouter at import, after every other overlay; k0 from ARC_OEA_K0 in the trial's server environment.
OEA_OLD = """        return result

    recipe.verify_and_extract_runtime = wrapped_extract"""
OEA_NEW = """        oea = Path(__file__).parents[1] / 'arc_infra/oea_runtime.py'
        if oea.exists():
            site = recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages/vllm'
            (site / 'arc_oea.py').write_bytes(oea.read_bytes())
            router = site / 'model_executor/layers/fused_moe/router/fused_topk_router.py'
            text = router.read_text()
            if 'arc_oea' not in text:
                router.write_text(text + '\\nfrom vllm.arc_oea import install as _arc_oea_install\\n'
                                  '_arc_oea_install(FusedTopKRouter)\\n')
        return result

    recipe.verify_and_extract_runtime = wrapped_extract"""
QSAT_NEW = """        trace = Path(__file__).parents[1] / 'arc_infra/qsa_trace.py'
        if trace.exists():
            site = recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages/vllm'
            (site / 'arc_qsa_trace.py').write_bytes(trace.read_bytes())
            indexer = site / 'models/qwen3_8_flash_next/nvidia/indexer_qsa.py'
            text = indexer.read_text()
            if 'arc_qsa_trace' not in text:
                indexer.write_text(text + '\\nfrom vllm.arc_qsa_trace import install as _arc_qsa_trace_install\\n'
                                   '_arc_qsa_trace_install(QSAIndexer)\\n')
        return result

    recipe.verify_and_extract_runtime = wrapped_extract"""
# Fewer QSA selections (18 Oct, 2K track; lossy): ARC_QSA_TOPK_TOKENS sets the indexer budget below the checkpoint's
# 2,048 tokens; the pinned top-k kernels serve 512/1,024/2,048 blocks, so other block counts take a torch top-k over
# the visible blocks (edited into the runtime after every other overlay).
QTOPK_NEW = """        site = recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia'
        ops = site / 'ops/qsa.py'
        text = ops.read_text()
        old = '        topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)\\n'
        if 'ARC_QSA_TOPK_TOKENS' not in text:
            if text.count(old) != 1:
                raise RuntimeError('ARC_QSA_TOPK_TOKENS: qsa.py top-k anchor not found once')
            text = text.replace(old, (
                '        if block_topk in (512, 1024, 2048):\\n'
                '            topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)\\n'
                '        else:  # ARC_QSA_TOPK_TOKENS (fewer selections): torch top-k over the visible blocks\\n'
                '            _lg = logits[:, :columns]\\n'
                '            _cols = torch.arange(columns, device=_lg.device)\\n'
                '            _lg = _lg.masked_fill(_cols[None, :] >= visible_blocks[:, None], float(\\"-inf\\"))\\n'
                '            _vals, _idx = _lg.topk(min(block_topk, columns), dim=1)\\n'
                '            blocks.fill_(-1)\\n'
                '            blocks[:, :_idx.shape[1]] = torch.where(torch.isinf(_vals), -1, _idx).to(torch.int32)\\n'))
            ops.write_text(text)
        ix = site / 'indexer_qsa.py'
        text = ix.read_text()
        old = '        self.token_topk = int(config.indexer_budget)\\n'
        if 'ARC_QSA_TOPK_TOKENS' not in text:
            if text.count(old) != 1:
                raise RuntimeError('ARC_QSA_TOPK_TOKENS: indexer budget anchor not found once')
            ix.write_text(text.replace(old, "        self.token_topk = int(__import__('os').environ.get("
                                            "'ARC_QSA_TOPK_TOKENS') or config.indexer_budget)\\n"))
        return result

    recipe.verify_and_extract_runtime = wrapped_extract"""
# Sorted QSA selections (21 Oct, P0 nondeterminism): the persistent top-k writes the selected blocks in an
# unspecified order, and the online softmax casts each tile's weights to BF16 against the running max, so the same
# set read in a different order gives a different output (in-place re-run rel 5.8e-3 on every QSA layer). Sorting each
# row's blocks (-1 padding last) fixes the read order: same set, same output.
# P0 residual (21 Oct, b2-det-v85p-x5sbi-fill-v1): the first divergence between cold passes is a QSA indexer whose
# input is bit-equal but whose selected set differs: persistent_topk breaks exact score ties at the k-th value in no
# fixed order. Keep every block above the k-th score, then the lowest-index blocks equal to it; output ascending, -1 last.
QSA_TIE_FN = """

def _arc_qsa_tie_fix(logits, visible, blocks):
    k, n = blocks.shape[1], logits.shape[1]
    col = torch.arange(n, device=logits.device, dtype=torch.int32).unsqueeze(0)
    live = col < visible.to(torch.int32).unsqueeze(1)
    sel = blocks.long()
    vals = torch.where(sel >= 0, logits.gather(1, sel.clamp(min=0)), torch.full(sel.shape, float('inf'), device=logits.device))
    t = vals.min(dim=1, keepdim=True).values
    gt = live & (logits > t)
    eq = live & (logits == t)
    need = k - gt.sum(dim=1, keepdim=True, dtype=torch.int32)
    take = gt | (eq & (eq.cumsum(dim=1, dtype=torch.int32) <= need))
    first = torch.where(take, col, n).topk(k, dim=1, largest=False, sorted=True).values
    blocks.copy_(torch.where(first >= n, -1, first).to(blocks.dtype))
"""
QSORT_BODY = """        site = recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia'
        ops = site / 'ops/qsa.py'
        text = ops.read_text()
        old = '        topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)\\n'
        if 'ARC_QSA_SORT' not in text:
            if text.count(old) != 1:
                raise RuntimeError('ARC_QSA_SORT: qsa.py top-k anchor not found once')
            ops.write_text(text.replace(old, old + (
                '        if __import__(\"os\").environ.get(\"ARC_QSA_TIE\") == \"1\":  # deterministic k-th ties (P0, 21 Oct)\\n'
                '            _arc_qsa_tie_fix(logits, visible_blocks, blocks)\\n'
                '        elif __import__(\"os\").environ.get(\"ARC_QSA_SORT\") == \"1\":  # fixed read order (P0, 21 Oct)\\n'
                '            _b = torch.where(blocks < 0, 2147483647, blocks).sort(dim=1).values\\n'
                '            blocks.copy_(torch.where(_b == 2147483647, -1, _b))\\n')) + QSA_TIE_SRC)
"""
QSORT_BODY = QSORT_BODY.replace("QSA_TIE_SRC", repr(QSA_TIE_FN))
# Deterministic Humming MoE (21 Oct, P0): Humming's heuristic configs in batch-invariant mode (no split-K / stream-K
# fix-up order dependence) for the MoE only, without vLLM's global VLLM_BATCH_INVARIANT.
HBI_BODY = """        moe = recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py'
        text = moe.read_text()
        old = 'use_batch_invariant=envs.VLLM_BATCH_INVARIANT,'
        if 'ARC_HUMMING_BI' not in text:
            if text.count(old) != 2:
                raise RuntimeError('ARC_HUMMING_BI: humming heuristics anchor not found twice')
            moe.write_text(text.replace(old, 'use_batch_invariant=envs.VLLM_BATCH_INVARIANT or __import__(\\"os\\").environ.get(\\"ARC_HUMMING_BI\\") == \\"1\\",'))
"""
EXTRACT_TAIL = """        return result

    recipe.verify_and_extract_runtime = wrapped_extract"""
# KV offload (18 Oct, milestone 2a; output-exact): kvoff/kvoff_runtime.py -> vllm/arc_kvoff.py, hooked into the block
# count, the KV allocation, the block tables and the mamba copy path; off unless ARC_KVOFF=1 (server env).
KVOFF_NEW = """        kvoff = Path(__file__).parents[1] / 'arc_infra/kvoff_runtime.py'
        if kvoff.exists():
            site = recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages/vllm'
            (site / 'arc_kvoff.py').write_bytes(kvoff.read_bytes())
            for rel, line in (('v1/core/kv_cache_utils.py', 'install_kv_cache_utils(__import__("sys").modules[__name__])'),
                              ('v1/worker/gpu_model_runner.py', 'install_runner(GPUModelRunner)'),
                              ('v1/worker/block_table.py', 'install_block_table(MultiGroupBlockTable)'),
                              ('v1/worker/mamba_utils.py', 'install_mamba_utils(__import__("sys").modules[__name__])'),
                              ('v1/worker/gpu/attn_utils.py', 'install_attn_utils_v2(__import__("sys").modules[__name__])'),
                              ('v1/worker/gpu/block_table.py', 'install_block_tables_v2(BlockTables)'),
                              ('v1/worker/gpu/model_runner.py', 'install_runner_v2(GPUModelRunner)')):
                path = site / rel
                text = path.read_text()
                if 'arc_kvoff' not in text:
                    name = line.split('(')[0]
                    path.write_text(text + f'\\nfrom vllm.arc_kvoff import {name} as _arc_kvoff_{name}\\n'
                                    + f'_arc_kvoff_{line}\\n')
        return result

    recipe.verify_and_extract_runtime = wrapped_extract"""
# GDN deferred commit (19 Oct, M2a; exact): gdn/arc_gdn_deferred.py -> vllm/arc_gdn_deferred.py + gdn_m2.cu built on
# the VM with the runtime's nvcc and --use_fast_math (M1: only the fast-math build matches the served op bit for bit), hooked into the Qwen GDN layer (spec paths) and the GDN metadata builder; ARC_GDN_DEFER=real|check.
# Tail entries for MTP mid-span stops (19 Oct; arc2 + throughput: end-hit 39-41% under MTP3 vs 96% at MTP1). After the
# stop token the continuation (newline, <|im_start|>, ...) is so predictable that the step accepts all drafts, so the
# stop lands mid-span and the in-flight step processes the m dropped tokens' bonus b: its running slot holds the state
# after the kept tokens + all m dropped tokens (d_{j+1}..d_k, b), as in the one-token spec_dropped_bonus case. The entry
# ends there with the dropped tokens as its last m tokens and is served when the next prompt continues with them.
GATEKV_PATCHES = [  # S1b (22 Oct): KV-metric gate, off unless the study sets gate_kv
    ("gate.py", """                 opening_burst_requests=1, opening=None, lane=None, background_slots=0):
""", """                 opening_burst_requests=1, opening=None, lane=None, background_slots=0, kv=None):
"""),
    ("gate.py", """        self.opening_burst_requests = opening_burst_requests
""", """        self.opening_burst_requests = opening_burst_requests
        # KV-metric gate (S1b): a measured per-request block overhead the layout's cost table misses (MTP3 ~2),
        # and no admission while vLLM's own KV usage is above ``limit`` (fresh 1 s sample, >= 1 request in flight).
        kv = kv or {}
        self.overhead_units = int(kv.get("overhead_units", 0))
        self.kv_limit = float(kv.get("limit", 0))
        self.kv_usage = self.kv_at = None
        self.kv_refusals = 0
"""),
    ("gate.py", """        return self.budget.cost(tokens) if self.budget else int(min(tokens, self.capacity))
""", """        return self.budget.cost(tokens) + self.overhead_units if self.budget else int(min(tokens, self.capacity))

    def _kv_hot(self):
        return (self.kv_limit > 0 and self.kv_usage is not None and self.now() - self.kv_at < 3.0
                and self.kv_usage > self.kv_limit)

    def observe_kv(self, usage):
        self.kv_usage, self.kv_at = float(usage), self.now()
        if not self._kv_hot():
            self._pump()

    async def kv_poll(self, metrics_url, interval=1.0):
        import httpx
        async with httpx.AsyncClient(trust_env=False) as client:
            while True:
                try:
                    r = await client.get(metrics_url, timeout=2)
                    for line in r.text.splitlines():
                        if line.startswith("vllm:kv_cache_usage_perc"):
                            self.observe_kv(float(line.rsplit(" ", 1)[1]))
                            break
                except Exception:
                    pass
                await asyncio.sleep(interval)
"""),
    ("gate.py", """        return inflight < self.slots and reserved + tokens <= self.capacity
""", """        if inflight > 0 and self._kv_hot():
            self.kv_refusals += 1
            return False
        return inflight < self.slots and reserved + tokens <= self.capacity
"""),
    ("gate.py", """        tokens = self.budget.cost(requested_tokens) if self.budget else int(min(tokens, self.capacity))""",
     """        tokens = self.budget.cost(requested_tokens) + self.overhead_units if self.budget else int(min(tokens, self.capacity))"""),
    ("gate.py", """            "affinity_admissions": self.affinity_admissions,
""", """            "affinity_admissions": self.affinity_admissions,
            **({"kv_limit": self.kv_limit, "overhead_units": self.overhead_units, "kv_refusals": self.kv_refusals,
                "kv_usage": self.kv_usage} if self.kv_limit or self.overhead_units else {}),
"""),
    ("trial.py", """lane=lane, background_slots=background_slots)
""", """lane=lane, background_slots=background_slots, kv=study.get("gate_kv"))
"""),
    ("trial.py", """        sampling = asyncio.create_task(monitor.run())
""", """        sampling = asyncio.create_task(monitor.run())
        kv_polling = (asyncio.create_task(gate.kv_poll(monitor.metrics_url))
                      if (study.get("gate_kv") or {}).get("limit") else None)
"""),
    ("trial.py", """            sampling.cancel()
            await asyncio.gather(sampling, return_exceptions=True)
""", """            sampling.cancel()
            await asyncio.gather(sampling, return_exceptions=True)
            if kv_polling is not None:
                kv_polling.cancel()
                await asyncio.gather(kv_polling, return_exceptions=True)
"""),
]
TAILMD_PATCHES = [
    ("""        extra_token = None
""", """        extra_tokens = []
"""),
    ("""            elif (SPEC_ACCEPT and accepted == self.spec and computed == request.num_tokens + self.spec + 1
                  and getattr(request, "_arc_dropped", None)):""",
     """            elif (SPEC_ACCEPT and accepted == self.spec and getattr(request, "_arc_dropped", None)
                  and 1 <= computed - request.num_tokens - self.spec == len(request._arc_dropped)):"""),
    ("""                extra_token = int(request._arc_dropped[0])
                L = request.num_tokens + 1
                column = (computed - 1) // B
                self.stats["spec_dropped_bonus"] += 1""",
     """                extra_tokens = [int(t) for t in request._arc_dropped]
                L = request.num_tokens + len(extra_tokens)
                column = (computed - 1) // B
                self.stats["spec_dropped_bonus" if len(extra_tokens) == 1 else f"spec_dropped_{len(extra_tokens)}"] += 1"""),
    ("""        if L % B == 0 or L < B or L > request.num_tokens + (extra_token is not None):""",
     """        if L % B == 0 or L < B or L > request.num_tokens + len(extra_tokens) or k_start_beyond(L, B, request):"""),
    ("""        tokens = list(request.all_token_ids[start:L - (extra_token is not None)])
        if extra_token is not None:
            tokens.append(extra_token)
        extra, _ = generate_block_hash_extra_keys(request, start, L - (extra_token is not None), 0)""",
     """        tokens = list(request.all_token_ids[start:L - len(extra_tokens)]) + extra_tokens
        extra, _ = generate_block_hash_extra_keys(request, start, L - len(extra_tokens), 0)"""),
    ("""def finished_by_length(request):""",
     """def k_start_beyond(L, B, request):
    # The entry's block start must lie within the request's own tokens (its parent hash and token slice).
    return (L // B) * B > request.num_tokens


def finished_by_length(request):"""),
]
GDND_NEW = """        gdnd = Path(__file__).parents[1] / 'arc_infra/gdn_deferred_runtime.py'
        if gdnd.exists():
            site = recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages/vllm'
            (site / 'arc_gdn_deferred.py').write_bytes(gdnd.read_bytes())
            import os as _os, subprocess as _sp
            nvcc = _os.environ.get('CUDACXX') or str(recipe.RUNTIME_ROOT / 'usr/local/cuda-13.0/bin/nvcc')
            _r = _sp.run([nvcc, '-O3', '-std=c++17', '--use_fast_math', '-gencode=arch=compute_120,code=sm_120', '--shared', '-Xcompiler',
                          '-fPIC', '-o', str(site / 'arc_libgdn2.so'), str(gdnd.parent / 'arc_gdn_m2.cu')],
                         capture_output=True, text=True)
            if _r.returncode:
                raise RuntimeError('gdn_m2.cu build failed: ' + _r.stderr[-2000:])
            print('ARC_GDN_DEFER built', nvcc, flush=True)
            for rel, line in (('model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py', 'install_layer(QwenGatedDeltaNetAttention)'),
                              ('v1/attention/backends/gdn_attn.py', 'install_builder(GDNAttentionMetadataBuilder)')):
                path = site / rel
                text = path.read_text()
                if 'arc_gdn_deferred' not in text:
                    name = line.split('(')[0]
                    path.write_text(text + f'\\nfrom vllm.arc_gdn_deferred import {name} as _arc_gdnd_{name}\\n'
                                    + f'_arc_gdnd_{line}\\n')
        return result

    recipe.verify_and_extract_runtime = wrapped_extract"""
GDNDUMP_NEW = """        gdump = Path(__file__).parents[1] / 'arc_infra/gdn_dump_runtime.py'
        if gdump.exists():
            site = recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages/vllm'
            (site / 'arc_gdn_dump.py').write_bytes(gdump.read_bytes())
            path = site / 'model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py'
            text = path.read_text()
            if 'arc_gdn_dump' not in text:
                path.write_text(text + '\\nfrom vllm.arc_gdn_dump import install_layer as _arc_gdn_dump_install\\n'
                                '_arc_gdn_dump_install(QwenGatedDeltaNetAttention)\\n')
            print('ARC_GDN_DUMP installed', flush=True)
        return result

    recipe.verify_and_extract_runtime = wrapped_extract"""
# FP8 PLE on the AutoRound checkpoint: the NVFP4 base checkpoint's ModelOpt FP8 table
# (a base asset on every VM; what v8.2 serves) through research_checkpoint's symlink copy; the worker's PLE picks the
# FP8 method under ARC_PLE_FP8=1 (edited into ple_layer.py after every other overlay, incl. ple_uva's hash-checked one).
NV_ROOT = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
NV_PLE = {f"model-plefp8-{i:05d}.safetensors": s for i, s in enumerate(
    [5200027199, 5200027202, 5200027198, 5200027189, 5200027190, 5200027190, 5200027190, 5200027190, 5200027190,
     4400023163])}
PLEF_OLD = """    command = recipe.server_command
    overlays = {}"""
PLEF_NEW = """    _arc_env = getattr(recipe, 'runtime_environment', None)
    if _arc_env is not None:
        # FP8 PLE (throughput arm): after the recipe's PLE patch and its hash check, and after the ple_uva overlay.
        def _arc_ple_fp8_environment(*args, **kwargs):
            out = _arc_env(*args, **kwargs)
            ple = recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py'
            text = ple.read_text()
            anchor = '    \\"\\"\\"Select global-scale FP8 only for quantized PLE checkpoint shards.\\"\\"\\"\\n'
            if 'ARC_PLE_FP8' not in text:
                if text.count(anchor) != 1:
                    raise RuntimeError('ple_layer.py: FP8 PLE anchor not found once')
                ple.write_text(text.replace(anchor, anchor + "    if __import__('os').environ.get('ARC_PLE_FP8') == '1' and "
                                            "prefix.endswith('ple_embedding.ngram_embedding'):\\n"
                                            "        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()\\n"))
            return out

        recipe.runtime_environment = _arc_ple_fp8_environment
    command = recipe.server_command
    overlays = {}"""
# Cache-budget admission: the equal scheduler admits a stream
# only while the live contexts of the active games, each plus a growth headroom, fit the cache that holds them whole.
# Under the write-through host tier that cache is max(GPU pool, host cache), not their sum. A game's context counts
# from its latest request until its next compaction or drain (the cached prefix is dead there). At an eviction over
# the budget the slot closes (the game re-queues); free room refills at the next tick, fewest tokens first.
CB_FIELD_OLD = "    context_evictions: int = 0\n"
CB_FIELD_NEW = "    context_evictions: int = 0\n    evicted_at_requests: int = -1  # len(requests) at the latest eviction\n"
CB_EVICT_OLD = """    def _context_evicted(self, kind, **details):
        self.context_evictions += 1
"""
CB_EVICT_NEW = """    def _context_evicted(self, kind, **details):
        self.context_evictions += 1
        self.evicted_at_requests = len(self.requests)
"""
CB_FILL_OLD = """    def _priority_fill(self, freed):
        \"\"\"Free slots (the release, or slots of finished games) take the best waiting games.\"\"\"
        while len(self.active()) < self.active_limit:
            ranked = self._priority_rank(self._priority_candidates())
            if not ranked:
                return
            game, value, details, _ = ranked[0]
            self._priority_admit(game, value, details, freed)
"""
CB_FILL_NEW = """    cache_budget_tokens = 0
    cache_headroom_tokens = 0
    cache_window_tokens = 0
    cache_min_slots = 4
    cache_cost = None  # tokens -> gate blocks (the served layout's BlockBudget.cost: the v8.2 gate's unit)
    cache_counts = None

    def _cache_resident(self, game):
        return game.latest_prompt_tokens if len(game.requests) > game.evicted_at_requests else 0

    def _cache_load(self, without=None):
        \"\"\"Blocks the active games' live contexts hold now (cost of each context, as the KV gate counts it).\"\"\"
        return sum(self.cache_cost(r) for r in (self._cache_resident(g) for g in self.active() if g is not without) if r)

    def _cache_projected(self, game):
        \"\"\"A game's context a headroom ahead, at most the window (admission counts growth; trimming does not).\"\"\"
        return self.cache_cost(min(self.cache_window_tokens, self._cache_resident(game) + self.cache_headroom_tokens))

    def _cache_room(self, game, without=None):
        if not self.cache_budget_tokens:
            return True
        active = sum(g is not without for g in self.active())
        need = sum(self._cache_projected(g) for g in self.active() if g is not without) + self._cache_projected(game)
        return active < self.cache_min_slots or need <= self.cache_budget_tokens

    def _cache_count(self, key):
        if self.cache_counts is None:
            self.cache_counts = {"fill_held_ticks": 0, "closed": 0, "trimmed": 0}
        self.cache_counts[key] += 1

    def _cache_trim(self):
        \"\"\"Over the budget between evictions (contexts grow): park the idle active game (no request in flight)
        with the most generated tokens until the load fits. Its context is re-prefilled once at its return.\"\"\"
        while self.cache_budget_tokens and len(self.active()) > self.cache_min_slots:
            load = self._cache_load()
            if load <= self.cache_budget_tokens:
                return
            idle = [g for g in self.active()
                    if not g.admission_attempts or g.admission_attempts[-1].get("status") != "admitted"]
            if not idle:
                return
            victim = max(idle, key=lambda g: (g.clock.completion_tokens(), g.index))
            self._cache_count("trimmed")
            self.priority_seq += 1
            value, _ = self.priority_value(victim, fresh=True)
            victim.priority_queued = self.priority_seq
            victim.priority_queued_score = value
            resident = self._cache_resident(victim)
            self.park(victim, "handover", extra={"priority": value, "to": None, "cache_budget": True, "trim": True,
                                                 "resident": resident})
            self._priority_note("cache_trim", victim, load=load, resident=resident, budget=self.cache_budget_tokens,
                                active=len(self.active()))

    def _priority_fill(self, freed):
        \"\"\"Free slots (the release, or slots of finished games) take the best waiting games.\"\"\"
        while len(self.active()) < self.active_limit:
            ranked = self._priority_rank(self._priority_candidates())
            if not ranked:
                return
            game, value, details, _ = ranked[0]
            if not self._cache_room(game):
                self._cache_count("fill_held_ticks")
                return
            self._priority_admit(game, value, details, freed)
"""
CB_HAND_OLD = """        if best is game:
            self.priority_counts["handover_kept"] += 1
"""
CB_HAND_NEW = """        if self.cache_budget_tokens and not self._cache_room(best, without=game):
            self._cache_count("closed")
            game.priority_queued = self.priority_seq
            game.priority_queued_score = value
            self.park(game, "handover", extra={"eviction": kind, "priority": value, "to": None, "cache_budget": True})
            self._priority_note("cache_close", game, eviction=kind, priority=value, load=self._cache_load(),
                                budget=self.cache_budget_tokens, active=len(self.active()), **extra)
            return
        if best is game:
            self.priority_counts["handover_kept"] += 1
"""
CB_TICK_OLD = """        self._priority_fill("finished")
        if self.priority_fade_wall is None:
"""
CB_TICK_NEW = """        self._cache_trim()
        self._priority_fill("finished")
        if self.priority_fade_wall is None:
"""
CB_PROG_OLD = """                "tail_fade_wall_seconds": self.priority_fade_wall,
"""
CB_PROG_NEW = """                "tail_fade_wall_seconds": self.priority_fade_wall,
                **({"cache_budget": {"budget": self.cache_budget_tokens, "load": self._cache_load(),
                                     **(self.cache_counts or {})}} if self.cache_budget_tokens else {}),
"""
CB_TRIAL_OLD = """            priority=study.get("priority"),
        )
"""
CB_TRIAL_NEW = """            priority=study.get("priority"),
        )
        if study.get("cache_budget"):
            if block_budget is None:
                raise ValueError("cache_budget needs the block-admission layout")
            cb, layout = study["cache_budget"], block_budget.layout
            # Units: gate blocks (BlockBudget.cost of a context, every KV group incl. the QSA ring and GDN state).
            gpu = int(block_budget.capacity)
            per_block = float(layout["planned_cache_bytes"]) / int(layout["num_blocks"])
            import os as _os
            gib = float(_os.environ.get("ARENA_CPU_CACHE_GIB") or study["cache_experiment"].get("cpu_cache_gib") or 0)
            host = int(gib * 2**30 / per_block * float(cb["host_margin"]))
            scheduler.cache_cost = block_budget.cost
            scheduler.cache_budget_tokens = int(max(gpu, host) * float(cb["fraction"]))  # blocks
            scheduler.cache_headroom_tokens = int(cb["headroom_tokens"])
            scheduler.cache_window_tokens = int(study["windows"][0])
            scheduler.cache_min_slots = int(cb["min_slots"])
            self.gate_log({"event": "cache_budget", "gpu_blocks": gpu, "host_blocks": host, "bytes_per_block":
                           round(per_block, 1), "budget_blocks": scheduler.cache_budget_tokens,
                           "full_window_blocks": block_budget.cost(int(study["windows"][0])), **cb})
"""
# A cache-budget trim parks a game between evictions (cb v2 audit: "1 priority handovers but 0 context evictions").
CB_AUDIT_OLD = """                        raise ValueError('handover park without the priority policy')
                    priority_handovers += 1
"""
CB_AUDIT_NEW = """                        raise ValueError('handover park without the priority policy')
                    priority_handovers += 0 if event.get('trim') else 1  # cache-budget trims: no eviction
"""
# vLLM's offloading spec defaults to offload_prompt_only=True: KV of generated tokens never reaches host RAM, so a
# multi-turn context loses each turn's completion (and the turn's tail) once the GPU evicts it.
DECODE_OLD = "                'kv_connector_extra_config': {'cpu_bytes_to_use': cpu_gib * 1024**3},"
DECODE_NEW = "                'kv_connector_extra_config': {'cpu_bytes_to_use': cpu_gib * 1024**3, 'offload_prompt_only': False},"
OFFLOAD_GUARD_OLD = """    if settings['cpu_cache_gib']:
        raise NotImplementedError('Pinned offloading scheduler rejects QSA CircularBufferSpec; CPU cache needs backend support')"""
OFFLOAD_GUARD_NEW = """    # cpu_cache_gib: served through arc_infra.arc_offload_runtime (vllm/arc_offload.py), which leaves the QSA ring out."""
# The startup probe follows the per-trial ARENA_CPU_CACHE_GIB override (a no-offload trial must not demand CPU loads).
PROBE_OLD = "            if study['cache_experiment']['cpu_cache_gib']:"
PROBE_ENV_OLD = '''"--study", str(bundle / "configs/comparison.json"), "--base-url", args.base_url],
                               env=env, check=True, timeout=300)'''
PROBE_ENV_NEW = '''"--study", str(bundle / "configs/comparison.json"), "--base-url", args.base_url],
                               env={**env, **extra}, check=True, timeout=300)'''
PROBE_NEW = ("            if int(__import__('os').environ.get('ARENA_CPU_CACHE_GIB', "
             "study['cache_experiment']['cpu_cache_gib'])):")
TOKEN_STOP_OLD = '''        return EarlyStop(cfg, read_score=read, paths=stop_paths(self.output), log=self.gate_log)
'''
TOKEN_STOP_NEW = '''        stop = EarlyStop(cfg, read_score=read, paths=stop_paths(self.output), log=self.gate_log)
        token_stop = int(study.get("token_stop") or 0)
        if token_stop:
            # Serving ablation: an instance finishes at token_stop generated tokens (its slot goes to
            # the next game at once); the trial ends when every instance has finished.
            check = stop.check

            def check_tokens(elapsed):
                reason = check(elapsed)
                if reason is not None:
                    return reason
                for game in games:
                    if game.finished_reason is None and game.clock.completion_tokens() >= token_stop:
                        game.note_finished("token_stop")
                        self.gate_log({"event": "token_stop_game", "game": game.game_id,
                                       "tokens": int(game.clock.completion_tokens()), "elapsed": round(elapsed, 1)})
                if all(game.finished_reason is not None for game in games):
                    stop.reason = "token_stop"
                    self.gate_log({"event": "early_stop", "reason": "token_stop", "token_stop": token_stop,
                                   "elapsed": round(elapsed, 1)})
                    return stop.reason
                return None

            stop.check = check_tokens
        return stop
'''


PROFILE_OLD = """            def check_tokens(elapsed):
                reason = check(elapsed)
"""
PROFILE_NEW = """            profile_at = sorted(float(t) for t in study.get("profile_at") or [])
            profile_url = self.base_url[:-3] if self.base_url.rstrip("/").endswith("/v1") else self.base_url
            profile_url = profile_url.rstrip("/").removesuffix("/v1")

            def fire_profile(at):
                # Step profile: the server's torch profiler (3-step delay, 40 steps) at play depth.
                import threading, time, urllib.request

                def run():
                    for path in ("/start_profile", "/stop_profile"):
                        try:
                            urllib.request.urlopen(urllib.request.Request(profile_url + path, data=b"", method="POST"),
                                                   timeout=600).read()
                            self.gate_log({"event": "profile", "path": path, "at": at})
                        except Exception as error:  # never fatal
                            self.gate_log({"event": "profile_error", "path": path, "error": repr(error)[:300]})
                        time.sleep(30)
                threading.Thread(target=run, daemon=True).start()

            def check_tokens(elapsed):
                while profile_at and elapsed >= profile_at[0]:
                    fire_profile(profile_at.pop(0))
                reason = check(elapsed)
"""
TOPK_OLD = "    config = vllm_config.model_config.hf_text_config\n    current = getattr(config, \"num_experts\", None)\n"
TOPK_NEW = ("    config = vllm_config.model_config.hf_text_config\n"
            "    _topk = os.environ.get(\"ARC_TOPK\")\n"
            "    if _topk and getattr(config, \"num_experts_per_tok\", None) != int(_topk):  # lossy top-k (18 Oct)\n"
            "        log(\"ARC_TOPK \" + json.dumps({\"from\": getattr(config, \"num_experts_per_tok\", None), \"to\": int(_topk)}), flush=True)\n"
            "        config.num_experts_per_tok = int(_topk)\n"
            "    current = getattr(config, \"num_experts\", None)\n")

# Startup cache probe (18 Oct): a client-side transport error (httpx ReadError with the server healthy, gi3m3x v2)
# failed the boot; each probe POST now retries transport errors 3 times with 1/2/4 s backoff (same request, output-exact).
PROBE_RETRY_ANCHOR = "\n\nasync def run(server_dir, study, base_url):\n"
PROBE_RETRY_HELPER = '''

async def _post(client, url, **kwargs):
    """client.post with transport-error retries (startup fragility, 18 Oct): same request, 1/2/4 s backoff."""
    for attempt in range(4):
        try:
            return await client.post(url, **kwargs)
        except httpx.TransportError:
            if attempt == 3:
                raise
            await asyncio.sleep(2 ** attempt)
'''


KV4_PAGE_OLD = "    page = 10240 * (3 + int(spec_tokens)) * 2 + 48 * 128 * 128 * 2\n"
KV4_PAGE_NEW = "    page = 10240 * (3 + int(spec_tokens)) * 2 + 48 * 128 * 128 * int(state_bytes)\n"
KV4_SIG_OLD = "def kv4_block_tokens(mode, spec_tokens=0):\n"
KV4_SIG_NEW = "def kv4_block_tokens(mode, spec_tokens=0, state_bytes=2):\n"
KV4_CALL_OLD = "str(kv4_block_tokens(mode, spec_tokens))]"
KV4_CALL_NEW = ("str(kv4_block_tokens(mode, spec_tokens, 2 if spec['mamba_ssm_cache_dtype'] == 'bfloat16' or not spec_tokens "
                "else 4))]")


PLE_SYNC_ANCHOR = (
    "            req_group = torch.full(\n"
    "                (m.num_reqs,),\n"
    "                2,\n"
    "                dtype=torch.int64,\n"
    "                device=query_start_loc.device,\n"
    "            )\n"
    "            req_group[spec_req_idx] = 0\n"
    "            req_group[async_tensor_h2d(decode_req_idx_cpu, device=query_start_loc.device)] = 1\n"
    "            token_group = torch.repeat_interleave(req_group, query_lens)\n"
    "            token_perm = torch.argsort(token_group, stable=True)\n")
PLE_SYNC_NEW = (
    "            # ARC (24 Oct): the token permutation from the CPU masks, copied async; the GPU repeat_interleave\n"
    "            # without output_size synced the stream on every MTP mixed step. Stable argsort: identical indices.\n"
    "            req_group_cpu = torch.full((m.num_reqs,), 2, dtype=torch.int64)\n"
    "            req_group_cpu[spec_req_idx_cpu] = 0\n"
    "            req_group_cpu[decode_req_idx_cpu] = 1\n"
    "            token_perm = async_tensor_h2d(\n"
    "                torch.argsort(torch.repeat_interleave(req_group_cpu, query_lens_cpu), stable=True),\n"
    "                device=query_start_loc.device,\n"
    "            )\n")
PLE_SYNC_NEW_AB = (
    "            if __import__('os').path.exists('/tmp/arc-ab-flag'):  # ARC A/B (24 Oct): the pinned GPU path\n"
    "                req_group = torch.full(\n"
    "                    (m.num_reqs,),\n"
    "                    2,\n"
    "                    dtype=torch.int64,\n"
    "                    device=query_start_loc.device,\n"
    "                )\n"
    "                req_group[spec_req_idx] = 0\n"
    "                req_group[async_tensor_h2d(decode_req_idx_cpu, device=query_start_loc.device)] = 1\n"
    "                token_group = torch.repeat_interleave(req_group, query_lens)\n"
    "                token_perm = torch.argsort(token_group, stable=True)\n"
    "            else:\n"
    + "".join("    " + line + "\n" for line in PLE_SYNC_NEW.rstrip("\n").split("\n")))
TOGGLE_NEW = """            toggle_s = float(study.get("toggle_ab_seconds") or 0)
            toggle_state = {"next": 0.0, "on": False}

            def check_tokens(elapsed):
                # A/B toggle (24 Oct): the flag file flips every toggle_s of play; server code reads it per step
                # (flag present = arm B, the pinned path). Each flip is logged with its play time.
                if toggle_s and elapsed >= toggle_state["next"]:
                    import os
                    try:
                        if toggle_state["on"]:
                            os.remove("/tmp/arc-ab-flag")
                        else:
                            open("/tmp/arc-ab-flag", "w").close()
                        toggle_state["on"] = not toggle_state["on"]
                        self.gate_log({"event": "ab_toggle", "at": elapsed, "flag": toggle_state["on"]})
                    except Exception as error:  # never fatal
                        self.gate_log({"event": "ab_toggle_error", "at": elapsed, "error": repr(error)[:300]})
                    toggle_state["next"] += toggle_s
                reason = check(elapsed)
"""
PLE_EDITS_OLD = "    ],\n    # The edits rely on this import of the pinned file (checked, not added).\n"


def patch_ple_sync(path, ab=False):
    """Append the PLE mixed-branch sync fix to stack_kernels_patch.SHORT_CONV's edits (applied after the async_h2d
    edits, so its anchor is their output). ``ab``: the pinned GPU path while /tmp/arc-ab-flag exists."""
    text = path.read_text()
    if text.count(PLE_EDITS_OLD) != 1:
        raise SystemExit("ple-sync: SHORT_CONV edits anchor not found once")
    repl = PLE_SYNC_NEW_AB if ab else PLE_SYNC_NEW
    text = text.replace(PLE_EDITS_OLD, f"        ({PLE_SYNC_ANCHOR!r},\n         {repl!r}, 1),\n" + PLE_EDITS_OLD)
    path.write_text(text)


GDN_MASK_ANCHOR = "            initial_state[~prefill_has_initial_state, ...] = 0\n"
GDN_MASK_NEW = (
    "            # ARC (24 Oct): masked_fill_ with a broadcast mask; boolean index_put_ ran nonzero (a stream sync) in\n"
    "            # every GDN layer of every step carrying a prefill. Same elements set to +0: bitwise identical.\n"
    "            initial_state.masked_fill_(\n"
    "                ~prefill_has_initial_state.to(initial_state.device).view(-1, *([1] * (initial_state.dim() - 1))), 0\n"
    "            )\n"
)
GDN_MASK_NEW_AB = (
    "            if __import__('os').path.exists('/tmp/arc-ab-flag'):  # ARC A/B (24 Oct): the pinned path\n"
    "                initial_state[~prefill_has_initial_state, ...] = 0\n"
    "            else:\n" + "".join("    " + l + "\n" for l in GDN_MASK_NEW.splitlines())
)
GDN_TARGET_OLD = "TARGETS = (SHORT_CONV,)\n"
GDN_TARGETS_OLD = "    return [SHORT_CONV] if spec['async_h2d'] else []\n"


def patch_gdn_mask(path, ab=False):
    """GDN prefill sync fix (24 Oct): a second stack_kernels target (the pinned qwen_gdn_linear_attn.py, sha-guarded)
    served with async_h2d. ``ab``: the pinned boolean-index path while /tmp/arc-ab-flag exists."""
    text = path.read_text()
    if text.count(GDN_TARGET_OLD) != 1 or text.count(GDN_TARGETS_OLD) != 1:
        raise SystemExit("gdn-mask: stack_kernels_patch anchors not found once")
    target = ("GDN_MASK = {\n"
              "    'relative': 'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py',\n"
              "    'sha256': '81b4dcd0952492375c93bffc2cdf45f10b45ab5e117f2e1d949a147d144e64f0',\n"
              f"    'edits': [({GDN_MASK_ANCHOR!r}, {(GDN_MASK_NEW_AB if ab else GDN_MASK_NEW)!r}, 1)],\n"
              "}\n")
    text = text.replace(GDN_TARGET_OLD, target + "TARGETS = (SHORT_CONV, GDN_MASK)\n")
    text = text.replace(GDN_TARGETS_OLD, "    return [SHORT_CONV, GDN_MASK] if spec['async_h2d'] else []\n")
    path.write_text(text)


def patch_gdn_fp32(path):
    """fp32 GDN recurrent state (24 Oct): with mamba_ssm_cache_dtype null the MTP draft hook serves the checkpoint's
    fp32 state, so the 4-bit block must hold a 4-byte-per-element state page (MTP-2: 6,208 tokens, not 3,200)."""
    text = path.read_text()
    for old, new in ((KV4_PAGE_OLD, KV4_PAGE_NEW), (KV4_SIG_OLD, KV4_SIG_NEW), (KV4_CALL_OLD, KV4_CALL_NEW)):
        if text.count(old) != 1:
            raise SystemExit(f"gdn-fp32: anchor not found once in {path}: {old.strip()[:60]}")
        text = text.replace(old, new)
    path.write_text(text)


def patch_probe_retry(path):
    text = path.read_text()
    if "async def _post(client, url" in text:
        return False
    if text.count(PROBE_RETRY_ANCHOR) != 1 or "await client.post(" not in text:
        raise SystemExit(f"cache probe layout changed: {path}")
    text = text.replace("await client.post(", "await _post(client, ")  # call sites first, then the helper
    text = text.replace(PROBE_RETRY_ANCHOR, PROBE_RETRY_HELPER + PROBE_RETRY_ANCHOR)
    path.write_text(text)
    return True

PYSPY_NEW = """            pyspy_at = sorted(float(t) for t in study.get("pyspy_at") or [])

            def fire_pyspy(at):
                # Engine sampling profile (17 Oct): py-spy (static binary shipped in src/arena/bin) on the EngineCore
                # process(es) and their worker children, 60 s, non-blocking, all threads incl. idle; raw stacks into serving/torch-profile.
                import pathlib, subprocess, threading

                def run():
                    try:
                        exe = str(pathlib.Path(__file__).resolve().parent / "bin" / "py-spy")
                        pids = subprocess.run(["pgrep", "-f", "EngineCore"], capture_output=True, text=True).stdout.split()
                        # and the GPU worker(s): EngineCore's child processes (the engine only waits on them)
                        pids += [c for e in list(pids) for c in subprocess.run(["pgrep", "-P", e], capture_output=True,
                                                                               text=True).stdout.split()]
                        out = pathlib.Path(self.server_dir) / "torch-profile"
                        out.mkdir(parents=True, exist_ok=True)
                        procs = [(pid, subprocess.Popen([exe, "record", "--pid", pid, "--duration", "60", "--rate", "250",
                                                         "--threads", "--idle", "--nonblocking", "--format", "raw",
                                                         "--output", str(out / f"pyspy-{int(at)}-{pid}.txt")],
                                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
                                 for pid in pids[:5]]
                        for pid, proc in procs:
                            text = proc.communicate(timeout=300)[0]
                            self.gate_log({"event": "pyspy", "at": at, "pid": pid, "rc": proc.returncode, "log": text[-300:]})
                    except Exception as error:  # never fatal
                        self.gate_log({"event": "pyspy_error", "at": at, "error": repr(error)[:300]})
                threading.Thread(target=run, daemon=True).start()

            def check_tokens(elapsed):
                while pyspy_at and elapsed >= pyspy_at[0]:
                    fire_pyspy(pyspy_at.pop(0))
                reason = check(elapsed)
"""
SAMPLER_NEW = """            sampler_at = sorted(float(t) for t in study.get("sampler_at") or [])

            def fire_sampler(at):
                # In-process stack sampler (22 Oct; py-spy cannot attach to the GPU worker): a trigger file that the
                # arc_sampler vLLM plugin in every server process picks up within 1 s; 60 s of main-thread stacks each.
                import pathlib
                try:
                    out = pathlib.Path(self.server_dir) / "torch-profile" / "sampler"
                    out.mkdir(parents=True, exist_ok=True)
                    (out / f"go-{int(at)}").write_text("")
                    self.gate_log({"event": "sampler", "at": at})
                except Exception as error:  # never fatal
                    self.gate_log({"event": "sampler_error", "at": at, "error": repr(error)[:300]})

            def check_tokens(elapsed):
                while sampler_at and elapsed >= sampler_at[0]:
                    fire_sampler(sampler_at.pop(0))
                reason = check(elapsed)
"""
SAMPLER_ENV_OLD = """    env["PYTHONPATH"] = os.pathsep.join(
        [str(cutlass_nested), str(site)]
"""
SAMPLER_ENV_NEW = """    # In-process stack sampler plugin (22 Oct): an entry-point dist on the path; triggers land in the server dir.
    env["ARC_SAMPLER_DIR"] = "/tmp/arena/serving/torch-profile/sampler"
    os.makedirs(env["ARC_SAMPLER_DIR"], exist_ok=True)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(next((c for c in [Path(e).resolve() / "arena" / "sampler" for e in sys.path if e]
                   + [Path(__file__).resolve().parents[1] / "arena" / "sampler"] if (c / "arc_sampler.py").is_file()),
                  Path("/opt/arc-validation/src/arena/sampler"))), str(cutlass_nested), str(site)]
"""
RAMP_TRIAL_OLD = """            scheduler.cache_min_slots = int(cb["min_slots"])
"""
RAMP_TRIAL_NEW = """            scheduler.cache_min_slots = int(cb["min_slots"])
            scheduler.ramp_seconds = float(cb.get("ramp_seconds") or 0)
            scheduler.ramp_start = int(cb.get("ramp_start") or 0)
"""
RAMP_FILL_OLD = """        while len(self.active()) < self.active_limit:
            ranked = self._priority_rank(self._priority_candidates())"""
RAMP_FILL_NEW = """        limit = self.active_limit
        if getattr(self, "ramp_seconds", 0) and freed != "handover":
            # Staggered first windows (17 Oct): slots open linearly from ramp_start to active_limit over ramp_seconds,
            # so games cross the context window (and drain) at spread times instead of in waves.
            done = self.global_clock.elapsed() / self.ramp_seconds
            limit = min(self.active_limit, self.ramp_start + int(done * (self.active_limit - self.ramp_start)))
        while len(self.active()) < limit:
            ranked = self._priority_rank(self._priority_candidates())"""
GRAPH_OLD = "not 1 <= v <= 256 for v in sizes)"
GRAPH_NEW = "not 1 <= v <= 4096 for v in sizes)"
GRAPH_MAX_OLD = "not 1 <= graph_capture_size <= 256:"
GRAPH_MAX_NEW = "not 1 <= graph_capture_size <= 4096:"
SETUP_ENV_OLD = """            minimum=0,
            maximum=256,
        )"""
SETUP_ENV_NEW = """            minimum=0,
            maximum=4096,
        )"""
SETUP_BOUND_OLD = '"max_cudagraph_capture_size": (0, 256),'
SETUP_BOUND_NEW = '"max_cudagraph_capture_size": (0, 4096),'
EVIDENCE_OLD = """        for name in ("cache-probe",):"""
EVIDENCE_NEW = """        for name in ("cache-probe", "torch-profile"):"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



STEADY_HOOK_ANCHOR = '                if self.spec.get("kernel_profile"):\n'
STEADY_HOOK = '''                if self.spec.get("steady_replay"):
                    # Steady-state replay bench (18 Oct; arena.steady_replay).
                    from .steady_replay import steady_replay

                    def save_steady():
                        (self.output / "bench.json").write_text(json.dumps(report, indent=2))

                    try:
                        report["steady_replay"] = await steady_replay(self, client, gate, budget, monitor,
                                                                      self.spec["steady_replay"], save=save_steady)
                    except Exception as exc:  # noqa: BLE001 - the report is still written
                        self.errors.append(f"steady_replay: {type(exc).__name__}: {str(exc)[:300]}")
                    save_steady()
'''


def add_steady_replay(out, study, a):
    """Steady-state replay bench: the served stack unchanged, the arena trial replaced by one
    serving-bench trial (named duck-transfer, the launcher's trial) that runs arena.steady_replay on recorded
    late-run histories, and no grid points."""
    contexts = Path(a.steady_replay)
    data = json.loads(contexts.read_text())
    if data.get("format") != "arc-steady-replay-v1":
        raise SystemExit("--steady-replay needs arc-steady-replay-v1 contexts")
    shutil.copyfile(Path(__file__).resolve().parent / "steady_replay.py", out / "src/arena/steady_replay.py")
    (out / "bench").mkdir(exist_ok=True)
    shutil.copyfile(contexts, out / "bench/steady-contexts.json")
    bench = out / "src/arena/throughput_bench.py"
    text = bench.read_text()
    if text.count(STEADY_HOOK_ANCHOR) != 1:
        raise SystemExit("throughput_bench layout changed: no kernel_profile hook anchor")
    bench.write_text(text.replace(STEADY_HOOK_ANCHOR, STEADY_HOOK + STEADY_HOOK_ANCHOR, 1))
    lo, hi = (int(x) for x in a.steady_start.split(","))
    spec = {"contexts_file": "bench/steady-contexts.json", "start_tokens": [lo, hi], "warm_seconds": a.steady_warm,
            "measure_seconds": a.steady_measure, "games": len(data["games"]), "seed": 0, "bin_seconds": 60,
            "reserve_tokens": a.steady_reserve or None, "length_scale": a.steady_length_scale}
    if a.steady_phases:
        spec["phases"] = json.loads(a.steady_phases)
    if a.steady_profile:  # "AT1,AT2,...:ITERS": torch-profile the live replay (mixed steps) at those offsets
        ats, iters = a.steady_profile.split(":")
        spec["profile_windows"] = [float(x) for x in ats.split(",")]
        spec["profile_seconds"] = 60
    wall = (a.steady_warm + a.steady_measure + 120) * max(1, len(spec.get("phases") or [])) + 60
    env = dict(study["trials"][0].get("server_env") or {})
    if a.steady_profile:
        env.update(ARENA_TORCH_PROFILER="1")
        his.patch(out / "src/arena/serving.py", "'max_iterations': 40,",
                  f"'max_iterations': {int(a.steady_profile.split(':')[1])},")
    study["trials"] = [{"name": "duck-transfer", "kind": "serving-bench", "competition": False, "wall_seconds": wall,
                        "server_env": env}]
    study["throughput_bench"] = {"prompt_tokens": [], "concurrency": [], "shared_prefix": None,
                                 "steady_replay": spec}
    study["wall_seconds"] = wall
    if study.get("b200"):
        study["b200"]["stop"]["wall_seconds"] = wall
    study["competition"]["preview_seconds"] = wall
    study["steady_replay"] = {"contexts": str(contexts), "contexts_sha256": sha(contexts),
                              "source": data["source"], "games": len(data["games"])}

GPU_HOOK = '''                if self.spec.get("gpu_script"):
                    # GPU microbenchmark (18 Oct): a bundle script run beside the idle server in the agent venv.
                    import os as _os
                    import sys as _sys
                    g = self.spec["gpu_script"]
                    root = Path(__file__).resolve().parents[2]
                    # The server's own runtime environment (python, PYTHONPATH with the extracted vLLM runtime,
                    # CUDA libraries), recorded by arena.serving as kv8-runtime-env.json (ARC_RECORD_RUNTIME_ENV).
                    rt = json.loads((Path(self.server_dir) / "kv8-runtime-env.json").read_text())
                    env = {**_os.environ, **rt["env"]}
                    env["PYTHONPATH"] = _os.pathsep.join([str(root / "src"), env.get("PYTHONPATH", "")])
                    env.update(ARC_BENCH_BASE_URL=self.base_url, ARC_BENCH_MODEL=self.model, ARC_BENCH_EFFORT=self.effort)
                    with open(self.output / "gpu-script.log", "w") as log:
                        proc = await asyncio.create_subprocess_exec(
                            rt["python"], str(root / g["script"]), *g["args"], "--out", str(self.output / g["out"]),
                            stdout=log, stderr=log, cwd=str(root), env=env)
                        code = await proc.wait()
                    report["gpu_script"] = {"returncode": code, "out": g["out"]}
                    if code:
                        self.errors.append(f"gpu_script exited {code}")
                    (self.output / "bench.json").write_text(json.dumps(report, indent=2))
'''


def add_gpu_script(out, study, a):
    """GPU microbenchmark bundle (18 Oct): one serving-bench trial whose bench runs a script from bench/ (copied with
    its inputs) next to the idle server; the server keeps a small memory share (--server-env)."""
    script, *inputs = a.gpu_script.split(",")
    bench_dir = out / "bench"
    bench_dir.mkdir(exist_ok=True)
    shutil.copyfile(script, bench_dir / Path(script).name)
    for item in inputs:
        src = Path(item)
        (shutil.copytree if src.is_dir() else shutil.copyfile)(src, bench_dir / src.name)
    path = out / "src/arena/throughput_bench.py"
    text = path.read_text()
    if text.count(STEADY_HOOK_ANCHOR) != 1:
        raise SystemExit("throughput_bench layout changed: no kernel_profile hook anchor")
    path.write_text(text.replace(STEADY_HOOK_ANCHOR, GPU_HOOK + STEADY_HOOK_ANCHOR, 1))
    serving = out / "src/arena/serving.py"
    text = serving.read_text()
    for old_text, new_text in (("    if environment is not None and (ple_uva or kv8):\n",
                                "    if environment is not None and (ple_uva or kv8 or os.environ.get('ARC_RECORD_RUNTIME_ENV')):\n"),
                               ("            if kv8:\n                from .kv8_checks import record_runtime_env\n",
                                "            if kv8 or os.environ.get('ARC_RECORD_RUNTIME_ENV'):\n"
                                "                from .kv8_checks import record_runtime_env\n")):
        if text.count(old_text) != 1:
            raise SystemExit("serving.py layout changed: runtime-env anchors")
        text = text.replace(old_text, new_text)
    serving.write_text(text)
    wall = a.gpu_seconds
    env = dict(study["trials"][0].get("server_env") or {}, ARC_RECORD_RUNTIME_ENV="1")
    study["trials"] = [{"name": "duck-transfer", "kind": "serving-bench", "competition": False, "wall_seconds": wall,
                        "server_env": env}]
    study["throughput_bench"] = {"prompt_tokens": [], "concurrency": [], "shared_prefix": None,
                                 "gpu_script": {"script": f"bench/{Path(script).name}",
                                                "args": json.loads(a.gpu_args), "out": "gpu-script.json"}}
    if a.bench_profile:  # decode torch profile after the script (throughput_bench.profile)
        study["throughput_bench"]["profile"] = json.loads(a.bench_profile)
        env["ARENA_TORCH_PROFILER"] = "1"
    study["wall_seconds"] = wall
    if study.get("b200"):
        study["b200"]["stop"]["wall_seconds"] = wall
    study["competition"]["preview_seconds"] = wall


# Strict bundle check (PTH's c1play-v1 served the wrong model because a loader ignored a key):
# every server / serving_kernels / dense_kernels key must be read somewhere (a quoted literal in the bundle's Python or
# the launchers), and a top-level study key must be read, inherited unchanged from the base study, or a provenance
# record this builder writes. The expert overlay must cover every expert shard of every decoder layer (runtime check).
LAUNCHERS = [Path(__file__).resolve().parents[1] / "modal_validation.py",
             Path(__file__).resolve().parents[3] / "nebius/scripts/nebius_validation.py"]
RECORD_KEYS = {"serving_ablation", "reply_reserve_tokens", "server_patch", "study_patch", "qsa_dump", "fp8_dense", "fp4_dense", "cg_exact", "tool_repair", "keepalive_until",
               "tail_multidrop", "gdn_defer", "gdn_dump", "kvoff", "ple_prefill_bound", "probe_retry", "steady_replay",
               "cache_budget_record", "notebook_seconds", "dense_capture"}
OVERLAY_OLD = """    omits = bool(manifest.get("omits_pruned"))"""
OVERLAY_NEW = """    # every expert shard of every decoder layer (strict check, 19 Oct): a partial overlay fails before serving
    import re as _re
    _layers = int((json.loads((base_dir / "config.json").read_text()).get("text_config") or {}).get("num_hidden_layers")
                  or json.loads((base_dir / "config.json").read_text()).get("num_hidden_layers") or 0)
    _expected = {f for f in set(weight_map.values()) if "-experts-" in f
                 and (m := _re.match(r"layer-0*(\\d+)-", f)) and int(m.group(1)) < _layers}
    if not _layers or set(shards) != _expected or sorted(manifest.get("layers") or []) != list(range(_layers)):
        raise RuntimeError(f"partial expert overlay: {len(shards)} shards / {len(manifest.get('layers') or [])} layers "
                           f"against {len(_expected)} expert shards of {_layers} decoder layers")
    omits = bool(manifest.get("omits_pruned"))"""


def check_bundle(out, study, base_study):
    text = "\n".join(p.read_text(errors="ignore") for p in [*out.rglob("*.py"), *LAUNCHERS] if p.is_file())
    import re
    read = set(re.findall(r"""['"]([A-Za-z_][A-Za-z0-9_]*)['"]""", text))
    server = study.get("server") or {}
    bad = [f"server.{k}" for k in server if k not in read]
    bad += [f"server.serving_kernels.{k}" for k in (server.get("serving_kernels") or {}) if k not in read]
    bad += [f"server.dense_kernels.{k}" for k in (server.get("dense_kernels") or {}) if k not in read]
    bad += [k for k in study if k not in read and k not in RECORD_KEYS
            and not (k in base_study and base_study[k] == study[k])]
    if bad:
        raise SystemExit(f"bundle check: keys nothing in the bundle or launchers reads: {bad}")
    if server.get("expert_overlay"):
        ov = out / "src/arena/expert_overlay.py"
        if "partial expert overlay" not in ov.read_text():
            raise SystemExit("bundle check: the expert overlay coverage check is not in src/arena/expert_overlay.py")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--v82", required=True)
    ap.add_argument("--arm", required=True, choices=("base", "R", "P", "K", "Kb", "Kf", "D", "DKf", "DKb", "O", "AR", "AV2C", "T25", "T12", "T50", "T62", "T37A", "DKbT12", "PROF", "CG", "TICK4", "CGT4"))
    ap.add_argument("--rl", help="rl-pool-v1-dev-p1 bundle (arm R: its server block)")
    ap.add_argument("--instances", type=int, default=100)
    ap.add_argument("--stop-tokens", type=int, default=75_000)
    ap.add_argument("--wall", type=int, required=True, help="gameplay backstop seconds")
    ap.add_argument("--streams", type=int, help="override the arm's stream count")
    ap.add_argument("--extra", default="", help="levers stacked on the arm, comma-separated: CG, TICK4, PROF, IXF8 (AV2C: FP8 indexer keys), T0 (no prune, all-expert excl5 INT3)")
    ap.add_argument("--kvoff", default="", help="TOTAL,DEV[,SLOTS]: attention KV in host RAM (TOTAL blocks, or xF times "
                    "the profiled count), DEV device pages for the GDN states, SLOTS 4-token GPU cache groups per "
                    "layer (0/absent: no cache; kvoff/kvoff_runtime.py; exact)")
    ap.add_argument("--cpu-gib", type=int, default=0, help="host-RAM KV offload (arc2 f44c34f9 ArcOffloadingConnector: the "
                    "QSA ring group left out); ARENA_CPU_CACHE_GIB in a trial's server_env overrides it per trial")
    ap.add_argument("--offload-trace", action="store_true", help="with --cpu-gib: log per-request offload lookups/finishes")
    ap.add_argument("--reserve", type=int, default=0, help="KV-gate reply reserve per request in place of the reply "
                    "cap (exact output; the engine preempts if replies outgrow it)")
    ap.add_argument("--ucap", type=int, default=0, help="batch expert-union cap: decode/verify steps read at most N "
                    "experts (top-N by summed router probability; ARC_UCAP; lossy)")
    ap.add_argument("--oea", type=int, default=0, help="batch-aware expert routing: each token keeps its top-K0 experts, "
                    "the rest filled from the step's union (ARC_OEA_K0; lossy)")
    ap.add_argument("--fp8-dense", default="", help="FP8 W8A8 for these dense families (gdn_in_proj_qkvz,qsa_qkv,attn_out)")
    ap.add_argument("--fp4-dense", default="", help="NVFP4 W4A4 for these dense families (fp4/fp4_dense.py; not with --fp8-dense)")
    ap.add_argument("--qsa-dump", type=int, default=0, help="QSA pick capture (qsa/qsa_dump.py): save every Nth eager QSA call's picks/addressing (max 24, after 480 s)")
    ap.add_argument("--kv", default="", choices=("", "int4", "fp8", "bf16"), help="diagnostic: KV cache dtype override after the arm")
    ap.add_argument("--indexer", default="", choices=("", "fp8", "bf16"), help="diagnostic: QSA indexer key dtype override after the arm")
    ap.add_argument("--kv4-pv-fp32", action="store_true", help="kv4 attention: FP32 tile weights as BF16 hi+lo MMAs")
    ap.add_argument("--humming-bi", action="store_true", help="Humming MoE heuristics in batch-invariant mode (deterministic)")
    ap.add_argument("--qsa-tie", action="store_true", help="QSA top-k ties at the k-th score broken by lowest block index "
                    "(deterministic selection; output sorted, so it implies --qsa-sort's read order)")
    ap.add_argument("--int2cold", action="store_true", help="item 4: x5's INT3 overlay with the 160 coldest kept experts "
                    "per layer re-rounded to INT2, served as hot uint3 + cold uint2 Humming tensors (int2/)")
    ap.add_argument("--qsa-sort", action="store_true", help="sort each row's QSA block selections (deterministic read order)")
    ap.add_argument("--divtrace-ops", action="store_true", help="with --divtrace: op-level trace of MoE reruns (layers 0, 1, 4)")
    ap.add_argument("--divtrace-sel", action="store_true", help="with --divtrace: record MoE top-k / QSA selections per call")
    ap.add_argument("--divtrace-pad", default="", help="with --divtrace: capture sizes; per-row modules re-run on inputs padded "
                    "to the next size, first M rows compared bit for bit (CG padding exactness)")
    ap.add_argument("--divtrace-rerun", default="mlp.experts", help="with --divtrace: module suffixes re-run in place (comma list)")
    ap.add_argument("--htune", default="", help="item 3: Humming MoE tile sweep in the worker (htune_runtime) under DIR; "
                    "pair with --gpu-script scripts/dfr_exact/htune/htune.py")
    ap.add_argument("--htune-seconds", type=int, default=2400)
    ap.add_argument("--divtrace", default="", help="diagnostic: per-module output checksums to DIR while DIR/tag exists (eager)")
    ap.add_argument("--prune-toggle", default="", help="screen mode: plain routing with experts dropped per layer while PATH lists them (JSON {layer: [served ids]}), eager")
    ap.add_argument("--fp8-bf16", default="", help="with --fp8-dense: LAYER:FAMILY,... kept BF16")
    ap.add_argument("--int3-fp4-down", action="store_true", help="B2 dn4: Humming int3 gate/up, NVFP4 down (needs a dn4 overlay)")
    ap.add_argument("--int3-fp4-layers", default="", help="with --int3-fp4-down: comma list of decoder layers served all-NVFP4 (B2 b)")
    ap.add_argument("--fp8-w16", action="store_true", help="with --fp8-dense: weight-only FP8 (Marlin W8A16), BF16 activations")
    ap.add_argument("--fp8-toggle", default="", help="with --fp8-dense: screen mode, BF16 kept, eager; BF16 while PATH exists")
    ap.add_argument("--fp8-calib", default="", help="with --fp8-dense: calibrated codes/scales dataset <owner>/<name> "
                    "(fp8/fp8_calib.py output; served from /kaggle/input/datasets/<owner>/<name>)")
    ap.add_argument("--keepalive-until", type=float, default=0, help="tail keepalive lifetime (s; served 45)")
    ap.add_argument("--notebook-seconds", type=int, default=0, help="launcher timeout (s); sizes the Nebius bound")
    ap.add_argument("--gate-kv", default="", metavar="LIMIT,OVERHEAD",
                    help="S1b: no admission while vLLM KV usage > LIMIT; OVERHEAD blocks added to each request's cost")
    ap.add_argument("--tail-multidrop", action="store_true", help="request-end tail entries for MTP stops with 2+ dropped tokens")
    ap.add_argument("--gdn-dump", default="", help="LAYERS: dump GDN chunk-op inputs/outputs of these model layers at prefill (gdn/arc_gdn_dump.py; GDN-state precision test)")
    ap.add_argument("--gdn-defer", default="", help="MODE: GDN deferred commit (gdn/arc_gdn_deferred.py + gdn_m2.cu built on the VM), real or check")
    ap.add_argument("--bench-profile", default="", help="with --gpu-script: JSON {prompt_tokens, concurrency, output_tokens}: forced-decode torch profiles after the script")
    ap.add_argument("--gpu-script", default="", help="SCRIPT[,INPUT...]: run SCRIPT (copied to bench/ with its inputs) "
                    "as the bench, beside the idle server (GPU microbenchmarks)")
    ap.add_argument("--gpu-args", default="[]", help="JSON list of the script's arguments (paths relative to the bundle)")
    ap.add_argument("--gpu-seconds", type=int, default=1800)
    ap.add_argument("--qsa-topk", type=int, default=0, help="QSA selected tokens per query (checkpoint 2,048; lossy): "
                    "ARC_QSA_TOPK_TOKENS, torch top-k for block counts the pinned kernels lack")
    ap.add_argument("--qsa-trace", type=int, default=0, help="QSA selection trace (KV-offload locality): eager server, "
                    "record N decode steps of selected 4-token groups after ARC_QSA_TRACE_START (qsa_trace.py)")
    ap.add_argument("--server-env", default="", help="KEY=VALUE[,KEY=VALUE] added to every trial's server env "
                    "(e.g. VLLM_MARLIN_INPUT_DTYPE=fp8: W4A8-FP8 Marlin MoE, lossy)")
    ap.add_argument("--cb-fraction", type=float, default=0.95, help="cache budget: fraction of max(gpu, host) blocks")
    ap.add_argument("--cb-host-margin", type=float, default=0.9, help="cache budget: host blocks factor (0: GPU-only budget)")
    ap.add_argument("--cache-budget", type=int, default=0, metavar="HEADROOM",
                    help="admit streams while live contexts + HEADROOM each fit max(GPU pool, host cache)")
    ap.add_argument("--topk", type=int, default=0, help="AV2C: num_experts_per_tok override (lossy)")
    ap.add_argument("--ple-fp8", action="store_true", help="AV2C: the NVFP4 checkpoint's FP8 PLE table (lossy)")
    ap.add_argument("--prune", default="", help="AV2C: expert-prune mask under data/expert_prune (lossy)")
    ap.add_argument("--watchdog", action="store_true",
                    help="engine watchdog: the trial restarts a dead vLLM server and games wait for it")
    ap.add_argument("--watchdog-test-kill", type=int, default=0,
                    help="with --watchdog: kill the EngineCore this many seconds into play (test)")
    ap.add_argument("--ple-bound", action="store_true",
                    help="bound the PLE short-conv prefill's padded packing (grouped prefills; same math)")
    ap.add_argument("--offload-eagle-tail", action="store_true",
                    help="with --cpu-gib: store the draft group's trailing full chunk once final (ARC_OFFLOAD_EAGLE_TAIL)")
    ap.add_argument("--offload-decode", action="store_true", help="with --cpu-gib: also offload generated tokens' KV")
    ap.add_argument("--add-file", action="append", default=[], help="REL=SRC: copy SRC into the bundle at REL")
    ap.add_argument("--study-patch", default="", help="JSON deep-merged into the whole study (null deletes a key)")
    ap.add_argument("--server-patch", default="", help="JSON deep-merged into the server block (null deletes a key)")
    ap.add_argument("--mtp", type=int, help="MTP draft depth override (exact: acceptance only)")
    ap.add_argument("--min-full-windows", type=int, help="override focus min_full_windows (MTP 4's extra state slots cost ~2%% of the pool)")
    ap.add_argument("--max-batched", type=int, default=0,
                    help="chunked-prefill size (max_num_batched_tokens; base 4,096); with CG the piecewise graphs extend to it")
    ap.add_argument("--pyspy", default="", help="play seconds (comma-separated) at which to record 60 s of py-spy on the "
                    "EngineCore process(es); not with PROF")
    ap.add_argument("--sampler", default="", help="play seconds (comma-separated) at which every server process samples "
                    "60 s of its main-thread Python stacks (in-process plugin; the GPU worker included); not with PROF/--pyspy")
    ap.add_argument("--toggle-ab", type=float, default=0, help="flip /tmp/arc-ab-flag every N s of play (A/B fixes read it; not with PROF/--pyspy/--sampler)")
    ap.add_argument("--gdn-mask-fix", action="store_true", help="GDN prefill: masked_fill_ for the no-initial-state rows (removes a stream sync per GDN layer per prefill step; exact)")
    ap.add_argument("--sync-audit", default="", help="play seconds (comma-separated; needs --toggle-ab, whose flag marks play start): every server process logs each synchronizing CUDA op with its stack for 60 s")
    ap.add_argument("--ple-sync-fix", action="store_true", help="PLE short-conv metadata: token permutation built on the CPU (removes a stream sync per MTP mixed step; exact)")
    ap.add_argument("--gdn-fp32", action="store_true", help="serve the checkpoint's fp32 GDN recurrent state (mamba_ssm_cache_dtype null; 4-bit block resized)")
    ap.add_argument("--graph-sizes", default="", help="comma list replacing server.cudagraph_capture_sizes (fewer graphs, smaller graph pool; the largest stays the capture bound)")
    ap.add_argument("--trim-budget", type=int, default=0, help="context budget cap in tokens (study.trim.budget_tokens): "
                    "drains trigger at this prompt size and drain to half of it; the window and the KV gate are unchanged (lossy harness change)")
    ap.add_argument("--tick", type=float, default=0, help="admission tick seconds (base 2.0): new requests enter in batches; "
                    "each batch costs one host-bound mixed-step sequence (gi3prof 17 Oct: ~0.75 s per batch)")
    ap.add_argument("--ramp", type=int, default=0, metavar="SECONDS",
                    help="with --cache-budget: open the stream slots linearly over SECONDS from a quarter (staggered drains)")
    ap.add_argument("--steady-replay", default="", metavar="CONTEXTS",
                    help="steady-state replay bench (scripts/steady_replay_inputs.py output) in place of the arena "
                         "trial: late-run serving load, tok/s only (scripts/dfr_exact/steady_replay.py)")
    ap.add_argument("--steady-warm", type=int, default=480, help="steady replay: seconds before the measured window")
    ap.add_argument("--steady-measure", type=int, default=600, help="steady replay: measured seconds")
    ap.add_argument("--steady-start", default="30000,95000", help="steady replay: start context range (tokens)")
    ap.add_argument("--steady-length-scale", type=float, default=0.5, help="steady replay: min_tokens = recorded reply "
                    "length x this (v3 at 1.0: replies averaged 2.0K against 976 recorded)")
    ap.add_argument("--steady-profile", default="", help="steady replay: 'AT1,AT2:ITERS' torch-profile ITERS engine steps of the live replay starting at each offset (s)")
    ap.add_argument("--steady-phases", default="", help="steady replay: JSON list of per-phase overrides run in turn on "
                    "one server (e.g. [{\"label\":\"base\"},{\"label\":\"r4k\",\"reserve_tokens\":4096}])")
    ap.add_argument("--steady-reserve", type=int, default=0, help="steady replay: KV-gate reply reserve (default: the cap)")
    ap.add_argument("--gate-blocks", type=float, help="override gate_blocks_per_stream (the exact KV gate still bars preemption)")
    a = ap.parse_args()
    out, v82 = Path(a.out), Path(a.v82)
    if out.exists():
        raise SystemExit(f"{out} exists")
    shutil.copytree(v82, out, ignore=shutil.ignore_patterns(*SKIP))
    for old, new in ((his.PRIORITY_OLD, his.PRIORITY_NEW), (his.SETTINGS_OLD, his.SETTINGS_NEW),
                     (his.SETTINGS2_OLD, his.SETTINGS2_NEW)):
        his.patch(out / "src/arena/priority.py", old, new)
    for old, new in ((his.VALUE_OLD, his.VALUE_NEW), (his.RANK_OLD, his.RANK_NEW), (his.PARKED_OLD, his.PARKED_NEW),
                     (his.NOTE_OLD, his.NOTE_NEW)):
        his.patch(out / "src/arena/scheduler.py", old, new)
    his.patch(out / "src/arena/trial.py", TOKEN_STOP_OLD, TOKEN_STOP_NEW)
    extras = tuple(x for x in a.extra.split(",") if x)
    if set(extras) - {"CG", "TICK4", "PROF", "IXF8", "DW", "KV4", "BFD", "DG", "T0"}:
        raise SystemExit("--extra takes CG, TICK4, PROF, IXF8, DW, KV4, BFD")
    big_sizes = max((int(x) for x in a.graph_sizes.split(",") if x), default=0) > 256  # exact CG (24 Oct)
    if "CG" in COMBOS.get(a.arm, (a.arm,)) + extras or big_sizes:
        bound = str(max(4096, a.max_batched))  # --max-batched past 4,096 extends the graph bound with it
        his.patch(out / "src/arena/serving.py", GRAPH_OLD, GRAPH_NEW.replace("4096", bound))
        his.patch(out / "src/arena/serving.py", GRAPH_MAX_OLD, GRAPH_MAX_NEW.replace("4096", bound))
        # The fork's hash is computed at stage time (serving_fork.fork_identity), so the fork may carry this bound.
        his.patch(out / "src/arc_infra/serving_setup.py", SETUP_ENV_OLD, SETUP_ENV_NEW.replace("4096", bound))
        his.patch(out / "src/arc_infra/serving_setup.py", SETUP_BOUND_OLD, SETUP_BOUND_NEW.replace("4096", bound))
    if a.watchdog:
        sys.path.insert(0, str(Path(__file__).parent / "watchdog"))
        import patches as watchdog_patches
        shutil.copyfile(Path(__file__).parent / "watchdog/engine_watchdog.py", out / "src/arena/engine_watchdog.py")
        for rel, old_text, new_text in watchdog_patches.PATCHES:
            his.patch(out / rel, old_text, new_text)
    if a.ple_bound:
        # 17 Oct: the PLE short-conv prefill packs all of a step's prefills padded to the longest query; one long chunk
        # with many short prefills asked for 1.06-1.48 GiB (CUDA OOM, engine dead, in 7 AV2C runs). Grouped above
        # 128 MiB per tensor (scripts/dfr_exact/ple_prefill_bound.py, appended to the worker's arc_expert_prune).
        rt = out / "src/arc_infra/expert_prune_runtime.py"
        his.patch(rt, "    p = the_plan or settings()\n    speculative =",
                  "    p = the_plan or settings()\n    if the_plan is None:\n        install_ple_prefill_bound(log=log)\n"
                  "    speculative =")
        rt.write_text(rt.read_text() + "\n\n" + (Path(__file__).parent / "ple_prefill_bound.py").read_text())
    if a.cpu_gib:
        # Host-RAM KV offload for the hybrid cache (arc2 7ccf5cd0): vLLM's OffloadingConnector refuses the QSA ring
        # (CircularBufferSpec); ArcOffloadingConnector hands it the prefix-cacheable groups only.
        shutil.copyfile(Path(__file__).parent / "offload/arc_offload_runtime.py", out / "src/arc_infra/arc_offload_runtime.py")
        for old, new in OFFLOAD_PATCHES:
            his.patch(out / "src/arena/serving.py", old, new)
        his.patch(out / "src/arena/compatibility.py", OFFLOAD_GUARD_OLD, OFFLOAD_GUARD_NEW)
        his.patch(out / "src/arena/cache_probe.py", PROBE_OLD, PROBE_NEW)
        his.patch(out / "bootstrap_arena.py", PROBE_ENV_OLD, PROBE_ENV_NEW)
        # v8.5 pools (~5.5M tokens) hold all 25 x 26K-word probe prefixes: no eviction, so no host load at the probe
        # (v85f off32 v1 died there, 21 Oct); gameplay exercises the round trip (record keeps a note).
        his.patch(out / "src/arena/cache_probe.py", PROBE_RT_OLD, PROBE_RT_NEW)
        if a.offload_decode:
            his.patch(out / "src/arena/serving.py", DECODE_OLD, DECODE_NEW)
    if a.pyspy:
        if a.arm == "PROF" or "PROF" in extras:
            raise SystemExit("--pyspy rides the PROF hook's anchor: not with PROF")
        his.patch(out / "src/arena/trial.py", PROFILE_OLD, PYSPY_NEW)
        his.patch(out / "bootstrap.py", EVIDENCE_OLD, EVIDENCE_NEW)
        (out / "src/arena/bin").mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).parent / "pyspy/py-spy", out / "src/arena/bin/py-spy")
        (out / "src/arena/bin/py-spy").chmod(0o755)
    if a.sampler:
        if a.pyspy or a.arm == "PROF" or "PROF" in extras:
            raise SystemExit("--sampler rides the PROF hook's anchor: not with PROF or --pyspy")
        his.patch(out / "src/arena/trial.py", PROFILE_OLD, SAMPLER_NEW)
        his.patch(out / "bootstrap.py", EVIDENCE_OLD, EVIDENCE_NEW)
        his.patch(out / "src/arc_infra/serving_setup.py", SAMPLER_ENV_OLD, SAMPLER_ENV_NEW)
        shutil.copytree(Path(__file__).parent / "sampler", out / "src/arena/sampler", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    if a.toggle_ab:
        if a.pyspy or a.sampler or a.arm == "PROF" or "PROF" in extras:
            raise SystemExit("--toggle-ab rides the PROF hook's anchor: not with PROF, --pyspy or --sampler")
        his.patch(out / "src/arena/trial.py", PROFILE_OLD, TOGGLE_NEW)
    if a.arm == "PROF" or "PROF" in extras:
        his.patch(out / "src/arena/trial.py", PROFILE_OLD, PROFILE_NEW)
        his.patch(out / "bootstrap.py", EVIDENCE_OLD, EVIDENCE_NEW)
    study = json.loads((out / "configs/comparison.json").read_text())
    if len(study["games"]) != 20 or study.get("level_budget_policy") != "priority" or study["active_limit"] != 48:
        raise SystemExit("not the v82tc-w128 c110 bundle")
    server = study["server"]
    kernels = server["serving_kernels"]
    streams = 48
    if a.arm in ("R", "AR", "AV2C"):
        ref = json.loads((Path(a.rl) / "configs/comparison.json").read_text())["server"]
        if ref.get("expert_prune") or ref.get("expert_overlay") or ref["serving_kernels"]["moe_backend"] != "marlin":
            raise SystemExit("--rl is not the unpruned stock NVFP4 reference profile")
        draft = server.get("draft_head")
        server.clear()
        server.update(json.loads(json.dumps(ref)))
        if draft:
            server["draft_head"] = draft  # the v8.2 draft head (exact either way; acceptance only)
        study["model_datasets"] = [d for d in study.get("model_datasets") or [] if d != INT3]
        # The reference profile's memory setting; no stall dump (it rides on the prune patch).
        for trial in study["trials"]:
            env = trial.get("server_env") or {}
            env.pop("ARENA_STALL_DUMP", None)
            env["ARENA_GPU_MEMORY_UTILIZATION"] = "0.96"
            trial["server_env"] = env
        study.pop("stall_dump_seconds", None)
        streams = 21
        if a.arm in ("AR", "AV2C"):
            # Intel AutoRound W4A16 (the 5 Oct registry from the throughput-arm branch: size-checked shards, hashed
            # small files); no expert overlay or prune mask may apply to it (arena.serving refuses both).
            src = Path(__file__).resolve().parent / "autoround"
            shutil.copyfile(src / "research_checkpoint.py", out / "src/arena/research_checkpoint.py")
            shutil.copyfile(src / "intel-autoround-w4a16.json", out / "src/arena/models/intel-autoround-w4a16.json")
            prefix = "flash-input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround"
            study["research_checkpoint"] = {"repo": "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound",
                                            "revision": "4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897",
                                            "path": f"/assets/{prefix}/transformers/default/1"}
            study["nebius_assets"] = [prefix]
    elif a.arm == "P":
        server.pop("expert_overlay")
        kernels.pop("expert_int3")
        kernels["moe_backend"] = "marlin"
        study["model_datasets"] = [d for d in study.get("model_datasets") or [] if d != INT3]
        streams = 48  # the served pool decides (focus_oversubscription density): about 40
    elif a.arm == "K":
        kernels.pop("kv_cache_4bit")
        kernels["kv_cache_8bit"] = "fp8"
        kernels["indexer_fp8"] = False
        streams = 48  # the served pool decides (focus_oversubscription density): about 20-24
    elif a.arm == "Kb":
        kernels["indexer_fp8"] = False
    elif a.arm in ("Kf", "DKf"):
        kernels.pop("kv_cache_4bit")
        kernels["kv_cache_8bit"] = "fp8"
        streams = 27
    elif a.arm == "D":
        server["dense_kernels"]["int8_weights"] = []
    elif a.arm == "DKb":
        # BF16 dense + BF16 indexer keys (int4 KV kept); the served pool caps the 48 streams via the gate.
        server["dense_kernels"]["int8_weights"] = []
        kernels["indexer_fp8"] = False
    if a.kv:  # divtrace ablations (21 Oct, P0): KV dtype after the arm
        kernels.pop("kv_cache_4bit", None); kernels.pop("kv_cache_8bit", None)
        if a.kv == "int4":
            kernels["kv_cache_4bit"] = "int4"
        elif a.kv == "fp8":
            kernels["kv_cache_8bit"] = "fp8"
    if a.indexer:
        kernels["indexer_fp8"] = a.indexer == "fp8"
    if a.arm == "AV2C":
        # A v2's whole serving configuration on our vLLM stack: his AutoRound W4A16 checkpoint
        # (AR), BF16 dense (no int8 dense weights), BF16 indexer keys (his QSA pool keeps them BF16), FP8 KV, 3 MTP
        # draft steps (his NEXTN 3), his 10 streams; sampler / reply cap / effort are the v8.2 harness's in both.
        kernels = server["serving_kernels"]  # R/AR replaced the server block: the old kernels dict is stale
        server["dense_kernels"]["int8_weights"] = []
        kernels["indexer_fp8"] = False
        kernels.pop("kv_cache_4bit", None)
        kernels["kv_cache_8bit"] = "fp8"
        server["mtp_tokens"] = 3
        sizes = server["cudagraph_capture_sizes"]
        server["cudagraph_capture_sizes"] = sorted(set(sizes) | {40, 64, 80, 96, 104})
        server["max_cudagraph_capture_size"] = 104
        streams = 10
        if "IXF8" in extras:  # FP8 indexer keys (Kb ablation: no quality cost)
            kernels["indexer_fp8"] = True
        if "KV4" in extras:  # 2K track (16 Oct): v8.2's int4 KV (2x the pool; lossy: KL .0325, Kf ablation no score cost)
            kernels.pop("kv_cache_8bit", None)
            kernels["kv_cache_4bit"] = "int4"
        if "DW" in extras:  # item 3: the large BF16 projections on the tuned Triton mid-M path
            # (cuBLAS/CUTLASS sm80 today at M > 32: 13.7% of the 16-stream step); kept per shape and bucket only when
            # >= 3% faster than the pinned path and within its FP32 error (dense_wide_runtime tuner).
            dense = server["dense_kernels"]
            dense["mid_wide_shapes"] = sorted({tuple(x) for x in dense.get("mid_wide_shapes") or []}
                                              | {(16384, 2560), (13312, 2560), (2560, 6144), (10240, 2560)})
            dense["mid_wide_shapes"] = [list(x) for x in dense["mid_wide_shapes"]]
            dense["mid_wide_buckets"] = sorted(set(dense.get("mid_wide_buckets") or []) | {104})
    parts = COMBOS.get(a.arm, (a.arm,)) + extras
    if a.mtp is not None:
        server["mtp_tokens"] = a.mtp
    for item in a.add_file:  # REL=SRC copied into the bundle (e.g. another build's expert-prune mask)
        rel, src = item.split("=", 1)
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out / rel)
    if a.server_patch:  # deep merge into the server block; null deletes (determinism bisect, 19 Oct)
        def merge(dst, src):
            for k, v in src.items():
                if v is None:
                    dst.pop(k, None)
                elif isinstance(v, dict) and isinstance(dst.get(k), dict):
                    merge(dst[k], v)
                else:
                    dst[k] = v
        merge(server, json.loads(a.server_patch))
        study["server_patch"] = json.loads(a.server_patch)
        mask = (server.get("expert_prune") or {}).get("mask")
        if mask and not (out / mask).exists():  # a patched mask ships from data/expert_prune, sha-checked
            src = Path(__file__).resolve().parents[2] / "data/expert_prune" / Path(mask).name
            (out / mask).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out / mask)
            if sha(out / mask) != server["expert_prune"]["sha256"]:
                raise SystemExit(f"{mask}: sha256 does not match the server patch")
    if a.study_patch:  # the same on the whole study (e.g. cache_experiment.prefix_cache.state_cotenant)
        merge_study = json.loads(a.study_patch)

        def merge2(dst, src):
            for k, v in src.items():
                if v is None:
                    dst.pop(k, None)
                elif isinstance(v, dict) and isinstance(dst.get(k), dict):
                    merge2(dst[k], v)
                else:
                    dst[k] = v
        merge2(study, merge_study)
        study["study_patch"] = merge_study
    study["cache_experiment"]["cpu_cache_gib"] = a.cpu_gib
    if a.offload_trace:  # per-request connector records (ARC_OFFLOAD_LOOKUP / ARC_OFFLOAD_FINISH) in the server log
        for trial in study["trials"]:
            trial.setdefault("server_env", {})["ARC_OFFLOAD_TRACE"] = "1"
    if a.server_env:
        for trial in study["trials"]:
            trial.setdefault("server_env", {}).update(dict(kv.split("=", 1) for kv in a.server_env.split(",")))
    if a.prune:
        if a.arm != "AV2C":
            raise SystemExit("--prune is for AV2C (other arms carry their masks by level)")
        src = Path(__file__).resolve().parents[2] / "data/expert_prune" / a.prune
        mask = json.loads(src.read_text())
        kept = 512 - int(mask["pruned_per_layer"])
        rel = f"configs/expert-prune/{a.prune}"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out / rel)
        server["expert_prune"] = {"mask": rel, "sha256": sha(out / rel)}
        his.patch(out / "src/arena/serving.py", PRUNE_GUARD_OLD, PRUNE_GUARD_NEW)
        if a.cpu_gib:
            his.patch(out / "src/arena/cache_probe.py", PROBE_RT_OLD, PROBE_RT_NEW)
        dense = server["dense_kernels"]
        dense["mid_wide_shapes"] = [list(x) for x in sorted({tuple(x) for x in dense.get("mid_wide_shapes") or []}
                                                            | {(kept, 2560)})]  # the pruned router
    if a.reserve:
        # Reply reserve (18 Oct): late replies average ~1K tokens (median 280) but each request reserved the 16K cap,
        # filling the gate at 46 in flight. The engine's own preemption covers a reply that outgrows the reserve.
        his.patch(out / "src/arena/gate.py", "            return self.next_estimate + self.cap\n",
                  f"            return self.next_estimate + min(self.cap, {int(a.reserve)})  # reply reserve (18 Oct)\n")
        study["reply_reserve_tokens"] = a.reserve
    if a.divtrace:  # divergence trace (21 Oct): exact per-module output checksums while a tag file exists, eager
        rt = out / "src/arc_infra/dense_kernels_runtime.py"
        rt.write_text(rt.read_text() + (Path(__file__).parent / "det/divtrace_runtime.py").read_text())
        his.patch(rt, '''    overlay handled ``module`` (the pinned SM103-only body is then skipped)."""
    config = load_config()
''', '''    overlay handled ``module`` (the pinned SM103-only body is then skipped)."""
    install_divtrace(module)
    config = load_config()
''')
        for trial in study["trials"]:
            trial.setdefault("server_env", {}).update(ARC_DIVTRACE=a.divtrace, ARENA_ENFORCE_EAGER="1",
                                                      ARC_DIVTRACE_DUMP="model.layers.0.mlp.experts:0",
                                                      ARC_DIVTRACE_RERUN=a.divtrace_rerun,
                                                      ARC_DIVTRACE_SEL="1" if a.divtrace_sel else "0",
                                                      ARC_DIVTRACE_OPS="1" if a.divtrace_ops else "0",
                                                      ARC_DIVTRACE_PAD=a.divtrace_pad)
    if a.htune:  # Humming MoE tile sweep (item 3, 22 Oct): worker hook + background sweep; server config unchanged
        if a.divtrace:
            raise SystemExit("--htune: not with --divtrace (same enable() anchor)")
        rt = out / "src/arc_infra/dense_kernels_runtime.py"
        rt.write_text(rt.read_text() + (Path(__file__).parent / "htune/htune_runtime.py").read_text())
        his.patch(rt, '''    overlay handled ``module`` (the pinned SM103-only body is then skipped)."""
    config = load_config()
''', '''    overlay handled ``module`` (the pinned SM103-only body is then skipped)."""
    install_htune(module)
    config = load_config()
''')
        for trial in study["trials"]:
            trial.setdefault("server_env", {}).update(ARC_HTUNE=a.htune, ARC_HTUNE_SECONDS=str(a.htune_seconds))
    if a.prune_toggle:  # expert-drop KL screens (21 Oct): the plain-routing hook with a per-layer drop file, eager
        if a.oea or a.ucap:
            raise SystemExit("--prune-toggle is a screen mode: not with --oea/--ucap")
        shutil.copyfile(Path(__file__).parent / "oea/oea_runtime.py", out / "src/arc_infra/oea_runtime.py")
        his.patch(out / "src/arena/serving.py", OEA_OLD, OEA_NEW)
        for trial in study["trials"]:
            trial.setdefault("server_env", {}).update(ARC_PRUNE_TOGGLE=a.prune_toggle, ARENA_ENFORCE_EAGER="1")
    if a.oea or a.ucap:  # OEA k0 / union cap (arc_infra/oea_runtime.py -> vllm/arc_oea.py)
        shutil.copyfile(Path(__file__).parent / "oea/oea_runtime.py", out / "src/arc_infra/oea_runtime.py")
        his.patch(out / "src/arena/serving.py", OEA_OLD, OEA_NEW)
        for trial in study["trials"]:
            trial.setdefault("server_env", {})["ARC_OEA_K0"] = str(a.oea)
            if a.ucap:
                trial["server_env"]["ARC_UCAP"] = str(a.ucap)
    if a.qsa_topk:
        if a.oea or a.ucap or a.qsa_trace or a.qsa_topk % 4 or not 256 <= a.qsa_topk < 2048:
            raise SystemExit("--qsa-topk: a multiple of 4 in [256, 2048), alone on the extract anchor")
        his.patch(out / "src/arena/serving.py", OEA_OLD, QTOPK_NEW)
        for trial in study["trials"]:
            trial.setdefault("server_env", {})["ARC_QSA_TOPK_TOKENS"] = str(a.qsa_topk)
        study["qsa_topk_tokens"] = a.qsa_topk
    if a.kv4_pv_fp32:
        for trial in study["trials"]:
            trial.setdefault("server_env", {})["ARC_KV4_PV_FP32"] = "1"
    if a.qsa_tie:
        a.qsa_sort = True
    if a.qsa_sort or a.humming_bi:
        if a.oea or a.ucap or a.qsa_trace or a.qsa_topk or a.kvoff:
            raise SystemExit("--qsa-sort / --humming-bi: alone on the extract anchor")
        his.patch(out / "src/arena/serving.py", OEA_OLD,
                  (QSORT_BODY if a.qsa_sort else "") + (HBI_BODY if a.humming_bi else "") + EXTRACT_TAIL)
        for trial in study["trials"]:
            if a.qsa_sort:
                trial.setdefault("server_env", {})["ARC_QSA_SORT"] = "1"
            if a.qsa_tie:
                trial["server_env"]["ARC_QSA_TIE"] = "1"
            if a.humming_bi:
                trial.setdefault("server_env", {})["ARC_HUMMING_BI"] = "1"
    if a.int2cold:  # item 4: the 8 Oct INT2-cold serving kernel (branch swkv 1d40f520) on x5
        src_ds = server.get("expert_overlay", {}).get("dataset")
        if src_ds not in INT2COLD_FOR or kernels.get("expert_int3") != 16:
            raise SystemExit("--int2cold: needs an INT3 overlay with a re-rounded INT2-cold copy (INT2COLD_FOR)")
        if a.humming_bi:
            raise SystemExit("--int2cold: not with --humming-bi (both edit fused_humming_moe.py)")
        here = Path(__file__).resolve().parent / "int2"
        for name in ("expert_int2_patch.py", "expert_int2_runtime.py"):
            shutil.copyfile(here / name, out / "src/arc_infra" / name)
        import subprocess
        # offset hunks (the offload connector's lines above them) apply cleanly; only fuzz or rejects fail the build
        subprocess.run(["patch", "-p1", "-N", "-s", "-F", "0", "--no-backup-if-mismatch", "-i", str(here / "wiring.patch")],
                       cwd=out, check=True)
        if list(out.rglob("*.rej")) or list(out.rglob("*.orig")):
            raise SystemExit("--int2cold: the wiring patch left .rej/.orig files")
        kernels["expert_int2"] = True
        target = INT2COLD_FOR[src_ds]
        server["expert_overlay"] = dict(target)
        if study.get("server_patch", {}).get("expert_overlay"):
            study["server_patch"]["expert_overlay"] = dict(target)
        study["model_datasets"] = [target["dataset"] if d == src_ds else d for d in study["model_datasets"]]
    if a.kvoff:
        if a.oea or a.ucap or a.qsa_trace or a.qsa_topk or a.cpu_gib:
            raise SystemExit("--kvoff: alone on the extract anchor, and without the --cpu-gib connector")
        total, dev, *slots = a.kvoff.split(",")
        shutil.copyfile(Path(__file__).parent / "kvoff/kvoff_runtime.py", out / "src/arc_infra/kvoff_runtime.py")
        his.patch(out / "src/arena/serving.py", OEA_OLD, KVOFF_NEW)
        for trial in study["trials"]:
            trial.setdefault("server_env", {}).update(ARC_KVOFF="1", ARC_KVOFF_TOTAL_BLOCKS=total,
                                                      ARC_KVOFF_DEV_BLOCKS=dev,
                                                      ARC_KVOFF_CACHE_SLOTS=slots[0] if slots else "0")
        study["kvoff"] = {"total_blocks": total, "dev_gdn_blocks": int(dev), "cache_slots": int(slots[0]) if slots else 0}
    if a.fp8_dense:  # FP8 W8A8 dense projections (fp8/fp8_dense.py)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fp8 import fp8_dense
        fams = a.fp8_dense.split(",")
        rt = out / "src/arc_infra/dense_kernels_runtime.py"
        rt.write_text(rt.read_text() + fp8_dense.RUNTIME_CODE)
        for name, old, new in fp8_dense.PATCHES:
            his.patch(out / "src/arc_infra" / name, old, new)
        dk = study["server"].get("dense_kernels")
        if not dk or not dk.get("skinny"):
            raise SystemExit("--fp8-dense needs the served dense_kernels overlay (skinny on)")
        dk["fp8_w8a8"] = sorted(set(fams))
        study["fp8_dense"] = {"families": dk["fp8_w8a8"], "toggle": a.fp8_toggle or None}
    if a.fp4_dense:  # NVFP4 W4A4 dense projections (fp4/fp4_dense.py)
        if a.fp8_dense:
            raise SystemExit("--fp4-dense and --fp8-dense are alternatives")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fp4 import fp4_dense
        fams = a.fp4_dense.split(",")
        rt = out / "src/arc_infra/dense_kernels_runtime.py"
        rt.write_text(rt.read_text() + fp4_dense.RUNTIME_CODE)
        for name, old, new in fp4_dense.PATCHES:
            his.patch(out / "src/arc_infra" / name, old, new)
        dk = study["server"].get("dense_kernels")
        if not dk or not dk.get("skinny"):
            raise SystemExit("--fp4-dense needs the served dense_kernels overlay (skinny on)")
        dk["fp4_w4a4"] = sorted(set(fams))
        study["fp4_dense"] = {"families": dk["fp4_w4a4"]}
    if a.fp8_calib:
        if not a.fp8_dense or a.fp8_calib.count("/") != 1:
            raise SystemExit("--fp8-calib needs --fp8-dense and an <owner>/<name> dataset")
        for trial in study["trials"]:
            trial.setdefault("server_env", {}).update(ARC_FP8_CALIB=f"/kaggle/input/datasets/{a.fp8_calib}")
        study["model_datasets"] = [*study.get("model_datasets", []), a.fp8_calib]
        study["fp8_dense"]["calib"] = a.fp8_calib
    if a.int3_fp4_down:  # B2 dn4 (23 Oct): int3 gate/up, NVFP4 down on Humming (overlay must carry NVFP4 down)
        if kernels.get("moe_backend") != "humming" or not kernels.get("expert_int3"):
            raise SystemExit("--int3-fp4-down needs the Humming int3 arm")
        rt = out / "src/arc_infra/expert_int3_runtime.py"
        text = rt.read_text()
        old_rt = ("    if schema.b_dtype != dtypes.float4e2m1 or schema.weight_scale_group_size != 16:\n"
                  "        return schema, tensors\n")
        new_rt = old_rt + (
            "    if os.environ.get('ARC_EXPERT_INT3_FP4_DOWN') == '1':  # B2 dn4: w2 (and w13 of listed layers) stay NVFP4\n"
            "        import re, sys\n"
            "        caller = sys._getframe(1).f_locals  # humming_utils._convert_sublayer_to_humming\n"
            "        sub, lname = caller.get('sublayer_name'), str(getattr(caller.get('layer'), 'layer_name', ''))\n"
            "        if sub not in ('w13', 'w2'):\n"
            "            raise RuntimeError(f'ARC_EXPERT_INT3_FP4_DOWN: caller sublayer_name {sub!r}')\n"
            "        keep = {int(x) for x in os.environ.get('ARC_EXPERT_INT3_FP4_LAYERS', '').split(',') if x}\n"
            "        m = re.search(r'layers\\.(\\d+)\\.', lname)\n"
            "        if keep and m is None:\n"
            "            raise RuntimeError(f'ARC_EXPERT_INT3_FP4_LAYERS: no layer index in {lname!r}')\n"
            "        if sub == 'w2' or (m is not None and int(m.group(1)) in keep):\n"
            "            print('ARC_EXPERT_INT3_FP4_DOWN keep NVFP4', sub, lname, tuple(tensors['weight'].shape), flush=True)\n"
            "            return schema, tensors\n")
        if text.count(old_rt) != 1:
            raise SystemExit("expert_int3_runtime anchor not found")
        rt.write_text(text.replace(old_rt, new_rt))
        for trial in study["trials"]:
            trial.setdefault("server_env", {})["ARC_EXPERT_INT3_FP4_DOWN"] = "1"
            if a.int3_fp4_layers:
                trial["server_env"]["ARC_EXPERT_INT3_FP4_LAYERS"] = ",".join(str(int(x)) for x in a.int3_fp4_layers.split(","))
    if a.fp8_w16:  # A1 (23 Oct): weight-only FP8 through Marlin, BF16 activations
        if not a.fp8_dense:
            raise SystemExit("--fp8-w16 needs --fp8-dense")
        for trial in study["trials"]:
            trial.setdefault("server_env", {})["ARC_FP8_W16"] = "1"
        study["fp8_dense"]["w16"] = True
    if a.fp8_bf16:  # per-layer BF16 kept inside the FP8 families (21 Oct)
        if not a.fp8_dense:
            raise SystemExit("--fp8-bf16 needs --fp8-dense")
        for trial in study["trials"]:
            trial.setdefault("server_env", {})["ARC_FP8_BF16"] = a.fp8_bf16
    if a.fp8_toggle:
        if not a.fp8_dense:
            raise SystemExit("--fp8-toggle needs --fp8-dense")
        for trial in study["trials"]:
            trial.setdefault("server_env", {}).update(ARC_FP8_TOGGLE=a.fp8_toggle, ARENA_ENFORCE_EAGER="1")
    if a.qsa_dump:  # QSA pick capture for the query-union tiling bench (qsa/qsa_dump.py)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from qsa import qsa_dump
        rt = out / "src/arc_infra/kv4_runtime.py"
        rt.write_text(rt.read_text() + qsa_dump.RUNTIME_CODE)
        for name, old, new in qsa_dump.PATCHES:
            his.patch(out / "src/arc_infra" / name, old, new)
        for trial in study["trials"]:
            trial.setdefault("server_env", {}).update(ARC_QSA_DUMP=f"/tmp/arena/trials/{trial['name']}/qsa-dump",
                                                      ARC_QSA_DUMP_EVERY=str(a.qsa_dump))
        study["qsa_dump"] = {"every": a.qsa_dump}
    if a.keepalive_until:  # idle-prefix protection: tail keepalive refreshes past 45 s (prefix_tail_runtime)
        his.patch(out / "src/arena/prefix_tail_runtime.py",
                  "KEEPALIVE_AFTER, KEEPALIVE_EVERY, KEEPALIVE_UNTIL = 5.0, 5.0, 45.0",
                  f"KEEPALIVE_AFTER, KEEPALIVE_EVERY, KEEPALIVE_UNTIL = 5.0, 5.0, {float(a.keepalive_until)}")
        study["keepalive_until"] = float(a.keepalive_until)
    if a.notebook_seconds:  # the launcher's timeout and spend bound (default inherited: 86,100 s)
        study["notebook_seconds"] = int(a.notebook_seconds)
    if a.gate_kv:
        lim, over = a.gate_kv.split(",")
        if not (0.5 < float(lim) < 1.0 and 0 <= int(over) <= 8):
            raise SystemExit("--gate-kv LIMIT in (0.5, 1), OVERHEAD 0..8 blocks")
        study["gate_kv"] = {"limit": float(lim), "overhead_units": int(over)}
        for name, old, new in GATEKV_PATCHES:
            his.patch(out / "src/arena" / name, old, new)
    if a.tail_multidrop:
        for old, new in TAILMD_PATCHES:
            his.patch(out / "src/arena/prefix_tail_runtime.py", old, new)
        study["tail_multidrop"] = True
    if a.gdn_defer:
        if a.oea or a.ucap or a.qsa_trace or a.qsa_topk or a.kvoff:
            raise SystemExit("--gdn-defer: alone on the extract anchor")
        mode = a.gdn_defer
        if mode not in ("real", "check"):
            raise SystemExit("--gdn-defer MODE with MODE real or check")
        cu = Path(__file__).parent / "gdn/gdn_m2.cu"
        shutil.copyfile(Path(__file__).parent / "gdn/arc_gdn_deferred.py", out / "src/arc_infra/gdn_deferred_runtime.py")
        shutil.copyfile(cu, out / "src/arc_infra/arc_gdn_m2.cu")
        his.patch(out / "src/arena/serving.py", OEA_OLD, GDND_NEW)
        for trial in study["trials"]:
            trial.setdefault("server_env", {}).update(
                ARC_GDN_DEFER=mode, ARC_GDN_DEFER_W=str(int(server.get("mtp_tokens") or 1) + 1),
                ARC_GDN_DEFER_STATS=f"/tmp/arena/trials/{trial['name']}/gdn-defer-stats.json")
        study["gdn_defer"] = {"mode": mode, "cu_sha256": sha(cu)}
    if a.gdn_dump:
        if a.gdn_defer or a.oea or a.qsa_trace:
            raise SystemExit("--gdn-dump: alone on the extract anchor")
        shutil.copyfile(Path(__file__).parent / "gdn/arc_gdn_dump.py", out / "src/arc_infra/gdn_dump_runtime.py")
        his.patch(out / "src/arena/serving.py", OEA_OLD, GDNDUMP_NEW)
        for trial in study["trials"]:
            trial.setdefault("server_env", {}).update(
                ARC_GDN_DUMP_LAYERS=a.gdn_dump, ARC_GDN_DUMP_DIR=f"/tmp/arena/trials/{trial['name']}/gdn-dump")
        study["gdn_dump"] = {"layers": a.gdn_dump, "source_sha256": sha(Path(__file__).parent / "gdn/arc_gdn_dump.py")}
    if a.qsa_trace:  # qsa_trace.py -> vllm/arc_qsa_trace.py, hooked onto QSAIndexer; eager (graphs skip Python)
        if a.oea or a.ucap:
            raise SystemExit("--qsa-trace and --oea/--ucap patch the same anchor")
        shutil.copyfile(Path(__file__).parent / "qsa_trace.py", out / "src/arc_infra/qsa_trace.py")
        his.patch(out / "src/arena/serving.py", OEA_OLD, QSAT_NEW)
        for trial in study["trials"]:
            trial.setdefault("server_env", {}).update(
                ARENA_ENFORCE_EAGER="1", ARC_QSA_TRACE_DIR=f"/tmp/arena/trials/{trial['name']}/qsa-trace",
                ARC_QSA_TRACE_STEPS=str(a.qsa_trace))
        study["qsa_trace_steps"] = a.qsa_trace
    if a.cache_budget:  # cache-budget admission (equal scheduler; CB_* patches)
        if a.arm == "O" or study.get("level_budget_policy") != "priority":
            raise SystemExit("--cache-budget is wired for the equal priority scheduler")
        sched, trial = out / "src/arena/scheduler.py", out / "src/arena/trial.py"
        for old_text, new_text in ((CB_FIELD_OLD, CB_FIELD_NEW), (CB_EVICT_OLD, CB_EVICT_NEW), (CB_FILL_OLD, CB_FILL_NEW),
                                   (CB_HAND_OLD, CB_HAND_NEW), (CB_PROG_OLD, CB_PROG_NEW), (CB_TICK_OLD, CB_TICK_NEW)):
            his.patch(sched, old_text, new_text)
        his.patch(trial, CB_TRIAL_OLD, CB_TRIAL_NEW)
        his.patch(out / "src/arena/scheduler_audit.py", CB_AUDIT_OLD, CB_AUDIT_NEW)
        probe = out / "src/arena/cache_probe.py"
        if a.cpu_gib and PROBE_RT_OLD in probe.read_text():  # large pools hold every probe prefix (Gi3 v1)
            his.patch(probe, PROBE_RT_OLD, PROBE_RT_NEW)
        study["cache_budget"] = {"headroom_tokens": a.cache_budget, "fraction": a.cb_fraction, "host_margin": a.cb_host_margin,
                                 "min_slots": 4}
        if a.ramp:
            his.patch(trial, RAMP_TRIAL_OLD, RAMP_TRIAL_NEW)
            his.patch(sched, RAMP_FILL_OLD, RAMP_FILL_NEW)
            study["cache_budget"].update(ramp_seconds=a.ramp, ramp_start=max(4, (a.streams or 40) // 4))
    if a.topk and a.arm == "AV2C":  # 2K track G2: fewer routed experts per token via the checkpoint copy's config (lossy)
        study["research_checkpoint"]["text_overrides"] = {"num_experts_per_tok": a.topk}
    elif a.topk:
        # NVFP4 arms (18 Oct, 2K track): the pinned config stays; the worker's prepare_config sets the target's
        # num_experts_per_tok from ARC_TOPK before the model is built (the MTP draft keeps its own config).
        if not a.prune and a.arm not in ("T50", "T62", "T25", "T12", "base", "T37A"):
            raise SystemExit("--topk on an NVFP4 arm needs the expert-prune runtime (a masked arm)")
        his.patch(out / "src/arc_infra/expert_prune_runtime.py", TOPK_OLD, TOPK_NEW)
        for trial in study["trials"]:
            trial.setdefault("server_env", {})["ARC_TOPK"] = str(a.topk)
    if a.ple_fp8:
        if a.arm != "AV2C":
            raise SystemExit("--ple-fp8 is for AV2C (the NVFP4 arms serve the FP8 table already)")
        study["research_checkpoint"]["ple_fp8"] = {"root": NV_ROOT, "files": NV_PLE}
        his.patch(out / "src/arena/serving.py", PLEF_OLD, PLEF_NEW)
        for trial in study["trials"]:
            trial.setdefault("server_env", {})["ARC_PLE_FP8"] = "1"
    if a.offload_eagle_tail:  # one chunk more per restored turn (arc_offload_runtime: _storable_chunks)
        for trial in study["trials"]:
            trial.setdefault("server_env", {})["ARC_OFFLOAD_EAGLE_TAIL"] = "1"
    if "DKb" in parts and len(parts) > 1:
        server["dense_kernels"]["int8_weights"] = []
        kernels["indexer_fp8"] = False
    level = next((p for p in parts if p in LEVELS), None)
    if level:
        name, kept = LEVELS[level]
        src = Path(__file__).resolve().parents[2] / "data/expert_prune" / name
        mask = json.loads(src.read_text())
        if mask["pruned_per_layer"] != 512 - kept or any(len(v) != 512 - kept for v in mask["pruned"].values()):
            raise SystemExit(f"{name} does not keep {kept}")
        old = out / server["expert_prune"]["mask"]
        old.unlink()
        rel = f"configs/expert-prune/{name}"
        shutil.copyfile(src, out / rel)
        server["expert_prune"] = {"mask": rel, "sha256": sha(out / rel)}
        server["expert_overlay"] = dict(INT3_ALL)
        study["model_datasets"] = [INT3_ALL["dataset"] if d == INT3 else d for d in study["model_datasets"]]
        dense = server["dense_kernels"]
        if dense["mid_wide_shapes"] != [[320, 2560]]:
            raise SystemExit("the base router shape changed")
        dense["mid_wide_shapes"] = [[kept, 2560]]  # the pruned router on the same mid-M path as base's [320, 2560]
        probe = out / "src/arena/cache_probe.py"
        if a.cpu_gib and kept <= 256 and PROBE_RT_OLD in probe.read_text():  # Gi3 v1 died at the probe
            his.patch(probe, PROBE_RT_OLD, PROBE_RT_NEW)
    if "T0" in parts:  # B0 split (23 Oct): no prune, all 512 experts on the all-expert excl5 INT3 overlay
        if level or a.arm in ("R", "AR", "AV2C", "P", "T37A") or "expert_prune" not in server:
            raise SystemExit("T0 goes on a pruned INT3 arm without another prune level")
        (out / server.pop("expert_prune")["mask"]).unlink()
        server["expert_overlay"] = dict(INT3_ALL)
        study["model_datasets"] = [INT3_ALL["dataset"] if d == INT3 else d for d in study["model_datasets"]]
        dense = server["dense_kernels"]
        if dense["mid_wide_shapes"] != [[320, 2560]]:
            raise SystemExit("the base router shape changed")
        dense["mid_wide_shapes"] = [[512, 2560]]
        # The stall dump rides on the prune patch (arena.serving.with_stall_dump): off without a prune (b0t0i3-v2).
        for trial in study["trials"]:
            (trial.get("server_env") or {}).pop("ARENA_STALL_DUMP", None)
        study.pop("stall_dump_seconds", None)
    if "BFD" in extras:  # BF16 dense on a v8.2-stack arm (int8 dense cost score in the D ablation)
        server["dense_kernels"]["int8_weights"] = []
    if a.arm == "T37A":
        # Confound check: base's t37 mask unchanged, on the all-expert excl5 INT3 overlay of T25/T12.
        server["expert_overlay"] = dict(INT3_ALL)
        study["model_datasets"] = [INT3_ALL["dataset"] if d == INT3 else d for d in study["model_datasets"]]
    if "CG" in parts:
        # Exact throughput: steps with a prefill over 104 tokens ran without a CUDA graph (276-310 ms,
        # GPU busy 17-42%, PROF run); piecewise graphs up to max_num_batched_tokens (4,096). Same compiled pieces.
        sizes = server["cudagraph_capture_sizes"]
        if sizes[-1] != 104 or server["max_num_batched_tokens"] != 4096:
            raise SystemExit("base capture sizes changed")
        server["cudagraph_capture_sizes"] = sizes + [128, 160, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536,
                                                     2048, 2560, 3072, 4096]
        server["max_cudagraph_capture_size"] = 4096
    if a.max_batched:
        server["max_num_batched_tokens"] = a.max_batched
        if "CG" in parts:
            sizes = server["cudagraph_capture_sizes"]
            step = 1024 if a.max_batched > 4096 else 512
            server["cudagraph_capture_sizes"] = sorted({x for x in sizes if x <= a.max_batched}
                                                       | set(range(4096 + step, a.max_batched + 1, step)) | {a.max_batched})
            server["max_cudagraph_capture_size"] = a.max_batched
    patch_probe_retry(out / "src/arena/cache_probe.py")
    if a.toggle_ab:
        study["toggle_ab_seconds"] = float(a.toggle_ab)
    if a.ple_sync_fix:
        if not (server.get("stack_kernels") or {}).get("async_h2d"):
            raise SystemExit("--ple-sync-fix extends stack_kernels.async_h2d")
        patch_ple_sync(out / "src/arc_infra/stack_kernels_patch.py", ab=bool(a.toggle_ab))
    if a.gdn_mask_fix:
        if not (server.get("stack_kernels") or {}).get("async_h2d"):
            raise SystemExit("--gdn-mask-fix rides stack_kernels.async_h2d")
        patch_gdn_mask(out / "src/arc_infra/stack_kernels_patch.py", ab=bool(a.toggle_ab))
    if a.sync_audit:
        if not a.toggle_ab or a.sampler:
            raise SystemExit("--sync-audit needs --toggle-ab (play start) and is not with --sampler")
        his.patch(out / "bootstrap.py", EVIDENCE_OLD, EVIDENCE_NEW)
        his.patch(out / "src/arc_infra/serving_setup.py", SAMPLER_ENV_OLD,
                  f'    env["ARC_SYNC_AUDIT_AT"] = "{a.sync_audit}"\n' + SAMPLER_ENV_NEW)
        shutil.copytree(Path(__file__).parent / "sampler", out / "src/arena/sampler", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    if a.gdn_fp32:
        kernels = server["serving_kernels"]
        if kernels.get("mamba_ssm_cache_dtype") != "bfloat16" or kernels.get("kv_cache_4bit") != "int4" or not server.get("mtp_tokens"):
            raise SystemExit("--gdn-fp32 expects the MTP int4 stack with bf16 GDN state")
        kernels["mamba_ssm_cache_dtype"] = None
        patch_gdn_fp32(out / "src/arc_infra/serving_kernels_patch.py")
    study["probe_retry"] = "transport errors: 3 retries, 1/2/4 s"
    if a.trim_budget:
        study["trim"] = {**(study.get("trim") or {}), "budget_tokens": a.trim_budget}
    if a.tick:
        study["cache_experiment"]["prefix_cache"]["admission_tick"] = float(a.tick)
    if a.pyspy:
        study["pyspy_at"] = [float(x) for x in a.pyspy.split(",")]
    if a.sampler:
        study["sampler_at"] = [float(x) for x in a.sampler.split(",")]
    if "TICK4" in parts:
        # Exact throughput: admission tick 2 -> 4 s, half the mixed (eager) steps; scheduling only.
        study["cache_experiment"]["prefix_cache"]["admission_tick"] = 4.0
    if "PROF" in parts:
        # Steady-state step profile of the v8.2 serving: base unchanged, torch profiler armed (no cost
        # until /start_profile), captures at 40, 60 and 75 min of play; telemetry as every run.
        for trial in study["trials"]:
            trial.setdefault("server_env", {})["ARENA_TORCH_PROFILER"] = "1"
        study["profile_at"] = study.get("profile_at") or ([2400, 3600, 4500] if a.wall > 5000 else [a.wall // 2])  # 60-min probes: one capture at 30 min
    if a.arm == "DKf":
        # A v2's dense and cache together: BF16 dense projections + FP8 KV at Kf's gate; the served pool
        # (smaller with BF16 dense) caps the 27 focus streams through the gate.
        server["dense_kernels"]["int8_weights"] = []
    streams = a.streams or streams
    if streams > server["max_num_seqs"]:  # the engine must admit every stream (pruned pools hold 28+)
        server["max_num_seqs"] = streams
    if "DG" in parts and "CG" not in parts:
        # Decode graphs for every decode step (16 Oct): a decode step carries max_num_seqs x (1 + MTP) tokens; past
        # the 104-token graph it ran eager (v85cb56 at 56 x 2: 537 tok/s vs DKb 1,514; subprobe 40 x 4: ~590).
        need = server["max_num_seqs"] * (1 + server["mtp_tokens"])
        sizes = server["cudagraph_capture_sizes"]
        extra_sizes = list(range(112, need + 16, 16)) if need > sizes[-1] else []
        if need > 256:
            raise SystemExit(f"decode graphs: {need} tokens over the 256 bound")
        server["cudagraph_capture_sizes"] = sorted(set(sizes) | set(extra_sizes))
        server["max_cudagraph_capture_size"] = server["cudagraph_capture_sizes"][-1]
    if a.graph_sizes:
        sizes = sorted({int(x) for x in a.graph_sizes.split(",")})
        if "DG" in parts:
            # 24 Oct: with MTP-k a decode step is n x (1 + k) tokens; FULL decode graphs exist only at sizes that are
            # multiples of 1 + k, so the MTP1-era list (all even) sent MTP2 decode to PIECEWISE (P32 trace).
            need = server["max_num_seqs"] * (1 + server["mtp_tokens"])
            if sizes[-1] < need:
                raise SystemExit(f"--graph-sizes with DG must reach the decode bound {need}")
            server["max_cudagraph_capture_size"] = sizes[-1]
        elif sizes[-1] != server["max_cudagraph_capture_size"]:
            raise SystemExit(f"--graph-sizes must end at the capture bound {server['max_cudagraph_capture_size']}")
        server["cudagraph_capture_sizes"] = sizes
    if True:
        # History blocks (arena_variant, hero_v2) name the base's int3-t37 overlay, and the Nebius launcher fetches every
        # {"dataset": ...} it finds (17 Oct: 39.6 GiB unused per AV2C/T-arm VM); the served overlay stays in server and
        # model_datasets. Keep the record, drop the key.
        def scrub(node):
            if isinstance(node, dict):
                if isinstance(node.get("dataset"), str):
                    node["dataset_record"] = node.pop("dataset")
                for child in node.values():
                    scrub(child)
            elif isinstance(node, list):
                for child in node:
                    scrub(child)
        scrub(study.get("arena_variant")); scrub(study.get("hero_v2"))
    if a.watchdog:
        study["engine_watchdog"] = {"test_kill_seconds": a.watchdog_test_kill or None}
    if a.ple_bound:
        study["ple_prefill_bound"] = {"limit_mib": 128, "source": "scripts/dfr_exact/ple_prefill_bound.py"}
    study.update(name=out.name, status="prepared-not-launched", active_limit=streams,
                 generation_slots=server["max_num_seqs"], wall_seconds=a.wall, token_stop=a.stop_tokens)
    if study.get("focus_oversubscription"):
        # Streams per arm from the served KV pool: focus = min(target, capacity / the validated gate blocks per stream),
        # refused below min_full_windows (v7.1's 8 for the smaller pools of R and K).
        study["focus_oversubscription"]["focus_slots"] = streams
        if a.arm in ("R", "AR", "K", "Kf", "DKf", "AV2C"):
            study["focus_oversubscription"].update(min_full_windows=8, min_slots=8)
        if a.arm in ("Kf", "DKf"):
            # The FP8 pool's blocks hold 1,616 tokens (int4: 3,136): rescale the gate density to that layout
            # (floor(0.96 x 2,051) / 27, the v8.4 rule) so the 27 streams fit without evicting waiting prefixes.
            study["focus_oversubscription"]["gate_blocks_per_stream"] = 72.888
        if a.gate_blocks:
            study["focus_oversubscription"]["gate_blocks_per_stream"] = a.gate_blocks
        if a.min_full_windows:
            study["focus_oversubscription"]["min_full_windows"] = a.min_full_windows
    if a.arm == "O":
        study["priority"].update(equal=False, park_tokens=0, level_cap_tokens=300000, tail_fade=False)
    else:
        study["priority"].update(equal=True, park_tokens=0, level_cap_tokens=0, tail_fade=False)
    study["competition"].update(preview_instances=a.instances, preview_seconds=a.wall)
    if study.get("b200"):
        study["b200"]["stop"]["wall_seconds"] = a.wall
    study["serving_ablation"] = {
        "arm": a.arm, "extra": list(extras), "cpu_cache_gib": a.cpu_gib, "offload_decode": a.offload_decode, "offload_eagle_tail": a.offload_eagle_tail, "oea_k0": a.oea, "ple_fp8": a.ple_fp8, "cache_budget": a.cache_budget, "max_batched": a.max_batched or None, "tick": a.tick or None, "trim_budget": a.trim_budget or None, "graph_sizes": a.graph_sizes or None, "gdn_fp32": a.gdn_fp32 or None, "ple_sync_fix": a.ple_sync_fix or None, "gdn_mask_fix": a.gdn_mask_fix or None, "sync_audit": a.sync_audit or None, "toggle_ab": a.toggle_ab or None, "topk": a.topk or None, "prune": a.prune or None, "server_env": a.server_env or None, "mtp_tokens": server.get("mtp_tokens"), "base_bundle": v82.name, "stop_tokens": a.stop_tokens, "instances": a.instances,
        "streams": streams, "scheduler": "old priority (equal off, level cap 300K)" if a.arm == "O" else "equal (priority.equal): fewest generated tokens first, hand-over at drains, "
        "compactions and finished games; an instance finishes at stop_tokens; the trial ends when all have finished",
        "server_change": {"base": "none", "R": "all 512 experts, stock NVFP4, Marlin (rl-pool reference profile)",
                          "P": "t37 prune, stock NVFP4 kept experts, Marlin (no int3 overlay)",
                          "K": "kv_cache_8bit fp8 in place of int4, indexer keys BF16",
                          "Kb": "indexer keys BF16 (int4 KV kept)",
                          "Kf": "kv_cache_8bit fp8 in place of int4 (FP8 indexer keys kept), 27 streams, gate 72.888",
                          "D": "BF16 dense projections (dense_kernels.int8_weights empty)",
                          "DKf": "BF16 dense projections + FP8 KV (Kf's 27 streams and gate 72.888; A v2's dense and cache)",
                          "DKb": "BF16 dense projections + BF16 indexer keys (int4 KV kept)",
                          "T25": "25% pruned (384 kept, excl5 mask), all-expert excl5 INT3 overlay (unmasked fit)",
                          "T50": "50% pruned (256 kept, excl5 mask), all-expert excl5 INT3 overlay (unmasked fit)",
                          "T62": "62.5% pruned (192 kept, excl5 mask), all-expert excl5 INT3 overlay (unmasked fit)",
                          "T12": "12.5% pruned (448 kept, excl5 mask), all-expert excl5 INT3 overlay (unmasked fit)",
                          "T37A": "base's t37 mask (37.5% pruned) on the all-expert excl5 INT3 overlay (unmasked fit)",
                          "DKbT12": "BF16 dense + BF16 indexer keys + 12.5% pruned (448 kept, excl5 mask, all-expert excl5 INT3)",
                          "PROF": "none (base serving with the torch profiler armed; captures at 40/60/75 min)",
                          "CG": "CUDA-graph capture sizes to 4,096 tokens (piecewise graphs for steps with a prefill)",
                          "TICK4": "admission tick 4 s (base 2 s)",
                          "CGT4": "CUDA-graph capture sizes to 4,096 + admission tick 4 s",
                          "AR": "Intel AutoRound W4A16 checkpoint (all 512 experts W4A16, rest BF16), R's profile",
                          "AV2C": "A v2's serving configuration on vLLM: AutoRound W4A16 (AR) + BF16 dense + BF16 indexer "
                                  "keys + FP8 KV + 3 MTP draft steps, 10 streams, gate density from --gate-blocks",
                          "O": "none (scheduler arm: the long run's priority scheduler, level cap 300K)"}[a.arm]}
    if a.steady_replay:
        add_steady_replay(out, study, a)
    if a.gpu_script:
        add_gpu_script(out, study, a)
    ov = out / "src/arena/expert_overlay.py"
    if ov.exists() and "partial expert overlay" not in ov.read_text():
        his.patch(ov, OVERLAY_OLD, OVERLAY_NEW)
    check_bundle(out, study, json.loads((v82 / "configs/comparison.json").read_text()))
    (out / "configs/comparison.json").write_text(json.dumps(study, indent=2) + "\n")
    identity = json.loads((v82 / "MANIFEST.json").read_text())
    identity["files"] = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
                         if p.is_file() and p.suffix != ".ipynb" and "__pycache__" not in p.parts}
    identity.update(dependencies_prepared=True, study=out.name)
    (out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps({"bundle": str(out), "arm": a.arm, "streams": streams, "files": len(identity["files"]),
                      "manifest_sha256": sha(out / "MANIFEST.json")}))


if __name__ == "__main__":
    main()
