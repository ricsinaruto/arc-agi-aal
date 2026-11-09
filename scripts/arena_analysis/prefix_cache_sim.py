"""CPU replay of the pinned vLLM scheduler + KV cache manager on append-only multi-game chats (29 September).

Evidence for docs/validation-20260927/prefix-cache-v2.md. It drives the *pinned* ``AsyncScheduler`` (the extracted
runtime's own ``vllm/v1/core``, optionally patched by ``arena.cache_reclamation`` and ``arena.prefix_cache_patch``)
with a fake model runner: each game sends request n+1 = prompt_n + reply_n + new tokens, ``--gap`` steps after
request n ends, with identical token ids (no template round trip), so every shortfall is the server's.
The KV layout is the hero's shape (one attention group, the QSA ring's 4-token CircularBufferSpec group, three GDN
groups and the PLE conv group, all Mamba groups in "align" mode) and ``cache_config.block_size`` is set to the
ring's 4 as ``EngineCore._initialize_kv_caches`` does on the served layout (``--cc-block 4``).

A worker-semantics tracker follows one Mamba group: a block holds the state after the last token computed while
it was the running column, and a boundary crossing copies the running state forward (the align precopy). At each
request's admission it checks that the hit block holds the state at exactly the hit length; ``wrong_state``
counts hits that resume from a state short of their prefix (``skipped_tokens``). At the end it checks the block
pool: no leaked references, no duplicate free-queue entries.

Setup (not in the project environment):
    # the extracted pinned runtime (runs/serving-research or a bundle's layer) and a CPU torch venv with vLLM's
    # pure-Python dependencies (regex, msgspec, cbor2, pydantic, transformers, pyzmq, uvloop, ...)
    python scripts/arena_analysis/prefix_cache_sim.py overlay --runtime <site-packages> --out /tmp/ov_all \
        reclaim align keep demote
    PYTHONPATH=/tmp/ov_all:<deps> <venv>/bin/python scripts/arena_analysis/prefix_cache_sim.py run \
        --block 64 --batched 164 --cc-block 4 --games 24 --requests 12 --blocks 700 --gap 60 --system 166 \
        --new-abs 20 140 --reply-abs 30 160 --demote-min 256
"""
import argparse
import random
import shutil
import sys
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def overlay(args):
    """Symlink farm of the pinned vllm package with ``v1/core`` copied, then the arena patches applied."""
    src = Path(args.runtime)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    def farm(folder, target):
        target.mkdir(parents=True, exist_ok=True)
        for child in folder.iterdir():
            rel = child.relative_to(src).as_posix()
            if rel == "vllm/v1/core":
                shutil.copytree(child, target / child.name)
            elif "vllm/v1/core".startswith(rel + "/"):
                farm(child, target / child.name)
            else:
                (target / child.name).symlink_to(child)

    farm(src / "vllm", out / "vllm")
    sys.path.insert(0, str(ROOT / "src"))
    from arena import cache_reclamation, prefix_cache_patch

    if "reclaim" in args.patches:
        cache_reclamation.apply(out, out)
    options = {"mamba_align_fix": "align" in args.patches, "mamba_keep_state": "keep" in args.patches,
               "mamba_demote_superseded": "demote" in args.patches}
    if "tail" in args.patches:
        options["request_end_tail"] = True
    for name, key in (("keepboundary", "tail_keep_boundary"), ("keepalive", "tail_keepalive")):
        if name in args.patches:
            options[key] = True
    if any(options.values()):
        print(prefix_cache_patch.apply(out, out, options))


def make_scheduler(a):
    import torch
    from vllm.config import CacheConfig, ObservabilityConfig, ParallelConfig, SchedulerConfig
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec,
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )

    cache_config = CacheConfig(block_size=a.block, cache_dtype="auto", enable_prefix_caching=True,
                               mamba_cache_mode="align", prefix_match_unit=a.hash_unit)
    cache_config.num_gpu_blocks = a.blocks
    if a.cc_block:
        cache_config.block_size = a.cc_block  # EngineCore: min(group block sizes) = the ring's block
    sched = SchedulerConfig(max_num_seqs=a.max_seqs, max_num_batched_tokens=a.batched, max_model_len=262144,
                            enable_chunked_prefill=True, async_scheduling=not a.sync, is_encoder_decoder=False)
    model_config = SimpleNamespace(max_model_len=262144, is_encoder_decoder=False, is_diffusion=False,
                                   enable_return_routed_experts=False, return_sampling_mask=False)
    vc = SimpleNamespace(
        cache_config=cache_config, scheduler_config=sched, model_config=model_config,
        ec_manager_config=SimpleNamespace(get_encoder_cache_manager_obj=lambda: None), ec_transfer_config=None,
        is_encoder_only=False, kv_events_config=None, kv_transfer_config=None, lora_config=None,
        max_concurrent_batches=1 if a.sync else 2, max_in_flight_tokens=a.batched * (1 if a.sync else 2),
        num_lookahead_tokens=0, num_speculative_tokens=0, observability_config=ObservabilityConfig(),
        parallel_config=ParallelConfig(), speculative_config=None, use_v2_model_runner=False)
    groups = [KVCacheGroupSpec(["attn"], FullAttentionSpec(block_size=a.block, num_kv_heads=1, head_size=64,
                                                           dtype=torch.bfloat16)),
              KVCacheGroupSpec(["ring"], CircularBufferSpec(block_size=4, num_kv_heads=1, head_size=64,
                                                            dtype=torch.bfloat16))]
    groups += [KVCacheGroupSpec([f"gdn{i}"], MambaSpec(block_size=a.block, shapes=((4, 32),), dtypes=(torch.float32,),
                                                       mamba_cache_mode="align")) for i in range(3)]
    groups.append(KVCacheGroupSpec(["ple"], MambaSpec(block_size=a.block, shapes=((2, 16),), dtypes=(torch.float32,),
                                                      mamba_cache_mode="align")))
    kc = KVCacheConfig(num_blocks=a.blocks, kv_cache_tensors=[], kv_cache_groups=groups)
    cls = Scheduler if a.sync else AsyncScheduler
    return cls(vllm_config=vc, kv_cache_config=kc,
               structured_output_manager=SimpleNamespace(should_advance=lambda *x, **k: False,
                                                         grammar_init=lambda *x, **k: None),
               block_size=a.block, hash_block_size=a.hash_unit or a.block,
               mm_registry=SimpleNamespace(supports_multimodal_inputs=lambda m: False), log_stats=True)


def run(a):
    if a.demote_min is not None:
        import vllm.v1.core.single_type_kv_cache_manager as manager_module

        if hasattr(manager_module.MambaManager, "_arc_demote_min_tokens"):
            manager_module.MambaManager._arc_demote_min_tokens = a.demote_min
    if a.tail_placement is not None:
        import vllm.v1.core.arc_tail_cache as tail_module

        tail_module.PLACEMENT = a.tail_placement
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

    s = make_scheduler(a)
    if a.step_seconds is not None and getattr(s.kv_cache_manager, "_arc_tail", None) is not None:
        s.kv_cache_manager._arc_tail.clock = lambda: step * a.step_seconds
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    unit = a.hash_unit or a.block  # --prefix-match-unit (hero-v2 precheck): hashes every `unit` tokens
    hasher = get_request_block_hasher(unit, hash_fn)
    hits = {}
    record = s.kv_cache_manager.record_prefix_cache_stats

    def recording(request, num_hits):
        hits[request.request_id] = num_hits
        return record(request, num_hits)

    s.kv_cache_manager.record_prefix_cache_stats = recording
    rng = random.Random(a.seed)
    B = a.block
    system = [rng.randrange(1, 30000) for _ in range(a.system)]
    games = []
    for g in range(a.games):
        plan = []
        for _ in range(a.requests):
            new = [rng.randrange(1, 30000) for _ in range(rng.randrange(*a.new_abs))]
            reply = [rng.randrange(1, 30000) for _ in range(rng.randrange(*a.reply_abs))]
            plan.append((new, reply))
        games.append({"id": g, "plan": plan, "n": 0, "tokens": list(system), "wait": g, "active": None, "log": []})

    def submit(game):
        new, reply = game["plan"][game["n"]]
        prompt = game["tokens"] + new
        req = Request(f"g{game['id']}-r{game['n']}", prompt, SamplingParams(max_tokens=len(reply), ignore_eos=True),
                      None, block_hasher=hasher)
        game["active"] = {"req": req, "reply": reply, "emitted": 0, "P": len(prompt)}
        s.add_request(req)

    groups = s.kv_cache_config.kv_cache_groups
    managers = s.kv_cache_manager.coordinator.single_type_managers
    mamba_gids = [i for i, g in enumerate(groups) if isinstance(g.kv_cache_spec, MambaSpec)]
    ring_gids = [i for i, m in enumerate(managers) if type(m).__name__ == "CircularBufferManager"]
    attn_gids = [i for i, m in enumerate(managers) if type(m).__name__ == "FullAttentionManager"]
    ring = groups[ring_gids[0]].kv_cache_spec.block_size if ring_gids else 0
    # Worker semantics per block id: attention/ring slots hold (sequence, position) written there; a Mamba block
    # holds (sequence, tokens) = the state after that many tokens of that sequence.
    slots, state = {}, {}
    running_col, checks = {}, Counter()
    seqs = {}  # request id -> its full token sequence (prompt + planned reply)
    queue, step = deque(), 0

    def same_prefix(seq, rid, n):
        return seqs[seq][:n] == seqs[rid][:n]

    def check_hit(rid, hit):
        """The hit blocks must hold exactly the new request's first ``hit`` tokens (all groups)."""
        bad = []
        for gid in attn_gids:
            table = managers[gid].req_to_blocks[rid]
            for pos in range(hit):
                got = slots.get(table[pos // B].block_id, {}).get(pos % B)
                if got is None or got[1] != pos or seqs[got[0]][pos] != seqs[rid][pos]:
                    bad.append(("attention", gid, pos))
                    break
        for gid in ring_gids:
            table = managers[gid].req_to_blocks[rid]
            for pos in range(hit - hit % ring, hit):  # the open compression group
                got = slots.get(table[0].block_id, {}).get(pos % ring)
                if got is None or got[1] != pos or seqs[got[0]][pos] != seqs[rid][pos]:
                    bad.append(("ring", gid, pos))
                    break
        for gid in mamba_gids:
            table = managers[gid].req_to_blocks[rid]
            column = hit // B if hit % B else hit // B - 1
            got = state.get(table[column].block_id)
            if got is None or got[1] != hit or not same_prefix(got[0], rid, hit):
                bad.append(("mamba", gid, got and got[1]))
                checks["skipped_tokens"] += hit - (got[1] if got else 0)
        return bad

    while True:
        for game in games:
            if game["active"] is None and game["n"] < a.requests:
                if game["wait"] <= 0:
                    submit(game)
                    act = game["active"]
                    seqs[act["req"].request_id] = list(act["req"].prompt_token_ids) + act["reply"]
                else:
                    game["wait"] -= 1
        if all(g["active"] is None and g["n"] >= a.requests for g in games) and not queue:
            break
        if s.has_requests():
            so = s.schedule()
            # Worker order: zero new blocks, run the queued copies, then the forward.
            for block_id in so.new_block_ids_to_zero or ():
                slots.pop(block_id, None)
                state.pop(block_id, None)
            for copy in getattr(so, "kv_cache_block_copies", None) or ():
                checks["cow_copies"] += 1
                src, dst = copy.src_block_id, copy.dst_block_id
                if src in slots:
                    slots[dst] = dict(slots[src])
                if src in state:
                    state[dst] = state[src]
            for rid, count in so.num_scheduled_tokens.items():
                req = s.requests.get(rid)
                if req is None or rid not in managers[attn_gids[0]].req_to_blocks:
                    continue
                after = req.num_computed_tokens
                before = after - count
                mamba_blocks = managers[mamba_gids[0]].req_to_blocks.get(rid, ()) if mamba_gids else ()
                if rid in running_col and running_col[rid] >= len(mamba_blocks):
                    # preempted and re-admitted (its blocks were freed): track it as a new admission
                    checks["readmitted_after_preemption"] += 1
                    del running_col[rid]
                if rid not in running_col:
                    hit = hits.get(rid, 0)
                    running_col[rid] = (hit // B if hit % B else hit // B - 1) if hit else -1
                    if hit:
                        checks["hits"] += 1
                        if hit % B:
                            checks["partial_hits"] += 1
                            checks["partial_hit_tokens"] += hit % B
                        bad = check_hit(rid, hit)
                        if bad:
                            checks["wrong_state"] += 1
                            checks["wrong_" + bad[0][0]] += 1
                for gid in attn_gids:
                    table = managers[gid].req_to_blocks[rid]
                    for pos in range(before, after):
                        slots.setdefault(table[pos // B].block_id, {})[pos % B] = (rid, pos)
                for gid in ring_gids:
                    table = managers[gid].req_to_blocks[rid]
                    for pos in range(max(before, after - ring), after):
                        slots.setdefault(table[0].block_id, {})[pos % ring] = (rid, pos)
                dst = (after + B - 1) // B - 1
                for gid in mamba_gids:
                    blocks = managers[gid].req_to_blocks[rid]
                    src = running_col[rid]
                    # The align precopy carries the running state into the new column; the step then advances it.
                    state[blocks[dst].block_id] = (rid, after)
                running_col[rid] = dst
            sampled = {}
            for rid in so.num_scheduled_tokens:
                act = games[int(rid.split("-")[0][1:])]["active"]
                if act is None or act["req"].request_id != rid:
                    sampled[rid] = []
                    continue
                req = act["req"]
                if req.num_computed_tokens >= req.num_prompt_tokens and act["emitted"] < len(act["reply"]):
                    sampled[rid] = [act["reply"][act["emitted"]]]
                    act["emitted"] += 1
                else:
                    sampled[rid] = []
            queue.append((so, sampled))
        step += 1
        if len(queue) >= (1 if a.sync else 2) or (queue and not s.has_requests()):
            so, sampled = queue.popleft()
            ids = list(so.num_scheduled_tokens)
            s.update_from_output(so, ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                                                       sampled_token_ids=[sampled[r] for r in ids]))
            for game in games:
                act = game["active"]
                if act is not None and act["req"].is_finished():
                    game["log"].append({"P": act["P"], "G": len(act["reply"]), "H": hits.get(act["req"].request_id)})
                    game["tokens"] = list(act["req"].prompt_token_ids) + act["reply"]
                    game["n"] += 1
                    game["active"] = None
                    game["wait"] = a.idle_gap if game["id"] < a.idle_games else a.gap
        if step > 500000:
            raise SystemExit("no progress")
    pool = s.kv_cache_manager.block_pool
    free_ids = [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]
    checks.update(leaked_blocks=sum(1 for b in pool.blocks if not b.is_null and b.ref_cnt != 0),
                  free_queue_dups=len(free_ids) - len(set(free_ids)), free_blocks=pool.get_num_free_blocks(),
                  prefix_hit_tokens=sum(v for v in hits.values() if v))
    labels, short, short_unit, above = Counter(), 0, 0, 0
    # Per group of games (idle = the long-tool games, others = the churning ones): follow-up requests, tokens of the
    # previous request's sequence that were not served, requests that lost at least one block of it, and requests
    # served from the shared system prefix or less.
    groups_report = {name: Counter() for name in ("idle", "others")}
    for game in games:
        log = game["log"]
        for n in range(1, len(log)):
            p, r = log[n - 1], log[n]
            ideal = min(p["P"] + p["G"], r["P"] - 1) // B * B
            short += max(0, ideal - r["H"])
            report = groups_report["idle" if game["id"] < a.idle_games else "others"]
            lost = max(0, min(p["P"] + p["G"], r["P"] - 1) - r["H"])
            report.update(requests=1, lost_tokens=lost, lost_a_block=lost >= B, from_system_prefix=r["H"] <= a.system)
            # With a fine unit the previous prompt's last unit boundary is cached too (its partial tail).
            ideal_unit = max(ideal, min(p["P"], r["P"] - 1) // unit * unit)
            short_unit += max(0, ideal_unit - r["H"])
            above += max(0, r["H"] - ideal)
            end = p["P"] + p["G"]
            labels["request_end" if r["H"] in (end - 1, end) and r["H"] > ideal
                   else "ideal" if r["H"] == ideal else "prev_prompt_end" if r["H"] == p["P"] // B * B
                   else "earlier_request" if n >= 2 and r["H"] in (log[n - 2]["P"] // B * B,
                                                                  (log[n - 2]["P"] + log[n - 2]["G"]) // B * B)
                   else "other"] += 1
    tail = getattr(s.kv_cache_manager, "_arc_tail", None)
    if tail is not None:
        checks.update({"tail_" + k: v for k, v in tail.stats.items()})
    if a.idle_games:
        print({name: dict(c) for name, c in groups_report.items()})
    print({"hit_vs_ideal": dict(labels), "shortfall_tokens": short, "hash_unit": unit,
           "shortfall_vs_unit_ideal": short_unit, "hit_tokens_above_block_ideal": above, "steps": step,
           **dict(checks)})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    o = sub.add_parser("overlay")
    o.add_argument("--runtime", required=True, help="site-packages directory holding the pinned vllm package")
    o.add_argument("--out", required=True)
    o.add_argument("patches", nargs="*", choices=("reclaim", "align", "keep", "demote", "tail", "keepboundary", "keepalive"))
    r = sub.add_parser("run")
    r.add_argument("--block", type=int, default=64)
    r.add_argument("--blocks", type=int, default=700)
    r.add_argument("--batched", type=int, default=164)
    r.add_argument("--max-seqs", type=int, default=25)
    r.add_argument("--cc-block", type=int, default=4)
    r.add_argument("--games", type=int, default=24)
    r.add_argument("--requests", type=int, default=12)
    r.add_argument("--gap", type=int, default=60)
    r.add_argument("--system", type=int, default=166)
    r.add_argument("--new-abs", type=int, nargs=2, default=(20, 140))
    r.add_argument("--reply-abs", type=int, nargs=2, default=(30, 160))
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--idle-games", type=int, default=0, help="the first N games wait --idle-gap steps (a long tool)")
    r.add_argument("--idle-gap", type=int, default=0)
    r.add_argument("--step-seconds", type=float, default=None, help="tail index clock: seconds per step")
    r.add_argument("--sync", action="store_true")
    r.add_argument("--demote-min", type=int, default=None)
    r.add_argument("--tail-placement", choices=("front", "back", "swap"), default=None,
                   help="request_end_tail: where a new entry's blocks enter the free queue (default: the runtime's)")
    r.add_argument("--hash-unit", type=int, default=None,
                   help="--prefix-match-unit: hash granularity below the block (partial-tail hits); default: the block")
    args = ap.parse_args()
    overlay(args) if args.command == "overlay" else run(args)


if __name__ == "__main__":
    main()
