"""Prefix-cache v2 estimates from recorded c110 runs (29 September, CPU only; prefix-cache-v2.md).

Input: the per-request JSON of ``prefill_attribution.py`` (exact single-request hits, prompts, replies, phases).

``hits``: where exact single-request hits land relative to the previous requests' Mamba block boundaries
(phase 2, gap < 5 s): the ideal (previous prompt + reply), the previous prompt end, a state inside the previous
reply (a decode crossing), or older boundaries; and how much of the shortfall the align fix recovers (a hit is then
at least the previous request's prompt-end boundary, whose state the fixed split writes).

``replay``: a first-order page-level LRU replay of the recorded requests (timing as recorded, not re-simulated).
Units are pool blocks (hero: 860 of ~21 MB; a block holds 1,600 tokens of attention KV for all attention layers,
or one Mamba group's state, so one reusable state costs 4 blocks, as much as 6,400 tokens of attention KV).
- admission: the game's prefix is reusable iff all its attention blocks and its newest useful state are resident,
  else only the shared system prefix hits;
- a request allocates attention blocks, one running state per Mamba group + the ring, and 4 blocks per state it
  writes: prefill chunk ends (3,200-token aligned chunks with the align fix; 4,096-token budget chunks on the pinned
  split) and decode crossings;
- all written states but the newest useful one are garbage, freed while the request runs into the LRU tail (pinned)
  or the head (``demote``); the newest useful state is freed at the prefill end / last crossing (pinned) or at the
  request end (``keep``); attention blocks at the request end;
- useful state: with the align fix the newest of {prompt-end state, decode crossings}; on the pinned split only
  decode crossings (without one the request leaves only the state it resumed from, freed at its start);
- ``protect``: the arena marks the first K games (scheduler-aware eviction by priority hints): their blocks are
  evicted only when no other game's evictable block is left;
- ``cpu``: every evicted prefix is restorable from host memory (capacity not binding; a 1,600-token block is
  ~21 MB across the attention layers, so 110 games of ~20K tokens hold ~40 GB).

    uv run python scripts/arena_analysis/prefix_cache_replay.py hits <attribution.json>...
    uv run python scripts/arena_analysis/prefix_cache_replay.py replay <attribution.json>... [--out result.json]
"""
import argparse
import json
import math
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path


def by_game(requests):
    games = defaultdict(list)
    for r in requests:
        if r["purpose"] == "gameplay":
            games[r["game"]].append(r)
    return games


def hits(d):
    """Exact single hits of phase-2 continuations (gap < 5 s) against the previous requests' boundaries."""
    B, shared = d["summary"]["block"], d["summary"]["shared_prefix"]
    labels, reply = Counter(), Counter()
    short = recovered = 0
    for gp in by_game(d["requests"]).values():
        for n, r in enumerate(gp):
            if (n < 1 or r.get("cat") != "continuation" or r.get("H") is None or r["phase"] != "phase2"
                    or (r.get("gap") or 0) >= 5 or r["H"] <= shared):
                continue
            p, H, P = gp[n - 1], r["H"], r["P"]

            def floor(x, P=P):
                return min(x, P - 1) // B * B
            A, D = floor(p["P"]), floor(p["P"] + p["G"])
            if H == D:
                label = "ideal (previous prompt + reply)"
            elif H == A:
                label = "previous prompt end"
            elif A < H < D:
                label = "inside the previous reply"
            elif n >= 2 and H in (floor(gp[n - 2]["P"]), floor(gp[n - 2]["P"] + gp[n - 2]["G"])):
                label = "a boundary of the request before"
            else:
                label = "older or other"
            labels[label] += 1
            if D > A:
                reply["reply crossed a boundary"] += 1
                reply["hit inside or at the end of that reply"] += A < H <= D
            else:
                reply["reply crossed no boundary"] += 1
                reply["hit at the previous prompt end"] += H == A
            ideal = r["ideal_hit"]
            short += max(0, ideal - H)
            recovered += max(0, min(ideal, max(H, A)) - H)
    lag = sum(r["est"].get("lag_reply", 0) + r["est"].get("lag_old", 0) for r in d["requests"])
    return {"singles": sum(labels.values()), "hit_vs_boundaries": dict(labels.most_common()), "reply": dict(reply),
            "single_shortfall_tokens": short, "align_fix_recovers_tokens": recovered,
            "align_fix_recovered_share": round(recovered / max(1, short), 3),
            "attributed_state_lag_tokens": round(lag),
            "projected_align_fix_recovery_tokens": round(lag * recovered / max(1, short))}


def wrong_state_hits(d):
    """Exact single hits that no correct state explains on the pinned split (chunk-end states under a later hash).

    Correct reusable states there: the shared system prefix and decode crossings (k*B inside an earlier reply of the
    same game since its last prompt rewrite)."""
    B, shared = d["summary"]["block"], d["summary"]["shared_prefix"]
    games = defaultdict(list)
    for r in d["requests"]:
        games[r["game"]].append(r)
    c = Counter()
    for rs in games.values():
        crossings = set()
        for r in rs:
            if r["cat"] in ("first", "post_compaction"):
                crossings = set()
            H = r.get("H")
            if H is not None and r["cat"] == "continuation":
                c[f"{r['phase']} hits"] += 1
                c[f"{r['phase']} " + ("shared prefix only" if H <= shared else "decode-crossing state (correct)"
                                      if H in crossings else "chunk-end state (wrong)")] += 1
            if r["purpose"] == "gameplay":
                crossings |= set(range((r["P"] // B + 1) * B, r["P"] + r["G"], B))
    return dict(sorted(c.items()))


def replay(reqs, pages_total, B, shared, align_fix, keep, demote, cpu, protect=0, chunk=4096, groups=4):
    step = chunk // B * B if align_fix else chunk
    events = sorted([(r["start"], 1, j) for j, r in enumerate(reqs)] + [(r["end"], 3, j) for j, r in enumerate(reqs)])
    lru = OrderedDict()  # key -> blocks; first = evicted first
    free, pinned, game_state = pages_total, {}, {}
    lost, stats, seq = defaultdict(float), defaultdict(float), [0]
    first_seen = {}

    def key(kind, g):
        seq[0] += 1
        return (kind, g, seq[0])

    def alloc(n):
        nonlocal free
        take = min(n, free)
        free -= take
        n -= take
        while n > 0 and lru:
            victim = next((k for k in lru if first_seen.get(k[1], 0) >= protect), None) if protect else None
            p = lru.pop(victim) if victim is not None else lru.popitem(last=False)[1]
            use = min(n, p)
            n -= use
            free += p - use
        stats["overcommit"] += n

    def release(k, pages, head=False):
        lru[k] = pages
        if head:
            lru.move_to_end(k, last=False)

    for _, kind, j in events:
        r = reqs[j]
        g, P, G = r["game"], r["P"], r["G"]
        if kind == 1:
            first_seen.setdefault(g, len(first_seen))
            want = r.get("ideal_hit", 0) if r["cat"] == "continuation" else min(shared, P)
            gs = game_state.get(g)
            hit, reuse = (shared if r["cat"] != "first" else min(shared, P)), 0
            if gs and r["cat"] == "continuation" and gs["fa"] in lru and gs["st"] in lru:
                hit = max(shared, min(want, gs["length"]))
                reuse = lru.pop(gs["fa"]) + lru.pop(gs["st"])
            elif gs and r["cat"] == "continuation" and cpu:
                hit = max(shared, min(want, gs["length"]))
                stats["restored_tokens"] += hit - shared
                reuse = sum(lru.pop(k) for k in (gs["fa"], gs["st"]) if k in lru)
            elif gs:
                free += sum(lru.pop(k) for k in (gs["fa"], gs["st"]) if k in lru)
            if r["cat"] == "continuation":
                lost[r["phase"]] += max(0, want - hit)
                stats["continuations"] += 1
                stats["game_prefix_hits"] += hit > shared
            T = P + G
            A = P // B * B
            prefill_states = (max(0, math.ceil((A - hit) / step)) if A > hit else 0) if align_fix \
                else max(0, math.ceil((P - hit) / step) - 1)
            crossings = max(0, (T - 1) // B - P // B)
            need = math.ceil(T / B) + groups + 1
            pinned[j] = {"prefill_states": prefill_states, "crossings": crossings, "hit": hit}
            alloc(max(0, need - reuse) + groups * (prefill_states + crossings))
            useful = crossings > 0 or (align_fix and prefill_states > 0)
            for _ in range(prefill_states + crossings - int(useful)):
                release(key("garbage", g), groups, head=demote)
            if useful and not keep:
                pinned[j]["st"] = key("st", g)
                release(pinned[j]["st"], groups)
        else:
            info = pinned.pop(j)
            T = P + G
            useful = info["crossings"] > 0 or (align_fix and info["prefill_states"] > 0)
            if useful:
                st_key = info.get("st") or key("st", g)
                if keep:
                    release(st_key, groups)
                free += groups + 1
            else:
                st_key = key("st", g)
                release(st_key, groups, head=not keep)
                free += 1
            fa_key = key("fa", g)
            lru[fa_key] = math.ceil(T / B)
            game_state[g] = {"fa": fa_key, "st": st_key, "length": T if useful else info["hit"]}
    return {"lost_prefix_tokens": {k: round(v) for k, v in lost.items()}, **{k: round(v) for k, v in stats.items()}}


ARMS = {"pinned": dict(align_fix=False, keep=False, demote=False, cpu=False),
        "align": dict(align_fix=True, keep=False, demote=False, cpu=False),
        "align+keep": dict(align_fix=True, keep=True, demote=False, cpu=False),
        "align+keep+demote": dict(align_fix=True, keep=True, demote=True, cpu=False),
        "align+keep+cpu tier": dict(align_fix=True, keep=True, demote=False, cpu=True),
        "pinned+protect first 20": dict(align_fix=False, keep=False, demote=False, cpu=False, protect=20),
        "pinned+protect first 40": dict(align_fix=False, keep=False, demote=False, cpu=False, protect=40),
        "align+keep+demote+protect 20": dict(align_fix=True, keep=True, demote=True, cpu=False, protect=20)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("hits", "wrong", "replay"))
    ap.add_argument("attribution", nargs="+")
    ap.add_argument("--pages", type=int, default=None, help="pool blocks (default: 860 hero, 875 BF16 KV)")
    ap.add_argument("--out")
    args = ap.parse_args()
    result = {}
    for path in args.attribution:
        d = json.loads(Path(path).read_text())
        name = Path(path).stem
        if args.command == "hits":
            result[name] = hits(d)
        elif args.command == "wrong":
            result[name] = wrong_state_hits(d)
        else:
            B, shared = d["summary"]["block"], d["summary"]["shared_prefix"]
            pages = args.pages or (860 if B == 1600 else 875)
            attributed = defaultdict(float)
            for r in d["requests"]:
                for k in ("d_evicted", "lag_reply", "lag_old"):
                    attributed[f"{r['phase']}.{k}"] += r["est"].get(k, 0)
            result[name] = {"pool_blocks": pages, "attributed": {k: round(v) for k, v in attributed.items()},
                            "arms": {arm: replay(d["requests"], pages, B, shared, **kw) for arm, kw in ARMS.items()}}
        print(name, json.dumps(result[name], indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
