"""Prefix-cache v3 estimates from recorded hero-v2 runs (29 September, CPU only; prefix-cache-v3.md).

Input: the per-request JSON of ``prefill_attribution.py`` (prompts, replies, the ideal hit, phases, timing) and the
trial's ``admission.jsonl`` (park / admit events). The model is prefix cache v2's first-order page-level LRU replay
(``prefix_cache_replay.replay``, align + keep + demote: the served hero-v2 cache) with timing as recorded, extended by:

``tail`` (``request_end_tail``): a request's end leaves an entry of 1 + ``groups`` pages (the ring and the running
Mamba states; the partial attention block is already counted with the attention pages) at the LRU tail, and with
placement "swap" the kept boundary state goes to the head instead ("front": the entry goes to the head; "back": both
at the tail). A continuation whose game's attention pages and entry are resident hits at the previous request's end
(``min(prevEnd, P - 1)``) instead of its last whole block; otherwise it falls back to the v2 rule.

``ping`` (arena keep-alive, ``cache_experiment.keep_alive``): every ``tick`` seconds the first ``games`` parked games
of an order are pinged when their prefix was last touched ``every`` seconds ago or more: a resident prefix moves to the
LRU tail (a max_tokens=1 request on the game's pending prompt); an evicted one is re-prefilled (``ping_prefill``, the
cost). Orders: ``oracle`` (the recorded next re-admission time, an upper bound) and ``readmissions`` (hero-v2's backlog
key from the log: fewest admissions, most levels, roster order).

Gains are uncached prompt tokens saved against the v2 arm, net of ping prefill, and generated tok/s at the in-run
prefill cost (hero-v2 2h fit: 0.081 ms per uncached token in phase 2, 0.092 in the opening; gain = s / (engine s - s)).

    uv run python scripts/arena_analysis/prefix_cache_v3_replay.py --attribution attr.json \
        --admission <trial>/admission.jsonl --pages 1335 --out result.json
"""
import argparse
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path

COST_S = {"opening": 0.092e-3, "phase2": 0.081e-3}


def load_parks(path):
    """game -> [(park wall s, next admit wall s or None, admits before the park, levels at the park, index)]."""
    events = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    events = [e for e in events if e.get("event") in ("park", "admit") and "reason" in e]
    admits = defaultdict(int)
    spans = defaultdict(list)
    open_park = {}
    order = {}
    for e in sorted(events, key=lambda e: e["wall_seconds"]):
        g = e["game"]
        order.setdefault(g, len(order))
        if e["event"] == "admit":
            admits[g] += 1
            if g in open_park:
                spans[g].append({**open_park.pop(g), "admit": e["wall_seconds"]})
        else:
            open_park[g] = {"park": e["wall_seconds"], "admits": admits[g], "levels": e.get("levels", 0),
                            "index": order[g]}
    for g, span in open_park.items():
        spans[g].append({**span, "admit": None})
    return spans


def replay(reqs, pages_total, B, shared, *, tail=None, ping=None, groups=4, chunk=4096, parks=None, wall_offset=0.0,
           ttl=None):
    """Page-level LRU replay (the v2 served arm) with the v3 options. ``tail``: None or a placement; ``ping``:
    None or {"games", "every", "tick", "order"}; ``ttl``: an entry not used within ``ttl`` seconds of its request's end
    moves to the head of the free queue (still hittable until reused)."""
    step = chunk // B * B
    events = [(r["start"], 1, j) for j, r in enumerate(reqs)] + [(r["end"], 3, j) for j, r in enumerate(reqs)]
    if tail and ttl is not None:
        events += [(r["end"] + ttl, 4, j) for j, r in enumerate(reqs)]
    if ping:
        t = ping["tick"]
        last = max(r["end"] for r in reqs)
        while t < last:
            events.append((t, 2, -1))
            t += ping["tick"]
    events.sort()
    lru = OrderedDict()
    free = pages_total
    pinned, game_state = {}, {}
    lost, stats, seq = defaultdict(float), defaultdict(float), [0]
    touched = {}
    next_req = defaultdict(list)
    for j, r in enumerate(reqs):
        next_req[r["game"]].append(j)

    def key(kind, g):
        seq[0] += 1
        return (kind, g, seq[0])

    def alloc(n):
        nonlocal free
        take = min(n, free)
        free -= take
        n -= take
        while n > 0 and lru:
            p = lru.popitem(last=False)[1]
            use = min(n, p)
            n -= use
            free += p - use
        stats["overcommit"] += n

    def release(k, pages, head=False):
        lru[k] = pages
        if head:
            lru.move_to_end(k, last=False)

    def resident(gs, *names):
        return gs is not None and all(gs.get(n) in lru for n in names)

    def parked_at(t):
        out = []
        for g, spans in (parks or {}).items():
            for s in spans:
                if s["park"] + wall_offset <= t and (s["admit"] is None or t < s["admit"] + wall_offset):
                    out.append((g, s))
        return out

    entry_of = {}
    for t, kind, j in events:
        if kind == 4:
            k = entry_of.pop(j, None)
            if k is not None and k in lru:
                lru.move_to_end(k, last=False)
                stats["ttl_demoted"] += 1
            continue
        if kind == 2:
            candidates = parked_at(t)
            if ping["order"] == "oracle":
                candidates = [c for c in candidates if c[1]["admit"] is not None]
                candidates.sort(key=lambda c: c[1]["admit"])
            else:
                candidates.sort(key=lambda c: (c[1]["admits"], -c[1]["levels"], c[1]["index"]))
            for g, span in candidates[:ping["games"]]:
                if t - touched.get(g, -1e9) < ping["every"]:
                    continue
                gs = game_state.get(g)
                if gs is None:
                    continue
                touched[g] = t
                stats["pings"] += 1
                names = [n for n in ("fa", "st", "tail") if gs.get(n) is not None]
                if resident(gs, "fa", "st") or resident(gs, "fa", "tail"):
                    for n in names:
                        if gs[n] in lru:
                            lru.move_to_end(gs[n])
                    stats["ping_refresh"] += 1
                elif ping.get("refresh_only"):
                    stats["ping_skipped_evicted"] += 1
                else:
                    # Evicted: the ping re-prefills the game's pending prompt (the work its re-admission would do).
                    for n in names:
                        if gs[n] in lru:
                            free += lru.pop(gs[n])
                    length = gs["length"]
                    alloc(math.ceil(length / B) + groups)
                    gs["fa"], gs["st"], gs["tail"] = key("fa", g), key("st", g), None
                    lru[gs["fa"]] = math.ceil(length / B)
                    lru[gs["st"]] = groups
                    stats["ping_prefill_tokens"] += max(0, length - shared)
                    stats["ping_refill"] += 1
            continue
        r = reqs[j]
        g, P, G = r["game"], r["P"], r["G"]
        if kind == 1:
            touched[g] = t
            cont = r["cat"] == "continuation"
            want = r.get("ideal_hit", 0) if cont else min(shared, P)
            want_tail = min(r.get("prevEnd", 0), P - 1) if cont and tail else want
            gs = game_state.get(g)
            hit, reuse = (shared if r["cat"] != "first" else min(shared, P)), 0
            used_tail = False
            if cont and tail and resident(gs, "fa", "tail") and gs.get("tail_end") is not None:
                hit = max(shared, min(want_tail, gs["tail_end"]))
                used_tail = hit > want
                reuse = sum(lru.pop(gs[n]) for n in ("fa", "st", "tail") if gs.get(n) in lru)
            elif cont and resident(gs, "fa", "st"):
                hit = max(shared, min(want, gs["length"]))
                reuse = sum(lru.pop(gs[n]) for n in ("fa", "st", "tail") if gs.get(n) in lru)
            elif gs:
                free += sum(lru.pop(gs[n]) for n in ("fa", "st", "tail") if gs.get(n) in lru)
            if cont:
                lost[r["phase"]] += max(0, want_tail - hit)
                stats["continuations"] += 1
                stats["tail_hits"] += used_tail
                stats["tail_tokens"] += max(0, hit - want) if used_tail else 0
                stats[f"uncached_{r['phase']}"] += P - hit
            T = P + G
            A = P // B * B
            prefill_states = max(0, math.ceil((A - hit) / step)) if A > hit else 0
            crossings = max(0, (T - 1) // B - P // B)
            need = math.ceil(T / B) + groups + 1
            pinned[j] = {"prefill_states": prefill_states, "crossings": crossings, "hit": hit}
            alloc(max(0, need - reuse) + groups * (prefill_states + crossings))
            useful = crossings > 0 or prefill_states > 0
            for _ in range(prefill_states + crossings - int(useful)):
                release(key("garbage", g), groups, head=True)
        else:
            touched[g] = t
            info = pinned.pop(j)
            T = P + G
            useful = info["crossings"] > 0 or info["prefill_states"] > 0
            st_key = key("st", g)
            tail_key = None
            fa_key = key("fa", g)
            if tail:
                # The running states + ring become the entry; the kept boundary state (only when the request wrote
                # one) goes to the head under "swap", else to the tail (v2). The entry goes after the attention pages
                # (the runtime's after_free), so it is never evicted before the pages it needs.
                if useful:
                    release(st_key, groups, head=(tail == "swap"))
                else:
                    st_key = None
                lru[fa_key] = math.ceil(T / B)
                tail_key = key("tail", g)
                release(tail_key, groups + 1, head=(tail == "front"))
                entry_of[j] = tail_key
            elif useful:
                release(st_key, groups)  # v2 keep: the newest boundary state ages with the attention pages
                free += groups + 1  # the running states and the ring, unhashed
            else:
                release(st_key, groups, head=True)  # the state it resumed from (v2 rule)
                free += 1
            if fa_key not in lru:
                lru[fa_key] = math.ceil(T / B)
            game_state[g] = {"fa": fa_key, "st": st_key, "tail": tail_key, "tail_end": T if tail else None,
                             "length": T if useful else info["hit"]}
    out = {"lost_prefix_tokens": {k: round(v) for k, v in lost.items()}, **{k: round(v) for k, v in stats.items()}}
    return out


def arms(ping_games=(5, 10, 20), every=(120.0, 240.0, 480.0), tick=30.0):
    result = {"v2 (served hero-v2)": {}, "v3 tail swap": {"tail": "swap"}, "v3 tail back": {"tail": "back"},
              "v3 tail front": {"tail": "front"}}
    for ttl in (10.0, 30.0, 60.0, 120.0, 300.0):
        result[f"v3 tail swap ttl={ttl:g}"] = {"tail": "swap", "ttl": ttl}
    for order in ("oracle", "readmissions"):
        for k in ping_games:
            spec = {"games": k, "every": every[1], "tick": tick, "order": order}
            result[f"v2 + ping {order} K={k}"] = {"ping": spec}
            for e in every:
                refresh = {**spec, "every": e, "refresh_only": True}
                result[f"v2 + ping-refresh {order} K={k} every={e:g}"] = {"ping": refresh}
                result[f"v3 tail swap + ping-refresh {order} K={k} every={e:g}"] = {"tail": "swap", "ping": refresh}
    return result


def summarize(name, rows, base, wall, B):
    base_unc = base["uncached_opening"] + base["uncached_phase2"]
    out = {}
    for arm, row in rows.items():
        saved = {ph: base[f"uncached_{ph}"] - row[f"uncached_{ph}"] for ph in ("opening", "phase2")}
        cost = row.get("ping_prefill_tokens", 0)
        seconds = sum(saved[ph] * COST_S[ph] for ph in saved) - cost * COST_S["phase2"]
        out[arm] = {"saved_opening": saved["opening"], "saved_phase2": saved["phase2"], "ping_prefill": cost,
                    "net_saved": saved["opening"] + saved["phase2"] - cost,
                    "net_saved_share_of_uncached": round((saved["opening"] + saved["phase2"] - cost) / base_unc, 4),
                    "engine_seconds_saved": round(seconds, 1),
                    "tok_s_gain": round(seconds / (wall - seconds), 4), "pings": row.get("pings", 0),
                    "tail_hits": row.get("tail_hits", 0), "continuations": row.get("continuations", 0)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--admission", required=True)
    ap.add_argument("--pages", type=int, required=True, help="pool blocks (cache-layout.json num_blocks)")
    ap.add_argument("--wall", type=float, default=None, help="engine seconds for the tok/s projection (default: run span)")
    ap.add_argument("--tick", type=float, default=30.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    d = json.loads(Path(a.attribution).read_text())
    B, shared = d["summary"]["block"], d["summary"]["shared_prefix"]
    reqs = d["requests"]
    parks = load_parks(a.admission)
    # Request times are on the game-log clock, the admission log on the arena wall clock (same origin, the trial start).
    wall = a.wall or (max(r["end"] for r in reqs) - min(r["start"] for r in reqs))
    rows = {arm: replay(reqs, a.pages, B, shared, parks=parks, **kw) for arm, kw in arms(tick=a.tick).items()}
    base = rows["v2 (served hero-v2)"]
    result = {"attribution": a.attribution, "block": B, "pages": a.pages, "wall": round(wall, 1),
              "summary": summarize(Path(a.attribution).stem, rows, base, wall, B), "arms": rows}
    print(json.dumps(result["summary"], indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
