"""Envs-side watcher of a v7 hero run (checklist item 7), beside RL's loop watcher (.rl-tools/v7/v7_watch.py: errors,
NaN, stalls, group drops, Mode B errors). Reads the run's feed and harvest.jsonl from the volume every 5 min, prints one
status line, and on a rule prints ALERT and exits 2 (the RL agent inspects and decides; it never stops a run). Exit 0
at v7_end. Re-arm with the fired rule in ACKED to keep watching the others.
usage: v7_envs_watch.py watch RUN_DIR [ACKED,RULES] [SHARES_JSON]   (RUN_DIR: validation/<run> on arc-prolong-runs)
       v7_envs_watch.py backtest FEED.jsonl [HARVEST.jsonl] [SHARES_JSON]

Rules, pre-registered 24 Oct before the hero (hero: 8 groups of 12 per ~35-min step, 4 explorers admitted 150 s
apart, starts used <= 2x while <= 4 versions old). Measured inputs: hero-v3 card-2 read (ap-cSH14, real speed): first
L2+ explorer start 1,061 s after explore_start, 7 segments in 47 min; CPU legs xv7e pfA / pfD2, xv7f pfR (stub speed).
- band: of the last 24 groups of >= 8 tries, share with 0 < clears < tries below 0.20. Expected 0.42 for uniform
  picks over the hero sets' seed priors (1 - p^12 - (1-p)^12; the curriculum picks for spread, so higher); 0.20 is
  ~2.2 SE (n 24) below it.
- fresh: a drawn start more than `stale` versions older than its group, or drawn more than `max_uses` times (both
  impossible by design: a freshness bug), from group_start x harvest.jsonl played_by.
- deep_dry: from step 6, fewer than 8 drawable L2+ starts (one step's groups) in 3 consecutive envs rows.
- first_deep: no L2+ explorer start 3,180 s (3 x the read's 1,061 s) after explore_start.
- explore_silent (two signals): explorers on, and for 6,000 s (2 x a 65,536-token segment at ~22 tok/s) no
  explore_seg AND no explore_starts row.
- explore_levels: >= 8 finished chains and at most 1 reached any level.
- drops / scan: dropped starts > 50% of (dropped + added) at >= 10 events in 4 h; >= 3 explore_scan_failed in 1 h.
- mix: after 48 groups, a source with target share >= 0.10 below half its target, or public more than 0.15 under.
- rotation: active games below 80% of the setup's, or more than 15 games rotated within 8 steps.
- l1: from step 6, more than half of the last 24 groups started at level 1 (deep starts not flowing).
- starved: >= 5 starved rows (scheduler had room, no group could be made) within 15 min.
- mode_b_idle: 3 consecutive Mode B blocks that played 0 tries without an error (xv7c pfA class)."""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BAND_N, BAND_MIN_TRIES, BAND_LOW = 24, 8, 0.20
DEEP_FROM_STEP, DEEP_MIN, DEEP_ROWS = 6, 8, 3
FIRST_DEEP_S = 3180
SILENT_S = 6000
LEVEL_CHAINS = 8
DROP_SHARE, DROP_MIN, DROP_H = 0.5, 10, 4
SCAN_FAILS = 3
MIX_AFTER, MIX_MIN_SHARE, MIX_HALF, PUBLIC_UNDER = 48, 0.10, 0.5, 0.15
ACTIVE_FLOOR, ROT_MAX, ROT_STEPS = 0.8, 15, 8
L1_FROM_STEP, L1_N, L1_SHARE = 6, 24, 0.5
STARVED_N, STARVED_S = 5, 900
IDLE_BLOCKS = 3
HERO_SHARES = {"public": 0.50, "generated": 0.21, "puzzlescript": 0.14, "sonpham_interactive": 0.11, "witness": 0.04}
RULES = ("band", "fresh", "deep_dry", "first_deep", "explore_silent", "explore_levels", "drops", "scan", "mix",
         "rotation", "l1", "starved", "mode_b_idle")


def _ep(e):
    return float(e.get("epoch") or 0)


def _step(events):
    return max([int(e.get("step") or 0) for e in events if e.get("kind") == "step"] or [0])


def decision(events, harvest, now, cfg=None, acked=(), only=None):
    """("warn", rule, why) | ("done", None, why) | None. ``harvest``: harvest.jsonl rows; cfg: {"shares", "stale",
    "max_uses"}; ``acked``: rules already alerted (the agent re-armed past them); ``only``: one rule (backtests)."""
    cfg = cfg or {}
    on = lambda r: (only is None or only == r) and r not in acked  # noqa: E731
    ev = [e for e in events if _ep(e) <= now]
    for e in ev:
        if e.get("kind") == "v7_end":
            return "done", None, f"v7_end why={e.get('why')} step={e.get('step')}"
    step = _step(ev)

    if on("band"):
        gd = [e for e in ev if e.get("kind") == "group_done" and int(e.get("tries") or 0) >= BAND_MIN_TRIES][-BAND_N:]
        if len(gd) == BAND_N:
            band = sum(1 for e in gd if 0 < int(e["clears"]) < int(e["tries"]))
            if band / BAND_N < BAND_LOW:
                zero = sum(1 for e in gd if int(e["clears"]) == 0)
                return "warn", "band", (f"band share {band}/{BAND_N} of the last groups (< {BAND_LOW}); none cleared "
                                        f"{zero}, all cleared {BAND_N - band - zero}")

    if on("fresh") and harvest:
        stale, uses_max = int(cfg.get("stale", 4)), int(cfg.get("max_uses", 2))
        born = {(r["attempt"], int(r["request"])): int(r.get("played_by") or 0) for r in harvest}
        uses = {}
        for e in ev:
            if e.get("kind") != "group_start" or not e.get("start"):
                continue
            key = (e["start"][0], int(e["start"][1]))
            if key not in born:
                continue  # a seed (bank) start: not in the pool's rows
            uses[key] = uses.get(key, 0) + 1
            lag = int(e.get("version") or 0) - born[key]
            if lag > stale:
                return "warn", "fresh", f"{e.get('gid')} drew a start {lag} versions old (stale {stale}): {key[0][-60:]}"
            if uses[key] > uses_max:
                return "warn", "fresh", f"{e.get('gid')} drew a start a {uses[key]}th time (max_uses {uses_max})"

    rows = [e for e in ev if e.get("kind") == "envs"]
    if on("deep_dry"):
        tail = [r for r in rows if int(r.get("step") or 0) >= DEEP_FROM_STEP][-DEEP_ROWS:]
        deep = [sum(int(n) for lv, n in (r.get("fresh_by_level") or {}).items() if int(lv) >= 2) for r in tail]
        if len(tail) == DEEP_ROWS and all(d < DEEP_MIN for d in deep):
            return "warn", "deep_dry", f"drawable L2+ starts {deep} at steps {[r['step'] for r in tail]} (< {DEEP_MIN})"

    xs = next((e for e in ev if e.get("kind") == "explore_start"), None)
    stopped = any(e.get("kind") == "explore_stop" for e in ev)
    if xs is not None and not stopped:
        t0 = _ep(xs)
        if on("first_deep") and now - t0 > FIRST_DEEP_S and not any(e.get("kind") == "explore_first_deep" for e in ev):
            return "warn", "first_deep", f"no L2+ explorer start {(now - t0) / 60:.0f} min after explorers started"
        last = max([_ep(e) for e in ev if e.get("kind") in ("explore_seg", "explore_starts")] or [t0])
        if on("explore_silent") and now - last > SILENT_S:
            return "warn", "explore_silent", f"no explorer segment or start for {(now - last) / 60:.0f} min"
    plays = [e for e in ev if e.get("kind") == "explore_play"]
    if on("explore_levels") and len(plays) >= LEVEL_CHAINS and sum(1 for e in plays if int(e.get("levels") or 0)) <= 1:
        return "warn", "explore_levels", f"{len(plays)} explorer chains, {sum(1 for e in plays if e.get('levels'))} reached a level"

    win = lambda h: [e for e in ev if now - 3600 * h <= _ep(e)]  # noqa: E731
    w = win(DROP_H)
    drops = sum(1 for e in w if e.get("kind") == "explore_drop")
    added = sum(int(e.get("new") or 0) for e in w if e.get("kind") == "explore_starts")
    if on("drops") and drops + added >= DROP_MIN and drops / (drops + added) > DROP_SHARE:
        return "warn", "drops", f"explorer starts dropped {drops}/{drops + added} in {DROP_H} h"
    fails = [e for e in win(1) if e.get("kind") == "explore_scan_failed"]
    if on("scan") and len(fails) >= SCAN_FAILS:
        return "warn", "scan", f"{len(fails)} explorer scan/launch failures in 1 h: {str(fails[-1].get('error'))[:120]}"

    if on("mix") and rows:
        made = rows[-1].get("groups_by_source") or {}
        total = sum(made.values())
        shares = cfg.get("shares") or HERO_SHARES
        norm = sum(shares.values())
        if total >= MIX_AFTER:
            for src, sh in shares.items():
                target, got = sh / norm, made.get(src, 0) / total
                if src == "public" and got < target - PUBLIC_UNDER:
                    return "warn", "mix", f"public {got:.2f} of {total} groups vs target {target:.2f}"
                if target >= MIX_MIN_SHARE and got < MIX_HALF * target:
                    return "warn", "mix", f"{src} {got:.2f} of {total} groups vs target {target:.2f}"

    setup = next((e for e in ev if e.get("kind") == "v7e_setup"), None)
    if on("rotation") and setup and rows:
        if int(rows[-1].get("active") or 0) < ACTIVE_FLOOR * int(setup.get("active") or 0):
            return "warn", "rotation", f"active games {rows[-1].get('active')} of {setup.get('active')} at setup"
        recent = [e for e in ev if e.get("kind") == "rotate" and int(e.get("step") or 0) > step - ROT_STEPS]
        n = len(recent)  # one rotate row per game swapped out
        if n > ROT_MAX:
            return "warn", "rotation", f"{n} games rotated out within {ROT_STEPS} steps"

    if on("l1") and step >= L1_FROM_STEP:
        gs = [e for e in ev if e.get("kind") == "group_start"][-L1_N:]
        l1 = sum(1 for e in gs if not e.get("start"))
        if len(gs) == L1_N and l1 / L1_N > L1_SHARE:
            return "warn", "l1", f"{l1}/{L1_N} recent groups started at level 1 (deep starts not flowing)"

    starved = [e for e in ev if e.get("kind") == "starved" and now - _ep(e) <= STARVED_S]
    if on("starved") and len(starved) >= STARVED_N:
        return "warn", "starved", f"{len(starved)} starved rows in {STARVED_S // 60} min (room for groups, none made)"

    offs = [e for e in ev if e.get("kind") == "block_off" and not e.get("error")][-IDLE_BLOCKS:]
    if on("mode_b_idle") and len(offs) == IDLE_BLOCKS and all(int(e.get("served") or 0) == 0 for e in offs):
        return "warn", "mode_b_idle", f"last {IDLE_BLOCKS} Mode B blocks played 0 tries"
    return None


def status(events):
    rows = [e for e in events if e.get("kind") == "envs"]
    gd = [e for e in events if e.get("kind") == "group_done" and int(e.get("tries") or 0) >= BAND_MIN_TRIES][-BAND_N:]
    band = sum(1 for e in gd if 0 < int(e["clears"]) < int(e["tries"]))
    last = rows[-1] if rows else {}
    return (f"step {_step(events)} band {band}/{len(gd)} fresh {last.get('fresh_by_level')} pool {last.get('pool')} "
            f"explore {last.get('explore')} segs {sum(1 for e in events if e.get('kind') == 'explore_seg')} "
            f"chains {sum(1 for e in events if e.get('kind') == 'explore_play')} mix {last.get('groups_by_source')}")


def jsonl(text):
    out = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def backtest(events, harvest, cfg, step_s=300, tail_s=1800):
    """Replay at the watcher's 5-min passes, through ``tail_s`` past the last row: the first rule that fires (None if
    the feed stays quiet)."""
    t0, end = min(_ep(e) for e in events), max(_ep(e) for e in events)
    for now in range(int(t0) + 60, int(end) + tail_s + 1, step_s):
        d = decision(events, harvest, now, cfg)
        if d and d[0] == "done":
            return None
        if d:
            return {"rule": d[1], "at_min": round((now - t0) / 60, 1), "why": d[2]}
    return None


def main(argv):
    if argv[1] == "backtest":
        events = jsonl(Path(argv[2]).read_text())
        harvest = jsonl(Path(argv[3]).read_text()) if len(argv) > 3 and argv[3] else []
        cfg = {"shares": json.loads(argv[4])} if len(argv) > 4 else {}
        print(json.dumps(backtest(events, harvest, cfg)))
        return 0
    run_dir = argv[2]
    acked = tuple(x for x in (argv[3] if len(argv) > 3 else "").split(",") if x)
    cfg = {"shares": json.loads(argv[4])} if len(argv) > 4 else {}
    tmp = Path(tempfile.mkdtemp())
    read_ok = time.time()

    def get(rel):
        out = tmp / Path(rel).name
        subprocess.run(["uv", "run", "--extra", "modal", "modal", "volume", "get", "--force", "arc-prolong-runs", rel,
                        str(out)], capture_output=True, timeout=300, check=True)
        return out.read_text(errors="replace")

    while True:
        try:
            events = jsonl(get(f"{run_dir}/arena/trials/rl-loop/rl/feed.jsonl"))
            harvest = jsonl(get(f"{run_dir}/harvest.jsonl"))
            read_ok = time.time()
        except Exception as error:  # noqa: BLE001 - a failed read is no data, never "quiet"
            if time.time() - read_ok > 1800:
                print(f"ALERT {run_dir}: envs watcher blind 30 min ({type(error).__name__})", flush=True)
                return 2
            time.sleep(300)
            continue
        print(f"[{time.strftime('%H:%M')}] {status(events)}", flush=True)
        d = decision(events, harvest, time.time(), cfg, acked)
        if d and d[0] == "done":
            print(d[2], flush=True)
            return 0
        if d:
            print(f"ALERT {run_dir}: {d[1]}: {d[2]} (re-arm with {','.join(acked + (d[1],))})", flush=True)
            return 2
        time.sleep(300)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
