"""Envs hero watcher rules (scripts/posttrain/v7_envs_watch.py): each rule fires on a doctored feed and stays quiet on
the healthy one it was made from."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("v7_envs_watch", Path(__file__).resolve().parents[1] / "scripts/posttrain/v7_envs_watch.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)

T0 = 1_000_000.0
STEP_S = 2100  # hero step ~35 min


def healthy(steps=8):
    """A hero-shaped feed: 8 groups of 12 per step (half from fresh pool starts), explorers with starts and levels,
    envs rows with a deep pool, mix on target, blocks that play."""
    ev = [{"kind": "v7_start", "epoch": T0}, {"kind": "v7e_setup", "epoch": T0, "active": 60},
          {"kind": "explore_start", "epoch": T0 + 1}]
    harvest, n = [], 0
    shares = {"public": 4, "generated": 2, "puzzlescript": 1, "sonpham_interactive": 1}
    made = {}
    for s in range(1, steps + 1):
        t = T0 + s * STEP_S
        for k in range(8):
            n += 1
            start = None
            if k % 2:
                att = f"/runs/x/harvest/x{n:04d}"
                harvest.append({"attempt": att, "request": 5, "played_by": s - 1})
                start = [att, 5]
            ev.append({"kind": "group_start", "epoch": t - 1800 + k, "gid": f"g{n:05d}", "start": start, "version": s - 1})
            ev.append({"kind": "group_done", "epoch": t - 60 + k, "tries": 12, "clears": 3 + k % 6})
        for src, w in shares.items():
            made[src] = made.get(src, 0) + w
        ev += [{"kind": "explore_seg", "epoch": t - 900}, {"kind": "explore_starts", "epoch": t - 800, "new": 2},
               {"kind": "explore_play", "epoch": t - 700, "levels": 1 + s % 2},
               {"kind": "block_off", "epoch": t - 300, "served": 10},
               {"kind": "step", "epoch": t, "step": s},
               {"kind": "envs", "epoch": t, "step": s, "fresh_by_level": {"1": 5, "2": 20, "3": 9},
                "groups_by_source": dict(made), "active": 60}]
        if s == 1:
            ev.append({"kind": "explore_first_deep", "epoch": t - 1000})
    return sorted(ev, key=lambda e: e["epoch"]), harvest


CFG = {"shares": {"public": 0.5, "generated": 0.25, "puzzlescript": 0.125, "sonpham_interactive": 0.125}}


def fires(ev, harvest, rule, now=None):
    now = now or max(e["epoch"] for e in ev) + 60
    d = W.decision(ev, harvest, now, CFG, only=rule)
    return d is not None and d[0] == "warn" and d[1] == rule


def test_healthy_hero_feed_is_quiet_for_every_rule():
    ev, h = healthy()
    for now in range(int(T0) + 60, int(max(e["epoch"] for e in ev)) + 61, 300):
        assert W.decision(ev, h, now, CFG) is None, now


def test_band_fires_on_all_or_none_groups():
    ev, h = healthy()
    for e in ev:
        if e["kind"] == "group_done":
            e["clears"] = 12 if e["epoch"] % 2 else 0
    assert fires(ev, h, "band")


def test_fresh_fires_on_a_stale_or_overused_draw():
    ev, h = healthy()
    old = dict(next(e for e in ev if e["kind"] == "group_start" and e["start"]))
    late = {**old, "epoch": ev[-1]["epoch"], "gid": "late", "version": old["version"] + 5}
    assert fires(ev + [late], h, "fresh")
    again = [{**old, "epoch": ev[-1]["epoch"] + i, "gid": f"again{i}"} for i in range(2)]
    assert fires(ev + again, h, "fresh")


def test_deep_dry_fires_when_the_l2_pool_stays_under_one_step():
    ev, h = healthy()
    for e in ev:
        if e["kind"] == "envs" and e["step"] >= 6:
            e["fresh_by_level"] = {"1": 5, "2": 3}
    assert fires(ev, h, "deep_dry")


def test_first_deep_and_silent_explorers():
    ev, h = healthy()
    no_deep = [e for e in ev if e["kind"] != "explore_first_deep"]
    assert fires(no_deep, h, "first_deep")
    t_mid = T0 + 3 * STEP_S
    silent = [e for e in ev if not (e["kind"] in ("explore_seg", "explore_starts") and e["epoch"] > t_mid)]
    assert fires(silent, h, "explore_silent")
    one_signal = [e for e in ev if not (e["kind"] == "explore_seg" and e["epoch"] > t_mid)]  # starts still land
    assert not fires(one_signal, h, "explore_silent")


def test_explore_levels_drops_scan():
    ev, h = healthy()
    for e in ev:
        if e["kind"] == "explore_play":
            e["levels"] = 0
    assert fires(ev, h, "explore_levels")
    ev, h = healthy()
    end = ev[-1]["epoch"]
    assert fires(ev + [{"kind": "explore_drop", "epoch": end - i} for i in range(20)], h, "drops")
    assert fires(ev + [{"kind": "explore_scan_failed", "epoch": end - i, "error": "x"} for i in range(3)], h, "scan")


def test_mix_rotation_l1_starved_mode_b():
    ev, h = healthy()
    last = [e for e in ev if e["kind"] == "envs"][-1]
    last["groups_by_source"] = {"public": 60, "generated": 2, "puzzlescript": 0, "sonpham_interactive": 2}
    assert fires(ev, h, "mix")
    ev, h = healthy()
    [e for e in ev if e["kind"] == "envs"][-1]["active"] = 40
    assert fires(ev, h, "rotation")
    ev, h = healthy()
    assert fires(ev + [{"kind": "rotate", "epoch": ev[-1]["epoch"], "step": 8, "out": f"g{i}"} for i in range(16)],
                 h, "rotation")
    ev, h = healthy()
    for e in [e for e in ev if e["kind"] == "group_start"][-24:]:
        e["start"] = None
    assert fires(ev, h, "l1")
    ev, h = healthy()
    end = ev[-1]["epoch"]
    assert fires(ev + [{"kind": "starved", "epoch": end - 60 * i} for i in range(5)], h, "starved")
    assert not fires(ev + [{"kind": "starved", "epoch": end - 400 * i} for i in range(5)], h, "starved")
    ev, h = healthy()
    for e in [e for e in ev if e["kind"] == "block_off"][-3:]:
        e["served"] = 0
    assert fires(ev, h, "mode_b_idle")


def test_acked_rule_is_skipped_and_v7_end_is_done():
    ev, h = healthy()
    for e in [e for e in ev if e["kind"] == "block_off"][-3:]:
        e["served"] = 0
    now = ev[-1]["epoch"] + 60
    assert W.decision(ev, h, now, CFG)[1] == "mode_b_idle"
    assert W.decision(ev, h, now, CFG, acked=("mode_b_idle",)) is None
    assert W.decision(ev + [{"kind": "v7_end", "epoch": now - 1, "why": "max_steps"}], h, now, CFG)[0] == "done"
