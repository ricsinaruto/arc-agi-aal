"""The envs part of the hero run's loop config (docs/rl-plan.md "Hero run", docs/envs-plan.md section 10):
the 15 public trainable games + the seed round's 45 generated games active, the rest of the pool as reserve (rotated in
as active games stop teaching), every level with an action reference as a training set, the held-out 10 as reads
from the seed round's recorded starts, the source mix, the per-group curriculum, the start pool, explorers,
rotation and the v7 read keys. The RL arm's hero derive calls ``envs_hero_cfg`` and owns everything else.

Inputs are plain data: ``pool`` (the envs pool file, pool-v4.json: sets with refs/priors/caps, points, shares; built
with the 18 audited families and the held-out 10 left out), ``seed_study`` (the seed bundle's comparison.json: its
45 env games and the public games it recorded), ``points`` (the bundle's points: level counts and baselines).
"""
from __future__ import annotations

import collections

HELD = ("tr87", "r11l", "m0r0", "dc22", "s5i5", "su15", "sp80", "g50t", "lf52", "bp35")


def band(s):
    p = s.get("prior") or {}
    t, c = int(p.get("tries") or 0), int(p.get("clears") or 0)
    return bool(t) and 0.3 <= 0.8 * c / t <= 0.85 and c / t < 0.95  # the probe table's band rule (0.95+: easy)


def envs_hero_cfg(mg, *, pool, seed_study, points, seed_run, reserve=True, batch_groups=8, group_size=12):
    """Sets the envs keys on ``mg`` (in place). Returns what the bundle must ship: ``ship`` (game ids: active,
    reserve, held-out), ``points_add`` (pool points of games the bundle's points lack), ``reserve`` (codes, first in
    first), ``no_action_ref`` (levels left out: no reward), counts."""
    env = list(seed_study["envs_seed"]["env_games"])
    pub = sorted({r["game"] for r in seed_study["rl_loop"]["mg"]["record"]} - set(env) - set(HELD))
    if len(pub) != 15 or len(env) != 45:
        raise ValueError(f"expected 15 public + 45 env games, got {len(pub)} + {len(env)}")
    have = {p["id"].split("-")[0]: p for p in points}
    pool_pts = {p["id"].split("-")[0]: p for p in pool["points"]}
    lv = collections.defaultdict(dict)
    for s in pool["sets"]:
        lv[s["game"]][int(s["level"])] = s
    if set(lv) & set(HELD):
        raise ValueError(f"held-out games in the pool: {sorted(set(lv) & set(HELD))}")
    res = []
    if reserve:
        cands = [g for g in lv if g not in pub and g not in env and (pool_pts.get(g) or {}).get("board_hash")]
        res = sorted(cands, key=lambda g: (-sum(band(s) for s in lv[g].values()), g))  # most band levels first
    sets, no_ref = [], []
    for g in pub + sorted(env) + res:
        pt = have.get(g) or pool_pts.get(g)
        if pt is None:
            raise ValueError(f"{g}: no point")
        base = pt.get("baselines") or []
        for level in range(1, int(pt["total_levels"]) + 1):
            s = lv[g].get(level)
            if s is None:
                raise ValueError(f"{g}-L{level}: no pool row")
            if not ((s.get("ref") or {}).get("baseline") or (level <= len(base) and base[level - 1])):
                no_ref.append(f"{g}-L{level}")  # no action reference: the reward is undefined
                continue
            row = {k: v for k, v in s.items() if k != "fresh"}
            row.update(role="train", **({"fresh": True} if level == 1 else {}), **({"reserve": True} if g in res else {}))
            sets.append(row)
    reads = []
    for c in HELD:
        total = int((have.get(c) or {}).get("total_levels") or 0)
        if not total:
            raise ValueError(f"held-out {c}: not in the bundle's points")
        reads.append({"name": f"{c}-L1", "game": c, "level": 1, "role": "test", "fresh": True, "read": True,
                      "tries": 0, "round_tries": 0, "cap": 64000})
        for level in range(2, total + 1):
            reads.append({"name": f"{c}-L{level}", "game": c, "level": level, "role": "test", "read": "screen",
                          "starts": 2, "tries": 0, "round_tries": 0, "cap": 64000, "max_prompt": 10 ** 9})
    active = pub + sorted(env)
    levels = sorted({s["level"] for s in sets if s["level"] >= 2})
    for k in ("screens", "depth_bins", "unseen_read", "half_split", "held_starts", "fixed_fresh", "no_repeat"):
        mg.pop(k, None)
    mg.update(
        sets=sets + reads, group_size=group_size,
        screen=f"/runs/validation/{seed_run}/arena/trials/rl-loop/rl/screen.json",
        screen_root=f"/runs/validation/{seed_run}/arena", seed_starts="screen",
        fixed_sets=[r["name"] for r in reads],
        # explorers: 4 live, KV counted at a 131,072 context, admitted staggered (mg.admit_stagger_s,
        # shared with the bench tries: v7_envs.admit_staggered); 300K whole
        # games as one-level segments (seg_tokens), public games the full budget, others stop at 2x the level's cap
        explore={"streams": 4, "budget": 300000, "games": active, "per_game": 2, "full_sources": ["public"],
                 "stuck": 2.0, "wall": 28800, "kv_share": 0.30, "seg_tokens": 65536, "context": 131072},
        running_caps={},  # 1.25 x p80 of clears, 4K..262K (play_explorer.CAP_KEYS)
        # per group (v7 plan_one): one hero step = 96 tries = 2/3 of a Run 1 update (144), so half_life and
        # reprobe_every in steps are 1.5x Run 1's (RL 23 Oct); park/retire off as in Run 1 (rotation has its own lines)
        curriculum={"groups": batch_groups, "group_size": group_size, "shares": dict(pool["shares"]), "seed": 1,
                    "token_share": 1.0, "park_p": 0.0, "retire_p": 1.01, "family_cap": 0.17, "half_life": 3.0,
                    "reprobe_every": 6, "window": 12},
        # start pool: used at most twice, only while <= 4 versions behind the trainer (~2.3 h at 35-min
        # steps); every level a start can be cut into; no set_level jumps or grafts
        harvest={"levels": levels, "from_round": 1, "switch_round": 1, "train_starts": 6, "train_tries": group_size,
                 "max_uses": 2, "fill": False, "board_check": True, "stale": 4},
        rotation={"keep": ["public"], "min_explored": 1, "every": 1, "retire_p": 0.95, "park_p": 0.10,
                  "min_tries": 2 * group_size},
    )
    v7 = dict(mg.get("v7") or {})
    v7.update(group_size=group_size, batch_groups=batch_groups, read_every=8, read_per_start=2, read_fresh_tries=4,
              read_share=0.10, read_base=True)
    mg["v7"] = v7
    mg["async"] = True
    ship = sorted({(have.get(g) or pool_pts[g])["game"] for g in active + res} | {have[c]["game"] for c in HELD})
    return {"ship": ship, "points_add": [pool_pts[g] for g in active + res if g not in have], "reserve": res,
            "active": active, "no_action_ref": no_ref, "train_sets": len(sets), "reads": len(reads),
            "by_source": dict(collections.Counter(s["source"] for s in sets if not s.get("reserve"))),
            "reserve_by_source": dict(collections.Counter(s["source"] for s in sets if s.get("reserve") and s["level"] == 1))}
