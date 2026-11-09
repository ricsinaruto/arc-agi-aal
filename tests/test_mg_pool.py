import pytest

from arena import mg_pool, rl_loop


def registry():
    games = [
        {"game_id": "q301-aa", "source": "writer", "family": "writer:q301", "split": "train", "levels": 4,
         "solution_lengths": [10, 20, 30, 40]},
        {"game_id": "p001-bb", "source": "puzzlescript", "family": "ps:a", "split": "train", "levels": 3,
         "solution_lengths": []},
        {"game_id": "w001-cc", "source": "witness", "family": "witness:tw02", "split": "he", "levels": 3},
        {"game_id": "m0r0-dd", "source": "public", "family": "public:m0r0", "split": "train", "levels": 3},
        {"game_id": "cd82-ee", "source": "public", "family": "public:cd82", "split": "dev_b", "levels": 3,
         "solution_lengths": [50, 60, 70]},
        {"game_id": "q016-ff", "source": "writer", "family": "writer:q016", "split": "train", "levels": 2,
         "excluded": "near-copy of m0r0"},
    ]
    return {"games": games, "held_out": list(mg_pool.HELD_OUT_10)}


def row(gid, level, reached, clears, tokens=(10000, 12000, 30000), actions=25):
    return {"game_id": gid, "level": level, "reached": reached, "clears": clears, "tokens": list(tokens),
            "median_actions": actions}


def test_chain_refs_caps_and_held_out_guard():
    tab = {"levels": [row("q301-aa", 1, 6, 6), row("q301-aa", 2, 6, 3), row("q301-aa", 3, 6, 6), row("q301-aa", 4, 6, 0),
                      row("p001-bb", 1, 6, 3), row("p001-bb", 2, 1, 1),
                      row("w001-cc", 1, 6, 3), row("m0r0-dd", 1, 6, 3), row("cd82-ee", 1, 12, 7),
                      row("q016-ff", 1, 6, 3)]}
    out = mg_pool.build(registry(), [tab], hashes={"q301-aa": "h1"})
    names = [s["name"] for s in out["sets"]]
    # q301: L1 easy feeder, L2 band, L3 easy dropped (the chain ends at its deepest frontier level), L4 low stops it
    assert names == ["cd82-L1", "p001-L1", "q301-L1", "q301-L2"]
    q2 = next(s for s in out["sets"] if s["name"] == "q301-L2")
    assert q2["source"] == "generated" and q2["family"] == "writer:q301" and "fresh" not in q2
    assert q2["ref"] == {"tokens": 12000, "actions": 25, "baseline": 26, "measured": True, "n": 3} and q2["cap"] == 15000
    assert q2["prior"] == {"tries": 6, "clears": 3, "median_to_clear": 12000}
    p1 = next(s for s in out["sets"] if s["name"] == "p001-L1")
    assert p1["fresh"] and p1["ref"]["baseline"] == 25  # no solution: the median actions
    assert {p["id"] for p in out["points"]} == {"cd82-game", "p001-game", "q301-game"}
    assert out["report"]["skipped_held_out"] == 1 and out["report"]["no_board_hash"] == 2
    assert set(out["shares"]) == {"public", "generated", "puzzlescript"}
    rl_loop.check_mg({"sets": out["sets"], "curriculum": {"shares": out["shares"], "groups": 12, "group_size": 12}})


def test_held_out_game_in_a_training_split_is_refused():
    reg = registry()
    reg["games"].append({"game_id": "tr87-zz", "source": "public", "family": "public:tr87", "split": "public_train",
                         "levels": 2})
    out = mg_pool.build(reg, [{"levels": [row("tr87-zz", 1, 6, 3)]}])
    assert not any(s["game"] == "tr87" for s in out["sets"]) and out["report"]["skipped_held_out"] == 2


def test_screen_rows_with_medians_and_cap_override():
    reg = {"games": [{"game_id": "sc25-aa", "source": "public", "family": "public:sc25", "split": "public_train",
                      "levels": 3, "solution_lengths": [36, 6, 32]}]}
    extra = [{"game_id": "sc25-aa", "level": 1, "reached": 34, "clears": 34, "median_tokens": 25370, "median_actions": 41.5,
              "cap": 38000},
             {"game_id": "sc25-aa", "level": 2, "reached": 30, "clears": 21, "median_tokens": 4210, "p80_tokens": 8000,
              "median_actions": 5}]
    out = mg_pool.build(reg, [], extra=extra)
    s1, s2 = out["sets"]
    assert s1["cap"] == 38000 and s1["ref"]["tokens"] == 25370 and s1["ref"]["baseline"] == 47
    assert s2["cap"] == 10000 and s2["prior"] == {"tries": 30, "clears": 21, "median_to_clear": 4210}


def test_admit_reached_keeps_every_reached_level():
    tab = {"levels": [row("q301-aa", 1, 6, 6), row("q301-aa", 2, 6, 3), row("q301-aa", 3, 6, 6),
                      row("q301-aa", 4, 6, 0, tokens=()), row("p001-bb", 1, 6, 3), row("p001-bb", 2, 1, 1)]}
    out = mg_pool.build(registry(), [tab], admit="reached")
    names = [s["name"] for s in out["sets"]]
    assert names == ["p001-L1", "p001-L2", "q301-L1", "q301-L2", "q301-L3", "q301-L4"]
    q4 = next(s for s in out["sets"] if s["name"] == "q301-L4")
    assert q4["cap"] == 64000 and q4["ref"]["tokens"] is None and q4["ref"]["measured"] is False
    assert mg_pool.placeholder_refs(out["sets"]) == []


def test_admit_all_every_level_nodata_cap_and_owner_mix():
    tab = {"levels": [row("q301-aa", 1, 6, 6, tokens=(60000, 70000, 90000)), row("q301-aa", 2, 6, 0, tokens=())]}
    out = mg_pool.build(registry(), [tab], admit="all", cap_max=262144, cap_nodata=64000, always=("cd82",))
    names = [s["name"] for s in out["sets"]]
    assert names == ["cd82-L1", "cd82-L2", "cd82-L3", "q301-L1", "q301-L2", "q301-L3", "q301-L4"]
    by = {s["name"]: s for s in out["sets"]}
    assert by["q301-L1"]["cap"] == 88000  # p80 x 1.25 above the old 64K clamp
    assert by["q301-L2"]["cap"] == 64000 and by["q301-L4"]["cap"] == 64000 and by["q301-L4"]["prior"]["tries"] == 0
    assert abs(sum(mg_pool.SHARES.values()) - 1.0) < 1e-9 and mg_pool.SHARES["public"] == 0.5


def test_no_placeholder_refs_built_or_left_after_apply():
    """24 Oct: 324 hero sets carried ref.tokens = 64,000 (the no-data cap), saturating RL's token term."""
    tab = {"levels": [row("q301-aa", 1, 6, 6), row("q301-aa", 2, 6, 0, tokens=())]}
    out = mg_pool.build(registry(), [tab], admit="all", cap_max=262144, cap_nodata=64000, always=("cd82",))
    assert mg_pool.placeholder_refs(out["sets"]) == []
    by = {s["name"]: s for s in out["sets"]}
    assert by["q301-L2"]["ref"]["tokens"] is None and by["q301-L2"]["cap"] == 64000
    old = [{**s, "ref": {k: v for k, v in s["ref"].items() if k not in ("measured", "n")}} for s in out["sets"]]
    for s in old:
        if not s["ref"]["tokens"]:
            s["ref"]["tokens"] = 64000  # the pre-24 Oct table
    assert "q301-L2" in mg_pool.placeholder_refs(old) and "q301-L1" not in mg_pool.placeholder_refs(old)
    table = {"refs": {"q301-L2": {"tokens": 2587, "actions": 9, "n": 3, "src": {"seed": 3}}}}
    new, rep = mg_pool.apply_refs(old, table)
    nb = {s["name"]: s["ref"] for s in new}
    assert nb["q301-L2"] == {"tokens": 2587, "actions": 9, "baseline": 26, "measured": True, "n": 3, "src": {"seed": 3}}
    assert nb["q301-L1"]["tokens"] == 12000 and nb["q301-L1"]["src"] == "env-table"
    assert nb["q301-L3"] == {"tokens": None, "actions": None, "baseline": 39, "measured": False}
    assert mg_pool.placeholder_refs(new) == [] and "q301-L3" in rep["unmeasured"]
    basis = {"opt": 30, "opt_src": "solution_lengths", "act_per_opt": 1.5, "tok_per_act": 400.0, "scope": "game"}
    new, rep = mg_pool.apply_refs(old, {**table, "derived": {"q301-L3": {"tokens": 18000, "actions": 45.0,
                                                                         "derived": True, "measured": False, "basis": basis}}})
    d3 = {s["name"]: s["ref"] for s in new}["q301-L3"]
    assert d3["tokens"] == 18000 and d3["derived"] and d3["measured"] is False and rep["derived"] == ["q301-L3"]
    assert mg_pool.placeholder_refs(new) == []
    assert mg_pool.placeholder_refs([{"name": "x-L1", "ref": {"tokens": 5, "measured": False}}]) == ["x-L1"]


def test_shipped_refs_table_has_no_placeholders():
    import json
    from pathlib import Path
    table = json.loads((Path(__file__).resolve().parents[1] / "data/envs/mg-refs-v1.json").read_text())
    assert table["refs"] and not set(table["refs"]) & set(table["no_base"])
    for name, r in table["refs"].items():
        assert r["n"] >= 1 and r["tokens"] not in mg_pool.PLACEHOLDER_TOKENS and 0 < r["tokens"] < 262144, name
        assert r["actions"] and r["src"], name
    sets = [{"name": n, "role": "train", "ref": {"tokens": 64000, "actions": None, "baseline": 10},
             "prior": {"tries": 0, "clears": 0}} for n in list(table["refs"]) + list(table["no_base"])]
    new, rep = mg_pool.apply_refs(sets, table)
    assert mg_pool.placeholder_refs(new) == [] and len(rep["from_table"]) == len(table["refs"])
    assert len(rep["derived"]) == len(table["derived"]) and set(table["derived"]) <= set(table["no_base"])
    for name, d in table["derived"].items():
        assert d["derived"] and not d["measured"] and d["tokens"] not in mg_pool.PLACEHOLDER_TOKENS, name
        assert 0 < d["tokens"] < 262144 and d["basis"]["scope"] in ("game", "family"), name
