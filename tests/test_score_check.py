import json
import math

from pathlib import Path

import pytest

from arena import kv_golden, score_check as sc


def dump(values):
    return {"probes": [{"label": "p", "logprobs": list(values)}]}


def dumps(move_t, move_s, gap=0.0, reload=0.0, noise=0.01):
    """Per token: trainer base b, trainer policy b + m, server the same plus ``gap`` and a scaled move."""
    n = len(move_t)
    base = [-1.0 - 0.01 * i for i in range(n)]
    wobble = [noise * (-1) ** i for i in range(n)]
    t = [b + m for b, m in zip(base, move_t)]
    s = [b + m + gap + 0.1 for b, m in zip(base, move_s)]  # the server's constant offset: the base gap
    served = [b + 0.1 for b in base]
    return {"s_policy": dump(s), "s_policy2": dump(a + w for a, w in zip(s, wobble)), "s_base": dump(served), "s_base2": dump(served),
            "s_policy_together": dump(s), "s_policy_reload": dump(a + reload * (-1) ** i + w for i, (a, w) in enumerate(zip(s, wobble))),
            "s_base_reload": dump(served), "t_policy": dump(t), "t_policy2": dump(a + w for a, w in zip(t, wobble)), "t_base": dump(base)}


MOVE = [0.5 * ((i % 7) - 3) for i in range(70)]


def test_updates_reach_play():
    got = sc.summary(dumps(MOVE, MOVE))
    assert got["verdict"] == "reach" and got["uncertain"]["slope"] == pytest.approx(1.0)
    assert got["uncertain"]["r"] == pytest.approx(1.0)


def test_server_that_does_not_move_is_a_bug():
    still = [0.001 * (-1) ** i for i in range(70)]
    assert "move" in sc.summary(dumps(MOVE, still))["verdict"]

def test_half_move_and_gap_are_bugs():
    half = [m * 0.3 for m in MOVE]
    assert "move" in sc.summary(dumps(MOVE, half))["verdict"]
    d = dumps(MOVE, MOVE)
    d["s_policy"] = dump(v + 1.0 for v in d["s_policy"]["probes"][0]["logprobs"])
    assert "gap" in sc.summary(d)["verdict"]


def test_reload_floor_is_a_bug_and_small_moves_say_so():
    assert "reload" in sc.summary(dumps(MOVE, MOVE, reload=0.5))["verdict"]
    tiny = [m * 0.01 for m in MOVE]
    assert sc.summary(dumps(tiny, tiny))["verdict"] == "move_too_small"


def test_tokens_with_nan_and_short_dumps_drop():
    d = dumps(MOVE, MOVE)
    d["t_policy"] = dump(d["t_policy"]["probes"][0]["logprobs"][:60])
    d["s_base"]["probes"][0]["logprobs"][0] = math.nan
    got = sc.summary(d, {"p": [-1.0] * 70})
    assert got["dropped"] == 11 and got["all"]["tokens"] == 59 and got["all"]["engine_vs_scored"]["tokens"] == 59


def test_trainer_row_takes_the_probes_reply():
    probes = [{"label": "a", "request": 4, "tokens": [1, 2]}, {"label": "b", "request": 9, "tokens": [3]}]
    got = sc.trainer_row({"a": {"replies": [{"request": 2, "start": 1, "raw": [0.0]}, {"request": 4, "start": 9, "raw": [-1.0, -2.0]}]}}, probes)
    assert got["probes"][0]["logprobs"] == [-1.0, -2.0] and got["probes"][1]["logprobs"] == []


def test_probes_carry_the_rollout_and_request(tmp_path):
    path = tmp_path / "a0" / "rollout.jsonl"
    path.parent.mkdir()
    rows = [{"n": i, "gameplay": True, "keep": 0, "messages": [{"role": "user", "content": str(i)}], "template": {},
             "tokens": [i, i + 1], "logprobs": [-0.5, -0.25], "usage": {"prompt_tokens": 100 * (i + 1)}} for i in range(3)]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    got = sc.probes_of({"rollouts": [str(path)], "depths": [250], "per_depth": 1})
    assert len(got) == 1 and got[0]["label"].startswith("00-") and got[0]["request"] == 1
    assert got[0]["engine"] == [-0.5, -0.25] and got[0]["rollout"] == str(path)


def test_each_probe_names_its_own_rollout(tmp_path):
    paths = []
    for name, size in (("a0", 100), ("a1", 200)):
        path = tmp_path / name / "rollout.jsonl"
        path.parent.mkdir()
        rows = [{"n": i, "gameplay": True, "keep": 0, "messages": [{"role": "user", "content": str(i)}], "template": {},
                 "tokens": [i, i + 1], "logprobs": [-0.5, -0.25], "usage": {"prompt_tokens": size * (i + 1)}} for i in range(3)]
        path.write_text("\n".join(json.dumps(r) for r in rows))
        paths.append(path)
    got = sc.probes_of({"rollouts": [str(p) for p in paths], "depths": [150, 450], "per_depth": 1})
    assert {p["label"].split("-")[1]: Path(p["rollout"]).parent.name for p in got} == {"a0": "a0", "a1": "a1"}


def test_settings_need_rollouts_and_depths():
    assert sc.settings({}) is None
    with pytest.raises(ValueError):
        sc.settings({"score_check": {"rollouts": []}})
