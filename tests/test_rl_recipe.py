"""recipe: the human-anchored reward (mg.reward_ref "human")."""
import math
import types

from arena import rl_loop

BASE, CAP = 20.0, 4000


def fake(log_rhae_min=None, clears=((1000, 20), (2000, 40), (3000, 60))):
    mg = {"reward_ref": "human", "log_clip": 16.0, "log_fail": 32.0, "rhae_cap": 1.15, "cap": 64000,
          "log_rhae_min": log_rhae_min}
    self = types.SimpleNamespace(cfg={"mg": mg}, mg_group_ref={"name": "s", "clears": list(clears)},
                                 mg_level_ref=lambda name, game, level: {"baseline": BASE})
    self.mg_human = types.MethodType(rl_loop.Loop.mg_human, self)
    return self


def row(actions=None, tokens=None):
    return {"point": "ab12-game-L1", "level": 1, "token_budget": CAP,
            "level_rows": [{"level": 1, "actions": actions}] if actions else [], "_t": tokens}


def test_rhae_term_is_against_the_human_baseline_and_tokens_against_the_group_median():
    lp = fake()
    r = lp.mg_human(row(20, 2000), "s", 2000)  # RHAE 1 at the baseline, median tokens: 0
    assert math.isclose(r, 0.0, abs_tol=1e-12)
    r = lp.mg_human(row(10, 1000), "s", 1000)  # RHAE capped at 1.15, half the median tokens
    assert math.isclose(r, math.log(1.15) + math.log(2.0))
    r = lp.mg_human(row(40, 2000), "s", 2000)  # 2x baseline actions: RHAE 0.25
    assert math.isclose(r, math.log(0.25))


def test_fail_is_a_median_actions_clear_at_twice_the_cap_and_clears_never_below():
    lp = fake()
    fail = lp.mg_human(row(), "s", None)
    want = math.log((BASE / 40) ** 2) + math.log(2000 / (2 * CAP))  # median actions 40, median tokens 2,000
    assert math.isclose(fail, want)
    spam = lp.mg_human(row(2000, 3900), "s", 3900)  # 100x baseline actions: raw far below F
    assert spam == fail
    assert lp.mg_human(row(40, CAP), "s", CAP) > fail  # a clear at the median actions within the cap beats F


def test_rhae_floor_knob_off_by_default():
    raw = fake().mg_human(row(2000, 3900), "s", 3900)
    floored = fake(log_rhae_min=32).mg_human(row(2000, 3900), "s", 3900)
    assert floored >= -math.log(32) + math.log(2000 / 3900) - 1e-12 and floored >= raw
    assert fake().cfg["mg"]["log_rhae_min"] is None


def test_group_without_clears_is_flat():
    lp = fake(clears=())
    assert lp.mg_human(row(), "s", None) == -math.log(32)


def test_below_worst_fail_is_log2_under_the_groups_worst_clear():
    """mg.fail_rule "below_worst": a fail scores log(fail_margin) (default 2) below the worst
    clear of its group; every clear stays above it."""
    lp = fake()
    lp.cfg["mg"]["fail_rule"] = "below_worst"
    worst = math.log((BASE / 60) ** 2) + math.log(2000 / 3000)  # the (3000 tokens, 60 actions) clear
    fail = lp.mg_human(row(), "s", None)
    assert math.isclose(fail, worst - math.log(2.0))
    assert all(lp.mg_human(row(a, t), "s", t) > fail for t, a in ((1000, 20), (2000, 40), (3000, 60)))
    lp.cfg["mg"]["fail_margin"] = 4.0
    assert math.isclose(lp.mg_human(row(), "s", None), worst - math.log(4.0))
