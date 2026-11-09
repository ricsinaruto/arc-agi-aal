import json

from arena import rl_loop
from arena.serving import lora_flags, validate_lora


def test_reward_and_leave_one_out():
    assert rl_loop.attempt_reward({"level_rows": [{"score": 100.0}, {"score": 130.0}]}) == 2.15
    assert rl_loop.attempt_reward({"level_rows": []}) == 0.0
    assert [round(v, 6) for v in rl_loop.leave_one_out([0.0, 1.0, 2.0, 3.0])] == [-2.0, -0.666667, 0.666667, 2.0]


def test_parity_detects_a_scaled_shift():
    base = [[-1.0, -2.0, -0.5, -3.0]]
    shifted = [[-0.9, -2.2, -0.4, -2.5]]
    check = rl_loop.parity(base, shifted, base, shifted)
    assert check["shift_corr"] == 1.0 and check["shift_slope"] == 1.0 and check["base_mean_abs"] == 0.0
    swapped = rl_loop.parity(base, [[-1.1, -1.8, -0.6, -3.5]], base, shifted)
    assert swapped["shift_corr"] < 0


def test_stop_rules(tmp_path):
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.cfg = rl_loop.settings({"mode": "loop"})
    loop.base = {"reward_mean": 2.3, "eval": {"lp85": {"mean": 2.2, "se": 0.17}}}
    loop.history = [{"round": 0, "stats": {"reward_mean": 2.3, "tokens_per_reply": 800, "actions_per_level": 40,
                                           "capped_share": 0.01}}]
    loop.started = __import__("time").time()
    row = {"stats": {"reward_mean": 2.5, "tokens_per_reply": 2000, "actions_per_level": 40, "capped_share": 0.01},
           "update": {"windows": [{"mismatch": {"tokens": 100, "mean_abs": 0.2, "mean": 0.0}}]}}
    assert loop.check_stop(row)[0] == "mismatch"
    row["update"]["windows"][0]["mismatch"]["mean_abs"] = 0.01
    stop, alerts = loop.check_stop(row)
    assert stop is None and any("tokens_per_reply" in a for a in alerts)
    loop.history.append({"stats": {"reward_mean": 4.6}})
    row["stats"]["reward_mean"] = 4.6
    loop.history.append(row)
    assert loop.check_stop(row)[0] == "pass_reached"
    row2 = {"stats": {"reward_mean": 2.0}, "eval": {"lp85": {"mean": 1.4, "se": 0.17}}}
    loop.history = loop.history[:1]
    assert loop.check_stop(row2)[0] == "untrained_eval_down"


def test_lora_flags():
    lora = validate_lora({})
    argv = lora_flags(["python", "serve", "m", "--moe-backend", "humming"], lora)
    assert argv[argv.index("--moe-backend") + 1] == "marlin"
    assert "--enable-mixed-moe-lora-format" in argv and "--enable-sleep-mode" in argv
    assert argv[argv.index("--max-lora-rank") + 1] == "32"
    try:
        lora_flags(["python", "--speculative-config", json.dumps({"method": "mtp"})], lora)
    except ValueError:
        pass
    else:
        raise AssertionError("LoRA with MTP must be refused")
