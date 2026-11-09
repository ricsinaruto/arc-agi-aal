"""train_dash.play_progress_fig: groups done per play phase, reset at each update, phase ends marked."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/ops"))
import train_dash as T  # noqa: E402


def test_play_progress_resets_at_update_and_marks_it():
    rows = [{"t": 0, "m": {"play.live_groups_done": 1}}, {"t": 600, "m": {"play.live_groups_done": 2}},
            {"t": 900, "m": {"train.step": 1, "play.groups": 2}}, {"t": 1200, "m": {"play.live_groups_done": 1}}]
    clock = {r["t"]: r["t"] / 3600 for r in rows}
    f = T.play_progress_fig(rows, clock)
    line = f["traces"][0]
    assert line["y"] == [1, 2, None, 0, 1, 1] and line["line"]["shape"] == "hv"
    marks = [t for t in f["traces"] if t["name"] == "update (phase end)"]
    assert len(marks) == 1 and marks[0]["x"] == [0.25, 0.25] and f["traces"][1]["y"] == [16, 16]
    assert T.play_progress_fig([{"t": 0, "m": {"infra.running": 3}}], {0: 0.0}) is None


def test_leg_kd_fig_shows_only_the_runs_own_steps():
    own = 1000
    rows = [{"t": 900, "m": {"train.step": 9, "train.kd": 0.5}, "_leg": "parent"},
            {"t": 1001, "m": {"train.step": 10, "train.kd": 0.5}, "_leg": "parent"},  # a parent's late row
            {"t": 1002, "m": {"train.step": 11}, "_leg": "kid"}]
    rows += [{"t": 1010 + s, "m": {"train.step": s, "train.distill_kl" if s == 16 else "train.kd": 0.1 + s / 1000}, "_leg": "kid"} for s in range(11, 17)]
    rows += [{"t": 1100, "m": {"train.step": 12, "train.kd": 0.2}, "_leg": "kid"}]  # a redone step: latest value wins
    f = T.leg_kd_fig(rows, "kid", own)
    raw, mean = f["traces"]
    assert raw["x"] == [11, 12, 13, 14, 15, 16] and raw["y"][1] == 0.2 and raw["y"][5] == 0.116
    assert raw["opacity"] < 1 and "opacity" not in mean and mean["x"] == raw["x"]
    assert f["rng"] is None and f["xr"] == [11, 16]


def test_leg_kd_fig_keeps_restarts_of_the_leg_and_marks_them():
    rows = [{"t": 1, "m": {"train.step": 5, "train.kd": 0.9}, "_leg": "x-v0"}]
    rows += [{"t": 10 + s, "m": {"train.step": s, "train.kd": 0.5}, "_leg": "x-v1"} for s in range(6, 12)]
    rows += [{"t": 100 + s, "m": {"train.step": s, "train.kd": 0.4}, "_leg": "x-v1b"} for s in range(11, 15)]
    f = T.leg_kd_fig(rows, "x-v1b", 111)
    raw = f["traces"][0]
    assert raw["x"] == list(range(6, 15)) and raw["y"][5] == 0.4  # the redone step 11 keeps the restart's value; x-v0 stays out
    assert [t["x"] for t in f["traces"] if t["name"].startswith("restart")] == [[11, 11]]


def test_leg_kd_fig_without_a_chain_uses_every_step_and_few_steps_stay_solid():
    rows = [{"t": s, "m": {"train.step": s, "train.kd": 0.1}} for s in (1, 2)]
    f = T.leg_kd_fig(rows, "solo", None)
    assert len(f["traces"]) == 1 and f["traces"][0]["opacity"] == 1 and f["traces"][0]["x"] == [1, 2]
    assert T.leg_kd_fig([{"t": 1, "m": {"train.step": 1}}], "solo", None) is None


def test_a_rolled_back_update_is_abandoned_before_the_new_branch_reaches_it():
    starts = [(None, None), (500, 18)]
    d = {(0, 17): (1, 0.1), (0, 18): (2, 0.2), (0, 21): (3, 0.3), (1, 18): (4, 0.4)}
    live, gone = T.split_branches(d, starts)
    assert [(u, v) for u, v, _ in live] == [(17, 0.1), (18, 0.4)]
    assert [(u, v) for u, v, _ in gone[0]] == [(18, 0.2), (21, 0.3)]


def test_vs_control_uses_live_pairs_only():
    starts = [(None, None), (500, 18)]
    uobs = {"train.control_diff": {(0, 21): (3, -1.64285), (1, 25): (9, -0.067522)},
            "train.vs_control": {(0, 21): (3, 0.1934), (1, 25): (9, 0.4252)}}
    T.live_control(uobs, starts)
    assert uobs["train.vs_control"][(1, 25)] == (9, 0.9347)  # sc25-L1 alone, not the mean with the abandoned 0.19
    assert uobs["train.vs_control"][(0, 21)] == (3, 0.1934)  # the faded point keeps its value
