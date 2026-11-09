"""v8.4: the equal scheduler with parking, on the bundle's patched source
(docs/validation-20261009/v84_equal.py). Run with PYTHONPATH=dist/arena-flash-competition-v84-sub/src."""
import pytest

import arena.scheduler

import test_arena_priority as tp

if "_equal_parked" not in open(arena.scheduler.__file__).read():
    pytest.skip("unpatched arena source (run against the v8.4 bundle's src)", allow_module_level=True)


def test_parked_games_take_any_free_stream_once_no_unparked_game_waits():
    now, wall, games, sched, log = tp.build(count=4, slots=2, equal=True, park_tokens=10_000, floor=True,
                                            tail_fade=False, level_cap_tokens=0)
    g0, g1, g2, g3 = games
    assert tp.active(games) == ["g0", "g1"]
    tp.charge(now, g0, 15_000)  # g0 crosses the park line while running: it keeps its slot until its next drain
    tp.at(now, sched, 1)
    assert tp.active(games) == ["g0", "g1"]
    g0.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert tp.active(games) == ["g1", "g2"]  # unstarted g2 (0 tokens) before parked g0
    tp.charge(now, g1, 5_000)
    g1.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert tp.active(games) == ["g2", "g3"]
    tp.charge(now, g2, 12_000)
    tp.charge(now, g3, 3_000)
    g2.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert tp.active(games) == ["g1", "g3"]  # unparked g1 (5K) before parked g0 (15K); g2 now parked (12K)
    # Only parked games wait: a finished game's stream goes to the parked game with the fewest tokens, at once.
    g3.finished_reason = "won"
    tp.at(now, sched, 2)
    assert tp.active(games) == ["g1", "g2"]
    # A running unparked game hands over at a drain only to an unparked game with fewer tokens: none waits, it keeps.
    tp.charge(now, g1, 1_000)
    g1.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert tp.active(games) == ["g1", "g2"] and sched.priority_counts["handover_kept"] >= 1
    # End of the run: the last unparked game ends; its stream goes to the remaining parked game, nothing idles.
    g1.finished_reason = "won"
    tp.at(now, sched, 3)
    assert tp.active(games) == ["g0", "g2"]
    # Among parked games the one with fewer tokens leads at a handover: g2 (12K) keeps before g0 (15K) gets it back.
    g2.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert tp.active(games) == ["g0", "g2"]
    assert sched.priority_counts.get("equal_parked", 0) >= 2


def test_a_capped_game_sorts_behind_parked_ones_under_equal():
    now, wall, games, sched, log = tp.build(count=3, slots=1, equal=True, park_tokens=10_000, floor=True,
                                            tail_fade=False, level_cap_tokens=0, level_cap_requests=5,
                                            level_cap_grant_requests=2)
    g0, g1, g2 = games
    assert tp.active(games) == ["g0"]
    g0.requests.extend({} for _ in range(5))  # a cheap loop: few tokens, so not parked by tokens
    tp.charge(now, g0, 100)
    g0.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert g0.level_capped and tp.active(games) == ["g1"]
    tp.charge(now, g1, 15_000)  # g1 crosses the park line
    g1.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert tp.active(games) == ["g2"]
    tp.charge(now, g2, 15_000)
    g2.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert tp.active(games) == ["g1"] and g0.state == "parked"  # a parked game before the capped one
    g1.finished_reason = "won"
    g2.finished_reason = "won"
    tp.at(now, sched, 30)
    assert tp.active(games) == ["g0"] and g0.state == "active"  # the capped game returns only to a slot that would idle
    assert g0.level_cap_request_limit == 7


def test_the_token_level_cap_sorts_behind_under_equal_and_a_clear_lifts_it():
    now, wall, games, sched, log = tp.build(count=3, slots=1, equal=True, park_tokens=500_000, floor=True,
                                            tail_fade=False, level_cap_tokens=20_000, level_cap_grant_tokens=10_000,
                                            level_cap_requests=1_000, level_cap_grant_requests=100)
    g0, g1, g2 = games
    tp.charge(now, g0, 25_000)
    g0.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert g0.level_capped and tp.active(games) == ["g1"]
    tp.charge(now, g1, 5_000)
    g1.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert tp.active(games) == ["g2"]  # unstarted g2, then g1 (5K), before the capped g0
    tp.charge(now, g2, 6_000)
    g2.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert tp.active(games) == ["g1"] and g0.state == "parked"
    g0.report_level(1, wall.elapsed())
    assert not g0.level_capped


def test_park_tokens_count_per_level_and_a_clear_unparks():
    now, wall, games, sched, log = tp.build(count=3, slots=1, equal=True, park_tokens=10_000, floor=True,
                                            tail_fade=False, level_cap_tokens=0)
    g0, g1, g2 = games
    tp.charge(now, g0, 8_000)
    g0.report_level(1, wall.elapsed())  # 8K on the first level, cleared
    tp.charge(now, g0, 8_000)  # 16K in the game, 8K on this level: not parked
    g0.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert not sched._equal_parked(g0) and sched.priority_counts.get("equal_parked", 0) == 0
    assert tp.active(games) == ["g1"]
    tp.charge(now, g1, 20_000)
    g1.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert sched._equal_parked(g1) and sched.priority_counts["equal_parked"] == 1
    assert tp.active(games) == ["g2"]
    tp.charge(now, g2, 12_000)
    g2.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert sched._equal_parked(g2) and tp.active(games) == ["g0"]  # g0 unparked, more game tokens than the parked two
    tp.charge(now, g0, 3_000)  # 11K on the level
    g0.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert sched._equal_parked(g0) and sched.priority_counts["equal_parked"] == 3
    assert tp.active(games) == ["g2"]  # all parked: fewest game tokens
    g2.report_level(1, wall.elapsed())
    assert not sched._equal_parked(g2) and not g2.equal_parked_noted
    tp.charge(now, g2, 10_000)
    g2.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert sched.priority_counts["equal_parked"] == 4  # a new level parks and notes again


def test_a_level_at_both_lines_is_capped_only_not_parked_twice():
    now, wall, games, sched, log = tp.build(count=3, slots=1, equal=True, park_tokens=20_000, floor=True,
                                            tail_fade=False, level_cap_tokens=20_000, level_cap_grant_tokens=10_000)
    g0, g1, g2 = games
    tp.charge(now, g0, 25_000)
    g0.note_context_drain(tokens_before=120_000, tokens_after=60_000)
    assert g0.level_capped and not sched._equal_parked(g0) and tp.active(games) == ["g1"]
    assert sched.level_cap_counts["capped"] == 1 and sched.priority_counts.get("equal_parked", 0) == 0
    tp.charge(now, g1, 21_000)
    g1.note_context_drain(tokens_before=120_000, tokens_after=60_000)  # the same line: capped, after g0 by spend
    assert g1.level_capped and tp.active(games) == ["g2"]
    g0.report_level(1, wall.elapsed())  # a clear lifts both
    assert not g0.level_capped and not sched._equal_parked(g0)

