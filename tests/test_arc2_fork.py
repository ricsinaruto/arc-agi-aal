"""arc2.fork: reopen variants fork from a base run's first passes."""
import asyncio

from test_arc2_conv import TASK, _close, call, engine

from arc2 import fork, tasks
from arc2.loop import TaskRun, guesses


def _base():
    """Two entries: a (S1 then a pinned S2, one program before S1) and b (never submitted)."""
    a = {"task": "t#1", "spent": 9000, "candidates": [
        {"code": "p1", "program": 1, "tokens_at": 1000, "t": 5.0, "tests": [[[9]]], "verified": False, "passed": 0, "pairs": 2, "accuracy": 0.0},
        {"code": "s1a", "submitted": 1.5, "pinned": True, "submit_n": 2, "submit_at": 1, "tokens_at": 2000, "t": 6.0,
         "tests": [[[7, 8, 9]]], "clock": 1.0, "verified": False, "passed": 0, "pairs": 2, "accuracy": 0.0},
        {"code": "s1b", "submitted": 2, "submit_n": 1, "submit_at": 1, "tokens_at": 2000, "t": 6.0, "tests": [[[1]]], "verified": False, "passed": 0, "pairs": 2, "accuracy": 0.0},
        {"code": "s2a", "submitted": 1, "submit_n": 2, "submit_at": 2, "tokens_at": 8000, "t": 9.0, "tests": [[[2]]], "verified": False, "passed": 0, "pairs": 2, "accuracy": 0.0},
        {"code": "p2", "program": 2, "tokens_at": 7000, "t": 8.0, "tests": [[[3]]], "verified": False, "passed": 0, "pairs": 2, "accuracy": 0.0}]}
    b = {"task": "t#2", "spent": 5000, "candidates": [{"code": "x", "tokens_at": 100, "t": 1.0, "tests": [[[4]]], "verified": False, "passed": 0, "pairs": 2, "accuracy": 0.0}]}
    events = [{"event": "reply", "task": "t#1", "program": 1, "code": "def transform(g):\n    return [[9]]",
               "tokens_at": 1000},
              {"event": "reply", "task": "t#1", "program": 2, "code": "def transform(g):\n    return [[3]]",
               "tokens_at": 7000}]
    return [a, b], events


def test_snapshot_keeps_submission_one_and_what_came_before():
    rows, events = _base()
    snap = fork.snapshot(rows, events)
    a, b = snap["entries"]["t#1"], snap["entries"]["t#2"]
    assert snap["first_pass_tokens"] == 2000 + 5000
    assert a["submitted"] and a["first_submit_tokens"] == a["spent"] == 2000
    ranks = {c["code"]: c["submitted"] for c in a["candidates"] if c.get("submitted")}
    assert ranks == {"s1a": 1, "s1b": 2}  # the pin undone; S2 and the program after S1 dropped
    assert [c["code"] for c in a["candidates"] if not c.get("submitted")] == ["p1"]
    assert a["programs"] == ["def transform(g):\n    return [[9]]"]
    assert all(c["t"] == fork.FORK_T and "clock" not in c for c in a["candidates"])
    assert not b["submitted"] and b["spent"] == 5000 and b["candidates"][0]["tests"] == [[[4]]]


def test_apply_starts_entries_at_their_first_reopen_and_the_pass_plays():
    rows, events = _base()
    snap = fork.snapshot(rows, events)
    a, b = TaskRun("t#1", TASK, {"conversation": True}), TaskRun("t#2", TASK, {"conversation": True})
    eng, sent = engine([call("submit([[2, 3]])")], submit_mode="fresh", fresh_each=True, fresh_text="naive",
                       keep_first=True, reopen_order="first_submit")
    assert fork.apply(eng, [a, b], snap) == 7000
    assert b.done == "fork_base" and b.spent == 5000
    assert a.awaiting_reopen and not a.parked and a.submits == 1 and a.spent == 2000 and a.done is None
    assert eng.queue_item(a, 0) == ((1, 0), -1)  # a reopen token, not a backlog entry
    assert eng.pick_reopen([a, b]) == 0
    first = a.messages[1]["content"]
    assert tasks.CONV_NAIVE in first and "7 8 9" in first  # the naive pass shows S1's answer
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))  # no hand reset: the pass plays from the fork state
    inputs = [t["input"] for t in TASK["test"]]
    assert a.submits == 2 and [g for g, _ in guesses(a.candidates, inputs)[0]] == [[[2, 3]], [[7, 8, 9]]]
    _close(a)


def test_a_forked_pass_keeps_its_slot_after_a_reply_without_submit():
    """fk-naive1 (29 Oct): apply() left ``parked`` set, so every pass broke after one reply and starved."""
    from test_arc2_conv import WRONG
    rows, events = _base()
    snap = fork.snapshot(rows, events)
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([WRONG, call("submit([[5]])")], submit_mode="fresh", fresh_each=True, fresh_text="naive",
                    keep_first=True)
    fork.apply(eng, [a], snap)
    assert eng.pick_reopen([a]) == 0
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    assert not a.parked and a.submits == 1 and not a.awaiting_reopen  # the worker holds the slot and goes on
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    assert a.submits == 2 and a.parked and a.awaiting_reopen
    _close(a)


def test_midpass_first_ranks_a_pass_that_left_its_slot_before_new_reopen_tokens():
    a = TaskRun("t#1", TASK, {"conversation": True})
    a.submits, a.reopened, a.awaiting_reopen, a.spent = 1, True, False, 50000
    off, _ = engine([], submit_mode="fresh", fresh_each=True, midpass_first=False)
    on, _ = engine([], submit_mode="fresh", fresh_each=True, midpass_first=True)
    assert off.priority(a) == (1, 50000) and on.priority(a) == (1, -1)
    assert on.queue_item(a, 3) < ((1, 0), -1)  # ahead of a reopen token
    a.awaiting_reopen = True
    assert on.queue_item(a, 3) == ((1, 0), -1)


def test_fork_b2_closes_submitted_entries_and_attempts_start_unsubmitted_first():
    """fk-b2: close mode ends a submitted entry at its first submit (no reopen pass); a never-submitted base entry
    stays ended unplayed; once applied, every entry counts as started, so fresh attempts fill the slots, the
    never-submitted entry first (``multi_ns_first``), and an attempt's submit merges into its entry."""
    rows, events = _base()
    rows[0]["candidates"][1]["tokens_at"] = rows[0]["candidates"][2]["tokens_at"] = 9000  # ttfs 9000 > b's 5000
    snap = fork.snapshot(rows, events)
    a, b = TaskRun("t#1", TASK, {"conversation": True}), TaskRun("t#2", TASK, {"conversation": True})
    eng, _ = engine([call("submit([[2, 3]])")], submit_check=True, multi_attempt=True, multi_ns_first=True)
    fork.apply(eng, [a, b], snap)
    assert a.done == "submitted" and a.submits == 1 and a.first_submit_tokens == 9000 and a.messages is None
    assert b.done == "fork_base" and not getattr(b, "awaiting_reopen", False)
    extra = eng.attempts_due([a, b], 1)
    assert [x.id for x in extra] == ["t#b2x2"] and extra[0].arm["attempt_k"] == 2 and extra[0].entry is b
    assert eng.attempts_due([a, b], 1)[0].entry is a  # b's attempt runs: a is next
    eng2, _ = engine([call("submit([[2, 3]])")], submit_check=True, multi_attempt=True)
    a2, b2 = TaskRun("t#1", TASK, {"conversation": True}), TaskRun("t#2", TASK, {"conversation": True})
    fork.apply(eng2, [a2, b2], snap)
    assert eng2.attempts_due([a2, b2], 1)[0].entry is a2  # one scale: ttfs 9000 > spent 5000


def test_fork_b2_attempt_on_a_never_submitted_entry_plays_and_merges():
    rows, events = _base()
    snap = fork.snapshot(rows, events)
    a, b = TaskRun("t#1", TASK, {"conversation": True}), TaskRun("t#2", TASK, {"conversation": True})
    eng, _ = engine([call("submit([[2, 3]])")], multi_attempt=True, multi_ns_first=True)
    fork.apply(eng, [a, b], snap)
    (att,) = eng.attempts_due([a, b], 1)
    asyncio.run(eng.advance(att, 10 ** 9, replies=1))
    inputs = [t["input"] for t in TASK["test"]]
    assert att.done == "submitted" and b.merge_gen == 1
    assert [g for g, _ in guesses(b.candidates, inputs)[0]][0] == [[2, 3]]
    _close(att)


ROUNDS3 = {"submit_mode": "provisional", "firm": True, "open_order": "rounds", "reopen_order": "rounds",
           "swap": "budget", "budget_floor": True, "compact_on_swap": True, "compact_over": 65536,
           "same_answer_swap": False, "force_submit_after": 0, "same_answer_scope": "round", "budget_context": True,
           "compacted_first": True, "no_repeat": True, "round_prompt": True, "keep_first": True}


def test_fork_rounds_enters_round_two_with_the_round_prompt_and_ns_entries_play_on():
    """fk-rounds: a submitted entry re-enters the backlog at round 2 in rounds order (a new conversation opened by the
    round prompt with its earlier answers; budget = tokens to first submit, floor one reply); with ``fork_ns_play`` a
    never-submitted entry plays on from a new conversation, ahead of every submitted one, keeping its base tokens."""
    rows, events = _base()
    snap = fork.snapshot(rows, events)
    a, b = TaskRun("t#1", TASK, {"conversation": True}), TaskRun("t#2", TASK, {"conversation": True})
    eng, sent = engine([call("submit([[7, 8, 9]])"), call("submit([[2, 3]])"), call("submit([[4, 4]])")],
                       fork_ns_play=True, **ROUNDS3)
    fork.apply(eng, [a, b], snap)
    assert a.round == 2 and a.submits == 1 and not a.parked and not a.awaiting_reopen and a.done is None
    assert b.done is None and b.spent == 5000 and b.messages is None
    assert eng.queue_item(b, 1) < eng.queue_item(a, 0)  # never submitted first
    assert eng.queue_item(a, 0)[0][:2] == (1, 2)
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))  # the round prompt opens round 2; the repeat is refused
    assert "7 8 9" in sent[0]["messages"][2]["content"] and a.submits == 1 and a.repeats_refused == 1
    assert eng.round_budget(a, {**eng.settings, **a.arm}) == eng.cap  # ttfs 2000 under the one-reply floor
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    assert a.submits == 2
    asyncio.run(eng.advance(b, 10 ** 9, replies=1))
    assert b.submits == 1 and b.spent > 5000 and b.first_submit_tokens == b.spent
    _close(a)
    _close(b)
