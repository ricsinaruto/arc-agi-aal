"""Harness-verified negatives (``arena.negatives``; harness_fixes ``negative_check``): claims read from a reply, mapped
to the level's record, answered once, quiet when the record supports the claim; generic text; off by default."""

from arena import harness_fixes as hf
from arena import negatives as ng

KEYS = ("UP", "DOWN", "SPACE", "MOUSE", "UNDO")


def test_claims_are_read_with_their_targets():
    text = ("The slot at (22,34) is not clickable. UP does nothing here, so I go down the other side.\n"
            "It cannot be filled at all. With that the level is unsolvable. Moving up the ladder works.")
    assert ng.claims(text, KEYS) == [("cell", 22, 34), ("key", "UP"), ("recent",), ("level",)]
    assert ng.claims("Click row 5, col 60: no effect.", KEYS) == [("cell", 5, 60)]
    assert ng.claims("I placed it at (3,4) and it works; nothing is blocked, and UP is not refused.", KEYS) == []
    assert ng.claims("That is impossible in one call, so two calls.", KEYS) == []
    assert ng.claims("The door is blocked by the wall; then SPACE opens it.", KEYS) == [("recent",)]  # SPACE after it
    message = {"reasoning": "ok", "tool_calls": [{"function": {"arguments": '{"code": "playbook(\\"UNDO has no effect\\")"}'}}]}
    assert ng.claims(ng.reply_text(message), KEYS) == [("key", "UNDO")]
    assert ng.claims(ng.reply_text({"reasoning": "ok", "tool_calls": [{"function": {"arguments": '{"code": "x = \\"UNDO has no effect\\""}'}}]}), KEYS) == []


def test_the_record_answers_once_and_stays_quiet_when_it_supports_the_claim():
    record, answered = ng.Record(), set()
    record.executed(1, "MOUSE(row=22, col=34)", "a", False)
    for _ in range(3):
        record.refused(1, "MOUSE(row=23, col=35)")
    text, details = ng.respond(record, 1, [("cell", 22, 34)], answered, KEYS)
    assert text.split("\n") == [
        ng.HEAD,
        "- MOUSE at or within 2 cells of row 22, col 34: executed once on this level, in 1 distinct board state; 0 of "
        "those changed the board; 3 further attempts were refused by the stale-state guard and never ran.",
        ng.TAIL]
    assert details == [{"claim": ["cell", 22, 34], "n": 1, "boards": 1, "changed": 0, "refused": 3}]
    assert ng.respond(record, 1, [("cell", 22, 34), ("cell", 23, 35)], answered, KEYS)[0] is None  # said already
    # A second level starts clean; a tested negative (twice, two boards, no change) is left alone.
    record.executed(2, "UP", "a", False)
    record.executed(2, "UP", "b", False)
    text, details = ng.respond(record, 2, [("key", "UP")], answered, KEYS)
    assert text is None and details[0]["quiet"] is True
    record.executed(2, "DOWN", "a", False)
    record.executed(2, "DOWN", "a", False)  # twice on the same board: one test
    assert "DOWN: executed 2 times on this level, in 1 distinct board state; 0 of those" in ng.respond(
        record, 2, [("key", "DOWN")], answered, KEYS)[0]
    assert "- SPACE: never executed on this level." in ng.respond(record, 2, [("key", "SPACE")], answered, KEYS)[0]


def test_the_untargeted_negative_and_the_verdict_use_the_record():
    record, answered = ng.Record(), set()
    record.mark()
    record.executed(0, "MOUSE(row=10, col=10)", "a", True)
    record.executed(0, "MOUSE(row=40, col=40)", "b", False)
    record.refused(0, "UNDO")
    record.mark()  # the reply after that batch
    text, _ = ng.respond(record, 0, [("recent",), ("level",)], answered, KEYS)
    lines = text.split("\n")
    assert lines[1].startswith("- MOUSE at or within 2 cells of row 40, col 40: executed once")
    assert lines[2] == "- UNDO: never executed on this level; 1 further attempt was refused by the stale-state guard and never ran."
    assert lines[3] == ("- This level so far: 2 actions executed over 2 distinct actions or cells; 1 of those changed the "
                        "board at least once, 2 were executed only once; 1 attempts were refused by the stale-state guard "
                        "and never ran; offered and never executed on this level: UP, DOWN, SPACE, UNDO.")
    assert ng.respond(record, 0, [("level",)], answered, KEYS)[0] is None  # the verdict once per level
    assert ng.respond(ng.Record(), 0, [("level",)], set(), KEYS)[0] is None  # nothing played: nothing to say


def test_generic_text_and_off_by_default():
    assert "negative_check" in hf.OPTIONAL_FIXES and "negative_check" not in hf.settings(True)["fixes"]
    for text in (ng.HEAD, ng.TAIL, ng.NEGATIVE.pattern, ng.VERDICT.pattern):
        assert not any(word in text.lower() for word in ("pipe", "slot ", "palette", "sb26", "box"))
