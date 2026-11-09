"""arc2.multi (b2): backlog order and N-attempt merge, shared by the RTX and 4xL4 harnesses."""
from arc2 import multi


def a(k, running=False, submitted=True, ttfs=None, spent=0):
    return {"k": k, "running": running, "submitted": submitted, "first_submit_tokens": ttfs, "spent": spent}


def test_pick_one_scale_ttfs_or_spent_longest_first_and_none_with_an_attempt_running():
    """one key for every entry: tokens to first submit (most recent attempt), or tokens spent so far
    while nothing has submitted; longest first. An unsubmitted entry at 50K ranks below a submitted one at 60K and
    overtakes it past 60K."""
    entries = [[a(1, ttfs=60000)], [a(1, running=True, submitted=False, spent=50000)]]
    assert multi.pick(entries) == 0
    entries[1] = [a(1, running=True, submitted=False, spent=61000)]
    assert multi.pick(entries) == 1
    entries = [
        [a(1, ttfs=30000)],
        [a(1, ttfs=90000)],
        [a(1, ttfs=200000), a(2, running=True, submitted=False)],   # its fresh attempt still runs: not in the backlog
    ]
    assert multi.pick(entries) == 1
    entries[1] = [a(1, ttfs=90000), a(2, ttfs=10000)]               # re-entry ranks by its most recent attempt
    assert multi.pick(entries) == 0
    assert multi.pick([[a(1, running=True, submitted=False, spent=0), a(2, running=True, submitted=False)]]) is None


def test_merge_ranks_by_verified_then_votes_then_earliest_and_two_attempts_is_b1s_rule():
    A, B, C = [[1]], [[2]], [[3]]
    # two attempts, differ, neither verified: attempt 1 first (b1); the verified one first otherwise
    assert multi.merge([{"tests": [A]}, {"tests": [B]}], 1) == [[A, B]]
    assert multi.merge([{"tests": [A]}, {"tests": [B], "verified": True}], 1) == [[B, A]]
    assert multi.merge([{"tests": [A]}, {"tests": [A]}], 1) == [[A]]  # agree: kept
    # three attempts: two agree on B -> B first, A (earliest of the rest) second
    assert multi.merge([{"tests": [A]}, {"tests": [B]}, {"tests": [B]}], 1) == [[B, A]]
    # verified beats votes; None (no answer) is skipped; per output
    got = multi.merge([{"tests": [A, C]}, {"tests": [B, C]}, {"tests": [B, None]}, {"tests": None},
                       {"tests": [C, A], "verified": True}], 2)
    assert got == [[C, B], [A, C]]


def test_pick_ns_first_puts_unsubmitted_entries_above_every_submitted_one():
    """fk-b2: with ``ns_first`` an entry no attempt has submitted ranks first even
    when a submitted entry's tokens to first submit are larger; off, one scale."""
    entries = [[a(1, ttfs=188000)], [a(1, submitted=False, spent=156000)]]
    assert multi.pick(entries) == 0 and multi.pick(entries, ns_first=True) == 1
    entries[1] = [a(1, submitted=False, spent=156000), a(2, ttfs=20000)]  # its fresh attempt submitted: one scale
    assert multi.pick(entries, ns_first=True) == 0
