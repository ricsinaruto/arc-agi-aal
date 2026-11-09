"""v7 async loop core (arena.async_rl): pinning, staleness drops, resume dedupe, scheduler throttle and ramp, reads,
the card-2 controller and the per-group balance."""
import json

import pytest

from arena.async_rl import (
    Book,
    Buffer,
    Card2,
    Group,
    Reads,
    Scheduler,
    balance_scale,
    ess,
    lag_hist,
)


def group(gid, version, size=3, unit="u"):
    return Group(gid=gid, unit=unit, start=["a", 1], version=version, size=size, created=0.0)


def test_group_ready_only_when_every_member_ends():
    g = group("g1", 0)
    assert not g.member_done(0, {"reward": 1.0}, 1.0)
    assert not g.member_done(0, {"reward": 1.0}, 2.0)  # the same member again does not count twice
    assert not g.member_done(1, {"reward": 0.0}, 3.0)
    assert g.member_done(2, {"reward": 0.5}, 4.0)
    assert g.status == "ready" and g.finished == 4.0


def test_book_serve_every_users_and_slots():
    book = Book(max_loras=3, serve_every=2)
    assert not book.publish(1) and book.served == 0
    assert book.publish(2) and book.served == 2
    book.acquire(0, 10)
    book.acquire(2, 10)
    assert book.live() == [0, 2] and book.slots_ok()
    book.release(0, 10)
    assert book.unloadable([0, 2]) == [0]
    with pytest.raises(ValueError):
        book.release(2, 11)
    with pytest.raises(ValueError):
        book.publish(2)
    assert book.lag(0) == 2 and book.lag(9) is None


def test_book_waits_for_a_slot_before_serving():
    book = Book(max_loras=3, serve_every=2)
    book.acquire(0, 10)
    assert book.publish(2) and book.served == 2
    book.acquire(2, 10)
    book.publish(3)
    assert book.publish(4) and book.served == 4  # live {0, 2, 4}
    book.acquire(4, 10)
    book.publish(5)
    assert not book.publish(6) and book.served == 4 and book.pending == 6  # {0, 2, 4} + 6 is over 3
    assert book.take_pending() is None and book.slots_ok()
    restored = Book.load_state(book.state_dict())
    assert restored.pending == 6
    book.release(0, 10)
    assert book.take_pending() == 6 and book.served == 6 and book.pending is None
    assert book.live() == [2, 4, 6] and book.slots_ok()
    book.rollback(4)
    assert book.pending is None and book.served == 4
    with pytest.raises(ValueError):
        Book(max_loras=1)


def test_book_rollback_drops_branch_from_lineage():
    book = Book(max_loras=8)
    for v in (1, 2, 3):
        book.publish(v)
    book.rollback(1)
    assert book.lineage == [0, 1] and book.current == 1 and book.lag(3) is None
    book.publish(2)  # the new branch reuses the number after the restored state
    assert book.lineage == [0, 1, 2]


def test_buffer_drops_stale_and_off_lineage_groups_whole():
    book = Book(max_loras=12)
    buf = Buffer(s_max=2, batch_groups=4)
    for gid, v in (("old", 0), ("ok", 1)):
        buf.add(group(gid, v))
    for v in (1, 2, 3):
        book.publish(v)
    g, lag = buf.next_group(book)
    assert g.gid == "ok" and lag == 2
    assert buf.closed == {"old": "dropped"} and buf.drops[0]["lag"] == 3
    assert buf.next_group(book) == (None, None)


def test_buffer_resume_dedupe_round_trip():
    book = Book(max_loras=12)
    buf = Buffer(s_max=8, batch_groups=4)
    buf.add(group("a", 0))
    buf.add(group("b", 0))
    g, _ = buf.next_group(book)
    buf.close(g, "consumed")
    state = json.loads(json.dumps({"buffer": buf.state_dict(), "book": book.state_dict()}))
    again = Buffer.load_state(state["buffer"])
    assert not again.add(group("a", 0))  # consumed before the crash: never trained twice
    assert not again.add(group("b", 0))  # still waiting in the restored buffer
    assert [x.gid for x in again.ready] == ["b"]
    assert Book.load_state(state["book"]).lineage == [0]


def test_scheduler_ramp_throttle_and_queue():
    s = Scheduler(group_size=10, max_ready=8, ramp_s=90, in_flight_target=110)
    assert s.want(0, queued=0, in_flight=0, ready_groups=0) == 1
    s.made(0)
    assert s.want(30, queued=0, in_flight=10, ready_groups=0) == 0  # ramp: one group per 90 s
    assert s.want(91, queued=0, in_flight=10, ready_groups=0) == 1
    assert s.want(200, queued=0, in_flight=110, ready_groups=0) == 1 and s.ramped
    assert s.want(201, queued=12, in_flight=110, ready_groups=0) == 0  # a group's tries already wait
    assert s.want(202, queued=0, in_flight=110, ready_groups=8) == 0  # bounded queue
    assert s.want(203, queued=0, in_flight=120, ready_groups=0) == 0  # hard in-flight cap, whatever the pool reports
    s.queue_groups = 2
    assert s.want(204, queued=0, in_flight=120, ready_groups=0) == 1


def test_reads_due_and_land():
    r = Reads(every=8)
    assert not r.due(0) and not r.due(7) and r.due(8)
    r.start(8, version=8, tries=2)
    assert not r.due(8)
    assert r.try_done(8) is None
    assert r.try_done(8)["version"] == 8 and r.open == {}
    assert Reads.load_state(json.loads(json.dumps(r.state_dict()))).every == 8


def test_card2_plays_only_with_slack_and_ratio():
    c = Card2(min_block_s=360, max_block_s=1200, ratio_on=1.3, horizon_s=1800, switch_s=40)
    assert c.decide(0, 0, []) == ("train", 0.0)  # no trainer rate yet
    c.observe_step(40, 480)  # 5.0 tries/min
    for i in range(10):  # server 1: one group of 10 per 3 min = 3.33 tries/min (ratio 1.5)
        c.observe_group(i * 180.0, 10)
    now = 1800.0
    # in the next 30 min: 10 ready + 5 groups due -> 60 tries = 12 min of training -> ~17 min idle
    mode, block = c.decide(now, 10, [(now + 300 * k, 10) for k in range(1, 6)] + [(now + 4000, 10)])
    assert mode == "play" and 360 <= block <= 1200
    assert c.decide(now, 200, [])[0] == "train"  # a full buffer: train
    fast = Card2(ratio_on=1.3)
    fast.observe_step(40, 480)
    for i in range(10):  # server 1 at 6.7 tries/min: the trainer is the bottleneck
        fast.observe_group(i * 90.0, 10)
    assert fast.decide(900.0, 0, [])[0] == "train"


def test_card2_start_block_plays_until_the_first_group_is_due():
    c = Card2(min_block_s=900, max_block_s=1200, horizon_s=1800, switch_s=40)
    assert c.decide(0, 0, [(2700, 12)]) == ("train", 0.0)  # off by default
    c = Card2(min_block_s=900, max_block_s=1200, horizon_s=1800, switch_s=40, start_block=True)
    assert c.decide(0, 0, [(2700, 12), (2800, 12)]) == ("play", 1800)  # up to the horizon at the start
    assert c.decide(0, 0, [(1500, 12)]) == ("play", 1460)
    assert c.decide(0, 0, []) == ("play", 1760)  # nothing in flight yet: the horizon
    assert c.decide(0, 0, [(800, 12)])[0] == "train"  # a group due within min_block_s
    assert c.decide(0, 12, [(2700, 12)])[0] == "train"  # a group ready: train
    c.observe_step(96, 600)  # a trainer rate: the normal rule (no P1 yet -> train)
    assert c.decide(0, 0, [(2700, 12)])[0] == "train"


def test_helpers():
    assert lag_hist([0, 2, 2, None]) == {"0": 1, "2": 2, "None": 1}
    assert ess([1, 1, 1]) == pytest.approx(1.0) and ess([2, 0, 0]) == pytest.approx(1 / 3)
    assert balance_scale(3.0, 6.0) == 0.5 and balance_scale(0.0, 6.0) == 1.0


def test_fake_trainer_accumulate_then_step(tmp_path):
    from arena import fake_trainer

    roll = tmp_path / "r" / "rollout.jsonl"
    roll.parent.mkdir()
    roll.write_text(json.dumps({"tokens": [1, 2, 3], "gameplay": True}) + "\n")
    fake = fake_trainer.Fake(16384, folder=tmp_path / "f")
    spec = {"name": "t1", "rollout": str(roll), "w": 1.0}
    assert fake.run("step", {"lr": 1e-5})["ok"] is False
    a = fake.run("accumulate", {"attempts": [spec], "lr": 1e-5, "first": True, "prefix_chunk": 16384, "keep_old": ["t1"]})
    b = fake.run("accumulate", {"attempts": [spec], "lr": 1e-5, "prefix_chunk": 16384})
    assert a["ok"] and b["acc_calls"] == 2 and fake.updates == 0
    out = fake.run("step", {"lr": 1e-5, "save": str(tmp_path / "ad"), "prefix_chunk": 16384})
    assert out["ok"] and out["applied"] and fake.updates == 1 and (tmp_path / "ad").exists()
    k3 = fake.run("k3_score", {"attempts": [spec]})  # pi_old kept by the first accumulate call survives the second
    assert k3["ok"] and k3["approx_kl"] is not None and not fake.run("k3_score", {"attempts": [spec]})["ok"]
    assert fake.run("acc_reset", {})["ok"] and fake.run("step", {"lr": 1e-5})["ok"] is False
    assert fake.run("unpark", {}) == {"already": True}
    fake.run("park", {})
    assert fake.run("unpark", {})["unparked"] and not fake.parked


def test_a_long_read_never_stops_new_versions_serving():
    """(v6g steps 4-6 trained on version-3 data): a read open across 3 steps holds its own slot
    (read_slots) and the base model (free) none; the newest version still serves with max_loras group versions."""
    from arena.async_rl import Book

    b = Book(3, read_slots=1, free={0})
    b.acquire(0, 17, read=True)  # the base read: no LoRA slot at all
    b.acquire(1, 12)
    assert b.publish(1) and b.served == 1
    b.acquire(1, 1, read=True)  # a step-1 read that stays open
    for v in (2, 3, 4):
        assert b.publish(v) and b.served == v, v
        b.acquire(v, 12)
        if v >= 3:  # groups drain lag 1 behind (version v-2's tries finish)
            b.release(v - 2, 12)
        assert b.slots_ok()
    assert 1 in b.live() and 1 not in b.unloadable([0, 1, 3, 4])
    b.release(1, 1, read=True)
    assert b.unloadable([0, 1, 3, 4]) == [1] and b.live() == [0, 3, 4]
    old = Book(3)  # the old book: the reads counted as group slots and the version waited
    old.acquire(0, 17)
    old.acquire(2, 12)
    old.acquire(3, 12)
    old.current = 3
    assert not old.publish(4) and old.served == 0


def test_book_state_keeps_read_slots():
    from arena.async_rl import Book

    b = Book(3, read_slots=1)
    assert Book.load_state(b.state_dict()).read_slots == 1 and Book.load_state({**b.state_dict(), "read_slots": 0}, read_slots=2).read_slots == 2


def test_buffer_drops_all_reported_once_and_restored_drops_seen():
    """v6h: one next_group call dropped 12 stale replayed groups and only the last reached the feed."""
    from arena.async_rl import Book, Buffer, Group

    book = Book(3, 1, version=6)
    buf = Buffer(2, 8)
    for i in range(3):
        buf.add(Group(gid=f"g{i:05d}", unit="u-L1", start=None, version=3, size=12, created=0.0))
    assert buf.next_group(book, 0.0) == (None, None)
    assert [d["gid"] for d in buf.drops if d.get("_seen") is None] == ["g00000", "g00001", "g00002"]
    again = Buffer.load_state(buf.state_dict())
    assert all(d["_seen"] for d in again.drops)


def test_book_load_state_takes_the_resumed_legs_max_loras():
    """v6j pfR2: the saved Book's max_loras (3) won over the leg's 4 (s_max 3 needs lags 0..3 live)."""
    from arena.async_rl import Book

    saved = Book(3, version=0).state_dict()
    assert Book.load_state(saved).max_loras == 3
    assert Book.load_state(saved, max_loras=4, read_slots=1).max_loras == 4
