"""LLM-ranked backlog: evidence table, ranking parse, scheduler order, the ranker's request."""

import asyncio
import json

import pytest

from arena.clock import ActiveClock, GlobalClock
from arena.gate import KVGate
from arena.llm_rank import LLMRanker, evidence, messages, parse_ranking
from arena.scheduler import GameRecord, Scheduler


class Now:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def setup(n=5):
    now = Now()
    wall = GlobalClock(3600, now=now)
    games = [GameRecord(f"g{i}xx-000{i}", i, ActiveClock(wall, now=now)) for i in range(n)]
    sched = Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="llm")
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    now.value = 100.0
    games[0].note_level(1, 100.0)
    games[1].note_level(1, 100.0)
    sched.tick()
    return now, wall, games, sched


def test_parse_ranking_matches_prefixes_and_drops_unknown_and_repeats():
    ids = ["ar25-1", "tn36-2", "sb26-3"]
    text = 'noise {"analysis": "x", "ranking": ["tn36", "zz99", "tn36-2", "ar25"]} tail'
    assert parse_ranking(text, ids) == (["tn36-2", "ar25-1"], None)
    assert parse_ranking("no json", ids)[1] == "no JSON object"
    assert parse_ranking('{"ranking": "ar25"}', ids)[1] == "no ranking list"
    assert parse_ranking('{"ranking": ["zz99"]}', ids)[1] == "no known ids"


def test_llm_order_replaces_readmissions_but_parked_clears_stay_first():
    now, wall, games, sched = setup()
    assert sched.phase == "steady"
    waiting = [g.game_id for g in sched.backlog()]
    assert waiting == [g.game_id for g in games[2:]]  # readmissions/roster fallback before a ranking
    sched.set_llm_ranking([games[4].game_id, games[2].game_id])
    assert [g.game_id for g in sched.backlog()] == [games[4].game_id, games[2].game_id, games[3].game_id]
    games[3].pending_parked_progress = True
    assert sched.backlog()[0] is games[3]
    assert sched.summary()["backlog_policy"] == "llm" and sched.events[-1]["event"] == "llm_rank"
    with pytest.raises(ValueError, match="fixed-2x"):
        Scheduler(games, wall, backlog_policy="llm")


def test_evidence_and_messages_cover_every_unfinished_game():
    now, wall, games, sched = setup()
    games[2].note_action("a")
    games[2].note_action("a")
    rows = evidence(sched)
    assert [r["state"] for r in rows] == ["focus", "focus", "parked", "parked", "parked"]
    assert rows[2]["actions_level"] == 2 and rows[2]["distinct_boards_level"] == 1 and rows[2]["actions_since_new_board"] == 1
    msgs, _ = messages(sched, wall.elapsed(), wall.remaining())
    assert msgs[0]["role"] == "system" and "g2xx" in msgs[1]["content"] and "2 focus slots" in msgs[1]["content"]


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class FakeClient:
    def __init__(self, reply):
        self.reply, self.posts = reply, []

    async def get(self, path):
        return FakeResponse({"data": [{"id": "served"}]})

    async def post(self, path, json):
        self.posts.append((path, json))
        return FakeResponse({"choices": [{"message": {"content": self.reply}}], "usage": {"completion_tokens": 20}})


def test_ranker_request_is_tool_less_thinking_off_through_the_gate_and_installs_the_order():
    now, wall, games, sched = setup()
    reply = json.dumps({"analysis": "g4 moves", "ranking": ["g4xx", "g3xx", "g2xx"]})
    client = FakeClient(reply)
    gate = KVGate(100000, 8)
    ranker = LLMRanker(sched, gate, client, model=None, wall=wall, now=now)
    record = asyncio.run(ranker.rank_once())
    path, payload = client.posts[0]
    assert path == "chat/completions" and payload["model"] == "served" and "tools" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False} and payload["max_tokens"] == 768
    assert record["ranking"] == ["g4xx", "g3xx", "g2xx"] and record["error"] is None
    assert [g.game_id[:4] for g in sched.backlog()] == ["g4xx", "g3xx", "g2xx"]
    assert gate.inflight == 0 and gate.reserved == 0
    bad = LLMRanker(sched, gate, FakeClient("sorry"), model="m", wall=wall, now=now)
    assert asyncio.run(bad.rank_once())["error"] == "no JSON object" and bad.errors == 1
    assert sched.llm_rankings == 1


def test_thompson_backlog_prefers_clears_demotes_fruitless_and_is_seeded():
    now = Now()
    wall = GlobalClock(3600, now=now)
    games = [GameRecord(f"g{i}xx-000{i}", i, ActiveClock(wall, now=now)) for i in range(6)]
    sched = Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="thompson", thompson_seed=3)
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    now.value = 100.0
    games[0].note_level(1, 100.0)
    games[1].note_level(1, 100.0)
    sched.tick()
    # Opening parks count as fruitless admissions for the level-0 games.
    assert all(g.fruitless == 1 for g in games[2:]) and games[0].fruitless == 0
    games[2].levels, games[2].fruitless = 3, 0
    games[5].fruitless = 20
    firsts = [sched.backlog()[0] for _ in range(200)]
    assert firsts.count(games[2]) > 150 and games[5] not in firsts
    orders = {tuple(g.index for g in sched.backlog()) for _ in range(50)}
    assert len(orders) > 1  # level-0 ties are drawn, not roster order
    again = Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="thompson", thompson_seed=3)
    assert [g.index for g in again.backlog()] == [g.index for g in Scheduler(
        games, wall, budget_policy="fixed-2x", backlog_policy="thompson", thompson_seed=3).backlog()]
    games[4].pending_parked_progress = True
    assert sched.backlog()[0] is games[4]
    with pytest.raises(ValueError, match="fixed-2x"):
        Scheduler(games, wall, backlog_policy="thompson")
