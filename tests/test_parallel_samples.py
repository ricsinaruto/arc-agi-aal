"""hero-v2 ``parallel_samples`` (arena.parallel_samples; docs/validation-20260927/parallel-samples-plans.md): settings,
the gate's exact sibling reservations and yield, winner selection and aborts on a fake engine, budget charging, a
randomized no-preemption check, the agent path with compaction, and the hero-v2 composition."""

import asyncio
import copy
import json
import random
from types import SimpleNamespace as NS

import httpx
import pytest
from conftest import observation, tool_reply
from test_arena_duck import arena_agent, text_reply

from arena import parallel_samples as ps
from arena.cache import BlockBudget
from arena.duck import ArenaTransferDuckAgent
from arena.gate import KVGate

BLOCK = 100
FIXED = 9  # the served layout's per-request recurrent-state blocks


def layout(max_len=20000, blocks=400):
    rows = []
    for tokens in sorted({1, max_len, *range(BLOCK, max_len, BLOCK)}):
        groups = [-(-tokens // BLOCK), 1, 2, 2, 2, 2]
        rows.append({"tokens": tokens, "blocks": sum(groups), "groups": groups})
    groups = [{"type": "FullAttentionSpec", "block_size": BLOCK}, {"type": "CircularBufferSpec", "block_size": 4},
              *({"type": "MambaSpec", "block_size": BLOCK} for _ in range(4))]
    return {"schema": 1, "num_blocks": blocks, "max_model_len": max_len, "lookahead": 0, "costs": rows,
            "groups": groups}


def budget(blocks=400):
    return BlockBudget(layout(blocks=blocks))


# --- settings and accounting ---------------------------------------------------------------------------------------

def test_settings_default_off_and_validate():
    assert ps.settings(None) is None and ps.settings(False) is None
    full = ps.settings(True)
    assert full == {**ps.DEFAULTS, "abort_grace_seconds": 0.5, "late_admission_seconds": 0.0, "retry_seconds": 0.25}
    assert full["k"] == 4 and full["min_levels"] == 2 and full["max_num_seqs"] == 48
    assert ps.settings({"k": 3})["k"] == 3
    for bad in ({"k": 1}, {"k": 9}, {"min_levels": -1}, {"max_num_seqs": 64}, {"observe_every": 1}, {"bogus": 1},
                {"abort_grace_seconds": 0}, "yes", {"k": 2.0}):
        with pytest.raises(ValueError):
            ps.settings(bad)
    assert ps.graph_capture_size(48) == 48 and ps.graph_capture_size(41) == 48 and ps.graph_capture_size(26) == 32


def test_a_sibling_reserves_its_own_blocks_and_shares_the_full_prompt_blocks():
    b = budget()
    # 1,234 prompt tokens: 12 full attention blocks shared; the sibling's own: the tail block, the reply and 9 fixed.
    assert b.cost(1234 + 1000) == 23 + FIXED
    assert ps.shared_blocks(b, 1234) == 12
    assert ps.sibling_units(b, 1234, 1000) == 23 + FIXED - 12
    # The run-B layout: a 90K prompt with the 16K cap at 3,136-token blocks.
    run_b = NS(layout={"groups": [{"type": "FullAttentionSpec", "block_size": 3136}, {"type": "MambaSpec",
                                                                                    "block_size": 3136}]},
               cost=lambda tokens: -(-(-(-tokens // 1024) * 1024) // 3136) + FIXED)
    assert run_b.cost(90_000 + 16_384) == 34 + FIXED and ps.shared_blocks(run_b, 90_000) == 28
    assert ps.sibling_units(run_b, 90_000, 16_384) == 15
    assert ps.shared_blocks(NS(layout=None), 5000) == 0


def test_try_acquire_takes_exact_units_keeps_headroom_and_a_free_slot():
    gate = KVGate(360, slots=4, budget=budget())
    gate.capacity = 360
    log = []
    gate.log = log.append
    first = gate.try_acquire("g", 2234, headroom_tokens=10256, units=20, event="sibling_admit")
    assert first is not None and first.tokens == 20 and gate.reserved == 20 and log[-1]["event"] == "sibling_admit"
    # headroom: one 10,256-token request (112 blocks) must still fit beside it.
    assert gate.try_acquire("g", 2234, headroom_tokens=10256, units=229) is None
    assert gate.try_acquire("g", 2234, headroom_tokens=10256, units=228) is not None
    # a slot stays free for the next request: 4 slots, 2 in flight, a third would leave one.
    assert gate.try_acquire("g", 100, units=1) is not None
    assert gate.try_acquire("g", 100, units=1) is None
    gate.release_background(first, event="sibling_release")
    assert log[-1]["event"] == "sibling_release" and gate.reserved == 229
    # keep_alive's defaults are unchanged: budget cost, background_admit / background_release.
    ping = gate.try_acquire("k", 150)
    assert ping.tokens == budget().cost(150) and log[-1]["event"] == "background_admit"


async def test_background_slots_keep_the_regular_slots_and_cap_the_siblings():
    """With background_slots the regular requests keep all their slots whatever the siblings hold, the siblings use
    at most background_slots sequences, and block holds (slot=False) take no sequence."""
    gate = KVGate(1000, slots=3, background_slots=2)
    siblings = [gate.try_acquire("s", 10, units=10) for _ in range(3)]
    assert siblings[2] is None and gate.background_inflight == 2 and gate.inflight == 2
    hold = gate.try_acquire("h", 0, units=5, slot=False, event="grace_hold")
    assert hold is not None and gate.inflight == 2 and gate.reserved == 25
    regular = [await gate.acquire(f"r{i}", 10) for i in range(3)]  # all three regular slots, beside the siblings
    assert gate.inflight == 5 and gate.snapshot()["background_inflight"] == 2
    waiting = asyncio.create_task(gate.acquire("r3", 10))
    await asyncio.sleep(0)
    assert not waiting.done()  # the fourth regular request waits for a regular slot, not for a sibling
    gate.release_background(siblings[0])
    await asyncio.sleep(0)
    assert not waiting.done()
    gate.release(regular[0])
    assert (await waiting).tokens == 10
    gate.release_background(hold)
    assert gate.reserved == 10 * 4 and gate.inflight == 4
    # background_slots 0 (every earlier build): the snapshot has no new keys.
    assert "background_slots" not in KVGate(10, slots=2).snapshot()


async def test_nothing_is_admitted_beside_a_waiting_request_and_the_hooks_yield():
    gate = KVGate(100, slots=8)
    held = await gate.acquire("a", 80)
    freed = []

    def hook(deficit, need_slot):
        freed.append((deficit, need_slot))
        return deficit, 0

    gate.queue_hooks.append(hook)
    waiter = asyncio.create_task(gate.acquire("b", 50))
    await asyncio.sleep(0)
    assert freed == [(30, False)]  # 80 + 50 - 100
    assert gate.try_acquire("c", 1) is None  # a request waits: no background admission
    gate.release(held)
    reservation = await waiter
    assert reservation.tokens == 50


# --- the sampler on a fake engine ------------------------------------------------------------------------------------

class Workspace:
    def __init__(self):
        self.events = []

    def append(self, kind, **data):
        self.events.append({"kind": kind, **data})


def reply(kind, tokens=100, prompt=1234, cached=None):
    usage = {"completion_tokens": tokens, "prompt_tokens": prompt}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    if kind == "tool":
        response = tool_reply("print(1)", usage=usage)
    elif kind == "text":
        response = text_reply("World model: thinking.", usage=usage)
    elif kind == "markup":
        response = text_reply("<tool_call><function=python><parameter=code>print(2)</parameter></function>"
                              "</tool_call>", usage=usage)
    else:
        response = httpx.Response(int(kind), request=httpx.Request("POST", "http://local/v1/chat/completions"),
                                  json={"error": "boom"})
    return response


class Engine:
    """``base(payload, record_extra=...)``: the ping answers at once; sample i (launch order) follows ``script[i]`` =
    (seconds, kind[, tokens]). Records every call and every cancellation."""

    def __init__(self, script, ping="ok"):
        self.script, self.ping = list(script), ping
        self.calls, self.cancelled, self.finished = [], [], []

    async def __call__(self, payload, record_extra=None):
        if payload.get("max_tokens") == 1:
            self.calls.append(("ping", record_extra))
            if self.ping == "raise":
                raise httpx.ConnectError("down")
            return reply("tool" if self.ping == "ok" else self.ping, tokens=1, cached=1200)
        index = len([c for c in self.calls if c[0] == "sample"])
        self.calls.append(("sample", record_extra))
        seconds, kind, *rest = self.script[index] if index < len(self.script) else (0.01, "tool")
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            self.cancelled.append(index)
            raise
        self.finished.append(index)
        if kind == "raise":
            raise httpx.ReadError("reset")
        return reply(kind, tokens=rest[0] if rest else 100, cached=1200)


def sampler_for(spec=None, *, levels=2, capacity=400, slots=12, prompt=1234, window=10000):
    from duck_transfer.loader import load

    gate = KVGate(capacity, slots=slots, budget=budget(blocks=int(capacity / 0.9) + 1))
    gate.capacity = capacity
    events = []
    gate.log = events.append
    game = NS(levels=levels, game_id="g1")
    slot = NS(gate=gate, next_estimate=prompt)
    agent = NS(game=game, client=NS(slots=slot, client=NS(records_extra=True)), window=window, cap=1000,
               module=load(), workspace=Workspace(), steps=7)
    sampler = ps.ParallelSampler(agent, {**ps.settings(True), "abort_grace_seconds": 0.05, "retry_seconds": 0.02,
                                         "prefix_warmup": True, "late_admission_seconds": 30.0,
                                         **(spec or {})})
    agent.sampler = sampler
    return sampler, gate, agent, events


async def settle(gate, seconds=0.12):
    await asyncio.sleep(seconds)
    return gate.reserved, gate.inflight


async def test_the_first_tool_call_wins_and_the_others_are_aborted():
    sampler, gate, agent, events = sampler_for()
    engine = Engine([(0.3, "tool", 900), (0.05, "tool", 120), (0.2, "tool"), (0.25, "text")])
    response = await sampler.complete({"messages": [], "max_tokens": 1000}, engine)
    assert response.json()["usage"]["completion_tokens"] == 120  # sample 1 (a sibling) won
    await asyncio.sleep(0.01)  # the cancellations land at the next loop turns
    assert sorted(engine.cancelled) == [0, 2, 3] and engine.finished == [1]
    # The ping went first, then the primary and the three reserved siblings, each tagged.
    assert [c[1]["parallel_role"] for c in engine.calls] == ["ping", "primary", "sibling", "sibling", "sibling"]
    assert await settle(gate) == (0, 0)  # every sibling reservation released after the grace
    admits = [e for e in events if e["event"] == "sibling_admit"]
    assert len(admits) == 3 and all(e["tokens"] == ps.sibling_units(gate.budget, 1234, 1000) for e in admits)
    stats = sampler.summary()
    assert stats["groups"] == 1 and stats["winner_sibling"] == 1 and stats["primary_aborted"] == 1
    assert stats["siblings_aborted"] == 2 and stats["winner_completion_tokens"] == 120
    # The aborted primary's blocks stay covered for the grace by a hold of its full cost (the reply does not wait).
    assert stats["aborted_tokens_estimated"] > 0 and stats["grace_holds"] == 1 and stats["grace_seconds"] == 0
    assert [e for e in events if e["event"] == "grace_hold"][0]["tokens"] == gate.budget.cost(1234 + 1000)
    assert stats["sibling_winner_full_hits"] == 1  # 1,200 cached = the prompt's 12 full blocks
    record = [e for e in agent.workspace.events if e["kind"] == "parallel_samples"][0]
    assert record["winner"] == 1 and record["k"] == 4 and record["ping"]["cached_tokens"] == 1200
    assert {s["i"]: s["status"] for s in record["samples"]} == {0: "aborted", 1: "won", 2: "aborted", 3: "aborted"}


async def test_a_reply_without_a_tool_call_waits_for_a_sibling_that_acts():
    sampler, gate, agent, _ = sampler_for()
    engine = Engine([(0.02, "text", 50), (0.15, "tool", 300), (0.4, "tool"), (0.45, "tool")])
    response = await sampler.complete({"messages": [], "max_tokens": 1000}, engine)
    assert response.json()["usage"]["completion_tokens"] == 300
    await asyncio.sleep(0.01)  # the cancellations land at the next loop turns
    assert engine.finished[:2] == [0, 1] and sorted(engine.cancelled) == [2, 3]
    # Markup the public loop recovers counts as a tool call.
    sampler2, *_ = sampler_for()
    engine2 = Engine([(0.2, "tool", 10), (0.02, "markup", 20), (0.3, "tool"), (0.3, "tool")])
    assert (await sampler2.complete({"messages": []}, engine2)).json()["usage"]["completion_tokens"] == 20
    assert await settle(gate) == (0, 0)


async def test_when_no_sample_acts_the_first_finished_reply_is_used():
    sampler, gate, _, _ = sampler_for({"k": 3})
    engine = Engine([(0.1, "text", 11), (0.03, "text", 22), (0.06, "500")])
    response = await sampler.complete({"messages": []}, engine)
    assert response.json()["usage"]["completion_tokens"] == 22 and sampler.stats["winner_without_tool_call"] == 1
    assert await settle(gate) == (0, 0)


async def test_errors_never_decide_alone_and_the_primary_error_path_is_kept():
    # A failing primary with an acting sibling: the sibling's reply.
    sampler, gate, _, _ = sampler_for({"k": 2})
    engine = Engine([(0.01, "raise"), (0.05, "tool", 77)])
    assert (await sampler.complete({"messages": []}, engine)).json()["usage"]["completion_tokens"] == 77
    # Every sample failing: the primary's own exception, as with one sample.
    sampler, gate, _, _ = sampler_for({"k": 2})
    with pytest.raises(httpx.ReadError):
        await sampler.complete({"messages": []}, Engine([(0.01, "raise"), (0.02, "raise")]))
    assert gate.reserved == 0 and gate.inflight == 0  # the failed siblings' reservations came back
    # Only non-2xx replies: the primary's response takes its ordinary path (e.g. a context rejection).
    sampler, gate, _, _ = sampler_for({"k": 2})
    response = await sampler.complete({"messages": []}, Engine([(0.03, "400"), (0.01, "500")]))
    assert response.status_code == 400
    assert await settle(gate) == (0, 0)


async def test_a_cancelled_group_releases_everything_after_the_grace():
    """The wall end cancels the game's request mid-group: running samples are aborted, ended ones released, and the
    aborted primary's blocks stay covered by a hold for the grace."""
    sampler, gate, _, events = sampler_for()
    engine = Engine([(0.5, "tool"), (0.02, "raise"), (0.5, "tool"), (0.5, "tool")])
    task = asyncio.create_task(sampler.complete({"messages": []}, engine))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [e["event"] for e in events].count("grace_hold") == 1
    assert await settle(gate) == (0, 0)


async def test_no_room_or_a_failed_ping_falls_back_to_the_single_request():
    sampler, gate, _, events = sampler_for(capacity=100)  # a full window (112 blocks) no longer fits beside
    engine = Engine([(0.01, "tool", 5)])
    assert (await sampler.complete({"messages": []}, engine)).json()["usage"]["completion_tokens"] == 5
    assert [c[0] for c in engine.calls] == ["sample"] and engine.calls[0][1] is None  # untagged: the plain request
    assert sampler.stats["groups_without_room"] == 1 and sampler.stats["groups"] == 0
    for ping in ("400", "raise"):
        sampler, gate, _, _ = sampler_for()
        engine = Engine([(0.01, "tool", 6)], ping=ping)
        if ping == "raise":
            with pytest.raises(httpx.ConnectError):
                await sampler.complete({"messages": []}, engine)
        else:
            assert (await sampler.complete({"messages": []}, engine)).json()["usage"]["completion_tokens"] == 6
            assert sampler.stats["ping_failures"] == 1 and len(engine.calls) == 2
        assert gate.reserved == 0 and gate.inflight == 0


async def test_a_waiting_regular_request_makes_the_siblings_yield():
    # A 3,000-token window: a full request (3,256 tokens) costs 42 blocks of headroom.
    sampler, gate, agent, events = sampler_for(capacity=200, window=3000)
    other = await gate.acquire("other", 5000)  # 59 blocks
    engine = Engine([(0.6, "tool", 400), (0.5, "tool", 500), (0.5, "tool", 500), (0.5, "tool", 500)])
    task = asyncio.create_task(sampler.complete({"messages": []}, engine))
    await asyncio.sleep(0.05)
    # "other" + three siblings of 20 blocks (the primary runs on the game's own reservation, not taken here).
    assert gate.inflight == 4 and gate.reserved == 59 + 3 * 20 and len(engine.calls) == 5
    # A regular request needing 82 blocks (81 are left) waits: one sibling yields, and the request gets in after the
    # grace, long before the group ends.
    waiting = asyncio.create_task(gate.acquire("late", 7300))
    await asyncio.sleep(0.01)
    assert not waiting.done() and sampler.stats["siblings_yielded"] == 1
    admitted = await asyncio.wait_for(waiting, 0.3)
    assert not task.done() and admitted.tokens == gate.budget.cost(7300) == 82
    response = await task
    assert response.json()["usage"]["completion_tokens"] == 500  # a remaining sibling won
    # No late sibling after a yield, and nothing was admitted beside the waiting request.
    assert len(engine.calls) == 5 and sampler.stats["late_samples"] == 0
    record = [e for e in agent.workspace.events if e["kind"] == "parallel_samples"][0]
    assert [s.get("yielded", False) for s in record["samples"]] == [False, False, False, True]
    gate.release(admitted)
    gate.release(other)
    assert await settle(gate) == (0, 0)


async def test_late_siblings_join_while_the_group_is_young():
    sampler, gate, _, _ = sampler_for(capacity=300)
    other = await gate.acquire("other", 15000)  # 159 blocks: 159 + 112 headroom leaves room for one sibling
    engine = Engine([(0.4, "tool", 400), (0.35, "tool"), (0.05, "tool", 33), (0.3, "tool")])
    task = asyncio.create_task(sampler.complete({"messages": []}, engine))
    await asyncio.sleep(0.05)
    assert len(engine.calls) == 3  # ping, primary, one sibling
    gate.release(other)
    response = await task
    assert response.json()["usage"]["completion_tokens"] == 33  # a late sibling won
    assert sampler.stats["late_samples"] == 2 and sampler.stats["samples_launched"] == 4
    assert await settle(gate) == (0, 0)


async def test_observed_groups_let_the_losers_finish_and_measure_the_time_saved():
    sampler, gate, agent, events = sampler_for({"observe_every": 2})
    engine = Engine([(0.01, "tool")] * 4)
    await sampler.complete({"messages": []}, engine)  # group 1: aborts as usual
    engine = Engine([(0.2, "tool", 500), (0.05, "tool", 50), (0.1, "text", 60), (0.3, "tool", 70)])
    response = await sampler.complete({"messages": []}, engine)
    assert response.json()["usage"]["completion_tokens"] == 50 and not engine.cancelled
    await asyncio.sleep(0.4)
    assert sorted(engine.finished) == [0, 1, 2, 3]
    observed = [e for e in agent.workspace.events if e["kind"] == "parallel_observed"]
    assert len(observed) == 1 and not observed[0]["censored"] and len(observed[0]["durations"]) == 4
    stats = sampler.summary()
    assert stats["observed_groups"] == 1 and stats["observe_holds"] == 1
    assert stats["saved_vs_slowest"]["groups"] == 1 and stats["saved_vs_slowest"]["mean_seconds"] > 0.2
    assert stats["finished_loser_tokens"] == 3 * 100 + 500 + 60 + 70  # group 1's simultaneous finishers too
    assert gate.reserved == 0 and gate.inflight == 0  # the primary's hold and the siblings' own released


async def test_randomized_load_never_exceeds_the_pool_and_regular_requests_barely_wait():
    """Several games sharing one gate; a fake engine allocates physical blocks as vLLM would (prompt blocks shared by
    a group's samples after the ping, own tail / reply / state blocks per request, aborts processed after a lag
    shorter than the grace). The physical use must never exceed the gate capacity (no preemption), every
    reservation must be released, and regular requests wait at most about one grace."""
    rng = random.Random(7)
    lay = layout(max_len=20000, blocks=1400)
    gate = KVGate(int(0.9 * 1400), slots=24, budget=BlockBudget(lay))
    groups_blocks = {}  # prompt key -> [shared blocks, users]
    physical = {"now": 0, "peak": 0}
    waits = []

    def use(delta):
        physical["now"] += delta
        physical["peak"] = max(physical["peak"], physical["now"])
        assert physical["now"] <= gate.capacity, "the engine would preempt"

    class Engine2:
        def __init__(self, name):
            self.name = name

        async def __call__(self, payload, record_extra=None):
            prompt = payload["prompt"]
            key = (self.name, payload["turn"])
            shared = ps.shared_blocks(gate.budget, prompt)
            entry = groups_blocks.setdefault(key, [shared, 0])
            if entry[1] == 0:
                use(entry[0])
            entry[1] += 1
            own = gate.budget.cost(prompt) - shared
            use(own)
            held = own

            def free():
                use(-held)
                entry[1] -= 1
                if entry[1] == 0:
                    use(-entry[0])

            try:
                if payload.get("max_tokens") == 1:
                    await asyncio.sleep(0.005)
                    result = reply("tool", tokens=1, prompt=prompt, cached=0)
                else:
                    length = rng.choice([50, 200, 400, 900, 2000])
                    for step in range(0, length, 100):
                        await asyncio.sleep(0.004)
                        grown = gate.budget.cost(prompt + step + 100) - shared
                        use(grown - held)
                        held = grown
                    result = reply("tool" if rng.random() < 0.8 else "text", tokens=length, prompt=prompt)
            except asyncio.CancelledError:
                async def free_later():
                    await asyncio.sleep(0.01)  # the engine processes an abort a step or two later (< the grace)
                    free()
                asyncio.get_running_loop().create_task(free_later())
                raise
            free()  # a finished request's blocks are freed before its reply is sent
            return result

    async def game(name, levels, turns):
        from duck_transfer.loader import load

        slot = NS(gate=gate, next_estimate=0)
        agent = NS(game=NS(levels=levels, game_id=name), client=NS(slots=slot, client=NS(records_extra=False)),
                   window=12000, cap=2000, module=load(), workspace=Workspace(), steps=0)
        sampler = ps.ParallelSampler(agent, {**ps.settings(True), "abort_grace_seconds": 0.05,
                                             "retry_seconds": 0.01, "observe_every": 3, "prefix_warmup": True})
        engine = Engine2(name)
        for turn in range(turns):
            prompt = rng.randint(1000, 9500)
            slot.next_estimate = prompt
            started = asyncio.get_running_loop().time()
            reservation = await gate.acquire(name, prompt + 2000)
            waits.append(asyncio.get_running_loop().time() - started)
            try:
                payload = {"prompt": prompt, "turn": turn, "max_tokens": 2000}
                if sampler.eligible():
                    await sampler.complete(payload, engine)
                else:
                    await engine(payload)
            finally:
                gate.release(reservation)
            await asyncio.sleep(rng.random() * 0.01)
        await sampler.close()
        return sampler.summary()

    results = await asyncio.gather(*(game(f"g{i}", i % 4, 12) for i in range(10)))
    await asyncio.sleep(0.2)
    assert gate.reserved == 0 and gate.inflight == 0 and physical["now"] == 0
    total = ps.aggregate(results)
    assert total["groups"] > 15 and total["samples_launched"] > 2 * total["groups"]
    assert total["siblings_aborted"] > 0 and total["observed_groups"] > 0
    assert physical["peak"] <= gate.capacity and physical["peak"] > 0.5 * gate.capacity  # the pool was under load
    assert max(waits) < 1.0


# --- the agent path: budget charging, the winner's reply in the trajectory, compaction ------------------------------

def boards(n):
    """``n`` distinct boards after the initial one (each action changes the board)."""
    import numpy as np

    result = []
    for i in range(n):
        board = np.zeros((64, 64), dtype=np.uint8)
        board[3 + i, 7] = 5
        result.append(observation(board))
    return result


class GroupTransport:
    """Gameplay requests of the agent: a ping (max_tokens 1) answers at once; within a group the i-th sample waits
    ``delays[i]`` and acts with its own call id, so the trajectory shows which one won."""

    def __init__(self, delays, tokens):
        self.delays, self.tokens = delays, tokens
        self.payloads, self.cancelled = [], []
        self.group, self.index = -1, 0

    async def post(self, url, **kwargs):
        payload = copy.deepcopy(kwargs["json"])
        self.payloads.append(payload)
        if payload.get("tool_choice") == "none":  # a compaction summary
            return text_reply("World model: summary. Plan: act.")
        if payload.get("max_tokens") == 1:
            self.group += 1
            self.index = 0
            return text_reply("", usage={"completion_tokens": 1, "prompt_tokens": 1000})
        index, self.index = self.index, self.index + 1
        try:
            await asyncio.sleep(self.delays[index] if index < len(self.delays) else 0.01)
        except asyncio.CancelledError:
            self.cancelled.append((self.group, index))
            raise
        return tool_reply("action([{'action':'UP'}])", call_id=f"g{self.group}s{index}",
                          content=f"World model: sample {index}.",
                          usage={"completion_tokens": self.tokens[index], "prompt_tokens": 1000})


async def test_the_winner_alone_enters_the_trajectory_and_is_charged(tmp_path):
    from arena.clock import ActiveClock, GlobalClock

    transport = GroupTransport(delays=[0.3, 0.2, 0.02, 0.25], tokens=[900, 800, 70, 600])
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    wall = GlobalClock(900)
    game.clock = ActiveClock(wall, budget_clock="tokens")
    wall.release()
    game.clock.release()
    game.levels = 2
    agent.apply_parallel_samples({"abort_grace_seconds": 0.05, "retry_seconds": 0.01, "prefix_warmup": True})
    agent.config.max_actions = 2
    env.observations = iter(boards(2))
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 2
    gameplay = [p for p in transport.payloads if p.get("max_tokens") != 1 and p.get("tool_choice") != "none"]
    second = gameplay[-1]["messages"]
    # The second turn's context holds the first group's winner (sample 2) verbatim, and no loser.
    assistants = [m for m in second if m["role"] == "assistant"]
    assert [m["tool_calls"][0]["id"] for m in assistants] == ["g0s2"]
    assert "sample 2" in assistants[0]["content"] and "sample 1" not in json.dumps(second)
    # The token clock charged the winners only: two groups x 70 tokens.
    assert game.clock.budget_tokens() == 140
    stats = result["parallel_samples"]
    assert stats["groups"] == 2 and stats["winner_sibling"] == 2 and stats["winner_completion_tokens"] == 140
    assert sorted(transport.cancelled) == [(0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (1, 3)]
    await asyncio.sleep(0.1)
    assert gate.reserved == 0 and gate.inflight == 0


async def test_below_min_levels_the_request_is_the_plain_single_request(tmp_path):
    transport = GroupTransport(delays=[0.01] * 4, tokens=[5] * 4)
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    agent.apply_parallel_samples(True)
    result = await agent.run()
    assert result["error"] is None and all(p.get("max_tokens") != 1 for p in transport.payloads)
    assert result["parallel_samples"]["requests_eligible"] == 0
    # Off: no sampler, no row.
    control, *_ = arena_agent(tmp_path / "c", GroupTransport([0.01], [5]), ArenaTransferDuckAgent)
    assert control.sampler is None and "parallel_samples" not in control.result()
    control.executor.shutdown()


async def test_compaction_after_a_sibling_won_summarizes_the_winner_reply(tmp_path):
    from test_harness_fixes import LAYOUT, counter, events_of, text_of

    from arena.compaction import SUMMARY_PREAMBLE

    transport = GroupTransport(delays=[0.2, 0.02, 0.2, 0.2], tokens=[900, 70, 600, 600])
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, window=20000)
    gate.budget = BlockBudget({**LAYOUT, "groups": [{"type": "FullAttentionSpec", "block_size": 1000}]})
    gate.capacity = 90
    agent.apply_harness_fixes(["exact_trigger"])
    game.levels = 3
    agent.apply_parallel_samples({"abort_grace_seconds": 0.05, "retry_seconds": 0.01, "prefix_warmup": True})
    agent.config.max_actions = 2
    env.observations = iter(boards(2))

    def rule(messages):
        if text_of(messages[-1]).startswith("Your context is full."):
            return 3600  # the summary request
        if any(text_of(m).startswith(SUMMARY_PREAMBLE) for m in messages):
            return 1200  # after the compaction
        return 3105 if len(messages) > 2 else 1000  # over the 3,104 budget after the first tool round

    agent.client.count_tokens, _ = counter(rule)
    result = await agent.run()
    assert result["error"] is None and game.compactions == 1
    summary = [p for p in transport.payloads if p.get("tool_choice") == "none"][0]
    assert "sample 1" in json.dumps(summary["messages"]) and "sample 0" not in json.dumps(summary["messages"])
    assert not [e for e in events_of(agent, "compaction") if e["mechanical_fallback"]]
    await asyncio.sleep(0.1)
    assert gate.reserved == 0


# --- hero-v2 composition ------------------------------------------------------------------------------------------------

def test_hero_v2_component_raises_sequences_and_graphs_and_keeps_run_b_otherwise():
    from test_competition_run_b import RUN_B, SERVED, hero
    from test_competition_run_b import budget as served_budget

    from arena import hero_v2
    from arena.capacity import competition_focus_fit

    on = hero_v2.compose(hero(), {**RUN_B, "token_budgets": False, "parallel_samples": True, "action_plans": True,
                                  "density_blocks": 1266}, served_blocks=SERVED)
    # The server runs 48 sequences with graphs to 48; the regular requests keep run B's 26 generation slots and the
    # siblings get the other 22 (KVGate background_slots, arena.trial).
    assert on["active_limit"] == 24 and on["server"]["max_num_seqs"] == 48 and on["generation_slots"] == 26
    assert on["server"]["max_cudagraph_capture_size"] == 48 and on["action_plans"] is True
    assert on["parallel_samples"] == ps.settings(True)
    record = on["hero_v2"]["components"]["parallel_samples"]
    assert record["generation_slots"] == 26 and record["background_slots"] == 22
    # The startup density keeps run B's 24 streams on a pool of at least 1,266 blocks (1,268 measured at 48/48).
    for blocks, streams in ((1270, 24), (1268, 24), (1266, 24), (1265, 23)):
        fit = {"active_limit": 24, "focus_oversubscription": on["focus_oversubscription"]}
        competition_focus_fit(served_budget(blocks), 130816, fit)
        assert fit["active_limit"] == streams
    off = hero_v2.compose(hero(), {**RUN_B, "token_budgets": False}, served_blocks=SERVED)
    explicit = hero_v2.compose(hero(), {**RUN_B, "token_budgets": False, "parallel_samples": False,
                                        "action_plans": False}, served_blocks=SERVED)
    assert json.dumps(off, sort_keys=True) == json.dumps(explicit, sort_keys=True)
    assert "parallel_samples" not in off and "action_plans" not in off
    assert "parallel_samples" not in off["hero_v2"]["requested"] and "action_plans" not in off["hero_v2"]["components"]
    with pytest.raises(ValueError, match="prefix_cache"):
        hero_v2.compose(hero(), {**RUN_B, "token_budgets": False, "prefix_cache": False, "parallel_samples": True},
                        served_blocks=SERVED)
    with pytest.raises(ValueError):
        hero_v2.compose(hero(), {**RUN_B, "token_budgets": False, "parallel_samples": {"k": 1}}, served_blocks=SERVED)


async def test_no_warmup_reserves_full_context_aborts_losers_and_does_not_measure_them():
    sampler, gate, agent, events = sampler_for({'prefix_warmup': False, 'late_admission_seconds': 0})
    engine = Engine([(0.3, 'tool'), (0.02, 'tool'), (0.3, 'tool'), (0.3, 'tool')])
    await sampler.complete({'messages': [], 'max_tokens': 1000}, engine)
    assert len(engine.calls) == 4 and all(c[0] == 'sample' for c in engine.calls)
    reservations = [e for e in events if e['event'] == 'sibling_admit']
    assert len(reservations) == 3
    assert all(e['tokens'] == gate.budget.cost(1234 + 1000) for e in reservations)
    assert sampler.stats['pings'] == sampler.stats['observed_groups'] == sampler.stats['late_samples'] == 0
    assert await settle(gate) == (0, 0)
    assert engine.finished == [1]


async def test_error_is_recorded_even_if_group_is_cancelled():
    sampler, gate, agent, events = sampler_for({'prefix_warmup': False})
    engine = Engine([(0.01, 'raise'), (0.3, 'tool'), (0.3, 'tool'), (0.3, 'tool')])
    task = asyncio.create_task(sampler.complete({'max_tokens': 1000}, engine))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sampler.stats['sample_errors'] == 1
    assert any(r['kind'] == 'parallel_sample_error' for r in agent.workspace.events)
    assert await settle(gate) == (0, 0)
