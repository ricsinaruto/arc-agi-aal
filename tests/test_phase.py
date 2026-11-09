import asyncio
import json

import httpx
import pytest

from arena import phase
from arena.clock import ActiveClock, GlobalClock
from arena.phase import PhaseController, PhaseGate, PhaseTime


class FakeServer(httpx.AsyncBaseTransport):
    """N play servers behind one transport: a request waits until ``release`` (or forever while held)."""

    def __init__(self):
        self.started, self.cancelled, self.done = [], 0, []
        self.release = asyncio.Event()

    async def handle_async_request(self, request):
        body = json.loads(request.content)
        self.started.append((request.url.port, body["id"]))
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        self.done.append(body["id"])
        return httpx.Response(200, json={"id": body["id"], "port": request.url.port})


def payload(i, start="s0"):
    return {"id": i, "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": start}]}


async def post(client, i, start="s0"):
    r = await client.post("http://x/v1/chat/completions", json=payload(i, start))
    return r.json()


def test_abort_and_resend_after_training():
    async def go():
        server = FakeServer()
        gate = PhaseGate(["http://a:1", "http://b:2"], inner=server)
        client = httpx.AsyncClient(transport=gate)
        tries = [asyncio.create_task(post(client, i, f"s{i % 3}")) for i in range(6)]
        await asyncio.sleep(0.01)
        assert len(server.started) == 6
        assert gate.close() == 6
        await asyncio.wait_for(gate.drained(), 1)
        assert server.cancelled == 6 and not any(t.done() for t in tries)
        late = asyncio.create_task(post(client, 99))  # arrives mid-training: held, never sent
        await asyncio.sleep(0.01)
        assert len(server.started) == 6 and gate.stats["held"] == 7  # 6 aborted + 1 new
        server.release.set()
        gate.open()
        out = await asyncio.wait_for(asyncio.gather(*tries, late), 1)
        assert sorted(o["id"] for o in out) == [0, 1, 2, 3, 4, 5, 99]
        assert gate.stats["aborted"] == 6 and gate.stats["resent"] == 6
        # sticky: each try key went to the same server both times; keys spread over both servers
        ports = {}
        for port, i in server.started:
            ports.setdefault(i % 3 if i != 99 else 0, set()).add(port)
        assert all(len(p) == 1 for p in ports.values())
        assert {p for s in ports.values() for p in s} == {1, 2}
        await client.aclose()
    asyncio.run(go())


def test_caller_cancel_cancels_the_server_request():
    async def go():
        server = FakeServer()
        gate = PhaseGate(inner=server)
        client = httpx.AsyncClient(transport=gate)
        t = asyncio.create_task(post(client, 1))
        await asyncio.sleep(0.01)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        await asyncio.wait_for(gate.drained(), 1)
        assert server.cancelled == 1 and gate.stats["aborted"] == 0
    asyncio.run(go())


def test_phase_time_freezes_try_clocks():
    t = [100.0]
    pt = PhaseTime(now=lambda: t[0])
    wall = GlobalClock(1800, now=pt.monotonic)
    game = ActiveClock(wall, pt.monotonic)
    wall.release()
    game.release()
    t[0] += 50
    pt.pause()
    t[0] += 600  # a train phase
    assert wall.elapsed() == 50 and game.active_seconds() == 50
    pt.resume()
    t[0] += 10
    assert wall.elapsed() == 60 and wall.remaining() == 1740 and pt.paused_total == 600


class Fake:
    def __init__(self, log, name):
        self.log, self.name = log, name

    async def sleep(self):
        self.log.append(f"{self.name}.sleep")

    async def wake(self):
        self.log.append(f"{self.name}.wake")

    async def load(self, adapter, path):
        self.log.append(f"{self.name}.load:{adapter}:{path}")


def test_controller_order_and_clock():
    async def go():
        log, pt = [], PhaseTime()
        server = FakeServer()
        gate = PhaseGate(inner=server)
        client = httpx.AsyncClient(transport=gate)
        live = asyncio.create_task(post(client, 7))
        await asyncio.sleep(0.01)

        async def park():
            log.append("park")

        async def unpark():
            log.append("unpark")

        async def update():
            assert pt.paused and not gate.is_open and len(server.started) == 1
            log.append("update")
            return "/runs/a/v1"

        ctl = PhaseController([gate], [Fake(log, "s0"), Fake(log, "s1")], park, unpark, clock=pt)
        server.release.set()  # would finish now, but the gate already aborted it
        row = await ctl.train(update)
        assert row["stop"]["aborted"] == 1
        assert log == ["s0.sleep", "s1.sleep", "unpark", "update", "park", "s0.wake", "s1.wake",
                       "s0.load:policy:/runs/a/v1", "s1.load:policy:/runs/a/v1"]
        assert (await asyncio.wait_for(live, 1))["id"] == 7 and not pt.paused and ctl.phase == "play"
        await client.aclose()
    asyncio.run(go())


def test_failed_update_keeps_train_phase():
    async def go():
        async def noop():
            return None

        async def bad():
            raise RuntimeError("step failed")

        ctl = PhaseController([PhaseGate()], [], noop, noop, clock=PhaseTime())
        with pytest.raises(RuntimeError):
            await ctl.train(bad)
        assert ctl.phase == "train"
        await ctl.to_play(None)
        assert ctl.phase == "play"
    asyncio.run(go())


def test_before_load_runs_after_wake_and_before_load():
    """mg.async_adapter_save: the trainer's background adapter write is awaited after park and wake, before the
    servers load it; a failed write raises there (nothing loads, the gates stay closed)."""
    async def go():
        log = []

        async def park():
            log.append("park")

        async def noop():
            return None

        async def landed(path):
            log.append(f"save_join:{path}")

        ctl = PhaseController([PhaseGate()], [Fake(log, "s0")], park, noop, clock=PhaseTime())
        await ctl.to_train()
        ctl.before_load = landed
        row = await ctl.to_play("/a/v1")
        assert log == ["s0.sleep", "park", "s0.wake", "save_join:/a/v1", "s0.load:policy:/a/v1"]
        assert "save_wait_s" in row and ctl.phase == "play"

        async def failed(path):
            raise RuntimeError("background adapter save failed")

        log.clear()
        await ctl.to_train()
        ctl.before_load = failed
        with pytest.raises(RuntimeError):
            await ctl.to_play("/a/v2")
        assert "s0.load:policy:/a/v2" not in log and ctl.phase == "train"
    asyncio.run(go())


def test_module_time_matches_monotonic_without_pauses():
    # order-free: rl_seq.run tests in this process pause the module clock, so its paused_total carries over
    import time
    assert abs(PhaseTime().monotonic() - time.monotonic()) < 1.0
    assert not phase.TIME.paused and abs(phase.monotonic() + phase.TIME.paused_total - time.monotonic()) < 1.0


def test_reclose_before_held_requests_resume_keeps_them_held():
    """H5 step 2 (26 Oct): a play phase that ended at once (open, then close with no await between) let the held
    requests, woken by open, go out after close: aborted 0, then the drain waited 82 s for them to finish."""
    async def go():
        server = FakeServer()
        gate = PhaseGate(["http://a:1"], inner=server)
        client = httpx.AsyncClient(transport=gate)
        gate.close()
        tries = [asyncio.create_task(post(client, i)) for i in range(4)]
        await asyncio.sleep(0.01)
        assert gate.stats["held"] == 4 and not server.started
        gate.open()
        assert gate.close() == 0  # nothing ran between open and close
        await asyncio.wait_for(gate.drained(), 1)
        await asyncio.sleep(0.01)
        assert not server.started  # still held: none went out in the closed phase
        server.release.set()
        gate.open()
        out = await asyncio.wait_for(asyncio.gather(*tries), 1)
        assert sorted(o["id"] for o in out) == [0, 1, 2, 3]
        await client.aclose()
    asyncio.run(go())


def chat(i, start, own, n_start=6, size=4000):
    """A carried try: its group's restored start (n_start messages of ``size`` chars), then its own turns."""
    msgs = [{"role": "system", "content": "sys"}]
    msgs += [{"role": "user" if k % 2 else "assistant", "content": f"{start}{k}" * (size // 4)} for k in range(n_start)]
    msgs += [{"role": "assistant", "content": f"own{i}-{k}" * 200} for k in range(own)]
    return {"id": i, "model": "policy", "max_tokens": 4000, "messages": msgs}


class CacheServer(httpx.AsyncBaseTransport):
    """A prefix-caching play server: a request computes the prompt chars past its deepest cached message prefix; its
    prefixes are cached once its prefill is done (vLLM: blocks become hits only after a request computed them)."""

    def __init__(self, per_char=2e-6, decode_s=0.05):
        self.cache, self.computed, self.order, self.warm_ids = set(), 0, [], []
        self.per_char, self.decode_s = per_char, decode_s

    async def handle_async_request(self, request):
        body = json.loads(request.content)
        if request.url.path.endswith("/tokenize"):  # vLLM's reply: no id
            return httpx.Response(200, json={"count": 3, "max_model_len": 9, "tokens": [1, 2, 3]})
        chars, hashes = phase.prompt_chain(request.content)
        hit = max((k for k in range(len(hashes)) if hashes[k] in self.cache), default=-1)
        new = chars[-1] - (chars[hit] if hit >= 0 else 0)
        self.computed += new
        warm = body["max_tokens"] == 1
        self.order.append(("warm" if warm else body["id"], round(new / chars[-1], 2)))
        if warm:
            self.warm_ids.append(request.headers.get("x-request-id"))
            assert body["request_id"] == self.warm_ids[-1] and "stream" not in body
        await asyncio.sleep(new * self.per_char)
        self.cache.update(hashes)
        if not warm:
            await asyncio.sleep(self.decode_s)
        usage = {"prompt_tokens": chars[-1], "prompt_tokens_details": {"cached_tokens": chars[-1] - new}}
        rid = "chatcmpl-" + (request.headers.get("x-request-id") or "x")  # vLLM: chatcmpl-<X-Request-Id>
        return httpx.Response(200, json={"id": rid if warm else body["id"], "usage": usage})


async def carried_switch(leader_first, groups=4, size=8):
    """Play, a train phase (close + drain), then open: every try of ``groups`` carried groups resends its turn."""
    server = CacheServer()
    gate = PhaseGate(inner=server, leader_first=leader_first)
    client = httpx.AsyncClient(transport=gate)
    gate.close()  # the train phase: every try's next turn is held
    tries = [asyncio.create_task(client.post("http://x/v1/chat/completions",
                                             json=chat(g * size + t, f"g{g}-", own=1 + t % 3)))
             for g in range(groups) for t in range(size)]
    await asyncio.sleep(0.01)
    assert len(gate.waiting) == groups * size and not server.order
    gate.open()
    out = await asyncio.wait_for(asyncio.gather(*tries), 5)
    assert all(r.status_code == 200 for r in out)
    await client.aclose()
    return server, gate


def test_leader_plan_clusters_carried_groups():
    a = [json.dumps(chat(i, "A", own=1 + i)).encode() for i in range(3)]
    b = [json.dumps(chat(10 + i, "B", own=2 - i)).encode() for i in range(2)]
    lone = json.dumps(chat(20, "C", own=1)).encode()
    short = json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode()
    plan = phase.leader_plan(a + b + [lone, short, b"not json"], min_share=0.5, min_chars=1000)
    assert plan == [[0, 1, 2], [4, 3]]  # shortest prompt leads; the lone try and the unshared ones are released
    assert phase.leader_plan(a + b, min_share=0.5, min_chars=10 ** 6) == []  # shared prefix under min_chars


def test_leader_first_warms_each_group_once():
    async def go():
        burst, _ = await carried_switch(None)
        lead, gate = await carried_switch({"min_chars": 1000})
        return burst, lead, gate
    burst, lead, gate = asyncio.run(go())
    # the burst (real-v15 phase 16): all 32 tries compute their group's start side by side
    assert all(share == 1.0 for _, share in burst.order)
    # leader-first: one warm-up per group first, then every try hits its group's start
    warm = [k for k, (who, _) in enumerate(lead.order) if who == "warm"]
    assert warm == [0, 1, 2, 3] and all(w.startswith(phase.WARM_ID) for w in lead.warm_ids)
    assert all(share < 0.3 for who, share in lead.order[4:])
    assert lead.computed < 0.35 * burst.computed
    row = gate.plans[-1]
    assert row["held"] == 32 and row["clusters"] == 4 and row["followers"] == 32 and row["warm"] == 4
    assert row["warm_failed"] == 0 and row["follow_timeout"] == 0 and row["warm_cached"] == 0


def test_leader_first_close_during_warmup_reholds():
    async def go():
        server = CacheServer(per_char=2e-5)  # warm-ups still prefilling at the close
        gate = PhaseGate(inner=server, leader_first={"min_chars": 1000})
        client = httpx.AsyncClient(transport=gate)
        gate.close()
        tries = [asyncio.create_task(client.post("http://x/v1/chat/completions", json=chat(i, "A", own=1 + i)))
                 for i in range(4)]
        await asyncio.sleep(0.01)
        gate.open()
        await asyncio.sleep(0.01)
        assert [w for w, _ in server.order] == ["warm"] and not gate.waiting
        gate.close()  # a phase ends mid warm-up: the warm-up is cancelled and the cluster held again
        await asyncio.wait_for(gate.drained(), 1)
        await asyncio.sleep(0.01)
        assert not gate.warming and len(gate.waiting) == 4 and not gate.follow
        server.per_char = 2e-6
        gate.open()
        out = await asyncio.wait_for(asyncio.gather(*tries), 5)
        assert [r.status_code for r in out] == [200] * 4
        assert [w for w, _ in server.order[:2]] == ["warm", "warm"] and len(gate.plans) == 2
        await client.aclose()
    asyncio.run(go())


def test_leader_first_failed_warmup_releases_cluster():
    class Refuse(CacheServer):
        async def handle_async_request(self, request):
            if json.loads(request.content)["max_tokens"] == 1:
                return httpx.Response(500, json={})
            return await super().handle_async_request(request)

    async def go():
        server = Refuse()
        gate = PhaseGate(inner=server, leader_first={"min_chars": 1000})
        client = httpx.AsyncClient(transport=gate)
        gate.close()
        tries = [asyncio.create_task(client.post("http://x/v1/chat/completions", json=chat(i, "A", own=1)))
                 for i in range(3)]
        await asyncio.sleep(0.01)
        gate.open()
        out = await asyncio.wait_for(asyncio.gather(*tries), 5)
        assert [r.status_code for r in out] == [200] * 3 and gate.plans[-1]["warm_failed"] == 1
    asyncio.run(go())


def test_leader_first_off_keeps_the_burst():
    gate = PhaseGate()
    assert gate.leader_first is None
    gate = PhaseGate(leader_first=True)
    assert gate.leader_first == phase.LEADER_FIRST


def test_leader_first_stops_when_the_server_ignores_the_warm_id():
    class NoIds(CacheServer):
        async def handle_async_request(self, request):
            got = await super().handle_async_request(request)
            body = json.loads(got.content)
            body["id"] = "chatcmpl-random"
            return httpx.Response(200, json=body)

    async def go():
        server = NoIds()
        gate = PhaseGate(inner=server, leader_first={"min_chars": 1000})
        client = httpx.AsyncClient(transport=gate)
        gate.close()
        tries = [asyncio.create_task(client.post("http://x/v1/chat/completions", json=chat(i, "A", own=1)))
                 for i in range(3)]
        await asyncio.sleep(0.01)
        gate.open()
        await asyncio.wait_for(asyncio.gather(*tries), 5)
        assert gate.plans[-1]["warm_id_bad"] == 1 and gate.leader_first is None  # the next opens burst as before
    asyncio.run(go())


def test_kv_gate_hold_admits_nothing_new_while_in_flight():
    from arena.gate import KVGate

    async def go():
        gate = KVGate(100, slots=10)
        a = await gate.acquire("g", 10)
        gate.hold(True)
        waiter = asyncio.ensure_future(gate.acquire("g", 10))
        await asyncio.sleep(0.01)
        assert not waiter.done()
        gate.hold(False)  # released: the queue is pumped
        b = await asyncio.wait_for(waiter, 1)
        gate.release(a)
        gate.release(b)
        gate.hold(True)  # nothing in flight: a held gate still admits one (no deadlock)
        c = await asyncio.wait_for(gate.acquire("g", 10), 1)
        assert gate.inflight == 1 and c.tokens == 10
    asyncio.run(go())


def test_leader_first_clusters_chat_requests_only():
    """N1c (29 Oct): the bench's /tokenize calls carry a try's messages; one led a cluster, its warm-up went to
    /tokenize (a reply without an id) and leader_first switched itself off."""
    async def go():
        server = CacheServer()
        gate = PhaseGate(inner=server, leader_first={"min_chars": 1000})
        client = httpx.AsyncClient(transport=gate)
        gate.close()
        tok = asyncio.create_task(client.post("http://x/tokenize", json={**chat(99, "A", own=0), "max_tokens": 4000}))
        tries = [asyncio.create_task(client.post("http://x/v1/chat/completions", json=chat(i, "A", own=1 + i)))
                 for i in range(3)]
        await asyncio.sleep(0.01)
        gate.open()
        await asyncio.wait_for(asyncio.gather(tok, *tries), 5)
        row = gate.plans[-1]
        assert row["held"] == 3 and row["followers"] == 3 and row["warm"] == 1 and "warm_id_bad" not in row
        assert gate.leader_first is not None
    asyncio.run(go())


def test_switch_reads_card_memory_asleep_and_awake(tmp_path):
    """Per-phase VRAM (29 Oct, +12 GiB vLLM residue after sleep): read after the sleep (before unpark) and after the
    wake (before the load); a failing reader never stops the switch."""
    from arena import rl_seq

    async def go():
        log = []

        async def noop():
            return None

        async def mem():
            log.append("mem")
            return [12.5, 3.0]

        async def update():
            return "/runs/a/v1"

        ctl = PhaseController([PhaseGate()], [Fake(log, "s0")], noop, noop, clock=PhaseTime(), memory=mem)
        row = await ctl.train(update)
        assert row["stop"]["asleep_gib"] == [12.5, 3.0] and row["start"]["awake_gib"] == [12.5, 3.0]
        assert log == ["s0.sleep", "mem", "s0.wake", "mem", "s0.load:policy:/runs/a/v1"]

        async def bad():
            raise RuntimeError("smi")

        ctl = PhaseController([PhaseGate()], [Fake(log, "s0")], noop, noop, clock=PhaseTime(), memory=bad)
        row = await ctl.train(update)
        assert "asleep_gib" not in row["stop"] and "awake_gib" not in row["start"]
    asyncio.run(go())
    smi = tmp_path / "smi"
    smi.write_text("#!/bin/sh\necho 12800\necho 3072\n")
    smi.chmod(0o755)
    assert rl_seq.cards_used(str(smi)) == [12.5, 3.0] and rl_seq.cards_used(str(tmp_path / "none")) is None
