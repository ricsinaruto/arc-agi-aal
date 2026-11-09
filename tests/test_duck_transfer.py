"""The public Duck loop bridged into the fleet: notes transfer, batching, draining, trimming."""

import asyncio
import json
import time
from copy import deepcopy
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from conftest import duck_agent_for, observation, tool_reply

from arc_infra.perception import Observation


async def test_transfer_notes_reach_next_level_without_extra_model_call(tmp_path):
    class Transport:
        calls = 0

        async def post(self, url, **kwargs):
            self.calls += 1
            assert self.calls <= 2
            if self.calls == 2:
                messages = json.dumps(kwargs["json"]["messages"])
                assert "Prior-level hypotheses, not verified for the new level" in messages
                assert "Actions: click toggles one tile" in messages
            return tool_reply(
                "action([{'action':'MOUSE','row':4,'col':9}])",
                call_id=f"click-{self.calls}",
                content="World model: tiles have two states\nAction model: click toggles one tile",
                usage={"completion_tokens": 50},
            )

    transport = Transport()
    agent, env = duck_agent_for(tmp_path, transport, levels=1)
    env.observations = iter([observation(levels=1), observation(levels=5, state="WIN")])
    agent.config.max_actions = 2
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 2
    assert result["agent_profile"] == "duck-transfer"
    assert transport.calls == len(env.calls) == 2
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    transfers = [e for e in events if e["kind"] == "transfer_notes"]
    assert len(transfers) == 1 and transfers[0]["step"] == 1


async def test_transfer_notes_lifecycle_preserves_explicit_model_notes(tmp_path):
    agent, _ = duck_agent_for(tmp_path, None)
    analyzer = agent.analyzer(asyncio.get_running_loop())
    analyzer._ensure_session(tmp_path / "state.json")
    knowledge = analyzer._summarized_knowledge
    knowledge.update(world_model="first rule", action_model="move by two", current_plan="old route")
    analyzer._last_step_summary = {"board_changed": True}
    analyzer._update_summarized_knowledge_from_step_summary()
    assert not knowledge["cross_level_notes"]
    analyzer._last_step_summary = {"level_transition": True}
    analyzer._update_summarized_knowledge_from_step_summary()
    assert "first rule" in knowledge["cross_level_notes"]
    assert not knowledge["world_model"] and not knowledge["current_plan"]
    knowledge["world_model"] = "updated rule " * 300
    analyzer._update_summarized_knowledge_from_step_summary()
    assert "first rule" not in knowledge["cross_level_notes"]
    assert len(knowledge["cross_level_notes"]) < 1650
    knowledge.update(world_model="updated rule", goal_model="fill the row", current_plan="click twice")
    analyzer._last_step_summary = {"game_over": True, "end_action_num": 7, "executed_actions": ["UP", "UP"]}
    analyzer._update_summarized_knowledge_from_step_summary()
    # v5: GAME_OVER keeps the mechanics and the transfer note; only the plan is dropped.
    assert "first rule" not in knowledge["cross_level_notes"] and knowledge["cross_level_notes"]
    assert knowledge["world_model"].startswith("updated rule") and knowledge["goal_model"] == "fill the row"
    assert knowledge["current_plan"] == ""
    assert knowledge["recent_findings"].startswith("GAME_OVER at step 7 after actions [UP, UP]")
    analyzer._last_step_summary = {"run_complete": True}
    analyzer._update_summarized_knowledge_from_step_summary()
    assert not knowledge["cross_level_notes"]
    knowledge.update(cross_level_notes="model verified this rule", world_model="different hypothesis")
    analyzer._last_step_summary = {"level_transition": True}
    analyzer._update_summarized_knowledge_from_step_summary()
    assert knowledge["cross_level_notes"] == "model verified this rule"
    analyzer._ensure_session(tmp_path / "new-game/state.json")
    assert not any(analyzer._summarized_knowledge.values())
    agent.executor.shutdown()


async def test_batch_stops_at_level_change_and_invalid_click_executes_nothing(tmp_path):
    agent, env = duck_agent_for(tmp_path, None, levels=1)
    agent.config.max_actions = 100
    agent.env = env
    agent.obs = Observation.from_raw(env.observation_space)
    agent.state_path = tmp_path / "tool_runtime_state.json"
    agent.write_duck_state()
    invalid = await agent.step_duck(
        {"actions": [{"action": "UP"}, {"action": "MOUSE", "row": -1, "col": 3}]}
    )
    assert invalid["error"] and not env.calls
    result = await agent.step_duck({"actions": [{"action": "UP"}] * 3})
    assert result["level_completed"] and result["stopped_early"]
    assert result["executed_count"] == 1 and len(env.calls) == 1
    assert result["requested_count"] == 3 and result["stop_reason"] == "level_completed"
    assert agent.duck_history[-1].frame.level == 2
    agent.stopping.set()
    result = await agent.step_duck({"action": "UP"})
    assert not result["executed"] and len(env.calls) == 1


async def test_cancel_drains_model_bridge_before_scorecard_can_close(tmp_path):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class Transport:
        async def post(self, *args, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    agent, env = duck_agent_for(tmp_path, Transport())
    task = asyncio.create_task(agent.run())
    await asyncio.wait_for(entered.wait(), 3)
    task.cancel()
    result = await asyncio.wait_for(task, 3)
    assert result["status"] == "time_limit" and result["error"] is None
    assert cancelled.is_set() and agent.chat_future is None
    assert agent.sandbox.closed and agent.sandbox.process is None
    assert agent.worker is None or agent.worker.done()  # the actor thread was drained
    assert not env.calls


@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize("error_class", [httpx.ReadTimeout, httpx.ConnectTimeout, RuntimeError])
async def test_request_timeout_at_deadline_is_not_a_cleanup_failure(tmp_path, expired, error_class):
    class Transport:
        async def post(self, *args, **kwargs):
            if expired:
                agent.deadline = time.monotonic() - 1
            raise error_class("scripted request failure")

    agent, env = duck_agent_for(tmp_path, Transport())
    result = await asyncio.wait_for(agent.run(), 3)
    expected_deadline = expired and issubclass(error_class, httpx.TimeoutException)
    assert result["status"] == ("time_limit" if expected_deadline else "error")
    assert (result["error"] is None) == expected_deadline
    assert not env.calls and result["actions"] == 0
    assert agent.worker is None or agent.worker.done()
    assert agent.chat_future is None and agent.sandbox.closed
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    assert sum(e["kind"] == "deadline_request_timeout" for e in events) == int(expected_deadline)


async def test_deadline_timeout_preserves_actions_without_replaying(tmp_path):
    class Transport:
        calls = 0

        async def post(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return tool_reply("action([{'action':'UP'}])")
            agent.deadline = time.monotonic() - 1
            raise httpx.ReadTimeout("deadline")

    transport = Transport()
    agent, env = duck_agent_for(tmp_path, transport)
    agent.config.max_actions = 2
    result = await asyncio.wait_for(agent.run(), 3)
    assert result["status"] == "time_limit" and result["error"] is None
    assert result["actions"] == len(env.calls) == 1 and transport.calls == 2


async def test_cancelled_worker_drains_an_expired_uncaught_request(tmp_path):
    """Compaction can raise outside the public analyzer's exception handler."""
    class Transport:
        async def post(self, *args, **kwargs):
            agent.deadline = time.monotonic() - 1
            task.cancel()  # race between runner cancellation and the HTTP timeout
            raise httpx.ReadTimeout("deadline")

    agent, env = duck_agent_for(tmp_path, Transport())

    def analyzer(loop):
        def analyze(*args, **kwargs):
            return asyncio.run_coroutine_threadsafe(agent.request_completion({}), loop).result()
        return SimpleNamespace(analyze=analyze)

    agent.analyzer = analyzer
    task = asyncio.create_task(agent.run())
    result = await asyncio.wait_for(task, 3)
    assert result["status"] == "time_limit" and result["error"] is None
    assert agent.worker is None and not env.calls
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    assert any(e["kind"] == "worker_deadline_drained" for e in events)


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("early"), RuntimeError("bug"), None])
async def test_worker_cleanup_does_not_hide_early_failures_or_hangs(tmp_path, failure):
    agent, _ = duck_agent_for(tmp_path, None)
    agent.deadline = time.monotonic() - 1
    agent.status = "time_limit"
    worker = agent.worker = asyncio.get_running_loop().create_future()
    if failure is not None:
        worker.set_exception(failure)
    try:
        await agent.drain_worker(timeout=0.01)
        expected = type(failure).__name__ if failure else "TimeoutError"
        assert agent.status == "error" and agent.error == f"Duck worker cleanup: {expected}"
        # A cleanup wait timeout is not a completed actor, nor authority to replay it.
        assert (agent.worker is None) == (failure is not None)
    finally:
        worker.cancel()
        agent.executor.shutdown()


async def test_a_worker_that_ran_into_the_deadline_is_a_time_limit_not_a_cleanup_error(tmp_path):
    """dfr-combined v3 Kaggle preview (3 October): a request returned at the wall, its tool call then raised the
    deadline's TimeoutError; the game ended on time, so the status stays time_limit (an event records the drain)."""
    agent, _ = duck_agent_for(tmp_path, None)
    agent.deadline = time.monotonic() - 1
    agent.status = "time_limit"
    worker = agent.worker = asyncio.get_running_loop().create_future()
    worker.set_exception(TimeoutError())
    try:
        await agent.drain_worker(timeout=0.01)
        assert agent.status == "time_limit" and agent.error is None and agent.worker is None
        events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
        assert any(e["kind"] == "worker_deadline_timeout" for e in events)
    finally:
        agent.executor.shutdown()


async def test_a_deadline_timeout_before_the_wall_is_still_a_cleanup_error(tmp_path):
    agent, _ = duck_agent_for(tmp_path, None)
    agent.deadline = time.monotonic() + 60
    worker = agent.worker = asyncio.get_running_loop().create_future()
    worker.set_exception(TimeoutError())
    try:
        await agent.drain_worker(timeout=0.01)
        assert agent.status == "error" and agent.error == "Duck worker cleanup: TimeoutError"
    finally:
        agent.executor.shutdown()


async def test_cancelling_cleanup_waiter_is_not_proof_worker_stopped(tmp_path):
    agent, _ = duck_agent_for(tmp_path, None)
    agent.stopping.set()
    worker = agent.worker = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(agent.drain_worker())
    try:
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert agent.worker is worker and not worker.done()
    finally:
        worker.cancel()
        agent.executor.shutdown()


async def test_25_duck_games_share_eight_slots_and_drain_all_workers(tmp_path):
    active = peak = 0

    class Transport:
        async def post(self, *args, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return tool_reply("action([{'action':'UP'}])", usage={"completion_tokens": 100, "prompt_tokens": 2000})

    slots = asyncio.Semaphore(8)
    agents = []
    for i in range(25):
        agent, _ = duck_agent_for(tmp_path / str(i), Transport())
        agent.client.slots = slots
        agents.append(agent)
    results = await asyncio.wait_for(asyncio.gather(*(a.run() for a in agents)), 12)
    assert all(r["actions"] == 1 and r["error"] is None for r in results)
    assert 1 < peak <= 8 and active == 0
    assert all(a.worker is None and a.chat_future is None for a in agents)


async def test_duck_loop_recovers_explicit_context_rejection(tmp_path):
    payloads = []

    class Transport:
        async def post(self, *args, **kwargs):
            payloads.append(deepcopy(kwargs["json"]))
            if len(payloads) == 1:
                assert "old evidence" in str(payloads[-1]["messages"])
                return httpx.Response(
                    400,
                    text="Maximum context length exceeded; reduce the length of the input prompt",
                    request=httpx.Request("POST", "http://local/chat"),
                )
            assert len(payloads) == 2
            assert "old evidence" not in str(payloads[-1]["messages"])
            assert "RUNTIME PLANNING BUDGET" not in str(payloads[-1]["messages"])
            return tool_reply("action([{'action':'UP'}])")

    agent, env = duck_agent_for(tmp_path, Transport())
    make_analyzer = agent.analyzer

    def with_previous_history(loop):
        analyzer = make_analyzer(loop)
        analyzer._ensure_session(agent.state_path)
        analyzer._history_messages = [
            {"role": "user", "content": "old evidence"},
            {"role": "assistant", "content": "old hypothesis"},
        ]
        return analyzer

    agent.analyzer = with_previous_history
    result = await agent.run()
    assert result["error"] is None and result["actions"] == len(env.calls) == 1
    assert result["model_calls"] == 1 and agent.backend_error is None
    # The retried request still carries the current observation (a user message).
    assert any(m["role"] == "user" for m in payloads[1]["messages"])
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    assert sum(e["kind"] == "context_rejection" for e in events) == 1


@pytest.mark.parametrize(
    "status,detail",
    [
        (400, "Maximum context length exceeded"),  # No removable history: fail once.
        (400, "Invalid tool schema"),
        (500, "Maximum context length backend failure"),
    ],
)
async def test_rejected_or_ambiguous_requests_do_not_loop_or_execute_actions(
    tmp_path, status, detail
):
    calls = 0

    class Transport:
        async def post(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            assert calls == 1
            return httpx.Response(
                status, text=detail, request=httpx.Request("POST", "http://local/chat")
            )

    agent, env = duck_agent_for(tmp_path, Transport())
    result = await asyncio.wait_for(agent.run(), 5)
    assert result["status"] == "error" and result["error"]
    assert calls == 1 and result["actions"] == 0 and not env.calls
    assert agent.worker is None and agent.chat_future is None


def long_turn(exchanges, tool_chars):
    """system, one observation, then ``exchanges`` python-tool exchanges with big outputs."""
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": [{"type": "text", "text": "observation 1"}]},
    ]
    for i in range(exchanges):
        messages.append(
            {
                "role": "assistant",
                "content": f"inspect {i}",
                "tool_calls": [
                    {"id": f"c{i}", "type": "function", "function": {"name": "python", "arguments": "{}"}}
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"out{i} " * tool_chars})
    return messages


async def test_long_no_action_turn_keeps_the_latest_observation(tmp_path):
    agent, _ = duck_agent_for(tmp_path, None)
    analyzer = agent.analyzer(asyncio.get_running_loop())
    analyzer._context_budget_tokens = 3000
    old = [{"role": "user", "content": "old observation"}, {"role": "assistant", "content": "old"}]
    messages = [long_turn(0, 0)[0], *old, *long_turn(6, 400)[1:]]
    before = deepcopy(messages)
    trimmed = analyzer._trim_messages_for_context(messages, tools=None)
    assert messages == before  # never mutated in place
    roles = [m["role"] for m in trimmed]
    assert roles[0] == "system" and roles[1] == "user"
    assert trimmed[1]["content"][0]["text"] == "observation 1"
    assert "old observation" not in json.dumps(trimmed)
    assert roles[-2:] == ["assistant", "tool"] and trimmed[-2]["content"] == "inspect 5"
    assert analyzer._estimate_request_input_tokens(trimmed, tools=None) <= 3000
    # Unlike the public agent, the history never collapses to the system prompt alone.
    forced = analyzer._force_reduce_messages(trimmed)
    assert forced != trimmed and forced[1]["content"][0]["text"] == "observation 1"
    # One huge tool output is truncated rather than dropping the observation.
    single = [long_turn(0, 0)[0], *long_turn(1, 5000)[1:]]
    trimmed = analyzer._trim_messages_for_context(single, tools=None)
    assert [m["role"] for m in trimmed] == ["system", "user", "assistant", "tool"]
    assert trimmed[-1]["content"].endswith(analyzer.TRUNCATION_MARK)
    assert single[-1]["content"].startswith("out0 out0")  # the original is intact
    agent.competition_clock = None
    agent.executor.shutdown()


async def test_timing_note_lands_on_the_latest_observation_after_trimming(tmp_path):
    from duck_transfer.timing import CompetitionClock

    agent, _ = duck_agent_for(tmp_path, None)
    analyzer = agent.analyzer(asyncio.get_running_loop())
    analyzer._context_budget_tokens = 1500
    trimmed = analyzer._trim_messages_for_context(long_turn(8, 300), tools=None)
    annotated = CompetitionClock(0.0, 25).annotate(trimmed, 10.0)
    assert annotated[1]["content"][-1]["text"].startswith("\n\n[RUNTIME PLANNING BUDGET")
    assert json.dumps(annotated).count("RUNTIME PLANNING BUDGET") == 1
    agent.executor.shutdown()


def sandbox_for(agent):
    return agent.sandbox


async def test_python_state_persists_across_tool_calls(tmp_path):
    class Transport:
        calls = 0

        async def post(self, url, **kwargs):
            self.calls += 1
            payload = kwargs["json"]
            assert payload["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "low"}
            assert payload["max_tokens"] == 8192
            prompt = payload["messages"][0]["content"]
            assert "starts fresh" not in prompt and "ephemeral" not in prompt
            assert "Python state persists" in prompt
            assert "persistent interpreter" in payload["tools"][0]["function"]["description"]
            if self.calls == 1:
                return tool_reply("def f():\n    return 41\nx = f()\nprint('first', x)")
            return tool_reply("print('second', x + 1)\naction([{'action':'UP'}])", call_id="act")

    transport = Transport()
    agent, env = duck_agent_for(tmp_path, transport)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1 and transport.calls == 2
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    python = [e for e in events if e["kind"] == "duck_python"]
    assert python[0]["result"]["stdout"] == "first 41\n" and python[0]["sandbox_calls"] == 1
    assert python[1]["result"]["stdout"] == "second 42\n" and python[1]["restarts"] == 0
    assert not any(e["kind"] == "duck_python_reset" for e in events)
    replies = [e for e in events if e["kind"] == "model"]
    assert replies[0]["reasoning_effort"] == "low" and "thinking_budget" not in replies[0]
    assert agent.sandbox.process is None and agent.sandbox.closed


def test_timeout_resets_python_state_and_close_reaps_the_child():
    from duck_transfer.history import preserve_transition_results
    from duck_transfer.loader import load
    from duck_transfer.sandbox import RESET_AFTER_TIMEOUT, StatefulSandbox, stateful_bootstrap

    module = load()
    original = module.run_sandboxed_python.__globals__
    events = []
    sandbox = StatefulSandbox(
        stateful_bootstrap(preserve_transition_results(original["_SANDBOX_BOOTSTRAP"])),
        color_chars=original["ARC_COLOR_CHARS"],
        on_event=lambda kind, **data: events.append((kind, data)),
    )
    state = {
        "current_frame": {"grid": [[0] * 64] * 64, "step": 3, "level": 1},
        "history": [],
        "valid_actions": ["UP"],
        "last_action_result": {},
    }
    run = lambda code, timeout=5, s=state: sandbox.run(  # noqa: E731
        code=code, timeout_seconds=timeout, initial_state=s, action_handler=None
    )
    first = run("x = 1\nresult = 7\nprint(current_frame.step)")
    assert first["stdout"] == "3\n" and first["result"] == 7 and first["error"] == ""
    pid = sandbox.process.pid
    later = {**state, "current_frame": {"grid": [[0] * 64] * 64, "step": 9, "level": 2}}
    second = run("print(x, result, current_frame.step, current_frame.level)", s=later)
    assert second["stdout"] == "1 None 9 2\n"  # definitions persist, runtime state is fresh
    assert sandbox.process.pid == pid
    timed_out = run("while True:\n    pass", timeout=1)
    assert RESET_AFTER_TIMEOUT in timed_out["error"] and sandbox.restarts == 1
    assert [(kind, data) for kind, data in events if kind not in ("duck_python_heap", "duck_python_rng")] == [
        ("duck_python_reset", {"reason": "timeout", "restarts": 1})]
    assert sandbox.process is None
    third = run("print(x)")
    assert "NameError" in third["error"] and sandbox.process.pid != pid
    process = sandbox.process
    sandbox.close()
    assert sandbox.closed and sandbox.process is None and sandbox.tempdir is None
    assert process.poll() is not None
    assert run("print(1)")["error"] == "Sandbox closed"


async def test_runtime_state_stays_in_memory_and_prompt_log_is_written_once_per_turn(tmp_path):
    class Transport:
        calls = 0

        async def post(self, url, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return tool_reply("print(len(history), history[-1].frame.step)")
            return tool_reply("action([{'action':'UP'}])\nprint(len(history))", call_id="act")

    transport = Transport()
    agent, env = duck_agent_for(tmp_path, transport)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1
    # One write at the start (analyze needs the file) and one at the end; never per action.
    assert result["state_flushes"] == 2 and agent.state_flushes == 2
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    python = [e for e in events if e["kind"] == "duck_python"]
    assert python[0]["result"]["stdout"] == "1 0\n" and python[0]["history_synced"] == 1
    # The action reply carried only the new entry; the child holds the full history.
    assert python[1]["result"]["stdout"] == "2\n" and python[1]["history_synced"] == 2
    assert agent.sandbox.delta_entries == 2  # the rendered tool output omits the raw fields
    # Two model requests, one turn, one prompt-log snapshot.
    assert transport.calls == 2 and result["prompt_log_writes"] == 1
    assert (agent.workspace.root / "prompts" / "prompt.log").exists() or any(
        p.name == "prompt.log" for p in agent.workspace.root.rglob("prompt.log")
    )
    from inference.agent.runtime_state import load_runtime_state

    frame, history = load_runtime_state(agent.state_path)
    assert frame.step == 1 and [entry.action for entry in history] == ["", "UP"]


def test_evicted_history_payloads_rebuild_exactly_for_a_sandbox_restart(tmp_path):
    agent, env = duck_agent_for(tmp_path, None)
    agent.obs = Observation.from_raw(env.observation_space)
    expected = []
    try:
        for step in range(agent.HISTORY_PAYLOAD_CACHE * 3):
            agent.steps = step
            agent.obs.board[0, 0] = step % 16
            agent.write_duck_state("UP", {"executed": True, "action_num": step})
            entry = agent.duck_history[-1]
            raw = agent.module._ascii_history_view_payload([entry])[0]
            raw["result"] = dict(agent.transition_results[step])
            expected.append(raw)
            assert json.dumps(agent.history_payload(entry)) == json.dumps(raw)
        assert len(agent.frame_payloads) == agent.HISTORY_PAYLOAD_CACHE
        assert 0 not in agent.frame_payloads
        # A fresh child needs the entire history, including entries evicted from the render cache.
        actual = agent.sandbox._delta({"history": agent.ascii_history(agent.duck_history)})
        assert actual["history_base"] == 0
        assert json.dumps(actual["history"]) == json.dumps(expected)
        assert len(agent.duck_history) == len(expected)
        assert len(agent.frame_payloads) == agent.HISTORY_PAYLOAD_CACHE
        assert agent.sandbox._delta({"history": agent.ascii_history(agent.duck_history)})["history"] == []
    finally:
        agent.sandbox.close()
        agent.executor.shutdown()


def test_sandbox_delta_history_keeps_frames_and_memoises_segmentation():
    from duck_transfer.history import preserve_transition_results
    from duck_transfer.loader import load
    from duck_transfer.sandbox import StatefulSandbox, stateful_bootstrap

    module = load()
    original = module.run_sandboxed_python.__globals__
    sandbox = StatefulSandbox(
        stateful_bootstrap(preserve_transition_results(original["_SANDBOX_BOOTSTRAP"])),
        color_chars=original["ARC_COLOR_CHARS"],
    )
    grid = [[0] * 64 for _ in range(64)]
    grid[3][3] = 5

    def frame(step):
        return {"grid": grid, "step": step, "level": 1, "ascii": "", "shape": [64, 64]}

    def state(step, count):
        return {
            "current_frame": frame(step),
            "history": [{"action": "UP" if i else "", "frame": frame(i)} for i in range(count)],
            "valid_actions": ["UP"],
            "last_action_result": {},
        }

    sent = []
    send = sandbox._send
    sandbox._send = lambda payload: (sent.append(payload), send(payload))
    run = lambda code, s, timeout=5: sandbox.run(  # noqa: E731
        code=code, timeout_seconds=timeout, initial_state=s, action_handler=None
    )
    first = run(
        "print(len(history), len(current_frame.segmentation['nodes']), "
        "len(history[0].frame.segmentation['nodes']))",
        state(1, 2),
    )
    assert first["stdout"] == "2 2 2\n" and first["error"] == ""
    assert first["segmentation"] == {"hits": 1, "misses": 1}  # same grid, one containment run
    assert len(sent[-1]["state"]["history"]) == 2 and sent[-1]["state"]["history_base"] == 0
    second = run(
        "print(len(history), history[0].frame.segmentation is history[1].frame.segmentation)",
        state(2, 3),
    )
    assert second["stdout"] == "3 True\n" and second["history_size"] == 3
    assert len(sent[-1]["state"]["history"]) == 1 and sent[-1]["state"]["history_base"] == 2
    assert sandbox.synced == 3 and sandbox.delta_entries == 3
    # A shorter history (fresh public-shaped state) is resent in full and replaces the child's.
    third = run("print(len(history))", state(0, 1))
    assert third["stdout"] == "1\n" and sent[-1]["state"]["history_base"] == 0
    pid = sandbox.process.pid
    timed_out = run("while True:\n    pass", state(0, 1), timeout=1)
    assert "timed out" in timed_out["error"] and sandbox.synced == 0
    fourth = run("print(len(history))", state(2, 3))
    assert fourth["stdout"] == "3\n" and sandbox.process.pid != pid
    assert sent[-1]["state"]["history_base"] == 0 and len(sent[-1]["state"]["history"]) == 3
    sandbox.close()


async def test_token_estimate_is_linear_and_costs_images_a_constant(tmp_path, monkeypatch):
    agent, _ = duck_agent_for(tmp_path, None)
    analyzer = agent.analyzer(asyncio.get_running_loop())
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 12000}}
    text = {"type": "text", "text": "observation " * 30}
    messages = [
        {"role": "system", "content": "rules"},
        *[{"role": "user", "content": [text, image]} for _ in range(8)],
    ]
    estimate = analyzer._estimate_request_input_tokens(messages, tools=None)
    per_user = analyzer._message_tokens(messages[1])
    assert per_user < 400 + len(text["text"]) // 3 + 40  # the image counts as 400 tokens
    assert estimate == analyzer._message_tokens(messages[0]) + 8 * per_user
    # Memoised: a second pass serialises nothing.
    import duck_transfer.trimming as trimming

    monkeypatch.setattr(trimming.json, "dumps", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert analyzer._estimate_request_input_tokens(messages, tools=None) == estimate
    # Eight images fit a budget that the public estimator (about 3,300 per image) would blow.
    analyzer._context_budget_tokens = 8000
    assert analyzer._trim_messages_for_context(messages, tools=None) == messages
    agent.executor.shutdown()


async def test_tokenizer_precheck_trims_before_sending_and_calibrates_the_estimate(tmp_path):
    """v5b: the server tokenizer counts every request; an over-budget prompt is trimmed
    client-side (the public recovery path) and the counts calibrate the estimator."""
    # The first request cannot be trimmed; the over-budget count lands on the second, whose
    # long tool output the recovery path truncates before resending.
    counts = iter([100, 70_000, 100, 100])
    seen = []

    def tokenize(payload):
        seen.append(payload)
        return next(counts)

    class Transport:
        calls = 0
        template_kwargs = []

        async def post(self, url, **kwargs):
            self.calls += 1
            self.template_kwargs.append((kwargs.get("json") or {}).get("chat_template_kwargs"))
            if self.calls == 1:
                return tool_reply("print('x' * 5000)", usage={"completion_tokens": 50, "prompt_tokens": 100 + 700})
            return tool_reply("action([{'action':'UP'}])", call_id="act", usage={"completion_tokens": 5, "prompt_tokens": 800})

    transport = Transport()
    agent, env = duck_agent_for(tmp_path, transport, tokenize=tokenize)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    prechecks = [e for e in events if e["kind"] == "context_precheck"]
    assert len(prechecks) == 1 and prechecks[0]["count"] == 70_000 and prechecks[0]["images"] == 1
    assert prechecks[0]["limit"] == 65536 - 8192 - 512 and prechecks[0]["estimate"] > prechecks[0]["limit"]
    assert result["context_prechecks"] == 1 and result["token_count_errors"] == 0
    # The tokenizer saw text only: every image part became "[image]", with the tools attached.
    assert all(
        part.get("type") != "image_url"
        for payload in seen for m in payload["messages"] if isinstance(m.get("content"), list) for part in m["content"]
    )
    assert any("[image]" in json.dumps(payload["messages"]) for payload in seen) and "tools" in seen[0]
    # v5d: the count uses the request's own chat template kwargs (v5c counted with
    # preserve_thinking false while the request left it undefined, missing every earlier
    # reasoning block), and its character total prices non-content fields like the trimmer.
    assert [p["chat_template_kwargs"] for p in seen[:2]] == transport.template_kwargs[:2]
    assert transport.template_kwargs[0].get("preserve_thinking") is not False
    # The rejected request was recovered (public path) and the model was still called twice.
    assert transport.calls == 2 and any(e["kind"] == "context_rejection" for e in events)
    # v5c calibration: image tokens come from the PNG geometry (a 64x64 board at upscale 4 is
    # 256x256 pixels: 8x8 vision tokens plus two markers), the residual of prompt_tokens
    # against text count + image tokens is tracked, and every sample is logged.
    samples = [e for e in events if e["kind"] == "token_count"]
    assert len(samples) == 2 and result["token_count_samples"] == 2
    assert samples[0]["count"] == 100 and samples[0]["images"] == 1 and samples[0]["image_tokens"] == 66
    assert samples[0]["prompt_tokens"] == 800 and samples[0]["residual"] == 800 - 100 - 66
    assert result["image_tokens"] == 66 and result["count_residual"] > 0
    # The second sample's characters include the assistant tool call and tool result fields.
    assert samples[1]["text_chars"] > samples[0]["text_chars"] + 100
    assert 1.0 <= result["chars_per_token"] <= 6.0 and result["chars_per_token"] != 3.0
    assert prechecks[0]["estimate"] >= 70_000 + 66
    analyzer = agent._analyzer
    assert analyzer._chars_per_token == agent.chars_per_token and analyzer._image_tokens == agent.image_tokens
    agent.executor.shutdown()


@pytest.mark.parametrize("grid", [((0, 6), (8, 15)), ((0, 6), (8,)), ((0, 255), (8, -1)),
                                 ((0, "6"), (8.8, 15))])
def test_grid_image_colors_and_padding(grid):
    import base64
    import io

    from PIL import Image

    from inference.agent.runtime_state import Frame
    from inference.agent.vision_context import frame_to_png_data_url

    url = frame_to_png_data_url(Frame(grid=grid, step=0, level=1), upscale=4)
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert image.mode == "RGB" and image.size == (8, 8)
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert image.getpixel((0, 7)) == (249, 60, 49)
    assert image.getpixel((7, 0)) == ((255, 255, 255) if grid[0][1] == 255 else (229, 58, 163))
    assert image.getpixel((7, 7)) == ((163, 86, 214) if len(grid[1]) > 1 and grid[1][1] == 15
                                   else (255, 255, 255))


def test_image_tokens_follow_the_png_geometry():
    from duck_transfer.trimming import image_tokens_for, png_dimensions
    from inference.agent.runtime_state import Frame
    from inference.agent.vision_context import frame_to_png_data_url

    frame = Frame(grid=tuple(tuple(0 for _ in range(64)) for _ in range(64)), step=0, level=1)
    url = frame_to_png_data_url(frame, upscale=4)
    assert png_dimensions(url) == (256, 256)
    assert image_tokens_for({"type": "image_url", "image_url": {"url": url}}, 400) == 66
    assert image_tokens_for({"type": "image_url", "image_url": {"url": frame_to_png_data_url(frame, upscale=16)}}, 400) == 1026
    assert image_tokens_for({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}, 400) == 400
    assert image_tokens_for({"type": "image_url", "image_url": {"url": "not a url"}}, 123) == 123


async def test_token_count_failure_does_not_block_the_request(tmp_path):
    def tokenize(payload):
        raise RuntimeError("tokenizer down")

    class Transport:
        async def post(self, url, **kwargs):
            return tool_reply("action([{'action':'UP'}])", call_id="act")

    agent, env = duck_agent_for(tmp_path, Transport(), tokenize=tokenize)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1
    assert result["token_count_errors"] >= 1 and result["context_prechecks"] == 0
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    assert any(e["kind"] == "token_count_error" for e in events)
    assert result["chars_per_token"] == 3.0 and result["image_tokens"] == 400  # nothing to calibrate from
    assert result["token_count_samples"] == 0 and result["count_residual"] == 0
    agent.executor.shutdown()


async def test_estimate_follows_the_calibrated_ratio(tmp_path):
    agent, _ = duck_agent_for(tmp_path, None)
    analyzer = agent.analyzer(asyncio.get_running_loop())
    messages = [{"role": "user", "content": "x" * 3000}]
    base = analyzer._estimate_request_input_tokens(messages)
    assert 1001 <= base <= 1020  # 3000 characters at 3 per token, plus the role field
    analyzer._chars_per_token = 1.5
    assert analyzer._estimate_request_input_tokens(messages) >= 2001  # re-memoised on the new ratio
    analyzer._chars_per_token = 3.0
    assert analyzer._estimate_request_input_tokens(messages) == base
    agent.executor.shutdown()


async def test_batch_guard_keeps_the_executed_receipt_and_refuses_a_level_start_reset(tmp_path):
    agent, env = duck_agent_for(tmp_path, None, levels=0)
    env.observations = iter([observation(), observation(), observation()])
    agent.config.max_actions = 1
    agent.env = env
    agent.obs = Observation.from_raw(env.observation_space)
    agent.state_path = tmp_path / "tool_runtime_state.json"
    agent.write_duck_state()
    refused = await agent.step_duck({"actions": [{"action": "RESET"}, {"action": "UP"}]})
    assert refused["stop_reason"] == "reset_refused" and not refused["executed"]
    assert "first action of a fresh level" in refused["stop_detail"] and not env.calls
    result = await agent.step_duck({"actions": [{"action": "UP"}, {"action": "UP"}]})
    # The second item hit max_actions: the first action's receipt is not marked done.
    assert result["executed"] and result["executed_count"] == 1 and len(env.calls) == 1
    assert result["stop_reason"] == "terminal_or_budget" and result["done"] is False
    assert result["transition"]["changed_cells"] == 0 and result["stopped_early"]
    agent.config.max_actions = 5
    allowed = await agent.step_duck({"actions": [{"action": "RESET"}]})
    assert allowed["executed"] and env.reset_count == 1


async def test_receipts_carry_transitions_and_no_change_repeats_are_refused(tmp_path):
    board = np.zeros((64, 64), dtype=np.uint8)
    moved = board.copy()
    moved[4, 9] = 9
    agent, env = duck_agent_for(tmp_path, None)
    env.observations = iter([observation(moved), observation(moved), observation(moved)])
    agent.config.max_actions = 10
    agent.env = env
    agent.obs = Observation.from_raw(env.observation_space)
    agent.state_path = tmp_path / "tool_runtime_state.json"
    agent.write_duck_state()
    first = await agent.step_duck({"actions": [{"action": "MOUSE", "row": 4, "col": 9}]})
    transition = first["transition"]
    assert transition["changed_cells"] == 1 and transition["change_bbox"] == [4, 9, 5, 10]
    assert transition["objects"]["counts"]["appeared"] == 1 and transition["no_change_repeat"] is False
    assert agent.transition_results[1]["transition"]["changed_cells"] == 1
    second = await agent.step_duck({"actions": [{"action": "UP"}]})
    assert second["executed"] and second["transition"]["changed_cells"] == 0
    assert second["tried_no_effect_here"] == ["UP"]
    third = await agent.step_duck({"actions": [{"action": "UP"}]})
    assert not third["executed"] and third["stop_reason"] == "no_change_repeat"
    assert "UP already produced no board change" in third["error"] and len(env.calls) == 2
    assert third["skipped_repeats"] == ["UP"] and third["transition"] == {"no_change_repeat": True}
    mixed = await agent.step_duck({"actions": [{"action": "UP"}, {"action": "RIGHT"}]})
    assert mixed["executed_actions"] == ["RIGHT"] and mixed["skipped_repeats"] == ["UP"]
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    assert sum(e["kind"] == "action_skipped" for e in events) == 2
    # Two refusals in one turn; a third insistence executes, flagged, so the turn can end.
    env.observations = iter([observation(moved)])
    insisted = await agent.step_duck({"actions": [{"action": "UP"}]})
    assert insisted["executed"] and insisted["transition"]["no_change_repeat"] is True
    assert agent.repeat_actions == 1 and len(env.calls) == 4
    # Duck's compaction keeps the new receipt keys.
    analyzer = agent.analyzer(asyncio.get_running_loop())
    compact = analyzer._compact_action_result(second)
    assert compact["transition"]["changed_cells"] == 0 and compact["tried_no_effect_here"] == ["UP"]
    assert agent.transitions.hits >= 1
    agent.executor.shutdown()


def test_stuck_tracker_notices_on_cadence():
    from duck_transfer.feedback import StuckTracker

    cfg = SimpleNamespace(stuck_no_change_actions=3, stuck_turns=8, stuck_notice_turns=2)
    tracker = StuckTracker(cfg, now=0.0)
    for changed in (True, False, False):
        tracker.note_action(changed)
    tracker.note_turn()
    assert tracker.notice(2, ["UP"], now=10.0) is None
    tracker.note_action(False)
    assert tracker.stuck()
    text = tracker.notice(2, ["UP", "MOUSE(row=1, col=2)"], now=30.0)
    assert text.startswith("Stuck check: 1 turns and 4 actions on level 2 (30s), 3 consecutive")
    assert "UP, MOUSE(row=1, col=2)" in text and "RESET" in text
    tracker.note_turn()
    assert tracker.notice(2, [], now=40.0) is None  # within the cadence
    tracker.note_turn()
    assert tracker.notice(2, [], now=50.0) is not None and tracker.notices == 2
    tracker.reset_level(now=60.0)
    assert not tracker.stuck() and tracker.notice(3, [], now=61.0) is None


async def test_user_prompt_boilerplate_moves_to_the_system_prompt(tmp_path):
    from duck_transfer.prompting import USER_PROMPT_REMOVALS
    from inference.agent.runtime_state import Frame

    agent, env = duck_agent_for(tmp_path, None)
    agent.env = env
    agent.obs = Observation.from_raw(env.observation_space)
    analyzer = agent.analyzer(asyncio.get_running_loop())
    system = analyzer._system_prompt
    assert "Per-turn discipline" in system and "`transition`" in system
    assert "labelled with its step" in system and "Python state persists" in system
    analyzer._ensure_session(tmp_path / "state.json")
    analyzer._summarized_knowledge["world_model"] = "rows toggle"
    frame = Frame(grid=tuple(tuple(0 for _ in range(64)) for _ in range(64)), step=0, level=1)
    prompt = analyzer._build_user_prompt(
        0, valid_actions=["UP", "MOUSE"], current_frame=frame, history_entries=[], previous_step_summary=None
    )
    for line in USER_PROMPT_REMOVALS:
        assert line not in prompt
    assert "emit the tool call itself" not in prompt and "emit the tool call itself" in system
    assert "If you use MOUSE" not in prompt and "Stuck check" not in prompt
    assert "Current state: step 1, level 1." in prompt and "Valid actions right now:" in prompt
    assert "Working world model carried from earlier turns:" in prompt and "end of world model." in prompt
    message = analyzer._build_user_message(prompt, frame)
    assert message["content"][0]["text"].endswith("Grid image of the board at step 0 (before this turn's actions):")
    assert message["content"][1]["type"] == "image_url"
    agent.executor.shutdown()


async def test_only_the_newest_turn_carries_an_image_and_the_world_model(tmp_path):
    requests = []

    moves = ["UP", "RIGHT", "MOUSE"]  # boards never change, so vary the action per turn

    class Transport:
        async def post(self, url, **kwargs):
            requests.append(kwargs["json"]["messages"])
            move = moves[(len(requests) - 1) % len(moves)]
            return tool_reply(
                f"action([{{'action':'{move}', 'row': 1, 'col': 1}}])",
                call_id=f"act-{len(requests)}",
                content="World model: rows toggle\nPlan: press UP",
            )

    agent, env = duck_agent_for(tmp_path, Transport())
    env.observations = iter([observation(), observation(), observation()])
    agent.config.max_actions = 3
    agent.config.stuck_turns = 2
    agent.config.stuck_notice_turns = 1
    agent.stuck = __import__("duck_transfer.feedback", fromlist=["StuckTracker"]).StuckTracker(agent.config)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 3 and len(requests) == 3
    third = requests[2]
    users = [m for m in third if m["role"] == "user"]
    images = sum(
        1 for m in third if isinstance(m.get("content"), list)
        for part in m["content"] if part.get("type") == "image_url"
    )
    assert images == 1 and isinstance(users[-1]["content"], list)
    assert json.dumps(third).count("Working world model carried from earlier turns:") == 1
    stubs = [m["content"] for m in users[:-1]]
    assert all(isinstance(stub, str) and stub.startswith("[earlier turn: ") for stub in stubs)
    assert "grid image at step 0 omitted" in stubs[0] and "Current state: step 1, level 1." in stubs[0]
    assert "The code executed 1 action in the previous sequence." in stubs[1]
    assert "rows toggle" not in stubs[1]
    newest = users[-1]["content"][0]["text"]
    assert "rows toggle" in newest and "Stuck check: 3 turns" in newest
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    assert sum(e["kind"] == "stuck_notice" for e in events) == 2  # turns 2 and 3, cadence 1


async def test_game_over_prompt_explains_the_automatic_reset_and_keeps_knowledge(tmp_path):
    seen = []

    class Transport:
        async def post(self, url, **kwargs):
            text = json.dumps(kwargs["json"]["messages"])
            seen.append(text)
            return tool_reply(
                "action([{'action':'UP'}])",
                call_id=f"act-{len(seen)}",
                content="World model: the red block dies on the border",
            )

    agent, env = duck_agent_for(tmp_path, Transport())
    env.observations = iter(
        [observation(state="GAME_OVER"), observation(), observation()]
    )
    agent.config.max_actions = 2
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 2 and result["resets"] == 1
    assert "The game is over." not in seen[1]
    assert "GAME_OVER at step 1 after actions [UP]; the level was reset automatically" in seen[1]
    assert "fresh start of level 1" in seen[1] and "red block dies" in seen[1]
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    kinds = [e["kind"] for e in events]
    assert "auto_reset" in kinds and "knowledge_retained" in kinds
    retained = next(e for e in events if e["kind"] == "knowledge_retained")
    assert retained["keys"] == ["world_model"]
    assert not any(e["kind"] == "full_reset" for e in events)


def test_loader_applies_the_analyzer_environment_even_after_an_early_public_import(monkeypatch):
    # 25-26 September arena runs: arena.snapshot imported inference.agent before the loader ran, so
    # the public tool_agent kept its default of 12 model requests per analyzer turn.
    import importlib
    import sys

    from duck_transfer import loader

    for key in loader.DUCK_ENV:
        monkeypatch.delenv(key, raising=False)
    for name in [m for m in sys.modules if m.startswith("inference.agent")]:
        monkeypatch.delitem(sys.modules, name)
    public = importlib.import_module("inference.agent.tool_agent")  # public defaults: 12 steps, 32K
    assert public._LOCAL_ANALYZER_TOOL_STEPS == 12
    monkeypatch.setitem(loader.DUCK_ENV, "LOCAL_ANALYZER_CONTEXT_WINDOW", "131072")
    module = loader.load()
    assert module is public and module.DUCK_TRANSFER_LATE_IMPORT is True
    assert module._LOCAL_ANALYZER_TOOL_STEPS == 0 and module._LOCAL_ANALYZER_CONTEXT_WINDOW == 131072
    assert module._LOCAL_ANALYZER_ENABLE_THINKING is True and module._LOCAL_ANALYZER_TOP_K == 20
    # arena.duck no longer imports the public package at module level.
    for name in [m for m in sys.modules if m.startswith("inference.agent") or m.startswith("arena.")]:
        monkeypatch.delitem(sys.modules, name)
    importlib.import_module("arena.duck")
    assert "inference.agent.tool_agent" not in sys.modules


def test_buffered_transcript_preserves_sections_and_flushes_on_error(tmp_path, monkeypatch):
    import builtins

    from duck_transfer.plumbing import PlumbingMixin

    namespace = {}
    exec("""
def analyze(self, state_path, action_num, **kwargs):
    for text in ["  first ", "", " second\\nline "]:
        _append_transcript_section(state_path, "REPLY", text)
    raise RuntimeError("interrupted turn")
""", namespace)
    def render(label, content):
        return f"[{label}]\n{content.strip()}\n\n" if content.strip() else ""

    module = SimpleNamespace(ToolAgent=SimpleNamespace(analyze=namespace["analyze"]),
                             _render_transcript_section=render)
    agent = PlumbingMixin()
    agent._owner = SimpleNamespace(module=module, runtime_state=lambda *args: None)
    path = tmp_path / "transcript.txt"
    path.write_text("existing\n")
    opened = []
    original = builtins.open
    def tracked(file, *args, **kwargs):
        result = original(file, *args, **kwargs)
        if file == path:
            opened.append(result)
        return result
    monkeypatch.setattr(builtins, "open", tracked)
    with pytest.raises(RuntimeError, match="interrupted turn"):
        agent.analyze(path, 0)
    assert len(opened) == 1 and opened[0].closed
    assert path.read_text() == "existing\n[REPLY]\nfirst\n\n[REPLY]\nsecond\nline\n\n"


def test_cache_telemetry_is_bounded_passive_and_throttled():
    import functools
    import os

    from duck_transfer.sandbox import CACHE_TELEMETRY

    scope = {"os": os}
    exec(CACHE_TELEMETRY, scope)
    scope["_heap_time"] = SimpleNamespace(monotonic=lambda: 100)
    calls = []

    @functools.lru_cache(None)
    def retained(key):
        calls.append(key)
        return object()

    expected = retained(1)
    retained.cache_info = lambda: (_ for _ in ()).throw(AssertionError("Model override must not run"))

    class Untrusted:
        def __getattribute__(self, name):
            raise AssertionError("Telemetry must not inspect model-defined attributes")

    runtime = {"retained": retained, "alias": retained, "unknown": Untrusted()}
    sample = scope["_cache_telemetry"](runtime, 60)
    assert sample["cache_count"] == sample["cache_entries"] == 1
    assert sample["caches"] == [{"name": "retained", "hits": 0, "misses": 1, "maxsize": None, "currsize": 1}]
    assert retained(1) is expected and calls == [1]
    assert scope["_cache_telemetry"](runtime, 60) is None
    scope["_heap_time"].monotonic = lambda: 161
    for index in range(12):
        runtime[f"cache{index}"] = functools.lru_cache(10)(lambda key: key)
    sample = scope["_cache_telemetry"](runtime, 60)
    assert sample["cache_count"] == 13 and len(sample["caches"]) == 8
    assert scope["_cache_telemetry"](runtime, 0) is None


@pytest.mark.parametrize("fixes", [None, ()])
@pytest.mark.parametrize("fail", [False, True])
def test_sandbox_cache_telemetry_logs_without_changing_tool_results(fixes, fail):
    from duck_transfer.loader import load
    from duck_transfer.sandbox import StatefulSandbox, stateful_bootstrap

    original = load().run_sandboxed_python.__globals__
    events = []
    sandbox = StatefulSandbox(
        stateful_bootstrap(original["_SANDBOX_BOOTSTRAP"], fixes=fixes),
        color_chars=original["ARC_COLOR_CHARS"],
        on_event=lambda kind, **data: events.append((kind, data)),
    )
    try:
        result = sandbox.run(
            code="from functools import lru_cache\n@lru_cache(None)\ndef memo(x): return x+1\nprint(memo(2))"
                 + ("\nraise ValueError('expected')" if fail else ""),
            timeout_seconds=5, initial_state={}, action_handler=None,
        )
        assert bool(result["error"]) is fail and result["stdout"] == "3\n"
        assert "cache_telemetry" not in result
        heap = [data for kind, data in events if kind == "duck_python_heap"]
        assert len(heap) == 1 and heap[0]["cache_entries"] == 1
        result = sandbox.run(code="print(memo(2)); raise ValueError('expected')", timeout_seconds=5,
                             initial_state={}, action_handler=None)
        assert result["stdout"] == "3\n" and "expected" in result["error"]
        assert len([kind for kind, _ in events if kind == "duck_python_heap"]) == 1
    finally:
        sandbox.close()


def test_runtime_state_streaming_roundtrip_and_atomic_failure(tmp_path, monkeypatch):
    from inference.agent import runtime_state as state

    path = tmp_path / "runtime.json"
    frames = [state.Frame(((0, 1), (2, 3)), step, 1) for step in range(3)]
    history = [state.HistoryEntry("UP ☀", frame) for frame in frames]
    state.write_runtime_state(path, current_frame=frames[-1], history=history)
    assert state.load_runtime_state(path) == (frames[-1], history)
    before = path.read_bytes()
    original = state.history_entry_to_payload

    def fail_second(entry):
        if entry.frame.step == 1:
            raise ValueError("serialization failed")
        return original(entry)

    monkeypatch.setattr(state, "history_entry_to_payload", fail_second)
    with pytest.raises(ValueError, match="serialization failed"):
        state.write_runtime_state(path, current_frame=frames[0], history=history)
    assert path.read_bytes() == before
    state.write_runtime_state(path, current_frame=None, history=[])
    assert state.load_runtime_state(path) == (None, [])


async def test_finished_game_state_flush_keeps_event_loop_responsive(tmp_path, monkeypatch):
    import threading

    agent, _ = duck_agent_for(tmp_path, None)
    agent.config.max_actions = 0
    started, release = threading.Event(), threading.Event()
    original = agent.flush_duck_state
    calls = 0

    def flush():
        nonlocal calls
        calls += 1
        if calls == 2:
            started.set()
            assert release.wait(1), "final state flush blocked the event loop"
        original()

    async def other_game():
        while not started.is_set():
            await asyncio.sleep(.001)
        release.set()

    monkeypatch.setattr(agent, "flush_duck_state", flush)
    result, _ = await asyncio.wait_for(asyncio.gather(agent.run(), other_game()), 3)
    assert result["error"] is None and calls == 2


async def test_cancellation_waits_for_final_state_and_finished_event(tmp_path, monkeypatch):
    import threading

    agent, _ = duck_agent_for(tmp_path, None)
    agent.config.max_actions = 0
    started, release = threading.Event(), threading.Event()
    original = agent.flush_duck_state
    calls = 0

    def flush():
        nonlocal calls
        calls += 1
        if calls == 2:
            started.set()
            assert release.wait(2)
        original()

    monkeypatch.setattr(agent, "flush_duck_state", flush)
    task = asyncio.create_task(agent.run())
    try:
        async with asyncio.timeout(2):
            while not started.is_set():
                await asyncio.sleep(.001)
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(.01)
            assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
        assert events[-1]["kind"] == "finished" and events[-1]["state_flushes"] == 2
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


def test_sandbox_hash_seed_is_fixed_so_set_print_order_replays():
    """v6m10 R9: -I (implies -E) ignored PYTHONHASHSEED, so a printed set's order differed between sandboxes."""
    from duck_transfer.history import preserve_transition_results
    from duck_transfer.loader import load
    from duck_transfer.sandbox import StatefulSandbox, stateful_bootstrap

    module = load()
    original = module.run_sandboxed_python.__globals__
    grid = [[0] * 64 for _ in range(64)]
    frame = {"grid": grid, "step": 0, "level": 1, "ascii": "", "shape": [64, 64]}
    state = {"current_frame": frame, "history": [], "valid_actions": ["UP"], "last_action_result": {}}
    outs = []
    for _ in range(3):
        sandbox = StatefulSandbox(stateful_bootstrap(preserve_transition_results(original["_SANDBOX_BOOTSTRAP"])),
                                  color_chars=original["ARC_COLOR_CHARS"])
        out = sandbox.run(code="print(hash('N'), {'N', 'W'}, {'a', 'b', 'c', 'd'})", timeout_seconds=10,
                          initial_state=state, action_handler=None)
        sandbox.close() if hasattr(sandbox, "close") else None
        outs.append(out["stdout"])
    assert outs[0] == outs[1] == outs[2], outs
    assert outs[0].split()[0] == str(-6883747592455401683)  # hash('N') at PYTHONHASHSEED=0


@pytest.mark.parametrize("fixes", [None, ("modules",)])
def test_sandbox_random_replays_per_call_and_clock_code_is_flagged(fixes):
    """R leg 25 Oct: 14 of 30 bad restored starts replayed tool code drawing from the unseeded ``random`` (g00145 a009:
    a random MOUSE hill-climb clicked elsewhere on replay). Each call is seeded from the call count, history length
    and code: a fresh sandbox replaying the same calls prints the same draws; repeated calls still differ."""
    from duck_transfer.history import preserve_transition_results
    from duck_transfer.loader import load
    from duck_transfer.sandbox import StatefulSandbox, stateful_bootstrap

    module = load()
    original = module.run_sandboxed_python.__globals__
    frame = {"grid": [[0] * 64 for _ in range(64)], "step": 0, "level": 1, "ascii": "", "shape": [64, 64]}
    state = {"current_frame": frame, "history": [], "valid_actions": ["UP"], "last_action_result": {}}
    codes = ["import random\nprint(random.random(), random.choice('abcdefgh'))"] * 2 + \
            ["import random\nrandom.shuffle(xs := list(range(9)))\nprint(xs)", "import time\nprint(time.time() > 0)"]
    runs, events = [], []
    for _ in range(2):
        seen = []
        sandbox = StatefulSandbox(stateful_bootstrap(preserve_transition_results(original["_SANDBOX_BOOTSTRAP"]),
                                                     fixes=fixes),
                                  color_chars=original["ARC_COLOR_CHARS"],
                                  on_event=lambda kind, **data: seen.append(kind))
        runs.append([sandbox.run(code=c, timeout_seconds=10, initial_state=state, action_handler=None)["stdout"]
                     for c in codes])
        sandbox.close()
        events.append(seen)
    assert runs[0] == runs[1], runs
    assert runs[0][0] != runs[0][1]  # the same code twice: a new draw
    assert events[0].count("duck_python_rng") == 1 and events[0].count("duck_python_clock") == 1


def test_restore_unsafe_from_clock_and_unseeded_random(tmp_path):
    """A clock read never replays; ``random`` replays only in a seeded recording (duck_python_rng)."""
    from arena import restore

    def write(log, replies):
        (tmp_path / "log.jsonl").write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in log))
        rows = [{"n": i, "step": step, "reply": {"tool_calls": [{"function": {"name": "python",
                                                                            "arguments": json.dumps({"code": code})}}]}}
                for i, (step, code) in enumerate(replies)]
        (tmp_path / "rollout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    replies = [(0, "print(1)"), (4, "import random\nprint(random.random())"), (9, "import time\nt0 = time.time()")]
    write([], replies)
    assert restore.unsafe_from(tmp_path)[0] == 4  # unseeded recording: the first random draw
    write([], replies[:1] + replies[2:])
    assert restore.unsafe_from(tmp_path)[0] == 9  # and its first clock read
    write([{"kind": "duck_python_rng", "step": 0}, {"kind": "duck_python_clock", "step": 9}], replies)
    assert restore.unsafe_from(tmp_path)[0] == 9  # seeded: random replays, the clock does not
    write([{"kind": "duck_python_rng", "step": 0}], replies[:2])
    assert restore.unsafe_from(tmp_path)[0] is None
