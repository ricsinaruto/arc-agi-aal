"""Fresh independent games released behind a shared gameplay-start barrier."""

import asyncio
import json
import time
import traceback
from pathlib import Path

from arc_infra.arcade import make_arcade
from arc_infra.metrics import Monitor
from arc_infra.serving_checks import wait_idle

from .agent import Agent
from .bridge import Bridge
from .clock import Clock
from .evidence import Archive
from .kernel import Kernel
from .model import RequestLayer, assemble
from .pi_agent import PiAgent


def prolong_summary(archive):
    """Per-call PRO-LONG statistics (baseline P) from `prolong_call` events."""
    calls = [
        json.loads(payload)
        for kind, payload in archive.db.execute("SELECT kind,payload FROM events ORDER BY id")
        if kind == "prolong_call"
    ]
    if not calls:
        return None
    loaded = [c for c in calls if c["loaded"]]
    return {
        "calls": len(calls),
        "plans_loaded": len(loaded),
        "nudges": sum(c["nudge"] for c in calls),
        "turn_cap_hits": sum(c["turn_cap_hit"] for c in calls),
        "model_errors": sum(bool(c["model_error"]) for c in calls),
        "tool_turns": [c["tool_turns"] for c in calls],
        "plan_lengths": [c["plan_length"] for c in loaded],
        "call_seconds": [round(c["seconds"], 1) for c in calls],
    }


def retrodict_summary(archive):
    """Per-invocation Retrodict statistics (baseline R) from `retrodict_call` events."""
    rows = [
        (kind, json.loads(payload))
        for kind, payload in archive.db.execute("SELECT kind,payload FROM events ORDER BY id")
        if kind in ("retrodict_call", "retrodict_metrics", "retrodict_escalation", "retrodict_tool")
    ]
    calls = [data for kind, data in rows if kind == "retrodict_call"]
    if not calls:
        return None
    metrics = next((data for kind, data in rows if kind == "retrodict_metrics"), {})
    tools = {}
    for kind, data in rows:
        if kind == "retrodict_tool":
            tools[data["name"]] = tools.get(data["name"], 0) + 1
    return {
        "invocations": len(calls),
        "fresh_sessions": sum(c["fresh_session"] for c in calls),
        "model_requests": [c["model_requests"] for c in calls],
        "context_tokens": [c["context_tokens"] for c in calls],
        "truncated": sum(c["truncated"] for c in calls),
        "budget_notices": sum(c.get("budget_notices", 0) for c in calls),
        "length_cuts": sum(c.get("length_cuts", 0) for c in calls),
        "call_seconds": [round(c["seconds"], 1) for c in calls],
        "tool_calls": tools,
        "stop_reason": metrics.get("stop_reason"),
        "parse_retries": metrics.get("parse_retries"),
        "surprises": metrics.get("surprises"),
        "escalation_tier": metrics.get("escalation_tier"),
        "escalations": sum(kind == "retrodict_escalation" for kind, _ in rows),
    }


def schema_summary(archive):
    """Baseline S statistics; None for other baselines."""
    if not (archive.root / "workspace" / "run" / "events.jsonl").exists():
        return None
    from schema_qwen.game import schema_summary as summarize

    return summarize(archive)


def metrics(archive):
    result = {
        "tool_calls": 0,
        "tool_operations": archive.count("calls"),
        "model_requests": 0,
        "model_responses": 0,
        "tokens": 0,
        "input_tokens": 0,
        "inference_wait_seconds": 0,
        "inference_seconds": 0,
        "compaction_requests": 0,
        "compaction_seconds": 0,
        "context_trims": 0,
        "repairs": 0,
        "tool_seconds": 0,
        "code_definitions": 0,
        "reused_function_calls": 0,
        "neutral_continuations": 0,
        "interrupted_generations_without_usage": 0,
        "execution_errors": 0,
        "written_file_executions": 0,
        "repeated_file_executions": 0,
        "actions_rejected": 0,
        "actions_rejected_by_reason": {},
        "outputs_suppressed": 0,
        "action_delays_seconds": [],
    }
    import ast

    known = set()
    written, executed_files = set(), set()
    for i in range(archive.count("calls")):
        recorded = archive.transcript(i)
        result["execution_errors"] += recorded.get("status") == "error" or recorded.get(
            "exit_code", 0
        ) not in (None, 0)
        try:
            operation = json.loads(recorded["code"])
            if isinstance(operation, dict) and operation.get("operation") == "write":
                written.add(Path(operation["path"]).name)
            if isinstance(operation, dict) and operation.get("operation") == "bash":
                for name in written:
                    if name in operation["command"]:
                        result["written_file_executions"] += 1
                        result["repeated_file_executions"] += name in executed_files
                        executed_files.add(name)
        except ValueError:
            pass
        try:
            tree = ast.parse(archive.transcript(i)["code"])
            nodes = list(ast.walk(tree))
            result["reused_function_calls"] += sum(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in known
                for n in nodes
            )
            definitions = {
                n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            known.update(definitions)
            result["code_definitions"] += len(definitions)
        except SyntaxError:
            pass
        if recorded.get("namespace_lost"):
            known.clear()
    previous_observation_seconds = 0
    for i in range(archive.count()):
        current = archive.metadata(i)
        if current["action_id"] is not None:
            result["action_delays_seconds"].append(
                current["elapsed_seconds"] - previous_observation_seconds
            )
        previous_observation_seconds = current["elapsed_seconds"]
    awaiting_repair = False
    for kind, payload in archive.db.execute("SELECT kind,payload FROM events ORDER BY id"):
        data = json.loads(payload)
        if kind == "request":
            result["repairs"] += awaiting_repair
            awaiting_repair = False
            result["model_requests"] += 1
            result["input_tokens"] += data["input_tokens"]
            result["inference_wait_seconds"] += data["waiting_seconds"]
            result["compaction_requests"] += bool(data["summary"])
        elif kind == "response":
            result["model_responses"] += 1
            result["inference_seconds"] += data["inference_seconds"]
            if data["summary"]:
                result["compaction_seconds"] += data["inference_seconds"]
            response = data["response"]
            parsed = assemble(response.encode()) if isinstance(response, str) else response
            result["tool_calls"] += sum(
                len(c.get("message", {}).get("tool_calls") or []) for c in parsed.get("choices", [])
            )
            if isinstance(response, str):
                response = [
                    json.loads(line[6:])
                    for line in response.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                usage = next(
                    (chunk["usage"] for chunk in reversed(response) if chunk.get("usage")), {}
                )
            else:
                usage = response.get("usage", {})
            result["tokens"] += usage.get("completion_tokens", 0)
        elif kind == "context_trim":
            result["context_trims"] += 1
        elif kind == "tool_error":
            awaiting_repair = True
        elif kind in {"python_time", "tool_time"}:
            result["tool_seconds"] += data["seconds"]
            call = archive.transcript(data["call_id"])
            awaiting_repair |= call.get("status") == "error" or call.get("exit_code", 0) not in (
                None,
                0,
            )
        elif kind == "pi_event" and data["event"].get("type") == "tool_execution_end":
            awaiting_repair |= bool(data["event"].get("isError"))
        elif kind == "neutral_continuation":
            result["neutral_continuations"] += 1
        elif kind == "action_rejected":
            result["actions_rejected"] += 1
            reasons = result["actions_rejected_by_reason"]
            reasons[data["reason"]] = reasons.get(data["reason"], 0) + 1
        elif kind == "output_suppressed":
            result["outputs_suppressed"] += 1
        elif kind == "generation_interrupted":
            result["interrupted_generations_without_usage"] += 1
            result["inference_seconds"] += data["inference_seconds"]
    return result


async def trial(
    config,
    baseline,
    games,
    games_dir,
    output,
    *,
    pi_dir,
    node="node",
    layer_factory=RequestLayer,
    arcade_factory=make_arcade,
):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    setup_started = time.monotonic()
    clock = Clock(config.gameplay_seconds)
    layer = layer_factory(config, clock)
    bridges, workers, agents = [], [], []
    arc, card = None, None
    monitor, monitoring = None, None
    report = {"baseline": baseline, "config": config.to_dict(), "games": [], "status": "setup"}
    try:
        async with asyncio.timeout(60):
            report["tokenizer_preflight"] = await layer.preflight()
        arc = arcade_factory("offline", games_dir, output / "recordings")
        discovered = {g.game_id for g in arc.get_environments()}
        if any(game not in discovered for game in games) or len(set(games)) != len(games):
            raise ValueError(f"Explicit games missing or duplicated: {games}")
        card = arc.open_scorecard(tags=["qwen-minimal", baseline])
        # Environment creation is setup; model admission starts only after every worker is ready.
        for game in games:
            env = await asyncio.to_thread(
                arc.make, game, seed=config.seed, scorecard_id=card, save_recording=False
            )
            if env is None:
                raise RuntimeError(f"Environment creation failed: {game}")
            archive = Archive(output / game, export=baseline == "C")
            # P, R and S drain their own plans across boundaries (prolong_qwen.runner,
            # retrodict_qwen.runner, schema_qwen.arc3); the bridge stops only for A/B/C.
            bridge = Bridge(
                env,
                archive,
                clock,
                label=f"{baseline} {game}",
                boundaries=baseline in "ABC",
                objects=config.object_view or config.transition_summary,
                actions_per_turn=config.actions_per_turn,
                reject_repeats=config.reject_no_change_repeats,
            )
            bridges.append(bridge)
            await bridge.initialize()
            worker = Kernel(bridge, config) if baseline == "B" else None
            if worker:
                workers.append(worker)
                await worker.start()
            if baseline == "C":
                agent = PiAgent(bridge, layer, pi_dir=pi_dir, node=node)
                workers.append(agent)
                await agent.start()
            elif baseline == "P":
                from prolong_qwen.agent import ProlongPi
                from prolong_qwen.runner import ProlongGame

                pi = ProlongPi(bridge, layer, pi_dir=pi_dir, node=node)
                workers.append(pi)
                await pi.start()
                agent = ProlongGame(bridge, pi, config)
            elif baseline == "R":
                from retrodict_qwen.game import RetrodictGame

                agent = RetrodictGame(bridge, layer, config)
                workers.append(agent)
            elif baseline == "S":
                from schema_qwen.agent import SchemaPi
                from schema_qwen.game import SchemaGame

                pi = SchemaPi(bridge, layer, pi_dir=pi_dir, node=node)
                workers.append(pi)
                await pi.start()
                agent = SchemaGame(bridge, pi, config)
            else:
                agent = Agent(baseline, bridge, layer, worker)
            agents.append(agent)
        if hasattr(layer, "client"):
            monitor = Monitor(output, config.base_url)
            await monitor.sample()
        report["setup_seconds"] = time.monotonic() - setup_started
        print(
            json.dumps(
                {
                    "event": "starting_trial",
                    "trial": baseline,
                    "games": list(games),
                    "gameplay_seconds": config.gameplay_seconds,
                    "setup_seconds": round(report["setup_seconds"], 1),
                }
            ),
            flush=True,
        )
        clock.release()
        if monitor:
            monitor.start = clock.start
            monitoring = asyncio.create_task(monitor.run())
        for bridge in bridges:
            bridge.archive.event("gameplay_start", monotonic=clock.start, timing=clock.feedback())
        outcomes = await asyncio.gather(*(agent.run() for agent in agents), return_exceptions=True)
        report["gameplay_seconds"] = clock.elapsed()
        if monitor:
            monitoring.cancel()
            await asyncio.gather(monitoring, return_exceptions=True)
            await wait_idle(config.base_url)
            await monitor.sample()
            report["backend_throughput"] = monitor.summary()
        report["status"] = (
            "complete" if not any(isinstance(x, BaseException) for x in outcomes) else "error"
        )
        scorecard = await asyncio.to_thread(arc.close_scorecard, card)
        card = None
        scores = scorecard.model_dump(mode="json", exclude={"api_key"})
        (output / "scorecard.json").write_text(json.dumps(scores, indent=2))
        score_by_game = {entry["id"]: entry for entry in scores.get("environments", [])}
        for game, bridge, outcome in zip(games, bridges, outcomes, strict=True):
            entry = {
                "game": game,
                "scorecard": score_by_game.get(game),
                "score": score_by_game[game]["score"] if game in score_by_game else None,
                "actions": bridge.actions,
                "levels_completed": bridge.archive.metadata()["levels_completed"],
                "clears": bridge.clears,
                "metrics": metrics(bridge.archive),
                "prolong": prolong_summary(bridge.archive),
                "retrodict": retrodict_summary(bridge.archive),
                "schema": schema_summary(bridge.archive),
                "error": str(outcome) if isinstance(outcome, BaseException) else None,
            }
            entry["metrics"]["tokens_per_gameplay_second"] = entry["metrics"]["tokens"] / max(
                clock.elapsed(), 0.001
            )
            entry["metrics"]["per_game_token_count_is_lower_bound"] = bool(
                entry["metrics"]["interrupted_generations_without_usage"]
            )
            report["games"].append(entry)
        if any(game["score"] is None for game in report["games"]):
            raise RuntimeError(
                "Scorecard omitted a selected game; do not report a partial-suite mean"
            )
        report["mean_score_percent"] = sum(game["score"] for game in report["games"]) / len(games)
        report["total_actions"] = sum(game["actions"] for game in report["games"])
        report["total_cleared_levels"] = sum(game["levels_completed"] for game in report["games"])
        print(
            json.dumps(
                {
                    "event": "trial_result",
                    "trial": baseline,
                    "status": report["status"],
                    "gameplay_seconds": round(report["gameplay_seconds"], 1),
                    "mean_score_percent": report["mean_score_percent"],
                    "total_actions": report["total_actions"],
                    "total_cleared_levels": report["total_cleared_levels"],
                    "games": [
                        {
                            "game": g["game"],
                            "score": g["score"],
                            "actions": g["actions"],
                            "levels_completed": g["levels_completed"],
                            "error": g["error"],
                        }
                        for g in report["games"]
                    ],
                }
            ),
            flush=True,
        )
        return report
    except BaseException:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
        raise
    finally:
        if monitoring:
            monitoring.cancel()
            await asyncio.gather(monitoring, return_exceptions=True)
        for worker in workers:
            await worker.close()
        if card is not None:
            await asyncio.to_thread(arc.close_scorecard, card)
        await layer.close()
        for bridge in bridges:
            bridge.archive.close()
        (output / "report.json").write_text(json.dumps(report, indent=2))


async def comparison(config, baselines, games, games_dir, output, *, pi_dir, node="node"):
    reports = []
    for baseline in baselines:
        await wait_idle(config.base_url)
        reports.append(
            await trial(
                config,
                baseline,
                games,
                games_dir,
                Path(output) / baseline,
                pi_dir=pi_dir,
                node=node,
            )
        )
        await wait_idle(config.base_url)
        Path(output, "comparison.json").write_text(json.dumps(reports, indent=2))
        if reports[-1]["status"] != "complete":
            raise RuntimeError(
                f"Trial {baseline} failed correctness checks; inspect its report before further GPU trials"
            )
    Path(output, "comparison.json").write_text(json.dumps(reports, indent=2))
    return reports
