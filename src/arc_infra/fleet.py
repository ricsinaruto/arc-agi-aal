"""Independent public-game sessions sharing one model server and one fleet clock.

A fleet is 25 single-replica sessions, 110 cyclic-replica sessions, or an explicit list of
roster entries (selected games). Sessions never share game state, scores or KV cache.
"""

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import httpx

from .arcade import make_arcade, source_provenance
from .model import ModelClient


def roster(game_ids, total=110):
    games = sorted(set(game_ids))
    if len(games) != 25 or total not in (25, 110):
        raise ValueError("This study requires 25 public games and either 25 or 110 sessions")
    return [
        {"game_id": games[i % 25], "replica": i // 25, "instance_id": f"r{i // 25}-{games[i % 25]}"}
        for i in range(total)
    ]


def selected_roster(games, available):
    """One replica of each selected game, in the given order."""
    games = list(games)
    if not games or len(set(games)) != len(games):
        raise ValueError("Select distinct games")
    missing = [game for game in games if game not in set(available)]
    if missing:
        raise ValueError(f"Selected games are not in the public set: {missing}")
    return [{"game_id": game, "replica": 0, "instance_id": f"r0-{game}"} for game in games]


class SessionHTTP:
    """Keep frozen request construction; add only a replica cache salt and telemetry."""

    # ``post(..., record_extra={...})`` adds fields to the request's telemetry record (the arena's parallel samples tag
    # their ping, primary and siblings); never sent to the server.
    records_extra = True

    def __init__(self, shared, replica, instance_id, requests):
        self.shared = shared
        self.replica, self.instance_id, self.requests = replica, instance_id, requests
        self.base_url = shared.base_url
        self.tokenizer_retries = 0
        self.generation_retries = 0

    async def post(self, url, **kwargs):
        extra = kwargs.pop("record_extra", None)
        if str(url).rstrip("/").endswith("chat/completions"):
            # Different games within one replica still share their common instructions.
            # Copies of the SAME game cannot hit each other's KV cache. Stable per turn.
            payload = dict(kwargs["json"])
            payload["cache_salt"] = f"arc-public-replica-{self.replica}"
            kwargs["json"] = payload
            record = {"instance_id": self.instance_id, "started_monotonic": time.monotonic()}
            if extra:
                record.update(extra)
            record["image_count"] = sum(
                part.get("type") == "image_url"
                for message in payload.get("messages", [])
                if isinstance(message.get("content"), list)
                for part in message["content"]
                if isinstance(part, dict)
            )
            self.requests.append(record)
            try:
                try:
                    response = await self.shared.post(url, **kwargs)
                except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                    # One replay (24 September): a transport failure on a generation request
                    # used to end the game for the whole trial (lp85 at 5 s, effort-low v1).
                    # A duplicate generation costs GPU time at worst; it issues no action.
                    remaining = kwargs["timeout"] - (time.monotonic() - record["started_monotonic"])
                    if remaining <= 1.0:
                        raise
                    record["transport_retry"] = type(exc).__name__
                    self.generation_retries += 1
                    await asyncio.sleep(1.0)
                    response = await self.shared.post(url, **{**kwargs, "timeout": remaining - 1.0})
                record["status"] = response.status_code
                if not response.is_success:
                    record["response_error"] = response.text[:4000]
                if response.is_success:
                    data = response.json()
                    record["usage"] = data.get("usage", {})
                    record["finish_reason"] = data["choices"][0]["finish_reason"]
                return response
            except BaseException as exc:
                record["error"] = type(exc).__name__
                raise
            finally:
                record["seconds"] = time.monotonic() - record["started_monotonic"]
        if str(url).rstrip("/").endswith("/tokenize"):
            # Read-only requests are safe to replay after transport failure. Never
            # replay an ambiguous generation: it may still be using the GPU.
            end = time.monotonic() + kwargs["timeout"]
            for attempt in range(3):
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Gameplay deadline reached during tokenizer recovery")
                try:
                    return await self.shared.post(url, **{**kwargs, "timeout": remaining})
                except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError):
                    if attempt == 2:
                        raise
                    self.tokenizer_retries += 1
                    await asyncio.sleep(min(0.1 * (attempt + 1), max(0, end - time.monotonic())))
        return await self.shared.post(url, **kwargs)


class SessionClient(ModelClient):
    def __init__(self, config, shared, slots, entry, requests, tool_schema=None):
        # No per-session transport or semaphore: every session shares these resources.
        self.config, self.tool_schema = config, tool_schema
        self.slots = slots
        self.client = SessionHTTP(shared, entry["replica"], entry["instance_id"], requests)


def fleet_summary(results, elapsed):
    actions = sum(r["actions"] for r in results)
    groups = defaultdict(list)
    for result in results:
        groups[result["game_id"]].append(result["score"])
    return {
        "sessions": len(results),
        "actions": actions,
        "actions_per_second": actions / elapsed,
        "actions_per_game_per_900s": actions / len(results) * 900 / elapsed,
        "mean_score": sum(r["score"] for r in results) / len(results),
        "mean_score_equal_weight_25_games": sum(
            sum(scores) / len(scores) for scores in groups.values()
        )
        / len(groups),
        "levels": sum(r["levels"] for r in results),
        "completed_replies": sum(r.get("completed_replies", r["model_calls"]) for r in results),
        "completed_reply_tokens": sum(r["completion_tokens"] for r in results),
        "zero_action_sessions": sum(r["actions"] == 0 for r in results),
        "failed_sessions": sum(r["status"] == "error" or bool(r["error"]) for r in results),
    }


async def run_replicas(
    config,
    games_dir,
    output,
    interpreter,
    monitor,
    *,
    agent_class,
    agent_profile,
    profile_source,
    sessions=110,
    games=None,
    clock_sessions=None,
    tool_schema=None,
):
    """Play a public fleet: ``sessions`` of 25 or 110, or the explicit ``games`` list.

    ``clock_sessions`` sets the fleet size the agents' planning clock is normalized to
    (default: the number of sessions actually played).
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if games is None and sessions not in (25, 110):
        raise ValueError("Select 25 or 110 independent public sessions, or explicit games")
    replicas = 1 if games is not None else (sessions + 24) // 25
    arcs = [
        make_arcade("offline", games_dir, output / "recordings" / f"r{r}") for r in range(replicas)
    ]
    available = [g.game_id for g in arcs[0].get_environments()]
    entries = (
        selected_roster(games, available) if games is not None else roster(available, sessions)
    )
    cards = [
        arc.open_scorecard(tags=["fleet", agent_profile, f"replica-{i}"])
        for i, arc in enumerate(arcs)
    ]
    requests = []
    slots = asyncio.Semaphore(config.parallel_games)
    shared = httpx.AsyncClient(
        base_url=config.base_url.rstrip("/") + "/",
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
        trust_env=False,
    )
    await monitor.sample()
    start = time.monotonic()
    start_epoch = time.time()
    monitor.start = start
    deadline = start + config.gameplay_seconds
    agents = [
        agent_class(
            entry["game_id"],
            arcs[entry["replica"]],
            cards[entry["replica"]],
            SessionClient(config, shared, slots, entry, requests, tool_schema=tool_schema),
            config,
            output / "games" / entry["instance_id"],
            interpreter,
            deadline,
        )
        for entry in entries
    ]
    timing_policies = {}
    for entry, agent in zip(entries, agents, strict=True):
        configure = getattr(agent, "configure_fleet_timing", None)
        if configure is not None:
            timing_policies[entry["instance_id"]] = configure(start, clock_sessions or len(entries))
    if games is not None:
        mode = "offline_selected_games"
    else:
        mode = "offline_replicated_public" if sessions == 110 else "offline_public_25"
    metadata = {
        "mode": mode,
        "roster": entries,
        "roster_order": "sorted cyclic suites; all sessions eligible together"
        if games is None
        else "explicit selected games, one replica each",
        "planning_budget": timing_policies,
        "roster_sha256": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
        "config": config.to_dict(),
        "source": source_provenance(),
        "agent_profile": {**profile_source, "name": agent_profile},
        "cache_isolation": "one salt per replica cohort; no same-game cross-copy KV reuse",
        "score_scope": f"{len(entries)} public sessions; not a submission",
        "start_monotonic": start,
        "start_epoch": start_epoch,
        "admission": {"mode": "fixed", "limit": config.parallel_games},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    tasks = [asyncio.create_task(agent.run()) for agent in agents]
    sampling = asyncio.create_task(monitor.run())
    try:
        while time.monotonic() < deadline and any(not t.done() for t in tasks):
            await asyncio.wait(
                [t for t in tasks if not t.done()],
                timeout=min(15, max(0, deadline - time.monotonic())),
            )
            progress = {
                "event": "progress",
                "elapsed_seconds": time.monotonic() - start,
                "actions": sum(a.steps for a in agents),
                "levels": sum(a.obs.levels if a.obs else 0 for a in agents),
                "completed_replies": sum(a.turns for a in agents),
                "completed_reply_tokens": sum(a.completion_tokens for a in agents),
                "finished_sessions": sum(t.done() for t in tasks),
            }
            (output / "progress.json").write_text(json.dumps(progress, indent=2))
            with (output / "progress.jsonl").open("a") as file:
                file.write(json.dumps(progress) + "\n")
            print(json.dumps(progress), flush=True)
    finally:
        # Capture accepted engine output, including unfinished replies, before cancellation.
        sampling.cancel()
        await asyncio.gather(sampling, return_exceptions=True)
        await monitor.sample()
        window = time.monotonic() - start
        for task in tasks:
            if not task.done():
                task.cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        await shared.aclose()
        closed = [
            await asyncio.to_thread(arc.close_scorecard, card)
            for arc, card in zip(arcs, cards, strict=True)
        ]
    scores = {}
    for replica, card in enumerate(closed):
        if card is None:
            raise RuntimeError(f"Failed to close replica {replica} scorecard")
        (output / f"scorecard-r{replica}.json").write_text(
            json.dumps(card.model_dump(mode="json", exclude={"api_key"}), indent=2)
        )
        scores.update({(replica, env.id): env.score for env in card.environments})
    expected_pairs = {(entry["replica"], entry["game_id"]) for entry in entries}
    if set(scores) != expected_pairs:
        raise RuntimeError("Scorecard sessions do not match the complete session roster")
    results = [
        {
            **agent.result(),
            **entry,
            "score": scores[(entry["replica"], entry["game_id"])],
            "tokenizer_retries": agent.client.client.tokenizer_retries,
            "generation_retries": agent.client.client.generation_retries,
        }
        for agent, entry in zip(agents, entries, strict=True)
    ]
    report = {
        **metadata,
        "gameplay_seconds": window,
        **fleet_summary(results, window),
        "engine": monitor.summary(),
        "games": results,
        "task_errors": [
            f"{type(o).__name__}: {o}"
            for o in outcomes
            if isinstance(o, BaseException) and not isinstance(o, asyncio.CancelledError)
        ],
    }
    (output / "requests.json").write_text(json.dumps(requests, indent=2))
    (output / "results.json").write_text(json.dumps(report, indent=2))
    return report
