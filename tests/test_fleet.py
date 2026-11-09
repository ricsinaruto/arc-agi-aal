"""Fleet rosters, session HTTP semantics, scoring summary and engine telemetry."""

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

from arc_infra.config import Config
from arc_infra.fleet import SessionHTTP, fleet_summary, roster, run_replicas, selected_roster
from arc_infra.metrics import Monitor, parse_metrics
from arc_infra.session import GameSession

ROOT = Path(__file__).resolve().parents[1]


def test_rosters_cover_25_110_and_selected_games():
    games = [f"game{i:02}" for i in range(25)]
    entries = roster(list(reversed(games)), 110)
    assert [e["game_id"] for e in entries] == games * 4 + games[:10]
    assert [e["replica"] for e in entries] == [i // 25 for i in range(110)]
    assert roster(games, 25) == entries[:25]
    with pytest.raises(ValueError):
        roster(games[:24], 25)
    selected = selected_roster(["game03", "game01"], games)
    assert selected == [
        {"game_id": "game03", "replica": 0, "instance_id": "r0-game03"},
        {"game_id": "game01", "replica": 0, "instance_id": "r0-game01"},
    ]
    with pytest.raises(ValueError, match="not in the public set"):
        selected_roster(["other"], games)
    with pytest.raises(ValueError, match="distinct"):
        selected_roster(["game01", "game01"], games)


async def test_cache_salt_is_stable_and_request_semantics_unchanged():
    observed = []

    def handler(request):
        import json

        observed.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop"}], "usage": {}})

    async with httpx.AsyncClient(
        base_url="http://model/v1/", transport=httpx.MockTransport(handler)
    ) as shared:
        records = []
        first = SessionHTTP(shared, 0, "r0-game", records)
        duplicate = SessionHTTP(shared, 1, "r1-game", records)
        payload = {"temperature": 0.7, "messages": [{"role": "user", "content": "board"}]}
        await first.post("chat/completions", json=payload)
        await first.post("chat/completions", json=payload)
        await duplicate.post("chat/completions", json=payload)
    assert "cache_salt" not in payload
    assert observed[0] == observed[1]
    assert observed[0]["cache_salt"] != observed[2]["cache_salt"]
    assert {k: v for k, v in observed[2].items() if k != "cache_salt"} == payload
    assert len(records) == 3 and all(r["status"] == 200 for r in records)


@pytest.mark.parametrize("failures", [1, 4])
async def test_only_read_only_tokenizer_transport_errors_are_retried(failures):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) <= failures:
            raise httpx.ReadError("interrupted", request=request)
        return httpx.Response(200, json={"count": 100, "usage": {}, "choices": [{"finish_reason": "stop"}]})

    async with httpx.AsyncClient(
        base_url="http://model/", transport=httpx.MockTransport(handler)
    ) as shared:
        client = SessionHTTP(shared, 0, "game", [])
        if failures == 1:
            response = await client.post("/tokenize", json={}, timeout=3.0)
            assert response.json()["count"] == 100
            assert len(calls) == 2 and client.tokenizer_retries == 1
        else:
            with pytest.raises(httpx.ReadError):
                await client.post("/tokenize", json={}, timeout=3.0)
            assert len(calls) == 3 and client.tokenizer_retries == 2
        calls.clear()
        # A generation request is replayed exactly once after a transport failure.
        if failures == 1:
            response = await client.post("chat/completions", json={"messages": []}, timeout=3.0)
            assert response.json()["count"] == 100
            assert len(calls) == 2 and client.generation_retries == 1
            assert client.requests[-1]["transport_retry"] == "ReadError"
        else:
            with pytest.raises(httpx.ReadError):
                await client.post("chat/completions", json={"messages": []}, timeout=3.0)
            assert len(calls) == 2 and client.generation_retries == 1


def test_summary_includes_all_110_and_base_game_weighting():
    entries = roster([f"game{i:02}" for i in range(25)])
    results = [
        {
            **entry,
            "score": 1.0 if entry["game_id"] == "game00" else 0.0,
            "actions": 2,
            "levels": 0,
            "model_calls": 1,
            "completion_tokens": 500,
            "status": "timeout",
            "error": None,
        }
        for entry in entries
    ]
    summary = fleet_summary(results, 300)
    assert summary["sessions"] == 110 and summary["actions"] == 220
    assert summary["actions_per_game_per_900s"] == 6
    assert summary["mean_score_equal_weight_25_games"] == pytest.approx(1 / 25)
    assert summary["mean_score"] == pytest.approx(5 / 110)
    assert summary["failed_sessions"] == 0


def test_throughput_uses_accepted_engine_tokens_and_preserves_source_labels(tmp_path):
    start = parse_metrics("""
vllm:generation_tokens_total{engine="0"} 100
vllm:generation_tokens_created{engine="0"} 99999999
vllm:prompt_tokens_by_source_total{source="local_compute",engine="0"} 200
vllm:prompt_tokens_by_source_total{source="local_cache_hit",engine="0"} 400
vllm:num_preemptions_total{engine="0"} 1
""")
    end = {
        **start,
        "vllm:generation_tokens_total": 10100,
        "vllm:num_preemptions_total": 4,
        "vllm:kv_cache_usage_perc": 0.95,
        "vllm:num_requests_running": 48,
        "vllm:prompt_tokens_by_source_total:local_compute": 2200,
    }
    monitor = Monitor(tmp_path)
    monitor.samples = [
        {"elapsed_seconds": 0, "metrics": start},
        {"elapsed_seconds": 100, "metrics": end},
    ]
    result = monitor.summary()
    assert result["accepted_generation_tokens_per_second"] == 100
    assert result["preemptions"] == 3
    assert result["computed_prompt_tokens"] == 2000
    assert result["cached_prompt_tokens"] == 0
    assert result["peak_running_requests"] == 48
    assert result["peak_cache_fraction"] == 0.95


class IdleSession(GameSession):
    """Registers its scorecard session, then stops without acting."""

    async def run(self):
        self.env = await asyncio.to_thread(
            self.arc.make, self.game_id, scorecard_id=self.card, save_recording=False
        )
        self.status = "time_limit"
        return self.result()


class OfflineMonitor:
    async def sample(self):
        pass

    async def run(self):
        await asyncio.Event().wait()

    def summary(self):
        return {"valid": False}


@pytest.mark.skipif(
    not (ROOT / "environment_files").is_dir(), reason="public environment files not downloaded"
)
@pytest.mark.parametrize("games", [None, ["ft09-0d8bbf25", "lp85-305b61c3"]])
async def test_fleets_initialize_and_audit_every_session(tmp_path, games):
    report = await run_replicas(
        Config(gameplay_seconds=10, image_mode="none"),
        ROOT / "environment_files",
        tmp_path,
        sys.executable,
        OfflineMonitor(),
        agent_class=IdleSession,
        agent_profile="idle",
        profile_source={"kind": "test"},
        sessions=25,
        games=games,
    )
    expected = 25 if games is None else len(games)
    assert report["mode"] == ("offline_public_25" if games is None else "offline_selected_games")
    assert report["sessions"] == expected and report["failed_sessions"] == 0
    assert report["mean_score"] == 0 and report["actions"] == 0
    assert report["agent_profile"]["name"] == "idle" and report["start_epoch"] > 0
    assert [g["game_id"] for g in report["games"]] == (games or sorted(g["game_id"] for g in report["games"]))
    assert len(list(tmp_path.glob("scorecard*.json"))) == 1
