import json
from types import SimpleNamespace

import httpx

from arc_infra.fleet import fleet_summary
from arena import pi
from arena.clock import ActiveClock, GlobalClock
from arena.gate import GateSlot, KVGate
from arena.scheduler import GameRecord
from qwen_baselines.config import Config
from qwen_baselines.evidence import Archive
from qwen_baselines.runner import metrics


def test_response_metrics_do_not_count_interrupted_requests_as_completed(tmp_path):
    archive = Archive(tmp_path)
    for summary in (False, True, False):
        archive.event('request', input_tokens=100, waiting_seconds=2, summary=summary)
    archive.event('response', inference_seconds=3, summary=False,
                  response={'choices': [], 'usage': {'completion_tokens': 40}})
    archive.event('response', inference_seconds=4, summary=True,
                  response={'choices': [], 'usage': {'completion_tokens': 60}})
    archive.event('generation_interrupted', summary=False, completion_usage_known=False,
                  inference_seconds=2)
    result = metrics(archive)
    assert result['model_requests'] == 3 and result['model_responses'] == 2
    assert result['tokens'] == 100 and result['compaction_seconds'] == 4
    assert result['interrupted_generations_without_usage'] == 1
    row = dict(game_id='game', score=1, actions=5, levels=1, status='time_limit', error=None,
               model_calls=result['model_requests'], completed_replies=result['model_responses'],
               completion_tokens=result['tokens'])
    report = fleet_summary([row], 1800)
    assert report['completed_replies'] == 2 and report['completed_reply_tokens'] == 100
    assert report['mean_score'] == 1 and report['actions'] == 5
    # Duck and historical adapters retain their existing completed-call semantics.
    row.pop('completed_replies')
    row['model_calls'] = 2
    assert fleet_summary([row], 1800)['completed_replies'] == 2
    archive.close()


async def test_pi_summary_wall_cost_includes_counting_and_inference(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(pi, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    wall = GlobalClock(1800)
    game = GameRecord('game', 0, ActiveClock(wall))
    wall.release()
    game.clock.release()

    def endpoint(request):
        if request.url.path == '/tokenize':
            now[0] += 1
            return httpx.Response(200, json={'count': 500, 'max_model_len': 262144})
        now[0] += 4
        body = json.loads(request.content)
        assert body['chat_template_kwargs']['enable_thinking'] is False
        return httpx.Response(200, json={'choices': [{'message': {'content': 'Summary'}}]})

    async with httpx.AsyncClient(base_url='http://localhost/v1/',
                                transport=httpx.MockTransport(endpoint)) as client:
        slot = GateSlot(KVGate(300000, 25), game, cap=16384, window=131072)
        layer = pi.ArenaRequestLayer(Config(output_tokens_cap=16384), game, slot, client=client)
        archive = Archive(tmp_path)
        payload = layer.payload([{'role': 'user', 'content': 'Summarize'}], summary=True)
        await layer.complete_payload(payload, archive, summary=True)
        assert game.compactions == 1 and game.compaction_seconds == 6
        archive.close()
