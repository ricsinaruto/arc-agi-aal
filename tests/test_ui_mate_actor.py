"""Native UI-Mate protocol over real ARC bridges; all model responses are CPU mocks."""

import asyncio
import hashlib
import importlib.util
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import jinja2
import pytest
from test_qwen_core import Environment, bridge_at

from arena.clock import ActiveClock, GlobalClock
from arena.gate import GateSlot, KVGate
from arena.pi import ArenaBridge, ArenaRequestLayer
from arena.scheduler import GameRecord
from arena.trial import qwen_config
from qwen_baselines.evidence import Archive
from ui_mate_actor import native
from ui_mate_actor.agent import NativeAgent
from ui_mate_actor.interface import execute, system_prompt, translate

ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT / 'configs/arena_30m_ui_mate_9b.json').read_text())


def xml(**fields):
    return '<tool_call>\n<function=computer_use>\n' + '\n'.join(
        f'<parameter={k}>\n{json.dumps(v) if isinstance(v, list) else v}\n</parameter>'
        for k, v in fields.items()
    ) + '\n</function>\n</tool_call>'


def test_native_source_identity_and_prompt_scope():
    provenance = json.loads((ROOT / 'src/ui_mate_actor/SOURCE.json').read_text())
    assert hashlib.sha256((ROOT / 'src/ui_mate_actor/native.py').read_bytes()).hexdigest() == provenance['native_sha256']
    prompt = system_prompt(1800)
    schema = json.loads(prompt.split('<tools>\n', 1)[1].split('\n</tools>', 1)[0])
    assert schema['function']['name'] == 'computer_use'
    assert schema['function']['parameters']['properties']['action']['enum'] == ['left_click', 'press', 'wait', 'finished']
    assert '1800 seconds' in prompt and 'not proof of progress' in prompt
    assert native.RESPONSE_FORMAT in prompt
    assert 'MEMORY:' not in prompt and 'World model:' not in prompt


@pytest.mark.parametrize('coordinate,cell', [([0, 999], (0, 63)), ([999, 0], (63, 0)), ([500, 500], (32, 32))])
def test_coordinate_edges_and_orientation(coordinate, cell):
    assert translate({'action': 'left_click', 'coordinate': coordinate}) == [
        {'action': 'ACTION6', 'x': cell[0], 'y': cell[1]}]


def test_upstream_compatible_key_and_numeric_string_cleanup():
    assert translate({'action':'press','keys':"keys=['up']"}) == [{'action':'ACTION1'}]
    assert translate({'action':'left_click','coordinate':['999','0']}) == [{'action':'ACTION6','x':63,'y':0}]


@pytest.mark.parametrize('call', [
    {'action': 'left_click', 'coordinate': [1000, 0]},
    {'action': 'left_click', 'coordinate': [True, 0]},
    {'action': 'left_click', 'coordinate': [float('nan'), 0]},
    {'action': 'press', 'keys': ['1', 'ctrl']},
    {'action': 'type', 'text': '__import__("os").system("false")'},
    {'action': 'wait', 'time': 'inf'}, {'action': 'wait', 'time': -1},
    {'action': 'wait', 'time': True}, {'action': 'finished', 'status': 'maybe'},
])
def test_unsupported_or_malformed_calls_rejected(call):
    with pytest.raises(ValueError):
        translate(call)


async def test_action7_order_reset_and_level_boundary(tmp_path):
    bridge = await bridge_at(tmp_path, clear_at=2)
    result = await execute(xml(action='press', keys=['7', '1', '2']) + xml(action='press', keys=['3']), bridge)
    assert bridge.actions == 2 and bridge.archive.metadata(1)['action']['action'] == 'ACTION7'
    assert result['results'][1]['receipt']['level_completed']
    assert all(r['skipped'] for r in result['results'][2:])
    await bridge.begin_turn()
    result = await execute(xml(action='press', keys=['r', '1']), bridge)
    assert result['results'][0]['receipt']['reset'] and result['results'][1]['skipped']
    bridge.archive.close()


async def test_partial_execution_errors_and_false_success_do_not_rewrite_state(tmp_path):
    bridge = await bridge_at(tmp_path)
    result = await execute(xml(action='press', keys=['1']) + xml(action='press', keys=['5']) +
                           xml(action='press', keys=['2']), bridge)
    assert bridge.actions == 1 and result['results'][0]['receipt']['action']['action'] == 'ACTION1'
    assert 'Unavailable' in result['results'][1]['error'] and result['results'][2]['skipped']
    await bridge.begin_turn()
    result = await execute(xml(action='finished', status='success') + xml(action='press', keys=['1']), bridge)
    assert result['finished_claim'] == 'success' and result['results'][1]['skipped']
    assert bridge.actions == 1 and bridge.archive.metadata()['status'] != 'WIN'
    result = await execute('I am still considering the screenshot.', bridge)
    assert 'continuation' in result and bridge.actions == 1
    bridge.archive.close()


async def test_reasoning_examples_and_truncated_thinking_never_act(tmp_path):
    bridge = await bridge_at(tmp_path)
    example = xml(action='press',keys=['2'])
    result = await execute('<think>Maybe use '+example+'</think><action>Try up</action>'+
                           xml(action='press',keys=['1']), bridge)
    assert bridge.actions == 1 and result['results'][0]['receipt']['action']['action']=='ACTION1'
    result = await execute('<think>I could use '+example, bridge)
    assert 'continuation' in result and bridge.actions == 1
    bridge.archive.close()


async def test_cancellation_preserves_issued_action_without_replay(tmp_path):
    bridge = await bridge_at(tmp_path)
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    original = bridge.env.step
    def slow(*args):
        loop.call_soon_threadsafe(entered.set)
        time.sleep(.03)
        return original(*args)
    bridge.env.step = slow
    task = asyncio.create_task(execute(xml(action='press', keys=['1', '2']), bridge))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bridge.actions == bridge.env.n == 1 and bridge.archive.count() == 2
    bridge.archive.close()


def setup_agent(tmp_path, client, *, seconds=1800):
    wall = GlobalClock(seconds)
    clock = ActiveClock(wall)
    game = GameRecord('test', 0, clock)
    cfg = qwen_config(STUDY, 'ui-mate-native', window=131072, cap=16384, wall=seconds, base_url='http://model/v1')
    gate = KVGate(262144, slots=25)
    layer = ArenaRequestLayer(cfg, game, GateSlot(gate, game, cap=16384, window=131072), client=client)
    bridge = ArenaBridge(Environment(), Archive(tmp_path), clock, game=game)
    agent = NativeAgent(bridge, layer, game=game, window=131072)
    return agent, wall, gate


@pytest.mark.parametrize('profile', ['ui-mate-9b', 'ui-mate-27b'])
async def test_native_loop_real_request_layer_and_receipts(tmp_path, profile):
    requests = []
    def endpoint(request):
        body = json.loads(request.content)
        if request.url.path == '/tokenize':
            render_native_request(profile, body)
            assert body['chat_template_kwargs']['enable_thinking']
            return httpx.Response(200, json={'count': 200, 'max_model_len': 262144})
        requests.append(body)
        text = xml(action='press', keys=['7']) if len(requests) == 1 else xml(action='finished', status='success')
        chunks = [{'choices': [{'index': 0, 'delta': {'reasoning_content': 'native thought'}}]},
                  {'choices': [{'index': 0, 'delta': {'content': '<action>Try next step</action>' + text}, 'finish_reason': 'stop'}]},
                  {'choices': [], 'usage': {'prompt_tokens': 200, 'completion_tokens': 20}}]
        return httpx.Response(200, content=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n')
    async with httpx.AsyncClient(base_url='http://model/v1/', transport=httpx.MockTransport(endpoint)) as client:
        agent, wall, gate = setup_agent(tmp_path, client)
        from arena.candidate_profiles import PROFILES
        agent.layer.config = replace(agent.layer.config, model=PROFILES[profile].model,
                                     model_revision=PROFILES[profile].revision)
        await agent.bridge.initialize()
        wall.release()
        agent.game.clock.release()
        await agent.run()
        assert agent.bridge.actions == 1 and len(requests) == 2
        assert all('tools' not in r and r['max_tokens'] == 16384 for r in requests)
        assert requests[0]['temperature'] == .7
        assert 'Actual gameplay clock' in requests[0]['messages'][-1]['content']
        assert 'ACTION7' in json.dumps(requests[1]['messages'][-2])
        assert '<think>native thought</think>' in json.dumps(requests[1]['messages'])
        assert agent.game.finished_reason == 'model_finished_success'
        assert agent.bridge.archive.metadata()['status'] != 'WIN'
        assert gate.inflight == gate.reserved == 0
        agent.bridge.archive.close()


async def test_native_history_collapse_and_token_backstop_preserve_latest_observation(tmp_path):
    async with httpx.AsyncClient(base_url='http://unused/') as client:
        agent, _, _ = setup_agent(tmp_path, client)
        await agent.bridge.initialize()
        for i in range(110):
            obs = agent.observation([])
            obs['content'][0]['text'] += f' historical-{i}'
            agent.exchanges.append([obs, {'role': 'assistant', 'content': f'<think>reason-{i}</think><action>Inspect</action>'}])
        latest = agent.observation([])
        latest['content'][0]['text'] += ' LATEST_BOARD'
        seen = []
        async def count(payload):
            seen.append(payload)
            return 120000 if len(agent.exchanges) > 3 else 1000
        agent.layer.count = count
        payload = await agent.payload(latest)
        assert len(agent.exchanges) == 3 and agent.omitted == 107
        assert 'LATEST_BOARD' in json.dumps(payload['messages'][-1])
        images = [b for m in payload['messages'] if isinstance(m.get('content'), list)
                  for b in m['content'] if b['type'] == 'image_url']
        assert len(images) == 4
        assert 'reason-109' in json.dumps(payload) and 'reason-1<' not in json.dumps(payload)
        assert all(m['role'] != 'tool' for m in payload['messages'])
        agent.bridge.archive.close()


async def test_five_images_preserve_earliest_retained_and_latest(tmp_path):
    async with httpx.AsyncClient(base_url='http://unused/') as client:
        agent, _, _ = setup_agent(tmp_path, client)
        await agent.bridge.initialize()
        for i in range(8):
            obs = {'role':'user','content':[{'type':'image_url','image_url':{'url':str(i)}}]}
            agent.exchanges.append([obs, {'role':'assistant','content':'<action>Inspect</action>'}])
        current = {'role':'user','content':[{'type':'image_url','image_url':{'url':'current'}}]}
        messages = agent.messages(current)
        images = [b['image_url']['url'] for m in messages if isinstance(m.get('content'), list)
                  for b in m['content'] if b['type']=='image_url']
        assert images == ['0','5','6','7','current']
        assert len(agent.exchanges[1][0]['content']) == 1  # Rendering does not mutate archived exchange.
        agent.bridge.archive.close()


async def test_parked_game_cannot_execute_remaining_plan(tmp_path):
    async with httpx.AsyncClient(base_url='http://unused/') as client:
        agent, wall, _ = setup_agent(tmp_path, client)
        await agent.bridge.initialize()
        wall.release()
        agent.game.clock.release()
        original = agent.bridge.step
        async def park_after_step(**kwargs):
            result = await original(**kwargs)
            agent.game.clock.pause()
            return result
        agent.bridge.step = park_after_step
        task = asyncio.create_task(execute(xml(action='press', keys=['1','2']), agent.bridge))
        for _ in range(100):
            if agent.bridge.actions:
                break
            await asyncio.sleep(.005)
        assert agent.bridge.actions == 1
        await asyncio.sleep(.01)
        assert not task.done() and agent.bridge.actions == 1
        agent.game.clock.resume()
        result = await asyncio.wait_for(task, 1)
        assert len(result['results']) == 2 and agent.bridge.actions == 2
        agent.bridge.archive.close()


async def test_full_trial_initialization_barrier_and_native_worker_dispatch(tmp_path, monkeypatch):
    from test_arena_trial import OfflineMonitor

    from arena import trial as trial_module

    games = STUDY['games'][:2]
    made = []
    class Arcade:
        def get_environments(self):
            return [SimpleNamespace(game_id=g) for g in games]
        def open_scorecard(self, **kwargs):
            return 'card'
        def make(self, game_id, **kwargs):
            made.append(game_id)
            return Environment()
        def close_scorecard(self, card):
            # A success claim without an environmental clear must retain score zero.
            return SimpleNamespace(environments=[SimpleNamespace(id=g, score=0.) for g in games],
                                   model_dump=lambda **kwargs: {'environments': games})
    requests = []
    def endpoint(request):
        body = json.loads(request.content)
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'count':200,'max_model_len':262144})
        assert made == games  # Every environment exists before any gameplay generation.
        requests.append(body)
        delta = {'choices':[{'index':0,'delta':{'content':xml(action='finished',status='success')},'finish_reason':'stop'}],
                 'usage':{'prompt_tokens':200,'completion_tokens':20}}
        return httpx.Response(200,content='data: '+json.dumps(delta)+'\n\ndata: [DONE]\n\n')
    client = httpx.AsyncClient(base_url='http://model/v1/',transport=httpx.MockTransport(endpoint))
    monkeypatch.setattr(trial_module.httpx,'AsyncClient',lambda **kwargs: client)
    monkeypatch.setattr(trial_module,'make_arcade',lambda *args: Arcade())
    monkeypatch.setattr(trial_module,'kv_pool_tokens',lambda _: 1_000_000)
    monkeypatch.setattr(trial_module,'GpuMonitor',OfflineMonitor)
    study = {**STUDY,'games':games,'wall_seconds':2,'cache_admission':'tokens'}
    trial = trial_module.Trial(study,{'name':'native','kind':'ui-mate-native'},source=ROOT,
        games_dir='unused',output=tmp_path/'trial',server_dir=tmp_path,base_url='http://model/v1',log=lambda *a,**k:None)
    report = await trial.run()
    assert report['status'] == 'complete' and len(requests) == 2
    assert report['actions'] == report['levels'] == report['mean_score_equal_weight_25_games'] == 0
    assert all(g['finished_reason']=='model_finished_success' and g['status']=='model_finished' for g in report['games'])
    assert report['config']['interface']=='ui-mate-native-arc'


@pytest.mark.parametrize('profile',['ui-mate-9b','ui-mate-27b'])
def test_separate_native_bundle_preserves_paired_default(tmp_path, profile):
    spec = importlib.util.spec_from_file_location('ui_native_builder', ROOT/'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output, study = builder.build('native-test', output=tmp_path/'bundle', paired_30m=True,
                                  candidate_profile=profile, ui_mate_native=True)
    assert [t['kind'] for t in study['trials']] == ['ui-mate-native']
    assert len(study['games']) == 25 and study['wall_seconds'] == 1800 and study['windows'] == [131072]
    assert (output/'src/ui_mate_actor/LICENSE').is_file()
    assert (output/'src/ui_mate_actor/SOURCE.json').is_file()
    assert json.loads((output/'MANIFEST.json').read_text())['files']['src/ui_mate_actor/native.py']
    cfg = qwen_config(study,'ui-mate-native',window=131072,cap=16384,wall=1800,base_url='http://model/v1')
    assert cfg.actions_per_turn is None and not cfg.reject_no_change_repeats
    assert cfg.image_pixels == study['native_actor']['image_pixels']


@pytest.mark.skipif(not (ROOT/'environment_files').is_dir(), reason='public CPU environments not installed')
async def test_native_controls_on_all_25_real_public_environments(tmp_path):
    from arc_infra.arcade import make_arcade
    from qwen_baselines.bridge import Bridge
    from qwen_baselines.clock import Clock

    arc = make_arcade('offline',ROOT/'environment_files',tmp_path/'recordings')
    card = arc.open_scorecard(tags=['ui-mate-native-cpu-bridge-validation'])
    checked = []
    for game_id in STUDY['games']:
        env = arc.make(game_id,scorecard_id=card,save_recording=False)
        clock = Clock(1800)
        archive = Archive(tmp_path/game_id)
        bridge = Bridge(env,archive,clock)
        await bridge.initialize()
        clock.release()
        legal = archive.metadata()['legal_actions']
        action = next(a for a in legal if a!='RESET')
        call = xml(action='left_click',coordinate=[500,500]) if action=='ACTION6' else xml(action='press',keys=[action[-1]])
        result = await execute(call,bridge)
        assert bridge.actions==1 and archive.count()==2
        assert result['results'][0]['receipt']['action']['action']==action
        assert archive.metadata()['levels_completed']==env.observation_space.levels_completed
        assert archive.metadata()['frame_count']==len(env.observation_space.frame)
        # Exercise explicit reset against the real competition-compatible wrapper too.
        await bridge.begin_turn()
        if archive.metadata()['status']!='WIN':
            reset = await execute(xml(action='press',keys=['r','1']),bridge)
            assert reset['results'][0]['receipt']['reset'] and reset['results'][1]['skipped']
        checked.append(game_id)
        archive.close()
    closed = arc.close_scorecard(card)
    assert set(e.id for e in closed.environments)==set(checked) and len(checked)==25


def render_native_request(profile, body):
    from arena.candidate_profiles import PROFILES
    source = ROOT / f'tests/fixtures/{profile}-chat-template.jinja'
    manifest = json.loads(PROFILES[profile].manifest.read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['files']['chat_template.jinja']['sha256']
    def fail(message):
        raise ValueError(message)
    return jinja2.Environment().from_string(source.read_text()).render(
        messages=body['messages'], add_generation_prompt=True, raise_exception=fail,
        **body['chat_template_kwargs'])


@pytest.mark.parametrize('profile', ['ui-mate-9b', 'ui-mate-27b'])
async def test_native_query_survives_all_history_eviction(tmp_path, profile):
    from arena.candidate_profiles import PROFILES
    async with httpx.AsyncClient(base_url='http://unused/') as client:
        agent, wall, _ = setup_agent(tmp_path, client)
        agent.layer.config = replace(agent.layer.config, model=PROFILES[profile].model,
                                     model_revision=PROFILES[profile].revision)
        await agent.bridge.initialize()
        wall.release()
        agent.game.clock.release()
        initial = agent.observation([])
        # Reproduce the exact old malformed transcript, independent of its repair.
        old = agent.layer.payload([agent.system, initial])
        with pytest.raises(ValueError, match='No user query found'):
            render_native_request(profile, agent.layer.fresh(old))
        first = agent.messages(initial)
        task = first[1]
        assert task['role'] == 'user' and isinstance(task['content'], str)
        assert not task['content'].startswith('<tool_response>')
        agent.exchanges = [[initial, {'role': 'assistant', 'content': '<think>prior reasoning</think>Inspect.'}]] * 110
        async def count(payload):
            text = render_native_request(profile, payload)
            assert task['content'] in text and '<|vision_start|>' in text
            return 120000 if agent.exchanges else 1000
        agent.layer.count = count
        payload = await agent.payload(initial)
        assert not agent.exchanges and agent.omitted == 110
        assert payload['messages'][1] == task
        assert sum(m == task for m in payload['messages']) == 1
        assert '110 earlier complete exchanges' in json.dumps(payload)
        render_native_request(profile, agent.layer.fresh(payload))
        agent.bridge.archive.close()


def structured(*arguments, content=None, name='computer_use', reasoning=None):
    return {'role': 'assistant', 'content': content, 'reasoning': reasoning,
            'tool_calls': [{'id': f'call-{i}', 'type': 'function', 'function': {
                'name': name, 'arguments': json.dumps(args)}} for i, args in enumerate(arguments)]}


async def test_captured_vllm_response_executes_and_remains_in_history(tmp_path):
    from ui_mate_actor.interface import response_text
    fixture = json.loads((ROOT/'tests/fixtures/ui-mate-native-structured-response.json').read_text())
    message = fixture['response']['choices'][0]['message']
    bridge = await bridge_at(tmp_path)
    # Exact former behavior reproduces the silent zero-action failure.
    old = await execute(message['content'], bridge)
    assert old['calls'] == 0 and bridge.actions == 0
    result = await execute(message, bridge)
    assert result['calls'] == 1 and bridge.actions == 1
    assert bridge.archive.metadata()['action'] == {'action': 'ACTION6', 'x': 37, 'y': 44}
    restored = response_text(message)
    assert native.extract_xml_tool_calls(restored) == [{'action': 'left_click', 'coordinate': [585, 694]}]
    assert message['reasoning'] in restored
    bridge.archive.close()


async def test_structured_calls_prevent_duplicates_and_preserve_boundary_evidence(tmp_path):
    bridge = await bridge_at(tmp_path, clear_at=2)
    message = structured({'action': 'press', 'keys': '["7", "1", "2"]'},
                         {'action': 'press', 'keys': ['3']},
                         content=xml(action='press', keys=['4']),
                         reasoning='Example only: '+xml(action='press', keys=['4']))
    result = await execute(message, bridge)
    assert bridge.actions == 2
    assert bridge.archive.metadata(1)['action']['action'] == 'ACTION7'
    assert bridge.archive.metadata(2)['level_completed']
    assert all(r.get('skipped') for r in result['results'][2:])
    bridge.archive.close()


@pytest.mark.parametrize('message', [
    structured({'action': 'press', 'keys': ['1']}, name='bash'),
    {'tool_calls': [{'function': {'name': 'computer_use', 'arguments': '{bad'}}]},
    {'tool_calls': [{'function': {'name': 'computer_use', 'arguments': '[]'}}]},
    {'tool_calls': {'function': {'name': 'computer_use'}}},
    structured({'action': 'left_click', 'coordinate': '[true, 2]'}),
])
async def test_malformed_structured_calls_never_fall_back_to_text_actions(tmp_path, message):
    bridge = await bridge_at(tmp_path)
    message['content'] = xml(action='press', keys=['1'])
    result = await execute(message, bridge)
    assert bridge.actions == 0 and result['calls'] == 1
    assert result['results'][0]['error']
    bridge.archive.close()


@pytest.mark.parametrize('profile', ['ui-mate-9b', 'ui-mate-27b'])
async def test_actual_native_loop_handles_structured_stream_and_next_request(tmp_path, profile):
    from arena.candidate_profiles import PROFILES
    fixture = json.loads((ROOT/'tests/fixtures/ui-mate-native-structured-response.json').read_text())
    first = fixture['response']['choices'][0]['message']
    requests = []
    def endpoint(request):
        body = json.loads(request.content)
        if request.url.path == '/tokenize':
            render_native_request(profile, body)
            return httpx.Response(200,json={'count':200,'max_model_len':262144})
        requests.append(body)
        msg = first if len(requests)==1 else structured({'action':'finished','status':'success'})
        delta = {k:v for k,v in msg.items() if k in ['content','reasoning'] and v is not None}
        delta['tool_calls'] = [{**t,'index':i} for i,t in enumerate(msg['tool_calls'])]
        chunks = [{'choices':[{'index':0,'delta':delta,'finish_reason':'tool_calls'}]},
                  {'choices':[],'usage':{'prompt_tokens':200,'completion_tokens':160}}]
        return httpx.Response(200,content=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n')
    async with httpx.AsyncClient(base_url='http://model/v1/',transport=httpx.MockTransport(endpoint)) as client:
        agent,wall,gate=setup_agent(tmp_path,client)
        agent.layer.config=replace(agent.layer.config,model=PROFILES[profile].model,model_revision=PROFILES[profile].revision)
        await agent.bridge.initialize()
        wall.release()
        agent.game.clock.release()
        await agent.run()
        assert len(requests)==2 and agent.bridge.actions==1
        assert agent.metrics['no_action_replies']==0 and agent.metrics['tool_calls']==2
        assert agent.game.finished_reason=='model_finished_success'
        assert '<function=computer_use>' in json.dumps(requests[1]['messages'])
        assert 'ACTION6' in json.dumps(requests[1]['messages'])
        assert gate.inflight==gate.reserved==0
        agent.bridge.archive.close()
