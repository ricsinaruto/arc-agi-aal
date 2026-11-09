"""Validate an arena bundle inside Linux x86_64 with no network or GPU.

Mount this script and the bundle read-only into python:3.12.11-slim-bookworm.
Use Docker --platform linux/amd64 --network none --read-only --tmpfs /tmp:exec.
This checks packaging and imports, not model loading, sandbox execution or gameplay.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--games-dir', type=Path,
                        help='Also exercise the competition gateway with public environments (CPU only)')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Run inside a Linux x86_64 container, not on the host')
    bundle = args.bundle.resolve()
    sys.path.insert(0, str(bundle))
    from bootstrap import install_agent_environment, verify
    from bootstrap_arena import stage_pi

    started = time.monotonic()
    identity = verify(bundle)
    with tempfile.TemporaryDirectory(prefix='arc-offline-') as temporary:
        work = Path(temporary)
        env = {**os.environ, 'PYTHONPATH': str(bundle / 'src'), 'PYTHONNOUSERSITE': '1',
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PIP_NO_INDEX': '1',
               'MPLBACKEND': 'Agg', 'PYTHONDONTWRITEBYTECODE': '1'}
        python = install_agent_environment(bundle, work / 'agent-venv', env)
        pi, node = stage_pi(bundle, work)
        program = """import json,sys,numpy,scipy,arc_agi,arena.candidate_serving
from pathlib import Path
from arena.trial import duck_config,qwen_config
s=json.load(open(sys.argv[1]))
if s.get('competition'):
    from arena.competition import open_arcade, ScorecardOwner
    from taaf.competition_arcade import CompetitionArcadeServer
    assert callable(open_arcade) and callable(ScorecardOwner) and callable(CompetitionArcadeServer)
    if sys.argv[2]:
        from arcengine import GameAction
        import os
        os.environ.pop('KAGGLE_IS_COMPETITION_RERUN', None)
        with open_arcade(s,sys.argv[2],Path('/tmp/gateway-validation')) as (resolved,owner):
            card=owner.open_scorecard(tags=['offline-cpu-validation'])
            envs=[owner.make(g,scorecard_id=card,save_recording=False) for g in resolved['games']]
            assert len(envs)==110 and all(e is not None for e in envs)
            snapshot=numpy.array(envs[len(s['games'])].observation_space.frame[-1],copy=True)
            assert numpy.array_equal(snapshot,envs[0].observation_space.frame[-1])
            action=envs[0].observation_space.available_actions[0]
            result=envs[0].step(GameAction.from_id(action),data={'x':32,'y':32} if action==6 else {})
            assert result is not None and numpy.array_equal(snapshot,envs[len(s['games'])].observation_space.frame[-1])
            closed=owner.close_scorecard(card)
            assert {g.id for g in closed.environments}==set(resolved['games'])
        print('CPU competition gateway: 110 independent games, action, and scorecard close passed')
if s.get('action_plans'):
    # Plans offline (30 September competition build): run_plan in the persistent sandbox (both loops), no network.
    from duck_transfer.history import preserve_transition_results
    from duck_transfer.loader import load
    from duck_transfer.sandbox import StatefulSandbox, stateful_bootstrap, with_plans
    from arena.action_plans import PROMPT
    assert 'run_plan' in PROMPT
    g=load().run_sandboxed_python.__globals__
    public=preserve_transition_results(g['_SANDBOX_BOOTSTRAP'])
    def board(step):
        return {'current_frame':{'ascii':'','step':step,'level':1,'shape':[2,2],'grid':[[step%10]*2]*2},
                'history':[],'valid_actions':['UP'],'last_action_result':{}}
    for fixes in (False, True):
        box=StatefulSandbox(with_plans(stateful_bootstrap(public)),color_chars=g['ARC_COLOR_CHARS'])
        if fixes:
            box.configure_fixes(with_plans(stateful_bootstrap(public,fixes=('modules',))),interrupt=True,grace=3)
        steps=[0]
        def handler(actions):
            steps[0]+=1
            r={'executed':True,'action_display':actions[0]['action'],'board_changed':True,'level_completed':False,
               'game_over':False,'done':False,'state':'NOT_FINISHED','level':1}
            return {'action_result':r,'state':board(steps[0])}
        out=box.run(code="import json\\nprint(json.dumps(run_plan(['UP','UP'])))",timeout_seconds=30,
                    initial_state=board(0),action_handler=handler)
        box.close()
        plan=json.loads(out['stdout'])
        assert not out['error'] and plan['executed']==2 and steps[0]==2, out
    print('Action plans: run_plan executed 2 actions in both sandbox loops offline')
native=any(t['kind']=='ui-mate-native' for t in s['trials'])
for window in s['windows']:
    common=dict(window=window,cap=s['completion_cap'],wall=s['wall_seconds'],base_url='http://localhost/v1')
    c=qwen_config(s,'ui-mate-native' if native else 'pi',**common)
    d=duck_config(s,Path(sys.argv[1]).parent.parent,**common)
    assert c.compaction_tokens+c.output_tokens_cap==d.max_model_len==window
    assert c.output_tokens_cap==d.max_output_tokens==s['completion_cap']
    assert c.context_window==s['server']['max_model_len']
    print('Configured context validated:',window,'output:',s['completion_cap'],'native:',c.context_window)
if native:
    from ui_mate_actor.agent import NativeAgent
    from ui_mate_actor.interface import system_prompt
    from ui_mate_actor.preflight import check, ProbeEnvironment
    assert 'computer_use' in system_prompt(1800)
    assert callable(check) and ProbeEnvironment().observation_space.frame[0].shape == (64,64)
print(c.model,c.reasoning_effort,'native actor import OK' if native else '')
"""
        probe = subprocess.run([python, '-c', program, str(bundle / 'configs/comparison.json'),
                                str(args.games_dir) if args.games_dir else ''],
                               env=env, capture_output=True, text=True, check=True, timeout=300)
        js = subprocess.run([str(node), '--input-type=module', '-e',
            "const m=await import('@earendil-works/pi-coding-agent'); "
            "if(typeof m.createAgentSession!=='function')throw Error('missing SDK'); "
            "const {configureHttpDispatcher}=await import(new URL('core/http-dispatcher.js',"
            "import.meta.resolve('@earendil-works/pi-coding-agent'))); "
            "configureHttpDispatcher(0); "
            "console.log(process.version,process.arch,'Pi SDK and HTTP dispatcher import OK');"],
            cwd=pi, env={**env, 'PATH': str(node.parent) + ':' + os.environ['PATH']},
            capture_output=True, text=True, check=True, timeout=120)
        result = {'status': 'passed', 'kind': 'Linux x86_64 offline bundle check, no GPU',
                  'source_commit': identity['git_commit'], 'verified_files': len(identity['files']),
                  'imports': probe.stdout.strip(), 'node_pi': js.stdout.strip(),
                  'seconds': time.monotonic() - started}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
