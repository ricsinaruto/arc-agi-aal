#!/bin/bash
# play_metrics_watch.sh RUN: every 10 min reads the play-check measures (tool-call share, games acting, actions per game)
# from the run's game logs; alerts and exits (the agent decides) if fewer than half of >= 500 replies call a tool, or if
# the cap or loop share passes 5% at >= 300 replies
cd .; run=$1
while true; do
  ./.claude/worktrees/sft-heal/launch/pth/vm_gone.sh $run && { echo "$(date -u +%H:%M) VM gone"; exit 0; }
  r=$(python3 ./.claude/worktrees/sft-heal/launch/pth/watch/play_read.py $run 2>/dev/null)
  echo "$(date -u +%H:%M) $(echo "$r" | python3 -c "import json,sys;d=json.load(sys.stdin);g=max(d['games'],1);print('replies',d['replies'],'tool_share',d['tool_call_share'],'games_acting',d['games_with_actions_share'],'actions_per_game',round(d['actions_total']/g,2),'cap',d.get('cap_share'),'loop',d.get('loop_share'))" 2>/dev/null)"
  # loops / caps (19 Oct): healthy warm1 sampling cap 1.5% / loop 1.1%, OPD round 1 35% / 35%; alert above 5% at >= 300 replies
  if echo "$r" | python3 -c "import json,sys;d=json.load(sys.stdin);sys.exit(0 if d['replies']>=300 and max(d.get('cap_share') or 0,d.get('loop_share') or 0)>0.05 else 1)" 2>/dev/null; then
    echo "ALERT $run: cap or loop share above 5% ($(echo "$r" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('cap_share'),d.get('loop_share'))"))"; exit 1
  fi
  if echo "$r" | python3 -c "import json,sys;d=json.load(sys.stdin);sys.exit(0 if d['replies']>=500 and (d['tool_call_share'] or 0)<0.5 else 1)" 2>/dev/null; then
    echo "ALERT $run: tool-call share below 50%"; exit 1
  fi
  sleep 600
done
