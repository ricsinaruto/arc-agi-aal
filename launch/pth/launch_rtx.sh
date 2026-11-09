#!/bin/bash
# launch_rtx.sh RUN BUNDLE_DIR "WHAT" (MV=<the building branch's modal_validation.py>, default sft-prune): queue row, Nebius RTX launch, record; watchers: play_watch, play metrics, hang guard
cd .
RUN=$1; B=$2; export RUN WHAT="$3"
# every RTX play matches the v87b submission stack (KV gate admission, stream handling); a
# serving bench or other non-play bundle opts out with V87B_STACK=0 V87B_WHY="<reason>"
if [ "${V87B_STACK:-1}" = 1 ]; then
  python3 ./.claude/worktrees/sft-heal/launch/pth/check_v87b_stack.py $B | tail -8; [ ${PIPESTATUS[0]} = 0 ] || { echo "REFUSED: not the v87b stack (check_v87b_stack.py)"; exit 1; }
else
  [ -n "${V87B_WHY:-}" ] || { echo "REFUSED: V87B_STACK=0 needs V87B_WHY"; exit 1; }; echo "v87b stack check skipped: $V87B_WHY"
fi
# held-out reads (HELDOUT_POOL="<pool index> ..."): refuse a roster that shares a game with a training pool (27 Oct)
if [ -n "${HELDOUT_POOL:-}" ]; then
  ./.venv/bin/python ./.claude/worktrees/sft-heal/launch/pth/check_heldout_roster.py $B $HELDOUT_POOL || { echo "REFUSED: held-out roster overlaps training"; exit 1; }
fi
# refuse a bundle whose code would ignore part of its study (c1play-v1)
ARC_NEBIUS_SCRIPTS=./.claude/worktrees/nebius-opd/scripts python3 ./.claude/worktrees/sft-heal/launch/pth/check_play_bundle.py $B | tail -3; [ ${PIPESTATUS[0]} = 0 ] || { echo "REFUSED: bundle check"; exit 1; }
# refuse a bundle whose run paths name another play (24 Oct: g1-v18 = v22's paths, relaunched as v24/v26/v27; the plays
# would follow v22's exchange dir)
python3 - "$B" "$RUN" <<'PY' || { echo "REFUSED: bundle paths name another run (build_play.py --rename-play OLD=$RUN)"; exit 1; }
import json, re, sys
s = open(sys.argv[1] + "/configs/comparison.json").read()
other = sorted({m for m in re.findall(r"/runs/validation/(b2-[A-Za-z0-9_.-]+)/", s) if m != sys.argv[2]})
sys.exit(f"bundle paths name {other}, not {sys.argv[2]}" if other else 0)
PY
# queue row end = boot (baked image 1,450 s measured, else 2,450 s) + every trial's wall + server restarts (~1,130 s) + teardown 300 s
export DUR=$(python3 -c "
import json,sys; c=json.load(open('$B/configs/comparison.json'))
walls=[json.load(open('$B/'+t['study'])).get('wall_seconds', c['wall_seconds']) if isinstance(t,dict) and t.get('study') else c['wall_seconds'] for t in (c.get('trials') or [None])] if c.get('chain') else [c['wall_seconds']]
print(int((1450 if c.get('nebius_image_id') else 2450) + sum(walls) + 1130 * (c.get('chain') or {}).get('server_restarts', 0) + 300))")
# run rows only through records: nebius_validation's reserve() writes docs/runs/<run>.json and the
# live queue row (what = ARC_WHAT); this script only pulls first and pushes the record after
./.venv/bin/python ./tools/ops/records.py pull > ~/arc-sft-data/mon/records-$RUN.log 2>&1
export ARC_WHAT="$WHAT" ARC_ARM=sft
L=$(date +%s)
ARC_NEBIUS_SCRIPTS=${ARC_NEBIUS_SCRIPTS:-./.claude/worktrees/nebius-opd/scripts} ARC_NEBIUS_CONFIG=./runs/ops/nebius.json tools/ops/nlaunch.sh ${MV:-./.claude/worktrees/sft-prune/scripts/modal_validation.py} b200-run $B $RUN ${REASON:+--reason "$REASON"} 2>&1 | tail -1 | cut -c1-160
IID=$(python3 -c "import json;print(json.load(open('docs/validation-nebius/nebius-$RUN.json')).get('instance_id') or '')" 2>/dev/null)
# the launcher's reserve writes no attempts row and its queue row bypasses records: write both through records
[ -n "$IID" ] && ./.venv/bin/python ./.claude/worktrees/sft-heal/launch/pth/rec_attempt.py $RUN >> ~/arc-sft-data/mon/records-$RUN.log 2>&1
./.venv/bin/python ./tools/ops/records.py show $RUN 2>/dev/null | grep -q '"ledger"' || echo "ALERT: no run record for $RUN (reserve did not write docs/runs)"
./.venv/bin/python ./tools/ops/records.py push $RUN >> ~/arc-sft-data/mon/records-$RUN.log 2>&1
echo "WATCH: HANG=$((DUR + 1800)) ./.claude/worktrees/sft-heal/launch/pth/watch_rtx.sh $RUN $L ${MINTOK:-1000}  (run as the agent's background task)"
cd .; git add docs/validation-nebius/nebius-$RUN.json 2>/dev/null && git commit -q -m "records: $RUN

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- docs/validation-nebius/nebius-$RUN.json && { for i in 1 2 3; do git push -q && break; ./.venv/bin/python ./tools/ops/records.py pull >/dev/null 2>&1; done; }
