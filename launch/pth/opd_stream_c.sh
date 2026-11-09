#!/bin/bash
# opd_stream_c.sh: OPD layout C: 2 trainer B300 + 1 FP8 one-card teacher B300 (chains 2, chain
# lock, fla pin, auto-restart) + the student play on 1 Nebius RTX (rows reload every 2 steps, age cap 8). Order: the
# play boots first (25-41 min), the exact-args CPU preflight starts at once, the round launches beside it (PARALLEL_PF,
# pf_guard) once the play writes rollouts. Layout knobs: HCARDS/HTRANKS/HRES (3+1: HCARDS=4 HTRANKS=3 HRES=..-r3/train).
set -u
O=./.claude/worktrees/sft-heal/launch/pth; M=~/arc-sft-data/mon; V=/runs/validation
D=$V/sft-heal-round-sA-D-1e5/train; NAME=${NAME:-opdS-c4}; PLAY=${PLAY:-b2-sft4x-opdS-g1-v13}; PB=${PB:-sft4x-opdS-g1-v9}
H2=$V/sft-heal-round-tA-dh10/teacher
# PLAY="LIVE,BACKLOG...": the round never waits on the live play when recorded plays exist; it starts on
# their version-0 backlog while the live play boots (the live play is PENDING in launch_gates and leads the confirm path)
LIVE=${PLAY%%,*}; READY=$(echo "$PLAY" | cut -s -d, -f2- | tr ',' ' '); READY=${READY:-$LIVE}
# READY_PLAYS (c7h, 24 Oct): the plays the round waits on, when the confirm play (first) is live and a later one boots
[ -n "${READY_PLAYS:-}" ] && READY=$(echo "$READY_PLAYS" | tr ',' ' ')
[ "$READY" = "$LIVE" ] || export ARC_HEAL_STREAM_PLAY_PENDING=${ARC_HEAL_STREAM_PLAY_PENDING:-$LIVE}
BASES=${BASES:-b2-sft4x-opdS-bg1-v8,b2-sft4x-opdS-bg1-v6,b2-sft4x-opdS-bg1-v5,b2-sft4x-opdS-bg1-v4,b2-sft4x-opdS-bg1-v3,b2-mfu1-bg1h-v2,b2-mfu1-bg1-v5-boot}
export ARC_HEAL_STREAM_BASE_FILL=${ARC_HEAL_STREAM_BASE_FILL-'/runs/heal-data/opd-*/b-*'} ARC_HEAL_STREAM_PRUNE=/opt/arc-validation/rl1/heal_prune-t75-excl5-v1.json ARC_HEAL_LOOP_SHARE=${LOOP_SHARE:-0} ARC_HEAL_STREAM_BASE=$BASES ARC_HEAL_STREAM_PLAY=$PLAY ARC_HEAL_STREAM_MAX_AGE=${MAX_AGE:-8} ARC_HEAL_STREAM_BAR_STEP=${BAR_STEP:-340} ARC_HEAL_STREAM_BAR_TEACHER=2960
export TRUNC=${TRUNC:-0} BATCH=${BATCH:-200000} CKAT=${CKAT:-8}
# a resume (HSTEP0) checkpoints only after its start step (c7e, 24 Oct: the plan refused 1,6 below step 7)
CKAT=$(echo "$CKAT" | tr ',' '\n' | awk -v s=${HSTEP0:-0} '$1 > s' | paste -sd, -)
E="HEAL_CAPTURE_CACHE=1,HEAL_GRAD_SYNC=step,HEAL_EXPERT_ROWS=1,HEAL_SAMPLED=1,HEAL_OPD_LOG=1,HEAL_DELTA_EVERY=${DELTA_EVERY:-1},HEAL_STREAM_STALL=1800,HEAL_ALLOW_TOPK_TTOK=1,HEAL_STREAM_BASE_SHARE=${BSHARE:-0.2},HEAL_HELDOUT2=$H2,HEAL_HELDOUT_REVERSE=1,HEAL_SAMPLED_METRIC=1,HEAL_CLASS_SPLIT=1,HEAL_MID_READ_EVERY=${MIDR:-8},HEAL_MID_READ_FILES=5,HEAL_SHARD_READS=1,HEAL_SAMPLED_MODE=topk,HEAL_INERT_PREFIX=1,HEAL_STREAM_COALESCE=1,HEAL_CHUNK_AFTER=5:16384${EXTRA_E:+,$EXTRA_E}${DELTA_RUN:+,HEAL_DELTA_RUN=$DELTA_RUN}"
E=$E WIN=${HWIN:-32768} CKAT=$CKAT STEP0=${HSTEP0:-0} TOTAL=${HSSTEPS:-500} python3 $O/prelaunch_check.py || { echo "ALERT prelaunch check failed"; exit 1; }
run() {  # the exact launch flags, shared by the preflight and the round
  env "$@" NAME=$NAME KIND=heal LR=${HLR:-3.5e-6} REV=0 FWD=${HFWD:-0.25} TDIR=$V/sft-heal-round-tA-r1/teacher BUNDLE=${HBUNDLE:-dist/sft-heal-round-v64} \
    ESFT=0.12 CARDS=${HCARDS:-3} TRANKS=${HTRANKS:-2} STUDENT=$D/student RESUME=${HRES:-$D} STEP0=${HSTEP0:-0} ROUTING=$D/routing.pt SCHED=wsd SSTEPS=${HSSTEPS:-500} WARM=0 HTOTAL=${HSSTEPS:-500} \
    WIN=${HWIN:-32768} TMODEL=/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540 TSPREAD=1 TENV=${TENV:-RL1_EXPERTS_FP8=1,RL1_STREAM_CHAINS=1} TO=${TO:-25200} DUR=${DUR:-16200} GATE=0 \
    SMOKE_STEPS=2 ARC_HEAL_ENV=$E \
    WHAT="${HWHAT:-OPD layout C: student D plays ft09 ls20 re86 sb26 vc33 on one Nebius RTX ($PLAY, weights applied in place every step); ${HTRANKS:-2} B300 train on its streamed 32K chunks (80%) and recorded base-play chunks (20%) with sampled top-128 reverse KL + forward KD 0.25, lr 3.5e-6 WSD, re-warm 2; one B300 scores everything with the FP8-expert teacher (1 chain; chains > 1 refused). Why: the long OPD run at >90% GPU util. Judged vs D on dh10 sampled reverse / forward / call-class KL and teacher-H, mid reads every 8 steps; util, data age and data wait watched.}" \
    NOTE="OPD layout C (2 trainer + 1 teacher B300 + 1 RTX)" $O/launch_heal.sh
}
PF=${PFREUSE:-sft-heal-round-$NAME-pf${PFTAG:-}}
python3 $O/check_play_paths.py ./.claude/worktrees/sft-prune/dist/$PB $LIVE sft-heal-round-$NAME ${DELTA_RUN:-} || { echo "ALERT play bundle paths do not match this launch"; exit 1; }
# play first: it boots while the preflight runs, so the trainers start on live rollouts
[ "${SKIP_PLAY:-}" = 1 ] || (cd . && ARC_NEBIUS_SCRIPTS=./.claude/worktrees/nebius-opd/scripts \
  REASON="OPD layout C student play: 5 games x 4 copies to 500K tokens (~3.75 h of play) for the long OPD run" \
  $O/launch_rtx.sh $LIVE ./.claude/worktrees/sft-prune/dist/$PB \
  "OPD layout C play: student D (1 RTX) plays ft09 ls20 re86 sb26 vc33 to 500K tokens each, recording engine log-probs and weight versions per reply; rollouts stream to the bucket every 15 s for the B300 round sft-heal-round-$NAME, whose weights it applies in place every step ('rows' reload). Judged by the round's KL bars and data age." 2>&1 | tail -2)
[ "${REUSE_PF:-}" = 1 ] || { rm -f $M/preflight-$PF.log; NAME=$NAME-pf${PFTAG:-} run PREFLIGHT=1 > $M/pfl-$NAME.out 2>&1 & }
until grep -q "PREFLIGHT CALL\|PREFLIGHT OK\|PREFLIGHT FAILED" $M/preflight-$PF.log 2>/dev/null; do sleep 20; done
[ "${PF_ONLY:-}" = 1 ] && { echo "preflight $PF launched"; exit 0; }
! grep -q "PREFLIGHT FAILED" $M/preflight-$PF.log || { echo "ALERT preflight FAILED before launch"; exit 1; }
until python3 $O/play_ready.py $READY > $M/playready-$NAME.log 2>&1; do sleep 60; done
echo "$(date -u +%H:%M) play writing rollouts: $(tr '\n' ' ' < $M/playready-$NAME.log)"
out=$(run PARALLEL_PF=1 PFRUN=$PF 2>&1 | tail -1); echo "$(date -u +%H:%M) round: $out"
echo "$out" | grep -v "Merge with strategy" | grep -q "REFUSED\|no passing\|failed" && { echo "ALERT round not launched: $out"; exit 1; }
A=$(echo "$out" | grep -o 'ap-[A-Za-z0-9]*' | head -1)
[ -n "$A" ] || A=$(grep -o 'ap-[A-Za-z0-9]\{22\}' $M/launch-sft-heal-round-$NAME.log 2>/dev/null | head -1)
[ -n "$A" ] || { echo "ALERT round not launched"; exit 1; }
# c4 (23 Oct): the app runs the CPU plan first; a refusal there leaves an app id but no round
! grep -q "REFUSED" $M/launch-sft-heal-round-$NAME.log 2>/dev/null || { echo "ALERT round refused in its app $A: $(grep REFUSED $M/launch-sft-heal-round-$NAME.log | tail -1)"; exit 1; }
(./.venv/bin/python ./tools/ops/records.py ledger sft-heal-round-$NAME --set app_id="\"$A\"" && ./.venv/bin/python ./tools/ops/records.py queue sft-heal-round-$NAME --set app_id="\"$A\"" && ./.venv/bin/python ./tools/ops/records.py push sft-heal-round-$NAME) >> $M/records-sft-heal-round-$NAME.log 2>&1
echo "APP $A"
$O/pf_guard.sh $PF $A > $M/pfguard-$NAME.log 2>&1 &
