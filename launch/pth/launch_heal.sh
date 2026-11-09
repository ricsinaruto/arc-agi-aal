#!/bin/bash
# launch_heal.sh: Stage A trainer runs on 2 B300 from the tA-v1 teacher files (no teacher phase): lr probes
# no wall-clock limits on training; TO is only a crash ceiling well above the expected length.
# (KIND=probe: TOTAL steps, no save, held-out reply KL on teacher-H) and heals (KIND=heal: auto schedule, checkpoints, save).
# RUN=sft-heal-round-<NAME>; TDIR = teacher folder(s) under tA-v1; LR; WIN backprop window (16K).
# Gate: mon/smoke-ok-trainer (B300 smoke v6+ passed with bundle v19) + mon/preflight-$RUN.log PREFLIGHT OK.
RUN=sft-heal-round-${NAME:?}; TO=${TO:?crash ceiling in seconds, well above the expected length}; KIND=${KIND:?probe or heal}; T=/runs/validation/sft-heal-round-tA-v1
if [ "$KIND" = probe ]; then  # PSAVE=1: the probe saves its adapter at the end (a play eval of it later)
  X="ARC_HEAL_TOTAL_STEPS=${TOTAL:-6} ARC_HEAL_NO_SAVE=$([ "${PSAVE:-}" = 1 ] || echo 1) ARC_HEAL_CHECKPOINT_SECONDS=0 ARC_HEAL_HELDOUT=$T/teacher-H ARC_HEAL_HELDOUT_FILES=20"
else
  # <=30 min while a save costs 115-444 s (optimum ~sqrt(2 x save x MTBF) = 60-80 min); 10 min after v25's sharded save
  X="ARC_HEAL_TOTAL_STEPS=${HTOTAL:-0} ARC_HEAL_NO_SAVE= ARC_HEAL_CHECKPOINT_SECONDS=${CKPTS:-1800} ARC_HEAL_CHECKPOINT_AT=${CKAT:-} ARC_HEAL_HELDOUT=$T/teacher-H ARC_HEAL_HELDOUT_FILES=20"
fi
# MFU S1+S3 adopted 21 Oct (B300 reads: step -15%, peak -13.5 GiB, held-out unchanged); ARC_HEAL_ENV= turns them off
export ARC_HEAL_ENV=${ARC_HEAL_ENV-HEAL_CAPTURE_CACHE=1,HEAL_GRAD_SYNC=step,HEAL_EXPERT_ROWS=1}
# WSD (OPD, 21 Oct): SCHED=wsd SSTEPS=<whole-run steps> [DFROM= DSTEPS=: a decay branch]; re-warm REWARM (2) steps on every start/resume
if [ "${SCHED:-cosine}" = wsd ]; then
  ARC_HEAL_ENV="${ARC_HEAL_ENV:+$ARC_HEAL_ENV,}HEAL_SCHEDULE_STEPS=${SSTEPS:?whole-run steps},HEAL_WSD_DECAY=0,HEAL_REWARM=${REWARM:-2}${DSTEPS:+,HEAL_DECAY_FROM=$DFROM,HEAL_DECAY_STEPS=$DSTEPS}"
fi
# GATE=auto: the real-trainer gate runs only when the bundle's GPU-path digest has not
# passed a gate before (scripts/posttrain/gpu_digest.py; passes recorded in mon/gate-passed-gpu.txt by gate_record.sh)
GDIG=$(cd ./.claude/worktrees/sft-heal && python3 scripts/posttrain/gpu_digest.py ${BUNDLE:-dist/sft-heal-round-v19}/rl1 2>/dev/null)
if [ "${GATE:-}" = auto ]; then
  if grep -q "^$GDIG " ~/arc-sft-data/mon/gate-passed-gpu.txt 2>/dev/null; then GATE=; echo "GPU digest $GDIG passed a gate before: no gate"
  else GATE=1; echo "GPU digest $GDIG new: gate runs"; fi
fi
export $X ARC_RL1_BUNDLE=${BUNDLE:-dist/sft-heal-round-v19} ARC_HEAL_WINDOW=${WIN:-16384} ARC_HEAL_TEACHER_DIR=${TDIR:?} ARC_HEAL_EXTRA_TEACHER= \
ARC_RL1_AUTHORIZATION="Stage A $KIND on 2 B300" \
ARC_HEAL_SEGMENTS=/runs/heal-data/stageA-v1 ARC_HEAL_STUDENT=${STUDENT:-/runs/validation/sft-heal-round-cut1-v2/train/student} \
ARC_HEAL_FROZEN=/assets/sft-heal/student-p75d4 ARC_HEAL_RESUME=${RESUME-/runs/validation/sft-heal-round-cut1-v2/train} \
ARC_HEAL_STEP0=${STEP0:-0} ARC_HEAL_GENERAL_WINDOWS=256 ARC_HEAL_TRAIN_SECONDS=${TRAINS:-1000000} \
ARC_HEAL_BATCH_TOKENS=${BATCH:-250000} ARC_HEAL_POSITIONS=reply ARC_HEAL_REVERSE=${REV:-0} ARC_HEAL_FORWARD=${FWD:-1} ARC_HEAL_LR=${LR:?} \
ARC_HEAL_MTP_WEIGHT=0.1 ARC_HEAL_ESFT=${ESFT:-0.12} ARC_HEAL_SCHEDULE=${SCHED:-cosine} ARC_HEAL_WARMUP=${WARM:-0.1} \
ARC_HEAL_TRUNCATE_LOOPS=${TRUNC:-1} ARC_HEAL_CARDS=${CARDS:-2} ARC_HEAL_TRANKS=${TRANKS:-} ARC_HEAL_TEACHER_ENV=${TENV:-} ARC_HEAL_TIMEOUT=$TO ARC_HEAL_TEACHER_TOKENS=0 ARC_HEAL_SMOKE= ARC_HEAL_TEACHER_ONLY= \
ARC_HEAL_TEACHER_MODEL=${TMODEL:-} ARC_HEAL_TEACHER_SPREAD=${TSPREAD:-1} ARC_HEAL_ROUTING=${ROUTING:-} ARC_HEAL_GATE=${GATE:-} ARC_HEAL_SMOKE_STEPS=${SMOKE_STEPS:-2}
# typo-class guard (21 Oct, v30 crashed 4 B300 arms on a duplicate event kwarg): static checks of the bundle's trainer
( cd ./.claude/worktrees/sft-heal && HEAL_KD_SRC=$ARC_RL1_BUNDLE/rl1/heal_kd.py ./.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_heal_event_kwargs.py >/dev/null 2>&1 ) || { echo "REFUSED: tests/test_heal_event_kwargs.py fails"; exit 1; }
# code manifest (settings/state-only launches need no preflight): the bundle's trainer code + the launcher
codehash() { cat ./.claude/worktrees/sft-heal/${ARC_RL1_BUNDLE}/rl1/*.py ./.claude/worktrees/sft-heal/scripts/posttrain/modal_rl1.py 2>/dev/null | shasum -a 256 | cut -c1-16; }
PASSED=~/arc-sft-data/mon/preflight-passed-code.txt
if [ "$PREFLIGHT" = 1 ]; then
  # 1 held-out file on the tiny student: 20 real files with the reference GDN kernels outran round A's 2,400 s (21 Oct)
  # queue row at launch (every app, preflights included, has a described row from the start)
  ( PFL=~/arc-sft-data/mon/preflight-$RUN.log; for i in $(seq 60); do APP=$(grep -o 'ap-[A-Za-z0-9]\{20,\}' $PFL 2>/dev/null | head -1); [ -n "$APP" ] && break; sleep 10; done
    ./.venv/bin/python ./tools/ops/records.py reserve pf-$RUN --arm sft --platform modal --gpu "CPU (preflight)" --gpu-count 0 --bound 2 --max-seconds 5400 --kind cpu-preflight \
      --auth "launch-gate preflight of $RUN" --set app_id="\"$APP\"" \
      --what "CPU launch-gate preflight (tiny model, gloo ranks, Modal CPU) of ${ARC_RL1_BUNDLE##*/} with $RUN's own launcher arguments: start, steps, checkpoint, resume, next round. A FAILED verdict stops the run it gates." \
      >> ~/arc-sft-data/mon/records-$RUN.log 2>&1 && ./.venv/bin/python ./tools/ops/records.py push pf-$RUN >> ~/arc-sft-data/mon/records-$RUN.log 2>&1 ) &
  export RUN ARC_HEAL_HELDOUT_FILES=1; cd ./.claude/worktrees/sft-heal && ./.venv/bin/modal run --detach scripts/posttrain/heal_preflight.py \
    ${PFDATA:+--data $PFDATA} ${PFGENERAL:+--general $PFGENERAL} > ~/arc-sft-data/mon/preflight-$RUN.log 2>&1
  grep -q "PREFLIGHT OK\|PREFLIGHT FAILED" ~/arc-sft-data/mon/preflight-$RUN.log || ./.claude/worktrees/sft-heal/launch/pth/pf_fetch.sh $RUN
  tail -1 ~/arc-sft-data/mon/preflight-$RUN.log; grep -q "PREFLIGHT OK" ~/arc-sft-data/mon/preflight-$RUN.log && echo "$(codehash) $RUN" >> $PASSED
  (./.venv/bin/python ./tools/ops/records.py settle pf-$RUN completed --cause completed --set note="\"$(tail -1 ~/arc-sft-data/mon/preflight-$RUN.log | cut -c1-120 | tr -d '"\\')\"" && ./.venv/bin/python ./tools/ops/records.py push pf-$RUN) >> ~/arc-sft-data/mon/records-$RUN.log 2>&1; exit 0
fi
[ -f ~/arc-sft-data/mon/smoke-ok-trainer ] || { echo "no passing B300 trainer smoke"; exit 1; }
# no B300 before its plays write rollouts; live GPUs reconcile with the queue and caps
[ -n "${DRY:-}" ] || ./.claude/worktrees/sft-heal/launch/pth/launch_gates.sh || exit 1
# a resume needs one optimizer shard per rank (opd-r1b, 22 Oct: a 2-rank D resumed on 4 cards crashed in the gate)
if [ -n "${RESUME:-}" ]; then NR=$(./.venv/bin/modal volume ls arc-prolong-runs "${RESUME#/runs}" 2>/dev/null | grep -c "optim-rank[0-9]*\.pt$")
  [ "$NR" = "${TRANKS:-${CARDS:-2}}" ] || { echo "REFUSED: $RESUME has $NR optimizer shards for ${TRANKS:-${CARDS:-2}} trainer ranks"; exit 1; }; fi
# faster gate: PARALLEL_PF=1 launches beside a running preflight; pf_guard.sh stops the run if it fails
if grep -q "^$(codehash) " $PASSED 2>/dev/null; then echo "code manifest $(codehash) passed before ($(grep "^$(codehash) " $PASSED | tail -1 | cut -d" " -f2)): no preflight needed"
elif [ "$PARALLEL_PF" = 1 ]; then PFR=${PFRUN:-$RUN}  # PFRUN: the same bundle's running preflight under another run name
  grep -q "PREFLIGHT CALL" ~/arc-sft-data/mon/preflight-$PFR.log || { echo "no running preflight for $PFR"; exit 1; }
  ! grep -q "PREFLIGHT FAILED" ~/arc-sft-data/mon/preflight-$PFR.log || { echo "preflight failed for $PFR"; exit 1; }
else grep -q "PREFLIGHT OK" ~/arc-sft-data/mon/preflight-$RUN.log || { echo "no passing preflight for $RUN"; exit 1; }; fi
export RUN AUTH="$ARC_RL1_AUTHORIZATION" BOUND=$(python3 -c "print(($TO + 60) * (${CARDS:-2} * 0.001972) * 1.15)") MAXS=$TO DUR=${DUR:?expected seconds for the queue row end (TO is only the crash ceiling)} GPUROW=B300:${CARDS:-2} \
WHAT="${WHAT:?}" NOTE="${NOTE:?}"
# run rows only through records.py: gate + ledger + live row, then push that record alone
cd . && ./.venv/bin/python ./tools/ops/records.py pull > ~/arc-sft-data/mon/records-$RUN.log 2>&1
./.venv/bin/python ./tools/ops/records.py reserve $RUN --arm sft --platform modal --gpu "${CARDS:-2} x B300" --gpu-count ${CARDS:-2} --bound $BOUND --max-seconds $MAXS \
  --kind heal-round --auth "$AUTH" --what "$WHAT" --set note="$(python3 -c 'import json,os;print(json.dumps(os.environ["NOTE"]))')" \
  --set end_epoch=$(( $(date +%s) + DUR )) >> ~/arc-sft-data/mon/records-$RUN.log 2>&1 \
  || { echo "REFUSED: records.py reserve: $(tail -1 ~/arc-sft-data/mon/records-$RUN.log)"; exit 1; }
./.venv/bin/python ./tools/ops/records.py push $RUN >> ~/arc-sft-data/mon/records-$RUN.log 2>&1
cd .claude/worktrees/sft-heal
./.venv/bin/modal run scripts/posttrain/modal_rl1.py --run-id $RUN --dry-run > ~/arc-sft-data/mon/dry-$RUN.log 2>&1 || { echo "dry run failed: mon/dry-$RUN.log"; exit 1; }
./.venv/bin/modal run --detach scripts/posttrain/modal_rl1.py --run-id $RUN > ~/arc-sft-data/mon/launch-$RUN.log 2>&1
APP=$(grep -o 'ap-[A-Za-z0-9]*' ~/arc-sft-data/mon/launch-$RUN.log | head -1); echo "$RUN $APP $(grep -c Error ~/arc-sft-data/mon/launch-$RUN.log) errors"
[ "${GATE:-}" = 1 ] && [ -n "$GDIG" ] && nohup ./.claude/worktrees/sft-heal/launch/pth/gate_record.sh $RUN $GDIG > ~/arc-sft-data/mon/gate-record-$RUN.log 2>&1 &
cd . && git add docs/validation-rl/modal-$RUN.json && git commit -q -m "records: $RUN

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- docs/validation-rl/modal-$RUN.json && { for i in 1 2 3; do git push -q && break; ./.venv/bin/python ./tools/ops/records.py pull >/dev/null 2>&1; done; }
