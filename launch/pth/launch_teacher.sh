#!/bin/bash
# launch_teacher.sh: Stage A BF16 teacher-only pass on 2 B300 (chunk 8K, staggered loads) over /heal-data/stageA-v1 (R = ra-/rb-,
# D = da-/db-, h10ref = h- segments); files split into teacher-R/-D/-H afterwards (ops/split_teacher.py). Gate: the B300
# smoke passed with the same teacher settings (mon/smoke-ok-<smoke run>).
RUN=sft-heal-round-tA-${VER:-v1}; TO=${TO:-3300}
export ARC_RL1_BUNDLE=dist/sft-heal-round-v18 ARC_HEAL_TEACHER_DIR= ARC_HEAL_EXTRA_TEACHER= \
ARC_RL1_AUTHORIZATION="${AUTHZ:-(Stage A approved; BF16 teacher re-scores every teacher file): Stage A teacher pass, R + D + h10ref}" \
ARC_HEAL_SEGMENTS=${SEGMENTS:-/runs/heal-data/stageA-v1} ARC_HEAL_STUDENT=/runs/validation/sft-heal-round-cut1-v2/train/student \
ARC_HEAL_FROZEN=/assets/sft-heal/student-p75d4 ARC_HEAL_RESUME=/runs/validation/sft-heal-round-cut1-v2/train \
ARC_HEAL_STEP0=0 ARC_HEAL_TOTAL_STEPS=0 ARC_HEAL_GENERAL_WINDOWS=256 ARC_HEAL_TRAIN_SECONDS=4500 \
ARC_HEAL_BATCH_TOKENS=250000 ARC_HEAL_POSITIONS=reply ARC_HEAL_REVERSE=0 ARC_HEAL_FORWARD=1 ARC_HEAL_LR=2e-5 \
ARC_HEAL_MTP_WEIGHT=0.1 ARC_HEAL_ESFT=0.12 ARC_HEAL_SCHEDULE=cosine ARC_HEAL_WARMUP=0.1 ARC_HEAL_CHECKPOINT_SECONDS=1800 \
ARC_HEAL_TRUNCATE_LOOPS=1 ARC_HEAL_CARDS=${CARDS:-4} ARC_HEAL_TIMEOUT=$TO ARC_HEAL_TEACHER_TOKENS=0 ARC_HEAL_SMOKE= ARC_HEAL_TEACHER_ONLY=1 \
ARC_HEAL_TEACHER_MODEL=/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540 ARC_HEAL_TEACHER_CHUNK=4096 ARC_HEAL_STAGGER=1 ARC_HEAL_TEACHER_SPREAD=2
if [ "$PREFLIGHT" = 1 ]; then
  export RUN; cd ./.claude/worktrees/sft-heal && ./.venv/bin/modal run --detach scripts/posttrain/heal_preflight.py \
    --data /runs/validation/sft-heal-round-c1opd-v1/teacher --general /runs/validation/sft-heal-round-c1opd-v1/general > ~/arc-sft-data/mon/preflight-$RUN.log 2>&1
  grep -q "PREFLIGHT OK\|PREFLIGHT FAILED" ~/arc-sft-data/mon/preflight-$RUN.log || ./.claude/worktrees/sft-heal/launch/pth/pf_fetch.sh $RUN
  tail -1 ~/arc-sft-data/mon/preflight-$RUN.log; exit 0
fi
[ -f ~/arc-sft-data/mon/smoke-ok-teacher ] || { echo "no passing B300 smoke"; exit 1; }
export RUN AUTH="$ARC_RL1_AUTHORIZATION" BOUND=$(python3 -c "print(($TO + 60) * (${CARDS:-4} * 0.001972) * 1.15)") MAXS=$TO DUR=$TO GPUROW=B300:${CARDS:-4} \
WHAT="${WHAT:-Stage A teacher pass: the full-precision (BF16) Flash teacher scores every recorded play session for Stage A (R: 15 training games, D: 144 generated games, plus the held-out h10 reference), top-128 next-token probabilities per position. These are the training targets for the two heals. ~45 min.}" \
NOTE="${NOTE:-Stage A teacher-only: BF16 teacher (expert-bf16 de4b8e4d), chunk 8K, staggered loads, k 128, 258 segments (R 94, D 144, h10ref 20), ~17.5M window tokens upper bound}"
./.venv/bin/python ./tools/ops/records.py reserve $RUN --arm sft --platform modal --gpu "${CARDS:-4} x B300" --gpu-count ${CARDS:-4} --bound $BOUND \
  --max-seconds $MAXS --kind heal-round --auth "$AUTH" --what "$WHAT" --set note="\"$NOTE\"" || { echo "RESERVE REFUSED"; exit 1; }
./.venv/bin/python ./tools/ops/records.py push $RUN
cd ./cd .claude/worktrees/sft-heal
./.venv/bin/modal run scripts/posttrain/modal_rl1.py --run-id $RUN --dry-run > ~/arc-sft-data/mon/dry-$RUN.log 2>&1 || { echo "dry run failed: mon/dry-$RUN.log"; exit 1; }
./.venv/bin/modal run --detach scripts/posttrain/modal_rl1.py --run-id $RUN > ~/arc-sft-data/mon/launch-$RUN.log 2>&1
APP=$(grep -o 'ap-[A-Za-z0-9]*' ~/arc-sft-data/mon/launch-$RUN.log | head -1); echo "$RUN $APP $(grep -c Error ~/arc-sft-data/mon/launch-$RUN.log) errors"
echo "WATCH: MAX_SECONDS=$TO python3 ./.claude/worktrees/sft-heal/launch/pth/opd_watch.py $RUN $APP  (teacher phase: silence + over-plan rules)"
cd . && git add docs/validation-rl/modal-$RUN.json && git commit -q -m "records: $RUN

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- docs/validation-rl/modal-$RUN.json && { for i in 1 2 3; do git push -q && break; ./.claude/worktrees/sft-heal/launch/pth/gpull.sh; done; }
