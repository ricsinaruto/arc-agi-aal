#!/bin/bash
# c7k: QSA ties=index in teacher + trainer, student-chain verify (KL bar VKL), step held-out curve, live play list, keep 2 ckpts.
# NEW_FLAGS="" drops the four c7k flags; PFTAG_SET names a separate preflight; READY_SET: plays to wait on
# (default: PLAY_SET's version-0 backlogs). Usage: DSTART=1 HBUNDLE_SET=dist/sft-heal-round-vNN VKL=<2 x floor> ACK="..." launch_c7k.sh (then trainer_knobs.py for the 2nd play)
S=<scratch>
# DSTART=1 (owner restart rule): from D at step 0, start reads recomputed under the new ties; else c7g ckpt-$CK
if [ "${DSTART:-}" = 1 ]; then RESARG=""; STEP0=0; SR=""; CK=0
else CK=${CK:?checkpoint step}; RESARG=/runs/validation/${RES_RUN:-sft-heal-round-opdS-c7g}/train/ckpt-$CK; STEP0=$CK
  SR="HEAL_START_READS=/runs/heal-data/start-reads/opdS-D-r3.json,"; fi
cd ~/arc-sft-data
ARC_RESTART_ACK="${ACK:?restart cause}" \
ARC_RL1_AUTHORIZATION="c7j fix-and-relaunch (QSA tie fix) = ${RES_RUN:-c7g} ckpt-$CK on v82, fwd KD 0.5, lr 7e-6" \
NAME=opdS-c7k DELTA_RUN=sft-heal-round-opdS-c7e PLAY=${PLAY_SET:-b2-sft4x-opdS-g1-v30,b2-sft4x-opdS-g1-v8,b2-sft4x-opdS-g1-v9,b2-sft4x-opdS-g1-v10,b2-sft4x-opdS-g1-v14} \
READY_PLAYS=${READY_SET-} PB=${PB_SET:-sft4x-opdS-g1-v23} SKIP_PLAY=1 PFTAG=${PFTAG_SET:-} \
BATCH=250000 HLR=7e-6 HFWD=0.5 HBUNDLE=${HBUNDLE_SET:?bundle} \
TENV="RL1_EXPERTS_FP8=1,RL1_STREAM_CHAINS=1,ARC_TEACHER_CACHE=1,HEAL_STREAM_NEWEST=1,RL1_QSA_TIES=index,HEAL_VERIFY_KL=${VKL:?verify KL bar (2 x the v7 floor)}" HCARDS=4 HTRANKS=3 \
HRES=$RESARG HSTEP0=$STEP0 BAR_STEP=300 HWIN=${HWIN_SET:-65536} MIDR=16 \
CKAT=$(cat ./.claude/worktrees/sft-heal/launch/pth/ckat6.txt) ARC_HEAL_STREAM_CHUNK=${STREAM_CHUNK:-16384} \
EXTRA_E="${SR}HEAL_HELDOUT2_BEH=/runs/heal-data/start-reads/opdS-D-r3-heldout2-beh.pt,HEAL_LOAD_PREFETCH=1,HEAL_MID_GENERAL=1,HEAL_ASYNC_CKPT=1,HEAL_STREAM_BALANCE=1,HEAL_STREAM_PREFETCH=1,HEAL_COUNT_COLLECTIVES=1,HEAL_HOST_EVERY=30,HEAL_HYPER_FAST=1,HEAL_DELTA_FAST=1,HEAL_SKIP_INERT_LOSS=1,HEAL_KD_ROWS=1,HEAL_FLA_CONV=1,HEAL_MID_INT4=1,HEAL_STREAM_STEPPLAN=1,HEAL_HOST_TIMERS=1,HEAL_MID_READ_AT_RESUME=1,HEAL_QSA_TIES=index${NEW_FLAGS-,HEAL_KEEP_CKPT=2,HEAL_STEP_HELDOUT=16384,HEAL_MID_TIES_AB=1,HEAL_FRESH_HOLDOUT=0.02}" \
HWHAT="OPD long run c7k (c7j fix-and-relaunch: c7j stopped on a teacher verify failure; cause: bf16 call-shape noise of the teacher, not a cache bug (stream checks v4-v7); QSA ties go to the lowest block index): from D (step 0; owner restart rule) or c7g ckpt-$CK; teacher verify bar = KL <= 2 x the measured bf16 call-shape floor (v7). Per-step held-out KD curve (16K dh10 slice per rank). Forward KD weight 0.5 (was 0.25: at c7h step 16 teacher-H forward KL was +22% vs D while sampled reverse fell 23%), lr 7e-6, 250K trained positions, 64K window, 16K teacher pieces, step plan (rank lockstep), newest-first teacher, checkpoints every 6. Live play v30 (all 15 training games) after version-0 backlogs v8-v14. Mid reads at resume and every 16 steps, incl. the served-int4 variant. Judged: teacher-H and dh10 forward KL vs D step 0 (stop: two consecutive mid reads above step 0 + 5%), median data age <= 4, grad norm, step time vs ~120 s planned, peak memory <= 245 GiB." \
  bash ./.claude/worktrees/sft-heal/launch/pth/opd_stream_c.sh
