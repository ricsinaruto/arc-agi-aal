#!/bin/bash
# v9 FP8 dense calibration (docs/v9-submission.md #4): v8.7b's recipe (fp8_calib.fit per key, 49 layers x in/out) on
# ckpt-268's own weights and its all-25 dense capture
cd ./.claude/worktrees/dfr-exact || exit 1
export ARC_FP8_CAPTURE_VOLUME=${CAPVOL:-arc-run-tp-v9-refit-c268-v1}
./.venv/bin/modal run scripts/dfr_exact/fp8/modal_fp8_calib.py --capture-run b2-tp-cap-v85p-p75c268-all25-v1 \
  --out-id arc-fp8-calib-p75c268-all25-v1 --weights /runs/validation/sft-p75-cur20-c231-v1/train/ckpt-268/student \
  --summary-path <scratch>/v9/fp8-summary.json ${KEYS:+--keys $KEYS}
