#!/bin/bash
# launch_stream_check.sh RUN BUNDLE: the one-card teacher continuation check (modal_rl1.stream_check, c7j diagnosis).
# Env: SC_FILES (volume .pt paths), SC_EXTRA (JSON: ref_single, subs, offsets, indexer, attention, state_check, kernels),
# SC_NEAR, SC_LIMIT, SC_PIN, SC_TIES. Reserve the run record first (records.py reserve). SC_EXTRA options must be
# within the gate test's default set (test_stream_check_entry_on_cpu), or widen that set first.
RUN=${1:?run}; B=${2:?bundle}
V=/runs/validation
cd ./.claude/worktrees/sft-heal
# rule 4 gate (<=5 min, Modal CPU): the Modal function's own body on the tiny teacher through to its first op output
./.venv/bin/python ./tools/ops/test_modal.py --tree . --label sc-gate-$RUN -n 0 \
  -- tests/test_stream_pipe.py -k stream_check_entry > ~/arc-sft-data/mon/gate-$RUN.log 2>&1 \
  || { echo "GATE FAILED: ~/arc-sft-data/mon/gate-$RUN.log"; exit 1; }
ARC_RL1_BUNDLE=$B ARC_VALIDATION_ROOT=. \
ARC_SC_FILES=${SC_FILES:-$V/sft-heal-round-tA-dh10/teacher/k020--e100.pt,$V/sft-heal-round-tA-v1/teacher-H/h-k019--e016.pt} \
ARC_HEAL_TEACHER_MODEL=/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540 \
ARC_SC_NEAR=${SC_NEAR:-16384} ARC_SC_LIMIT=${SC_LIMIT:-24000} ARC_SC_PIN=${SC_PIN:-1} ARC_SC_TIES=${SC_TIES:-topk} \
ARC_SC_EXTRA="${SC_EXTRA:-{\}}" \
  ./.venv/bin/modal run --detach scripts/posttrain/modal_rl1.py --run-id $RUN \
  > ~/arc-sft-data/mon/launch-$RUN.log 2>&1
