#!/bin/zsh
# launch_p75.sh ARM: base12 on c120, p75 checkpoint in its trained format. Default nebius runner: it
# mirrors arena/live/control.json, so live knobs work (nebius-opd has no mirror)
ARM=$1; V=${2:-1}
R=arc2-rtx-c120-hx-base12-p75$ARM-cpu32-v9b-p$V-nebius-$(date -u +%Y%m%d)
B=./.claude/worktrees/arc2/dist/arc2-rtx-c120-hx-base12-p75$ARM-cpu32-v9b
S=${ARC2_SCRATCH:-./.claude/worktrees/arc2/dist/ops}; mkdir -p $S/hx
WHAT="ARC-AGI-2 c120 read to 10M tokens: harness BASE12 unchanged, with the last p75 checkpoint ckpt-1047 in its trained format (48 layers, 128 experts, top-k 10, INT4 experts, INT8 dense, int4 KV, MTP 2), 83 streams, 32 GiB tier. First p75 play on ARC-2. Judged against base12c 138, the 100 and the 20 tasks reported separately."
REASON="base12 to 10M on c120 with p75 ckpt-1047; c120 reads end at 10M, expected ~2 h 15 with boot"
APPROVAL='"run base12 to 10M tokens on c120, 1 RTX, with the weights of the last p75 checkpoint"; "launch without waiting for me once gates pass"'
cd .
python3 .claude/worktrees/arc2/scripts/arc2_weights/check_rtx_stack.py $B --v9b || exit 1
ARC_NEBIUS_SCRIPTS=./.claude/worktrees/nebius/scripts python3 .claude/worktrees/sft-heal/launch/pth/check_play_bundle.py $B | tail -2; [[ ${pipestatus[1]} == 0 ]] || exit 1
ARC_ARM=arc2 ARC_WHAT="$WHAT" ARC_OPS_SCRATCH=./.claude/worktrees/arc2/dist/ops ARC_NEBIUS_CONFIG=./.claude/worktrees/arc2/dist/ops/state/nebius-372.json perl -e 'alarm 1500; exec @ARGV' tools/ops/nlaunch.sh .claude/worktrees/arc2/scripts/modal_validation.py arc2-run $B $R --reason "$REASON" --approval "$APPROVAL" > $S/hx/launch-p75$ARM.log 2>&1
tail -3 $S/hx/launch-p75$ARM.log | cut -c1-300; python3 tools/ops/records.py push $R 2>&1 | tail -1
date -u
