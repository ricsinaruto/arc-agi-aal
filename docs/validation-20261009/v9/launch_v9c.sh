#!/bin/bash
# V9C 24 h data runs (24 h runs exempt from the RTX cap). Usage: launch_v9c.sh RUN
cd .
RUN=$1; B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=arc3sub
export ARC_ALLOWANCE_EXEMPT="24 h long runs are exempt from the RTX cap"
export REASON="two 24 h V9C data runs (c893 INT3 dn4g8 refit on the 15, INT8 dense, own MTP, GDN + QSA, drain jitter, no CPU tier, density streams) on disjoint rosters; decision: a level at 300K or a game at 500K generated tokens ends the game; v2 of v1 (stopped in assets, before play, cap not live-settable); per-request records (arena.sft_record)"
export APPROVAL="launch once the checklist passes; length approved; 24 h runs exempt from the RTX cap"
export V87B_STACK=0 V87B_WHY="v9 stack by owner spec: window 151,040, no CPU KV, density 47.9, target 60, MTP 2; KV gate, prefix cache and block budget code as v87b"
case $RUN in *tr15x10*) R="the 15 training games x 10 clones = 150 entries";; *) R="the 10 held-out games x 11 clones = 110 entries";; esac
WHAT="V9C data collection for SFT: $R, up to 24 h, the V9 stack with c893 INT3 dn4g8 refit weights and INT8 dense. A level at 300K or a game at 500K generated tokens ends that game; the run ends when every entry has ended or at the envelope. Every model request is recorded as served (arena.sft_record). Why: on-policy data for scoring. Judged by KV ~.8 and prefix hit >= .95 at steady state, 0 preemptions, and complete records under games/*/sft."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6 | cut -c1-260
date -u; date +%s
