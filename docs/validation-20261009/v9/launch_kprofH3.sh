#!/bin/bash
# kprofH3: TURNS44 worker stacks by one-shot py-spy dumps; ordered item, fix and rerun of kprofH2
cd .
RUN=b2-kprofH3-v9-v1
B=./.claude/worktrees/v84sub/dist/$RUN
export ARC_ARM=arc3sub
export REASON="kprofH2 lost its TURNS44 point (py-spy record hang, TimeoutError in the SIGINT wait); fix the class and rerun once to split the ~16 ms per step left unexplained in play"
export APPROVAL="fix-and-relaunch passing the checklist; 1 RTX"
export V87B_STACK=0 V87B_WHY="serving bench, no play: v9 tree host profile, TURNS44 point only"
WHAT="kprofH3: profiles where the v9 server's time goes on real-size turns: 44 streams of play-like requests, sampling the worker's Python stacks with one-shot py-spy dumps at 2 Hz for 240 s. Why: the worker fills 54.7 of 57.5 ms per step and ~16 ms is unexplained. Judged by ms per step by stage and which stages a fix can remove."
bash <scratch>/limits/launch_rtx_appr.sh $RUN $B "$WHAT" 2>&1 | tail -6 | cut -c1-300
date -u; date +%s
