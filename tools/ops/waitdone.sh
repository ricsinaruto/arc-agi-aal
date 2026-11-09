#!/bin/bash
# waitdone.sh <run id> <local dir under <scratch>/mon> <max minutes>: wait for a Modal post-training job's posttrain/<run>/summary.json
# (written when the job ends; result.json appears early and partial), then copy summary and result from volume arc-prolong-runs
ROOT=$(cd "$(dirname "$0")/../.." && pwd); S=${ARC_OPS_SCRATCH:-$ROOT/runs/ops}; mkdir -p "$S/launch" "$S/mon"
D=$S/mon/$2; mkdir -p $D; n=0; rm -f $D/$1-summary.json
until (cd $ROOT && uv run -q --extra modal modal volume get arc-prolong-runs /posttrain/$1/summary.json $D/$1-summary.json --force >/dev/null 2>&1) || [ $n -ge $3 ]; do sleep 60; n=$((n+1)); done
(cd $ROOT && uv run -q --extra modal modal volume get arc-prolong-runs /posttrain/$1/result.json $D/$1.json --force >/dev/null 2>&1)
if [ -f $D/$1-summary.json ]; then echo "$(date -u +%H:%M) $1 done after $n min: $(head -c 500 $D/$1-summary.json | tr '\n' ' ')"; else echo "$(date -u +%H:%M) $1 NOT done after $n min"; exit 1; fi
