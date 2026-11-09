#!/bin/bash
# pf_guard.sh RUN APP: a run launched beside its preflight; when the preflight ends FAILED, stop the app (verified) and alert
RUN=$1; APP=$2; L=~/arc-sft-data/mon/preflight-$RUN.log
until grep -q "PREFLIGHT OK\|PREFLIGHT FAILED" $L; do sleep 60; done
if grep -q "PREFLIGHT OK" $L; then echo "$(date -u +%H:%M) $RUN preflight OK"; exit 0; fi
cd . && .venv/bin/modal app stop -y $APP; sleep 20
echo "ALERT $(date -u +%H:%M) $RUN preflight FAILED -> stopped $APP: $(.venv/bin/modal app list --json | python3 -c "import sys,json;print([a['state'] for a in json.load(sys.stdin) if a['app_id']=='$APP'])")"
