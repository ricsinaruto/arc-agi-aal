#!/bin/bash
# pf_fetch.sh RUN: a preflight whose client lost its connection (19 Oct) -> its result from arc-prolong-runs:/preflight/TAG.json
# (TAG from the log's "PREFLIGHT CALL id tag" line), appended to mon/preflight-RUN.log with the PREFLIGHT OK/FAILED verdict
L=~/arc-sft-data/mon/preflight-$1.log; M=./.venv/bin/modal
TAG=$(grep -o "PREFLIGHT CALL [^ ]* [^ ]*" $L | tail -1 | awk '{print $4}'); [ -n "$TAG" ] || { echo "no PREFLIGHT CALL line in $L"; exit 1; }
D=$(mktemp -d); for i in $(seq 720); do
  $M volume get --force arc-prolong-runs /preflight/$TAG.json $D/r.json >/dev/null 2>&1 && break; sleep 30; done
[ -s $D/r.json ] || { echo "no result for $TAG after 6 h" | tee -a $L; exit 1; }
cat $D/r.json >> $L
python3 -I -c "import json,sys; print('PREFLIGHT OK' if json.load(open(sys.argv[1]))['ok'] else 'PREFLIGHT FAILED')" $D/r.json | tee -a $L
