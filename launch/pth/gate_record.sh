#!/bin/bash
# gate_record.sh RUN DIGEST: waits for RUN's real-trainer gate verdict (smoke.json on the runs volume) and records the
# bundle's GPU-path digest in mon/gate-passed-gpu.txt when it passed (launch_heal.sh GATE=auto reads the list)
RUN=$1; DIG=$2; MV=./.venv/bin/modal; T=$(mktemp -d)
for i in $(seq 360); do  # up to 6 h
  if $MV volume get --force arc-prolong-runs /validation/$RUN/smoke.json $T/smoke.json >/dev/null 2>&1; then
    if python3 -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') else 1)" $T/smoke.json; then
      echo "$DIG $RUN $(date -u +%FT%H:%MZ)" >> ~/arc-sft-data/mon/gate-passed-gpu.txt; echo "gate passed: $DIG recorded"
    else echo "gate failed: $DIG not recorded"; fi
    rm -rf $T; exit 0
  fi
  sleep 60
done
echo "no gate verdict in 6 h"; rm -rf $T
