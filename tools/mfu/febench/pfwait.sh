#!/bin/zsh
# emits phase changes of a febench run status until it ends (phase done/error) or the app stops
run=$1; app=$2; last=""
D=${FE_STATUS_DIR:-${0:A:h}/status}/st-$run
mkdir -p $D
while true; do
  ./.venv/bin/modal volume get --force arc-prolong-runs /validation/mfu2-fe/$run/status.json $D/status.json >/dev/null 2>&1
  ph=$(python3 -c "import json;d=json.load(open('$D/status.json'));print(d.get('phase'), d.get('error','')[:200] if isinstance(d.get('error'),str) else '')" 2>/dev/null)
  if [ "$ph" != "$last" ]; then echo "$(date +%H:%M:%S) $ph"; last=$ph; fi
  case "$ph" in done*|*rror*|finished*) exit 0;; esac
  st=$(./.venv/bin/modal app list 2>/dev/null | grep $app | grep -c stopped)
  if [ "$st" = "1" ]; then echo "$(date +%H:%M:%S) app stopped, last phase $ph"; exit 0; fi
  sleep 30
done
