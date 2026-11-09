#!/bin/zsh
# fe-launch.sh RECORD FE_RUN ARMS [febench args...]: detached febench launch (env FE_TIMEOUT, ARC_RL1_BUNDLE,
# FE_BOOT_WT = a src/ snapshot of the branch to boot, from the caller), then records.py attach RECORD LOG so the app id always lands on the run record.
set -e
rec=$1 run=$2 arms=$3; shift 3
here=${0:A:h}
root=${ARC_ROOT:-.}
# GPU ceiling from measured phase times (sizing.py): unset FE_TIMEOUT takes it; a smaller one is refused
need=$(python3 $here/sizing.py $arms "$@")
if (( need > 0 )); then
  if [[ -z $FE_TIMEOUT ]]; then export FE_TIMEOUT=$need
  elif (( FE_TIMEOUT < need )); then echo "FE_TIMEOUT $FE_TIMEOUT < sized $need s for $arms (sizing.py): refused"; exit 1; fi
  echo "FE_TIMEOUT $FE_TIMEOUT (sized $need)"
fi
mkdir -p ${FE_LOG_DIR:=$here/status}; log=$FE_LOG_DIR/launch-$run-$(date +%s).log
(cd $here && $root/.venv/bin/modal run --detach febench.py --run $run --arms $arms "$@") > $log 2>&1 || { tail -20 $log; exit 1; }
grep -E "call_id" $log
python3 $root/tools/ops/records.py attach $rec $log
