#!/bin/bash
# wait_copy.sh "<run id> <dest folder under the scratch dir>" ... : wait for each Nebius run's execution.json (the end-of-run marker), then copy its results
ROOT=$(cd "$(dirname "$0")/../.." && pwd); [ -z "$ARC_OPS_SCRATCH" ] && [ -f "$ROOT/runs/ops/ops.env" ] && . "$ROOT/runs/ops/ops.env"; S=${ARC_OPS_SCRATCH:-$ROOT/runs/ops}; mkdir -p "$S/launch" "$S/mon"
R=${ARC_NEBIUS_RUNS:-nebius-arc:arc-runs-bucket/runs/validation}
for pair in "$@"; do r=${pair% *}; d=${pair#* }; n=0
  until rclone lsf $R/$r/ 2>/dev/null | grep -q "execution.json" || [ $n -ge 400 ]; do sleep 60; n=$((n+1)); done
  free=$(df -g "$S" | awk 'NR==2{print $4}')
  if [ "${free:-0}" -lt 100 ]; then echo "$r: not copied, only ${free} GB free (bucket keeps it: $R/$r)"; continue; fi
  # results only: per-game logs and big server logs stay in the bucket (one run is 2-10 GB in full; 12 Oct the disk filled)
  rclone copy $R/$r $S/$d --exclude "**/games/**" --exclude "**/trials/*/attempts/**" --max-size 200M 2>/dev/null; echo "$r -> $d n=$n $(date -u +%H:%M) $(grep -o '"exit_code": [0-9-]*' $S/$d/execution.json 2>/dev/null | head -1)"
done
