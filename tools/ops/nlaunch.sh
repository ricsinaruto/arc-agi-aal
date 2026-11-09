#!/bin/zsh
# nlaunch.sh <modal_validation.py of the branch that built the bundle> <test kind> <bundle dir> <run id> [extra launcher args...]
# Launches one run on a Nebius fixed RTX PRO 6000 without waiting. Extra args: --reason "<why>" for 2-4 h of play, --approval "<explicit approval>" above 4 h.
# Add the run's row to docs/modal-queue.json before calling. The log goes to <scratch>/launch/<run id>.log.
ROOT=$(cd "$(dirname "$0")/../.." && pwd); S=${ARC_OPS_SCRATCH:-$ROOT/runs/ops}; mkdir -p "$S/launch" "$S/mon"
MV=$1; T=$2; B=$3; R=$4; shift 4
export ARC_VALIDATION_ROOT=$ROOT ARC_VALIDATION_KIND=arena ARC_NEBIUS_CONFIG=${ARC_NEBIUS_CONFIG:-$S/nebius.json}
NS=${ARC_NEBIUS_SCRIPTS:-$ROOT/.claude/worktrees/nebius/scripts}
# dfr-exact bundles: strict key / overlay check before anything is uploaded (19 Oct)
case "$MV" in *worktrees/dfr-exact/*) python3 "$(dirname "$MV")/dfr_exact/check_bundle.py" "$B" > /dev/null || { echo "$R exit=1 bundle check failed"; exit 1; } ;; esac
ARC_MODAL_VALIDATION=$MV ARC_VALIDATION_TEST=$T ARC_VALIDATION_BUNDLE=$B $ROOT/.venv/bin/python $NS/nebius_validation.py launch --run-id $R "$@" --no-wait > $S/launch/$R.log 2>&1
echo "$R exit=$? $(grep -o '"charged_bound_usd": [0-9.]*' $S/launch/$R.log | head -1) $(tail -c 90 $S/launch/$R.log | tr -d '\n')"
