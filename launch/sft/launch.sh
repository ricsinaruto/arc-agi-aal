#!/bin/bash
# trunk/sft launch (one B300): CPU preflight beside the run, records only through records.py.
#   AUTH="<owner go>" BOUND=<$> MAXS=<s> bash launch/sft/launch.sh launch/sft/<run>.json
set -euo pipefail
CFG=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")  # absolute: the script cds to the main checkout and back
WT=$(cd "$(dirname "$0")/../.." && pwd)
ARC=.
PY=$ARC/.venv/bin/python
REC=$ARC/tools/ops/records.py
RUN=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['run'])" "$CFG")
WHAT=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['what'])" "$CFG")
LOG=~/arc-sft-data/mon/$RUN
mkdir -p "$LOG"
# per-run volumes: "volume": "run" in the config -> arc-run-<run> at /runs, inputs at /in/<vol>/ (vol.py)
VOLENV=$($PY "$WT/scripts/sft/vol.py" env "$CFG") || exit 1
export $VOLENV
VOLARG=(); [ "$ARC_SFT_VOLUME" = arc-prolong-runs ] || VOLARG=(--volume "$ARC_SFT_VOLUME")
[ -n "$ARC_SFT_INPUT_VOLUMES" ] || unset ARC_SFT_INPUT_VOLUMES
[ "$ARC_SFT_VOLUME" != arc-prolong-runs ] || unset ARC_SFT_VOLUME
NGPU=$($PY -c "import json,sys; s=json.load(open(sys.argv[1])); print(len((s.get('score') or {}).get('shards') or []) or int((s.get('score') or {}).get('cards', s.get('gpus', 1))))" "$CFG")  # scoring: one B300 per card; training: "gpus" (shard.py)
cd $ARC && $PY $REC pull > "$LOG/records.log" 2>&1
TRAINS=$($PY -c "import json,re,sys; t=json.load(open(sys.argv[1])).get('train') or {}; m=re.search(r'ckpt-(\d+)', t.get('init_ckpt') or ''); print(int(t.get('total_steps', 0) > (int(m.group(1)) if m else 0)))" "$CFG")
EXP=${SFT_EXPORT:-$ARC/tools/ops/export_sft.py}
if [ "$TRAINS" = 1 ]; then  # --help runs its imports: a broken exporter stops the launch, before any spend
  [ -f "$EXP" ] && $PY "$EXP" --help > /dev/null 2>"$LOG/export-check.log" \
    || { echo "EXPORTER BROKEN at $EXP: $(tail -1 "$LOG/export-check.log" 2>/dev/null)" >&2; exit 1; }
fi
$PY $REC reserve "$RUN" --arm sft --platform modal --gpu "$NGPU x B300" --gpu-count "$NGPU" --bound "$BOUND" --max-seconds "$MAXS" \
  --auth "$AUTH" --what "$WHAT" ${VOLARG[@]+"${VOLARG[@]}"} >> "$LOG/records.log" 2>&1 || { echo "REFUSED: $(tail -1 "$LOG/records.log")"; exit 1; }
RECVOL=$($PY $REC volume "$RUN" | tail -1)
[ "$RECVOL" = "${ARC_SFT_VOLUME:-arc-prolong-runs}" ] || { echo "VOLUME MISMATCH: record $RECVOL, config ${ARC_SFT_VOLUME:-arc-prolong-runs}" >&2; exit 1; }
$PY $REC push "$RUN" >> "$LOG/records.log" 2>&1
cd "$WT"
$ARC/.venv/bin/modal run --detach scripts/sft/modal_sft.py --config "$CFG" --pf > "$LOG/preflight.log" 2>&1 || echo "PREFLIGHT NOT STARTED: see $LOG/preflight.log" >&2  # spawned; poll .runops/$RUN/preflight.json
$ARC/.venv/bin/modal run --detach scripts/sft/modal_sft.py --config "$CFG" > "$LOG/launch.log" 2>&1
APP=$(grep -o "ap-[A-Za-z0-9]*" "$LOG/launch.log" | head -1 || true)
[ -n "$APP" ] || { echo "NO APP ID: see $LOG/launch.log" >&2; exit 1; }
cd $ARC && $PY $REC running "$RUN" --set app_id="$APP" >> "$LOG/records.log" 2>&1 \
  || echo "RECORD APP ID FAILED: $(tail -1 "$LOG/records.log")" >&2
# checklist 7: a training run (total_steps past its init step) gets watch.py + the dashboard exporter from launch, here,
# so neither can be skipped; the agent also waits on the watcher's alerts (watch.py alerts file / its own waiter).
if [ "$TRAINS" = 1 ]; then
  PLAN="$LOG/watch-plan.json"
  $PY - "$RUN" "$APP" "$WHAT" "$PLAN" "$CFG" <<'EOF'
import json, os, sys, time
from pathlib import Path
run, app, what, plan, cfg = sys.argv[1:6]
scratch = Path(os.environ.get("ARC_OPS_SCRATCH") or "<scratch>")
t = json.load(open(cfg))["train"]
# held-out reads hold the step (measured 2 Nov: slice + ARC-2 125-260 s, start reads ~800 s at 2-4 B300)
pauses = [p for p in ({"every": t.get("eval_steps") or 0, "s": 300}, {"every": t.get("arc2_heldout_every") or 0, "s": 480},
                      {"at": sorted(set(t.get("eval_full_steps") or []) | set(t.get("arc2_heldout_steps") or [])), "s": 1800})
          if p.get("every") or p.get("at")]
Path(plan).write_text(json.dumps({"run": run, "arm": "sft", "launched": int(time.time()), "boot_s": 1200, "cadence_s": 15,
    "platform": "modal", "app_id": app, "metrics_remote": f"modal:{os.environ.get('ARC_SFT_VOLUME') or 'arc-prolong-runs'}:{t['out'].removeprefix('/runs/')}/metrics.jsonl",
    "progress_key": "train.step", "progress_k": 4, "silence_s": 900, "end_marker": "\"event\": \"done\"",
    "expect": {"train.step": 1200}, "owner_agent": "trunk/sft agent", "progress_pauses": pauses}, indent=1))
meta = scratch / "train-metrics" / f"{run}.meta.json"
meta.parent.mkdir(parents=True, exist_ok=True)
cont = os.environ.get("SFT_CONTINUES", "")
meta.write_text(json.dumps({"run": run, "arm": "sft", "what": what[:300], "status": "running", "cards": int(json.load(open(cfg)).get("gpus", 1)),
                            **({"continues": cont.split(",")} if cont else {})}))
EOF
  nohup $PY -u $ARC/tools/ops/watch.py run "$PLAN" --no-exit-on-alert >> "$LOG/watch.log" 2>&1 &  # nohup: an exit on alert would wake no one
  nohup bash "$WT/launch/sft/keep_exporter.sh" "$PY" "$EXP" "$RUN" "$LOG" > /dev/null 2>&1 &
  echo "WATCH: plan $PLAN (watch.log, export.log in $LOG)" >&2
fi
echo "$APP"
