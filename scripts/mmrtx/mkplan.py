"""mkplan.py ref|stagea|kl RUN APP_ID LAUNCH_EPOCH -> plan-RUN.json for tools/ops/watch.py run"""
import json, sys
kind, run, app, launched = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
if kind == "ref":
    plan = {"run": run, "arm": "mmrtx", "launched": launched, "boot_s": 900, "cadence_s": 30, "platform": "modal",
            "app_id": app, "metrics_remote": f"modal:mmrtx-models:ref/{run}/metrics.jsonl",
            "targets": {"infra.vram_gib.*": 140, "infra.gpu_util.*": 40}, "play_key": "ref.progress",
            "progress_key": "ref.progress", "counters": {}, "end_marker": "RUN COMPLETE"}
elif kind == "kl":
    plan = {"run": run, "arm": "mmrtx", "launched": launched, "boot_s": 900, "cadence_s": 30, "platform": "modal",
            "app_id": app, "metrics_remote": f"modal:mmrtx-models:kl/{run}/metrics.jsonl",
            "targets": {"infra.vram_gib.*": 80, "infra.gpu_util.*": 40}, "play_key": "kl.progress",
            "progress_key": "kl.progress", "counters": {}, "end_marker": "RUN COMPLETE"}
else:
    plan = {"run": run, "arm": "mmrtx", "launched": launched, "boot_s": 900, "cadence_s": 30, "platform": "modal",
            "app_id": app, "metrics_remote": f"modal:mmrtx-models:stagea/{run}/metrics.jsonl",
            "targets": {"infra.vram_gib.*": 80, "infra.gpu_util.*": 40}, "play_key": "stagea.progress",
            "progress_key": "stagea.progress", "counters": {}, "end_marker": "RUN COMPLETE"}
json.dump(plan, open(f"plan-{run}.json", "w"), indent=1)
print(f"plan-{run}.json")
