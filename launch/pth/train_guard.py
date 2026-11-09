"""Live guard for an SFT training job: train_guard.py RUN APP_ID.
Prints every update and validation read; stops the app (and verifies it stopped) on a failed or non-finite update,
two loss spikes in a row (> 3x the median of the first three losses), no trainer 25 min after the feed starts, or
no feed event for 15 min once training has started. The validation early stop and the time stop run inside the job."""
import json, math, os, statistics, subprocess, sys, time
MV = "./.venv/bin/modal"
RUN, APP = sys.argv[1], sys.argv[2]
D = f"~/arc-sft-data/mon/{RUN}"
os.makedirs(D, exist_ok=True)


def get(remote, local):
    return subprocess.run([MV, "volume", "get", "--force", "arc-prolong-runs", remote, local], capture_output=True).returncode == 0


def app_state():
    out = subprocess.run([MV, "app", "list", "--json"], capture_output=True, text=True).stdout
    try:
        return next((a["state"] for a in json.loads(out) if a["app_id"] == APP), "gone")
    except Exception:
        return "unknown"


def stop(reason):
    print(f"GUARD STOP: {reason}", flush=True)
    subprocess.run([MV, "app", "stop", "-y", APP], capture_output=True)
    for _ in range(20):
        s = app_state()
        if s in ("stopped", "gone"):
            print(f"GUARD: app {APP} {s} (verified)", flush=True)
            return
        time.sleep(15)
    print(f"GUARD: app {APP} still {app_state()} after stop", flush=True)


seen, losses, spikes, began = 0, [], 0, time.time()
first_event = last_event = None
trainer_up = False
while True:
    if get(f"/validation/{RUN}/arena/trials/rl-loop/rl/feed.jsonl", f"{D}/feed.jsonl"):
        lines = open(f"{D}/feed.jsonl").read().splitlines()
        for line in lines[seen:]:
            e = json.loads(line)
            k = e.get("kind")
            last_event = time.time()
            first_event = first_event or last_event
            if k == "trainer":
                trainer_up = True
                print("trainer up", flush=True)
            elif k == "sft_update":
                loss = e.get("loss")
                print(f"{e['label']} lr {e.get('lr', 0):.2e} loss {loss} s {e.get('train_seconds')} util {e.get('gpu_util_mean')} "
                      f"peak {e.get('card_peak_gib')} val {e.get('val_nll')} fit {e.get('fit_nll')} kl {e.get('heldout_kl')}", flush=True)
                if not e.get("ok") or loss is None or not math.isfinite(float(loss)):
                    stop(f"update failed or non-finite at {e['label']}: {e.get('error') or e.get('stop')}")
                    sys.exit(0)
                losses.append(float(loss))
                if len(losses) > 3:
                    spikes = spikes + 1 if float(loss) > 3 * statistics.median(losses[:3]) else 0
                    if spikes >= 2:
                        stop(f"two loss spikes in a row at {e['label']} ({loss})")
                        sys.exit(0)
            elif k in ("sft_val", "sft_t3", "fit", "gate", "error", "stop", "sft_resume", "sft_probe_base"):
                print(k, json.dumps({x: y for x, y in e.items() if x not in ("t", "epoch", "kind", "loaded", "train")})[:500], flush=True)
        seen = len(lines)
    if get(f"/validation/{RUN}/modal-identity.json", f"{D}/id.json"):
        i = json.load(open(f"{D}/id.json"))
        if i.get("exit_code") is not None:
            print(f"END exit {i['exit_code']} elapsed {i.get('elapsed_seconds')}; app {app_state()}", flush=True)
            break
    now = time.time()
    if first_event and not trainer_up and now - first_event > 1500:
        stop("no trainer 25 min after the feed started")
        break
    if trainer_up and last_event and now - last_event > 900:
        stop("no feed event for 15 min")
        break
    time.sleep(60)
