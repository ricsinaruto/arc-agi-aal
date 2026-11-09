"""Live reader of a running Kaggle kernel's stdout (kernels_logs_stream; no secrets in the kernel).

Run under scripts/kaggle_cli.py --python. OUT/<slug>.log gets every stdout line, OUT/<slug>.lines every progress line;
OUT/<slug>.events gets alerts, STOP lines (the run's stop rules, the agent decides) and a summary every 30 min, for a waking task
(a file: kaggle_cli.py holds a child's stdout until it exits). --replay FILE backtests the rules on a saved log."""
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

SILENT = 300  # s with no stdout: the stream is taken as hung
NUM = dict(tokens=r"tokens ([\d.]+)M", tps=r"tok/s (\d+) \(5 min", both=r"both ([\d.]+) recompute", gpu=r"hit GPU ([\d.]+)",
           cpu=r"CPU-tier ([\d.]+)", ram=r"host RAM (\d+)/", kvmax=r"KV 5min [\d.]+ max ([\d.]+)", kv=r"\| KV ([\d.]+) \|",
           restarts=r"engine restarts (\d+)", alloc=r"alloc fails (\d+)", preempt=r"preemptions ([\d.]+)", running=r"running ([\d.]+) waiting",
           reserved=r"reserved ([\d.]+)/", cap=r"reserved [\d.]+/([\d.]+)", right=r"right (\d+)/", mtp=r"MTP accept ([\d.]+)", task=r"blocks/task now ([\d.]+)", peak=r"peak mean ([\d.]+)", ptask=r"peak task (\d+)")


def parse(line):
    head = re.match(r"\[(\d+):(\d+) elapsed\] tokens", line)
    if not head:
        return None
    row = {"minute": int(head[1]) * 60 + int(head[2])}
    for key, pattern in NUM.items():
        found = re.search(pattern, line)
        if found:
            row[key] = float(found[1])
    return row


class Rules:
    """Thresholds from v52c (death: KV .95-.99, preemptions, tokens frozen, hit below .90) and healthy v52 / v55bm2 logs."""

    def __init__(self):
        self.rows, self.fired = [], {}

    def fire(self, key, text, every=10):
        minute = self.rows[-1]["minute"]
        if minute - self.fired.get(key, -999) >= every:
            self.fired[key] = minute
            return [f"ALERT [{minute // 60}:{minute % 60:02d}] {key}: {text}"]
        return []

    def step(self, row):
        self.rows.append(row)
        out, last = [], self.rows[-11:]
        if row["minute"] < 20:  # boot and ramp
            return out
        old = last[0]
        if len(last) == 11 and row.get("tokens") == old.get("tokens") and (row.get("running") or 0) >= 1 and (row.get("tps") or 0) < 50:
            out += self.fire("stalled", f"tokens flat at {row.get('tokens')}M for 10 min, running {row.get('running')}, engine tok/s {row.get('tps')}")
        if row.get("restarts", 0) > old.get("restarts", 0):
            out += self.fire("engine_restart", f"restarts {row['restarts']:.0f}")
        if row.get("alloc", 0) > old.get("alloc", 0):
            out += self.fire("alloc_fail", f"ARC_CACHE_ALLOCATION_FAILURE total {row['alloc']:.0f}")
        if row.get("preempt", 0) > old.get("preempt", 0):
            out += self.fire("preemptions", f"{old.get('preempt', 0):.0f} -> {row['preempt']:.0f} in 10 min, KV {row.get('kv')}")
        if max(row.get("kvmax", 0), row.get("kv", 0)) >= .95:
            out += self.fire("kv_high", f"KV >= .95, now {row.get('kv')} 5-min max {row.get('kvmax')}, gate {row.get('reserved')}/{row.get('cap')}")
        if len(last) == 11 and sum(r.get("both", 1) for r in last) / 11 < .90:
            out += self.fire("hit_low", f"hit both mean < .90 over 10 min, now GPU {row.get('gpu')} CPU {row.get('cpu')} both {row.get('both')}", 30)
        if len(last) == 11 and all(r.get("tps", 999) < 300 for r in last) and row.get("restarts", 0) == old.get("restarts", 0):
            out += self.fire("slow", f"engine tok/s < 300 for 10 min, now {row.get('tps')}", 30)
        if row.get("ram", 0) >= 170:
            out += self.fire("host_ram", f"{row['ram']:.0f} GiB used of 189")
        return out + self.stops(row)

    def stops(self, row):
        out, hour = [], self.rows[-31:]
        if row.get("restarts", 0) >= 2:
            out += self.fire("STOP second_engine_death", f"restarts {row['restarts']:.0f}", 30)
        if row.get("ram", 0) >= 170:
            out += self.fire("STOP host_ram", f"{row['ram']:.0f} GiB", 30)
        if len(hour) == 31 and all(r.get("tps", 999) < 50 for r in hour) and row.get("restarts", 0) > hour[0].get("restarts", 0) - 1 and row.get("restarts", 0):
            out += self.fire("STOP restart_over_30_min", f"engine tok/s < 50 for 30 min, restarts {row['restarts']:.0f}", 30)
        slow = self.rows[-21:]
        if len(slow) == 21 and all(r.get("tps", 999) < 300 for r in slow) and row.get("restarts", 0) == slow[0].get("restarts", 0):
            out += self.fire("STOP slow", f"engine tok/s < 300 for 20 min outside a restart, now {row.get('tps')}", 30)
        return out


def summary(row):
    keys = ("tokens", "tps", "running", "kv", "kvmax", "reserved", "cap", "gpu", "cpu", "both", "mtp", "preempt", "ram", "right", "restarts", "alloc", "task", "peak", "ptask")
    return f"[{row['minute'] // 60}:{row['minute'] % 60:02d}] " + " ".join(f"{k} {row[k]:g}" for k in keys if k in row)


def replay(path):
    rules = Rules()
    seen = set()
    for raw in Path(path).read_text(errors="replace").splitlines():
        raw = raw.strip().lstrip("[,").rstrip("]")
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        row = parse(str(event.get("data", ""))) if isinstance(event, dict) else None
        if row and row["minute"] not in seen:
            seen.add(row["minute"])
            for alert in rules.step(row):
                print(alert)
    print("rows", len(rules.rows), summary(rules.rows[-1]) if rules.rows else "")


def follow(slug, out):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    name = slug.split("/")[-1]
    rules, last, silent = Rules(), -1.0, time.time()
    path = out / f"{name}.events"
    said = set(path.read_text().splitlines()) if path.exists() else set()

    def say(text):  # a reconnect replays the feed: each line once
        if text not in said:
            said.add(text)
            with path.open("a") as events:
                events.write(text + "\n")

    def status():
        try:
            return str(getattr(api.kernels_status(slug), "status", ""))
        except Exception as exc:  # noqa: BLE001 - a failed status call is no end
            print(f"status {type(exc).__name__}: {str(exc)[:120]}", file=sys.stderr, flush=True)
            return ""

    def watchdog():  # the stream can hang without an exception: the process ends and liveread.sh starts it again
        while True:
            time.sleep(30)
            state = status()
            if any(word in state.upper() for word in ("COMPLETE", "ERROR", "CANCEL")):
                say(f"KERNEL ENDED {state[:160]}")
                os._exit(0)
            if time.time() - silent > SILENT:
                if rules.rows and rules.rows[-1]["minute"] >= 15:
                    say(f"ALERT feed silent {SILENT} s at elapsed {rules.rows[-1]['minute']} min, kernel {state[:100]}: reconnecting")
                os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()
    with (out / f"{name}.log").open("w") as log, (out / f"{name}.lines").open("w") as lines:  # the feed replays from the start
        while True:
            try:
                for event in api.kernels_logs_stream(slug):
                    when = float(event.get("time") or 0)
                    if when <= last or event.get("stream_name") != "stdout":
                        continue
                    last, silent, text = when, time.time(), str(event.get("data", "")).rstrip("\n")
                    log.write(text + "\n")
                    log.flush()
                    row = parse(text)
                    if row:
                        lines.write(text + "\n")
                        lines.flush()
                        for alert in rules.step(row):
                            say(alert)
                        if row["minute"] % 30 == 0:
                            say(summary(row))
            except Exception as exc:  # noqa: BLE001 - a dropped feed reconnects
                print(f"feed {type(exc).__name__}: {str(exc)[:120]}", file=sys.stderr, flush=True)
            time.sleep(30)


if __name__ == "__main__":
    if sys.argv[1] == "--replay":
        replay(sys.argv[2])
    else:
        follow(sys.argv[1], sys.argv[2])
