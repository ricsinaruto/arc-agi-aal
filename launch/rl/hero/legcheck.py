"""v6c CPU legs, first play minutes: groups, live tries (engine_admission), park check, Mode B blocks, reads, alerts."""
import json, subprocess, sys, time
M = "./.venv/bin/modal"
D = "<scratch>/rl-arm/hero"
RUNP = sys.argv[1]
for tag in sys.argv[2:]:
    f = f"{D}/{RUNP}{tag}-chk.jsonl"
    subprocess.run([M, "volume", "get", "--force", "arc-prolong-runs",
                    f"validation/preflight/{RUNP}{tag}/arena/trials/rl-loop/rl/feed.jsonl", f], capture_output=True)
    try:
        rows = [json.loads(l) for l in open(f) if l.strip()]
    except (OSError, ValueError):
        print(tag, "no feed yet"); continue
    k = lambda n: [r for r in rows if r.get("kind") == n]
    g = k("group_start")
    t0 = g[0]["epoch"] if g else None
    play = round((rows[-1]["epoch"] - t0) / 60, 1) if t0 else 0
    adm = k("engine_admission")
    pc = k("park_check")
    print(tag, f"play {play} min | groups {len(g)} done {len(k('group_done'))} | live(adm) {[a.get('live') for a in adm[-3:]]} | "
          f"park {[(p.get('identical'), p.get('error')) for p in pc]} | blocks {len(k('block_on'))} | steps {len(k('step'))} | "
          f"reads {[(r.get('step'), len(r.get('sets') or {}), len(r.get('extra_sets') or [])) for r in k('read')]} | "
          f"wreads {[(r.get('step'), r.get('score'), r.get('errors')) for r in k('wread')]} | chain {[r.get('start_source') for r in k('envs')][-1:]} | "
          f"alerts {[str(r.get('why'))[:80] for r in k('alert')][:3]} | errors {len(k('error'))}", flush=True)
