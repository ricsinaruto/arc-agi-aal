"""play_read.py RUN [K]: closed-loop measures over a Nebius arena run's games (one log at a time, deleted after reading):
gameplay replies with a tool call / all, games with >= 1 action, actions, levels; and the same over each game's first K
gameplay replies (default 20; games with >= K replies), so runs at different speeds compare at matched replies; plus
cap share (finish_reason length) and loop share (>= 50% repeated 8-grams in the last 3,000 chars of reasoning + text)."""
import json, os, subprocess, sys, tempfile
run = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
base = f"nebius-arc:arc-runs-bucket/runs/validation/{run}/arena/working/trials/duck-transfer/games/"
games = subprocess.run(["rclone", "lsf", base], capture_output=True, text=True).stdout.split()
per = []
with tempfile.TemporaryDirectory() as tmp:
    for g in games:
        path = os.path.join(tmp, "log.jsonl")
        if subprocess.run(["rclone", "copyto", base + g + "log.jsonl", path], capture_output=True).returncode:
            continue
        n = c = act = lv = cap = loop = 0; nk = ck = actk = None
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") == "model" and r.get("purpose") == "gameplay":
                n += 1; c += bool((r.get("message") or {}).get("tool_calls"))
                cap += r.get("finish_reason") == "length"
                m = r.get("message") or {}
                w = (str(m.get("reasoning") or "") + " " + str(m.get("content") or ""))[-3000:].split()
                grams = [" ".join(w[i:i + 8]) for i in range(max(0, len(w) - 8))]
                loop += len(grams) >= 50 and 1 - len(set(grams)) / len(grams) >= 0.5
                if n == K:
                    nk, ck, actk = n, c, act
            elif r.get("kind") == "observation":
                if int(r.get("step") or 0) >= 1:
                    act += 1
                lv = max(lv, int(r.get("levels") or 0))
        os.remove(path)
        per.append({"g": g.strip("/"), "n": n, "c": c, "act": act, "lv": lv, "nk": nk, "ck": ck, "actk": actk, "cap": cap,
                    "loop": loop})
tot = sum(p["n"] for p in per); k = [p for p in per if p["nk"]]
out = {"run": run, "games": len(per), "replies": tot,
       "tool_call_share": round(sum(p["c"] for p in per) / tot, 4) if tot else None,
       "games_with_actions": sum(p["act"] > 0 for p in per),
       "games_with_actions_share": round(sum(p["act"] > 0 for p in per) / max(len(per), 1), 4),
       "actions_total": sum(p["act"] for p in per),
       "actions_per_100_replies": round(100 * sum(p["act"] for p in per) / tot, 2) if tot else None,
       "levels_total": sum(p["lv"] for p in per),
       "cap_share": round(sum(p["cap"] for p in per) / tot, 4) if tot else None,
       "loop_share": round(sum(p["loop"] for p in per) / tot, 4) if tot else None,
       f"first{K}": {"games": len(k), "tool_call_share": round(sum(p["ck"] for p in k) / max(sum(p["nk"] for p in k), 1), 4),
                     "games_acting_share": round(sum(p["actk"] > 0 for p in k) / max(len(k), 1), 4),
                     "actions_per_game": round(sum(p["actk"] for p in k) / max(len(k), 1), 2)}}
print(json.dumps(out))
