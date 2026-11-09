"""50 real v9-harness gameplay requests at ~100K (SFT a2 records), up to 2 distinct turns per game, nearest 100K first.
Usage: mkctx50.py TURNS_DIR OUT"""
import glob, json, sys
from arena.sft_record import rebuild
LO, HI, MID, N = 88000, 115000, 100000, 50
picked = []
for f in sorted(glob.glob(f"{sys.argv[1]}/*/sft/turns-a2-0001.jsonl")):
    cand = []
    for m, rec, msgs, ids in rebuild([f]):
        if rec.get("purpose") == "gameplay" and LO <= len(ids) <= HI:
            cand.append({"messages": msgs, "tools": m["tools"], "prompt_tokens": len(ids), "game": m["instance"]})
    picked += sorted(cand, key=lambda c: abs(c["prompt_tokens"] - MID))[:2]
rows = sorted(picked, key=lambda c: abs(c["prompt_tokens"] - MID))[:N]
assert len(rows) == N, len(rows)
json.dump({"source": "b2-v9sft-p75c268-kf-all25x5-24h-v2 turns-a2-0001: 50 x ~100K", "contexts": rows}, open(sys.argv[2], "w"))
n = [c["prompt_tokens"] for c in rows]
print(len(rows), "min", min(n), "max", max(n), "mean", round(sum(n) / N), "pool", len(picked), "games", len({c["game"] for c in rows}))
