"""The conversations live at one moment of a recorded run, for ``l4_steady``: messages at that moment and every later
request's change, in a fixed shuffled order (nested: the first N of any test are the first N of a larger one), then the
conversations admitted after it. Args: SERVED_PARTS_GLOB EVENTS ADMISSION BUNDLE_SRC T OUT.GZ"""
import glob
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

GAP_MAX = 35.0  # s: the recorded idle before a request; the RTX gate's waits are near 0, python's limit is 30


def main():
    parts, events, admission, bundle, moment, out = sys.argv[1:7]
    moment = float(moment)
    sys.path.insert(0, bundle)
    from arc2.served import rebuild

    rows = defaultdict(list)
    for part in sorted(glob.glob(parts)):
        for row, payload in rebuild(part):
            rows[row["task"]].append((row, payload))
    replies = defaultdict(list)
    for line in Path(events).read_text(errors="replace").splitlines():
        if line.endswith("}"):
            e = json.loads(line)
            if e.get("event") == "reply":
                replies[e["task"]].append(e)
    assert set(rows) == set(replies) and all(len(rows[t]) == len(replies[t]) for t in rows)
    held, later = {}, []
    for line in Path(admission).read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "admit":
            if e["t"] <= moment:
                held[e["game"]] = e["t"]
            else:
                later.append(e)
        elif e["event"] == "release" and e["t"] <= moment:
            held.pop(e["game"], None)

    def conversation(task, start):
        """Requests from index ``start`` on; ``gap``: the idle before each after the first."""
        chain, ends = rows[task], replies[task]
        keep = []
        for i in range(start + 1, len(chain)):
            row, payload = chain[i]
            began = ends[i]["t"] - ends[i]["seconds"]
            keep.append({"keep": row["keep"], "append": payload["messages"][row["keep"]:], "max_tokens": row["max_tokens"],
                         "gap": round(min(GAP_MAX, max(0.0, began - ends[i - 1]["t"])), 2)})
        return {"task": task, "messages": chain[start][1]["messages"], "max_tokens": chain[start][0]["max_tokens"],
                "rows": keep, "context_tokens": ends[start]["prompt_tokens"]}

    live = []
    for task in held:
        ended = [i for i, e in enumerate(replies[task]) if e["t"] <= moment]
        start = (ended[-1] + 1) if ended else 0
        if start < len(rows[task]):
            live.append(conversation(task, start))
    random.Random(12).shuffle(live)
    seen = {c["task"] for c in live}
    following = []
    for e in later:
        task = e["game"]
        if task in seen or task not in rows:
            continue
        seen.add(task)
        after = [i for i, r in enumerate(replies[task]) if r["t"] > e["t"]]
        if after:
            following.append(conversation(task, after[0]))
    meta = {"moment": moment, "live": len(live), "following": len(following),
            "context_mean": sum(c["context_tokens"] for c in live) // max(1, len(live))}
    with gzip.open(out, "wt", compresslevel=6) as stream:
        json.dump({"meta": meta, "conversations": live + following}, stream)
    print(meta)
    for n in (34, 40, 45, 50, 60):
        print(n, sum(c["context_tokens"] for c in live[:n]))


if __name__ == "__main__":
    main()
