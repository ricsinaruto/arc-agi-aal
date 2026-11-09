"""v8.2 live read-out from a partial copy of a running arena run. argv: copy_dir (python -I)
copy_dir holds live/progress.json, serving/vllm-openai-server.log and trials/duck-transfer/games/*/{log.jsonl,duck-transcript.txt}."""
import sys, json, re, glob, collections, os
D = sys.argv[1]
try:
    p = json.load(open(D + "/live/progress.json"))
    print("progress:", {k: p.get(k) for k in ("phase", "elapsed_seconds", "active", "levels", "parked", "finished")}, "gate", p.get("gate"),
          "drain", p.get("context_drain"), "loop_rule", p.get("loop_rule"))
    print("  live:", json.dumps(p.get("live"))[:700])
except Exception as exc:
    print("no progress.json:", exc)
kinds = collections.Counter(); per = collections.defaultdict(collections.Counter); drains = []; shown = []
code_hits = collections.Counter(); out_hits = collections.Counter(); animated = 0; frames_max = 0; tokens = 0; errors = collections.Counter()
for f in glob.glob(D + "/trials/duck-transfer/games/*/log.jsonl"):
    game = f.split("/")[-2]
    for line in open(f, errors="replace"):
        if not line.startswith('{"kind":"'):
            continue
        kind = line[9:line.index('"', 9)]
        kinds[kind] += 1; per[kind][game] += 1
        if kind == "context_drain":
            d = json.loads(line); drains.append((d.get("tokens_before"), d.get("tokens_after"), d.get("target"), d.get("estimated_after"), d.get("note_chars")))
        elif kind == "playbook_shown":
            d = json.loads(line); shown.append((d.get("chars"), d.get("writes")))
        elif kind == "observation":
            n = line.count("]],[[") + 1 if '"frames":[[' in line else 1
            # frames is a list of grids; count grids cheaply
            d = json.loads(line); n = len(d.get("frames") or [])
            animated += n > 1; frames_max = max(frames_max, n)
        elif kind == "duck_python":
            d = json.loads(line); code = d.get("code") or ""; res = json.dumps(d.get("result") or "")
            for name, pat in (("playbook(text)", r"playbook\(\s*[^)\s]"), ("playbook()", r"playbook\(\s*\)"), ("note(", r"(?<![a-z_])note\("), ("animation_frames(", r"animation_frames\("),
                              ("animation_timeline(", r"animation_timeline\("), ("animated_steps(", r"animated_steps\("), ("last_animation_", r"last_animation_"), ("expect=", r"expect=")):
                code_hits[name] += bool(re.search(pat, code))
            for name, pat in (("[playbook] saved", r"\[playbook\] saved"), ("NameError note", r"name 'note' is not defined"), ("TRANSIENT EFFECT", r"TRANSIENT EFFECT"),
                              ("Traceback", r"Traceback"), ("animation unavailable", r"no longer stored|No animation is stored")):
                out_hits[name] += bool(re.search(pat, res))
        elif kind in ("error", "analyzer_error", "harness_error", "task_error"):
            errors[line[:200]] += 1
games = len(glob.glob(D + "/trials/duck-transfer/games/*/log.jsonl"))
print("games with logs", games, "events", {k: (kinds[k], len(per[k])) for k in ("model", "duck_python", "observation", "context_drain", "playbook_shown", "loop_rule_fresh_context", "auto_reset", "cap_hit_followup", "dfr_noop_stop", "malformed_tool_calls_dropped") if kinds[k]})
if drains:
    print("drains", len(drains), "before/after/target median", sorted(d[0] for d in drains)[len(drains)//2], sorted(d[1] for d in drains)[len(drains)//2], drains[0][2],
          "over target by >25%:", sum(1 for d in drains if d[1] - d[2] > max(4096, d[2] // 4)), "estimate above target:", sum(1 for d in drains if (d[3] or 0) > d[2]), "note chars median", sorted(d[4] for d in drains)[len(drains)//2])
if shown:
    print("playbook shown in cut notes", len(shown), "chars median/max", sorted(s[0] for s in shown)[len(shown)//2], max(s[0] for s in shown), "empty", sum(1 for s in shown if not s[0]))
print("python calls using:", dict(code_hits)); print("python output:", dict(out_hits))
print("animated actions (frames > 1):", animated, "max frames", frames_max)
if errors: print("ERROR EVENTS:", errors.most_common(5))
pat = {"carried block": r"Working world model carried", "mandate": r"BEFORE EXECUTING NEW ACTIONS", "label sentence": r"world-model update lines",
       "Prior-level": r"Prior-level hypotheses", "note(key, text)": r"note\(key, text\)", "notes block": r"Your notes \(persist",
       "memory framing": r"Three things survive every cut", "solvable line": r"Every level can be completed with the actions offered",
       "animation line": r"animation_frames\(x\)", "playbook in cut note": r"Your playbook, as you last wrote it", "playbook empty in note": r"You have not written a playbook yet",
       "level-clear playbook line": r"rewrite your playbook: fold", "playbook reminder": r"have not written your playbook since the last cut",
       "death v2 first": r"GAME_OVER at step \d+ after [^:\n]+: the attempt on level", "death v2 later": r"GAME_OVER at step \d+ after [^:\n]+: death \d+ on level",
       "old death text": r"BAR RULE|Do NOT re-submit", "old follow-up": r"You have not acted yet", "new follow-up": r"Your previous reply contained no tool call",
       "STEP BACK": r"STEP BACK \(", "transient note": r"were NOT no-ops", "stuck notice": r"\[Stuck|No progress for"}
count = collections.Counter(); files = glob.glob(D + "/trials/duck-transfer/games/*/duck-transcript.txt")
for f in files:
    text = open(f, errors="replace").read()
    for name, pt in pat.items():
        n = len(re.findall(pt, text)); count[name] += n; count[name + "#"] += bool(n)
print("transcripts:", len(files)); print("  " + "; ".join(f"{k}: {count[k]} in {count[k + '#']} games" for k in pat))
marks = collections.Counter(); last = {}; acc = []
for f in glob.glob(D + "/serving/vllm-openai-server.log") + glob.glob(D + "/trials/duck-transfer/vllm-openai-server.log"):
    for line in open(f, errors="replace"):
        for m in ("ARC_TAIL_COLLAPSE", "ARC_REPLY_IDS_STATS", "ARC_STATE_COTENANT", "ARC_DRAFT_HEAD_OVERLAY", "ARC_DRAFT_HEAD_PATCH", "ARC_TAIL_STATS", "ARC_TAIL_INVALID"):
            if m in line:
                marks[m] += 1; last[m] = line[line.index(m):].strip()[:520]
        low = line.lower()
        if "keepalive" in low or "keep_alive" in low or "keep-alive" in low:
            marks["keepalive lines"] += 1; last["keepalive"] = line.strip()[-320:]
        if "preempt" in low: marks["preempt lines"] += 1; last["preempt"] = line.strip()[-200:]
        m = re.search(r"[Aa]cceptance rate: ([0-9.]+)%", line)
        if m: acc.append(float(m.group(1)))
print("server log marks:", dict(marks))
for k in ("ARC_REPLY_IDS_STATS", "ARC_STATE_COTENANT", "ARC_DRAFT_HEAD_OVERLAY", "ARC_TAIL_STATS", "keepalive", "preempt"):
    if k in last: print("  last", k, last[k] if k not in last[k] else last[k])
if acc: print("  draft acceptance %%: last %.1f, median %.1f, min %.1f, n %d" % (acc[-1], sorted(acc)[len(acc) // 2], min(acc), len(acc)))
