"""Harness, compaction and prompt audit of one duck-transfer arena trial (29 September, CPU only).

Written for the hero-v2 2 h run (docs/validation-20260927/hero-v2-analysis/harness.md); it reads only the recorded
files of a trial (``arena.json``, ``requests.json``, ``public-roster.json`` and every game's ``log.jsonl`` and
``prompts/prompt.log``) and prints one JSON document with:

- ``compactions``: every summary (tokens, finish, sections present, stated level/step against the truth, helper
  recall, ending), the tokens between compactions and the prompt size before/after;
- ``post_compaction``: the K gameplay replies before and after each compaction (python errors, actions,
  no-effect actions, refused repeats, re-definitions of existing helpers, tokens per action);
- ``structure``: the message layout of every ``prompt.log`` snapshot that carries a compaction;
- ``errors``: python-tool errors by class, timeouts/resets, refused actions, length stops, tool-less replies,
  deadline timeouts, per game;
- ``turns``: reasoning tokens per reply, completion tokens per action, actions per reply, no-effect streaks,
  the carried world model (assistant ``content``) and the stuck notices;
- ``roundtrip``: prompt(n) - prompt(n-1) - completion(n-1) for consecutive gameplay requests (a re-sent history
  that drops or re-renders the previous reply shows up as a negative or short delta);
- ``levels``: per level clear, the first replies on the new level (helper reuse, inspection-only replies);
- ``parks``: behaviour of the first replies after every re-admission.

``--compare`` adds the summary-quality table for other trials (e.g. thinking-on and thinking-off compaction runs).

    python scripts/arena_analysis/harness_audit.py --trial <run>/<run>/arena/trials/duck-transfer \
        [--compare label=<trial> ...] [--out audit.json]
"""

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

WINDOW = 6  # gameplay replies before/after a compaction, a re-admission or a level start
SECTIONS = {
    "world": r"world model",
    "goal": r"goal model",
    "action": r"action model",
    "current": r"current level|current state|level and state",
    "tried": r"tried|without effect|no[- ]effect|no-op",
    "helpers": r"helper|python session|defined in|functions",
    "plan": r"\bplan\b",
    "cross": r"cross[- ]level",
}
DEF_RE = re.compile(r"^def\s+([A-Za-z_]\w*)\s*\(", re.M)
ASSIGN_RE = re.compile(r"^([A-Za-z_]\w*)\s*=", re.M)
ERR_RE = re.compile(r"(\w+Error|\w+Exception)\b")
LEVEL_RE = re.compile(r"\blevel\s*(\d+)", re.I)
STEP_RE = re.compile(r"\bstep\s*(\d+)", re.I)


def pct(values, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)):
    v = sorted(values)
    if not v:
        return {}
    return {f"p{int(q * 100)}": v[min(len(v) - 1, int(q * len(v)))] for q in qs} | {"max": v[-1], "n": len(v),
                                                                                       "mean": round(sum(v) / len(v), 1)}


def board_key(obs):
    frames = obs.get("frames") or []
    frame = frames[-1] if frames else []
    return hashlib.sha1("\n".join(frame).encode()).hexdigest()


def load_trial(trial):
    trial = Path(trial)
    arena = json.loads((trial / "arena.json").read_text())
    roster = {}
    if (trial / "public-roster.json").is_file():
        roster = {r["game_id"]: r["source_game_id"][:4] for r in json.loads((trial / "public-roster.json").read_text())}
    records = {g["game_id"]: g for g in arena["games"]}
    games = {}
    for game_dir in sorted((trial / "games").iterdir()):
        log = game_dir / "log.jsonl"
        if not log.is_file():
            continue
        events = [json.loads(line) for line in log.open()]
        prompt = game_dir / "prompts" / "prompt.log"
        games[game_dir.name] = {"events": events, "source": roster.get(game_dir.name, game_dir.name),
                                "record": records.get(game_dir.name, {}),
                                "prompt_log": prompt.read_text() if prompt.is_file() else None}
    return arena, games


def summary_of(message, mode):
    """The handoff text as the harness built it (arena.compaction.prefilled_summary_text / summary_text)."""
    def text(v):
        if isinstance(v, list):
            v = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in v)
        return str(v or "").strip()

    if mode == "prefilled-empty":
        parts = []
        for key in ("reasoning_content", "reasoning", "content"):
            t = text(message.get(key))
            if t and t not in parts:
                parts.append(t)
        return ("World model: " + "\n".join(parts)) if parts else ""
    return text(message.get("content"))


def annotate(game):
    """One pass over a game's events: replies, actions and compactions with their context."""
    replies, actions, comps, pythons = [], [], [], []
    levels, board, seen = 0, None, defaultdict(set)
    defined, resets = {}, 0
    last_reply = None
    tokens = 0  # completion tokens, every purpose
    pending_comp_model = None
    allowance = None
    for ev in game["events"]:
        kind = ev["kind"]
        if kind == "exact_admission_count":
            allowance = ev.get("output_allowance")
        if kind == "observation":
            key = board_key(ev)
            new_levels = int(ev.get("levels", levels))
            if ev.get("action") is not None:
                a = ev["action"]
                label = a.get("action") + (f"({a.get('y')},{a.get('x')})" if a.get("x") is not None else "")
                no_effect = key == board and new_levels == levels
                revisit = new_levels == levels and key in seen[levels] and not no_effect
                actions.append({"step": ev["step"], "label": label, "no_effect": no_effect, "revisit": revisit,
                                "cleared": new_levels > levels, "level": levels + 1, "reply": last_reply,
                                "state": ev.get("state"), "tokens": tokens})
                if last_reply is not None:
                    replies[last_reply]["actions"] += 1
                    replies[last_reply]["no_effect"] += no_effect
                    replies[last_reply]["cleared"] += new_levels > levels
            levels, board = new_levels, key
            seen[levels].add(key)
        elif kind == "model":
            usage = ev.get("usage") or {}
            ct = int(usage.get("completion_tokens") or 0)
            tokens += ct
            if ev.get("purpose") == "compaction":
                pending_comp_model = {**ev, "allowance": allowance}
                continue
            msg = ev.get("message") or {}
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            replies.append({"i": len(replies), "step": ev["step"], "prompt": int(usage.get("prompt_tokens") or 0),
                            "completion": ct, "reasoning": int(reasoning), "finish": ev.get("finish_reason"),
                            "tool_calls": len(msg.get("tool_calls") or []),
                            "content": (msg.get("content") or "").strip(), "level": levels + 1, "actions": 0,
                            "no_effect": 0, "cleared": 0, "py_errors": 0, "py_error_kinds": [], "redefs": 0,
                            "defs": 0, "skipped": 0, "time": ev["time"], "tokens_before": tokens - ct,
                            "comps_before": len(comps)})
            last_reply = len(replies) - 1
        elif kind == "duck_python":
            res = ev.get("result") or {}
            code = ev.get("code") or ""
            names = DEF_RE.findall(code)
            rec = {"step": ev["step"], "error": res.get("error"), "reply": last_reply, "defs": names}
            pythons.append(rec)
            if last_reply is not None:
                r = replies[last_reply]
                r["defs"] += len(names)
                r["redefs"] += sum(n in defined for n in names)
                if res.get("error"):
                    r["py_errors"] += 1
                    m = ERR_RE.findall(res["error"])
                    r["py_error_kinds"].append(m[-1] if m else ("timeout" if "timed out" in res["error"] else "other"))
            for n in names:
                defined[n] = ev["step"]
        elif kind == "duck_python_reset":
            resets += 1
            defined = {}
        elif kind == "action_skipped" and last_reply is not None:
            replies[last_reply]["skipped"] += 1
        elif kind == "compaction":
            m = pending_comp_model or {}
            comps.append({"event": ev, "model": m, "step": ev["step"], "level": levels + 1, "levels": levels,
                          "reply_index": len(replies), "tokens": tokens, "defined": dict(defined),
                          "resets": resets, "time": ev["time"]})
            pending_comp_model = None
    return {"replies": replies, "actions": actions, "comps": comps, "pythons": pythons}


def summary_quality(text, truth_level, truth_step, defined, finish):
    low = text.lower()
    found = {k: bool(re.search(p, low)) for k, p in SECTIONS.items()}
    cur = re.split(r"current level|current state|level and state", low, maxsplit=1)
    cur = cur[1][:600] if len(cur) > 1 else low
    lv = LEVEL_RE.findall(cur)
    st = STEP_RE.findall(cur)
    helpers = [n for n in defined if len(n) > 2]
    mentioned = [n for n in helpers if re.search(r"\b" + re.escape(n) + r"\b", text)]
    tail = text.rstrip()[-1:] if text.strip() else ""
    return {
        "chars": len(text),
        "sections": found,
        "section_count": sum(found.values()),
        "stated_level": int(lv[0]) if lv else None,
        "level_ok": (int(lv[0]) == truth_level) if lv else None,
        "level_stated": bool(lv),
        "stated_step": int(st[0]) if st else None,
        "step_err": (int(st[0]) - truth_step) if st else None,
        "helpers_defined": len(helpers),
        "helpers_mentioned": len(mentioned),
        "ends_clean": finish != "length" and tail in ".)]`'\"!?*|>0123456789",
        "last_chars": text.rstrip()[-80:],
        "coords": len(re.findall(r"\(\s*\d+\s*,\s*\d+\s*\)", text)),
    }


def compaction_table(games, mode):
    rows = []
    for gid, g in games.items():
        a = g["ann"]
        for k, c in enumerate(a["comps"]):
            ev, m = c["event"], c["model"]
            usage = ev.get("usage") or {}
            text = summary_of(m.get("message") or {}, mode) if m else ""
            q = summary_quality(text, c["level"], c["step"], c["defined"], m.get("finish_reason"))
            nxt = [r for r in a["replies"] if r["i"] >= c["reply_index"]]
            prv = [r for r in a["replies"] if r["i"] < c["reply_index"]]
            rows.append({
                "game": gid, "source": g["source"], "k": k, "step": c["step"], "level": c["level"],
                "estimate": ev.get("estimate"), "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
                "finish": m.get("finish_reason"), "seconds": ev.get("seconds"), "attempts": ev.get("attempts"),
                "mechanical": ev.get("mechanical_fallback"), "summary_chars_logged": ev.get("summary_chars"),
                "post_prompt": nxt[0]["prompt"] if nxt else None, "pre_prompt": prv[-1]["prompt"] if prv else None,
                "pre_completion": prv[-1]["completion"] if prv else None,
                "tokens_at": c["tokens"], "resets_before": c["resets"],
                "headroom": (m["allowance"] - (usage.get("completion_tokens") or 0)) if m.get("allowance") else None,
                **q,
            })
    return rows


def window_stats(replies):
    acts = sum(r["actions"] for r in replies)
    comp = sum(r["completion"] for r in replies)
    return {"replies": len(replies), "actions": acts, "no_effect": sum(r["no_effect"] for r in replies),
            "py_errors": sum(r["py_errors"] for r in replies), "redefs": sum(r["redefs"] for r in replies),
            "defs": sum(r["defs"] for r in replies), "skipped": sum(r["skipped"] for r in replies),
            "zero_action_replies": sum(r["actions"] == 0 for r in replies), "completion": comp,
            "clears": sum(r["cleared"] for r in replies),
            "name_errors": sum(k == "NameError" for r in replies for k in r["py_error_kinds"])}


def add(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def rates(s):
    return {**s, "py_error_rate": round(s["py_errors"] / max(1, s["replies"]), 3),
            "actions_per_reply": round(s["actions"] / max(1, s["replies"]), 2),
            "no_effect_share": round(s["no_effect"] / max(1, s["actions"]), 3),
            "zero_action_share": round(s["zero_action_replies"] / max(1, s["replies"]), 3),
            "completion_per_action": round(s["completion"] / max(1, s["actions"])),
            "redef_per_reply": round(s["redefs"] / max(1, s["replies"]), 2)}


def post_compaction(games, k=WINDOW):
    before, after, n = {}, {}, 0
    per = []
    for gid, g in games.items():
        a = g["ann"]
        idx = [c["reply_index"] for c in a["comps"]]
        for j, i in enumerate(idx):
            lo = idx[j - 1] if j else 0
            hi = idx[j + 1] if j + 1 < len(idx) else len(a["replies"])
            pre = a["replies"][max(lo, i - k):i]
            post = a["replies"][i:min(hi, i + k)]
            if len(pre) < k or len(post) < k:
                continue
            n += 1
            sb, sa = window_stats(pre), window_stats(post)
            before, after = add(before, sb), add(after, sa)
            per.append({"game": gid, "step": a["comps"][j]["step"], "before": sb, "after": sa})
    return {"windows": n, "k": k, "before": rates(before) if n else None, "after": rates(after) if n else None,
            "per": per}


def baseline_windows(games, k=WINDOW):
    """Every run of k consecutive replies not touching a compaction: the no-compaction reference."""
    s, n = {}, 0
    for g in games.values():
        a = g["ann"]
        cut = {c["reply_index"] for c in a["comps"]}
        reps = a["replies"]
        for i in range(0, len(reps) - k + 1, k):
            if any(i < c <= i + k for c in cut):
                continue
            s = add(s, window_stats(reps[i:i + k]))
            n += 1
    return {"windows": n, **rates(s)}


def structure(games):
    out = []
    for gid, g in games.items():
        text = g["prompt_log"]
        if not text or "[Context compaction]" not in text:
            continue
        body = text.split("[MODEL INPUT]", 1)[1]
        roles = re.findall(r"^\[(SYSTEM|USER|ASSISTANT|TOOL RESULT)[^\]]*\]\s*$", body, re.M)
        blocks = re.split(r"^\[(?:SYSTEM|USER|ASSISTANT|TOOL RESULT)[^\]]*\]\s*$", body, flags=re.M)[1:]
        first_user = blocks[1] if len(blocks) > 1 else ""
        second_user_i = next((i for i, r in enumerate(roles) if r == "USER" and i > 1), None)
        second = blocks[second_user_i] if second_user_i is not None else ""
        out.append({
            "game": gid, "messages": len(roles), "role_head": roles[:4],
            "summary_second": roles[:2] == ["SYSTEM", "USER"] and first_user.lstrip().startswith("[Context compaction]"),
            "observation_third": len(roles) > 2 and roles[2] == "USER",
            "third_starts": second.strip()[:40] if second_user_i == 2 else None,
            "compaction_headers": body.count("[Context compaction]"),
            "user_turns_after_summary": roles.count("USER") - 1,
        })
    return out


def errors(games, requests):
    per_game = {}
    kinds = Counter()
    examples = defaultdict(list)
    for gid, g in games.items():
        c = Counter()
        for ev in g["events"]:
            kind = ev["kind"]
            if kind == "duck_python":
                res = ev.get("result") or {}
                err = res.get("error")
                if err:
                    if "timed out" in err:
                        k = "python_timeout"
                    elif "not allowed in the sandbox" in err:
                        k = "sandbox_import_blocked"
                    elif "syntax error" in err.lower():
                        k = "SyntaxError"
                    elif "non-empty `code`" in err:
                        k = "empty_code"
                    elif re.search(r"NameError: name '(open|globals|locals|id|vars|exec|eval|__import__|compile)'", err):
                        k = "sandbox_builtin_missing"
                    else:
                        m = ERR_RE.findall(err)
                        k = m[-1] if m else "other"
                    c[k] += 1
                    kinds[k] += 1
                    if len(examples[k]) < 3:
                        examples[k].append({"game": gid, "step": ev["step"], "error": err[-200:]})
                    if (ev.get("code") or "").count("action(") and not res.get("stdout"):
                        c["error_in_acting_call"] += 1
            elif kind == "model":
                if ev.get("finish_reason") == "length":
                    c["length_stop"] += 1
                if ev.get("purpose") == "gameplay" and not (ev.get("message") or {}).get("tool_calls"):
                    c["toolless_reply"] += 1
                if ev.get("purpose") == "gameplay" and len((ev.get("message") or {}).get("tool_calls") or []) > 1:
                    c["multi_tool_reply"] += 1
            elif kind in ("action_skipped",):
                c[f"skipped:{ev.get('reason')}"] += 1
            elif kind in ("context_rejection", "context_precheck", "token_count_error", "compaction_empty",
                          "compaction_rejected", "compaction_prefix_refused", "deadline_request_timeout",
                          "duck_python_reset", "auto_reset", "worker_request_cancelled", "aside_cap_fitted",
                          "state_flush_failed", "worker_deadline_drained"):
                c[kind] += 1
        per_game[gid] = dict(c)
    totals = Counter()
    for c in per_game.values():
        totals.update(c)
    req = Counter((r.get("status"), r.get("error"), r.get("finish_reason")) for r in requests)
    return {"totals": dict(totals), "python_error_kinds": dict(kinds), "examples": dict(examples),
            "requests": {str(k): v for k, v in req.items()},
            "per_game": {g: c for g, c in per_game.items() if c}}


def turns(games):
    reasoning, completion, per_action, apr = [], [], {}, Counter()
    streaks, stuck_secs, content_nonempty, wm_labels = [], [], 0, 0
    total = 0
    for gid, g in games.items():
        a = g["ann"]
        for r in a["replies"]:
            total += 1
            reasoning.append(r["reasoning"])
            completion.append(r["completion"])
            apr[min(r["actions"], 10)] += 1
            content_nonempty += bool(r["content"])
            wm_labels += "world model" in r["content"].lower()
        acts = len(a["actions"])
        if acts:
            per_action[gid] = round(sum(r["completion"] for r in a["replies"]) / acts)
        run, best, prev = 0, (0, None), None
        for x in a["actions"]:
            if x["no_effect"] and x["label"] == prev:
                run += 1
            else:
                run = 1 if x["no_effect"] else 0
            if run > best[0]:
                best = (run, x["label"])
            prev = x["label"]
        streaks.append((best[0], gid, best[1]))
        for ev in g["events"]:
            if ev["kind"] == "stuck_notice":
                m = re.search(r"\((\d+)s\), (\d+) consecutive", ev["text"])
                if m:
                    stuck_secs.append(int(m.group(1)))
    acts_total = sum(len(g["ann"]["actions"]) for g in games.values())
    no_eff = sum(x["no_effect"] for g in games.values() for x in g["ann"]["actions"])
    rev = sum(x["revisit"] for g in games.values() for x in g["ann"]["actions"])
    return {"replies": total, "reasoning_tokens": pct(reasoning), "completion_tokens": pct(completion),
            "reasoning_share": round(sum(reasoning) / max(1, sum(completion)), 3),
            "replies_reasoning_ge_8k": sum(x >= 8192 for x in reasoning),
            "completion_per_action_by_game": pct(list(per_action.values())),
            "actions_per_reply": dict(sorted(apr.items())), "actions": acts_total,
            "no_effect_actions": no_eff, "revisit_actions": rev,
            "longest_same_action_no_effect_streaks": sorted(streaks, reverse=True)[:10],
            "content_nonempty_replies": content_nonempty, "content_with_world_model_label": wm_labels,
            "stuck_notice_seconds": pct(stuck_secs)}


def roundtrip(games):
    deltas, neg = [], []
    for gid, g in games.items():
        prev = prev_finish = None
        for ev in g["events"]:
            if ev["kind"] == "compaction":
                prev = None
            elif ev["kind"] == "model" and ev.get("purpose") == "gameplay":
                u = ev.get("usage") or {}
                if prev is not None:
                    d = u["prompt_tokens"] - prev["prompt_tokens"] - prev["completion_tokens"]
                    deltas.append(d)
                    if d < 20:
                        neg.append({"game": gid, "step": ev["step"], "delta": d,
                                    "previous_completion": prev["completion_tokens"],
                                    "previous_finish": prev_finish})
                prev, prev_finish = u, ev.get("finish_reason")
    return {"deltas": pct(deltas, (0.0, 0.01, 0.05, 0.5, 0.95)), "below_20": neg}


def level_starts(games, k=WINDOW):
    """First k replies of every level >= 2 against the level's later replies."""
    first, later, reuse = {}, {}, Counter()
    for gid, g in games.items():
        a = g["ann"]
        by_level = defaultdict(list)
        for r in a["replies"]:
            by_level[r["level"]].append(r)
        for lv, reps in by_level.items():
            if lv < 2:
                continue
            first = add(first, window_stats(reps[:k]))
            if len(reps) > k:
                later = add(later, window_stats(reps[k:]))
        # helper reuse: python calls in the first replies of a level that call a helper defined on an earlier level
        defined_at = {}
        level_of_reply = {r["i"]: r["level"] for r in a["replies"]}
        seen_level_calls = Counter()
        for ev, p in zip([e for e in g["events"] if e["kind"] == "duck_python"], a["pythons"]):
            lv = level_of_reply.get(p["reply"], 1)
            code = ev.get("code") or ""
            if lv >= 2 and seen_level_calls[lv] < 3:
                seen_level_calls[lv] += 1
                old = [n for n, at in defined_at.items() if at < lv and re.search(r"\b" + re.escape(n) + r"\s*\(", code)
                       and n not in DEF_RE.findall(code)]
                reuse["first_calls"] += 1
                reuse["first_calls_reusing_old_helper"] += bool(old)
                reuse["first_calls_redefining"] += bool(DEF_RE.findall(code))
            for n in DEF_RE.findall(code):
                defined_at.setdefault(n, lv)
    return {"first": rates(first) if first else None, "later": rates(later) if later else None,
            "helper_reuse": dict(reuse)}


def parks(games, k=WINDOW):
    out = {"readmissions": 0, "with_replies_after": 0}
    agg = {}
    gaps = []
    for gid, g in games.items():
        rec = g["record"]
        a = g["ann"]
        adm = rec.get("admissions") or []
        parked_at = None
        for e in adm:
            if e["event"] == "park":
                parked_at = e["wall_seconds"]
            elif e["event"] == "admit" and parked_at is not None:
                gap = e["wall_seconds"] - parked_at
                if gap < 1:  # regrant with no real pause
                    parked_at = None
                    continue
                out["readmissions"] += 1
                gaps.append(round(gap))
                # replies after the re-admission: those recorded after it (the first is the request that waited
                # in the gate while the game was parked)
                start = g.get("epoch0")
                if start is None:
                    parked_at = None
                    continue
                t_admit = start + e["wall_seconds"]
                post = [r for r in a["replies"] if r["time"] >= t_admit][:k]
                if post:
                    out["with_replies_after"] += 1
                    agg = add(agg, window_stats(post))
                parked_at = None
    out["pause_seconds"] = pct(gaps)
    out["after"] = rates(agg) if agg else None
    return out


def _depth(reps, i):
    """Analysis-only replies since the last acting reply (how deep into a turn reply ``i`` starts)."""
    n, k = 0, i - 1
    while k >= 0 and not reps[k]["actions"]:
        n, k = n + 1, k - 1
    return n


def _to_action(reps, i):
    t = 0
    for n, r in enumerate(reps[i:], 1):
        t += r["completion"]
        if r["actions"]:
            return t, n
    return None


def reground(games):
    """Completion tokens from a compaction to the next action, against mid-turn points at the same depth into the
    turn (0, 1-2, 3-5, 6+ analysis-only replies before) that are not near a compaction."""
    post, base = defaultdict(list), defaultdict(list)
    for g in games.values():
        reps = g["ann"]["replies"]
        cuts = {c["reply_index"] for c in g["ann"]["comps"]}
        for i in range(len(reps)):
            v = _to_action(reps, i)
            if v is None:
                continue
            d = _depth(reps, i)
            b = 0 if d == 0 else 1 if d <= 2 else 2 if d <= 5 else 3
            if i in cuts:
                post[b].append(v)
            elif not any(0 <= i - c < 15 for c in cuts) and not any(i < c <= i + v[1] for c in cuts):
                base[b].append(v)
    out, excess = {}, 0.0
    for b in sorted(post):
        p, q = [t for t, _ in post[b]], [t for t, _ in base[b]]
        mp, mq = sum(p) / len(p), (sum(q) / len(q) if q else None)
        if mq is not None:
            excess += (mp - mq) * len(p)
        out[str(b)] = {"post_n": len(p), "post_mean": round(mp), "post_median": sorted(p)[len(p) // 2],
                       "post_replies": round(sum(n for _, n in post[b]) / len(p), 1), "base_n": len(q),
                       "base_mean": round(mq) if mq is not None else None,
                       "base_median": sorted(q)[len(q) // 2] if q else None}
    return {"by_depth": out, "excess_tokens": round(excess), "compactions": sum(len(v) for v in post.values())}


def stuck_notices(games, epoch0):
    """Seconds the stuck notice claims for the level against the pause (parked) seconds inside that span."""
    rows = []
    for gid, g in games.items():
        rec = g["record"]
        start_of = {e["level"]: e["wall_seconds"] for e in rec.get("time_to_level") or []}
        parked, pk = [], None
        for e in rec.get("admissions") or []:
            if e["event"] == "park":
                pk = e["wall_seconds"]
            elif e["event"] == "admit" and pk is not None:
                parked.append((pk, e["wall_seconds"]))
                pk = None
        if pk is not None:
            parked.append((pk, float("inf")))
        for ev in g["events"]:
            if ev["kind"] != "stuck_notice" or epoch0 is None:
                continue
            m = re.search(r"on level (\d+) \((\d+)s\), (\d+) consecutive", ev["text"])
            if not m:
                continue
            lv, secs, streak = int(m.group(1)), int(m.group(2)), int(m.group(3))
            wall = ev["time"] - epoch0
            start = start_of.get(lv - 1, 0.0)
            pause = sum(max(0.0, min(b, wall) - max(a, start)) for a, b in parked)
            rows.append({"game": gid, "level": lv, "claimed": secs, "parked_inside": round(pause), "streak": streak})
    return {"n": len(rows), "turn_triggered": sum(r["streak"] == 0 for r in rows),
            "parked_ge_600": sum(r["parked_inside"] >= 600 for r in rows),
            "worst": sorted(rows, key=lambda r: -r["parked_inside"])[:5]}


def analysis_stretches(games):
    """Analysis-only replies before an acting reply, and whether a level clears within the next 3 acting replies."""
    out = defaultdict(lambda: {"acting_replies": 0, "clear_within_3": 0, "tokens": 0, "actions": 0})
    for g in games.values():
        reps = g["ann"]["replies"]
        n = t = 0
        for i, r in enumerate(reps):
            if not r["actions"]:
                n, t = n + 1, t + r["completion"]
                continue
            b = "0" if n == 0 else "1-2" if n <= 2 else "3-7" if n <= 7 else "8+"
            o = out[b]
            o["acting_replies"] += 1
            o["clear_within_3"] += any(q["cleared"] for q in [q for q in reps[i:] if q["actions"]][:3])
            o["tokens"] += t + r["completion"]
            o["actions"] += r["actions"]
            n = t = 0
    total = sum(r["completion"] for g in games.values() for r in g["ann"]["replies"] if not r["actions"])
    return {"analysis_only_tokens": total,
            **{k: {**v, "clear_rate": round(v["clear_within_3"] / v["acting_replies"], 3),
                   "tokens_per_stretch": round(v["tokens"] / v["acting_replies"]),
                   "actions_per_acting_reply": round(v["actions"] / v["acting_replies"], 1)}
               for k, v in sorted(out.items())}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True)
    parser.add_argument("--compare", nargs="*", default=[], help="label=trial_dir (summary quality only)")
    parser.add_argument("--out")
    args = parser.parse_args()
    trial = Path(args.trial)
    arena, games = load_trial(trial)
    mode = (arena.get("compaction_preflight") or {}).get("mode") or "gameplay"
    epoch0 = None
    adm = trial / "admission.jsonl"
    if adm.is_file():
        first = json.loads(adm.open().readline())
        epoch0 = first["epoch"] - first["wall_seconds"]
    for g in games.values():
        g["ann"] = annotate(g)
        g["epoch0"] = epoch0
    requests = json.loads((trial / "requests.json").read_text()) if (trial / "requests.json").is_file() else []
    comps = compaction_table(games, mode)
    res = {"trial": str(trial), "mode": mode, "compaction_preflight": arena.get("compaction_preflight")}
    res["compactions"] = {
        "n": len(comps), "games": len({c["game"] for c in comps}),
        "finish": dict(Counter(c["finish"] for c in comps)),
        "completion_tokens": pct([c["completion_tokens"] or 0 for c in comps]),
        "reasoning_tokens": pct([c["reasoning_tokens"] for c in comps]),
        "chars": pct([c["chars"] for c in comps]),
        "seconds": pct([c["seconds"] or 0 for c in comps]),
        "estimate": pct([c["estimate"] or 0 for c in comps]),
        "prompt_tokens": pct([c["prompt_tokens"] or 0 for c in comps]),
        "post_prompt": pct([c["post_prompt"] for c in comps if c["post_prompt"]]),
        "section_count": dict(Counter(c["section_count"] for c in comps)),
        "sections_missing": dict(Counter(k for c in comps for k, v in c["sections"].items() if not v)),
        "level_ok": dict(Counter(str(c["level_ok"]) for c in comps)),
        "headroom": pct([c["headroom"] for c in comps if c["headroom"] is not None], (0.0, 0.01, 0.05, 0.5)),
        "coords": pct([c["coords"] for c in comps]),
        "step_err": pct([abs(c["step_err"]) for c in comps if c["step_err"] is not None]),
        "helper_recall": round(sum(c["helpers_mentioned"] for c in comps) / max(1, sum(c["helpers_defined"] for c in comps)), 3),
        "ends_clean": sum(c["ends_clean"] for c in comps),
        "mechanical": sum(bool(c["mechanical"]) for c in comps),
        "retries": sum((c["attempts"] or 1) > 1 for c in comps),
        "rows": comps,
    }
    gaps = []
    for g in games.values():
        comps_g = g["ann"]["comps"]
        for a, b in zip(comps_g, comps_g[1:]):
            gaps.append(b["tokens"] - a["tokens"])
    res["compactions"]["completion_tokens_between"] = pct(gaps)
    res["compactions"]["first_at_tokens"] = pct([g["ann"]["comps"][0]["tokens"] for g in games.values() if g["ann"]["comps"]])
    res["post_compaction"] = post_compaction(games)
    res["baseline_windows"] = baseline_windows(games)
    res["structure"] = structure(games)
    res["errors"] = errors(games, requests)
    res["turns"] = turns(games)
    res["roundtrip"] = roundtrip(games)
    res["levels"] = level_starts(games)
    res["parks"] = parks(games)
    res["reground"] = reground(games)
    res["stuck_notices"] = stuck_notices(games, epoch0)
    res["analysis_stretches"] = analysis_stretches(games)
    compare = {}
    for spec in args.compare:
        label, path = spec.split("=", 1)
        a2, g2 = load_trial(path)
        m2 = (a2.get("compaction_preflight") or {}).get("mode")
        for g in g2.values():
            g["ann"] = annotate(g)
        rows = compaction_table(g2, m2 or "gameplay")
        if not rows:
            continue
        compare[label] = {
            "n": len(rows), "mode": m2 or ("gameplay" if statistics.median(r["reasoning_tokens"] for r in rows) > 0 else "off"),
            "completion_tokens": statistics.median(r["completion_tokens"] or 0 for r in rows),
            "reasoning_tokens": statistics.median(r["reasoning_tokens"] for r in rows),
            "chars": statistics.median(r["chars"] for r in rows),
            "coords": statistics.median(r["coords"] for r in rows),
            "sections_mean": round(sum(r["section_count"] for r in rows) / len(rows), 2),
            "level_stated": round(sum(r["level_stated"] for r in rows) / len(rows), 3),
            "level_wrong": sum(r["level_ok"] is False for r in rows),
            "step_abs_err_median": statistics.median([abs(r["step_err"]) for r in rows if r["step_err"] is not None] or [0]),
            "helper_recall": round(sum(r["helpers_mentioned"] for r in rows) / max(1, sum(r["helpers_defined"] for r in rows)), 3),
            "finish_length": sum(r["finish"] == "length" for r in rows),
            "post_compaction": {k: v for k, v in (post_compaction(g2)["after"] or {}).items()},
            "pre_compaction": {k: v for k, v in (post_compaction(g2)["before"] or {}).items()},
            "baseline": baseline_windows(g2),
            "reground": reground(g2),
        }
    res["compare"] = compare
    text = json.dumps(res, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text)
    brief = {k: v for k, v in res.items() if k not in ("structure",)}
    brief["compactions"] = {k: v for k, v in res["compactions"].items() if k != "rows"}
    brief["post_compaction"] = {k: v for k, v in res["post_compaction"].items() if k != "per"}
    brief["errors"] = {k: v for k, v in res["errors"].items() if k != "per_game"}
    print(json.dumps(brief, indent=1, default=str))


if __name__ == "__main__":
    main()
