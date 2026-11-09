"""Audit of the post-training data before any training run (CPU only; reads data/posttrain and runs/).

Usage:
    uv run --with tokenizers python scripts/posttrain/audit_data.py [--out docs/posttrain/data-audit.json]

Sources audited (all in the shared record shape of ``posttrain.sft_records``):

* ``segment``: data/posttrain/segments/segments.jsonl (Flash one-hour runs, 14 dev games) via
  ``sft_records.segment_records``;
* ``teacher``: data/posttrain/teacher/segments.jsonl (Flash on the 20 generated mechanics);
* ``solver``: data/posttrain/solver/records.jsonl (BFS solver through the harness);
* ``aux``: data/posttrain/aux/tasks.jsonl (single-turn predict / reach / transition tasks).

Tokens are counted with the Qwen3.6-35B-A3B tokenizer (``tokenizer.json`` from the local Hugging Face
cache; Ornith-1.5-35B-A3B is the same architecture, assumed the same vocabulary) over a rendering that
follows the Qwen3.6 chat template: the tool block in the system turn, ``<think>`` reasoning kept only on
assistant turns after the last user message, tool results as ``<tool_response>`` user turns, and 64
tokens per 256x256 board image (patch 16, merge 2). Without ``tokenizers`` it falls back to
characters / 4.1 and says so in the output.

Quality checks join every segment request to its run log (``runs/<run>/.../games/<game>/log.jsonl``,
parsed with ``posttrain.segments.game_turns``) through the tool-call ids, which gives the served
``finish_reason``, the real prompt length, the python results and the environment actions (with the
frames before and after each action).

The same log reader summarises baseline runs (Ornith one hour; Flash one-hour base runs; evolve-v4's
holdout-only Flash run) per game group (dev / holdout).
"""

import argparse
import base64
import collections
import difflib
import glob
import hashlib
import json
import re
import statistics
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from posttrain import segments as seg  # noqa: E402
from posttrain.sft_records import segment_records  # noqa: E402
from posttrain.split import DEV, GENERATED_HOLDOUT_MECHANICS, HOLDOUT, prefix  # noqa: E402

DATA = ROOT / "data" / "posttrain"
TOKENIZER_GLOB = str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/*/tokenizer.json")
IMAGE_TOKENS = 64  # 256x256 PNG, patch 16, merge 2 (checked against the first image below)
COMPLETION_CAP = 16384
HOLDOUT_RE = re.compile(r"\b(" + "|".join(HOLDOUT) + r")\b")
ALL_GAME_RE = re.compile(r"\b(" + "|".join(DEV + HOLDOUT + ("ft09", "lp85")) + r")\b")
EVAL_MECHANICS = GENERATED_HOLDOUT_MECHANICS  # held-out generated mechanics (+ q1NN skins), posttrain.split
BASELINE_RUNS = {
    "ornith-v1 (23 games)": "runs/kaggle-arena-duck-60m-r23-ornith-v1/arena/trials/duck-transfer",
    "flash base-v4 (23 games)": "runs/kaggle-arena-duck-60m-r23-base-v4/arena/trials/duck-transfer",
    "flash base-v5 (23 games)": "runs/kaggle-arena-duck-60m-r23-base-v5/arena/trials/duck-transfer",
    "flash base-v2 modal (23 games)": "runs/modal-validation/arena-duck-60m-r23-base-v2-modal-20260925/arena/trials/duck-transfer",
    "flash base-v3 modal (23 games)": "runs/modal-validation/arena-duck-60m-r23-base-v3-modal-20260925/arena/trials/duck-transfer",
    "flash evolve-v4 holdout-baseline (9 holdout games)": "runs/kaggle-arena-duck-60m-r23-evolve-v4/arena/evolve/holdout-baseline/duck-transfer",
    "flash evolve-v4 round-00 (14 dev games)": "runs/kaggle-arena-duck-60m-r23-evolve-v4/arena/evolve/round-00/duck-transfer",
}


# --- token counting --------------------------------------------------------------------------------


class Tok:
    def __init__(self):
        self.cache = {}
        self.exact = False
        self.path = None
        try:
            import tokenizers

            paths = sorted(glob.glob(TOKENIZER_GLOB))
            if paths:
                self.tk = tokenizers.Tokenizer.from_file(paths[0])
                self.exact, self.path = True, paths[0]
        except ImportError:
            pass

    def n(self, text):
        if not text:
            return 0
        key = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).digest() if len(text) > 256 else text
        hit = self.cache.get(key)
        if hit is None:
            hit = len(self.tk.encode(text, add_special_tokens=False).ids) if self.exact else int(len(text) / 4.1)
            self.cache[key] = hit
        return hit


TOK = None


def texts_and_images(content):
    if isinstance(content, str):
        return content, 0
    if isinstance(content, list):
        parts, images = [], 0
        for p in content:
            if isinstance(p, dict) and p.get("type") == "image_url":
                images += 1
                parts.append("<|vision_start|><|vision_end|>")
            elif isinstance(p, dict):
                parts.append(p.get("text") or "")
        return "".join(parts), images
    return "", 0


def tool_call_text(calls):
    out = []
    for call in calls or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {"raw": fn.get("arguments") or ""}
        body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                       for k, v in (args.items() if isinstance(args, dict) else []))
        out.append(f"<tool_call>\n<function={fn.get('name', '')}>\n{body}</function>\n</tool_call>")
    return "\n".join(out)


TOOL_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>")


def tools_block(tools):
    if not tools:
        return ""
    return ("# Tools\n\nYou have access to the following functions:\n\n<tools>"
            + "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools) + "\n</tools>"
            + TOOL_INSTRUCTIONS)


def message_tokens(message, keep_reasoning):
    """(tokens, images) of one message in the Qwen3.6 template rendering."""
    role = message.get("role")
    text, images = texts_and_images(message.get("content"))
    n = 5 + TOK.n(text) + images * IMAGE_TOKENS  # <|im_start|>role\n ... <|im_end|>\n
    if role == "assistant":
        if keep_reasoning:
            n += 4 + TOK.n((message.get("reasoning_content") or "").strip())
        n += TOK.n(tool_call_text(message.get("tool_calls")))
    elif role == "tool":
        n += 4  # <tool_response> wrappers
    return n, images


def record_tokens(messages, tools, target):
    """Prompt tokens with every history reasoning rendered (as the Flash server rendered the teacher's
    requests: within 1% of the logged prompt_tokens; Ornith's template does the same), prompt tokens with
    history reasoning before the last user turn dropped (the Qwen3.6 template default), images and the
    target tokens (reasoning / rest) of one request."""
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=0)
    base = TOK.n(tools_block(tools)) + 6  # system header + generation prompt
    prompt = drop = base
    images = 0
    for i, m in enumerate(messages):
        n_keep, im = message_tokens(m, keep_reasoning=True)
        n_drop, _ = message_tokens(m, keep_reasoning=i > last_user)
        prompt += n_keep
        drop += n_drop
        images += im
    reasoning = TOK.n((target.get("reasoning") or "").strip()) + 4
    rest = TOK.n(target.get("content") or "") + TOK.n(tool_call_text(target.get("tool_calls"))) + 1
    return prompt, drop, images, reasoning, rest


# --- helpers ---------------------------------------------------------------------------------------


def pct(values, qs=(50, 90, 99)):
    if not values:
        return {}
    s = sorted(values)
    out = {f"p{q}": s[min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))] for q in qs}
    out.update(n=len(s), mean=round(statistics.fmean(s), 1), max=s[-1], sum=sum(s))
    return out


def buckets(values, edges=(8192, 16384, 32768, 65536, 131072)):
    out = collections.OrderedDict()
    for e in edges:
        out[f"<={e // 1024}K"] = sum(v <= e for v in values)
    out[f">{edges[-1] // 1024}K"] = sum(v > edges[-1] for v in values)
    return out


def share_top(counter, k=3):
    total = sum(counter.values()) or 1
    top = counter.most_common(k)
    return {"top": [(g, v, round(v / total, 3)) for g, v in top], "top_share": round(sum(v for _, v in top) / total, 3)}


def action_key(action):
    if action is None:
        return None
    if isinstance(action, dict):
        return (action.get("action"), action.get("x"), action.get("y"))
    return (str(action), None, None)


def final_frame(obs):
    frames = obs.get("frames") or []
    return tuple(frames[-1]) if frames else None


def border_only(a, b, margin=3):
    """True when frames a and b differ only within ``margin`` cells of the edge (a HUD/timer proxy)."""
    if a is None or b is None:
        return False
    for r, (ra, rb) in enumerate(zip(a, b)):
        if ra == rb:
            continue
        for c, (x, y) in enumerate(zip(ra, rb)):
            if x != y and margin <= r < 64 - margin and margin <= c < 64 - margin:
                return False
    return True


# --- run logs --------------------------------------------------------------------------------------


class GameLog:
    """One game's log: requests (with actions) and the per-action frame effects."""

    def __init__(self, path):
        records = seg.read_log(path)
        self.turns, self.obs, self.extras = seg.game_turns(records)
        self.requests = [r for t in self.turns for r in t["requests"]]
        self.by_call, self.by_reasoning = {}, {}
        for r in self.requests:
            for call in r["tool_calls"]:
                if call.get("id"):
                    self.by_call[call["id"]] = r
            self.by_reasoning.setdefault(r["reasoning"][:400], r)
        self.effect = {}  # step -> dict(action, noop, border_only, state_key, reset)
        prev = None
        for o in self.obs:
            frame = final_frame(o)
            if o.get("action") is not None and prev is not None:
                pframe = final_frame(prev)
                key = (hash(pframe), action_key(o["action"]))
                self.effect[o["step"]] = {
                    "action": o["action"].get("action"), "noop": frame == pframe,
                    "border_only": frame != pframe and border_only(pframe, frame), "key": key,
                    "reset": o["action"].get("action") == "RESET" or prev.get("state") == "GAME_OVER",
                    "game_over": o.get("state") == "GAME_OVER"}
            prev = o

    def find(self, message):
        for call in message.get("tool_calls") or []:
            if call.get("id") in self.by_call:
                return self.by_call[call["id"]]
        return self.by_reasoning.get((message.get("reasoning_content") or "").strip()[:400])


def trial_map():
    out = {}
    for trial in seg.find_trials(ROOT):
        out[seg.run_name(trial)] = Path(trial)
    return out


def python_error(results):
    for res in results:
        r = res.get("result") or {}
        if r.get("error") or (r.get("returncode") not in (0, None)):
            return str(r.get("error") or r.get("stdout") or "")[:160]
    return None


# --- segment-style sources -------------------------------------------------------------------------


def audit_segments(path, source, trials, examples):
    rows, seg_rows = [], []
    logs = {}
    target_hashes = collections.Counter()
    for line in open(path):
        s = json.loads(line)
        run, game_dir = s["run"], s["id"].split(":")[-2]
        trial = trials.get(run)
        log = None
        if trial is not None:
            lp = trial / "games" / game_dir / "log.jsonl"
            if lp.is_file():
                if lp not in logs:
                    logs.clear()  # one game at a time keeps memory flat (segments are grouped by run)
                    logs[lp] = GameLog(lp)
                log = logs[lp]
        records = segment_records(s)
        # user-group sizes: requests between two user messages (request cap check)
        first = int(s.get("target_start") or 1)  # full-context segments: earlier levels are context only
        groups, cur = [], 0
        for m in s["messages"][first:]:
            if m["role"] == "user":
                if cur:
                    groups.append(cur)
                cur = 0
            elif m["role"] == "assistant":
                cur += 1
        groups.append(cur)
        assistant_idx = [i for i, m in enumerate(s["messages"]) if m["role"] == "assistant" and i >= first]
        seen_keys, last_reset_step = set(), None
        if log is not None:
            for step in range(s["start_step"] + 1, s["clear_step"] + 1):
                e = log.effect.get(step)
                if e and e["reset"]:
                    last_reset_step = step
        seg_row = {"id": s["id"], "game": s["game"], "level": s["level"], "run": run, "variant": s.get("variant"),
                   "request_cap": s.get("request_cap"), "max_model_len": s.get("max_model_len"),
                   "user_source": s.get("user_source"), "caveats": s.get("caveats") or [],
                   "sequence_count": s.get("sequence_count", 1), "action_count": s["action_count"],
                   "baseline": s.get("baseline_actions"), "efficiency": s.get("efficiency"),
                   "level_score": s.get("level_score"), "starts_after_compaction": s.get("starts_after_compaction"),
                   "compactions": len(s.get("compaction_indices") or []), "first_prompt_tokens": s.get("first_prompt_tokens"),
                   "cap_hit_groups": sum(1 for g in groups if g >= 12) if s.get("request_cap") else 0,
                   "linked": log is not None, "last_reset_step": last_reset_step,
                   "actions": [json.dumps(a, sort_keys=True) for a in s["actions"]], "records": len(records)}
        seg_rows.append(seg_row)
        for rec, msg_index in zip(records, assistant_idx):
            msg = s["messages"][msg_index]
            prompt, drop, images, rtok, otok = record_tokens(rec["messages"], rec["tools"], rec["target"])
            t = rec["target"]
            th = hashlib.md5(((t["reasoning"] or "") + (t["content"] or "") + tool_call_text(t["tool_calls"])).encode()).hexdigest()
            target_hashes[th] += 1
            code = " ".join(json.loads(c["function"]["arguments"] or "{}").get("code", "") if c["function"]["arguments"].strip().startswith("{") else ""
                            for c in t["tool_calls"])
            text_blob = "\n".join([t["reasoning"] or "", t["content"] or "", code])
            ctx_text = "\n".join(texts_and_images(m.get("content"))[0] for m in rec["messages"])
            row = {"source": source, "id": rec["id"], "game": s["game"], "level": s["level"], "segment": s["id"],
                   "prompt": prompt, "prompt_drop": drop, "images": images, "reasoning": rtok, "rest": otok,
                   "target": rtok + otok, "total": prompt + rtok + otok, "total_drop": drop + rtok + otok, "hash": th, "has_call": bool(t["tool_calls"]),
                   "holdout_in_target": sorted(set(HOLDOUT_RE.findall(text_blob))),
                   "holdout_in_context": sorted(set(HOLDOUT_RE.findall(ctx_text))),
                   "other_games_in_target": sorted(set(ALL_GAME_RE.findall(text_blob)) - {s["game"]}),
                   "mentions_notes": bool(re.search(r"cross-game|other games?|notes? (from|about) (an)?other", text_blob, re.I)),
                   "variant": s.get("variant"), "caveats": s.get("caveats") or []}
            row["chunk"] = max([c for c in (s.get("compaction_indices") or []) if c <= msg_index], default=1)
            row["context_starts_after_compaction"] = bool(s.get("compaction_indices")) and any(
                c <= msg_index for c in s["compaction_indices"])
            req = log.find(msg) if log is not None else None
            if req is not None:
                rec_log = req["record"]
                usage = rec_log.get("usage") or {}
                row.update(linked=True, finish=rec_log.get("finish_reason"),
                           served_prompt=usage.get("prompt_tokens"), served_completion=usage.get("completion_tokens"),
                           served_reasoning=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                           request_step=rec_log.get("step"), py_error=python_error(req["python"]),
                           n_actions=len(req["actions"]))
                effects = [log.effect.get(st) for st in req["actions"]]
                effects = [e for e in effects if e]
                row["noop_actions"] = sum(e["noop"] for e in effects)
                row["border_only_actions"] = sum(e["border_only"] for e in effects)
                rep = 0
                for e in effects:
                    if e["key"] in seen_keys:
                        rep += 1
                    seen_keys.add(e["key"])
                row["repeat_actions"] = rep
                row["resets"] = sum(e["reset"] for e in effects)
                step = rec_log.get("step") or 0
                row["dead_end"] = last_reset_step is not None and (
                    (req["actions"] and max(req["actions"]) <= last_reset_step) or (not req["actions"] and step < last_reset_step))
                if row["finish"] == "length" and len(examples.setdefault("truncated", [])) < 3:
                    examples["truncated"].append({"id": rec["id"], "served_completion": row["served_completion"],
                                                  "reasoning_tail": (t["reasoning"] or "")[-160:]})
                if row["py_error"] and len(examples.setdefault("py_error", [])) < 3:
                    examples["py_error"].append({"id": rec["id"], "error": row["py_error"]})
            else:
                row.update(linked=False)
            rows.append(row)
    for r in rows:
        r["dup_target"] = target_hashes[r["hash"]] > 1
    return rows, seg_rows


def classify(row):
    """Mutually exclusive usefulness class of one request (for the wasted-token proxy)."""
    if not row.get("linked"):
        return "unlinked"
    if row.get("finish") == "length":
        return "truncated"
    if row.get("dead_end"):
        return "dead_end"
    n = row.get("n_actions") or 0
    if n == 0:
        return "error_no_action" if row.get("py_error") else "analysis_only"
    if row["noop_actions"] == n:
        return "all_noop"
    if row["repeat_actions"] == n:
        return "all_repeat"
    return "effective_action"


def near_duplicates(seg_rows, threshold=0.9):
    by_level = collections.defaultdict(list)
    for r in seg_rows:
        by_level[(r["game"], r["level"])].append(r)
    flagged, pairs = set(), 0
    for rows in by_level.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i]["actions"], rows[j]["actions"]
                if min(len(a), len(b)) / max(len(a), len(b), 1) < threshold:
                    continue
                if difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= threshold:
                    pairs += 1
                    flagged.add(rows[j]["id"])
    return {"threshold": threshold, "pairs": pairs, "segments_with_earlier_near_duplicate": len(flagged)}


def packed_tokens(rows):
    """Training tokens when each segment (split at compactions) is one sequence with loss on every kept
    assistant turn: valid when history reasoning is rendered (Ornith's template; Qwen3.6 with
    preserve_thinking), because each request's prompt is then a prefix of the next one."""
    longest = collections.Counter()
    for r in rows:
        key = (r["segment"], r["chunk"])
        longest[key] = max(longest[key], r["total"])
    return {"sequences": len(longest), "tokens": sum(longest.values()), "buckets": buckets(list(longest.values()))}


def summarize_source(rows, seg_rows, name):
    out = {"records": len(rows), "segments": len(seg_rows)}
    for key in ("prompt", "prompt_drop", "target", "reasoning", "rest", "total", "images"):
        out[key] = pct([r[key] for r in rows])
    out["total_buckets"] = buckets([r["total"] for r in rows])
    out["train_tokens_per_epoch"] = sum(r["total"] for r in rows)
    out["train_tokens_per_epoch_drop_history_reasoning"] = sum(r["total_drop"] for r in rows)
    out["packed"] = packed_tokens(rows)
    out["target_tokens"] = sum(r["target"] for r in rows)
    games = collections.Counter()
    games_records = collections.Counter()
    for r in rows:
        games[r["game"]] += r["target"]
        games_records[r["game"]] += 1
    out["per_game_target_tokens"] = dict(games.most_common())
    out["per_game_records"] = dict(games_records.most_common())
    out["per_game_segments"] = dict(collections.Counter(r["game"] for r in seg_rows).most_common())
    out["concentration_target_tokens"] = share_top(games)
    out["concentration_segments"] = share_top(collections.Counter(r["game"] for r in seg_rows))
    lv = collections.Counter((r["game"], r["level"]) for r in seg_rows)
    out["per_level_segments_top"] = [(f"{g} L{level}", n) for (g, level), n in lv.most_common(10)]
    out["levels_covered"] = len(lv)
    out["level_hist"] = dict(sorted(collections.Counter(r["level"] for r in seg_rows).items()))
    # quality
    linked = [r for r in rows if r.get("linked")]
    out["linked_records"] = len(linked)
    out["finish_reason"] = dict(collections.Counter(r.get("finish") for r in linked))
    out["truncated_target_tokens"] = sum(r["target"] for r in linked if r.get("finish") == "length")
    out["segments_with_truncated_turn"] = len({r["segment"] for r in linked if r.get("finish") == "length"})
    out["py_error_records"] = sum(1 for r in linked if r.get("py_error"))
    out["no_tool_call_records"] = sum(1 for r in rows if not r["has_call"])
    out["no_action_records"] = sum(1 for r in linked if not r.get("n_actions"))
    classes = collections.Counter()
    class_tokens = collections.Counter()
    for r in rows:
        c = classify(r)
        classes[c] += 1
        class_tokens[c] += r["target"]
    tot = sum(class_tokens.values()) or 1
    out["request_classes"] = {c: {"records": classes[c], "target_tokens": class_tokens[c],
                                  "token_share": round(class_tokens[c] / tot, 3)} for c in classes}
    acts = sum(r.get("n_actions") or 0 for r in linked)
    out["actions_linked"] = acts
    out["noop_action_share"] = round(sum(r.get("noop_actions", 0) for r in linked) / max(acts, 1), 3)
    out["border_only_action_share"] = round(sum(r.get("border_only_actions", 0) for r in linked) / max(acts, 1), 3)
    out["repeat_action_share"] = round(sum(r.get("repeat_actions", 0) for r in linked) / max(acts, 1), 3)
    out["dup_target_records"] = sum(1 for r in rows if r["dup_target"])
    out["segments_with_reset"] = sum(1 for s in seg_rows if s["last_reset_step"] is not None)
    out["near_duplicates"] = near_duplicates(seg_rows)
    out["sequence_copies"] = {"segments_with_copies": sum(s["sequence_count"] > 1 for s in seg_rows),
                              "copies_total": sum(s["sequence_count"] for s in seg_rows)}
    eff = [s["action_count"] / s["baseline"] for s in seg_rows if s.get("baseline")]
    out["actions_over_baseline"] = pct([round(e, 2) for e in eff])
    out["segments_at_or_below_baseline"] = sum(e <= 1.0 for e in eff)
    out["segments_over_2x_baseline"] = sum(e > 2.0 for e in eff)
    # heterogeneity
    out["variants"] = dict(collections.Counter(s["variant"] for s in seg_rows).most_common())
    out["request_cap"] = dict(collections.Counter(str(s["request_cap"]) for s in seg_rows))
    out["cap_hit_groups"] = sum(s["cap_hit_groups"] for s in seg_rows)
    out["max_model_len"] = dict(collections.Counter(s["max_model_len"] for s in seg_rows))
    out["user_source"] = dict(collections.Counter(s["user_source"] for s in seg_rows))
    out["caveats"] = dict(collections.Counter(c for s in seg_rows for c in s["caveats"]))
    out["starts_after_compaction"] = sum(bool(s["starts_after_compaction"]) for s in seg_rows)
    out["segments_with_compaction"] = sum(s["compactions"] > 0 for s in seg_rows)
    out["records_context_after_compaction"] = sum(r["context_starts_after_compaction"] for r in rows)
    # context mismatch: reconstructed vs served prompt
    ratios = [r["prompt"] / r["served_prompt"] for r in linked if r.get("served_prompt")]
    out["reconstructed_over_served_prompt"] = {k: (round(v, 3) if isinstance(v, float) else v)
                                               for k, v in pct([round(x, 3) for x in ratios], (10, 50, 90)).items()
                                               if k != "sum"}
    out["records_reconstructed_below_half_served"] = sum(x < 0.5 for x in ratios)
    out["served_prompt"] = {k: v for k, v in pct([r["served_prompt"] for r in linked if r.get("served_prompt")]).items() if k != "sum"}
    # leakage
    out["holdout_ids_in_targets"] = sum(bool(r["holdout_in_target"]) for r in rows)
    out["holdout_ids_in_context"] = sum(bool(r["holdout_in_context"]) for r in rows)
    out["holdout_id_examples"] = sorted({g for r in rows for g in r["holdout_in_target"] + r["holdout_in_context"]})
    out["other_game_ids_in_targets"] = sum(bool(r["other_games_in_target"]) for r in rows)
    notes_rows = [r for r in rows if any("cross_game_notes" in c for c in r["caveats"])]
    out["cross_game_notes_records"] = len(notes_rows)
    out["cross_game_notes_records_mentioning_notes"] = sum(r["mentions_notes"] for r in notes_rows)
    out["records_mentioning_notes_any"] = sum(r["mentions_notes"] for r in rows)
    return out


# --- proposed filters and per-level cap ----------------------------------------------------------------

DROP_CLASSES = ("truncated", "dead_end", "error_no_action", "all_noop", "all_repeat")
DROP_CAVEATS = ("carried_context_omitted:cross_game_notes", "carried_context_omitted:memory_block",
                "carried_context_omitted:game_snapshot")
DROP_VARIANTS = ("fixed-2x-procedure",)  # its user prompts carry an instruction block the base harness lacks


def filtered(rows, seg_rows, cap=None, holdout_games=()):
    """Records kept by the proposed rules: drop the request classes above, segments whose targets saw
    omitted variant context or a non-base prompt, held-out generated mechanics, and keep at most ``cap``
    segments per (game, level) (highest level score, then fewest target tokens)."""
    seg_ok = {s["id"]: s for s in seg_rows
              if not any(c in DROP_CAVEATS for c in s["caveats"]) and s["variant"] not in DROP_VARIANTS
              and s["game"] not in holdout_games}
    tokens = collections.Counter()
    for r in rows:
        tokens[r["segment"]] += r["target"]
    if cap:
        by_level = collections.defaultdict(list)
        for s in seg_ok.values():
            by_level[(s["game"], s["level"])].append(s)
        keep = set()
        for group in by_level.values():
            group.sort(key=lambda s: (-(s["level_score"] or 0), tokens[s["id"]]))
            keep.update(s["id"] for s in group[:cap])
        seg_ok = {k: v for k, v in seg_ok.items() if k in keep}
    packed = packed_tokens([r for r in rows if r["segment"] in seg_ok])
    kept = [r for r in rows if r["segment"] in seg_ok and classify(r) not in DROP_CLASSES]
    games = collections.Counter()
    for r in kept:
        games[r["game"]] += r["target"]
    return {"segments": len(seg_ok), "records": len(kept), "target_tokens": sum(r["target"] for r in kept),
            "reasoning_tokens": sum(r["reasoning"] for r in kept),
            "train_tokens": sum(r["total"] for r in kept), "packed_train_tokens": packed["tokens"],
            "packed_sequences": packed["sequences"], "games": len(games),
            "concentration_target_tokens": share_top(games),
            "total_buckets": buckets([r["total"] for r in kept])}


# --- system prompt / harness version ----------------------------------------------------------------


def norm_system(text):
    # the chat template trims the system text, so a trailing newline is not a difference
    return hashlib.md5(re.sub(r"\d+", "#", (text or "").strip()).encode()).hexdigest()[:10]


def harness_signatures(paths_sources):
    """Distinct (digit-normalised) system prompts and tool schemas per source."""
    out = {}
    for name, path, kind in paths_sources:
        sys_c, tools_c = collections.Counter(), collections.Counter()
        for line in open(path):
            r = json.loads(line)
            if kind == "aux":
                continue
            sys_c[norm_system(r["messages"][0]["content"])] += 1
            tools_c[hashlib.md5(json.dumps(r.get("tools"), sort_keys=True).encode()).hexdigest()[:10]] += 1
        out[name] = {"system_prompts": dict(sys_c), "tool_schemas": dict(tools_c)}
    return out


def template_lines(path, base_variant="reclaim-fixed-2x", min_segments=2):
    """Harness instruction lines of the user prompts (no digits, >= 40 characters, in >= ``min_segments``
    segments of a variant) that the base variant's segments never contain."""
    per_variant = collections.defaultdict(lambda: collections.defaultdict(set))
    for line in open(path):
        s = json.loads(line)
        v = s.get("variant") or "None"
        for m in s["messages"]:
            if m["role"] != "user":
                continue
            text, _ = texts_and_images(m["content"])
            for ln in text.splitlines():
                ln = ln.strip()
                if len(ln) >= 40 and not re.search(r"\d", ln):
                    per_variant[v][ln[:120]].add(s["id"])
    base = set(per_variant.get(base_variant, {}))
    out = {}
    for v, lines in per_variant.items():
        extra = sorted(((ln, len(ids)) for ln, ids in lines.items() if ln not in base and len(ids) >= min_segments),
                       key=lambda x: -x[1])
        if extra:
            out[v] = {"lines_not_in_base": len(extra), "examples": extra[:4]}
    return out


# --- solver and aux ---------------------------------------------------------------------------------


def audit_solver(path):
    rows = []
    reasoning_forms = collections.Counter()
    leaks = 0
    for line in open(path):
        r = json.loads(line)
        prompt, drop, images, rtok, otok = record_tokens(r["messages"], r["tools"], r["target"])
        reasoning_forms[re.sub(r"\d+|\(.*\)", "#", r["target"]["reasoning"] or "")[:60]] += 1
        ctx = "\n".join(texts_and_images(m.get("content"))[0] for m in r["messages"])
        leaks += bool(HOLDOUT_RE.search(ctx + json.dumps(r["target"])))
        rows.append({"game": r["game_id"].split("-")[0], "level": r["level"], "prompt": prompt, "prompt_drop": drop, "images": images,
                     "reasoning": rtok, "rest": otok, "target": rtok + otok, "total": prompt + rtok + otok,
                     "purpose": r.get("purpose"), "skin": int(r["game_id"][1:4]) > 100})
    out = {"records": len(rows)}
    for key in ("prompt", "prompt_drop", "target", "reasoning", "total", "images"):
        out[key] = pct([x[key] for x in rows])
    out["total_buckets"] = buckets([x["total"] for x in rows])
    out["train_tokens_per_epoch"] = sum(x["total"] for x in rows)
    out["target_tokens"] = sum(x["target"] for x in rows)
    out["reasoning_templates"] = len(reasoning_forms)
    out["reasoning_template_examples"] = reasoning_forms.most_common(3)
    out["purpose"] = dict(collections.Counter(x["purpose"] for x in rows))
    out["games"] = len({x["game"] for x in rows})
    out["skin_records"] = sum(x["skin"] for x in rows)
    out["holdout_leaks"] = leaks
    return out


def audit_aux(path):
    rows, leaks = [], 0
    for line in open(path):
        r = json.loads(line)
        leaks += bool(HOLDOUT_RE.search(r["prompt"] + str(r["answer"])))
        p, a = TOK.n(r["prompt"]), TOK.n(str(r["answer"]))
        rows.append({"family": r["family"], "game": prefix(r["game"]), "prompt": p, "answer": a})
    out = {"tasks": len(rows), "holdout_leaks": leaks,
           "games": sorted({x["game"] for x in rows}),
           "generated_games": sum(1 for x in rows if re.match(r"q\d{3}", x["game"]))}
    for fam in sorted({x["family"] for x in rows}):
        sub = [x for x in rows if x["family"] == fam]
        out[fam] = {"tasks": len(sub), "prompt": pct([x["prompt"] for x in sub]), "answer": pct([x["answer"] for x in sub])}
    out["tokens_total"] = sum(x["prompt"] + x["answer"] for x in rows)
    out["answer_tokens_total"] = sum(x["answer"] for x in rows)
    return out


# --- baseline runs ---------------------------------------------------------------------------------


def audit_run(trial):
    trial = ROOT / trial
    if not (trial / "games").is_dir():
        return None
    card = {}
    sc = trial / "scorecard.json"
    if sc.is_file():
        for env in json.loads(sc.read_text()).get("environments", []):
            card[env["id"]] = env
    groups = collections.defaultdict(lambda: collections.Counter())
    per_game = {}
    for game_dir in sorted((trial / "games").iterdir()):
        lp = game_dir / "log.jsonl"
        if not lp.is_file():
            continue
        gid = game_dir.name
        group = "holdout" if prefix(gid) in HOLDOUT else "dev" if prefix(gid) in DEV else "other"
        log = GameLog(lp)
        c = collections.Counter()
        for r in log.requests:
            rec = r["record"]
            u = rec.get("usage") or {}
            c["replies"] += 1
            c["completion"] += u.get("completion_tokens") or 0
            c["length"] += rec.get("finish_reason") == "length"
            c["no_call"] += not r["tool_calls"]
            c["py_error"] += bool(python_error(r["python"]))
            c["with_actions"] += bool(r["actions"])
        for e in log.effect.values():
            c["actions"] += 1
            c["noop"] += e["noop"]
        env = card.get(gid) or {}
        c["clears"] = env.get("levels_completed") or 0
        c["score_x1000"] = int(1000 * (env.get("score") or 0))
        c["games"] = 1
        per_game[gid] = {"clears": c["clears"], "score": round(env.get("score") or 0, 2), "actions": c["actions"],
                         "replies": c["replies"], "completion_tokens": c["completion"]}
        groups[group].update(c)
    out = {}
    for g, c in groups.items():
        out[g] = {"games": c["games"], "clears": c["clears"], "mean_game_score": round(c["score_x1000"] / 1000 / max(c["games"], 1), 2),
                  "completion_tokens": c["completion"], "clears_per_10k_tokens": round(1e4 * c["clears"] / max(c["completion"], 1), 3),
                  "replies": c["replies"], "tokens_per_reply": round(c["completion"] / max(c["replies"], 1)),
                  "actions": c["actions"], "tokens_per_action": round(c["completion"] / max(c["actions"], 1)),
                  "length_share": round(c["length"] / max(c["replies"], 1), 3),
                  "no_tool_call_share": round(c["no_call"] / max(c["replies"], 1), 3),
                  "py_error_share": round(c["py_error"] / max(c["replies"], 1), 3),
                  "replies_with_actions_share": round(c["with_actions"] / max(c["replies"], 1), 3),
                  "noop_action_share": round(c["noop"] / max(c["actions"], 1), 3)}
    out["per_game"] = per_game
    return out


# --- main ------------------------------------------------------------------------------------------


def first_image_size(path):
    for line in open(path):
        for m in json.loads(line)["messages"]:
            if isinstance(m.get("content"), list):
                for p in m["content"]:
                    if isinstance(p, dict) and p.get("type") == "image_url":
                        raw = base64.b64decode(p["image_url"]["url"].split(",", 1)[1][:64] + "==")
                        return struct.unpack(">II", raw[16:24])
    return None


def main():
    global TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/posttrain/data-audit.json"))
    ap.add_argument("--skip-runs", action="store_true", help="skip the baseline-run summaries")
    args = ap.parse_args()
    TOK = Tok()
    trials = trial_map()
    examples = {}
    result = {"tokenizer": {"exact": TOK.exact, "path": TOK.path,
                            "note": "Qwen3.6-35B-A3B tokenizer, Qwen3.6 chat-template rendering" if TOK.exact
                            else "estimate: characters / 4.1"},
              "image_size_px": first_image_size(DATA / "segments/segments.jsonl"), "image_tokens": IMAGE_TOKENS}
    print("segments (dev)...", flush=True)
    dev_rows, dev_segs = audit_segments(DATA / "segments/segments.jsonl", "segment", trials, examples)
    result["segment"] = summarize_source(dev_rows, dev_segs, "segment")
    print("teacher (generated)...", flush=True)
    t_rows, t_segs = audit_segments(DATA / "teacher/segments.jsonl", "teacher", trials, examples)
    result["teacher"] = summarize_source(t_rows, t_segs, "teacher")
    # teacher: first level vs later levels (discovery vs execution)
    first = [r for r in t_rows if r["level"] == 1]
    later = [r for r in t_rows if r["level"] > 1]
    result["teacher"]["level1_vs_later_target_tokens"] = {"level1": sum(r["target"] for r in first),
                                                         "later": sum(r["target"] for r in later)}
    result["filtered"] = {
        "segment_no_cap": filtered(dev_rows, dev_segs),
        "segment_cap8": filtered(dev_rows, dev_segs, cap=8),
        "segment_cap4": filtered(dev_rows, dev_segs, cap=4),
        "teacher_all": filtered(t_rows, t_segs),
        "teacher_minus_eval_mechanics": filtered(t_rows, t_segs, holdout_games=EVAL_MECHANICS),
    }
    first = [r for r in dev_rows if r["level"] == 1]
    result["segment"]["level1_target_tokens"] = sum(r["target"] for r in first)
    # efficient portion: tokens in the final attempt (after the last reset) and effective-action requests
    print("solver / aux...", flush=True)
    result["solver"] = audit_solver(DATA / "solver/records.jsonl")
    result["aux"] = audit_aux(DATA / "aux/tasks.jsonl")
    result["harness"] = harness_signatures([
        ("segment", DATA / "segments/segments.jsonl", "seg"), ("teacher", DATA / "teacher/segments.jsonl", "seg"),
        ("solver", DATA / "solver/records.jsonl", "rec")])
    result["variant_prompt_lines"] = template_lines(DATA / "segments/segments.jsonl")
    result["examples"] = examples
    # per-variant quality for the dev segments
    per_var = collections.defaultdict(collections.Counter)
    for r in dev_rows:
        v = r["variant"] or "None"
        per_var[v]["records"] += 1
        per_var[v]["target"] += r["target"]
        per_var[v]["truncated"] += r.get("finish") == "length"
    result["segment"]["per_variant"] = {v: dict(c) for v, c in sorted(per_var.items(), key=lambda kv: -kv[1]["records"])}
    if not args.skip_runs:
        print("baseline runs...", flush=True)
        result["runs"] = {name: audit_run(path) for name, path in BASELINE_RUNS.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
