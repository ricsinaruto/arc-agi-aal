"""The pilot SFT mixture from the fixed post-training sources (data audit, sections 4 and 5).

Sources (training games only, ``split.check_training_games``):

- ``dev``: Flash level-clear segments on the 14 dev games (``data/posttrain/segments``), rebuilt with
  the full context the teacher saw (``posttrain.segments``);
- ``teacher``: Flash level-clear segments on the generated training mechanics and skins
  (``data/posttrain/teacher``; every run in ``segments.TEACHER_RUNS``);
- ``aux``: engine-verified predict / reach tasks (``data/posttrain/aux`` and ``aux-generated``). Their
  answers carry no model reasoning: every aux item is flagged ``no_reasoning``;
- ``solver``: BFS-solver requests through the harness (``data/posttrain/solver``), off by default
  (``Mixture.solver``); their template reasoning gets no loss (the loss covers the tool call only).

The evaluation split holds the held-out generated mechanics (``split.GENERATED_HOLDOUT``): their teacher
segments (loss checks), their solver requests (the BFS-optimal next action) and their aux tasks.

Filters (``filter_segments``, audit section 4):

- drop segments from runs that injected context the records do not carry: cross-game notes, game
  snapshot, memory block (``caveats`` ``carried_context_omitted:*`` or ``context_injections``) and the
  stuck procedure's prompt block (study key ``stuck_procedure``);
- drop near duplicates: same game and level and action-sequence similarity >= ``near_duplicate``;
- keep at most ``cap_per_level`` segments per (game, level).
  Both keep the best segments first: fewest actions, then fewest completion tokens, then id;
- targets cut by the completion cap (``finish_reason = length``) get no loss (they stay as context in
  packed sequences, as the teacher saw them); python-error targets are kept (realistic) and counted.

Mixture (``Mixture``, by target = loss tokens): every teacher segment, ``teacher_epochs`` times; dev at
most ``dev_max_share`` of the total and aux at most ``aux_max_share`` (solver ``solver_max_share`` when
on). Within a budget, segments are taken level by level in rounds (the best segment of every (game,
level), then the second best, ...) and aux tasks game by game, so a smaller budget lowers the depth per
level instead of dropping levels.

Rendering: the Ornith-1.5 chat template (``TEMPLATE_PATH``; it renders every earlier reasoning block, as
Flash's server did), with tool-call arguments as objects (as vLLM passes them). Two formats:

- ``*_packed.jsonl``: one sequence per segment chunk: the context of the chunk's last request and
  every target, with loss on each target. A chunk ends where a request's context is not the previous
  request plus its reply (a compaction). Valid because the template keeps earlier reasoning, so each
  request's prompt is a prefix of the sequence (checked for every target);
- ``*_turns.jsonl``: one record per target: the exact request the teacher's server received (with the
  generation prompt) followed by the target.

Each line: ``id``, ``source``, ``split``, ``format``, ``game_id``, ``level``, ``segment``, ``repeat``,
``weight``, ``no_reasoning``, ``text`` (rendered; one ``<|vision_start|><|image_pad|><|vision_end|>``
per board image), ``images`` (data URLs in order), ``loss_spans`` (character ``[start, end)`` pairs:
the reasoning after the generation prompt's ``<think>\\n``, the content, the tool calls and
``<|im_end|>``), ``tokens`` and ``loss_tokens``.

Tokens are counted with the local Qwen3.6-35B-A3B tokenizer (Ornith-1.5 is assumed to share it) when
``tokenizers`` is importable (``uv run --with tokenizers``), each image as ``IMAGE_TOKENS`` pad tokens;
otherwise characters / 4.1 (``stats["tokenizer"]["exact"]`` says which). Every rebuilt prompt is compared
with the ``prompt_tokens`` the teacher's server logged for that request (``prompt_check``).
"""

import collections
import copy
import difflib
import glob
import hashlib
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .segments import TEACHER_RUNS
from .sft_records import segment_contexts, segment_records
from .split import check_eval_games, check_training_games, prefix

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "posttrain"
TEMPLATE_PATH = REPO / "tests" / "fixtures" / "ornith15-35b-chat-template.jinja"
TOKENIZER_GLOB = str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/*/tokenizer.json")
IMAGE_TOKENS = 64  # a 256x256 board image: patch 16, merge 2
IMAGE_PAD = "<|image_pad|>"
IM_START = "<|im_start|>"
CHARS_PER_TOKEN = 4.1
RATIO_OUTLIER = 0.05
LIGHT_KEYS = ("id", "source", "segment", "game_id", "level", "tokens", "loss_tokens", "repeat")  # |rebuilt / served prompt tokens - 1| above this is reported
LENGTH_EDGES = (8192, 16384, 32768, 65536, 131072)


@dataclass
class Sources:
    """Input files per source. The teacher file holds every run of ``segments.TEACHER_RUNS``."""

    dev: tuple = (DATA / "segments" / "segments.jsonl",)
    teacher: tuple = (DATA / "teacher" / "segments.jsonl",)
    aux: tuple = (DATA / "aux" / "tasks.jsonl", DATA / "aux-generated" / "tasks.jsonl")
    solver: tuple = (DATA / "solver" / "records.jsonl",)
    eval_teacher: tuple = (DATA / "teacher" / "eval_segments.jsonl",)
    eval_solver: tuple = (DATA / "solver" / "eval_records.jsonl",)
    eval_aux: tuple = (DATA / "aux-eval" / "tasks.jsonl",)


@dataclass
class Mixture:
    """Per-source budgets (shares of the mixture's target tokens) and filters."""

    dev_max_share: float = 0.70
    teacher_epochs: int = 2
    aux_max_share: float = 0.10
    aux_families: tuple = ("predict", "reach")  # transition tasks have no reference answer
    solver: bool = False
    solver_max_share: float = 0.02
    cap_per_level: int = 8
    near_duplicate: float = 0.9
    weights: dict = field(default_factory=lambda: {"dev": 1.0, "teacher": 1.0, "aux": 1.0, "solver": 1.0})


# --- rendering -------------------------------------------------------------------------------------


def _raise(message):
    raise ValueError(message)


def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # transformers' chat-template filter (jinja's own escapes <, >, & and ').
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def template_message(message):
    """A copy of ``message`` as the server hands it to the template: tool-call arguments parsed."""
    out = dict(message)
    if message.get("tool_calls"):
        calls = []
        for call in message["tool_calls"]:
            call = copy.deepcopy(call)
            function = call.get("function") or {}
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"] or "{}")
            calls.append(call)
        out["tool_calls"] = calls
    return out


def assistant_message(target):
    message = {"role": "assistant", "content": target.get("content") or "",
               "reasoning_content": target.get("reasoning") or ""}
    if target.get("tool_calls"):
        message["tool_calls"] = target["tool_calls"]
    return message


class ChatTemplate:
    """The Ornith-1.5 chat template, rendered as transformers (and vLLM) render it."""

    def __init__(self, path=TEMPLATE_PATH):
        import jinja2
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                           extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        self.path = Path(path)
        self.template = env.from_string(self.path.read_text())

    def render(self, messages, tools=None, *, generation_prompt=False):
        return self.template.render(messages=[template_message(m) for m in messages], tools=tools or None,
                                    add_generation_prompt=generation_prompt, add_vision_id=False)

    def target_text(self, target):
        """The rendered reply after the generation prompt, up to and including ``<|im_end|>``."""
        probe = [{"role": "user", "content": "x"}]
        head = self.render(probe, generation_prompt=True)
        full = self.render([*probe, assistant_message(target)])
        if not full.startswith(head):
            raise ValueError("the generation prompt is not a prefix of the rendered reply")
        return full[len(head):].removesuffix("\n")


def image_urls(messages):
    return [part["image_url"]["url"] for m in messages if isinstance(m.get("content"), list)
            for part in m["content"] if isinstance(part, dict) and part.get("type") == "image_url"]


# --- token counting --------------------------------------------------------------------------------


class TokenCounter:
    """Tokens of rendered text. The text is split at ``<|im_start|>`` (a special token, so the pieces
    tokenize independently) and pieces are cached; each image placeholder counts ``IMAGE_TOKENS``."""

    def __init__(self, path=None):
        self.cache, self.exact, self.path, self.tokenizer = {}, False, None, None
        try:
            import tokenizers
        except ImportError:
            return
        paths = [str(path)] if path else sorted(glob.glob(TOKENIZER_GLOB))
        if paths and Path(paths[0]).is_file():
            self.tokenizer = tokenizers.Tokenizer.from_file(paths[0])
            self.exact, self.path = True, paths[0]

    def piece(self, text):
        if not text:
            return 0
        key = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).digest() if len(text) > 128 else text
        hit = self.cache.get(key)
        if hit is None:
            hit = (len(self.tokenizer.encode(text, add_special_tokens=False).ids) if self.exact
                   else int(len(text) / CHARS_PER_TOKEN))
            self.cache[key] = hit
        return hit

    def count(self, text):
        pieces = text.split(IM_START)
        return len(pieces) - 1 + sum(self.piece(p) for p in pieces) + text.count(IMAGE_PAD) * (IMAGE_TOKENS - 1)


# --- helpers ---------------------------------------------------------------------------------------


def read_jsonl(paths):
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def pct(values, qs=(10, 50, 90)):
    if not values:
        return {"n": 0}
    s = sorted(values)
    out = {f"p{q}": round(s[min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))], 4) for q in qs}
    out.update(n=len(s), mean=round(statistics.fmean(s), 4), min=round(s[0], 4), max=round(s[-1], 4))
    return out


def buckets(values, edges=LENGTH_EDGES):
    out = {f"<={e // 1024}K": sum(v <= e for v in values) for e in edges}
    out[f">{edges[-1] // 1024}K"] = sum(v > edges[-1] for v in values)
    return out


def share_top(counter, k=3):
    total = sum(counter.values()) or 1
    top = counter.most_common(k)
    return {"top": [[g, v, round(v / total, 3)] for g, v in top], "top_share": round(sum(v for _, v in top) / total, 3)}


def stable_key(text):
    return hashlib.sha1(str(text).encode()).hexdigest()


# --- filters ---------------------------------------------------------------------------------------


def injected_context(segment):
    """Names of the context the teacher saw but the record does not carry (or a non-base prompt)."""
    names = {c.split(":", 1)[1] for c in segment.get("caveats") or [] if c.startswith("carried_context_omitted:")}
    names |= set(segment.get("context_injections") or [])
    return sorted(names)


def rank_key(segment):
    """Best first: fewest actions, then fewest completion tokens, then id."""
    return (segment["action_count"], segment.get("completion_tokens") or 0, segment["id"])


def _action_keys(segment):
    return [json.dumps(a, sort_keys=True) for a in segment["actions"]]


def near_duplicate(a, b, threshold):
    if min(len(a), len(b)) / max(len(a), len(b), 1) < threshold:
        return False
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= threshold


def filter_segments(segments, *, cap_per_level=8, threshold=0.9):
    """``(kept, report)``: segments without injected context, near duplicates removed and at most
    ``cap_per_level`` per (game, level), best first (``rank_key``). Deterministic."""
    report = {"input": len(segments), "injected_context": collections.Counter(), "injected_context_segments": 0,
              "near_duplicates": 0, "over_level_cap": 0}
    clean = []
    for segment in segments:
        names = injected_context(segment)
        if names:
            report["injected_context_segments"] += 1
            report["injected_context"].update(names)
        else:
            clean.append(segment)
    groups = collections.defaultdict(list)
    for segment in clean:
        groups[(segment["game"], segment["level"])].append(segment)
    kept = []
    for key in sorted(groups):
        chosen = []
        for segment in sorted(groups[key], key=rank_key):
            actions = _action_keys(segment)
            if any(near_duplicate(actions, _action_keys(other), threshold) for other in chosen):
                report["near_duplicates"] += 1
                continue
            chosen.append(segment)
        report["over_level_cap"] += max(0, len(chosen) - cap_per_level)
        kept += chosen[:cap_per_level]
    report["injected_context"] = dict(sorted(report["injected_context"].items()))
    report["kept"] = len(kept)
    return kept, report


# --- items -----------------------------------------------------------------------------------------


def packed_chunks(contexts):
    """Group targets into packed sequences: ``contexts`` is ``[(target index, context indices)]`` in
    request order; a target joins the current sequence when its context extends the previous
    target's context plus that target (so every request is a prefix of the sequence), else it starts
    a new one (a compaction). Returns lists of positions into ``contexts``."""
    chunks, previous = [], None
    for n, (index, seen) in enumerate(contexts):
        if previous is not None and seen[:len(previous)] == previous:
            chunks[-1].append(n)
        else:
            chunks.append([n])
        previous = [*seen, index]
    return chunks


def segment_items(segment, source, template, counter, *, split="train"):
    """Rendered packed chunks and per-turn items of one segment, and per-target check rows."""
    records = segment_records(segment, split=split)
    contexts = segment_contexts(segment)
    old_contexts = segment_contexts(segment, context="segment")
    if len(records) != len(contexts):
        raise ValueError(f"{segment['id']}: {len(records)} records for {len(contexts)} targets")
    messages, tools = segment["messages"], segment.get("tools")
    base = {"source": source, "split": split, "game_id": segment["game_id"], "level": segment["level"],
            "segment": segment["id"], "no_reasoning": False}
    packed, turns, checks = [], [], []
    for chunk in packed_chunks(contexts):
        last_index, last_seen = contexts[chunk[-1]]
        sequence = [messages[p] for p in [*last_seen, last_index]]
        text = template.render(sequence, tools)
        spans, loss_tokens = [], 0
        for n in chunk:
            record = records[n]
            prompt = template.render(record["messages"], tools, generation_prompt=True)
            target = template.target_text(record["target"])
            if not text.startswith(prompt) or text[len(prompt):len(prompt) + len(target)] != target:
                raise ValueError(f"{record['id']}: the request is not a prefix of its packed sequence")
            prompt_tokens = counter.count(prompt)
            turn_text = prompt + target
            tokens = counter.count(turn_text)
            truncated = record.get("finish_reason") == "length"
            row = {"id": record["id"], "source": source, "game": prefix(segment["game_id"]),
                   "level": segment["level"], "prompt_tokens": prompt_tokens,
                   "served_prompt_tokens": record.get("served_prompt_tokens"),
                   "target_tokens": tokens - prompt_tokens, "truncated": truncated,
                   "python_error": bool(record.get("python_error")), "segment": segment["id"]}
            old_seen = old_contexts[n][1]
            row["segment_only_prompt_tokens"] = prompt_tokens if old_seen == contexts[n][1] else counter.count(
                template.render([messages[p] for p in old_seen], tools, generation_prompt=True))
            checks.append(row)
            if truncated:
                continue
            spans.append([len(prompt), len(prompt) + len(target)])
            loss_tokens += row["target_tokens"]
            turns.append({**base, "id": record["id"], "format": "turn", "text": turn_text,
                          "images": image_urls(record["messages"]), "loss_spans": [[len(prompt), len(turn_text)]],
                          "tokens": tokens, "loss_tokens": row["target_tokens"],
                          "served_prompt_tokens": record.get("served_prompt_tokens"),
                          "python_error": row["python_error"]})
        if spans:
            packed.append({**base, "id": f"{segment['id']}:C{chunk[0] + 1}", "format": "packed", "text": text,
                           "images": image_urls(sequence), "loss_spans": spans, "tokens": counter.count(text),
                           "loss_tokens": loss_tokens, "targets": len(spans)})
    return packed, turns, checks


def single_turn_item(item_id, source, game_id, level, messages, tools, target, template, counter, *,
                     split="train", loss="all", **extra):
    """One single-request item (aux, solver): the same text in both formats."""
    prompt = template.render(messages, tools, generation_prompt=True)
    target_text = template.target_text(target)
    text = prompt + target_text
    start = len(prompt)
    if loss == "tool_call":  # the loss covers the tool call(s) and the end of turn only
        at = target_text.find("<tool_call>")
        start = len(prompt) + (at if at >= 0 else 0)
    tokens = counter.count(text)
    return {"id": item_id, "source": source, "split": split, "game_id": game_id, "level": level,
            "segment": None, "no_reasoning": True, "text": text, "images": image_urls(messages),
            "loss_spans": [[start, len(text)]], "tokens": tokens,
            "loss_tokens": tokens - counter.count(text[:start]), **extra}


def aux_item(task, template, counter, *, split="train"):
    messages = [{"role": "user", "content": task["prompt"]}]
    return single_turn_item(task["id"], "aux", task["game"], task["level"], messages, None,
                            {"reasoning": "", "content": task["answer"], "tool_calls": []}, template, counter,
                            split=split, family=task["family"])


def solver_item(record, template, counter, *, split="train"):
    return single_turn_item(record["id"], "solver", record["game_id"], record["level"], record["messages"],
                            record["tools"], record["target"], template, counter, split=split, loss="tool_call",
                            purpose=record.get("purpose"))


# --- selection -------------------------------------------------------------------------------------


def solve_budgets(fixed, available, shares):
    """Token budgets per capped source: each at most its share of the total, where the total is
    ``fixed`` plus every capped source's budget (a fixed point; the shares sum below 1)."""
    if sum(shares.values()) >= 1:
        raise ValueError(f"shares must sum below 1: {shares}")
    total = fixed + sum(available.values())
    budgets = dict(available)
    for _ in range(200):
        budgets = {k: min(available[k], shares[k] * total) for k in available}
        new = fixed + sum(budgets.values())
        if abs(new - total) < 0.5:
            break
        total = new
    return budgets


def select_rounds(groups, budget, cost):
    """Take items group by group in rounds (every group's best, then every group's second, ...),
    skipping an item that no longer fits. ``groups``: ``{key: [items, best first]}``."""
    selected, remaining, depth = [], budget, 0
    while True:
        left = False
        for key in sorted(groups):
            items = groups[key]
            if depth >= len(items):
                continue
            left = True
            if cost(items[depth]) <= remaining:
                selected.append(items[depth])
                remaining -= cost(items[depth])
        if not left:
            return selected
        depth += 1


# --- the build -------------------------------------------------------------------------------------


def _check_rows_summary(rows):
    ratio = [(r["prompt_tokens"] / r["served_prompt_tokens"], r) for r in rows if r.get("served_prompt_tokens")]
    before = [r["segment_only_prompt_tokens"] / r["served_prompt_tokens"] for r in rows if r.get("served_prompt_tokens")]
    later = [x for x, r in ratio if r["level"] > 1]
    later_before = [r["segment_only_prompt_tokens"] / r["served_prompt_tokens"] for r in rows
                    if r.get("served_prompt_tokens") and r["level"] > 1]
    outliers = sorted(((abs(x - 1), x, r) for x, r in ratio if abs(x - 1) > RATIO_OUTLIER), key=lambda t: -t[0])
    return {
        "records": len(rows),
        "with_served_prompt": len(ratio),
        "rebuilt_over_served": pct([x for x, _ in ratio]),
        "rebuilt_over_served_level1": pct([x for x, r in ratio if r["level"] == 1]),
        "rebuilt_over_served_level2plus": pct(later),
        "segment_only_over_served": pct(before),
        "segment_only_over_served_level2plus": pct(later_before),
        "segment_only_below_half_served": sum(x < 0.5 for x in before),
        "rebuilt_below_half_served": sum(x < 0.5 for x, _ in ratio),
        "outliers": len(outliers),
        "outlier_examples": [{"id": r["id"], "ratio": round(x, 4), "prompt_tokens": r["prompt_tokens"],
                              "served_prompt_tokens": r["served_prompt_tokens"]} for _, x, r in outliers[:15]],
        "truncated_targets": sum(r["truncated"] for r in rows),
        "truncated_target_tokens": sum(r["target_tokens"] for r in rows if r["truncated"]),
        "python_error_targets": sum(r["python_error"] for r in rows),
        "python_error_target_tokens": sum(r["target_tokens"] for r in rows if r["python_error"]),
    }


def _format_summary(items):
    tokens = [i["tokens"] for i in items]
    return {"items": len(items), "tokens": sum(tokens), "loss_tokens": sum(i["loss_tokens"] for i in items),
            "length": pct(tokens, (50, 90, 99)), "buckets": buckets(tokens)}


class _Spool:
    """Rendered items on disk (the per-turn texts of every candidate do not fit comfortably in
    memory), with a light index in memory."""

    def __init__(self, path):
        self.path = Path(path)
        self.handle = open(self.path, "w")
        self.index = []

    def add(self, item):
        self.handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.index.append({k: item.get(k) for k in ("id", "source", "segment", "game_id", "level", "tokens",
                                                     "loss_tokens", "split")})

    def close(self):
        self.handle.close()

    def lines(self):
        with open(self.path) as handle:
            for line in handle:
                yield json.loads(line)


def build(out_dir, *, sources=None, mixture=None, template=None, counter=None, write=True, log=print):
    """Build the mixture and the evaluation split into ``out_dir``; returns the stats."""
    sources, mixture = sources or Sources(), mixture or Mixture()
    template, counter = template or ChatTemplate(), counter or TokenCounter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"config": {"mixture": asdict(mixture),
                        "sources": {k: [str(Path(p).relative_to(REPO)) if Path(p).is_relative_to(REPO) else str(p)
                                        for p in v] for k, v in asdict(sources).items()},
                        "teacher_runs": list(TEACHER_RUNS),
                        "template": str(template.path.relative_to(REPO)) if template.path.is_relative_to(REPO)
                        else str(template.path)},
             "tokenizer": {"exact": counter.exact, "path": counter.path,
                           "note": "Qwen3.6-35B-A3B tokenizer, Ornith-1.5 chat template, 64 tokens per image"
                           if counter.exact else "estimate: characters / 4.1, 64 tokens per image"}}
    spools = {name: _Spool(out / f".candidates-{name}.jsonl") for name in ("packed", "turns")}

    # 1. segment sources: filter, render, check against the served prompts
    filters, checks, segs = {}, {}, {}
    for source in ("dev", "teacher"):
        segments = list(read_jsonl(getattr(sources, source)))
        check_training_games(s["game_id"] for s in segments)
        kept, filters[source] = filter_segments(segments, cap_per_level=mixture.cap_per_level,
                                                threshold=mixture.near_duplicate)
        rows = []
        for segment in kept:
            packed, turns, check = segment_items(segment, source, template, counter)
            for item in packed:
                spools["packed"].add(item)
            for item in turns:
                spools["turns"].add(item)
            rows += check
        segs[source] = kept
        checks[source] = rows
        log(f"{source}: {len(segments)} segments -> {len(kept)} after filters, {len(rows)} targets")
    stats["filters"] = filters
    stats["prompt_check"] = {s: _check_rows_summary(r) for s, r in checks.items()}

    # 2. single-turn sources
    aux_tasks = [t for t in read_jsonl(sources.aux) if t.get("answer") and t["family"] in mixture.aux_families]
    check_training_games({t["game"] for t in aux_tasks})
    for task in aux_tasks:
        spools["turns"].add(aux_item(task, template, counter))
    solver_records = list(read_jsonl(sources.solver)) if mixture.solver else []
    check_training_games({r["game_id"] for r in solver_records})
    for record in solver_records:
        spools["turns"].add(solver_item(record, template, counter))
    for spool in spools.values():
        spool.close()

    # 3. budgets (target tokens) and selection
    turn_index = spools["turns"].index
    loss_by_segment = collections.Counter()
    for row in turn_index:
        if row["segment"]:
            loss_by_segment[row["segment"]] += row["loss_tokens"]
    teacher_tokens = sum(loss_by_segment[s["id"]] for s in segs["teacher"])
    fixed = mixture.teacher_epochs * teacher_tokens
    aux_rows = [r for r in turn_index if r["source"] == "aux"]
    solver_rows = [r for r in turn_index if r["source"] == "solver"]
    available = {"dev": sum(loss_by_segment[s["id"]] for s in segs["dev"]),
                 "aux": sum(r["loss_tokens"] for r in aux_rows)}
    shares = {"dev": mixture.dev_max_share, "aux": mixture.aux_max_share}
    if mixture.solver:
        available["solver"] = sum(r["loss_tokens"] for r in solver_rows)
        shares["solver"] = mixture.solver_max_share
    budgets = solve_budgets(fixed, available, shares)
    dev_groups = collections.defaultdict(list)
    for segment in sorted(segs["dev"], key=rank_key):
        dev_groups[(segment["game"], segment["level"])].append(segment)
    dev_chosen = select_rounds(dev_groups, budgets["dev"], lambda s: loss_by_segment[s["id"]])
    chosen_segments = {s["id"]: 1 for s in dev_chosen}
    chosen_segments.update({s["id"]: mixture.teacher_epochs for s in segs["teacher"]})

    def by_game(rows):
        groups = collections.defaultdict(list)
        for row in sorted(rows, key=lambda r: stable_key(r["id"])):
            groups[prefix(row["game_id"])].append(row)
        return groups

    chosen_single = {r["id"] for r in select_rounds(by_game(aux_rows), budgets["aux"], lambda r: r["loss_tokens"])}
    if mixture.solver:
        chosen_single |= {r["id"] for r in select_rounds(by_game(solver_rows), budgets["solver"],
                                                         lambda r: r["loss_tokens"])}

    def repeats(item):
        if item["segment"]:
            return chosen_segments.get(item["segment"], 0)
        return 1 if item["id"] in chosen_single else 0

    # 4. write both formats (single-turn items appear in both, so each file is a whole mixture)
    summary = {"packed": [], "turns": []}
    handles = {name: open(out / f"train_{name}.jsonl", "w") for name in summary} if write else {}

    def emit(name, items):
        for item in items:
            for repeat in range(repeats(item)):
                row = {**item, "repeat": repeat, "weight": mixture.weights.get(item["source"], 1.0)}
                if not item["segment"]:
                    row["format"] = "single"
                summary[name].append({k: row[k] for k in LIGHT_KEYS})
                if write:
                    handles[name].write(json.dumps(row, ensure_ascii=False) + "\n")

    emit("packed", spools["packed"].lines())
    emit("packed", (i for i in spools["turns"].lines() if not i["segment"]))
    emit("turns", spools["turns"].lines())
    for handle in handles.values():
        handle.close()
    for spool in spools.values():
        spool.path.unlink()

    # 5. stats of the mixture
    turns = summary["turns"]
    loss = collections.Counter()
    for row in turns:
        loss[row["source"]] += row["loss_tokens"]
    total_loss = sum(loss.values()) or 1
    mix = {}
    for source in ("dev", "teacher", "aux", "solver"):
        rows_t = [r for r in turns if r["source"] == source]
        rows_p = [r for r in summary["packed"] if r["source"] == source]
        mix[source] = {
            "target_tokens": loss[source], "share": round(loss[source] / total_loss, 4),
            "target_tokens_unique": sum(r["loss_tokens"] for r in rows_t if r["repeat"] == 0),
            "segments": len({r["segment"] for r in rows_t if r["segment"]}),
            "games": len({prefix(r["game_id"]) for r in rows_t}),
            "per_turn": _format_summary(rows_t), "packed": _format_summary(rows_p),
        }
    mix["budgets_target_tokens"] = {k: int(v) for k, v in budgets.items()}
    mix["available_target_tokens"] = {"teacher": teacher_tokens, **available}
    mix["teacher_epochs"] = mixture.teacher_epochs
    mix["total_target_tokens"] = sum(loss.values())
    mix["per_turn"] = _format_summary(turns)
    mix["packed"] = _format_summary(summary["packed"])
    stats["mixture"] = mix
    dev_loss = collections.Counter()
    dev_segments = collections.Counter()
    for row in turns:
        if row["source"] == "dev":
            dev_loss[prefix(row["game_id"])] += row["loss_tokens"]
    for segment in dev_chosen:
        dev_segments[segment["game"]] += 1
    levels = collections.Counter((s["game"], s["level"]) for s in dev_chosen)
    stats["concentration"] = {
        "dev_target_tokens": share_top(dev_loss), "dev_segments": share_top(dev_segments),
        "dev_segments_per_game": dict(dev_segments.most_common()),
        "dev_levels_covered": len(levels), "dev_levels_available": len(dev_groups),
        "dev_max_segments_per_level": max(levels.values(), default=0),
        "dev_level_hist": dict(sorted(collections.Counter(s["level"] for s in dev_chosen).items())),
        "generated_share_of_target_tokens": round(
            sum(r["loss_tokens"] for r in turns if prefix(r["game_id"]).startswith("q")) / total_loss, 4),
    }
    aux_games = collections.Counter(prefix(r["game_id"]) for r in turns if r["source"] == "aux")
    stats["aux"] = {"tasks_available": len(aux_rows), "tasks_selected": sum(aux_games.values()),
                    "generated_games": sum(1 for g in aux_games if g.startswith("q")),
                    "dev_games": sum(1 for g in aux_games if not g.startswith("q")),
                    "families": dict(collections.Counter(t["family"] for t in aux_tasks)),
                    "no_reasoning": True}

    # 6. the evaluation split: held-out generated mechanics only
    stats["eval"] = build_eval(out, sources, template, counter, write=write)
    if write:
        (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    return stats


def build_eval(out, sources, template, counter, *, write=True):
    """Teacher segments (loss checks), solver requests (next-action checks) and aux tasks of the
    held-out generated mechanics, in both formats."""
    packed, turns, rows = [], [], []
    segments = list(read_jsonl(sources.eval_teacher))
    check_eval_games(s["game_id"] for s in segments)
    for segment in segments:
        p, t, c = segment_items(segment, "teacher", template, counter, split="eval")
        packed += p
        turns += t
        rows += c
    solver = list(read_jsonl(sources.eval_solver))
    check_eval_games(r["game_id"] for r in solver)
    single = [solver_item(r, template, counter, split="eval") for r in solver]
    aux = [t for t in read_jsonl(sources.eval_aux) if t.get("answer")]
    check_eval_games(t["game"] for t in aux)
    single += [aux_item(t, template, counter, split="eval") for t in aux]
    for item in single:
        item["format"] = "single"
    for item in [*packed, *turns, *single]:
        item.setdefault("repeat", 0)
        item.setdefault("weight", 1.0)
    if write:
        for name, items in (("packed", [*packed, *single]), ("turns", [*turns, *single])):
            with open(Path(out) / f"eval_{name}.jsonl", "w") as handle:
                for item in items:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {"games": sorted({i["game_id"] for i in [*packed, *turns, *single]}),
            "teacher_segments": len(segments), "packed": _format_summary(packed), "turns": _format_summary(turns),
            "solver_requests": len(solver), "aux_tasks": len(aux),
            "prompt_check": _check_rows_summary(rows)}
