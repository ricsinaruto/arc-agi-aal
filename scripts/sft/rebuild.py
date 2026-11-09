"""trunk/sft prompt rebuild: the served requests of an unrecorded play, rebuilt from its log.jsonl
by replaying the run's own harness (the bundle it played with), as rollout.jsonl rows build.py reads.

    .venv/bin/python scripts/sft/rebuild.py gate --runs RUN[,RUN] [--per-game 2] --out gate.json   # recorded runs
    .venv/bin/python scripts/sft/rebuild.py rebuild --manifest manifest.json --out rebuilt.json      # the 43 runs

One Modal CPU job per (run, trial, instance). The job runs the bundle's ``arena.level_bench.LevelBenchAgent`` (the
arena's Duck-transfer agent with the trim bed's scripted replies, hx7) on a fresh game: every gameplay reply comes from
the log's ``model`` rows in order, every tool call runs again in the sandbox, every action replays on the game, and the
server's tokenizer (``/tokenize``) is the served model's processor. Each request the harness sends is written as a
rollout row (``keep`` + new messages, tools, template, the logged reply and usage, ``rebuilt: true``).

Checks per instance (``checks``): the replay's own log against the recorded one (boards, levels and actions per step,
python tool results, the tokenizer counts of every gameplay request, drains), the final state, and with a recording
(``gate``) every rebuilt request against the recorded one (messages, tools, template). Wall-clock text: the stuck
notice's active seconds are the logged clock's (``model_request`` runtime at the request; the level start interpolated
at its observation)."""
import argparse
import collections
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
BUCKET, ENDPOINT, REGION = "arc-runs-bucket", "https://storage.eu-south1.nebius.cloud", "eu-south1"
PROCESSORS = ("/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1",
              "/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540")
BUNDLES = "/runs/sft-data/bundles"   # one folder per bundle source hash: src/, configs/ (no wheels)
REBUILT = "/runs/sft-data/rebuilt"   # <run>/<trial>/<k>/rollout.jsonl + checks.json
MAX_MODEL_LEN = 262144
REPLAY_TOOL_SECONDS = 300
INVENTORY_HOOKS = []  # the current replay's inventory check (StatefulSandbox.inventory is wrapped once per process)
STUCK_SECONDS = re.compile(r"\((\d+)s of active play\)")
LOG_KINDS = ("observation", "duck_python", "token_count", "context_drain", "auto_reset", "length_truncation",
             "duck_python_interrupt", "playbook_shown", "compaction_inventory", "hx_reflect", "dfr_noop_stop",
             "stuck_notice")
LOG_SKIP = {"time", "seconds", "rss_bytes", "elapsed", "wall_seconds", "issued_gameplay_seconds"}
# tool code whose effect does not replay (duck_transfer.sandbox.CLOCK_CODE, arena.restore.RANDOM_CODE; the pool's
# bundles predate the seeded sandbox): the interpreter state after it may differ from the played one
CLOCK_CODE = re.compile(r"\b(?:time\.(?:time|time_ns|perf_counter|perf_counter_ns|monotonic|monotonic_ns|process_time)"
                        r"|perf_counter|monotonic|process_time|datetime\.(?:now|utcnow|today)|date\.today)\s*\(")
RANDOM_CODE = re.compile(r"\brandom\b")
NUMERIC = re.compile(r"[0-9\s,.()\[\]{}+-]*")
# code that builds a set (the only source of str-hash order: per-process random in the pool's sandboxes)
SET_CODE = re.compile(r"\b(?:set|frozenset)\s*\(|\{[^{}:]*\bfor\b[^{}:]*\}|\{\s*['\"][^'\"]*['\"]\s*[,}]|\.union\(|"
                      r"\.intersection\(|\.difference\(")
QUOTED = re.compile(r"'[^'\n]*'|\"[^\"\n]*\"")
OBSERVERS = "Your observers after the last action"
CONTAINERS = ("list", "tuple", "dict", "set", "frozenset", "deque", "defaultdict", "OrderedDict", "Counter")

app = modal.App("sft-rebuild")
runs = modal.Volume.from_name("arc-prolong-runs")
assets = modal.Volume.from_name("arc-prolong-assets")


def _image():
    req = os.environ.get("SFT_REBUILD_REQUIREMENTS")  # the bundles' locked requirements (identical across the pool)
    img = modal.Image.debian_slim(python_version="3.12").apt_install("rclone", "procps")
    if req:
        img = img.pip_install_from_requirements(req)
    return (img.pip_install("torch==2.14.0", "torchvision==0.29.0", index_url="https://download.pytorch.org/whl/cpu")
            .pip_install("transformers==5.18.0", "tokenizers", "safetensors", "huggingface_hub>=0.34")
            .add_local_file(str(HERE / "data.py"), "/root/sft_data.py"))


image = _image() if modal.is_local() else modal.Image.debian_slim(python_version="3.12")


# ---- pure parts (unit-tested) ---------------------------------------------------------------------------------------

def read_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def script_of(rows):
    """The log's replies: gameplay replies in order (the trim bed's script) and the others (summaries, asides) in
    order, each ``{message, finish_reason, usage}``."""
    game, other = [], []
    for r in rows:
        if r.get("kind") != "model":
            continue
        reply = {"message": r["message"], "finish_reason": r.get("finish_reason"), "usage": r.get("usage") or {}}
        (game if r.get("purpose", "gameplay") == "gameplay" else other).append(reply)
    return game, other


def action_text(action):
    """An observation row's action as the level bench's ``bench_actions`` text (None: no action)."""
    if action is None or action == "None":
        return None
    if isinstance(action, str):
        return action
    name = action.get("action") or action.get("name")
    return name + (f" {action['x']} {action['y']}" if name == "ACTION6" else "")


def timeline(rows):
    """``(request times and active seconds, observation time per step)`` of the log: the clock the stuck notice read."""
    requests = [(float(r["time"]), float((r.get("runtime") or {}).get("elapsed_seconds") or 0.0))
                for r in rows if r.get("kind") == "model_request"]
    observed = {}
    for r in rows:
        if r.get("kind") == "observation":
            observed.setdefault(int(r["step"]), float(r["time"]))
    return requests, observed


def active_at(requests, t):
    """Active seconds at wall time ``t``: the last request's, plus the wall time since, at most up to the next
    request's (parked time is not active)."""
    before = [(rt, e) for rt, e in requests if rt <= t]
    if not before:
        return requests[0][1] if requests else 0.0
    rt, e = before[-1]
    nxt = next(((rt2, e2) for rt2, e2 in requests if rt2 > t), None)
    return e + (t - rt if nxt is None else min(t - rt, nxt[1] - e))


def rollout_rows(captures):
    """Rebuilt requests as rollout.jsonl rows: ``keep`` messages shared with the previous request, the new ones,
    ``tools`` when they change."""
    out, previous, tools = [], [], object()
    for n, c in enumerate(captures):
        msgs = c["payload"]["messages"]
        keep = 0
        for a, b in zip(previous, msgs):
            if a != b:
                break
            keep += 1
        row = {"n": n, "step": c["step"], "gameplay": c["gameplay"], "keep": keep, "messages": msgs[keep:],
               "template": c["payload"].get("chat_template_kwargs") or {}, "reply": c["reply"]["message"],
               "finish_reason": c["reply"].get("finish_reason"), "usage": c["reply"].get("usage") or {},
               "rebuilt": True, "count": c.get("count")}
        if c["payload"].get("tools") != tools:
            tools = c["payload"].get("tools")
            row["tools"] = tools
        out.append(row)
        previous = msgs
    return out


def requests_of(rows):
    """Full requests of rollout rows: ``[(messages, tools, template)]`` (every row, gameplay or not)."""
    messages, tools, out = [], None, []
    for row in rows:
        messages = messages[: row["keep"]] + row["messages"]
        tools = row["tools"] if "tools" in row else tools
        out.append({"messages": messages, "tools": tools, "template": row.get("template") or {},
                    "gameplay": row.get("gameplay", True), "n": row.get("n"), "step": row.get("step")})
    return out


def classify(a, b):
    """The class of one differing message pair (rebuilt, recorded)."""
    ta, tb = json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True)
    if STUCK_SECONDS.sub("(Ns of active play)", ta) == STUCK_SECONDS.sub("(Ns of active play)", tb):
        return "stuck_seconds"
    if a.get("role") != b.get("role"):
        return "role"
    if a.get("role") == "tool":
        return "tool_output"
    if a.get("role") == "assistant":
        return "assistant"
    if a.get("role") == "system":
        return "system"
    return "user_text"


def compare(rebuilt, recorded):
    """Per recorded gameplay request, the rebuilt one: ``{"exact", "mismatches": [{n, class, ...}], "missing"}``.
    Pairs requests in order (gameplay with gameplay, the others with the others)."""
    out = {"requests": 0, "exact": 0, "mismatches": [], "missing": 0, "extra": 0}
    for kind in (True, False):
        a = [r for r in requests_of(rebuilt) if r["gameplay"] is kind]
        b = [r for r in requests_of(recorded) if r["gameplay"] is kind]
        out["missing"] += max(0, len(b) - len(a))
        out["extra"] += max(0, len(a) - len(b))
        for x, y in zip(a, b):
            out["requests"] += 1
            if x["messages"] == y["messages"] and x["tools"] == y["tools"] and x["template"] == y["template"]:
                out["exact"] += 1
                continue
            if x["tools"] != y["tools"]:
                cls, at = "tools", None
            elif x["template"] != y["template"]:
                cls, at = "template", None
            else:
                at = next((i for i, (p, q) in enumerate(zip(x["messages"], y["messages"])) if p != q),
                          min(len(x["messages"]), len(y["messages"])))
                if at >= min(len(x["messages"]), len(y["messages"])):
                    cls = "message_count"
                else:
                    cls = classify(x["messages"][at], y["messages"][at])
            entry = {"n": y["n"], "step": y["step"], "gameplay": kind, "class": cls, "message": at}
            if at is not None and at < min(len(x["messages"]), len(y["messages"])):
                p, q = json.dumps(x["messages"][at], sort_keys=True), json.dumps(y["messages"][at], sort_keys=True)
                c = next((i for i, (u, v) in enumerate(zip(p, q)) if u != v), min(len(p), len(q)))
                entry["rebuilt"], entry["recorded"] = p[max(0, c - 120):c + 200], q[max(0, c - 120):c + 200]
            out["mismatches"].append(entry)
    return out


def norm(r):
    return {k: v for k, v in r.items() if k not in LOG_SKIP}


def hash_ordered(data):
    """Inventory values whose text depends on the interpreter's hash seed (random per sandbox in the pool's bundles):
    sets of two or more non-numeric items."""
    return [name for name, kind, size, text in (data or {}).get("values") or []
            if kind in ("set", "frozenset") and (size or 0) >= 2
            and not NUMERIC.fullmatch(str(text).replace("frozenset", ""))]


def strings_ordered(text):
    """A container text with two or more strings: its order may come from iterating a set of strings."""
    text = str(text)
    return any(c in text for c in "[{(") and len(QUOTED.findall(text)) >= 2


def hash_risky(data):
    """Inventory values whose order may depend on the hash seed once the code has built a set: containers of strings."""
    return [name for name, kind, size, text in (data or {}).get("values") or []
            if kind in CONTAINERS and (size or 0) >= 2 and strings_ordered(text)]


def observer_lines(messages):
    """The hx observer report lines of these messages (user-defined functions' output; not in the log)."""
    out = []
    for m in messages:
        content = m.get("content")
        parts = [content] if isinstance(content, str) else [p.get("text") or "" for p in content or []
                                                            if isinstance(p, dict)]
        out += [line for part in parts for line in str(part).splitlines() if line.startswith(OBSERVERS)]
    return out


def log_diff(replayed, recorded):
    """The replay's log against the recorded one, per kind in ``LOG_KINDS``: rows compared in order on their fields
    without times and durations."""
    out = {}
    for kind in LOG_KINDS:
        a = [norm(r) for r in replayed if r.get("kind") == kind]
        b = [norm(r) for r in recorded if r.get("kind") == kind]
        if not a and not b:
            continue
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        entry = {"replayed": len(a), "recorded": len(b), "first_diff": first}
        if first is not None:
            x, y = a[first], b[first]
            keys = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
            entry["keys"] = keys
            entry["example"] = {k: [str(x.get(k))[:300], str(y.get(k))[:300]] for k in keys[:4]}
        out[kind] = entry
    return out


# ---- the replay (inside the container, with the bundle's src on sys.path) ----------------------------------------

class Tokenizer:
    """vLLM's ``/tokenize`` for chat requests: the served model's chat template and processor."""

    def __init__(self, folder):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(folder)
        self.tok = self.processor.tokenizer
        spec = __import__("importlib.util").util.spec_from_file_location("sft_data", "/root/sft_data.py")
        self.data = __import__("importlib.util").util.module_from_spec(spec)
        spec.loader.exec_module(self.data)

    text_images = False  # the played server's answer to an image counted as text (arc_infra.model text_images)

    def count(self, payload):
        """The prompt's token count; None: refused as the played server refused an image placeholder sent as text
        (the client then counts with the image; the log shows which: its token_count rows' image_tokens)."""
        if not self.text_images and "<|image_pad|>" in json.dumps(payload["messages"]) and not any(
                isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])
                for m in payload["messages"]):
            return None
        messages, images = self.data.served_messages(payload["messages"])
        kw = dict(payload.get("chat_template_kwargs") or {})
        cont = bool(payload.get("continue_final_message"))
        text = self.tok.apply_chat_template(messages, tools=payload.get("tools"), tokenize=False,
                                            add_generation_prompt=payload.get("add_generation_prompt", not cont),
                                            continue_final_message=cont, **kw)
        if images:
            return int(self.processor(text=[text], images=images, return_tensors="pt")["input_ids"].shape[1])
        return len(self.tok(text, add_special_tokens=False)["input_ids"])


def replay(bundle, study, rows, game_id, workdir, games_dir, tokenizer, recorded=None, layout=None):
    """Replay one recorded instance; returns ``(rollout rows, checks)``."""
    import asyncio
    import time

    import httpx

    sys.path[:0] = [str(Path(bundle) / "src")]
    os.environ["PYTHONPATH"] = str(Path(bundle) / "src")
    from arena.trim_bed import settings as trim_settings

    from arc_infra.arcade import make_arcade
    from arc_infra.fleet import SessionClient
    from arc_infra.perception import board_hash
    from arena.cache import BlockBudget
    from arena.clock import ActiveClock, GlobalClock
    from arena.compaction import DEFAULT_COMPACTION_THINKING
    from arena.gate import GateSlot, KVGate
    from arena.level_bench import LevelBenchAgent
    from arena.scheduler import GameRecord
    from arena.trial import duck_config
    from duck_transfer import loader
    from duck_transfer.agent import GameplayRequestExpired  # noqa: F401 - the agent's stop path
    from duck_transfer.sandbox import StatefulSandbox

    for key in ("cross_game_notes", "duck_harness", "explorer", "first_trigger_stagger", "parallel_samples",
                "action_plans", "competition_clock"):
        if study.get(key):
            raise ValueError(f"rebuild: study key {key!r} is not replayed")
    game_rows, other_rows = script_of(rows)
    requests, observed = timeline(rows)
    tokenizer.text_images = any(int(r.get("image_tokens") or 0) > 0 for r in rows if r.get("kind") == "token_count")
    obs = [r for r in rows if r.get("kind") == "observation"]
    last = obs[-1]
    stop = {"steps": int(last["step"]), "levels": int(last["levels"]),
            "board_hash": board_hash([[int(c, 16) for c in row] for row in last["frames"][-1]])}
    actions = [a for a in (action_text(r.get("action")) for r in obs) if a is not None]
    captures, counts, others = [], [], list(other_rows)
    py_rows = [r for r in rows if r.get("kind") == "duck_python"]
    subs = []  # tool results that differed from the logged ones (the logged one is what the prompt carries)
    # The cut: the first request that may not be the played one is not emitted, nor any after it. Taints (tool code
    # reading the clock or ``random``, a timeout, a substituted result) leave the interpreter state uncertain; the
    # first inventory with values after one cuts. A replayed log row that differs from the recorded one cuts at once.
    # sets: calls whose code builds a set; orders: calls whose output showed a different hash order than the play's
    cut, taint, sets, orders = {"at": None, "reason": None}, [], [], []
    expect = {k: [norm(r) for r in rows if r.get("kind") == k] for k in LOG_KINDS}
    seen = collections.Counter()

    def mark(reason):
        if cut["at"] is None:
            cut["at"], cut["reason"], cut["drains"] = len(captures), reason, seen["context_drain"]

    def watch(workspace):
        base = workspace.append

        def append(kind, **payload):
            base(kind, **payload)
            if kind in expect:
                i = seen[kind]
                seen[kind] += 1
                row = norm(json.loads(json.dumps({"kind": kind, **payload}, ensure_ascii=False, default=str)))
                if i >= len(expect[kind]) or row != expect[kind][i]:
                    mark(f"log:{kind}")
        workspace.append = append

    def handler(request):
        path = request.url.path
        body = json.loads(request.content or b"{}")
        if path.endswith("/tokenize"):
            n = tokenizer.count(body)
            if n is None:
                return httpx.Response(400, json={"error": "rebuild: image placeholder sent as text"})
            counts.append(n)
            return httpx.Response(200, json={"count": n, "max_model_len": MAX_MODEL_LEN})
        if path.endswith("chat/completions"):
            if not others:
                return httpx.Response(500, json={"error": "rebuild: no recorded reply left"})
            reply = others.pop(0)
            captures.append({"payload": copy.deepcopy(body), "reply": reply, "gameplay": False, "step": agent.steps})
            return httpx.Response(200, json={"choices": [{"index": 0, "message": reply["message"],
                                                          "finish_reason": reply["finish_reason"]}],
                                             "usage": reply["usage"]})
        return httpx.Response(404, json={"error": path})

    class Agent(LevelBenchAgent):
        def bench_script_finish(self):
            super().bench_script_finish()
            if self.bench_outcome is None:
                self.bench_outcome = "rebuilt"

        async def request_completion(self, payload, *args, **kwargs):
            gameplay = (self.request_purpose or ("summary" if not payload.get("tools") else "gameplay")) == "gameplay"
            previous = captures[-1]["payload"]["messages"] if captures else []
            msgs = payload.get("messages") or []
            shared = next((j for j, (a, b) in enumerate(zip(previous, msgs)) if a != b), min(len(previous), len(msgs)))
            lines = observer_lines(msgs[shared:])
            if lines and taint:
                mark(f"observers_after_{taint[0]['class']}")
            elif lines and orders and any(strings_ordered(line) for line in lines):
                mark("observers_hash_order")
            if gameplay and self.bench_script_at < len(self.bench_script["replies"]):
                captures.append({"payload": copy.deepcopy(payload), "gameplay": True, "step": self.steps,
                                 "reply": self.bench_script["replies"][self.bench_script_at],
                                 "count": counts[-1] if counts else None})
            return await super().request_completion(payload, *args, **kwargs)

        def run_python_tool(self, execute, state_path, arguments):
            """The tool runs (state, actions); its result is the logged one when they differ: python's set order
            (per-process hash seed) and wall-clock timeouts do not replay. The steps and boards are checked apart."""
            import dataclasses

            i = len(self.py_seen)
            self.py_seen.append(i)
            code = str(arguments.get("code") or "")
            for name, pattern in (("clock", CLOCK_CODE), ("random", RANDOM_CODE)):
                if pattern.search(code):
                    taint.append({"class": name, "call": i, "step": self.steps})
            if SET_CODE.search(code):
                sets.append(i)
            # a call that finished in the play gets a long timeout here (a slower replay host must not time it out);
            # one that timed out keeps the played 30 s (its state is tainted either way)
            plumb = getattr(execute, "__self__", None)
            played_out = i < len(py_rows) and "timed out" in json.dumps(py_rows[i].get("result"))
            saved = getattr(plumb, "_python_timeout", None)
            if saved is not None and not played_out:
                plumb._python_timeout = REPLAY_TOOL_SECONDS
            try:
                out = super().run_python_tool(execute, state_path, arguments)
            finally:
                if saved is not None:
                    plumb._python_timeout = saved
            if i >= len(py_rows) or (py_rows[i].get("code") or "") != code:  # a call without code logs None
                subs.append({"call": i, "class": "unlogged_call" if i >= len(py_rows) else "code_misaligned"})
                mark("tool_misaligned")
                return out
            logged = py_rows[i].get("result")
            mine = json.loads(out.content)
            if "timed out" in json.dumps(logged):
                taint.append({"class": "timeout_played", "call": i, "step": self.steps})
            if mine == logged:
                return out
            fmt = next((f for f in (lambda v: json.dumps(v, indent=2), lambda v: json.dumps(v, indent=2,
                        ensure_ascii=False)) if f(mine) == out.content), None)
            text = json.dumps(mine) + json.dumps(logged)
            cls = ("timeout" if "timed out" in text else "set_order" if sorted(json.dumps(mine)) == sorted(
                json.dumps(logged)) else "other")
            subs.append({"call": i, "step": self.steps, "class": cls, "format": fmt is not None,
                         "mine": json.dumps(mine)[:300], "logged": json.dumps(logged)[:300]})
            if cls == "set_order":  # hash order seen: an inventory or observer showing ordered strings cuts after it
                orders.append(i)
            else:
                taint.append({"class": cls, "call": i, "step": self.steps})
            if fmt is None:
                mark("tool_unformatted")
                return out
            return dataclasses.replace(out, content=fmt(logged)) if dataclasses.is_dataclass(out) else out

    def check_inventory(data):
        """The live definitions the drain note renders: their values carry the interpreter state into the prompt."""
        if (data or {}).get("values") and taint:
            mark(f"inventory_after_{taint[0]['class']}")
        elif hash_ordered(data) or (orders and hash_risky(data)):
            mark("inventory_hash_order")

    if not getattr(StatefulSandbox, "_rebuild_hooked", False):
        base_inventory = StatefulSandbox.inventory

        def inventory(self, *args, **kwargs):
            data = base_inventory(self, *args, **kwargs)
            for hook in INVENTORY_HOOKS:
                hook(data)
            return data
        StatefulSandbox.inventory, StatefulSandbox._rebuild_hooked = inventory, True
    INVENTORY_HOOKS[:] = [check_inventory]

    async def main():
        nonlocal agent
        window, cap = int(study["windows"][0]), int(study["completion_cap"])
        cfg = duck_config(study, bundle, window=window, cap=cap, wall=float(study["wall_seconds"]),
                          base_url="http://127.0.0.1:1234/v1")
        loader.configure(cfg, sampling=study.get("sampling"))
        # the replay's wall: the played one x 20, so a slower replay host never reaches the time limit (the prompts carry
        # no wall-clock text but the stuck notice's, which reads the logged clock)
        clock = GlobalClock(float(study["wall_seconds"]) * 20)
        game = GameRecord(game_id, 0, ActiveClock(clock, budget_clock=study.get("budget_clock", "inference")))
        game.ready = asyncio.Event()
        # the run's block admission (hybrid_blocks): the agent counts every request exactly through the gate's budget
        budget = (BlockBudget(layout, float(study["kv_margin"])) if study.get("cache_admission") == "hybrid_blocks"
                  else None)
        if study.get("cache_admission") == "hybrid_blocks" and layout is None:
            raise ValueError("rebuild: hybrid_blocks admission needs the run's serving/cache-layout.json")
        gate = KVGate(budget.capacity if budget else 10 ** 9, slots=1, log=lambda entry: None, budget=budget)
        slot = GateSlot(gate, game, cap=cap, window=window)
        shared = httpx.AsyncClient(base_url=cfg.base_url.rstrip("/") + "/", transport=httpx.MockTransport(handler),
                                   trust_env=False)
        entry = {"game_id": game_id, "replica": 0, "instance_id": "r0-" + game_id}
        client = SessionClient(cfg, shared, slot, entry, [])
        arc = owner if owner is not None else make_arcade("offline", games_dir, Path(workdir) / "recordings")
        card = arc.open_scorecard(tags=["sft-rebuild"])
        agent = Agent(game_id, arc, card, client, cfg, Path(workdir) / "game", sys.executable,
                      time.monotonic() + float(study["wall_seconds"]) + 3600)
        agent.py_seen = []
        watch(agent.workspace)
        agent.attach(game, window=window, cap=cap, notes=None, memory=study.get("memory_block") or None,
                     escalation=study.get("model_escalation") or None, snapshot=study.get("game_snapshot") or None,
                     harness=None, procedure=study.get("stuck_procedure") or None, explorer=None,
                     critical_errors=bool(study.get("critical_errors")), harness_v2=study.get("harness_v2") or None,
                     adaptive_thinking=study.get("adaptive_thinking") or None,
                     worldmodel_notice=bool(study.get("worldmodel_notice")),
                     harness_fixes=study.get("harness_fixes") or None, parallel_samples=None, action_plans=False,
                     context_drain=study.get("drain_half") or None, dfr_harness=study.get("dfr_harness") or None,
                     dfr_loop=study.get("dfr_loop") or None, hx_harness=study.get("hx_harness") or None,
                     loop_rule=study.get("loop_rule") or None)
        agent.compaction_keep_observations = int(study.get("compaction_keep_observations", 1))
        agent.compaction_keep_fraction = study.get("compaction_keep_fraction")
        agent.compaction_summary_cap = study.get("compaction_summary_cap")
        agent.compaction_full_prefix = bool(study.get("compaction_full_prefix"))
        agent.trim = trim_settings(study.get("trim"))
        agent.compaction_thinking = study.get("compaction_thinking", DEFAULT_COMPACTION_THINKING)
        agent.bench_setup(start_level=0, budget=10 ** 12, through=True,
                          script={"replies": game_rows, "stop": stop, "actions": actions})
        stuck = agent.stuck
        if hasattr(stuck, "active_seconds") and hasattr(stuck, "level_active"):
            # the logged clock: the request's active seconds, the level start's at its observation
            stuck.active_seconds = lambda: (requests[agent.bench_script_at][1]
                                            if agent.bench_script_at < len(requests) else requests[-1][1])
            base_reset = stuck.reset_level

            def reset_level(now=None):
                base_reset(now)
                t = observed.get(int(agent.steps))
                stuck.level_active = active_at(requests, t) if t is not None and agent.steps else 0.0
            stuck.reset_level = reset_level
        if hasattr(stuck, "notice"):
            # the notice text is logged (stuck_notice): the played one is used; its seconds read the played clock
            base_notice, logged_notices = stuck.notice, [r.get("text") for r in rows if r.get("kind") == "stuck_notice"]

            def notice(*args, **kwargs):
                text = base_notice(*args, **kwargs)
                if text is None:
                    return None
                i = agent.notices_seen
                agent.notices_seen += 1
                logged = logged_notices[i] if i < len(logged_notices) else None
                if logged is None or STUCK_SECONDS.sub("", logged) != STUCK_SECONDS.sub("", text):
                    mark("stuck_notice")
                    return text
                return logged
            agent.notices_seen = 0
            stuck.notice = notice
        clock.release()
        game.clock.release()
        result = await agent.run()
        await shared.aclose()
        return result

    agent = owner = None
    started = time.monotonic()
    if study.get("competition"):
        # the run's arcade: the local competition gateway of public clones (k000 ...), as arena.run opens it
        from arena.competition import open_arcade

        with open_arcade(study, games_dir, Path(workdir) / "arcade") as (resolved, owner):
            if game_id not in resolved["games"]:
                raise ValueError(f"rebuild: {game_id} is not in the clone roster")
            result = asyncio.run(main())
    else:
        result = asyncio.run(main())
    replayed = read_log(agent.workspace.log)
    end = agent.bench_script_end or {}
    checks = {"status": (result or {}).get("status"), "error": (result or {}).get("error"),
              "outcome": agent.bench_outcome, "script_end": end, "replies": len(game_rows),
              "scripted": agent.bench_script_at, "other_replies": len(other_rows), "others_left": len(others),
              "seconds": round(time.monotonic() - started, 1), "log": log_diff(replayed, rows),
              "tool_substitutions": subs, "taints": taint[:20], "set_calls": len(sets), "cut": cut["reason"]}
    kept = len(captures) if cut["at"] is None else cut["at"]
    tails = {"token_count": 1, "exact_admission_count": 1}  # the next request's count, sent before the run ended
    # finished: the script's end state, or the run's time limit after every reply (the log watch checked every board)
    finished = bool(end.get("ok")) or checks["status"] == "time_limit"
    checks.update(kept=kept, captured=len(captures), kept_drains=cut.get("drains", seen["context_drain"]),
                  game_replies=len(game_rows),
                  whole=cut["at"] is None and finished and agent.bench_script_at == len(game_rows) and not others
                  and all(v["first_diff"] is None and 0 <= v["recorded"] - v["replayed"] <= tails.get(k, 0)
                          for k, v in checks["log"].items()))
    return rollout_rows(captures), checks


def first_difference(a, b):
    """The first request index where two rollouts differ (full messages, tools, template), or the shorter length."""
    x, y = requests_of(a), requests_of(b)
    return next((i for i, (p, q) in enumerate(zip(x, y)) if (p["messages"], p["tools"], p["template"])
                 != (q["messages"], q["tools"], q["template"])), min(len(x), len(y)))


def settle(replays, recorded=None):
    """Two replays of one instance (independent sandboxes: their own hash seeds, clocks and timings) -> ``(rollout,
    checks)``: kept up to each one's cut and to the first request where they differ; with a recording, the kept
    requests compared with it."""
    (a, ca), (b, cb) = replays
    agree = first_difference(a, b)
    kept = min(ca["kept"], cb["kept"], agree)
    out = a[:kept]
    checks = {**ca, "kept": kept, "replica": {k: cb.get(k) for k in ("kept", "cut", "whole", "status", "seconds")},
              "agree": agree, "cut": ca["cut"] if kept == ca["kept"] and ca["cut"] else
              (cb["cut"] if kept == cb["kept"] and cb["cut"] else ("replicas_differ" if kept < len(a) else None))}
    checks["whole"] = bool(ca["whole"] and cb["whole"] and kept == len(a))
    checks["kept_reply_tokens"] = sum((r["usage"] or {}).get("completion_tokens") or 0 for r in out if r["gameplay"])
    if recorded is not None:
        # a recorded request whose reply the log never got (the run ended in its tool call) has no rebuild
        logged = [r for r in recorded if r.get("gameplay", True)][: ca["game_replies"]]
        tail = sum(r.get("gameplay", True) for r in recorded) - len(logged)
        keep_rows = sorted([r for r in recorded if not r.get("gameplay", True)] + logged, key=lambda r: int(r["n"]))
        checks["compare"] = {**compare(out, keep_rows[:kept]), "unlogged_tail": tail}
    # a cut after the last request (the run's final tool call, unlogged) keeps every request
    checks["ok"] = (checks["whole"] or kept < len(a) or bool(ca["cut"] or cb["cut"])) and (
        recorded is None or (not checks["compare"]["mismatches"] and not checks["compare"]["missing"]))
    return out, checks


# ---- Modal ----------------------------------------------------------------------------------------------------------

def rclone(*args):
    return subprocess.run(["rclone", *args], check=True, capture_output=True, text=True).stdout


def rclone_env():
    os.environ.update(RCLONE_CONFIG_ARC_TYPE="s3", RCLONE_CONFIG_ARC_PROVIDER="Other", RCLONE_CONFIG_ARC_ENV_AUTH="true",
                      RCLONE_CONFIG_ARC_ENDPOINT=ENDPOINT, RCLONE_CONFIG_ARC_REGION=REGION,
                      RCLONE_CONFIG_ARC_NO_CHECK_BUCKET="true")


_tok = []


@app.function(image=image, timeout=600, retries=0, cpu=(0.125, 2), volumes={"/runs": runs})
def saved() -> dict:
    """Instances already rebuilt: ``{run/trial/k: checks}`` from the volume (a relaunch skips them)."""
    runs.reload()
    return {"/".join(f.parts[-4:-1]): json.loads(f.read_text()) for f in Path(REBUILT).glob("*/*/*/checks.json")}


@app.function(image=image, timeout=2700, retries=0, cpu=(4, 8), single_use_containers=True,
              max_containers=int(os.environ.get("SFT_REBUILD_MAX_CONTAINERS", 25)),  # workspace <= 80 (owner)
              volumes={"/runs": runs, "/assets": assets.read_only()},
              secrets=[modal.Secret.from_name("nebius-s3")])
def rebuild_one(job: dict) -> dict:
    """One instance: ``job`` = {run, trial, k, game_id, bundle, compare, save}."""
    import shutil
    import traceback

    rclone_env()
    tmp = Path("/tmp") / f"{job['run']}__{job['k']}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    src = f"arc:{BUCKET}/runs/validation/{job['run']}/arena"
    try:
        rclone("copyto", f"{src}/working/trials/{job['trial']}/games/{job['k']}/log.jsonl", str(tmp / "log.jsonl"))
        if job.get("compare"):
            rclone("copyto", f"{src}/working/trials/{job['trial']}/games/{job['k']}/rollout.jsonl",
                   str(tmp / "rollout.jsonl"))
        rclone("copyto", f"{src}/trials/study.json", str(tmp / "study.json"))
        subprocess.run(["rclone", "copyto", f"{src}/serving/cache-layout.json", str(tmp / "cache-layout.json")],
                       capture_output=True)
        layout = json.loads((tmp / "cache-layout.json").read_text()) if (tmp / "cache-layout.json").exists() else None
        if not Path("/tmp/games").exists():
            rclone("copy", f"arc:{BUCKET}/assets/games", "/tmp/games")
        if not _tok:
            _tok.append(Tokenizer(next(d for d in PROCESSORS if Path(d, "tokenizer_config.json").exists())))
        bundle = Path(BUNDLES) / job["bundle"]
        study = json.loads((bundle / "configs/comparison.json").read_text())
        recorded = [json.loads(line) for line in open(tmp / "rollout.jsonl")] if job.get("compare") else None
        rows = read_log(tmp / "log.jsonl")
        game_id = job["k"] if study.get("competition") else job["game_id"]
        out, checks = settle([replay(bundle, study, rows, game_id, tmp / f"work{i}", "/tmp/games", _tok[0], recorded,
                                     layout) for i in range(2)], recorded)
    except Exception as exc:  # noqa: BLE001 - every instance outcome is recorded
        return {"job": job, "checks": {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                       "trace": traceback.format_exc()[-3000:]}}
    if job.get("save"):
        dst = Path(REBUILT) / job["run"] / job["trial"] / job["k"]
        dst.mkdir(parents=True, exist_ok=True)
        with open(dst / "rollout.jsonl", "w") as f:
            for row in out:
                f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
        (dst / "checks.json").write_text(json.dumps(checks, indent=1))
        runs.commit()
    shutil.rmtree(tmp, ignore_errors=True)
    return {"job": job, "checks": checks, "rows": len(out),
            "reply_tokens": sum((r["usage"] or {}).get("completion_tokens") or 0 for r in out if r["gameplay"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gate", "rebuild"))
    ap.add_argument("--jobs", required=True, help="JSON list of jobs (rebuild_one's job dicts)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = json.loads(Path(a.jobs).read_text())
    for j in jobs:
        j["compare"] = a.mode == "gate"
        j["save"] = a.mode == "rebuild"
    with modal.enable_output(), app.run():
        tool = Path(subprocess.run(["git", "-C", str(HERE), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                   capture_output=True, text=True).stdout.strip()).parent / "tools/ops/records.py"
        subprocess.run([sys.executable, str(tool), "cpu-app", app.app_id], check=False)
        done = saved.remote() if a.mode == "rebuild" else {}
        todo = [j for j in jobs if f"{j['run']}/{j['trial']}/{j['k']}" not in done]
        print(f"{len(jobs) - len(todo)} instances already rebuilt; {len(todo)} to run")
        got = dict(zip(map(id, todo), rebuild_one.map(todo, return_exceptions=True)))
    res = []
    for j in jobs:
        c = done.get(f"{j['run']}/{j['trial']}/{j['k']}")
        g = {"job": j, "checks": c, "reply_tokens": c.get("kept_reply_tokens", 0)} if c is not None else got[id(j)]
        res.append(g if isinstance(g, dict) else {"job": j, "checks": {"ok": False, "error": repr(g)[:500]}})
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"jobs": len(res), "ok": sum(bool(r["checks"].get("ok")) for r in res)}))


if __name__ == "__main__":
    main()
