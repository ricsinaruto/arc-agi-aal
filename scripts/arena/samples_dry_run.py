"""CPU dry run of the run-B + parallel_samples + action_plans study through the real arena trial (30 September).

    uv run --extra dev python scripts/arena/samples_dry_run.py [--games 6] [--wall 120] [--output DIR]

The feature bundle's study (scripts/build_arena.py, the launch components) runs ``arena.trial.Trial`` on a few real
offline games (``environment_files``) against a fake vLLM server on localhost: ``/v1/chat/completions`` answers after
a random decode time (cancellable: a client disconnect aborts the handler, as vLLM's ``with_cancellation`` does),
with a python tool call that acts (``action``), plans (``run_plan``) or inspects; ``/tokenize``, ``/detokenize`` and
``/metrics`` are served. The server keeps an engine-style block ledger on the study's served layout (int4, 3,136-token
blocks, 1,268 blocks at 48 sequences): identical prompts in flight share their full prompt blocks, every request owns
its tail, reply and 9 state blocks, and an aborted request's blocks are freed 50 ms after the disconnect. The dry run
fails if that ledger ever exceeds 0.9 x the pool (a preemption on the real engine), if more requests run than
``max_num_seqs``, or if the trial report misses the feature totals. ``min_levels`` is 0 here so every game samples
(offline games at level 0 would otherwise never reach the default of 2).
"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import random
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
# The offline games (git-ignored): in the checkout, or in the main checkout of a .claude/worktrees/<name> worktree.
GAMES_DIR = next((p / "environment_files" for p in (ROOT, *ROOT.parents) if (p / "environment_files").is_dir()),
                 ROOT / "environment_files")
POOL_BLOCKS, BLOCK, FIXED, MAX_LEN = 1268, 3136, 9, 262144
ABORT_LAG = 0.05


def layout():
    rows = []
    for tokens in sorted({1, MAX_LEN, *range(1024, MAX_LEN, 1024)}):
        groups = [-(-tokens // BLOCK), 1, 2, 2, 2, 2]
        rows.append({"tokens": tokens, "blocks": sum(groups), "groups": groups})
    groups = [{"type": "FullAttentionSpec", "block_size": BLOCK, "page_size_bytes": 1, "layers": []},
              {"type": "CircularBufferSpec", "block_size": 4, "page_size_bytes": 1, "layers": []},
              *({"type": "MambaSpec", "block_size": BLOCK, "page_size_bytes": 1, "layers": []} for _ in range(4))]
    return {"schema": 1, "num_blocks": POOL_BLOCKS, "max_model_len": MAX_LEN, "planned_cache_bytes": None,
            "lookahead": 0, "groups": groups, "costs": rows, "rule": "dry run"}


def cost(tokens):
    return -(-(-(-max(1, tokens) // 1024) * 1024) // BLOCK) + FIXED


class FakeEngine:
    def __init__(self, seed=0, seqs=48):
        self.rng = random.Random(seed)
        self.seqs = seqs
        self.running = 0
        self.peak_running = 0
        self.prefixes = {}  # prompt key -> [shared blocks, users]
        self.physical = 0
        self.peak_physical = 0
        self.violations = []
        self.stats = {"requests": 0, "pings": 0, "aborted": 0, "completed": 0, "generated": 0, "prompt": 0,
                      "cached": 0, "run_plan_replies": 0, "action_replies": 0, "inspect_replies": 0}
        self.seen = set()

    margin = 0.9  # the study's kv_margin (0.96 for the campaign blocks arms)

    def use(self, delta):
        self.physical += delta
        self.peak_physical = max(self.peak_physical, self.physical)
        if self.physical > int(self.margin * POOL_BLOCKS):
            self.violations.append({"physical": self.physical, "time": time.monotonic()})

    def reply_code(self, text):
        match = re.search(r"Valid actions right now: ([^\n]+?)\.", text)
        names = [n.strip() for n in (match.group(1).split(",") if match else ["UP"]) if n.strip()]

        def one():
            name = self.rng.choice(names)
            if name == "MOUSE":
                return {"action": "MOUSE", "row": self.rng.randrange(64), "col": self.rng.randrange(64)}
            return name
        roll = self.rng.random()
        if roll < 0.3:
            self.stats["run_plan_replies"] += 1
            plan = [one() for _ in range(self.rng.randint(2, 5))]
            return f"r = run_plan({plan!r})\nprint(r['stop_reason'], r['executed'])"
        if roll < 0.75:
            self.stats["action_replies"] += 1
            return f"print(action([{one()!r}]))"
        self.stats["inspect_replies"] += 1
        return "print(current_frame.shape)"

    async def chat(self, payload):
        text = json.dumps(payload.get("messages"), sort_keys=True)
        prompt = max(1000, len(text) // 3)
        key = hashlib.sha1((payload.get("cache_salt", "") + text).encode()).hexdigest()
        shared = prompt // BLOCK
        entry = self.prefixes.setdefault(key, [shared, 0])
        cached = shared * BLOCK if key in self.seen else 0
        self.seen.add(key)
        self.running += 1
        self.peak_running = max(self.peak_running, self.running)
        if self.running > self.seqs:
            self.violations.append({"running": self.running})
        if entry[1] == 0:
            self.use(entry[0])
        entry[1] += 1
        held = cost(prompt) - shared
        self.use(held)
        self.stats["requests"] += 1
        self.stats["prompt"] += prompt
        self.stats["cached"] += cached
        released = False

        def free():
            nonlocal released
            if released:
                return
            released = True
            self.running -= 1
            self.use(-held)
            entry[1] -= 1
            if entry[1] == 0:
                self.use(-entry[0])
                del self.prefixes[key]

        try:
            cap = int(payload.get("max_tokens") or 16384)
            if cap == 1:
                self.stats["pings"] += 1
                await asyncio.sleep(0.02)
                length = 1
                message = {"role": "assistant", "content": None, "reasoning": "T"}
                finish = "length"
            else:
                length = min(cap, int(self.rng.lognormvariate(6.3, 1.0)))  # median ~550 tokens
                steps = max(1, length // 200)
                for step in range(steps):
                    await asyncio.sleep(0.2 * 200 / 400)  # 400 tok/s per stream in the dry run
                    grown = cost(prompt + (step + 1) * 200) - shared
                    self.use(grown - held)
                    held = grown
                code = self.reply_code(text)
                message = {"role": "assistant", "content": "World model: dry run.", "reasoning": "thinking " * 20,
                           "tool_calls": [{"id": f"call{self.stats['requests']}", "type": "function",
                                           "function": {"name": "python", "arguments": json.dumps({"code": code})}}]}
                finish = "tool_calls"
        except asyncio.CancelledError:
            self.stats["aborted"] += 1

            async def later():
                await asyncio.sleep(ABORT_LAG)
                free()
            asyncio.get_running_loop().create_task(later())
            raise
        free()
        self.stats["completed"] += 1
        self.stats["generated"] += length
        return {"choices": [{"index": 0, "finish_reason": finish, "message": message}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": length, "total_tokens": prompt + length,
                          "prompt_tokens_details": {"cached_tokens": cached}}}


async def serve(engine):
    from aiohttp import web

    async def chat(request):
        payload = await request.json()
        return web.json_response(await engine.chat(payload))

    async def tokenize(request):
        payload = await request.json()
        count = max(1000, len(json.dumps(payload.get("messages"), sort_keys=True)) // 3)
        return web.json_response({"count": count, "max_model_len": MAX_LEN, "tokens": list(range(count))})

    async def detokenize(request):
        return web.json_response({"prompt": ""})

    async def metrics(request):
        s = engine.stats
        text = "\n".join([
            f"vllm:num_requests_running {engine.running}", "vllm:num_requests_waiting 0",
            f"vllm:generation_tokens_total {s['generated']}", "vllm:num_preemptions_total 0",
            f"vllm:kv_cache_usage_perc {engine.physical / POOL_BLOCKS}",
            f"vllm:prompt_tokens_total {s['prompt']}"]) + "\n"
        return web.Response(text=text)

    app = web.Application()
    app.add_routes([web.post("/v1/chat/completions", chat), web.post("/tokenize", tokenize),
                    web.post("/detokenize", detokenize), web.get("/metrics", metrics)])
    runner = web.AppRunner(app, handler_cancellation=True, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def feature_study(tmp):
    from arena import hero_v2

    spec = importlib.util.spec_from_file_location("build_arena_dry", ROOT / "scripts/build_arena.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    components = {"kernels_phase6": False, "prefix_cache": True, "kv_cache": "int4", "focus_oversubscription": True,
                  "prompt_tokens_details": True, "backlog_fix": True, "harness_fixes": True, "window": 130816,
                  "focus_slots": 24, "parallel_samples": True, "action_plans": True, "density_blocks": 1266}
    _, study = build.build("arena-hero-v2-samples-plans-dry", output=tmp / "bundle", paired_30m=True,
                           arena_variant=hero_v2.BASE, hero_v2_served_blocks=1335, opening_burst_requests=3,
                           exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], wall_seconds=7200,
                           clone_instances=110, hero_v2=components)
    return json.loads((tmp / "bundle/configs/comparison.json").read_text())


async def main_async(args):
    from arc_infra.arcade import make_arcade
    from arena import trial as trial_module
    from arena.trial import Trial

    gates = []

    class RecordedGate(trial_module.KVGate):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            gates.append(self)

    trial_module.KVGate = RecordedGate  # the trial's gate, read after the abort-grace timers

    tmp = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="samples-dry-"))
    tmp.mkdir(parents=True, exist_ok=True)
    # --study: a built bundle's comparison.json (e.g. the blocks3-samples2 campaign arm) instead of the run-B build.
    study = json.loads(Path(args.study).read_text()) if args.study else feature_study(tmp)
    arc = make_arcade("offline", GAMES_DIR, tmp / "recordings")
    ids = sorted(g.game_id for g in arc.get_environments())
    study["games"] = ids[: args.games]
    study["wall_seconds"] = float(args.wall)
    study.pop("competition", None)
    study["parallel_samples"] = {**study["parallel_samples"], "min_levels": args.min_levels}
    if study.get("blocks") is not None:
        # The blocks scheduler on the small offline roster: the same block count and partition, the stream count
        # scaled to the roster (a block is the stream count), the dry run's wall.
        count, blocks = len(study["games"]), int(study["blocks"]["blocks"])
        slots = -(-count // blocks)
        study["blocks"] = {**study["blocks"], "roster_games": count, "slots": slots, "wall_seconds": float(args.wall)}
        study["active_limit"] = slots
        study.pop("early_stop", None)
    FakeEngine.margin = float(study.get("kv_margin", 0.9))
    server_dir = tmp / "server"
    server_dir.mkdir(exist_ok=True)
    (server_dir / "server.log").write_text("ARC_CACHE_LAYOUT " + json.dumps(layout()) + "\n")
    engine = FakeEngine(seed=args.seed, seqs=study["server"]["max_num_seqs"])
    runner, port = await serve(engine)
    try:
        trial = Trial(study, study["trials"][0], source=ROOT, games_dir=GAMES_DIR, output=tmp / "trial",
                      server_dir=server_dir, base_url=f"http://127.0.0.1:{port}/v1", arcade=arc,
                      log=lambda *a, **k: None)
        report = await trial.run()
        await asyncio.sleep(1.5)  # the siblings' abort-grace releases (0.5 s) land after the report's snapshot
    finally:
        await runner.cleanup()
    samples, plans = report.get("parallel_samples") or {}, report.get("action_plans") or {}
    result = {"output": str(tmp), "end_reason": report.get("end_reason"), "games": len(study["games"]),
              "engine": {**engine.stats, "peak_running": engine.peak_running, "peak_physical": engine.peak_physical,
                         "gate_capacity": int(engine.margin * POOL_BLOCKS), "violations": engine.violations[:5],
                         "physical_after": engine.physical},
              "gate": report.get("gate"),
              "gate_after_grace": {"reserved": gates[0].reserved, "inflight": gates[0].inflight} if gates else None,
              "samplers_after": [{"game": w.game.game_id, "live": [(x.index, x.status, x.released, x.task.done())
                                                                   for x in w.sampler.live],
                                  "background": len(w.sampler.background)}
                                 for w in getattr(trial, "duck_workers", []) if w.sampler is not None],
              "parallel_samples": samples, "action_plans": plans,
              "task_errors": report.get("task_errors"),
              "game_errors": [r.get("error") for r in report.get("rows", []) if r.get("error")]}
    problems = []
    if engine.violations:
        problems.append("engine ledger exceeded the gate capacity or max_num_seqs")
    if not samples.get("groups"):
        problems.append("no parallel-sample group ran")
    if not plans.get("plan_calls"):
        problems.append("no run_plan call was recorded")
    if not gates or gates[0].reserved or gates[0].inflight:
        problems.append("gate reservations left after the abort grace")
    if report.get("task_errors"):
        problems.append("task errors")
    result["problems"] = problems
    (tmp / "dry-run.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--wall", type=float, default=120)
    ap.add_argument("--min-levels", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output")
    ap.add_argument("--study", help="a built bundle's configs/comparison.json (default: the run-B feature build)")
    args = ap.parse_args()
    from arena import hero_v2

    hero_v2._token_clock = lambda: True  # the token-clock scheduler is merged (the check predates the key)
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
