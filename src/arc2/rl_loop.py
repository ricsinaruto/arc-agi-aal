"""RL on a broad ARC pool (docs/arc2-arm.md, "RL with a verifiable reward"): GRPO rounds of ``arc2.loop`` attempts
against the served model and its current LoRA, the reward exact on held-out pairs, updates by the RL arm's trainer
service (``arena.rl_loop.Trainer``; ``rl1_trainer.py`` with the routed experts frozen). Trial kind ``arc2-rl``.

A round: ``tasks`` band tasks (base pass rate strictly between 0 and 1) x ``group`` episodes; an episode is one attempt
(``max_attempts`` 1) capped at ``attempt_replies`` replies of at most ``cap`` tokens. Reward 1 when the attempt's
verified program also reproduces every held-out pair (``arc2.rl_pool.reward``), else 0. Advantage: reward minus the
group mean (no std division); groups with no spread are dropped. The server sleeps through the updates (one card).

Every reply is recorded with its sampled token ids and the engine's top log-probabilities (``RLTransport``) in the
episode's ``rollout.jsonl``, the format ``rl1_trainer`` reads (``arena.rollout_record``)."""

import asyncio
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

from . import rl_pool, sandbox
from .loop import Engine, TaskRun

DEFAULTS = {
    "pool": "data/rl/pool.jsonl", "eval_ids": None, "eval_pool": None, "screen_from": None,  # eval_ids: pool ids held out of training (a JSON list file)
    "screen_tasks": 200, "screen_samples": 4, "tasks": 16, "group": 8, "rounds": 2, "per_update": 16,
    "effort": "medium", "cap": 16384, "attempt_replies": 3, "images": True, "image_side": 224, "temperature": 0.7,
    "concurrency": 84, "lr": 2.5e-5, "clip": 1.0, "max_windows": 64, "eval_tasks": 0, "eval_samples": 2,
    "trainer_timeout": 5400, "update_timeout": 3600, "reserve_seconds": 900, "dense_only": True, "sandbox_pythons": [],
    # trainer_command (arena.rl_loop): trainer_python, trainer_script, model_dir, rank, optimizer, loop_lora, tis_cap;
    # train_window: its --window (the play window is the study's)
    "train_window": 32768,
    # extra update fields (rl1_trainer.op_update): shared prompt prefix per task, batched tails, as the RL arm's mg5g
    "update_extra": {"share_prefix": True, "batch_tokens": 32768, "batch_max": 8},
    "profile": False,  # one profiled window pass after the first round's updates (rl1_trainer.op_profile)
    # long run: band = 0 < rate < 1 and rate <= band_hi over a task's last rate_window rewards
    "band_hi": 1.0, "rate_window": 16,
    "screen_prior": None,  # a folder of an earlier screen's episodes (<task key>/<k>/episode.json + rollout.jsonl)
    "round0_from_screen": False,  # round 0 trains on the screen groups of its picked band tasks (no round-0 play)
    "eval_every": 0,  # eval_pool every this many rounds (with the adapter of the round's play), beside the play
    "eval_seconds": 1800,  # the final eval's estimate (the deadline check before another round)
    "probes": 0, "kl_stop": 0.05,  # probe windows (half outside the band): KL to base after every round; stop above
    "checkpoint": False,  # adapter + trainer state + state.json every round under $ARC_VALIDATION_OUTPUT/ckpt
    "adapter_name": None,  # None: r00, r01, ...; else one served name swapped each round
    "trainer_env": {},  # extra trainer environment (split_gemm knobs: RL1_SWITCH_INTERVAL, RL1_PREFETCH)
    # long-v2 (18 Oct): ~1,100 episodes at once stalled the server's frontend; at most this many episodes run at once
    # (0: all), so they finish as they go; eval_prior: an earlier base eval's episodes (missing ones replayed)
    "episode_cap": 0, "eval_prior": None, "local_count": False,
    # a round's training play ends this many seconds in (0: none); unfinished episodes are cut and left out (the
    # slowest few of 128, up to 15 sequential requests each, set the round's length)
    "round_play_seconds": 0,
    # c240 split: prior_hist, {task: [rewards]} of earlier base screens, seeds the rates of tasks the run
    # has none for (a resume onto a new pool); eval_base_fresh: on a resume, the base eval of this run's eval_pool runs
    # beside the first round's play (the old run's evals were on another set); try_unsolved: tasks whose only rewards
    # are at most this many zeros count as band until they get more
    "prior_hist": None, "eval_base_fresh": False, "try_unsolved": 0,
    # pilots (docs/posttrain-practice.md items 10, 15, 13.5; the RL arm's mg5 controller):
    # advantage "loo" (reward minus the mean of the group's others); push_balance: negatives scaled so the push-down
    # mass (|weight| x completion tokens) equals the push-up mass in each round; kl_target: after every optimizer
    # step the probes' KL to the last accepted state, above its floor: past kl_rollback_at x target the update is
    # undone, past kl_stop_at x target it stops; lr / 1.5 after either, x 1.5 (to lr_max) after a full update under
    # target / 1.5; probe_rollouts: probe windows given as recorded rollouts (untrained tasks); fixed_tasks: a JSON
    # list of pool ids played every round (no screen); update_rounds: rounds from this number on play only
    "advantage": "mean", "push_balance": False, "kl_target": None, "kl_stop_at": 1.5, "kl_rollback_at": 3.0,
    "lr_max": 1e-4, "lr_min": 2.5e-6, "probe_rollouts": [], "fixed_tasks": None, "update_rounds": None,
    # kl_floor: the probes' perturbation floor from a dose run (KL after any non-zero update, flat in lr and steps:
    # bf16 rounding / routing flips), a number or {probe name: number}; each probe's floor is the larger of it and
    # the identical-weights floor
    "kl_floor": 0.0,
    # lr_rule: "step" (x 1.5 under target / 1.5, / 1.5 after a rollback or stop) or "dose":
    # KL grows with lr squared, so the next round's lr is lr x sqrt(kl_target / the round's excess), within
    # [lr_min, lr_max] (a play rollback, which measures no excess, still divides by 1.5)
    "lr_rule": "step",
    # looser drift rules: kl_stop_at None (no per-step stop); kl_rollback_excess: an update is
    # undone only when the guard probes' excess passes this many nats (in place of kl_rollback_at x target);
    # play_guard: the round's play against round 0's base play: mean reply tokens or cap-hit share over ratio x base,
    # or format / tool-call failures up sharply (no-code reply share over failures_ratio x base and base + failures_points) -> the last update is
    # undone (the state before it), the round trains nothing, lr / 1.5; kl_stop None: no run stop on kl_base
    "kl_rollback_excess": None, "play_guard": None,
    # trained_probes: after round 0, one success and one failure (the shortest) of this many trained tasks join the
    # probes as won-/lost- windows (base references taken before the first update; not in the controller's KL): a
    # reward-free check that updates push trained successes up and failures down
    "trained_probes": 0,
    # held_successes: the shortest success of every trained task with >= 2 successes is held
    # out of training once (held-<task>), its trainer NLL scored before its task's first update and after every round
    # (flattening: the KL controller cannot see it); not in the controller
    "held_successes": False,
    # signal yield: spread_prior {source: [a, b]} (Beta priors fitted on
    # unselected base screens): band = posterior P(0.1 <= p <= 0.9) over a task's last rate_window rewards >=
    # spread_min (re-estimated every round, so saturated tasks drop); a round plays tasks + oversample groups, then up
    # to extra_play more batches from unplayed band tasks while fewer than min_spread groups have spread; at most
    # `tasks` groups with spread are trained
    "spread_prior": None, "spread_min": 0.0, "oversample": 0, "min_spread": 0, "extra_play": 1,
    # an episode's completion-token cap (checked between replies; 0: none), from solved lengths ~p90;
    # a capped episode is a valid failure
    "episode_tokens": 0,
    # signed False: positives only (the trainer drops w <= 0: TOPR's positive branch, the
    # negatives are what flattened P1); warmup_steps: the first optimizer steps of the run at lr x step / warmup_steps
    "signed": True, "warmup_steps": 0,
    # resume_lr: a resumed run's lr from its first round on, in place of the state's (a dose
    # set from the measured KL: KL grows with lr^2)
    "resume_lr": None,
}
# Keys read outside DEFAULTS: arena.rl_loop.trainer_command's, and the screen/sampling switches read with .get.
TRAINER_KEYS = {"trainer_python", "trainer_script", "model_dir", "rank", "optimizer", "loop_lora", "tis_cap", "attention",
                "window"}
OPTIONAL_KEYS = {"sample_major", "screen_have", "screen_shuffle"}


def play_broken(replies, base, guard):
    """The play-breakage rules (``play_guard``) that fire for a round's reply stats against the base round's; {} if none.
    Format / tool-call failures: the share of replies with no program or tool call (``no_code_share``)."""
    guard = {"ratio": 2.0, "failures_ratio": 2.0, "failures_points": 0.05, **(guard if isinstance(guard, dict) else {})}
    out = {}
    for key in ("mean_tokens", "at_cap_share"):
        now, then = replies.get(key), base.get(key)
        if now is not None and then and now > guard["ratio"] * then:
            out[key] = [now, then]
    now, then = replies.get("no_code_share"), base.get("no_code_share")
    if now is not None and then is not None and now > max(guard["failures_ratio"] * then, then + guard["failures_points"]):
        out["no_code_share"] = [now, then]
    return out


def config(study, trial):
    """The run's arc2_rl settings; a key no code reads refuses (19 Oct: PTH lost a play check to an ignored key)."""
    given = {**study.get("arc2_rl", {}), **trial.get("arc2_rl", {})}
    unknown = sorted(set(given) - set(DEFAULTS) - TRAINER_KEYS - OPTIONAL_KEYS)
    if unknown:
        raise ValueError(f"arc2_rl keys no code reads: {unknown}")
    return {**DEFAULTS, **given}

SETTINGS = {"tool": True, "augment": 1, "confirm_attempts": 0, "max_attempts": 1}


def digest(messages):
    return [hashlib.sha1(json.dumps(m, sort_keys=True).encode()).digest() for m in messages]


class RLTransport:
    """The trial's transport without the gate: ``count`` by /tokenize, ``complete`` with the rollout fields asked and
    every reply recorded in the episode's file. ``model``: the served name (base or a loaded adapter)."""

    def __init__(self, client, model, served_model, effort, temperature, concurrency):
        self.client, self.model, self.served_model = client, model, served_model
        self.effort, self.temperature = effort, temperature
        self.slots = asyncio.Semaphore(int(concurrency))
        self.files, self.previous, self.counts = {}, {}, {}
        self.stats = {"replies": 0, "tokens": 0, "without_logprobs": 0, "count_mismatches": 0}
        # local_count (long-v3, 19 Oct): after the first reply, an attempt's count is the last reply's usage plus an
        # estimate of the messages added since, instead of /tokenize (the server's one frontend process renders every
        # image prompt; counting each turn doubled that load and it stalled 1-2 min at a time)
        self.local_count, self.known = False, {}

    def kwargs(self, settings):
        return {"enable_thinking": True, "reasoning_effort": settings.get("effort") or self.effort}

    async def count(self, messages, settings):
        from . import tasks

        known = self.known.get(id(messages)) if self.local_count else None
        if known and len(messages) >= known[0] and digest(messages[:1]) == known[2]:
            self.stats["estimated"] = self.stats.get("estimated", 0) + 1
            return known[1] + sum(estimate_tokens(m) for m in messages[known[0]:])

        response = await self.client.post("/tokenize", json={
            "model": self.served_model, "messages": messages, "chat_template_kwargs": self.kwargs(settings),
            **({"tools": [tasks.PYTHON_TOOL]} if settings.get("tool") else {})}, timeout=300)
        response.raise_for_status()
        return int(response.json()["count"])

    async def complete(self, task_id, messages, count, max_tokens, settings):
        from arena.rollout_record import ids_digest, parse_logprobs

        from . import tasks
        from .loop import ContextFull

        payload = tasks.chat_request(self.model, messages, max_tokens,
                                     temperature=settings.get("temperature") or self.temperature,
                                     effort=self.kwargs(settings)["reasoning_effort"],
                                     tools=[tasks.PYTHON_TOOL] if settings.get("tool") else None)
        extras = ({"logprobs": True, "top_logprobs": 20, "return_tokens_as_token_ids": True, "return_token_ids": True}
                  if getattr(self, "record_logprobs", True) else {"return_token_ids": True})
        async with self.slots:
            t0 = time.monotonic()
            response = await self.client.post("/v1/chat/completions", json={**payload, **extras}, timeout=3600)
        if response.status_code == 400 and "context" in response.text.lower():
            raise ContextFull(response.text[:200])
        response.raise_for_status()
        data = response.json()
        choice, usage = data["choices"][0], data.get("usage") or {}
        tokens, values, top = parse_logprobs((choice.get("logprobs") or {}).get("content"), 20)
        if choice.get("token_ids") and not tokens:
            tokens = list(choice["token_ids"])
        marks = digest(messages)
        keep = 0
        for old, new in zip(self.previous.get(task_id) or [], marks):
            if old != new:
                break
            keep += 1
        row = {"n": self.counts.get(task_id, 0), "step": 0, "gameplay": bool(payload.get("tools")), "keep": keep,
               "messages": messages[keep:], "template": payload.get("chat_template_kwargs") or {},
               "sampling": {**{k: payload.get(k) for k in ("temperature", "top_p", "top_k", "max_tokens")},
                            **{k: payload[k] for k in ("min_p", "presence_penalty", "frequency_penalty",
                                                       "repetition_penalty") if k in payload}},
               "reply": choice["message"], "finish_reason": choice.get("finish_reason"), "usage": usage,
               "prompt_digest": ids_digest(data["prompt_token_ids"]) if data.get("prompt_token_ids") else None,
               "tokens": tokens, "logprobs": values, "top": top}
        if task_id not in self.previous:
            row["tools"] = payload.get("tools")
        with open(self.files[task_id], "a") as stream:
            stream.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
        self.previous[task_id] = marks
        if usage.get("prompt_tokens") is not None:  # covers the prompt and this reply (appended next by the engine)
            self.known[id(messages)] = (len(messages) + 1, int(usage["prompt_tokens"]) + int(usage.get("completion_tokens") or 0),
                                        digest(messages[:1]))
        self.counts[task_id] = self.counts.get(task_id, 0) + 1
        self.stats["replies"] += 1
        self.stats["count_mismatches"] += len(tokens) != usage.get("completion_tokens")
        self.stats["tokens"] += len(tokens)
        self.stats["without_logprobs"] += not values
        message = {k: v for k, v in choice["message"].items() if k not in ("token_ids",)}
        return {"message": message, "finish_reason": choice.get("finish_reason"), "usage": usage, "t0": t0,
                "t1": time.monotonic()}


def estimate_tokens(message):
    """Rough tokens of one chat message (an over-estimate: ~3 characters a token, 300 an image, 16 of framing)."""
    parts = message.get("content")
    parts = [{"type": "text", "text": parts}] if isinstance(parts, str) else (parts or [])
    text = sum(len(p.get("text") or "") for p in parts if isinstance(p, dict) and p.get("type") == "text")
    images = sum(1 for p in parts if isinstance(p, dict) and p.get("type") in ("image_url", "image"))
    calls = len(json.dumps(message.get("tool_calls"))) if message.get("tool_calls") else 0
    return 16 + (text + calls) // 3 + 300 * images


def episode_reward(row, run):
    """1.0 when the attempt's first verified program is also right on every held-out pair; the program's outputs on
    the held-out inputs are the candidate's ``tests`` (the episode runs on copy 0)."""
    verified = next((c for c in run.candidates if c["verified"] and not c.get("review")), None)
    if verified is None:
        return 0.0
    return float(all(t == p["output"] for t, p in zip(verified["tests"], row["holdout"]))
                 and len(verified["tests"]) == len(row["holdout"]))


def advantages(groups, loo=False):
    """``{task: [rewards]}`` -> ``{task: [advantages]}`` for groups with spread (reward minus the group mean;
    ``loo``: minus the mean of the other rewards, n / (n - 1) times as large)."""
    out = {}
    for task, rewards in groups.items():
        if max(rewards) > min(rewards):
            mean = sum(rewards) / len(rewards)
            scale = len(rewards) / (len(rewards) - 1) if loo else 1.0
            out[task] = [round(scale * (r - mean), 6) for r in rewards]
    return out


def in_band_probability(rewards, prior, lo=0.1, hi=0.9, steps=400):
    """P(lo <= p <= hi) under Beta(a + successes, b + failures), by the midpoint rule."""
    a, b = prior
    k = sum(rewards)
    A, B = a + k, b + len(rewards) - k
    xs = [(i + 0.5) / steps for i in range(steps)]
    logs = [(A - 1) * math.log(x) + (B - 1) * math.log(1 - x) for x in xs]
    top = max(logs)
    w = [math.exp(v - top) for v in logs]
    return sum(wi for x, wi in zip(xs, w) if lo <= x <= hi) / sum(w)


def balance(specs):
    """Push balance (posttrain-practice item 15): negative weights scaled so sum |w| x tokens over the push-down
    episodes equals the push-up sum (``tokens``: the episode's completion tokens). Returns the scale."""
    up = sum(s["w"] * s["tokens"] for s in specs if s["w"] > 0)
    down = sum(-s["w"] * s["tokens"] for s in specs if s["w"] < 0)
    if not up or not down:
        return 1.0
    scale = up / down
    for s in specs:
        if s["w"] < 0:
            s["w"] = round(s["w"] * scale, 6)
    return round(scale, 4)


async def play(rows, samples, *, transport, execute, explore, cfg, folder, label, deadline=None, offsets=None):
    """``samples`` episodes of every row: ``{task id: [(reward, rollout path, tokens)]}``. ``offsets``: episodes a task
    already has elsewhere (numbering and the sample-major pass start after them)."""
    offsets = offsets or {}
    settings = {**SETTINGS, "effort": cfg["effort"], "cap": cfg["cap"], "attempt_replies": cfg["attempt_replies"],
                "images": cfg["images"], "image_side": cfg["image_side"], "temperature": cfg["temperature"]}

    counts = samples if isinstance(samples, dict) else {row["id"]: int(samples) for row in rows}

    async def one(row, k):
        if getattr(transport, "episodes", None) is not None:
            async with transport.episodes:
                return await episode(row, k)
        return await episode(row, k)

    async def episode(row, k):
        key = f"{label}/{row['id'].replace(':', '_')}/{k + offsets.get(row['id'], 0)}"
        path = Path(folder) / key / "rollout.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        transport.files[key] = path
        engine = Engine(complete=transport.complete, count=transport.count, execute=execute,
                        window=cfg["window"], cap=cfg["cap"], settings=settings, explore=explore, deadline=deadline)
        run = TaskRun(key, rl_pool.model_view(row))
        failed = False
        budget = int(cfg.get("episode_tokens") or 0) or 10 ** 9
        try:
            await engine.advance(run, budget)
        except Exception as exc:  # noqa: BLE001 - a failed episode is recorded and left out
            (path.parent / "error.txt").write_text(f"{type(exc).__name__}: {exc}"[:2000])
            failed = True
        if not failed and run.done is None and run.spent >= budget:
            run.done = "cap"  # the episode's token cap: a failure that counts (not a deadline cut)
        reward = episode_reward(row, run)
        # left out of groups and rates: request errors and cut-offs are not failures (pipe-v3's screen: 143 of 256
        # zeros were episodes the deadline cut; long-v2: 32 of the first 1,000 ended on request errors)
        valid = valid_episode(run.done, failed)
        (path.parent / "episode.json").write_text(json.dumps({"task": row["id"], "reward": reward, "spent": run.spent,
                                                              "attempts": run.attempts, "done": run.done, "valid": valid,
                                                              "verified": any(c["verified"] for c in run.candidates),
                                                              "reply_tokens": [int(r.get("completion_tokens") or 0)
                                                                               for r in run.replies],
                                                              "at_cap": sum(1 for r in run.replies
                                                                            if r.get("finish") == "length"),
                                                              "no_code": sum(1 for r in run.replies
                                                                             if r.get("outcome") == "no_code")}))
        if not valid:
            transport.stats["invalid"] = transport.stats.get("invalid", 0) + 1
            return row["id"], None
        return row["id"], (reward, str(path), run.spent)

    if cfg.get("sample_major"):  # every task's first episode before any second (a cut keeps whole passes)
        order = sorted(((row, k) for row in rows for k in range(counts.get(row["id"], 0))),
                       key=lambda item: item[1] + offsets.get(item[0]["id"], 0))  # stable: pool order within a pass
    else:
        order = [(row, k) for row in rows for k in range(counts.get(row["id"], 0))]
    results = await asyncio.gather(*(one(row, k) for row, k in order))
    out = {}
    for task, item in results:
        if item is not None:
            out.setdefault(task, []).append(item)
    return out


def reply_stats(paths):
    """Mean reply length, share of replies at the reply cap, and among failures the share that used up their replies
    (vs the episode token cap), from the round's episode records."""
    lengths, at_cap, fails, out_of_replies, capped, no_code = [], 0, 0, 0, 0, 0
    for p in paths:
        try:
            e = json.loads((Path(p).parent / "episode.json").read_text())
        except (OSError, ValueError):
            continue
        lengths += e.get("reply_tokens") or []
        at_cap += int(e.get("at_cap") or 0)
        no_code += int(e.get("no_code") or 0)
        if not e.get("reward"):
            fails += 1
            out_of_replies += e.get("done") == "replies"
            capped += e.get("done") == "cap"
    return {"replies": len(lengths), "mean_tokens": round(sum(lengths) / max(1, len(lengths))),
            "at_cap_share": round(at_cap / max(1, len(lengths)), 3), "failures": fails,
            "no_code_share": round(no_code / max(1, len(lengths)), 3),  # replies with no program or tool call
            "out_of_replies_share": round(out_of_replies / max(1, fails), 3),
            "episode_cap_share": round(capped / max(1, fails), 3)}


def valid_episode(done, failed=False):
    """An episode counts when its attempt ended on its own: a verified program or its replies used up (not a
    deadline cut, ``done`` None, nor request errors)."""
    return not failed and done not in (None, "errors")


def load_prior(folder):
    """An earlier screen's episodes: ``{task id: [(reward, rollout path, tokens)]}``."""
    out = {}
    for path in sorted(Path(folder).glob("*/*/episode.json")):
        episode = json.loads(path.read_text())
        if not valid_episode(episode.get("done"), (path.parent / "error.txt").exists()):
            continue
        out.setdefault(episode["task"], []).append((float(episode["reward"]), str(path.parent / "rollout.jsonl"),
                                                     int(episode.get("spent") or 0)))
    return out


def window_stats(windows):
    """Trainer-vs-server log-probability mismatch over an update's windows (``rl1_trainer.mismatch``)."""
    rows = [w.get("mismatch") or {} for w in windows]
    tokens = sum(r.get("tokens", 0) for r in rows)
    if not tokens:
        return {"windows": len(windows), "tokens": 0}
    mean = lambda key: round(sum(r.get(key, 0) * r.get("tokens", 0) for r in rows) / tokens, 5)  # noqa: E731
    return {"windows": len(windows), "tokens": tokens, "mean_abs": mean("mean_abs"), "mean": mean("mean"),
            "beyond_half": mean("beyond_half"), "trained": sum(int(w.get("trained") or 0) for w in windows)}


async def run_arc2_rl(study, trial, *, source, output, server_dir, base_url, deadline_epoch, python=None):
    """The loop. Resume: ``$ARC2_RL_RESUME``, an earlier run's ckpt folder (its state.json: band, probes, the last
    round's trainer state and adapter); the base KL references are scored again on the fresh trainer first."""
    import httpx

    from arena.rl_loop import Server, Trainer, trainer_command

    cfg = config(study, trial)
    cfg["window"] = int(study["windows"][0])  # the play window; the trainer's is train_window
    output, source = Path(output), Path(source)
    work = Path("/tmp/arc2-rl")
    work.mkdir(parents=True, exist_ok=True)
    feed = output / "rl"
    feed.mkdir(parents=True, exist_ok=True)
    ckroot = Path(os.environ.get("ARC_VALIDATION_OUTPUT") or output) / "ckpt"
    resume = os.environ.get("ARC2_RL_RESUME")
    report = {"status": "running", "settings": cfg, "rounds": [], "started_epoch": time.time(), "evals": []}
    from .gpu import GpuSampler

    gpu = GpuSampler()

    def save():
        report["updated_epoch"] = time.time()
        report["gpu"] = gpu.summary()
        text = json.dumps(report, indent=1, default=str)
        (output / "rl-report.json").write_text(text)
        (feed / "status.json").write_text(text)  # the launcher mirrors rl/status.json to the volume (live watch)

    def mark(phase):
        gpu.mark(phase)
        report.update(phase=phase, phase_epoch=time.time())
        save()

    def until(reserve):
        """The engines' monotonic deadline: ``reserve`` seconds before the trial's end."""
        return time.monotonic() + (deadline_epoch - reserve - time.time())

    def stop_requested():
        flag = os.environ.get("ARENA_STOP_REQUEST")
        return bool(flag) and Path(flag).exists()

    tried = []
    for python in [python] + list(cfg.get("sandbox_pythons") or []):  # the Axolotl venv when this one lacks scipy
        check = sandbox.self_test(python)
        if not check["ok"]:  # as arc2.trial: the looser worker when the strict one cannot run here
            check = {**sandbox.self_test(python, strict=False), "first": check}
        tried.append({"python": python, "ok": check["ok"]})
        if check["ok"]:
            break
    check["tried"] = tried
    (output / "sandbox-check.json").write_text(json.dumps(check, indent=1, default=str))
    if not check["ok"]:
        report.update(status="error", error="sandbox self-test failed")
        save()
        return report
    strict = check["strict"]
    runner = asyncio.Semaphore(32)

    async def execute(code, inputs, **kw):
        async with runner:
            return await asyncio.to_thread(sandbox.run, code, inputs, python=python, strict=strict, **kw)

    async def explore(cells, code, task, **kw):
        async with runner:
            return await asyncio.to_thread(sandbox.run_cells, cells, code, task, python=python, strict=strict, **kw)

    pool = [json.loads(line) for line in open(source / cfg["pool"])]
    held = set(json.loads((source / cfg["eval_ids"]).read_text())) if cfg.get("eval_ids") else set()
    train_pool = [r for r in pool if r["id"] not in held]
    eval_pool = ([json.loads(line) for line in open(source / cfg["eval_pool"])] if cfg.get("eval_pool")  # h240
                 else [r for r in pool if r["id"] in held])[: int(cfg["eval_tasks"])]
    if {r["id"] for r in eval_pool} & {r["id"] for r in train_pool}:
        raise ValueError("eval tasks in the training pool")
    rng = random.Random(0)
    served = study.get("served_model") or "Qwen/Qwen3.8-Flash-Next-NVFP4"
    server = Server(base_url, served)
    client = httpx.AsyncClient(base_url=server.root, trust_env=False, timeout=3600,  # pipe-v3: the default 100-connection pool held play under ~97 running
                               limits=httpx.Limits(max_connections=None, max_keepalive_connections=None))
    command, env = trainer_command({**cfg, "window": int(cfg["train_window"])}, source, work)
    if os.environ.get("ARC2_PREFLIGHT"):  # launch gate: the scripted trainer (arc2.preflight) on a CPU container
        command = [sys.executable, "-m", "arc2.preflight", "trainer", "--dir", str(Path(work) / "trainer")]
        env = {**env, "PYTHONPATH": str(Path(source) / "src")}
    if cfg["dense_only"]:  # rl1_trainer: no routed-expert adapters (the served stacks prune and requantize them)
        env["RL1_DENSE_ONLY"] = "1"
    env.update({k: str(v) for k, v in (cfg.get("trainer_env") or {}).items()})
    reserve = float(cfg["reserve_seconds"])
    trainer = None
    hist, base_rates, probes, floor, held_ref = {}, {}, [], {}, {}  # hist: task -> rewards (latest last); base_rates: the screen's
    lr = float(cfg["lr"])
    steps_done = 0  # optimizer steps sent in this run and the runs it resumes (the warm-up's clock)
    fixed = ([r for r in train_pool if r["id"] in set(json.loads((source / cfg["fixed_tasks"]).read_text()))]
             if cfg.get("fixed_tasks") else None)

    def rate(task):
        window = hist[task][-int(cfg["rate_window"]):]
        return sum(window) / len(window)

    def in_band(task):
        if cfg.get("spread_prior"):
            prior = cfg["spread_prior"].get(task.split(":")[0]) or cfg["spread_prior"].get("default")
            return bool(prior) and in_band_probability(hist.get(task, [])[-int(cfg["rate_window"]):], prior) \
                >= float(cfg["spread_min"])
        if task in hist and rate(task) == 0 and len(hist[task]) <= int(cfg["try_unsolved"] or 0):
            return True
        return task in hist and 0 < rate(task) < 1 and rate(task) <= float(cfg["band_hi"])

    def transport_for(model):
        transport = RLTransport(client, model, served, cfg["effort"], cfg["temperature"], cfg["concurrency"])
        transport.episodes = asyncio.Semaphore(int(cfg["episode_cap"])) if int(cfg["episode_cap"] or 0) else None
        transport.local_count = bool(cfg.get("local_count"))
        return transport

    async def serve_again():
        """Park the trainer's state in host memory, then wake the server (arena.rl_loop.Loop.wake)."""
        park = await trainer.must("park")
        return {"park_gib": park.get("gib"), "park_seconds": park.get("seconds"), "wake_seconds": await server.wake()}

    async def evaluate(label, transport, prior=None):
        began = time.time()
        prior = prior or {}
        need = {r["id"]: max(0, int(cfg["eval_samples"]) - len(prior.get(r["id"], []))) for r in eval_pool}
        ev = await play(eval_pool, need, transport=transport, execute=execute, explore=explore,
                        cfg=cfg, folder=feed, label=label, deadline=until(reserve / 2))
        ev = {r["id"]: prior.get(r["id"], [])[: int(cfg["eval_samples"])] + ev.get(r["id"], []) for r in eval_pool}
        ev = {k: v for k, v in ev.items() if v}
        rewards = [r for items in ev.values() for r, _, _ in items]
        mean = sum(rewards) / max(1, len(rewards))
        return {"label": label, "mean": round(mean, 4), "episodes": len(rewards),
                "se": round((mean * (1 - mean) / max(1, len(rewards) - 1)) ** 0.5, 4),
                "tasks_any": sum(any(r for r, _, _ in items) for items in ev.values()),
                "seconds": round(time.time() - began)}

    async def base_kl():
        """Probe references at the base (the trainer's adapter is still zero), then the pass-to-pass floor."""
        if not probes or floor:
            return
        await trainer.must("score", windows=probes, label="base")
        again = (await trainer.must("score", windows=probes))["windows"]
        given = cfg["kl_floor"] or 0.0
        floor.update({name: max(float(given.get(name, 0.0) if isinstance(given, dict) else given),
                                float(row.get("kl_base") or 0.0)) for name, row in again.items()})
        report["kl_floor"] = dict(floor)

    async def restore(path):
        """``load_checkpoint``, then every probe scored at the restored state: the trainer forgets each probe's
        unaccepted score on a load (rl1_trainer.forget_unaccepted), and an ``accept_reference`` with any probe
        unscored fails (P5b crashed so after its resume); the score also makes the restored state the probes'
        KL-to-previous reference, as the state the server plays."""
        loaded = await trainer.must("load_checkpoint", path=str(path))
        if probes:
            await trainer.must("score", windows=probes)
        return loaded

    def write_state(number, adapter_path, state_path):
        state = {"round": number, "adapter": str(adapter_path), "state": str(state_path), "hist": hist, "lr": lr,
                 "steps": steps_done, "base_replies": report.get("base_replies"), "held_ref": dict(held_ref),
                 "base_rates": base_rates, "probes": probes, "eval_base": report.get("eval_base"),
                 "rounds": [{k: r.get(k) for k in ("round", "mean_reward", "lift", "groups_with_spread", "probes")}
                            for r in report["rounds"]], "evals": report["evals"], "epoch": time.time()}
        tmp = ckroot / "state.json.tmp"
        tmp.write_text(json.dumps(state, indent=1, default=str))
        tmp.replace(ckroot / "state.json")
        for old in sorted(ckroot.glob("r*/state")):  # the last two trainer states stay; adapters stay
            if old.parent.name not in (f"r{number:02d}", f"r{number - 1:02d}"):
                shutil.rmtree(old, ignore_errors=True)

    try:
        mark("trainer_load")
        report["sleep_for_trainer_seconds"] = await server.sleep()  # the trainer loads beside a sleeping server
        began = time.time()
        trainer = Trainer(work / "trainer", command, env, output / "trainer.log")
        state = await trainer.ready(cfg["trainer_timeout"])
        report["trainer"] = {"seconds": round(time.time() - began, 1), "loaded": state.get("loaded")}
        adapter, start, groups0 = None, 0, {}
        name = lambda number: cfg["adapter_name"] or f"r{number:02d}"  # noqa: E731
        if resume:
            old = json.loads((Path(resume) / "state.json").read_text())
            hist.update(old["hist"])
            base_rates.update(old["base_rates"])
            probes.extend(old["probes"])
            report["eval_base"], report["evals"] = old.get("eval_base"), list(old.get("evals") or [])
            if cfg["eval_base_fresh"]:
                report["eval_base"], report["evals"] = None, []
            report["prior_rounds"] = old.get("rounds")
            lr = float(cfg.get("resume_lr") or old.get("lr") or lr)
            steps_done = int(old.get("steps") or 0)
            # the play guard's base and the held successes' references: the state's, else the old run's live
            # status (states written before 19 Oct 21:00 UTC lack them)
            status = Path(resume).parent / "arena/trials/rl-loop/rl/status.json"
            earlier = json.loads(status.read_text()) if status.is_file() else {}
            if old.get("base_replies") or earlier.get("base_replies"):
                report["base_replies"] = old.get("base_replies") or earlier.get("base_replies")
            held_ref.update(old.get("held_ref") or earlier.get("held_ref") or {})
            if held_ref:
                report["held_ref"] = dict(held_ref)
            await base_kl()  # before the state is loaded: the adapter is zero
            if cfg.get("kl_target") is not None:  # the state before the old run's first update: a play-guard rollback
                pre = work / "pre-update"         # in the first resumed round returns to it, as in a fresh run
                shutil.rmtree(pre, ignore_errors=True)
                await trainer.must("checkpoint", path=str(pre))
            loaded = await restore(old["state"])
            report["serve"] = await serve_again()
            start = int(old["round"]) + 1
            adapter = name(start - 1)
            report["resumed"] = {"from": resume, "round": old["round"], "loaded": loaded,
                                 "load_seconds": await server.load(adapter, old["adapter"])}
            save()
        else:
            report["serve"] = await serve_again()
            mark("screen")
            base = transport_for(served)
            began = time.time()
            evaluation = None
            if fixed is not None:  # pilots: the same tasks every round, no screen
                screen = []
                if cfg.get("screen_prior"):  # an earlier run's base play of these tasks (a relaunch keeps its round 0)
                    prior = load_prior(cfg["screen_prior"] if Path(str(cfg["screen_prior"])).is_absolute()
                                       else source / cfg["screen_prior"])
                    groups0 = {r["id"]: prior.get(r["id"], [])[: int(cfg["group"])] for r in fixed}
                    hist.update({task: [x[0] for x in items] for task, items in groups0.items() if items})
                    report["screen_prior_episodes"] = sum(len(v) for v in groups0.values())
                if eval_pool:
                    evaluation = await evaluate("eval-base", base)
            elif cfg.get("screen_from"):  # rates of an earlier screen (task id -> pass rate): no screen play here
                rates = json.loads((source / cfg["screen_from"]).read_text())
                screen = [r for r in train_pool if r["id"] in rates]
                hist.update({task: [value] * 4 for task, value in rates.items()})
                if eval_pool:
                    evaluation = await evaluate("eval-base", base)
            else:
                screen = rng.sample(train_pool, min(len(train_pool), int(cfg["screen_tasks"])))
                prior = load_prior(cfg["screen_prior"] if Path(str(cfg["screen_prior"])).is_absolute()
                                   else source / cfg["screen_prior"]) if cfg.get("screen_prior") else {}
                need = {r["id"]: max(0, int(cfg["screen_samples"]) - len(prior.get(r["id"], []))) for r in screen}
                jobs = [play(screen, need, transport=base, execute=execute, explore=explore, cfg=cfg, folder=feed,
                             label="screen", deadline=until(reserve))]
                if eval_pool:  # the base eval beside the screen (one reply pool; the tails overlap)
                    jobs.append(evaluate("eval-base", base, load_prior(cfg["eval_prior"]) if cfg.get("eval_prior") else None))
                done = await asyncio.gather(*jobs)
                found, evaluation = done[0], (done[1] if eval_pool else None)
                groups0 = {r["id"]: prior.get(r["id"], [])[: int(cfg["screen_samples"])] + found.get(r["id"], [])
                           for r in screen}
                hist.update({task: [x[0] for x in items] for task, items in groups0.items() if items})
                report["screen_prior_episodes"] = sum(len(v) for v in prior.values())
            base_rates.update({task: rate(task) for task in hist})
            band = [r for r in screen if in_band(r["id"])]
            seconds = time.time() - began
            report["screen"] = {"tasks": len(screen), "seconds": round(seconds), "tokens": base.stats["tokens"],
                                "tokens_per_second": round(base.stats["tokens"] / max(1.0, seconds)),
                                "replies": base.stats["replies"], "without_logprobs": base.stats["without_logprobs"],
                                "count_mismatches": base.stats["count_mismatches"],
                                "solved_all": sum(v == 1 for v in base_rates.values()),
                                "solved_none": sum(v == 0 for v in base_rates.values()), "band": len(band),
                                "band_counts": {str(k): sum(round(v * int(cfg["screen_samples"])) == k
                                                            for v in base_rates.values())
                                                for k in range(int(cfg["screen_samples"]) + 1)},
                                "rates": base_rates}
            if evaluation:
                report["eval_base"] = evaluation
            if int(cfg["probes"]) and groups0:
                outside = sorted(t for t in groups0 if groups0[t] and not in_band(t))
                inside = sorted(t for t in groups0 if groups0[t] and in_band(t))
                count = int(cfg["probes"])
                chosen = [("untrained", t) for t in rng.sample(outside, min(len(outside), (count + 1) // 2))]
                chosen += [("band", t) for t in rng.sample(inside, min(len(inside), count - len(chosen)))]
                for kind, task in chosen:
                    path = Path(max(groups0[task], key=lambda x: x[2])[1])  # the longest episode
                    label = f"{kind}-{task.replace(':', '_')}"
                    if cfg["checkpoint"]:  # a resumed run reads the probes from the volume
                        (ckroot / "probes").mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, ckroot / "probes" / f"{label}.jsonl")
                        path = ckroot / "probes" / f"{label}.jsonl"
                    probes.append({"rollout": str(path), "w": 1.0, "name": label})
                report["probes"] = probes
            save()
        if cfg.get("prior_hist"):
            seeded = json.loads((source / cfg["prior_hist"]).read_text())
            new = {t: [float(x) for x in v] for t, v in seeded.items() if t not in hist and v}
            hist.update(new)
            base_rates.update({t: rate(t) for t in new})
            report["prior_hist"] = {"tasks": len(new), "band": sum(in_band(r["id"]) for r in train_pool)}
            save()
        if cfg.get("probe_rollouts") and not probes:
            probes.extend({"rollout": str(p), "w": 1.0, "name": f"untrained-{i}"}
                          for i, p in enumerate(cfg["probe_rollouts"]))
            report["probes"] = probes
        control = cfg.get("kl_target") is not None and bool(probes)
        last_round = last_play = 0.0
        for number in range(start, int(cfg["rounds"])):
            if stop_requested():
                report["stop"] = "stop request"
                break
            # the next round's need: a play-only measuring round needs the last play, not a whole round; the final
            # eval's estimate only when there is an eval pool
            plays_only = cfg.get("update_rounds") is not None and number >= int(cfg["update_rounds"])
            need = (last_play if plays_only else last_round) + (float(cfg["eval_seconds"]) if eval_pool else 0.0)
            if time.time() > deadline_epoch - 2 * reserve or (
                    number > start and time.time() + need > deadline_epoch - reserve):
                report["stop"] = "deadline"
                break
            band = fixed if fixed is not None else [r for r in train_pool if in_band(r["id"])]
            if not band:
                report["stop"] = "empty band"
                break
            round_began = time.time()
            mark("play")
            picked = band if fixed is not None else rng.sample(
                band, min(len(band), int(cfg["tasks"]) + int(cfg["oversample"] or 0)))
            transport = transport_for(adapter or served)
            began = time.time()
            due = bool(eval_pool and int(cfg["eval_every"]) and number > 0 and number % int(cfg["eval_every"]) == 0)
            base_due = bool(resume and eval_pool and cfg["eval_base_fresh"] and not report.get("eval_base"))
            evaluation = None
            if number == 0 and cfg["round0_from_screen"] and groups0:
                found = {r["id"]: groups0[r["id"]] for r in picked}  # base episodes, already in hist
            else:
                cut = until(reserve)
                if int(cfg["round_play_seconds"] or 0):
                    cut = min(cut, time.monotonic() + int(cfg["round_play_seconds"]))
                jobs = [play(picked, int(cfg["group"]), transport=transport, execute=execute, explore=explore,
                             cfg=cfg, folder=feed, label=f"r{number:02d}", deadline=cut)]
                if due:
                    jobs.append(evaluate(f"eval-r{number:02d}", transport))
                if base_due:
                    jobs.append(evaluate("eval-base", transport_for(served)))
                done = await asyncio.gather(*jobs)
                found, evaluation = done[0], (done[1] if due else None)
                if base_due:
                    report["eval_base"] = done[-1]
                extra = 0
                while fixed is None and extra < int(cfg["extra_play"] or 0) and int(cfg["min_spread"] or 0):
                    have = len(advantages({t: [r for r, _, _ in v] for t, v in found.items()}))
                    rest = [r for r in band if r["id"] not in found and r not in picked]
                    if have >= int(cfg["min_spread"]) or not rest or time.monotonic() > until(reserve) - 600:
                        break
                    extra += 1
                    more = rng.sample(rest, min(len(rest), max(4, 2 * (int(cfg["min_spread"]) - have))))
                    picked = picked + more
                    cut = until(reserve)
                    if int(cfg["round_play_seconds"] or 0):
                        cut = min(cut, time.monotonic() + int(cfg["round_play_seconds"]))
                    found.update(await play(more, int(cfg["group"]), transport=transport, execute=execute,
                                            explore=explore, cfg=cfg, folder=feed, label=f"r{number:02d}x{extra}",
                                            deadline=cut))
                for task, items in found.items():
                    hist.setdefault(task, []).extend(r for r, _, _ in items)
            groups = {task: [r for r, _, _ in items] for task, items in found.items()}
            adv = advantages(groups, loo=cfg["advantage"] == "loo")
            spread_found = len(adv)
            if fixed is None and len(adv) > int(cfg["tasks"]):  # dynamic sampling: train at most `tasks` groups
                adv = {t: adv[t] for t in [r["id"] for r in picked if r["id"] in adv][:int(cfg["tasks"])]}
            specs = [{"rollout": path, "w": a, "name": f"r{number:02d}:{task}:{k}", "whole": True, "share": task,
                      "tokens": max(1, int(spent or 1))}
                     for task, items in found.items() if task in adv
                     for k, ((_, path, spent), a) in enumerate(zip(items, adv[task])) if a != 0]
            push_scale = balance(specs) if cfg["push_balance"] else None
            measure = cfg.get("update_rounds") is not None and number >= int(cfg["update_rounds"])
            if measure:  # a play-only round (the pilot's read after its updates)
                specs = []
            for spec in specs:
                spec.pop("tokens", None)
            seconds = time.time() - began
            mean_reward = sum(sum(v) for v in groups.values()) / max(1, sum(len(v) for v in groups.values()))
            expected = sum(base_rates.get(t, 0.0) for t in groups) / max(1, len(groups))
            row = {"round": number, "model": adapter or "base", "play_seconds": round(seconds),
                   "tokens": transport.stats["tokens"],
                   "tokens_per_second": round(transport.stats["tokens"] / max(1.0, seconds)),
                   "mean_reward": round(mean_reward, 4), "base_rate": round(expected, 4),
                   "lift": round(mean_reward - expected, 4), "band": len(band),
                   "groups_with_spread": spread_found, "groups_played": len(groups), "groups_trained": len(adv),
                   "trained_episodes": len(specs), "updates": [],
                   "left_out": transport.stats.get("invalid", 0),  # this round's transport: cut or failed episodes
                   "push_scale": push_scale, "measure": measure, "lr": lr}
            if evaluation:
                row["eval"] = evaluation
                report["evals"].append({"round": number, "updates_before": number, **evaluation})
            if specs and number == 0 and int(cfg["trained_probes"] or 0) and not floor and not any(
                    p["name"].startswith("won-") for p in probes):
                for task in sorted(adv)[:int(cfg["trained_probes"])]:
                    for kind, want in (("won", 1.0), ("lost", 0.0)):
                        items = [x for x in found[task] if float(x[0]) == want]
                        if not items:
                            continue
                        path, label = Path(min(items, key=lambda x: x[2])[1]), f"{kind}-{task.replace(':', '_')}"
                        if cfg["checkpoint"]:
                            (ckroot / "probes").mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(path, ckroot / "probes" / f"{label}.jsonl")
                            path = ckroot / "probes" / f"{label}.jsonl"
                        probes.append({"rollout": str(path), "w": 1.0, "name": label})
                report["probes"] = probes
                save()
            row["replies"] = reply_stats([p for items in found.values() for _, p, _ in items])
            restored = False
            if number == 0 and adapter is None and not report.get("base_replies"):
                report["base_replies"] = row["replies"]
            elif cfg["play_guard"] and report.get("base_replies") and adapter is not None:
                broken = play_broken(row["replies"], report["base_replies"], cfg["play_guard"])
                row["play_guard"] = broken
                if broken and control and (work / "pre-update").is_dir():
                    # the adapter this round played broke play: back to the state before its update, no training
                    await restore(work / "pre-update")
                    lr = max(float(cfg["lr_min"]), lr / 1.5)
                    row["control"] = {"action": "rollback_play", "broken": broken, "lr_next": lr}
                    specs, restored = [], True
                    row["trained_episodes"] = 0
            new_held = []
            if specs and cfg["held_successes"]:
                have = {p["name"] for p in probes}
                for task in sorted(adv):
                    label = f"held-{task.replace(':', '_')}"
                    wins = [x for x in found[task] if float(x[0]) == 1.0]
                    if label in have or len(wins) < 2:
                        continue
                    pick = min(wins, key=lambda x: x[2])
                    specs = [sp for sp in specs if sp["rollout"] != pick[1]]
                    path = Path(pick[1])
                    if cfg["checkpoint"]:
                        (ckroot / "probes").mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, ckroot / "probes" / f"{label}.jsonl")
                        path = ckroot / "probes" / f"{label}.jsonl"
                    new_held.append({"rollout": str(path), "w": 1.0, "name": label})
                probes.extend(new_held)
                report["probes"] = probes
                row["trained_episodes"] = len(specs)
            row["positive_episodes"] = sum(1 for sp in specs if sp["w"] > 0)  # all a positives-only step trains
            if specs:
                row["sleep_seconds"] = await server.sleep()
                mark("update")
                await base_kl()
                if new_held:  # each held success's NLL before its task's first update
                    first = (await trainer.must("score", windows=new_held))["windows"]
                    held_ref.update({n: (v.get("mismatch") or {}).get("trainer_nll") for n, v in first.items()})
                    report["held_ref"] = dict(held_ref)
                rng.shuffle(specs)
                action, excess = "full", None
                if control:  # the state before this update, for a rollback
                    pre = work / "pre-update"
                    shutil.rmtree(pre, ignore_errors=True)
                    await trainer.must("checkpoint", path=str(pre))
                steps = list(range(0, len(specs), int(cfg["per_update"])))
                for j, i in enumerate(steps):
                    warm = int(cfg["warmup_steps"] or 0)
                    step_lr = lr * min(1.0, (steps_done + 1) / warm) if warm else lr
                    steps_done += 1
                    result = await trainer.must("update", timeout=cfg["update_timeout"], lr=step_lr, clip=cfg["clip"],
                                                attempts=specs[i:i + int(cfg["per_update"])], signed=bool(cfg["signed"]),
                                                max_windows=int(cfg["max_windows"]), **(cfg["update_extra"] or {}))
                    row["updates"].append({"lr": step_lr, "step": steps_done, **{k: result.get(k) for k in (
                        "loss", "grad_norm", "update_norm", "adapter_norm", "trained_tokens", "train_seconds",
                        "seconds", "errors", "memory", "prefix_reused", "window_wait_seconds")},
                        "mismatch": window_stats(result.get("windows") or []),
                        "windows": [{k: w.get(k) for k in ("name", "tokens", "targets", "trained", "w", "mismatch", "tis",
                                                          "prefix_seconds", "prefix_reused", "batch", "seconds")}
                                    for w in result.get("windows") or []]})
                    if control:
                        scored = (await trainer.must("score", windows=[p for p in probes if not p["name"].startswith(
                            ("won-", "lost-", "held-"))], hold_previous=True))["windows"]
                        guard = scored
                        excess = sum(max(0.0, float(v.get("kl_previous") or 0.0) - floor.get(n, 0.0))
                                     for n, v in guard.items()) / max(1, len(guard))
                        row["updates"][-1]["kl_excess"] = round(excess, 5)
                        limit = (float(cfg["kl_rollback_excess"]) if cfg["kl_rollback_excess"] is not None
                                 else float(cfg["kl_rollback_at"]) * float(cfg["kl_target"]))
                        if excess > limit:
                            await restore(pre)
                            action = "rollback"
                        elif (cfg["kl_stop_at"] is not None and j < len(steps) - 1
                              and excess > float(cfg["kl_stop_at"]) * float(cfg["kl_target"])):
                            action = "stop"
                    report["rounds_partial"] = row
                    save()
                    if action != "full":
                        break
                if control:
                    row["control"] = {"action": action, "excess": None if excess is None else round(excess, 5),
                                      "lr": lr, "rule": cfg["lr_rule"]}
                    if cfg["lr_rule"] == "dose" and excess is not None:
                        lr = min(float(cfg["lr_max"]), max(float(cfg["lr_min"]), lr * math.sqrt(
                            float(cfg["kl_target"]) / max(excess, 1e-9))))
                    elif action != "full":
                        lr = max(float(cfg["lr_min"]), lr / 1.5)
                    elif excess is not None and excess < float(cfg["kl_target"]) / 1.5:
                        lr = min(float(cfg["lr_max"]), lr * 1.5)
                    row["control"]["lr_next"] = lr
                    if action != "rollback":
                        await trainer.must("accept_reference")
                trained = sum(u.get("trained_tokens") or 0 for u in row["updates"])
                busy = sum(u.get("train_seconds") or 0 for u in row["updates"])
                row["trainer_tokens_per_second"] = round(trained / max(1.0, busy), 1)
                if cfg["profile"] and number == 0:
                    mark("profile")
                    prof = await trainer.call("profile", timeout=900, spec={**specs[0], "w": 1.0, "whole": False},
                                              warm=True)
                    row["profile"] = {k: v for k, v in prof.items() if k != "traceback"}
                    report["rounds_partial"] = row
                    save()
                if probes and row.get("control", {}).get("action") != "rollback":
                    scored = (await trainer.must("score", windows=probes, hold_previous=control))["windows"]
                    row["probes"] = {n: {k: v.get(k) for k in ("kl_base", "kl_previous", "logratio_base", "mismatch")}
                                     for n, v in scored.items()}
                    drops = [held_ref[n] - (v.get("mismatch") or {}).get("trainer_nll") for n, v in scored.items()
                             if n in held_ref and held_ref[n] is not None
                             and (v.get("mismatch") or {}).get("trainer_nll") is not None]
                    if drops:  # > 0: held-back successes more likely than before their task's first update
                        row["held"] = {"windows": len(drops), "mean_nll_drop": round(sum(drops) / len(drops), 5),
                                       "worse": sum(1 for d in drops if d < 0)}
                    high = {n: v.get("kl_base") for n, v in scored.items() if n.startswith("untrained")
                            and cfg["kl_stop"] is not None
                            and (v.get("kl_base") or 0) > floor.get(n, 0.0) + float(cfg["kl_stop"])}
                    if high:
                        row["kl_stop"] = high
            if specs and row.get("control", {}).get("action") == "rollback":  # nothing kept: serve the last adapter
                mark("serve")
                row["serve"] = await serve_again()
            elif specs or restored:  # restored: the state before the broken update, saved and served as this round's
                if cfg["checkpoint"]:
                    path = ckroot / f"r{number:02d}" / "adapter"
                else:
                    path = output / "adapters" / f"r{number:02d}"
                saved = await trainer.must("save", path=str(path))
                row["saved"] = {k: saved.get(k) for k in ("path", "tensors", "gib", "seconds") if k in saved}
                if cfg["checkpoint"]:
                    state_path = ckroot / f"r{number:02d}" / "state"
                    row["checkpoint"] = await trainer.must("checkpoint", path=str(state_path))
                mark("serve")
                row["serve"] = await serve_again()
                adapter = name(number)
                row["load_seconds"] = await server.load(adapter, saved.get("path") or path)
                if cfg["checkpoint"]:
                    report["rounds"].append(row)  # in state.json's summary
                    write_state(number, saved.get("path") or path, state_path)
                    report["rounds"].pop()
            last_round = time.time() - round_began
            last_play = 1.25 * float(row.get("play_seconds") or 0.0)  # play varies round to round
            row["round_seconds"] = round(last_round)
            report.pop("rounds_partial", None)
            report["rounds"].append(row)
            save()
            if row.get("kl_stop"):
                report["stop"] = f"kl_base above floor + {cfg['kl_stop']}: {row['kl_stop']}"
                break
        if eval_pool and adapter and time.time() < deadline_epoch - reserve / 2:
            mark("eval")
            report["eval_final"] = await evaluate("eval-final", transport_for(adapter))
            report["eval_final"]["updates_before"] = len([r for r in report["rounds"] if r.get("updates")]) + len(
                report.get("prior_rounds") or [])
        report["status"] = "complete"
    except Exception as exc:  # noqa: BLE001 - the report says what failed
        import traceback

        report.update(status="error", error=f"{type(exc).__name__}: {exc}"[:2000],
                      traceback=traceback.format_exc()[-4000:])
    finally:
        save()
        if trainer is not None:
            trainer.close()
        await client.aclose()
    return report


async def trainer_speed(cfg, source, rollouts, output):
    """Trainer only (no server): lr-0 updates over the same recorded arc2 episodes with the next window encoded
    beside the GPU loop (the trainer's default), with a 0.1 ms GIL switch interval, and inline; ``speed.json``
    after every step."""
    from arena.rl_loop import Trainer, trainer_command

    output, work = Path(output), Path("/tmp/arc2-speed")
    work.mkdir(parents=True, exist_ok=True)
    report = {"rollouts": rollouts, "steps": []}

    def save():
        (output / "speed.json").write_text(json.dumps(report, indent=1, default=str))

    command, env = trainer_command({**cfg, "window": int(cfg["train_window"])}, source, work)
    if os.environ.get("ARC2_PREFLIGHT"):  # launch gate: the scripted trainer (arc2.preflight) on a CPU container
        command = [sys.executable, "-m", "arc2.preflight", "trainer", "--dir", str(Path(work) / "trainer")]
        env = {**env, "PYTHONPATH": str(Path(source) / "src")}
    env["RL1_DENSE_ONLY"] = "1"
    trainer = Trainer(work / "trainer", command, env, output / "trainer.log")
    specs = [{"rollout": r, "w": 1.0 if i % 2 == 0 else -1.0, "name": f"s{i}", "whole": True,
              "share": Path(r).parent.parent.name} for i, r in enumerate(rollouts)]
    try:
        began = time.time()
        report["load"] = await trainer.ready(cfg["trainer_timeout"])
        report["load_seconds"] = round(time.time() - began, 1)
        save()
        update = {"lr": 0.0, "clip": 1.0, "attempts": specs, "signed": True, "max_windows": 64}
        # speed-v4: the split GEMM loop is launch bound; does the encoder thread's GIL hold stall it?
        plans = [("warm", "update", {**update, "attempts": specs[:1], "max_windows": 1}),
                 ("update_prefetch", "update", update),
                 ("knobs_fast_switch", "knobs", {"switch_interval": 0.0001}),
                 ("update_prefetch_fast_switch", "update", update),
                 ("knobs_inline", "knobs", {"switch_interval": 0.005, "prefetch": False}),
                 ("update_inline", "update", update)]
        for label, op, payload in plans:
            since = time.time()
            out = await trainer.call(op, timeout=900, **payload)
            row = {"step": label, "seconds": round(time.time() - since, 1), "ok": out.get("ok", True),
                   "error": out.get("error")}
            if op == "exact":
                row.update({k: v for k, v in out.items() if k not in ("traceback",)})
            elif op == "update":
                row.update({k: out.get(k) for k in ("trained_tokens", "train_seconds", "prefix_reused", "memory",
                                                    "window_wait_seconds")},
                           tokens_per_second=round((out.get("trained_tokens") or 0) / max(1.0, out.get("train_seconds") or 1), 1),
                           mismatch=window_stats(out.get("windows") or []),
                           windows=[{k: w.get(k) for k in ("name", "tokens", "targets", "trained", "batch", "prefix_reused",
                                                           "prefix_seconds", "seconds", "mismatch")} for w in out.get("windows") or []])
            else:
                row.update({k: v for k, v in out.items() if k != "traceback"})
            report["steps"].append(row)
            save()
        report["status"] = "complete"
    except Exception as exc:  # noqa: BLE001
        report.update(status="error", error=f"{type(exc).__name__}: {exc}"[:2000])
    finally:
        save()
        trainer.close()
    return report


def _by_source(done):
    out = {}
    for task, rewards in done.items():
        out.setdefault(task.split(":")[0], []).append(rewards)
    return out


async def run_screen(study, trial, *, source, output, base_url, deadline_epoch, python=None):
    """Pre-screen only (trial kind ``arc2-screen``, 19 Oct): ``samples`` base episodes of every pool task on any
    served stack (no trainer, no sleep), the first pass over all tasks before the second, cut at the trial's
    deadline; per-task rewards in ``screen.json``; live counts in arena/live/arc2-<trial>.json."""
    import httpx

    cfg = config(study, trial)
    cfg["window"] = int(study["windows"][0])
    output, source = Path(output), Path(source)
    feed = output / "rl"
    feed.mkdir(parents=True, exist_ok=True)
    live = os.environ.get("ARC_VALIDATION_OUTPUT")
    live = Path(live) / "arena" / "live" if live else None
    from .gpu import GpuSampler

    gpu = GpuSampler()
    gpu.mark("screen")
    report = {"status": "running", "settings": cfg, "started_epoch": time.time()}
    check = sandbox.self_test(python)
    if not check["ok"]:
        check = {**sandbox.self_test(python, strict=False), "first": check}
    strict = check["strict"]
    runner = asyncio.Semaphore(32)

    async def execute(code, inputs, **kw):
        async with runner:
            return await asyncio.to_thread(sandbox.run, code, inputs, python=python, strict=strict, **kw)

    async def explore(cells, code, task, **kw):
        async with runner:
            return await asyncio.to_thread(sandbox.run_cells, cells, code, task, python=python, strict=strict, **kw)

    rows = [json.loads(line) for line in open(source / cfg["pool"])]
    if cfg.get("screen_shuffle", True):
        random.Random(0).shuffle(rows)
    # screen_have (a top-up): {task: episodes it already has}; each task plays only up to screen_samples in all
    have = json.loads((source / cfg["screen_have"]).read_text()) if cfg.get("screen_have") else {}
    need = {r["id"]: max(0, int(cfg["screen_samples"]) - int(have.get(r["id"], 0))) for r in rows}
    served = study.get("served_model") or "Qwen/Qwen3.8-Flash-Next-NVFP4"
    root = base_url.rstrip("/").rsplit("/v1", 1)[0]
    client = httpx.AsyncClient(base_url=root, trust_env=False, timeout=3600,
                               limits=httpx.Limits(max_connections=None, max_keepalive_connections=None))
    transport = RLTransport(client, served, served, cfg["effort"], cfg["temperature"], cfg["concurrency"])
    transport.episodes = asyncio.Semaphore(int(cfg["episode_cap"])) if int(cfg["episode_cap"] or 0) else None
    transport.local_count, transport.record_logprobs = bool(cfg.get("local_count")), False
    began = time.time()

    def save(final=False):
        done = {}
        for path in feed.glob("screen/*/*/episode.json"):
            episode = json.loads(path.read_text())
            if episode.get("valid", True):
                done.setdefault(episode["task"], []).append(episode["reward"])
        seconds = max(1.0, time.time() - began)
        counts = {}
        for rewards in done.values():
            counts[str(int(sum(rewards)))] = counts.get(str(int(sum(rewards))), 0) + 1
        report.update(tasks=len(rows), episodes_planned=sum(need.values()), tasks_done=len(done), episodes=sum(len(v) for v in done.values()),
                      tokens=transport.stats["tokens"], tokens_per_second=round(transport.stats["tokens"] / seconds),
                      invalid=transport.stats.get("invalid", 0), replies=transport.stats["replies"],
                      not_all_solved=sum(1 for v in done.values() if min(v) < 1 and max(v) > 0) if done else 0,
                      none_solved=sum(1 for v in done.values() if max(v) == 0), counts_by_solved=counts,
                      play_seconds=round(seconds), gpu=gpu.summary(), updated_epoch=time.time(),
                      by_source={src: {"tasks": len(v), "episodes": sum(len(x) for x in v),
                                       "solved": sum(sum(x) for x in v), "mixed": sum(1 for x in v if min(x) < 1 < 2 * max(x))}
                                 for src, v in _by_source(done).items()})
        if final:
            report["rewards"] = done
        text = json.dumps(report, indent=1, default=str)
        (output / "screen.json").write_text(text)
        if live is not None:
            live.mkdir(parents=True, exist_ok=True)
            (live / f"arc2-{trial['name']}.json").write_text(text)

    async def ticker():
        while True:
            await asyncio.sleep(60)
            save()

    tick = asyncio.create_task(ticker())
    try:
        deadline = time.monotonic() + (deadline_epoch - 120 - time.time())
        await play(rows, need, transport=transport, execute=execute, explore=explore,
                   cfg={**cfg, "sample_major": True}, folder=feed, label="screen", deadline=deadline, offsets=have)
        report["status"] = "complete"
    except Exception as exc:  # noqa: BLE001
        report.update(status="error", error=f"{type(exc).__name__}: {exc}"[:2000])
    finally:
        tick.cancel()
        save(final=True)
        await client.aclose()
    return report


async def dose_curve(cfg, source, folder, output, *, lrs=(1e-5, 3e-5, 1e-4), steps=4, batch=12, tasks=8, seed=0):
    """P0: trainer only, on recorded base episodes (``folder``: <task>/<k>/episode.json +
    rollout.jsonl). Tasks with spread and >= 2 successes give one held-back success each and train on the rest
    (leave-one-out, push balance); from the same fresh adapter, ``steps`` updates of ``batch`` episodes at each lr,
    after each: KL to base on probes of all-solved tasks and the trainer's NLL of the held-back successes.
    ``dose.json`` after every step."""
    from arena.rl_loop import Trainer, trainer_command

    output, work = Path(output), Path("/tmp/arc2-dose")
    work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    if str(folder).endswith(".json"):  # an index {task: [[reward, rollout, tokens]]} (no per-file volume reads)
        prior = {t: [tuple(x) for x in v] for t, v in json.loads(Path(folder).read_text()).items()}
    else:
        prior = load_prior(folder)
    usable = sorted(t for t, items in prior.items()
                    if len(items) >= 6 and 2 <= sum(r for r, _, _ in items) < len(items))
    picked = rng.sample(usable, min(tasks, len(usable)))
    solved = sorted(t for t, items in prior.items() if items and all(r for r, _, _ in items))
    probes = [{"rollout": prior[t][0][1], "w": 1.0, "name": f"untrained-{t}"} for t in rng.sample(solved, min(2, len(solved)))]
    held, specs = [], []
    for t in picked:
        items = list(prior[t])
        k = max(i for i, (r, _, _) in enumerate(items) if r)
        held.append({"rollout": items[k][1], "w": 1.0, "name": f"held-{t}"})
        rest = items[:k] + items[k + 1:]
        adv = advantages({t: [r for r, _, _ in rest]}, loo=True).get(t)
        if adv:
            specs += [{"rollout": p, "w": a, "name": f"{t}:{i}", "whole": True, "share": t, "tokens": max(1, int(n or 1))}
                      for i, ((_, p, n), a) in enumerate(zip(rest, adv)) if a]
    scale = balance(specs)
    for s in specs:
        s.pop("tokens")
    rng.shuffle(specs)
    report = {"tasks": picked, "probes": [p["name"] for p in probes], "held": len(held), "train_episodes": len(specs),
              "push_scale": scale, "lrs": list(lrs), "steps": steps, "batch": batch, "curves": {}}

    def save():
        (output / "dose.json").write_text(json.dumps(report, indent=1, default=str))

    command, env = trainer_command({**cfg, "window": int(cfg["train_window"])}, source, work)
    if os.environ.get("ARC2_PREFLIGHT"):  # launch gate: the scripted trainer (arc2.preflight) on a CPU container
        command = [sys.executable, "-m", "arc2.preflight", "trainer", "--dir", str(Path(work) / "trainer")]
        env = {**env, "PYTHONPATH": str(Path(source) / "src")}
    env["RL1_DENSE_ONLY"] = "1"
    env.update({k: str(v) for k, v in (cfg.get("trainer_env") or {}).items()})
    trainer = Trainer(work / "trainer", command, env, output / "trainer.log")
    windows = probes + held

    def read(scored):
        return {n: {"kl_base": v.get("kl_base"), "nll": (v.get("mismatch") or {}).get("trainer_nll"),
                    "engine_nll": (v.get("mismatch") or {}).get("engine_nll")} for n, v in scored.items()}

    try:
        began = time.time()
        await trainer.ready(cfg["trainer_timeout"])
        report["load_seconds"] = round(time.time() - began, 1)
        start = work / "start"
        await trainer.must("checkpoint", path=str(start))  # the fresh adapter
        report["base"] = read((await trainer.must("score", windows=windows, label="base"))["windows"])
        report["floor"] = read((await trainer.must("score", windows=windows))["windows"])
        save()
        for lr in lrs:
            await trainer.must("load_checkpoint", path=str(start))
            curve = report["curves"][str(lr)] = []
            for step in range(steps):
                chunk = [specs[(step * batch + i) % len(specs)] for i in range(min(batch, len(specs)))]
                result = await trainer.must("update", timeout=cfg["update_timeout"], lr=lr, clip=cfg["clip"],
                                            attempts=chunk, signed=True, max_windows=int(cfg["max_windows"]),
                                            **(cfg["update_extra"] or {}))
                scored = read((await trainer.must("score", windows=windows))["windows"])
                kl = [v["kl_base"] or 0.0 for n, v in scored.items() if n.startswith("untrained")]
                gain = [(report["base"][n]["nll"] or 0.0) - (v["nll"] or 0.0) for n, v in scored.items() if n.startswith("held")]
                curve.append({"step": step + 1, "trained_tokens": result.get("trained_tokens"),
                              "train_seconds": result.get("train_seconds"), "grad_norm": result.get("grad_norm"),
                              "update_norm": result.get("update_norm"), "loss": result.get("loss"),
                              "probe_kl": round(sum(kl) / max(1, len(kl)), 5),
                              "held_nll_drop": round(sum(gain) / max(1, len(gain)), 5), "windows": scored})
                save()
        report["status"] = "complete"
    except Exception as exc:  # noqa: BLE001
        import traceback

        report.update(status="error", error=f"{type(exc).__name__}: {exc}"[:2000], traceback=traceback.format_exc()[-3000:])
    finally:
        save()
        trainer.close()
    return report


def chunk_name(setting):
    return setting.get("tag") or f"{setting.get('select', 'qsa')}@{int(setting['chunk'])}"


def pick_tries(folder, per_task):
    """The first ``per_task`` recorded tries (by number) of every task in ``folder`` (<task>/<k>/rollout.jsonl)."""
    picked = []
    for task in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        tries = sorted((p for p in task.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
        tries = [p / "rollout.jsonl" for p in tries if (p / "rollout.jsonl").is_file() and (p / "rollout.jsonl").stat().st_size]
        picked += [str(p) for p in tries[:per_task]]
    return picked


def pool_chunks(windows, settings):
    """``chunk_check`` windows pooled per setting: the trainer-vs-sampler gap (token-weighted), and per token the
    actor / raw log-probability gap to the setting's reference (mean |d|, mean d, max |d|, share beyond 0.5 nats) and
    each reply's mean gap (GSPO's log s)."""
    out = {}
    for key in map(chunk_name, settings):
        cells = [w["settings"][key] for w in windows if key in (w.get("settings") or {}) and "error" not in w["settings"][key]]
        if not cells:
            continue
        samp = [c["sampler"] for c in cells if (c.get("sampler") or {}).get("tokens")]
        n = sum(s["tokens"] for s in samp)
        row = {"windows": len(cells), "seconds": round(sum(c["seconds"] for c in cells), 1),
               "sampler": {"tokens": n, **{f: round(sum(s[f] * s["tokens"] for s in samp) / n, 5)
                                           for f in ("mean_abs", "mean", "beyond_half")}} if n else {}}
        for label in ("gap", "gap_default"):
            gaps = [c[label] for c in cells if label in c]
            if not gaps:
                continue
            pooled = {"against": gaps[0]["against"]}
            for field in ("actor", "raw"):
                count = sum(g[field]["count"] for g in gaps)
                pooled[field] = {"tokens": count, "mean_abs": round(sum(g[field]["abs_sum"] for g in gaps) / max(1, count), 6),
                                 "mean": round(sum(g[field]["sum"] for g in gaps) / max(1, count), 6),
                                 "max_abs": round(max(g[field]["abs_max"] for g in gaps), 4),
                                 "beyond_half": round(sum(g[field]["beyond_half"] for g in gaps) / max(1, count), 6)}
            log_s = sorted(abs(v) for g in gaps for v in (g.get("replies") or {}).get("log_s", []))
            if log_s:
                pooled["reply_log_s"] = {"replies": len(log_s), "abs_mean": round(sum(log_s) / len(log_s), 6),
                                         "p90": round(log_s[int(0.9 * (len(log_s) - 1))], 6), "max": round(log_s[-1], 6),
                                         "outside_3e-4": round(sum(v > 3e-4 for v in log_s) / len(log_s), 4)}
            row[label] = pooled
        out[key] = row
    return out


async def chunk_check(cfg, source, rollouts, output, *, settings, spec=None, resume=None, deadline=None):
    """rl1-chunk port (the RL trainer's log-probs move with prefix chunking at identical weights):
    trainer only, base weights, on recorded tries: each try's window at every prefix chunk x QSA selection setting
    (rl1_trainer ``op_chunk_check``), one window per call; ``chunk.json`` after every window. Windows already in
    ``resume`` (an earlier run's chunk.json) are kept and skipped."""
    from arena.rl_loop import Trainer, trainer_command

    output, work = Path(output), Path("/tmp/arc2-chunk")
    work.mkdir(parents=True, exist_ok=True)
    spec = {"window": 32768, "floor": 1024, "margin": 1024, "credited_only": True, **(spec or {})}
    report = {"settings": settings, "spec": spec, "rollouts": rollouts, "windows": [], "started": time.time()}
    if resume and Path(resume).is_file():
        report["windows"] = [w for w in json.loads(Path(resume).read_text()).get("windows", [])
                             if w.get("settings") and not any("error" in c for c in w["settings"].values())]
        report["resumed_from"] = str(resume)
    done = {w["name"] for w in report["windows"]}

    def save():
        report["pooled"] = pool_chunks(report["windows"], settings)
        (output / "chunk.json").write_text(json.dumps(report, indent=1, default=str))

    command, env = trainer_command({**cfg, "window": int(cfg["train_window"])}, source, work)
    if os.environ.get("ARC2_PREFLIGHT"):  # launch gate: the scripted trainer (arc2.preflight) on a CPU container
        command = [sys.executable, "-m", "arc2.preflight", "trainer", "--dir", str(Path(work) / "trainer")]
        env = {**env, "PYTHONPATH": str(Path(source) / "src")}
    env["RL1_DENSE_ONLY"] = "1"
    env.update({k: str(v) for k, v in (cfg.get("trainer_env") or {}).items()})
    import threading

    from .gpu import GpuSampler

    sampler, stop = GpuSampler(every=5.0, recent=300.0), threading.Event()

    def card():  # the watcher's second signal beside new windows: the card's recent util and memory
        while not stop.wait(30):
            try:
                rows = list(sampler.rows)
                (output / "card.json").write_text(json.dumps({"t": time.time(), "phase": sampler.phase,
                                                              "recent": sampler.stats([r for r in rows if r[0] >= time.time() - 300])}))
            except Exception:  # noqa: BLE001 - a missed sample file is not an error
                pass

    threading.Thread(target=card, daemon=True).start()
    trainer = Trainer(work / "trainer", command, env, output / "trainer.log")
    try:
        began = time.time()
        sampler.mark("load")
        await trainer.ready(cfg["trainer_timeout"])
        report["load_seconds"] = round(time.time() - began, 1)
        sampler.mark("windows")
        save()
        for path in rollouts:
            name = "-".join(Path(path).parts[-3:-1])  # <task>-<k>
            if name in done:
                continue
            if deadline and time.time() > deadline - 180:
                report["cut_short"] = f"deadline before {name}"
                break
            out = await trainer.must("chunk_check", timeout=900, settings=settings,
                                     attempts=[{"rollout": path, "w": 1.0, "name": name, **spec}])
            report["windows"] += out["windows"]
            save()
        report["status"] = "partial" if report.get("cut_short") else "complete"
    except Exception as exc:  # noqa: BLE001
        import traceback

        report.update(status="error", error=f"{type(exc).__name__}: {exc}"[:2000], traceback=traceback.format_exc()[-3000:])
    finally:
        stop.set()
        sampler.stop.set()
        report["gpu"] = sampler.summary()
        save()
        trainer.close()
    return report


def snr_calls(positives):
    """P5's positives per task in episode order, alternately into two halves (odd tasks start the other half, so odd
    counts balance): one gradient call per (task, half)."""
    by_task = {}
    for p in positives:
        by_task.setdefault(p["task"], []).append(p)
    calls = []
    for index, task in enumerate(sorted(by_task)):
        items = sorted(by_task[task], key=lambda p: int(p["k"]))
        for half in (0, 1):
            part = items[(half ^ (index % 2))::2]
            if part:
                calls.append({"name": f"{task.replace(':', '_')}-h{half}", "task": task, "half": half, "episodes": len(part),
                              "attempts": [{"rollout": p["rollout"], "w": float(p["w"]), "name": f"r00:{task}:{p['k']}",
                                            "whole": True, "share": task} for p in part]})
    return calls


def next_dose(distance, kl, target, low=0.35, high=4.0):
    """A scope's second displacement: its untrained-probe KL aimed at ``target`` (KL ~ D^2); a first dose already near
    the target gets a second point at twice (or half) the distance."""
    ratio = math.sqrt(target / kl) if kl and kl > 0 else high
    ratio = min(max(ratio, low), high)
    if abs(ratio - 1.0) < 0.3:
        ratio = 2.0 if (kl or 0.0) < 2 * target else 0.5
    return round(distance * ratio, 4)


def snr_point(base, scored, sets):
    """A scored state against the base pass: the untrained probes' mean KL to base; per probe set the mean KL, the mean
    trainer-NLL change and that change per unit of the untrained KL."""
    kls = [v["kl_base"] for n, v in scored.items() if n in sets["untrained"] and v.get("kl_base") is not None]
    kl = sum(kls) / len(kls) if kls else None
    row = {"kl": round(kl, 6) if kl is not None else None}
    for label, members in sets.items():
        d = [v["nll"] - base[n]["nll"] for n, v in scored.items()
             if n in members and v.get("nll") is not None and (base.get(n) or {}).get("nll") is not None]
        k = [v["kl_base"] for n, v in scored.items() if n in members and v.get("kl_base") is not None]
        row[f"d_nll_{label}"] = round(sum(d) / len(d), 6) if d else None
        row[f"kl_{label}"] = round(sum(k) / len(k), 6) if k else None
        row[f"per_kl_{label}"] = round(row[f"d_nll_{label}"] / kl, 4) if d and kl else None
    row["windows"] = scored
    return row


def snr_fit(points, label, target):
    """Two (KL, dNLL) points of one scope: dNLL = p sqrt(KL) + q KL (dNLL = a D + b D^2 with KL ~ D^2): p the first-order
    (directional) part, q the curvature (flattening) part, and the value at the target KL."""
    pts = [(p["kl"], p[f"d_nll_{label}"]) for p in points if p.get("kl") and p.get(f"d_nll_{label}") is not None]
    if len(pts) < 2:
        return None
    (k1, y1), (k2, y2) = pts[:2]
    a, b, c, d = math.sqrt(k1), k1, math.sqrt(k2), k2
    det = a * d - b * c
    if abs(det) < 1e-12:
        return None
    p, q = (y1 * d - b * y2) / det, (a * y2 - c * y1) / det
    return {"p": round(p, 5), "q": round(q, 4), "at_target": round(p * math.sqrt(target) + q * target, 6),
            "per_kl_at_target": round((p * math.sqrt(target) + q * target) / target, 4)}


class SnrGate(Exception):
    """A real-trainer check failed (restored start, single AdamW step, repeat gradient): no (b) on a broken path."""


async def snr_check(cfg, source, plan, output, *, resume=None, deadline=None, work="/tmp/arc2-snr"):
    """arc2 RL diagnostic: trainer only, base model, P5's 56 positive episodes.
    (a) from one start state (fresh adapter, B factors nudged to 1e-12 as RL's mg5g so every state scores on one path),
    one AdamW step per (task, half), each checkpointed: exp_avg = 0.1 g gives the call's gradient (arc2.snr_math
    extract, which also checks the real trainer's restore path: the adapter moved by exactly -lr g / (|g| + eps) from
    the reloaded start, one step, update count 1; a repeated call must give the same gradient); then the halves' cosine
    and signal share, within- vs across-task cosines (snr_math analyze).
    (b) the full update's first AdamW step restricted to attention / gdn / shared expert, a random-sign control and the
    raw gradient direction, each at two displacements (the second aimed at the target KL), scored on the untrained,
    held and lost probes against the nudged base; also the B = 0 state against that base (the zero-path artifact).
    ``snr.json`` after every step. ``resume``: an earlier run's output folder; its gradients and scored points are kept
    when its start state is byte-identical."""
    if plan.get("mode") == "bank":
        return await grad_bank(cfg, source, plan, output, resume=resume, deadline=deadline, work=work)
    if plan.get("mode") == "dose":
        return await bank_dose(cfg, source, plan, output, resume=resume, deadline=deadline, work=work)
    from arena.rl_loop import Trainer, trainer_command

    output, work = Path(output), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (output / "grads").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lr, clip = float(plan.get("lr", 1e-4)), float(plan.get("clip", cfg.get("clip", 1.0)))
    doses = {"first": 2.0, "target_kl": 0.02, "low": 0.35, "high": 4.0, **(plan.get("doses") or {})}
    calls = snr_calls(plan["positives"])
    report = {"started": t0, "status": "running", "phase": "load", "lr": lr, "clip": clip, "doses": doses,
              "calls_planned": [{k: c[k] for k in ("name", "task", "half", "episodes")} for c in calls],
              "steps": [], "calls": {}, "points": {}}
    old = {}
    if resume and (Path(resume) / "snr.json").is_file():
        old = json.loads((Path(resume) / "snr.json").read_text())
        report["resumed_from"] = str(resume)
    windows = [{"rollout": p["rollout"], "w": 1.0, "name": p["name"]} for label in ("untrained", "held", "lost")
               for p in plan["probes"].get(label, [])]
    sets = {label: {p["name"] for p in plan["probes"].get(label, [])} for label in ("untrained", "held", "lost")}
    python = os.environ.get("ARC2_SNR_PYTHON") or cfg["trainer_python"]
    script = Path(source) / "src/arc2/snr_math.py"

    def save():
        tmp = output / "snr.json.tmp"
        tmp.write_text(json.dumps(report, indent=1, default=str))
        tmp.replace(output / "snr.json")

    def step(op, began, **fields):
        report["steps"].append({"op": op, "t": round(time.time() - t0, 1), "seconds": round(time.time() - began, 1), **fields})
        save()

    async def tensors(*args, timeout=900):
        proc = await asyncio.create_subprocess_exec(python, str(script), *map(str, args), stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
        if proc.returncode:
            raise RuntimeError(f"snr_math {args[0]}: " + " | ".join(err.decode(errors="replace").strip().splitlines()[-3:]))
        return json.loads(out.decode().strip().splitlines()[-1])

    def read(scored):
        return {n: {"kl_base": v.get("kl_base"), "nll": (v.get("mismatch") or {}).get("trainer_nll"),
                    "engine_nll": (v.get("mismatch") or {}).get("engine_nll"), "tokens": v.get("tokens")}
                for n, v in scored.items()}

    command, env = trainer_command({**cfg, "window": int(cfg["train_window"])}, source, work)
    if os.environ.get("ARC2_PREFLIGHT"):  # launch gate: the scripted trainer (arc2.preflight) on a CPU container
        command = [sys.executable, "-m", "arc2.preflight", "trainer", "--dir", str(Path(work) / "trainer")]
        env = {**env, "PYTHONPATH": str(Path(source) / "src")}
    env["RL1_DENSE_ONLY"] = "1"
    env.update({k: str(v) for k, v in (cfg.get("trainer_env") or {}).items()})
    import threading

    from .gpu import GpuSampler

    sampler, stop = GpuSampler(every=5.0, recent=300.0), threading.Event()

    def card():  # the watcher's second signal beside new steps
        while not stop.wait(30):
            try:
                rows = list(sampler.rows)
                (output / "card.json").write_text(json.dumps({"t": time.time(), "phase": sampler.phase,
                                                              "recent": sampler.stats([r for r in rows if r[0] >= time.time() - 300])}))
            except Exception:  # noqa: BLE001 - a missed sample file is not an error
                pass

    def phase(name):
        report["phase"] = name
        sampler.mark(name)
        save()

    threading.Thread(target=card, daemon=True).start()
    save()
    trainer = Trainer(work / "trainer", command, env, output / "trainer.log")
    try:
        sampler.mark("load")
        await trainer.ready(cfg["trainer_timeout"])
        report["load_seconds"] = round(time.time() - t0, 1)
        phase("base")
        fresh, start = work / "fresh", work / "start"
        began = time.time()
        await trainer.must("checkpoint", path=str(fresh))  # B = 0 exactly: the P runs' base path
        step("checkpoint_fresh", began)
        began = time.time()
        nudged = await trainer.must("nudge", value=float(plan.get("nudge", 1e-12)))
        step("nudge", began, nudged=nudged.get("nudged"))
        began = time.time()
        base = read((await trainer.must("score", windows=windows, label="base"))["windows"])
        report["base"] = base
        step("score_base", began)
        began = time.time()
        await trainer.must("checkpoint", path=str(start))
        report["start_sha"] = (await tensors("sha", start))["sha256"]
        same = bool(old) and old.get("start_sha") == report["start_sha"]
        report["resume_reuse"] = same
        step("checkpoint_start", began, sha=report["start_sha"][:16])
        phase("calls")
        smallest = min(calls, key=lambda c: (c["episodes"], c["name"]))
        todo = calls + [{**smallest, "name": smallest["name"] + "~repeat", "repeat_of": smallest["name"]}]
        for call in todo:
            name, grad = call["name"], output / "grads" / f"{call['name']}.safetensors"
            kept = (old.get("calls") or {}).get(name)
            if same and kept and kept.get("ok") and (Path(resume) / "grads" / grad.name).is_file():
                shutil.copyfile(Path(resume) / "grads" / grad.name, grad)
                report["calls"][name] = {**kept, "resumed": True}
                step("call_kept", time.time(), call=name)
                continue
            if deadline and time.time() > deadline - 240:
                report["cut_short"] = f"deadline before call {name}"
                break
            began = time.time()
            loaded = await trainer.must("load_checkpoint", path=str(start))
            up = await trainer.must("update", timeout=cfg["update_timeout"], lr=lr, clip=clip, attempts=call["attempts"],
                                    signed=False, max_windows=int(plan.get("max_windows", 64)), **(cfg.get("update_extra") or {}))
            folder = work / "calls" / name
            await trainer.must("checkpoint", path=str(folder))
            ex = await tensors("extract", start, folder, grad, "--lr", lr, "--clip", clip, "--grad-norm", up["grad_norm"])
            shutil.rmtree(folder, ignore_errors=True)
            rec = {"task": call["task"], "half": call["half"], "episodes": call["episodes"], "divisor": up.get("trained_tokens"),
                   "grad_norm": up.get("grad_norm"), "loss": up.get("loss"), "train_seconds": up.get("train_seconds"),
                   "windows": len(up.get("windows") or []), "errors": up.get("errors"), "loaded_updates": loaded.get("updates"),
                   "extract": ex, "repeat_of": call.get("repeat_of")}
            why = [w for w, bad in (("load_checkpoint did not restore update 0", loaded.get("updates") != 0),
                                    ("update errors", bool(up.get("errors"))), ("no trained tokens", not up.get("trained_tokens")),
                                    ("not one AdamW step", ex["steps"] != [1.0] or ex["updates"] != 1),
                                    ("adapter moved off -lr g/(|g|+eps)", ex["step_rel_err"] > 1e-3)) if bad]
            rec["ok"] = not why
            report["calls"][name] = rec
            step("call", began, call=name, tokens=up.get("trained_tokens"), tok_s=round((up.get("trained_tokens") or 0)
                 / max(1e-6, float(up.get("train_seconds") or 0)), 1), ok=rec["ok"])
            if why:
                raise SnrGate(f"{name}: {', '.join(why)}")
        if report.get("cut_short"):
            report["status"] = "partial"
            return report
        phase("math")
        began = time.time()
        done = [dict(report["calls"][c["name"]], name=c["name"], grad=str(output / "grads" / f"{c['name']}.safetensors"))
                for c in calls]
        spec = {"start": str(start), "directions": str(work / "directions.safetensors"), "seed": int(plan.get("seed", 0)),
                "calls": [{k: c[k] for k in ("name", "task", "half", "divisor", "episodes", "grad")} for c in done],
                "repeat": {"of": smallest["name"], "grad": str(output / "grads" / f"{smallest['name']}~repeat.safetensors")}}
        (work / "spec.json").write_text(json.dumps(spec))
        report["analysis"] = await tensors("analyze", work / "spec.json", output / "analysis.json", timeout=1800)
        step("analyze", began)
        rep = report["analysis"].get("repeat") or {}
        if (rep.get("cos") or 0.0) < 0.999:
            raise SnrGate(f"repeat call cosine {rep.get('cos')} < 0.999")
        report["gate"] = {"ok": True, "calls": len(done) + 1, "repeat_cos": rep.get("cos"),
                          "step_rel_err_max": max(report["calls"][c]["extract"]["step_rel_err"] for c in report["calls"])}
        phase("points")
        began = time.time()
        zero = old.get("zero_path") if same and old.get("zero_path") else None
        if zero is None:
            await trainer.must("load_checkpoint", path=str(fresh))
            zero = snr_point(base, read((await trainer.must("score", windows=windows))["windows"]), sets)
        report["zero_path"] = zero
        step("zero_path", began, kl=zero.get("kl"))
        scopes_info = report["analysis"].get("scopes") or {}

        async def point(scope, distance):
            name = f"{scope}@{distance:g}"
            kept = (old.get("points") or {}).get(name)
            if same and kept:
                report["points"][name] = {**kept, "resumed": True}
                step("point_kept", time.time(), point=name)
                return report["points"][name]
            if deadline and time.time() > deadline - 120:
                report.setdefault("skipped", []).append(name)
                save()
                return None
            began = time.time()
            folder = work / "states" / name
            await tensors("state", start, work / "directions.safetensors", scope, distance, folder)
            loaded = await trainer.must("load_checkpoint", path=str(folder))
            row = {"scope": scope, "D": distance, "adapter_norm": loaded.get("adapter_norm"),
                   "lr_equiv": round(distance * float((scopes_info.get(scope) or {}).get("lr_per_unit_D") or 0.0), 8),
                   **snr_point(base, read((await trainer.must("score", windows=windows))["windows"]), sets)}
            shutil.rmtree(folder, ignore_errors=True)
            report["points"][name] = row
            step("point", began, point=name, kl=row.get("kl"), held=row.get("d_nll_held"), lost=row.get("d_nll_lost"),
                 untrained=row.get("d_nll_untrained"))
            return row

        for scope in plan.get("scopes") or ["all", "random", "attention", "gdn", "shared_expert", "sgd"]:
            first = await point(scope, float(doses["first"]))
            if first and first.get("kl") is not None:
                await point(scope, next_dose(float(doses["first"]), first["kl"], float(doses["target_kl"]),
                                             float(doses["low"]), float(doses["high"])))
        report["status"] = "partial" if report.get("skipped") else "complete"
    except SnrGate as exc:
        report.update(status="gate_failed", gate={"ok": False, "why": str(exc)[:1000]})
    except Exception as exc:  # noqa: BLE001
        import traceback

        report.update(status="error", error=f"{type(exc).__name__}: {exc}"[:2000], traceback=traceback.format_exc()[-3000:])
    finally:
        stop.set()
        sampler.stop.set()
        report["gpu"] = sampler.summary()
        report["summary"] = snr_summary(report, doses)
        report["elapsed_seconds"] = round(time.time() - t0, 1)
        save()
        trainer.close()
    return report


async def bank_dose(cfg, source, plan, output, *, resume=None, deadline=None, work="/tmp/arc2-snr"):
    """arc2 RL plan step 3: dose-response of one fixed batch. From one start (fresh adapter, B
    nudged to 1e-12; LoRA A is drawn anew in each trainer process, so every gradient is taken in this run), each planned
    update is one real trainer update of the batch with the estimator's weights (signed: negative advantages included),
    one AdamW step from the start, its gradient read back from exp_avg with the step identity checked (as the bank's
    calls). Each planned direction is an update's first AdamW step on a scope (snr_math combine; ``random``: random
    signs on one direction's support), scored at its planned displacements on held successes and same-task failures
    the batch did not contain and on untrained tasks' tries, against the nudged base. ``dose.json`` after every step;
    ``resume`` keeps an earlier run's gradients and points when its start sha is the same."""
    from arena.rl_loop import Trainer, trainer_command

    output, work = Path(output), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (output / "grads").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lr, clip = float(plan.get("lr", 1e-4)), float(plan.get("clip", cfg.get("clip", 1.0)))
    max_windows = int(plan.get("max_windows", 128))
    report = {"started": t0, "status": "running", "mode": "dose", "phase": "load", "lr": lr, "clip": clip, "steps": [],
              "updates": {}, "points": {}}
    old = {}
    if resume and (Path(resume) / "dose.json").is_file():
        old = json.loads((Path(resume) / "dose.json").read_text())
        report["resumed_from"] = str(resume)
    windows = [{"rollout": p["rollout"], "w": 1.0, "name": p["name"]} for label in ("untrained", "held", "lost")
               for p in plan["probes"].get(label, [])]
    sets = {label: {p["name"] for p in plan["probes"].get(label, [])} for label in ("untrained", "held", "lost")}
    episodes = {e["name"]: e for e in plan["episodes"]}
    python = os.environ.get("ARC2_SNR_PYTHON") or cfg["trainer_python"]
    script = Path(source) / "src/arc2/snr_math.py"

    def save():
        tmp = output / "dose.json.tmp"
        tmp.write_text(json.dumps(report, indent=1, default=str))
        tmp.replace(output / "dose.json")

    def step(op, began, **fields):
        report["steps"].append({"op": op, "t": round(time.time() - t0, 1), "seconds": round(time.time() - began, 1), **fields})
        save()

    async def tensors(*args, timeout=1800):
        proc = await asyncio.create_subprocess_exec(python, str(script), *map(str, args), stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
        if proc.returncode:
            raise RuntimeError(f"snr_math {args[0]}: " + " | ".join(err.decode(errors="replace").strip().splitlines()[-3:]))
        return json.loads(out.decode().strip().splitlines()[-1])

    def read(scored):
        return {n: {"kl_base": v.get("kl_base"), "nll": (v.get("mismatch") or {}).get("trainer_nll"),
                    "engine_nll": (v.get("mismatch") or {}).get("engine_nll"), "tokens": v.get("tokens")}
                for n, v in scored.items()}

    command, env = trainer_command({**cfg, "window": int(cfg["train_window"])}, source, work)
    if os.environ.get("ARC2_PREFLIGHT"):  # launch gate: the scripted trainer (arc2.preflight) on a CPU container
        command = [sys.executable, "-m", "arc2.preflight", "trainer", "--dir", str(Path(work) / "trainer")]
        env = {**env, "PYTHONPATH": str(Path(source) / "src")}
    env["RL1_DENSE_ONLY"] = "1"
    env.update({k: str(v) for k, v in (cfg.get("trainer_env") or {}).items()})
    import threading

    from .gpu import GpuSampler

    sampler, stop = GpuSampler(every=5.0, recent=300.0), threading.Event()

    def card():  # the watcher's second signal beside new steps
        while not stop.wait(30):
            try:
                rows = list(sampler.rows)
                (output / "card.json").write_text(json.dumps({"t": time.time(), "phase": sampler.phase,
                                                              "recent": sampler.stats([r for r in rows if r[0] >= time.time() - 300])}))
            except Exception:  # noqa: BLE001 - a missed sample file is not an error
                pass

    def phase(name):
        report["phase"] = name
        sampler.mark(name)
        save()

    threading.Thread(target=card, daemon=True).start()
    save()
    trainer = Trainer(work / "trainer", command, env, output / "trainer.log")
    try:
        sampler.mark("load")
        await trainer.ready(cfg["trainer_timeout"])
        report["load_seconds"] = round(time.time() - t0, 1)
        phase("base")
        start = work / "start"
        began = time.time()
        nudged = await trainer.must("nudge", value=float(plan.get("nudge", 1e-12)))
        await trainer.must("checkpoint", path=str(start))
        report["start_sha"] = (await tensors("sha", start))["sha256"]
        same = bool(old) and old.get("start_sha") == report["start_sha"]
        report["resume_reuse"] = same
        step("checkpoint_start", began, nudged=nudged.get("nudged"), sha=report["start_sha"][:16])
        began = time.time()
        if same and old.get("base"):
            base = old["base"]
            step("base_kept", began)
        else:
            base = read((await trainer.must("score", windows=windows, label="base"))["windows"])
            step("score_base", began)
        report["base"] = base
        phase("updates")
        for name, up_plan in plan["updates"].items():
            grad = output / "grads" / f"{name}.safetensors"
            kept = (old.get("updates") or {}).get(name)
            if same and kept and kept.get("ok") and (Path(resume) / "grads" / grad.name).is_file():
                if Path(resume).resolve() != output.resolve():
                    shutil.copyfile(Path(resume) / "grads" / grad.name, grad)
                report["updates"][name] = {**kept, "resumed": True}
                step("update_kept", time.time(), update=name)
                continue
            began = time.time()
            loaded = await trainer.must("load_checkpoint", path=str(start))
            attempts = [{"rollout": episodes[e]["rollout"], "w": float(w), "name": e, "whole": True, "share": episodes[e]["task"]}
                        for e, w in up_plan["weights"].items() if float(w)]
            up = await trainer.must("update", timeout=cfg["update_timeout"], lr=lr, clip=clip, signed=True, attempts=attempts,
                                    max_windows=max_windows, **(cfg.get("update_extra") or {}))
            folder = work / "updates" / name
            await trainer.must("checkpoint", path=str(folder))
            ex = await tensors("extract", start, folder, grad, "--lr", lr, "--clip", clip, "--grad-norm", up["grad_norm"])
            shutil.rmtree(folder, ignore_errors=True)
            rec = {"episodes": len(attempts), "divisor": up.get("trained_tokens"), "grad_norm": up.get("grad_norm"),
                   "loss": up.get("loss"), "train_seconds": up.get("train_seconds"), "windows": len(up.get("windows") or []),
                   "errors": up.get("errors"), "loaded_updates": loaded.get("updates"),
                   "extract": {k: ex.get(k) for k in ("step_rel_err", "v_rel_err", "steps", "updates", "coef", "norm",
                                                      "update_norm", "norm2_by_group")}}
            why = [w for w, bad in (("load_checkpoint did not restore update 0", loaded.get("updates") != 0),
                                    ("update errors", bool(up.get("errors"))), ("no trained tokens", not up.get("trained_tokens")),
                                    ("max_windows reached (windows may be cut)", rec["windows"] >= max_windows),
                                    ("not one AdamW step", ex["steps"] != [1.0] or ex["updates"] != 1),
                                    ("adapter moved off -lr g/(|g|+eps)", ex["step_rel_err"] > 1e-3)) if bad]
            rec["ok"] = not why
            report["updates"][name] = rec
            step("update", began, update=name, tokens=up.get("trained_tokens"), tok_s=round((up.get("trained_tokens") or 0)
                 / max(1e-6, float(up.get("train_seconds") or 0)), 1), ok=rec["ok"])
            if why:
                raise SnrGate(f"{name}: {', '.join(why)}")
        phase("directions")
        began = time.time()
        spec = {"start": str(start), "directions_out": str(work / "directions.safetensors"), "seed": int(plan.get("seed", 0)),
                "random_of": plan.get("random_of"),
                "directions": {d: {"weights": {row["update"]: 1.0}, "scope": row.get("scope", "all")}
                               for d, row in plan["directions"].items()},
                "grads": {u: str(output / "grads" / f"{u}.safetensors") for u in plan["updates"]}}
        (work / "combine.json").write_text(json.dumps(spec))
        report["directions"] = await tensors("combine", work / "combine.json", output / "directions.json")
        step("combine", began, n=len(plan["directions"]))
        phase("points")

        async def point(name, distance):
            key = f"{name}@{distance:g}"
            kept = (old.get("points") or {}).get(key)
            if same and kept:
                report["points"][key] = {**kept, "resumed": True}
                step("point_kept", time.time(), point=key)
                return
            if deadline and time.time() > deadline - 150:
                report.setdefault("skipped", []).append(key)
                save()
                return
            began = time.time()
            folder = work / "states" / key.replace(":", "_")
            await tensors("state", start, work / "directions.safetensors", name, distance, folder)
            loaded = await trainer.must("load_checkpoint", path=str(folder))
            info = report["directions"].get(name.split(":", 1)[-1]) or {}
            row = {"direction": name, "D": distance, "adapter_norm": loaded.get("adapter_norm"),
                   "lr_equiv": round(distance * float(info.get("lr_per_unit_D") or 0.0), 8) if not name.startswith("sgd:") else None,
                   **snr_point(base, read((await trainer.must("score", windows=windows))["windows"]), sets)}
            shutil.rmtree(folder, ignore_errors=True)
            report["points"][key] = row
            step("point", began, point=key, kl=row.get("kl"), held=row.get("d_nll_held"), lost=row.get("d_nll_lost"),
                 untrained=row.get("d_nll_untrained"))

        for name, distances in plan["points"].items():
            for distance in distances:
                await point(name, float(distance))
        report["status"] = "partial" if report.get("skipped") else "complete"
    except SnrGate as exc:
        report.update(status="gate_failed", gate={"ok": False, "why": str(exc)[:1000]})
    except Exception as exc:  # noqa: BLE001
        import traceback

        report.update(status="error", error=f"{type(exc).__name__}: {exc}"[:2000], traceback=traceback.format_exc()[-3000:])
    finally:
        stop.set()
        sampler.stop.set()
        report["gpu"] = sampler.summary()
        report["elapsed_seconds"] = round(time.time() - t0, 1)
        save()
        trainer.close()
    return report


async def grad_bank(cfg, source, plan, output, *, resume=None, deadline=None, work="/tmp/arc2-snr"):
    """arc2 RL plan step 1: one gradient per recorded episode (successes and failures alike,
    weight 1, no baseline) from the nudged base, as snr_check's calls: one AdamW step from an empty state, the
    gradient read back from exp_avg with the step identity checked, a repeated call at cos >= 0.999. Then Gram
    matrices per (group, layer) block (snr_math gram), from which every estimator (positive-only, signed GRPO),
    reward (success, success with a token cost) and LoRA scope is compared offline by split-half shared signal.
    ``bank.json`` after every step; ``resume`` keeps an earlier run's gradients when the start state is the same."""
    from arena.rl_loop import Trainer, trainer_command

    output, work = Path(output), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (output / "grads").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lr, clip = float(plan.get("lr", 1e-4)), float(plan.get("clip", cfg.get("clip", 1.0)))
    episodes = plan["episodes"]
    credit = plan.get("credit")  # "answer": loss only after </think> (arc2 3'; dist/ops/trainer_patch/answer_credit.py)
    report = {"started": t0, "status": "running", "mode": "bank", "phase": "load", "lr": lr, "clip": clip, "credit": credit,
              "episodes_planned": len(episodes), "steps": [], "calls": {}}
    old = {}
    if resume and (Path(resume) / "bank.json").is_file():
        old = json.loads((Path(resume) / "bank.json").read_text())
        report["resumed_from"] = str(resume)
    python = os.environ.get("ARC2_SNR_PYTHON") or cfg["trainer_python"]
    script = Path(source) / "src/arc2/snr_math.py"

    def save():
        tmp = output / "bank.json.tmp"
        tmp.write_text(json.dumps(report, indent=1, default=str))
        tmp.replace(output / "bank.json")

    def step(op, began, **fields):
        report["steps"].append({"op": op, "t": round(time.time() - t0, 1), "seconds": round(time.time() - began, 1), **fields})
        save()

    async def tensors(*args, timeout=900):
        proc = await asyncio.create_subprocess_exec(python, str(script), *map(str, args), stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
        if proc.returncode:
            raise RuntimeError(f"snr_math {args[0]}: " + " | ".join(err.decode(errors="replace").strip().splitlines()[-3:]))
        return json.loads(out.decode().strip().splitlines()[-1])

    command, env = trainer_command({**cfg, "window": int(cfg["train_window"])}, source, work)
    if os.environ.get("ARC2_PREFLIGHT"):  # launch gate: the scripted trainer (arc2.preflight) on a CPU container
        command = [sys.executable, "-m", "arc2.preflight", "trainer", "--dir", str(Path(work) / "trainer")]
        env = {**env, "PYTHONPATH": str(Path(source) / "src")}
    env["RL1_DENSE_ONLY"] = "1"
    env.update({k: str(v) for k, v in (cfg.get("trainer_env") or {}).items()})
    import threading

    from .gpu import GpuSampler

    sampler, stop = GpuSampler(every=5.0, recent=300.0), threading.Event()

    def card():  # the watcher's second signal beside new steps
        while not stop.wait(30):
            try:
                rows = list(sampler.rows)
                (output / "card.json").write_text(json.dumps({"t": time.time(), "phase": sampler.phase,
                                                              "recent": sampler.stats([r for r in rows if r[0] >= time.time() - 300])}))
            except Exception:  # noqa: BLE001 - a missed sample file is not an error
                pass

    def phase(name):
        report["phase"] = name
        sampler.mark(name)
        save()

    threading.Thread(target=card, daemon=True).start()
    save()
    trainer = Trainer(work / "trainer", command, env, output / "trainer.log")
    pending = []
    try:
        sampler.mark("load")
        await trainer.ready(cfg["trainer_timeout"])
        report["load_seconds"] = round(time.time() - t0, 1)
        phase("base")
        start = work / "start"
        began = time.time()
        nudged = await trainer.must("nudge", value=float(plan.get("nudge", 1e-12)))
        await trainer.must("checkpoint", path=str(start))
        report["start_sha"] = (await tensors("sha", start))["sha256"]
        same = bool(old) and old.get("start_sha") == report["start_sha"]
        report["resume_reuse"] = same
        step("checkpoint_start", began, nudged=nudged.get("nudged"), sha=report["start_sha"][:16])
        phase("calls")
        calls = [{"name": e["name"], "episode": e} for e in episodes]
        smallest = min(calls, key=lambda c: (int(c["episode"].get("tokens") or 0), c["name"]))
        repeat = {**smallest, "name": smallest["name"] + "~repeat", "repeat_of": smallest["name"]}
        todo, gate_at = calls + [repeat], None
        if plan.get("gate_first"):  # the real-trainer gate before the bank: the smallest try, its repeat and, with
            # credit, the same try with full credit (the mask must change the gradient)
            todo = [smallest, repeat] + ([{**smallest, "name": smallest["name"] + "~full", "full_of": smallest["name"]}]
                                         if credit else [])
            gate_at = len(todo)
            todo += [c for c in calls if c["name"] != smallest["name"]]
        max_windows = int(plan.get("max_windows", 16))

        def grad_of(name):
            return output / "grads" / f"{name.replace(':', '_')}.safetensors"

        async def gate_check():
            for task_ in pending:
                await task_
            for name in [smallest["name"], repeat["name"]] + ([smallest["name"] + "~full"] if credit else []):
                if not (report["calls"].get(name) or {}).get("ok"):
                    raise SnrGate(f"gate call {name} not ok: {report['calls'].get(name)}")
            began = time.time()
            spec = {"start": str(start), "episodes": [{"name": smallest["name"], "grad": str(grad_of(smallest["name"]))}]
                    + ([{"name": smallest["name"] + "~full", "grad": str(grad_of(smallest["name"] + "~full"))}] if credit else []),
                    "repeat": {"of": smallest["name"], "grad": str(grad_of(repeat["name"]))}}
            (work / "gate-spec.json").write_text(json.dumps(spec))
            g = await tensors("gram", work / "gate-spec.json", work / "gate-gram.json")
            out = {"repeat_cos": (g.get("repeat") or {}).get("cos")}
            if credit:
                blocks = json.loads((work / "gate-gram.json").read_text())["blocks"].values()
                k00, k01, k11 = (sum(b[i][j] for b in blocks) for i, j in ((0, 0), (0, 1), (1, 1)))
                out.update(cos_answer_full=round(k01 / math.sqrt(k00 * k11), 6) if k00 * k11 > 0 else None,
                           norm_ratio=round(math.sqrt(k00 / k11), 6) if k11 > 0 else None,
                           credited_share=report["calls"][smallest["name"]].get("credited_share"))
            report["gate_first"] = out
            step("gate", began, **out)
            if (out["repeat_cos"] or 0.0) < 0.999:
                raise SnrGate(f"gate: repeat cosine {out['repeat_cos']} < 0.999")
            if credit and (out.get("cos_answer_full") is None or out["cos_answer_full"] > 0.99):
                raise SnrGate(f"gate: answer credit left the gradient unchanged (cos {out.get('cos_answer_full')})")

        async def finish(name, call, e, up, loaded, folder, grad, began):
            # the extract (a CPU subprocess) overlaps the next call's update; its checks gate the run as before
            ex = await tensors("extract", start, folder, grad, "--lr", lr, "--clip", clip, "--grad-norm", up["grad_norm"])
            shutil.rmtree(folder, ignore_errors=True)
            cred = [w.get("credited") for w in (up.get("windows") or [])]
            masked = bool(credit) and not call.get("full_of")
            rec = {"task": e["task"], "reward": e.get("reward"), "divisor": up.get("trained_tokens"),
                   "credited": sum(c or 0 for c in cred) if masked else None,
                   "credited_share": round(sum(c or 0 for c in cred) / max(1, sum(int(w.get("targets") or 0)
                                           for w in (up.get("windows") or []))), 5) if masked else None,
                   "full_of": call.get("full_of"),
                   "grad_norm": up.get("grad_norm"), "loss": up.get("loss"), "train_seconds": up.get("train_seconds"),
                   "windows": len(up.get("windows") or []), "errors": up.get("errors"), "loaded_updates": loaded.get("updates"),
                   "extract": {k: ex.get(k) for k in ("step_rel_err", "v_rel_err", "steps", "updates", "coef", "norm",
                                                      "update_norm", "norm2_by_group")},
                   "repeat_of": call.get("repeat_of")}
            why = [w for w, bad in (("load_checkpoint did not restore update 0", loaded.get("updates") != 0),
                                    ("update errors", bool(up.get("errors"))), ("no trained tokens", not up.get("trained_tokens")),
                                    ("windows outside 1..max_windows-1 (maybe cut)", not 1 <= rec["windows"] < max_windows),
                                    ("not one AdamW step", ex["steps"] != [1.0] or ex["updates"] != 1),
                                    ("adapter moved off -lr g/(|g|+eps)", ex["step_rel_err"] > 1e-3),
                                    ("credit asked but the trainer counted none (patch missing)", masked and any(c is None for c in cred)),
                                    ("credited share > 0.5 (mask not applied)", masked and (rec["credited_share"] or 0) > 0.5)) if bad]
            if masked and not why and rec["credited"] == 0:  # every reply cut at the cap: no answer tokens, nothing to bank
                rec.update(ok=False, skip="no answer tokens")
                report["calls"][name] = rec
                step("call", began, call=name, tokens=up.get("trained_tokens"), ok=False, skip=rec["skip"])
                return
            rec["ok"] = not why
            report["calls"][name] = rec
            step("call", began, call=name, tokens=up.get("trained_tokens"), tok_s=round((up.get("trained_tokens") or 0)
                 / max(1e-6, float(up.get("train_seconds") or 0)), 1), ok=rec["ok"])
            if why:
                raise SnrGate(f"{name}: {', '.join(why)}")

        for index, call in enumerate(todo):
            if index == gate_at:
                await gate_check()
            for task_ in [p for p in pending if p.done()]:
                task_.result()  # a failed check stops before the next call
            name, e = call["name"], call["episode"]
            grad = output / "grads" / f"{name.replace(':', '_')}.safetensors"
            kept = (old.get("calls") or {}).get(name)
            if same and kept and kept.get("ok") and (Path(resume) / "grads" / grad.name).is_file():
                if Path(resume).resolve() != output.resolve():
                    shutil.copyfile(Path(resume) / "grads" / grad.name, grad)
                report["calls"][name] = {**kept, "resumed": True}
                step("call_kept", time.time(), call=name)
                continue
            if deadline and time.time() > deadline - 300:
                report["cut_short"] = f"deadline before call {name}"
                break
            began = time.time()
            loaded = await trainer.must("load_checkpoint", path=str(start))
            attempt = {"rollout": e["rollout"], "w": 1.0, "name": e["name"], "whole": True, "share": e["task"]}
            if credit and not call.get("full_of"):
                attempt["credit"] = credit
            up = await trainer.must("update", timeout=cfg["update_timeout"], lr=lr, clip=clip, signed=False,
                                    attempts=[attempt], max_windows=max_windows, **(cfg.get("update_extra") or {}))
            folder = work / "calls" / name.replace(":", "_")
            await trainer.must("checkpoint", path=str(folder))
            while sum(not p.done() for p in pending) >= 2:
                await asyncio.sleep(0.2)
            report["calls"][name] = {"pending": True}  # keeps the plan's order in bank.json
            pending.append(asyncio.create_task(finish(name, call, e, up, loaded, folder, grad, began)))
        for task_ in pending:
            await task_
        phase("math")
        began = time.time()
        done = [c for c in calls if (report["calls"].get(c["name"]) or {}).get("ok")]
        spec = {"start": str(start), "episodes": [{"name": c["name"], "grad": str(output / "grads" / f"{c['name'].replace(':', '_')}.safetensors")}
                                                  for c in done]}
        rep = report["calls"].get(smallest["name"] + "~repeat")
        if rep and rep.get("ok") and any(c["name"] == smallest["name"] for c in done):
            spec["repeat"] = {"of": smallest["name"], "grad": str(output / "grads" / f"{(smallest['name'] + '~repeat').replace(':', '_')}.safetensors")}
        (work / "gram-spec.json").write_text(json.dumps(spec))
        report["gram"] = await tensors("gram", work / "gram-spec.json", output / "gram.json", timeout=1800)
        step("gram", began, n=len(done))
        repeat_cos = (report["gram"].get("repeat") or {}).get("cos")
        if spec.get("repeat") and (repeat_cos or 0.0) < 0.999:
            raise SnrGate(f"repeat call cosine {repeat_cos} < 0.999")
        report["gate"] = {"ok": True, "calls": len(done) + bool(spec.get("repeat")), "repeat_cos": repeat_cos,
                          "step_rel_err_max": max((c["extract"]["step_rel_err"] for c in report["calls"].values()
                                                   if c.get("extract")), default=None)}
        report["status"] = "partial" if report.get("cut_short") else "complete"
    except SnrGate as exc:
        report.update(status="gate_failed", gate={"ok": False, "why": str(exc)[:1000]})
    except Exception as exc:  # noqa: BLE001
        import traceback

        report.update(status="error", error=f"{type(exc).__name__}: {exc}"[:2000], traceback=traceback.format_exc()[-3000:])
    finally:
        if pending:  # let running extracts end: a cancelled communicate() at loop shutdown can hang asyncio.run
            await asyncio.wait(pending, timeout=600)
            for task_ in pending:
                if task_.done() and not task_.cancelled():
                    task_.exception()
        stop.set()
        sampler.stop.set()
        report["gpu"] = sampler.summary()
        report["elapsed_seconds"] = round(time.time() - t0, 1)
        save()
        trainer.close()
    return report


def snr_summary(report, doses):
    """The numbers the arm log needs: (a) halves, within/across, repeat; the zero-path KL; (b) per scope its points and
    the two-point fit per probe set at the target KL."""
    out = {"status": report.get("status")}
    a = report.get("analysis") or {}
    if a:
        h = a.get("halves") or {}
        out["a"] = {"halves_cos": h.get("cos"), "signal_share_full": h.get("signal_share"), "adam_cos": h.get("adam_cos"),
                    "sign_agree": h.get("sign_agree"), "groups_cos": {g: r.get("cos") for g, r in (h.get("groups") or {}).items()},
                    "within_task_cos": (a.get("within_task_summary") or {}).get("cos_mean"),
                    "across_task_cos": (a.get("across_task") or {}).get("cos_mean"),
                    "repeat_cos": (a.get("repeat") or {}).get("cos"), "episodes": h.get("episodes"), "tokens": h.get("tokens")}
    if report.get("zero_path"):
        z = report["zero_path"]
        out["zero_path"] = {"kl": z.get("kl"), "kl_held": z.get("kl_held"), "kl_lost": z.get("kl_lost"),
                            "per_probe": {n: v.get("kl_base") for n, v in (z.get("windows") or {}).items()}}
    scopes = {}
    for row in (report.get("points") or {}).values():
        scopes.setdefault(row["scope"], []).append(row)
    target = float(doses.get("target_kl", 0.02))
    out["b"] = {s: {"points": [{k: r.get(k) for k in ("D", "lr_equiv", "kl", "d_nll_held", "d_nll_lost", "d_nll_untrained")}
                               for r in sorted(rows, key=lambda r: r["D"])],
                    **{label: snr_fit(sorted(rows, key=lambda r: r["D"]), label, target) for label in ("held", "lost", "untrained")}}
                for s, rows in scopes.items()}
    return out
