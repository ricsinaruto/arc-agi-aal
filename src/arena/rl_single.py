"""One-level RL: sb26 level ``l2_level`` only, fresh adapter, continuous token reward.

- starts: the level's starts in a stopped run's recorded base games (``lvl_resume.folder``/games-00), before the
  game's first compaction or timed-out python call; ``l2_held`` of them held out for evaluation, the rest trained
  round-robin, ``l2_starts`` per round;
- tries: ``l2_tries`` restored tries per train start and ``l2_eval_tries`` per held-out start, every try capped at
  ``l2_cap`` tokens (bounds play time only);
- reward: a clear scores T_ref / tokens (T_ref: the mean tokens of the group's clears), no clear 0; advantage:
  leave-one-out inside the start's group; per group the ``l2_per_side`` largest positive and negative advantages are
  trained (cost), positives as plain likelihood, negatives with the ratio truncated to [0, 1] (TOPR), each try's mean
  log-likelihood counted once (per-try normalisation), one round of lag (play on GPU 0, training on GPU 1);
- metric each round: on the held-out starts, clear rate within the cap and median tokens to clear (round 1 is the
  base: its adapter is the fresh one);
- final: ``l2_final_tries`` tries per held-out start for the adapter and the base on the same seeds, plus
  ``l2_final_games`` whole games per arm.
"""

import asyncio
import json
import os
import random
import re
import shutil
import statistics
from pathlib import Path

from .rl_ladder import game_metrics, level_of


def clear_tokens(row):
    """Tokens to clear the level for a cleared try, else None."""
    level_rows = row.get("level_rows") or []
    if row.get("outcome") not in ("cleared", "won") or not level_rows:
        return None
    return int(level_rows[0].get("tokens") or row.get("completion_tokens") or 0) or None


def clear_rhae(row):
    """The cleared level's score as a fraction (RHAE, min(1.15, (baseline / actions)^2)), else None."""
    level_rows = row.get("level_rows") or []
    if row.get("outcome") not in ("cleared", "won") or not level_rows or level_rows[0].get("score") is None:
        return None
    return float(level_rows[0]["score"]) / 100.0


def group_advantages(rows, reward="tokens"):
    """Per try: reward and leave-one-out advantage. ``tokens``: T_ref / tokens for a clear (T_ref: the mean tokens of
    the group's clears), 0 otherwise; ``rhae_tokens``: the level's RHAE x T_ref / tokens (score per
    token, scale-free)."""
    tokens = [clear_tokens(r) for r in rows]
    clears = [t for t in tokens if t]
    if not clears or len(rows) < 2:
        return [0.0] * len(rows), [0.0] * len(rows)
    ref = statistics.mean(clears)
    rewards = [ref / t if t else 0.0 for t in tokens]
    if reward == "rhae_tokens":
        rewards = [r * (clear_rhae(row) or 0.0) for r, row in zip(rewards, rows)]
    total, n = sum(rewards), len(rewards)
    return rewards, [r - (total - r) / (n - 1) for r in rewards]


CITES = re.compile(r"\b(my|the|these|those|earlier|previous|old) notes?\b|\bhint\b|earlier play|previous play|"
                   r"notes (say|said|show)|from (my|the) notes", re.I)


def hint_citations(folder, rows):
    """Per try, the share of its live replies whose reasoning cites the hint (``CITES``): (tries citing at least
    once, live replies citing, live replies)."""
    tries = cited = replies = 0
    for row in rows:
        path = Path(folder) / "attempts" / row["label"] / "rollout.jsonl"
        if not path.is_file():
            continue
        hit = False
        with open(path) as file:
            for line in file:
                reply = json.loads(line)
                if reply.get("scripted"):
                    continue
                text = (reply.get("reply") or {}).get("reasoning") or (reply.get("reply") or {}).get("reasoning_content") or ""
                replies += 1
                if CITES.search(text):
                    cited += 1
                    hit = True
        tries += hit
    return {"tries_citing": tries, "of_tries": len(rows), "replies_citing": cited, "replies": replies,
            "reply_share": round(cited / replies, 4) if replies else None}


def held_metrics(rows):
    toks = [clear_tokens(r) for r in rows]
    clears = [t for t in toks if t]
    rhae = [v for v in (clear_rhae(r) for r in rows) if v is not None]
    actions = [int(r["level_rows"][0].get("actions") or 0) for r in rows if clear_tokens(r)]
    spent = sum(int(r.get("completion_tokens") or 0) for r in rows)
    return {"tries": len(rows), "clears": len(clears), "clear_rate": round(len(clears) / len(rows), 3) if rows else None,
            "median_to_clear": statistics.median(clears) if clears else None,
            "median_actions": statistics.median(actions) if actions else None,
            "mean_rhae": round(statistics.mean(rhae), 4) if rhae else None,
            "rhae_per_10k": round(sum(rhae) / spent * 1e4, 4) if spent else None}


class SingleLevelMixin:
    def l2_start_pool(self):
        from .restore import level_starts, log_rows
        from .rl_loop import usable

        level = int(self.cfg["l2_level"])
        root = getattr(self, "l2_source", None) or Path(self.cfg["lvl_resume"]["folder"]) / "games-00"
        rows = [r for r in json.loads((root / "level-bench.json").read_text())["attempts"] if usable(r)]
        out = []
        for row in sorted(self.game_rows(rows, self.cfg["train_game"]), key=lambda r: r["label"]):
            attempt = root / "attempts" / row["label"]
            stops = [int(e.get("step") or 0) for e in log_rows(attempt / "log.jsonl",
                                                               ("compaction", "playbook_handoff", "duck_python_interrupt"))]
            first = min(stops) if stops else None
            for request, done, step in level_starts(attempt):
                if done + 1 == level and (first is None or step < first):
                    out.append((attempt, int(request)))
        rng = random.Random(int(self.cfg["l2_seed"]))
        rng.shuffle(out)
        held = out[: int(self.cfg["l2_held"])]
        return held, out[int(self.cfg["l2_held"]):], rows

    def l2_specs(self, picks, kind, cap):
        level = int(self.cfg["l2_level"])
        return [(attempt, request, f"L{level}-{kind}-{attempt.name[:4]}-k{request:03d}", False, cap)
                for attempt, request in picks]

    def l2_groups(self, rows):
        groups = {}
        for r in self.game_rows(rows, self.cfg["train_game"]):
            if (r.get("restore") or {}).get("ok") and level_of(r) is not None:
                groups.setdefault(r["point"], []).append(r)
        return groups

    async def l2_record(self):
        """Base games from level 1, each stopped by a small token budget soon after level ``l2_level`` begins: one
        start per game (the source of ``l2_start_pool``)."""
        rec = self.cfg["l2_record"]
        g = self.cfg["train_game"]
        folder, rows, play = await self.bench("l2-record", [g], int(rec["games"]), {g: int(rec["budget"])},
                                              model=self.server.model, seed_base=int(rec.get("seed", 9000)),
                                              concurrency=int(rec["games"]), reserve=120)
        self.l2_source = Path(folder)
        self.feed.emit("l2_record", folder=str(folder), games=len(rows), play=play,
                       reached=sum(1 for r in rows if len(r.get("level_rows") or []) >= int(self.cfg["l2_level"]) - 1))
        return folder

    async def l2_rounds(self, start, probes):
        g, cap = self.cfg["train_game"], int(self.cfg["l2_cap"])
        if self.cfg["l2_record"] and getattr(self, "l2_source", None) is None:
            await self.l2_record()
        held, train, _ = self.l2_start_pool()
        if not train or not held:
            raise RuntimeError(f"level {self.cfg['l2_level']}: {len(train)} train and {len(held)} held-out starts")
        self.l2_held_starts = held
        self.feed.emit("l2_starts", held=[f"{a.name[:4]}-k{q:03d}" for a, q in held],
                       train=[f"{a.name[:4]}-k{q:03d}" for a, q in train], cap=cap)
        if self.serving_adapter is None:
            self.serving_adapter = self.adapters / "r00"
        kept, stop, played_by, k, streak = None, None, start - 1, int(self.cfg["l2_starts"]), 0
        self.l2_base = None
        for number in range(start, self.cfg["rounds"] + 1):
            began = asyncio.get_event_loop().time()
            picks = [train[((number - 1) * k + i) % len(train)] for i in range(min(k, len(train)))]
            tpath, tpoints = self.restore_points(self.l2_specs(picks, "T", cap), f"l2-train-{number:02d}")
            hpath, hpoints = self.restore_points(self.l2_specs(held, "H", cap), f"l2-held-{number:02d}")
            tries, etries = int(self.cfg["l2_tries"]), int(self.cfg["l2_eval_tries"])
            jobs = [self.bench(f"l2-train-{number:02d}", [g], tries, {g: cap}, points=tpath,
                               concurrency=len(tpoints) * tries),
                    self.bench(f"l2-held-{number:02d}", [g], etries, {g: cap}, points=hpath,
                               concurrency=len(hpoints) * etries, seed_base=int(self.cfg["l2_eval_seed"]))]
            if kept:
                jobs.append(self.l2_update(number, kept, probes))
            results = await asyncio.gather(*jobs)
            (tfolder, trows, tplay), (hfolder, hrows, hplay) = results[0], results[1]
            row = {"round": number, "played_by": played_by}
            if kept:
                update_row, serve = results[2]
                row["update"] = {k2: v for k2, v in update_row.items() if k2 not in ("traceback", "windows")}
                if not update_row.get("ok"):
                    stop = row["stop"] = update_row.get("stop") or "update_failed"
                    self.feed.emit("round", **row)
                    break
                kl = [p.get("kl_base") for p in (update_row.get("probes") or {}).values() if p.get("kl_base") is not None]
                limit = self.cfg["l2_kl_stop"]
                if limit is not None and kl and max(kl) > float(limit):
                    # KL guard: the new adapter is not served; the run ends after this round's
                    # row and the final is played by the last adapter within the limit
                    row["kl_guard"] = {"kl_base_max": max(kl), "limit": float(limit), "kept_serving": str(self.serving_adapter)}
                    stop = "kl_guard"
                else:
                    row["swap"] = {"load_seconds": await self.server.load("policy", serve)}
                    self.serving_adapter = serve
                    played_by = number
            # held-out metric (round 1: the fresh adapter, i.e. the base)
            hgroups = self.l2_groups(hrows)
            metric = held_metrics([r for grp in hgroups.values() for r in grp])
            metric["per_start"] = {p.split("-H-", 1)[-1]: held_metrics(grp) for p, grp in sorted(hgroups.items())}
            if self.l2_base is None:
                self.l2_base = metric
            row["held"] = metric
            bad = self.l2_bad_round(metric) if row["played_by"] > 0 else None  # the adapter that played this round
            row["early_stop_signal"] = bad
            streak = streak + 1 if bad else 0
            # train groups: rewards, leave-one-out advantages, the largest of each sign
            groups = self.l2_groups(trows)
            selected, weights, info, up, down = [], [], {}, 0.0, 0.0
            side = int(self.cfg["l2_per_side"])
            for point, grp in sorted(groups.items()):
                rewards, adv = group_advantages(grp, self.cfg["l2_reward"])
                order = sorted(range(len(grp)), key=lambda i: -adv[i])
                pick = [i for i in order[:side] if adv[i] > 0] + [i for i in order[::-1][:side] if adv[i] < 0]
                selected += [grp[i] for i in pick]
                weights += [adv[i] for i in pick]
                up += sum(adv[i] for i in pick if adv[i] > 0)
                down += sum(-adv[i] for i in pick if adv[i] < 0)
                m = held_metrics(grp)
                info[point.split("-T-", 1)[-1]] = {**m, "trained": len(pick)}
            all_rows = json.loads((Path(tfolder) / "level-bench.json").read_text())["attempts"]
            failed = sum(1 for r in all_rows if not (r.get("restore") or {}).get("ok"))
            row["train"] = {"play": tplay, "held_play": hplay, "restored": len(all_rows) - failed, "of": len(all_rows),
                            "starts": info, "trained": len(selected),
                            "balance": {"up": round(up, 4), "down": round(down, 4),
                                        "up_share": round(up / (up + down), 4) if up + down else None}}
            row["round_seconds"] = round(asyncio.get_event_loop().time() - began, 1)
            kept = (Path(tfolder), selected, weights) if selected else None
            self.feed.emit("round", **row)
            self.history.append(row)
            if failed > float(self.cfg["lvl_restore_fail"]) * max(1, len(all_rows)) and not info:
                stop = "restore"
                break
            if stop == "kl_guard":
                self.feed.emit("kl_guard", round=number, **row["kl_guard"])
                break
            if streak >= int(self.cfg["l2_stop_rounds"]):
                stop = "early_stop"
                self.feed.emit("early_stop", round=number, signal=bad, base=self.l2_base)
                break
            if self.left() < 3600 * float(self.cfg["l2_final_hours"]):
                stop = "wall"
                break
        return stop

    def l2_bad_round(self, metric):
        """Why a policy-played round counts toward the early stop, else None."""
        rate, ratio = self.cfg["l2_stop_rate"], self.cfg["l2_stop_median"]
        base = (self.l2_base or {}).get("median_to_clear")
        if rate is not None and metric["clear_rate"] is not None and metric["clear_rate"] <= float(rate):
            return f"clear rate {metric['clear_rate']} <= {rate}"
        if ratio is not None and base and metric["median_to_clear"] and metric["median_to_clear"] > float(ratio) * base:
            return f"median {metric['median_to_clear']} > {ratio} x base {base}"
        return None

    async def l2_update(self, number, kept, probes):
        """One pass (``l2_passes``) over the selected tries: one optimizer step per start (its tries share the
        prefix); signed (negatives TOPR), per-try normalisation."""
        folder, rows, weights = kept
        specs = [{**s, "w": float(w), "whole": True, "window": int(self.cfg["lvl_window"]), "floor": 1024,
                  "margin": 1024, "credited_only": True, "share": r.get("point") or r["label"]}
                 for s, w, r in zip(self.windows(folder, rows, prefix="t-"), weights, rows) if w != 0]
        groups = {}
        for spec in specs:
            groups.setdefault(spec["share"], []).append(spec)
        rng, batches, per = random.Random(number), [], max(1, int(self.cfg["l2_starts_per_step"]))
        for _ in range(int(self.cfg["l2_passes"])):
            keys = sorted(groups)
            rng.shuffle(keys)
            # one optimizer step per l2_starts_per_step starts (each start's tries stay together: shared prefix)
            batches += [[spec for key in keys[i:i + per] for spec in groups[key]] for i in range(0, len(keys), per)]
        merged, path = await self.dosed_update(number, None, None, None, probes, batches=batches, signed=True,
                                               probe_last=True, max_windows=100000,
                                               cmd_extra={"share_prefix": True,
                                                          "prefix_chunk": int(self.cfg["lvl_prefix_chunk"]),
                                                          "norm": self.cfg["l2_norm"],
                                                          "norm_cap": int(self.cfg["l2_cap"])})
        merged["trained_tries"] = len(specs)
        if merged.get("ok"):
            self.checkpoint(number)
            await self.save_checkpoint()
        return merged, path

    async def l2_finish(self, stop):
        """Paired held-out tries (adapter and base, same seeds) and whole games per arm."""
        if stop in ("nan_or_inf", "update_failed"):
            self.feed.emit("final", stop=stop, eval=None)
            return stop
        g, cap = self.cfg["train_game"], int(self.cfg["l2_cap"])
        n, seed = int(self.cfg["l2_final_tries"]), int(self.cfg["l2_final_seed"])
        path, points = self.restore_points(self.l2_specs(self.l2_held_starts, "H", cap), "l2-final")
        names = {"policy": "policy", "base": self.server.model}
        jobs = [self.bench(f"final-l2-{tag}", [g], n, {g: cap}, points=path, model=model, seed_base=seed,
                           concurrency=len(points) * n, reserve=120) for tag, model in names.items()]
        games = int(self.cfg["l2_final_games"])
        if games:
            jobs += [self.bench(f"final-games-{tag}", [g], games, {g: int(self.cfg["lvl_game_budget"])},
                                seed_base=int(self.cfg["lvl_final_seed"]), model=model, concurrency=games, reserve=120,
                                wall=int(self.cfg["lvl_game_wall"])) for tag, model in names.items()]
        results = await asyncio.gather(*jobs)
        out = {"seed_base": seed, "tries_per_start": n, "cap": cap}
        per = {}
        for (tag, _), (_, rows, play) in zip(names.items(), results[:2]):
            groups = self.l2_groups(rows)
            flat = [r for grp in groups.values() for r in grp]
            out[tag] = held_metrics(flat)
            out[f"{tag}_play"] = play
            per[tag] = {p.split("-H-", 1)[-1]: held_metrics(grp) for p, grp in groups.items()}
        out["per_start"] = {s: {"policy": per["policy"].get(s), "base": per["base"].get(s)}
                            for s in sorted(set(per["policy"]) | set(per["base"]))}
        pol, base = out["policy"], out["base"]
        if pol["median_to_clear"] and base["median_to_clear"]:
            out["token_ratio"] = round(base["median_to_clear"] / pol["median_to_clear"], 3)
        if games:
            for (tag, _), (_, rows, play) in zip(names.items(), results[2:]):
                metrics = game_metrics(self.game_rows(rows, g))
                out[f"games_{tag}"] = {k: v for k, v in metrics.items() if k != "per_game"}
        self.feed.emit("eval", label="final", **out)
        self.feed.emit("final", stop=stop, eval=out, base_round1=self.l2_base)
        return stop


def paired_summary(pairs, cap):
    """``pairs`` [(policy row, base row)] on the same start and seed; a try without a clear costs ``cap``; per pair
    the log of base / policy tokens."""
    import math

    cost = lambda r: clear_tokens(r) or cap
    logs = [math.log(cost(b) / cost(p)) for p, b in pairs]
    both = [(clear_tokens(p), clear_tokens(b)) for p, b in pairs]
    if not logs:
        return {"pairs": 0}
    mean = statistics.mean(logs)
    se = statistics.stdev(logs) / math.sqrt(len(logs)) if len(logs) > 1 else None
    return {"pairs": len(logs), "mean_log_base_over_policy": round(mean, 4), "se": None if se is None else round(se, 4),
            "token_ratio_geo": round(math.exp(mean), 3), "policy_cheaper": sum(v > 0 for v in logs),
            "policy_only_clear": sum(1 for p, b in both if p and not b), "base_only_clear": sum(1 for p, b in both if b and not p)}


class TransferMixin:
    """Transfer check: the adapter and the base on restored starts of levels the adapter never
    trained on, the same starts and seeds for both arms."""

    def transfer_starts(self, spec):
        from .restore import level_starts, log_rows
        from .rl_loop import usable

        pool = {}
        for src in spec["sources"]:
            root = Path(src)
            rows = [r for r in json.loads((root / "level-bench.json").read_text())["attempts"] if usable(r)]
            for row in sorted(self.game_rows(rows, self.cfg["train_game"]), key=lambda r: r["label"]):
                attempt = root / "attempts" / row["label"]
                stops = [int(e.get("step") or 0) for e in log_rows(attempt / "log.jsonl",
                                                                   ("compaction", "playbook_handoff", "duck_python_interrupt"))]
                first = min(stops) if stops else None
                spent = {int(l["level"]): l.get("tokens") for l in row.get("level_rows") or []}
                with open(attempt / "rollout.jsonl") as file:
                    prompts = [int((json.loads(line).get("usage") or {}).get("prompt_tokens") or 0) for line in file]
                for request, done, step in level_starts(attempt):
                    # max_prompt: starts whose prompt leaves room in the cache for every try at once
                    if (done + 1 in spec["levels"] and (first is None or step < first)
                            and prompts[int(request)] <= int(spec.get("max_prompt") or 10**9)):
                        pool.setdefault(done + 1, []).append((attempt, int(request), spent.get(done + 1)))
        picks = {}
        for level in spec["levels"]:
            cands = pool.get(level, [])
            random.Random(int(spec.get("pick_seed", 0)) * 100 + level).shuffle(cands)
            picks[level] = cands[: int(spec["starts"])]
        return pool, picks

    async def transfer_eval(self, spec):
        g, cap, n, seed = self.cfg["train_game"], int(spec["cap"]), int(spec["tries"]), int(spec["seed"])
        pool, picks = self.transfer_starts(spec)
        specs = [(a, q, f"L{level}-X-{a.parent.parent.name[-2:]}{a.name[:4]}-k{q:03d}", False, cap)
                 for level, chosen in sorted(picks.items()) for a, q, _ in chosen]
        self.feed.emit("transfer_starts", available={k: len(v) for k, v in sorted(pool.items())},
                       picked={k: [{"start": f"{a.parent.parent.name}/{a.name}/k{q:03d}", "source_tokens": t}
                                   for a, q, t in v] for k, v in sorted(picks.items())}, cap=cap, tries=n, seed=seed)
        path, points = self.restore_points(specs, "transfer")
        names = {"policy": "policy", "base": self.server.model}
        results = await asyncio.gather(*(self.bench(f"transfer-{tag}", [g], n, {g: cap}, points=path, model=model,
                                                    seed_base=seed, concurrency=len(points) * n, reserve=120)
                                         for tag, model in names.items()))
        out = {"cap": cap, "tries_per_start": n, "seed_base": seed, "starts": len(points)}
        by = {}
        for (tag, _), (folder, rows, play) in zip(names.items(), results):
            out[f"{tag}_play"] = play
            all_rows = json.loads((Path(folder) / "level-bench.json").read_text())["attempts"]
            out[f"{tag}_unusable"] = len(all_rows) - len(rows)
            groups = self.l2_groups(rows)
            flat = [r for grp in groups.values() for r in grp]
            out[tag] = held_metrics(flat)
            for r in flat:
                by.setdefault(tag, {})[(r["point"], r.get("pass"), r["label"].rsplit("-p", 1)[-1])] = r
            per_level = {}
            for r in flat:
                per_level.setdefault(level_of(r), []).append(r)
            out.setdefault("per_level", {})
            for level, grp in sorted(per_level.items()):
                out["per_level"].setdefault(str(level), {})[tag] = held_metrics(grp)
        keys = sorted(set(by.get("policy", {})) & set(by.get("base", {})), key=str)
        out["paired"] = paired_summary([(by["policy"][k], by["base"][k]) for k in keys], cap)
        for level in sorted({level_of(by["policy"][k]) for k in keys}):
            out["per_level"][str(level)]["paired"] = paired_summary(
                [(by["policy"][k], by["base"][k]) for k in keys if level_of(by["policy"][k]) == level], cap)
        self.feed.emit("eval", label="transfer", **out)
        return out


class OpdMixin:
    """On-policy distillation: the student (the policy without a hint) plays sb26 level ``l2_level``
    from the single-level starts; the teacher is the CURRENT policy with a hint in its system message (an
    iterated teacher): the shortest clear the policy has produced so far, as its action list and its own reasoning
    before the clear (``arena.restore.clear_hint``). The loss per sampled token is the reverse-KL advantage log
    q_teacher - log p_student (``rl1_trainer``, ``--loss opd``). Metric: held-out starts without a hint."""

    def shortest_clear(self, folder, rows, level, skip=()):
        """``(tokens, attempt folder)`` of the cheapest clear of ``level`` among ``rows`` of a bench folder."""
        from .restore import clear_hint

        best = None
        for row in rows:
            attempt = Path(folder) / "attempts" / row["label"]
            if row["label"] in skip or not (attempt / "rollout.jsonl").is_file():
                continue
            try:
                _, info = clear_hint(attempt, level)
            except Exception:  # an unreadable record is not a candidate
                continue
            if info and info["tokens"] and (best is None or info["tokens"] < best[0]):
                best = (info["tokens"], attempt)
        return best

    async def opd_gate(self):
        """Teacher gate: the held-out starts, ``opd_gate_tries`` tries each per arm on the base, the same seeds:
        no hint; the shortest recorded clear of the level as the hint; the shortest recorded clear of another level
        (mismatched). Clear rate, median tokens to clear and how often the reasoning cites the hint, per arm."""
        from .restore import clear_hint

        g, cap, level = self.cfg["train_game"], int(self.cfg["l2_cap"]), int(self.cfg["l2_level"])
        held, _, rows = self.l2_start_pool()
        source = Path(self.cfg["lvl_resume"]["folder"]) / "games-00"
        own = self.shortest_clear(source, rows, level)
        others = [(c, other) for other in self.cfg["opd_gate_other_levels"]
                  for c in [self.shortest_clear(source, rows, int(other))] if c]
        pick = min(others, key=lambda item: item[0][0]) if others else None
        arms = {"none": (None, None)}
        if own:
            arms["hint"] = clear_hint(own[1], level)
        if pick:
            arms["mismatched"] = clear_hint(pick[0][1], int(pick[1]))
        self.feed.emit("opd_gate_hints", **{name: {"info": info, "hint": hint} for name, (hint, info) in arms.items()})
        tries, seed = int(self.cfg["opd_gate_tries"]), int(self.cfg["opd_gate_seed"])
        jobs = []
        for name, (hint, _) in arms.items():
            path, points = self.restore_points(self.l2_specs(held, "H", cap), f"gate-{name}", hint=hint)
            jobs.append(self.bench(f"gate-{name}", [g], tries, {g: cap}, points=path, model=self.server.model,
                                   seed_base=seed, concurrency=len(points) * tries))
        results = await asyncio.gather(*jobs)
        out = {"tries_per_start": tries, "cap": cap, "seed_base": seed, "starts": len(held)}
        for (name, (hint, info)), (folder, rows_arm, play) in zip(arms.items(), results):
            groups = self.l2_groups(rows_arm)
            flat = [r for grp in groups.values() for r in grp]
            out[name] = {**held_metrics(flat), "play": play, "hint": info,
                         "per_start": {p.split("-H-", 1)[-1]: held_metrics(grp) for p, grp in sorted(groups.items())},
                         "citations": hint_citations(folder, flat) if hint else None}
        base = (out.get("none") or {}).get("median_to_clear")
        for name in ("hint", "mismatched"):
            if name in out and base and out[name].get("median_to_clear"):
                out[name]["token_ratio"] = round(base / out[name]["median_to_clear"], 3)
        self.feed.emit("opd_gate", **out)
        return out

    async def opd_rounds(self, start, probes):
        """As ``l2_rounds`` (train and held-out benches each round, the update one round behind), but every
        restored train try is trained by OPD against the hinted current policy; the hint is refreshed to the
        cheapest clear so far (the recorded base games', then every round's train tries')."""
        from .restore import clear_hint

        g, cap, level = self.cfg["train_game"], int(self.cfg["l2_cap"]), int(self.cfg["l2_level"])
        held, train, rows0 = self.l2_start_pool()
        if not train or not held:
            raise RuntimeError(f"level {level}: {len(train)} train and {len(held)} held-out starts")
        self.l2_held_starts = held
        self.feed.emit("l2_starts", held=[f"{a.name[:4]}-k{q:03d}" for a, q in held],
                       train=[f"{a.name[:4]}-k{q:03d}" for a, q in train], cap=cap)
        best = self.shortest_clear(Path(self.cfg["lvl_resume"]["folder"]) / "games-00", rows0, level)
        if best is None:
            raise RuntimeError(f"no recorded clear of level {level} for the first hint")
        if self.serving_adapter is None:
            self.serving_adapter = self.adapters / "r00"
        kept, stop, played_by, k = None, None, start - 1, int(self.cfg["l2_starts"])
        self.l2_base = None
        for number in range(start, self.cfg["rounds"] + 1):
            began = asyncio.get_event_loop().time()
            hint, hint_info = clear_hint(best[1], level)
            picks = [train[((number - 1) * k + i) % len(train)] for i in range(min(k, len(train)))]
            tpath, tpoints = self.restore_points(self.l2_specs(picks, "T", cap), f"l2-train-{number:02d}")
            hpath, hpoints = self.restore_points(self.l2_specs(held, "H", cap), f"l2-held-{number:02d}")
            tries, etries = int(self.cfg["l2_tries"]), int(self.cfg["l2_eval_tries"])
            jobs = [self.bench(f"l2-train-{number:02d}", [g], tries, {g: cap}, points=tpath,
                               concurrency=len(tpoints) * tries),
                    self.bench(f"l2-held-{number:02d}", [g], etries, {g: cap}, points=hpath,
                               concurrency=len(hpoints) * etries, seed_base=int(self.cfg["l2_eval_seed"]))]
            if kept:
                jobs.append(self.opd_update(number, kept, probes, hint))
            results = await asyncio.gather(*jobs)
            (tfolder, trows, tplay), (hfolder, hrows, hplay) = results[0], results[1]
            row = {"round": number, "played_by": played_by, "hint": hint_info if kept else None}
            if kept:
                update_row, serve = results[2]
                row["update"] = {k2: v for k2, v in update_row.items() if k2 not in ("traceback", "windows")}
                if not update_row.get("ok"):
                    stop = row["stop"] = update_row.get("stop") or "update_failed"
                    self.feed.emit("round", **row)
                    break
                row["swap"] = {"load_seconds": await self.server.load("policy", serve)}
                self.serving_adapter = serve
                played_by = number
            hgroups = self.l2_groups(hrows)
            metric = held_metrics([r for grp in hgroups.values() for r in grp])
            metric["per_start"] = {p.split("-H-", 1)[-1]: held_metrics(grp) for p, grp in sorted(hgroups.items())}
            if self.l2_base is None:
                self.l2_base = metric
            row["held"] = metric
            groups = self.l2_groups(trows)
            flat = [r for grp in groups.values() for r in grp]
            tokens = [int(r.get("completion_tokens") or 0) for r in flat]
            row["train"] = {"play": tplay, "held_play": hplay, **held_metrics(flat), "trained": len(flat),
                            "mean_try_tokens": round(statistics.mean(tokens)) if tokens else None}
            found = self.shortest_clear(tfolder, flat, level)
            if found and found[0] < best[0]:
                best = found
            row["best_clear_tokens"] = best[0]
            row["round_seconds"] = round(asyncio.get_event_loop().time() - began, 1)
            kept = (Path(tfolder), flat) if flat else None
            self.feed.emit("round", **row)
            self.history.append(row)
            if not flat:
                stop = "restore"
                break
            if self.left() < 3600 * float(self.cfg["l2_final_hours"]):
                stop = "wall"
                break
        return stop

    async def opd_update(self, number, kept, probes, hint):
        """One optimizer step per start over its tries (shared prefixes), every try weight 1 and per-try
        normalised; the teacher's hint is the round's."""
        folder, rows = kept
        specs = [{**s, "w": 1.0, "whole": True, "window": int(self.cfg["lvl_window"]), "floor": 1024, "margin": 1024,
                  "credited_only": True, "share": r.get("point") or r["label"]}
                 for s, r in zip(self.windows(folder, rows, prefix="t-"), rows)]
        groups = {}
        for spec in specs:
            groups.setdefault(spec["share"], []).append(spec)
        keys = sorted(groups)
        random.Random(number).shuffle(keys)
        merged, path = await self.dosed_update(number, None, None, None, probes, batches=[groups[key] for key in keys],
                                               probe_last=True, max_windows=100000,
                                               cmd_extra={"share_prefix": True,
                                                          "prefix_chunk": int(self.cfg["lvl_prefix_chunk"]),
                                                          "norm": "try",
                                                          "opd": {"hint": hint, "clip": float(self.cfg["opd_clip"])}})
        parts = [w.get("opd") or {} for w in merged.get("windows", [])]
        counts = [p.get("tokens") or 0 for p in parts]
        if sum(counts):
            merged["opd"] = {"reverse_kl": round(sum((p.get("reverse_kl") or 0) * c for p, c in zip(parts, counts))
                                                 / sum(counts), 5),
                             "up_share": round(sum((p.get("up_share") or 0) * c for p, c in zip(parts, counts))
                                               / sum(counts), 4),
                             "teacher_seconds": round(sum(float((w.get("teacher") or {}).get("seconds") or 0)
                                                          for w in merged.get("windows", [])), 1),
                             "tokens": sum(counts)}
        merged["trained_tries"] = len(specs)
        if merged.get("ok"):
            self.checkpoint(number)
            await self.save_checkpoint()
        return merged, path


class MultiMixin:
    """Multi-game baseline. ``mg`` spec: ``cap`` (tokens per try), ``sets``
    [{name, game, level, role train|test, fresh | sources [folders | "record:<game>"], starts, tries, seed,
    max_prompt, untrained_game, round_tries}], ``record`` [{game, games, budget, seed}].

    Mode ``screen`` (one card, base only): the record benches, then every set's tries; writes ``screen.json`` (per
    set: its points file, bench folder, per-start tries and clears, metrics). Mode ``loop`` with ``mg.screen`` (two
    cards): l4's method on the screen's mixed train starts, dynamic sampling (groups whose tries all cleared or all
    failed are not trained), the update one round behind; each played round also plays every test set (same
    starts and seeds as the screen's base)."""

    def mg_point(self, game):
        from .level_bench import load_points

        return load_points(self.source / self.study["level_bench"]["start_points"],
                           {**self.study["level_bench"], "games": [game], "split": "all"})[0]

    def mg_starts(self, spec, sources):
        """Restored starts of ``spec``'s level in ``sources``' usable base games of its game (before any compaction
        or python time-out, prompt <= max_prompt), shuffled by pick_seed, the first ``starts``."""
        from .restore import level_starts, log_rows
        from .rl_loop import usable

        pool = []
        for src in sources:
            root = Path(src)
            rows = [r for r in json.loads((root / "level-bench.json").read_text())["attempts"] if usable(r)]
            for row in sorted(self.game_rows(rows, spec["game"]), key=lambda r: r["label"]):
                attempt = root / "attempts" / row["label"]
                stops = [int(e.get("step") or 0) for e in log_rows(attempt / "log.jsonl",
                                                                   ("compaction", "playbook_handoff", "duck_python_interrupt"))]
                first = min(stops) if stops else None
                with open(attempt / "rollout.jsonl") as file:
                    prompts = [int((json.loads(line).get("usage") or {}).get("prompt_tokens") or 0) for line in file]
                for request, done, step in level_starts(attempt):
                    if (done + 1 == int(spec["level"]) and (first is None or step < first)
                            and prompts[int(request)] <= int(spec.get("max_prompt") or 10**9)):
                        pool.append((str(attempt), int(request)))
        random.Random(int(self.cfg["mg"].get("pick_seed", 0)) * 100 + int(spec["level"]) * 7 + len(spec["game"])).shuffle(pool)
        return pool[: int(spec["starts"])], len(pool)

    @staticmethod
    def mg_tag(spec, attempt, request):
        import hashlib

        return f"L{spec['level']}-{hashlib.md5(str(attempt).encode()).hexdigest()[:6]}-k{int(request):03d}"

    def mg_points(self, spec, starts, name):
        """The points file of one set: a fresh level-1 segment of its game, or its restored starts."""
        from .restore import load

        cap, base = int(spec.get("cap") or self.cfg["mg"]["cap"]), self.mg_point(spec["game"])
        if spec.get("fresh"):
            points = [{**base, "id": f"{base['id']}-L1-{spec['name']}", "through": False, "segment": True, "level": 1,
                       "level_index": 0, "prefix": [], "budget": cap}]
        else:
            points = []
            for attempt, request in starts:
                tag = self.mg_tag(spec, attempt, request)
                points.append({**base, "id": f"{base['id']}-{tag}", "level": load(attempt, request)["level"],
                               "restore": {"attempt": attempt, "request": int(request), "check_only": False, "budget": cap,
                                           **({"pin_actions": True} if spec.get("pin_actions") else {})}})
        path = self.work / f"{name}-points.json"
        path.write_text(json.dumps({"schema": 1, "points": points}))
        return path, points

    async def mg_play(self, name, spec, starts, tries, *, model, seed, concurrency=None):
        path, points = self.mg_points(spec, starts, name)
        folder, rows, play = await self.bench(name, [spec["game"]], int(tries),
                                              {spec["game"]: int(spec.get("cap") or self.cfg["mg"]["cap"])},
                                              points=path, model=model, seed_base=seed,
                                              concurrency=concurrency or len(points) * int(tries), reserve=120)
        rows = [r for r in self.game_rows(rows, spec["game"])
                if (not r.get("restore") or r["restore"].get("ok")) and level_of(r) is not None]
        return folder, rows, play

    @staticmethod
    def mg_groups(rows, size=4):
        """Tries grouped by start; a fresh set's tries (one start) in groups of ``size`` by pass."""
        groups = {}
        for r in rows:
            p = int(r.get("pass") or r["label"].rsplit("-p", 1)[-1])
            key = r["point"] if r.get("restore") else f"{r['point']}#g{p // size}"
            groups.setdefault(key, []).append(r)
        return groups

    def mg_select(self, folder, rows):
        """Dynamic sampling and l4's selection: per mixed group (some but not all tries cleared) the per_side largest
        positive and negative advantages (reward l2_reward). Returns ([(folder, row, weight, share)], info)."""
        out, info, side = [], {"groups": 0, "mixed": 0, "trained": 0}, int(self.cfg["l2_per_side"])
        for key, grp in sorted(self.mg_groups(rows, int(self.cfg["mg"].get("group_size") or 4)).items()):
            info["groups"] += 1
            clears = sum(1 for r in grp if clear_tokens(r))
            if not 0 < clears < len(grp):
                continue
            info["mixed"] += 1
            _, adv = group_advantages(grp, self.cfg["l2_reward"])
            order = sorted(range(len(grp)), key=lambda i: -adv[i])
            pick = [i for i in order[:side] if adv[i] > 0] + [i for i in order[::-1][:side] if adv[i] < 0]
            out += [(Path(folder), grp[i], adv[i], key) for i in pick]
        info["trained"] = len(out)
        return out, info

    async def mg_screen(self):
        mg, cap = self.cfg["mg"], int(self.cfg["mg"]["cap"])
        base_model, recorded = self.server.model, {}

        async def record(rec):
            folder, rows, play = await self.bench(f"rec-{rec['game']}", [rec["game"]], int(rec["games"]),
                                                  {rec["game"]: int(rec["budget"])}, model=base_model,
                                                  seed_base=int(rec.get("seed", 9100)), concurrency=int(rec["games"]),
                                                  reserve=120)
            recorded[rec["game"]] = str(folder)
            self.feed.emit("mg_record", game=rec["game"], folder=str(folder), play=play,
                           levels=sorted(len(r.get("level_rows") or []) for r in rows))

        async def play_set(spec):
            sources = [recorded[s.split(":", 1)[1]] if s.startswith("record:") else s for s in spec.get("sources") or []]
            starts, available = ([], 1) if spec.get("fresh") else self.mg_starts(spec, sources)
            folder, rows, play = await self.mg_play(f"screen-{spec['name']}", spec, starts, spec["tries"],
                                                    model=base_model, seed=int(spec.get("seed", mg.get("seed", 8000))))
            groups = self.mg_groups(rows)
            toks = [clear_tokens(r) for r in rows]
            at_caps = {c: sum(1 for t in toks if t and t <= c) for c in mg.get("at_caps") or []}
            out = {"role": spec["role"], "game": spec["game"], "level": spec["level"], "folder": str(folder),
                   "cap": int(spec.get("cap") or cap), "at_caps": at_caps, "tries": len(rows),
                   "compactions": sum(int(r.get("compactions") or 0) for r in rows),
                   "starts": [list(s) for s in starts], "available": available, "play": play,
                   "metrics": held_metrics(rows),
                   "groups": {k: {"tries": len(g), "clears": sum(1 for r in g if clear_tokens(r)),
                                  "mixed": 0 < sum(1 for r in g if clear_tokens(r)) < len(g)} for k, g in sorted(groups.items())}}
            self.feed.emit("mg_set", name=spec["name"], **{k: v for k, v in out.items() if k != "groups"},
                           mixed=sum(g["mixed"] for g in out["groups"].values()), of=len(out["groups"]))
            return spec["name"], out

        late = [s for s in mg["sets"] if any(x.startswith("record:") for x in s.get("sources") or [])]
        early = [s for s in mg["sets"] if s not in late]
        first = await asyncio.gather(*(record(r) for r in mg.get("record") or []), *(play_set(s) for s in early))
        second = await asyncio.gather(*(play_set(s) for s in late))
        sets = dict(x for x in list(first) + list(second) if x)
        screen = {"cap": cap, "spec": mg, "recorded": recorded, "sets": sets}
        (self.output / "rl" / "screen.json").write_text(json.dumps(screen, indent=1, default=str))
        self.feed.emit("final", stop=None, eval={n: {"metrics": s["metrics"], "mixed": sum(g["mixed"] for g in s["groups"].values()),
                                                     "of": len(s["groups"])} for n, s in sets.items()}, base=None)
        return {n: s["metrics"] for n, s in sets.items()}

    def mg_screen_file(self):
        """screen.json with its container paths (/tmp/arena/...) mapped to the screen run's volume copy
        (``mg.screen_root``, e.g. /runs/validation/rl1-screen-mg1/arena); ``mg.screens`` [{screen, root}] add later
        screens' sets."""
        mg = self.cfg["mg"]
        out = None
        for item in [{"screen": mg["screen"], "root": mg.get("screen_root")}] + list(mg.get("screens") or []):
            text, root = Path(item["screen"]).read_text(), item.get("root")
            one, raw = json.loads(text.replace('"/tmp/arena/', f'"{root.rstrip("/")}/') if root else text), json.loads(text)
            for n, s in one["sets"].items():  # screen rows' point ids hash the unmapped paths
                s["raw_starts"] = raw["sets"][n].get("starts")
            if out is None:
                out = one
            else:  # a later screen's sets replace or add to the earlier ones
                out["sets"].update(one["sets"])
        return out

    def mg_rows(self, folder):
        from .rl_loop import usable

        return [r for r in json.loads((Path(folder) / "level-bench.json").read_text())["attempts"] if usable(r)
                and (not r.get("restore") or r["restore"].get("ok")) and level_of(r) is not None]

    async def mg_update(self, number, kept, probes):
        specs = []
        for folder, row, w, share in kept:
            spec = self.windows(folder, [row], prefix="t-")[0]
            specs.append({**spec, "w": float(w), "whole": True, "window": int(self.cfg["lvl_window"]), "floor": 1024,
                          "margin": 1024, "credited_only": True, "share": share})
        groups = {}
        for spec in specs:
            groups.setdefault(spec["share"], []).append(spec)
        rng, per = random.Random(number), max(1, int(self.cfg["l2_starts_per_step"]))
        keys = sorted(groups)
        rng.shuffle(keys)
        # l4's dose per update (owner): at most max_groups groups (l4: 8 starts, 2 per step = 4 steps)
        keys = keys[: int(self.cfg["mg"].get("max_groups") or len(keys))]
        specs = [sp for key in keys for sp in groups[key]]
        batches = [[s for key in keys[i:i + per] for s in groups[key]] for i in range(0, len(keys), per)]
        merged, path = await self.dosed_update(number, None, None, None, probes, batches=batches, signed=True,
                                               probe_last=True, max_windows=100000,
                                               cmd_extra={"share_prefix": True, "prefix_chunk": int(self.cfg["lvl_prefix_chunk"]),
                                                          "norm": self.cfg["l2_norm"], "norm_cap": int(self.cfg["mg"]["cap"])})
        merged["trained_tries"] = len(specs)
        return merged, path

    def mg_tests(self, results, screen):
        """Per test set and pooled untrained games: the adapter's metrics beside the screen's base."""
        out, pools = {}, {"untrained_games": ([], []), "primary": ([], [])}
        for name, (folder, rows, play) in results.items():
            spec = next(s for s in self.cfg["mg"]["sets"] if s["name"] == name)
            cap = int(spec.get("cap") or self.cfg["mg"]["cap"])
            # the screen's base tries at this set's cap (the screen played a looser cap; the model never sees it)
            base_rows = [self.mg_censor(r, cap) for r in self.mg_rows(screen["sets"][name]["folder"])]
            out[name] = {"policy": held_metrics(rows), "base": held_metrics(base_rows), "play_s": play["seconds"]}
            for key, flag in (("untrained_games", "untrained_game"), ("primary", "primary")):
                if spec.get(flag):
                    pools[key][0].extend(rows)
                    pools[key][1].extend(base_rows)
        for key, (pol, base) in pools.items():
            out[key] = {"policy": held_metrics(pol), "base": held_metrics(base)}
        return out

    @staticmethod
    def mg_censor(row, cap):
        """A base try played at a looser cap, as it would end at ``cap``: a clear beyond the cap is a fail, its
        spent tokens the cap."""
        t = clear_tokens(row)
        if t is not None and t <= cap:
            return row
        return {**row, "outcome": "budget", "level_rows": [], "completion_tokens": min(int(row.get("completion_tokens") or cap), cap)}

    def mg_rules(self, number, plays, row, played_by):
        """stop rules, on one round: (a) the round played after 4 policy updates (and its
        predecessor) shows no gain on the training sets at their caps against the base rounds 1-2 (clear rate not
        higher and median tokens to clear not lower); (b) the primary unseen-level trend clearly worse than base in 2
        consecutive rounds (pooled clear rate below base - 0.2, or RHAE per 10K below 0.67 x base); (d) a bench's
        usable share below 0.75. (c) is the KL guard's second trip, in the round body. Returns the rule or None."""
        mg = self.cfg["mg"]
        for n, (folder, rows, play) in plays.items():
            share = len(rows) / play["attempts"] if play.get("attempts") else 1.0
            if share < float(mg.get("min_usable", 0.75)):
                row["rule_detail"] = {"bench": n, "usable_share": round(share, 3)}
                return "rule_d_usable"
        train = [r for n, (_, rows, _) in plays.items() if self.cfg["mg_sets"][n]["role"] == "train" for r in rows]
        self.mg_train_rows = getattr(self, "mg_train_rows", {})
        self.mg_train_rows[number] = (played_by, train)

        def per_request(rows):  # completion tokens per model request over a round's played tries
            asked = sum(int(r.get("admitted_requests") or r.get("gameplay_requests") or 0) for r in rows)
            return round(sum(int(r.get("completion_tokens") or 0) for r in rows) / asked, 1) if asked else None

        played = train + [r for n, (_, rows, _) in plays.items() if self.cfg["mg_sets"][n]["role"] == "test" for r in rows]
        base_tpr = per_request([r for k in (1, 2) for r in self.mg_train_rows.get(k, (0, []))[1]])
        row["tokens_per_request"] = {"round": per_request(played), "train": per_request(train), "base": base_tpr}
        limit = mg.get("length_guard")  # (e) above limit x base in 2 policy rounds running
        if limit and played_by >= 2 and base_tpr and train:  # training sets against the base's training rounds
            long = (per_request(train) or 0) > float(limit) * base_tpr
            self.mg_long = getattr(self, "mg_long", 0) + 1 if long else 0
            row["tokens_per_request"].update(over=long, streak=self.mg_long)
            if self.mg_long >= 2:
                row["rule_detail"] = row["tokens_per_request"]
                return "rule_e_length"
        if played_by >= 2 and played_by - 1 == int(mg.get("rule_a_updates", 4)):  # 4 updates: U2..U5 -> a5
            base = [r for k in (1, 2) for r in self.mg_train_rows.get(k, (0, []))[1]]
            pol = train + self.mg_train_rows.get(number - 1, (0, []))[1]
            b, p = held_metrics(base), held_metrics(pol)
            row["rule_a"] = {"base": b, "policy": p}
            better = ((p["clear_rate"] or 0) > (b["clear_rate"] or 0)
                      or (p["median_to_clear"] and b["median_to_clear"] and p["median_to_clear"] < b["median_to_clear"]))
            if not better:
                row["rule_detail"] = row["rule_a"]
                return "rule_a_no_gain"
        prim = (row.get("tests") or {}).get("primary")
        if prim and prim["policy"]["tries"]:
            p, b = prim["policy"], prim["base"]
            worse = ((p["clear_rate"] or 0) < (b["clear_rate"] or 0) - 0.2
                     or (b["rhae_per_10k"] and (p["rhae_per_10k"] or 0) < 0.67 * b["rhae_per_10k"]))
            self.mg_worse = getattr(self, "mg_worse", 0) + 1 if worse else 0
            row["rule_b"] = {"worse": worse, "streak": self.mg_worse}
            if self.mg_worse >= 2:
                row["rule_detail"] = prim
                return "rule_b_primary_worse"
        return None

    def mg_ok_starts(self, name, screen):
        """A set's screen starts that restored (at least one usable base try); a fresh set has none."""
        spec = next(s for s in self.cfg["mg"]["sets"] if s["name"] == name)
        points = {r["point"] for r in self.mg_rows(screen["sets"][name]["folder"])}
        return self.mg_matched(spec, screen["sets"][name], points)

    def mg_matched(self, spec, entry, points):
        """``entry``'s (mapped) starts whose screen point (tagged from the unmapped path) is in ``points``."""
        raw = entry.get("raw_starts") or entry["starts"]
        return [tuple(st) for st, r in zip(entry["starts"], raw)
                if any(p.endswith("-" + self.mg_tag(spec, r[0], r[1])) for p in points)]

    async def mg_loop(self):
        """l4's schedule: rounds 1-2 played by the base (round 1 beside the trainer's load), the update of round r on
        round r-1's selection, the adapter swapped after it; tests from round 3 (a trend; the decision final is a
        separate job); the last round plays tests only. Pool: the screen's starts with a clear."""
        mg = self.cfg["mg"]
        screen = self.mg_screen_file()
        sets = {s["name"]: s for s in mg["sets"]}
        self.cfg["mg_sets"] = sets
        pool = {}
        for name, s in sets.items():
            if s["role"] == "train" and not s.get("fresh"):
                rows = self.mg_rows(screen["sets"][name]["folder"])
                good = {r["point"] for r in rows if clear_tokens(r)}
                pool[name] = self.mg_matched(s, screen["sets"][name], good)
        tests = [n for n, s in sets.items() if s["role"] == "test"]
        lvl_rows = self.mg_rows(screen["sets"][tests[0]]["folder"])[:2]
        probes = self.windows(screen["sets"][tests[0]]["folder"], lvl_rows, prefix="heldout-")
        for n in [n for n in tests if sets[n].get("untrained_game")][:2]:
            probes += self.windows(screen["sets"][n]["folder"], self.mg_rows(screen["sets"][n]["folder"])[:1],
                                   prefix="untrained-")
        self.feed.emit("mg_pool", pool={n: len(v) for n, v in pool.items()}, probes=[p["name"] for p in probes])

        def train_jobs(number, model):
            jobs, names = [], []
            for name, s in sets.items():
                if s["role"] != "train":
                    continue
                k, st = int(mg.get("train_starts", 2)), [tuple(x) for x in pool.get(name, [])]
                st = [st[((number - 1) * k + i) % len(st)] for i in range(min(k, len(st)))] if st else []
                if s.get("fresh") or st:
                    jobs.append(self.mg_play(f"r{number:02d}-{name}", s, st,
                                             mg.get("fresh_tries", 8) if s.get("fresh") else mg.get("train_tries", 4),
                                             model=model, seed=None))
                    names.append(name)
            return jobs, names

        async def start():
            trainer = await self.start_trainer()
            self.feed.emit("trainer", **trainer)
            await self.trainer.must("score", windows=probes, label="base")

        self.serving_adapter, played_by, stop, data, last_round_s = None, 0, None, [], None
        last = int(mg.get("updates", 8)) + 1  # rounds 1..last; updates in rounds 2..last
        for number in range(1, last + 1):
            began = asyncio.get_event_loop().time()
            if number > 1 and self.left() < (last_round_s or 0) + 120:
                self.feed.emit("mg_time", round=number, left=round(self.left()), need=last_round_s)
                break
            model = "policy" if self.serving_adapter is not None else self.server.model
            jobs, names = train_jobs(number, model) if number < last else ([], [])
            if self.serving_adapter is not None:
                for name in tests:
                    s = sets[name]
                    st = self.mg_ok_starts(name, screen)
                    jobs.append(self.mg_play(f"r{number:02d}-{name}", s, st, s.get("round_tries", s["tries"]),
                                             model="policy", seed=int(s.get("seed", mg.get("seed", 8000)))))
                    names.append(name)
            extra = start() if number == 1 else (self.mg_update(number, data, probes) if data else None)
            results = await asyncio.gather(*jobs, *([extra] if extra is not None else []))
            row = {"round": number, "played_by": played_by}
            plays = dict(zip(names, results[: len(names)]))
            if any(sets[n]["role"] == "test" for n in plays):
                row["tests"] = self.mg_tests({n: r for n, r in plays.items() if sets[n]["role"] == "test"}, screen)
            new, tinfo = [], {}
            for n, (folder, rows, play) in plays.items():
                if sets[n]["role"] == "train":
                    sel, tinfo[n] = self.mg_select(folder, rows)
                    tinfo[n]["clears"] = f"{sum(1 for r in rows if clear_tokens(r))}/{len(rows)}"
                    new += sel
            row["train"] = tinfo
            rule = self.mg_rules(number, plays, row, played_by)
            if number > 1 and extra is not None:
                update_row, serve = results[-1]
                row["update"] = {k: v for k, v in update_row.items() if k not in ("traceback", "windows")}
                if not update_row.get("ok"):
                    stop = row["stop"] = update_row.get("stop") or "update_failed"
                else:
                    kl = [p.get("kl_base") for p in (update_row.get("probes") or {}).values() if p.get("kl_base") is not None]
                    if self.cfg["l2_kl_stop"] is not None and kl and max(kl) > float(self.cfg["l2_kl_stop"]):
                        # rule (c): the new adapter is not served; the second trip ends the run
                        self.kl_trips = getattr(self, "kl_trips", 0) + 1
                        row["kl_guard"] = {"kl_base_max": max(kl), "limit": float(self.cfg["l2_kl_stop"]),
                                           "kept_serving": str(self.serving_adapter), "trips": self.kl_trips}
                        if self.kl_trips >= 2:
                            stop = "rule_c_kl_guard_twice"
                    else:
                        row["swap"] = {"load_seconds": await self.server.load("policy", serve)}
                        self.serving_adapter, played_by = serve, number
            data = new
            if rule and stop is None:
                stop = row["stop"] = rule
                self.feed.emit("stop_rule", round=number, rule=rule, detail=row.get("rule_detail"))
            row["round_seconds"] = last_round_s = round(asyncio.get_event_loop().time() - began, 1)
            self.feed.emit("round", **row)
            self.history.append(row)
            if stop is not None:
                break
        if self.serving_adapter is not None and os.environ.get("ARC_VALIDATION_OUTPUT"):
            dest = Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "adapters" / Path(self.serving_adapter).name
            await asyncio.to_thread(shutil.copytree, self.serving_adapter, dest, dirs_exist_ok=True)
            self.feed.emit("mg_adapter", path=str(dest), played_by=played_by)
        self.feed.emit("final", stop=stop, eval=(self.history[-1] if self.history else {}).get("tests"), base=None)
        return stop

    async def mg_final(self):
        """Mode final with mg.final {set name: tries}: the adapter and the base on each set's screen starts with the
        same seeds, both at once; per set and pooled untrained games: metrics per arm and the paired log token ratio
        (a try without a clear at its set's cap)."""
        import math

        mg, sets = self.cfg["mg"], {s["name"]: s for s in self.cfg["mg"]["sets"]}
        screen = self.mg_screen_file()
        jobs, keys, starts = [], [], {n: self.mg_ok_starts(n, screen) for n in mg["final"]}
        size = {n: max(1, len(starts[n])) * int(t) for n, t in mg["final"].items()}
        total = mg.get("final_concurrency")  # tries in flight over all jobs, split by each job's share of the tries
        for name, tries in mg["final"].items():
            s = {**sets[name], "cap": sets[name].get("final_cap") or sets[name].get("cap")}  # the final's own cap
            conc = max(1, round(int(total) * size[name] / (2 * sum(size.values())))) if total else None
            for tag, model in (("policy", "policy"), ("base", self.server.model)):
                jobs.append(self.mg_play(f"final-{tag}-{name}", s, starts[name], int(tries), model=model,
                                         seed=int(s.get("final_seed", 20000)), concurrency=conc))
                keys.append((name, tag))
        results = dict(zip(keys, await asyncio.gather(*jobs)))
        out, pools = {}, {g: {"policy": [], "base": [], "pairs": []} for g in ("untrained_games", "primary")}
        for name in mg["final"]:
            cap = int(sets[name].get("final_cap") or sets[name].get("cap") or mg["cap"])
            by = {tag: {(r["point"], r["label"].rsplit("-p", 1)[-1]): r for r in results[(name, tag)][1]}
                  for tag in ("policy", "base")}
            pairs = [(by["policy"][k], by["base"][k]) for k in sorted(set(by["policy"]) & set(by["base"]))]
            logs = [math.log((clear_tokens(b) or cap) / (clear_tokens(p) or cap)) for p, b in pairs]
            out[name] = {"policy": held_metrics(results[(name, "policy")][1]), "base": held_metrics(results[(name, "base")][1]),
                         "paired": paired_summary(pairs, cap),
                         "play_s": {t: results[(name, t)][2]["seconds"] for t in ("policy", "base")}}
            for g in [g for g, key in (("untrained_games", "untrained_game"), ("primary", "primary")) if sets[name].get(key)]:
                pools[g]["policy"] += results[(name, "policy")][1]
                pools[g]["base"] += results[(name, "base")][1]
                pools[g]["pairs"] += logs
        for g, pooled in pools.items():
            v = pooled["pairs"]
            if not v:
                continue
            m = statistics.mean(v)
            se = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None
            out[g] = {"policy": held_metrics(pooled["policy"]), "base": held_metrics(pooled["base"]),
                      "pairs": len(v), "token_ratio_geo": round(math.exp(m), 3),
                      "ci95": [round(math.exp(m - 2 * se), 3), round(math.exp(m + 2 * se), 3)] if se else None,
                      "pair_sd": round(statistics.stdev(v), 4) if len(v) > 1 else None,
                      "policy_cheaper": sum(x > 0 for x in v)}
        self.feed.emit("eval", label="mg_final", **out)
        return out
