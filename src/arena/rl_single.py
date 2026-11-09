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
import random
import re
import statistics
from pathlib import Path

from .rl_ladder import game_metrics, level_of


def clear_tokens(row):
    """Tokens to clear the level for a cleared try, else None."""
    level_rows = row.get("level_rows") or []
    if row.get("outcome") not in ("cleared", "won") or not level_rows:
        return None
    return int(level_rows[0].get("tokens") or row.get("completion_tokens") or 0) or None


def group_advantages(rows):
    """Per try: reward (T_ref / tokens for a clear, 0 otherwise) and leave-one-out advantage."""
    tokens = [clear_tokens(r) for r in rows]
    clears = [t for t in tokens if t]
    if not clears or len(rows) < 2:
        return [0.0] * len(rows), [0.0] * len(rows)
    ref = statistics.mean(clears)
    rewards = [ref / t if t else 0.0 for t in tokens]
    total, n = sum(rewards), len(rewards)
    return rewards, [r - (total - r) / (n - 1) for r in rewards]


WORD = re.compile(r"[a-z]{2,}", re.I)
CITE_N = 5


def word_grams(text, n=CITE_N):
    words = [w.lower() for w in WORD.findall(text or "")]
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def reasoning_of(reply):
    reply = reply.get("reply") or {}
    return reply.get("reasoning") or reply.get("reasoning_content") or ""


def message_text(messages):
    out = []
    for m in messages or []:
        c = m.get("content")
        out.append(c if isinstance(c, str) else " ".join(p.get("text", "") for p in (c or []) if isinstance(p, dict)))
    return "\n".join(out)


def hint_grams(hint, background, harness):
    """Hint-specific phrases (SFT arm, 15 Oct; v2b's regex fired on the no-hint arm's own "notes"): the hint body's
    word 5-grams (letters only: state lists such as red=[O,p,_,b] are re-derived by every play of the level) minus
    those in ``harness`` (the request text without a hint) and in ``background`` (the reasoning of the source's
    other recorded plays of that game). Checked on v2b: the no-hint arm cites the a008 hint in 0 of 237 replies."""
    body = hint.split("\n", 1)[1] if "\n" in hint else hint
    return word_grams(body) - word_grams(harness) - background


def background_grams(folder, skip):
    out = set()
    for path in sorted(Path(folder).glob("attempts/*/rollout.jsonl")):
        if path.parent.name == skip:
            continue
        with open(path) as file:
            for line in file:
                out |= word_grams(reasoning_of(json.loads(line)))
    return out


def hint_citations(folder, rows, grams):
    """Per try, the live replies whose reasoning contains a hint-specific phrase (``hint_grams``): (tries citing at
    least once, live replies citing, live replies)."""
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
                replies += 1
                if word_grams(reasoning_of(reply)) & grams:
                    cited += 1
                    hit = True
        tries += hit
    return {"tries_citing": tries, "of_tries": len(rows), "replies_citing": cited, "replies": replies,
            "reply_share": round(cited / replies, 4) if replies else None, "phrases": len(grams)}


def held_metrics(rows):
    toks = [clear_tokens(r) for r in rows]
    clears = [t for t in toks if t]
    return {"tries": len(rows), "clears": len(clears), "clear_rate": round(len(clears) / len(rows), 3) if rows else None,
            "median_to_clear": statistics.median(clears) if clears else None}


class SingleLevelMixin:
    def l2_start_pool(self):
        from .restore import level_starts, log_rows
        from .rl_loop import usable

        level, root = int(self.cfg["l2_level"]), Path(self.cfg["lvl_resume"]["folder"]) / "games-00"
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

    async def l2_rounds(self, start, probes):
        g, cap = self.cfg["train_game"], int(self.cfg["l2_cap"])
        held, train, _ = self.l2_start_pool()
        if not train or not held:
            raise RuntimeError(f"level {self.cfg['l2_level']}: {len(train)} train and {len(held)} held-out starts")
        self.l2_held_starts = held
        self.feed.emit("l2_starts", held=[f"{a.name[:4]}-k{q:03d}" for a, q in held],
                       train=[f"{a.name[:4]}-k{q:03d}" for a, q in train], cap=cap)
        if self.serving_adapter is None:
            self.serving_adapter = self.adapters / "r00"
        kept, stop, played_by, k = None, None, start - 1, int(self.cfg["l2_starts"])
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
            # train groups: rewards, leave-one-out advantages, the largest of each sign
            groups = self.l2_groups(trows)
            selected, weights, info, up, down = [], [], {}, 0.0, 0.0
            side = int(self.cfg["l2_per_side"])
            for point, grp in sorted(groups.items()):
                rewards, adv = group_advantages(grp)
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
            if self.left() < 3600 * float(self.cfg["l2_final_hours"]):
                stop = "wall"
                break
        return stop

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
        rng, batches = random.Random(number), []
        for _ in range(int(self.cfg["l2_passes"])):
            keys = sorted(groups)
            rng.shuffle(keys)
            batches += [groups[key] for key in keys]
        merged, path = await self.dosed_update(number, None, None, None, probes, batches=batches, signed=True,
                                               probe_last=True, max_windows=100000,
                                               cmd_extra={"share_prefix": True,
                                                          "prefix_chunk": int(self.cfg["lvl_prefix_chunk"]),
                                                          "norm": "try"})
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
        mismatch = self.cfg.get("opd_gate_mismatch")
        if mismatch:  # another game's clear (SFT arm, 15 Oct: sb26's own later levels state its slot-order rule)
            folder = Path(mismatch["folder"])
            if not folder.is_absolute():
                folder = self.source / folder
            m_rows = [r for r in json.loads((folder / "level-bench.json").read_text())["attempts"]
                      if str(r["game"]).startswith(mismatch["game"])]
            found = self.shortest_clear(folder, m_rows, int(mismatch["level"]))
            pick = (found, int(mismatch["level"])) if found else None
        else:
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
        if self.cfg.get("opd_gate_shared"):
            # one bench for every arm (SFT arm, 15 Oct: v2b's three concurrent benches each gated against the whole
            # pool, oversubscribed it 3x, hit 12.5% of the prefix cache and were cut at the wall): one KV gate, the
            # arms' starts interleaved pass by pass on the same seeds; an arm is the G<arm> tag of its points
            merged = []
            for name, (hint, _) in arms.items():
                merged += self.restore_points(self.l2_specs(held, f"G{name}", cap), f"gate-{name}", hint=hint)[1]
            path = self.work / "gate-points.json"
            path.write_text(json.dumps({"schema": 1, "points": merged}))
            folder, rows_all, play = await self.bench("gate", [g], tries, {g: cap}, points=path, model=self.server.model,
                                                      seed_base=seed, reserve=int(self.cfg["opd_gate_reserve"]),
                                                      concurrency=int(self.cfg["opd_gate_concurrency"]))
            results = [(folder, [r for r in rows_all if f"-L{level}-G{name}-" in str(r.get("point"))], play)
                       for name in arms]
        else:
            jobs = []
            for name, (hint, _) in arms.items():
                path, points = self.restore_points(self.l2_specs(held, "H", cap), f"gate-{name}", hint=hint)
                jobs.append(self.bench(f"gate-{name}", [g], tries, {g: cap}, points=path, model=self.server.model,
                                       seed_base=seed, concurrency=len(points) * tries))
            results = await asyncio.gather(*jobs)
        # hint-specific phrases; every arm is also scored against the own hint's phrases (the no-hint arm: ~0)
        # the request text without a hint: the first live request of every no-hint try (its restored context)
        harness = []
        folder_none, rows_none, _ = results[list(arms).index("none")]
        for r in rows_none:
            path = Path(folder_none) / "attempts" / r["label"] / "rollout.jsonl"
            if path.is_file():
                with open(path) as file:
                    first = next((x for x in map(json.loads, file) if not x.get("scripted")), None)
                if first:
                    harness.append(message_text(first.get("messages")))
        harness = "\n".join(harness)
        grams = {}
        for name, (hint, info) in arms.items():
            if hint:
                src = Path(info["source"])
                grams[name] = hint_grams(hint, background_grams(src.parent.parent, src.name), harness)
        out = {"tries_per_start": tries, "cap": cap, "seed_base": seed, "starts": len(held),
               "shared_bench": bool(self.cfg.get("opd_gate_shared"))}
        for (name, (hint, info)), (folder, rows_arm, play) in zip(arms.items(), results):
            groups = self.l2_groups(rows_arm)
            flat = [r for grp in groups.values() for r in grp]
            out[name] = {**held_metrics(flat), "planned": len(held) * tries, "censored": len(held) * tries - len(flat),
                         "play": play, "hint": info,
                         "per_start": {re.split(r"-[GH][a-z]*-", p, maxsplit=1)[-1]: held_metrics(grp)
                                       for p, grp in sorted(groups.items())},
                         "citations": hint_citations(folder, flat, grams[name]) if name in grams else None,
                         "cites_own_hint": hint_citations(folder, flat, grams["hint"]) if "hint" in grams else None}
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
