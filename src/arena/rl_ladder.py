"""Level ladder: Flash wins all 8 sb26 levels in a tenth of the base's tokens.

Per round, on two cards (play on GPU 0, the trainer on GPU 1, one round of lag):

- whole games: ``lvl_games`` sb26 games from level 1 with the adapter of the round they start in (served under the
  name ``game``, so a batch keeps one adapter), up to ``lvl_game_budget`` tokens, in the background; tokens to win all
  8 levels is the primary metric, and every level start of these games is a practice state;
- level tries: per level, ``lvl_tries`` restored tries from one level start of the newest finished batch, each within
  the level's token budget ``B_l``; a pass is a clear within ``B_l``; passes are kept (positive only) with rank weights
  (fewer tokens, fewer actions: higher RHAE), mean 1;
- the update: every token of each kept try (one window per planned window end), ``lvl_passes`` passes over the round's
  kept tries in batches of ``lvl_batch``; then ``B_l`` shrinks by ``lvl_shrink`` (held while a level has fewer than
  ``lvl_hold_below`` passes).
"""

import asyncio
import json
import random
import statistics
import time
from pathlib import Path

LEVELS = 8


def level_of(row):
    """The level a restored try starts on (its point id carries ``-L<n>-``)."""
    return next((int(part[1:]) for part in str(row.get("point", "")).split("-") if part[:1] == "L" and part[1:].isdigit()),
                None)


def game_metrics(rows, levels=LEVELS):
    """Whole games: wins (all ``levels`` cleared), tokens to win (completion tokens at the last clear), levels and
    level score (sum of min(115, score) / 100, the bench's RHAE-style reward)."""
    out, wins = [], []
    for r in rows:
        level_rows = r.get("level_rows") or []
        won = len(level_rows) >= levels
        to_win = int(level_rows[levels - 1].get("at_tokens") or 0) if won else None
        score = round(sum(min(115.0, float(x.get("score") or 0.0)) for x in level_rows) / 100.0, 4)
        out.append({"label": r["label"], "won": won, "tokens_to_win": to_win, "levels": len(level_rows),
                    "tokens": int(r.get("completion_tokens") or 0), "score": score})
        if won:
            wins.append(to_win)
    return {"games": len(rows), "won": len(wins), "tokens_to_win": sorted(wins),
            "median_to_win": statistics.median(wins) if wins else None,
            "levels_mean": round(statistics.mean(o["levels"] for o in out), 3) if out else None,
            "score_mean": round(statistics.mean(o["score"] for o in out), 4) if out else None, "per_game": out}


def level_tokens(rows):
    """Per level the completion tokens each clear took (whole games)."""
    out = {}
    for r in rows:
        for x in r.get("level_rows") or []:
            out.setdefault(int(x["level"]), []).append(int(x.get("tokens") or 0))
    return out


def rank_weights(passes):
    """Rank weights over a level's passes: the mean of the token rank and the action rank (fewer is better), best
    first; weight 2 (n - i) / (n + 1) for the i-th best (mean 1, no cap to saturate)."""
    n = len(passes)
    if not n:
        return []
    tokens = sorted(range(n), key=lambda i: passes[i]["tokens"])
    actions = sorted(range(n), key=lambda i: passes[i]["actions"])
    rank = {i: (tokens.index(i) + actions.index(i)) / 2 for i in range(n)}
    order = sorted(range(n), key=lambda i: (rank[i], passes[i]["tokens"]))
    weights = [0.0] * n
    for position, i in enumerate(order):
        weights[i] = 2.0 * (n - position) / (n + 1)
    return weights


def try_record(row, budget):
    """One level try: cleared within the budget (a pass), its tokens and actions on the level."""
    level_rows = row.get("level_rows") or []
    tokens = int(row.get("completion_tokens") or 0)
    actions = int(level_rows[0].get("actions") or 0) if level_rows else int(row.get("total_actions") or 0)
    cleared = row.get("outcome") in ("cleared", "won") and bool(level_rows)
    return {"label": row["label"], "pass": cleared and tokens <= budget, "tokens": tokens, "actions": actions,
            "restored": bool((row.get("restore") or {}).get("ok"))}


class LadderMixin:
    def lvl_starts(self, batches, seed):
        """Per level one start (attempt folder, request) from the newest finished batch that reached it: only level
        starts before the game's first compaction or python time-out, never a start
        that failed to restore before (``lvl_bad``)."""
        from .restore import level_starts, log_rows

        rng, out, bad = random.Random(seed), [], getattr(self, "lvl_bad", set())
        for level in range(1, int(self.cfg["lvl_levels"]) + 1):
            for folder, rows in reversed(batches):
                options = []
                for row in rows:
                    attempt = Path(folder) / "attempts" / row["label"]
                    # a timed-out python call (rl1-loop-l1 round 3: the L8 start's game had 2) need not repeat on
                    # replay, nor does a compaction: starts come from before the first of either
                    compacted = [int(e.get("step") or 0) for e in log_rows(
                        attempt / "log.jsonl", ("compaction", "playbook_handoff", "duck_python_interrupt"))]
                    first = min(compacted) if compacted else None
                    for request, done, step in level_starts(attempt):
                        tag = f"L{level}-{Path(folder).name}-{attempt.name[:4]}-k{request:03d}"
                        key = (Path(folder).name, attempt.name, int(request))
                        if done + 1 == level and (first is None or step < first) and key not in bad:
                            options.append((attempt, request, tag))
                if options:
                    attempt, request, tag = rng.choice(sorted(options))
                    out.append((attempt, request, tag, False, int(self.budgets[level])))
                    break
        return out

    def lvl_take_tries(self, folder, rows, number):
        """One round's tries: drop the starts that failed to restore (half or more of their tries: logged, never
        picked again), then per level the passes within B_l, their rank weights (the best ``lvl_keep_max`` kept) and
        the next B_l. Returns (kept or None, the round row's ``tries`` part, restore failures outside dropped starts)."""
        g = self.cfg["train_game"]
        all_tries = json.loads((Path(folder) / "level-bench.json").read_text())["attempts"]
        by_point, source = {}, {}
        for r in all_tries:
            by_point.setdefault(r.get("point"), []).append(bool((r.get("restore") or {}).get("ok")))
            info = r.get("restore") or {}
            if info.get("source"):  # (batch folder, attempt, request): stable across copies of the run folder
                src = Path(info["source"])
                source[r.get("point")] = (src.parent.parent.name, src.name, int(info.get("request") or 0))
        self.lvl_bad = getattr(self, "lvl_bad", set())
        dropped = []
        for point, oks in by_point.items():
            if oks.count(False) * 2 >= len(oks):
                if point in source:
                    self.lvl_bad.add(source[point])
                dropped.append({"start": str(point).split("-game-", 1)[-1], "failed": oks.count(False), "of": len(oks),
                                "source": list(source.get(point) or [])})
        if dropped:
            self.feed.emit("lvl_bad_start", round=number, starts=dropped)
        bad_points = {p for p, oks in by_point.items() if oks.count(False) * 2 >= len(oks)}
        failed = sum(1 for r in all_tries if r.get("point") not in bad_points and not (r.get("restore") or {}).get("ok"))
        counted = sum(1 for r in all_tries if r.get("point") not in bad_points)
        by_level, kept_rows, kept_w = {}, [], []
        for r in self.game_rows(rows, g):
            if (r.get("restore") or {}).get("ok") and level_of(r) is not None and r.get("point") not in bad_points:
                by_level.setdefault(level_of(r), []).append(r)
        levels = {}
        for level, group in sorted(by_level.items()):
            records = [try_record(r, self.budgets[level]) for r in group]
            passes = [(r, rec) for r, rec in zip(group, records) if rec["pass"]]
            weights = rank_weights([rec for _, rec in passes])
            best = sorted(range(len(passes)), key=lambda i: -weights[i])[:int(self.cfg["lvl_keep_max"])]
            kept_rows += [passes[i][0] for i in best]  # per-start cap (practice S1): easy levels cannot dominate
            kept_w += [weights[i] for i in best]
            levels[level] = {"tries": len(group), "passes": len(passes), "kept": len(best), "budget": int(self.budgets[level]),
                             "pass_tokens_median": statistics.median([rec["tokens"] for _, rec in passes]) if passes else None,
                             "tokens_median": statistics.median([rec["tokens"] for rec in records])}
            if len(passes) >= int(self.cfg["lvl_hold_below"]):
                self.budgets[level] = max(256.0, self.budgets[level] * float(self.cfg["lvl_shrink"]))
        kept = (Path(folder), kept_rows, kept_w) if kept_rows else None
        part = {"restored": sum(1 for r in all_tries if (r.get("restore") or {}).get("ok")), "of": len(all_tries),
                "dropped_starts": dropped, "levels": levels, "kept": len(kept_rows)}
        return kept, part, (failed, counted)

    async def lvl_resume(self, probes):
        """Continue a stopped ladder (``lvl_resume``: state, round, folder with its games-* and tries-* folders,
        finished batches, that round's B_l before its shrink, tries folder): the state's adapter serves, the base
        metrics come from games-00, the round's tries are scored again (bad starts dropped) and trained next round."""
        from .rl_loop import usable

        res = self.cfg["lvl_resume"]
        g, root, number = self.cfg["train_game"], Path(res["folder"]), int(res["round"])
        loaded = load_seconds = None
        if res.get("state"):  # else a fresh adapter on the stopped run's base games and budgets
            loaded = await self.trainer.must("load_checkpoint", path=str(res["state"]))
            serve = self.adapters / f"r{number:02d}"
            await self.trainer.must("save", path=str(serve))
            load_seconds = await self.server.load("policy", serve)
            self.serving_adapter = serve

        def rows_of(name):
            return [r for r in json.loads((root / name / "level-bench.json").read_text())["attempts"] if usable(r)]
        self.lvl_base = game_metrics(self.game_rows(rows_of("games-00"), g))
        batches = [(root / name, self.game_rows(rows_of(name), g)) for name in res["batches"]]
        self.budgets = {int(k): float(v) for k, v in res["budgets"].items()}
        kept, part, failed, counted = None, None, 0, 0
        if res.get("tries"):
            kept, part, (failed, counted) = self.lvl_take_tries(root / res["tries"], rows_of(res["tries"]), number)
        self.feed.emit("lvl_resume", round=number, state=str(res["state"]), loaded=loaded, load_seconds=load_seconds,
                       batches=list(res["batches"]), tries=part, next_budgets={k: int(v) for k, v in self.budgets.items()},
                       failed_outside_dropped=[failed, counted])
        return number + 1, batches, kept

    def measured_budgets(self, rows):
        """B_l from base games on this harness: the median tokens of a level's clears where at least 6 cleared,
        else the configured value (the c100 medians)."""
        mine = level_tokens(rows)
        out = {}
        for level in range(1, int(self.cfg["lvl_levels"]) + 1):
            values = mine.get(level, [])
            out[level] = float(statistics.median(values)) if len(values) >= 6 else float(self.budgets[level])
        return out, {lv: {"n": len(v), "median": statistics.median(v)} for lv, v in sorted(mine.items()) if v}

    async def lvl_update(self, number, kept, probes):
        """Every token of each kept try, ``lvl_passes`` passes; a batch holds whole starts (about ``lvl_batch``
        tries), so the trainer runs each start's shared prefix once per step; each window is the try's own span
        (the restored history stays in the no-gradient prefix)."""
        folder, rows, weights = kept
        specs = [{**s, "w": float(w), "whole": True, "window": int(self.cfg["lvl_window"]), "floor": 1024,
                  "margin": 1024, "credited_only": True, "share": r.get("point") or r["label"]}
                 for s, w, r in zip(self.windows(folder, rows, prefix="t-"), weights, rows) if w > 0]
        groups = {}
        for spec in specs:
            groups.setdefault(spec["share"], []).append(spec)
        rng, batches, size = random.Random(number), [], int(self.cfg["lvl_batch"])
        for _ in range(int(self.cfg["lvl_passes"])):
            keys = sorted(groups)
            rng.shuffle(keys)
            batch = []
            for key in keys:
                batch += groups[key]
                if len(batch) >= size:
                    batches.append(batch)
                    batch = []
            if batch:
                batches.append(batch)
        merged, path = await self.dosed_update(number, None, None, None, probes, batches=batches, signed=False,
                                               probe_last=True, max_windows=100000,
                                               cmd_extra={"share_prefix": True,
                                                          "prefix_chunk": int(self.cfg["path_chunk"]),
                                                          "norm": self.cfg["lvl_norm"]})  # "try": each try counts once
        merged["kept"] = len(specs)
        if merged.get("ok"):
            self.checkpoint(number)
            await self.save_checkpoint()
        return merged, path

    async def lvl_rounds(self, start, probes):
        g, e = self.cfg["train_game"], self.cfg["eval_game"]
        self.budgets = {int(k): float(v) for k, v in (self.cfg["lvl_budgets"] or {}).items()}
        budget, tries = int(self.cfg["lvl_game_budget"]), int(self.cfg["lvl_tries"])
        base = self.server.model
        wall = int(self.cfg["lvl_game_wall"])
        kept = None
        if self.cfg.get("lvl_resume"):
            start, batches, kept = await self.lvl_resume(probes)
        else:
            folder0, rows0, play0 = await self.bench("games-00", [g], int(self.cfg["lvl_round0_games"]), {g: budget},
                                                     model=base, concurrency=int(self.cfg["lvl_round0_games"]), wall=wall)
            metrics0 = game_metrics(self.game_rows(rows0, g))
            self.lvl_base = metrics0
            self.feed.emit("lvl_games", round=0, model="base", play=play0,
                           **{k: v for k, v in metrics0.items() if k != "per_game"})
            if self.cfg["lvl_measure"]:
                measured, counts = self.measured_budgets(self.game_rows(rows0, g))
                self.feed.emit("lvl_budgets", configured=self.budgets, measured=measured, counts=counts)
                self.budgets = measured
            batches = [(folder0, self.game_rows(rows0, g))]
        if self.serving_adapter is None:  # a fresh adapter (not resumed): round zero's r00
            self.serving_adapter = self.adapters / "r00"
        pending, stop, played_by = [], None, start - 1  # in-flight batches: (round, adapter, name, task)
        lp85_task = None
        for number in range(start, self.cfg["rounds"] + 1):
            since = time.time()
            if len(pending) < int(self.cfg["lvl_batches"]):  # a new batch on this round's adapter, pinned by name
                busy = {b[2] for b in pending} | ({lp85_task[3]} if lp85_task else set())
                name = next(n for n in ("game-a", "game-b", "game-c") if n not in busy)
                await self.server.load(name, self.serving_adapter)
                pending.append((number, played_by, name, asyncio.create_task(
                    self.bench(f"games-{number:02d}", [g], int(self.cfg["lvl_games"]), {g: budget}, model=name,
                               concurrency=int(self.cfg["lvl_games"]), reserve=120, wall=wall))))
            starts = self.lvl_starts(batches, seed=number)
            path, points = self.restore_points(starts, f"tries-{number:02d}")
            seg = self.bench(f"tries-{number:02d}", [g], tries, {g: int(max(self.budgets.values()))}, points=path,
                             concurrency=min(len(points) * tries, int(self.cfg["lvl_seg_concurrency"])))
            every = int(self.cfg["lvl_lp85_every"] or 0)
            if every and (number - start + 1) % every == 0 and lp85_task is None:  # report only: in the background
                _, by, pinned, _ = pending[-1]  # the newest games batch's adapter, pinned by its name
                lp85_task = (number, by, asyncio.create_task(self.bench(
                    f"lp85-{number:02d}", [e], int(self.cfg["lvl_lp85_games"]), {e: self.cfg["eval_budget"]},
                    seed_base=int(self.cfg["eval_seed_base"]), reserve=120, model=pinned)), pinned)
            update = self.lvl_update(number, kept, probes) if kept else None
            jobs = [seg] + ([update] if update else [])
            results = await asyncio.gather(*jobs)
            seg_folder, seg_rows, seg_play = results[0]
            update_row, serve = results[1] if update else (None, None)
            lp85 = None
            if lp85_task is not None and lp85_task[2].done():
                lp85, lp85_round, lp85_by = lp85_task[2].result(), lp85_task[0], lp85_task[1]
                lp85_task = None
            row = {"round": number, "played_by": played_by, "budgets": {lv: int(b) for lv, b in self.budgets.items()}}
            if update_row is not None:
                row["update"] = {k: v for k, v in update_row.items() if k not in ("traceback", "windows")}
                if not update_row.get("ok"):
                    stop = row["stop"] = update_row.get("stop") or "update_failed"
                    self.feed.emit("round", **row)
                    break
                row["swap"] = {"load_seconds": await self.server.load("policy", serve)}
                self.serving_adapter = serve
                played_by = number
            # the tries: restores (failed starts dropped), passes, rank weights, next budgets
            kept, part, (failed, counted) = self.lvl_take_tries(seg_folder, seg_rows, number)
            row["tries"] = {"play": seg_play, **part}
            # the usable share leaves out dropped starts (their restore failures are logged, not infra)
            bad_tries = sum(d["of"] for d in part["dropped_starts"])
            usable_share = len(seg_rows) / max(1, part["of"] - bad_tries) if part["of"] > bad_tries else 1.0
            if lp85 is not None:
                from .rl_loop import attempt_reward, mean_se

                values = self.game_rows(lp85[1], e)
                row["lp85"] = dict(zip(("mean", "se"), mean_se([attempt_reward(r) for r in values])), n=len(values),
                                   started=lp85_round, played_by=lp85_by)
            # finished whole games: the metric and the next practice states
            for batch in [b for b in pending if b[3].done()]:
                number0, adapter0, name, task = batch
                folder, rows, play = task.result()
                metrics = game_metrics(self.game_rows(rows, g))
                self.feed.emit("lvl_games", round=number0, played_by=adapter0, play=play,
                               **{k: v for k, v in metrics.items() if k != "per_game"})
                row.setdefault("games", []).append({"round": number0, "played_by": adapter0, "won": metrics["won"],
                                                    "median_to_win": metrics["median_to_win"],
                                                    "levels_mean": metrics["levels_mean"]})
                batches.append((folder, self.game_rows(rows, g)))
                pending.remove(batch)
            self.feed.emit("round", **row)
            self.history.append(row)
            if failed > float(self.cfg["lvl_restore_fail"]) * max(1, counted):
                stop = "restore"  # scattered restore failures (not one bad start): stop
                break
            if usable_share < float(self.cfg["lvl_usable"]):
                stop = "infra"
                break
            hours = self.cfg["limits"].get("loop_hours")
            if hours and time.time() - self.loop_started + (time.time() - since) > 3600 * float(hours):
                stop = "loop_hours"  # another round like this one would end past the loop's wall
                break
            if self.left() < 3600 * float(self.cfg.get("lvl_final_hours", 1.5)):
                stop = "wall"
                break
        if lp85_task is not None:  # a report still running at the end is dropped
            lp85_task[2].cancel()
            self.feed.emit("lp85_report", round=lp85_task[0], cancelled=True)
        for number0, adapter0, name, task in pending:  # the last batches: kept when they end before the final
            try:
                folder, rows, play = await asyncio.wait_for(task, timeout=max(60.0, self.left() - 3600 * float(
                    self.cfg.get("lvl_final_hours", 1.5))))
                metrics = game_metrics(self.game_rows(rows, g))
                self.feed.emit("lvl_games", round=number0, played_by=adapter0, play=play,
                               **{k: v for k, v in metrics.items() if k != "per_game"})
            except asyncio.TimeoutError:
                self.feed.emit("lvl_games", round=number0, played_by=adapter0, cancelled=True)
        return stop

    async def lvl_finish(self, stop):
        """Final: ``lvl_final_games`` whole games each for the adapter and the base on the same seeds."""
        if stop in ("nan_or_inf", "update_failed"):
            self.feed.emit("final", stop=stop, eval=None)
            return stop
        g, n, seed = self.cfg["train_game"], int(self.cfg["lvl_final_games"]), int(self.cfg["lvl_final_seed"])
        budget = {g: int(self.cfg["lvl_game_budget"])}
        names = {"policy": "policy", "base": self.server.model}
        results = await asyncio.gather(*(self.bench(f"final-{tag}", [g], n, budget, seed_base=seed, model=model,
                                                    concurrency=n, reserve=120, wall=int(self.cfg["lvl_game_wall"]))
                                         for tag, model in names.items()))
        out = {"seed_base": seed, "games": n}
        per = {}
        for (tag, _), (_, rows, play) in zip(names.items(), results):
            metrics = game_metrics(self.game_rows(rows, g))
            out[tag] = {k: v for k, v in metrics.items() if k != "per_game"}
            out[f"{tag}_play"] = play
            per[tag] = {m["label"].split("-p")[-1]: m for m in metrics["per_game"]}
        both = sorted(set(per["policy"]) & set(per["base"]))
        pairs = [(per["policy"][k], per["base"][k]) for k in both]
        wins = [(a["tokens_to_win"], b["tokens_to_win"]) for a, b in pairs if a["won"] and b["won"]]
        out["paired"] = {"n": len(pairs), "levels_diff": round(statistics.mean(a["levels"] - b["levels"] for a, b in pairs), 3) if pairs else None,
                         "score_diff": round(statistics.mean(a["score"] - b["score"] for a, b in pairs), 4) if pairs else None,
                         "both_won": len(wins),
                         "token_ratio_median": round(statistics.median(b / a for a, b in wins), 3) if wins else None}
        self.feed.emit("eval", label="final", **out)
        self.feed.emit("final", stop=stop, eval=out, base=self.lvl_base and {k: v for k, v in self.lvl_base.items()
                                                                               if k != "per_game"})
        return stop
