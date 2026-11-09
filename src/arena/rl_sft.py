"""Cheapest-clear SFT on the rl1 stack (SFT arm T2/T3).

The corpus is recorded v8.2 play: per segment (one level clear of one game) an attempt folder whose ``rollout.jsonl``
holds every request from the game's start to the clear, rebuilt token-exact from the run's logs; the requests before
the level start are ``scripted`` (no-gradient context) and the level's own replies are trained (whole-segment
windows, plain log-likelihood at temperature 1: the rows carry no engine sampling sets).

Mode ``sft`` (``study["rl_loop"]["sft"]``):
- ``t2``: the sizing check (``fit`` at the run's window), learning-rate probes (per rate: the adapter reset, the base
  scored on the held-out windows, ``probe_steps`` updates on one fixed batch, the held-out KL to the base after each),
  then the served positive control: an adapter trained on a few level-1 clears, served, and fresh level-1 tries of
  those games, adapter and base on the same seeds.
- ``t3``: one pass over the corpus in batches (steps), held-out probes every ``probe_every`` steps, stop on NaN or
  the held-out KL hard stop; the adapter saved and served; the positive control; checkpoints after every step.
"""

import asyncio
import json
import random
import math
import statistics
import time
from pathlib import Path

from .rl_single import clear_tokens

SFT_DEFAULTS = {
    "phase": "t2", "fit": True,
    "corpus": "sft/train", "probe_corpus": "sft/probe",
    "window": 32768, "probe_window": 16384, "probe_count": 4, "prefix_chunk": 32768,
    "rates": [5e-5, 1.5e-4, 4e-4], "probe_steps": 2, "probe_quantiles": [0.75], "probe_seed": 0,
    "kl_stable": 0.02, "kl_stop": 0.05,
    # positive control: segments (folder names in the corpus) trained pc_steps times at the picked rate (x pc_lr_scale),
    # then fresh level-1 tries per game, adapter and base on the same seeds
    "pc": {"segments": [], "steps": 6, "lr_scale": 1.0, "games": {}, "tries": 6, "seed": 9300, "bar": 1.5},
    # t3
    "lr": None, "batch_windows": 8, "probe_every": 4, "max_steps": None, "passes": 1,
    # t3 schedule: linear warm-up over ``warmup`` of the run, then cosine to ``lr_floor`` x peak; the run's progress is
    # the larger of steps done / planned and training seconds used / the training budget (a time stop still decays)
    "warmup": 0.05, "lr_floor": 0.1,
    # audit 2 (17 Oct): "time" = the cosine above on max(steps, time) (gate v3: floor reached with 19 of 95 steps left);
    # "wsd" = step-driven warm-up, constant to ``stable`` of the steps, cosine to lr_floor; the time stop only saves.
    # save_passes: an adapter per pass (sft-adapter-p<n>); fit_probes: trained segments scored with the probes (their
    # NLL is the completion read, S8); the probe KL floor is two base scorings of the same weights
    "schedule": "time", "stable": 0.8, "save_passes": False, "fit_probes": 0,
    # resume {"adapter", "from_step"}: a saved adapter loaded after the base scoring, steps before from_step skipped
    # (same shuffles), a fresh AdamW re-warmed over ``rewarm`` steps (gate v4b, 17 Oct)
    "resume": None, "rewarm": 5,
    # utilization: the running adapter saved every ``save_every`` steps (5.2 GiB, ~9 s); probe_base False =
    # no base scoring and no probes (a speed check); fit_gate: trainer ``fit`` kwargs run before gate training (the
    # checkpointing mode that fills the card)
    "save_every": 4, "probe_base": True, "fit_gate": None,
    # val_corpus (replan 17 Oct): segments of held-out starts of the same game, scored with the probes ("val-"); the
    # best validation NLL's adapter is kept (sft-adapter-best) and training stops after ``val_patience`` probe points
    # without a new best
    "val_corpus": None, "val_patience": 2,
    "checkpointing": None,  # the trainer's mode for training (None: full); partial:N keeps N layers' activations
    "reserve_seconds": 1800,  # T3 stops training this long before the deadline (the positive control's play)
    # gate: one game's lower levels trained (the corpus), then adapter vs base on the same seeds
    # from restored starts of a base recording made first: the trained ``control`` level and held-out levels;
    # ``levels``: level -> try budget (3x the level median); ``starts`` x ``tries`` per level and arm
    "gate": {"game": "re86", "record": {"games": 12, "budget": 100000, "seed": 9400}, "levels": {},
             "control": None, "starts": 5, "tries": 4, "seed": 9500, "max_prompt": 110000, "pick_seed": 0,
             "bar_control": 1.5, "bar_heldout": 1.3, "drift": None, "mtp_from": None, "mtp_attempts": 4,
             "play_reserve": 2700,
             # restores replay the recorded actions (re-executed tool code chose others: replay_mismatch, gate v3);
             # reuse {"recording", "adapter"}: eval only; tries_per_start: overrides starts x tries / picked
             "pin_restore": False, "reuse": None, "tries_per_start": None, "train_only": False,
             # concurrency: streams per level job (None: all its tries at once); reuse.adapters {} = base only
             # (on-policy collection, replan 17 Oct)
             "concurrency": None, "record_only": False, "skip_base": False, "start_offset": 0},
}


def scheduled_lr(peak, progress, warmup, floor):
    """Linear warm-up to ``peak`` over the first ``warmup`` of the run, then cosine down to ``floor`` x peak."""
    if warmup > 0 and progress < warmup:
        return peak * max(progress / warmup, 0.02)
    tail = (progress - warmup) / max(1e-9, 1.0 - warmup)
    return peak * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, tail))))


def wsd_lr(peak, progress, warmup, stable, floor):
    """Step-driven: linear warm-up over ``warmup``, constant to ``stable``, then cosine down to ``floor`` x peak."""
    if warmup > 0 and progress < warmup:
        return peak * max(progress / warmup, 0.02)
    if progress < stable:
        return peak
    tail = (progress - stable) / max(1e-9, 1.0 - stable)
    return peak * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, tail))))


def merged(raw):
    out = json.loads(json.dumps(SFT_DEFAULTS))
    for key, value in (raw or {}).items():
        if key not in SFT_DEFAULTS:
            raise ValueError(f"Unknown sft setting {key!r}")
        out[key] = {**out[key], **value} if isinstance(SFT_DEFAULTS[key], dict) else value
    return out


class EarlyStop:
    """Validation-NLL early stop: ``see`` returns "best" (a new minimum), "wait", or "stop" after ``patience``
    points in a row without a new minimum."""

    def __init__(self, patience):
        self.patience, self.best, self.best_step, self.bad = int(patience), None, None, 0

    def see(self, step, value):
        if self.best is None or value < self.best:
            self.best, self.best_step, self.bad = float(value), int(step), 0
            return "best"
        self.bad += 1
        return "stop" if self.bad >= self.patience else "wait"


def censored_median(values, cap):
    """Median tokens to clear with unfinished tries counted at the cap (a lower bound of their cost)."""
    filled = sorted(cap if v is None else v for v in values)
    return statistics.median(filled) if filled else None


class SftMixin:
    def sft_cfg(self):
        return merged(self.cfg.get("sft"))

    def sft_specs(self, folder_names=None, *, root=None, window=None):
        """Whole-segment specs of the corpus (``folder_names``: a subset), weight = the segment's ``weight`` (an
        all-segments corpus) or its draws."""
        cfg = self.sft_cfg()
        root = Path(root or (self.source / cfg["corpus"]))
        specs = []
        for meta_path in sorted(root.glob("*/meta.json")):
            meta = json.loads(meta_path.read_text())
            if folder_names is not None and meta["name"] not in folder_names:
                continue
            specs.append({"rollout": str(meta_path.parent / "rollout.jsonl"),
                          "w": float(meta.get("weight") or meta.get("draws_used") or 1),
                          "whole": True, "window": int(window or cfg["window"]), "floor": 1024, "margin": 1024,
                          "credited_only": True, "share": meta["name"], "name": meta["name"]})
        return specs

    def sft_probes(self):
        cfg = self.sft_cfg()
        root = self.source / cfg["probe_corpus"]
        out = []
        for meta_path in sorted(root.glob("*/meta.json"))[: int(cfg["probe_count"])]:
            meta = json.loads(meta_path.read_text())
            out.append({"rollout": str(meta_path.parent / "rollout.jsonl"), "w": 1.0,
                        "window": int(cfg["probe_window"]), "credited_only": True, "name": "heldout-" + meta["name"]})
        if cfg["val_corpus"]:
            for meta_path in sorted((self.source / cfg["val_corpus"]).glob("*/meta.json")):
                meta = json.loads(meta_path.read_text())
                out.append({"rollout": str(meta_path.parent / "rollout.jsonl"), "w": 1.0,
                            "window": int(cfg["probe_window"]), "credited_only": True, "name": "val-" + meta["name"]})
        train = sorted((self.source / cfg["corpus"]).glob("*/meta.json"))
        for meta_path in random.Random(7).sample(train, min(int(cfg["fit_probes"]), len(train))):
            meta = json.loads(meta_path.read_text())
            out.append({"rollout": str(meta_path.parent / "rollout.jsonl"), "w": 1.0,
                        "window": int(cfg["probe_window"]), "credited_only": True, "name": "fit-" + meta["name"]})
        return out

    @staticmethod
    def probe_kl(result):
        rows = [r for k, r in (result or {}).items() if isinstance(r, dict) and "kl_base" in r
                and not str(k).startswith(("fit-", "val-"))]
        return round(sum(r["kl_base"] for r in rows) / len(rows), 6) if rows else None

    @staticmethod
    def fit_nll(result, prefix="fit-"):
        """Mean NLL of the trained probe segments (token-weighted): falls as the corpus is learned. ``prefix`` "val-":
        the held-out starts' segments (the early-stop read)."""
        rows = [r for k, r in (result or {}).items() if str(k).startswith(prefix) and isinstance(r, dict) and r.get("tokens")]
        n = sum(r["tokens"] for r in rows)
        return round(-sum(r["actor_mean"] * r["tokens"] for r in rows) / n, 5) if n else None

    async def sft_update(self, specs, lr, probes=None, label=""):
        cfg = self.sft_cfg()
        result = await self.trainer.call("update", timeout=int(self.cfg["update_timeout"]), attempts=specs, lr=lr,
                                         max_windows=100000, share_prefix=True, prefix_chunk=int(cfg["prefix_chunk"]),
                                         clip=float(self.cfg["clip"]), probes=probes)
        windows = result.get("windows") or []
        row = {"label": label, "lr": lr, "ok": result.get("ok"), "loss": result.get("loss"),
               "grad_norm": result.get("grad_norm"), "trained_tokens": result.get("trained_tokens"),
               "windows": len(windows), "errors": len(result.get("errors") or []),
               "train_seconds": result.get("train_seconds"), "gpu_util_mean": result.get("gpu_util_mean"),
               "card_peak_gib": result.get("card_peak_gib"), "torch_peak_gib": result.get("torch_peak_gib"),
               "window_seconds": [w.get("seconds") for w in windows], "window_tokens": [w.get("tokens") for w in windows],
               "nll": [w.get("sampled_nll_trainer") for w in windows], "stop": result.get("stop"),
               "heldout_kl": self.probe_kl(result.get("probes")), "fit_nll": self.fit_nll(result.get("probes")),
               "val_nll": self.fit_nll(result.get("probes"), "val-"), "autotune_pinned": result.get("autotune_pinned"), "autotune_late": result.get("autotune_pinned_late"),
               "error": result.get("error")}
        self.feed.emit("sft_update", **row)
        return row, result

    async def sft_fit(self):
        self.gpu.mark("fit")
        began = time.time()
        fit = await self.trainer.call("fit", timeout=3600, save=str(self.adapters / "perturbed"),
                                      **{k: v for k, v in self.cfg["fit"].items() if v is not None})
        fit_row = {k: v for k, v in fit.items() if k != "traceback"}
        self.feed.emit("fit", wall_seconds=round(time.time() - began, 1), **fit_row)
        return fit_row

    async def sft_probe_rates(self):
        cfg = self.sft_cfg()
        # one fixed batch of segments at the given quantiles of the last prompt, so the probe's steps cost about
        # what the pass's steps will
        specs = sorted(self.sft_specs(), key=lambda s: json.loads((Path(s["rollout"]).parent / "meta.json")
                                                                  .read_text())["last_prompt"])
        batch = [specs[i] for i in sorted({int(float(q) * (len(specs) - 1)) for q in cfg["probe_quantiles"]})]
        probes = self.sft_probes()
        out = {"batch": [s["name"] for s in batch], "rates": {}}
        for lr in cfg["rates"]:
            await self.trainer.must("reset", lr=lr)
            base = await self.trainer.call("score", windows=probes, label="base", timeout=1800)
            steps = []
            for step in range(int(cfg["probe_steps"])):
                row, _ = await self.sft_update(batch, lr, probes=probes, label=f"probe-{lr:g}-{step + 1}")
                steps.append(row)
                if not row["ok"]:
                    break
            losses = [s["loss"] for s in steps if s.get("loss") is not None]
            kl = steps[-1].get("heldout_kl") if steps else None
            stable = (len(steps) == int(cfg["probe_steps"]) and all(s["ok"] for s in steps) and len(losses) >= 2
                      and losses[-1] < losses[0] and kl is not None and kl <= float(cfg["kl_stable"]))
            out["rates"][f"{lr:g}"] = {"steps": steps, "stable": stable, "loss_drop": round(losses[0] - losses[-1], 5)
                                       if len(losses) >= 2 else None, "heldout_kl": kl,
                                       "base_probe": {k: v.get("actor_mean") for k, v in (base.get("windows") or {}).items()}}
        stable = [float(k) for k, v in out["rates"].items() if v["stable"]]
        out["pick"] = max(stable) if stable else None
        self.feed.emit("sft_probes", **out)
        return out

    async def sft_positive_control(self, lr, adapter=None):
        """Fresh level-1 tries per game, adapter vs base on the same seeds. ``adapter``: a saved adapter folder (T3's);
        else one trained here on the control's level-1 clears (``pc.segments``, ``pc.steps`` updates at ``lr``)."""
        cfg = self.sft_cfg()
        pc = cfg["pc"]
        steps = []
        if adapter is None:
            specs = self.sft_specs(set(pc["segments"]))
            await self.trainer.must("reset", lr=lr)
            for step in range(int(pc["steps"])):
                row, _ = await self.sft_update(specs, lr, label=f"pc-{step + 1}")
                steps.append(row)
                if not row["ok"]:
                    break
            path = self.adapters / "pc"
            saved = await self.trainer.must("save", path=str(path))
        else:
            path, saved, specs = Path(adapter), {"path": str(adapter)}, []
        swap = await self.swap_in(path)
        self.feed.emit("sft_pc_adapter", saved=saved, swap=swap, segments=[s["name"] for s in specs])
        names = {"policy": "policy", "base": self.server.model}
        jobs, keys = [], []
        for game, cap in sorted(pc["games"].items()):
            spec = {"name": f"pc-{game}", "game": game, "level": 1, "fresh": True, "cap": int(cap)}
            for tag, model in names.items():
                keys.append((game, tag, int(cap)))
                jobs.append(self.mg_play(f"pc-{game}-{tag}", spec, [], int(pc["tries"]), model=model,
                                         seed=int(pc["seed"])))
        results = await asyncio.gather(*jobs)
        out = {"steps": steps, "games": {}}
        from .rl_loop import reply_stats

        for (game, tag, cap), (folder, rows, play) in zip(keys, results):
            toks = [clear_tokens(r) for r in rows]
            replies = [reply_stats(Path(folder) / "attempts" / r["label"] / "rollout.jsonl",
                                   int(self.study["completion_cap"])) for r in rows]
            total = sum(x.get("replies") or 0 for x in replies)
            capped = sum((x.get("capped_share") or 0) * (x.get("replies") or 0) for x in replies)
            out["games"].setdefault(game, {})[tag] = {
                "tries": len(rows), "clears": sum(t is not None for t in toks), "tokens": toks,
                "median_censored": censored_median(toks, cap), "cap": cap, "play": play,
                "replies": total, "capped_share": round(capped / total, 4) if total else None}
        ratios = []
        for game, arms in out["games"].items():
            if "policy" in arms and "base" in arms and arms["policy"]["median_censored"]:
                arms["ratio"] = round(arms["base"]["median_censored"] / arms["policy"]["median_censored"], 3)
                ratios.append(arms["ratio"])
        pooled = {tag: censored_median([t for arms in out["games"].values() for t in arms.get(tag, {}).get("tokens", [])],
                                       max(int(c) for c in pc["games"].values())) for tag in names}
        out["ratio_min"] = min(ratios) if ratios else None
        out["pass"] = bool(ratios) and min(ratios) >= float(pc["bar"])
        out["pooled"] = pooled
        self.feed.emit("sft_pc", **{k: v for k, v in out.items() if k != "steps"})
        return out

    def gate_starts(self, folder, level, count):
        """Restored starts of ``level`` in a recorded base bench (usable games, before any compaction or python
        time-out, prompt <= max_prompt), shuffled by pick_seed, the first ``count``; and how many there were."""
        from .restore import level_starts, log_rows
        from .rl_loop import usable

        g = self.sft_cfg()["gate"]
        root, pool = Path(folder), []
        rows = [r for r in json.loads((root / "level-bench.json").read_text())["attempts"] if usable(r)]
        for row in sorted(self.game_rows(rows, g["game"]), key=lambda r: r["label"]):
            attempt = root / "attempts" / row["label"]
            stops = [int(e.get("step") or 0) for e in log_rows(attempt / "log.jsonl",
                                                               ("compaction", "playbook_handoff", "duck_python_interrupt",
                                                                "context_drain"))]
            first = min(stops) if stops else None
            with open(attempt / "rollout.jsonl") as file:
                prompts = [int((json.loads(line).get("usage") or {}).get("prompt_tokens") or 0) for line in file]
            for request, done, step in level_starts(attempt):
                if (done + 1 == int(level) and (first is None or step < first)
                        and prompts[int(request)] <= int(g["max_prompt"])):
                    pool.append((str(attempt), int(request)))
        random.Random(int(g["pick_seed"]) * 100 + int(level) * 7).shuffle(pool)
        offset = int(g.get("start_offset") or 0)  # round 2: one level's starts split over jobs (disjoint slices)
        return pool[offset: offset + int(count)], len(pool)

    async def sft_gate(self, lr):
        """Record base games, train the corpus (T3 schedule, probes), the untrained-text drift, then per level adapter
        and base from the same restored starts and seeds (both arms at once); reads: censored medians and ratios,
        clears, capped-reply share, and the untrained MTP head's draft acceptance on each arm's own tries (the rl1
        server drafts nothing)."""
        cfg = self.sft_cfg()
        g = cfg["gate"]
        game, base_model = g["game"], self.server.model
        out = {"game": game}
        rec, reuse = g["record"], g.get("reuse")
        if reuse:  # eval only: a saved recording and adapter, no training (the held-out re-read)
            adapters = reuse["adapters"] if reuse.get("adapters") is not None else {"policy": reuse["adapter"]}
            return await self.gate_play(g, out, Path(reuse["recording"]), adapters, base_model, None)
        if g.get("train_only"):  # gate v4: train and save (adapters per pass), the reads run as eval-only jobs
            await self.server.sleep()
            out["trainer"] = await self.start_trainer()
            self.feed.emit("trainer", **out["trainer"])
            if cfg["checkpointing"]:
                out["checkpointing"] = await self.trainer.must("checkpointing", mode=cfg["checkpointing"])
                self.feed.emit("checkpointing", **out["checkpointing"])
            if cfg["fit_gate"]:
                self.gpu.mark("fit")
                began = time.time()
                fit = await self.trainer.call("fit", timeout=3600, **cfg["fit_gate"])
                out["fit"] = {k: v for k, v in fit.items() if k != "traceback"}
                self.feed.emit("fit", wall_seconds=round(time.time() - began, 1), **out["fit"])
            out["train"] = await self.sft_t3(lr, reserve=float(g["play_reserve"]), control=False)
            if g.get("drift"):
                try:
                    out["drift"] = await self.trainer.call("text_drift", timeout=1800, **g["drift"])
                except Exception as exc:  # noqa: BLE001
                    out["drift"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                self.feed.emit("gate_drift", **out["drift"])
            self.feed.emit("gate", **{k: v for k, v in out.items() if k != "train"})
            return out
        recording = asyncio.ensure_future(self.bench(f"gate-rec-{game}", [game], int(rec["games"]),
                                                     {game: int(rec["budget"])}, model=base_model,
                                                     seed_base=int(rec["seed"]), concurrency=int(rec["games"]),
                                                     reserve=120))
        dual = bool(self.cfg.get("dual"))
        if dual:  # the trainer has its own card: it loads and trains while the server records
            out["trainer"] = await self.start_trainer()
            self.feed.emit("trainer", **out["trainer"])
            training = asyncio.ensure_future(self.sft_t3(lr, reserve=float(g["play_reserve"]), control=False))
        folder, rows, play = await recording
        out["record"] = {"folder": str(folder), "play": play, "games": len(rows),
                         "levels": sorted(len(r.get("level_rows") or []) for r in rows)}
        if g.get("record_only"):  # a fresh base recording (the replan's unseen eval starts), nothing else
            self.feed.emit("gate", **out)
            return out
        if dual:
            out["train"] = await training
        else:
            await self.server.sleep()
            out["trainer"] = await self.start_trainer()
            self.feed.emit("trainer", **out["trainer"])
            out["train"] = await self.sft_t3(lr, reserve=float(g["play_reserve"]), control=False)
        adapter = self.output / "rl" / "sft-adapter"
        if g.get("drift"):
            try:
                out["drift"] = await self.trainer.call("text_drift", timeout=1800, **g["drift"])
            except Exception as exc:  # noqa: BLE001  a failed probe must not cost the play read
                out["drift"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            self.feed.emit("gate_drift", **out["drift"])
        return await self.gate_play(g, out, Path(folder), {"policy": adapter}, base_model, await self.swap_in(adapter))

    async def gate_play(self, g, out, folder, adapters, base_model, swap):
        """Adapters ({arm: path}) and base from the same restored starts of the recording in ``folder`` (``swap``:
        None = load the adapters beside the awake server); per arm the paired read against base, and pooled reads.
        A restore counts when its game state is exact (pinned actions; a tool message's text may differ)."""
        from .rl_loop import reply_stats

        game = g["game"]
        starts = {level: self.gate_starts(folder, level, int(g["starts"])) for level in sorted(g["levels"], key=int)}
        out.setdefault("record", {"folder": str(folder)})["starts"] = {
            lv: {"used": len(s), "available": n} for lv, (s, n) in starts.items()}
        self.feed.emit("gate_record", **out["record"])
        if swap is None:
            swap = {name: {"load_seconds": await self.server.load(name, path), "adapter": str(path)}
                    for name, path in adapters.items()}
        arms = {**({} if g.get("skip_base") else {"base": base_model}), **{name: name for name in adapters}}  # skip_base: an adapter-only eval job (its base pairs come from a sibling job on the same starts and seeds)
        out["levels"] = {}
        jobs, keys = [], []
        for tag, model in arms.items():
            for level, (picked, _) in starts.items():
                if not picked:
                    continue
                tries = int(g.get("tries_per_start") or max(1, -(-int(g["starts"]) * int(g["tries"]) // len(picked))))
                spec = {"name": f"gate-L{level}", "game": game, "level": int(level), "cap": int(g["levels"][level]),
                        "pin_actions": bool(g.get("pin_restore"))}
                keys.append((tag, level))
                jobs.append(self.mg_play(f"gate-L{level}-{tag}", spec, picked, tries, model=model, seed=int((g.get("seeds") or {}).get(str(level), g["seed"])),
                                         concurrency=int(g["concurrency"]) if g.get("concurrency") else None))
        results = await asyncio.gather(*jobs)
        for (tag, level), (lfolder, lrows, lplay) in zip(keys, results):
            cap = int(g["levels"][level])
            report = json.loads((Path(lfolder) / "level-bench.json").read_text())["attempts"]
            tries = [r for r in report if r.get("outcome") not in ("error", "replay_mismatch")
                     and (not r.get("restore") or r["restore"].get("ok") or (r["restore"].get("state") or {}).get("ok"))]
            toks = [clear_tokens(r) for r in tries]  # None: not cleared (budget, wall-cut, ...): censored at the cap
            replies = [reply_stats(Path(lfolder) / "attempts" / r["label"] / "rollout.jsonl",
                                   int(self.study["completion_cap"])) for r in tries]
            total = sum(x.get("replies") or 0 for x in replies)
            capped = sum((x.get("capped_share") or 0) * (x.get("replies") or 0) for x in replies)
            out["levels"].setdefault(level, {})[tag] = {
                "tries": len(tries), "clears": sum(t is not None for t in toks), "tokens": toks, "cap": cap,
                "wall_cut": sum(1 for r in tries if r.get("outcome") == "censored"),
                "errors": len(report) - len(tries), "median_censored": censored_median(toks, cap), "play": lplay,
                "folder": str(lfolder), "replies": total, "capped_share": round(capped / total, 4) if total else None,
                "pairs": {f"{'-'.join(str(r.get('point')).split('-')[-2:])}#{r.get('pass')}": t for r, t in zip(tries, toks)},
                "cut": sorted(f"{'-'.join(str(r.get('point')).split('-')[-2:])}#{r.get('pass')}" for r in tries
                              if r.get("outcome") == "censored")}
        names = [t for t in arms if t != "base"]
        held = [lv for lv in out["levels"] if lv != str(g["control"])]
        start_means = {}
        for level, arms_ in out["levels"].items():  # paired by start and seed: log(base / adapter), unfinished at the cap
            if "base" not in arms_:
                continue
            arms_["paired"] = {}
            for name in names:
                if name not in arms_:
                    continue
                cap = arms_["base"]["cap"]
                # a wall-cut try (not a token cap) says nothing about tokens: its pair is dropped (the slower arm
                # would otherwise lose the pair at the cap)
                common = sorted((set(arms_["base"]["pairs"]) & set(arms_[name]["pairs"]))
                                - set(arms_["base"]["cut"]) - set(arms_[name]["cut"]))
                diffs = [math.log(min(arms_["base"]["pairs"][k] or cap, cap)) - math.log(min(arms_[name]["pairs"][k] or cap, cap))
                         for k in common]
                by_start = {}
                for k, d in zip(common, diffs):
                    by_start.setdefault(k.split("#")[0], []).append(d)
                means = [sum(v) / len(v) for v in by_start.values()]
                se = (statistics.stdev(means) / math.sqrt(len(means))) if len(means) > 1 else None
                if level in held:
                    start_means.setdefault(name, []).extend(means)
                arms_["paired"][name] = {"pairs": len(common), "starts": len(means),
                                         "ratio": round(math.exp(sum(diffs) / len(diffs)), 3) if diffs else None,
                                         "se_log": round(se, 4) if se is not None else None,
                                         "ci95": [round(math.exp(sum(means) / len(means) - 1.96 * se), 3),
                                                  round(math.exp(sum(means) / len(means) + 1.96 * se), 3)] if se else None}

        def pooled(tag):
            values = [t / out["levels"][lv][tag]["cap"] if t is not None else None
                      for lv in held for t in out["levels"][lv].get(tag, {}).get("tokens", [])]
            return censored_median(values, 1.0)

        out["heldout_paired"] = {}
        for name, means in start_means.items():  # held-out levels pooled, one mean per (level, start): 95% CI
            if len(means) > 1:
                mu, se = sum(means) / len(means), statistics.stdev(means) / math.sqrt(len(means))
                out["heldout_paired"][name] = {"starts": len(means), "ratio": round(math.exp(mu), 3), "se_log": round(se, 4),
                                               "ci95": [round(math.exp(mu - 1.96 * se), 3), round(math.exp(mu + 1.96 * se), 3)]}
        out["heldout_pooled"] = {tag: pooled(tag) for tag in arms}
        hp, first = out["heldout_pooled"], (names[-1] if names else None)
        out["heldout_ratio"] = round(hp["base"] / hp[first], 3) if first and hp.get(first) and hp.get("base") else None
        control = out["levels"].get(str(g["control"]), {})
        out["control_ratio"] = ((control.get("paired") or {}).get(first) or {}).get("ratio") if first else None
        out["pass_control"] = out["control_ratio"] is not None and out["control_ratio"] >= float(g["bar_control"])
        out["pass_heldout"] = out["heldout_ratio"] is not None and out["heldout_ratio"] >= float(g["bar_heldout"])
        if g.get("mtp_from") and self.left() > 900:  # the untrained MTP head's draft acceptance on each arm's own tries
            await self.server.sleep()  # the trainer's long forwards need the card
            out["mtp"] = {}
            for tag, on in (("base", "off"), (names[-1], "on")):
                segs = []
                for level in held + [str(g["control"])]:
                    arm = out["levels"].get(level, {}).get(tag)
                    if arm:
                        segs += [{"rollout": str(p), "name": f"{tag}-L{level}-{p.parent.name}"}
                                 for p in sorted(Path(arm["folder"]).glob("attempts/*/rollout.jsonl"))[: int(g["mtp_attempts"])]]
                try:
                    res = await self.trainer.call("topk", timeout=1800, segments=segs, k=1, adapter=on, save=False,
                                                  out=str(self.output / "rl" / f"mtp-{tag}"), mtp_from=g["mtp_from"],
                                                  max_tokens=600000)
                    out["mtp"][tag] = res.get("mtp_check") or {"error": str(res.get("error"))[:300]}
                except Exception as exc:  # noqa: BLE001
                    out["mtp"][tag] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        out["swap"] = swap
        self.feed.emit("gate", **{k: v for k, v in out.items() if k not in ("levels", "train")},
                       levels={lv: {t: {k: v for k, v in a.items() if k not in ("tokens", "folder")} if isinstance(a, dict) else a
                                    for t, a in arms_.items()} for lv, arms_ in out["levels"].items()})
        return out

    async def sft_t3(self, lr, reserve=None, control=True):
        """``passes`` passes over the corpus in batches of ``batch_windows // 2`` whole segments (shuffled per pass)."""
        cfg = self.sft_cfg()
        reserve = float(cfg["reserve_seconds"] if reserve is None else reserve)
        per = max(1, int(cfg["batch_windows"]) // 2)
        batches, pass_ends = [], set()
        for number in range(int(cfg["passes"])):  # each pass reshuffled
            specs = self.sft_specs()
            random.Random(1 + number).shuffle(specs)
            batches += [specs[i:i + per] for i in range(0, len(specs), per)]
            pass_ends.add(len(batches))
        if cfg["max_steps"]:
            batches = batches[: int(cfg["max_steps"])]
        probes = self.sft_probes() if cfg["probe_base"] else None
        await self.trainer.must("reset", lr=lr)
        scores = {}
        if probes:
            base = await self.trainer.call("score", windows=probes, label="base", timeout=1800)
            floor = await self.trainer.call("score", windows=probes, timeout=1800)  # same weights: the KL floor
            scores = {"kl_floor": self.probe_kl(floor.get("windows")), "fit_nll_base": self.fit_nll(base.get("windows"))}
            self.feed.emit("sft_probe_base", **scores)
        rows, stop, passes = [], None, []
        resume, skip = cfg.get("resume"), 0
        if resume:
            skip = int(resume["from_step"])
            loaded = await self.trainer.must("load_adapter", path=str(resume["adapter"]), lr=lr)
            self.feed.emit("sft_resume", **loaded, from_step=skip)
            passes = [{"pass": k + 1, "step": e, "resumed": True} for k, e in enumerate(sorted(pass_ends)) if e <= skip]
        early = EarlyStop(int(cfg["val_patience"]))
        began, budget = time.time(), max(1.0, self.left() - reserve)
        for number, batch in enumerate(batches, 1):
            if number <= skip:
                continue
            if self.left() < reserve:
                stop = "time"
                break
            if cfg["schedule"] == "wsd":  # the decay starts at ``stable`` of the steps or of the time, whichever is first
                steps_p = (number - 1) / len(batches)
                time_p = skip / len(batches) + (1 - skip / len(batches)) * (time.time() - began) / budget
                if time_p > steps_p and time_p >= float(cfg["stable"]):
                    steps_p = time_p
                step_lr = wsd_lr(lr, min(1.0, steps_p), float(cfg["warmup"]), float(cfg["stable"]), float(cfg["lr_floor"]))
                if skip and number - skip <= int(cfg["rewarm"]):  # fresh Adam moments after a resume
                    step_lr *= (number - skip) / (int(cfg["rewarm"]) + 1)
            else:
                progress = min(1.0, max((number - 1) / len(batches), (time.time() - began) / budget))
                step_lr = scheduled_lr(lr, progress, float(cfg["warmup"]), float(cfg["lr_floor"]))
            probe = probes if probes and (number % int(cfg["probe_every"]) == 0 or number == len(batches)
                                          or number in pass_ends) else None
            row, _ = await self.sft_update(batch, step_lr, probes=probe, label=f"t3-{number}")
            rows.append(row)
            if not row["ok"]:
                stop = row.get("stop") or "update_failed"
                break
            if row.get("heldout_kl") is not None and row["heldout_kl"] > float(cfg["kl_stop"]):
                stop = "heldout_kl"
                break
            if row.get("val_nll") is not None:
                verdict = early.see(number, row["val_nll"])
                if verdict == "best":
                    await self.trainer.must("save", path=str(self.output / "rl" / "sft-adapter-best"))
                self.feed.emit("sft_val", step=number, val_nll=row["val_nll"], verdict=verdict, best=early.best,
                               best_step=early.best_step)
                if verdict == "stop":
                    stop = "val_early"
                    break
            if cfg["save_passes"] and number in pass_ends:
                path = self.output / "rl" / f"sft-adapter-p{len(passes) + 1}"
                await self.trainer.must("save", path=str(path))
                passes.append({"pass": len(passes) + 1, "step": number, "path": str(path), "fit_nll": row.get("fit_nll"),
                               "heldout_kl": row.get("heldout_kl")})
                self.feed.emit("sft_pass", **passes[-1])
            if number % int(cfg["save_every"]) == 0:
                await self.trainer.must("save", path=str(self.output / "rl" / "sft-adapter"))
        saved = await self.trainer.must("save", path=str(self.output / "rl" / "sft-adapter"))
        out = {"steps": len(rows), "of": len(batches), "stop": stop, "saved": saved,
               "trained_tokens": sum(int(r.get("trained_tokens") or 0) for r in rows),
               "heldout_kl": [r["heldout_kl"] for r in rows if r.get("heldout_kl") is not None],
               "fit_nll": [r["fit_nll"] for r in rows if r.get("fit_nll") is not None], "passes": passes, **scores,
               "val_nll": [r["val_nll"] for r in rows if r.get("val_nll") is not None],
               "best": {"step": early.best_step, "val_nll": early.best} if early.best is not None else None}
        self.feed.emit("sft_t3", **out)
        if control and stop in (None, "time") and rows:
            out["pc"] = await self.sft_positive_control(lr, adapter=self.output / "rl" / "sft-adapter")
        return out

    async def sft_run(self):
        cfg = self.sft_cfg()
        report = {}
        if cfg["phase"] == "gate":  # records base games on the awake server before the trainer loads
            report["gate"] = await self.sft_gate(float(cfg["lr"]))
            return report
        await self.server.sleep()
        report["trainer"] = await self.start_trainer()
        self.feed.emit("trainer", **report["trainer"])
        if cfg["fit"]:  # off: the trainer's default (full checkpointing), the sizing read from the updates
            report["fit"] = await self.sft_fit()
        if cfg["phase"] == "t2":
            if cfg["lr"] is not None:  # rate fixed from an earlier probe run: positive control only
                lr = float(cfg["lr"])
            else:
                report["probes"] = await self.sft_probe_rates()
                lr = report["probes"]["pick"]
            if lr is None:
                report["verdict"] = "no stable rate"
                return report
            report["pc"] = await self.sft_positive_control(lr * float(cfg["pc"]["lr_scale"]))
        else:
            lr = float(cfg["lr"])
            report["t3"] = await self.sft_t3(lr)
        return report
