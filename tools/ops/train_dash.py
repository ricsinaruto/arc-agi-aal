"""Live training dashboards: one view per RL / PTH training run in docs/html/training-live.html (run list switches views),
rebuilt every dashboard pass (5 min) from <ARC_OPS_SCRATCH>/train-metrics/<run>.jsonl, which each arm's exporter appends to.

Contract (one JSON object per line): {"t": epoch_s, "m": {key: number}, "alert": "text" (optional)}; a <run>.meta.json
beside it: {"run", "arm", "what", "status" (live|ended|stopped), "cards"}. Keys group by prefix into panels (PANELS);
a key no panel claims is not plotted. Grounding for the metric set: docs/arm-logs (dashboard entry, 21 Oct).
"""
import fnmatch, re, json, os, pathlib, time
import paths

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = pathlib.Path(os.environ.get("ARC_OPS_SCRATCH", ROOT / "runs/ops")) / "train-metrics"
TRAIN_RUN_PREFIXES = ("rl1-loop-", "sft-heal-round-", "sft-p75-qad-", "sft-layers-")  # RL loop, OPD/heal rounds, trunk/sft runs
BOOKKEEPING = {"infra.cost_usd", "infra.host_ram_gib", "infra.cpu_cores", "infra.disk_gib"}
TRAIN_ARMS = ("rl", "sft", "pth")  # a run whose meta names one of these arms is listed whatever its name (28 Oct: sft-layers-p75-v2 missing)
OUT = ROOT / "docs/html/training-live.html"
KEEP_ENDED_S = 36 * 3600
STALE_S = 30 * 60  # a "live" run with no live queue row and no data this long is shown as ended


class _AllRuns:
    def __contains__(self, run):
        return True

# (panel title, key patterns, y label, section). Order is the page order.
PANELS = [
    ("GPU utilization", ["infra.gpu_util.*"], "%", "Infrastructure",
     'Share of time each GPU is computing. Healthy: near 100% while a phase runs. Long dips mean the card waits on something (data, the other card, a handoff).'),
    ("GPU memory", ["infra.vram_gib.*", "infra.vram_cap_gib", "infra.peak_alloc_gib", "infra.alloc_after_load_gib", "infra.teacher_peak_gib.*"], "GiB", "Infrastructure",
     'Memory in use per card (peak = highest point during a step); dotted line = card capacity. Close to the line is good (batch fills the card); touching it risks an out-of-memory crash.'),
    ("Throughput", ["infra.gen_tok_s", "infra.train_tok_s", "train.live_tok_s", "infra.teacher_tok_s"], "tokens/s", "Infrastructure",
     'Tokens processed per second for play (gen), training and the teacher. Should hold flat; a drop with GPU use also down points to a stall.'),
    ("Serving load", ["infra.running", "infra.waiting"], "requests", "Infrastructure",
     'Game requests being answered (running) and queued (waiting). Running should sit near the stream cap; a growing waiting count means the server is the bottleneck.'),
    ("KV cache use", ["infra.kv_frac"], "%", "Infrastructure",
     "How full the server's attention cache is. Near 100% means streams are limited by memory; low means room for more streams."),
    ("Data age", ["train.age_median", "train.age_max"], "updates", "Training",
     "Streaming OPD: how many weight updates old the trained play is (median and oldest chunk of each step), counted from the weights that sampled it. Alert above 3.5 (median) or 6 (max): the student would learn from play it no longer produces."),
    ("Tokens per step", ["train.tokens", "train.live_tokens"], "tokens", "Training",
     "Training tokens in each optimizer step (the batch size). Should hold steady at the planned size; a drop means windows were filtered out or data ran short."),
    ("Step time", ["time.step_s", "time.train_s", "time.checkpoint_s"], "s", "Infrastructure",
     'Seconds per training step (and per checkpoint save). The dotted line is the run median. Flat near it is healthy; steps well above it are slowdowns (data loading, memory pressure, a busy neighbour).'),
    ("Round time split", ["time.*"], "minutes", "Infrastructure",
     "RL: minutes per round spent playing, updating, handing off between cards and idle, side by side. In two-card runs play (card 1) and the update (card 2) run at the same time, so a round lasts as long as the longer of the two; a resume round has no update. Idle and handoff are GPU time wasted."),
    ("Host", ["infra.host_ram_gib", "infra.cpu_cores", "infra.disk_gib"], "", "Infrastructure",
     'Machine RAM and CPU cores in use. Watch for RAM creeping up toward the limit (a leak) before a crash.'),
    ("Cost so far", ["infra.cost_usd"], "$", "Infrastructure",
     "Money spent by the run since launch (GPU time at list price). Should rise in a straight line; compare with the run's budget."),
    ("Progress", ["train.step", "train.update", "train.updates_planned"], "", "Training",
     'Training steps completed (RL: 4 per update), updates completed and updates planned. Should climb steadily; a flat stretch means training stalled.'),
    ("Groups dropped", ["train.groups_dropped_frac", "train.groups_dropped_frac_step"], "%", "Training",
     "Share of played groups the trainer dropped (too stale for the lag limit, filtered, other), cumulative over the leg and per step. Above 10% (alert) means play is being wasted."),
    ("Held-out RHAE/tok vs base", ["eval.heldout_ratio", "eval.heldout_ratio_lo", "eval.heldout_ratio_hi", "eval.heldout_ratio_target"], "x", "Training",
     "RL's target read: held-out RHAE per token of the policy divided by the base model's on the same tries, with its 95% interval (dotted). The target is 2x."),
    ("RHAE/tok vs reference", ["train.rhae_tok_vs_ref.all", "eval.rhae_tok_vs_ref.all"], "x", "Training",
     "RL's target, all levels: each try's RHAE per token against its own level's base reference (geometric mean), so neither the level mix of a step nor L2's higher scores move it. Base = the starting model on the same levels; the target is 2x base."),
    ("Reference coverage", ["train.*cov*", "train.ref_coverage*"], "%", "Training",
     "RL: share of a step's tries whose level has a base reference, so they count in RHAE/tok vs reference (training and base lines). Low coverage makes that panel noisy."),
    ("Update progress", ["train.live_groups_done", "train.live_groups_total"], "groups", "Training",
     "RL, inside the current step: groups trained so far out of the step's total. A step lasts until its last group is played and trained; a slow climb means play (long tries) is the bottleneck."),
    ("Card 2", ["infra.card2_mode", "infra.card2_moved_tries"], "", "Infrastructure",
     "RL two-card runs: card 2's mode (1 = playing, 0 = training) and tries moved off card 2 at each switch to training. Each switch costs play time; moved tries re-read their context on card 1."),
    ("Advantage signal", ["train.live_reward_std", "train.live_w_abs_median", "train.live_gigpo_step_abs"], "", "Training",
     "RL, per trained group: spread of the try rewards (std), median size of the try advantage, and mean size of the per-turn (GiGPO) term. Learning signal needs spread; if the per-turn term is much larger than the try advantage it overrides the episode verdict."),
    ("Per-turn overrides", ["train.live_sign_flips"], "replies", "Training",
     "RL: replies whose per-turn (GiGPO) term flipped the sign of the try's advantage. Rule: over 20% of a step's replies lowers the per-turn weight."),
    ("RHAE per 10K tokens", ["train.rhae_per_10k.all", "eval.rhae_per_10k.all", "eval.rhae_per_10k.L13"], "", "Training",
     "All levels pooled (training tries; held-out shows L1+L3 until its all-level read is exported). RL's main metric: level score (RHAE) earned per 10,000 generated tokens, on this round's training tries and on held-out starts. Rising means the model clears levels with fewer tokens; this is the number the run has to double."),
    ("RHAE mean", ["train.rhae_mean.all", "eval.rhae_mean.all", "eval.rhae_mean.L13"], "", "Training",
     "All levels pooled (training tries; held-out shows L1+L3 until its all-level read is exported). Average level score per try (RHAE, capped at 1.15 by the official formula), regardless of tokens spent. It should hold or rise while RHAE per 10K rises; a fall means the model is saving tokens by playing worse."),
    ("Clear rate", ["train.clear_rate.all", "eval.clears_frac.all", "eval.clears_frac.L13"], "%", "Training",
     "All levels pooled (training tries; held-out shows L1+L3 until its all-level read is exported). Share of tries that cleared their level, on training tries (all levels) and on held-out starts. Near 100% on training tries means the starts are getting too easy to teach much."),
    ("Tokens per try", ["train.tokens_per_try.all", "eval.tokens_per_try.all", "eval.tokens_per_try.L13"], "tokens", "Training",
     "All levels: mean generated tokens per try (cleared or not), on training tries and held-out starts. Falling at a steady clear rate means cheaper play."),
    ("Actions per clear", ["train.actions_per_clear.all", "eval.actions_per_clear.all", "eval.actions_per_clear.L13"], "actions", "Training",
     "All levels pooled (training tries; held-out shows L1+L3 until its all-level read is exported). Game actions used per cleared level (the official score charges for actions). Falling at the same clear rate is more efficient play."),
    ("Policy sharpness", ["train.entropy"], "nats", "Training",
     "How spread out the model's next-token choices are on its own replies (entropy). A slow fall is normal in RL; a fast drop toward 0 means the policy is collapsing to one way of playing."),
    ("Play progress", ["play.*"], "", "Training",
     "RL, inside the current round: tries finished and levels cleared so far (resets each round). Should climb steadily during play; a flat line while the round is playing means play stalled."),
    ("Clears by level", ["train.clears_frac.*"], "%", "Training",
     "RL: share of this round's training tries that cleared, per level. Rising is learning; the same few levels pinned at 0% or 100% give no learning signal."),
    ("Useful groups", ["train.nondegenerate_frac", "train.groups_*"], "%", "Training",
     "RL: share of try groups whose tries split between clear and fail. Groups where every try clears still teach through token cost (the reward counts tokens), so this understates the signal; falling toward zero means the starts are too easy or too hard."),
    ("Advantage size", ["train.adv_abs_*"], "", "Training",
     "RL: median absolute advantage of the trained tries (how far each try's reward sits from its group's mean). It is not the reward: it falls when tries within a group score alike, which means less to learn from that round."),
    ("Tokens per try", ["train.try_tokens_*"], "tokens", "Training",
     "RL: tokens spent per try (median and 90th percentile). Falling at equal clears is the efficiency gain we want; a rising tail means longer, stuck games."),
    ("Update size", ["train.update_norm*", "train.kl_excess*"], "", "Training",
     "RL: how much each update moved the model (weight-change size and KL above the noise floor). The ramp rule raises the dose after quiet updates; spikes here come before breakage."),
    ("KL sensitivity", ["train.kl_k*"], "", "Training",
     "RL: KL moved per unit of squared learning rate (excess KL / lr²), per update; the dose controller aims with it. Flat is healthy. Doubling every update means the policy is sharpening fast, so a fixed step overshoots its KL target."),
    ("Loss", ["train.loss", "train.kd*", "train.policy_loss", "train.nll"], "", "Training",
     'Training objective; lower is better for SFT/distillation (loss, kd, nll). Should fall then flatten; sudden spikes mean bad batches or too-high learning rate. RL policy loss is noisy by nature: read reward instead.'),
    ("Draft head (MTP)", ["train.mtp_kl", "train.accept1"], "", "Training",
     "SFT: the speculative draft head's KL to the trunk (lower is better) and its first-token acceptance (higher is better) on each step's batch. A fall in acceptance means the draft no longer tracks the trunk."),
    ("Update call", ["train.call_util", "train.ahead_hit_frac"], "%", "Infrastructure",
     "RL: GPU use over the update's one trainer call, and the share of its windows taken pre-encoded from the early-ahead files (should be near 100%; 0% means every window was encoded at the call with the cards idle)."),
    ("Update idle", ["train.setup_s", "train.window_wait_s"], "s", "Infrastructure",
     "RL: seconds of the update call before the first window (setup) and waiting on window encodes. Near zero when early-ahead works."),
    ("MFU (estimate)", ["train.mfu"], "%", "Infrastructure",
     "SFT: analytic model FLOPs per step divided by step time and the B300's dense BF16 peak. Flat is healthy; the MFU arm's target is 10%."),
    ("Reward", ["train.reward_mean", "train.reward_std"], "", "Training",
     "Average score of the model's own games (RL). Should trend up. Reward spread near zero means every try scores the same, so there is nothing left to learn from."),
    ("Gradient norm", ["train.grad_norm*"], "", "Training",
     'Size of each weight update before clipping. Steady or slowly falling is healthy; sharp spikes are the earliest sign of instability.'),
    ("Learning rate", ["train.lr*"], "", "Training",
     'Step size schedule: warm-up ramp, then decay. Check it matches the plan; it should reach its floor near the last planned update.'),
    ("General-text check", ["train.kl_general*", "eval.general*"], "nats", "Results",
     "Heals: KL to the teacher on general (non-game) text, measured before training and again at the end. It must not rise by more than 2% (the damage guard); a fall means the heal also helps outside games."),
    ("Drift (KL)", ["train.kl_*"], "nats", "Training",
     'How far the model has moved from where it started (or from its target). Rising slowly is expected; a sudden jump means the model is changing faster than the reward justifies. kl_target (dotted) is the planned dose.'),
    ("Approx KL per step (k3)", ["train.k3_measure", "train.approx_kl"], "nats", "Training",
     "The trainer's KL estimate (k3) per update (k3_measure: measured after the step on a held sample; approx_kl on multi-step updates), one line per update. Informational (KL is not a stop signal)."),
    ("Per-token drift and off-policy share", ["train.k3_tok", "train.tis_out_share"], "", "Training",
     "k3_tok: mean per-token KL of the update on its own tries (a jump with flat k3 means a few tokens moved a lot); tis_out_share: share of tokens outside the TIS cap (alert above 5%)."),
    ("Clipping", ["train.clip_frac*", "train.tis_clip_frac", "train.live_tis_out_frac"], "%", "Training",
     'Share of tokens whose update was capped by the safety clip. A few % is normal; rising toward 20%+ means updates are too large or the data is stale. SFT: share of steps so far with grad norm over the clip (1.0).'),
    ("Drift from base", ["train.drift_gap_base*", "train.logprob_gap*", "train.nll_base_tokens*", "train.nll_trainer*"], "nats", "Training",
     "RL: how differently the trainer now scores the base model's recorded play (logprob gap vs the base's own logprobs, and the trainer's loss on base tokens). Grows as the policy moves away from base, which is expected; it is not a serving error. Earlier rounds show it as logprob_gap / nll_trainer."),
    ("Train/serve mismatch", ["train.tis_mean*", "train.routing_*"], "", "Training",
     "RL, on the update's own tries: trainer/server probability ratio (tis_mean, ~1 when they agree; at most one update of lag), how often the trainer's expert choice matches the server's (routing agree; ~0.6-0.7 normal, drops when the trainer is an update ahead) and routing coverage (should be 1). A jump at an update's first step is a real serving mismatch."),
    ("Policy vs control", ["train.vs_control"], "x", "Results",
     "RL: every 4 updates one policy group gets an untrained twin of 8 tries on the base model from the same start and budget; exp(mean over the last 4 pairs of policy mean logeq minus control mean logeq). 1.0 = no learning on the levels it plays; above 1 = the policy beats its own base there."),
    ("Minibatch checks", ["train.mb_k3_last_vs_first", "train.mb_clip_last_vs_first", "train.update_norm_sum"], "",
     "Training",
     "RL with 4 optimizer steps per update: k3 and TIS clip share of the last step over the first (bars: <= 3), and the update's summed update norm (bar <= 0.6). Empty while the run takes one step per update."),
    ("Update mix", ["train.top_game_share"], "%", "Training",
     "RL: the largest single game's share of the groups played for the update. Above 40% one game drives the "
     "update's RHAE/tok and tokens/try; above 50% those reads say more about the mix than the policy (U21-U22: sb26 11 of 16)."),
    ("Advantage share", ["train.adv_nonzero_share", "train.adv_up_share"], "%", "Training",
     "RL: share of the update's trained tokens with a non-zero advantage weight (and with a positive one). Low = most tokens carry no signal (groups or turns without spread)."),
    ("Reply length", ["train.reply_tokens_mean"], "tokens", "Training",
     'Average tokens per model reply. Slow change is fine; a steady climb or collapse often precedes reward hacking or degenerate output.'),
    ("Reply quality", ["train.truncation_frac", "train.noaction_frac"], "", "Training",
     "Truncation = share of replies cut off at the cap (above 10-15% destabilizes RL); no-action = share of replies that send no game action (only thinking or python), normal around 40% for this harness, a steady rise means the model stalls instead of playing."),
    ("Held-out reads", ["eval.*"], "", "Results",
     "Reads on games the model never trained on: the result that counts. Scores (rhae, clears): higher is better. Held-out KL (heals): how far the model's replies are from the teacher's, lower is better. Base = the same read on the starting model (dotted line, or 'base → now'); differences smaller than the read's noise are not real."),
]
PHASE_COLORS = {"time.play_s": "#10b981", "time.old_pass_s": "#8b5cf6", "time.update_s": "#3b82f6", "time.reload_s": "#eab308",
                "time.idle_s": "#ef4444", "time.handoff_s": "#64748b"}
COLORS = ["#3b82f6", "#f97316", "#10b981", "#ec4899", "#8b5cf6", "#14b8a6", "#eab308", "#64748b"]
ARM = {"rl": "#e11d48", "pth": "#7c3aed", "sft": "#7c3aed"}


END_STATES = ("done", "stopped", "completed", "ended", "failed", "cancelled", "canceled")
ended_state = lambda st: str(st or "").split("-")[0] in END_STATES  # "stopped-owner", "failed-oom" ... end too
NO_TRAIN_METRICS = (("sft-layers-", "layer-scoring read: writes layers.json, no training metrics by design"),)


def record(run):
    try:
        return json.loads((ROOT / "docs/runs" / f"{run}.json").read_text())
    except (OSError, ValueError):
        return {}


def record_ended(run, apps=None):
    """The run's own record says it ended (its state, or every ledger row settled), or its Modal app is listed stopped.
    None: no record (status then falls back to the queue and data age)."""
    rec = record(run)
    if not rec:
        return None
    rows = [x.get("row", {}) for x in rec.get("ledger", [])]
    if ended_state(rec.get("state")) or (rows and all(ended_state(r.get("state")) for r in rows)):
        return True
    ids = {r.get("app_id") for r in rows if r.get("app_id")}
    return bool(apps is not None and ids and all(apps.get(i) == "stopped" for i in ids))


def record_what(run, descriptions=None):
    """A title for a run without an exporter meta: the record's what, else the queue's description for the run (exact,
    else the longest matching prefix). The launch authorization is not a title (28 Oct)."""
    rec = record(run)
    w = (rec.get("what") or ((rec.get("queue") or {}).get("row") or {}).get("what")  # queue --set what= lands in queue.row
         or next((x["row"]["what"] for x in rec.get("ledger", []) if x.get("row", {}).get("what")), ""))
    if not w and descriptions:
        k = run if run in descriptions else max((k for k in descriptions if run.startswith(k)), key=len, default=None)
        w = descriptions.get(k, "") if k else ""
    return w if isinstance(w, str) else ""


def load():
    runs = []
    if not SRC.exists():
        return runs
    now = time.time()
    try:
        qlive = paths.read_json(ROOT / "docs/modal-queue.json").get("live", [])
        descriptions = paths.read_json(ROOT / "docs/modal-queue.json").get("descriptions") or {}
        queued_live = {e["run"] for e in qlive if not ended_state(e.get("state"))}
        row_ended = {e["run"] for e in qlive if ended_state(e.get("state"))}  # the queue says it ended: shown ended at once
        starts = {e["run"]: e["start_epoch"] for e in qlive if e.get("start_epoch")}
    except (OSError, ValueError):
        queued_live, starts, row_ended, descriptions = None, {}, set(), {}
    if queued_live is None:  # queue unreadable: trust the exporters' own status
        queued_live = _AllRuns()
    listing = paths.modal_apps()  # None = unknown, never "all stopped"
    apps = {a["app_id"]: a.get("state") for a in listing} if listing else None
    for f in sorted(SRC.glob("*.jsonl")):
        if f.name.endswith((".alerts.jsonl", ".extra.jsonl")):  # side files (alerts; exporter rows beside a watcher-mirrored file): merged below
            continue
        meta_f = f.with_suffix(".meta.json")
        if "-hread-" in f.stem:  # held-out reads are not training runs; their result plots on the RL run's card
            continue
        if not f.stem.startswith(TRAIN_RUN_PREFIXES):  # RL and OPD training runs only (the shared watcher writes rows for every run)
            try:
                listed = json.loads(meta_f.read_text()).get("arm") in TRAIN_ARMS
            except (OSError, ValueError):
                listed = False
            if not listed:
                continue
        rows, seen, by_t = [], set(), {}
        for g in (f, f.with_name(f.stem + ".extra.jsonl"), f.with_name(f.stem + ".alerts.jsonl")):
            for line in (g.read_text().splitlines() if g.exists() else []):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("m") and not r.get("alert") and r.get("t") in by_t:  # re-exported row (backfill, restarted watcher): merge, plot once
                    by_t[r["t"]]["m"].update(r["m"])
                    continue
                if r.get("m") and not r.get("alert") and r.get("t") is not None:
                    by_t[r["t"]] = r
                if r.get("alert"):
                    if (r.get("t"), r["alert"]) in seen:
                        r = {k: v for k, v in r.items() if k != "alert"}
                    seen.add((r.get("t"), r.get("alert")))
                if r.get("m") or r.get("alert"):
                    rows.append(r)
        meta_f = f.with_suffix(".meta.json")
        if not rows:  # launched, nothing measured yet: list it so the run is visible from its first minute
            if meta_f.exists():
                rows = [{"t": meta_f.stat().st_mtime, "m": {}}]
            else:
                continue
        meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
        meta.setdefault("run", f.stem)
        if not meta.get("what"):
            meta["what"] = record_what(meta["run"], descriptions)
        meta.setdefault("_start", starts.get(meta["run"]))  # this run's own launch: earlier (backfilled) history plots at negative hours
        # "last data" = the newest measured metric, not exporter bookkeeping (28 Oct: fresh host/cost rows hid a sampler
        # frozen at 0.04 h): rows with only cost/host keys count only when nothing else exists
        measured = [r.get("t", 0) for r in rows if set(r.get("m") or {}) - BOOKKEEPING]
        last = max(measured or [r.get("t", 0) for r in rows])
        if meta.get("_start") and last < meta["_start"] and meta["run"] in queued_live:
            rows.append({"t": meta["_start"], "m": {}})  # launched; only backfilled history so far
            last = meta["_start"]
            meta["what"] = " · ".join(filter(None, [meta.get("what"), "launched, waiting for its first live metrics"]))
        # status: the record's state and its app's (28 Oct: a stopped run showed live for 22 min); data age only without a record
        rec_end = record_ended(meta["run"], apps)
        if meta.get("status", "live") == "live" and (meta["run"] in row_ended or rec_end
                                                     or (rec_end is None and meta["run"] not in queued_live and now - last > STALE_S)):
            meta["status"] = "ended"
        # attempts of one record (relaunch under the same run id, 28 Oct real-v10 attempt 2): a marked restart on every plot;
        # the current attempt's own start is the side plots' "latest start" and the page reads "booting" until it plays/trains
        att = [a for a in record(meta["run"]).get("attempts", []) if a.get("start_epoch")]
        if att:  # 0 h = the first attempt's launch: earlier attempts plot at positive hours (full lineage)
            meta["_start"] = min(float(att[0]["start_epoch"]), meta.get("_start") or float("inf"))
        if len(att) > 1:
            meta["_attempt_stages"] = []
            for p_, a in zip(att, att[1:]):
                t_a = float(a["start_epoch"])
                meta["_attempt_stages"].append((t_a, f"attempt {a.get('n')} (after {p_.get('end_cause') or 'a restart'})"))
                rows.append({"t": t_a, "m": {}})
            if not ended_state(att[-1].get("state")):
                meta["_attempt_start"] = float(att[-1]["start_epoch"])
        runs.append((meta, sorted(rows, key=lambda r: r.get("t", 0)), last))
    # one run continued under new ids (preemption, crash, restart): meta "continues": [earlier run ids]. The latest leg's view
    # shows every leg's rows from the first leg's launch, and the earlier legs leave the run list (22 Oct, all rounds)
    by = {m["run"]: (m, rows, last) for m, rows, last in runs}
    def legs(run, seen=()):
        m = by[run][0]
        out = []
        for prev in m.get("continues") or []:
            prev = prev if isinstance(prev, dict) else {"run": prev}  # {"run", "until"}: the parent's rows up to the branch point
            if prev["run"] in by and prev["run"] not in seen:
                out += legs(prev["run"], seen + (run,))
                if prev.get("until"):
                    cut[prev["run"]] = prev["until"]
        return out + [run]
    cut = {}
    merged = set()
    for m, rows, last in list(by.values()):
        chain = legs(m["run"])
        if len(chain) > 1:
            merged.update(chain[:-1])
            allrows, at, cost0, upto, reads = [], {}, 0.0, 0.0, []
            for i, leg in enumerate(chain):  # legs carry backfilled copies of earlier history: one row per time, metrics merged
                legcost = 0.0
                # past the branch point, a parent's rounds played by a checkpoint the next legs resumed from still read that
                # checkpoint: kept for the by-checkpoint panels only (every read of a checkpoint on one line)
                res = next((by[c][0]["resume_from"] for c in chain[i + 1:] if by[c][0].get("resume_from") is not None), None)
                for r in by[leg][1]:
                    t = r.get("t", 0)
                    if t > cut.get(leg, float("inf")):
                        pb = (r.get("m") or {}).get("train.played_by")
                        if res is not None and isinstance(pb, (int, float)) and pb <= res:
                            reads.append(dict(r, _leg=leg))
                        continue
                    own = t > upto  # past the earlier legs: this leg's own data
                    if r.get("m") and not r.get("alert") and t in at:
                        if own:
                            at[t]["m"].update(r["m"])
                        continue
                    if r.get("m") and not r.get("alert"):
                        r = dict(r, m=dict(r["m"]), _leg=leg)
                        if own and isinstance(r["m"].get("infra.cost_usd"), (int, float)):  # each leg's cost starts at 0: add the earlier legs'
                            legcost = max(legcost, r["m"]["infra.cost_usd"])
                            r["m"]["infra.cost_usd"] += cost0
                        at[t] = r
                    elif not own:
                        continue  # an earlier leg's alert copied into this leg's file
                    allrows.append(r)
                cost0 += legcost
                upto = max([upto] + [r.get("t", 0) for r in by[leg][1] if r.get("t", 0) <= cut.get(leg, float("inf"))])
            allrows.sort(key=lambda r: r.get("t", 0))
            starts, seen = [], set()  # where each leg begins: a dotted vertical line labelled with the leg's "stage" (or its run id)
            for leg in chain:  # a leg's own start: its first row that no earlier leg has (legs carry backfilled history)
                ts = [r["t"] for r in by[leg][1] if r.get("t") is not None and r["t"] <= cut.get(leg, float("inf"))]
                own = [t for t in ts if t not in seen and t > max(seen, default=0)]
                if own:
                    starts.append((min(own), by[leg][0].get("stage") or leg))
                seen.update(ts)
            m["_stages"] = starts
            m["_own_start"] = starts[-1][0] if starts else None  # alerts before the current leg belong to an earlier leg
            first = by[chain[0]][0].get("_start") or allrows[0].get("t")
            m["_start"] = first
            m["legs"] = chain
            m["_reads"] = reads
            by[m["run"]] = (m, allrows, last)
    runs = [v for k, v in by.items() if k not in merged and not (v[0].get("status", "live") != "live" and now - v[2] > KEEP_ENDED_S)]
    return sorted(runs, key=lambda x: (x[0].get("status", "live") != "live", -x[2]))


HIDE = ["*.L2", "play.*.*", "play.played_by", "train.rhae_per_10k", "eval.rhae_per_10k", "train.rhae_mean", "eval.rhae_mean", "train.clear_rate", "eval.clears_frac", "train.actions_per_clear", "eval.actions_per_clear", "train.round", "train.step_in_update", "train.pass", "train.played_by", "play.round", "play.phase", "eval.rhae_*_base", "eval.clears_frac_base", "eval.actions_per_clear_base", "eval.rhae_mean_base"]  # RL: R1 is the base read already
PREFIX_NAMES = {"train": "training tries", "eval": "held-out", "infra": "", "time": "", "play": ""}
NAMES = {"eval.heldout_ratio": "policy ÷ base", "eval.heldout_ratio_lo": "95% low", "eval.heldout_ratio_hi": "95% high", "eval.heldout_ratio_target": "target 2x",
         "train.rhae_tok_vs_ref.all": "policy", "train.rhae_tok_vs_ref_base.all": "base model", "eval.rhae_tok_vs_ref.all": "held-out policy",
         "eval.rhae_tok_vs_ref_base.all": "held-out base", "train.clip_frac": "PPO clip", "train.live_tis_out_frac": "TIS outside range",
         "infra.gen_tok_s": "play (generation)", "train.live_tok_s": "trainer", "infra.train_tok_s": "trainer", "infra.teacher_tok_s": "teacher"}


def is_frac(k):
    return re.search(r"(_frac|_share|clear_rate)(_base|_target|_step)?($|\.)", k) is not None


GAP_S = 30 * 60  # pauses longer than this (a resume, backfilled pilot history) are cut out of the x axis (30 min)


def run_clock(rows, start=None):
    """Row time -> hours of run time, with pauses over GAP_S shrunk to 0.02 h; 0 = the run's own launch when known."""
    ts = sorted({r.get("t", 0) for r in rows})
    x, h, prev = {}, 0.0, None
    for t in ts:
        if prev is not None:
            d = t - prev
            h += 0.02 if d > GAP_S else d / 3600
        x[t], prev = h, t
    own = [x[t] for t in ts if start and t >= start - 300]
    z = own[0] if own else 0.0
    return {t: v - z for t, v in x.items()}


def series(rows, clock):
    out = {}
    for r in rows:
        m = r.get("m") or {}
        for k, v in m.items():
            if k.startswith("train.kl_k") and m.get("train.kl_excess", 1) <= 0:
                continue  # excess / lr² means nothing when the excess is at or below the floor
            if isinstance(v, (int, float)):
                out.setdefault(k, ([], []))
                out[k][0].append(round(clock[r["t"]], 4))
                out[k][1].append(v * 100 if is_frac(k) else v)
    return out


def at_clock(clock, t):
    """Any epoch -> run hours on the collapsed clock (the nearest earlier row's hour plus the time since it, gap-capped)."""
    ts = sorted(clock)
    if not ts:
        return 0.0
    prev = max((x for x in ts if x <= t), default=ts[0])
    d = max(0.0, t - prev)
    return round(clock[prev] + (0.02 if d > GAP_S else d / 3600), 4)


def update_obs(rows):
    """RL per-update values, branch-aware. The branch comes from the data: train.step going down is a
    rollback, and the earlier branch's rows from that update on are abandoned (never joined to the live line).
    Returns ({key: {(branch, update): (t, value)}}, [(t, update) where each branch starts])."""
    obs, starts, br, top, cur = {}, [(None, None)], 0, -1, None
    for r in rows:
        m = r.get("m") or {}
        s = m.get("train.step")
        if isinstance(s, (int, float)):
            s = int(s)
            if s < top:
                br += 1
                starts.append((r["t"], s))
                top = s
            top, cur = max(top, s), s
        if cur is None:
            continue
        for k, v in m.items():
            if (isinstance(v, (int, float)) or k == "train.top_game") and k.startswith(("train.", "eval.")) and not k.startswith("train.live_") and k != "train.step":
                obs.setdefault(k, {})[(br, cur)] = (r["t"], v)
    return obs, starts


def split_branches(d, starts=()):
    """{(branch, update): (t, v)} -> (live [(u, v, t)], {branch: [(u, v, t)]} abandoned): an update's live value is its
    latest branch's. starts (update_obs): an update at or past a later branch's first update was rolled back even
    before that branch reaches it."""
    dead = lambda b, u: any(s is not None and s <= u for _, s in starts[b + 1:])  # noqa: E731
    best = {}
    for (b, u), (t, v) in d.items():
        if not dead(b, u) and (u not in best or b > best[u][0]):
            best[u] = (b, t, v)
    live = sorted((u, v, t) for u, (b, t, v) in best.items())
    gone = {}
    for (b, u), (t, v) in d.items():
        if u not in best or best[u][0] != b:
            gone.setdefault(b, []).append((u, v, t))
    return live, {b: sorted(p) for b, p in gone.items()}


def live_control(uobs, starts, last=4):
    """train.vs_control on the live branch only: exp(mean of the last 4 live pairs' train.control_diff), as export_rl's
    control_pairs. A running exporter's state may still hold abandoned pairs; their faded points keep the exported value."""
    import math
    d = uobs.get("train.control_diff")
    if not d:
        return
    vals = uobs.setdefault("train.vs_control", {})
    live = split_branches(d, starts)[0]
    at = {(u, t): k for k, (t, _) in d.items() for u in [k[1]]}
    for i, (u, _, t) in enumerate(live):
        w = [x[1] for x in live[max(0, i - last + 1):i + 1]]
        vals[at[(u, t)]] = (t, round(math.exp(sum(w) / len(w)), 4))


def heldout_reads(meta):
    """Held-out RHAE/tok reads of this run's lineage, from the read runs' records (records.py result): one point per read
    step with its 95% CI (log scale, delta method from the read's and the base's standard errors)."""
    import glob, math
    legs = [short_leg(x) for x in (meta.get("legs") or [meta["run"]])]
    out = []
    for p in glob.glob(str(ROOT / "docs/runs/*hread*.json")):
        try:
            rec = json.loads(pathlib.Path(p).read_text())
        except (OSError, ValueError):
            continue
        run = rec.get("run", "")
        text = run + " " + str(((rec.get("queue") or {}).get("row") or {}).get("what") or "")
        st = re.search(r"-s(\d{4})(?:-|$)", run)
        if not st or not any(lg and lg in text for lg in legs):
            continue
        for res in rec.get("results") or []:
            mt = res.get("metrics") or {}
            if not isinstance(mt.get("ratio"), (int, float)):
                continue
            a, b, sd = mt.get("rhae10k"), mt.get("rhae10k_base"), mt.get("se_diff")
            sb = None
            try:
                base = json.loads((ROOT / "docs/runs" / f"{res.get('vs')}.json").read_text())
                sb = next((x["metrics"].get("se") for x in base.get("results") or [] if (x.get("metrics") or {}).get("se")), None)
            except (OSError, ValueError, TypeError):
                pass
            lo = hi = None
            if a and b and sd:
                sa = math.sqrt(max(sd ** 2 - sb ** 2, 0)) if sb and sd > sb else sd / math.sqrt(2)
                sb = sb if sb and sd > sb else sd / math.sqrt(2)
                rel = math.sqrt((sa / a) ** 2 + (sb / b) ** 2)
                lo, hi = mt["ratio"] * math.exp(-1.96 * rel), mt["ratio"] * math.exp(1.96 * rel)
            out.append({"step": int(st.group(1)), "ratio": mt["ratio"], "lo": lo, "hi": hi, "t": res.get("t"),
                        "verdict": res.get("verdict") or "", "run": run})
    return sorted(out, key=lambda x: (x["step"], x["t"] or 0))


def run_events(meta, rows, starts):
    """Attempts (record), rollbacks (branch starts) and incidents (crash/incident/gone alerts) for the timeline strip;
    marks within 30 min of each other are merged."""
    rec = record(meta["run"])
    att = [a for a in rec.get("attempts", []) if a.get("start_epoch")]
    last_t = max((r.get("t", 0) for r in rows), default=0)
    bands = []
    for i, a in enumerate(att):
        t0 = float(a["start_epoch"])
        t1 = float(a.get("end_epoch") or (att[i + 1]["start_epoch"] if i + 1 < len(att) else last_t))
        bands.append({"n": a.get("n", i + 1), "t0": t0, "t1": max(t1, t0), "state": a.get("state") or "",
                      "cause": a.get("end_cause") or ("running" if not ended_state(a.get("state")) else "")})
    marks = [(rollback_t(bands, t), f"rollback to s{s:04d}", "rollback") for t, s in starts if t]
    for r in rows:
        al = r.get("alert") or ""
        if re.search(r"INCIDENT|\[crash\]|\[gone\]|preempt", al):
            marks.append((r["t"], clip(al.replace("watch: ", ""), 90), "incident"))
    marks += [(b["t0"], f"attempt {b['n']} starts" + (f" (after {bands[i - 1]['cause']})" if i else ""), "attempt")
              for i, b in enumerate(bands)]
    marks.sort()
    merged = []
    for t, lab, kind in marks:
        if merged and t - merged[-1]["t"] <= 1800:
            merged[-1]["labels"].append(lab)
            merged[-1]["kinds"].add(kind)
        else:
            merged.append({"t": t, "labels": [lab], "kinds": {kind}})
    return bands, merged


def rollback_t(bands, t):
    """When a rollback happened: the start of the attempt that resumed from the earlier checkpoint (the record), not its
    first update row (a10's rollback 01:48 UTC read as 04:30, its first post-rollback update)."""
    return max((b["t0"] for b in bands if b["t0"] <= t), default=t)


def bar_line(xs, y, name, color="#64748b", dash="dot"):
    return {"x": [min(xs), max(xs)] if xs else [0, 1], "y": [y, y], "name": name, "type": "scatter", "mode": "lines",
            "line": {"width": 1.5, "color": color, "dash": dash}, "hoverinfo": "skip"}


def breaches(pts, test, name="over its bar"):
    """Red rings on the points that breach a pre-registered bar."""
    bad = [(x, y) for x, y in pts if y is not None and test(x, y)]
    if not bad:
        return None
    return {"x": [x for x, _ in bad], "y": [y for _, y in bad], "name": name, "type": "scatter", "mode": "markers",
            "marker": {"size": 11, "color": "rgba(0,0,0,0)", "line": {"width": 2, "color": "#dc2626"}},
            "hovertemplate": "U%{x}: %{y:.4g} " + name + "<extra></extra>"}


SECTION_ORDER = {"Key plots": -3, "This round, live": -2, "Latest update, by pass": -1, "Training": 0, "Infrastructure": 1, "Results": 2, "Other": 3}
VRAM_CAP_GIB = 268.6  # B300 usable, when the run does not report its cap
CHIPS = [("update", "train.update", "", 0), ("cost so far, $", "infra.cost_usd", "", 0), ("play tok/s", "infra.gen_tok_s", "", 0), ("train tok/s", "infra.train_tok_s", "", 0),
         ("train tok/s", "train.live_tok_s", "", 0), ("RHAE/tok vs ref", "train.rhae_tok_vs_ref.all", "x", 2), ("groups trained this step", "train.live_groups_done", "", 0),
         ("loss", "train.loss", "", 4), ("kd", "train.kd", "", 4), ("reward", "train.reward_mean", "", 3),
         ("grad norm", "train.grad_norm", "", 3), ("KL excess", "train.kl_excess", "", 4), ("training step", "train.step", "", 0),
         ("RHAE/10K, training", "train.rhae_per_10k.all", "", 3), ("RHAE/10K L1+L3, held-out", "eval.rhae_per_10k.L13", "", 3),
         ("entropy", "train.entropy", "", 3), ("held-out KL", "eval.heldout_reply_kl", "", 4)]


# per-round reads plotted by the checkpoint that played them: x = r07 etc., every read of one checkpoint on
# one vertical line (hover: leg and round), a dotted line through each checkpoint's mean
BY_CKPT = {"RHAE per 10K tokens", "RHAE mean", "Clear rate", "Actions per clear", "Tokens per try", "Clears by level", "Useful groups"}


def short_leg(leg):
    return re.sub(r"^rl1-loop-(m6-)?", "", leg or "")


def ckpt_fig(reads, keys, own_leg):
    pts, seen = {}, set()
    for r in reads:
        m = r.get("m") or {}
        pb = m.get("train.played_by")
        if not isinstance(pb, (int, float)):
            continue
        dk = (r.get("_leg") or own_leg, m.get("train.round"))
        for k in keys:
            if isinstance(m.get(k), (int, float)) and (dk, k) not in seen:
                seen.add((dk, k))
                pts.setdefault(k, []).append((int(pb), m[k] * 100 if is_frac(k) else m[k], f"{short_leg(dk[0])} R{int(dk[1]) if dk[1] is not None else '?'}"))
    if not pts:
        return None
    lab = lambda c: "base" if c == 0 else f"r{c:02d}"
    have = {p[0] for v in pts.values() for p in v}
    cats = [lab(c) for c in range(min(have), max(have) + 1)]  # unread checkpoints keep their slot
    traces = []
    for j, k in enumerate(sorted(pts)):
        c = COLORS[j % len(COLORS)]
        pre, suf = k.split(".", 1) if "." in k else ("", k)
        nm = PREFIX_NAMES.get(pre) if len({kk.split(".", 1)[0] for kk in pts}) > 1 and PREFIX_NAMES.get(pre) else (
            suf.rsplit(".", 1)[-1] if re.search(r"\.L\d+$", suf) else suf.replace("_", " "))
        p = sorted(pts[k])
        traces.append({"x": [lab(a) for a, _, _ in p], "y": [b for _, b, _ in p], "text": [t for _, _, t in p], "name": nm, "type": "scatter",
                       "mode": "markers", "marker": {"size": 8, "color": c}, "hovertemplate": "%{text}: %{y:.4g}<extra>" + nm + "</extra>"})
        mean = {}
        for a, b, _ in p:
            mean.setdefault(a, []).append(b)
        xs = sorted(mean)
        traces.append({"x": [lab(a) for a in xs], "y": [sum(mean[a]) / len(mean[a]) for a in xs], "name": nm + " mean", "type": "scatter",
                       "mode": "lines", "showlegend": False, "hoverinfo": "skip", "line": {"width": 1.5, "dash": "dot", "color": c}})
    ys = [y for t in traces for y in t["y"]]
    return {"traces": traces, "cats": cats, "rng": [0, max(ys) * 1.15] if ys and min(ys) >= 0 and max(ys) > 0 else None}


# the latest update's steps by pass, for feedback inside an update
PASS_METRICS = [("k3 per step", "train.approx_kl", "nats", "The trainer's KL estimate after each step; dotted lines = target and early-stop line."),
                ("Clipped share", "train.clip_frac", "%", "Share of replies the GSPO band clipped in each step; later passes clip more as the policy moves."),
                ("Policy loss", "train.policy_loss", "", "Policy loss per step; later passes over the same data should keep lowering it."),
                ("Entropy", "train.entropy", "nats", "Entropy of the model's replies per step; a fast fall across passes means the policy is collapsing."),
                ("Gradient norm", "train.grad_norm", "", "Gradient norm per step.")]


def pass_figs(rows):
    steps = [r["m"] for r in rows if isinstance((r.get("m") or {}).get("train.step_in_update"), (int, float))
             and isinstance(r["m"].get("train.round"), (int, float))]
    if not steps:
        return []
    ups = sorted({int(m["train.round"]) for m in steps})
    last, prev = ups[-1], (ups[-2] if len(ups) > 1 else None)
    figs = []
    for title, key, unit, helptext in PASS_METRICS:
        traces = []
        if prev is not None:
            p = sorted((m["train.step_in_update"], m[key]) for m in steps if int(m["train.round"]) == prev and isinstance(m.get(key), (int, float)))
            if p:
                traces.append({"x": [a for a, _ in p], "y": [b * 100 if unit == "%" else b for _, b in p], "name": f"U{prev}", "type": "scatter",
                               "mode": "lines", "line": {"width": 1.5, "dash": "dot", "color": "#94a3b8"}, "hovertemplate": f"U{prev} step %{{x}}: %{{y:.4g}}<extra></extra>"})
        cur = [m for m in steps if int(m["train.round"]) == last and isinstance(m.get(key), (int, float))]
        for j, ps in enumerate(sorted({int(m.get("train.pass") or 1) for m in cur})):
            p = sorted((m["train.step_in_update"], m[key]) for m in cur if int(m.get("train.pass") or 1) == ps)
            traces.append({"x": [a for a, _ in p], "y": [b * 100 if unit == "%" else b for _, b in p], "name": f"U{last} pass {ps}", "type": "scatter",
                           "mode": "lines+markers", "line": {"width": 2.2, "color": COLORS[j % len(COLORS)]}, "marker": {"size": 6, "color": COLORS[j % len(COLORS)]},
                           "hovertemplate": f"pass {ps}, step %{{x}}: %{{y:.4g}}<extra></extra>"})
        if not any(t["y"] for t in traces):
            continue
        ys = [y for t in traces for y in t["y"]]
        figs.append({"section": "Latest update, by pass", "title": f"{title} · U{last}", "unit": unit, "single": None, "bar": False,
                     "small": max(map(abs, ys)) < 1e-2, "help": helptext + f" x axis: optimizer step within update U{last}; one colour per pass, grey dotted = the previous update.",
                     "traces": traces, "rng": [0, max(ys) * 1.15] if min(ys) >= 0 and max(ys) > 0 else None, "legend": True, "y2": False, "xlab": "step in update"})
    return figs


# reads inside a round, from play_progress rows' per-set running sums (play.<set>.{tries,clears,tok_sum,
# act_sum,rhae_sum,planned}). Short tries finish first, so the current round is drawn against earlier rounds at the
# same share of tries done.
LIVE_METRICS = [("RHAE per 10K tokens", lambda s: s["rhae_sum"] / s["tok_sum"] * 1e4 if s["tok_sum"] else None, "Above 1 = better than that round."),
                ("Tokens per try", lambda s: s["tok_sum"] / s["tries"] if s["tries"] else None, "Below 1 = cheaper tries than that round."),
                ("Actions per clear (mean)", lambda s: s["act_sum"] / s["clears"] if s["clears"] else None, "Below 1 = fewer actions per clear than that round.")]


def live_figs(rows):
    rounds = {}
    for r in rows:
        m = r.get("m") or {}
        if not isinstance(m.get("play.round"), (int, float)):
            continue
        sets = {}
        for k, v in m.items():
            if k.startswith("play.") and k.count(".") >= 2 and isinstance(v, (int, float)):
                name, f = k[5:].rsplit(".", 1)
                sets.setdefault(name, {})[f] = v
        if sets:
            rounds.setdefault(int(m["play.round"]), []).append((r["t"], m.get("play.played_by", m.get("train.played_by")), sets))
    if not rounds:
        return []
    lv = lambda n: (re.search(r"[Ll](\d+)", n) or [None, None])[1]
    any_lv = any(lv(n) for v in rounds.values() for _, _, s in v for n in s)

    def pooled(sets, held):
        g = {f: 0.0 for f in ("tries", "clears", "tok_sum", "act_sum", "rhae_sum", "planned")}
        for n, s in sets.items():
            if bool(re.search(r"fixed|held", n, re.I)) == held and (not any_lv or lv(n) in ("1", "3")):
                for f in g:
                    g[f] += s.get(f) or 0
        return g if g["planned"] else None

    order = sorted(rounds)
    cur, prev = order[-1], order[-3:-1]
    lvs = sorted({lv(n) for v in rounds.values() for _, _, s in v for n in s if lv(n) in ("1", "3")})
    levels = "+".join(f"L{x}" for x in lvs) if lvs else "all levels"
    ck = lambda pb: f"r{int(pb):02d}" if isinstance(pb, (int, float)) else "?"
    figs = []

    def curve(rd, held, fn):
        pts, pb = [], None
        for _, pb_, sets in sorted(rounds[rd], key=lambda x: x[0]):
            g = pooled(sets, held)
            pb = pb_ if pb_ is not None else pb
            if g and g["tries"]:
                y = fn(g)
                if y is not None:  # a resume carries tries over: the share never passes 100%
                    pts.append((100 * g["tries"] / max(g["planned"], g["tries"]), y, int(g["tries"])))
        return sorted(pts), pb  # sorted per series: no line doubles back

    def at(pts, x):  # an earlier round's running value at the same share of tries done (linear between its points)
        if not pts or x < pts[0][0] or x > pts[-1][0]:
            return None
        for (x0, y0, _), (x1, y1, _) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return pts[-1][1]

    # running values fall as a round fills (short tries finish first), which reads as worsening. Plot the
    # current round as a ratio to each earlier round at the same share of tries done: finish order cancels out.
    for title, fn, better in LIVE_METRICS:
        traces = []
        for held in (False, True):
            cp, _ = curve(cur, held, fn)
            for i, p_ in enumerate(reversed(prev)):
                pp, ppb = curve(p_, held, fn)
                pts = [(round(x, 1), y / b, n) for x, y, n in cp if (b := at(pp, x))]
                if not pts:
                    continue
                nm = f"{'held-out' if held else 'training'} R{cur} ÷ R{p_}" + (f" ({ck(ppb)})" if ck(ppb) != "?" else "")
                col = (COLORS[0] if held else COLORS[1]) if i == 0 else ("#94a3b8" if held else "#fdba74")
                traces.append({"x": [a for a, _, _ in pts], "y": [b for _, b, _ in pts], "text": [c for _, _, c in pts], "name": nm,
                               "type": "scatter", "mode": "lines+markers", "line": {"width": 2.2 if i == 0 else 1.5, "color": col, "dash": "solid" if i == 0 else "dot"},
                               "marker": {"size": 5, "color": col}, "hovertemplate": nm + ": x%{y:.3f} after %{text} tries<extra></extra>"})
        if not traces:
            continue
        xs = [0, 100]
        traces.append({"x": xs, "y": [1, 1], "name": "same as earlier round", "type": "scatter", "mode": "lines",
                       "line": {"width": 1, "color": "#64748b", "dash": "dash"}, "hoverinfo": "skip"})
        cur_ck = next((ck(pb_) for _, pb_, _ in sorted(rounds[cur], key=lambda x: x[0]) if pb_ is not None), "?")
        ys = [y for t in traces for y in t["y"]]
        model = f" (model {cur_ck})" if cur_ck != "?" else ""
        figs.append({"section": "This round, live", "title": f"{title} · round {cur}{model} vs earlier rounds, at the same point", "unit": "x", "single": None, "bar": False, "small": False,
                     "help": f"This round's running value ({levels}) divided by an earlier round's at the same share of tries done, so finish order "
                             f"(short tries first) cancels out. {better} The last point is the round's final ratio so far.",
                     "traces": traces, "rng": [min(0.5, min(ys) * 0.9), max(1.5, max(ys) * 1.1)] if ys else None, "legend": True,
                     "y2": False, "xlab": "% of the round's tries done", "dtick": 10, "xr": [0, 100]})
    return figs


def _smooth(pts, win_s):
    """Centered moving mean over win_s seconds; pts sorted (t, v)."""
    out, j0, j1, tot = [], 0, 0, 0.0
    for t, _ in pts:
        while j1 < len(pts) and pts[j1][0] <= t + win_s / 2:
            tot += pts[j1][1]
            j1 += 1
        while pts[j0][0] < t - win_s / 2:
            tot -= pts[j0][1]
            j0 += 1
        out.append((t, tot / (j1 - j0)))
    return out


def _busy(m):
    """A sample of real work: a server running requests, or training tokens."""
    return any(isinstance(v, (int, float)) and v > 0 for k, v in m.items()
               if k.startswith("infra.running") or k in ("train.live_tok_s", "train.tok_s_live", "infra.train_tok_s"))


def mini_figs(rows, t0, cap):
    """small, heavily smoothed side plots since the latest start (current leg): throughput, GPU util + VRAM
    (mean over cards), KV cache use, round time split."""
    rows = [r for r in rows if r.get("t", 0) >= (t0 or 0) and r.get("m")]
    # 28 Oct: from the first productive sample (serving requests or training), not the boot ramp, and play keys only
    # while that server runs requests (updates leave gaps), so the side plots read the same series as Key plots
    first = next((i for i, r in enumerate(rows) if _busy(r["m"])), None)
    if first is None:
        return []
    rows = rows[first:]
    T0 = rows[0]["t"]
    runs_ = sorted(r["m"]["infra.running"] for r in rows if (r["m"].get("infra.running") or 0) > 0)
    floor = runs_[len(runs_) // 2] / 2 if runs_ else 0  # no ramp or drain at the edges of an update
    playing = lambda m: not any(k.startswith("infra.running") for k in m) or (m.get("infra.running") or 0) > floor

    def series(fn):
        pts = sorted((r["t"], v) for r in rows for v in [fn(r["m"])] if isinstance(v, (int, float)))
        return pts

    def mean_of(pre, scale=1.0):
        return lambda m: (sum(v for k, v in m.items() if k.startswith(pre) and isinstance(v, (int, float))) /
                          max(1, sum(1 for k, v in m.items() if k.startswith(pre) and isinstance(v, (int, float)))) * scale
                          if any(k.startswith(pre) for k in m) else None)

    def line(pts, name, color, win=600, ramp=False):
        segs = []
        for t, v in pts:  # segments split at gaps over 5 min (update phases), each smoothed on its own
            if not segs or t - segs[-1][-1][0] > 300:
                segs.append([])
            segs[-1].append((t, v))
        if ramp:  # play keys: a phase's first 4 min is prefill ramp, not throughput
            segs = [[p for p in sg if p[0] - sg[0][0] >= 240] for sg in segs]
            segs = [sg for sg in segs if sg]
        sm = [p for sg in segs for p in (_smooth(sg, win) if len(sg) > 3 else sg)]
        xs, ys = [], []
        for i, (t, v) in enumerate(sm):
            if xs and t - sm[i - 1][0] > 300:
                xs.append(None)
                ys.append(None)
            xs.append(round((t - T0) / 3600, 3))
            ys.append(round(v, 2))
        return {"x": xs, "y": ys, "name": name, "type": "scatter", "mode": "lines", "line": {"width": 2, "color": color},
                "hovertemplate": name + " %{y:.3g}<extra></extra>"}

    figs = []
    gen = series(lambda m: m.get("infra.gen_tok_s") if playing(m) else None)
    trn = (series(lambda m: m.get("train.live_tok_s")) or series(lambda m: m.get("train.tok_s_live"))
           or series(lambda m: m.get("infra.train_tok_s")))
    tr = [line(p, n, c, ramp=n == "play") for p, n, c in ((gen, "play", COLORS[0]), (trn, "train", COLORS[1])) if p]
    if tr:
        figs.append({"title": "Throughput (tok/s)", "traces": tr})
    util = series(mean_of("infra.gpu_util."))
    vram = series(mean_of("infra.vram_gib.", 100.0 / cap))
    tr = [line(p, n, c) for p, n, c in ((util, "util %", COLORS[2]), (vram, "VRAM %", COLORS[4])) if p]
    if tr:
        figs.append({"title": "GPU util + VRAM (mean over cards, %)", "traces": tr, "rng": [0, 105]})
    kv = series(lambda m: m["infra.kv_frac"] * 100 if isinstance(m.get("infra.kv_frac"), (int, float)) and playing(m) else None)
    if kv:
        figs.append({"title": "KV cache use (%)", "traces": [line(kv, "KV %", COLORS[5], ramp=True)], "rng": [0, 105]})
    rr = [r for r in rows if isinstance(r["m"].get("train.round"), (int, float)) and any(k in r["m"] for k in PHASE_COLORS)]
    seen, rounds = set(), []
    for r in rr:
        if r["m"]["train.round"] not in seen:
            seen.add(r["m"]["train.round"])
            rounds.append(r["m"])
    if rounds:
        tr = []
        for k, c in PHASE_COLORS.items():
            ys = [round((m.get(k) or 0) / 60, 1) for m in rounds]
            if any(ys):
                tr.append({"x": [f"R{int(m['train.round'])}" for m in rounds], "y": ys, "name": k[5:-2].replace("_", " "), "type": "bar",
                           "marker": {"color": c}, "hovertemplate": "%{x} " + k[5:-2].replace("_", " ") + ": %{y:.0f} min<extra></extra>"})
        figs.append({"title": "Round time split (min)", "traces": tr, "bar": True})
    return figs


RL_DROP = {"Data age", "Tokens per step", "Play progress", "Reference coverage", "Advantage signal", "Progress", "Loss",
           "Update size", "Update progress", "KL sensitivity", "Drift (KL)", "Useful groups", "Advantage size", "Reward", "Round time split", "Per-turn overrides"}


SFT_HELDOUT = [  # (title, unit, [(key, name)], help): SFT held-out evals vs training step, slice and full, step-0 base dotted
    ("Held-out reply KL", "nats", [("eval.heldout_reply_kl_slice", "slice (56)"), ("eval.heldout_reply_kl", "full (155)")],
     "KL from the teacher on held-out games' replies (lower is better): the 56-example slice every 24 steps and the full 155 at the start, middle and end. Dotted = the same read at step 0."),
    ("ARC-2 held-out KL", "nats", [("eval.arc2_heldout_kl", "20 held-out tasks (~56)")],
     "KL from the teacher on the fixed slice of the 20 ARC-2 tasks no student trains on (lower is better), read at the start, middle and end of a run. Dotted = the run's first read."),
    ("Held-out loss", "", [("eval.heldout_loss_slice", "slice (56)"), ("eval.heldout_loss", "full (155)")],
     "Held-out training objective (KL + draft-head KL; lower is better), slice and full. Dotted = step 0."),
    ("Draft head KL, held-out", "nats", [("eval.heldout_mtp_kl_slice", "slice (56)"), ("eval.heldout_mtp_kl", "full (155)")],
     "The draft head's KL to the trunk on held-out replies (lower is better), slice and full. Dotted = step 0."),
    ("Draft acceptance, held-out", "", [("eval.accept1_slice", "slice (56)"), ("eval.accept1", "full (155)")],
     "The draft head's first-token acceptance on held-out replies (higher is better), slice and full. Dotted = step 0."),
]


def sft_heldout_figs(rows):
    """SFT held-out evals as lines vs training step (eval rows carry train.step), each with its step-0 base as a dotted line."""
    pts = {}
    for r in rows:
        m = r.get("m") or {}
        if isinstance(m.get("train.step"), (int, float)):
            for k, v in m.items():
                if k.startswith("eval.") and isinstance(v, (int, float)):
                    pts.setdefault(k, {})[int(m["train.step"])] = v
    figs = []
    for title, unit, keys, helptext in SFT_HELDOUT:
        have = [(k, nm) for k, nm in keys if k in pts]
        if not have:
            continue
        end = max(s for k, _ in have for s in pts[k])
        start = min(s for k, _ in have for s in pts[k])  # a continued run's first read, not step 0
        traces = []
        for j, (k, nm) in enumerate(have):
            c = COLORS[j % len(COLORS)]
            xs = sorted(pts[k])
            traces.append({"x": xs, "y": [pts[k][s] for s in xs], "name": nm, "type": "scatter",
                           "mode": "lines+markers", "line": {"width": 2.2, "color": c}, "marker": {"size": 6, "color": c},
                           "hovertemplate": "step %{x}: %{y:.4g}<extra></extra>"})
            bases = pts.get(k + "_base", {})
            b = bases.get(0, bases[min(bases)] if bases else pts[k].get(0))
            if b is not None:
                traces.append({"x": [start, max(end, start + 1)], "y": [b, b], "name": nm + " step 0", "type": "scatter", "mode": "lines", "showlegend": False,
                               "line": {"width": 1.5, "color": c, "dash": "dot"}, "hoverinfo": "skip"})
        figs.append({"section": "Key plots", "title": title, "unit": unit, "help": helptext, "traces": traces, "single": None, "bar": False,
                     "small": False, "rng": None, "legend": True, "y2": False,  # autorange: a few % must show
                     "xlab": "training step", "dtick": 24})
    return figs


def leg_family(run):
    """A restart of one training leg keeps the id with a letter added (x-v1, x-v1b): the same family."""
    return re.sub(r"(v\d+)[a-z]?$", r"\1", run)


def leg_kd_fig(rows, run, own_start, same=()):
    """SFT, this leg only: training KD from the leg's first step, restarts of the leg (same family) included, no
    earlier legs, y axis on this leg alone so the slope reads; raw steps faint plus the Loss panel's centred step mean. A redone
    step keeps its latest value; a dotted line marks each restart."""
    pts, first = {}, {}
    fam = leg_family(run)
    for r in rows:
        m = r.get("m") or {}
        v = m.get("train.kd", m.get("train.distill_kl"))
        leg = r.get("_leg")
        if isinstance(m.get("train.step"), (int, float)) and isinstance(v, (int, float)) and v == v \
                and (leg is not None and leg != run and (leg_family(leg) == fam or leg in same) or r.get("t", 0) >= (own_start or 0) and leg in (None, run)):
            pts[int(m["train.step"])] = v
            first.setdefault(leg, int(m["train.step"]))
    if not pts:
        return None
    xs = sorted(pts)
    ys = [pts[x] for x in xs]
    n = len(ys)
    faint = n >= 5
    traces = [{"x": xs, "y": ys, "name": "raw steps", "type": "scatter", "mode": "lines" if n > 1 else "markers", "opacity": 0.3 if faint else 1,
               "line": {"width": 1, "color": COLORS[0]}, "marker": {"size": 8, "color": COLORS[0]}, "hovertemplate": "step %{x}: %{y:.4g}<extra>raw</extra>"}]
    if faint:
        w = max(5, min(25, n // 15))
        mean = [sum(ys[max(0, i - w // 2):i + w // 2 + 1]) / len(ys[max(0, i - w // 2):i + w // 2 + 1]) for i in range(n)]
        nm = f"{w // 2 * 2 + 1}-step mean"
        traces.append({"x": xs, "y": mean, "name": nm, "type": "scatter", "mode": "lines", "line": {"width": 2.2, "color": COLORS[0]},
                       "hovertemplate": "step %{x}: %{y:.4g}<extra>" + nm + "</extra>"})
    for leg, step in first.items():
        if step > xs[0]:
            traces.append({"x": [step, step], "y": [min(ys), max(ys)], "name": f"restart {leg}", "type": "scatter", "mode": "lines",
                           "line": {"width": 1.5, "color": COLORS[1], "dash": "dot"}, "hovertemplate": f"{leg} from step {step}<extra></extra>"})
    span = xs[-1] - xs[0]
    return {"section": "Key plots", "title": "Training KD · this leg", "unit": "nats", "single": None, "bar": False, "small": False, "rng": None,
            "legend": True, "y2": False, "xlab": "training step", "dtick": next((d for d in (1, 2, 5, 10, 20, 50, 100, 200, 500) if span / d <= 8), 1000),
            "xr": [xs[0], max(xs[-1], xs[0] + 1)], "traces": traces,
            "help": f"Training KD (train.kd) on this leg's steps, from its first step (s{xs[0]}), restarts of the leg included (dotted lines); no "
                    "earlier legs, y axis scaled to this leg so the slope reads. Raw steps faint, centred step mean solid. The other panels keep "
                    "the whole lineage."}


def prefix_fig(rows, clock):
    """RL prefix-cache hit rate: per server (infra.prefix_hit.s<port>, else the total), 10-min mean over play
    samples only (a sample counts when that server runs requests: updates leave gaps, not zeros), dotted 95% target."""
    keys = sorted({k for r in rows for k in (r.get("m") or {}) if k == "infra.prefix_hit" or k.startswith("infra.prefix_hit.s")})
    if any(k != "infra.prefix_hit" for k in keys):
        keys = [k for k in keys if k != "infra.prefix_hit"]
    traces, end = [], 0.0
    for j, k in enumerate(keys):
        port = k[len("infra.prefix_hit"):]  # "" or ".s1234"
        pts, run = [], None
        for r in rows:  # running carried forward: a backfilled hit row lands milliseconds off its sampler row
            m = r.get("m") or {}
            run = m.get("infra.running" + port, m.get("infra.running", run))
            v = m.get(k)
            if isinstance(v, (int, float)) and isinstance(run, (int, float)) and run > 0 and r["t"] in clock:
                pts.append((clock[r["t"]], 100 * v))
        xs, ys = [], []
        for i, (x, _) in enumerate(pts):
            if xs and x - xs[-1] > 1 / 3:  # no play for 20 min: break the line
                xs.append((xs[-1] + x) / 2)
                ys.append(None)
            win = [y for xx, y in pts[:i + 1] if x - xx <= 1 / 6]
            xs.append(x)
            ys.append(sum(win) / len(win))
        if xs:
            end = max(end, xs[-1])
            c = COLORS[j % len(COLORS)]
            traces.append({"x": xs, "y": ys, "name": ("server " + port[2:]) if port else "all servers", "type": "scatter",
                           "mode": "lines", "line": {"width": 2, "color": c}, "connectgaps": False,
                           "hovertemplate": "%{y:.1f}%<extra></extra>"})
    if not traces:
        return None
    traces.append({"x": [0, max(end, 0.01)], "y": [95, 95], "name": "target 95%", "type": "scatter", "mode": "lines",
                   "line": {"width": 1.5, "color": "#64748b", "dash": "dot"}, "hoverinfo": "skip"})
    return {"section": "Key plots", "title": "Prefix hit rate", "unit": "%", "traces": traces, "single": None, "bar": False,
            "small": False, "rng": None, "legend": True, "y2": False,
            "help": "Share of prompt tokens served from the prefix cache, per play server, 10-min mean over play only (updates "
                    "leave gaps). Dotted line = the 95% target (v87b recipe: mean 94-95%); below it, re-sent context is recomputed."}


PHASE_GROUPS = 16  # groups per play phase


def play_progress_fig(rows, clock):
    """RL play-phase progress: groups finished in the current play phase (export_rl's
    play.live_groups_done, counted from the feed's group_done rows and reset at each update) vs run time, out of 16, with
    each update (train.step: the phase boundary) marked."""
    pts, steps = [], []
    for r in rows:
        m = r.get("m") or {}
        if r.get("t") not in clock:
            continue
        x = clock[r["t"]]
        if isinstance(m.get("train.step"), (int, float)):
            steps.append((x, int(m["train.step"])))
            pts.append((x, None))  # the count restarts after the update
            pts.append((x, 0))
        v = m.get("play.live_groups_done")
        if isinstance(v, (int, float)):
            pts.append((x, v))
    if not any(y for _, y in pts):
        return None
    end = max(x for x, _ in pts)
    xs, ys = [x for x, _ in pts] + [end], [y for _, y in pts] + [pts[-1][1]]
    traces = [{"x": xs, "y": ys, "name": "groups done this phase", "type": "scatter", "mode": "lines+markers",
               "line": {"width": 2, "color": COLORS[0], "shape": "hv"}, "marker": {"size": 4}, "connectgaps": False,
               "hovertemplate": "%{y:.0f} of " + str(PHASE_GROUPS) + " groups<extra></extra>"},
              {"x": [0, max(end, 0.01)], "y": [PHASE_GROUPS, PHASE_GROUPS], "name": f"phase target {PHASE_GROUPS}",
               "type": "scatter", "mode": "lines", "line": {"width": 1.5, "color": "#64748b", "dash": "dot"},
               "hoverinfo": "skip"}]
    for i, (x, n) in enumerate(steps):
        traces.append({"x": [x, x], "y": [0, PHASE_GROUPS], "name": "update (phase end)", "type": "scatter",
                       "mode": "lines", "line": {"width": 1, "color": "#94a3b8", "dash": "dash"}, "showlegend": i == 0,
                       "legendgroup": "steps", "hovertemplate": f"update step {n}<extra></extra>"})
    return {"section": "Key plots", "title": "Play phase progress", "unit": "groups", "traces": traces, "single": None,
            "bar": False, "small": False, "rng": [0, PHASE_GROUPS + 1], "legend": True, "y2": False,
            "help": f"Groups finished in the current play phase, out of {PHASE_GROUPS} (from the feed's group_done rows); "
                    "the count restarts at each update (dashed line = the update that ends the phase)."}


# pre-registered bars as reference lines, breaching points ringed red (values as plotted: fractions x100)
RL_BARS = {"train.fail_rate_vs_ref.all": (2.0, lambda x, y: y > 2.0), "train.groups_dropped_frac_step": (10, lambda x, y: y > 10),
           "train.grad_norm": (0.40, lambda x, y: y > 0.40), "train.tis_out_share": (5, lambda x, y: y > 5),
           "train.top_game_share": (40, lambda x, y: y > 40)}
RL_REFS = {"train.grad_norm": [(0.40, "bar 0.40", "dash")], "train.tis_out_share": [(5, "bar 5%", "dash")],
           "train.top_game_share": [(40, "40%: watch the mix", "dot"), (50, "50%: mix-dominated", "dash")], "train.vs_control": [(1.0, "control = 1.0")],
           "train.fail_rate_vs_ref.all": [(2.0, "rollback bar 2.0", "dash")]}
# series always drawn in their own panel (no dual axes)
SPLIT = {"time.checkpoint_s": "Checkpoint save time", "train.truncation_frac": "Truncated replies", "train.tis_out_share": "Tokens outside the TIS cap"}


def split_scales(title, keys, vals):
    """No dual axes: keys in SPLIT get their own panel; series under a tenth of the panel's largest go to a
    second panel. Returns [(title, keys)]."""
    own = [k for k in keys if k in SPLIT and len(keys) > 1]
    rest = [k for k in keys if k not in own]
    top = {k: max(map(abs, [v for v in vals.get(k, []) if isinstance(v, (int, float))]), default=0) for k in rest}
    big = max(top.values(), default=0)
    small = [k for k in rest if big > 0 and top[k] < big / 10 and not k.endswith(("_target", "_stop", "_cap_gib", "_lo", "_hi"))]
    main = [k for k in rest if k not in small]
    out = [(title, main)] if main else []
    if small:
        out.append((title + " · " + ", ".join(k.rsplit(".", 1)[-1].replace("_", " ") for k in small), small))
    return out + [(SPLIT[k], [k]) for k in own]


def heldout_fig(reads, cur):
    """Held-out RHAE/tok vs the in-run base per read step, 95% CI as error bars, 1.0 = base, 2.0 = target."""
    if not reads:
        return None
    xs = [r["step"] for r in reads]
    ys = [r["ratio"] for r in reads]
    txt = [f"s{r['step']:04d}: {r['ratio']:.3f}" + (f" [{r['lo']:.2f}, {r['hi']:.2f}]" if r["lo"] else "") + (f" · {r['verdict']}" if r["verdict"] else "") for r in reads]
    tr = {"x": xs, "y": ys, "text": txt, "name": "policy ÷ base, 95% CI", "type": "scatter", "mode": "markers",
          "marker": {"size": 9, "color": COLORS[0]}, "hovertemplate": "%{text}<extra></extra>",
          "error_y": {"type": "data", "symmetric": False, "array": [(r["hi"] or r["ratio"]) - r["ratio"] for r in reads],
                      "arrayminus": [r["ratio"] - (r["lo"] or r["ratio"]) for r in reads], "color": COLORS[0], "thickness": 1.5, "width": 6}}
    span = [min(xs + [cur]) - 1, max(xs + [cur]) + 1]
    hi = max([r["hi"] or r["ratio"] for r in reads] + [2.0])
    return {"section": "Key plots", "title": "Held-out RHAE/tok vs base", "unit": "x", "upd": True, "single": None, "bar": False, "small": False,
            "traces": [tr, bar_line(span, 1.0, "base = 1.0"), bar_line(span, 2.0, "target 2x", "#16a34a", "dash")],
            "rng": [0, hi * 1.12], "legend": True, "y2": False, "xr": span,
            "help": "RL's target read: held-out RHAE per token of the checkpoint over the in-run base on the same tries, with its 95% interval "
                    "(log scale, from the read's and the base's standard errors). Target 2x. x axis: the checkpoint's update number."}


STAT_KEYS = [("RHAE/tok vs ref", "train.rhae_tok_vs_ref.all", "{:.2f}x"), ("fail ratio", "train.fail_rate_vs_ref.all", "{:.2f}x"),
             ("entropy", "train.entropy", "{:.3f}"), ("tokens/try vs ref", "train.tokens_per_try_vs_ref.all", "{:.2f}x"),
             ("groups dropped", "train.groups_dropped_frac_step", "{:.1%}"), ("grad norm", "train.grad_norm", "{:.3f}")]


def rl_stats(meta, bands, uobs, reads, starts=()):
    """Stat row above Key plots: attempt, update, the last update's values with deltas, next held-out read ETA."""
    out = []
    if bands:
        b = bands[-1]
        out.append(["attempt", f"a{b['n']}", f"{b['state']} since {paths.local(b['t0']):%H:%M}", ""])
    live = {k: split_branches(v, starts)[0] for k, v in uobs.items()}
    ups = live.get("train.rhae_tok_vs_ref.all") or max(live.values(), key=len, default=[])
    if not ups:
        return out
    cur, t_cur = ups[-1][0], ups[-1][2]
    out.append(["update", f"U{cur}", f"done {paths.local(t_cur):%H:%M}", ""])
    sh = [x for x in live.get("train.top_game_share") or [] if x[0] == cur]
    gm = [x for x in live.get("train.top_game") or [] if x[0] == cur]
    share = sh[0][1] if sh else None
    if share is not None:  # one game filling the update makes RHAE/tok and tokens/try mix reads
        out.append(["top game share", f"{share:.0%}" + (f" {gm[0][1]}" if gm else ""),
                    "over 40%: mix-heavy" if share > 0.4 else "largest game's share of groups", "warn" if share > 0.4 else ""])
    for lab, k, fmt in STAT_KEYS:
        p = live.get(k) or []
        if p and p[-1][0] == cur:
            v = p[-1][1]
            prev = p[-2][1] if len(p) > 1 and p[-2][0] == cur - 1 else None
            d = "" if prev is None else (f"{100 * (v - prev):+.1f} pp vs U{cur - 1}" if "%" in fmt else
                                         f"{100 * (v - prev) / abs(prev):+.0f}% vs U{cur - 1}" if prev else f"{v - prev:+.3g} vs U{cur - 1}")
            mixed = share is not None and share > 0.5 and k in ("train.rhae_tok_vs_ref.all", "train.tokens_per_try_vs_ref.all")
            out.append([lab, fmt.format(v), ("mix-dominated · " if mixed else "") + (d or "no previous update"), "warn" if mixed else ""])
    every = int((re.search(r"every (\d+) steps", meta.get("what") or "") or [0, 8])[1])
    nxt = cur if cur % every == 0 and not any(r["step"] == cur for r in reads) else (cur // every + 1) * every
    out = [x for x in out if x[0] != "top game share"] + [x for x in out if x[0] == "top game share"]  # after the values
    durs = sorted(b_[2] - a_[2] for a_, b_ in zip(ups[-6:], ups[-5:]) if b_[0] == a_[0] + 1)
    if durs:
        eta = t_cur + (nxt - cur) * durs[len(durs) // 2]
        out.append(["next held-out read", f"s{nxt:04d}", f"checkpoint ~{paths.local(eta):%H:%M}" if nxt > cur else "checkpoint saved, read pending", ""])
    return out


PLAY_ONLY = ("infra.gen_tok_s", "infra.running", "infra.waiting", "infra.kv_frac")  # zero while an update runs: not a drop


def update_phases(rows, clock):
    """Wall-time intervals (run hours) with play stopped (infra.running 0): an update runs. Shaded grey on infra panels."""
    out, start, prev = [], None, None
    for r in rows:
        v = (r.get("m") or {}).get("infra.running")
        if not isinstance(v, (int, float)) or r["t"] not in clock:
            continue
        if v == 0 and start is None:
            start = r["t"]
        elif v > 0 and start is not None:
            if r["t"] - start >= 300:
                out.append([clock[start], clock[r["t"]]])
            start = None
        prev = r["t"]
    if start is not None and prev and prev - start >= 300:
        out.append([clock[start], clock[prev]])
    return out


def run_view(meta, rows, last):
    clock = run_clock(rows, meta.get("_start"))
    leg0 = meta.get("_own_start")  # the run id's first row: the attempt override below narrows _own_start to an attempt
    # play-only series skip update phases (shaded) instead of a comb of drops to 0
    rows_w = [dict(r, m={k: v for k, v in r["m"].items() if not (k in PLAY_ONLY and r["m"].get("infra.running") == 0)})
              if (r.get("m") or {}).get("infra.running") == 0 else r for r in rows]
    ser = series(rows_w, clock)
    stages = [[round(clock[t], 3), lab] for t, lab in sorted(list(meta.get("_stages", [])) + list(meta.get("_attempt_stages", []))) if t in clock]
    t_att = meta.get("_attempt_start")
    if t_att and t_att > (meta.get("_own_start") or 0):
        meta = dict(meta, _own_start=t_att)
    status = meta.get("status", "live")
    if status == "live" and t_att and not any(r.get("t", 0) >= t_att and _busy(r.get("m") or {}) for r in rows):
        status = "booting"  # the current attempt has not played or trained yet
    own0 = meta.get("_own_start") or meta.get("_start")  # earlier legs' alerts go to the history line, not the live alert box
    roundat = {round(clock[r["t"]], 4): r["m"]["train.round"] for r in rows if isinstance((r.get("m") or {}).get("train.round"), (int, float))}
    if not ser:
        note = next((n for pre, n in NO_TRAIN_METRICS if meta["run"].startswith(pre)), None) or (
            "ended before any metrics" if meta.get("status") == "ended" else "waiting for its first metrics (container starting or queued for a GPU)")
        meta = dict(meta, what=" · ".join(filter(None, [meta.get("what"), note])))
    claimed, figs = set(), []
    is_rl = (meta.get("arm") or "").lower() == "rl" or meta["run"].startswith("rl")
    asyn = any("train.live_groups_total" in (r.get("m") or {}) for r in rows[-50:])  # async loop: no rounds to compare
    uobs, bstarts = update_obs(rows) if is_rl else ({}, [])
    live_control(uobs, bstarts)
    bands, marks = run_events(meta, rows, bstarts) if is_rl else ([], [])

    def nm_of(k, keys):
        pre, suf = k.split(".", 1) if "." in k else ("", k)
        multi = len({kk.split(".", 1)[0] for kk in keys}) > 1
        nm = PREFIX_NAMES.get(pre, pre) if multi and PREFIX_NAMES.get(pre) else (
            suf.rsplit(".", 1)[-1] if re.search(r"\.(L\d+|d\d+p?|all)$", suf) else suf.replace("_", " ") + (" %" if is_frac(k) else ""))
        return NAMES.get(k, nm)

    def upd_fig(title, keys, unit, section, helptext, names=None, refs=(), bars=None, band=None):
        """RL per-update panel: x = update number; the live branch as a line, a rolled-back branch as a
        faded 'abandoned' series never joined to it; refs = [(y, name)] reference lines; bars = {key: (bar, test)} marks
        the breaching points; band = key: dotted line at -20% of the previous update (entropy)."""
        traces, xs_all, gone_shown = [], [], False
        for j, k in enumerate(keys):
            c = COLORS[j % len(COLORS)]
            live, gone = split_branches(uobs[k], bstarts)
            f = 100 if is_frac(k) else 1
            nm = (names or {}).get(k) or nm_of(k, keys)
            xs, ys = [u for u, _, _ in live], [v * f for _, v, _ in live]
            xs_all += xs
            traces.append({"x": xs, "y": ys, "text": [f"{paths.local(t):%d %b %H:%M}" for _, _, t in live], "name": nm, "type": "scatter",
                           "mode": "lines+markers", "line": {"width": 2, "color": c}, "marker": {"size": 6, "color": c},
                           "hovertemplate": "U%{x} · %{text}: %{y:.4g}<extra>" + nm + "</extra>"})
            for b, p_ in sorted(gone.items()):
                rb = bstarts[b + 1][0] if b + 1 < len(bstarts) else None
                lab = f"abandoned (rolled back {paths.local(rollback_t(bands, rb)):%H:%M} {paths.tz()})" if rb else "abandoned branch"
                traces.append({"x": [u for u, _, _ in p_], "y": [v * f for _, v, _ in p_], "text": [f"{paths.local(t):%d %b %H:%M}" for _, _, t in p_],
                               "name": lab, "type": "scatter", "mode": "lines+markers", "opacity": 0.35,
                               "showlegend": not gone_shown, "line": {"width": 1.5, "color": c, "dash": "dot"}, "marker": {"size": 5, "color": c},
                               "hovertemplate": "U%{x} · %{text}: %{y:.4g}<extra>" + nm + ", " + lab + "</extra>"})
                gone_shown = True
                xs_all += [u for u, _, _ in p_]
            if bars and k in bars:
                b_, test = bars[k]
                tr = breaches(list(zip(xs, ys)), test)
                if tr:
                    traces.append(tr)
            if band == k and len(ys) > 1:
                lo = [None] + [y0 * 0.8 for y0 in ys[:-1]]
                traces.append({"x": xs, "y": lo, "name": "-20% of the previous update", "type": "scatter", "mode": "lines",
                               "line": {"width": 1.2, "color": "#94a3b8", "dash": "dot", "shape": "hvh"}, "hoverinfo": "skip"})
                tr = breaches([(x, y) for x, y, l in zip(xs, ys, lo) if l is not None and y < l], lambda x, y: True, "fell over 20%")
                if tr:
                    traces.append(tr)
        if not xs_all:
            return None
        for y, nm, *dash in refs:
            traces.append(bar_line(xs_all, y, nm, dash=(dash or ["dot"])[0]))
        ys = [y for t in traces for y in t["y"] if isinstance(y, (int, float))]
        rng = [0, max(ys) * 1.15] if ys and min(ys) >= 0 and max(ys) > 0 else None
        return {"section": section, "title": title, "unit": unit, "help": helptext + " x axis: update number (live branch solid; a rolled-back branch faded).",
                "traces": traces, "single": None, "bar": False, "small": max(map(abs, ys), default=1) < 1e-2, "rng": rng,
                "legend": len(traces) > 1, "y2": False, "upd": True}
    if is_rl:  # L1+L3 and pass/coverage details: superseded by the all-level and reference panels
        claimed.update(k for k in ser if fnmatch.fnmatch(k, "train.*.L13") or fnmatch.fnmatch(k, "train.*cov*") or k.startswith("play.live_")
                       or k in ("train.tries", "train.live_tokens", "train.live_reward_std", "train.live_w_abs_median", "train.live_gigpo_step_abs",
                                "infra.gpu_util.trainer_mean", "infra.peak_alloc_gib", "infra.card2_mode"))
    claimed.add("infra.alloc_after_load_gib")
    rlkey = []
    if is_rl and uobs:  # decision metrics first, each with its pre-registered bar
        hf = heldout_fig(heldout_reads(meta), max((u for v in uobs.values() for _, u in v), default=0))
        if hf:
            rlkey.append(hf)
            claimed.update(k for k in ser if k.startswith("eval.heldout_ratio"))
        tpt = "train.tokens_per_try_vs_ref.all" if "train.tokens_per_try_vs_ref.all" in uobs else "train.tokens_per_try.all"
        for title, keys, unit, helptext, kw in [
                ("RHAE/tok vs reference, by depth", ["train.rhae_tok_vs_ref.all", "train.rhae_tok_vs_ref.d1", "train.rhae_tok_vs_ref.d23", "train.rhae_tok_vs_ref.d4p"], "x",
                 "Training tries' RHAE per token against each level's base reference (geometric mean), all levels and by depth bucket (d1 = level 1, d23 = levels 2-3, d4p = level 4 on). 1.0 = the reference.",
                 {"names": {"train.rhae_tok_vs_ref.all": "all levels", "train.rhae_tok_vs_ref.d1": "depth 1", "train.rhae_tok_vs_ref.d23": "depth 2-3", "train.rhae_tok_vs_ref.d4p": "depth 4+"},
                  "refs": [(1.0, "reference = 1.0")]}),
                ("Fail ratio vs reference", ["train.fail_rate_vs_ref.all"], "x",
                 "Share of tries that fail their level, over the reference's fail share on the same levels. The pre-registered rollback bar is 2.0 (red ring = over it).",
                 {"names": {"train.fail_rate_vs_ref.all": "fail ratio"}, "refs": [(2.0, "rollback bar 2.0", "dash")], "bars": {"train.fail_rate_vs_ref.all": RL_BARS["train.fail_rate_vs_ref.all"]}}),
                ("Entropy", ["train.entropy"], "nats",
                 "Entropy of the policy's replies per update. Dotted step line = 20% below the previous update (the pre-registered drop bar; red ring = fell more than 20% in one update).",
                 {"names": {"train.entropy": "entropy"}, "band": "train.entropy"}),
                ("Tokens per try" + (" vs reference" if tpt.endswith("vs_ref.all") else ""), [tpt], "x" if tpt.endswith("vs_ref.all") else "tokens",
                 "Generated tokens per try" + (" over the reference's on the same levels; below 1.0 = cheaper tries." if tpt.endswith("vs_ref.all") else "."),
                 {"names": {tpt: "tokens/try"}, "refs": [(1.0, "reference = 1.0")] if tpt.endswith("vs_ref.all") else []}),
                ("Groups dropped", ["train.groups_dropped_frac_step"], "%",
                 "Share of the update's played groups the trainer dropped (lag, filters, other). Bar 10%.",
                 {"names": {"train.groups_dropped_frac_step": "dropped this update"}, "refs": [(10, "10% bar", "dash")],
                  "bars": {"train.groups_dropped_frac_step": RL_BARS["train.groups_dropped_frac_step"]}})]:
            ks = [k for k in keys if k in uobs]
            f = upd_fig(title, ks, unit, "Key plots", helptext, **kw) if ks else None
            if f:
                rlkey.append(f)
                claimed.update(ks)
        if rlkey:
            claimed.update(k for k in ser if k.startswith("train.groups_dropped_frac"))
    claimed.update(k for k in ser if "rhae_tok_vs_ref_base" in k or "rhae_tok_vs_ref_cov" in k or k.startswith("train.first_coverage"))
    for title, pats, unit, section, helptext in PANELS:  # unclaimed keys are not plotted (only useful plots)
        keys = [k for k in sorted(ser) if k not in claimed and any(fnmatch.fnmatch(k, p) for p in pats)
                and not any(fnmatch.fnmatch(k, h) for h in HIDE)]
        keys += [k for k in sorted(uobs) if k not in claimed and k not in keys and any(fnmatch.fnmatch(k, p) for p in pats)
                 and not any(fnmatch.fnmatch(k, h) for h in HIDE)]
        if not keys:
            continue
        claimed.update(keys)
        if title in ("Tokens per step", "Progress") or (is_rl and (title in RL_DROP or title in BY_CKPT)):  # no information / raw RL reads not comparable across steps
            continue
        if title.startswith("Approx KL") and is_rl:
            title += ", informational"
        if is_rl and section != "Infrastructure":  # per-update metrics on an update-number axis, branch-aware
            uk = [k for k in keys if k in uobs]
            if uk:
                for t_, ks in split_scales(title, uk, {k: [v for _, v, _ in split_branches(uobs[k], bstarts)[0]] for k in uk}):
                    f = upd_fig(t_, ks, unit, section, helptext, refs=[r for k in ks for r in RL_REFS.get(k, [])],
                                bars={k: RL_BARS[k] for k in ks if k in RL_BARS})
                    if f:
                        figs.append(f)
            keys = [k for k in keys if k not in uobs and k in ser]
            if not keys:
                continue
            title += " · live" if uk else ""
        if title in BY_CKPT and not asyn and not is_rl:  # async: served versions lag and mix within a step, one dot per checkpoint misleads
            ck = ckpt_fig([r for r in rows if "train.played_by" in (r.get("m") or {})] + meta.get("_reads", []), keys, meta["run"])
            if ck:
                figs.append({"section": section, "title": title + " · by checkpoint", "unit": unit, "single": None, "bar": False, "small": False,
                             "help": helptext + " x axis: the checkpoint that played the round (r07 = after update 7); each dot is one read "
                             "(hover: run and round), the dotted line joins each checkpoint's mean.",
                             "traces": ck["traces"], "rng": ck["rng"], "legend": True, "y2": False, "ck": ck["cats"]})
                continue
        keys = [k for k in keys if k in ser]
        if not keys:
            continue
        for title, keys in split_scales(title, keys, {k: ser[k][1] for k in keys}) if title != "Round time split" else [(title, keys)]:
            bar = title == "Round time split"
            top = {k: max(map(abs, ser[k][1]), default=0) for k in keys}
            big = max(top.values(), default=0)
            right = set()  # no dual axes; mixed scales are split into separate panels above
            traces = []
            for j, k in enumerate(keys):
                c = COLORS[j % len(COLORS)]
                pre, suf = k.split(".", 1) if "." in k else ("", k)
                multi = len({kk.split(".", 1)[0] for kk in keys}) > 1
                nm = PREFIX_NAMES.get(pre, pre) if multi and PREFIX_NAMES.get(pre) else (
                    suf.rsplit(".", 1)[-1] if re.search(r"\.L\d+$", suf) else suf.replace("_", " ") + (" %" if is_frac(k) else ""))
                tr = {"x": ser[k][0], "y": ser[k][1], "name": NAMES.get(k, nm),
                      "type": "bar" if bar else "scatter", "hovertemplate": "%{y:.4g}"}
                if bar:  # share of the round, seconds in the hover
                    tot = {}
                    for kk in keys:
                        for x, y in zip(*ser[kk]):
                            tot[x] = tot.get(x, 0) + y
                    rounds = sorted({x for kk in keys for x in ser[kk][0]})  # one row per round: label bars R1, R2, ...
                    tr.update(x=[f"R{int(roundat[x])}" if x in roundat else f"R{rounds.index(x) + 1}" for x in ser[k][0]], y=[y / 60 for y in ser[k][1]],
                              hovertemplate="%{x}: %{y:.0f} min", name=tr["name"].removesuffix(" s"))
                raw = None
                if not bar and (k.startswith("infra.") or k == "time.step_s"):  # live samples: break the line over gaps > 20 min
                    segs = []
                    for x, y in zip(*ser[k]):
                        if not segs or x - segs[-1][-1][0] > (0.05 if k in PLAY_ONLY else 1 / 3):  # play-only: break at update phases
                            segs.append([])
                        segs[-1].append((x, y))
                    dense = len(ser[k][0]) > 40 and k != "infra.cost_usd" and not k.endswith("_cap_gib")  # 60-s samples are noisy, plot a 10-min mean
                    gx, gy, rx, ry = [], [], [], []
                    for seg in segs:
                        if gx:
                            gx.append((gx[-1] + seg[0][0]) / 2)
                            gy.append(None)
                            rx.append(gx[-1])
                            ry.append(None)
                        for x, y in seg:
                            win = [v for u, v in seg if abs(u - x) <= 1 / 12] if dense else [y]
                            gx.append(x)
                            gy.append(sum(win) / len(win))
                            rx.append(x)
                            ry.append(y)
                    tr.update(x=gx, y=gy)
                    if dense:
                        tr["name"] += " (10-min mean)"
                        raw = {"x": rx, "y": ry, "name": tr["name"].replace(" (10-min mean)", " raw"), "type": "scatter", "mode": "lines",
                               "showlegend": False, "hoverinfo": "skip", "line": {"width": 1, "color": c}, "opacity": 0.22}
                elif title == "Loss" and not is_rl and len(ser[k][1]) > 10:  # per-step SFT loss is noisy, plot a centred mean
                    ys, w = ser[k][1], max(5, min(25, len(ser[k][1]) // 15))
                    tr["y"] = [sum(ys[max(0, i - w // 2):i + w // 2 + 1]) / len(ys[max(0, i - w // 2):i + w // 2 + 1]) for i in range(len(ys))]
                    tr["name"] += f" ({w // 2 * 2 + 1}-step mean)"
                    raw = {"x": ser[k][0], "y": ys, "name": tr["name"] + " raw", "type": "scatter", "mode": "lines",
                           "showlegend": False, "hoverinfo": "skip", "line": {"width": 1, "color": c}, "opacity": 0.22}
                if k == "time.checkpoint_s":
                    tr.update(mode="markers", marker={"size": 9, "symbol": "diamond", "color": COLORS[1]})
                if k in right:
                    tr["yaxis"], tr["name"] = "y2", tr["name"] + " (right axis)"
                if bar:
                    tr["marker"] = {"color": PHASE_COLORS.get(k, c)}
                else:
                    dotted = k.endswith(("_target", "_stop", "_cap_gib", "_base", "_lo", "_hi"))
                    if k == "time.checkpoint_s":
                        traces.append(tr)
                        continue
                    tr.update(mode="lines" if dotted or len(ser[k][0]) > 40 else "lines+markers",
                              line={"width": 1.5 if dotted else 2.2, "color": c, "dash": "dot" if dotted else "solid", "shape": "linear"},
                              marker={"size": 5, "color": c})
                    if len(keys) == 1 and min(ser[k][1]) >= 0 and not raw:
                        tr.update(fill="tozeroy", fillcolor=c + "1f")
                if raw and k not in right:
                    traces.append(raw)
                traces.append(tr)
            if "time.step_s" in keys and len(ser["time.step_s"][1]) > 2:
                med = sorted(ser["time.step_s"][1])[len(ser["time.step_s"][1]) // 2]
                traces.append({"x": [0, ser["time.step_s"][0][-1]], "y": [med, med], "name": f"median {med:,.0f} s", "mode": "lines",
                               "type": "scatter", "hoverinfo": "skip", "line": {"width": 1.5, "dash": "dot", "color": "#94a3b8"}})
            ys = [y for k in keys if k not in right for y in ser[k][1]]
            small = max(map(abs, ys), default=1) < 1e-2
            rng = [0, 100] if bar else [0, max(ys) * 1.15] if ys and min(ys) >= 0 and max(ys) > 0 else None  # flat series: keep zero in view
            single = None
            if not bar and max(len(ser[k][1]) for k in keys) <= 2:  # one or two readings (before/after): numbers read better than dots
                single = []
                for k in keys:
                    if (k.endswith("_base") and k[:-5] in ser) or not any(ser[k][1]):
                        continue  # shown beside its metric; an all-zero timing (no such phase yet) says nothing
                    base = ser.get(k + "_base")
                    (x0, *_), (y0, *_) = ser[k]
                    name = k.split(".", 1)[-1].replace("_", " ")
                    if base:
                        b, yl = base[1][-1], ser[k][1][-1]
                        single.append([name, f"{b:.4g} → {yl:.4g}", f"base → now: {100 * (yl - b) / abs(b):+.1f}%" if b else "base → now"])
                    elif len(ser[k][1]) == 1:
                        scored = k.startswith("eval.") or "rhae" in k  # a base exists only for scored reads, not timings or infra
                        single.append([name, f"{y0:.4g}", f"one reading, at {x0:.2f} h" + ("; no base reported" if scored else "")])
                    else:
                        y1 = ser[k][1][1]
                        single.append([name, f"{y0:.4g} → {y1:.4g}", f"{100 * (y1 - y0) / abs(y0):+.2f}% from {x0:.2f} h to {ser[k][0][1]:.2f} h" if y0 else ""])
            figs.append({"section": section, "title": title, "unit": unit, "help": helptext, "traces": traces, "single": single,
                         "bar": bar, "small": small, "rng": rng, "legend": len(traces) > 1, "y2": False})
    figs = ([] if asyn else live_figs(rows)) + pass_figs(rows) + figs
    figs.sort(key=lambda f: SECTION_ORDER.get(f["section"], 9))  # stable: panel order kept within a section
    # at most 6 essential plots on top (results, dosing, infra), everything else below
    key = []
    for t in KEY_PLOTS:
        f = next((f for f in figs if f["title"] == t and f not in key), None)
        if f and len(key) < 6:
            key.append(f)
    for f in key:
        figs.remove(f)
    key0 = list(key)
    held = sft_heldout_figs(rows)  # SFT held-out reads as lines vs step, right under Loss
    if held:  # the generic all-eval.* panel over time duplicates these on one unreadable axis
        figs = [f for f in figs if not f["title"].startswith("Held-out reads")]
    at = next((i + 1 for i, f in enumerate(key) if f["title"] == "Loss"), len(key))
    key = key[:at] + held + key[at:]
    if is_rl and rlkey:  # the decision metrics, then KV/prefix and play progress
        key = rlkey + [f for f in key if f["title"] in ("KV cache use",)]
        figs += [f for f in key0 if f not in key and f["title"] != "KV cache use"]
        figs.sort(key=lambda f: SECTION_ORDER.get(f["section"], 9))
    if is_rl:  # KV use and prefix hit rate side by side in Key plots on every RL page
        kv = next((f for f in figs if f["title"] == "KV cache use"), None) or next((f for f in key if f["title"] == "KV cache use"), None)
        figs = [f for f in figs if f is not kv]
        rest = [f for f in key if f is not kv]
        at = len(rest) if rlkey else min(2, len(rest) - len(rest) % 2)  # an even slot: the two share one row of the two-column grid
        pair = [f for f in (kv, prefix_fig(rows, clock)) if f]
        play = play_progress_fig(rows, clock)  # play-phase progress, right under the KV/prefix row
        key = rest[:at] + pair + ([play] if play else []) + rest[at:]
    leg_kd = None if is_rl else leg_kd_fig(rows, meta["run"], leg0, meta.get("same_leg") or ())  # first in Key plots for SFT; meta same_leg: earlier runs shown as this leg
    if leg_kd:
        key = [leg_kd] + key
    figs = [dict(f, section="Key plots") for f in key] + figs
    # one timeline strip (attempts, rollbacks, incidents, stages) and plain event lines in the wall-time panels
    marks += [{"t": t, "labels": [lab], "kinds": {"stage"}} for t, lab in sorted(list(meta.get("_stages", [])) + list(meta.get("_attempt_stages", [])))
              if t in clock and not (bands and lab.startswith("attempt"))]
    marks.sort(key=lambda m: m["t"])
    merged = []
    for m_ in marks:
        if merged and m_["t"] - merged[-1]["t"] <= 1800:
            merged[-1]["labels"] += [x for x in m_["labels"] if x not in merged[-1]["labels"]]
            merged[-1]["kinds"] |= m_["kinds"]
        else:
            merged.append({"t": m_["t"], "labels": list(dict.fromkeys(m_["labels"])), "kinds": set(m_["kinds"])})
    kind = lambda ks: next(k for k in ("incident", "rollback", "attempt", "stage") if k in ks)
    timeline = {"bands": [{"x0": at_clock(clock, b["t0"]), "x1": at_clock(clock, b["t1"]), "n": b["n"],
                           "txt": f"attempt {b['n']}: {paths.local(b['t0']):%d %b %H:%M}–{paths.local(b['t1']):%H:%M}" + (f" · {b['cause']}" if b["cause"] else "")}
                          for b in bands],
                "marks": [{"x": at_clock(clock, m_["t"]), "kind": kind(m_["kinds"]),
                           "txt": f"{paths.local(m_['t']):%d %b %H:%M}<br>" + "<br>".join(clip(x, 90) for x in m_["labels"][:6])} for m_ in merged]}
    shade = update_phases(rows, clock) if is_rl else []
    stats = rl_stats(meta, bands, uobs, heldout_reads(meta) if is_rl else [], bstarts) if is_rl and uobs else []
    head = {k: v[1][-1] for k, v in ser.items()}
    chips = []
    for label, key, unit, nd in CHIPS:
        vals = [v for k, v in head.items() if k == key or (key.endswith(".") and k.startswith(key))]
        if vals:
            chips.append([label, f"{sum(vals) / len(vals):,.{nd}f}{unit}"])
    card = lambda pre: sorted((k.rsplit(".", 1)[-1], v) for k, v in head.items() if k.startswith(pre) and k.rsplit(".", 1)[-1].isdigit())  # per-card keys only (not .trainer_mean)
    cap = head.get("infra.vram_cap_gib") or VRAM_CAP_GIB
    gauges = [g for g in (["GPU util", [[c, v / 100, f"{v:.0f}%"] for c, v in card("infra.gpu_util.")], "util"],
                          ["GPU memory", [[c, v / cap, f"{v:.0f} GiB"] for c, v in card("infra.vram_gib.")], "mem"]) if g[1]]
    arm = meta.get("arm") or next((a for pre, a in (("rl", "rl"), ("sft-heal-", "pth"), ("sft-", "sft"), ("pth", "pth"), ("mfu-", "pth"), ("hx4", "envs"), ("envs", "envs")) if meta["run"].startswith(pre)), "?")
    ncards = max(len(card("infra.gpu_util.")), len(card("infra.vram_gib."))) or meta.get("cards", "?")  # the cards that report, not the meta guess
    return {"run": meta["run"], "arm": arm.upper(), "armc": ARM.get(arm.lower(), "#64748b"),
            "what": meta.get("summary") or short_what(meta.get("what", "")), "status": status, "cards": ncards,
            "minis": mini_figs(rows, (not meta.get("same_leg") and meta.get("_own_start")) or meta.get("_start"), cap),
            "label": meta.get("label") or meta.get("stage") or clip(short_what(meta.get("summary") or meta.get("what") or ""), 60),
            "stages": stages, "timeline": timeline, "shade": shade, "stats": stats,
            "attempt": f"attempt {bands[-1]['n']} ({bands[-1]['state']})" if bands else "", "last": last, "hours": round(clock[last], 2), "x0": round(min(clock.values()), 2), "chips": chips, "gauges": gauges, "cap": cap, "figs": figs,
            "alerts": [[r["t"], clock[r["t"]], r["alert"]] for r in rows if r.get("alert") and not (own0 and r["t"] < own0 - 300)]}


KEY_PLOTS = ["Held-out RHAE/tok vs base", "RHAE/tok vs first visit", "Groups dropped", "RHAE/tok vs reference", "Loss", "Policy sharpness", "Gradient norm", "Clipping", "Throughput",
             # RL starts change every step (curriculum): raw training-try RHAE/tokens/clears are not comparable across steps
             "Drift (KL)", "GPU utilization", "GPU memory", "Step time"]


def clip(text, n):
    return text if len(text) <= n else text[:n - 1].rsplit(" ", 1)[0] + "…"


def short_what(what):
    """First sentence of the queue description, at most ~160 characters (the meta's own "summary" wins)."""
    w = re.split(r"(?<=[.;])\s", what.strip(), maxsplit=1)[0]
    return w if len(w) <= 160 else w[:157].rsplit(" ", 1)[0] + "…"


PAGE = r"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<meta http-equiv='refresh' content='60'><title>Training Live</title>
<script src='https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.0/plotly.min.js'></script>
<style>
:root{--bg:#f4f6fa;--panel:#fff;--fg:#0f172a;--mut:#64748b;--line:#e2e8f0;--grid:#eef2f7;--acc:#3b82f6;--sel:#eff6ff;--bad:#dc2626;--ok:#16a34a;--shadow:0 1px 2px rgba(15,23,42,.06),0 1px 3px rgba(15,23,42,.05)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0b1020;--panel:#131a2b;--fg:#e2e8f0;--mut:#94a3b8;--line:#243049;--grid:#1c2538;--acc:#60a5fa;--sel:#1a2540;--shadow:none}}
:root[data-theme="dark"]{--bg:#0b1020;--panel:#131a2b;--fg:#e2e8f0;--mut:#94a3b8;--line:#243049;--grid:#1c2538;--acc:#60a5fa;--sel:#1a2540;--shadow:none}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 Inter,system-ui,-apple-system,Segoe UI,sans-serif}
a{color:var(--acc);text-decoration:none} a:hover{text-decoration:underline}
.top{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;padding:14px 24px;border-bottom:1px solid var(--line);background:var(--panel)}
.top h1{font-size:17px;margin:0;font-weight:650} .top .meta{color:var(--mut);font-size:12.5px}
.wrap{display:grid;grid-template-columns:290px 1fr;min-height:calc(100vh - 52px);background:linear-gradient(to right,var(--panel) 289px,var(--line) 289px,var(--line) 290px,var(--bg) 290px)}
nav{padding:12px 10px;overflow-y:auto;height:100vh;position:sticky;top:0}
nav h4{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:10px 8px 6px;font-weight:600}
.ri{display:block;padding:9px 10px;border-radius:8px;margin-bottom:2px;color:var(--fg);cursor:pointer;border:1px solid transparent}
.ri:hover{background:var(--sel);text-decoration:none} .ri.on{background:var(--sel);border-color:var(--acc)}
.ri .n{font-weight:600;font-size:13px;overflow-wrap:anywhere} .ri .s{color:var(--mut);font-size:11.5px;margin-top:2px}
.pill{display:inline-block;color:#fff;border-radius:5px;padding:0 6px;font-size:10.5px;font-weight:700;margin-right:6px;vertical-align:1px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;vertical-align:1px;background:var(--mut)}
.dot.live{background:var(--ok);box-shadow:0 0 0 3px rgba(22,163,74,.18)} .dot.booting{background:#d97706;box-shadow:0 0 0 3px rgba(217,119,6,.18)} .dot.stopped{background:var(--bad)}
main{padding:20px 24px 40px;min-width:0}
.rh h2{font-size:20px;margin:0 0 4px;font-weight:650;word-break:break-all} .rh .what{color:var(--mut);max-width:980px}
.rh .facts{color:var(--mut);font-size:12.5px;margin-top:6px}
.tl{margin-top:10px} .tl .l{color:var(--mut);font-size:11.5px} .tlf{height:58px}
.tiles.stats{margin-top:12px;grid-template-columns:repeat(auto-fill,minmax(130px,1fr))} .stats .v{font-size:19px}
@media (max-width:600px){.tiles.stats{grid-template-columns:repeat(3,minmax(0,1fr));gap:6px} .stats .tile{padding:8px} .stats .v{font-size:16px} .stats .l{font-size:11px}}
.tile.warn{border-color:#d97706;border-left-width:4px} .tile.warn .l:first-child{color:#b45309}
.tile.mini{padding:8px 10px} .mf{height:170px}
.body2{display:grid;grid-template-columns:minmax(0,1fr) 270px;gap:18px;align-items:start;margin-top:8px}
.gcol{position:sticky;top:12px;display:flex;flex-direction:column;gap:10px;margin-top:36px} .gcol .gs{flex-direction:column;align-items:center}
@media (max-width:1100px){.body2{grid-template-columns:1fr} .gcol{position:static;flex-direction:row;flex-wrap:wrap;margin-top:0}}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}
.tile.g{grid-column:span 2} .gs{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:flex-end;margin-top:2px} .gs figure{margin:0;text-align:center}
.gs svg{display:block} .gs figcaption{color:var(--mut);font-size:11px;margin-top:-2px}
@media (max-width:1100px){.hdr.al{grid-template-columns:1fr}}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 14px;box-shadow:var(--shadow)}
.tile .v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums} .tile .l{color:var(--mut);font-size:12px}
.alerts{border:1px solid var(--bad);border-left-width:4px;background:var(--panel);border-radius:10px;padding:10px 14px;margin:0;max-height:260px;overflow-y:auto;font-size:13px;overflow-wrap:anywhere;max-width:100%}
.alerts .hd{color:var(--bad);font-weight:650} .alerts .ex{color:var(--mut);font-size:12px;margin:2px 0 6px}
.alerts ol{margin:0;padding-left:0;list-style:none} .alerts li{margin:3px 0;display:flex;gap:8px;align-items:baseline}
.alerts .no{flex:0 0 auto;background:var(--bad);color:#fff;border-radius:50%;width:18px;height:18px;font-size:11px;font-weight:700;display:inline-flex;align-items:center;justify-content:center}
.alerts .tm{color:var(--mut);flex:0 0 auto;font-variant-numeric:tabular-nums}
h3{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:26px 0 10px;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px 12px;box-shadow:var(--shadow);min-width:0}
.card .t{font-weight:600;font-size:13.5px} .card .u{color:var(--mut);font-weight:400;font-size:12px;margin-left:6px}
.card .fig{height:220px;margin-top:4px}
.card .one{padding:12px 4px 4px} .card .one .v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums} .card .one .l{color:var(--mut);font-size:12.5px}
.card .h{color:var(--mut);font-size:12px;line-height:1.45;border-top:1px dashed var(--line);padding-top:8px;margin-top:6px}
.empty{color:var(--mut);padding:40px}
@media (max-width:820px){.wrap{grid-template-columns:1fr;background:none} nav{position:static;height:auto;background:var(--panel);border-bottom:1px solid var(--line);border-right:0;border-bottom:1px solid var(--line);display:flex;gap:6px;overflow-x:auto;padding:8px 16px}
 nav h4{display:none} .ri{flex:0 0 220px} main{padding:16px} .top{padding:12px 16px} .grid{grid-template-columns:1fr}}
</style></head><body>
<div class='top'><h1>Training runs</h1><span class='meta'>Updated __NOW__ · rebuilt every 2 min · <a href='../../MODAL_STATUS.html'>&larr; What is running</a></span></div>
<div class='wrap'><nav id='nav'></nav><main id='main'></main></div>
<script>
const RUNS=__DATA__, NOW=__T__;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ago=t=>{const m=Math.round((NOW-t)/60);return m<60?m+' min ago':(m/60).toFixed(1)+' h ago'};
const hm=t=>{const d=new Date(t*1000);return String(d.getHours()).padStart(2,'0')+':'+String(d.getMinutes()).padStart(2,'0')};  // viewer's local time
function gauge(c,kind){const f=Math.max(0,Math.min(1,c[1])),L=Math.PI*26;  // semicircle gauge: util green when busy; memory green when full but not at the edge
 const col=kind==='util'?(f>=.8?'#16a34a':f>=.5?'#eab308':'#dc2626'):(f>=.99?'#dc2626':f>=.75?'#16a34a':'#eab308');
 return `<figure><svg width='76' height='44' viewBox='0 0 76 44'><path d='M8 38 A30 30 0 0 1 68 38' fill='none' stroke='var(--line)' stroke-width='7' stroke-linecap='round' pathLength='${L}'/>
 <path d='M8 38 A30 30 0 0 1 68 38' fill='none' stroke='${col}' stroke-width='7' stroke-linecap='round' pathLength='${L}' stroke-dasharray='${(L*f).toFixed(1)} ${L}'/>
 <text x='38' y='37' text-anchor='middle' font-size='13' font-weight='650' fill='var(--fg)'>${esc(c[2])}</text></svg><figcaption>${esc(c[0])}</figcaption></figure>`}
function nav(cur){let h='',grp=null;RUNS.forEach((r,i)=>{const g=(r.status==='live'||r.status==='booting')?'Live':'Recently ended';
 if(g!==grp){h+=`<h4>${g}</h4>`;grp=g}
 h+=`<a class='ri${i===cur?' on':''}' href='#${encodeURIComponent(r.run)}'><div class='n'><span class='pill' style='background:${r.armc}'>${esc(r.arm)}</span>${esc(r.label||r.run)}</div>${r.label?`<div class='s' style='word-break:break-all'>${esc(r.run)}</div>`:''}
 <div class='s'><span class='dot ${r.status}'></span>${r.status} · ${ago(r.last)}${r.gauges.length?' · GPU '+r.gauges[0][1].map(c=>c[2]).join('/'):r.chips.length?' · '+esc(r.chips[0][0])+' '+esc(r.chips[0][1]):''}</div></a>`});
 document.getElementById('nav').innerHTML=h||"<div class='empty'>No training run has exported metrics yet.</div>"}
function show(){const want=decodeURIComponent(location.hash.slice(1));let i=RUNS.findIndex(r=>r.run===want);if(i<0)i=0;nav(i);
 const m=document.getElementById('main');document.querySelectorAll('.fig,.mf').forEach(d=>Plotly.purge(d));
 if(!RUNS.length){m.innerHTML='';return}
 const r=RUNS[i];document.title=r.run+' · Training Live';
 let h=`<div class='rh'><h2><span class='pill' style='background:${r.armc}'>${esc(r.arm)}</span>${esc(r.run)}</h2><div class='what'>${esc(r.what)}</div>
 <div class='facts'><span class='dot ${r.status}'></span>${r.status} · last data ${ago(r.last)} · ${r.hours} h of run time · ${esc(r.cards)} card(s)${r.attempt?' · '+esc(r.attempt):''}</div></div>`;
 const tl=r.timeline||{bands:[],marks:[]};
 if(tl.bands.length||tl.marks.length)h+=`<div class='tl'><div class='l'>Timeline (run hours, pauses over 30 min cut): attempts as bands · <span style='color:#d97706'>&#9660;</span> rollback · <span style='color:var(--bad)'>&#10005;</span> incident · | attempt or leg start. Hover for details; panels show the same events as thin lines.</div><div id='tlf' class='tlf'></div></div>`;
 if((r.stats||[]).length)h+=`<div class='tiles stats'>`+r.stats.map(x=>`<div class='tile${x[3]?' '+x[3]:''}'><div class='l'>${x[3]?'&#9888; ':''}${esc(x[0])}</div><div class='v'>${esc(x[1])}</div><div class='l'>${esc(x[2])}</div></div>`).join('')+`</div>`;
 h+=`<div class='body2'><div>`;
 let sec=null;r.figs.forEach((f,j)=>{if(f.section!==sec){if(sec)h+='</div>';h+=`<h3>${esc(f.section)}</h3><div class='grid'>`;sec=f.section}
  h+=`<div class='card'><div class='t'>${esc(f.title)}<span class='u'>${esc(f.unit)}</span></div>`+(f.single?`<div class='one'>`+f.single.map(x=>`<div class='l'>${esc(x[0])}</div><div class='v'>${esc(x[1])}</div><div class='l' style='margin-bottom:10px'>${esc(x[2])}</div>`).join('')+`</div>`:`<div class='fig' id='f${j}'></div>`)+`<div class='h'>${esc(f.help)}</div></div>`});
 if(sec)h+='</div>';
 h+=`</div><aside class='gcol'><div class='l' style='color:var(--mut);font-size:11.5px'>Since the latest start: 10-min mean, play only</div>`+r.minis.map((f,j)=>`<div class='tile mini'><div class='l'>${esc(f.title)}</div><div class='mf' id='m${j}'></div></div>`).join('')+`</aside></div>`;
 m.innerHTML=h;window.scrollTo(0,0);
 const fg=css('--fg'),mut=css('--mut'),grid=css('--grid'),line=css('--line');
 r.minis.forEach((f,j)=>Plotly.newPlot('m'+j,f.traces,{margin:{l:34,r:6,t:4,b:f.traces.length>1||f.bar?40:20},font:{color:mut,size:10,family:'Inter,system-ui,sans-serif'},
  xaxis:f.bar?{type:'category',gridcolor:grid,linecolor:line}:{ticksuffix:' h',nticks:4,tickangle:0,gridcolor:grid,linecolor:line,zeroline:false},yaxis:{gridcolor:grid,linecolor:line,zeroline:false,range:f.rng,rangemode:'tozero'},
  barmode:'group',bargap:.2,showlegend:f.traces.length>1,legend:{orientation:'h',y:-0.22,yanchor:'top',font:{size:10,color:fg}},hovermode:'x unified',
  hoverlabel:{bgcolor:css('--panel'),bordercolor:line,font:{size:11.5,color:fg}},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)'},{displayModeBar:false,responsive:true}));
 const wallR=[Math.min(r.x0,0),Math.max(r.hours,0.05)*1.02],kcol={rollback:'#d97706',incident:css('--bad'),attempt:mut,stage:mut};
 const wallShapes=(r.shade||[]).map(s=>({type:'rect',layer:'below',x0:s[0],x1:s[1],yref:'paper',y0:0,y1:1,fillcolor:'rgba(100,116,139,.12)',line:{width:0}}))
  .concat(((r.timeline||{}).marks||[]).map(m=>({type:'line',x0:m.x,x1:m.x,yref:'paper',y0:0,y1:1,line:{color:kcol[m.kind]||mut,width:1,dash:m.kind==='stage'||m.kind==='attempt'?'dot':'solid'},opacity:.7})));
 const legRows=(f,el)=>{const ns=f.traces.filter(t=>t.showlegend!==false&&t.name).map(t=>String(t.name));if(!f.legend||!ns.length)return 0; // legend never scrolls (the stray grey bar was its scrollbar)
  const iw=Math.max(...ns.map(n=>n.length))*6.6+44;return Math.ceil(ns.length/Math.max(1,Math.floor((el.clientWidth-20)/iw)))};
 r.figs.forEach((f,j)=>{if(f.single||f.xlab)return;const el=document.getElementById('f'+j),n=legRows(f,el);f._lr=n;if(n>1)el.style.height=(200+n*21)+'px'});
 r.figs.forEach((f,j)=>f.single||Plotly.newPlot('f'+j,f.traces,{margin:{l:46,r:10,t:6,b:f.legend?(f.xlab?96:(f._lr?26+f._lr*21:48)):24},font:{color:mut,size:11,family:'Inter,system-ui,sans-serif'},
  xaxis:f.ck?{type:'category',categoryorder:'array',categoryarray:f.ck,gridcolor:grid,linecolor:line}:f.upd?{type:'linear',tickprefix:'U',dtick:updTick(f),range:f.xr||updRange(f),gridcolor:grid,linecolor:line,zeroline:false}:f.xlab?{type:'linear',range:f.xr,dtick:f.dtick||(f.traces.some(t=>t.x.length>12)?2:1),title:{text:f.xlab,font:{size:11}},gridcolor:grid,linecolor:line,zeroline:false}:{range:f.bar?undefined:wallR,ticksuffix:f.bar?'':' h',type:f.bar?'category':'linear',gridcolor:grid,linecolor:line,zeroline:false},
  yaxis:{gridcolor:grid,linecolor:line,zeroline:false,range:f.rng,exponentformat:f.small?'e':'none'},
  shapes:f.bar||f.ck||f.xlab||f.upd?[]:wallShapes,
  barmode:'group',bargap:.25,showlegend:f.legend,legend:f.xlab||!f._lr?{orientation:'h',y:f.xlab?-0.42:-0.14,yanchor:'top',font:{size:11,color:fg}}:{orientation:'h',yref:'container',y:(f._lr*21+2)/(200+Math.max(1,f._lr)*21),yanchor:'top',font:{size:11,color:fg}},
  hovermode:f.ck||f.xlab||f.title.startsWith('Held-out RHAE')?'closest':'x unified',hoverlabel:{bgcolor:css('--panel'),bordercolor:css('--line'),font:{size:12.5,color:fg}},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)'},
  {displayModeBar:false,responsive:true}));
 if(document.getElementById('tlf')){ // timeline strip: attempt bands, rollbacks, incidents
  const tr=[];tl.bands.forEach((b,k)=>tr.push({x:[b.x0,b.x1,b.x1,b.x0,b.x0],y:[0,0,1,1,0],fill:'toself',fillcolor:k%2?'rgba(100,116,139,.22)':'rgba(100,116,139,.10)',line:{width:0},mode:'lines',hoveron:'fills',text:b.txt,hoverinfo:'text',type:'scatter',showlegend:false}));
  const sym={rollback:'triangle-down',incident:'x',attempt:'line-ns-open',stage:'line-ns-open'};
  ['attempt','stage','rollback','incident'].forEach(kd=>{const ms=tl.marks.filter(m=>m.kind===kd);if(ms.length)tr.push({x:ms.map(m=>m.x),y:ms.map(()=>0.5),mode:'markers',type:'scatter',showlegend:false,
   marker:{symbol:sym[kd],size:kd==='attempt'||kd==='stage'?16:11,color:kcol[kd],line:{width:kd==='attempt'||kd==='stage'?1.5:2,color:kcol[kd]}},hovertext:ms.map(m=>m.txt),hoverinfo:'text'})});
  const span=wallR[1]-wallR[0];
  Plotly.newPlot('tlf',tr,{margin:{l:46,r:10,t:2,b:18},font:{color:mut,size:10,family:'Inter,system-ui,sans-serif'},xaxis:{range:wallR,ticksuffix:' h',gridcolor:grid,linecolor:line,zeroline:false},
   yaxis:{range:[0,1],visible:false,fixedrange:true},hovermode:'closest',hoverlabel:{bgcolor:css('--panel'),bordercolor:line,font:{size:12,color:fg}},
   annotations:tl.bands.filter(b=>b.x1-b.x0>span*(innerWidth<700?0.08:0.025)).map(b=>({x:(b.x0+b.x1)/2,y:0.92,yanchor:'top',text:'a'+b.n,showarrow:false,font:{size:10,color:mut}})),
   paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)'},{displayModeBar:false,responsive:true})}}
function updRange(f){const xs=f.traces.flatMap(t=>t.x).filter(x=>typeof x==='number');return xs.length?[Math.min(...xs)-0.5,Math.max(...xs)+0.5]:undefined}
function updTick(f){const r=f.xr||updRange(f)||[0,1];const n=r[1]-r[0];return n<=16?1:n<=40?2:5}
window.addEventListener('hashchange',show);show();
</script></body></html>"""


def build():
    runs = [run_view(m, rows, last) for m, rows, last in load()]
    page = (PAGE.replace("__NOW__", f"{paths.local():%Y-%m-%d %H:%M} {paths.tz()}").replace("__T__", str(int(time.time())))
            .replace("__DATA__", json.dumps(runs, separators=(",", ":")).replace("</", "<\\/")))
    OUT.write_text(page)
    return len(runs)


if __name__ == "__main__":
    print(build(), "runs")
