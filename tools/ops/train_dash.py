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
TRAIN_RUN_PREFIXES = ("rl1-loop-", "sft-heal-round-")  # RL loop and OPD/heal rounds
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
    ("RHAE/tok vs first visit", ["train.rhae_tok_vs_first", "train.rhae_tok_vs_first.all"], "x", "Training",
     "Training progress that survives the changing level mix: each level's RHAE per token this step divided by the first group ever played on it, geometric mean over levels. Above 1 = the policy plays levels more efficiently than when it first met them."),
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
    ("Approx KL per step (k3)", ["train.k3_measure", "train.approx_kl", "train.k3_target", "train.k3_stop"], "nats", "Training",
     "The trainer's KL estimate (k3) per update (k3_measure: measured after the step on a held sample; approx_kl on multi-step updates), against the dose target and the early-stop line. Above the stop line the update is too large; far below the target means the dose is small."),
    ("Per-token drift and off-policy share", ["train.k3_tok", "train.tis_out_share"], "", "Training",
     "k3_tok: mean per-token KL of the update on its own tries (a jump with flat k3 means a few tokens moved a lot); tis_out_share: share of tokens outside the TIS cap (alert above 5%)."),
    ("Clipping", ["train.clip_frac*", "train.tis_clip_frac", "train.live_tis_out_frac"], "%", "Training",
     'Share of tokens whose update was capped by the safety clip. A few % is normal; rising toward 20%+ means updates are too large or the data is stale.'),
    ("Drift from base", ["train.drift_gap_base*", "train.logprob_gap*", "train.nll_base_tokens*", "train.nll_trainer*"], "nats", "Training",
     "RL: how differently the trainer now scores the base model's recorded play (logprob gap vs the base's own logprobs, and the trainer's loss on base tokens). Grows as the policy moves away from base, which is expected; it is not a serving error. Earlier rounds show it as logprob_gap / nll_trainer."),
    ("Train/serve mismatch", ["train.tis_mean*", "train.routing_*"], "", "Training",
     "RL, on the update's own tries: trainer/server probability ratio (tis_mean, ~1 when they agree; at most one update of lag), how often the trainer's expert choice matches the server's (routing agree; ~0.6-0.7 normal, drops when the trainer is an update ahead) and routing coverage (should be 1). A jump at an update's first step is a real serving mismatch."),
    ("Reply length", ["train.reply_tokens_mean"], "tokens", "Training",
     'Average tokens per model reply. Slow change is fine; a steady climb or collapse often precedes reward hacking or degenerate output.'),
    ("Reply quality", ["train.truncation_frac", "train.noaction_frac", "train.filler_per_1k"], "", "Training",
     "Truncation = share of replies cut off at the cap (above 10-15% destabilizes RL); no-action = share of replies that send no game action (only thinking or python), normal around 40% for this harness, a steady rise means the model stalls instead of playing; filler (padding phrases per 1,000 tokens): lower is better."),
    ("Held-out reads", ["eval.*"], "", "Results",
     "Reads on games the model never trained on: the result that counts. Scores (rhae, clears): higher is better. Held-out KL (heals): how far the model's replies are from the teacher's, lower is better. Base = the same read on the starting model (dotted line, or 'base → now'); differences smaller than the read's noise are not real."),
]
PHASE_COLORS = {"time.play_s": "#10b981", "time.old_pass_s": "#8b5cf6", "time.update_s": "#3b82f6", "time.reload_s": "#eab308",
                "time.idle_s": "#ef4444", "time.handoff_s": "#64748b"}
COLORS = ["#3b82f6", "#f97316", "#10b981", "#ec4899", "#8b5cf6", "#14b8a6", "#eab308", "#64748b"]
ARM = {"rl": "#e11d48", "pth": "#7c3aed", "sft": "#7c3aed"}


def record_ended(run):
    """The run's own record (docs/runs/<run>.json) has every ledger row settled: ended, whatever its exporter still writes."""
    try:
        rows = [x.get("row", {}) for x in json.loads((ROOT / "docs/runs" / f"{run}.json").read_text()).get("ledger", [])]
    except (OSError, ValueError):
        return False
    return bool(rows) and all(r.get("state") in ("done", "stopped", "completed", "ended", "failed", "cancelled", "canceled") for r in rows)


def load():
    runs = []
    if not SRC.exists():
        return runs
    now = time.time()
    try:
        qlive = paths.read_json(ROOT / "docs/modal-queue.json").get("live", [])
        END = ("done", "stopped", "completed", "ended", "failed", "cancelled", "canceled")
        queued_live = {e["run"] for e in qlive if e.get("state") not in END}
        row_ended = {e["run"] for e in qlive if e.get("state") in END}  # the queue says it ended: shown ended at once
        starts = {e["run"]: e["start_epoch"] for e in qlive if e.get("start_epoch")}
    except (OSError, ValueError):
        queued_live, starts, row_ended = None, {}, set()
    if queued_live is None:  # queue unreadable: trust the exporters' own status
        queued_live = _AllRuns()
    for f in sorted(SRC.glob("*.jsonl")):
        if f.name.endswith(".alerts.jsonl"):  # side file of alerts: merged into its run below, never a run of its own
            continue
        if not f.stem.startswith(TRAIN_RUN_PREFIXES):  # RL and OPD training runs only (the shared watcher writes rows for every run)
            continue
        rows, seen, by_t = [], set(), {}
        for g in (f, f.with_name(f.stem + ".alerts.jsonl")):
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
        meta.setdefault("_start", starts.get(meta["run"]))  # this run's own launch: earlier (backfilled) history plots at negative hours
        last = max(r.get("t", 0) for r in rows)
        if meta.get("_start") and last < meta["_start"] and meta["run"] in queued_live:
            rows.append({"t": meta["_start"], "m": {}})  # launched; only backfilled history so far
            last = meta["_start"]
            meta["what"] = (meta.get("what") or "") + " · launched, waiting for its first live metrics"
        if meta.get("status", "live") == "live" and (meta["run"] in row_ended or record_ended(meta["run"]) or (meta["run"] not in queued_live and now - last > STALE_S)):
            meta["status"] = "ended"  # exporter never wrote its end: the queue row ended, or no live row and no data for 30 min
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
    return re.search(r"(_frac|clear_rate)(_base|_target|_step)?($|\.)", k) is not None


GAP_S = 45 * 60  # pauses longer than this (a resume, backfilled pilot history) are cut out of the x axis


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
        if key == "train.approx_kl":
            for lk, nm in (("train.k3_target", "target"), ("train.k3_stop", "stop")):
                v = next((m[lk] for m in reversed(cur) if isinstance(m.get(lk), (int, float))), None)
                xs = [m["train.step_in_update"] for m in cur]
                if v and xs:
                    traces.append({"x": [1, max(xs)], "y": [v, v], "name": nm, "type": "scatter", "mode": "lines", "hoverinfo": "skip",
                                   "line": {"width": 1.5, "dash": "dot", "color": "#ef4444" if nm == "stop" else "#f59e0b"}})
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
                if y is not None:
                    pts.append((100 * g["tries"] / g["planned"], y, int(g["tries"])))
        return pts, pb

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
                nm = f"{'held-out' if held else 'training'} R{cur} ÷ R{p_} ({ck(ppb)})"
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
        figs.append({"section": "This round, live", "title": f"{title} · round {cur} (model {cur_ck}) vs earlier rounds, at the same point", "unit": "x", "single": None, "bar": False, "small": False,
                     "help": f"This round's running value ({levels}) divided by an earlier round's at the same share of tries done, so finish order "
                             f"(short tries first) cancels out. {better} The last point is the round's final ratio so far.",
                     "traces": traces, "rng": [min(0.5, min(ys) * 0.9), max(1.5, max(ys) * 1.1)] if ys else None, "legend": True,
                     "y2": False, "xlab": "% of the round's tries done", "dtick": 10})
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


def mini_figs(rows, t0, cap):
    """small, heavily smoothed side plots since the latest start (current leg): throughput, GPU util + VRAM
    (mean over cards), KV cache use, round time split."""
    rows = [r for r in rows if r.get("t", 0) >= (t0 or 0) and r.get("m")]
    if not rows:
        return []
    T0 = rows[0]["t"]

    def series(fn):
        pts = sorted((r["t"], v) for r in rows for v in [fn(r["m"])] if isinstance(v, (int, float)))
        return pts

    def mean_of(pre, scale=1.0):
        return lambda m: (sum(v for k, v in m.items() if k.startswith(pre) and isinstance(v, (int, float))) /
                          max(1, sum(1 for k, v in m.items() if k.startswith(pre) and isinstance(v, (int, float)))) * scale
                          if any(k.startswith(pre) for k in m) else None)

    def line(pts, name, color, win=1800):
        sm = _smooth(pts, win) if len(pts) > 3 else pts
        xs, ys = [], []
        for i, (t, v) in enumerate(sm):  # break over gaps > 20 min
            if xs and t - sm[i - 1][0] > 1200:
                xs.append(None)
                ys.append(None)
            xs.append(round((t - T0) / 3600, 3))
            ys.append(round(v, 2))
        return {"x": xs, "y": ys, "name": name, "type": "scatter", "mode": "lines", "line": {"width": 2, "color": color},
                "hovertemplate": name + " %{y:.3g}<extra></extra>"}

    figs = []
    gen = series(lambda m: m.get("infra.gen_tok_s"))
    trn = (series(lambda m: m.get("train.live_tok_s")) or series(lambda m: m.get("train.tok_s_live"))
           or series(lambda m: m.get("infra.train_tok_s")))
    tr = [line(p, n, c) for p, n, c in ((gen, "play", COLORS[0]), (trn, "train", COLORS[1])) if p]
    if tr:
        figs.append({"title": "Throughput (tok/s)", "traces": tr})
    util = series(mean_of("infra.gpu_util."))
    vram = series(mean_of("infra.vram_gib.", 100.0 / cap))
    tr = [line(p, n, c) for p, n, c in ((util, "util %", COLORS[2]), (vram, "VRAM %", COLORS[4])) if p]
    if tr:
        figs.append({"title": "GPU util + VRAM (mean over cards, %)", "traces": tr, "rng": [0, 105]})
    kv = series(lambda m: m["infra.kv_frac"] * 100 if isinstance(m.get("infra.kv_frac"), (int, float)) else None)
    if kv:
        figs.append({"title": "KV cache use (%)", "traces": [line(kv, "KV %", COLORS[5])], "rng": [0, 105]})
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


def run_view(meta, rows, last):
    clock = run_clock(rows, meta.get("_start"))
    ser = series(rows, clock)
    stages = [[round(clock[t], 3), lab] for t, lab in meta.get("_stages", []) if t in clock]
    own0 = meta.get("_own_start") or meta.get("_start")  # earlier legs' alerts go to the history line, not the live alert box
    roundat = {round(clock[r["t"]], 4): r["m"]["train.round"] for r in rows if isinstance((r.get("m") or {}).get("train.round"), (int, float))}
    if not ser:
        meta = dict(meta, what=(meta.get("what") or "") + " · waiting for its first metrics (container starting or queued for a GPU)")
    claimed, figs = set(), []
    is_rl = (meta.get("arm") or "").lower() == "rl" or meta["run"].startswith("rl")
    asyn = any("train.live_groups_total" in (r.get("m") or {}) for r in rows[-50:])  # async loop: no rounds to compare
    if is_rl:  # L1+L3 and pass/coverage details: superseded by the all-level and reference panels
        claimed.update(k for k in ser if fnmatch.fnmatch(k, "train.*.L13") or fnmatch.fnmatch(k, "train.*cov*") or k.startswith("play.live_")
                       or k in ("train.tries", "train.live_tokens", "train.live_reward_std", "train.live_w_abs_median", "train.live_gigpo_step_abs",
                                "infra.gpu_util.trainer_mean", "infra.peak_alloc_gib", "infra.card2_mode"))
    claimed.add("infra.alloc_after_load_gib")
    claimed.update(k for k in ser if "rhae_tok_vs_ref_base" in k or "rhae_tok_vs_ref_cov" in k or k.startswith("train.first_coverage"))
    for title, pats, unit, section, helptext in PANELS:  # unclaimed keys are not plotted (only useful plots)
        keys = [k for k in sorted(ser) if k not in claimed and any(fnmatch.fnmatch(k, p) for p in pats)
                and not any(fnmatch.fnmatch(k, h) for h in HIDE)]
        if not keys:
            continue
        claimed.update(keys)
        if title in ("Tokens per step", "Progress") or (is_rl and title in RL_DROP):  # constant batch size / step vs time: no information
            continue
        if title in BY_CKPT and not asyn:  # async: served versions lag and mix within a step, one dot per checkpoint misleads
            ck = ckpt_fig([r for r in rows if "train.played_by" in (r.get("m") or {})] + meta.get("_reads", []), keys, meta["run"])
            if ck:
                figs.append({"section": section, "title": title + " · by checkpoint", "unit": unit, "single": None, "bar": False, "small": False,
                             "help": helptext + " x axis: the checkpoint that played the round (r07 = after update 7); each dot is one read "
                             "(hover: run and round), the dotted line joins each checkpoint's mean.",
                             "traces": ck["traces"], "rng": ck["rng"], "legend": True, "y2": False, "ck": ck["cats"]})
                continue
        bar = title == "Round time split"
        top = {k: max(map(abs, ser[k][1]), default=0) for k in keys}
        big = max(top.values(), default=0)
        right = {k for k in keys if not bar and big > 0 and top[k] < big / 10}  # mixed scales: small series on a right axis
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
                    if not segs or x - segs[-1][-1][0] > 1 / 3:
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
                if k.endswith("_base") and k[:-5] in ser:
                    continue  # shown beside its metric
                base = ser.get(k + "_base")
                (x0, *_), (y0, *_) = ser[k]
                name = k.split(".", 1)[-1].replace("_", " ")
                if base:
                    b, yl = base[1][-1], ser[k][1][-1]
                    single.append([name, f"{b:.4g} → {yl:.4g}", f"base → now: {100 * (yl - b) / abs(b):+.1f}%" if b else "base → now"])
                elif len(ser[k][1]) == 1:
                    single.append([name, f"{y0:.4g}", f"one reading, at {x0:.2f} h; no base reported"])
                else:
                    y1 = ser[k][1][1]
                    single.append([name, f"{y0:.4g} → {y1:.4g}", f"{100 * (y1 - y0) / abs(y0):+.2f}% from {x0:.2f} h to {ser[k][0][1]:.2f} h" if y0 else ""])
        figs.append({"section": section, "title": title, "unit": unit, "help": helptext, "traces": traces, "single": single,
                     "bar": bar, "small": small, "rng": rng, "legend": len(traces) > 1, "y2": bool(right)})
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
    figs = [dict(f, section="Key plots") for f in key] + figs
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
    arm = meta.get("arm") or next((a for pre, a in (("rl", "rl"), ("sft-", "pth"), ("pth", "pth"), ("mfu-", "pth"), ("hx4", "envs"), ("envs", "envs")) if meta["run"].startswith(pre)), "?")
    ncards = max(len(card("infra.gpu_util.")), len(card("infra.vram_gib."))) or meta.get("cards", "?")  # the cards that report, not the meta guess
    return {"run": meta["run"], "arm": arm.upper(), "armc": ARM.get(arm.lower(), "#64748b"),
            "what": meta.get("summary") or short_what(meta.get("what", "")), "status": meta.get("status", "live"), "cards": ncards,
            "minis": mini_figs(rows, meta.get("_own_start") or meta.get("_start"), cap),
            "label": meta.get("label") or meta.get("stage") or clip(short_what(meta.get("summary") or meta.get("what") or ""), 60),
            "stages": stages, "last": last, "hours": round(clock[last], 2), "x0": round(min(clock.values()), 2), "chips": chips, "gauges": gauges, "cap": cap, "figs": figs,
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
.dot.live{background:var(--ok);box-shadow:0 0 0 3px rgba(22,163,74,.18)} .dot.stopped{background:var(--bad)}
main{padding:20px 24px 40px;min-width:0}
.rh h2{font-size:20px;margin:0 0 4px;font-weight:650;word-break:break-all} .rh .what{color:var(--mut);max-width:980px}
.rh .facts{color:var(--mut);font-size:12.5px;margin-top:6px}
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
<div class='top'><h1>Training runs</h1><span class='meta'>Updated __NOW__ UTC · rebuilt every 5 min · <a href='../../MODAL_STATUS.html'>&larr; What is running</a></span></div>
<div class='wrap'><nav id='nav'></nav><main id='main'></main></div>
<script>
const RUNS=__DATA__, NOW=__T__;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ago=t=>{const m=Math.round((NOW-t)/60);return m<60?m+' min ago':(m/60).toFixed(1)+' h ago'};
const hm=t=>new Date(t*1000).toISOString().slice(11,16);
function gauge(c,kind){const f=Math.max(0,Math.min(1,c[1])),L=Math.PI*26;  // semicircle gauge: util green when busy; memory green when full but not at the edge
 const col=kind==='util'?(f>=.8?'#16a34a':f>=.5?'#eab308':'#dc2626'):(f>=.99?'#dc2626':f>=.75?'#16a34a':'#eab308');
 return `<figure><svg width='76' height='44' viewBox='0 0 76 44'><path d='M8 38 A30 30 0 0 1 68 38' fill='none' stroke='var(--line)' stroke-width='7' stroke-linecap='round' pathLength='${L}'/>
 <path d='M8 38 A30 30 0 0 1 68 38' fill='none' stroke='${col}' stroke-width='7' stroke-linecap='round' pathLength='${L}' stroke-dasharray='${(L*f).toFixed(1)} ${L}'/>
 <text x='38' y='37' text-anchor='middle' font-size='13' font-weight='650' fill='var(--fg)'>${esc(c[2])}</text></svg><figcaption>${esc(c[0])}</figcaption></figure>`}
function nav(cur){let h='',grp=null;RUNS.forEach((r,i)=>{const g=r.status==='live'?'Live':'Recently ended';
 if(g!==grp){h+=`<h4>${g}</h4>`;grp=g}
 h+=`<a class='ri${i===cur?' on':''}' href='#${encodeURIComponent(r.run)}'><div class='n'><span class='pill' style='background:${r.armc}'>${esc(r.arm)}</span>${esc(r.label||r.run)}</div>${r.label?`<div class='s' style='word-break:break-all'>${esc(r.run)}</div>`:''}
 <div class='s'><span class='dot ${r.status}'></span>${r.status} · ${ago(r.last)}${r.gauges.length?' · GPU '+r.gauges[0][1].map(c=>c[2]).join('/'):r.chips.length?' · '+esc(r.chips[0][0])+' '+esc(r.chips[0][1]):''}</div></a>`});
 document.getElementById('nav').innerHTML=h||"<div class='empty'>No training run has exported metrics yet.</div>"}
function show(){const want=decodeURIComponent(location.hash.slice(1));let i=RUNS.findIndex(r=>r.run===want);if(i<0)i=0;nav(i);
 const m=document.getElementById('main');document.querySelectorAll('.fig,.mf').forEach(d=>Plotly.purge(d));
 if(!RUNS.length){m.innerHTML='';return}
 const r=RUNS[i];document.title=r.run+' · Training Live';
 let h=`<div class='rh'><h2><span class='pill' style='background:${r.armc}'>${esc(r.arm)}</span>${esc(r.run)}</h2><div class='what'>${esc(r.what)}</div>
 <div class='facts'><span class='dot ${r.status}'></span>${r.status} · last data ${ago(r.last)} · ${r.hours} h of run time · ${esc(r.cards)} card(s)</div></div><div class='body2'><div>`;
 let sec=null;r.figs.forEach((f,j)=>{if(f.section!==sec){if(sec)h+='</div>';h+=`<h3>${esc(f.section)}</h3><div class='grid'>`;sec=f.section}
  h+=`<div class='card'><div class='t'>${esc(f.title)}<span class='u'>${esc(f.unit)}</span></div>`+(f.single?`<div class='one'>`+f.single.map(x=>`<div class='l'>${esc(x[0])}</div><div class='v'>${esc(x[1])}</div><div class='l' style='margin-bottom:10px'>${esc(x[2])}</div>`).join('')+`</div>`:`<div class='fig' id='f${j}'></div>`)+`<div class='h'>${esc(f.help)}</div></div>`});
 if(sec)h+='</div>';
 h+=`</div><aside class='gcol'><div class='l' style='color:var(--mut);font-size:11.5px'>Since the latest start, smoothed</div>`+r.minis.map((f,j)=>`<div class='tile mini'><div class='l'>${esc(f.title)}</div><div class='mf' id='m${j}'></div></div>`).join('')+`</aside></div>`;
 m.innerHTML=h;window.scrollTo(0,0);
 const fg=css('--fg'),mut=css('--mut'),grid=css('--grid'),line=css('--line');
 r.minis.forEach((f,j)=>Plotly.newPlot('m'+j,f.traces,{margin:{l:34,r:6,t:4,b:f.traces.length>1||f.bar?40:20},font:{color:mut,size:10,family:'Inter,system-ui,sans-serif'},
  xaxis:f.bar?{type:'category',gridcolor:grid,linecolor:line}:{ticksuffix:' h',gridcolor:grid,linecolor:line,zeroline:false},yaxis:{gridcolor:grid,linecolor:line,zeroline:false,range:f.rng,rangemode:'tozero'},
  barmode:'group',bargap:.2,showlegend:f.traces.length>1,legend:{orientation:'h',y:-0.22,yanchor:'top',font:{size:10,color:fg}},hovermode:'x unified',
  hoverlabel:{bgcolor:css('--panel'),bordercolor:line,font:{size:11.5,color:fg}},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)'},{displayModeBar:false,responsive:true}));
 r.figs.forEach((f,j)=>f.single||Plotly.newPlot('f'+j,f.traces,{margin:{l:46,r:f.y2?46:10,t:6,b:f.legend?(f.xlab?96:48):24},font:{color:mut,size:11,family:'Inter,system-ui,sans-serif'},
  xaxis:f.ck?{type:'category',categoryorder:'array',categoryarray:f.ck,gridcolor:grid,linecolor:line}:f.xlab?{type:'linear',dtick:f.dtick||(f.traces.some(t=>t.x.length>12)?2:1),title:{text:f.xlab,font:{size:11}},gridcolor:grid,linecolor:line,zeroline:false}:{range:f.bar?undefined:[Math.min(r.x0,0),Math.max(r.hours,0.05,...(r.stages||[]).map(s=>s[0]+0.25))*1.02],ticksuffix:f.bar?'':' h',type:f.bar?'category':'linear',gridcolor:grid,linecolor:line,zeroline:false},
  yaxis:{gridcolor:grid,linecolor:line,zeroline:false,range:f.rng,exponentformat:f.small?'e':'none'},
  shapes:f.bar||f.ck||f.xlab?[]:(r.stages||[]).map(s=>({type:'line',x0:s[0],x1:s[0],yref:'paper',y0:0,y1:1,line:{color:mut,width:1,dash:'dot'}})),
  annotations:f.bar||f.ck||f.xlab?[]:(r.stages||[]).map((s,k)=>({x:s[0],yref:'paper',y:k%2?0.9:1,text:s[1],showarrow:false,xanchor:'left',yanchor:'top',xshift:2,font:{size:10,color:mut}})),
  barmode:'group',bargap:.25,showlegend:f.legend,legend:{orientation:'h',y:f.xlab?-0.42:-0.14,yanchor:'top',font:{size:11,color:fg}},
  hovermode:f.ck||f.xlab?'closest':'x unified',hoverlabel:{bgcolor:css('--panel'),bordercolor:css('--line'),font:{size:12.5,color:fg}},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  ...(f.y2?{yaxis2:{overlaying:'y',side:'right',showgrid:false,zeroline:false,rangemode:'tozero'}}:{})},
  {displayModeBar:false,responsive:true}))}
window.addEventListener('hashchange',show);show();
</script></body></html>"""


def build():
    runs = [run_view(m, rows, last) for m, rows, last in load()]
    page = (PAGE.replace("__NOW__", time.strftime("%Y-%m-%d %H:%M", time.gmtime())).replace("__T__", str(int(time.time())))
            .replace("__DATA__", json.dumps(runs, separators=(",", ":")).replace("</", "<\\/")))
    OUT.write_text(page)
    return len(runs)


if __name__ == "__main__":
    print(build(), "runs")
