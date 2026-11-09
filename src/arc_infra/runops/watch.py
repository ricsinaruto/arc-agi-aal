"""Shared run watcher rules: a run declares its plan; the watcher compares plan with
actual and wakes the owning agent. It never stops a run.

Pure logic: ``Watch(plan).check(snapshot)`` returns the alerts due now. ``tools/ops/watch.py`` gathers the snapshots
(metrics mirror, app/VM state, restart-guard state, log tails) and replays logs for backtests.

Plan (JSON; the "plan" block of the run record):
  run, arm, launched (epoch), boot_s (launch -> play, from a prior run's measured boot), cadence_s (metrics cadence),
  targets: {metric key or glob: planned steady value | {"plan": v, "steady": true}}; e.g. "infra.running" (live tries or
    streams), "infra.gen_tok_s", "infra.gpu_util.*" (%), "infra.vram_gib.*" (GiB). Checked at 10 and 20 min of play
    (median of the last 5 min against the plan, below ``RATE_FLOOR`` = alert). "steady" targets are also checked
    continuously after 20 min (15-min median).
  play_key: the metric whose first value > 0 marks play start (default: the first non-glob target).
  host: {"next_alloc_gib": the next big host allocation (e.g. a checkpoint's host copy), "min_avail_gib": 8,
    "total_gib": host RAM, when the run exports used RAM (infra.host_ram_gib) rather than available,
    "switch": {"phase_s", "switch_gib", "floor_gib"}: RL runs, during a play phase (``seq.phase`` 1): available host
    projected to the phase's end (slope over the last 10 min, at least 5 min of it; phase_s of play expected per
    phase) less the switch's own need (switch_gib) below floor_gib = alert "host-switch" (real-v7 27 Oct: vLLM's host
    heaps grow ~20 GiB per M tokens and the switch adds ~50 GiB)}.
  util_exempt: gpu_util keys outside rule (c) (a card idle by design between phases).
  rate_floor: the plan@ marks' and steady targets' floor (default ``RATE_FLOOR``; real-v5: 0.8). A steady
    target's {"phase_from_s": s} judges it only from s into the current play phase (kv_frac: each phase starts short).
  each: {key: {"min": v} | {"max": v}}: every new value of a per-phase key judged as it lands (switch times, setup,
    train.call_util: one row per phase or step, too few for the medians). With "sustain_s" and/or "confirm"
    ({rate key: fraction of its plan}) a sampled key fires only when every value of the last sustain_s (that much of
    history) is past the limit AND each confirm key's median there is under its fraction of plan (two signals; real-v11
    28 Oct: api_cpu 1.02 on one server at 2,900-3,300 tok/s vs plan 2,900 was no fault). {"after_s": s} judges only
    values from s after the first play value (not launch + boot_s), {"once": true} only the first of them (a read at a
    mark: arc2 fork runs' play.replies_per_attempt {"min": 3, "after_s": 600, "once": true}, 29 Oct).
  silence_s: metric silence threshold (default max(4 x cadence, 600 s)).
  counters: {cumulative metric key: seconds} (or a list, at ``freeze_s``): rule "frozen" fires when a counter stops
    advancing that long while rows keep arriving and work is in flight throughout (infra.running > 0 at every row of
    that span, or unreported: v15 29 Oct, a play start after a 21-min train phase is no hang); RL: 120 s (v17 29 Oct:
    a vLLM worker hang, gen 0 at 10 running, died at the 5-min RPC timeout); default
    {"infra.gen_tokens": freeze_s} (the arena adapter's engine token counter). A *tok_s target repeating one exact
    value of at least 25% of its plan for freeze_s fires it too (a status field copied from a hung engine).
  freeze_s: default max(300 s, 5 x cadence).
  progress_key: a training-progress counter (e.g. "train.step"): rule "progress" fires when it has not advanced for
    progress_k (default 2.5) x the median interval between its past advances (needs two advances; floor 120 s), while
    the app is up. Catches a hang at full GPU util (c7f: NCCL, 24 Oct).
  progress_pauses: [{"every": N} | {"at": [v, ...]}, "s": seconds]: designed pauses after a progress value (SFT held-out
    reads after step % N == 0 or at pass ends); a stall at a matching value gets the matching entries' seconds on top
    of its limit (sft a2r2t v3 / p75 2 Nov: slice + ARC-2 reads of 125-260 s fired the 4 x 38-52 s limit).
  expect: {metric key or glob: seconds after play start}: rule "expect" fires once when no value of that metric has
    arrived by then (a read the run must make, e.g. the step-0 base read "eval.heldout_rhae10k": its absence leaves the
    run with no anchor; 24 Oct v6e).
  starve (always on when the plan targets a live count and a gen rate): the clock starts at server ready (first
    ``ready_key`` value > 0, default "infra.gen_tok_s"), not launch; from 5 min after ready, fires when the 5-min mean of
    ``live_key`` (default the play_key, e.g. "infra.running") stays under 50% of its plan AND gen tok/s stays under 50%
    of its plan at every sample of the last 5 min, checked from 5 to 15 min after ready (two signals; quiet during boot,
    a ramp that crosses half plan, and later phase lulls). 24 Oct: hero-v6/v6b starved ~10 min before plan@10m.
  no_play (default on with an "arena" sampler source; false turns it off; {"passes": 2, "util": 30}): past launch +
    boot_s, while the file is still written (newest row <= 3 cadences old), the last ``passes`` rows each show no play:
    gen tok/s under 5% of its plan (0 without a plan) with no non-exempt card at ``util`` % or more (a re-prefill
    runs the card at full util), or, on a row without a gen rate, every non-exempt card under ``util`` %. Catches a run dead after boot ~2 min in (t3-v1-attempt2, 26 Oct) instead of at its silence. A loop
    whose server idles for more than ``passes`` rows between play phases sets ``passes`` above that.
  phases (RL loops; rows with ``seq.phase``, 1 play / 0 train, from the feed adapter): play metrics (gen, running,
    waiting, KV, GPU util) inside a train phase are left out of the play rules (marks, rate, util, frozen, starve,
    no-play), and a running train phase is judged by its own rules: "train-long" (the phase past ``train_max_s``, or
    past 2x the longest measured train phase once one has ended) and "train-idle" (no train progress row,
    seq.train_rows / train.step, for ``train_row_s`` (default 600; before the run's first progress row, the model
    load, ``train_load_s``, default 1500) AND every non-exempt card under 30% util for the last two rows: two signals). "train-cold": mean util of the
    busiest non-exempt card under 30% over ``train_cold_s`` (default 900) with no progress row in that span. "phase-churn": 3 or more play starts within 5 min (phases closing as fast as they
    open: t3-v3's empty failed groups, 26 Oct, invisible to the per-phase rules).
  servers (N-card RL runs; per-server rows ``infra.running.s<port>``, ``infra.gen_tok_s.s<port>``,
    ``infra.gen_tokens.s<port>`` from the sampler's vllm_servers, two or more ports): "starve:<port>" (that server's
    running AND gen tok/s under 50% of its 1/N plan share, same clock as starve), "no-play:<port>" (its gen tok/s
    ~0 AND its card's util under the floor for the last ``passes`` rows while the run as a whole plays; card =
    0 for :1234, port - 1235 otherwise) and "frozen:infra.gen_tokens.s<port>" (its counter flat for freeze_s while it
    has requests running). The unsuffixed keys stay the totals.
  confirm: {rate key: [keys]}: the plan@ marks and the steady "rate" rule fire for that key only when one of the
    listed keys (each with its own target) is also under ``RATE_FLOOR`` of its plan over the same window (two
    signals; arc2 conv1 27 Oct: lumpy reply-completion tokens read 801 tok/s at 95% util). Or {rate key: {key:
    {"max": v} | {"min": v}}}: only when every listed key's mean over the window is under max / over min (RL:
    {"infra.running": {"infra.kv_frac": {"max": 0.7}, "infra.waiting": {"min": 0}}}, the KV gate binding is no fault).
  boot: no play past 1.5 x boot_s, with a second signal: a good read showing none, or reads failing for READ_FAIL_S
    (a boot that never writes its file). Never after play was seen; a failed read keeps the last good rows.
  rate_window_s (arc2 roster feed, tools/ops/watch.py): the token rate of each row is taken over at least this many
    seconds of snapshots (default: between consecutive snapshots).
  follow_record: true: launched and app_id come from the run record's current attempt (records.py reserve of an ended
    run = attempt N+1), so one plan and one watcher cover every relaunch; rule state resets at each new attempt.

Metric rows: {"t": epoch, "m": {key: number}, "alert": text (optional)}, as tools/ops/train_dash.py reads them.
"""
import fnmatch
import re
import statistics
from dataclasses import dataclass, field

RATE_FLOOR = 0.7  # rule (c): a run below 70% of its planned rate at 20 min
UTIL_FLOOR = 30.0  # rule (c): a B300 below 30% util for 15 min outside a known phase
UTIL_WINDOW_S = 900
HOST_MARGIN = 1.2  # host available below 1.2 x the next big allocation
READ_FAIL_S = 900  # failed metric reads this long count as a boot signal; shorter is a blip of the read itself
REMIND_S = 1800  # an alert that stays true is repeated every 30 min
MARKS_MIN = (10, 20)
CRASH = re.compile(r"SyntaxError|IndentationError|ImportError|ModuleNotFoundError|CalledProcessError|"
                   r"CUDA out of memory|OutOfMemoryError|\bKilled\b|Segmentation fault|RUNOPS INCIDENT|"
                   r"Traceback \(most recent call last\)|^\w*(Error|Exception): ")
WARMUP_FALLBACK = '"event": "keep_layers_down"'  # info until the run's first "step" event; an alert after it
# a run's own clean end (plan "end_marker" adds to it); a run that goes without one is an alert, not an END (24 Oct)
END_MARKER = r"RUNOPS END ok|\bclean end\b|\bRUN COMPLETE\b|\btraining complete\b"


PLAN_KEYS = {"run", "arm", "launched", "boot_s", "cadence_s", "targets", "play_key", "host", "util_exempt", "silence_s",
             "platform", "app_id", "vm", "metrics_remote", "runops_state", "logs", "owner_agent", "arena", "counters",
             "freeze_s", "end_marker", "progress_key", "progress_k", "follow_record", "expect", "ready_key",
             "live_key", "no_play", "train_max_s", "train_row_s", "train_load_s", "train_cold_s", "rate_floor", "each", "metrics_lag_s", "confirm", "rate_window_s",
             "progress_pauses"}


def validate_plan(plan):
    """Raises ValueError on a plan the watcher cannot judge; returns the plan (a run record's "plan" block is
    accepted whole: {"plan": {...}})."""
    plan = plan.get("plan", plan) if isinstance(plan, dict) else plan
    missing = ({"run", "boot_s"} if plan.get("follow_record") else {"run", "launched", "boot_s"}) - set(plan)
    unknown = set(plan) - PLAN_KEYS
    if missing or unknown:
        raise ValueError(f"plan: missing {sorted(missing)}, unknown {sorted(unknown)}")
    for k, v in (plan.get("targets") or {}).items():
        target, _ = _target(v)
        if not isinstance(target, (int, float)) or target <= 0:
            raise ValueError(f"plan target {k}: {v!r}")
    if plan.get("platform") == "modal" and not plan.get("follow_record") and not str(plan.get("app_id", "")).startswith("ap-"):
        raise ValueError("plan: a Modal run needs app_id (ap-...)")
    return plan


@dataclass
class Snapshot:
    now: float
    rows: list  # metric rows, any order
    live: bool | None = None  # app / VM up (None: unknown)
    remote_age: float | None = None  # seconds since the run's own remote metrics/progress file changed
    beat_age: float | None = None  # seconds since the restart guard's last heartbeat
    incidents: list = field(default_factory=list)  # restart-guard incidents
    log_text: str = ""  # tail of the run's logs
    ended: bool = False  # the run record says ended / stopped
    clean_end: bool | None = None  # the run's own clean-end evidence (guard status ok, end marker, final.json ok)
    log_age: float | None = None  # seconds since the freshest of the plan's remote logs changed (None: unknown)
    read_failed: bool = False  # this pass could not read a metrics source (its rows are the last good ones, or none)


@dataclass
class Alert:
    rule: str
    text: str


def _target(spec):
    return (spec, False) if isinstance(spec, (int, float)) else (spec["plan"], bool(spec.get("steady")))


class Watch:
    def __init__(self, plan):
        self.plan = plan
        self.run = plan["run"]
        self.targets = {k: _target(v) for k, v in (plan.get("targets") or {}).items()}
        self.target_opts = {k: v for k, v in (plan.get("targets") or {}).items() if isinstance(v, dict)}
        self.rate_floor = float(plan.get("rate_floor", RATE_FLOOR))
        self.each_seen = {}  # "each" rule: key -> time of the last value judged
        plain = [k for k in self.targets if not any(c in k for c in "*?[")]
        self.play_key = plan.get("play_key") or (plain[0] if plain else None)
        cadence = float(plan.get("cadence_s", 60))
        self.silence_s = float(plan.get("silence_s") or max(4 * cadence, 600))
        self.boot_s = float(plan.get("boot_s", 900))
        self.freeze_s = float(plan.get("freeze_s") or max(300, 5 * cadence))
        c = plan.get("counters", ["infra.gen_tokens"])
        self.counters = {k: float(v) for k, v in c.items()} if isinstance(c, dict) else dict.fromkeys(c, self.freeze_s)
        self.active = {}  # rule -> last fired time
        self.marks_done = set()
        self.seen = set()  # incidents, run alerts, crash lines already relayed
        self.played = False  # play was seen this attempt: boot is over, whatever a later read returns
        self.read_fail_since = None  # unbroken failed reads began here

    # --- helpers ---------------------------------------------------------------------------------------------
    def _fire(self, out, now, rule, cond, text):
        if not cond:
            self.active.pop(rule, None)
            return
        last = self.active.get(rule)
        if last is None or now - last >= REMIND_S:
            self.active[rule] = now
            out.append(Alert(rule, text))

    def _once(self, out, key, rule, text):
        key = tuple(map(str, key))  # as restored from state()
        if key not in self.seen:
            self.seen.add(key)
            out.append(Alert(rule, text))

    @staticmethod
    def _series(rows, pattern):
        """{key: [(t, v)]} for keys matching ``pattern``."""
        out = {}
        for r in rows:
            for k, v in (r.get("m") or {}).items():
                if isinstance(v, (int, float)) and fnmatch.fnmatchcase(k, pattern):
                    out.setdefault(k, []).append((r["t"], float(v)))
        return out

    @staticmethod
    def _window(points, start, end, carry=True):
        vals = [v for t, v in points if start <= t <= end]
        if not vals and carry:  # carry the last value forward (a gauge reported only on change)
            prior = [v for t, v in points if t <= end]
            vals = prior[-1:]
        return vals

    def _value(self, rows, pattern, start, end, agg=statistics.median, min_points=1, carry=True):
        """The pattern's value over [start, end]: per key ``agg``, averaged over matching keys (cards)."""
        per = [agg(w) for w in (self._window(p, start, end, carry) for p in self._series(rows, pattern).values())
               if len(w) >= min_points]
        return sum(per) / len(per) if per else None

    def _confirmed(self, rows, key, start, end):
        """Plan ``confirm``: a second listed key is also under the floor of its own plan in the window (or none set).
        A dict {key: {"max": v} | {"min": v}} instead: every listed key's window mean is under max / over min
        (real-v11 28 Oct: running 237 vs 304 with KV at 0.79 and gen on plan is the KV gate binding as contexts grow;
        low running is a fault only with KV < 0.7 and requests waiting)."""
        keys = (self.plan.get("confirm") or {}).get(key)
        if not keys:
            return True
        if isinstance(keys, dict):
            for other, lim in keys.items():
                v = self._value(rows, other, start, end, agg=statistics.mean)  # mean: any waiting counts
                if v is None or (lim.get("max") is not None and not v < float(lim["max"])) or (
                        lim.get("min") is not None and not v > float(lim["min"])):
                    return False
            return True
        for other in keys:
            target = self.targets.get(other)
            v = self._value(rows, other, start, end) if target else None
            if v is not None and v < self.rate_floor * target[0]:
                return True
        return False

    def play_start(self, rows):
        launched = float(self.plan.get("launched") or (rows[0]["t"] if rows else 0))
        if self.play_key:
            first = next((t for t, v in self._series(rows, self.play_key).get(self.play_key, []) if v > 0), None)
        else:
            first = rows[0]["t"] if rows else None
        return first, (None if first is None else max(first, launched + self.boot_s)), launched

    # --- rules -----------------------------------------------------------------------------------------------
    def check(self, s):
        out, now = [], s.now
        # only this attempt's rows: a resumed run appends to its earlier attempt's file (sft-p75-qad-v2 27 Oct: attempt
        # 2's boot read attempt 1's last step as "stuck 370 min" and its last row as "silent 364 min")
        since = float(self.plan["launched"]) if self.plan.get("launched") is not None else float("-inf")
        rows = sorted((r for r in s.rows if r.get("t") is not None and since <= r["t"] <= now), key=lambda r: r["t"])
        all_rows, trains = rows, self._train_phases(rows, now)
        rows = self._play_only(rows, trains)  # the play rules below never see a train phase's idle servers
        first_play, play, launched = self.play_start(rows)

        if s.live is False and not s.clean_end:
            self._once(out, ("unclean-end",), "unclean-end",
                       f"app/VM gone before a clean end ({(now - launched) / 60:.1f} min after launch; no end marker, "
                       f"guard status or final.json): treat as a crash")
        self._fire(out, now, "gone", s.live is False and not s.ended and bool(s.clean_end),
                   "app/VM ended cleanly but the run record is still open: settle it (records.py settle)")
        what = f"'{self.play_key}' > 0" if self.play_key else "metrics"
        self.played = self.played or first_play is not None or bool(self.marks_done)
        self.read_fail_since = (self.read_fail_since if self.read_fail_since is not None else now) if s.read_failed else None
        # two signals (arc3sub 30 Oct: a failed read fired [boot] 1,000 min into a healthy run): no play past the deadline,
        # and either a good read that shows none or reads failing for READ_FAIL_S (a boot that never writes its file)
        believed = not s.read_failed or now - self.read_fail_since >= READ_FAIL_S
        self._fire(out, now, "boot", not self.played and now > launched + 1.5 * self.boot_s and believed,
                   f"no {what} {(now - launched) / 60:.0f} min after launch (plan boot {self.boot_s / 60:.0f} min)"
                   + (f"; metrics unreadable for {(now - self.read_fail_since) / 60:.0f} min" if s.read_failed else ""))

        self._starve(out, now, rows)  # clocked on server ready, not launch + boot
        if play is not None:
            for mark in MARKS_MIN:
                # marks count play time: an rl_seq train phase (servers asleep) stops the clock (H4 26 Oct: plan@10m
                # read a window that was 13 s of play tail + 5 min of train phase after a 298 s play phase)
                end = self._play_clock(play, trains, 60 * mark)
                if end is None or now < end or mark in self.marks_done:
                    continue
                self.marks_done.add(mark)
                for key, (target, _) in self.targets.items():
                    agg = max if key.startswith("infra.vram") else statistics.median
                    begin = self._play_clock(play, trains, 60 * mark - 300)
                    after = (self.target_opts.get(key) or {}).get("phase_from_s")
                    if after is not None:  # kv_frac: a lockstep phase start ramps (real-v12 plan@10m read min 5-10)
                        ends = [b for a, b in trains if b is not None and b <= end]
                        begin = max(begin, max(ends + [play]) + float(after))
                        if begin >= end:
                            continue
                    v = self._value(rows, key, begin, end, agg)
                    if v is None:
                        out.append(Alert(f"plan@{mark}m", f"{key}: no data at {mark} min of play (plan {target:g})"))
                    elif v < self.rate_floor * target and self._confirmed(rows, key, begin, end):
                        out.append(Alert(f"plan@{mark}m", f"{key} {v:.4g} vs plan {target:g} at {mark} min of play "
                                                          f"({v / target:.0%} < {self.rate_floor:.0%})"))
            for key, secs in (self.plan.get("expect") or {}).items():
                if now >= play + float(secs) and not self._series(rows, key):
                    self._once(out, ("expect", key), f"expect:{key}",
                               f"no {key} {(now - play) / 60:.0f} min into play (expected within {float(secs) / 60:.0f}"
                               f" min): the run has no such read")
            if now >= play + 1200:
                for key, (target, steady) in self.targets.items():
                    if steady:
                        begin, need = now - 900, 600
                        after = (self.target_opts.get(key) or {}).get("phase_from_s")
                        if after is not None:  # judged only from this long into the current play phase (kv_frac)
                            ends = [b for a, b in trains if b is not None and b <= now]
                            begin, need = max(begin, max(ends + [play]) + float(after)), 300
                        fresh = [t for pts in self._series(rows, key).values() for t, _ in pts if t >= begin]
                        spans = bool(fresh) and max(fresh) - min(fresh) >= need  # a stale window is the silence rule's
                        v = self._value(rows, key, begin, now, min_points=5, carry=False) if spans else None
                        low = v is not None and v < self.rate_floor * target and self._confirmed(rows, key, begin, now)
                        self._fire(out, now, f"rate:{key}", low,
                                   f"{key} 15-min median {v if v is None else round(v, 3)} vs plan {target:g}")
            exempt = set(self.plan.get("util_exempt") or [])
            for key, pts in self._series(rows, "infra.gpu_util.*").items():
                recent = [(t, v) for t, v in pts if t >= now - UTIL_WINDOW_S]
                spans = recent and recent[-1][0] - recent[0][0] >= 0.66 * UTIL_WINDOW_S and recent[0][0] >= play
                low = bool(spans) and key not in exempt and statistics.mean(v for _, v in recent) < UTIL_FLOOR
                self._fire(out, now, f"util:{key}", low, f"{key} mean {statistics.mean(v for _, v in recent):.0f}% "
                           f"over 15 min (< {UTIL_FLOOR:.0f}%)" if low else "")

            self._frozen(out, now, rows)
        self._each(out, now, all_rows, first_play)

        self._progress(out, now, all_rows, s)
        if not (trains and trains[-1][1] is None):  # no play is expected inside a running train phase
            self._no_play(out, now, rows, launched)
        self._train(out, now, all_rows, trains)

        if rows:
            age = now - rows[-1]["t"]
            stale = age > self.silence_s and not s.ended
            second = [n for n, a in (("remote file", s.remote_age), ("heartbeat", s.beat_age)) if a is not None
                      and a > self.silence_s] + (["app/VM down"] if s.live is False else [])
            alive_log = s.log_age is not None and s.log_age <= self.silence_s and s.live is not False
            # a fresh run log means the metrics file syncs late (Nebius uploads it less often), not a silent run
            self._fire(out, now, "silent", stale and bool(second) and not alive_log,
                       f"no metrics for {age / 60:.0f} min; confirmed by {', '.join(second)}")
            self._fire(out, now, "feed", stale and s.remote_age is not None and s.remote_age <= self.silence_s,
                       f"local metrics {age / 60:.0f} min old but the remote file is fresh: the mirror/feed is broken")

        host = self.plan.get("host") or {}
        avail = self._last(rows, "infra.host_avail_gib")
        if avail is None and host.get("total_gib") and self._last(rows, "infra.host_ram_gib") is not None:
            avail = host["total_gib"] - self._last(rows, "infra.host_ram_gib")
        nxt = self._last(rows, "infra.next_alloc_gib") or host.get("next_alloc_gib")
        self._fire(out, now, "host-ram", avail is not None and bool(nxt) and avail < HOST_MARGIN * nxt,
                   f"host available {avail} GiB < {HOST_MARGIN} x next allocation {nxt} GiB")
        floor = host.get("min_avail_gib", 8)
        self._fire(out, now, "host-floor", avail is not None and avail < floor,
                   f"host available {avail} GiB below the {floor} GiB floor")
        if isinstance(host.get("switch"), dict):
            proj = self._switch_projection(rows, now, host)
            self._fire(out, now, "host-switch", proj is not None and proj["at_switch"] < proj["floor"],
                       "host available at this phase's switch projected {at_switch:.0f} GiB < {floor:.0f} (now "
                       "{avail:.0f}, {slope:+.1f} GiB/min, {left:.0f} s of play left, switch {switch:.0f} GiB)"
                       .format(**proj) if proj else "")

        errs = [(r["t"], r["m"]["play.errors"]) for r in rows if isinstance((r.get("m") or {}).get("play.errors"), (int, float))]
        recent = [e for t, e in errs if t >= now - 600]
        rising = len(errs) > 1 and bool(recent) and recent[-1] > min([e for t, e in errs if t < now - 600] or [0])
        self._fire(out, now, "failures", rising, f"play failures/request errors rising: {errs[-1][1]:.0f} now"
                   if errs else "")

        for inc in s.incidents:
            self._once(out, ("incident", inc.get("id")), "restart",
                       f"{inc.get('id')} cause={inc.get('cause')} blocked={inc.get('blocked')} why={inc.get('why')}")
        for r in rows:
            if r.get("alert") and not str(r["alert"]).startswith("watch:"):
                self._once(out, ("row", r["t"], r["alert"]), "run-alert", str(r["alert"])[:300])
        stepped = False  # SFT trainer: keep_layers_down before the first step is the warm-up's designed memory fallback
        for line in s.log_text.splitlines():
            stepped = stepped or '"event": "step"' in line
            if CRASH.search(line) and not (WARMUP_FALLBACK in line and not stepped):
                self._once(out, ("crash", line.strip()), "crash", line.strip()[:300])
        return out

    PLAY_KEYS = ("infra.gen_tok_s", "infra.gen_tokens", "infra.running", "infra.waiting", "infra.kv_frac",
                 "infra.gpu_util.*", "infra.gen_tok_s.s*", "infra.gen_tokens.s*", "infra.running.s*",
                 "infra.waiting.s*", "infra.kv_frac.s*")

    def _servers(self, rows):
        """Ports with per-server play metrics (``<key>.s<port>``, N-card RL runs), sorted; [] on one server."""
        ports = {k.rsplit(".s", 1)[1] for k in self._series(rows, "infra.running.s*")}
        ports |= {k.rsplit(".s", 1)[1] for k in self._series(rows, "infra.gen_tok_s.s*")}
        return sorted(ports, key=lambda p: int(p) if p.isdigit() else 0) if len(ports) > 1 else []

    def _train_phases(self, rows, now):
        """[(start, end or None)] of train phases from the ``seq.phase`` rows."""
        out = []
        for t, v in self._series(rows, "seq.phase").get("seq.phase", []):
            if v == 0 and (not out or out[-1][1] is not None):
                out.append((t, None))
            elif v == 1 and out and out[-1][1] is None:
                out[-1] = (out[-1][0], t)
        return out

    @staticmethod
    def _play_clock(play, trains, seconds):
        """The time at which ``seconds`` of play (time outside train phases) have passed since ``play``; None while a
        train phase that has not ended holds the clock short of it."""
        t, left = play, float(seconds)
        for a, b in sorted(trains):
            if (b is not None and b <= t):
                continue
            if a > t:
                if t + left <= a:
                    return t + left
                left -= a - t
            if b is None:
                return None
            t = max(t, b)
        return t + left

    def _play_only(self, rows, trains):
        if not trains:
            return rows
        inside = lambda t: any(a <= t and (b is None or t < b) for a, b in trains)
        out = []
        for r in rows:
            if inside(r["t"]):
                m = {k: v for k, v in (r.get("m") or {}).items()
                     if not any(fnmatch.fnmatchcase(k, p) for p in self.PLAY_KEYS)}
                r = {**r, "m": m}
            out.append(r)
        return out

    def _each(self, out, now, rows, play=None):
        """Plan ``each``: {key: {"min": v} | {"max": v}}: every new value of a per-phase key (one row per phase or
        step: switch times, setup, the update's whole-call util) judged as it lands."""
        for key, lim in (self.plan.get("each") or {}).items():
            pts = [(t, v) for t, v in self._series(rows, key).get(key, []) if t <= now]
            if lim.get("sustain_s") or lim.get("confirm"):
                self._each_sustained(out, now, rows, key, lim, pts)
                continue
            if lim.get("after_s") is not None:
                if play is None:
                    continue
                pts = [(t, v) for t, v in pts if t >= play + float(lim["after_s"])]
            new = [(t, v) for t, v in pts if t > self.each_seen.get(key, float("-inf"))]
            if lim.get("once") and key in self.each_seen:
                new = []
            new = new[:1] if lim.get("once") else new
            if not new:
                continue
            self.each_seen[key] = new[-1][0]
            for t, v in new:
                if lim.get("min") is not None and v < float(lim["min"]):
                    out.append(Alert(f"each:{key}", f"{key} {v:.4g} < min {float(lim['min']):g}"))
                elif lim.get("max") is not None and v > float(lim["max"]):
                    out.append(Alert(f"each:{key}", f"{key} {v:.4g} > max {float(lim['max']):g}"))

    def _each_sustained(self, out, now, rows, key, lim, pts):
        win = float(lim.get("sustain_s") or 0)
        recent = [v for t, v in pts if t >= now - win]
        if lim.get("max") is not None:
            past = [v > float(lim["max"]) for v in recent]
        else:
            past = [v < float(lim["min"]) for v in recent]
        bad = bool(past) and all(past) and bool(pts) and pts[0][0] <= now - win
        why = []
        for ck, frac in (lim.get("confirm") or {}).items():
            plan = (self.targets.get(ck) or (0, False))[0]
            vals = [v for t, v in self._series(rows, ck).get(ck, []) if now - max(win, 300) <= t <= now]
            med = statistics.median(vals) if vals else None
            if not plan or med is None or med >= float(frac) * plan:
                bad = False
                break
            why.append(f"{ck} median {med:.4g} < {float(frac):g} x plan {plan:g}")
        lim_txt = f"> max {float(lim['max']):g}" if lim.get("max") is not None else f"< min {float(lim['min']):g}"
        self._fire(out, now, f"each:{key}", bad, f"{key} {lim_txt} for {win:.0f} s" + (f" and {'; '.join(why)}" if why
                                                                                           else ""))

    def _train(self, out, now, rows, trains):
        starts = [t for t, v in self._series(rows, "seq.phase").get("seq.phase", []) if v == 1 and now - 300 <= t <= now]
        self._fire(out, now, "phase-churn", len(starts) >= 3, f"{len(starts)} play phases started in the last 5 min: "
                   "phases close as fast as they open (failing groups?)" if len(starts) >= 3 else "")
        live = trains[-1] if trains and trains[-1][1] is None else None
        done = [b - a for a, b in trains if b is not None]
        ceiling = 2 * max(done) if done else self.plan.get("train_max_s")
        long = live is not None and ceiling is not None and now - live[0] > float(ceiling)
        self._fire(out, now, "train-long", long, f"train phase running {(now - live[0]) / 60:.0f} min (ceiling "
                   f"{float(ceiling) / 60:.0f} min: {'2x the longest measured phase' if done else 'plan train_max_s'})"
                   if long else "")
        idle = False
        if live is not None:
            prog = [t for k in ("seq.train_rows", "train.step") for t, _ in self._series(rows, k).get(k, [])]
            last = max([t for t in prog if t >= live[0]] or [live[0]])
            exempt = set(self.plan.get("util_exempt") or [])
            util = list({r["t"]: r for r in rows if r["t"] >= live[0]  # one row per sample time (v15 29 Oct 13:26:
                         and any(k.startswith("infra.gpu_util.") for k in r.get("m") or {})}.values())[-2:]  # an
            # exported 0-util sample twice was "two cold reads" mid-accumulate at 100% util: two distinct samples)
            cold = len(util) == 2 and all(max(v for k, v in r["m"].items() if k.startswith("infra.gpu_util.")
                                              and k not in exempt) < UTIL_FLOOR for r in util)
            # before the run's first progress row the trainer is still loading the model (cards cold ~10-13 min:
            # U4 / U4b 27 Oct): train_load_s from the phase start, then train_row_s per row
            loading = not prog
            limit = float(self.plan.get("train_load_s", 1500) if loading else self.plan.get("train_row_s", 600))
            idle = now - last > limit and cold
            self._fire(out, now, "train-idle", idle, f"train phase: no progress row for {(now - last) / 60:.0f} min "
                       f"{'(trainer load) ' if loading else ''}and the cards under {UTIL_FLOOR:.0f}% util: trainer "
                       "hang?" if idle else "")
            # compute (c): 15 min under 30% in a train phase, whatever train_row_s allows. A mean over the
            # window, not two samples: a10's old pass has 22-32% of its 1-s samples under 30 at a 64-72% mean
            span = float(self.plan.get("train_cold_s", 900))
            seen = {r["t"]: max(v for k, v in r["m"].items() if k.startswith("infra.gpu_util.") and k not in exempt)
                    for r in rows if now - span <= r["t"] <= now and r["t"] >= live[0]
                    and any(k.startswith("infra.gpu_util.") and k not in exempt for k in r.get("m") or {})}
            mean = sum(seen.values()) / len(seen) if seen else None
            frozen = (not loading and last <= now - span and now - live[0] >= span and len(seen) >= 4
                      and now - min(seen) >= 0.8 * span and mean < UTIL_FLOOR)
            self._fire(out, now, "train-cold", frozen, f"train phase: cards at {mean:.0f}% mean util for "
                       f"{span / 60:.0f} min with no progress row: trainer hang?" if frozen else "")

    def _no_play(self, out, now, rows, launched):
        cfg = self.plan.get("no_play", bool((self.plan.get("arena") or {}).get("sampler")))
        if not cfg:
            return
        cfg = cfg if isinstance(cfg, dict) else {}
        passes, floor = int(cfg.get("passes", 2)), float(cfg.get("util", UTIL_FLOOR))
        gen_plan = (self.targets.get("infra.gen_tok_s") or (0, False))[0]
        exempt = set(self.plan.get("util_exempt") or [])
        cadence = float(self.plan.get("cadence_s", 60))
        recent = [r for r in rows if r["t"] >= launched + self.boot_s and any(
            k == "infra.gen_tok_s" or k.startswith("infra.gpu_util.") for k in r.get("m") or {})][-passes:]

        def idle(m):  # two signals: gen rate low and no busy card; v15 29 Oct: a phase-start
            # re-prefill ran at gen 96 tok/s with the card at 94-100% util for 5 min and fired
            gen = m.get("infra.gen_tok_s")
            util = [v for k, v in m.items() if k.startswith("infra.gpu_util.") and k not in exempt]
            cold = not util or max(util) < floor
            if gen is not None:
                return (gen <= 0.05 * gen_plan if gen_plan else gen <= 0) and cold
            return bool(util) and cold

        low = (len(recent) == passes and now - recent[-1]["t"] <= 3 * cadence and all(idle(r["m"]) for r in recent))
        self._fire(out, now, "no-play", low, f"no play for {passes} passes past boot ({(now - launched) / 60:.0f} min "
                   f"after launch): gen tok/s {round(recent[-1]['m'].get('infra.gen_tok_s') or 0)}, util "
                   f"{[round(v) for k, v in sorted(recent[-1]['m'].items()) if k.startswith('infra.gpu_util.')]}"
                   if low else "")
        # N cards (H2 26 Oct): one server idle while the others play. Two signals per server: its own gen rate (0, or
        # under 5% of its plan share) and its card's util under the floor, for the last ``passes`` rows
        ports = self._servers(rows)
        for port in ports:
            card = f"infra.gpu_util.{0 if int(port) == 1234 else int(port) - 1235}"
            gen_key = f"infra.gen_tok_s.s{port}"
            mine = [r for r in rows if r["t"] >= launched + self.boot_s and gen_key in (r.get("m") or {})][-passes:]

            def quiet(m):
                gen = m[gen_key]
                cold = card not in exempt and m.get(card) is not None and m[card] < floor
                return (gen <= 0.05 * gen_plan / len(ports) if gen_plan else gen <= 0) and cold

            alone = (len(mine) == passes and now - mine[-1]["t"] <= 3 * cadence and all(quiet(r["m"]) for r in mine)
                     and not low)  # the whole run idle is the no-play rule's
            self._fire(out, now, f"no-play:{port}", alone,
                       f"server :{port} idle for {passes} passes while the run plays: its gen tok/s "
                       f"{round(mine[-1]['m'][gen_key])}, {card} {round(mine[-1]['m'][card])}%" if alone else "")

    def _frozen(self, out, now, rows):
        """A hung engine behind a live status loop: rows keep coming, a cumulative counter stops (or a rate field
        repeats one exact value). Quiet when nothing is in flight (a drained run) or when rows themselves stop
        (the silence rule's case)."""
        if not rows or now - rows[-1]["t"] > self.silence_s:
            return
        running = self._last(rows, "infra.running")
        busy = running is None or running > 0
        since = None  # start of the current unbroken run of infra.running > 0 rows (a play start right after a
        for t, v in reversed(self._series(rows, "infra.running").get("infra.running", [])):  # train phase is no hang)
            if v <= 0:
                break
            since = t
        keys = dict(self.counters)
        rates = {k: t for k, (t, _) in self.targets.items() if k.endswith("tok_s") and k not in keys}
        keys.update(dict.fromkeys(rates, self.freeze_s))
        for key, limit in keys.items():
            pts = self._series(rows, key).get(key, [])
            frozen = False
            if pts and now - pts[-1][0] <= self.silence_s:
                changed = pts[0][0]
                for (_, a), (t, b) in zip(pts, pts[1:]):
                    if b != a:
                        changed = t
                if since is not None:
                    changed = max(changed, since)
                flat = [v for t, v in pts if t >= changed]
                counter = key in self.counters
                frozen = (busy and pts[-1][0] - changed >= limit and len(flat) >= 3
                          and (counter or flat[-1] >= 0.25 * rates[key]))  # a slow tail is the rate rule's
            self._fire(out, now, f"frozen:{key}", frozen,
                       f"{key} stuck at {pts[-1][1]:.6g} for {(pts[-1][0] - changed) / 60:.0f} min while rows keep "
                       f"arriving (running {running}): engine hang?" if frozen else "")
        for port in self._servers(rows):  # N cards: one server's engine hung while the others' tokens hide it
            key = f"infra.gen_tokens.s{port}"
            pts = self._series(rows, key).get(key, [])
            mine = self._last(rows, f"infra.running.s{port}")
            frozen = False
            if pts and now - pts[-1][0] <= self.silence_s and mine is not None and mine > 0:
                changed = pts[0][0]
                for (_, a), (t, b) in zip(pts, pts[1:]):
                    if b != a:
                        changed = t
                frozen = pts[-1][0] - changed >= self.freeze_s and sum(1 for t, _ in pts if t >= changed) >= 3
            self._fire(out, now, f"frozen:{key}", frozen,
                       f"server :{port} tokens stuck at {pts[-1][1]:.6g} for {(pts[-1][0] - changed) / 60:.0f} min "
                       f"with {mine:g} running: engine hang?" if frozen else "")

    def _starve(self, out, now, rows):
        live_key = self.plan.get("live_key") or self.play_key
        gen_key = self.plan.get("ready_key") or "infra.gen_tok_s"
        if live_key not in self.targets or gen_key not in self.targets:
            return
        gen = self._series(rows, gen_key).get(gen_key, [])
        floor = 0.05 * self.targets[gen_key][0]  # a warm-up probe's trickle (ovf3: 36 tok/s for 12 min) is not ready
        ready = next((t for t, v in gen if v > floor), None)
        if ready is None or not ready + 300 <= now <= ready + 900:  # the first play minutes; later lulls (a
            return                                                    # sync loop's train phase) are the rate rules'
        lives = [v for t, v in self._series(rows, live_key).get(live_key, []) if now - 300 <= t <= now]
        gens = [v for t, v in gen if now - 300 <= t <= now]
        live_plan, gen_plan = self.targets[live_key][0], self.targets[gen_key][0]
        cond = (len(lives) >= 2 and len(gens) >= 2 and max(lives) < 0.5 * live_plan  # every sample of the last 5 min
                and max(gens) < 0.5 * gen_plan)                                       # under half plan: not a ramp
        live_v, gen_v = (statistics.mean(lives) if lives else None), (statistics.mean(gens) if gens else None)
        self._fire(out, now, "starve", cond,
                   f"starved {(now - ready) / 60:.0f} min after server ready: {live_key} 5-min mean "
                   f"{live_v if live_v is None else round(live_v, 1)} < 50% of {live_plan:g} and {gen_key} "
                   f"{gen_v if gen_v is None else round(gen_v)} < 50% of {gen_plan:g}")
        ports = self._servers(rows)  # N cards: each server against its share of the plan (a starved card hides
        for port in ports:           # under the total while the others run full, H2 26 Oct)
            lv = [v for t, v in self._series(rows, f"{live_key}.s{port}").get(f"{live_key}.s{port}", []) if now - 300 <= t <= now]
            gv = [v for t, v in self._series(rows, f"{gen_key}.s{port}").get(f"{gen_key}.s{port}", []) if now - 300 <= t <= now]
            share = len(ports)
            low = (len(lv) >= 2 and len(gv) >= 2 and max(lv) < 0.5 * live_plan / share and max(gv) < 0.5 * gen_plan / share)
            self._fire(out, now, f"starve:{port}", low,
                       f"server :{port} starved: {live_key} 5-min max {max(lv) if lv else None:g} < 50% of its "
                       f"share {live_plan / share:g} and {gen_key} {round(max(gv)) if gv else None} < 50% of "
                       f"{gen_plan / share:g}" if low else "")

    def _progress(self, out, now, rows, s):
        key = self.plan.get("progress_key")
        if not key or s.live is False or s.ended:
            return
        pts = self._series(rows, key).get(key, [])
        adv = [t for (_, a), (t, b) in zip(pts, pts[1:]) if b > a]
        if pts:
            adv = [pts[0][0]] + adv
        gaps = [b - a for a, b in zip(adv, adv[1:])]
        if not gaps:
            return
        limit = max(120.0, float(self.plan.get("progress_k", 2.5)) * statistics.median(gaps))
        v = pts[-1][1]
        limit += sum(float(p.get("s", 0)) for p in self.plan.get("progress_pauses", [])
                     if (p.get("every") and v % p["every"] == 0) or v in p.get("at", []))
        # a metrics file Nebius uploads late (bootstrap.log current, train-metrics minutes behind: hspeed-a-v2, 26 Oct)
        # lags by up to metrics_lag_s (default one cadence on Nebius): that much of the time since the newest row is
        # not counted as a stall
        allow = self.plan.get("metrics_lag_s", float(self.plan.get("cadence_s", 60))
                              if self.plan.get("platform") == "nebius" and self.plan.get("metrics_remote") else 0.0)
        lag = min(max(0.0, now - max(r["t"] for r in rows)), float(allow))
        idle = now - adv[-1] - lag
        self._fire(out, now, f"progress:{key}", idle > limit,
                   f"{key} stuck at {pts[-1][1]:g} for {idle / 60:.1f} min (> {limit / 60:.1f} min = "
                   f"{self.plan.get('progress_k', 2.5)} x median step interval {statistics.median(gaps) / 60:.1f} min)")

    def state(self):
        """JSON-able rule state, so a restarted watcher does not re-fire or forget its marks."""
        return {"active": self.active, "marks_done": sorted(self.marks_done),
                "seen": [list(k) for k in self.seen], "each_seen": self.each_seen, "played": self.played,
                "read_fail_since": self.read_fail_since}

    def restore(self, st):
        self.active = dict(st.get("active") or {})
        self.marks_done = set(st.get("marks_done") or [])
        self.seen = {tuple(k) for k in st.get("seen") or []}
        self.each_seen = dict(st.get("each_seen") or {})
        self.played = bool(st.get("played") or self.marks_done)
        self.read_fail_since = st.get("read_fail_since")

    def _host_avail(self, rows, host):
        """[(t, available GiB)]: infra.host_avail_gib, else total_gib - infra.host_ram_gib."""
        pts = self._series(rows, "infra.host_avail_gib").get("infra.host_avail_gib")
        if pts:
            return pts
        used = self._series(rows, "infra.host_ram_gib").get("infra.host_ram_gib") or []
        return [(t, host["total_gib"] - v) for t, v in used] if host.get("total_gib") else []

    def _switch_projection(self, rows, now, host):
        """host.switch: the current play phase's available host at its switch, None outside play or too early."""
        sw = host["switch"]
        phase = self._series(rows, "seq.phase").get("seq.phase") or []
        if not phase or phase[-1][1] != 1:
            return None
        start = phase[-1][0]
        pts = [(t, v) for t, v in self._host_avail(rows, host) if start <= t <= now and t >= now - 600]
        if len(pts) < 3 or pts[-1][0] - pts[0][0] < 300:
            return None
        n = len(pts)
        mt, mv = sum(t for t, _ in pts) / n, sum(v for _, v in pts) / n
        var = sum((t - mt) ** 2 for t, _ in pts)
        slope = sum((t - mt) * (v - mv) for t, v in pts) / var if var else 0.0
        left = max(float(sw.get("phase_s", 1600)) - (now - start), 0.0)
        avail, need = pts[-1][1], float(sw.get("switch_gib", 50))
        return {"avail": avail, "slope": slope * 60, "left": left, "switch": need,
                "at_switch": avail + min(slope, 0.0) * left - need,
                "floor": float(sw.get("floor_gib", host.get("min_avail_gib", 8)))}

    @staticmethod
    def _last(rows, key):
        for r in reversed(rows):
            v = (r.get("m") or {}).get(key)
            if isinstance(v, (int, float)):
                return float(v)
        return None
