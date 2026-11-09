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
    "total_gib": host RAM, when the run exports used RAM (infra.host_ram_gib) rather than available}.
  util_exempt: gpu_util keys outside rule (c) (a card idle by design between phases).
  silence_s: metric silence threshold (default max(4 x cadence, 600 s)).
  counters: {cumulative metric key: seconds} (or a list, at ``freeze_s``): rule "frozen" fires when a counter stops
    advancing that long while rows keep arriving and work is in flight (infra.running > 0 or unreported); default
    {"infra.gen_tokens": freeze_s} (the arena adapter's engine token counter). A *tok_s target repeating one exact
    value of at least 25% of its plan for freeze_s fires it too (a status field copied from a hung engine).
  freeze_s: default max(300 s, 5 x cadence).
  progress_key: a training-progress counter (e.g. "train.step"): rule "progress" fires when it has not advanced for
    progress_k (default 2.5) x the median interval between its past advances (needs two advances; floor 120 s), while
    the app is up. Catches a hang at full GPU util (c7f: NCCL, 24 Oct).
  expect: {metric key or glob: seconds after play start}: rule "expect" fires once when no value of that metric has
    arrived by then (a read the run must make, e.g. the step-0 base read "eval.heldout_rhae10k": its absence leaves the
    run with no anchor; 24 Oct v6e).
  starve (always on when the plan targets a live count and a gen rate): the clock starts at server ready (first
    ``ready_key`` value > 0, default "infra.gen_tok_s"), not launch; from 5 min after ready, fires when the 5-min mean of
    ``live_key`` (default the play_key, e.g. "infra.running") stays under 50% of its plan AND gen tok/s stays under 50%
    of its plan at every sample of the last 5 min, checked from 5 to 15 min after ready (two signals; quiet during boot,
    a ramp that crosses half plan, and later phase lulls). 24 Oct: hero-v6/v6b starved ~10 min before plan@10m.
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
REMIND_S = 1800  # an alert that stays true is repeated every 30 min
MARKS_MIN = (10, 20)
CRASH = re.compile(r"SyntaxError|IndentationError|ImportError|ModuleNotFoundError|CalledProcessError|"
                   r"CUDA out of memory|OutOfMemoryError|\bKilled\b|Segmentation fault|RUNOPS INCIDENT|"
                   r"Traceback \(most recent call last\)|^\w*(Error|Exception): ")
# a run's own clean end (plan "end_marker" adds to it); a run that goes without one is an alert, not an END (24 Oct)
END_MARKER = r"RUNOPS END ok|\bclean end\b|\bRUN COMPLETE\b|\btraining complete\b"


PLAN_KEYS = {"run", "arm", "launched", "boot_s", "cadence_s", "targets", "play_key", "host", "util_exempt", "silence_s",
             "platform", "app_id", "vm", "metrics_remote", "runops_state", "logs", "owner_agent", "arena", "counters",
             "freeze_s", "end_marker", "progress_key", "progress_k", "follow_record", "expect", "ready_key",
             "live_key"}


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
        rows = sorted((r for r in s.rows if r.get("t") is not None and r["t"] <= now), key=lambda r: r["t"])
        first_play, play, launched = self.play_start(rows)

        if s.live is False and not s.clean_end:
            self._once(out, ("unclean-end",), "unclean-end",
                       f"app/VM gone before a clean end ({(now - launched) / 60:.1f} min after launch; no end marker, "
                       f"guard status or final.json): treat as a crash")
        self._fire(out, now, "gone", s.live is False and not s.ended and bool(s.clean_end),
                   "app/VM ended cleanly but the run record is still open: settle it (records.py settle)")
        what = f"'{self.play_key}' > 0" if self.play_key else "metrics"
        self._fire(out, now, "boot", first_play is None and now > launched + 1.5 * self.boot_s,
                   f"no {what} {(now - launched) / 60:.0f} min after launch (plan boot {self.boot_s / 60:.0f} min)")

        self._starve(out, now, rows)  # clocked on server ready, not launch + boot
        if play is not None:
            for mark in MARKS_MIN:
                end = play + 60 * mark
                if now < end or mark in self.marks_done:
                    continue
                self.marks_done.add(mark)
                for key, (target, _) in self.targets.items():
                    agg = max if key.startswith("infra.vram") else statistics.median
                    v = self._value(rows, key, end - 300, end, agg)
                    if v is None:
                        out.append(Alert(f"plan@{mark}m", f"{key}: no data at {mark} min of play (plan {target:g})"))
                    elif v < RATE_FLOOR * target:
                        out.append(Alert(f"plan@{mark}m", f"{key} {v:.4g} vs plan {target:g} at {mark} min of play "
                                                          f"({v / target:.0%} < {RATE_FLOOR:.0%})"))
            for key, secs in (self.plan.get("expect") or {}).items():
                if now >= play + float(secs) and not self._series(rows, key):
                    self._once(out, ("expect", key), f"expect:{key}",
                               f"no {key} {(now - play) / 60:.0f} min into play (expected within {float(secs) / 60:.0f}"
                               f" min): the run has no such read")
            if now >= play + 1200:
                for key, (target, steady) in self.targets.items():
                    if steady:
                        fresh = [t for pts in self._series(rows, key).values() for t, _ in pts if t >= now - 900]
                        spans = bool(fresh) and max(fresh) - min(fresh) >= 600  # a stale window is the silence rule's
                        v = self._value(rows, key, now - 900, now, min_points=5, carry=False) if spans else None
                        self._fire(out, now, f"rate:{key}", v is not None and v < RATE_FLOOR * target,
                                   f"{key} 15-min median {v if v is None else round(v, 3)} vs plan {target:g}")
            exempt = set(self.plan.get("util_exempt") or [])
            for key, pts in self._series(rows, "infra.gpu_util.*").items():
                recent = [(t, v) for t, v in pts if t >= now - UTIL_WINDOW_S]
                spans = recent and recent[-1][0] - recent[0][0] >= 0.66 * UTIL_WINDOW_S and recent[0][0] >= play
                low = bool(spans) and key not in exempt and statistics.mean(v for _, v in recent) < UTIL_FLOOR
                self._fire(out, now, f"util:{key}", low, f"{key} mean {statistics.mean(v for _, v in recent):.0f}% "
                           f"over 15 min (< {UTIL_FLOOR:.0f}%)" if low else "")

            self._frozen(out, now, rows)

        self._progress(out, now, rows, s)

        if rows:
            age = now - rows[-1]["t"]
            stale = age > self.silence_s and not s.ended
            second = [n for n, a in (("remote file", s.remote_age), ("heartbeat", s.beat_age)) if a is not None
                      and a > self.silence_s] + (["app/VM down"] if s.live is False else [])
            self._fire(out, now, "silent", stale and bool(second),
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
        for line in s.log_text.splitlines():
            if CRASH.search(line):
                self._once(out, ("crash", line.strip()), "crash", line.strip()[:300])
        return out

    def _frozen(self, out, now, rows):
        """A hung engine behind a live status loop: rows keep coming, a cumulative counter stops (or a rate field
        repeats one exact value). Quiet when nothing is in flight (a drained run) or when rows themselves stop
        (the silence rule's case)."""
        if not rows or now - rows[-1]["t"] > self.silence_s:
            return
        running = self._last(rows, "infra.running")
        busy = running is None or running > 0
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
                flat = [v for t, v in pts if t >= changed]
                counter = key in self.counters
                frozen = (busy and pts[-1][0] - changed >= limit and len(flat) >= 3
                          and (counter or flat[-1] >= 0.25 * rates[key]))  # a slow tail is the rate rule's
            self._fire(out, now, f"frozen:{key}", frozen,
                       f"{key} stuck at {pts[-1][1]:.6g} for {(pts[-1][0] - changed) / 60:.0f} min while rows keep "
                       f"arriving (running {running}): engine hang?" if frozen else "")

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
        idle = now - adv[-1]
        self._fire(out, now, f"progress:{key}", idle > limit,
                   f"{key} stuck at {pts[-1][1]:g} for {idle / 60:.1f} min (> {limit / 60:.1f} min = "
                   f"{self.plan.get('progress_k', 2.5)} x median step interval {statistics.median(gaps) / 60:.1f} min)")

    def state(self):
        """JSON-able rule state, so a restarted watcher does not re-fire or forget its marks."""
        return {"active": self.active, "marks_done": sorted(self.marks_done),
                "seen": [list(k) for k in self.seen]}

    def restore(self, st):
        self.active = dict(st.get("active") or {})
        self.marks_done = set(st.get("marks_done") or [])
        self.seen = {tuple(k) for k in st.get("seen") or []}

    @staticmethod
    def _last(rows, key):
        for r in reversed(rows):
            v = (r.get("m") or {}).get(key)
            if isinstance(v, (int, float)):
                return float(v)
        return None
