"""Momentum commitment with a preemptible lane (``fixed-2x-commit``, 26 September).

Measured over 1,088 level attempts in 29 one-hour runs, clears per 10K completion tokens peak at
10-25K tokens on a level (level 1: 0.27; deeper: 0.27) and fall off past 50K (level 1: 0.03; deeper:
0.10), while a typical run leaves ten games under 10K tokens on level 1 and uncleared deep attempts
take a third of all tokens. The variant is built around that curve:

- Focus slots (``focus_slots``, 2) go to games that clear levels: a clear in the lane promotes the
  game at once (``arena.scheduler.Scheduler``, reason ``lane_clear``). A focus game keeps its slot
  while its tokens on the current level stay within its stall cap, ``clamp(stall_multiple x median
  tokens per level it has cleared, stall_min_tokens, stall_max_tokens)``; past the cap it is demoted
  to the lane (park reason ``stalled_level``), and only a clear brings it back.
- The lane (``CommitLane``) serves every other game: level-1 games under ``level1_floor_tokens``
  first (fewest tokens first), then deep games (fewest tokens on their level first), then level-1
  games past the floor.
- Lane requests are admitted on an estimate of the cache actually in use instead of full-length
  reservations, up to ``lane_max_inflight``, and carry vLLM priority 1 (focus and opening requests
  priority 0) on a server started with ``--scheduling-policy priority``. vLLM may then preempt lane
  requests when the cache runs short; it picks the running request with the largest
  ``(priority, arrival)``, so while a lane request runs, a focus request is never the victim.
  ``PreemptionAttribution`` separates those lane preemptions from any that could have hit a focus
  request (``preemptions_lane_policy`` in the trial report).
"""

import json
import math
import re
from pathlib import Path

# Optional settings, absent from the defaults so that fixed-2x-commit studies stay unchanged:
# replace_unstalled (default True) False = a lane clear takes only a free or stalled focus slot and
# otherwise queues (fixed-2x-commit-queue).
COMMIT_OPTIONAL = {"replace_unstalled": bool}
COMMIT_DEFAULTS = {
    "focus_slots": 2,
    "stall_multiple": 2.5,
    "stall_min_tokens": 25_000,
    "stall_max_tokens": 60_000,
    "level1_floor_tokens": 25_000,
    "lane_kv_fraction": 0.92,
    "lane_max_inflight": 4,
    "lane_headroom_tokens": 6_000,  # until the run has observed lane_headroom_min_samples completions
    "lane_headroom_quantile": 0.75,
    "lane_headroom_min_samples": 8,
    # Generation of an in-flight request is estimated as this rate x its age (at most the cap). The
    # decode bench at 32K prompts: 110 tok/s for one stream, 120 per stream for two, 96 for three;
    # a rate at the top of that range over-estimates what a request holds, never under-estimates it.
    "decode_tokens_per_second": 120.0,
    # A counter increment is attributed to the gate's in-flight mix from this long before the
    # previous metrics sample (metrics reach the endpoint slightly after the preemption).
    "preemption_slack_seconds": 2.0,
}

COUNTER = "vllm:num_preemptions_total"
ATTRIBUTION_RULE = (
    "vLLM (priority policy) preempts the running request with the largest (priority, arrival); lane requests "
    "carry priority 1 and focus/opening requests 0, so a preemption can hit a focus request only when no lane "
    "request is running. Every increment of vllm:num_preemptions_total between two metrics samples is "
    "attributed to the gate's in-flight mix over [previous sample - slack, this sample]: 'focus_exposed' if at "
    "any moment a focus request was in flight with no lane request, else 'lane_policy' if a lane request was in "
    "flight, else 'unattributed' (nothing in flight). focus_exposed + unattributed + increments the monitor did "
    "not attribute count against preemption_free; lane_policy preemptions do not.")


def commit_settings(raw=None):
    """Validated ``commit`` settings (``fixed-2x-commit``)."""
    merged = {**COMMIT_DEFAULTS, **(raw or {})}
    unknown = set(merged) - set(COMMIT_DEFAULTS) - set(COMMIT_OPTIONAL)
    if unknown:
        raise ValueError(f"Unknown commit settings: {sorted(unknown)}")
    for key, kind in COMMIT_OPTIONAL.items():
        if key in merged and type(merged[key]) is not kind:
            raise ValueError(f"commit.{key} must be a {kind.__name__}")

    def number(key, low, high=math.inf, integer=False):
        value = merged[key]
        kinds = (int,) if integer else (int, float)
        if type(value) not in kinds or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"commit.{key} must be {'an integer' if integer else 'a number'} in [{low}, {high}]")

    number("focus_slots", 1, 8, integer=True)
    number("stall_multiple", 0.5, 20)
    number("stall_min_tokens", 1, integer=True)
    number("stall_max_tokens", merged["stall_min_tokens"], integer=True)
    number("level1_floor_tokens", 0, integer=True)
    number("lane_kv_fraction", 0.1, 1.0)
    number("lane_max_inflight", 1, 16, integer=True)
    number("lane_headroom_tokens", 1, integer=True)
    number("lane_headroom_quantile", 0.05, 1.0)
    number("lane_headroom_min_samples", 1, integer=True)
    number("decode_tokens_per_second", 1)
    number("preemption_slack_seconds", 0, 30)
    return merged


class CommitLane:
    """Lane roles, order and occupancy parameters for ``KVGate`` (``fixed-2x-commit``).

    The gate admits focus requests exactly as before on the focus requests' own full-length
    reservations (lane reservations do not count against them: lane requests yield to focus requests
    in vLLM), so a focus request never waits at the gate on a lane request. A lane request is admitted
    when, in cache units (``budget.cost`` of a length, or tokens without a block budget),

        estimate + cost(its prompt + headroom) <= lane_kv_fraction x pool

    with at most ``lane_max_inflight`` lane requests and one per game. ``estimate`` (``KVGate.lane_estimate``)
    is the sum over in-flight requests of cost(prompt + decode_tokens_per_second x age, at most the cap),
    plus, for every focus game without a request in flight, cost(latest prompt + headroom): its next
    request is a tool call away and vLLM admits a waiting request only when its whole prompt fits
    (``scheduler_reserve_full_isl``), and a waiting request never preempts a running one. ``headroom`` is
    the ``lane_headroom_quantile`` of the completion lengths this run has observed (every request, focus
    and lane), ``lane_headroom_tokens`` until ``lane_headroom_min_samples`` are in.
    """

    preemptible = True  # the gate switches to the estimate rule for this lane

    def __init__(self, games, *, window, cap, pool_units, **settings):
        self.cfg = commit_settings(settings)
        self.games = {game.game_id: game for game in games}
        self.window, self.cap = int(window), int(cap)
        self.pool_units = int(pool_units)
        if self.pool_units <= 0:
            raise ValueError("The commit lane needs a positive cache pool")
        self.max_inflight = self.cfg["lane_max_inflight"]
        self.completions = []  # completion tokens of every finished request (sequential per game)
        self._open = {}  # game id -> its completion tokens when its latest request was admitted

    def is_lane(self, game_id):
        game = self.games.get(game_id)
        return game is not None and game.lane

    def order(self, game_id):
        """Level-1 games under the floor (fewest tokens first), then deep games (fewest tokens on the
        level first), then level-1 games past the floor; roster order breaks ties."""
        game = self.games[game_id]
        tokens = game.tokens_on_level()
        if game.levels == 0:
            tier = 0 if tokens < self.cfg["level1_floor_tokens"] else 2
        else:
            tier = 1
        return (tier, tokens, game.index)

    def focus_games(self):
        return [game for game in self.games.values() if game.state == "active" and not game.lane]

    def budget_units(self):
        return int(self.pool_units * self.cfg["lane_kv_fraction"])

    def generated(self, seconds):
        return int(min(self.cap, self.cfg["decode_tokens_per_second"] * max(0.0, seconds)))

    def headroom(self):
        if len(self.completions) < self.cfg["lane_headroom_min_samples"]:
            return min(self.cap, self.cfg["lane_headroom_tokens"])
        ordered = sorted(self.completions)
        rank = max(1, math.ceil(self.cfg["lane_headroom_quantile"] * len(ordered)))  # nearest rank
        return max(1, min(self.cap, ordered[rank - 1]))

    def note_admission(self, game_id):
        """The gate calls this at every admission. A game's requests are sequential and a reply is
        charged before the game's next admission, so the completion tokens charged since its previous
        admission are that request's."""
        game = self.games.get(game_id)
        if game is None:
            return
        tokens = game.clock.completion_tokens()
        opened = self._open.get(game_id)
        if opened is not None and tokens > opened:
            self.completions.append(tokens - opened)
        self._open[game_id] = tokens

    def summary(self):
        return {"window": self.window, "cap": self.cap, "pool_units": self.pool_units,
                "lane_budget_units": self.budget_units(), "max_inflight": self.max_inflight,
                "headroom_tokens": self.headroom(), "completions_observed": len(self.completions)}


class PreemptionAttribution:
    """Attributes vLLM preemption-counter increments to the gate's in-flight mix (``ATTRIBUTION_RULE``).
    ``observe`` is the engine monitor's ``on_sample`` callback."""

    def __init__(self, gate, *, slack_seconds=COMMIT_DEFAULTS["preemption_slack_seconds"]):
        self.gate, self.slack = gate, float(slack_seconds)
        self.last_value = self.last_time = None
        self.counts = {"lane_policy": 0, "focus_exposed": 0, "unattributed": 0}
        self.events = []
        self.missed_samples = 0

    def observe(self, item):
        if item is None or COUNTER not in item.get("metrics", {}):
            self.missed_samples += 1
            return
        value, now = float(item["metrics"][COUNTER]), self.gate.now()
        if self.last_value is None:
            self.last_value, self.last_time = value, now
            return
        exposure = self.gate.exposure(self.last_time - self.slack, now)  # also prunes the gate's timeline
        delta = value - self.last_value
        if delta > 0:
            kind = ("focus_exposed" if exposure["focus_only"] else "lane_policy" if exposure["lane"]
                    else "unattributed")
            self.counts[kind] += int(round(delta))
            self.events.append({"elapsed_seconds": round(item.get("elapsed_seconds", 0.0), 3),
                                "preemptions": int(round(delta)), "attribution": kind, **exposure})
        self.last_value, self.last_time = value, now

    def counted(self, total):
        """Preemptions that count against ``preemption_free``: every increment not attributed to the
        lane policy, including any the samples did not attribute (``total`` is the monitor's delta)."""
        attributed = sum(self.counts.values())
        return self.counts["focus_exposed"] + self.counts["unattributed"] + max(0, int(total or 0) - attributed)

    def summary(self):
        return {"rule": ATTRIBUTION_RULE, "slack_seconds": self.slack, **self.counts,
                "missed_samples": self.missed_samples, "events": list(self.events)}


def server_scheduling_policy(server_dir):
    """``(policy, source)`` of the running vLLM server: the recorded argv first, else the server log's
    non-default args; vLLM's default (``fcfs``) when neither names a policy; ``(None, None)`` when
    neither record exists."""
    server_dir = Path(server_dir)
    identity = server_dir / "vllm-server-identity.json"
    if identity.is_file():
        argv = json.loads(identity.read_text()).get("argv") or []
        if "--scheduling-policy" in argv[:-1]:
            return argv[argv.index("--scheduling-policy") + 1], "vllm-server-identity.json argv"
        return "fcfs", "vllm-server-identity.json argv (no --scheduling-policy)"
    from .capacity import server_log_lines

    for line in server_log_lines(server_dir):
        if "non-default args:" in line:
            match = re.search(r"'scheduling_policy': '(\w+)'", line)
            return (match[1] if match else "fcfs"), "server log non-default args"
    return None, None
