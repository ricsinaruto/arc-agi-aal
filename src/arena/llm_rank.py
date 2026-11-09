"""LLM-ranked backlog (ideas.md point 4, 26 September): every few minutes of phase 2 the served
model reads a compact evidence table of every unfinished game and ranks the waiting games; the
scheduler admits from the backlog in that order.

Hard rules stay in code: level budgets, parking, the opening switch and the preference for a game
that cleared a level while parked are unchanged; only the order of the waiting games is the
model's. Before the first ranking (and for games a ranking leaves out) the readmissions order
applies. Each ranking is one tool-less, thinking-off request admitted through the KV gate as
``scheduler`` and recorded with its evidence, reply and latency.
"""

import asyncio
import json
import re
import time

SYSTEM = (
    "You allocate GPU time in an ARC-AGI-3 arena. Several puzzle games share one model server; only "
    "a few 'focus' games play at a time and the rest wait. A game's score comes from the levels it "
    "clears: level k is worth k points of weight and later levels are harder. Time spent on a game "
    "that never clears its next level is wasted.\n"
    "You see evidence per game: state, levels cleared of the total, T1 (busy seconds its first level "
    "took), busy seconds and actions spent on its current level, distinct boards produced on the "
    "current level, actions since its last new board (high = stuck), and how often it was admitted.\n"
    "Rank the WAITING games (state parked or backlog) by how likely each is to clear its next level "
    "soon if given the next free slot. Prefer games with evidence of progress (clears, fast T1, many "
    "distinct boards per action, few actions since a new board). Deprioritise games that were "
    "admitted repeatedly without clearing anything.\n"
    'Reply with one JSON object and nothing else: {"analysis": "<at most three sentences>", '
    '"ranking": ["<game id>", ...]} listing every waiting game id exactly once.'
)
# Token budgets (29 September): the same evidence in completion tokens instead of busy seconds.
SYSTEM_TOKENS = SYSTEM.replace(
    "T1 (busy seconds its first level took), busy seconds and actions spent",
    "T1 (completion tokens its first level took), completion tokens and actions spent")
KEYS = ["id", "state", "levels", "t1_s", "busy_s_total", "busy_s_level", "actions_level",
        "distinct_boards_level", "actions_since_new_board", "admissions"]
TOKEN_KEYS = {"t1_s": "t1_tokens", "busy_s_total": "tokens_total", "busy_s_level": "tokens_level"}


def evidence(scheduler):
    """One row per unfinished game (focus games included as context)."""
    tokens = getattr(scheduler, "token_clock", False)
    rows = []
    for game in scheduler.games:
        if game.state == "finished":
            continue
        state = {"active": "focus"}.get(game.state, game.state)
        if game.state == "active" and game.pool:
            state = "audition"
        row = {
            "id": game.game_id[:4],
            "state": state,
            "levels": f"{game.levels}/{game.total_levels}" if game.total_levels else str(game.levels),
            "t1_s": None if game.t1 is None else round(game.t1),
            "busy_s_total": round(game.clock.budget_seconds()),
            "busy_s_level": round(game.spent_on_level()),
            "actions_level": game.actions_on_level,
            "distinct_boards_level": len(game.frames_on_level),
            "actions_since_new_board": game.actions_since_novel,
            "admissions": game.readmissions,
        }
        if tokens:
            row = {TOKEN_KEYS.get(k, k): v for k, v in row.items()}
        if game.pending_parked_progress:
            row["cleared_while_parked"] = True
        rows.append(row)
    return rows


def table(rows, tokens=False):
    keys = [TOKEN_KEYS.get(k, k) for k in KEYS] if tokens else KEYS
    lines = [" | ".join(keys)]
    for row in rows:
        lines.append(" | ".join("-" if row.get(k) is None else str(row[k]) for k in keys)
                     + (" | cleared while parked" if row.get("cleared_while_parked") else ""))
    return "\n".join(lines)


def messages(scheduler, wall_seconds, remaining_seconds):
    rows = evidence(scheduler)
    tokens = getattr(scheduler, "token_clock", False)
    user = (f"Elapsed {round(wall_seconds)} s, {round(remaining_seconds)} s of the run left, "
            f"{scheduler.active_limit} focus slots.\n\n{table(rows, tokens)}")
    system = SYSTEM_TOKENS if tokens else SYSTEM
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], rows


def parse_ranking(text, waiting_ids):
    """Waiting game ids in the model's order; ids are matched on their four-character prefix.
    Returns ``(ranking, error)``; unknown and repeated ids are dropped."""
    by_prefix = {g[:4]: g for g in waiting_ids}
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return [], "no JSON object"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return [], f"invalid JSON: {exc.msg}"
    ranked = data.get("ranking") if isinstance(data, dict) else None
    if not isinstance(ranked, list):
        return [], "no ranking list"
    out = []
    for item in ranked:
        game = by_prefix.get(str(item).strip()[:4])
        if game is not None and game not in out:
            out.append(game)
    return out, None if out else "no known ids"


class LLMRanker:
    def __init__(self, scheduler, gate, client, *, model, wall, interval_seconds=180.0, cap_tokens=768,
                 temperature=0.7, log=None, now=time.monotonic):
        self.scheduler, self.gate, self.client = scheduler, gate, client
        self.model, self.wall = model, wall
        self.interval, self.cap, self.temperature = float(interval_seconds), int(cap_tokens), temperature
        self.log, self.now = log, now
        self.rankings = []
        self.errors = 0

    def payload(self, msgs):
        return {"model": self.model, "messages": msgs, "stream": False, "max_tokens": self.cap,
                "temperature": self.temperature, "chat_template_kwargs": {"enable_thinking": False}}

    async def rank_once(self):
        scheduler = self.scheduler
        waiting = [g.game_id for g in scheduler.games if g.state in ("backlog", "parked")]
        if not waiting:
            return None
        msgs, rows = messages(scheduler, self.wall.elapsed(), self.wall.remaining())
        if self.model is None:  # the served model's id, as the agents address it
            listing = await self.client.get("models")
            listing.raise_for_status()
            self.model = listing.json()["data"][0]["id"]
        estimate = sum(len(m["content"]) for m in msgs) // 3 + 64
        started = self.now()
        reservation = await self.gate.acquire("scheduler", estimate + self.cap)
        try:
            response = await self.client.post("chat/completions", json=self.payload(msgs))
            response.raise_for_status()
            data = response.json()
        finally:
            self.gate.release(reservation)
        text = str(data["choices"][0]["message"].get("content") or "")
        ranking, error = parse_ranking(text, waiting)
        record = {"wall_seconds": round(self.wall.elapsed(), 3), "seconds": round(self.now() - started, 3),
                  "evidence": rows, "reply": text[:2000], "ranking": [g[:4] for g in ranking],
                  "error": error, "usage": data.get("usage", {})}
        self.rankings.append(record)
        if ranking:
            scheduler.set_llm_ranking(ranking)
        else:
            self.errors += 1
        if self.log:
            self.log({"event": "llm_rank", **{k: v for k, v in record.items() if k != "evidence"}})
        return record

    async def run(self):
        """Rank at the opening switch and every ``interval`` seconds of phase 2 after it."""
        last = None
        while self.wall.remaining() > 0:
            if self.scheduler.phase != "opening" and (last is None or self.wall.elapsed() - last >= self.interval):
                last = self.wall.elapsed()
                try:
                    await self.rank_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - a failed ranking keeps the previous order
                    self.errors += 1
                    self.rankings.append({"wall_seconds": round(self.wall.elapsed(), 3),
                                          "error": f"{type(exc).__name__}: {exc}"[:300]})
            await asyncio.sleep(1.0)

    def summary(self):
        return {"interval_seconds": self.interval, "cap_tokens": self.cap, "rankings": len(self.rankings),
                "errors": self.errors, "history": self.rankings}
