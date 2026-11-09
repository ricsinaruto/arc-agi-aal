"""Model-free graph explorer for waiting games (variant ``fixed-2x-explorer``).

Measured 26 September on the one-hour base runs: the two phase-2 streams decode about 230 tokens/s
between them (four or eight streams reach only 290 / 360), the hour yields about 1,000 actions in
all, and a game outside the focus slots gets 3-28 actions all hour, below every level-1 baseline
(7-71 actions). Seven roster games never cleared level 1 in 17 runs. The environments themselves
step at 2,000-4,000 actions per second on one CPU core and sit idle while their games wait.

Offline, a frame-hash graph explorer (one click per connected component, simple keys, breadth-first
walks to the nearest state with untried actions, level reset on GAME_OVER or a closed graph)
cleared level 1 of eight roster games within 20,000 actions (5-15 CPU seconds each): cd82, m0r0,
sp80, su15 and tn36, which the model clears in at most 3 of 17 runs, plus r11l, tu93 and vc33. The
winning paths it knows are short (4-62 actions against baselines of 7-55).

A level the explorer clears scores about nothing (the squared action efficiency), so it only runs
on waiting games still at level 0 (level weight 1 of 21-55) and its value is what follows: the game
is flagged as cleared while parked (readmitted first) and the model receives the demonstration
(the shortest known winning sequence and the region it changed) for the next levels. All of its
actions are taken in the scored environment, with level resets only.
"""

import collections
import hashlib
import time

import numpy as np

KEYS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7")
COLOR_CHARS = "WwgGcBMPRbSYOrNp"  # the letter palette of the Duck board views


def components(frame, limit=64):
    """One representative cell per connected single-colour component (4-connectivity), smallest
    components first; the most common colour is taken as background and skipped."""
    from scipy import ndimage

    colors, counts = np.unique(frame, return_counts=True)
    background = colors[int(np.argmax(counts))]
    found = []
    for color in colors:
        if color == background:
            continue
        labels, n = ndimage.label(frame == color)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(labels == i)
            k = len(ys) // 2
            found.append((len(ys), int(ys[k]), int(xs[k])))
    found.sort()
    return found[:limit]


def frame_key(frame, levels):
    return hashlib.blake2b(np.ascontiguousarray(frame).tobytes() + bytes([levels % 256]), digest_size=12).hexdigest()


class LevelExplorer:
    """Graph exploration of one level. ``next_action()`` returns ``(name, x, y)``; the driver takes
    it and reports the outcome to ``observe``. Pure logic: no environment access."""

    def __init__(self, frame, levels, available, *, click_limit=64):
        self.levels = int(levels)
        self.available = set(available)
        self.click_limit = click_limit
        self.untried = {}
        self.edges = collections.defaultdict(dict)
        self.frames = {}
        self.start = self.state = frame_key(frame, self.levels)
        # Roots of the winning-path search: the board the explorer started from and every board a level
        # reset returned to (the true level start when the model had already played part of the level;
        # explorer-v1 lost 4 of its 6 demonstrations searching from the mid-level board only).
        self.roots = {self.start}
        self.frame = frame
        self.plan = []
        self.effects = collections.Counter()  # action kind -> board changes caused
        self.kinds = collections.Counter()  # action kind -> times taken
        self._ensure(self.state, frame)

    def _actions(self, frame):
        acts = [(a, None, None) for a in KEYS if a in self.available]
        if "ACTION6" in self.available:
            acts += [("ACTION6", x, y) for _, y, x in components(frame, self.click_limit)]
        return acts

    def _ensure(self, key, frame):
        if key not in self.untried:
            self.untried[key] = self._actions(frame)
            self.frames[key] = frame

    def _path(self, source, target_test):
        prev = {source: None}
        queue = collections.deque([source])
        while queue:
            node = queue.popleft()
            if target_test(node):
                path = []
                while prev[node] is not None:
                    node, action = prev[node]
                    path.append(action)
                return list(reversed(path))
            for action, nxt in self.edges[node].items():
                if nxt not in prev:
                    prev[nxt] = (node, action)
                    queue.append(nxt)
        return None

    def next_action(self):
        if not self.plan:
            if self.untried[self.state]:
                self.plan = [self.untried[self.state].pop(0)]
            else:
                path = self._path(self.state, lambda node: bool(self.untried.get(node)))
                self.plan = path if path else [("RESET", None, None)]
        return self.plan.pop(0)

    def kind(self, action, frame):
        name, x, y = action
        return name if name != "ACTION6" else f"click {COLOR_CHARS[int(frame[y, x])]}"

    def observe(self, action, frame, levels):
        """Record the outcome of ``action``; returns True when it cleared the level."""
        levels = int(levels)
        if levels > self.levels:
            return True
        key = frame_key(frame, levels)
        if action[0] == "RESET":
            self.plan = []
            self.roots.add(key)
        else:
            kind = self.kind(action, self.frame)
            self.kinds[kind] += 1
            if key != self.state:
                self.effects[kind] += 1
            expected = self.edges[self.state].get(action)
            self.edges[self.state][action] = key
            if expected is not None and expected != key:
                self.plan = []  # not deterministic from here: re-plan
        self._ensure(key, frame)
        self.state, self.frame = key, frame
        return False

    def winning_path(self, action):
        """Shortest known path from the level start to the state the clear came from, plus the
        winning action."""
        paths = [p for p in (self._path(root, lambda node: node == self.state) for root in self.roots) if p is not None]
        return None if not paths else [*min(paths, key=len), action]


def explore_level(env, frame, levels, available, *, max_actions, max_seconds, should_stop, click_limit=64,
                  now=time.monotonic):
    """Explore the current level of ``env`` (``step(GameAction, data)``/``reset()``) until it clears,
    the action or time budget ends, or ``should_stop()``; an uncleared level is reset to its start.
    Runs synchronously (call it in a worker thread). Returns the outcome and the last raw frame."""
    from arcengine import GameAction

    explorer = LevelExplorer(frame, levels, available, click_limit=click_limit)
    start_frame = frame
    started = now()
    actions = resets = game_overs = 0
    raw = None
    cleared = False
    path = None
    prewin = None
    stop_reason = "budget"
    while True:
        if actions >= max_actions or now() - started >= max_seconds:
            break
        if should_stop():
            stop_reason = "admitted"
            break
        action = explorer.next_action()
        name, x, y = action
        if name == "RESET":
            raw = env.reset()
            resets += 1
        else:
            raw = env.step(GameAction[name], {"x": x, "y": y} if name == "ACTION6" else {})
        actions += 1
        if raw is None:
            stop_reason = "no_observation"
            break
        new_frame = np.array(raw.frame[-1], dtype=np.uint8) if raw.frame else explorer.frame
        new_levels = int(raw.levels_completed)
        if explorer.observe(action, new_frame, new_levels):
            cleared, stop_reason = True, "cleared"
            prewin = explorer.frame
            path = explorer.winning_path(action)
            break
        state = getattr(raw.state, "name", str(raw.state))
        if state == "WIN":
            stop_reason = "won"
            break
        if state == "GAME_OVER":
            game_overs += 1
            raw = env.reset()
            resets += 1
            actions += 1
            new_frame = np.array(raw.frame[-1], dtype=np.uint8)
            explorer.observe(("RESET", None, None), new_frame, int(raw.levels_completed))
    if not cleared and raw is not None and actions:
        raw = env.reset()  # leave the level at its start for the model
        resets += 1
        actions += 1
    effects = sorted(((kind, explorer.effects[kind], taken) for kind, taken in explorer.kinds.items()),
                     key=lambda row: (-row[1] / max(1, row[2]), row[0]))
    return {
        "cleared": cleared,
        "stop_reason": stop_reason,
        "actions": actions,
        "resets": resets,
        "game_overs": game_overs,
        "states": len(explorer.untried),
        "seconds": round(now() - started, 3),
        "path": path,
        "start_frame": start_frame,
        "prewin_frame": prewin,
        "effects": effects,
        "raw": raw,
    }


def compress(labels):
    """``['UP', 'UP', 'LEFT']`` -> ``'UP x2, LEFT'``."""
    out = []
    for label in labels:
        if out and out[-1][0] == label:
            out[-1][1] += 1
        else:
            out.append([label, 1])
    return ", ".join(label if n == 1 else f"{label} x{n}" for label, n in out)


def region_text(before, after, *, max_side=24):
    """Letter-coded rows of the bounding box of cells that differ (clipped to ``max_side``)."""
    diff = np.argwhere(before != after)
    if not len(diff):
        return None
    r0, c0 = diff.min(axis=0)
    r1, c1 = diff.max(axis=0) + 1
    r1, c1 = min(r1, r0 + max_side), min(c1, c0 + max_side)

    def rows(grid):
        return "\n".join("".join(COLOR_CHARS[int(v)] for v in row) for row in grid[r0:r1, c0:c1])

    return (f"rows {r0}-{r1 - 1}, cols {c0}-{c1 - 1} ({len(diff)} cells differ)\n"
            f"at the level start:\n{rows(before)}\njust before the winning action:\n{rows(after)}")


def report_message(result, level, display):
    """The carried user message for the model (``level`` = the explored level, 1-based)."""
    head = (f"[Automatic explorer, level {level}] While this game waited for a model slot, a model-free "
            f"explorer (systematic trial of every key and one click per object, no model calls) took "
            f"{result['actions']} actions on level {level} and saw {result['states']} distinct boards.")
    def name(kind):
        return kind if kind.startswith("click ") else display((kind, None, None))

    rows = result["effects"]
    if rows and all(changed == taken for _, changed, taken in rows):
        effects = ["every action changed the board (a timer or counter is likely on screen)"]
    else:
        effects = [f"{name(kind)} changed the board {changed} of {taken} times" for kind, changed, taken in rows[:8]]
    if result["cleared"]:
        labels = [display(a) for a in result["path"]] if result["path"] else []
        lines = [head, f"It CLEARED level {level}; the board you see now is level {level + 1}. The turns "
                       f"below describe level {level} and are superseded.",
                 f"Shortest winning sequence it found from the level-{level} start ({len(labels)} actions): "
                 f"{compress(labels)}." if labels else "The winning sequence was not recoverable."]
        region = (region_text(result["start_frame"], result["prewin_frame"])
                  if result["prewin_frame"] is not None else None)
        if region:
            lines.append("What that sequence changed before the winning action (letter-coded):\n" + region)
        lines.append("Infer the goal and the mechanics from this demonstration, check them on the new "
                     "board with python, and play the new level with as few actions as you can "
                     "(every action counts toward the level's score).")
    else:
        lines = [head, f"It did not clear level {level}; the level was reset to its start (the board and "
                       f"your earlier findings about the start still apply)."]
    if effects:
        lines.append("Action effects it measured: " + "; ".join(effects) + ".")
    return {"role": "user", "content": "\n".join(lines)}


class ExplorerService:
    """Runs the explorer on waiting games after the opening switch, one game at a time."""

    def __init__(self, scheduler, agents, wall, settings, log=None):
        self.scheduler, self.agents, self.wall = scheduler, agents, wall
        self.max_level = int(settings.get("max_level", 0))
        self.max_actions = int(settings.get("max_actions", 20000))
        self.max_seconds = float(settings.get("max_seconds", 90))
        self.min_remaining = float(settings.get("min_remaining_seconds", 120))
        self.log = log
        self.done = set()  # (game_id, level) already explored
        self.runs = []

    def candidate(self):
        for agent in self.agents:
            game = agent.game
            if (game.state == "parked" and game.finished_reason is None and game.levels <= self.max_level
                    and (game.game_id, game.levels) not in self.done and agent.explorer_waiting
                    and not agent.env_lock.locked()):
                return agent
        return None

    async def explore(self, agent):
        game = agent.game
        self.done.add((game.game_id, game.levels))
        async with agent.env_lock:
            if not (game.clock.parked and agent.explorer_waiting):
                self.done.discard((game.game_id, game.levels))
                return None
            record = await agent.run_explorer(
                max_actions=self.max_actions, max_seconds=self.max_seconds,
                should_stop=lambda: not game.clock.parked or self.wall.remaining() < 5)
        self.runs.append(record)
        if self.log:
            self.log({"event": "explorer", **record})
        return record

    async def run(self):
        import asyncio

        while self.wall.remaining() > self.min_remaining:
            agent = None if self.scheduler.phase == "opening" else self.candidate()
            if agent is None:
                await asyncio.sleep(1.0)
                continue
            try:
                await self.explore(agent)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one failed exploration never ends the trial
                entry = {"game": agent.game.game_id, "error": f"{type(exc).__name__}: {exc}"[:300]}
                self.runs.append(entry)
                if self.log:
                    self.log({"event": "explorer_error", **entry})

    def summary(self):
        return {"max_level": self.max_level, "max_actions": self.max_actions, "max_seconds": self.max_seconds,
                "explored": len(self.runs), "cleared": sum(1 for r in self.runs if r.get("cleared")),
                "actions": sum(r.get("actions", 0) for r in self.runs), "runs": self.runs}
