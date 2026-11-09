"""Solver policy: plays a generated game through the real harness with the game's own solution.

Data source (a) of the post-training arm ("perfect trajectories"). The policy answers the requests
that the arena Duck-transfer harness (``ArenaTransferDuckAgent``) sends through
``posttrain.relay.RelayTransport``, so the saved request/response pairs have exactly the format of
a model's turns in the arena. It is a solver, not a player: it reads the breadth-first-search
solution from the game's hidden notes (``notes/<prefix>.json``) and never looks at the board.

Every reply is programmatic:

- a gameplay request gets one python tool call in the harness's action idiom
  (``action(['UP', 'LEFT'])``, ``action([{'action': 'MOUSE', 'row': r, 'col': c}])``) that runs the
  next solver actions of the current level (by default the rest of the level; the harness stops a
  batch at a level clear anyway), plus a short reasoning line and a ``Plan:`` line built from the
  level, the step and the planned action names;
- a tool-less request (the arena's compaction summary) gets a short summary built from the same
  state.

The position in the solution is derived from the harness's own state line (``Current state: step S,
level L``): ``S - 1`` actions have been executed, all of them solver actions, so the policy is
stateless and checks that the level the harness reports is the level the solution expects.
"""

import json
import re
from pathlib import Path

from .games.admission import display_xy

KEY_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE", 7: "ACTION7"}
STATE_RE = re.compile(r"Current state: step (\d+), level (\d+)")
MAX_PLAN = 48  # the harness's action-batch maximum (configs/duck_controls.yaml max_plan)
DEFAULT_CHUNK = 8


def game_source(games_root, game_id):
    prefix, version = str(game_id).split("-", 1)
    return (Path(games_root) / "environment_files" / prefix / version / f"{prefix}.py").read_text()


def level_sizes(games_root, game_id):
    """(width, height) of every level's grid, from the game file's ``LEVELS``."""
    from .games.store import load_class

    prefix = str(game_id).split("-")[0]
    cls = load_class(prefix, game_source(games_root, game_id))
    return [(len(spec["map"][0]), len(spec["map"])) for spec in cls.LEVELS]


def model_action(action, cell, size):
    """The harness's model-facing action for one solver step (MOUSE in 64x64 display cells)."""
    if action == 6:
        x, y = display_xy(size[0], size[1], cell)
        return {"action": "MOUSE", "row": y, "col": x}
    return KEY_NAMES[action]


def label(action):
    return f"MOUSE({action['row']},{action['col']})" if isinstance(action, dict) else action


def load_plan(games_root, game_id):
    """Per-level lists of harness actions from the game's notes (checked against its baselines)."""
    games_root = Path(games_root)
    prefix = str(game_id).split("-")[0]
    notes = json.loads((games_root / "notes" / f"{prefix}.json").read_text())
    if notes["game_id"] != game_id:
        raise ValueError(f"notes are for {notes['game_id']}, not {game_id}")
    sizes = level_sizes(games_root, game_id)
    solution = notes["solution"]
    if len(solution) != len(sizes):
        raise ValueError(f"{game_id}: {len(solution)} solution levels for {len(sizes)} game levels")
    version = str(game_id).split("-", 1)[1]
    metadata = json.loads((games_root / "environment_files" / prefix / version / "metadata.json").read_text())
    if [len(path) for path in solution] != metadata["baseline_actions"]:
        raise ValueError(f"{game_id}: solution lengths differ from baseline_actions")
    return [[model_action(a, c, size) for a, c in path] for path, size in zip(solution, sizes, strict=True)]


def latest_state(messages):
    """(step, level) from the newest ``Current state`` line, and whether a tool result follows it."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        match = STATE_RE.search(content or "") if message.get("role") == "user" else None
        if match:
            tool_after = any(m.get("role") == "tool" for m in messages[index + 1:])
            return int(match.group(1)), int(match.group(2)), tool_after
    raise ValueError("no 'Current state' line in the request")


def action_code(actions):
    batch = ", ".join(repr(a) for a in actions)
    return f"r = action([{batch}])\nprint(r.get('state'), r.get('score'), r.get('stop_reason'))"


class SolverPolicy:
    """Answers relay requests with the solution's next actions.

    A level's solution is split into ``ceil(n / chunk)`` near-equal batches (``chunk`` = 0: the
    rest of the level in one call, up to the harness maximum). The default of 8 keeps batches in
    the range Flash uses (teacher runs: median 1, 90th percentile 9 actions per executing call).
    """

    def __init__(self, plan, *, chunk=DEFAULT_CHUNK, reasoning=True):
        self.plan = plan
        self.chunk = chunk
        self.reasoning = reasoning
        self.starts = [sum(len(p) for p in plan[:i]) for i in range(len(plan))]
        self.total = sum(len(p) for p in plan)

    def position(self, executed):
        """(level index, index within the level) after ``executed`` solver actions."""
        for level in range(len(self.plan) - 1, -1, -1):
            if executed >= self.starts[level]:
                return level, executed - self.starts[level]
        raise ValueError(executed)

    def answer(self, payload):
        messages = payload.get("messages") or []
        step, level, tool_after = latest_state(messages)
        executed = step - 1
        if not payload.get("tools"):
            return {"reasoning": "", "content": self.summary(executed, level)}
        if tool_after:
            raise RuntimeError(f"step {step}: the previous solver call did not execute actions "
                               f"(tool result: {str(messages[-1].get('content'))[:500]})")
        if executed >= self.total:
            raise RuntimeError(f"step {step}: the solution is used up but the game is still running")
        index, offset = self.position(executed)
        if index + 1 != level:
            raise RuntimeError(f"step {step}: harness reports level {level}, solution expects {index + 1}")
        remaining = self.plan[index][offset:]
        actions = remaining[:self.batch_end(len(self.plan[index]), offset) - offset]
        names = ", ".join(label(a) for a in actions)
        rest = "the rest of the level" if len(actions) == len(remaining) else f"{len(actions)} of {len(remaining)} left"
        reasoning = (f"Level {level}, step {step}: next {len(actions)} solution actions ({rest})."
                     if self.reasoning else "")
        return {"reasoning": reasoning, "content": f"Plan: {names}.", "code": action_code(actions)}

    def batch_end(self, length, offset):
        """End (exclusive) of the batch that starts at ``offset`` in a level of ``length`` actions."""
        size = min(self.chunk or MAX_PLAN, MAX_PLAN)
        count = -(-length // size)
        bounds = [-(-i * length // count) for i in range(1, count + 1)]  # front-loaded: 5 -> 3, 2
        return next(b for b in bounds if b > offset)

    def summary(self, executed, level):
        index, offset = self.position(min(executed, self.total - 1))
        remaining = self.plan[index][offset:]
        return (f"Current level and state: level {level}, step {executed + 1}; {offset} of "
                f"{len(self.plan[index])} solution actions of this level done.\n"
                f"Plan: {', '.join(label(a) for a in remaining)}.")


async def serve(relay_dir, policy, finished, *, poll_seconds=0.02):
    """Answer every pending relay turn in ``relay_dir`` with ``policy`` until ``finished`` is set.

    The answer is written to a temporary name and renamed, so the transport never reads a partial
    file. Returns the number of turns answered.
    """
    import asyncio

    relay_dir = Path(relay_dir)
    answered = 0
    while not finished.is_set():
        marker = relay_dir / "PENDING"
        try:
            turn = int(marker.read_text())
        except (FileNotFoundError, ValueError):
            await asyncio.sleep(poll_seconds)
            continue
        stem = relay_dir / f"turn-{turn:04d}"
        response = stem.with_suffix(".response.json")
        if response.exists():
            await asyncio.sleep(poll_seconds)
            continue
        payload = json.loads(stem.with_suffix(".request.json").read_text())
        answer = policy.answer(payload)
        partial = stem.with_suffix(".response.tmp")
        partial.write_text(json.dumps(answer, ensure_ascii=False, indent=1))
        partial.rename(response)
        answered += 1
    return answered
