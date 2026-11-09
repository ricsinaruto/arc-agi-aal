"""carry_live.py TREE: what every next v9b build carries: prefix hit in live_progress (c35de6ece, whole file), the
active_limit knob (ab5cf2297, anchor patch: the bundle's scheduler.py differs from that commit's parent) and the
request level cap 1,000 / grant 100 and park_tokens per level (cap_patch). Data runs also set level_cap_tokens 300,000 (mkV9bsft)."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TICK = '    def tick(self):\n        """Apply the rules once; safe to call every second and after level or finish events."""\n'
GUARD = "        if self._closed:\n            return\n"


def rep(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def cap_patch(sched, prio):
    """level_cap_requests: the level cap also counts requests, a loop guard. Counted from game.requests:
    gate.note_request runs only under a lane. ``equal`` exists in the bundle trees only."""
    sched = rep(sched, "    level_cap_parked_total: float | None = None\n",
                "    level_cap_parked_total: float | None = None\n    level_started_requests: int = 0\n"
                "    level_cap_request_limit: int = 0\n")
    sched = rep(sched, "        self.allowance_base = 0.0  # a new clear: a fresh allowance from the level's start\n",
                "        self.allowance_base = 0.0  # a new clear: a fresh allowance from the level's start\n"
                "        self.level_started_requests = len(self.requests)\n")
    sched = rep(sched, "    def tokens_on_level(self):\n",
                "    def requests_on_level(self):\n        return len(self.requests) - self.level_started_requests\n\n"
                "    def tokens_on_level(self):\n")
    a = sched.index("    def _level_cap_over(self, game):")
    b = sched.index("    def _level_cap_settle(self, game):")
    sched = sched[:a] + '''    def _level_cap_over(self, game):
        """At an eviction: cap ``game`` if its level reached a cap (tokens or requests); True when no grant is left."""
        cap, cap_requests = self.priority_cfg["level_cap_tokens"], self.priority_cfg["level_cap_requests"]
        if not cap and not cap_requests:
            return False
        spent, asked = game.spent_on_level(), game.requests_on_level()
        if not game.level_capped and ((cap and spent >= cap) or (cap_requests and asked >= cap_requests)):
            game.level_capped = True
            game.level_cap_limit = spent
            game.level_cap_request_limit = asked
            self.level_cap_counts["capped"] += 1
            self._priority_note("level_capped", game, tokens_on_level=int(spent), requests_on_level=asked, cap=cap,
                                cap_requests=cap_requests)
        return game.level_capped and bool((cap and spent >= game.level_cap_limit)
                                          or (cap_requests and asked >= game.level_cap_request_limit))

    def _level_cap_grant(self, game):
        game.level_cap_limit = game.spent_on_level() + self.priority_cfg["level_cap_grant_tokens"]
        game.level_cap_request_limit = game.requests_on_level() + self.priority_cfg["level_cap_grant_requests"]

''' + sched[b:]
    sched = rep(sched, '        return {"cap": self.priority_cfg["level_cap_tokens"], "grant": self.priority_cfg["level_cap_grant_tokens"],\n',
                '        return {"cap": self.priority_cfg["level_cap_tokens"], "grant": self.priority_cfg["level_cap_grant_tokens"],\n'
                '                "cap_requests": self.priority_cfg["level_cap_requests"],\n')
    sched = rep(sched, 'if self.priority_cfg["level_cap_tokens"] else {})}',
                'if self.priority_cfg["level_cap_tokens"] or self.priority_cfg["level_cap_requests"] else {})}')
    old = '            if self.priority_cfg["equal"]:\n                return (int(self._equal_parked(game)), -c[1], c[3])\n'
    if old in sched:
        sched = sched.replace(old, '            if self.priority_cfg["equal"] and not game.level_capped:\n'
                                   '                return (int(self._equal_parked(game)), -c[1], c[3])\n'
                                   '            if self.priority_cfg["equal"]:  # behind parked games too\n'
                                   '                return (2, -1.0 if game is holder else game.spent_on_level(), c[3])\n')
    old = "return bool(cap) and game.readmissions > 0 and game.clock.completion_tokens() >= cap"
    if old in sched:  # park_tokens per level, not per game; a capped game is only capped, never both
        sched = rep(sched, old, "return (bool(cap) and game.readmissions > 0 and not game.level_capped\n"
                                "                and game.tokens_on_level() >= cap)")
        sched = rep(sched, "        self.level_started_requests = len(self.requests)\n",
                    "        self.level_started_requests = len(self.requests)\n        self.equal_parked_noted = False\n")
        sched = rep(sched, 'self._priority_note("equal_parked", game, tokens=int(game.clock.completion_tokens()),\n',
                    'self._priority_note("equal_parked", game, tokens=int(game.clock.completion_tokens()),\n'
                    '                                tokens_on_level=int(game.tokens_on_level()),\n')
        prio = rep(prio, "parked at park_tokens (0 = never)", "parked at park_tokens on a level (0 = never)")
    prio = rep(prio, '    "level_cap_grant_tokens": 100000,\n',
               ' "level_cap_grant_tokens": 100000,\n    # The same cap in requests: a loop guard, set far above any legitimate level.\n'
               '    "level_cap_requests": 0,\n    "level_cap_grant_requests": 100,\n')
    prio = rep(prio, '("level_cap_grant_tokens", 1)', '("level_cap_grant_tokens", 1), ("level_cap_requests", 0), ("level_cap_grant_requests", 1)')
    return sched, prio


def apply(tree):
    tree = Path(tree)
    shutil.copyfile(ROOT / "src/arena/live_progress.py", tree / "src/arena/live_progress.py")
    p = tree / "src/arena/scheduler.py"
    text = p.read_text()
    assert "_live_knobs" not in text and text.count(TICK + GUARD) == 1
    src = subprocess.run(["git", "-C", str(ROOT), "show", "ab5cf2297:src/arena/scheduler.py"], capture_output=True,
                         text=True, check=True).stdout
    block = re.search(r"    KNOB_CAP = 64.*?(?=    def tick\(self\):)", src, re.S).group(0)
    text = text.replace(TICK + GUARD, block + TICK + GUARD + "        self._live_knobs()\n")
    q = tree / "src/arena/priority.py"
    text, prio = cap_patch(text, q.read_text())
    p.write_text(text)
    q.write_text(prio)
    c = tree / "configs/comparison.json"
    study = json.loads(c.read_text())
    study["priority"].update(level_cap_requests=1000, level_cap_grant_requests=100)
    c.write_text(json.dumps(study, indent=2) + "\n")


if __name__ == "__main__":
    apply(sys.argv[1])
