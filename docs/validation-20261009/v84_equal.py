"""v8.4: the long runs' equal scheduler with parking, as source patches on the v8.2 arena.

Copied from the dfr-exact branch (scripts/dfr_exact/build_v82_his_serving.py, used by the equal-scheduler long runs
A v2 / B v3 and the serving ablation), unchanged.
"""
from pathlib import Path

# Equal scheduler, as ``priority.equal`` inside the priority policy's state machine: the
# hand-over points stay (context drains/compactions, finished games); the waiting game with the fewest generated tokens
# (clock.completion_tokens; unstarted 0, ties in arrival = roster order) takes a free slot; a game with park_tokens or
# more ranks after every game under it (parked), fewest tokens first among parked games.
PRIORITY_OLD = '''    "level_cap_tokens": 0,
    "level_cap_grant_tokens": 100000,
}
'''
PRIORITY_NEW = '''    "level_cap_tokens": 0,
    "level_cap_grant_tokens": 100000,
    # Equal scheduler: fewest generated tokens first, parked at park_tokens (0 = never).
    "equal": False,
    "park_tokens": 0,
}
'''
SETTINGS_OLD = '''    for key in ("refresh_queue", "tail_fade", "normalize_score", "floor"):'''
SETTINGS_NEW = '''    for key in ("refresh_queue", "tail_fade", "normalize_score", "floor", "equal"):'''
SETTINGS2_OLD = '''    for key, least in (("level_cap_tokens", 0), ("level_cap_grant_tokens", 1)):'''
SETTINGS2_NEW = '''    for key, least in (("level_cap_tokens", 0), ("level_cap_grant_tokens", 1), ("park_tokens", 0)):'''
VALUE_OLD = '''        from .priority import UNSTARTED_BASE, score

        if game.readmissions == 0:'''
VALUE_NEW = '''        from .priority import UNSTARTED_BASE, score

        if self.priority_cfg["equal"]:
            tokens = 0 if game.readmissions == 0 else int(game.clock.completion_tokens())
            return -tokens, {"tokens": tokens, "parked": self._equal_parked(game)}
        if game.readmissions == 0:'''
RANK_OLD = '''        def key(c):
            game = c[0]
            if not game.level_capped:'''
RANK_NEW = '''        def key(c):
            game = c[0]
            if self.priority_cfg["equal"]:
                return (int(self._equal_parked(game)), -c[1], c[3])
            if not game.level_capped:'''
PARKED_OLD = '''    # --- level cap (priority.level_cap_tokens) ---------------------------------------------
'''
PARKED_NEW = '''    def _equal_parked(self, game):
        cap = self.priority_cfg["park_tokens"]
        return bool(cap) and game.readmissions > 0 and game.clock.completion_tokens() >= cap

    # --- level cap (priority.level_cap_tokens) ---------------------------------------------
'''
NOTE_OLD = '''        over = self._level_cap_over(game)
'''
NOTE_NEW = '''        over = self._level_cap_over(game)
        if self.priority_cfg["equal"] and self._equal_parked(game) and not getattr(game, "equal_parked_noted", False):
            game.equal_parked_noted = True
            self.priority_counts["equal_parked"] = self.priority_counts.get("equal_parked", 0) + 1
            self._priority_note("equal_parked", game, tokens=int(game.clock.completion_tokens()),
                                park_tokens=self.priority_cfg["park_tokens"])
'''



def patch(path, old, new):
    path = Path(path)
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"{path}: patch anchor not found once")
    path.write_text(text.replace(old, new))


def apply(out):
    """Patch ``out/src/arena`` (priority.py: the equal and park_tokens settings; scheduler.py: ranking and parking)."""
    for old, new in ((PRIORITY_OLD, PRIORITY_NEW), (SETTINGS_OLD, SETTINGS_NEW), (SETTINGS2_OLD, SETTINGS2_NEW)):
        patch(Path(out) / "src/arena/priority.py", old, new)
    for old, new in ((VALUE_OLD, VALUE_NEW), (RANK_OLD, RANK_NEW), (PARKED_OLD, PARKED_NEW), (NOTE_OLD, NOTE_NEW)):
        patch(Path(out) / "src/arena/scheduler.py", old, new)
