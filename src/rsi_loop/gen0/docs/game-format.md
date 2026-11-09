# Writing a game (ARC engine format)

A game you add with `rsi.add_game(path, role)` is a folder:

    <path>/environment_files/<prefix>/<version>/<prefix>.py      the game (imports arcengine and numpy only)
    <path>/environment_files/<prefix>/<version>/metadata.json    {"game_id": "<prefix>-<version>", "title": ..., "default_fps": 10, "tags": [], "baseline_actions": []}
    <path>/solution.json                                         per level, the actions that clear it: [["ACTION4", "ACTION4"], ["ACTION6@3,5"], ...]

`<prefix>` is 4 characters not used by a pool or practice game (the call refuses a taken prefix). The vet replays
the solution twice (both must reach WIN with identical boards) and plays 3 random games of 300 actions: if any clears
level 2, the game is refused as trivial (the pool's rule). A pool game you add is scored like the bundled ones, with
your solution's action counts per level as the reference (the human baseline of RHAE): a long solution makes the
game easy to score on, so give your shortest one. The bundled pool games always stay in the pool.

Test a game before adding it: your kernel has `arcengine` and `arc_agi`; load it with
`arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=".../environment_files")` and
`make("<game_id>")`, then `step` through your solution.

Template (a 16x16 grid, the player walks to the goal; two levels):

```python
from arcengine import ARCBaseGame, Camera, Level, Sprite


def make_level(px, py, gx, gy):
    player = Sprite(pixels=[[9]], name="player", visible=True, collidable=True, tags=["player"])
    goal = Sprite(pixels=[[14]], name="goal", visible=True, collidable=False, tags=["goal"])
    return Level(sprites=[player.set_position(px, py), goal.set_position(gx, gy)], grid_size=(16, 16))


LEVELS = [make_level(1, 1, 5, 1), make_level(1, 1, 14, 12)]


class Tmpl(ARCBaseGame):
    def __init__(self):
        # game_id prefix, levels, camera (x, y, width, height, background, padding, displays), ..., available actions
        super().__init__("tmpl", LEVELS, Camera(0, 0, 16, 16, 5, 4, []), False, 1, [1, 2, 3, 4])

    def on_set_level(self, level):
        self.player = self.current_level.get_sprites_by_tag("player")[0]
        self.goal = self.current_level.get_sprites_by_tag("goal")[0]

    def step(self):
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(self.action.id.value, (0, 0))
        x = min(15, max(0, self.player.x + dx))
        y = min(15, max(0, self.player.y + dy))
        self.player.set_position(x, y)
        if (x, y) == (self.goal.x, self.goal.y):
            self.next_level()
        self.complete_action()
```

The arcengine package source is in your kernel's site-packages (`import arcengine; arcengine.__file__`).
