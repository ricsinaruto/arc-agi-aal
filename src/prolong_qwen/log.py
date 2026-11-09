"""The PRO-LONG game log, written exactly as upstream's runner writes it."""

from pathlib import Path

from .game_state import GameState


class GameLog:
    """`logs.txt`, `current_board.txt` and `run_info.txt` in the agent workspace."""

    def __init__(self, workspace, grid_mode="ascii"):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.path = self.workspace / "logs.txt"
        self.path.write_text("")
        self.state = GameState(grid_mode=grid_mode)

    def observe(self, frames, status, score):
        self.state.record_env_update({"frame": frames, "state": status, "score": score})

    def append(self, text):
        with open(self.path, "a", encoding="utf-8") as output:
            output.write(text)

    def initial(self, level, attempt, score):
        grid = self.state.render_board()
        if grid:
            self.append(
                f"{'=' * 80}\n"
                f"Action 0 | Level {level} | Attempt {attempt} | INITIAL STATE | Score: {score}\n\n"
                f"[INITIAL BOARD STATE]\n{grid}\n\n"
            )

    def action(self, number, level, attempt, plan_index, plan_total, score, name, data):
        plan_info = f" | Plan Step {plan_index}/{plan_total}" if plan_total > 0 else ""
        text = (
            f"\n{'=' * 80}\n"
            f"Action {number} | Level {level} | Attempt {attempt}{plan_info} | Score: {score}\n\n"
        )
        hint = self.state.consume_hint_block()
        if hint:
            text += f"{hint}\n"
        if name == "ACTION6":
            import json

            text += f"Tool Call: {name}({json.dumps(data)})\n"
        else:
            text += f"Tool Call: {name}({{}})\n"
        self.append(text)

    def post_board(self):
        grid = self.state.render_board()
        if grid:
            self.append(f"[POST-ACTION BOARD STATE]\n{grid}\n\n")

    def current_board(self, score, action_number):
        grid = self.state.render_board(include_animation=False)
        with open(self.workspace / "current_board.txt", "w", encoding="utf-8") as output:
            output.write(f"Score: {score}\nAction: {action_number}\n\n")
            if grid:
                output.write(f"[CURRENT BOARD STATE]\n{grid}\n")

    def run_info(self, game_id, command):
        (self.workspace / "run_info.txt").write_text(f"game_id: {game_id}\ncommand: {command}\n")
