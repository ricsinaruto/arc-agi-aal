"""A game's record in a level-bench try (``GameRecord``): its clock, admissions and requests; the gate and the agent
report into it."""

from dataclasses import dataclass, field


@dataclass
class GameRecord:
    game_id: str
    index: int
    clock: object
    admission_attempts: list = field(default_factory=list)  # the gate's admission rows
    requests: list = field(default_factory=list)  # one per admitted request (requests.json)
    compactions: int = 0
    max_prompt_tokens: int = 0
    ready: object = None  # asyncio.Event set by the agent once its environment exists
