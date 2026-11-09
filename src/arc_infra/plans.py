"""Validated game actions."""

from dataclasses import dataclass, field


@dataclass
class Action:
    name: str
    x: int | None = None
    y: int | None = None
    expect: list[list[int]] = field(default_factory=list)
    expect_levels: int | None = None
    allow_repeat: bool = False

    def payload(self):
        result = {"action": self.name}
        if self.name == "ACTION6":
            result.update(x=self.x, y=self.y)
        if self.expect:
            result["expect"] = self.expect
        if self.expect_levels is not None:
            result["expect_levels"] = self.expect_levels
        if self.allow_repeat:
            result["allow_repeat"] = True
        return result

    def identity(self):
        return self.name, self.x, self.y


def integer(value, lower, upper, label):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{label} must be an integer in {lower}..{upper}")
    return value
