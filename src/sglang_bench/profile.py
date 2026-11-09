"""Server profiles and sweep points for the SGLang benchmark.

Pennyroyal's ``next`` recipe admits four requests with 24 Mamba slots and an 824,384-token
KV pool; its C6 profile uses 6 / 36 / 1,048,576. The sweep keeps the 6-slots-per-request
ratio and plays as many games as the server admits, so no request ever queues.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

MAMBA_SLOTS_PER_ADMITTED = 6
RECIPE_POOL_TOKENS = 824_384
C6_POOL_TOKENS = 1_048_576
PAGE_SIZE = 64
CORE_GAMES = ("ft09-0d8bbf25", "lp85-305b61c3", "ar25-0c556536")
PUBLIC_NOTEBOOK_CONFIG = Path(__file__).resolve().parents[2] / "configs/duck_public_notebook.json"


@dataclass(frozen=True)
class ServerProfile:
    admitted: int
    mamba_slots: int
    max_total_tokens: int
    online_fp8: bool = True
    context_length: int = 524_288
    served_model_name: str = "pennyroyal"
    port: int = 8001
    host: str = "127.0.0.1"
    page_size: int = PAGE_SIZE
    chunked_prefill_size: int = 4096
    mm_preprocess_device: str = "cpu"
    ple_backend: str = "ram"
    weight_loader_disable_mmap: bool = False
    weight_loader_threads: int | None = None

    def validate(self, *, allow_custom_slots: bool = False) -> ServerProfile:
        if type(self.weight_loader_disable_mmap) is not bool:
            raise ValueError("weight_loader_disable_mmap must be a boolean")
        if self.weight_loader_threads is not None and (
            type(self.weight_loader_threads) is not int or self.weight_loader_threads < 1
        ):
            raise ValueError("weight_loader_threads must be a positive integer or None")
        for name in ("admitted", "mamba_slots", "max_total_tokens", "context_length", "port"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.max_total_tokens % self.page_size:
            raise ValueError(f"max_total_tokens must align to page size {self.page_size}")
        if not allow_custom_slots and self.mamba_slots != MAMBA_SLOTS_PER_ADMITTED * self.admitted:
            raise ValueError("mamba_slots must be six per admitted request (the recipe's ratio)")
        if self.mm_preprocess_device != "cpu" and not self.mm_preprocess_device.startswith("cuda:"):
            raise ValueError("mm_preprocess_device must be cpu or cuda:N")
        if self.ple_backend != "ram":
            raise ValueError("Only the RAM PLE backend is packaged")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ServerProfile:
        return cls(**data).validate(allow_custom_slots=True)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def for_admitted(admitted: int, *, online_fp8: bool = True) -> ServerProfile:
    """The recipe's capacity for ``admitted`` concurrent requests."""
    pool = C6_POOL_TOKENS if admitted >= 6 else RECIPE_POOL_TOKENS
    return ServerProfile(
        admitted=admitted,
        mamba_slots=MAMBA_SLOTS_PER_ADMITTED * admitted,
        max_total_tokens=pool,
        online_fp8=online_fp8,
    ).validate()


RECIPE_DEFAULT = for_admitted(4)


def public_game_ids(config_path: Path | str = PUBLIC_NOTEBOOK_CONFIG) -> list[str]:
    return list(json.loads(Path(config_path).read_text())["public_game_ids"])


def games_for(admitted: int, game_ids: list[str] | None = None) -> list[str]:
    """The three protocol games, then the public notebook's order for the rest."""
    game_ids = public_game_ids() if game_ids is None else list(game_ids)
    missing = [g for g in CORE_GAMES if g not in game_ids]
    if missing:
        raise ValueError(f"Public game list lacks {missing}")
    extra = [g for g in game_ids if g not in CORE_GAMES]
    if admitted > len(CORE_GAMES) + len(extra):
        raise ValueError(f"Only {len(game_ids)} public games for {admitted} admitted requests")
    return list(CORE_GAMES)[:admitted] + extra[: max(0, admitted - len(CORE_GAMES))]


def sweep_points(
    admitted: tuple[int, ...] = (3, 4, 6), game_ids: list[str] | None = None, *, online_fp8: bool = True
) -> list[dict]:
    return [
        {
            "name": f"c{count}",
            "profile": for_admitted(count, online_fp8=online_fp8).to_dict(),
            "games": games_for(count, game_ids),
        }
        for count in admitted
    ]


def smoke_point(game: str = CORE_GAMES[0]) -> dict:
    """The recipe as shipped (4 / 24 / 824,384) on one game."""
    return {"name": "smoke", "profile": RECIPE_DEFAULT.to_dict(), "games": [game]}


def with_overrides(profile: ServerProfile, **changes) -> ServerProfile:
    return replace(profile, **changes).validate(allow_custom_slots=True)
