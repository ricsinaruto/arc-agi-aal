"""Offline competition-like arcade construction and public game discovery."""

import hashlib
import json
import logging
import os
from pathlib import Path

from arc_agi import Arcade, OperationMode


class CompetitionLikeOfflineArcade(Arcade):
    def make(self, *args, **kwargs):
        env = super().make(*args, **kwargs)
        if env is not None:
            response = env.observation_space
            if response is not None and not response.full_reset:
                # arc-agi 0.9.9's API does this when ONLY_RESET_LEVELS suppresses
                # the initial full_reset flag. Mirror its scorecard registration
                # for the direct offline wrapper; no game state is changed.
                self.scorecard_manager.update_scorecard(response.guid, response, True)
        return env


def source_provenance():
    files = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "files": files,
    }


def make_arcade(mode, games_dir, recordings_dir):
    # Competition Mode permits only level resets. Match it in offline benchmarks.
    os.environ["ONLY_RESET_LEVELS"] = "true"
    logger = logging.getLogger("arc.environment")
    logger.setLevel(logging.WARNING)
    if mode == "competition":
        return Arcade(
            operation_mode=OperationMode.COMPETITION,
            arc_base_url="http://gateway:8001",
            arc_api_key=os.environ.get("ARC_API_KEY", "test-key-123"),
            environments_dir="/tmp/arc-no-local-games",
            recordings_dir=str(recordings_dir),
            logger=logger,
        )
    return CompetitionLikeOfflineArcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(Path(games_dir).resolve()),
        recordings_dir=str(recordings_dir),
        logger=logger,
    )


def find_games(input_root):
    candidates = sorted(Path(input_root).rglob("environment_files"))
    valid = [p for p in candidates if p.is_dir() and len(list(p.glob("*/*.py"))) >= 25]
    # SDK assets have a version directory beneath each game in some packages.
    if not valid:
        valid = [p for p in candidates if p.is_dir() and len(list(p.rglob("*.py"))) >= 25]
    if len(valid) != 1:
        raise ValueError(f"Expected one public environment directory; found {valid}")
    return valid[0]
