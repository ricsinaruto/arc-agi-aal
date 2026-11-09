"""Observation rendering and system-prompt rules for baselines A and B."""

import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from arc_infra.perception import PALETTE
from qwen_baselines import prompts
from qwen_baselines.config import Config
from qwen_baselines.evidence import Archive, grid_text


def rgb(color_id):
    c = PALETTE[color_id]
    return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))


def archive_with_animation(tmp_path):
    archive = Archive(tmp_path / "game")
    first = np.zeros((64, 64), dtype=np.uint8)
    first[0, 1] = 3
    settled = np.full((64, 64), 5, dtype=np.uint8)
    settled[0, 0] = 9
    archive.append({"status": "NOT_FINISHED"}, [first, settled])
    return archive, first, settled


def decode(part):
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    return Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


def test_observation_settled_only_and_memory(tmp_path):
    archive, first, settled = archive_with_animation(tmp_path)
    content = prompts.observation(archive, settled_only=True, memory="tiles toggle")["content"]
    assert content[0]["text"] == "Memory from your previous turn:\ntiles toggle"
    assert content[1]["text"].startswith("Observation: ")
    images = [p for p in content if p["type"] == "image_url"]
    labels = [p["text"] for p in content if p["type"] == "text"][2:]
    assert len(images) == 1 and labels == [
        "obs-000000 frame 1: settled (1 intermediate frames omitted)"
    ]
    assert decode(images[0]).getpixel((0, 0)) == rgb(9)
    plain = prompts.observation(archive, settled_only=True)["content"]
    assert plain[0]["text"].startswith("Observation: ")
    viewed = prompts.observation(archive, settled_only=True, objects=archive.object_view())
    texts = [p["text"] for p in viewed["content"] if p["type"] == "text"]
    assert texts[1].startswith("Objects (settled frame):\nobjects: ")
    assert "bbox=[" in texts[1] and texts[2].startswith("obs-000000 frame 1")


def test_memory_block_extraction():
    assert prompts.memory_block("thinking...\nMEMORY:\nrule: click toggles\nplan: x") == (
        "rule: click toggles\nplan: x"
    )
    assert prompts.memory_block("MEMORY: one line") == "one line"
    assert prompts.memory_block("no block here; MEMORY: mid-line does not count") is None
    assert prompts.memory_block("MEMORY:\n") is None and prompts.memory_block(None) is None
    assert prompts.memory_block([{"type": "text", "text": "a\nMEMORY:\nb"}]) == "b"


def test_observation_is_images_only_at_512(tmp_path):
    archive, first, settled = archive_with_animation(tmp_path)
    content = prompts.observation(archive)["content"]
    assert content[0]["text"].startswith("Observation: ")
    assert not any(p.get("text", "").startswith("Settled grid:") for p in content)
    images = [p for p in content if p["type"] == "image_url"]
    labels = [p["text"] for p in content if p["type"] == "text"][1:]
    assert len(images) == 2 and labels == [
        "obs-000000 frame 0: intermediate",
        "obs-000000 frame 1: settled",
    ]
    frame0, frame1 = (decode(p) for p in images)
    assert frame0.size == frame1.size == (512, 512)
    # 8 pixels per cell, nearest neighbour: cell (x=1, y=0) starts at pixel x=8.
    assert frame0.getpixel((0, 0)) == rgb(0) and frame0.getpixel((8, 0)) == rgb(3)
    assert frame0.getpixel((15, 7)) == rgb(3) and frame0.getpixel((16, 0)) == rgb(0)
    assert frame1.getpixel((0, 0)) == rgb(9) and frame1.getpixel((8, 8)) == rgb(5)


def test_observation_legacy_grid_and_256(tmp_path):
    archive, first, settled = archive_with_animation(tmp_path)
    content = prompts.observation(archive, text_grid=True, pixels=256)
    grid = [p["text"] for p in content["content"] if p.get("text", "").startswith("Settled grid:")]
    assert grid == ["Settled grid:\n" + grid_text(settled)]
    images = [decode(p) for p in content["content"] if p["type"] == "image_url"]
    assert [i.size for i in images] == [(256, 256), (256, 256)]


def test_system_prompt_rules():
    a = prompts.system("A")
    assert "At most 20 actions are executed per response" in a
    assert "act takes an ordered list of 1-20 actions" in a
    assert "8 pixels per cell" in a and "Color legend (ID=hex RGB): 0=#" in a and "10=#" in a
    assert "Grid symbols" not in a
    for text in (a, prompts.system("B"), prompts.system("C")):
        assert "HUD element such as" in text and "re-read the new" in text
        assert "stop probing and search" in text and "MEMORY:" in text
        assert "below 500" in text
    assert "End every response with" in a and "End every turn" in prompts.system("C")
    b = prompts.system("B")
    assert "act takes" not in b and "segment(board)" in b and "never print a whole board" in b
    assert "only the settled frame image" in b
    assert "from segmentation import segment" in prompts.PI_API
    assert prompts.ONE_ACTION.strip() in prompts.system("A", actions_per_turn=1)
    legacy = prompts.system("A", text_grid=True, image_pixels=256, actions_per_turn=None)
    assert "Grid symbols 0-9,A-F" in legacy and "A=#" in legacy
    assert "At most" not in legacy and "ordered list" in legacy
    assert "At most 3 actions" in prompts.system("A", actions_per_turn=3)
    schema = prompts.tools("A")[0]["function"]["parameters"]
    assert schema["required"] == ["actions"] and schema["properties"]["actions"]["maxItems"] == 20
    item = schema["properties"]["actions"]["items"]
    assert item["required"] == ["action"] and "x" in item["properties"]


def test_config_validation():
    with pytest.raises(ValueError, match="multiple of 64"):
        Config(image_pixels=100)
    with pytest.raises(ValueError, match="at least 1"):
        Config(actions_per_turn=0)
    legacy = Config(text_grid=True, image_pixels=256, actions_per_turn=None, preserve_thinking=True)
    assert legacy.to_dict()["actions_per_turn"] is None
    assert Config().to_dict()["preserve_thinking"] is False
    cfg = Config()
    assert cfg.actions_per_turn == 20 and cfg.history_tokens == 65536
    assert cfg.compaction_tokens == 65536
    # v4 mechanisms default on for B/C except the per-observation object view (off since
    # the 18 September B/C v4 result) and off in the faithful-port profiles.
    assert not cfg.object_view and cfg.transition_summary and cfg.suppress_grid_dumps
    assert cfg.reject_no_change_repeats and cfg.object_view_limit == 40
    assert "object view" in prompts.system("C", object_view=True)
    assert "object view" not in prompts.system("C") and "transition summary" in prompts.system("C")
    with pytest.raises(ValueError, match="object_view_limit"):
        Config(object_view_limit=0)
    from prolong_qwen.profile import P_DEFAULTS
    from retrodict_qwen.profile import R_DEFAULTS
    from schema_qwen.profile import S_DEFAULTS

    for profile in (P_DEFAULTS, R_DEFAULTS, S_DEFAULTS):
        off = Config(**profile)
        assert not (off.object_view or off.transition_summary or off.suppress_grid_dumps)
        assert not off.reject_no_change_repeats and off.actions_per_turn is None
    with pytest.raises(ValueError, match="history_tokens"):
        Config(history_tokens=300000)
    assert Config(history_tokens=None).history_tokens is None
