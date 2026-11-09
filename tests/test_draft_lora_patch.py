"""ARC_DRAFT_LORA (RL stage D serving side): the anchor patch on the pinned vLLM files (ARC_PINNED_SITE or
~/arc-shared/pinned-vllm; skipped if absent), its composition with expert_mask compact and lora_pin,
an unrecognized tree left unwritten, and rl_loop.draft_files (the adapter folder's draft files). Merge numerics:
tests/test_draft_lora_runtime.py."""
import ast
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc_infra import draft_lora_patch as P  # noqa: E402

SITE = Path(os.environ.get("ARC_PINNED_SITE") or Path.home() / "arc-shared/pinned-vllm")
PINNED = all((SITE / rel).is_file() for rel in P.EDITS)
needs_pinned = pytest.mark.skipif(not PINNED, reason="pinned vLLM files not available")


def copy_site(tmp_path, rels=tuple(P.EDITS)):
    for rel in rels:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SITE / rel, tmp_path / rel)
    (tmp_path / "vllm").mkdir(exist_ok=True)
    return tmp_path


def function(text, cls, name):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found")


def line_of(node, source):
    hits = [n.lineno for n in ast.walk(node) if isinstance(n, ast.stmt) and ast.unparse(n) == source]
    assert len(hits) == 1, source
    return hits[0]


@needs_pinned
def test_patch_applies_to_pinned_and_is_idempotent(tmp_path, capsys):
    site = copy_site(tmp_path / "site")
    ident = P.apply(site, tmp_path / "out")
    assert ident and set(ident) == {"helper_sha256", *P.EDITS}
    assert "ARC_DRAFT_LORA_PATCH " in capsys.readouterr().out
    assert (site / P.HELPER).read_bytes() == P.runtime_source()
    for rel in P.EDITS:
        text = (site / rel).read_text()
        compile(text, rel, "exec")
        assert text.count(P.TAG) == len(P.EDITS[rel])
    assert json.loads((tmp_path / "out/draft-lora-overlay.json").read_text()) == ident
    assert not list((tmp_path / "out").glob("*-patch.json"))  # untraced files: the boot-cache key stays
    assert P.apply(site, tmp_path / "out") == ident  # a restart on the patched tree


@needs_pinned
def test_hooks_land_in_the_right_functions():
    wm = P.patched_source(P.WORKER, (SITE / P.WORKER).read_bytes()).decode()
    load = function(wm, "WorkerLoRAManager", "_load_adapter")
    stamp = line_of(load, "lora.is_3d_lora_weight = lora_request.is_3d_lora_weight")
    assert line_of(load, "lora._arc_draft = _arc_dl.read(lora_path)") == stamp + 2  # inside the try

    mm = P.patched_source(P.MANAGER, (SITE / P.MANAGER).read_bytes()).decode()
    act = function(mm, "LoRAModelManager", "activate_adapter")
    assert ast.unparse(act.body[-2]) == "_arc_dl.activate(lora_model)" and ast.unparse(act.body[-1]) == "return True"
    assert isinstance(act.body[-4], ast.For)  # after every module's set_lora / reset_lora
    early = act.body[1]  # an already-active id returns before (per-batch re-activation never re-merges)
    assert isinstance(early, ast.If) and ast.unparse(early.body[0]) == "return False"
    deact = function(mm, "LoRAModelManager", "_deactivate_adapter")
    assert ast.unparse(deact.body[1]) == "_arc_dl.deactivate(lora_id)"
    clear = function(mm, "LoRAModelManager", "remove_all_adapters")
    assert ast.unparse(clear.body[2]) == "_arc_dl.deactivate_all()"

    sp = P.patched_source(P.SPECULATOR, (SITE / P.SPECULATOR).read_bytes()).decode()
    load = [ast.unparse(s) for s in function(sp, "DraftModelSpeculator", "load_model").body]
    i = load.index("self.model = self.load_draft_model(target_model, target_attn_layer_names)")
    assert load[i + 2] == "_arc_dl.register_draft(self.model)"


@needs_pinned
def test_composes_with_expert_mask_compact_and_lora_pin(tmp_path):
    from arc_infra import expert_mask_patch, lora_pin_patch

    original = (SITE / P.MANAGER).read_bytes()
    target = next(t for t in expert_mask_patch.COMPACT_TARGETS if t["relative"] == P.MANAGER)
    compact = expert_mask_patch.compact_source(target, original)  # sha-guarded: runs first in every boot
    anchor = dict(lora_pin_patch.TARGETS)[P.MANAGER]
    pin = lambda b: lora_pin_patch.patched_source(b, anchor, P.MANAGER)  # noqa: E731
    ours = lambda b: P.patched_source(P.MANAGER, b)  # noqa: E731
    for first, second in ((pin, ours), (ours, pin)):
        text = second(first(compact)).decode()
        compile(text, P.MANAGER, "exec")
        assert P.TAG in text and lora_pin_patch.ENV in text and text.count(P.TAG) == 3
    # the whole tree in boot order: lora_pin, draft_lora
    rels = set(P.EDITS) | {rel for rel, _ in lora_pin_patch.TARGETS}
    site = copy_site(tmp_path / "site", tuple(rels))
    lora_pin_patch.apply(site, tmp_path / "out")
    assert P.apply(site, tmp_path / "out") is not None


def test_unrecognized_file_writes_nothing(tmp_path, capsys):
    for rel in P.EDITS:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x = 1\n")
    assert P.apply(tmp_path, tmp_path / "out") is None
    assert "ARC_DRAFT_LORA_PATCH_SKIPPED" in capsys.readouterr().out
    assert all((tmp_path / rel).read_text() == "x = 1\n" for rel in P.EDITS)
    assert not (tmp_path / P.HELPER).exists()


@needs_pinned
def test_half_recognized_tree_writes_nothing(tmp_path):
    site = copy_site(tmp_path)
    (site / P.SPECULATOR).write_text("x = 1\n")
    before = {rel: (site / rel).read_bytes() for rel in (P.WORKER, P.MANAGER)}
    assert P.apply(site, tmp_path / "out") is None
    assert all((site / rel).read_bytes() == data for rel, data in before.items())


def test_serving_setup_gates_on_the_flag():
    src = (ROOT / "src/arc_infra/serving_setup.py").read_text()
    assert '("ARC_DRAFT_LORA", "draft_lora", "draft_lora_patch", "apply", True)' in src


# ----------------------------------------------------------------------------- the adapter folder's draft files
def safetensors_header(path, names):
    head = json.dumps({n: {"dtype": "BF16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]}
                       for i, n in enumerate(names)}).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"\0" * 2 * len(names))


def test_draft_files(tmp_path):
    from arena.rl_loop import draft_files

    assert draft_files(tmp_path) == 0
    with pytest.raises(ValueError, match="no draft_lora.json"):
        draft_files(tmp_path, need=True)
    mods = {"fc_hidden": {"vllm": "model.fc_hidden", "shard": None},
            "layers.0.self_attn.q_proj": {"vllm": "model.layers.0.self_attn.qkv_proj", "shard": "q"}}
    (tmp_path / "draft_lora.json").write_text(json.dumps({"format": "arc-draft-lora-v1", "r": 4, "lora_alpha": 8,
                                                          "modules": mods}))
    with pytest.raises(ValueError, match="unreadable"):  # json without its tensors
        draft_files(tmp_path)
    keys = [f"base_model.model.mtp.{m}.lora_{p}.weight" for m in mods for p in "AB"]
    safetensors_header(tmp_path / "draft_lora.safetensors", keys[:3])
    with pytest.raises(ValueError, match="3 tensors for 2 modules"):
        draft_files(tmp_path)
    safetensors_header(tmp_path / "draft_lora.safetensors", keys)
    assert draft_files(tmp_path, need=True) == 2
