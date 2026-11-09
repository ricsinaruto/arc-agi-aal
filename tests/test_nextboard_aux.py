import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))

import nextboard_aux as nb  # noqa: E402
from arena import nextboard_pack as nbp  # noqa: E402

TOKENIZER = Path.home() / "arc-sft-data" / "tok-QKsX"  # Flash's own tokenizer and chat template (qwen4_exp)


def board(cells=()):
    rows = [["0"] * 64 for _ in range(64)]
    for r, c, v in cells:
        rows[r][c] = v
    return ["".join(r) for r in rows]


def obs(step, b, levels=0, state="NOT_FINISHED", action=None):
    return {"kind": "observation", "step": step, "frames": [board(), b], "levels": levels, "state": state,
            "action": action}


def write_log(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))


def ladder(folder):
    """Level 0: three moves (one no-op), a clear; level 1: two moves, a RESET, one move."""
    a = lambda n, **k: {"action": n, **k}
    rows = [obs(0, board()), obs(1, board([(1, 1, "3")]), action=a("ACTION1")),
            obs(2, board([(1, 1, "3")]), action=a("ACTION2")),
            obs(3, board([(1, 1, "3"), (2, 2, "4")]), action=a("ACTION6", x=2, y=2)),
            obs(4, board([(5, 5, "9")]), levels=1, action=a("ACTION1")),  # the clear: next level's board
            obs(5, board([(5, 6, "9")]), levels=1, action=a("ACTION4")),
            obs(6, board([(5, 7, "9")]), levels=1, action=a("ACTION4")),
            obs(7, board([(5, 5, "9")]), levels=1, action=a("RESET")),
            obs(8, board([(6, 5, "9")]), levels=1, action=a("ACTION2"))]
    write_log(folder / "log.jsonl", [{"kind": "harness", "x": 1}] + rows)
    return folder


def test_level_attempts_split_and_clear_left_out(tmp_path):
    atts = nbp.level_attempts(ladder(tmp_path / "t1") / "log.jsonl", min_steps=1)
    assert [(a["level"], len(a["actions"]), a["end"]) for a in atts] == [(0, 3, "level_change"), (1, 2, "reset"),
                                                                         (1, 1, "eof")]
    assert atts[0]["actions"][2] == {"action": "ACTION6", "x": 2, "y": 2}
    assert len(atts[0]["boards"]) == 4  # first board + one per action; the clearing move's board is not in it


def test_render_targets_and_board_every(tmp_path):
    att = nbp.level_attempts(ladder(tmp_path / "t1") / "log.jsonl")[0]
    part = nbp.render(att, board_every=2)[0]
    turns = part["turns"]
    assert turns[0]["user"].startswith(nbp.INTRO) and turns[0]["user"].endswith("Action: UP")
    assert turns[1]["user"] == "Action: DOWN" and turns[1]["target"] == "NO CHANGE" and turns[1]["changed"] == 0
    assert turns[2]["user"].startswith("Board now:") and turns[2]["user"].endswith("Action: MOUSE(row=2, col=2)")
    assert turns[0]["target"].startswith("@ row=1 col=1")
    assert [len(p["turns"]) for p in nbp.render(att, max_steps=2)] == [2, 1]
    assert nbp.render(att, max_steps=2)[1]["turns"][0]["user"].startswith(nbp.INTRO)


def test_build_skips_held_out_and_counts(tmp_path):
    t1, t2 = ladder(tmp_path / "a"), ladder(tmp_path / "b")
    rows, summary = nbp.build([("sk48", t1), ("tr87", t2), ("q586", tmp_path / "missing")], k=8, min_steps=1)
    assert {r["game"] for r in rows} == {"sk48"}
    assert summary["attempts"] == 3 and summary["steps"] == 6 and summary["copy_em"] == round(1 / 6, 4)
    rows2, _ = nbp.build([("sk48", t1), ("cn04", t2)], k=8, min_steps=1)
    assert len(rows2) == 3  # the second try repeats every transition: nothing new, not picked
    nbp.write(rows, tmp_path / "out" / "r01.jsonl")
    assert len((tmp_path / "out" / "r01.jsonl").read_text().splitlines()) == 3


class CharTok:
    """One token per character, special tokens whole (the joins are exact by construction)."""
    SPECIAL = ("<|im_start|>", "<|im_end|>")

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for k, s in enumerate(self.SPECIAL):
                if text.startswith(s, i):
                    ids.append(100000 + k)
                    i += len(s)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return {"input_ids": ids}


def row_of(tmp_path):
    att = nbp.level_attempts(ladder(tmp_path / "t1") / "log.jsonl")[0]
    return nbp.render(att, board_every=2)[0]


def test_encode_spans_and_cut(tmp_path):
    row = row_of(tmp_path)
    tok = CharTok()
    enc = nb.encode(tok, row)
    assert enc["ids"] == tok(nb.text_of(row))["input_ids"]
    for (s, e, step), turn in zip(enc["spans"], row["turns"]):
        assert enc["ids"][s:e] == tok(turn["target"] + "<|im_end|>")["input_ids"]
    row["turns"][1]["train"] = False
    enc2 = nb.encode(tok, row)
    assert [st for _, _, st in enc2["spans"]] == [0, 2]
    short = nb.encode(tok, row, max_tokens=enc["spans"][1][1] + 1)
    assert len(short["spans"]) == 1  # step 1 is untrained; step 2 does not fit


def test_made_and_stats(tmp_path):
    import torch

    enc = nb.encode(CharTok(), row_of(tmp_path))
    m = nb.made(enc)
    pos = m["arrays"]["positions"]
    assert m["tail"] == len(enc["ids"]) and int(pos.numel()) == sum(e - s for s, e, _ in enc["spans"])
    assert int(pos.min()) >= 1 and bool((m["arrays"]["support"] == -1).all())
    n = int(pos.numel())
    hit = torch.ones(n, dtype=torch.bool)
    s0, e0, _ = enc["spans"][0]
    hit[1] = False  # inside step 0's body
    rows = nb.step_stats(enc, -0.5 * torch.ones(n), hit)
    assert [r["em"] for r in rows] == [False, True, True] and rows[0]["body"] is False
    p = nb.pooled(rows)
    assert p["steps"] == 3 and p["em"] == round(2 / 3, 4) and p["nll"] == 0.5 and p["changed"]["steps"] == 2


@pytest.mark.skipif(not TOKENIZER.exists(), reason="Flash tokenizer not on this machine")
def test_encode_matches_flash_template(tmp_path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    row = row_of(tmp_path)
    messages = []
    for t in row["turns"]:
        messages += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["target"]}]
    rendered = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    assert rendered == nb.text_of(row)
    enc = nb.encode(tok, row)
    assert enc["ids"] == tok(rendered, add_special_tokens=False)["input_ids"]
    for (s, e, _), t in zip(enc["spans"], row["turns"]):
        assert tok.decode(enc["ids"][s:e]) == t["target"] + "<|im_end|>"
    # the reply opens where a no-think generation prompt ends
    prompt = tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert tok(prompt, add_special_tokens=False)["input_ids"] == enc["ids"][:enc["spans"][0][0]]


def test_fake_trainer_and_keys(tmp_path):
    from arena import fake_trainer
    from arena.rl_loop import check_mg

    rows, _ = nbp.build([("sk48", ladder(tmp_path / "a"))], k=4, min_steps=1)
    path = nbp.write(rows, tmp_path / "r01.jsonl")
    assert fake_trainer.nb_fake(str(path), 2)["windows"] == 2
    assert fake_trainer.nb_fake(str(path), None)["windows"] == 3
    check_mg({"nextboard": {"weight": 0.1, "windows": 2, "read": "x", "read_every": 2}})
    with pytest.raises(ValueError):
        check_mg({"nextboard": {"wieght": 0.1}})


def test_trainer_aux_pass_scale_and_cursor(tmp_path):
    """rl1_trainer.Trainer.aux_pass with a stand-in window pass: the step's summed loss is weight x mean target NLL,
    the cursor walks the file and wraps, a read covers every row without moving it."""
    torch = pytest.importorskip("torch")
    try:
        import rl1_trainer
    except Exception as error:  # the trainer's own imports (flash_qlora_job, rl_job) need the training stack
        pytest.skip(f"rl1_trainer not importable here: {error!r}")
    rows, _ = nbp.build([("sk48", ladder(tmp_path / "a"))], k=4, min_steps=1)
    path = str(nbp.write(rows, tmp_path / "r01.jsonl"))

    class T:
        aux_rows = rl1_trainer.Trainer.aux_rows
        aux_pass = rl1_trainer.Trainer.aux_pass
        aux_device = "cpu"
        tokenizer = CharTok()
        seen = []

        def window_pass(self, made, scale=None):
            n = int(made["arrays"]["positions"].numel())
            raw = -0.25 * torch.ones(n)
            self.seen.append((made["name"], n, scale))
            return {"raw": raw, "hit": torch.ones(n, dtype=torch.bool),
                    "loss": None if scale is None else float(-(raw.sum()) * scale)}

    t = T()
    out = t.aux_pass({"file": path, "weight": 0.2, "windows": 2})
    assert out["windows"] == 2 and abs(out["loss"] - 0.2 * 0.25) < 1e-9 and out["em"] == 1.0
    assert t.aux_cursor[path] == 2
    t.aux_pass({"file": path, "weight": 0.2, "windows": 2})
    assert t.aux_cursor[path] == 1  # 3 rows: wrapped
    t.seen.clear()
    read = t.aux_pass({"file": path}, grad=False)
    assert read["windows"] == 3 and all(s is None for _, _, s in t.seen) and "sk48" in read["by_game"]
    assert t.aux_cursor[path] == 1


def test_patch_text_matches_aux_tasks():
    """The scipy-free patch equals posttrain.aux_tasks.patch_text on random sparse and blocky changes."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from inference.utils.grid_utils import format_grid_ascii
    from posttrain.aux_tasks import patch_text

    rng = np.random.default_rng(0)
    for k in range(300):
        a = rng.integers(0, 16, (64, 64))
        b = a.copy()
        for _ in range(int(rng.integers(0, 6))):
            r, c, h, w = (int(x) for x in rng.integers(0, 64, 4))
            b[r:r + 1 + h % 9, c:c + 1 + w % 9] = rng.integers(0, 16)
        sparse = rng.random((64, 64)) < (0.002 if k % 2 else 0.02)
        b[sparse] = (b[sparse] + 1) % 16
        assert nbp.patch_text(a, b) == patch_text(a, b)
    assert nbp.ascii_board(a) == format_grid_ascii(a.tolist())
