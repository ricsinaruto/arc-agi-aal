"""heal_kd plan: the CPU dry run over the real files refuses each past harness failure. Needs torch
(Modal CPU)."""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import heal_kd  # noqa: E402

K = 4


def fake(path, n, reply_from=None, tok_logp=True, loop=False):
    """An op_topk-format file of n tokens; reply positions from reply_from on (None: none)."""
    g = torch.Generator().manual_seed(len(path.name) + n)
    ids = torch.randint(10, 1000, (n,), generator=g)
    reply = torch.zeros(n, dtype=torch.bool)
    if reply_from is not None:
        reply[reply_from:] = True
        if loop:  # a degenerate repeat filling the reply
            ids[reply_from:] = torch.tensor([7, 8, 9, 11] * n)[: n - reply_from]
    d = {"ids": ids.int(), "image": torch.zeros(n, dtype=torch.bool), "reply": reply,
         "top_logp": torch.full((n, K), -1.5).half(), "top_ids": torch.randint(10, 1000, (n, K), generator=g).int(),
         "tail": torch.full((n,), -2.0).half()}
    if tok_logp:
        d["tok_logp"] = torch.full((n,), -1.0)
    torch.save(d, path)
    return int(reply[1:].sum())


def folder(tmp, name, specs):
    f = tmp / name
    f.mkdir(parents=True, exist_ok=True)
    rows = [{"file": n, "reply_tokens": fake(f / n, *s[:2], **(s[2] if len(s) > 2 else {})),
             "tokens": s[0]} for n, *s in specs]
    (f / "index.json").write_text(json.dumps({"rows": rows}))
    return f


def plan(tmp, monkeypatch, teacher, *extra, world=2):
    out = tmp / f"out{len(list(tmp.glob('out*')))}"
    monkeypatch.setattr(sys, "argv", ["heal_kd", "plan", "--teacher", str(teacher), "--out", str(out),
                                      "--world", str(world), "--batch-tokens", "1000", "--warmup", "0.1", *extra])
    code = 0
    try:
        heal_kd.main()
    except SystemExit as e:
        code = e.code
    got = json.loads((out / "plan.json").read_text())
    return code, got


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ("HEAL_SAMPLED", "HEAL_HELDOUT2", "HEAL_SHARD_READS", "HEAL_READ_SELECT", "HEAL_MID_READ_EVERY",
              "HEAL_REWARM", "HEAL_SCHEDULE_STEPS", "HEAL_ALLOW_TOPK_TTOK", "HEAL_PREREAD_FROM"):
        monkeypatch.delenv(k, raising=False)


def test_healthy_plan_passes(tmp_path, monkeypatch):
    t = folder(tmp_path, "t", [(f"s-{i}.pt", 900, 100) for i in range(8)])
    code, got = plan(tmp_path, monkeypatch, t, "--total-steps", "0")
    assert code in (0, None) and not got["problems"]
    assert got["steps_fit"] == 4 and got["total_steps"] == 3 and got["slots"] == 4  # a 2 x 799 slot fills a step


def test_no_reply_within_cut_refused(tmp_path, monkeypatch):
    # the r1b preflight's 12K cut: 8 of 67 files kept a reply; here none does
    t = folder(tmp_path, "t", [(f"s-{i}.pt", 600, None) for i in range(4)])
    code, got = plan(tmp_path, monkeypatch, t, "--total-steps", "2")
    assert code == 1 and any("no file has a reply" in p for p in got["problems"])
    assert any("schedule" in p for p in got["problems"])


def test_resume_world_mismatch_refused(tmp_path, monkeypatch):
    # opd-r1b: the 2-rank optimizer state resumed on 4 cards
    t = folder(tmp_path, "t", [(f"s-{i}.pt", 900, 100) for i in range(16)])
    r = tmp_path / "ckpt"
    r.mkdir()
    for i in range(2):
        (r / f"optim-rank{i}.pt").write_bytes(b"")
    code, got = plan(tmp_path, monkeypatch, t, "--total-steps", "2", "--resume", str(r), world=4)
    assert code == 1 and any("optimizer ranks [0, 1], launch world 4" in p for p in got["problems"])
    code, got = plan(tmp_path, monkeypatch, t, "--total-steps", "2", "--resume", str(r), world=2)
    assert not got["problems"]


def test_sampled_file_without_tok_logp_refused(tmp_path, monkeypatch):
    t = folder(tmp_path, "t", [("s-0.pt", 900, 100, {"tok_logp": False}), ("b-0.pt", 900, 100, {"tok_logp": False})])
    monkeypatch.setenv("HEAL_SAMPLED", "1.0")
    code, got = plan(tmp_path, monkeypatch, t, "--total-steps", "1")
    assert code == 1 and [p for p in got["problems"] if "tok_logp" in p and "1 files (s-0.pt)" in p]
    assert not [p for p in got["problems"] if "b-0.pt" in p]  # teacher-sampled files keep the top-k lookup


def test_schedule_and_checkpoint_steps_checked(tmp_path, monkeypatch):
    t = folder(tmp_path, "t", [(f"s-{i}.pt", 900, 100) for i in range(4)])
    code, got = plan(tmp_path, monkeypatch, t, "--total-steps", "9", "--checkpoint-at", "12")
    assert code == 1
    assert any("schedule would not complete" in p for p in got["problems"])
    assert any("--checkpoint-at [12]" in p for p in got["problems"])


def test_loop_share_must_match_the_cut(tmp_path, monkeypatch):
    t = folder(tmp_path, "t", [("s-0.pt", 900, 100, {"loop": True}), ("s-1.pt", 900, 100)])
    code, got = plan(tmp_path, monkeypatch, t, "--total-steps", "0", "--truncate-loops")
    assert got["loop_removed"] > 0 and any("--loop-share 0.0" in p for p in got["problems"])


def test_read_without_replies_refused(tmp_path, monkeypatch):
    t = folder(tmp_path, "t", [(f"s-{i}.pt", 900, 100) for i in range(4)])
    h = folder(tmp_path, "h", [("h-k000--e1.pt", 500, None), ("h-k001--e1.pt", 500, 300)])
    monkeypatch.setenv("HEAL_SHARD_READS", "1")
    code, got = plan(tmp_path, monkeypatch, t, "--total-steps", "2", "--heldout", str(h), "--heldout-files", "2")
    assert code == 1 and got["reads"]["heldout"]["no_reply"] == ["h-k000--e1.pt"]
    assert got["reads"]["heldout"]["per_rank"] == [1, 1]
