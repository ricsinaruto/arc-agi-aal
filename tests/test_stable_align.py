"""stable_align (MFU-2 24 Oct): the deterministic re-sort against an emulated atomic align, and the hash-guarded edit."""

import os
import shutil
from pathlib import Path

import pytest

from arc_infra import stable_align as sa

torch = pytest.importorskip("torch")
PINNED = Path(os.environ.get("ARC_PINNED_VLLM", "./.claude/worktrees/throughput-arm/keep/vllm-full"))


def stable():
    ns = {"torch": torch}
    exec(sa.SOURCE, ns)
    return ns["_arc_stable_align"]


def atomic_align(topk, num_experts, block, gen, garbage=-7):
    """vLLM's layout with a random within-segment order (what the atomics give): segments in expert order, each
    padded to ``block`` with ``numel``; blocks after the end hold ``garbage``."""
    flat = topk.flatten()
    numel = flat.numel()
    n = numel + num_experts * (block - 1)
    blocks = -(-n // block)
    ids = torch.full((n,), garbage, dtype=torch.int32)
    eids = torch.full((blocks,), garbage, dtype=torch.int32)
    at = 0
    for e in range(num_experts):
        mine = (flat == e).nonzero().flatten()
        if not mine.numel():
            continue
        size = -(-mine.numel() // block) * block
        seg = torch.full((size,), numel, dtype=torch.int32)
        seg[:mine.numel()] = mine[torch.randperm(mine.numel(), generator=gen)].to(torch.int32)
        ids[at:at + size] = seg
        eids[at // block:(at + size) // block] = e
        at += size
    return ids, eids, torch.tensor([at], dtype=torch.int32)


@pytest.mark.parametrize("tokens,k,experts,block", [(37, 8, 64, 16), (300, 8, 256, 16), (5, 2, 4, 4), (64, 8, 16, 64)])
def test_identical_after_any_atomic_order_and_layout_kept(tokens, k, experts, block):
    fix = stable()
    gen = torch.Generator().manual_seed(tokens)
    topk = torch.stack([torch.randperm(experts, generator=gen)[:k] for _ in range(tokens)]).to(torch.int32)
    outs = []
    for seed in range(4):
        ids, eids, ntpp = atomic_align(topk, experts, block, torch.Generator().manual_seed(seed))
        before = ids.clone()
        fix(ids, eids, ntpp, block, topk.numel())
        end = int(ntpp)
        assert torch.equal(ids[end:], before[end:])  # past the end: untouched
        for b in range(end // block):  # same segments, each now ascending (padding last)
            e = int(eids[b])
            seg = (eids[:end // block] == e).nonzero().flatten()
            lo, hi = int(seg[0]) * block, (int(seg[-1]) + 1) * block
            assert torch.equal(ids[lo:hi], torch.sort(before[lo:hi]).values)
        outs.append(ids)
    assert all(torch.equal(outs[0], o) for o in outs[1:])
    flat = topk.flatten()
    want = torch.cat([(flat == e).nonzero().flatten() for e in range(experts)])
    got = outs[0][:int(ntpp)]
    assert torch.equal(got[got < topk.numel()].long(), want)  # = index order within each expert


def test_lora_rows_each_sorted():
    fix = stable()
    gen = torch.Generator().manual_seed(1)
    topk = torch.stack([torch.randperm(16, generator=gen)[:4] for _ in range(20)]).to(torch.int32)
    rows = [atomic_align(topk, 16, 8, torch.Generator().manual_seed(s)) for s in range(3)]
    ids = torch.stack([r[0] for r in rows]).flatten().clone()
    eids = torch.stack([r[1] for r in rows]).flatten().clone()
    ntpp = torch.cat([r[2] for r in rows])
    fix(ids, eids, ntpp, 8, topk.numel())
    ids = ids.view(3, -1)
    assert torch.equal(ids[0], ids[1]) and torch.equal(ids[1], ids[2])
    single = rows[0][0].clone()
    fix(single, rows[0][1], rows[0][2], 8, topk.numel())
    assert torch.equal(single, ids[0])


@pytest.mark.skipif(not PINNED.exists(), reason="no local copy of the pinned vLLM")
def test_patch_restart_and_refusal(tmp_path):
    site = tmp_path / "site"
    (site / "vllm").mkdir(parents=True)
    shutil.copy(PINNED / sa.RELATIVE, site / sa.RELATIVE)
    identity = sa.apply(site, tmp_path / "out")
    text = (site / sa.RELATIVE).read_text()
    assert text.count("if _ARC_STABLE_ALIGN:  # ARC") == 2 and "def _arc_stable_align(" in text
    assert sa.apply(site, tmp_path / "out") == identity
    assert sa.apply(site, tmp_path / "fresh")["patched_sha256"] == identity["patched_sha256"]  # later trial, same VM
    shutil.copy(PINNED / sa.RELATIVE, site / sa.RELATIVE)  # runtime extracted again under the same marker
    assert sa.apply(site, tmp_path / "out")["patched_sha256"] == identity["patched_sha256"]
    (site / sa.RELATIVE).write_text(text + "# changed\n")
    with pytest.raises(RuntimeError):
        sa.apply(site, tmp_path / "out")  # marker says patched, file changed
    with pytest.raises(RuntimeError):
        sa.patched_source(b"not vllm")


def test_qsa_blocks_sorted_padding_last_and_patch():
    ns = {"torch": torch}
    exec(sa.QSA_SOURCE, ns)
    a = torch.tensor([[5, -1, 2, 9], [-1, -1, 3, 0]], dtype=torch.int32)
    b = torch.tensor([[9, 2, -1, 5], [0, -1, 3, -1]], dtype=torch.int32)
    ns["_arc_sort_blocks"](a)
    ns["_arc_sort_blocks"](b)
    assert torch.equal(a, b) and a.tolist() == [[2, 5, 9, -1], [0, 3, -1, -1]]
    if not (PINNED / sa.QSA_RELATIVE).exists():
        pytest.skip("no local copy of the pinned vLLM")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        site = Path(tmp)
        (site / sa.QSA_RELATIVE).parent.mkdir(parents=True)
        shutil.copy(PINNED / sa.QSA_RELATIVE, site / sa.QSA_RELATIVE)
        assert sa.apply_qsa(site)["pinned"]
        assert sa.apply_qsa(site) == {"already": True}
        assert (site / sa.QSA_RELATIVE).read_text().count("            _arc_sort_blocks(blocks)\n") == 1


def test_flags_read_per_process(monkeypatch):
    """ARC_STABLE_PATCH=1 edits the shared tree; each server process turns a fix on by its own env at import."""
    for src, name, env in ((sa.SOURCE, "_ARC_STABLE_ALIGN", sa.ENV), (sa.QSA_SOURCE, "_ARC_STABLE_QSA", sa.QSA_ENV)):
        monkeypatch.delenv(env, raising=False)
        ns = {"torch": torch}
        exec(src, ns)
        assert ns[name] is False
        monkeypatch.setenv(env, "1")
        ns = {"torch": torch}
        exec(src, ns)
        assert ns[name] is True


def test_qsa_ties_lowest_column_first(monkeypatch):
    monkeypatch.setenv("ARC_QSA_TIES", "index")
    ns = {"torch": torch}
    exec(sa.QSA_SOURCE, ns)
    assert ns["_ARC_QSA_TIES"] is True
    logits = torch.tensor([[0.0, 0.5, 0.0, -0.0, float("-inf"), 0.0, 2.0, 1e-30]])
    before = logits.clone()
    ns["_arc_tie_by_index"](logits)
    pos = before > 0
    assert torch.equal(logits[pos], before[pos]) and logits[0, 4] == float("-inf")
    assert (logits[~pos & torch.isfinite(before)] < 0).all()
    # budget 5: the three positives, then zeros by lowest column (0, 2), whatever the kernel's claim order
    top = torch.topk(logits, 5, dim=-1).indices[0].tolist()
    assert sorted(top) == [0, 1, 2, 6, 7]
    assert len(set(logits[0, [0, 2, 3, 5]].tolist())) == 4  # distinct


def test_qsa_edit_places_ties_before_topk(tmp_path):
    src = PINNED / sa.QSA_RELATIVE
    if not src.exists():
        pytest.skip("no pinned qsa.py")
    text = sa.qsa_patched_source(src.read_bytes()).decode()
    i_tie, i_topk, i_sort = text.index("_arc_tie_by_index(logits)\n"), text.index(sa.QSA_ANCHOR), text.index("_arc_sort_blocks(blocks)\n")
    assert i_tie < i_topk < i_sort
