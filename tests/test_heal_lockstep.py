"""MFU-1 levers vs rank-local control flow (c7f desync, 24 Oct): with asymmetric rank inputs (a padding-only rank, an
inert prefix, a rank live from row 1, a one-row live tail), every rank of dp_sequence must issue the same sequence of
FSDP-gathering calls (loss / MTP / extend forwards, backwards), whatever HEAL_SKIP_INERT_LOSS / HEAL_CAPTURE_CACHE."""
import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import sys as _sys  # trunk 24 Oct: PTH's trainer copies (scripts/posttrain/pth) shadow RL's for this module
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "pth"))
for _m in ("rl1_trainer", "flash_qlora_job", "rl1_sparse"):
    _sys.modules.pop(_m, None)

W, CHUNK = 64, 16
SCENARIOS = {  # per rank: None (padding only) or (total, prefix)
    "mixed": [(300, 129), None, (100, 1)],
    "all_inert": [(300, 129), None, (200, 100)],
    "tail_row": [(200, 199), (130, 70), None],  # rank 0's only live row is its last: a one-row loss window
}


def _rank(rank, world, port, scenario, env, out):
    import torch.distributed as dist
    import transformers

    import heal_kd

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), **env)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    seq, w = [], torch.ones(1, requires_grad=True)

    class Cache:
        def __init__(self, *a, **k):
            pass

    transformers.DynamicCache = Cache
    heal_kd.commit_captured = lambda runner: Cache()

    class Runner:
        training, mtp = True, None
        config = type("C", (), {"text_config": None})()

        def eval(self):
            self.training = False

        def train(self, was=True):
            self.training = was

        def __call__(self, mode, ids, a, b, cache, held=None, piece=None, mtp_cache=None, capture=False, lazy=False):
            seq.append(mode)
            if heal_kd.mtp_inputs(ids, a, b) is not None:  # the real Runner's MTP head forward (an FSDP unit)
                seq.append("mtp")
            if mode == "extend":
                return cache
            loss = (w * float(b - a)).sum()
            loss.register_hook(lambda g: seq.append("bwd"))
            return loss, 0.0, 0.0, 0.0, 0.0, 0.0

    def data(total):
        ids = torch.zeros(1, total, dtype=torch.long)
        top = (torch.zeros(total, 2), torch.zeros(total, 2, dtype=torch.int32), torch.zeros(total))
        return {"ids": ids, "teacher": top, "reply": torch.zeros(total), "mask": torch.zeros(total)}

    spec = SCENARIOS[scenario][rank]
    mine = None
    if spec is not None:
        d = data(spec[0])
        mine = {"data": d, "mask": d["mask"], "piece": None, "prefix": spec[1], "general": False}
    dummy = {**data(16), "piece": None}
    heal_kd.dp_sequence(Runner(), mine, dummy, window=W, chunk=CHUNK, scale=1.0, mtp_weight=0.0)
    every = [None] * world
    dist.all_gather_object(every, seq)
    if rank == 0:
        Path(out).write_text(json.dumps(every))
    dist.destroy_process_group()


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("skip,capture", [("0", "0"), ("1", "0"), ("1", "1")])
def test_every_rank_issues_the_same_calls(tmp_path, scenario, skip, capture):
    import torch.multiprocessing as mp

    env = {"HEAL_INERT_PREFIX": "1", "HEAL_SKIP_INERT_LOSS": skip, "HEAL_CAPTURE_CACHE": capture}
    out = tmp_path / "seq.json"
    port = 29620 + sorted(SCENARIOS).index(scenario) * 4 + int(skip) * 2 + int(capture)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank, args=(r, 3, port, scenario, env, str(out))) for r in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
        assert p.exitcode == 0
    seqs = json.loads(out.read_text())
    assert seqs[0] == seqs[1] == seqs[2], [len(s) for s in seqs]
    if skip == "1" and scenario == "all_inert":  # the shared inert window 0 lost its loss call
        assert seqs[0][0] != "loss"
