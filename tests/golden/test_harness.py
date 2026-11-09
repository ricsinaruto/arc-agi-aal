"""Golden (a/b): the agent harness's requests (messages, tools, template, sampling) stay byte-identical through rl-min.
Fresh level-bench tries of the hero study on two games against the replay stub (recorded ls20 replies, instant),
one at a time so the stub's reply order is fixed; ls20 runs long enough for a context drain. Recorded from base 38a45c57d."""
import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest
from common import B, ROOT, check, config, fixture, source_dir

GAMES = {"ls20": 100000, "wa30": 8000}
KEYS = ("messages", "tools", "template", "sampling", "keep", "purpose", "gameplay")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# rl-min C (28 Oct): the main run's study (rl_seq real-v8; recorded on base 1477cee88 with its original config)
RUNS = {"rl1-loop-seq-real-v8": "harness-real-v8"}


@pytest.mark.parametrize("run", sorted(RUNS))
def test_harness_requests(tmp_path, run):
    pytest.importorskip("arc_agi")
    corpus, games = fixture("g00172-ls20-L2"), fixture("harness-games")
    import arena
    from arena.level_bench import run_level_bench

    cfg = config(run)
    for k, v in cfg["files"].items():  # duck_controls.yaml and the notebook config beside the seeds
        if k.startswith("configs/"):
            (tmp_path / k).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / k).write_bytes((ROOT / v).read_bytes())
    source = source_dir(cfg, tmp_path)
    port = free_port()
    src = os.path.dirname(os.path.dirname(arena.__file__))  # the tree under test (PYTHONPATH), base or this one
    stub = subprocess.Popen([sys.executable, "-m", "arena.replay_stub", "serve", "--working", str(tmp_path / "stub"),
                             "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONPATH": src,
                                 "ARC_PREFLIGHT_STUB": json.dumps({"corpus": [str(corpus)], "speed": 0})})
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        study = B.inline(cfg["study"])
        study["served_model"], study["wall_seconds"] = "policy", 900
        study["level_bench"] = {**study["level_bench"], "games": list(GAMES), "passes": 1, "concurrency": 1,
                                "capacity_tokens": 4_000_000, "seed_base": 7, "attempt_wall_seconds": 600,
                                "min_wall_seconds": 10,
                                "token_budget_by_point": {f"{g}-game": b for g, b in GAMES.items()}}
        out = tmp_path / "lb"
        report = asyncio.run(run_level_bench(study, {"name": "hg", "kind": "level-bench"}, source=source,
                                             games_dir=str(games), output=out, server_dir=str(tmp_path / "stub"),
                                             analysis_python=sys.executable, base_url=f"http://127.0.0.1:{port}/v1",
                                             deadline_epoch=time.time() + 1200))
    finally:
        stub.kill()
    got = {"outcomes": [[a["label"], a["outcome"], a["completion_tokens"], a["gameplay_requests"], a.get("compactions"),
                         a.get("context_drain")] for a in report["attempts"]]}
    for att in sorted((out / "attempts").iterdir()):
        rows = [json.loads(line) for line in open(att / "rollout.jsonl")]
        got[att.name] = [hashlib.sha1(json.dumps([r.get(k) for k in KEYS], sort_keys=True).encode()).hexdigest()[:16]
                         for r in rows]
    check(RUNS[run], got)
