"""Parallel boot: the redesign's servers on cards 1..N-1 start beside server 1's boot, not after
it is ready (real-v9: server 2 booted 305 s after server 1's 410 s).

``python -m arena.server_start STUDY SERVER_DIR`` (modal_rl1's boot thread, once server 1 is running and its compile
caches are complete) spawns each card's server with ``rl_loop.SecondServer.start``, the call ``SecondServer.boot``
makes itself; boot then waits on that live process instead of spawning another."""
import json
import sys
from pathlib import Path


def servers(study, server_dir):
    """The SecondServer objects rl_seq.run builds: same cards, ports, names and per-card env."""
    from .rl_loop import SecondServer
    from .rl_seq import card_env, settings

    cfg = settings((study.get("rl_loop") or {}).get("mg") or {}) or {}
    per_card = card_env(cfg)
    return [SecondServer(server_dir, "parallel", card=i, port=1235 + i, gpu_util=None, name=f"server{i + 1}",
                         **({"env": per_card[i]} if per_card.get(i) else {})) for i in range(1, int(cfg.get("cards") or 1))]


def start(study_path, server_dir):
    rows = []
    for server in servers(json.loads(Path(study_path).read_text()), Path(server_dir)):
        if server.start() is None:  # the loop's boot got there first
            rows.append({"card": int(server.card), "skipped": "claimed"})
            continue
        rows.append({"card": int(server.card), "port": server.port, "pid": server.process.pid})
    return rows


if __name__ == "__main__":
    print(json.dumps({"event": "server_start", "servers": start(sys.argv[1], sys.argv[2])}), flush=True)
