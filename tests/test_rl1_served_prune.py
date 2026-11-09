"""v17 a3 U20 (30 Oct): the R3 store is cached per process for the run. Across updates with a served_prune between
them, every later window must still find its own record (served_share 1.0) and a pruned donor must never raise."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))

from arena import served_route as sr  # noqa: E402

sys.modules.setdefault("served_route", sr)  # the bundle copies it beside rl1_encode
import rl1_encode  # noqa: E402

SHARED = np.arange(1000, 1064)  # the system prompt every try starts with


def write(d, name, start, tokens, meta):
    t = np.asarray(tokens, dtype=np.int32)
    ids = np.full((len(t), 2, 1), 7, dtype=np.int16)
    (d / f"{name}.rt").write_bytes(np.array([start, len(t), 2, 1], dtype=np.int32).tobytes() + t.tobytes() + ids.tobytes())
    (d / f"{name}.mm.json").write_text(json.dumps(meta))


def play(d, update, tries=4):
    """Each try: one from-scratch end request (its window) and a mid-context request that is a donor for others."""
    out = []
    for i in range(tries):
        target = np.concatenate([SHARED, np.arange(5000 + 100 * (update * tries + i), 5000 + 100 * (update * tries + i) + 40)])
        rid = f"chatcmpl-u{update}t{i}"
        write(d, f"{rid}-abcdefgh", 0, target, {"lora": "p"})
        h = sr.prefix_hashes(target)
        write(d, f"chatcmpl-u{update}d{i}-abcdefgh", 32, target[32:70], {"lora": "p", "prefix_hash": format(int(h[32]), "x")})
        out.append((rid, target))
    return out


def test_cached_store_after_prune_keeps_full_coverage(tmp_path):
    stores = {}
    for update in range(4):
        tries = play(tmp_path, update)
        found = 0
        for rid, target in tries:
            spec = {"served_route": str(tmp_path)}
            rows = [{"n": 0, "rid": rid, "keep": 0, "messages": [{}]}]
            made = {"replies": [{"request": 0}]}
            table, audit, _ = rl1_encode.served_table(spec, rows, made, target, stores)
            found += table is not None and audit.get("unresolved") == 0
        assert found == len(tries), (update, found)  # served_share 1.0 for requests written after the store was built
        for f in list(tmp_path.glob(f"chatcmpl-u{update}*")):  # served_prune: the trained tries' records go
            f.unlink()
    assert len(stores) == 1  # one cached store all along
