"""Single-card start-up cuts (MFU-2 23 Oct, ovf2's 19.4 min from launch to round-1 play): mg.r1_first (round 1 plays
before the trainer exists; the trainer path runs beside the round-end sleep), mg.base_cache (the probes' base
references stored per key and checked), mg.park_prepin (the park's pinned buffers on a thread)."""
import asyncio

import pytest

from arena import level_bench
from test_rl_startup import Server, Trainer, first, make_loop, trainer_ops


class CardServer(Server):
    async def sleep(self):
        self.log.append(("server.sleep",))
        return 0.1

    async def wake(self):
        self.log.append(("server.wake",))
        return 0.1


def fresh(tmp_path, monkeypatch, log, mg=None, **kw):
    loop = make_loop(tmp_path, monkeypatch, log, mg={"resume": None, "single_card": True, "max_rounds": 3, **(mg or {})},
                     **kw)
    loop.server = CardServer(log)
    loop.mg_init = False
    return loop


def steps(log):
    keep = ("trainer_start", "trainer_ready", "trainer", "server.sleep", "server.wake", "play_start", "play_end")
    return [e for e in log if e[0] in keep]


def test_keys_off_fresh_single_card_loads_the_trainer_before_round_1(tmp_path, monkeypatch):
    log = []
    loop = fresh(tmp_path, monkeypatch, log)
    assert asyncio.run(loop.mg_loop()) == "max_rounds"
    assert first(log, ("server.sleep",)) < first(log, ("trainer_start",)) < first(log, ("play_start", "r01-tr"))
    assert "r1_first" not in loop.feed.kinds()


def test_r1_first_plays_round_1_before_the_trainer_and_loads_it_at_the_round_end(tmp_path, monkeypatch):
    log = []
    loop = fresh(tmp_path, monkeypatch, log, mg={"r1_first": True})
    assert asyncio.run(loop.mg_loop()) == "max_rounds"
    s = steps(log)
    assert first(s, ("play_start", "r01-tr")) < first(s, ("trainer_start",))
    assert first(s, ("play_end", "r01-tr")) < first(s, ("server.sleep",)) < first(s, ("trainer_start",))
    assert log.count(("trainer_start",)) == 1
    # the trainer path of start() ran once (base references, r00), then park + wake before round 2 plays
    ops = trainer_ops(log)
    assert ops.count("score") >= 1 and "checkpoint" in ops and "park" in ops
    assert first(log, ("trainer", "park")) < first(log, ("server.wake",)) < first(log, ("play_start", "r02-tr"))
    kinds = loop.feed.kinds()
    assert kinds.index("r1_first") < kinds.index("r1_first_trainer") < kinds.index("round")


def test_r1_first_is_off_for_a_resume_a_seeded_u1_or_two_cards(tmp_path, monkeypatch):
    log = []
    loop = make_loop(tmp_path, monkeypatch, log, mg={"r1_first": True, "single_card": True})
    loop.server = CardServer(log)
    asyncio.run(loop.mg_loop())
    assert "r1_first" not in loop.feed.kinds()
    log2 = []
    loop2 = fresh(tmp_path / "b", monkeypatch, log2, mg={"r1_first": True, "single_card": False})
    asyncio.run(loop2.mg_loop())
    assert "r1_first" not in loop2.feed.kinds()  # two cards: the trainer loads beside round 1 as today


def test_park_prepin_starts_after_the_trainer_loads(tmp_path, monkeypatch):
    log = []
    loop = fresh(tmp_path, monkeypatch, log, mg={"park_prepin": True})
    asyncio.run(loop.mg_loop())
    ops = trainer_ops(log)
    assert ops[0] == "prepin" and ops.index("prepin") < ops.index("park")
    assert "park_prepin" in loop.feed.kinds()


class CacheTrainer(Trainer):
    """score/base_export/base_import as the real ops' contract; ``check_kl`` is the cache check's KL to base."""

    def __init__(self, log, check_kl=0.0):
        super().__init__(log)
        self.check_kl, self.stored = check_kl, {}

    async def call(self, op, timeout=3600, **payload):
        self.ops.append((op, payload))
        self.log.append(("trainer", op, payload.get("label")))
        if op == "base_export":
            from pathlib import Path
            Path(payload["path"]).parent.mkdir(parents=True, exist_ok=True)
            Path(payload["path"]).write_text(payload["key"])
            return {"windows": len(payload["names"])}
        if op == "base_drop":
            return {"dropped": len(payload["names"])}
        if op == "base_import":
            from pathlib import Path
            if getattr(self, "import_error", None):
                raise self.import_error
            if Path(payload["path"]).read_text() != payload["key"]:
                raise ValueError("key")
            return {"windows": len(payload["names"]), "names": payload["names"]}
        if op == "score":
            kl = self.check_kl if payload.get("label") == "cache_check" else 0.0
            return {"windows": {w["name"]: {"kl_base": kl, "logratio_base": getattr(self, "check_lr", 0.0)}
                                for w in payload["windows"]}}
        return {"ok": True}


def cache_loop(tmp_path, monkeypatch, log, check_kl=0.0, **mg):
    loop = fresh(tmp_path, monkeypatch, log, mg={"base_cache": str(tmp_path / "cache"), "kl_floor": 0.02, **mg})
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text("{}")
    loop.cfg["model_dir"] = str(model)
    loop.trainer = CacheTrainer(log, check_kl)
    monkeypatch.setenv("ARC_RL1_TRAINER_KEY", "tk1")
    probes = []
    for i, size in enumerate((300, 100)):
        f = tmp_path / "probes" / f"p{i}" / "rollout.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x" * size)
        probes.append({"rollout": str(f), "w": 1.0, "name": f"heldout-p{i}", "label": f"p{i}"})
    return loop, probes


def labels(log):
    return [(e[1], e[2]) for e in log if e[0] == "trainer"]


def test_base_cache_miss_scores_and_stores_then_a_hit_loads_and_checks_the_shortest_probe(tmp_path, monkeypatch):
    log = []
    loop, probes = cache_loop(tmp_path, monkeypatch, log)
    row = asyncio.run(loop.mg_base_reference(probes))
    assert row["hit"] is False and labels(log) == [("score", "base"), ("base_export", None)]
    log.clear()
    row = asyncio.run(loop.mg_base_reference(probes))
    assert row["hit"] is True and labels(log) == [("base_import", None), ("score", "cache_check")]
    assert row["limit"] == 2e-4
    checked = loop.trainer.ops[-1][1]
    assert [w["name"] for w in checked["windows"]] == ["heldout-p1"] and checked["hold_previous"] is True


@pytest.mark.parametrize("kl,lr", [(1e-3, 0.0), (0.0, -1e-3)])
def test_base_cache_failed_check_rescores(tmp_path, monkeypatch, kl, lr):
    log = []
    loop, probes = cache_loop(tmp_path, monkeypatch, log)
    asyncio.run(loop.mg_base_reference(probes))
    loop.trainer.check_kl, loop.trainer.check_lr = kl, lr  # a 1e-3 shift: above the reload-check bar 2e-4
    log.clear()
    row = asyncio.run(loop.mg_base_reference(probes))
    assert row["hit"] is False and ("score", "base") in labels(log)
    assert any(k == "alert" for k, _ in loop.feed.events)
    seq = labels(log)  # the imported copy is dropped before the fresh base score (v6j2)
    assert seq.index(("base_drop", None)) < seq.index(("score", "base"))


def test_base_cache_refused_import_drops_before_scoring_afresh(tmp_path, monkeypatch):
    log = []
    loop, probes = cache_loop(tmp_path, monkeypatch, log)
    asyncio.run(loop.mg_base_reference(probes))
    loop.trainer.import_error = RuntimeError("Expected all tensors to be on the same device")
    log.clear()
    row = asyncio.run(loop.mg_base_reference(probes))
    assert row["hit"] is False
    assert labels(log)[:3] == [("base_import", None), ("base_drop", None), ("score", "base")]


@pytest.mark.parametrize("change", ["probe", "trainer_fast", "trainer_key", "model", "path_chunk"])
def test_base_cache_key_moves_with_every_input(tmp_path, monkeypatch, change):
    log = []
    loop, probes = cache_loop(tmp_path, monkeypatch, log)
    key = loop.mg_base_cache_key(probes)
    if change == "probe":
        (tmp_path / "probes" / "p0" / "rollout.jsonl").write_text("y" * 300)
    elif change == "trainer_fast":
        loop.cfg["mg"]["trainer_fast"] = {"prefetch": 2}
    elif change == "trainer_key":
        monkeypatch.setenv("ARC_RL1_TRAINER_KEY", "tk2")
    elif change == "model":
        (tmp_path / "model" / "config.json").write_text('{"a": 1}')
    else:
        loop.cfg["path_chunk"] = 12345
    assert loop.mg_base_cache_key(probes) not in (key, None)


def test_base_cache_off_without_a_trainer_key_or_for_a_warm_start(tmp_path, monkeypatch):
    log = []
    loop, probes = cache_loop(tmp_path, monkeypatch, log)
    monkeypatch.delenv("ARC_RL1_TRAINER_KEY")
    assert loop.mg_base_cache_key(probes) is None
    monkeypatch.setenv("ARC_RL1_TRAINER_KEY", "tk1")
    loop.mg_init = True
    assert loop.mg_base_cache_key(probes) is None
    row = asyncio.run(loop.mg_base_reference(probes))
    assert row["hit"] is False and labels(log) == [("score", "base")]


def test_new_start_keys_are_known_and_default_off():
    from arena import rl_loop
    assert {"r1_first", "base_cache", "park_prepin"} <= rl_loop.MG_KEYS
    mg = rl_loop.settings({"mode": "loop"}).get("mg") or {}
    assert not any(mg.get(k) for k in ("r1_first", "base_cache", "park_prepin"))


def test_host_load_spawns_the_trainer_at_round_1_and_opens_its_gate_at_the_round_end(tmp_path, monkeypatch):
    log = []
    loop = fresh(tmp_path, monkeypatch, log, mg={"r1_first": True, "host_load": True, "host_load_threads": 3})
    monkeypatch.setenv("ARC_RL1_TRAINER_KEY", "tk1")
    monkeypatch.setenv("ARC_PARK_RECORDS", str(tmp_path / "records"))
    spawned = {}

    def spawn_trainer(gate=None, extra_env=None):
        log.append(("spawn",))
        spawned.update(gate=gate, env=extra_env)
        return "early"

    async def start_trainer(gate=None, extra_env=None, spawned=None):
        log.append(("start_trainer", spawned, gate.is_file() if (gate := spawned and loop.work / "trainer-gpu-gate-r1") else None))
        loop.trainer = Trainer(log)
        return {"seconds": 0.1, "loaded": {"host_load": {"on": True, "fallback": True, "error": "x"}}}

    loop.spawn_trainer, loop.start_trainer = spawn_trainer, start_trainer
    assert asyncio.run(loop.mg_loop()) == "max_rounds"
    assert first(log, ("spawn",)) < first(log, ("play_start", "r01-tr")) < first(log, ("server.sleep",))
    assert ("start_trainer", "early", True) in log  # adopted after the sleep, gate written
    env = spawned["env"]
    assert env["RL1_HOST_LOAD"] == "1" and env["RL1_HOST_LOAD_THREADS"] == "3"
    assert env["RL1_LAYOUT_RECORD"] == str(tmp_path / "records" / "layout-tk1.json")
    assert not spawned["gate"].exists() or first(log, ("server.sleep",)) < first(log, ("start_trainer", "early", True))
    assert any(k == "alert" and "host load" in d.get("why", "") for k, d in loop.feed.events)


def test_host_load_needs_a_trainer_key(tmp_path, monkeypatch):
    log = []
    loop = fresh(tmp_path, monkeypatch, log, mg={"r1_first": True, "host_load": True})
    monkeypatch.delenv("ARC_RL1_TRAINER_KEY", raising=False)
    loop.spawn_trainer = lambda **k: log.append(("spawn",))
    asyncio.run(loop.mg_loop())
    assert ("spawn",) not in log


def test_base_cache_true_resolves_to_the_volume_root_not_a_relative_dir(tmp_path, monkeypatch):
    log = []
    loop, probes = cache_loop(tmp_path, monkeypatch, log)
    loop.cfg["mg"]["base_cache"] = True
    monkeypatch.setenv("ARC_BASE_CACHE_ROOT", str(tmp_path / "vol"))
    monkeypatch.chdir(tmp_path)
    row = asyncio.run(loop.mg_base_reference(probes))
    assert row["hit"] is False and not (tmp_path / "True").exists()
    assert (tmp_path / "vol" / f"{row['key']}.pt").is_file()
    log.clear()
    assert asyncio.run(loop.mg_base_reference(probes))["hit"] is True


def test_fake_trainer_base_drop_answers(tmp_path):
    from arena import fake_trainer

    fake = fake_trainer.Fake(16384, folder=tmp_path / "f")  # a4cc9b2a5's twin read an undefined name
    assert fake.run("base_drop", {"names": ["p0", "p1"]}) == {"dropped": 2}
