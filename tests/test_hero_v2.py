"""hero-v2: component composition, streams from the served layout, the GPTQ overlay under the prune filter (synthetic
safetensors shards), live progress, the build and the Modal tests."""

import hashlib
import importlib.util
import json
import struct
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arena import hero_v2
from arena import live_progress as lp
from arena.competition import ScorecardOwner
from arena.expert_overlay import MANIFEST, materialise
from arena.variants import EXPERT_OVERLAY_GPTQ_ARC, EXPERT_PRUNE_VARIANTS, apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
READY = {"token_budgets": False, "kernels_phase6": False}  # phase 6 not merged; token budgets: test_arena_token_budget


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hero():
    return apply_variant(CONTROL, hero_v2.BASE)


def pretend_merged(monkeypatch):
    """The token clock and the phase-6 plan as their branches define them (for the composition only)."""
    from arc_infra.dense_kernels_patch import PHASE5
    from arena.clock import ActiveClock
    from arena.scheduler import Scheduler

    monkeypatch.setattr(hero_v2, "_token_clock", lambda: True)
    monkeypatch.setattr(ActiveClock, "TOKEN_PURPOSES", ("gameplay",), raising=False)
    monkeypatch.setattr(Scheduler, "TOKEN_FLOOR", 1920, raising=False)
    monkeypatch.setattr(hero_v2, "_phase6_plan", lambda: ("PHASE6_FUSED", {**PHASE5, "fused_splitk": True}))


def test_defaults_ask_for_every_improvement_and_refuse_what_is_not_merged(monkeypatch):
    monkeypatch.setattr(hero_v2, "_token_clock", lambda: False)
    monkeypatch.setattr(hero_v2, "_phase6_plan", lambda: (None, None))
    with pytest.raises(hero_v2.ComponentUnavailable, match="token_budgets"):
        hero_v2.compose(hero())
    # Phase 6 is off by default (no measured gain, kernels-phase6.md section 6); asked for, it must be merged.
    assert hero_v2.DEFAULT_COMPONENTS["kernels_phase6"] is False
    monkeypatch.setattr(hero_v2, "_token_clock", lambda: True)
    with pytest.raises(hero_v2.ComponentUnavailable, match="kernels_phase6"):
        hero_v2.compose(hero(), {"kernels_phase6": True})
    with pytest.raises(ValueError, match="composes on"):
        hero_v2.compose(CONTROL, READY)


def test_full_composition_records_every_component_and_the_difference_from_the_hero(monkeypatch):
    pretend_merged(monkeypatch)
    base = hero()
    original = json.loads(json.dumps(base))
    study = hero_v2.compose(base)
    assert base == original
    server, record = study["server"], study["hero_v2"]
    assert record["on"] == ["compaction_thinking", "expert_prune", "gptq_arc", "live_progress",
                            "reasoning_history", "serving_tweaks", "stall_dump", "token_budgets", "topk_sampler"]
    assert study["reasoning_history"] == "verbatim"
    assert record["components"]["reasoning_history"] == {"on": True, "mode": "verbatim"}
    assert study["compaction_thinking"] == "prefilled-empty"
    assert record["components"]["compaction_thinking"] == {"on": True, "mode": "prefilled-empty"}
    # The follow-up components (29 September) default to hero-v2's own settings: recorded, off.
    assert record["off"] == sorted(["harness_fixes", "kernels_phase6", "kv8_prefill", "prefix_cache", "backlog_fix", "budget_policy",
                                    "keep_alive", "request_end_tail",
                                    "focus_oversubscription", "focus_slots", "kv_cache", "prefix_match_unit",
                                    "prompt_tokens_details", "window"])
    assert "harness_fixes" not in study and record["components"]["harness_fixes"] == {"on": False}
    assert "smoke" in record["components"]["prefix_cache"]["reason"]
    assert "prefix_cache" not in study["cache_experiment"]
    assert study["budget_clock"] == "tokens" and study["token_budget"] == {
        "unit": "completion_tokens", "purposes": ["gameplay"], "floor_tokens": 1920, "reference_tokens_per_busy_second": 64.0}
    assert server["serving_kernels"] == {**base["server"]["serving_kernels"], "ple_uva": True, "embed_host": True}
    assert server["serving_kernels"]["topk_sampler"] is True and server["serving_kernels"]["moe_backend"] == "marlin"
    assert server["dense_kernels"] == base["server"]["dense_kernels"] and "fused_splitk" not in server["dense_kernels"]
    prune = EXPERT_PRUNE_VARIANTS["fixed-2x-mtp0-wide-kv8-hero-prune12"]
    assert server["expert_prune"] == {"mask": prune["mask"], "sha256": prune["sha256"]}
    assert server["expert_overlay"] == EXPERT_OVERLAY_GPTQ_ARC
    assert study["stall_dump_seconds"] == 90 and study["live_progress"] == hero_v2.LIVE_PROGRESS
    # prune12's served 1,245 blocks + 60 (host embedding): 1,174 at the gate, 23 windows of 51; 25 sequences stay.
    assert record["layout"]["blocks"] == 1305 and record["layout"]["windows_fit"] == 23
    assert study["active_limit"] == 23 and server["max_num_seqs"] == study["generation_slots"] == 25
    assert study["require_focus_windows_fit"] and study["focus_fit_fallback"] == {"min_slots": 14}
    # Everything the hero sets and hero-v2 does not change is the hero's.
    for key in ("windows", "opening_switch_level1_games", "level_budget_policy", "reasoning_effort", "temperature",
                "completion_cap", "games", "kv_margin", "cache_admission"):
        assert study[key] == base[key]
    variant = study["arena_variant"]
    assert variant["name"] == hero_v2.NAME and variant["reference_variant"] == hero_v2.BASE
    assert set(variant["reference_difference"]) == {
        "budget_clock", "token_budget", "server.serving_kernels", "server.expert_prune",
        "server.expert_overlay", "live_progress", "active_limit", "focus_fit_fallback", "compaction_thinking",
        "reasoning_history"}
    assert variant["reference_difference"]["reasoning_history"] == {"before": "normalized", "after": "verbatim"}
    assert variant["reference_difference"]["compaction_thinking"] == {"before": "gameplay", "after": "prefilled-empty"}
    assert variant["base_reference"]["reference_variant"] == "reclaim-fixed-2x"


@pytest.mark.parametrize("components,served,active,sequences", [
    ({"expert_prune": "prune12"}, None, 23, 25),
    ({"expert_prune": "prune12", "serving_tweaks": False}, None, 21, 25),
    ({"expert_prune": "prune25"}, None, 30, 30),        # 1,667 + 60: 30 windows, sequences raised to 30
    ({"expert_prune": "prune25", "serving_tweaks": False}, None, 29, 29),
    ({"expert_prune": None, "stall_dump": False}, None, 16, 25),
    ({"expert_prune": "prune12"}, 1290, 22, 25),       # a measured pool replaces the projection
    ({"expert_prune": "prune25"}, 2000, 32, 32),       # capped at the CUDA-graph capture size
])
def test_streams_are_the_most_64k_windows_the_served_layout_fits(components, served, active, sequences):
    layout = hero_v2.served_layout(hero_v2.normalize({**READY, **components}), served)
    assert (layout["active_limit"], layout["max_num_seqs"]) == (active, sequences)
    assert layout["active_limit"] * hero_v2.WINDOW_BLOCKS <= layout["gate_capacity"]
    assert hero_v2.SEQ_FLOOR <= layout["max_num_seqs"] <= hero_v2.SEQ_CAP


def test_raised_sequences_reach_the_server_and_the_generation_slots(monkeypatch):
    study = hero_v2.compose(hero(), {**READY, "expert_prune": "prune25"})
    assert study["active_limit"] == 30 and study["server"]["max_num_seqs"] == study["generation_slots"] == 30
    assert study["arena_variant"]["reference_difference"]["server.max_num_seqs"] == {"before": 25, "after": 30}
    from arc_infra import serving

    source = (ROOT / "src/arc_infra/serving.py").read_text()
    assert "25 <= generation_slots <= 96" in source and serving.SERVER_ENV["TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE"] == "32"


def test_component_values_are_validated():
    with pytest.raises(ValueError, match="Unknown hero-v2 component"):
        hero_v2.normalize({"pruning": True})
    with pytest.raises(ValueError, match="rides on the expert-prune"):
        hero_v2.normalize({"expert_prune": False})
    with pytest.raises(ValueError, match="expert_prune is false"):
        hero_v2.normalize({"expert_prune": "prune50"})
    with pytest.raises(ValueError, match="30..240"):
        hero_v2.normalize({"stall_dump": 10})
    assert hero_v2.normalize({"expert_prune": False, "stall_dump": False})["expert_prune"] is None
    study = hero_v2.compose(hero(), {**READY, "topk_sampler": False, "gptq_arc": False, "live_progress": False})
    assert study["server"]["serving_kernels"]["topk_sampler"] is False
    assert "expert_overlay" not in study["server"] and "live_progress" not in study
    assert study["hero_v2"]["components"]["gptq_arc"] == {"on": False}


# --- GPTQ overlay x pruning on synthetic shards ------------------------------------------------------------------

def write_safetensors(path, tensors):
    header, offset, blobs = {}, 0, []
    for name, array in tensors.items():
        data = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    raw = json.dumps(header).encode()
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def read_safetensors(path):
    data = Path(path).read_bytes()
    size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + size])
    body = data[8 + size:]
    return {name: np.frombuffer(body[s:e], dtype=np.float32).reshape(spec["shape"])
            for name, spec in header.items() if name != "__metadata__" for s, e in [spec["data_offsets"]]}


def synthetic_checkpoint(root, experts=8, layers=2):
    """Per-layer expert shards named like the served checkpoint's, a router and shared tensors elsewhere."""
    base = root / "base"
    base.mkdir()
    weight_map = {}
    other = {}
    for layer in range(layers):
        shard = f"layer-{layer:05d}-experts-0000-{experts - 1:04d}.safetensors"
        tensors = {}
        for e in range(experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for kind in ("weight", "weight_scale"):
                    name = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}.{kind}"
                    tensors[name] = np.full((2, 3), 100 * layer + e, dtype=np.float32)
        write_safetensors(base / shard, tensors)
        weight_map.update({n: shard for n in tensors})
        other[f"model.language_model.layers.{layer}.mlp.gate.weight"] = np.arange(experts * 4, dtype=np.float32).reshape(
            experts, 4) + layer
        other[f"model.language_model.layers.{layer}.mlp.shared_expert.gate_proj.weight"] = np.ones((2, 2))
    write_safetensors(base / "model-rest.safetensors", other)
    weight_map.update({n: "model-rest.safetensors" for n in other})
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (base / "config.json").write_text("{}")
    return base, weight_map


def synthetic_overlay(root, base, weight_map):
    """Replacement shards for every expert shard: same names, shapes and sizes, values + 0.5 (the 'GPTQ' weights)."""
    overlay = root / "overlay"
    overlay.mkdir()
    shards = {}
    for shard in sorted({s for s in weight_map.values() if s.startswith("layer-")}):
        tensors = {k: v + 0.5 for k, v in read_safetensors(base / shard).items()}
        write_safetensors(overlay / shard, tensors)
        shards[shard] = {"bytes": (overlay / shard).stat().st_size,
                         "sha256": hashlib.sha256((overlay / shard).read_bytes()).hexdigest()}
    (overlay / MANIFEST).write_text(json.dumps({"format": "arc-expert-overlay-v1", "shards": shards, "complete": True}))
    return overlay, hashlib.sha256((overlay / MANIFEST).read_bytes()).hexdigest()


def loader_stream(model_dir):
    """The loader's view: every shard the index names, read through the served model path (the farm)."""
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in sorted(set(index.values())):
        yield from read_safetensors(model_dir / shard).items()


def test_overlay_shards_then_the_prune_filter_serve_the_kept_gptq_experts_renumbered(tmp_path):
    rt = load("arc_expert_prune_hero_v2", ROOT / "src/arc_infra/expert_prune_runtime.py")
    base, weight_map = synthetic_checkpoint(tmp_path)
    overlay, digest = synthetic_overlay(tmp_path, base, weight_map)
    record = materialise(base, overlay, tmp_path / "farm", digest, log=lambda *a, **k: None)
    assert record["shards"] == 2
    pruned = {"0": [1, 5], "1": [0, 7]}
    plan = rt.plan(8, pruned)
    out = dict(rt.filter_weights(loader_stream(tmp_path / "farm"), plan, log=lambda *a, **k: None))
    for layer, drop in ((0, {1, 5}), (1, {0, 7})):
        kept = [e for e in range(8) if e not in drop]
        for rank, e in enumerate(kept):
            value = out[f"model.language_model.layers.{layer}.mlp.experts.{rank}.down_proj.weight"]
            assert np.all(value == 100 * layer + e + 0.5)  # the overlay's tensor of the original expert
        assert f"model.language_model.layers.{layer}.mlp.experts.{len(kept)}.gate_proj.weight" not in out
        router = out[f"model.language_model.layers.{layer}.mlp.gate.weight"]
        assert router.shape == (6, 4) and np.all(router[:, 0] == np.array(kept) * 4 + layer)
    assert np.all(out["model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"] == 1)
    counts = hero_v2.overlay_prune_plan(weight_map, json.loads((overlay / MANIFEST).read_text())["shards"], pruned,
                                        experts=8)
    assert counts == {0: {"kept": 36, "dropped": 12}, 1: {"kept": 36, "dropped": 12}}


def test_overlay_prune_plan_refuses_names_the_filter_would_not_rename():
    weight_map = {"model.language_model.layers.0.mlp.experts.gate_up_proj": "layer-00000-experts.safetensors",
                  "model.language_model.layers.0.mlp.experts.0.down_proj.weight": "layer-00000-experts.safetensors"}
    with pytest.raises(ValueError, match="does not rename"):
        hero_v2.overlay_prune_plan(weight_map, ["layer-00000-experts.safetensors"], {"0": [1]}, experts=4)
    with pytest.raises(ValueError, match="not in the checkpoint index"):
        hero_v2.overlay_prune_plan(weight_map, ["other.safetensors"], {"0": [1]}, experts=4)
    assert hero_v2.overlay_prune_plan(
        {"mtp.layers.0.mlp.experts.1.gate_proj.weight": "s", "model.language_model.layers.0.mlp.experts.3.up_proj.weight": "s"},
        ["s"], {"0": [3]}, experts=4) == {0: {"kept": 0, "dropped": 1}}


PLE_SITE = Path(__import__("os").environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))


def test_ple_uva_gate_accepts_a_pruned_config_through_the_checkpoint_count():
    """hero-v2 smoke v1: ple_uva + pruning built the PLE in the worker after num_experts became 448, and the pinned
    exact NVFP4-PLE FP8 gate (num_experts == 512) fell back to the stock loader. The overlay's gate edit reads
    ``arc_unpruned_<field>`` first (set by the prune overlay's prepare_config)."""
    from arc_infra.serving_kernels_patch import PLE, patched_source

    anchor, replacement, count = PLE["edits"][-1]
    assert count == 1 and "arc_unpruned_" in replacement
    expected = {"num_experts": 512, "hidden_size": 2560}
    line = replacement.strip().splitlines()[-1]

    def rejects(config):
        return any(eval(line, {"config": config, "field": f, "expected": e}) for f, e in expected.items())

    assert not rejects(SimpleNamespace(num_experts=448, arc_unpruned_num_experts=512, hidden_size=2560))
    assert not rejects(SimpleNamespace(num_experts=512, hidden_size=2560))           # unpruned: unchanged
    assert rejects(SimpleNamespace(num_experts=448, hidden_size=2560))                # the v1 failure
    path = PLE_SITE / PLE["relative"]
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == PLE["sha256"]:
        text = patched_source(PLE, path.read_bytes()).decode()
        assert text.count('getattr(config, "arc_unpruned_" + field') == 1


# --- live progress ------------------------------------------------------------------------------------------------

def sample(t, generated, running=20, preempt=0):
    return {"elapsed_seconds": t, "metrics": {"vllm:generation_tokens_total": generated, "vllm:num_requests_running":
                                              running, "vllm:num_requests_waiting": 1,
                                              "vllm:kv_cache_usage_perc": 0.4, "vllm:num_preemptions_total": preempt}}


def test_engine_snapshot_rates_over_the_last_minute_and_the_run():
    samples = [sample(0, 0), sample(60, 30000), sample(120, 36000, preempt=2)]
    snap = lp.engine_snapshot(samples)
    assert snap["tokens_per_second_recent"] == 100.0 and snap["tokens_per_second_mean"] == 300.0
    assert snap["running"] == 20 and snap["preemptions"] == 2 and lp.engine_snapshot([]) is None


class FakeOwner:
    """A research-gateway peek_score is a class attribute, as on ScorecardOwner."""

    calls = 0

    def peek_score(self, card):
        FakeOwner.calls += 1
        envs = [SimpleNamespace(id="sb26-a", score=40.0), SimpleNamespace(id="re86-b", score=10.0),
                SimpleNamespace(id="lf52-c", score=0.0)]
        return SimpleNamespace(environments=envs, total_levels_completed=7)


def test_live_fields_read_the_score_every_score_seconds_and_never_raise():
    now = [0.0]
    games = [SimpleNamespace(game_id=g, levels=lv, compactions=0, compaction_seconds=0, error=None,
                             clock=SimpleNamespace(completion_tokens=lambda t=t: t, budget_tokens=lambda t=t: t))
             for g, lv, t in (("sb26-a", 5, 1000), ("re86-b", 2, 500), ("lf52-c", 0, 250), ("cn04-d", 0, 250))]
    live = lp.LiveProgress({"interval_seconds": 60, "score_seconds": 300}, arc=FakeOwner(), card="c",
                           monitor=SimpleNamespace(samples=[sample(0, 0), sample(90, 9000)]), games=games,
                           clock=lambda: now[0])
    FakeOwner.calls = 0
    fields = live.fields(15.0)
    assert fields["score"]["score"] == 12.5 and fields["score"]["levels"] == 7 and fields["score"]["scored_games"] == 2
    assert fields["levels_by_game"] == {"sb26-a": 5, "re86-b": 2} and fields["completion_tokens"] == 2000
    assert fields["engine"]["tokens_per_second_mean"] == 100.0
    now[0] = 100.0
    live.fields(115.0)
    assert FakeOwner.calls == 1          # cached until score_seconds pass
    now[0] = 301.0
    live.fields(316.0)
    assert FakeOwner.calls == 2
    broken = lp.LiveProgress(lp.settings(True), arc=SimpleNamespace(), card="c", monitor=None, games=games)
    assert broken.fields(1.0)["score"] is None  # no peek and not an offline arcade: no score, no error
    with pytest.raises(ValueError):
        lp.settings({"interval_seconds": 5})


def test_trial_progress_carries_the_live_block_with_real_game_records(tmp_path):
    from arena.clock import ActiveClock, GlobalClock
    from arena.scheduler import GameRecord
    from arena.trial import Trial

    wall = GlobalClock(7200)
    wall.release()
    games = [GameRecord(g, i, ActiveClock(wall)) for i, g in enumerate(("sb26-a", "re86-b", "lf52-c"))]
    games[0].levels, games[0].state = 3, "active"
    games[0].clock.charge(0.0, completion_tokens=1234)
    trial = object.__new__(Trial)
    trial.output, trial.log = tmp_path, lambda *a, **k: None
    trial.live = lp.LiveProgress(lp.settings(True), arc=FakeOwner(), card="c", monitor=SimpleNamespace(samples=[]),
                                 games=games)
    gate = SimpleNamespace(snapshot=lambda: {"reserved": 1, "inflight": 1, "waiting": 0}, lane=None)
    trial.progress(wall, games, SimpleNamespace(phase="steady"), gate)
    written = json.loads((tmp_path / "progress.json").read_text())
    assert written["levels"] == 3 and written["active"] == ["sb26-a"]
    assert written["live"]["levels_by_game"] == {"sb26-a": 3} and written["live"]["completion_tokens"] == 1234
    assert written["live"]["score"]["score"] == pytest.approx(50 / 3)
    trial.live = None
    trial.progress(wall, games, SimpleNamespace(phase="steady"), gate)
    assert "live" not in json.loads((tmp_path / "progress.json").read_text())


def test_scorecard_owner_peeks_only_its_card_on_the_local_research_gateway(monkeypatch):
    from arc_agi import scorecard as sc

    monkeypatch.setattr(sc.EnvironmentScorecard, "from_scorecard", classmethod(lambda cls, card, envs: ("card", card)))
    arcade = SimpleNamespace(scorecard_manager=SimpleNamespace(scorecards={"c1": "SC"}), available_environments=[])
    owner = ScorecardOwner(SimpleNamespace(), Path("/tmp"))
    owner.card = "c1"
    assert owner.peek_score("c1") is None          # no local gateway (the hidden rerun never sets one)
    owner.local_gateway = SimpleNamespace(_arcade=arcade)
    assert owner.peek_score("c1") == ("card", "SC")
    assert owner.peek_score("other") is None


def test_publish_copies_progress_and_the_server_tail_and_prints_one_line(tmp_path):
    trial = tmp_path / "trials/duck-transfer"
    trial.mkdir(parents=True)
    progress = {"event": "progress", "elapsed_seconds": 1800.0, "phase": "steady", "active": ["a", "b"], "levels": 9,
                "live": {"score": {"score": 0.0123, "at_elapsed_seconds": 1790.0}, "games_with_levels": 4,
                         "completion_tokens": 1_200_000, "levels_by_game": {"sb26-a": 5},
                         "engine": {"tokens_per_second_recent": 610.0}}}
    (trial / "progress.json").write_text(json.dumps(progress))
    log = tmp_path / "server.log"
    log.write_text("".join(f"line {i}\n" for i in range(1000)))
    printed = []
    lp.publish(trial / "progress.json", tmp_path / "out", "duck-transfer",
               log=lambda text, **k: printed.append(json.loads(text)), server_log=log)
    live = tmp_path / "out/live"
    assert json.loads((live / "progress.json").read_text())["levels"] == 9
    assert len((live / "progress.jsonl").read_text().splitlines()) == 1
    tail = (live / "server-tail.log").read_text().splitlines()
    assert len(tail) == 300 and tail[-1] == "line 999"
    assert printed[0]["event"] == "live_progress" and printed[0]["score"] == 0.0123 and printed[0]["tok_s_recent"] == 610.0
    stop = threading.Event()
    thread = threading.Thread(target=lp.tail, args=(trial / "progress.json", tmp_path / "out2", "duck-transfer", stop,
                                                    3600), kwargs={"log": lambda *a, **k: None})
    thread.start()
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive() and (tmp_path / "out2/live/progress.json").is_file()  # one last copy at the end


def test_bootstrap_starts_the_tailer_only_for_live_arena_trials():
    source = (ROOT / "scripts/arena/bootstrap.py").read_text()
    assert "start_live_progress(bundle, study, trial, working, args.output)" in source
    namespace = {}
    start = source[source.index("def start_live_progress"):source.index("def trial_lifecycle_lines")]
    exec(compile("from pathlib import Path\n" + start, "bootstrap_part", "exec"), namespace)
    fn = namespace["start_live_progress"]
    assert fn(ROOT, {}, {"name": "x", "kind": "duck-transfer"}, "/tmp/w", "/tmp/o") is None
    assert fn(ROOT, {"live_progress": {"interval_seconds": 60}}, {"name": "x", "kind": "serving-bench"}, "/tmp/w",
              "/tmp/o") is None


# --- build and Modal ------------------------------------------------------------------------------------------------

def builder():
    return load("build_arena_hero_v2", ROOT / "scripts/build_arena.py")


def build_args(**extra):
    return dict(paired_30m=True, arena_variant=hero_v2.BASE, opening_burst_requests=3,
                exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], clone_instances=110, **extra)


def test_build_hero_v2_run_bundle(tmp_path):
    b = builder()
    _, study = b.build("hero-v2-test", output=tmp_path / "bundle", wall_seconds=7200, hero_v2=dict(READY),
                       **build_args())
    written = json.loads((tmp_path / "bundle/configs/comparison.json").read_text())
    assert written == study and study["arena_variant"]["name"] == hero_v2.NAME
    assert study["wall_seconds"] == 7200 and study["competition"]["preview_seconds"] == 7200
    assert study["competition"]["preview_instances"] == 110 and study["trials"][0]["name"] == "duck-transfer"
    env = study["trials"][0]["server_env"]
    assert env == {"ARENA_STALL_DUMP": "90", "ARENA_READY_GUARD": "180,900"}
    assert (tmp_path / "bundle" / study["server"]["expert_prune"]["mask"]).is_file()
    assert study["active_limit"] == 23 and study["live_progress"]
    assert study["compaction_thinking"] == "prefilled-empty"
    assert study["hero_v2"]["components"]["compaction_thinking"] == {"on": True, "mode": "prefilled-empty"}
    with pytest.raises(ValueError, match="contradicts"):
        b.build("hero-v2-flag", output=tmp_path / "flag", wall_seconds=7200, hero_v2=dict(READY),
                compaction_thinking="gameplay", **build_args())
    _, hero_gameplay = b.build("hero-v2-gameplay", output=tmp_path / "gameplay", wall_seconds=7200,
                               hero_v2={**READY, "compaction_thinking": "gameplay"}, **build_args())
    assert "compaction_thinking" not in hero_gameplay  # the hero's mode, written as no key
    assert hero_gameplay["hero_v2"]["components"]["compaction_thinking"] == {"on": False, "mode": "gameplay"}
    modal = load("modal_validation_hero_v2", ROOT / "scripts/modal_validation.py")
    assert modal.validation_timeout("arena", "hero-v2-2h") == 9600
    assert modal.arena_arguments("hero-v2-2h") == ["--trial", "duck-transfer"]
    with pytest.raises(ValueError, match="clone roster"):
        b.build("hero-v2-bad", output=tmp_path / "bad", hero_v2={}, paired_30m=True, arena_variant=hero_v2.BASE)


def test_build_hero_v2_smoke_bundle(tmp_path):
    record = tmp_path / "hero-smoke/trials"
    (record / "hero").mkdir(parents=True)
    rows = {"concurrency": 1, "seconds": 1.0, "rows": [{"index": 0, "tokens": [1, 2], "positions": []}]}
    (record / "hero/kv8-quality.json").write_text(json.dumps({
        "greedy": {"concurrency": 16, "seconds": 1.0, "rows": [{"index": 0, "prompt_ids": [1], "tokens": [2]}]},
        "teacher_forced": rows, "reference": "configs/kernel-quality/reference-ac-bf16.json",
        "summary": {"vs_reference": {"teacher_forced": {"mean_kl": 0.072}}, "self_repeat": {"teacher_forced": {}}}}))
    hero_server = hero()["server"]
    (record / "study.json").write_text(json.dumps({"trials": [
        {"trial": {"name": "hero", "kind": "serving-bench", "server_env": {"ARENA_TORCH_PROFILER": "1"}},
         "report": {"server": hero_server}}]}))
    b = builder()
    output, study = b.build("hero-v2-smoke-test", output=tmp_path / "bundle", wall_seconds=900, hero_v2=dict(READY),
                            hero_v2_smoke=True, hero_v2_reference=str(record / "hero/kv8-quality.json"), **build_args())
    assert [t["name"] for t in study["trials"]] == ["hero-v2-check", "duck-transfer"]
    assert all(t["server_env"] == {"ARENA_STALL_DUMP": "90", "ARENA_READY_GUARD": "180,900"} for t in study["trials"])
    bench = study["throughput_bench"]
    assert bench["points"] == [[64512, 23]] and bench["sanity"] and bench["quality"]["repeat_teacher"]
    reference = json.loads((output / bench["quality"]["reference"]).read_text())
    assert reference["role"] == "gate" and reference["server"] == hero_server and reference["teacher_forced"] == rows
    assert study["opening_max_wall_seconds"] == 450 and study["opening_cap_absolute"] is True
    assert study["competition"]["cleanup_seconds"] == 240 and study["hero_v2"]["smoke"]["soak_seconds"] == 900
    modal = load("modal_validation_hero_v2_smoke", ROOT / "scripts/modal_validation.py")
    assert modal.validation_timeout("arena", "hero-v2-smoke") == 2460 and modal.arena_arguments("hero-v2-smoke") == []
    assert study["kernel_timeout"] == 2460
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (2460 + 60) * rate * 1.15 <= 4.0      # the smoke's ledger bound
    assert (9600 + 60) * rate * 1.15 < 15.3      # the 2 h run's


def test_prefix_cache_component_switches_on_the_three_patches(monkeypatch):
    from arena.variants import HERO_PREFIX_CACHE

    pretend_merged(monkeypatch)
    study = hero_v2.compose(hero(), {"prefix_cache": True})
    assert study["cache_experiment"]["prefix_cache"] == HERO_PREFIX_CACHE
    assert "prefix_cache" in study["hero_v2"]["on"]
    assert study["arena_variant"]["reference_difference"]["cache_experiment.prefix_cache"]["after"] == HERO_PREFIX_CACHE


def test_reasoning_history_component_can_select_the_old_normalization(monkeypatch):
    pretend_merged(monkeypatch)
    study = hero_v2.compose(hero(), {"reasoning_history": "normalized"})
    assert study["reasoning_history"] == "normalized" and "reasoning_history" in study["hero_v2"]["off"]
    with pytest.raises(ValueError, match="reasoning_history"):
        hero_v2.compose(hero(), {"reasoning_history": "raw"})


# --- compaction preflight ------------------------------------------------------------------------------------------

class FakeTemplateServer:
    """/tokenize and /detokenize over a toy chat template, one token per character. ``drop_old_reasoning``: reasoning
    only after the last user message (a template without preserved thinking), which breaks the summary's prefix."""

    def __init__(self, drop_old_reasoning=False, think_prefill=True):
        self.drop_old_reasoning, self.think_prefill = drop_old_reasoning, think_prefill

    def render(self, messages, add_generation_prompt=False, continue_final_message=False, **_):
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        text = ""
        for i, m in enumerate(messages):
            content = m["content"] if isinstance(m["content"], str) else "".join(p["text"] for p in m["content"])
            final = continue_final_message and i == len(messages) - 1
            if m["role"] == "assistant":
                reasoning = m.get("reasoning", "") if (not self.drop_old_reasoning or i > last_user) else ""
                think = f"<think>\n{reasoning}\n</think>\n\n" if not final or self.think_prefill else ""
                text += "<|im_start|>assistant\n" + think + content + ("" if final else "<|im_end|>\n")
            else:
                text += f"<|im_start|>{m['role']}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n<think>\n"
        return text

    async def post(self, url, json):
        if url.endswith("/tokenize"):
            tokens = [ord(c) for c in self.render(**{k: v for k, v in json.items() if k != "model"})]
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"tokens": tokens})
        text = "".join(chr(t) for t in json["tokens"])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"prompt": text})


async def test_compaction_preflight_accepts_an_append_only_template_and_rejects_the_divergent_ones():
    from arena.compaction_preflight import run_preflight

    args = dict(model="Qwen/Qwen3.8-Flash-Next-NVFP4", effort="xhigh")
    good = await run_preflight(FakeTemplateServer(), "http://127.0.0.1:1234/v1", **args)
    assert good["ok"] and good["mode"] == "prefilled-empty" and set(good["shapes"]) == {"tool_tail", "user_tail"}
    assert all(row["empty_think_then_prefill"] for row in good["shapes"].values())
    dropped = await run_preflight(FakeTemplateServer(drop_old_reasoning=True), "http://x/v1", **args)
    assert dropped["mode"] == "gameplay" and "tool_tail: summary_shares_conversation" in dropped["reason"]
    no_think = await run_preflight(FakeTemplateServer(think_prefill=False), "http://x/v1", **args)
    assert no_think["mode"] == "gameplay" and "empty_think_then_prefill" in no_think["reason"]

    class Broken:
        async def post(self, url, json):
            raise RuntimeError("no tokenize endpoint")

    broken = await run_preflight(Broken(), "http://x/v1", **args)
    assert broken["mode"] == "gameplay" and "no tokenize endpoint" in broken["reason"]


class FlashTemplateServer(FakeTemplateServer):
    """The served Flash-Next template (tests/fixtures/qwen38-flash-chat-template.jinja) rendered as vLLM does:
    ``reasoning`` handed over as ``reasoning_content``, string tool arguments parsed, transformers 5.x
    ``continue_final_message`` (cut at the final message's end tag)."""

    TAG = "CONTINUE_FINAL_MESSAGE_TAG "

    def render(self, messages, add_generation_prompt=False, continue_final_message=False, tools=None,
               chat_template_kwargs=None, **_):
        import jinja2

        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
        env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
        rows = []
        for m in messages:
            m = dict(m)
            if "reasoning" in m:
                m["reasoning_content"] = m.pop("reasoning")
            if m.get("tool_calls"):
                m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": json.loads(
                    c["function"]["arguments"])}} for c in m["tool_calls"]]
            rows.append(m)
        if continue_final_message:
            rows[-1] = {**rows[-1], "content": rows[-1]["content"] + self.TAG}
        text = env.from_string((ROOT / "tests/fixtures/qwen38-flash-chat-template.jinja").read_text()).render(
            messages=rows, tools=tools, add_generation_prompt=add_generation_prompt, **(chat_template_kwargs or {}))
        if continue_final_message:
            at = text.rindex(self.TAG.strip())
            text = text[:at] if text[at:at + len(self.TAG)] == self.TAG else text[:at].rstrip()
        return text


async def test_compaction_preflight_passes_on_the_served_flash_template():
    pytest.importorskip("jinja2")
    from arena.compaction_preflight import run_preflight

    result = await run_preflight(FlashTemplateServer(), "http://x/v1", model="Qwen/Qwen3.8-Flash-Next-NVFP4",
                                 effort="xhigh")
    assert result["ok"], result["reason"]
    for row in result["shapes"].values():
        assert row["assistant_turn"].startswith("<think>\n\n</think>\n\nWorld model:")
