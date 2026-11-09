"""tools/ops/export_rl.carry_target: a resumed leg without target rows emits its lineage's latest, once."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_OPS_SCRATCH", str(tmp_path))
    sys.path.insert(0, str(ROOT / "tools" / "ops"))
    spec = importlib.util.spec_from_file_location("export_rl_t", ROOT / "tools/ops/export_rl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_carry_target_from_the_lineage_then_not_again(tmp_path, monkeypatch):
    ex = load(tmp_path, monkeypatch)
    d = tmp_path / "train-metrics"
    d.mkdir()
    (d / "a.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"t": 1, "m": {"train.rhae_per_10k.L13": 1.0, "infra.running": 3}},
        {"t": 2, "m": {"train.rhae_per_10k.L13": 2.5, "eval.heldout_rhae10k": 7.0}}]) + "\n")
    (d / "b.meta.json").write_text(json.dumps({"continues": [{"run": "a"}]}))
    (d / "c.meta.json").write_text(json.dumps({"continues": [{"run": "b"}]}))
    out = ex.metrics.writer("c")
    got = ex.carry_target("c", out)
    assert got == {"train.rhae_per_10k.L13": 2.5, "eval.heldout_rhae10k": 7.0}
    out.close() if hasattr(out, "close") else None
    rows = ex.metrics.read_rows(d / "c.jsonl")
    assert any((r.get("m") or {}).get("train.rhae_per_10k.L13") == 2.5 for r in rows)
    assert ex.carry_target("c", out) == {}  # own rows now: no second carry


def test_rl_seq_run_exports_sampler_rows_once_each(tmp_path, monkeypatch):
    """An rl_seq run has no RL-watcher pass files: infra rows (util, VRAM, both servers' running/KV/gen tok/s, host
    RAM, cost) come from the runops sampler and cpulog; a second pass adds only newer rows (H4 26 Oct had none)."""
    ex = load(tmp_path, monkeypatch)
    smp = [{"t": 60, "epoch": 1000.0, "util": [90.0, 80.0], "mem_gib": [200.0, 190.0],
            "vllm_servers": {"1234": {"num_requests_running": 100.0, "generation_tokens_total": 0.0, "kv_cache_usage_perc": 0.5},
                             "1236": {"num_requests_running": 110.0, "generation_tokens_total": 0.0, "kv_cache_usage_perc": 0.6}}},
           {"t": 120, "epoch": 1060.0, "util": [95.0, 85.0], "mem_gib": [201.0, 191.0],
            "vllm_servers": {"1234": {"num_requests_running": 100.0, "generation_tokens_total": 60000.0, "kv_cache_usage_perc": 0.5},
                             "1236": {"num_requests_running": 110.0, "generation_tokens_total": 30000.0, "kv_cache_usage_perc": 0.6}}}]
    files = {"validation/r/live/sampler.jsonl": "\n".join(json.dumps(r) for r in smp[:1]),
             "validation/r/util/cpulog.txt": "1000 cpu 1 2 3 mem_gib 120.0 anon_gib 1\n"}
    monkeypatch.setattr(ex, "volume_text", lambda p: files.get(p, ""))
    st = {"rate": [900.0, 0.004]}
    out = ex.metrics.writer("r")
    ex.once("r", tmp_path, st, out)
    files["validation/r/live/sampler.jsonl"] = "\n".join(json.dumps(r) for r in smp)
    ex.once("r", tmp_path, st, out)
    rows = ex.metrics.read_rows(tmp_path / "train-metrics" / "r.jsonl")
    keys = {k for r in rows for k in r["m"]}
    assert {"infra.gpu_util.0", "infra.vram_gib.1", "infra.host_ram_gib", "infra.cost_usd"} <= keys
    assert sum(1 for r in rows if "infra.gpu_util.0" in r["m"]) == 2  # no repeats across passes
    assert any(r["m"].get("infra.gen_tok_s.s1234") == 1000.0 for r in rows)


def test_r3_offload_rows_reach_the_dashboard(tmp_path, monkeypatch):
    """rl_seq's {"kind": "r3_offload", "metrics": {...}} rows map to infra.r3_offload_* keys (H5 watch target)."""
    ex = load(tmp_path, monkeypatch)
    m = ex.feed_metrics({"kind": "r3_offload", "epoch": 1.0,
                         "metrics": {"infra.r3_offload_bound": 1, "infra.r3_offload_refused.partial_tail": 3}}, {})
    assert m["infra.r3_offload_bound"] == 1 and m["infra.r3_offload_refused.partial_tail"] == 3


def test_seq_phase_rows_export_mtp_guard_and_switch(tmp_path, monkeypatch):
    export_rl = load(tmp_path, monkeypatch)
    """dl2 (26 Oct): the stage D bars (MTP acceptance per phase) and the back-off guard reach the dashboard."""
    e = {"kind": "seq_phase", "play_seconds": 360.0, "stop": {"switch_s": 88.3}, "start": {"switch_s": 27.3},
         "mtp": {"accept": 0.61, "pos1": 0.72, "mean_len": 2.2, "gen_tok_s": 1580.0, "drafts": 10},
         "guard": {"kv_peak": 0.34, "preemptions": 0, "polls": 40}}
    m = export_rl.feed_metrics(e, {})
    assert m["infra.draft_acceptance"] == 0.61 and m["infra.draft_pos1"] == 0.72 and m["infra.draft_mean_len"] == 2.2
    assert m["infra.phase_gen_tok_s"] == 1580.0 and m["infra.phase_kv_peak"] == 0.34
    assert m["infra.phase_preemptions"] == 0 and m["infra.switch_stop_s"] == 88.3 and m["play.phase_seconds"] == 360.0
    assert export_rl.feed_metrics({"kind": "seq_phase", "mtp": None}, {}) == {}


def test_rl_seq_feed_gives_train_rhae_per_10k(tmp_path, monkeypatch):
    """rl_seq runs (27 Oct): group_done carries its tries and seq_accumulate names the step's groups, so the step row
    exports train.rhae_per_10k.L13 / .all with no group folder on the volume (D1/D2 exported none)."""
    ex = load(tmp_path, monkeypatch)
    monkeypatch.setenv("ARC_RL_DIST", str(tmp_path / "nodist"))
    tries = [{"level": 1, "completion_tokens": 10000, "cleared": i < 6, "actions": 20, "level_score": 50.0,
              "total_levels": 4, "gameplay_requests": 10} for i in range(12)]
    feed = [{"kind": "group_done", "gid": "p1g1", "set": "sb26-L1", "version": 0, "tries": 12, "clears": 6,
             "attempts": tries},
            {"kind": "seq_accumulate", "one_call": True, "groups": 1, "windows": 12},
            {"kind": "seq_accumulate", "gid": "p1g1", "set": "sb26-L1", "tries": 12, "trained": 12},
            {"kind": "step", "step": 1, "version": 1, "grad_norm": 0.1}]
    feed = [{**e, "t": float(i), "epoch": 1000.0 + i} for i, e in enumerate(feed)]
    monkeypatch.setattr(ex, "volume_text", lambda path: "\n".join(json.dumps(e) for e in feed) if path.endswith(
        "feed.jsonl") else "")
    got = []
    out = type("O", (), {"emit": lambda self, m, t=None, **kw: got.append(m)})()
    passes = tmp_path / "passes"
    passes.mkdir()
    (passes / "r-watch-passes.jsonl").write_text("")
    ex.once("r", passes, {}, out)
    m = {k: v for row in got for k, v in row.items()}
    assert abs(m["train.rhae_per_10k.L13"] - 1e4 * 6 * 2.0 / 120000) < 1e-9 and "train.rhae_per_10k.all" in m


def test_settled_run_retries_a_missed_read_before_exiting(tmp_path, monkeypatch):
    """gate-v5 (28 Oct): the pass after settle skipped the sampler read (another reader / slow volume) and the run's
    last sampler rows were never exported. A settled run's exporter repeats the pass while a read was missed."""
    ex = load(tmp_path, monkeypatch)
    calls = []

    def once(run, passes, st, out):
        calls.append(1)
        if len(calls) < 3:
            ex.MISSED.add("validation/x/live/sampler.jsonl")  # skipped twice, then read

    monkeypatch.setattr(ex, "once", once)
    monkeypatch.setattr(ex, "settled", lambda run: True)
    monkeypatch.setattr(ex, "queue_row", lambda run: {})
    monkeypatch.setattr(ex, "carry_target", lambda run, out: {})
    monkeypatch.setattr(ex.time, "sleep", lambda s: None)
    monkeypatch.setattr(ex.paths, "read_json", lambda p: {})
    monkeypatch.setattr(sys, "argv", ["export_rl.py", "r", "--passes", str(tmp_path), "--out", "r"])
    ex.main()
    assert len(calls) == 3  # missed, missed, clean -> exit
    calls.clear()
    monkeypatch.setattr(ex, "once", lambda *a: (calls.append(1), ex.MISSED.add("p")))
    ex.main()
    assert len(calls) == ex.FINAL_TRIES  # never clean: bounded


def test_new_attempt_folder_reads_its_feed_from_the_start(tmp_path, monkeypatch):
    """real-v11 attempt 3 (28 Oct): the cursor of attempt 2's feed skipped attempt 3's first 79 rows (boot, r3)."""
    ex = load(tmp_path, monkeypatch)
    feeds = {"r-attempt2": [{"kind": "r3_offload", "epoch": 1.0, "metrics": {"infra.r3_offload_bound": 1.0}}] * 3,
             "r-attempt3": [{"kind": "r3_offload", "epoch": 9.0, "metrics": {"infra.r3_offload_bound": 1.0}}]}
    monkeypatch.setattr(ex, "volume_text", lambda p: "\n".join(
        json.dumps(e) for f, es in feeds.items() if f"/{f}/" in p for e in es))
    monkeypatch.setattr(ex, "sampler_once", lambda *a: None)
    got = []
    out = type("O", (), {"emit": lambda self, m, t=None, alert=None: got.append((t, m))})()
    st = {}
    ex.OUT["r"] = "r-attempt2"
    ex.once("r", tmp_path, st, out)
    assert st["feed"] == 3 and len(got) == 3
    ex.OUT["r"] = "r-attempt3"
    ex.once("r", tmp_path, st, out)
    assert st["feed"] == 1 and st["feed_out"] == "r-attempt3" and got[-1] == (9.0, {"infra.r3_offload_bound": 1.0})


def test_one_exporter_per_run(tmp_path, monkeypatch):
    ex = load(tmp_path, monkeypatch)
    state = tmp_path / "r-export-state.json"
    assert ex.hold_instance(state)
    import subprocess
    code = ("import fcntl,sys; h=open(sys.argv[1],'w')\n"
            "try: fcntl.flock(h, fcntl.LOCK_EX|fcntl.LOCK_NB); print('got')\nexcept BlockingIOError: print('held')")
    out = subprocess.run([sys.executable, "-c", code, str(state.with_suffix(".lock"))], capture_output=True, text=True)
    assert out.stdout.strip() == "held"


def test_pending_only_resume_step_exports_with_replayed_tries(tmp_path, monkeypatch):
    """real-v12 (28 Oct): step 1 trained groups an earlier leg played (seq_replay, no group_done); the step row raised
    KeyError('tries') after the cursor passed it and was lost. Now its tries come from the replayed folders, and a
    row that raises is read again next pass."""
    ex = load(tmp_path, monkeypatch)
    att = [{"level": 1, "completion_tokens": 5000, "cleared": i < 2, "actions": 30, "level_score": 50, "total_levels": 4}
           for i in range(4)]
    feed = [{"kind": "seq_replay", "epoch": 1.0, "groups": 1, "tries": 4,
             "folders": ["/runs/validation/old/arena/trials/rl-loop/rl/p1g6-ka59-L1"]},
            {"kind": "seq_accumulate", "epoch": 2.0, "gid": "p1g6", "set": "ka59-L1", "tries": 4, "trained": 4},
            {"kind": "step", "epoch": 3.0, "step": 1, "lr": 3e-6, "trained": 4, "entropy": 0.3}]
    files = {"validation/new/arena/trials/rl-loop/rl/feed.jsonl": "\n".join(json.dumps(r) for r in feed),
             "validation/old/arena/trials/rl-loop/rl/p1g6-ka59-L1/level-bench.json": json.dumps({"attempts": att}, indent=1)}
    monkeypatch.setattr(ex, "volume_text", lambda p: files.get(p, ""))
    monkeypatch.setitem(ex.OUT, "new", "new")
    st = {"backfill": None}
    out = ex.metrics.writer("new")
    real = ex.feed_metrics
    boom = {"n": 1}

    def flaky(e, s):
        if e.get("kind") == "step" and boom["n"]:
            boom["n"] -= 1
            raise KeyError("x")
        return real(e, s)
    monkeypatch.setattr(ex, "feed_metrics", flaky)
    try:
        ex.once("new", tmp_path, st, out)
    except KeyError:
        pass
    assert st["feed"] == 2  # the step row was not passed
    ex.once("new", tmp_path, st, out)
    assert st["feed"] == 3
    rows = ex.metrics.read_rows(tmp_path / "train-metrics" / "new.jsonl")
    step = next(r["m"] for r in rows if r["m"].get("train.step") == 1)
    assert step["train.rhae_per_10k.L13"] == 1e4 * 2 * 2.0 / 20000


def test_unit_refs_fall_back_to_the_history_base(tmp_path, monkeypatch):
    """real-v12 (28 Oct): no screen, no set refs -> vs_ref was exp(-ln16 x fail rate); the bundle's history (v8.5p)
    now gives each unit its median tokens/actions to clear, so a clear's logeq is no longer 0 by construction."""
    ex = load(tmp_path, monkeypatch)
    dist = tmp_path / "dist" / "r"
    (dist / "configs").mkdir(parents=True)
    (dist / "data").mkdir()
    (dist / "configs/comparison.json").write_text(json.dumps(
        {"rl_loop": {"mg": {"sets": [{"name": "ab12-L1"}], "redesign": {"history": "data/h.json"}}}}))
    (dist / "configs/rl1_points.json").write_text(json.dumps({"points": [{"id": "ab12-x", "baselines": [10]}]}))
    rows = [{"game": "ab12", "level": 1, "cleared": True, "tokens": t, "actions": 20} for t in (1000, 2000, 3000)]
    rows.append({"game": "ab12", "level": 1, "cleared": False, "tokens": 9000, "actions": 50})
    (dist / "data/h.json").write_text(json.dumps({"rows": rows}))
    monkeypatch.setenv("ARC_RL_DIST", str(tmp_path / "dist"))
    monkeypatch.setattr(ex, "volume_text", lambda p: "")
    refs = ex.unit_refs("r")
    assert refs["screen"]["ab12-L1"] == {"median_to_clear": 2000, "median_actions": 20, "clear_rate": 0.75,
                                         "median_tokens": 2500.0}
    assert refs["base"] == {}  # no per-unit screen fetches for history units
    import math
    assert abs(ex.logeq(refs, "ab12-L1", [1, 1000, 1, 20, 0.0]) - math.log(2.0)) < 1e-9  # half the base tokens
    assert ex.logeq(refs, "ab12-L1", [1, 1000, 0, 0, 0.0]) == -math.log(16.0)


def test_a_reused_gid_takes_its_new_group_done_tries(tmp_path, monkeypatch):
    """real-v12 (28 Oct): a seq_replay seeded p1g6 with 16 carried tries; the run's own later p1g6 (8 tries) must
    replace them, not be skipped."""
    ex = load(tmp_path, monkeypatch)
    old = [{"level": 1, "completion_tokens": 100, "cleared": True, "actions": 5, "level_score": 50, "total_levels": 2}
           for _ in range(16)]
    new = [{"level": 1, "completion_tokens": 900, "cleared": False, "actions": 0, "level_score": 0, "total_levels": 2}
           for _ in range(8)]
    feed = [{"kind": "seq_replay", "epoch": 1.0, "folders": ["/runs/validation/old/arena/trials/rl-loop/rl/p1g6-ab12-L1"]},
            {"kind": "seq_accumulate", "epoch": 2.0, "gid": "p1g6", "set": "ab12-L1", "tries": 16, "trained": 16},
            {"kind": "step", "epoch": 3.0, "step": 1},
            {"kind": "group_done", "epoch": 4.0, "gid": "p1g6", "set": "ab12-L1", "version": 1, "tries": 8,
             "attempts": new},
            {"kind": "seq_accumulate", "epoch": 5.0, "gid": "p1g6", "set": "ab12-L1", "tries": 8, "trained": 8},
            {"kind": "step", "epoch": 6.0, "step": 2}]
    files = {"validation/new/arena/trials/rl-loop/rl/feed.jsonl": "\n".join(json.dumps(r) for r in feed),
             "validation/old/arena/trials/rl-loop/rl/p1g6-ab12-L1/level-bench.json": json.dumps({"attempts": old}, indent=1)}
    monkeypatch.setattr(ex, "volume_text", lambda p: files.get(p, ""))
    monkeypatch.setitem(ex.OUT, "new", "new")
    st, out = {}, ex.metrics.writer("new")
    ex.once("new", tmp_path, st, out)
    rows = ex.metrics.read_rows(tmp_path / "train-metrics" / "new.jsonl")
    s2 = next(r["m"] for r in rows if r["m"].get("train.step") == 2)
    assert s2["train.clear_rate.all"] == 0.0 and s2["train.tokens_per_try.all"] == 900


def test_ref_ratios_against_the_unit_reference(tmp_path, monkeypatch):
    """clear and fail rates as observed / expected under each unit's v8.5p clear rate, tokens per
    try as the geometric mean of tokens / the unit's median."""
    ex = load(tmp_path, monkeypatch)
    refs = {"screen": {"a-L1": {"clear_rate": 0.9, "median_tokens": 1000}, "b-L5": {"clear_rate": 0.5, "median_tokens": 4000}}}
    named = [("a-L1", [1, 1000, 1, 5, 1.0]), ("a-L1", [1, 1000, 1, 5, 1.0]),
             ("b-L5", [5, 2000, 1, 5, 1.0]), ("b-L5", [5, 8000, 0, 0, 0.0]), ("zz-L1", [1, 5, 0, 0, 0.0])]
    m = ex.ref_ratios(refs, named)
    assert abs(m["train.clear_rate_vs_ref.all"] - 3 / 2.8) < 1e-9  # expected 0.9 + 0.9 + 0.5 + 0.5
    assert abs(m["train.fail_rate_vs_ref.all"] - 1 / 1.2) < 1e-9
    assert abs(m["train.tokens_per_try_vs_ref.all"] - 1.0) < 1e-9  # 1, 1, 0.5, 2


def test_history_refs_carry_clear_rate_and_median_tokens(tmp_path, monkeypatch):
    ex = load(tmp_path, monkeypatch)
    rows = [{"game": "a", "level": 1, "cleared": True, "tokens": 100, "actions": 4},
            {"game": "a", "level": 1, "cleared": False, "tokens": 300, "actions": 9},
            {"game": "a", "level": 1, "cleared": False, "tokens": 900, "actions": 9, "censored": True}]
    (tmp_path / "h.json").write_text(json.dumps({"rows": rows}))
    r = ex.history_refs(tmp_path / "h.json")["a-L1"]
    assert abs(r["clear_rate"] - 1 / 3) < 1e-9 and r["median_tokens"] == 300 and r["median_to_clear"] == 100


def test_ref_logeq_counts_reference_fails_at_the_budget(tmp_path, monkeypatch):
    import math
    E = load(tmp_path, monkeypatch)
    refs = {"points": {}, "sets": {}, "screen": {"aa11-L1": {"median_to_clear": 100, "median_actions": 10}},
            "hist_rows": {"aa11-L1": [(1, 100, True, 10, False), (1, 300, True, 10, False),
                                      (1, 50, False, 5, True), (1, 400, False, 5, True)]}}
    # budget 200: the 300-token clear is a fail, the censored try at 50 is unknown, the one at 400 a fail
    v = E.ref_logeq(refs, "aa11-L1", 200)
    assert math.isclose(v, (0.0 - 2 * math.log(16)) / 3)
    # no budget: the history's own (1.25 x p80 of clears = 1.25 x 260 = 325): the 300 clear counts
    v2 = E.ref_logeq(refs, "aa11-L1", None)
    assert math.isclose(v2, (0.0 + math.log(100 / 300) - math.log(16)) / 3)
    assert E.ref_logeq(refs, "bb22-L1", 200) is None


def test_history_refs_long_runs_only(tmp_path, monkeypatch):
    E = load(tmp_path, monkeypatch)
    h = tmp_path / "h.json"
    h.write_text(json.dumps({"rows": [
        {"run": "b2-x-c100-15m-v1", "game": "aa11", "level": 1, "cleared": False, "tokens": 50, "actions": 3, "censored": True},
        {"run": "b2-x-c110-9h-v1", "game": "aa11", "level": 1, "cleared": True, "tokens": 100, "actions": 10},
        {"run": "b2-x-c100-300k-v1", "game": "aa11", "level": 1, "cleared": False, "tokens": 400, "actions": 9, "censored": True}]}))
    assert E.history_refs(h)["aa11-L1"]["clear_rate"] == 1 / 3
    assert E.history_refs(h, E.LONG_RUN)["aa11-L1"]["clear_rate"] == 1 / 2
    assert len(E.history_rows(h, E.LONG_RUN)["aa11-L1"]) == 2


def test_clear_and_fail_vs_ref_are_censoring_matched(tmp_path, monkeypatch):
    """clear/fail_vs_ref used v8.5p's unconditional clear rate, so budget cuts swung them. The expected
    clear rate is now v8.5p's under the group's budget (a clear past it is not one, a run ended before it unknown)."""
    ex = load(tmp_path, monkeypatch)
    rows = [(1, t, True, 5, False) for t in (100, 200, 300, 400)] + [(1, 50, False, 1, True)]
    refs = {"hist_rows": {"g-L1": rows}, "screen": {"g-L1": {"clear_rate": 0.8, "median_tokens": 250}}}
    assert ex.ref_clear_at(refs, "g-L1", 250) == 0.5 and ex.ref_clear_at(refs, "g-L1", 1000) == 1.0
    named = [("g-L1", (1, 250, 0, 0, 0.0)), ("g-L1", (1, 150, 1, 5, 1.0))]
    m = ex.ref_ratios(refs, named, [250, 250])
    assert m["train.clear_rate_vs_ref.all"] == 1.0 and m["train.fail_rate_vs_ref.all"] == 1.0


def test_vs_control_pairs_policy_and_twin_in_either_order(tmp_path, monkeypatch):
    """exp(mean of the last 4 (policy mean logeq - control mean logeq)), a pair counted once both rows
    are in, whichever comes first."""
    import math

    ex = load(tmp_path, monkeypatch)
    monkeypatch.setattr(ex, "unit_refs", lambda run: {})
    monkeypatch.setattr(ex, "logeq", lambda refs, name, r: r["v"])
    monkeypatch.setattr(ex, "try_rows", lambda text: json.loads(text)["attempts"])
    st = {"tries": {"p1g1": [{"v": 1.0}, {"v": 0.0}]}}
    ctl = {"kind": "control_done", "pair": "p1g1", "set": "g-L1", "step": 4, "attempts": [{"v": 0.0}, {"v": 0.0}]}
    assert ex.control_pairs(ctl, st, "r") == {"train.vs_control": round(math.exp(0.5), 4), "train.control_diff": 0.5,
                                              "train.control_pairs": 1}
    ctl2 = {**ctl, "pair": "p2g1", "step": 8, "attempts": [{"v": 1.0}]}
    assert ex.control_pairs(ctl2, st, "r") == {} and "p2g1" in st["ctl_wait"]  # the policy group still playing
    st["tries"]["p2g1"] = [{"v": 0.5}]
    m = ex.control_pairs({"kind": "group_done", "gid": "p2g1"}, st, "r")
    assert m["train.control_diff"] == -0.5 and m["train.vs_control"] == 1.0 and not st["ctl_wait"]


def test_minibatch_step_fields(tmp_path, monkeypatch):
    """4 steps per update: k3 and TIS clip of the last step vs the first, update-norm sum, grad max."""
    ex = load(tmp_path, monkeypatch)
    e = {"kind": "step", "step": 18, "epoch": 1.0, "update_norm_sum": 0.41, "grad_norm_max": 0.02,
         "minibatches": [{"approx_kl": 0.02, "clip_frac": 0.004}, {}, {}, {"approx_kl": 0.05, "clip_frac": 0.01}]}
    m = ex.feed_metrics(e, {})
    assert m["train.mb_k3_last_vs_first"] == 2.5 and m["train.mb_clip_last_vs_first"] == 2.5
    assert m["train.mb_clip_last"] == 0.01 and m["train.update_norm_sum"] == 0.41 and m["train.minibatches"] == 4
    assert "train.minibatches" not in ex.feed_metrics({"kind": "step", "step": 1, "epoch": 1.0}, {})


def test_attempt_reset_moves_dead_attempt_groups_out():
    """v17 29 Oct: attempt 1's played groups read as dropped at attempt 2's first step."""
    import export_rl as X

    st, m = {}, {}
    for i in range(3):
        X.group_counts({"kind": "group_done", "gid": f"a{i}", "tries": 8}, st, m)
    X.group_counts({"kind": "group_done", "gid": "z", "tries": 0}, st, m)
    st["grp"] = {"groups": 4}
    X.attempt_reset(st)
    assert "grp" not in st and st["gcount"]["lost_attempt"] == 4
    for i in range(2):
        X.group_counts({"kind": "group_done", "gid": f"b{i}", "tries": 8}, st, m)
    X.group_counts({"kind": "step", "step": 1}, st, m)
    assert m["train.groups_played_step"] == 2 and m["train.groups_dropped_step"] == 0
    assert m["train.groups_played"] == 2 and m["train.groups_lost_attempt"] == 4


def test_a_rollback_drops_the_control_pairs_of_the_abandoned_branch(tmp_path, monkeypatch):
    ex = load(tmp_path, monkeypatch)
    st = {"ctl_diffs": [[17, "a", -0.1], [21, "b", -1.64]], "ctl_wait": {"c": ["x", 0.0, 22]}, "ctl_top": 22}
    got = ex.control_pairs({"kind": "step", "step": 18}, st, "r")
    assert st["ctl_diffs"] == [[17, "a", -0.1]] and st["ctl_wait"] == {} and st["ctl_top"] == 18
    assert got["train.control_pairs"] == 1 and got["train.vs_control"] == 0.9048
    assert ex.control_pairs({"kind": "step", "step": 19}, st, "r") == {} and st["ctl_top"] == 19  # forward: no change
