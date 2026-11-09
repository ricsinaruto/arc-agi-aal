"""The redesign's held-out read (stage F): the read config (typed, roster, pending mask), the per-read bundle overlay,
the adapter hand-off (trainer copy -> fake bucket -> fetched copy), the scorer on synthetic reads, rl_seq's export
every N steps and the in-run dashboard rows."""
import asyncio
import json
import math
import random
import shutil
import struct
import sys
from pathlib import Path

import pytest

from arena import hread as H
from arena import level_bench, phase, rl_backlog, rl_seq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "launch/rl"))
sys.path.insert(0, str(ROOT / "launch/rl/heldout"))
import build as B  # noqa: E402
import config as C  # noqa: E402
import hread as R  # noqa: E402 (launch/rl/heldout/hread.py, the operator CLI)

CONFIG = ROOT / "launch/rl/runs/b2-rl-hread-rd-v1.json"


def safetensors(path, names):
    header = {n: {"dtype": "BF16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]} for i, n in enumerate(names)}
    raw = json.dumps(header).encode()
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0" * len(names))


def adapter(folder, draft=False, seed=0):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    names = [f"base_model.model.model.layers.{i}.mlp.experts.w1.lora_A.weight" for i in range(3)]
    safetensors(folder / "adapter_model.safetensors", names)
    if draft:  # stage D (rl1_draft.export): draft_lora.json + its A/B pair per module
        mods = ["fc_embedding", "layers.0.self_attn.q_proj"]
        (folder / "draft_lora.json").write_text(json.dumps({"format": "arc-draft-lora-v1", "r": 8, "lora_alpha": 16,
                                                            "modules": {m: {"vllm": m} for m in mods}}))
        safetensors(folder / "draft_lora.safetensors",
                    [f"base_model.model.mtp.{m}.lora_{p}.weight" for m in mods for p in "AB"])
    (folder / "adapter_config.json").write_text(json.dumps({"r": 16, "seed": seed}))
    (folder / "adapter-tensors.json").write_text(json.dumps({"tensors": len(names)}))
    return folder


# ---- config ---------------------------------------------------------------------------------------------------------

R15B = "b30b3b44ba4a7637209da4ed4bb96883d926688df912d78d1d177d505b58076e"  # main 57d060d2f prune-t37-r15b-nvfp4-v1
R15 = "7571ee74dd608ee8a0e5a3b8d609182f35593f5a1b8c0c35fa9bceb5da657b1a"


def pending_copy(tmp_path):
    d = json.loads(CONFIG.read_text())
    d["study"]["server"]["expert_mask"]["sha256"] = "PENDING-stage-G"
    path = tmp_path / "b2-rl-hread-rd-v1.json"
    path.write_text(json.dumps(d, indent=1) + "\n")
    return path


def test_read_config_is_typed_with_the_held_out_roster_and_the_train15_mask():
    cfg = C.load(CONFIG)
    assert C.checks(cfg) == [] and C.pending(cfg) == []
    mask = cfg.study["server"]["expert_mask"]  # the redesign template's keys (trunk/rl 91d482008), as training
    assert mask == {"mask": "data/expert_mask/prune-t37-r15b-nvfp4-v1.json", "sha256": R15B}
    assert B.sha256_file(ROOT / cfg.files[mask["mask"]]) == R15B
    t3 = C.load(ROOT / "launch/rl/runs/rl1-loop-seq-real-v8.json")
    assert t3.study["server"]["expert_mask"] == mask
    assert tuple(cfg.roster.ids) == tuple(sorted(rl_backlog.HELD_OUT)) and cfg.roster.size == 10
    assert not set(cfg.roster.ids) & set(rl_backlog.TRAIN)
    pts = json.loads((ROOT / "launch/rl/data/hread-rd-points-v1.json").read_text())["points"]
    seed = next(s for s in cfg.seeds if s.dest == cfg.study["level_bench"]["start_points"])
    assert B.tree_sha256(B.local(seed.path)) == seed.sha256
    assert [p["game"][:4] for p in pts] == list(cfg.roster.ids)  # launch.bundle_roster reads the file in order
    assert all(p["through"] and not p["prefix"] and p["level"] == 1 for p in pts)  # whole games from level 1
    lb = level_bench.settings(cfg.study["level_bench"])
    assert lb["token_budget"] == 200000 and lb["token_budget_by_point"] is None and lb["level_token_cap"] is None
    assert lb["arms"] == [{"name": "base"}] and lb["split"] == "all" and sorted(lb["games"]) == list(cfg.roster.ids)
    assert lb["concurrency"] == lb["passes"] * 10 == cfg.watch["expect"]["play.done"]  # N x 10 copies, all at once
    assert level_bench.load_points(ROOT / "launch/rl/data/hread-rd-points-v1.json", lb) == pts
    srv = cfg.study["server"]
    assert srv["mtp_tokens"] == 1 and srv["lora"]["mtp"] is True and srv["lora"]["moe_backend"] == "marlin"
    assert srv["serving_kernels"]["kv_cache_4bit"] == "int4" and srv["serving_kernels"]["draft_experts"] == "nvfp4"
    assert "draft_head" not in srv  # the frozen (checkpoint) draft; trunk/rl has no v8.5p overlay
    assert cfg.study["hread"]["every"] == 8 and cfg.study["load_adapters"] == [] and cfg.study["nebius_assets"] == []
    assert "ARENA_STALL_DUMP" not in cfg.study["trials"][0]["server_env"]  # rides on expert_prune on trunk/rl
    assert cfg.launch.backend == "nebius" and cfg.gpu == "RTX PRO 6000" and cfg.gpu_count == 1
    assert {"-pf", "-db"} <= set(cfg.preflight) and cfg.preflight["-db"]["ARC_RL1_DRY_BOOT"] == "1"


def test_pending_input_blocks_the_build(tmp_path):
    path = pending_copy(tmp_path)
    assert C.pending(C.load(path)) == ["study.server.expert_mask"]
    with pytest.raises(SystemExit, match="inputs pending"):
        B.build(path, tmp_path / "dist", gate=False)


# ---- per-read overlay -----------------------------------------------------------------------------------------------

VERIFY = '''import hashlib, json
from pathlib import Path
def verify(root):
    root = Path(root)
    m = json.loads((root / "MANIFEST.json").read_text())
    for k, v in m["files"].items():
        if hashlib.sha256((root / k).read_bytes()).hexdigest() != v:
            raise SystemExit("bad " + k)
'''


def fake_base_bundle(tmp, cfg):
    base = tmp / "dist" / cfg.bundle
    (base / "configs").mkdir(parents=True)
    (base / "configs/comparison.json").write_text(json.dumps(B.inline(C.launchcfg.to_dict(cfg.study)), indent=2) + "\n")
    (base / "bootstrap.py").write_text(VERIFY)
    (base / "src.txt").write_text("code")
    m = {"files": B.manifest(base), "git_commit": cfg.source, "spec_sha256": B.spec_sha256(cfg), "study": cfg.study["name"]}
    (base / "MANIFEST.json").write_text(json.dumps(m))
    return base


def test_per_read_bundle_changes_only_the_overlay(tmp_path, monkeypatch):
    cfg = C.load(CONFIG)
    base = fake_base_bundle(tmp_path, cfg)
    monkeypatch.setattr(R.B, "DIST", tmp_path / "dist")
    monkeypatch.setattr(R.L.B, "DIST", tmp_path / "dist")
    monkeypatch.setattr(R.B, "check_study", lambda out, study: level_bench.settings(study["level_bench"]))
    monkeypatch.setattr(R, "STATE", tmp_path / "state")
    train = "rl1-loop-rd-hero-v1"
    row = {"prefix": R.asset_prefix(train, 8), "digest": "d" * 64, "draft": None, "expert_mask_sha256": R15B, "src": f"validation/{train}/hread/s0008"}
    out0, rc0 = R.bundle(cfg, train, 0, gate=False)
    out8, rc8 = R.bundle(cfg, train, 8, row, gate=False)
    s0 = json.loads((out0 / "configs/comparison.json").read_text())
    s8 = json.loads((out8 / "configs/comparison.json").read_text())
    base_study = json.loads((base / "configs/comparison.json").read_text())
    assert R.overlay_problems(base_study, s0) == [] and R.overlay_problems(base_study, s8) == []
    assert s0["load_adapters"] == [] and s0["level_bench"]["arms"] == [{"name": "base"}]
    assert s8["load_adapters"] == [{"name": "policy", "path": "/kaggle/input/models/arc-rl/hread/rl1-loop-rd-hero-v1/s0008"}]
    assert s8["nebius_assets"] == [row["prefix"]] and s8["level_bench"]["arms"][0]["study"] == {"served_model": "policy"}
    assert s0["level_bench"]["seed_base"] == s8["level_bench"]["seed_base"]  # same seeds
    for out in (out0, out8):  # hard links: the same code bytes; a fresh manifest that verifies
        assert (out / "src.txt").stat().st_ino == (base / "src.txt").stat().st_ino
        m = json.loads((out / "MANIFEST.json").read_text())
        assert m["read"]["base_manifest_sha256"] == B.sha256_file(base / "MANIFEST.json")
    assert rc0.run == "b2-rl-hread-rd-v1-rd-hero-v1-s0000" and rc8.anchor.run == rc0.run
    assert f"arc-prolong-runs:{row['src']}" in rc8.data and C.checks(rc8) == []
    assert rc8.watch["metrics_remote"] == f"nebius:validation/{rc8.run}/arena/working/trials/heldout-read/train-metrics.jsonl"
    assert len(rc8.watch["vm"]) <= 59
    bad = json.loads(json.dumps(s8))
    bad["level_bench"]["passes"] = 4
    assert R.overlay_problems(base_study, bad) == ["level_bench.passes"]
    row16 = {**row, "prefix": R.asset_prefix(train, 16), "draft": "lora"}
    assert R.draft_merge(base_study)  # the config's servers merge a stage-D draft LoRA
    s16 = R.read_study(base_study, train, 16, row16)
    assert s16["hread"]["draft"] == "lora" and R.overlay_problems(base_study, s16) == []
    no_merge = json.loads(json.dumps(base_study))
    no_merge["trials"][0]["server_env"].pop("ARC_DRAFT_LORA")
    with pytest.raises(SystemExit, match="draft LoRA"):
        R.read_study(no_merge, train, 16, row16)
    assert R.read_study(no_merge, train, 8, row)["hread"]["draft"] is None  # no draft files: base draft, any server
    with pytest.raises(SystemExit, match="unknown draft"):
        R.read_study(base_study, train, 16, {**row16, "draft": "frozen"})
    assert "mask_check" not in base_study["hread"]  # only a speed bench turns the VM's mask check off
    bench = json.loads(json.dumps(base_study))
    bench["hread"]["mask_check"] = False
    with pytest.raises(SystemExit, match="speed bench"):
        R.read_study(bench, train, 8, row)
    for other in (R15, None):  # an adapter trained under r15 (or an unknown mask) never meets the r15b read
        with pytest.raises(SystemExit, match="expert mask"):
            R.read_study(base_study, train, 8, {**row, "expert_mask_sha256": other})


def test_vm_refuses_an_adapter_trained_under_another_mask(tmp_path):
    study = {"hread": {}, "server": {"expert_mask": {"sha256": R15B}}}
    ok = tmp_path / "ok"
    H.export_adapter(adapter(tmp_path / "a"), ok, {"expert_mask_sha256": R15B})
    old = tmp_path / "old"
    H.export_adapter(adapter(tmp_path / "b"), old, {"expert_mask_sha256": R15})
    assert H.mask_problem(ok, study) is None
    assert "trained under expert mask 7571ee74dd60" in H.mask_problem(old, study)
    assert "no hread-adapter.json" in H.mask_problem(adapter(tmp_path / "bare"), study)


# ---- hand-off ---------------------------------------------------------------------------------------------------------

def local_bucket(root):
    def put(local, remote, single=False):
        dest = Path(remote.replace("bucket:", str(root) + "/"))
        if single:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, dest)
        else:
            shutil.copytree(local, dest, dirs_exist_ok=True)

    def ls(remote):
        base = Path(remote.replace("bucket:", str(root) + "/"))
        return {str(p.relative_to(base)): p.stat().st_size for p in base.rglob("*") if p.is_file()}

    return put, ls


def test_handoff_trainer_to_bucket_to_vm_is_exact(tmp_path):
    src = adapter(tmp_path / "work/adapters/v0008")
    vol = tmp_path / "runs/validation/r/hread/s0008"
    man = H.export_adapter(src, vol, {"step": 8, "version": 8, "expert_mask_sha256": R15B})
    assert man["draft"] is None and set(man["files"]) == {"adapter_model.safetensors", "adapter_config.json",
                                                           "adapter-tensors.json"}
    assert json.loads((vol / H.MANIFEST).read_text())["digest"] == man["digest"] == H.digest(H.file_manifest(src))
    put, ls = local_bucket(tmp_path / "bucket")
    prefix = "flash-input/models/arc-rl/hread/r/s0008"
    row = H.publish_folder(vol, prefix, "bucket:", put, ls, tmp=tmp_path)
    assert row["digest"] == man["digest"] and row["files"] == 4
    bm = json.loads((tmp_path / "bucket/assets-manifests" / f"{H.asset_slug(prefix)}.json").read_text())
    vm = tmp_path / "vm/kaggle/input/models/arc-rl/hread/r/s0008"  # the VM's fetch: copy, then check vs the manifest
    shutil.copytree(tmp_path / "bucket/assets" / prefix, vm)
    assert H.verify_tree(vm, bm["files"]) == []
    assert H.digest({k: v for k, v in bm["files"].items() if k != H.MANIFEST}) == man["digest"]
    data = bytearray((vm / "adapter_model.safetensors").read_bytes())
    data[-1] ^= 1
    (vm / "adapter_model.safetensors").write_bytes(bytes(data))
    assert H.verify_tree(vm, bm["files"]) == ["adapter_model.safetensors: sha256 differs"]
    (vol / "adapter_config.json").write_text("{}")  # a volume copy that changed after the export: refused
    with pytest.raises(RuntimeError, match="hand-off manifest"):
        H.publish_folder(vol, prefix, "bucket:", put, ls, tmp=tmp_path)


def test_draft_lora_is_detected_and_a_bad_folder_refused(tmp_path):
    from arena.rl_loop import draft_files

    a = adapter(tmp_path / "a", draft=True)
    assert H.draft_kind(a) == "lora" and draft_files(a) == 2  # the trainer's own check agrees
    assert H.draft_kind(adapter(tmp_path / "b")) is None and draft_files(tmp_path / "b") == 0
    (a / "draft_lora.json").unlink()  # a partial pair would silently serve the base draft
    with pytest.raises(ValueError, match="partial draft"):
        H.draft_kind(a)
    e = adapter(tmp_path / "e", draft=True)
    safetensors(e / "draft_lora.safetensors", ["base_model.model.mtp.fc_embedding.lora_A.weight"])
    with pytest.raises(ValueError, match="1 tensors for 2 modules"):
        H.draft_kind(e)
    f = adapter(tmp_path / "f")
    safetensors(f / "adapter_model.safetensors", ["base_model.model.mtp.fc_hidden.lora_A.weight"])
    with pytest.raises(ValueError, match="mtp. tensors"):
        H.draft_kind(f)
    with pytest.raises(ValueError, match="partial draft"):
        H.export_adapter(a, tmp_path / "a-out")
    (tmp_path / "c").mkdir()
    with pytest.raises(ValueError, match="not a PEFT adapter"):
        H.export_adapter(tmp_path / "c", tmp_path / "d")


# ---- scorer -----------------------------------------------------------------------------------------------------------

def test_game_score_is_rl_loops():
    from arena.rl_loop import game_score

    rng = random.Random(1)
    for _ in range(200):
        total = rng.randint(1, 10)
        cleared = {lv: rng.choice([0.0, rng.uniform(0, 115)]) for lv in range(1, total + 1) if rng.random() < 0.6}
        assert H.game_score(cleared, total) == pytest.approx(game_score(cleared, total))


def whole(game, p, arm, levels, tokens, total=6, outcome="budget"):
    """A whole-game try: ``levels`` [(at_tokens, score)] for levels 1.. in order."""
    rows = [{"level": i + 1, "at_tokens": at, "tokens": at, "score": sc} for i, (at, sc) in enumerate(levels)]
    return {"game": f"{game}-abcd1234", "pass": p, "arm": arm, "outcome": outcome, "completion_tokens": tokens,
            "level_rows": rows, "total_levels": total}


def synthetic(seed=0, games=("sp80", "wa30", "sk48", "lf52"), lift=1.0):
    rng = random.Random(seed)
    base, pol = [], []
    for g in games:
        for p in range(3):
            n = rng.randint(0, 3)
            base.append(whole(g, p, "base", [(30_000 * (i + 1), 60.0) for i in range(n)], 200_000))
            m = min(6, n + (1 if rng.random() < 0.5 * lift else 0))
            won = m == 6
            pol.append(whole(g, p, "s0008", [(25_000 * (i + 1), 80.0) for i in range(m)],
                             25_000 * m if won else 200_000, outcome="won" if won else "budget"))
    return base, pol


def test_scorer_ratio_paired_se_and_matched_tokens():
    base, pol = synthetic()
    res = H.score(base, pol, "drop", budget=200_000)
    agg = res["aggregate"]
    assert agg["games"] == 4 and agg["pairs"] == 12 and res["unpaired"] == []
    # per game: ratio of sums x 1e4; the aggregate is the game mean
    for g, s in res["games"].items():
        b = [r for r in base if r["game"].startswith(g)]
        p = [r for r in pol if r["game"].startswith(g)]
        assert s["base"] == pytest.approx(sum(H.try_rhae(r) for r in b) / sum(r["completion_tokens"] for r in b) * 1e4, rel=1e-5)
        assert s["policy"] == pytest.approx(sum(H.try_rhae(r) for r in p) / sum(r["completion_tokens"] for r in p) * 1e4, rel=1e-5)
    assert agg["base"] == pytest.approx(sum(s["base"] for s in res["games"].values()) / 4, rel=1e-5)
    # paired SE: a brute-force bootstrap over pairs within games lands near the delta-method SE
    rng = random.Random(3)
    diffs = []
    for _ in range(3000):
        gs = []
        for g in res["games"]:
            pairs = [(b, p) for b, p in zip(base, pol) if b["game"].startswith(g)]
            pick = [rng.choice(pairs) for _ in pairs]
            st = H.ratio_stats(pick)
            gs.append(st["policy"] - st["base"])
        diffs.append(sum(gs) / len(gs))
    m = sum(diffs) / len(diffs)
    boot = math.sqrt(sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1))
    assert 0.5 < agg["se_diff"] / boot < 1.6  # delta method with n=3 per game vs bootstrap (n/(n-1) apart)
    assert agg["passes_bar"] == (agg["diff"] >= 2 * agg["se_diff"]) and agg["ratio"][1] <= agg["ratio"][0] <= agg["ratio"][2]
    mt = {r["tokens"]: r for r in res["matched_tokens"]}
    assert set(mt) == set(H.GRID)
    # at 50K the base can have cleared only level 1 (30K) and the policy levels 1-2 (25K, 50K)
    b50 = [H.try_rhae(r, 50_000) for r in base]
    assert all(v <= H.game_score({1: 60.0}, 6) / 100 + 1e-9 for v in b50)
    assert mt[200_000]["pairs"] == 12  # a won try counts at every T (its score is final)


def test_scorer_censored_modes_and_pairing_guard():
    base, pol = synthetic(1)
    pol[0] = {**pol[0], "outcome": "censored", "completion_tokens": 120_000}
    pol[1] = {**pol[1], "outcome": "error"}
    drop, part = H.score(base, pol, "drop"), H.score(base, pol, "partial")
    assert drop["aggregate"]["pairs"] == 10 and part["aggregate"]["pairs"] == 11
    assert len(drop["unpaired"]) == 2 and len(part["unpaired"]) == 1
    mt = {r["tokens"]: r["pairs"] for r in part["matched_tokens"]}
    assert mt[100_000] == 11 and mt[150_000] == 10  # the censored try is known up to its 120K tokens
    with pytest.raises(ValueError, match="two tries"):
        H.score(base + base[:1], pol)
    a = {"points": 10, "window": 1, "settings": {"passes": 3, "seed_base": 8500, "arms": [{"name": "base"}]}}
    b = {**a, "settings": {**a["settings"], "arms": [{"name": "s0008"}]}}
    assert H.same_read(a, b) == [] and H.same_read(a, {**b, "settings": {**b["settings"], "seed_base": 1}}) == ["settings.seed_base"]


def test_score_command_on_files(tmp_path, monkeypatch):
    base, pol = synthetic(2, lift=3.0)
    settings = {"token_budget": 200000, "passes": 3, "seed_base": 8500}
    for name, rows, arm in (("b.json", base, "base"), ("p.json", pol, "s0008")):
        (tmp_path / name).write_text(json.dumps({"points": 10, "settings": {**settings, "arms": [{"name": arm}]},
                                                 "attempts": rows}))
    monkeypatch.setattr(R, "STATE", tmp_path / "state")
    cfg = C.load(CONFIG)
    out = R.score(cfg, "local", 8, tmp_path / "b.json", tmp_path / "p.json", record=False)
    assert set(out["reads"]) == {"drop", "partial"} and out["reads"]["drop"]["aggregate"]["diff"] > 0
    assert isinstance(out["final_passes_bar"], bool)


# ---- rl_seq hand-off ----------------------------------------------------------------------------------------------------

def test_rl_seq_exports_the_adapter_every_n_steps(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tests"))
    import test_rl_seq as T

    import arena.restore as restore

    monkeypatch.setattr(rl_seq.harvest_mod, "snapshot", lambda a, d: Path(d).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(restore, "unsafe_from", lambda d: (None, None))
    monkeypatch.setattr(restore, "load", lambda d, r: {"level": int(Path(d).name.split("-L")[1].split("-")[0]) + 1})
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))

    class Trainer(T.FakeTrainer):
        async def call(self, op, **kw):
            out = await super().call(op, **kw)
            if op == "step":
                adapter(kw["save"], seed=self.steps)
            return out

    loop = T.FakeLoop(tmp_path, max_steps=4)
    loop.study = {"server": {"expert_mask": {"sha256": R15B}}}
    loop.cfg["mg"]["redesign"]["hread"] = {"every": 2}

    async def start_trainer():
        loop.log.append("start_trainer")
        loop.trainer = Trainer(loop.log)

    loop.start_trainer = start_trainer
    asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=phase.PhaseGate(inner=T.Upstream())), 30))
    due = loop.feed.of("hread_due")
    assert [r["step"] for r in due] == [0, 2, 4] and due[0]["path"] is None
    for r in due[1:]:
        folder = tmp_path / "vo" / "hread" / f"s{r['step']:04d}"
        man = json.loads((folder / H.MANIFEST).read_text())
        assert r["path"] == str(folder) and r["digest"] == man["digest"] and man["step"] == r["step"]
        assert H.verify_tree(folder, man["files"]) == [] and r["draft"] is None
        assert man["expert_mask_sha256"] == r["expert_mask_sha256"] == R15B  # the training mask, for the read's check
    assert not loop.feed.of("hread_error")


# ---- in-run exporter ----------------------------------------------------------------------------------------------------

def test_read_metrics_emit_the_dashboard_keys(tmp_path):
    (tmp_path / "vllm-gpu-memory.jsonl").write_text(json.dumps({"gpus": [{"utilization.gpu": 97.0, "memory.used": 95000.0,
                                                                          "memory.total": 97887.0}]}) + "\n")
    rm = H.ReadMetrics(tmp_path, {"step": 8, "usd_per_hour": 1.82, "boot_seconds": 1500,
                                  "base": {"rhae10k": 0.05, "se": 0.01}}, started=1000.0)
    base, pol = synthetic(4)
    line = {"event": "level_bench_progress", "done": 3, "running": 27, "failed": 0, "completion_tokens": 600_000}
    eng = {"vllm:kv_cache_usage_perc": 0.81, "vllm:prefix_cache_queries_total": 100.0, "vllm:prefix_cache_hits_total": 95.0}
    rm.emit(line, pol[:3], eng, now=1600.0)
    row = rm.emit({**line, "completion_tokens": 660_000, "event": "level_bench_final"}, pol[:3],
                  {**eng, "vllm:prefix_cache_queries_total": 200.0, "vllm:prefix_cache_hits_total": 194.0}, now=1720.0)
    m = row["m"]
    for k in ("eval.heldout_rhae10k", "eval.heldout_rhae10k_base", "eval.heldout_ratio", "infra.gen_tok_s",
              "infra.kv_frac", "infra.gpu_util.0", "infra.vram_gib.0", "infra.cost_usd", "infra.gen_tokens", "play.done"):
        assert k in m, k
    assert m["infra.gen_tok_s"] == pytest.approx(500.0) and m["infra.prefix_hit"] == pytest.approx(0.99)
    assert m["infra.cost_usd"] == pytest.approx(1.82 * (720 + 1500) / 3600)
    assert m["eval.read_step"] == 8 and row["note"] == "read end"
    rows = [json.loads(x) for x in (tmp_path / "train-metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(set(r) >= {"t", "m"} for r in rows)


def test_ref_ratios_whole_game_units():
    hist = [{"game": "aa11", "level": 1, "cleared": True, "tokens": 1000, "actions": 10},
            {"game": "aa11", "level": 1, "cleared": False, "tokens": 3000, "actions": 30, "censored": True},
            {"game": "aa11", "level": 2, "cleared": True, "tokens": 2000, "actions": 20}]
    refs = H.history_refs(hist)
    assert refs[("aa11", 1)]["clear_rate"] == 0.5 and refs[("aa11", 1)]["median_tokens"] == 2000
    assert refs[("aa11", 1)]["median_to_clear"] == 1000
    # cleared L1 at 500 tokens / 10 actions, stopped on L2 after 1500 more; L3 has no reference
    t1 = {"game": "aa11-x", "pass": 0, "outcome": "budget", "total_levels": 3, "completion_tokens": 2000,
          "total_actions": 25, "level_rows": [{"level": 1, "tokens": 500, "at_tokens": 500, "actions": 10}]}
    assert H.level_units(t1) == [("aa11", 1, 500, 10, True), ("aa11", 2, 1500, 15, False)]
    t2 = {"game": "aa11-x", "pass": 1, "outcome": "won", "total_levels": 3, "completion_tokens": 900, "total_actions": 9,
          "level_rows": [{"level": k, "tokens": 300, "at_tokens": 300 * k, "actions": 3} for k in (1, 2, 3)]}
    assert [u[4] for u in H.level_units(t2)] == [True, True, True]
    out = H.ref_ratios([t1, t2], refs, {"aa11": [10, 20, 30]})
    a = out["all"]
    assert (a["levels"], a["covered"], out["no_ref"]) == (5, 4, ["aa11-L3"])
    assert a["clear_rate_vs_ref"] == round(3 / 3.0, 4)  # expected 0.5 + 1 + 0.5 + 1
    assert a["fail_rate_vs_ref"] == 1.0  # 4 covered - 3 expected = 1 expected fail, 1 observed
    assert out["games"]["aa11"]["levels"] == 5


def test_ref_ratios_fail_ratio():
    refs = H.history_refs([{"game": "bb22", "level": 1, "cleared": c, "tokens": 100, "actions": 5} for c in (1, 1, 1, 0)])
    rows = [{"game": "bb22", "pass": i, "outcome": "budget", "total_levels": 2, "completion_tokens": 400, "total_actions": 4,
             "level_rows": []} for i in range(4)]
    a = H.ref_ratios(rows, refs)["all"]
    assert a["clear_rate_vs_ref"] == 0 and a["fail_rate_vs_ref"] == 4.0 and a["tokens_vs_ref"] == 4.0
    assert a["rhae_tok_vs_ref"] == a["rhae_tok_vs_ref_unmatched"] == round(16 ** -0.75, 4)  # no budget: as before


def test_ref_ratios_match_the_reference_censoring():
    """the base read's 0.65 was its 200K whole-try budget. A level cut at t tokens is compared with v8.5p
    at budget t: v8.5p's clears past t are fails there, its runs that ended before t unknown."""
    hist = [{"game": "bb22", "level": 1, "cleared": True, "tokens": t, "actions": 5} for t in (100, 200, 300, 400)]
    hist.append({"game": "bb22", "level": 1, "cleared": False, "tokens": 50, "actions": 1, "censored": True})
    refs = H.history_refs(hist)
    cut = {"game": "bb22", "label": "a", "outcome": "budget", "total_levels": 2, "completion_tokens": 250,
           "total_actions": 3, "level_rows": []}
    a = H.ref_ratios([cut], refs, budget=250)["all"]
    assert a["matched"] == 1 and a["clears_ref"] == 0.5  # within 250: 2 of 4 (the run cut at 50 left out)
    assert a["rhae_tok_vs_ref"] == round(math.exp(-math.log(16) - H.ref_at(("bb22", 1), refs[("bb22", 1)], [], 250)[0]), 4)
    assert a["rhae_tok_vs_ref"] > a["rhae_tok_vs_ref_unmatched"]
    assert H.unit_budgets([("g", 1, 100, 1, True), ("g", 2, 50, 1, False)], 200) == [200, 100]
    per = H.ref_ratios([cut], refs, budgets={"a": 1000})["all"]  # a training group's per-level budget
    assert per["clears_ref"] == 1.0
