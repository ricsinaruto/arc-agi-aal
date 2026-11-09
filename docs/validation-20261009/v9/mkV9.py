"""v9 ARC-AGI-3 candidate, 9 h held-out kf bundle.
Base: v84sub dist/arena-flash-competition-v87b-sub (fp8_dense + dn4g8 blocks), re-pointed by heldout_bundle.py at p75's
t75 nodrop mask and the v9 overlay (ckpt-268 experts INT3 g16 dn4g8 fit on all 25 + ckpt-268 dense/MTP shards), then:
- no draft-head overlay (p75's own MTP block rides in the dense shards);
- FP8 dense calibration refit on ckpt-268 (same families, same ARC_FP8_BF16 layers);
- no CPU KV tier (cpu_cache_gib 0, offload env dropped); streams from A''s per-stream KV (density rule, see STREAMS);
- drain jitter (v84-sub 77e18eef0): T ~ U[90,000, 134,000] per drain, keep 64,000; window 151,040;
- qsa_fast (kern-qsa b10d7b3aa: serving_kernels_patch.py + qsa_fast_runtime.py, serving_kernels.qsa_fast);
- GDN deferred commit (kern-gdn GDN_COMMIT) only with --gdn.
Usage: mkV9.py NAME OVERLAY OVERLAY_SHA CALIB [--gdn COMMIT]"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(".")
V84 = REPO / ".claude/worktrees/v84sub"
D = V84 / "dist"
WT = REPO / ".claude/worktrees/agent-aad6619d37e5f5311"
PY = str(REPO / ".venv/bin/python")
SRC = D / "arena-flash-competition-v87b-sub"
MASK = REPO / ".claude/worktrees/sft-prune/dist/qadv2-v87s-p75c116-ho10x6-100k-v1/configs/expert-prune/prune-t75-nodrop-excl5-v1.json"
MASK_SHA = "058b173af9498516060eeed69f64368a135e93883d6c76de074772f48de2ed1e"
JITTER_BASE, JITTER_COMMIT = "883f38985", "77e18eef0"
QSA_COMMIT, QSA_HELPER_SHA = "b10d7b3aa", "bc9823e69c8968f31e73662bec7b149af97dcb174e513d4455e525753af9bf68"
OLD_CALIB = "arc-fp8-calib-all25-v1"
# STREAMS (derivation, plan): A' 9 h h3-8.6 = pool 2,662 blocks, active 61, KV 0.889 -> 1.457% of the pool per active
# stream; KV 0.80 with the jitter's +3% mean context -> 0.80 / (1.457% x 1.03) = 53.3 streams on A''s pool, i.e.
# gate_blocks_per_stream = 0.96 x 2,662 / 53.3 = 47.9 in the density rule min(target, floor(capacity / gbps)).
GBPS, TARGET, SEQS = 47.9, 60, 64
WINDOW, LOW, HIGH, KEEP, SEED = 151040, 90000, 134000, 64000, 20261029

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("overlay")
ap.add_argument("overlay_sha")
ap.add_argument("calib", help="dataset name of the ckpt-268 FP8 calibration (richardcsaky/<calib>)")
ap.add_argument("--gdn", default="", help="kern-gdn commit (>= 07bd7f6c2) to add the GDN deferred commit")
ap.add_argument("--cpu-gib", type=int, default=0, help="CPU KV tier (ArcOffloadingConnector, offload_prompt_only false); 0 = none")
ap.add_argument("--gbps", type=float, default=GBPS, help="density-rule gate blocks per stream")
a = ap.parse_args()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_show(commit, path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], check=True, capture_output=True).stdout


assert sha(MASK) == MASK_SHA
sub, out = D / f"{a.name}-sub", D / a.name
for p in (sub, out):
    if p.exists():
        shutil.rmtree(p)
subprocess.run([PY, str(WT / "scripts/dfr_exact/int3/heldout_bundle.py"), str(SRC), str(sub), "--mask", str(MASK),
                "--overlay", a.overlay, "--overlay-sha", a.overlay_sha,
                "--strict-overlay", str(D / "v87b-sub-prev600/src/arena/expert_overlay.py")], check=True)
changed, removed = [], []

# --- source: drain jitter, qsa_fast, (GDN) ---
diff = subprocess.run(["git", "-C", str(V84), "diff", JITTER_BASE, JITTER_COMMIT, "--", "src/arena/context_drain.py",
                       "src/arena/duck.py"], check=True, capture_output=True, text=True).stdout
subprocess.run(["patch", "-p1", "--forward", "--no-backup-if-mismatch", "-d", str(sub)], input=diff, text=True, check=True)
changed += ["src/arena/context_drain.py", "src/arena/duck.py"]
skp = sub / "src/arc_infra/serving_kernels_patch.py"
assert sha(skp) == "34e4b646f8f8b744301288c03990d5b32001dd599f3c22246cc941eb95433a88", "v87b serving_kernels_patch changed"
skp.write_bytes(git_show(QSA_COMMIT, "src/arc_infra/serving_kernels_patch.py"))
(sub / "src/arc_infra/qsa_fast_runtime.py").write_bytes(git_show(QSA_COMMIT, "src/arc_infra/qsa_fast_runtime.py"))
assert sha(sub / "src/arc_infra/qsa_fast_runtime.py") == QSA_HELPER_SHA
changed += ["src/arc_infra/serving_kernels_patch.py", "src/arc_infra/qsa_fast_runtime.py"]
if a.gdn:
    for f in ("stack_kernels_patch.py", "gdn_defer_runtime.py", "gdn_defer.cu"):
        (sub / "src/arc_infra" / f).write_bytes(git_show(a.gdn, f"src/arc_infra/{f}"))
        changed.append(f"src/arc_infra/{f}")

# --- study ---
p = sub / "configs/comparison.json"
s = json.loads(p.read_text())
srv = s["server"]
head = srv.pop("draft_head")
assert head["overlay"]["path"] == "draft-head/overlay" and not head.get("dump"), head
removed = [str(f.relative_to(sub)) for f in (sub / "draft-head").rglob("*") if f.is_file()]
shutil.rmtree(sub / "draft-head")
kern = srv["serving_kernels"]
assert kern["kv_cache_4bit"] == "int4" and kern["indexer_fp8"] is False and kern["expert_int3"] == 16, kern
kern["qsa_fast"] = True
if a.gdn:
    assert srv["stack_kernels"] == {"async_h2d": True}, srv["stack_kernels"]
    srv["stack_kernels"] = {"async_h2d": True, "gdn_defer": True}
# FP8 dense calibration refit on ckpt-268
assert s["fp8_dense"]["calib"] == f"richardcsaky/{OLD_CALIB}"
s["fp8_dense"]["calib"] = f"richardcsaky/{a.calib}"
s["model_datasets"] = [d if d != f"richardcsaky/{OLD_CALIB}" else f"richardcsaky/{a.calib}" for d in s["model_datasets"]]
assert f"richardcsaky/{a.calib}" in s["model_datasets"], s["model_datasets"]
for t in s["trials"]:
    env = t["server_env"]
    assert env["ARC_FP8_CALIB"] == f"/tmp/arc-fp8-calib/{OLD_CALIB}", env["ARC_FP8_CALIB"]
    env["ARC_FP8_CALIB"] = f"/tmp/arc-fp8-calib/{a.calib}"
    assert env["ARC_EXPERT_INT3_FP4_DOWN"] == "1" and env["ARC_EXPERT_INT3_FP4_LAYERS"] == "20,23,28,31,32,33,35,40"
    for k in ("ARC_OFFLOAD_EAGLE_TAIL", "ARC_OFFLOAD_NO_EAGLE_POP"):  # CPU-tier only
        assert env[k] == "1"
        if not a.cpu_gib:
            env.pop(k)
# no CPU KV tier; streams
assert s["cache_experiment"]["cpu_cache_gib"] == 32
s["cache_experiment"]["cpu_cache_gib"] = a.cpu_gib
s["active_limit"] = TARGET
s["focus_oversubscription"]["focus_slots"] = TARGET
s["focus_oversubscription"]["gate_blocks_per_stream"] = a.gbps
s["generation_slots"] = SEQS
srv["max_num_seqs"] = SEQS
need = SEQS * (1 + int(srv["mtp_tokens"]))
assert int(srv["mtp_tokens"]) == 2
srv["cudagraph_capture_sizes"] = sorted(set(srv["cudagraph_capture_sizes"]) | set(range(176, need + 16, 16)))
srv["max_cudagraph_capture_size"] = max(srv["cudagraph_capture_sizes"])
assert srv["max_cudagraph_capture_size"] >= need, srv["cudagraph_capture_sizes"]
# drain jitter
assert s["windows"] == [130816] and s["completion_cap"] == 16384
s["windows"] = [WINDOW]
s["drain_half"] = {**s["drain_half"], "keep_tokens": KEEP, "jitter": {"low": LOW, "high": HIGH, "seed": SEED}}
assert WINDOW - 16384 - 512 >= HIGH
s["heldout"]["model"] = {
    "stack": "v9: p75 ckpt-268 (t75 nodrop), experts INT3 g16 dn4g8 GPTQ-refit on its native all-25 capture, dense/MTP "
             "shards of ckpt-268, FP8 dense calibration refit on ckpt-268, p75's own MTP head, qsa_fast"
             + (f", GDN deferred commit (kern-gdn {a.gdn})" if a.gdn else ", no GDN deferred commit")
             + (f", {a.cpu_gib} GiB CPU KV tier" if a.cpu_gib else ", no CPU KV tier") + ", drain jitter",
    "streams": f"density rule min({TARGET}, floor(capacity / {a.gbps})); {SEQS} seqs; derivation in mkV9.py STREAMS",
    "drain": f"trigger T ~ U[{LOW}, {HIGH}] per drain (sha256(seed|game|index), seed {SEED}), keep {KEEP}; window {WINDOW}",
    "note": "the all-25 fits include the held-out games: not a clean held-out read (same for v8.7b)"}
text = json.dumps(s, indent=2) + "\n"
p.write_text(text)
for f in ("experiment.ipynb", "kernel-metadata.json", "build-manifest.json"):
    q = sub / f
    if q.exists():
        q.write_text(q.read_text().replace(OLD_CALIB, a.calib))
m = json.loads((sub / "MANIFEST.json").read_text())
for r in removed:
    m["files"].pop(r, None)
for c in changed + ["configs/comparison.json"]:
    m["files"][c] = sha(sub / c)
(sub / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
subprocess.run([PY, str(V84 / "docs/validation-20261009/nebius_kaggle_faithful.py"), str(sub), str(out), "--live-score"],
               check=True)
print("built", out, "changed", changed, "removed", removed)
