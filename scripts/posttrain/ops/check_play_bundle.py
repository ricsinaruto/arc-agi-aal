"""check_play_bundle.py BUNDLE_DIR: refuse a play bundle whose code would ignore part of its study (c1play-v1, 19 Oct).
1. every study key (top level, research_checkpoint.*, server.*) is named as a string in the bundle's own code;
2. the bundle's checkpoint loader has a KNOWN key set covering research_checkpoint;
3. with heal_overlay set, the bundle's loader runs on the real config/index/overlay manifest (shards stubbed) and
   must map every overlay tensor. Prints CHECK OK / CHECK FAILED; exit 1 on failure."""
import hashlib, importlib.util, json, re, subprocess, sys, tempfile, types
from pathlib import Path

B = Path(sys.argv[1]).resolve()
BUCKET = "nebius-arc:arc-runs-bucket"
CACHE = Path("~/arc-sft-data/cache/bundle-check")
# records only: read at build time (build_arena / build_v82_ablation) or never
META = {"name", "notes", "note", "what", "why", "hero_v2", "arena_variant", "serving_ablation", "evidence", "v82",
        "context_policy", "harness_selection", "smoke_active", "ple_prefill_bound", "games_source", "kernel_timeout"}
# build_v82_ablation.py RECORD_KEYS (dfr-exact): records of patches applied at build time, not read at run time
META |= {"serving_ablation", "reply_reserve_tokens", "server_patch", "study_patch", "qsa_dump", "fp8_dense",
         "keepalive_until", "tail_multidrop", "gdn_defer", "kvoff", "ple_prefill_bound", "probe_retry", "steady_replay",
         "cache_budget_record", "notebook_seconds", "dense_capture"}
import os
LAUNCHER = Path(os.environ.get("ARC_NEBIUS_SCRIPTS", "./.claude/worktrees/nebius/scripts")) / "nebius_validation.py"  # reads nebius_assets
study = json.loads((B / "configs/comparison.json").read_text())
code = "\n".join(p.read_text(errors="ignore") for p in [*B.glob("src/**/*.py"), *B.glob("bootstrap*.py"), LAUNCHER])
fail = []

for prefix, block in [("", study), ("research_checkpoint.", study.get("research_checkpoint") or {}),
                      ("server.", study.get("server") or {})]:
    for k in block:
        if k not in META and not re.search(r"""['"]%s['"]""" % re.escape(k), code):
            fail.append(f"study key {prefix}{k} is not read by the bundle's code")

spec = study.get("research_checkpoint")
if spec:
    m = importlib.util.spec_from_file_location("bundle_rc", B / "src/arena/research_checkpoint.py")
    rc = importlib.util.module_from_spec(m); m.loader.exec_module(rc)
    known = getattr(rc, "KNOWN", None)
    if known is None:
        fail.append("checkpoint loader has no KNOWN key set (predates the unknown-key guard)")
    elif set(spec) - known:
        fail.append(f"checkpoint loader ignores {sorted(set(spec) - known)}")
    if spec.get("heal_overlay") and known is not None:
        def fetch(remote, name):
            d = CACHE / hashlib.sha256(remote.encode()).hexdigest()[:16]; d.mkdir(parents=True, exist_ok=True)
            if not (d / name).is_file():
                r = subprocess.run(["rclone", "copyto", f"{BUCKET}{remote}/{name}", str(d / name)], capture_output=True)
                if r.returncode:
                    return None
            return d / name if (d / name).is_file() else None  # rclone exits 0 on a missing remote file
        with tempfile.TemporaryDirectory() as t:
            root, ov, work = Path(t, "root"), Path(t, "ov"), Path(t, "work")
            for d in (root, ov, work):
                d.mkdir()
            for n in ("config.json", "model.safetensors.index.json", "quantization_config.json"):
                f = fetch(spec["path"], n)
                if f:
                    (root / n).write_bytes(f.read_bytes())
                elif n != "quantization_config.json":
                    fail.append(f"cannot fetch {spec['path']}/{n}")
            f = fetch(spec["heal_overlay"], "overlay.json")
            if not f:
                fail.append(f"cannot fetch {spec['heal_overlay']}/overlay.json")
            else:
                meta = json.loads(f.read_text()); (ov / "overlay.json").write_text(f.read_text())
                for name in set(meta["weight_map"].values()):
                    (ov / name).touch()
                for name in set(json.loads((root / "model.safetensors.index.json").read_text())["weight_map"].values()):
                    (root / name).touch()
                local = {k: v for k, v in spec.items() if k != "ple_fp8"} | {"path": str(root), "heal_overlay": str(ov)}
                try:
                    out = rc.configure(types.SimpleNamespace(SERVED_MODEL_NAME="check"), work, local)
                    if out.get("heal_overlay_tensors") != len(meta["weight_map"]) or not meta["weight_map"]:
                        fail.append(f"overlay maps {out.get('heal_overlay_tensors')} of {len(meta['weight_map'])} tensors")
                    else:
                        print("overlay tensors", out["heal_overlay_tensors"])
                except Exception as e:
                    fail.append(f"loader refused: {e}")

print("\n".join(fail) or "", "\nCHECK FAILED" if fail else "CHECK OK", B.name)
sys.exit(1 if fail else 0)
