"""p75 ckpt-893 INT3 dn4g8 refit on the 15 train games, 100K held-out read.
cap DATASET SHA: A's capture bundle (inputs-r15-v1, t75 nodrop mask), overlay = c893 NVFP4 RTN stand-in + dense/MTP.
read DATASET SHA [--gbps X]: SFT's b2-int8-p75c893-ho10x6-100k-v1 tree (token_stop, v9b harness) with the INT3 path p50 ran
(NVFP4 base, expert_overlay, expert_int3 16, humming, INT8 dense at load), dn4g8 env of v9."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

W = Path("./.claude/worktrees")
SP = Path("<scratch>")
DN4 = {"ARC_EXPERT_INT3_FP4_DOWN": "1", "ARC_EXPERT_INT3_FP4_LAYERS": "20,23,28,31,32,33,35,40"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cap(dataset, msha):
    src, out = W / "sft-prune/dist/cap-v85p-p75c176-v1", W / "sft-prune/dist/cap-v85p-p75c893-tr15-v1"
    old, new = "b2-sft-cap-v85p-p75c176-v1", "b2-tp-cap-v85p-p75c893-tr15-v1"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(src, out, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
    p = out / "configs/comparison.json"
    s = json.loads(p.read_text())
    assert s["expert_quant"]["inputs"]["source"] == "inputs-r15-v1"
    for f, h in s["expert_quant"]["inputs"]["sha256"].items():
        assert sha(out / "configs/expert-quant" / f) == h, f
    s["name"] = out.name
    s["server"]["expert_overlay"] = {"dataset": dataset, "manifest_sha256": msha, "check": "sha256"}
    s["model_datasets"] = [dataset]
    text = json.dumps(s, indent=2).replace(old, new).replace(src.name, out.name) + "\n"
    assert "p75c176" not in text, [x for x in text.splitlines() if "p75c176" in x][:5]
    p.write_text(text)
    manifest(out, ("kernel-metadata.json", "build-manifest.json"))
    print(out, s["server"]["expert_overlay"])


def read(dataset, msha, gbps):
    src = SP / "sftspeed/int8/b2-int8-p75c893-ho10x6-100k-v1"
    run = "b2-int3dn4g8-p75c893-ho10x6-100k-v1"
    out = W / "v84sub/dist" / run
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("__pycache__"))
    p = out / "configs/comparison.json"
    d = json.loads(p.read_text().replace(src.name, run))
    assert d["competition"]["rehearsal"] is True and d["token_stop"] == 115000
    for k in ("research_checkpoint", "nebius_assets", "nebius_asset_skip", "nebius_boot_disk_gib"):
        d.pop(k)
    s = d["server"]
    s["expert_overlay"] = {"dataset": dataset, "manifest_sha256": msha, "check": "size"}
    s["serving_kernels"].update(expert_int3=16, moe_backend="humming")
    s["dense_kernels"]["fp8_w8a8"] = []
    d["model_datasets"] = [dataset]
    env = d["trials"][0]["server_env"]
    env.pop("ARC_PLE_FP8")  # the research path's PLE switch; the NVFP4 base already serves FP8 PLE
    env.update(DN4)
    d["heldout"].pop("heal_overlay", None)
    d["heldout"].update(overlay=dataset, overlay_manifest_sha256=msha,
                        model={"stack": "p75 ckpt-893: experts INT3 g16 dn4g8 GPTQ refit on the 15 train games (native capture), "
                                         "INT8 dense at load from its BF16 dense, own MTP",
                               "student": "/runs/validation/sft-p75-kd-c613-v1d/train/ckpt-893/student"})
    if gbps:
        d["focus_oversubscription"]["gate_blocks_per_stream"] = gbps
    d["active_limit"] = 53; d["focus_oversubscription"]["focus_slots"] = 53  # the anchor's streams: ms/step and tok/s pair
    d["roster"] = "held-out read c893 INT3 dn4g8 refit on the 15: 10 held-out games x 6, 100K tokens per try"
    d["evidence"]["int3_read"] = "from b2-int8-p75c893-ho10x6-100k-v1: expert format only"
    p.write_text(json.dumps(d, indent=2) + "\n")
    manifest(out, ())
    print(out, run, d["active_limit"], d["focus_oversubscription"]["gate_blocks_per_stream"])


def manifest(out, skip):
    m = json.loads((out / "MANIFEST.json").read_text())
    m["files"] = {q.relative_to(out).as_posix(): sha(q) for q in sorted(out.rglob("*"))
                  if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"
                  and q.name not in skip and "dataset-upload" not in q.parts}
    (out / "MANIFEST.json").write_text(json.dumps(m, indent=1) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "cap":
        cap(a[1], a[2])
    else:
        read(a[1], a[2], float(a[a.index("--gbps") + 1]) if "--gbps" in a else 0)
