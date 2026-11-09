"""Export one draft-head checkpoint as a served overlay, as v8.5's step-300 head was made (export_dh_step.py, 7 Oct):
the run's final overlay manifest with the tensors swapped; patched-shard hashes not computed (the loader skips them).
usage: export_step.py TRAINER_DIR STEP OUT_DIR SOURCE_RUN   (TRAINER_DIR: overlay/, checkpoints/, draft-head-train.json)"""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from arena import weights_overlay as wo  # noqa: E402

src, step, out, run = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
state = torch.load(src / f"checkpoints/step-{step:05d}.pt", map_location="cpu")
tensors = {n: ("BF16", list(t.shape), t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes())
           for n, t in state.items()}
m = json.loads((src / "overlay/overlay-manifest.json").read_text())
assert set(tensors) == set(m["tensors"]), "checkpoint tensors differ from the run's export"
out.mkdir(parents=True, exist_ok=True)
wo.write_safetensors(out / wo.OVERLAY_FILE, tensors, metadata={"format": wo.FORMAT})
m["overlay"].update(bytes=(out / wo.OVERLAY_FILE).stat().st_size, sha256=wo.sha256_file(out / wo.OVERLAY_FILE))
for n, (_, _, data) in tensors.items():
    m["tensors"][n]["sha256"] = wo.sha256_bytes(data)
for shard in m["patched"].values():
    shard["sha256"] = None
report = json.loads((src / "draft-head-train.json").read_text())
m["curve_at_step"] = next((c for c in report.get("curve", []) if c.get("step") == step), None)
m["curve_before"] = next((c for c in report.get("curve", []) if c.get("step") == 0), None)
m.update(selected_step=step, source_run=run, identical_to_base=False,
         producer="arena.draft_head_train checkpoint, exported by scripts/rsi2/export_step.py "
                  "(patched-shard hashes not computed)")
for k in ("eval_after", "eval_before"):
    m.pop(k, None)
(out / wo.MANIFEST_FILE).write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
print(wo.sha256_file(out / wo.MANIFEST_FILE), m["overlay"]["sha256"], m["curve_at_step"] and
      m["curve_at_step"].get("curve", {}).get("primary", {}).get("p_accept_step"))
