"""In-run checkpoint reload check of a streaming heal round (no GPU gate phase; the first checkpoint is
reloaded and compared while training continues). Modal CPU, beside the round:

1. waits for the round's first checkpoint folder ``train/ckpt-<k>`` and at once copies the trainer's own export of the
   same step (``deltas/v<k>`` in the bucket; the exporter keeps only the newest few versions there);
2. once ``progress.json`` marks the checkpoint complete, joins every rank's ``params-rank<r>.pt`` shards and rebuilds
   the export from them with heal_delta's own rule: dense tensors and routers must equal the export bit for bit (BF16),
   sampled trained expert matrices must re-quantise to the exported int4 (codes, zeros, scales);
3. checks the optimizer shards (finite, step counters equal across ranks).

The export is the live trainer's weights at step k (gathered from the FSDP shards in memory), so a match means the
checkpoint files hold exactly the weights the round trains, joined in the right order. The restart code path that
loads them (heal_kd load_local_params) was proven on the same bundle by the real-trainer A/B resume check.

    modal run --detach scripts/posttrain/heal_reload_check.py --run-id sft-heal-round-NAME
"""
import json
import os
import random
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
TESTS = HERE.parents[1] / "tests" if len(HERE.parents) > 1 else HERE  # in the container the module sits in /root
NEBIUS_BUCKET = "arc-runs-bucket"
image = (modal.Image.debian_slim(python_version="3.12").apt_install("rclone")
         .pip_install("torch==2.14.0", "safetensors", "numpy", extra_index_url="https://download.pytorch.org/whl/cpu")
         .add_local_file(HERE / "heal_delta.py", "/opt/rc/heal_delta.py"))
for _p in sorted(HERE.glob("heal_prune-*.json")):
    image = image.add_local_file(_p, f"/opt/rc/{_p.name}")
app = modal.App("arc-heal-reload-check")
runs = modal.Volume.from_name("arc-prolong-runs")


def _rclone_env():
    os.environ.update(RCLONE_CONFIG_ARC_TYPE="s3", RCLONE_CONFIG_ARC_PROVIDER="Other", RCLONE_CONFIG_ARC_ENV_AUTH="true",
                      RCLONE_CONFIG_ARC_ENDPOINT="https://storage.eu-south1.nebius.cloud",
                      RCLONE_CONFIG_ARC_REGION="eu-south1", RCLONE_CONFIG_ARC_NO_CHECK_BUCKET="true")


def compare(ckpt, export, cut, *, experts=48, seed=0, log=print):
    """Rebuild ``export`` (served name -> tensor) from the joined shards of ``ckpt``; returns the verdict dict."""
    import torch

    import heal_delta

    prog = json.loads((ckpt / "progress.json").read_text())
    world = int(prog.get("world") or len(list(ckpt.glob("params-rank*.pt"))))
    shards = [torch.load(ckpt / f"params-rank{r}.pt", map_location="cpu", weights_only=True, mmap=True)
              for r in range(world)]
    names = [k for k in shards[0] if not k.endswith("::experts")]
    if any(sorted(s) != sorted(shards[0]) for s in shards):
        return {"ok": False, "error": "ranks hold different tensor names"}

    class Experts:  # rows-only shards (c7b 24 Oct): global expert id -> matrix, indexed like the full tensor
        def __init__(self, key):
            self.mats = {int(e): s[key][i] for s in shards for i, e in enumerate(s[key + "::experts"].tolist())}
            one = next(iter(self.mats.values())) if self.mats else None
            self.shape = (max(self.mats) + 1 if self.mats else 0, *(one.shape if one is not None else ()))

        def __getitem__(self, e):
            return self.mats[e]

    def joined(key):
        return Experts(key) if key + "::experts" in shards[0] else torch.cat([s[key] for s in shards], 0)
    dense = {"checked": 0, "equal": 0, "bad": []}
    fused, seen = [], set()
    for key in names:
        name = key[len("model."):] if key.startswith("model.") else key
        if heal_delta.FUSED.match(name):
            fused.append((name, key))
            continue
        full = joined(key)
        for served, t in heal_delta.entries(name, full, cut):
            seen.add(served)
            dense["checked"] += 1
            got = export.get(served)
            if got is not None and got.shape == t.shape and torch.equal(got, t):
                dense["equal"] += 1
            elif len(dense["bad"]) < 8:
                dense["bad"].append({"name": served, "missing": got is None,
                                     "shape": [list(t.shape), list(got.shape) if got is not None else None],
                                     "max_abs": float((got.float() - t.float()).abs().max())
                                     if got is not None and got.shape == t.shape else None})
    # trained expert rows present in the export (fused gate_up / down of each layer), sampled
    rows = []
    for name, key in fused:
        served, old = cut.served(name)
        base = served.rsplit(".mlp.experts.", 1)[0]
        part = "down_proj" if name.endswith("down_proj") else "gate_proj"
        if key + "::experts" in shards[0]:
            present = sorted(Experts(key).mats)
        else:
            present = range(sum(s[key].shape[0] for s in shards) if shards[0][key].dim() else 0)
        for e in present:
            orig = cut.experts[old][e] if cut.experts else e
            if f"{base}.mlp.experts.{orig}.{part}.qweight" in export:
                rows.append((name, key, e))
    random.Random(seed).shuffle(rows)
    exp = {"present_rows": len(rows), "checked": 0, "code_share_min": None, "zeros_share_min": None,
           "scale_rel_max": None, "bad": []}
    shares, zshares, srel, seq = [], [], [], []
    for name, key, e in rows[:experts]:
        full = joined(key)
        for served, t in heal_delta.entries(name, full, cut, rows=[e]):
            got = export.get(served)
            seen.add(served)
            if got is None or got.shape != t.shape:
                exp["bad"].append({"name": served, "missing": got is None})
                continue
            if served.endswith(".qweight"):
                # 8 codes per int32: compare the codes themselves
                a = torch.stack([(t >> (4 * i)) & 15 for i in range(8)])
                b = torch.stack([(got >> (4 * i)) & 15 for i in range(8)])
                shares.append(float((a == b).float().mean()))
            elif served.endswith(".qzeros"):
                zshares.append(float((t == got).float().mean()))
            else:
                srel.append(float(((t.float() - got.float()).abs() / got.float().abs().clamp_min(1e-12)).max()))
                seq.append(float((t == got).float().mean()))
        exp["checked"] += 1
    if shares:
        exp.update(code_share_min=min(shares), zeros_share_min=min(zshares) if zshares else None,
                   scale_rel_max=max(srel) if srel else None, scale_equal_min=min(seq) if seq else None)
    unmatched = sorted(k for k in export if k not in seen and not any(
        k.endswith(s) for s in (".qweight", ".qzeros", ".scales")))
    # optimizer shards: finite, equal step counters
    opt = {"files": 0, "finite": True, "steps_equal": True}
    counters = []
    for r in range(world):
        st = torch.load(ckpt / f"optim-rank{r}.pt", map_location="cpu", weights_only=False)
        opt["files"] += 1
        for v in st["adam"]["state"].values():
            for k, x in v.items():
                x = x.to_local() if hasattr(x, "to_local") else x
                if isinstance(x, torch.Tensor) and x.is_floating_point() and not bool(torch.isfinite(x).all()):
                    opt["finite"] = False
        counters.append((st["rows"]["t"], sorted({str(float(v["step"])) for v in st["adam"]["state"].values()
                                                  if "step" in v})))
        del st
    opt["steps_equal"] = all(c == counters[0] for c in counters)
    opt["counters"] = [str(c) for c in counters[:1]]
    ok = (dense["checked"] > 0 and dense["equal"] == dense["checked"] and not unmatched and exp["checked"] > 0
          and not exp["bad"] and (exp["code_share_min"] or 0) >= 0.999 and (exp["zeros_share_min"] or 0) >= 0.999
          and (exp.get("scale_equal_min") or 0) >= 0.99 and (exp["scale_rel_max"] if exp["scale_rel_max"] is not None else 1) <= 2e-3
          and opt["finite"] and opt["steps_equal"])
    return {"ok": ok, "world": world, "dense": dense, "experts": exp, "unmatched_export": unmatched[:8],
            "unmatched_count": len(unmatched), "optimizer": opt}


@app.function(image=image, timeout=6 * 3600, retries=0, volumes={"/runs": runs},
              secrets=[modal.Secret.from_name("nebius-s3")])
def check(run_id: str, wait_hours: float = 5.0, experts: int = 48):
    import sys

    import torch

    sys.path.insert(0, "/opt/rc")
    import heal_delta

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    _rclone_env()
    out = Path("/runs/validation") / run_id
    train = out / "train"
    report = {"run_id": run_id, "started_epoch": time.time()}
    verdict_path = out / "reload-check.json"

    def done(**kw):
        report.update(kw, elapsed_seconds=round(time.time() - report["started_epoch"], 1))
        verdict_path.write_text(json.dumps(report, indent=1))
        runs.commit()
        print("RELOAD CHECK", "OK" if report.get("ok") else "FAILED", json.dumps(report)[:2000], flush=True)
        return report

    deadline = time.time() + wait_hours * 3600
    step = None
    while time.time() < deadline:
        runs.reload()
        cks = sorted((int(p.name[5:]), p) for p in train.glob("ckpt-*") if p.name[5:].isdigit()) if train.is_dir() else []
        if cks:
            step = cks[0][0]
            break
        time.sleep(20)
    if step is None:
        return done(ok=False, error="no checkpoint within the wait")
    report["step"], report["ckpt_seen_epoch"] = step, time.time()
    local = Path(f"/tmp/v{step}")
    src = f"arc:{NEBIUS_BUCKET}/runs/validation/{run_id}/deltas/v{step}"
    for _ in range(30):  # the version may still be uploading: the manifest is copied last
        subprocess.run(["rclone", "copy", src, str(local), "--transfers", "16"], capture_output=True, text=True)
        if (local / "manifest.json").is_file():
            break
        time.sleep(20)
    if not (local / "manifest.json").is_file():
        return done(ok=False, error=f"export v{step} not fetched from the bucket")
    report["export_fetched_epoch"] = time.time()
    ckpt = train / f"ckpt-{step}"
    while not (ckpt / "progress.json").is_file():
        if time.time() > deadline:
            return done(ok=False, error=f"ckpt-{step} never completed")
        time.sleep(20)
        runs.reload()
    from safetensors.torch import load_file

    export = {}
    for f in json.loads((local / "manifest.json").read_text())["files"]:
        export.update(load_file(str(local / f)))
    prog = json.loads((ckpt / "progress.json").read_text())
    student = Path(prog["student"])
    cut = heal_delta.Cut(student / "config.json", os.environ.get("HEAL_DELTA_PRUNE") or None)
    try:
        verdict = compare(ckpt, export, cut, experts=experts)
    except Exception as exc:  # noqa: BLE001 - a crash is a failed check, reported
        verdict = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:1000]}
    return done(**verdict)


test_image = (modal.Image.debian_slim(python_version="3.12")
              .pip_install("torch==2.14.0", "safetensors", "numpy", "pytest", "modal",
                           extra_index_url="https://download.pytorch.org/whl/cpu")
              .add_local_file(HERE / "heal_delta.py", "/opt/t/scripts/posttrain/heal_delta.py")
              .add_local_file(Path(__file__).resolve(), "/opt/t/scripts/posttrain/heal_reload_check.py")
              .add_local_file(TESTS / "test_heal_reload_check.py", "/opt/t/tests/test_heal_reload_check.py"))


@app.function(image=test_image, timeout=900)
def unit_tests():
    """``modal run scripts/posttrain/heal_reload_check.py::unit_tests`` (no local torch)."""
    r = subprocess.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "/opt/t/tests"], capture_output=True,
                       text=True)
    print(r.stdout[-3000:], r.stderr[-1500:])
    return r.returncode


@app.local_entrypoint()
def main(run_id: str, wait_hours: float = 5.0, experts: int = 48):
    call = check.spawn(run_id, wait_hours, experts)
    print("RELOAD CHECK CALL", call.object_id, run_id, flush=True)
