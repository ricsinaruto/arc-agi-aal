"""Item 6 (22 Oct): REAP masks + router refit vs the t37 excl5 routed-mass mask, offline on the excl5 capture (B200).

Per layer: full routing from the BF16 gate on the capture's calibration rows; scores (mass, REAP, EAN); masks keeping
320 of 512; routed-output error vs the full BF16 model on the held-out eval rows for: the served excl5 mask (file),
mass/REAP/EAN masks, and router refits (bias-only, full) on the served mask and the REAP mask. Check: the served mask's
routing on the calibration rows reproduces the capture's own ids/weights (the server's routing).
Outputs under /runs/throughput/<run id>/: layer-NN.json, masks-<kind>.json (arc-expert-prune-v1), refit-<mask>-L.pt.

    modal run scripts/dfr_exact/reap/modal_reap.py --run-id reap-pre-v1 --synthetic   # CPU, synthetic tensors
    modal run --detach scripts/dfr_exact/reap/modal_reap.py --run-id tp-reap-excl5-v1-20261022  # B200, 48 layers
"""
import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
MASK = Path("./.claude/worktrees/dfr-exact/data/expert_prune/prune-t37-int3-requant-excl5-v1.json")
CAPTURE = "/runs/validation/b2-tp-cap-v85p-x5-dense-v2-nebius-20261020/expert-capture"
BF16 = "/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540"
E, H, INTER, KEEP = 512, 2560, 640, 320
TIMEOUT = 5400

app = modal.App("arc-reap-fit")
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "safetensors==0.6.2", "numpy")
         .add_local_file(str(HERE / "reap_core.py"), "/opt/reap/reap_core.py")
         .add_local_file(str(MASK), "/opt/reap/mask.json"))


def body(run_id, gpu, layers, synthetic, calib_rows, eval_rows):
    import numpy as np
    import torch

    sys.path.insert(0, "/opt/reap")
    import reap_core as rc

    dev = "cuda" if gpu else "cpu"
    out = Path("/runs/throughput") / run_id
    out.mkdir(parents=True, exist_ok=True)
    mask = json.load(open("/opt/reap/mask.json"))
    served = {int(k): sorted(set(range(E)) - set(map(int, v))) for k, v in mask["pruned"].items()}
    log_f = open(out / "reap.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n"); log_f.flush()

    if synthetic:
        g = torch.Generator().manual_seed(0)
        Hs, Is, Es = 64, 16, 512

        def load(layer):
            gu = torch.randn(Es, 2 * Is, Hs, generator=g) * 0.1
            dn = torch.randn(Es, Hs, Is, generator=g) * 0.1
            W = torch.randn(Es, Hs, generator=g)
            xc = torch.randn(4000, Hs, generator=g).bfloat16()
            xe = torch.randn(2000, Hs, generator=g).bfloat16()
            return gu, dn, W, xc, xe, None
        inter = Is
    else:
        from safetensors import safe_open
        index = json.load(open(Path(BF16) / "model.safetensors.index.json"))["weight_map"]
        cmeta = json.load(open(Path(CAPTURE) / "calib/meta.json"))
        emeta = json.load(open(Path(CAPTURE) / "eval/meta.json"))
        inter = INTER

        def tensor(name):
            with safe_open(str(Path(BF16) / index[name]), framework="pt", device="cpu") as fh:
                return fh.get_tensor(name)

        def load(layer):
            p = f"model.language_model.layers.{layer}.mlp."
            gu, dn, W = (tensor(p + n).to(dev) for n in ("experts.gate_up_proj", "experts.down_proj", "gate.weight"))
            root = Path(CAPTURE) / "calib"
            rows = int(cmeta["rows"][str(layer)]); take = min(rows, calib_rows)
            q = np.fromfile(root / f"L{layer:02d}.x", dtype=np.uint8, count=take * H).reshape(take, H)
            s = np.fromfile(root / f"L{layer:02d}.s", dtype=np.float32, count=take)
            ids = np.fromfile(root / f"L{layer:02d}.ids", dtype=np.int16, count=take * 10).reshape(take, 10)
            w = np.fromfile(root / f"L{layer:02d}.w", dtype=np.float32, count=take * 10).reshape(take, 10)
            xc = (torch.from_numpy(q).to(dev).view(torch.float8_e4m3fn).float() * torch.from_numpy(s).to(dev)[:, None]).bfloat16()
            root = Path(CAPTURE) / "eval"
            rows = int(emeta["rows"][str(layer)]); take = min(rows, eval_rows)
            xe = torch.from_numpy(np.fromfile(root / f"L{layer:02d}.x", dtype=np.int16, count=take * H).reshape(take, H)
                                  ).view(torch.bfloat16).to(dev)
            return gu, dn, W, xc, xe, (torch.from_numpy(ids.astype(np.int64)).to(dev), torch.from_numpy(w).to(dev))

    todo = [int(v) for v in layers.split(",")] if layers else list(range(48))
    masks_out = {k: {} for k in ("mass", "reap", "ean")}
    for layer in todo:
        t0 = time.time()
        if (out / f"layer-{layer:02d}.json").exists():
            prev = json.loads((out / f"layer-{layer:02d}.json").read_text())
            for k in masks_out:
                masks_out[k][layer] = prev["pruned"][k]
            continue
        gu, dn, W, xc, xe, cap = load(layer)
        Es = W.shape[0]
        keep_n = KEEP if not synthetic else KEEP * Es // E
        kept_served = served[layer] if not synthetic else list(range(Es - keep_n, Es))
        rec = {"layer": layer, "calib_rows": xc.shape[0], "eval_rows": xe.shape[0]}
        if cap is not None:  # does our pruned routing reproduce the server's?
            ids_c, w_c = cap
            lg = rc.kept_logits(xc[:8192].float() @ W.float().t(), kept_served)
            ids_m, w_m = rc.route(lg)
            kept_t = torch.as_tensor(kept_served, device=dev)
            ids_srv = kept_t[ids_c[:8192]]
            same = (torch.sort(ids_m, 1).values == torch.sort(ids_srv, 1).values).all(1).float().mean()
            same_g = (torch.sort(ids_m, 1).values == torch.sort(ids_c[:8192], 1).values).all(1).float().mean()
            rec["routing_check"] = {"rows": 8192, "same_top10_set_rank_ids": float(same), "same_top10_set_global_ids": float(same_g),
                                    "weight_sum_err": float((w_c[:8192].sum(1) - 1).abs().max())}
        sc = rc.scores(xc, W, gu, dn, inter)
        kept = {"served": kept_served}
        for k in ("mass", "reap", "ean"):
            kept[k] = rc.keep_top(sc[k], keep_n).tolist()
            masks_out[k][layer] = sorted(set(range(Es)) - set(kept[k]))
        rec["overlap_with_served"] = {k: len(set(kept[k]) & set(kept_served)) for k in ("mass", "reap", "ean")}
        rec["pruned_mass_share"] = {k: float(1 - sc["mass"][kept[k]].sum() / sc["mass"].sum()) for k in kept}
        errs = rc.mask_error(xe, W, gu, dn, inter, kept)
        refits = {}
        for base in ("served", "reap"):
            for full in (False, True):
                name = f"{base}+{'full' if full else 'bias'}"
                dW, b, extra, hist = rc.refit(xc, W, gu, dn, inter, kept[base], full=full,
                                              steps=60 if synthetic else (300 if gpu else 30), log=lambda m: log(f"L{layer} {name} {m}"))
                refits[name] = (kept[base], extra)
                rec.setdefault("refit_hist", {})[name] = hist
                torch.save({"kept": kept[base], "dW": dW.cpu() if full else None, "b": b.cpu()},
                           out / f"refit-{name}-L{layer:02d}.pt")
        errs.update(rc.mask_error(xe, W, gu, dn, inter, refits))
        rec["eval_rel_err"] = errs
        rec["pruned"] = {k: masks_out[k][layer] for k in masks_out}
        rec["seconds"] = round(time.time() - t0, 1)
        (out / f"layer-{layer:02d}.json").write_text(json.dumps(rec))
        log(f"L{layer} " + json.dumps({k: round(v, 4) for k, v in errs.items()}) + f" check {rec.get('routing_check')} "
            f"{rec['seconds']} s")
        runs.commit()
    for k, pruned in masks_out.items():
        if len(pruned) == 48:
            (out / f"masks-{k}.json").write_text(json.dumps({
                "format": "arc-expert-prune-v1", "experts": E, "fraction": 0.375, "pruned_per_layer": E - KEEP,
                "pruned": {str(l): v for l, v in sorted(pruned.items())}, "rank_by": k,
                "source_run": run_id, "screen": "excl5 capture b2-tp-cap-v85p-x5-dense-v2 (five games outside c110)"}))
    runs.commit()
    return {"layers_done": len(list(out.glob("layer-*.json")))}


@app.function(image=image, gpu="B200", timeout=TIMEOUT, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/assets": assets, "/runs": runs})
def fit(run_id: str, layers: str = "", calib_rows: int = 32768, eval_rows: int = 16384):
    return body(run_id, True, layers, False, calib_rows, eval_rows)


@app.function(image=image, timeout=1800, retries=0, max_containers=1, volumes={"/assets": assets, "/runs": runs})
def cpu(run_id: str, layers: str, synthetic: bool):
    return body(run_id, False, layers, synthetic, 2048, 1024)


@app.local_entrypoint()
def main(run_id: str, synthetic: bool = False, cpu_real: bool = False, layers: str = ""):
    if synthetic or cpu_real:
        print(json.dumps(cpu.remote(run_id, layers or "0", synthetic)))
        return
    call = fit.spawn(run_id, layers)
    print(json.dumps({"run_id": run_id, "function_call_id": call.object_id, "start_epoch": round(time.time()),
                      "timeout": TIMEOUT}))
