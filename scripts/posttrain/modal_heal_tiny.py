"""Modal CPU job: heal_kd.py's tiny check (2 FSDP ranks on gloo, fp32 and BF16 compute) on Linux.

    ARC_VALIDATION_ROOT=. modal run scripts/posttrain/modal_heal_tiny.py --run-id <id>

Trainer MFU S1 (HEAL_CAPTURE_CACHE / HEAL_RECOMPUTE): ``--capture cpu`` (bars on gloo, plus the tiny and train-tiny
checks with the flag on), ``--capture gpu`` (bars on 2 L4, NCCL), ``--capture mem`` (recompute memory read, 1 L40S);
HEAL_BASE_LOCAL=<the pre-S1 heal_kd.py> adds the flag-off identity bar.

Trainer MFU S2 (HEAL_COMPILE=layers): ``--capture gpu-compile`` (the S1 bars compiled, 2 L40S) and the one-card
proxy, e.g. ``modal run --detach modal_heal_tiny.py --proxy-gpu L40S --proxy-env HEAL_CAPTURE_CACHE=1 --proxy-cases
off,off,layers --proxy-args "--proxy-layers 4 --proxy-experts 16 --proxy-mtp-experts 16 --window 2048 --proxy-seqs
5000,3000 --mtp-weight 0.1 --lr 1e-5 --proxy-grads --proxy-profile"``.
"""
import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
FILES = ("heal_kd.py", "heal_delta.py", "heal_stream.py", "flash_config.json", "flash_qlora_job.py", "rl1_trainer.py", "rl1_sparse.py", "rl_job.py", "fla_pin.py", "rl1_hyper.py", "rl1_hyper_test.py", "rl1_sparse2.py", "bench_sparse2.py")
BASE = os.environ.get("HEAL_BASE_LOCAL") if modal.is_local() else None
app = modal.App("sft-heal-tiny", include_source=False)


def with_files(image):
    for name in FILES:
        image = image.add_local_file(str(HERE / name), f"/opt/heal/{name}", copy=True)
    if BASE:
        image = image.add_local_file(BASE, "/opt/heal/heal_kd_base.py", copy=True)
    return image.add_local_file(__file__, "/root/modal_heal_tiny.py", copy=True)


image = with_files(modal.Image.debian_slim(python_version="3.12")
                   .pip_install("torch==2.14.0", "transformers==5.17.0", "safetensors", "pillow", "torchvision==0.29.0",
                                "accelerate", extra_index_url="https://download.pytorch.org/whl/cpu"))
# the trainer image's transformers (5.18.0: the axolotl image; 5.17 drops mm_encoder_outputs silently)
image_518 = with_files(modal.Image.debian_slim(python_version="3.12")
                       .pip_install("torch==2.14.0", "transformers==5.18.0", "safetensors", "pillow", "torchvision==0.29.0",
                                    "accelerate", extra_index_url="https://download.pytorch.org/whl/cpu"))
gpu_image = with_files(modal.Image.debian_slim(python_version="3.12")
                       .pip_install("torch==2.14.0", "transformers==5.17.0", "safetensors", "pillow", "torchvision==0.29.0",
                                    "accelerate"))


def _capture_run(nproc, extra_env, extra_args=(), python=""):
    import subprocess
    import sys

    out = f"/tmp/capture-{abs(hash((nproc, tuple(sorted(extra_env.items())), tuple(extra_args))))}"
    env = {**os.environ, "PYTHONPATH": "/opt/heal", **extra_env}
    if Path("/opt/heal/heal_kd_base.py").exists():
        env["HEAL_BASE_FILE"] = "/opt/heal/heal_kd_base.py"
    p = subprocess.run([python or sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(nproc),
                        "/opt/heal/heal_kd.py", "capture-check", "--out", out, "--k", "16", *extra_args],
                       capture_output=True, text=True, cwd="/opt/heal", env=env)
    ev = Path(out) / "events.jsonl"
    events = [json.loads(line) for line in open(ev)] if ev.exists() else []
    return {"returncode": p.returncode, "env": extra_env, "args": list(extra_args),
            "cases": [e for e in events if "case" in e], "check": next((e for e in events if e.get("check")), None),
            "stderr": _first_traceback(p.stderr) if p.returncode else ""}


def _first_traceback(text, lines=60):
    rows = text.splitlines()
    first = next((i for i, row in enumerate(rows) if "Traceback" in row), 0)
    rows = [row for row in rows[first:] if "/torch/nn/modules/module.py" not in row and "_call_impl" not in row]
    ranks = [row for row in text.splitlines() if row.startswith("[rank") and "Warning" not in row]  # a rank's own
    # message (a SystemExit prints no traceback)
    return "\n".join(ranks[-20:] + ["..."] + rows[:12] + ["..."] + rows[-lines:])


@app.function(image=image, timeout=3600)
def capture_cpu():
    """The S1 bars on CPU (gloo, deterministic): BF16 and fp32 compute, recompute full / alternate / none."""
    results = {"bf16": _capture_run(2, {}), "fp32": _capture_run(2, {}, ["--fp32"])}
    for policy in ("alternate", "none"):
        results[f"bf16-{policy}"] = _capture_run(2, {"HEAL_RECOMPUTE": policy})
    for tag, extra in (("bf16-chunk40", ["--chunk", "40"]), ("fp32-chunk40", ["--fp32", "--chunk", "40"])):
        results[tag] = _capture_run(2, {}, extra)  # extend chunks off the GDN chunk grid: a real numeric difference
    results["tiny_on"] = tiny.local(capture=True)
    results["train_tiny_on"] = train_tiny.local(capture=True)
    return results


@app.function(image=image, timeout=1800)
def capture_cpu_chunk():
    """The off-grid extend chunking only (BF16), with the fp32-compute reference."""
    return {"bf16-chunk40": _capture_run(2, {}, ["--chunk", "40"]),
            "bf16-none-chunk40": _capture_run(2, {"HEAL_RECOMPUTE": "none"}, ["--chunk", "40"])}


# the B300 trainer's own software (modal_rl1: Axolotl image + SonicMoE pins; its GDN kernels, if any)
AXOLOTL_IMAGE = "axolotlai/axolotl@sha256:af1ff77a35170b794d5fd3d01d97d69fe11ba9a098bca43eabe0849ff3d2b010"
trainer_image = modal.Image.from_registry(AXOLOTL_IMAGE, add_python="3.12").entrypoint([])
for _cmd in ("quack-kernels==0.6.1 nvidia-cutlass-dsl==4.6.0 apache-tvm-ffi>=0.1.10,<0.2", "--no-deps triton==3.6.0"):
    trainer_image = trainer_image.run_commands("uv pip install --python /workspace/axolotl-venv/bin/python "
                                               + " ".join(f"'{item}'" for item in _cmd.split()))
trainer_image = with_files(trainer_image)
TRAINER_PYTHON = "/workspace/axolotl-venv/bin/python"


@app.function(image=trainer_image, gpu="L40S:2", timeout=2400)
def capture_gpu():
    """The S1 bars on 2 L40S in the trainer's software (NCCL, FSDP2, BF16 compute, Triton selected attention, fla)."""
    return {"bf16": _capture_run(2, {}, python=TRAINER_PYTHON),
            "bf16-chunk40": _capture_run(2, {}, ["--chunk", "40"], python=TRAINER_PYTHON),
            "bf16-none": _capture_run(2, {"HEAL_RECOMPUTE": "none"}, python=TRAINER_PYTHON)}


@app.function(image=trainer_image, gpu="L40S:2", timeout=2400)
def capture_gpu_compile():
    """The S1 capture bars with HEAL_COMPILE=layers (trainer MFU S2): every case compiled, 2 L40S, NCCL, FSDP2."""
    env = {"HEAL_COMPILE": "layers"}
    return {"bf16": _capture_run(2, env, python=TRAINER_PYTHON),
            "bf16-chunk40": _capture_run(2, env, ["--chunk", "40"], python=TRAINER_PYTHON)}


assets = modal.Volume.from_name("arc-prolong-assets")


@app.function(image=trainer_image, gpu="L40S", timeout=7200, volumes={"/assets": assets})
def proxy(args: str, env: str = "", cases: str = "off,layers", session: int = 0):
    """heal_kd.py proxy (trainer MFU S2) on one card, one process per case of ``cases`` (HEAL_COMPILE modes; case 0
    writes its first-step gradients, later cases compare), ``session`` random-length sequences after the last case;
    ``args`` the proxy arguments, ``env`` "K=V,K=V" (e.g. HEAL_CAPTURE_CACHE=1). Events stream to the log; returns
    every event, each case's stderr tail and the comparison against case 0."""
    import subprocess
    import threading

    extra = dict(pair.split("=", 1) for pair in env.split(",") if pair)
    run_env = {**os.environ, "PYTHONPATH": "/opt/heal", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **extra}
    modes = cases.split(",")
    events, errors = [], {}
    for i, mode in enumerate(modes):
        out = f"/tmp/proxy/case{i}"
        tail = ["--proxy-session", str(session)] if session and i == len(modes) - 1 else []
        p = subprocess.Popen([TRAINER_PYTHON, "-m", "torch.distributed.run", "--nproc-per-node", "1",
                              "/opt/heal/heal_kd.py", "proxy", "--out", out, *args.split(), "--proxy-cases", mode,
                              "--proxy-case-index", str(i), "--proxy-ref", "/tmp/proxy/ref", *tail],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd="/opt/heal", env=run_env)
        err = []
        reader = threading.Thread(target=lambda: err.extend(p.stderr), daemon=True)
        reader.start()
        for line in p.stdout:
            print(line[:3000], end="", flush=True)
        p.wait()
        reader.join(timeout=10)
        ev = Path(out) / "events.jsonl"
        events += [json.loads(line) for line in open(ev)] if ev.exists() else []
        text = "".join(err)
        rec = [ln.strip()[:400] for ln in text.splitlines() if "ecompil" in ln or "triggered by" in ln or "- " in ln[:12]]
        errors[f"{i}:{mode}"] = {"returncode": p.returncode, "recompiles": rec[:80],
                                 "stderr": _first_traceback(text) if p.returncode else text[-1500:]}
    done = [e for e in events if e.get("event") == "proxy_case"]
    verdict = {}
    if done and done[0]["case"] == 0:
        base = done[0]
        rel = lambda a, b: abs(a - b) / max(abs(b), 1e-12)  # noqa: E731
        for case in done[1:]:
            w = [max(rel(x["loss"], y["loss"]), rel(x["kd"], y["kd"]))
                 for x, y in zip(case.get("windows", []), base.get("windows", []))]
            verdict[f"{case['case']}:{case['mode']}"] = {
                "window_loss_max_rel": max(w) if w else None,
                "step_kd_rel": [rel(x["kd"], y["kd"]) for x, y in zip(case["steps"], base["steps"])],
                "grad": case.get("grad_vs_case0"),
                "speedup": round(base["step_s"] / case["step_s"], 3) if case.get("step_s") and base.get("step_s")
                else None}
    return {"events": events, "errors": errors, "verdict": verdict}


CONV_CHECK = r"""
import json, os, sys, time
import torch
sys.argv = ['heal_kd.py']
import heal_kd
from transformers.models.qwen4_exp import modeling_qwen4_exp as m
heal_kd.install_fla_conv()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cpu':  # TRITON_INTERPRET: no driver to benchmark autotune configs; keep one each
    import fla.modules.convolution  # noqa: F401
    import fla.modules.conv.triton.ops  # noqa: F401
    for name, mod in list(sys.modules.items()):
        if not name.startswith('fla.modules.conv'):
            continue
        for obj in list(vars(mod).values()):
            at = obj
            while at is not None and not hasattr(at, 'configs') and hasattr(at, 'fn'):
                at = at.fn
            if at is not None and isinstance(getattr(at, 'configs', None), list) and len(at.configs) > 1:
                at.configs = at.configs[:1]
shapes = [(8192, 32768), (8192, 32771)] if dev == 'cuda' else [(32, 70), (32, 73)]
res = []
for dt in ([torch.bfloat16] if dev == 'cuda' else [torch.float32, torch.bfloat16]):
    for C, T in shapes:
        g = torch.Generator(device=dev).manual_seed(0)
        x = torch.randn(1, T, C, device=dev, dtype=dt, generator=g)
        w = (torch.randn(C, 4, device=dev, generator=g) * 0.5).to(dt)
        gy = torch.randn(1, T, C, device=dev, dtype=dt, generator=g)
        outs, times = [], []
        for on in (False, True):
            heal_kd.FLA_CONV['on'] = on
            for rep in range(6 if dev == 'cuda' else 1):
                xi, wi = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
                if dev == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.time()
                y = m.causal_conv1d_fn(xi.transpose(1, 2), wi, None, activation='silu')
                y.transpose(1, 2).backward(gy)
                if dev == 'cuda':
                    torch.cuda.synchronize()
                if rep:
                    times.append(time.time() - t0)
            outs.append((y.detach().float(), xi.grad.float(), wi.grad.float()))
            times.append(None)
        row = {'dtype': str(dt), 'C': C, 'T': T, 'calls': heal_kd.FLA_CONV['calls'], 'fallback': heal_kd.FLA_CONV['fallback']}
        for name, a, b in zip(('y', 'dx', 'dw'), outs[0], outs[1]):
            row[name] = {'equal_frac': float((a == b).float().mean()), 'max_abs': float((a - b).abs().max()),
                         'rel': float((a - b).norm() / a.norm().clamp(min=1e-30))}
        ts = [t for t in times if t is not None]
        k = len(ts) // 2
        if ts:
            row['ms_off'], row['ms_on'] = round(1e3 * min(ts[:k]), 3), round(1e3 * min(ts[k:]), 3)
        res.append(row)
print(json.dumps(res))
"""


GPU_CHECK = r"""
import json, sys, time, torch
sys.path.insert(0, "/opt/heal")
import heal_delta
torch.manual_seed(0)
row = {"check": "int4_rows_cuda", "gpu": torch.cuda.get_device_name()}
for shape in ((15, 1280, 2560), (15, 2560, 640), (40, 1280, 2560)):
    ws = torch.randn(*shape, device="cuda") * 0.02
    got = heal_delta.int4_rtn_rows(ws, search=20)
    row[str(shape)] = sum(int(not torch.equal(a[e], b)) for e in range(shape[0])
                          for a, b in zip(got, heal_delta.int4_rtn(ws[e], search=20)))
ws = torch.randn(15, 1280, 2560, device="cuda") * 0.02
torch.cuda.synchronize(); t = time.time()
[[x.cpu() for x in heal_delta.int4_rtn(ws[e], search=20)] for e in range(15)]
torch.cuda.synchronize(); row["per_row_s"] = round(time.time() - t, 3); t = time.time()
[x.cpu() for x in heal_delta.int4_rtn_rows(ws, search=20)]
torch.cuda.synchronize(); row["batched_s"] = round(time.time() - t, 3)
row["ok"] = all(v == 0 for k, v in row.items() if k.startswith("("))
print(json.dumps(row))
"""


@app.function(image=trainer_image, timeout=900)
def import_check():
    """CPU: the trainer image imports every shipped module and parses heal_kd's proxy arguments (launch gate)."""
    import subprocess

    code = ("import sys; sys.argv=['heal_kd.py']; import heal_kd, heal_delta, heal_stream, rl1_hyper, rl1_trainer, rl1_sparse; "
            "print('imports ok', heal_kd.__file__)")
    p = subprocess.run([TRAINER_PYTHON, "-c", code], capture_output=True, text=True, cwd="/opt/heal",
                       env={**os.environ, "PYTHONPATH": "/opt/heal"})
    q = subprocess.run([TRAINER_PYTHON, "/opt/heal/bench_sparse2.py", "--mode", "test"], capture_output=True, text=True,
                       cwd="/opt/heal", env={**os.environ, "PYTHONPATH": "/opt/heal", "TRITON_INTERPRET": "1"})
    c = subprocess.run([TRAINER_PYTHON, "-c", CONV_CHECK], capture_output=True, text=True, cwd="/opt/heal",
                       env={**os.environ, "PYTHONPATH": "/opt/heal", "TRITON_INTERPRET": "1"})
    return {"rc": p.returncode, "out": p.stdout[-2000:], "err": p.stderr[-3000:],
            "union_interp": {"rc": q.returncode, "out": q.stdout[-4000:], "err": q.stderr[-2000:] if q.returncode else ""},
            "conv_interp": {"rc": c.returncode, "out": c.stdout[-4000:], "err": c.stderr[-2500:] if c.returncode else ""}}


@app.function(image=trainer_image, gpu="L40S", timeout=1800)
def gpu_checks(which: str = "delta,hyper"):
    """MFU-1 lever checks on the trainer's software: HEAL_DELTA_FAST's batched int4 vs per-row (bitwise), rl1_hyper's
    trainable-weight path vs the model's code (exacttrain, bitwise), rl1_sparse2 vs rl1_sparse vs the fp32 reference
    and speed (bench_sparse2 bench)."""
    import subprocess

    out = []
    cmds = {"delta": [[TRAINER_PYTHON, "-c", GPU_CHECK]],
            "hyper": [[TRAINER_PYTHON, "/opt/heal/rl1_hyper_test.py", "exacttrain", n] for n in ("4096", "1000")],
            "union": [[TRAINER_PYTHON, "/opt/heal/bench_sparse2.py", "--mode", "bench"]],
            "conv": [[TRAINER_PYTHON, "-c", CONV_CHECK]]}
    for cmd in [c for w in which.split(",") for c in cmds[w]]:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd="/opt/heal", env={**os.environ, "PYTHONPATH": "/opt/heal"})
        out.append({"cmd": cmd[1:2] + cmd[2:], "rc": p.returncode, "out": p.stdout[-8000:],
                    "err": p.stderr[-1500:] if p.returncode else ""})
        print(json.dumps(out[-1])[:3000], flush=True)
    return out


@app.function(image=trainer_image, gpu="L40S", timeout=3600)
def recompute_mem_trainer(depths: str = "4 8", windows: str = "1024 2048 3072"):
    return recompute_mem.local(depths, windows, python=TRAINER_PYTHON)


@app.function(image=gpu_image, gpu="L40S", timeout=3600)
def recompute_mem(depths: str = "4 8", windows: str = "1024 2048 4096", python: str = ""):
    import subprocess
    import sys

    python = python or sys.executable
    kernels = subprocess.run([python, "-c", "import importlib.util as u, torch, transformers; print({m: bool(u.find_spec(m)) "
                              "for m in ('fla', 'causal_conv1d', 'kernels')}, torch.__version__, transformers.__version__)"],
                             capture_output=True, text=True).stdout.strip()
    p = subprocess.run([python, "/opt/heal/heal_kd.py", "recompute-mem", "--out", "/tmp/mem", "--k", "32",
                        "--depths", *depths.split(), "--mem-windows", *windows.split()], capture_output=True, text=True,
                       cwd="/opt/heal",
                       env={**os.environ, "PYTHONPATH": "/opt/heal", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    ev = Path("/tmp/mem/events.jsonl")
    return {"returncode": p.returncode, "rows": [json.loads(line) for line in open(ev)] if ev.exists() else [],
            "stderr": "\n".join(p.stderr.splitlines()[-40:]) if p.returncode else "",
            "kernels": kernels, "stdout_tail": "\n".join(p.stdout.splitlines()[-5:])}


@app.function(image=image, timeout=1800)
def tiny(capture: bool = False):
    import subprocess
    import sys

    results = {}
    env = {**os.environ, "HEAL_CAPTURE_CACHE": "1" if capture else "0"}
    for tag, extra in (("fp32", ["--fp32"]), ("bf16", [])):
        out = f"/tmp/tiny-{tag}"
        p = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "2",
                            "/opt/heal/heal_kd.py", "tiny", "--out", out, "--steps", "12", *extra],
                           capture_output=True, text=True, env=env)
        events = [json.loads(line) for line in open(f"{out}/events.jsonl")] if Path(f"{out}/events.jsonl").exists() else []
        results[tag] = {"returncode": p.returncode, "checks": [e for e in events if "check" in e],
                        "stderr": p.stderr[-1500:] if p.returncode else ""}
    return results


@app.function(image=image, timeout=1800)
def train_tiny(capture: bool = False):
    """heal_kd.py train-tiny: run_train end to end on the tiny model (2 gloo ranks), phase 1 then resume (phase 2)."""
    import shutil
    import subprocess
    import sys

    import torch

    results = {}
    env = {**os.environ, "HEAL_CAPTURE_CACHE": "1" if capture else "0"}
    shutil.rmtree("/tmp/tt", ignore_errors=True)
    for phase, procs in ((1, 2), (2, 2), (3, 1), (3, 2), (3, 4), (4, 2), (5, 2)):  # dp4: the 4xB200 bench
        p = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(procs),
                            "/opt/heal/heal_kd.py", "train-tiny", "--out", "/tmp/tt", "--tiny-phase", str(phase),
                            "--k", "16", "--lr", "1e-3", "--esft", "0.5"],
                           capture_output=True, text=True, cwd="/opt/heal", env=env)
        name = {1: "round1", 2: "round2", 3: f"dp{procs}", 4: "ckA", 5: "ckB"}[phase]
        ev = Path(f"/tmp/tt/{name}/events.jsonl")
        results[f"{name}"] = {"returncode": p.returncode,
                                    "events": [json.loads(line) for line in open(ev)] if ev.exists() else [],
                                    "stderr": "\n".join([l for l in p.stderr.splitlines() if l.startswith("[rank0]")][-30:])
                                    if p.returncode else ""}
        if p.returncode:
            break
    for w in (2, 4):
        if not (Path("/tmp/tt/grads-w1.pt").exists() and Path(f"/tmp/tt/grads-w{w}.pt").exists()):
            continue
        g1, g2 = torch.load("/tmp/tt/grads-w1.pt"), torch.load(f"/tmp/tt/grads-w{w}.pt")
        worst = max((float((g1[k] - g2[k]).abs().max()) / max(float(g1[k].abs().max()), 1e-12), k) for k in g1)
        n1 = sum(float(v.pow(2).sum()) for v in g1.values()) ** 0.5
        n2 = sum(float(v.pow(2).sum()) for v in g2.values()) ** 0.5
        results[f"dp_equality_w{w}"] = {"tensors": len(g1), "same_names": set(g1) == set(g2), "norm_w1": n1, "norm_w2": n2,
                                  "worst_rel_diff": worst[0], "worst_tensor": worst[1]}
    a = next((e for e in results.get("ckA", {}).get("events", []) if e.get("event") == "step" and e["step"] == 3), None)
    b = next((e for e in results.get("ckB", {}).get("events", []) if e.get("event") == "step" and e["step"] == 3), None)
    if a and b:
        results["exact_resume"] = {k: [a[k], b[k]] for k in ("kd", "kd_reply", "grad_norm", "trained", "lr")}
        results["exact_resume"]["ok"] = all(abs(a[k] - b[k]) <= 1e-6 * max(1.0, abs(a[k]))
                                            for k in ("kd", "kd_reply", "grad_norm", "trained", "lr"))
        results["exact_resume"]["checkpoint"] = Path("/tmp/tt/ckA/ckpt-2/progress.json").exists()
    student = Path("/tmp/tt/round2/student")
    results["saved"] = sorted(x.name for x in student.iterdir())[:12] if student.exists() else []
    return results


@app.function(image=image.add_local_file(str(HERE / "heal_reshard.py"), "/opt/heal/heal_reshard.py", copy=True),
              timeout=3600)
def reshard_tiny(async_ckpt: bool = False):
    """heal_reshard.py end to end (22 Oct): A (2 ranks) checkpoints after step 2 and finishes step 3; B2 resumes the
    2-rank checkpoint, B4 resumes it resharded to 4 ranks; each redoes step 3 and saves. B4's step 3 (KD, grad norm)
    and saved student must match A's (B2: the same-world reference)."""
    import shutil
    import sys

    import torch

    sys.path.insert(0, "/opt/heal")
    import heal_reshard

    root = "/tmp/rs"
    shutil.rmtree(root, ignore_errors=True)
    out = {}
    a_env = {"HEAL_ASYNC_CKPT": "1"} if async_ckpt else None  # A writes its checkpoint asynchronously
    for name, phase, procs, tag, env in (("round1", 1, 2, "", None), ("ckA", 4, 2, "", a_env),
                                         ("ckB", 5, 2, "", {"HEAL_TINY_SAVE": "1"})):
        rc, err = _tt(phase, procs, root, tag=tag, env=env)
        if rc:
            return {name: err}
    experts = json.loads(Path(f"{root}/tiny-student/config.json").read_text())["text_config"]["num_experts"]
    out["async_ckpt"] = {"on": async_ckpt, "files": sorted(x.name for x in Path(f"{root}/ckA/ckpt-2").iterdir()),
                         "event": next((e for e in _events(f"{root}/ckA/events.jsonl") if e.get("event") == "checkpoint"), None)}
    out["reshard"] = heal_reshard.reshard(f"{root}/ckA/ckpt-2", f"{root}/ckAr4/ckpt-2", 4, experts)
    rc, err = _tt(5, 4, root, tag="r4", env={"HEAL_TINY_SAVE": "1"})
    if rc:
        return {**out, "ckBr4": err}
    step3 = {n: next((e for e in _events(f"{root}/{n}/events.jsonl") if e.get("event") == "step" and e["step"] == 3), {})
             for n in ("ckA", "ckB", "ckBr4")}
    out["step3"] = {n: {k: e.get(k) for k in ("kd", "kd_reply", "grad_norm", "lr")} for n, e in step3.items()}

    def student(n):
        from safetensors.torch import load_file
        d = {}
        for f in sorted(Path(f"{root}/{n}/student").glob("*.safetensors")):
            d.update(load_file(str(f)))
        return {k: v.float() for k, v in d.items()}

    # sensitivity: the same resharded checkpoint with its moments zeroed must NOT match (else the check is blind)
    shutil.copytree(f"{root}/ckAr4", f"{root}/ckAr4z")
    for f in Path(f"{root}/ckAr4z/ckpt-2").glob("optim-rank*.pt"):
        st = torch.load(f, weights_only=False)
        for s_ in st["adam"]["state"].values():
            for k in ("exp_avg", "exp_avg_sq"):
                s_[k] = torch.zeros_like(s_[k])
        st["rows"]["m"] = [torch.zeros_like(x) for x in st["rows"]["m"]]
        st["rows"]["v"] = [torch.zeros_like(x) for x in st["rows"]["v"]]
        torch.save(st, f)
    rc, err = _tt(5, 4, root, tag="r4z", env={"HEAL_TINY_SAVE": "1"})
    a = student("ckA")
    out["b2_vs_a"], out["b4_vs_a"] = _cmp_tensors(student("ckB"), a), _cmp_tensors(student("ckBr4"), a)
    out["b4zero_vs_a"] = _cmp_tensors(student("ckBr4z"), a) if not rc else err
    rel = lambda x, y: abs(x - y) / max(abs(x), abs(y), 1e-12)  # noqa: E731
    ea, e4 = step3["ckA"], step3["ckBr4"]
    out["ok"] = bool(ea and e4 and all(rel(ea[k], e4[k]) <= 1e-4 for k in ("kd", "kd_reply", "grad_norm"))
                     and out["b4_vs_a"]["same_names"] and out["b4_vs_a"]["max_rel"] <= 1e-4
                     and isinstance(out["b4zero_vs_a"], dict) and out["b4zero_vs_a"]["max_rel"] > 1e-3)
    return out


@app.function(image=image, timeout=3600)
def stream_tiny():
    """OPD stream end to end (22 Oct): phase 1 (data, round 1 routing), then phase 7 at 2 ranks: chunks fed while
    training (row0 chunks, engine b, versions), 3 steps, a weight delta per step. Checks: 3 steps with ages, the
    sampled term on the engine b, 3 delta versions with manifests and served names."""
    import shutil

    root = "/tmp/st"
    shutil.rmtree(root, ignore_errors=True)
    rc, err = _tt(1, 2, root)
    if rc:
        return {"round1": err}
    rc, err = _tt(7, 2, root)
    if rc:
        return {"stream": err}
    ev = _events(f"{root}/st/events.jsonl")
    steps = [e for e in ev if e.get("event") == "step"]
    deltas = sorted(Path(f"{root}/deltas").glob("v*/manifest.json"))
    mans = [json.loads(p.read_text()) for p in deltas]
    names = set()
    from safetensors import safe_open
    for p in sorted(Path(f"{root}/deltas").glob("v3/*.safetensors")):
        with safe_open(str(p), framework="pt") as f:
            names |= set(f.keys())
    out = {"steps": [{k: e.get(k) for k in ("step", "kd_reply", "age_median", "age_max", "stream_chunks",
                                            "opd_sampled_rev", "opd_sampled_positions")} for e in steps],
           "deltas": [{k: m[k] for k in ("version", "tensors", "bytes")} for m in mans],
           "delta_events": [e for e in ev if e.get("event") in ("delta_export", "delta_written")][:6],
           "served_names_sample": sorted(names)[:6], "layer4_absent": not any(".layers.4." in n for n in names)}
    out["ok"] = (len(steps) == 3 and [s.get("age_median") for s in steps] == [0, 1, 2]
                 and all((s.get("opd_sampled_positions") or 0) > 0 for s in steps)
                 and [m["version"] for m in mans] == [1, 2, 3] and out["layer4_absent"]
                 and any(".mlp.experts." in n and n.endswith(".qweight") for n in names))
    return out


@app.function(image=image_518, timeout=1800)
def stream_teacher_tiny():
    """Streaming teacher (heal_stream + rl1_trainer.stream_forward) on the tiny Qwen4-Exp teacher, CPU fp32: two
    sessions of growing rollout files with images and a drain; every chain's streamed pieces must equal one fresh
    pass (the hybrid cache carried across pieces), with and without cache eviction."""
    import sys
    import tempfile

    import torch

    sys.path.insert(0, "/opt/heal")
    import heal_kd
    import heal_stream
    import rl1_trainer

    model = heal_kd.tiny_teacher("/opt/heal/flash_config.json", vocab_size=248320).float().eval()  # image ids
    vc = model.config.vision_config
    image_id = model.config.image_token_id
    patch = vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2
    grid = (1, 4, 4)  # 16 patches, merge 2: 4 tokens
    per_image = 16 // vc.spatial_merge_size ** 2

    def encode(capture):
        ids, pix, grids = [], [], []
        text = [m["content"] for m in capture["payload"]["messages"]]
        start = None
        for i, part in enumerate(text + [capture["reply"]]):
            if i == len(text):
                start = len(ids)
            for tok in part.split():
                if tok.startswith("IMG"):
                    g = torch.Generator().manual_seed(int(tok[3:]))
                    pix.append(torch.randn(16, patch, generator=g))
                    grids.append(grid)
                    ids += [image_id] * per_image
                else:
                    ids.append(int(tok))
        t = torch.tensor([ids])
        inputs = {"input_ids": t, "mm_token_type_ids": (t == image_id).long()}
        if pix:
            inputs["pixel_values"] = torch.cat(pix)
            inputs["image_grid_thw"] = torch.tensor(grids)
        return inputs, start

    def rows_of(rng, n_requests, drain_at, salt):
        rows, messages, last = [], [], None
        for n in range(n_requests):
            user = " ".join(str(int(x)) for x in rng.integers(10, 1500, int(rng.integers(5, 40))))
            if n % 2:
                user += f" 5 IMG{salt * 100 + n} 6"
            if n == drain_at or last is None:
                keep, add = 0, [{"role": "system", "content": "1 2 3"}, {"role": "user", "content": user}]
                messages = add
            else:
                keep, add = len(messages), [{"role": "assistant", "content": last}, {"role": "user", "content": user}]
                messages = messages + add
            reply_ids = [int(x) for x in rng.integers(10, 1500, int(rng.integers(3, 30)))]
            last = " ".join(map(str, reply_ids))
            rows.append({"n": n, "keep": keep, "messages": add, "reply": last, "gameplay": True, "template": {},
                         "usage": {"prompt_tokens": 1, "completion_tokens": len(reply_ids)}, "tokens": reply_ids,
                         "logprobs": [-0.1] * len(reply_ids), "version": n // 3})
        return rows

    import numpy as np

    import flash_qlora_job as fq

    out = {}
    # image path: the model's own multimodal forward = range_piece's = positions_and_images'; images matter
    ids = [11, 12, model.config.vision_start_token_id] + [image_id] * per_image + [model.config.vision_end_token_id, 13, 14]
    t1 = torch.tensor([ids])
    g = torch.Generator().manual_seed(0)
    one = {"input_ids": t1, "mm_token_type_ids": (t1 == image_id).long(), "pixel_values": torch.randn(16, patch, generator=g),
           "image_grid_thw": torch.tensor([grid])}
    with torch.no_grad():
        native = model(**one).logits[0]
        n1 = t1.shape[1]
        mine = model(input_ids=t1, attention_mask=torch.ones_like(t1),
                     **rl1_trainer.range_piece(model.model, one, 0, n1, image_id, device="cpu")(0, n1)).logits[0]
        old = model(input_ids=t1, attention_mask=torch.ones_like(t1), **fq.positions_and_images(model, one)(0, n1)).logits[0]
        other = model(**dict(one, pixel_values=one["pixel_values"] * 3 + 1)).logits[0]
    diff = lambda a, b: float((a - b).abs().max())  # noqa: E731
    out["image_path"] = {"native_vs_range_piece": diff(native, mine), "native_vs_positions_and_images": diff(native, old),
                         "native_vs_other_pixels": diff(native, other)}
    for evict in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "roll"
            files = {}
            for g, (seed, n, drain) in {"g1/a0": (1, 10, 6), "g2/a0": (2, 8, None)}.items():
                (root / g).mkdir(parents=True)
                files[g] = rows_of(np.random.default_rng(seed), n, drain, seed)
            fwd = rl1_trainer.stream_forward(model, 16, image_id, device="cpu", autocast=False)
            t = heal_stream.StreamTeacher(root, Path(tmp) / "out", encode, fwd, k=16, chunk=48, image_token_id=image_id,
                                          idle=1e9, encoders=2,
                                          budget=heal_stream.lru_budget(lambda: True) if evict else None)

            seen = {}

            def pump():
                for _ in range(500):
                    t._scan()
                    while not t.events.empty():
                        t._handle(t.events.get_nowait())
                        seen.update({id(c): c for c in t.chains.values()})
                    chain, upto = t._due()
                    if chain is None:
                        return
                    t.score(chain, upto)
                    if t.budget is not None:
                        t.stats["evictions"] += t.budget([c for c in t.chains.values() if c is not chain])

            for half in (0, 1):
                for g, rows in files.items():
                    cut = len(rows) // 2
                    with open(root / g / "rollout.jsonl", "a") as f:
                        for row in (rows[:cut] if half == 0 else rows[cut:]):
                            f.write(json.dumps(row) + "\n")
                pump()
            for key in list(t.chains):
                t._handle(("idle", key, None, None))
            pump()
            checks = [t.verify(c) for c in seen.values() if len(c.files) >= 2]  # every chain with 2+ pieces
            images = sum(int((torch.load(f, weights_only=False)["ids"] == image_id).sum())
                         for f in (Path(tmp) / "out").glob("s-*.pt"))
            out["evict" if evict else "keep"] = {"stats": t.status(), "verify": checks, "image_tokens": images}
            if not evict:  # negative control: other pixels in the chains must show in the verify
                for c in seen.values():
                    if c.inputs.get("pixel_values") is not None:
                        c.inputs = dict(c.inputs, pixel_values=c.inputs["pixel_values"] * 3 + 1)
                out["control"] = max(t.verify(c)["top_logp_max_abs"] for c in seen.values() if len(c.files) >= 2)
    runs = [out["keep"], out["evict"]]
    worst = max((c["top_logp_max_abs"] for v in runs for c in v["verify"]), default=None)
    out["ok"] = (worst is not None and worst < 1e-3 and all(c["top1_agree"] == 1.0 for v in runs for c in v["verify"])
                 and out["evict"]["stats"]["recompute_tokens"] > 0 and out["keep"]["stats"]["breaks"] == 1
                 and out["keep"]["image_tokens"] > 0 and out["control"] > 1e-3
                 and out["image_path"]["native_vs_range_piece"] < 1e-4 and out["image_path"]["native_vs_other_pixels"] > 1e-3)
    out["worst_top_logp_max_abs"] = worst
    return out


def queue_row(run, state, *, what="", gpu="", dur=1800):
    """The app's live row in docs/modal-queue.json (every app has a described row with its GPU
    type and expected end), staged at start and closed at the end through ops/jedit.py; a failed write stops the
    launch."""
    import os
    import subprocess

    repo, ops = ".", "~/arc-sft-data/ops"
    env = {**os.environ, "RUN": run, "STATE": state, "WHAT": what, "GPU": gpu, "DUR": str(dur),
           "APP": getattr(app, "app_id", None) or ""}
    # jedit stages the row (index and working tree); the dashboard's next queue commit carries it, no merge here
    done = subprocess.run(["python3", f"{ops}/jedit.py", "docs/modal-queue.json", f"{ops}/edits/tinyrow.py"], cwd=repo,
                          env=env, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"queue row for {run} not written: {done.stderr[-400:]}")


COMM_CONFIGS = {  # trainer MFU S3: HEAL_* comm flags per case (HEAL_COUNT_COLLECTIVES on in every case)
    "off": {}, "off2": {},
    "step": {"HEAL_GRAD_SYNC": "step"},
    "step_k2": {"HEAL_GRAD_SYNC": "step", "HEAL_SYNC_EVERY": "2"},
    "rows": {"HEAL_EXPERT_ROWS": "1"},
    "step_rows": {"HEAL_GRAD_SYNC": "step", "HEAL_EXPERT_ROWS": "1"},
    "step_rows_pf": {"HEAL_GRAD_SYNC": "step", "HEAL_EXPERT_ROWS": "1", "HEAL_FSDP_PREFETCH": "1", "HEAL_RS_BUFFERS": "2"},
    "bf16": {"HEAL_REDUCE_DTYPE": "bf16"},
    "rows_bf16": {"HEAL_EXPERT_ROWS": "1", "HEAL_REDUCE_DTYPE": "bf16"},
    "step_rows_bf16": {"HEAL_GRAD_SYNC": "step", "HEAL_EXPERT_ROWS": "1", "HEAL_REDUCE_DTYPE": "bf16"},
}
RESUME_CONFIGS = ("step", "step_rows", "step_rows_pf")  # exact mid-round resume and dp1 == dp2 under the flags


def _tt(phase, procs, root, tag="", env=None, script="/opt/heal/heal_kd.py", python="", extra=()):
    import subprocess
    import sys

    p = subprocess.run([python or sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(procs), script,
                        "train-tiny", "--out", root, "--tiny-phase", str(phase), "--k", "16", "--lr", "1e-3",
                        "--esft", "0.5", *(["--tag", tag] if tag else []), *extra],
                       capture_output=True, text=True, cwd="/opt/heal",
                       env={**os.environ, "PYTHONPATH": "/opt/heal", **(env or {})})
    text = p.stdout + "\n" + p.stderr
    cut = text.find("torch/distributed/run.py")  # the launcher's own traceback: the children's output is before it
    rows = [r for r in (text[:cut] if cut > 0 else text).splitlines() if "arning" not in r and r.strip()]
    tail = "\n".join(rows[-60:])
    return p.returncode, (_first_traceback(p.stderr) + "\n--- last lines ---\n" + tail) if p.returncode else ""


def _events(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []


def _cmp_tensors(a, b):
    """a vs reference b (dicts of tensors): cosine, norm rel diff, worst per-tensor max-abs diff over the
    reference's max-abs."""
    names = sorted(set(a) & set(b))
    dot = sum(float((a[k].double() * b[k].double()).sum()) for k in names)
    na = sum(float(a[k].double().pow(2).sum()) for k in names) ** 0.5
    nb = sum(float(b[k].double().pow(2).sum()) for k in names) ** 0.5
    worst, where = 0.0, None
    for k in names:
        scale = float(b[k].abs().max())
        if scale == 0:
            continue
        r = float((a[k] - b[k]).abs().max()) / scale
        if r > worst:
            worst, where = r, k
    return {"cosine": dot / max(na * nb, 1e-300), "norm_rel": abs(na - nb) / max(nb, 1e-300), "max_rel": worst,
            "worst_tensor": where, "same_names": set(a) == set(b), "tensors": len(names)}


def _comm_run(root, python, base_root=None):
    import shutil

    import torch

    shutil.rmtree(root, ignore_errors=True)
    out = {"torch": str(torch.__version__)}
    rc, err = _tt(1, 2, root, python=python)
    if rc:
        return {"phase1": err}
    runs = {}
    for name, flags in COMM_CONFIGS.items():
        rc, err = _tt(6, 2, root, tag=name, env={**flags, "HEAL_COUNT_COLLECTIVES": "1"}, python=python)
        ev = _events(f"{root}/comm2{name}/events.jsonl")
        runs[name] = {"rc": rc, "err": err, "steps": [e for e in ev if e.get("event") == "step"],
                      "comm": [e for e in ev if e.get("event") == "comm_counts"],
                      "flags": next((e for e in ev if e.get("event") == "comm"), None)}
    ref = runs["off"]
    for name, r in runs.items():
        if r["rc"]:
            out[name] = {"rc": r["rc"], "err": r["err"]}
            continue
        row = {"steps": len(r["steps"]),
               "boundaries_equal": [(s["trained"], s["tokens"]) for s in r["steps"]]
               == [(s["trained"], s["tokens"]) for s in ref["steps"]],
               "grad_norm_rel_max": max((abs(s["grad_norm"] - t["grad_norm"]) / t["grad_norm"]
                                         for s, t in zip(r["steps"], ref["steps"])), default=None),
               "kd_rel_max": max((abs(s["kd"] - t["kd"]) / abs(t["kd"]) for s, t in zip(r["steps"], ref["steps"])),
                                 default=None),
               "synced_backwards": [s.get("synced_backwards") for s in r["steps"]],
               "rs_calls": [c["counts"].get("reduce_scatter_single", c["counts"].get("reduce_scatter_tensor", [0, 0]))[0]
                            for c in r["comm"]],
               "rs_mbytes": [round(c["counts"].get("reduce_scatter_single", c["counts"].get("reduce_scatter_tensor",
                                                                                            [0, 0]))[1] / 1e6, 3)
                             for c in r["comm"]],
               "ag_calls": [c["counts"].get("all_gather_single", c["counts"].get("all_gather_into_tensor", [0, 0]))[0]
                            for c in r["comm"]],
               "ar_calls": [c["counts"].get("all_reduce", [0, 0])[0] for c in r["comm"]],
               "counts_equal_across_ranks": all(c["equal"] for c in r["comm"]),
               "peak_gib": [c["peak_gib"] for c in r["comm"]][-1:],
               "expert_rows": (r["flags"] or {}).get("expert_rows")}
        grads, params = [], []
        for s in range(1, len(r["steps"]) + 1):
            ga, gb = f"{root}/steps2{name}/grads-step{s}.pt", f"{root}/steps2off/grads-step{s}.pt"
            pa, pb = f"{root}/steps2{name}/params-step{s}.pt", f"{root}/steps2off/params-step{s}.pt"
            if Path(ga).exists() and Path(gb).exists():
                grads.append(_cmp_tensors(torch.load(ga), torch.load(gb)))
            if Path(pa).exists() and Path(pb).exists():
                params.append(_cmp_tensors(torch.load(pa), torch.load(pb)))
        row["grads_vs_off"] = {"cosine_min": min((g["cosine"] for g in grads), default=None),
                               "norm_rel_max": max((g["norm_rel"] for g in grads), default=None),
                               "max_rel": max((g["max_rel"] for g in grads), default=None),
                               "worst": max(grads, key=lambda g: g["max_rel"])["worst_tensor"] if grads else None,
                               "same_names": all(g["same_names"] for g in grads)}
        row["params_vs_off"] = {"max_rel": max((g["max_rel"] for g in params), default=None),
                                "worst": max(params, key=lambda g: g["max_rel"])["worst_tensor"] if params else None}
        out[name] = row
    out["off_steps"] = [{k: s[k] for k in ("step", "trained", "tokens", "kd", "grad_norm", "lr")} for s in ref["steps"]]
    # exact mid-round resume and data-parallel equality (dp1 vs dp2, unequal lengths) under the flags
    for name in ("off", *RESUME_CONFIGS):
        flags = {**COMM_CONFIGS[name], "HEAL_COUNT_COLLECTIVES": "1"}
        res = {}
        for phase, procs in ((4, 2), (5, 2), (3, 1), (3, 2)):
            rc, err = _tt(phase, procs, root, tag=name, env=flags, python=python)
            if rc:
                res[f"phase{phase}-{procs}"] = err
                break
        a = next((e for e in _events(f"{root}/ckA{name}/events.jsonl") if e.get("event") == "step" and e["step"] == 3), None)
        b = next((e for e in _events(f"{root}/ckB{name}/events.jsonl") if e.get("event") == "step" and e["step"] == 3), None)
        if a and b:
            res["exact_resume_ok"] = all(abs(a[k] - b[k]) <= 1e-6 * max(1.0, abs(a[k]))
                                         for k in ("kd", "kd_reply", "grad_norm", "trained", "lr"))
            res["step3"] = [{k: x[k] for k in ("kd", "grad_norm", "trained")} for x in (a, b)]
        g1, g2 = f"{root}/grads-w1{name}.pt", f"{root}/grads-w2{name}.pt"
        if Path(g1).exists() and Path(g2).exists():
            res["dp1_vs_dp2"] = _cmp_tensors(torch.load(g2), torch.load(g1))
        out[f"resume_{name}"] = res
    if base_root:  # flags off vs the pre-change file: phases 3 (dp2 gradients) and 4 (3 steps) on the same data
        shutil.rmtree(base_root, ignore_errors=True)
        Path(base_root).mkdir(parents=True)
        for item in ("tiny-student", "teacher", "general"):
            shutil.copytree(f"{root}/{item}", f"{base_root}/{item}", symlinks=True)
        Path(f"{base_root}/round1").mkdir()
        shutil.copy(f"{root}/round1/routing.pt", f"{base_root}/round1/routing.pt")
        res = {}
        for phase in (3, 4):
            rc, err = _tt(phase, 2, base_root, script="/opt/heal/heal_kd_base.py", python=python)
            if rc:
                res[f"phase{phase}"] = err
        steps = lambda f: [{k: e[k] for k in ("step", "trained", "tokens", "kd", "kd_reply", "kd_mtp", "grad_norm", "lr")}
                           for e in _events(f) if e.get("event") == "step"]
        sa, sb = steps(f"{base_root}/ckA/events.jsonl"), steps(f"{root}/ckAoff/events.jsonl")
        res["loss_rows_identical"] = bool(sa) and sa == sb
        res["rows"] = [sa, sb]
        if Path(f"{base_root}/grads-w2.pt").exists():
            res["dp2_grads_vs_base"] = _cmp_tensors(torch.load(f"{root}/grads-w2off.pt"),
                                                    torch.load(f"{base_root}/grads-w2.pt"))
        out["base"] = res
    return out


# trainer MFU S4 (idle-GPU cuts): HEAL_* flags per case on top of the production flags (S1 capture, S3 step sync and
# expert rows); each group compares its cases with its own "off" on the same data (gloo, deterministic CPU)
S4_PROD = {"HEAL_CAPTURE_CACHE": "1", "HEAL_GRAD_SYNC": "step", "HEAL_EXPERT_ROWS": "1"}
S4_ALL = {"HEAL_HOST_TIMERS": "1", "HEAL_LAZY_STATS": "1", "HEAL_HOST_GROUP": "1", "HEAL_MASK_HOST": "1",
          "HEAL_PLE_TABLE": "1", "HEAL_PREFETCH": "1"}
RL1_PATH = {"HEAL_NO_FAST_ATTENTION": "0", "RL1_ATTENTION": "selected"}  # rl1's indexer + selected attention on CPU
S4_GROUPS = {
    "hf": (S4_PROD, {"off": {}, "off2": {}, "timers": {"HEAL_HOST_TIMERS": "1"}, "lazy": {"HEAL_LAZY_STATS": "1"},
                     "lazy_group": {"HEAL_LAZY_STATS": "1", "HEAL_HOST_GROUP": "1"}, "ple": {"HEAL_PLE_TABLE": "1"},
                     "prefetch": {"HEAL_PREFETCH": "1"}, "mask": {"HEAL_MASK_HOST": "1"}, "all": S4_ALL}, ()),
    "rl1": ({**S4_PROD, **RL1_PATH}, {"off": {}, "mask": {"HEAL_MASK_HOST": "1"}, "timers": {"HEAL_HOST_TIMERS": "1"},
                                      "all": S4_ALL}, ()),
    "nocap": ({**S4_PROD, "HEAL_CAPTURE_CACHE": "0"}, {"off": {}, "ple": {"HEAL_PLE_TABLE": "1"}, "all": S4_ALL}, ()),
    "loops": ({**S4_PROD, **RL1_PATH}, {"off": {}, "all": S4_ALL}, ("--truncate-loops",)),
    "window": ({"HEAL_CAPTURE_CACHE": "1"}, {"off": {}, "all": S4_ALL}, ()),  # HEAL_GRAD_SYNC=window, FSDP grads
}


@app.function(image=image, timeout=3600)
def s4_group(group: str):
    """One S4 group: phase 1 (setup), then phase 6 (a several-step round: own and general slots, unequal lengths, a
    padding rank, the flush step) per case; step rows, per-step gradients and parameters against the group's off;
    host events. Groups rl1 and loops also run exact mid-round resume (phases 4 then 5) and dp1 vs dp2 (phase 3)
    per case."""
    import shutil

    import torch

    base, cases, extra = S4_GROUPS[group]
    root = f"/tmp/s4{group}"
    shutil.rmtree(root, ignore_errors=True)
    rc, err = _tt(1, 2, root, env=base)
    if rc:
        return {"phase1": err}
    out, runs = {}, {}
    for name, flags in cases.items():
        rc, err = _tt(6, 2, root, tag=name, env={**base, **flags}, extra=extra)
        ev = _events(f"{root}/comm2{name}/events.jsonl")
        runs[name] = {"rc": rc, "err": err, "steps": [e for e in ev if e.get("event") == "step"],
                      "host": [e for e in ev if e.get("event") == "host"]}
    keys = ("step", "lr", "trained", "tokens", "kd", "kd_reply", "kd_mtp", "grad_norm", "loop_spans", "loop_removed",
            "synced_backwards")
    ref = runs["off"]
    for name, r in runs.items():
        if r["rc"]:
            out[name] = {"rc": r["rc"], "err": r["err"]}
            continue
        rows = [{k: s.get(k) for k in keys} for s in r["steps"]]
        ref_rows = [{k: s.get(k) for k in keys} for s in ref["steps"]]
        grads, params = [], []
        for s in range(1, len(r["steps"]) + 1):
            for kind, acc in (("grads", grads), ("params", params)):
                a, b = f"{root}/steps2{name}/{kind}-step{s}.pt", f"{root}/steps2off/{kind}-step{s}.pt"
                if Path(a).exists() and Path(b).exists():
                    acc.append(_cmp_tensors(torch.load(a), torch.load(b)))
        host = r["host"]
        out[name] = {"steps": len(rows), "rows_identical": rows == ref_rows,
                     "rows_diff": [(x, y) for x, y in zip(rows, ref_rows) if x != y][:2],
                     "grads_max_rel": max((g["max_rel"] for g in grads), default=None),
                     "params_max_rel": max((g["max_rel"] for g in params), default=None),
                     "dumps": [len(grads), len(params)], "host_events": len(host),
                     "host_sample": host[0]["ranks"][0] if host else None}
    out["off_rows"] = [{k: s.get(k) for k in keys} for s in ref["steps"]]
    if group in ("rl1", "loops"):
        for name in ("off", "all"):
            env = {**base, **cases[name]}
            res = {}
            for phase, procs in ((4, 2), (5, 2), (3, 1), (3, 2)):
                rc, err = _tt(phase, procs, root, tag=name, env=env, extra=extra)
                if rc:
                    res[f"phase{phase}-{procs}"] = err
                    break
            pick = lambda f: next((e for e in _events(f) if e.get("event") == "step" and e["step"] == 3), None)  # noqa
            a, b = pick(f"{root}/ckA{name}/events.jsonl"), pick(f"{root}/ckB{name}/events.jsonl")
            if a and b:
                res["exact_resume_ok"] = all(a[k] == b[k] for k in ("kd", "kd_reply", "grad_norm", "trained", "lr"))
                res["step3"] = [{k: x[k] for k in ("kd", "grad_norm", "trained")} for x in (a, b)]
            g1, g2 = f"{root}/grads-w1{name}.pt", f"{root}/grads-w2{name}.pt"
            if Path(g1).exists() and Path(g2).exists():
                res["dp1_vs_dp2"] = _cmp_tensors(torch.load(g2), torch.load(g1))
            out[f"resume_{name}"] = res
        ra, rb = out.get("resume_off", {}).get("step3"), out.get("resume_all", {}).get("step3")
        out["resume_all_vs_off"] = bool(ra and rb and ra == rb)
    return out


@app.function(image=image, timeout=3600)
def comm_cpu():
    """S3 bars on CPU (gloo, deterministic)."""
    return _comm_run("/tmp/cc", "", base_root="/tmp/ccb" if Path("/opt/heal/heal_kd_base.py").exists() else None)


@app.function(image=trainer_image, gpu="L40S:2", timeout=3600)
def comm_gpu():
    """S3 bars on 2 L40S in the trainer's software (NCCL)."""
    return _comm_run("/tmp/cc", TRAINER_PYTHON,
                     base_root="/tmp/ccb" if Path("/opt/heal/heal_kd_base.py").exists() else None)


NCCL_BW = r"""
import json, os, time
import torch
import torch.distributed as dist
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
rows = []
for mb in (16, 128, 512, 1024, 2048):  # per-rank shard (input) size; FSDP unit gathers are ~0.5 GB in per rank
    n = mb * 2**20 // 2
    x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(n * world, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        dist.all_gather_into_tensor(out, x)
    torch.cuda.synchronize(); dist.barrier()
    reps = 10
    t0 = time.time()
    for _ in range(reps):
        dist.all_gather_into_tensor(out, x)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    recv = n * 2 * (world - 1)
    rows.append({"in_mb": mb, "ms": round(dt * 1e3, 3), "recv_gbps": round(recv / dt / 1e9, 1),
                 "bus_gbps": round(n * 2 * world * (world - 1) / world / dt / 1e9, 1)})
if rank == 0:
    print("NCCLBW " + json.dumps({"world": world, "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL")},
                                  "rows": rows}), flush=True)
dist.destroy_process_group()
"""


@app.function(image=trainer_image, gpu="B300:2", timeout=900)
def nccl_bw(envs: str = ""):
    """MFU-1: all-gather bandwidth on Modal B300s in the trainer's software, NCCL_DEBUG=INFO transport lines; ``envs``:
    ;-separated cases of comma K=V NCCL settings (empty case = defaults)."""
    import re
    import subprocess

    import torch

    n = torch.cuda.device_count()
    Path("/tmp/nccl_bw.py").write_text(NCCL_BW)
    out = {"gpus": n, "smi": subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout[-3000:],
           "cases": []}
    for case in envs.split(";"):
        extra = dict(kv.split("=", 1) for kv in case.split(",") if kv)
        p = subprocess.run([TRAINER_PYTHON, "-m", "torch.distributed.run", "--nproc-per-node", str(n), "/tmp/nccl_bw.py"],
                           capture_output=True, text=True, env={**os.environ, "NCCL_DEBUG": "INFO", **extra}, timeout=600)
        res = next((json.loads(line[7:]) for line in p.stdout.splitlines() if line.startswith("NCCLBW ")), None)
        via = sorted(set(re.findall(r"via (\S+)", p.stdout + p.stderr)))
        out["cases"].append({"env": extra, "rc": p.returncode, "result": res, "via": via,
                             "err": (p.stderr[-1500:] if p.returncode else "")})
        print(json.dumps(out["cases"][-1])[:2000], flush=True)
    return out


test_image = image_518.pip_install("pytest", "httpx")
for _name in FILES:
    test_image = test_image.add_local_file(str(HERE / _name), f"/opt/repo/scripts/posttrain/{_name}", copy=True)
if modal.is_local():  # the container imports this file from /root (no repo above it)
    test_image = test_image.add_local_dir(str(HERE.parents[1] / "tests"), "/opt/repo/tests", copy=True,
                                          ignore=["**/__pycache__/**"])


@app.function(image=test_image, cpu=(0.125, 16), timeout=3600)
def inert_cpu(only: str = ""):
    """HEAL_INERT_PREFIX (trainer MFU S1b, 23 Oct) on CPU: the inert-check bars (2 gloo ranks, BF16 and fp32), the
    OPD stream leg (stream_tiny's phase 7) flag off vs on, and the heal unit tests. ``only``: these stream legs
    (comma list) and their derived reads, no inert-check bars."""
    import subprocess
    import sys

    out = {}
    d = "/tmp/inert"
    p = None if only else subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "2", "/opt/heal/heal_kd.py",
                        "inert-check", "--out", d, "--k", "16"], capture_output=True, text=True, cwd="/opt/heal",
                       env={**os.environ, "PYTHONPATH": "/opt/heal"})
    ev = [] if only else _events(f"{d}/events.jsonl")
    out["check"] = next((e for e in ev if e.get("check")), None)
    out["cases"] = [e for e in ev if "case" in e]
    if p is not None and p.returncode:
        out["check_err"] = _first_traceback(p.stderr)
    legs = {}
    for name, env in (("off", {}), ("on", {"HEAL_INERT_PREFIX": "1"}), ("pieces", {"HEAL_TINY_PIECES": "1"}),
                      ("pieces-s1", {"HEAL_TINY_PIECES": "1", "HEAL_INERT_PREFIX": "1"}),
                      ("pieces-s2", {"HEAL_TINY_PIECES": "1", "HEAL_INERT_PREFIX": "1", "HEAL_STREAM_COALESCE": "1"}),
                      ("knobs", {"HEAL_TINY_PIECES": "1", "HEAL_INERT_PREFIX": "1", "HEAL_STREAM_COALESCE": "1",
                                 "HEAL_INERT_CHECK": "1", "HEAL_CHUNK_AFTER": "2:32"}),
                      ("prefetch", {"HEAL_TINY_PIECES": "1", "HEAL_INERT_PREFIX": "1", "HEAL_STREAM_COALESCE": "1",
                                    "HEAL_STREAM_PREFETCH": "1"}),
                      ("balance", {"HEAL_TINY_PIECES": "1", "HEAL_INERT_PREFIX": "1", "HEAL_STREAM_COALESCE": "1",
                                   "HEAL_STREAM_PREFETCH": "1", "HEAL_STREAM_BALANCE": "1"}),
                      ("balance3", {"HEAL_TINY_PIECES": "1", "HEAL_INERT_PREFIX": "1", "HEAL_STREAM_COALESCE": "1",
                                    "HEAL_STREAM_PREFETCH": "1", "HEAL_STREAM_BALANCE": "1", "PROCS": "3"}),
                      # MFU-1 HEAL_SKIP_INERT_LOSS on c7's comm flags, 3 ranks: steps must equal the flag-off leg
                      ("prod3", {"HEAL_TINY_PIECES": "1", "HEAL_INERT_PREFIX": "1", "HEAL_STREAM_COALESCE": "1",
                                 "HEAL_STREAM_BALANCE": "1", "HEAL_GRAD_SYNC": "step", "HEAL_EXPERT_ROWS": "1",
                                 "HEAL_CAPTURE_CACHE": "1", "PROCS": "3"}),
                      ("prod3-skip", {"HEAL_TINY_PIECES": "1", "HEAL_INERT_PREFIX": "1", "HEAL_STREAM_COALESCE": "1",
                                      "HEAL_STREAM_BALANCE": "1", "HEAL_GRAD_SYNC": "step", "HEAL_EXPERT_ROWS": "1",
                                      "HEAL_CAPTURE_CACHE": "1", "HEAL_SKIP_INERT_LOSS": "1", "PROCS": "3"}),
                      ("prod3-align", {"HEAL_TINY_PIECES": "1", "HEAL_INERT_PREFIX": "1", "HEAL_STREAM_COALESCE": "1",
                                       "HEAL_STREAM_BALANCE": "1", "HEAL_GRAD_SYNC": "step", "HEAL_EXPERT_ROWS": "1",
                                       "HEAL_CAPTURE_CACHE": "1", "HEAL_SKIP_INERT_LOSS": "1", "HEAL_STREAM_ALIGN": "1",
                                       "HEAL_STREAM_PREFETCH": "1", "PROCS": "3"}),
                      # MFU-1 HEAL_KD_ROWS on the skip leg: LM head + KD on live rows only (equal at the fp32 floor)
                      ("prod3-rows", {"HEAL_TINY_PIECES": "1", "HEAL_INERT_PREFIX": "1", "HEAL_STREAM_COALESCE": "1",
                                      "HEAL_STREAM_BALANCE": "1", "HEAL_GRAD_SYNC": "step", "HEAL_EXPERT_ROWS": "1",
                                      "HEAL_CAPTURE_CACHE": "1", "HEAL_SKIP_INERT_LOSS": "1", "HEAL_KD_ROWS": "1",
                                      "PROCS": "3"}),
                      # MFU-1 HEAL_STREAM_STEPPLAN: the step's units grouped by call structure (rank 0 plans)
                      ("prod3-plan", {"HEAL_TINY_PIECES": "1", "HEAL_INERT_PREFIX": "1", "HEAL_STREAM_COALESCE": "1",
                                      "HEAL_STREAM_BALANCE": "1", "HEAL_GRAD_SYNC": "step", "HEAL_EXPERT_ROWS": "1",
                                      "HEAL_CAPTURE_CACHE": "1", "HEAL_SKIP_INERT_LOSS": "1", "HEAL_KD_ROWS": "1",
                                      "HEAL_STREAM_STEPPLAN": "1", "HEAL_STREAM_PREFETCH": "1", "PROCS": "3"})):
        if only and name not in only.split(","):
            continue
        root = f"/tmp/st-{name}"
        procs = int(env.pop("PROCS", 2)) if "PROCS" in env else 2
        rc, err = _tt(1, procs, root, env=env)
        if not rc:
            rc, err = _tt(7, procs, root, env=env)
        steps = [e for e in _events(f"{root}/st/events.jsonl") if e.get("event") == "step"]
        legs[name] = {"rc": rc, "err": err[-3000:] if rc else "",
                      "steps": [{k: e.get(k) for k in ("step", "kd", "kd_reply", "kd_mtp", "grad_norm", "trained",
                                                        "tokens", "age_median", "age_max", "stream_chunks",
                                                        "opd_sampled_positions", "tok_rank", "loss_tok_rank", "coalesce",
                                                        "finite", "chunk", "fe_rank", "fe_idle_share",
                                                        "prefetched")} for e in steps],
                      "inert_check": [e for e in _events(f"{root}/st/events.jsonl") if e.get("event") == "inert_check"],
                      "loaded": next((e for e in _events(f"{root}/st/events.jsonl") if e.get("event") == "loaded"), None)}
    out["stream"] = legs
    rel = lambda a, b: max((abs(x[k] - y[k]) / max(abs(y[k]), 1e-12) for x, y in zip(a, b)  # noqa: E731
                            for k in ("kd", "kd_reply", "kd_mtp", "grad_norm") if y.get(k)), default=None)
    if "prod3-rows" in legs and "prod3-skip" in legs:
        pr, ps = legs["prod3-rows"], legs["prod3-skip"]
        out["kd_rows"] = {"ok": not pr["rc"] and not ps["rc"] and len(pr["steps"]) == 3
                          and all(x.get("finite") for x in pr["steps"]),
                          "max_rel": rel(pr["steps"], ps["steps"]) if not pr["rc"] and not ps["rc"] else None,
                          "steps": [{k: x.get(k) for k in ("kd", "kd_reply", "kd_mtp", "grad_norm")} for x in pr["steps"]],
                          "base": [{k: x.get(k) for k in ("kd", "kd_reply", "kd_mtp", "grad_norm")} for x in ps["steps"]],
                          "err": pr["err"][-800:]}
    if "prod3-plan" in legs:
        pp = legs["prod3-plan"]
        out["step_plan"] = {"ok": not pp["rc"] and len(pp["steps"]) == 3 and all(x.get("finite") for x in pp["steps"]),
                            "fe_idle_share": [x.get("fe_idle_share") for x in pp["steps"]],
                            "trained": [x.get("trained") for x in pp["steps"]], "err": pp["err"][-1500:]}
        if "prod3-rows" in legs:
            out["step_plan"]["rows_fe_idle_share"] = [x.get("fe_idle_share") for x in legs["prod3-rows"]["steps"]]
    if only:
        return out
    if not legs["off"]["rc"] and not legs["on"]["rc"]:
        a, b = legs["on"]["steps"], legs["off"]["steps"]
        out["stream_max_rel"] = max((abs(x[k] - y[k]) / max(abs(y[k]), 1e-12) for x, y in zip(a, b)
                                     for k in ("kd", "kd_reply", "kd_mtp", "grad_norm") if y.get(k)), default=None)
        out["stream_same_steps"] = len(a) == len(b) == 3 and all(x["trained"] == y["trained"] for x, y in zip(a, b))
    if not legs["pieces"]["rc"] and not legs["pieces-s1"]["rc"]:
        a, b = legs["pieces-s1"]["steps"], legs["pieces"]["steps"]
        out["pieces_s1_max_rel"] = max((abs(x[k] - y[k]) / max(abs(y[k]), 1e-12) for x, y in zip(a, b)
                                        for k in ("kd", "kd_reply", "kd_mtp", "grad_norm") if y.get(k)), default=None)
    s2 = legs["pieces-s2"]
    if not s2["rc"]:
        last = s2["steps"][-1] if s2["steps"] else {}
        out["s2_ok"] = (len(s2["steps"]) == 3 and all(x.get("finite") for x in s2["steps"])
                        and (last.get("coalesce") or {}).get("pieces", 0) > (last.get("coalesce") or {}).get("units", 0))
    kn = legs["knobs"]  # HEAL_INERT_CHECK once on the first trained slot; HEAL_CHUNK_AFTER=2:32 from step 2
    out["knobs_ok"] = (not kn["rc"] and len(kn["steps"]) == 3 and all(x.get("finite") for x in kn["steps"])
                       and [x.get("chunk") for x in kn["steps"]] == [64, 32, 32] and len(kn["inert_check"]) == 1)
    if not kn["rc"] and not s2["rc"]:  # the extra no-grad pass leaves step 1 as the S2 leg trains it
        a, b = kn["steps"][:1], s2["steps"][:1]
        out["knobs_step1_max_rel"] = max((abs(x[k] - y[k]) / max(abs(y[k]), 1e-12) for x, y in zip(a, b)
                                          for k in ("kd", "kd_reply", "kd_mtp", "grad_norm") if y.get(k)), default=None)
    rel = lambda a, b: max((abs(x[k] - y[k]) / max(abs(y[k]), 1e-12) for x, y in zip(a, b)  # noqa: E731
                            for k in ("kd", "kd_reply", "kd_mtp", "grad_norm") if y.get(k)), default=None)
    pf, bl, b3 = legs["prefetch"], legs["balance"], legs["balance3"]
    out["lever1"] = {  # prefetch alone keeps the picks: steps equal the S2 leg; balance may re-pair units
        "prefetch_ok": not pf["rc"] and len(pf["steps"]) == 3 and all(x.get("finite") for x in pf["steps"]),
        "prefetch_vs_s2_max_rel": rel(pf["steps"], s2["steps"]) if not pf["rc"] and not s2["rc"] else None,
        "prefetched": [x.get("prefetched") for x in pf["steps"]],
        "balance_ok": not bl["rc"] and len(bl["steps"]) == 3 and all(x.get("finite") for x in bl["steps"]),
        "balance_vs_s2_max_rel": rel(bl["steps"], s2["steps"]) if not bl["rc"] and not s2["rc"] else None,
        "fe_idle_share": {"s2": [x.get("fe_idle_share") for x in s2["steps"]],
                          "balance": [x.get("fe_idle_share") for x in bl["steps"]],
                          "balance3": [x.get("fe_idle_share") for x in b3["steps"]]},
        "balance3_ok": not b3["rc"] and len(b3["steps"]) == 3 and all(x.get("finite") for x in b3["steps"]),
        "balance3_fe_rank": [x.get("fe_rank") for x in b3["steps"]]}
    pr, ps = legs["prod3"], legs["prod3-skip"]
    out["skip_inert"] = {"ok": not pr["rc"] and not ps["rc"] and len(ps["steps"]) == 3,
                         "steps_equal": [{k: x.get(k) for k in ("kd", "kd_reply", "kd_mtp", "grad_norm")} ==
                                         {k: y.get(k) for k in ("kd", "kd_reply", "kd_mtp", "grad_norm")}
                                         for x, y in zip(ps["steps"], pr["steps"])],
                         "max_rel": rel(ps["steps"], pr["steps"]) if not pr["rc"] and not ps["rc"] else None}
    pa = legs["prod3-align"]
    out["align"] = {"ok": not pa["rc"] and len(pa["steps"]) == 3 and all(x.get("finite") for x in pa["steps"]),
                    "fe_idle_share": [x.get("fe_idle_share") for x in pa["steps"]],
                    "skip_fe_idle_share": [x.get("fe_idle_share") for x in ps["steps"]], "err": pa["err"][-800:]}
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "--noconftest", "/opt/repo/tests/test_heal_stream.py",
                        "/opt/repo/tests/test_heal_knobs.py",
                        "/opt/repo/tests/test_heal_plan.py", "/opt/repo/tests/test_heal_kd_loss.py",
                        "/opt/repo/tests/test_heal_schedule.py"], capture_output=True, text=True, cwd="/opt/repo")
    out["pytest"] = {"rc": t.returncode, "tail": t.stdout[-2500:], "err": t.stderr[-1500:],
                     "files": sorted(p.name for p in Path("/opt/repo/tests").glob("*"))[:40]}
    return out


@app.function(image=image_518.add_local_file(str(HERE / "heal_reshard.py"), "/opt/heal/heal_reshard.py", copy=True),
              cpu=(0.125, 16), timeout=5400)
def tranks3_cpu():
    """A 3-rank trainer on CPU (gloo; 23 Oct, PTH): the tiny student's experts do not split evenly over 3 ranks. Per
    flag set (off; step sync + expert rows + capture): 3-rank A (3 steps, checkpoint after 2) and an exact resume B of
    step 3; a 2-rank checkpoint resharded to 3 ranks and a 3-rank one to 2, each resumed for step 3 and compared with
    its source run (step 3 KD / grad norm, saved student, rel <= 1e-4); the comm round (phase 6) and the OPD stream
    (phase 7, pieces, S1 + S2) at 3 ranks."""
    import shutil
    import sys

    sys.path.insert(0, "/opt/heal")
    import heal_reshard

    root = "/tmp/t3"
    shutil.rmtree(root, ignore_errors=True)
    out = {}
    rc, err = _tt(1, 2, root)
    if rc:
        return {"round1": err}
    experts = json.loads(Path(f"{root}/tiny-student/config.json").read_text())["text_config"]["num_experts"]
    out["num_experts"] = experts
    rel = lambda x, y: abs(x - y) / max(abs(x), abs(y), 1e-12)  # noqa: E731

    def step3(name):
        return next((e for e in _events(f"{root}/{name}/events.jsonl") if e.get("event") == "step" and e["step"] == 3), {})

    def student(n):
        from safetensors.torch import load_file
        d = {}
        for f in sorted(Path(f"{root}/{n}/student").glob("*.safetensors")):
            d.update(load_file(str(f)))
        return {k: v.float() for k, v in d.items()}

    def same(a, b):
        ea, eb = step3(a), step3(b)
        return {"step3": [{k: e.get(k) for k in ("kd", "kd_reply", "grad_norm", "trained")} for e in (ea, eb)],
                "ok": bool(ea and eb and all(rel(ea[k], eb[k]) <= 1e-4 for k in ("kd", "kd_reply", "grad_norm")))}

    prod = {"HEAL_GRAD_SYNC": "step", "HEAL_EXPERT_ROWS": "1", "HEAL_CAPTURE_CACHE": "1"}
    for name, env in (("off", {}), ("prod", prod)):
        res = {}
        save = {**env, "HEAL_TINY_SAVE": "1"}
        for label, phase, procs, tag, e in (("A3", 4, 3, f"w3{name}", env), ("B3", 5, 3, f"w3{name}", save),
                                            ("A2", 4, 2, f"w2{name}", env)):
            rc, err = _tt(phase, procs, root, tag=tag, env=e)
            if rc:
                res[label] = err[-4000:]
                break
        if any(isinstance(v, str) for v in res.values()):
            out[name] = res
            continue
        res["resume_w3"] = same(f"ckAw3{name}", f"ckBw3{name}")
        for label, src, world, tag in (("r2to3", f"ckAw2{name}", 3, f"r3{name}"), ("r3to2", f"ckAw3{name}", 2, f"r2{name}")):
            try:
                res[f"{label}_reshard"] = heal_reshard.reshard(f"{root}/{src}/ckpt-2", f"{root}/ckA{tag}/ckpt-2", world, experts)
            except SystemExit as exc:
                res[f"{label}_reshard"] = f"SystemExit: {exc}"
                continue
            rc, err = _tt(5, world, root, tag=tag, env=save)
            if rc:
                res[label] = err[-4000:]
                continue
            res[label] = same(src, f"ckB{tag}")
            res[label]["student"] = _cmp_tensors(student(f"ckB{tag}"), student(src))
            res[label]["ok"] = res[label]["ok"] and res[label]["student"]["max_rel"] <= 1e-4
        out[name] = res
    rc, err = _tt(6, 3, root, tag="w3", env={**prod, "HEAL_COUNT_COLLECTIVES": "1"})
    ev = _events(f"{root}/comm3w3/events.jsonl")
    out["comm3"] = {"rc": rc, "err": err[-3000:] if rc else "",
                    "steps": [{k: e.get(k) for k in ("step", "kd", "grad_norm", "trained", "tokens")}
                              for e in ev if e.get("event") == "step"],
                    "counts_equal": all(c.get("equal") for c in ev if c.get("event") == "comm_counts")}
    rc, err = _tt(7, 3, root, tag="w3", env={**prod, "HEAL_TINY_PIECES": "1", "HEAL_INERT_PREFIX": "1",
                                             "HEAL_STREAM_COALESCE": "1"})
    ev = _events(f"{root}/stw3/events.jsonl")
    out["stream3"] = {"rc": rc, "err": err[-3000:] if rc else "",
                      "steps": [{k: e.get(k) for k in ("step", "kd", "grad_norm", "finite", "trained", "tok_rank",
                                                       "coalesce", "age_max")} for e in ev if e.get("event") == "step"]}
    out["ok"] = (all(isinstance(out.get(n), dict) and all(isinstance(out[n].get(k), dict) and out[n][k]["ok"]
                                                          for k in ("resume_w3", "r2to3", "r3to2")) for n in ("off", "prod"))
                 and not out["comm3"]["rc"] and not out["stream3"]["rc"] and len(out["stream3"]["steps"]) == 3)
    return out


@app.function(image=image_518.add_local_file(str(HERE / "heal_reshard.py"), "/opt/heal/heal_reshard.py", copy=True),
              cpu=(0.125, 16), timeout=3600)
def tranks3_reshard_cpu():
    """The 2 -> 3 reshard's noise vs its sensitivity (prod flags): B3 (resumed from the resharded 2-rank checkpoint)
    against A2's saved student per tensor group (experts / dense), and the same with the moments zeroed (must be far
    off); A2 against its own 2-rank resume B2 (exact)."""
    import shutil
    import sys

    import torch

    sys.path.insert(0, "/opt/heal")
    import heal_reshard

    root, out = "/tmp/t3r", {}
    shutil.rmtree(root, ignore_errors=True)
    prod = {"HEAL_GRAD_SYNC": "step", "HEAL_EXPERT_ROWS": "1", "HEAL_CAPTURE_CACHE": "1"}
    save = {**prod, "HEAL_TINY_SAVE": "1"}
    for label, phase, procs, tag, env in (("round1", 1, 2, "", None), ("A2", 4, 2, "", prod), ("B2", 5, 2, "", save)):
        rc, err = _tt(phase, procs, root, tag=tag, env=env)
        if rc:
            return {label: err[-4000:]}
    experts = json.loads(Path(f"{root}/tiny-student/config.json").read_text())["text_config"]["num_experts"]
    out["reshard"] = heal_reshard.reshard(f"{root}/ckA/ckpt-2", f"{root}/ckAr3/ckpt-2", 3, experts)
    shutil.copytree(f"{root}/ckAr3", f"{root}/ckAr3z")
    for f in Path(f"{root}/ckAr3z/ckpt-2").glob("optim-rank*.pt"):
        st = torch.load(f, weights_only=False)
        for s_ in st["adam"]["state"].values():
            for k in ("exp_avg", "exp_avg_sq"):
                s_[k] = torch.zeros_like(s_[k])
        st["rows"]["m"] = [torch.zeros_like(x) for x in st["rows"]["m"]]
        st["rows"]["v"] = [torch.zeros_like(x) for x in st["rows"]["v"]]
        torch.save(st, f)
    for tag in ("r3", "r3z"):
        rc, err = _tt(5, 3, root, tag=tag, env=save)
        if rc:
            out[tag] = err[-4000:]
            return out

    def student(n):
        from safetensors.torch import load_file
        d = {}
        for f in sorted(Path(f"{root}/{n}/student").glob("*.safetensors")):
            d.update(load_file(str(f)))
        return {k: v.float() for k, v in d.items()}

    a = student("ckA")
    for name in ("ckB", "ckBr3", "ckBr3z"):
        b = student(name)
        groups = {"experts": [k for k in a if ".experts." in k], "dense": [k for k in a if ".experts." not in k]}
        out[name] = {g: _cmp_tensors({k: b[k] for k in ks}, {k: a[k] for k in ks}) for g, ks in groups.items()}
        # elementwise: the share of entries off by more than 1e-3 of their tensor's max
        off = sum(int(((b[k] - a[k]).abs() > 1e-3 * a[k].abs().max().clamp_min(1e-30)).sum()) for k in a)
        out[name]["entries_off_1e-3"] = off / sum(a[k].numel() for k in a)
    step3 = {n: next((e for e in _events(f"{root}/{n}/events.jsonl") if e.get("event") == "step" and e["step"] == 3), {})
             for n in ("ckA", "ckB", "ckBr3", "ckBr3z")}
    out["step3"] = {n: {k: e.get(k) for k in ("kd", "grad_norm", "trained")} for n, e in step3.items()}
    return out


@app.local_entrypoint()
def main(run_id: str = "sft-heal-tiny-v1", train: bool = False, capture: str = "", save: str = "", what: str = "",
         gpu: str = "", minutes: int = 30, proxy_args: str = "", proxy_env: str = "", proxy_gpu: str = "L40S",
         proxy_cases: str = "off,layers", proxy_session: int = 0):
    if not what or not gpu:
        raise SystemExit("--what '<plain description>' and --gpu '<type x count, e.g. L40S:2 or CPU>' are required: "
                         "every app writes its queue row")
    kind = "proxy" if proxy_args else (capture or ('train' if train else 'tiny'))
    row = f"{run_id}-{kind}-{int(__import__('time').time()) % 100000}"
    queue_row(row, "running", what=what, gpu=gpu, dur=minutes * 60)
    try:
        _main(train, capture, save, proxy_args, proxy_env, proxy_gpu, proxy_cases, proxy_session)
    finally:
        queue_row(row, "completed")


def _main(train, capture, save, proxy_args="", proxy_env="", proxy_gpu="L40S", proxy_cases="off,layers",
          proxy_session=0):
    if proxy_args:  # trainer MFU S2: heal_kd.py proxy on one card (L40S tests; no B300 from this agent)
        call = proxy.with_options(gpu=proxy_gpu).spawn(proxy_args, proxy_env, proxy_cases, proxy_session)
        print(json.dumps({"function": "proxy", "gpu": proxy_gpu, "call_id": call.object_id}), flush=True)
        result = call.get()
        if save:
            Path(save).write_text(json.dumps(result, indent=1))
        for e in result["events"]:
            if e.get("event") in ("proxy_case", "proxy_session"):
                print(json.dumps({k: v for k, v in e.items() if k not in ("windows", "sequences")})[:4000])
        print(json.dumps({k: v for k, v in result.items() if k != "events"}, indent=1)[:8000])
        return
    if capture:  # cpu, gpu or mem: spawned, the call id printed first, then the result
        fn = {"cpu": capture_cpu, "gpu": capture_gpu, "mem": recompute_mem, "gpu-compile": capture_gpu_compile,
              "mem-trainer": recompute_mem_trainer, "cpu-chunk": capture_cpu_chunk,
              "comm-cpu": comm_cpu, "comm-gpu": comm_gpu, "inert-cpu": inert_cpu,
              "tranks3-cpu": tranks3_cpu, "tranks3-reshard": tranks3_reshard_cpu}[capture]
        call = fn.spawn()
        print(json.dumps({"function": capture, "call_id": call.object_id}), flush=True)
        result = call.get()
        if save:
            Path(save).write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1)[:20000])
        return
    print(json.dumps(train_tiny.remote() if train else tiny.remote(), indent=1))
