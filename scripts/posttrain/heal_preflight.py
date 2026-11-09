"""Pre-launch check of one heal_round launch (PTH 19 Oct, after two B300 failures on untried paths): the bundle's own
heal_kd.py runs the launcher's own ``train`` arguments (heal_train_args from the ARC_HEAL_* settings; only the paths,
the batch and the time limit swapped) on 2 L4 (NCCL, FSDP2, CUDA as on B300) with a tiny real-vocabulary student on
real teacher and general files: round A one optimizer step from the tiny student and a save; round B resumes from A
(other files, so the ESFT selection may move) one step and a save; the saved student then reloads. Header checks of
the real student and frozen folders and, with a resume, of the real optimizer files.

  ARC_RL1_BUNDLE=dist/sft-heal-round-vN ARC_HEAL_...=... modal run scripts/posttrain/heal_preflight.py \
      [--data TEACHER_FOLDER] [--general GENERAL_FOLDER]
"""
import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ.get("ARC_RL1_BUNDLE", "dist/none")).resolve() if modal.is_local() else Path("/")
app = modal.App("sft-heal-preflight", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.14.0", "transformers==5.17.0", "safetensors", "pillow", "torchvision==0.29.0", "accelerate")
image_518 = modal.Image.debian_slim(python_version="3.12").apt_install("rclone").pip_install(  # stream preflight:
    "torch==2.14.0", "transformers==5.18.0", "safetensors", "pillow", "torchvision==0.29.0", "accelerate")  # trainer's
if modal.is_local():
    for name in ("heal_kd.py", "flash_qlora_job.py", "flash_config.json", "fla_pin.py"):
        image = image.add_local_file(str(BUNDLE / "rl1" / name), f"/opt/heal/{name}", copy=True)
    for name in ("heal_kd.py", "flash_qlora_job.py", "flash_config.json", "heal_stream.py", "heal_delta.py",
                 "rl1_trainer.py", "rl1_sparse.py", "rl_job.py", "rl_rollouts.py", "fla_pin.py", "rl1_hyper.py", "rl1_sparse2.py"):
        if (BUNDLE / "rl1" / name).exists():
            image_518 = image_518.add_local_file(str(BUNDLE / "rl1" / name), f"/opt/heal/{name}", copy=True)
    for f in sorted((BUNDLE / "rl1").glob("heal_prune-*.json")):  # the prune specs a launch names (real_paths check)
        image_518 = image_518.add_local_file(str(f), f"/opt/heal/{f.name}", copy=True)
    image = image.add_local_file(__file__, "/root/heal_preflight.py", copy=True)
    image = image.add_local_file(str(HERE / "modal_rl1.py"), "/root/modal_rl1.py", copy=True)  # run_smoke
    image_518 = image_518.add_local_file(str(HERE / "modal_rl1.py"), "/root/modal_rl1.py", copy=True)  # puller flags
    image_518 = image_518.add_local_file(__file__, "/root/heal_preflight.py", copy=True)
    image_518 = image_518.add_local_file(str(BUNDLE / "launch" / "pth" / "train_export.py"), "/opt/heal/train_export.py",
                                         copy=True)  # exporter_keys: the dashboard's own exporter
    # delta_bucket: the play side's own code (the Nebius runner's follow/copy, the play bundle's weights follower)
    image_518 = image_518.add_local_file(os.environ.get("ARC_PF_RUNNER", "./.claude/worktrees/"
                                                        "nebius-opd/scripts/nebius_runner.py"), "/root/nebius_runner.py", copy=True)
    image_518 = image_518.add_local_file(os.environ.get("ARC_PF_FOLLOWER", "./.claude/worktrees/"
                                                        "sft-prune/dist/sft4x-opdS-g1-v6/src/arena/weights_follow.py"),
                                         "/root/weights_follow.py", copy=True)


def header(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n))


def folder_check(folder):
    """Every indexed tensor in a readable shard header; the config's expert count matches the fused expert tensors."""
    folder = Path(folder)
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((folder / "config.json").read_text())["text_config"]
    missing, shapes = [], {}
    for shard in sorted(set(index.values())):
        h = header(folder / shard)
        shapes.update({k: v["shape"] for k, v in h.items() if k != "__metadata__"})
    missing = [k for k in index if k not in shapes]
    experts = {k: v[0] for k, v in shapes.items() if k.endswith(("experts.gate_up_proj", "experts.down_proj"))
               and not k.startswith("mtp")}
    bad = sorted({v for v in experts.values()} - {config["num_experts"]})
    return (not missing and not bad and bool(experts),
            {"tensors": len(index), "missing": missing[:5], "expert_tensors": len(experts),
             "num_experts": config["num_experts"], "expert_rows_off_config": bad,
             "layers": config["num_hidden_layers"], "top_k": config["num_experts_per_tok"]})


def resume_check(resume, student, world=2):
    import torch

    config = json.loads((Path(student) / "config.json").read_text())["text_config"]
    local = config["num_experts"] // world
    info, ok = {}, True
    for rank in range(world):
        state = torch.load(Path(resume) / f"optim-rank{rank}.pt", map_location="cpu", mmap=True, weights_only=False)
        rows = state["rows"]
        sizes = [len(r) for r in rows["rows"]]
        good = (rows["t"] >= 1 and len(rows["rows"]) == len(rows["m"]) == len(rows["v"])
                and all(m.shape[0] == n and v.shape == m.shape for m, v, n in zip(rows["m"], rows["v"], sizes))
                and all(len(set(r.tolist())) == len(r) and (not len(r) or int(r.max()) < local) for r in rows["rows"]))
        adam = state["adam"]
        group = sum(len(g["params"]) for g in adam["param_groups"])  # state only for parameters that had gradients
        good = good and 0 < len(adam["state"]) <= group
        info[f"rank{rank}"] = {"t": rows["t"], "expert_tensors": len(sizes), "rows_min_max": [min(sizes), max(sizes)],
                               "local_experts": local, "adam_states": len(adam["state"]), "adam_params": group}
        ok = ok and good
    return ok, info


# reserved 8 cores, 16 ceiling: a token request gets only spare host CPU (g1i-pf2 1.0, pf3 2.1 cores/rank)
PF_CPU = (8, 16)
PF_WINDOW = "1024"  # preflight windows (preflights <= 30 min; 4K windows on CPU took ~1.5 h a round)


def shrink(src, dst, min_tokens=2048, replies=2, max_tokens=12288):
    """A short copy of a teacher file for the CPU preflight (<= 30 min): the first ``min_tokens``
    tokens, extended to the end of its ``replies``-th reply span (training and the loop cut need reply positions), at
    most ``max_tokens``; never split inside an image (pixel_values / image_grid_thw cut to the whole images kept)."""
    import torch

    d = torch.load(src, map_location="cpu", weights_only=True)
    ids = d["ids"]
    total = int(ids.numel())
    cut = min(min_tokens, total)
    reply = d.get("reply")
    if reply is not None and bool(reply.any()):
        r = reply.bool().int()
        starts = ((r[1:] - r[:-1]) == -1).nonzero().flatten() + 1  # first token after each reply span
        ends = starts.tolist() + ([total] if bool(r[-1]) else [])
        if ends:
            cut = max(cut, min(ends[min(replies, len(ends)) - 1], max_tokens))
    cut = min(cut, total)
    if cut >= total:
        dst.symlink_to(src)
        return total
    img = d.get("image")
    n_img, rows = 0, 0
    if img is not None and bool(img.any()):
        img = img.bool()
        while 0 < cut < total and bool(img[cut - 1]) and bool(img[cut]):  # inside an image: back to its start
            cut -= 1
        if "image_grid_thw" in d:
            per = d["image_grid_thw"].prod(-1)
            merge = max(1, int(per.sum()) // max(1, int(img.sum())))
            cum = torch.cumsum(per // merge, 0)
            kept = int(img[:cut].sum())
            n_img = int((cum <= kept).sum())
            if n_img and int(cum[n_img - 1]) != kept:
                raise ValueError(f"{src.name}: image tokens {kept} do not end on an image ({cum[:n_img + 1].tolist()})")
            rows = int(per[:n_img].sum())
    out = {}
    for k, v in d.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
        elif k in ("pixel_values", "image_grid_thw") and not n_img:
            continue  # no whole image kept: a text-only file (empty image tensors break the vision path, opd-v37)
        elif k == "pixel_values":
            out[k] = v[:rows]
        elif k == "image_grid_thw":
            out[k] = v[:n_img]
        elif v.dim() >= 1 and v.shape[0] == total:
            out[k] = v[:cut].clone()
        elif v.dim() >= 2 and v.shape[0] == 1 and v.shape[1] == total:
            out[k] = v[:, :cut].clone()
        else:
            out[k] = v
    torch.save(out, dst)
    return cut


READ_CUT = {"min_tokens": 2048, "replies": 1, "max_tokens": 12288}  # held-out reads: 1 file, to the end of its first reply (the reply reads and class split need one; dh10 opens with ~8K tokens of prompt; at most 12K; 22 Oct)


def shrunk_dir(folder, out, n, **kw):
    """The ``n`` smallest teacher files of ``folder`` whose shrunk copy keeps reply positions (a held-out read's file
    with none tests nothing: dh10 k006 kept 0 in 12K, 22 Oct), shrunk, in ``out``; returns {name: tokens kept}."""
    import torch

    out.mkdir(parents=True, exist_ok=True)
    got = {}
    for f in sorted(Path(folder).glob("*.pt"), key=lambda f: f.stat().st_size):
        if len(got) >= n:
            break
        kept = shrink(f, out / f.name, **kw)
        if bool(torch.load(out / f.name, map_location="cpu", weights_only=True)["reply"][1:].any()):
            got[f.name] = kept
        else:
            (out / f.name).unlink()
    return got


def _cuda():
    import torch
    return torch.cuda.is_available()


def _preflight(args, kw, data, general):
    import shutil
    import subprocess

    res = {"checks": {}, "ok": True}
    nproc = int(kw.get("cards") or 2)  # the launch's ranks (22 Oct audit: every count below derives from the launch)

    def check(name, ok, info):
        res["checks"][name] = {"ok": bool(ok), **info}
        res["ok"] = res["ok"] and bool(ok)

    for key in ("student", "frozen_root"):
        try:
            check(key, *folder_check(kw[key]))
        except Exception as exc:  # noqa: BLE001
            check(key, False, {"error": f"{type(exc).__name__}: {exc}"[:400]})
    if kw.get("resume"):
        try:
            check("resume_files", *resume_check(kw["resume"], kw["student"], world=nproc))  # the launch's ranks (opd-r1b
            # resumed a 2-rank D on 4 cards: optim-rank2.pt missing, 22 Oct)
        except Exception as exc:  # noqa: BLE001
            check("resume_files", False, {"error": f"{type(exc).__name__}: {exc}"[:400]})
    root = Path("/tmp/pf")
    env = {**os.environ, "PYTHONPATH": "/opt/heal", "HEAL_NO_FAST_ATTENTION": "1", "HEAL_EXPERTS": "eager",
           "PYTHONUNBUFFERED": "1", "HEAL_ROUTE_CACHE": "/tmp/pf/routecache"}
    pairs = lambda s: dict(x.strip().partition("=")[::2] for x in (s or "").split(",") if x.strip())  # noqa: E731
    env.update(pairs(kw.get("heal_env")))  # the launch's trainer flags (ARC_HEAL_ENV) on every preflight process
    # Modal sets OMP/MKL_NUM_THREADS=1 (no CPU reserved) while ~17 cores are usable: torch ran 1 thread per rank and
    # round A took 63 min (opd-v31b, 22 Oct); give each rank its share
    threads = str(max(1, min(16, len(os.sched_getaffinity(0))) // nproc))
    env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, HEAL_THREADS=threads)  # + torch.set_num_threads
    READ_N = nproc if env.get("HEAL_SHARD_READS") == "1" else 1  # sharded reads: one real file per rank
    env["HEAL_PROBE_WINDOWS"] = "4"  # the general drift probe on 4 windows (<= 30 min)
    if env.get("HEAL_HELDOUT2"):  # its smallest file only: one whole-game dh10 file on the tiny CPU model outran 6,000 s
        # (opd-v31, 22 Oct; 20 files did too, opd-v28): the read path runs, on the least data
        res.setdefault("shrunk", {})["heldout2"] = shrunk_dir(env["HEAL_HELDOUT2"], root / "heldout2", READ_N, **READ_CUT)
        env["HEAL_HELDOUT2"], env["HEAL_HELDOUT2_FILES"] = str(root / "heldout2"), str(READ_N)
    if env.get("HEAL_SAMPLED"):  # the preflight's smallest files may be base-sampled: every file takes the sampled path
        env["HEAL_STUDENT_PREFIX"] = ""
    if env.get("HEAL_SHARD_READS") == "1":  # sharded reads: each read's first call is compared with the every-rank read
        env["HEAL_SHARD_CHECK"] = "1"
    p = subprocess.run([sys.executable, "/opt/heal/heal_kd.py", "preflight-student", "--out", str(root)],
                       capture_output=True, text=True, env=env, cwd="/opt/heal")
    check("tiny_student", p.returncode == 0, {"stderr": p.stderr[-1500:] if p.returncode else ""})
    if p.returncode:
        return res
    files = sorted(Path(data).glob("*.pt"), key=lambda f: f.stat().st_size)
    # the smallest real files, 2 lockstep slots per round (round A checkpoints after its first slot, B resumes past it):
    # 2 files per rank (a 4-rank preflight with 4 files ran one slot and never checkpointed, s0-b 21 Oct)
    # up to 3 slots (opd-r1 gate, 22 Oct): with auto steps the schedule's slack step left B a general window only
    # (kd_reply 0), so resume of the reply/sampled path went untested; now 3 steps (own, general, own), checkpoint
    # after step 2. Round-1 data: only 8 of 67 files have a reply in their first 12K (s- files start ~45-110K deep)
    got = shrunk_dir(data, root / "teacher-a", 3 * nproc)  # files with reply positions in their cut
    res.setdefault("shrunk", {})["teacher-a"] = got
    picks = [f for f in files if f.name in got]
    if len(got) < 2 * nproc:  # own, general, own needs 2 lockstep slots of reply files
        check("teacher_a_slots", False, {"reply_files": len(got), "need": 2 * nproc})
        return res
    if general:  # 4 probe windows + 4 trained (the general mix path), 512 tokens each
        (root / "general").mkdir()
        for f in sorted(Path(general).glob("*.pt"))[:4 + 3 * nproc]:  # + one window per rank per general step
            shrink(f, root / "general" / f.name, min_tokens=512, replies=0, max_tokens=512)
    if "--heldout" in args:  # the launch's held-out read on its smallest file, shrunk
        res["shrunk"]["heldout"] = shrunk_dir(args[args.index("--heldout") + 1], root / "heldout", READ_N, **READ_CUT)
    res["files"] = {"teacher": [f"{f.name}:{f.stat().st_size >> 20}MiB" for f in picks], "pool": len(files)}

    def round_args(student, teacher, out, resume, step0=None, skip=""):
        a, i = list(args), 0
        swap = {"--student": student, "--frozen-root": str(root / "tiny-student"), "--teacher": teacher, "--out": out,
                "--general": str(root / "general"), "--batch-tokens": "1", "--seconds": "100000",
                "--total-steps": "3", "--window": PF_WINDOW,  # own, general, own: B resumes after step 2 (22 Oct)
                "--heldout": str(root / "heldout"), "--heldout-files": str(READ_N)}
        # total steps auto: the launch's count does not fit the tiny data (s0-b2: 20 refused on 7 staged); the real
        # run's schedule check runs at its own start. Window 4K, no time cap (deviation, 21 Oct): on CPU with the
        # reference GDN kernels a 16K window outran --seconds 1800 before step 1 (s1-noise/s3-read: 5,588 s, no step)
        # a launch that resumes mid-run (c7d: --step0 6) still starts the tiny rounds at 0, unless a resume says so
        swap["--step0"] = str(step0 if step0 is not None else 0)
        while i < len(a):
            if a[i] in swap:
                a[i + 1] = swap[a[i]]
            if a[i] in ("--resume", "--routing", "--skip"):
                del a[i:i + 2]
                continue
            i += 1
        # one checkpoint after step 1 only (opd-v36, 22 Oct: per-second checkpoints kept only A's last, at its final
        # step, so round B resumed with nothing left to train)
        while "--checkpoint-at" in a:
            j = a.index("--checkpoint-at")
            del a[j:j + 2]
        if "--checkpoint-seconds" in a:
            a[a.index("--checkpoint-seconds") + 1] = "0"
        else:
            a += ["--checkpoint-seconds", "0"]
        a += ["--checkpoint-at", "2"]  # after the general step: B's first step is a reply slot (22 Oct)
        return a + (["--resume", resume] if resume else []) + (["--skip", skip] if skip else [])

    # PF_ONLY_SMOKE: rounds A / B / reload already passed for this launch (their log is kept); only the smoke path runs
    # a launch that saves nothing and never checkpoints (an lr probe) has no resume path: a stopped one reruns in full
    no_resume = "--no-save" in args and "--checkpoint-seconds" not in args and "--checkpoint-at" not in args
    if no_resume and not kw.get("pf_only_smoke"):
        check("round_b", True, {"skipped": "no save and no checkpoints in the launch: no resume path"})
    for tag in (() if kw.get("pf_only_smoke") else ("a",) if no_resume else ("a", "b")):
        if tag == "a":
            a = round_args(str(root / "tiny-student"), str(root / "teacher-a"), str(root / "a"), "")
        else:  # the restart path: round A's mid-round checkpoint, its consumed files skipped
            ck = sorted((root / "a").glob("ckpt-*"))
            if not ck or not (ck[-1] / "progress.json").exists():
                check("checkpoint_a", False, {"found": [c.name for c in ck]})
                return res
            step0 = json.loads((ck[-1] / "progress.json").read_text())["step"]
            start = ck[-1] / "student" if (ck[-1] / "student").is_dir() else root / "tiny-student"  # sharded: base + shards
            a = round_args(str(start), str(root / "teacher-a"), str(root / "b"), str(ck[-1]), step0,
                           str(ck[-1] / "progress.json"))
        import time
        began = time.time()
        # HEAL_COMPILE on CPU: inductor's first step took 5,917 s on the tiny model (s24-read, 21 Oct) -> 18,000 s
        ceiling = 18000 if pairs(kw.get("heal_env")).get("HEAL_COMPILE") else 6000
        try:
            p = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(nproc),
                                "/opt/heal/heal_kd.py", *a], capture_output=True, text=True, env=env, cwd="/opt/heal",
                               timeout=ceiling)  # crash ceiling (no time limits); rounds take 1,000-2,400 s
        except subprocess.TimeoutExpired as exc:  # a hang shows as a failed check, not a lost preflight
            p = subprocess.CompletedProcess(exc.cmd, "timeout", "", (exc.stderr or b"").decode(errors="ignore")
                                            if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
        ev = root / tag / "events.jsonl"
        events = [json.loads(line) for line in open(ev)] if ev.exists() else []
        steps = [e for e in events if e.get("event") == "step"]
        ok = (p.returncode == 0 and steps and all(e.get("finite") for e in steps)
              and any(e.get("event") == "round_done" for e in events)
              and ("--no-save" in a or all((root / tag / f"optim-rank{r}.pt").exists() for r in range(nproc)))
              and ("--heldout" not in a or any(e.get("event") == "heldout" for e in events)))
        marks = {e.get("event"): e for e in events if e.get("event") in ("checkpoint", "skipped")}
        if tag == "a":
            ok = ok and "checkpoint" in marks
            if pairs(kw.get("heal_env")).get("HEAL_CLASS_SPLIT") == "1" and "--heldout" in a:  # class-split reads
                ho = next((e for e in events if e.get("event") == "heldout"), {})
                split_ok = "by_class" in ho and any(e.get("event") == "heldout_before" for e in events)
                check("class_split", split_ok, {"by_class": ho.get("by_class")})
                ok = ok and split_ok
            import torch
            cached = sorted((root / "routecache").glob("*.pt"))  # the route-count cache was written and matches
            same = (len(cached) == 1 and (root / "a" / "routing.pt").exists()
                    and torch.equal(torch.load(cached[0]), torch.load(root / "a" / "routing.pt")))
            check("route_cache", same, {"files": [c.name for c in cached]})
            ok = ok and same
            if "--truncate-loops" in a:  # the loop cut ran on real reply spans
                ok = ok and steps[-1].get("loop_spans", 0) > 0
        else:
            ok = ok and marks.get("skipped", {}).get("files", 0) >= 1 and steps[0]["step"] > 1
            # the resumed reply path (opd-r1 gate, 22 Oct): B trains a reply step, equal to A's same step on CPU
            # (KD, reply KD, and with HEAL_OPD_LOG the trained objective and sampled-reverse estimate, rel 1e-3)
            rel = lambda x, y: abs(x - y) / max(abs(x), abs(y), 1e-9)  # noqa: E731
            # B's first step is computed before any update, on the checkpoint's weights: equal to A's (the re-warm lr
            # only acts after it)
            first = steps[0]
            a_same = next((e for e in steps_a if e["step"] == first["step"]), {})  # noqa: F821 - bound before the call (ruff sees the later del / the loop's first pass)
            keys = ["kd", "kd_reply"] + (["opd_objective", "opd_sampled_rev"] if env.get("HEAL_OPD_LOG") == "1" else [])
            same = {k: (a_same.get(k), first.get(k)) for k in keys}
            match = all(x is not None and y is not None and rel(x, y) <= 1e-3 for x, y in same.values())
            reply_ok = bool(first.get("kd_reply"))  # the reply (and sampled) path ran in the resumed step
            check("resume_reply", match and reply_ok, {"step": first["step"], "a_vs_b": same})
            ok = ok and match and reply_ok
        if tag == "a":
            steps_a = steps
        check(f"round_{tag}", ok, {"args": a, "steps": steps[:3], "marks": marks, "returncode": p.returncode,
                                   "seconds": round(time.time() - began), "stdout": "" if ok else (p.stdout or "")[-3000:],
                                   "stderr_tail": "" if ok else "\n".join((p.stderr or "").splitlines()[-40:]),
                                   "events": sorted({e.get("event") for e in events}),
                                   "stderr": "\n".join([l for l in p.stderr.splitlines() if "Traceback" in l or l.startswith(("  File", "    ")) or "Error" in l or "Exit" in l or "schedule" in l or "resume" in l][-40:] + p.stderr.splitlines()[-8:]) if not ok else ""})
        if not ok:
            return res
    code = ("import sys, os, glob, torch, heal_kd\n"
            "ck, base_path = sys.argv[1], sys.argv[2]\n"
            "base, _ = heal_kd.load_student(base_path, 'cpu')\n"
            "b = dict(base.named_parameters())\n"
            "if os.path.isdir(ck):\n"
            "    m, _ = heal_kd.load_student(ck, 'cpu')\n"
            "    heal_kd.restore_fp32(m, ck)\n"
            "    t = dict(m.named_parameters())\n"
            "else:  # a sharded checkpoint (v25): every rank's fp32 shards, dim-0 pieces in rank order\n"
            "    shards = [torch.load(f, map_location='cpu', weights_only=True)\n"
            "              for f in sorted(glob.glob(os.path.join(os.path.dirname(ck), 'params-rank*.pt')))]\n"
            "    t = {k[len('model.'):] if k.startswith('model.') else k: torch.cat([x[k] for x in shards], 0)\n"
            "         for k in shards[0] if not k.endswith('::experts')}\n"
            "finite = all(bool(torch.isfinite(p).all()) for p in t.values())\n"
            "same = [k for k in t if k in b and b[k].shape == t[k].shape]\n"
            "moved = sum(int(not torch.equal(t[k].float(), b[k].float())) for k in same)\n"
            "print({'tensors': len(t), 'compared': len(same), 'finite': finite, 'moved': moved})\n"
            "sys.exit(0 if finite and moved and same else 1)\n")
    p = subprocess.CompletedProcess([], 0, "skipped (PF_ONLY_SMOKE)", "") if kw.get("pf_only_smoke") else subprocess.run(
        [sys.executable, "-c", code, str(root / "b" / "student") if "--no-save" not in args else  # an lr probe saves no
         str(sorted((root / "a").glob("ckpt-*"))[-1] / "student"), str(root / "tiny-student")],  # student: A's checkpoint
        capture_output=True, text=True, env=env, cwd="/opt/heal")
    check("reload", p.returncode == 0, {"out": p.stdout.strip()[-300:], "stderr": p.stderr[-800:] if p.returncode else ""})
    if kw.get("diag_window") and res["ok"]:  # heal_round's diagnostics path: gradient shards + profile, then cosine
        import time as _t
        shards, outs = root / "gradshard", []
        for i, (win, extra) in enumerate(((None, {"HEAL_GRAD_SHARD_DUMP": str(root / "gradshard"),
                                                   "HEAL_PROFILE": kw.get("profile") or ""}),
                                          (kw["diag_window"], {"HEAL_GRAD_COSINE": str(root / "gradshard"), **pairs(kw.get("diag_env"))}))):
            a = round_args(str(root / "tiny-student"), str(root / "teacher-a"), str(root / f"diag{i}"), "")
            a[a.index("--window") + 1] = PF_WINDOW if win is None else str(2 * int(PF_WINDOW))  # tiny model: short windows
            began = _t.time()
            try:
                q = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(nproc),
                                    "/opt/heal/heal_kd.py", *a], capture_output=True, text=True,
                                   env={**env, **{k: v for k, v in extra.items() if v}}, cwd="/opt/heal", timeout=6000)
            except subprocess.TimeoutExpired:
                q = subprocess.CompletedProcess([], "timeout", "", "")
            ev = root / f"diag{i}" / "events.jsonl"
            outs.append(([json.loads(x) for x in open(ev)] if ev.exists() else [], q, round(_t.time() - began)))
        e0, e1 = outs[0][0], outs[1][0]
        cosr = next((e for e in e1 if e.get("event") == "grad_cosine"), None)
        prof = next((e for e in e0 if e.get("event") == "profile"), None)
        ok = (outs[0][1].returncode == 0 and outs[1][1].returncode == 0
              and any(e.get("event") == "grad_shards_saved" for e in e0) and cosr is not None
              and -1.0 <= cosr["cosine"] <= 1.0 and (not kw.get("profile") or not _cuda() or (prof is not None and prof.get("kernels", 0) > 0
                                                and 0 < prof.get("busy_s", 0) <= prof["wall_s"])))
        check("diag", ok, {"cosine": cosr, "profile": prof, "seconds": [o[2] for o in outs],
                           "stderr": "" if ok else "\n".join((outs[0][1].stderr or "").splitlines()[-20:]
                                                              + (outs[1][1].stderr or "").splitlines()[-20:])})
    if (kw.get("smoke") or kw.get("gate")) and res["ok"]:  # GATE=1: the round's first phase is run_smoke  # the launcher's own run_smoke (A / B / C) on the tiny student, real files
        import ast
        tree = ast.parse(Path("/root/modal_rl1.py").read_text())
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("run_smoke", "heal_train_args")]
        scope = {"json": json, "Path": Path, "os": os}
        exec(compile(ast.Module(body=fns, type_ignores=[]), "modal_rl1.py", "exec"), scope)
        sm = root / "smoke"
        (sm / "teacher").mkdir(parents=True)
        n_smoke = int(kw.get("smoke_steps") or 3)  # A's steps, C's 2 more, one slot of slack, each one file per rank
        want = max(10, (n_smoke + 3) * nproc)
        got_sm = shrunk_dir(data, sm / "teacher", want)  # reply files first (round-1 data has 8), then any to fill
        for f in [f for f in files if f.name not in got_sm][:want - len(got_sm)]:
            shrink(f, sm / "teacher" / f.name)

        def torchrun(kw2, out_dir):
            k = dict(kw2, frozen_root=str(root / "tiny-student"), batch_tokens=1)
            if k.get("heldout") and (root / "heldout").is_dir():  # the shrunk held-out file (<= 30 min preflight)
                k["heldout"], k["heldout_files"] = str(root / "heldout"), READ_N
            if k["student"] == kw["student"]:  # the real student and its optimizer do not fit the tiny model
                k["student"], k["resume"] = str(root / "tiny-student"), ""
            a = scope["heal_train_args"](k, teacher=str(sm / "teacher"), out=out_dir, general=str(root / "general"))
            if "--window" in a:  # orchestration only: eager attention on 48 GB cards took 1,394 s a 32K-window step
                a[a.index("--window") + 1] = PF_WINDOW
            return [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(nproc), "/opt/heal/heal_kd.py", *a]

        v = scope["run_smoke"](torchrun, kw, sm, env, 10800)  # 30 min a process
        tails = {n: "\n".join((sm / f"{n}.log").read_text(errors="ignore").splitlines()[-15:])
                 for n in ("smoke-a", "smoke-b", "smoke-b2", "smoke-c") if (sm / f"{n}.log").exists()}
        n = int(kw.get("smoke_steps") or 3)
        ok = v["ok"] and v["steps"] == [n, 1, 1, 2]  # A n steps, B and B2 redo step n, C 2
        check("smoke_orchestration", ok, {**{k: v.get(k) for k in ("exit", "steps", "exact_resume", "a_step3", "b_step3",
                                                                    "b2_step3", "grad_norm_noise", "memory_ok", "lr_a")}, "logs": {} if ok else tails})
    shutil.rmtree(root, ignore_errors=True)
    return res


PF_CHUNK = 256  # stream preflight pieces (the run: 32K; 256 on the shortened rollouts: g1i-pf7 got 2 pieces at 1024)


PF_TEXT = (2000, 300)  # stand-in rollouts: system / other text kept (chars); g1i-pf2 stream steps on 7-14K-token contexts took 715-1,600 s


def _shorten(src, dst, count=None):
    """A copy of a real rollout.jsonl with every text cut to PF_TEXT and only its first image kept (pf4 profile,
    22 Oct: tiny-model CPU time scales with context tokens). Rows are message deltas, so each message is cut once
    and the chain stays a prefix chain; the reply and its later history copy get the same cut."""
    seen = [False]

    def cut(m, lim):
        m = dict(m)
        for key in ("reasoning", "reasoning_content"):
            if isinstance(m.get(key), str):
                m[key] = m[key][:PF_TEXT[1]]
        c = m.get("content")
        if isinstance(c, str):
            m["content"] = c[:lim]
        elif isinstance(c, list):
            parts = []
            for q in c:
                if q.get("type") == "text":
                    parts.append({**q, "text": q.get("text", "")[:PF_TEXT[1]]})
                elif not seen[0]:
                    seen[0] = True
                    parts.append(q)
            m["content"] = parts or [{"type": "text", "text": "(frame)"}]
        return m

    rows = []
    for line in Path(src).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["messages"] = [cut(m, PF_TEXT[0] if m.get("role") == "system" else PF_TEXT[1]) for m in r.get("messages") or []]
        if isinstance(r.get("reply"), dict):
            r["reply"] = cut(r["reply"], PF_TEXT[1])
        rows.append(json.dumps(r) + "\n")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text("".join(rows))
    if count is not None:  # the served prompt count follows the cut (the encoder drops a request whose count differs)
        import sys as _s
        _s.path.insert(0, "/opt/heal")
        import heal_stream
        got = {c["request"]: count(c) for c in heal_stream.SessionTail(dst).read()}
        fixed = []
        for line in rows:
            r = json.loads(line)
            if r.get("n") in got and isinstance(r.get("usage"), dict):
                r["usage"] = {**r["usage"], "prompt_tokens": got[r["n"]]}
            fixed.append(json.dumps(r) + "\n")
        Path(dst).write_text("".join(fixed))
    return Path(dst)


def _err_tail(stderr, n=40):
    """A failed torchrun's own error lines (not just its ChildFailedError summary): rank tracebacks + the last lines."""
    lines = (stderr or "").splitlines()
    keep = [l for l in lines if ("Traceback" in l or "Error" in l or "Exception" in l or "/opt/heal/" in l
                                 or l.lstrip("[]rank0123456789: ").startswith(("raise ", "assert ")))
            and "torch/distributed/" not in l]
    return "\n".join(keep[-n:] + ["..."] + lines[-8:])


def _stream_rows(src, n_tokens, encode, out, split=3):
    """A real rollout.jsonl cut to its first gameplay requests (about ``n_tokens`` in its last request), each row given
    the fields the streaming harness adds (the sampled ids = the re-encoded reply, an engine log p, a weights
    version); written as two halves (``out``.a / .b: the play before and after the restart, which falls after
    gameplay request ``split``)."""
    import sys as _s

    _s.path.insert(0, "/opt/heal")
    import heal_stream

    rows, kept, half, extra = [], [], 0, 0
    base = "base-pf" in str(out)  # a base play records no sampled ids (the teacher's canonical path, 8-token rule)
    tail = heal_stream.SessionTail(src)
    lines = [json.loads(x) for x in Path(src).read_text().splitlines() if x.strip()]
    tail_rows = tail.read()
    by_n = {c["request"]: c for c in tail_rows}
    for row in lines:
        rows.append(row)
        cap = by_n.get(row.get("n"))
        if cap is None:
            continue
        made = encode(cap)
        if made is None:
            continue
        inputs, start = made
        ids = inputs["input_ids"][0]
        if extra:  # the server holds the earlier non-canonical sample: its counts run over the re-encoding
            row["usage"] = {**row["usage"], "prompt_tokens": row["usage"]["prompt_tokens"] + extra}
        tokens = ids[start:].tolist()
        if not base and len(kept) == 1 and len(tokens) >= 2:  # a non-canonical sample (one id more than the
            tokens, extra = tokens[:1] + tokens, extra + 1  # re-encoding), as the pruned student makes (23 Oct)
        if base:
            row.update(version=len(kept) // 2)
        else:
            row.update(tokens=tokens, logprobs=[-0.7] * len(tokens), version=len(kept) // 2)
        kept.append(row)
        if len(kept) == split:
            half = len(rows)  # the restart falls after gameplay request ``split``
        if len(kept) >= split + 2 and ids.shape[0] >= n_tokens:
            break
    if len(kept) < split + 1:  # too few for the split: halve the rows
        half = max(1, len(rows) // 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.a").write_text("".join(json.dumps(r) + "\n" for r in rows[:half]))
    Path(f"{out}.b").write_text("".join(json.dumps(r) + "\n" for r in rows[half:]))
    return {"rows": len(rows), "gameplay": len(kept)}


def _exporter_keys_check(events, mon, cards):
    """the dashboard exporter (launch/pth/train_export.lines) on round A's own step events
    carries every REQUIRED key. GPU util / VRAM come from the round's GPU sampler (none on CPU): one sampler-format
    row per step stands in, so only their schema is checked here (the 10-min launch check reads the real ones)."""
    import shutil

    try:
        import train_export as te

        run = mon / "pf"
        run.mkdir(parents=True, exist_ok=True)
        shutil.copy(events, run / "train_events.jsonl")
        steps = [json.loads(x) for x in open(events) if '"step"' in x]
        with open(run / "gpu.jsonl", "w") as f:
            for e in steps:
                if e.get("event") == "step" and isinstance(e.get("t"), (int, float)):
                    f.write(json.dumps({"epoch": 1.79e9 + e["t"], "util": [0.0] * cards, "mem_gib": [0.0] * cards}) + "\n")
        te.MON = mon
        rows = te.lines("pf")
        missing, dead = te.missing_keys(rows), te.dead_keys(rows)  # dead: present but 0/NaN on every row (c7m)
        return not missing and not dead, {"missing": missing, "dead": dead, "rows": len(rows)}
    except Exception as exc:  # noqa: BLE001 - reported as the failed check
        return False, {"error": f"{type(exc).__name__}: {exc}"[:400]}


def _stream_preflight(args, kw, segments, general, tag=""):
    """The streaming round's trainer path on CPU: the launch's own ``train`` arguments on the tiny
    student; the teacher stream (heal_stream + rl1_trainer.stream_forward on the tiny teacher, the real processor) over
    real rollouts given the streaming fields; round A trains 3 steps from the stream with weight deltas and a checkpoint;
    then the play continues, a restarted teacher scores only the new tokens and round B resumes from A's checkpoint."""
    import shutil
    import subprocess
    import time

    import torch

    sys.path.insert(0, "/opt/heal")
    res = {"checks": {}, "ok": True}
    # the round's own card layout (modal_rl1.stream_layout, ARC_HEAL_TRANKS): its trainer ranks run here on gloo
    try:
        tr, tdev, sdev, spread = _round_fn("stream_layout")(int(kw.get("cards") or 2), kw.get("tranks") or 0)
        teacher_extra = _round_fn("stream_teacher_env")(spread, kw.get("teacher_env") or "")
        layout = {"ok": True, "trainer_ranks": tr, "trainer_devices": tdev, "teacher_devices": sdev,
                  "teacher_spread": spread, "teacher_env": teacher_extra}
    except Exception as exc:  # noqa: BLE001 - reported as the failed check
        layout = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
    res["checks"]["layout"] = layout
    if not layout["ok"]:
        res["ok"] = False
        return res
    nproc = tr
    torch.set_num_threads(max(1, min(8, len(os.sched_getaffinity(0)))))  # the in-process teacher (Modal sets OMP=1)

    def check(name, ok, info):
        res["checks"][name] = {"ok": bool(ok), **info}
        res["ok"] = res["ok"] and bool(ok)

    # the launch's own schedule (c3, 23 Oct: --total-steps 0 gave a stream round 1 step; the tiny rounds below set
    # their own 3, which hid it): explicit, and its decay completes at the last step
    import heal_kd
    la = list(args)
    total = int(la[la.index("--total-steps") + 1]) if "--total-steps" in la else 0
    sched = la[la.index("--schedule") + 1] if "--schedule" in la else "wsd"
    why = heal_kd.stream_schedule_problem(total, sched, {**os.environ, **dict(x.strip().partition("=")[::2] for x in (kw.get("heal_env") or "").split(",") if x.strip())})
    check("schedule", why is None, {"total_steps": total, "schedule": sched, "error": why})
    if not res["ok"]:
        return res
    # every real path the launch names, resolved the way the round resolves it (opdS-g1e, 22 Oct: the tiny student has
    # no prune, so the delta exporter's prune spec for the real student was first resolved on the B300s and failed)
    import heal_delta
    real = {}
    try:
        cfg = json.loads((Path(kw["student"]) / "config.json").read_text())
        prune = (kw.get("stream_prune") or "").replace("/opt/arc-validation/rl1/", "/opt/heal/") or None
        cut = heal_delta.Cut(cfg, prune)
        n = (cfg.get("text_config") or cfg).get("num_experts")
        real["cut"] = {"prune": prune, "layers": len(cut.keep), "kept": sorted({len(v) for v in (cut.experts or {}).values()})}
        bad = [o for o in cut.keep if cut.experts and len(cut.experts.get(o, [])) != n]
        # the cut's recorded prune_file is the cutting container's path: a pruned student needs the launch to name one
        named = bool(prune) or not cfg.get("student_cut", {}).get("prune_file")
        real["cut_ok"] = named and not bad and (cut.experts is not None or not cfg.get("student_cut", {}).get("prune_file"))
        if not named:
            real["cut_error"] = "pruned student: ARC_HEAL_STREAM_PRUNE unset"
    except Exception as exc:  # noqa: BLE001 - reported as the failed check
        real["cut_ok"], real["cut_error"] = False, f"{type(exc).__name__}: {exc}"[:400]
    hpairs = dict(x.strip().partition("=")[::2] for x in (kw.get("heal_env") or "").split(",") if x.strip())
    paths = {"student": kw.get("student"), "resume": kw.get("resume"), "routing": kw.get("routing"),
             "teacher_model": kw.get("teacher_model"), "heldout2": hpairs.get("HEAL_HELDOUT2")}
    real["missing"] = [f"{k}={v}" for k, v in paths.items() if v and not Path(v).exists()]
    if kw.get("stream_base_fill"):  # the recorded base sessions the round copies in (a folder glob)
        import glob
        real["base_fill_sessions"] = len(glob.glob(kw["stream_base_fill"].rstrip("/") + "/rollout.jsonl"))
        if not real["base_fill_sessions"]:
            real["missing"].append(f"stream_base_fill={kw['stream_base_fill']}")
        else:  # the round's own copy (heal_stream.stream_fill), read back as the teacher tails it
            try:
                sys.path.insert(0, "/opt/heal")
                import heal_stream as _hs
                fill_root = Path("/tmp/pf-fill")
                shutil.rmtree(fill_root, ignore_errors=True)
                name = _hs.stream_fill(kw["stream_base_fill"], fill_root)
                copied = sorted((fill_root / name).glob("*/rollout.jsonl"))
                caps = len(_hs.SessionTail(copied[0]).read()) if copied else 0
                real["base_fill_copied"], real["base_fill_first_captures"] = len(copied), caps
                if len(copied) != real["base_fill_sessions"] or not caps:
                    real["missing"].append("stream_fill copy/read")
                shutil.rmtree(fill_root, ignore_errors=True)
            except BaseException as exc:  # noqa: BLE001 - stream_fill raises SystemExit on an empty glob
                real["missing"].append(f"stream_fill: {type(exc).__name__}: {exc}"[:300])
    check("real_paths", real["cut_ok"] and not real["missing"], real)
    try:
        check("quack_cache", *_quack_cache_check(Path("/tmp/pf-quack")))
    except Exception as exc:  # noqa: BLE001 - reported as the failed check
        check("quack_cache", False, {"error": f"{type(exc).__name__}: {exc}"[:400]})
    if "ARC_TEACHER_CACHE=1" in (kw.get("teacher_env") or ""):
        try:
            check("teacher_cache", *_teacher_cache_check(Path("/tmp/pf-tcache")))
        except Exception as exc:  # noqa: BLE001 - reported as the failed check
            check("teacher_cache", False, {"error": f"{type(exc).__name__}: {exc}"[:400]})
    if not res["ok"]:
        return res

    root = Path("/tmp/pfs")
    shutil.rmtree(root, ignore_errors=True)
    env = {**os.environ, "PYTHONPATH": "/opt/heal", "HEAL_NO_FAST_ATTENTION": "1", "HEAL_EXPERTS": "eager",
           "PYTHONUNBUFFERED": "1", "HEAL_ROUTE_CACHE": str(root / "routecache")}
    pairs = lambda s: dict(x.strip().partition("=")[::2] for x in (s or "").split(",") if x.strip())  # noqa: E731
    env.update(pairs(kw.get("heal_env")))
    for name, f in (("HEAL_START_READS", "start-reads.json"), ("HEAL_HELDOUT2_BEH", "heldout2-beh.pt")):
        if env.get(name):  # the run's caches stay the run's: the tiny student's go to the preflight's own folder
            env[name] = str(root / f)
    threads = str(max(1, min(16, len(os.sched_getaffinity(0))) // nproc))
    env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, HEAL_THREADS=threads, HEAL_STUDENT_PREFIX="s-")
    # the reads on the least data, as in _preflight (opdS-g1b-pf1, 22 Oct: 20 whole-game dh10 files in the routing
    # round's start reads outran the 6,000 s ceiling on the tiny CPU model, and pf_guard stopped g1f)
    read_n = nproc if env.get("HEAL_SHARD_READS") == "1" else 1
    env["HEAL_PROBE_WINDOWS"] = "4"
    if env.get("HEAL_HELDOUT2"):
        root.mkdir(parents=True, exist_ok=True)
        res.setdefault("shrunk", {})["heldout2"] = shrunk_dir(env["HEAL_HELDOUT2"], root / "heldout2", read_n, **READ_CUT)
        env["HEAL_HELDOUT2"], env["HEAL_HELDOUT2_FILES"] = str(root / "heldout2"), str(read_n)
    p = subprocess.run([sys.executable, "/opt/heal/heal_kd.py", "preflight-student", "--out", str(root)],
                       capture_output=True, text=True, env=env, cwd="/opt/heal")
    check("tiny_student", p.returncode == 0, {"stderr": p.stderr[-1500:] if p.returncode else ""})
    if p.returncode:
        return res
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    import flash_qlora_job as fq
    import heal_stream
    import rl1_trainer

    processor, tried = None, []
    for proc_dir in (kw.get("teacher_model"), kw["student"], kw.get("frozen_root"),
                     "/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540"):  # the BF16 teacher's folder
        if not proc_dir:
            continue
        try:
            processor = AutoProcessor.from_pretrained(proc_dir)
            break
        except Exception as exc:  # noqa: BLE001 - the next candidate
            tried.append(f"{proc_dir}: {type(exc).__name__}")
    check("processor", processor is not None, {"tried": tried})
    if processor is None:
        return res
    end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")  # the teacher's exact_inputs reply end
    encoded, _ = fq.example_maker(processor, [], tail=int(PF_WINDOW), phases=[int(PF_WINDOW)])
    import rl1_trainer as _rt  # the round's own slack (STREAM_PROMPT_SLACK) on the stand-in captures
    encode = lambda capture: encoded(capture, pair=True, device=None, slack=_rt.STREAM_PROMPT_SLACK)  # noqa: E731
    srcs = [f for f in sorted(Path(segments).glob("*/rollout.jsonl"), key=lambda f: f.stat().st_size)
            if sum(1 for x in f.open() if x.strip()) >= 8][:2]  # the smallest with a few gameplay requests (a restart half)
    if len(srcs) < 2:
        check("rollouts", False, {"segments": segments, "found": len(srcs)})
        return res
    base_share = float(pairs(kw.get("heal_env")).get("HEAL_STREAM_BASE_SHARE") or 0)
    plays = ["play-pf", "base-pf"] if base_share > 0 else ["play-pf", "play-pf"]  # the 2nd session: base play
    dests = {f.parent.name: root / "roll" / plays[i] / "t" / "games" / f.parent.name / "rollout.jsonl"
             for i, f in enumerate(srcs)}
    count = lambda c: fq.encode_request(processor, c)[1]["prompt_tokens"]  # noqa: E731
    # round A's pieces come from the play before the restart: it moves 2 requests later per rank above 2 (3 ranks x
    # S2 units drained the 10 pieces of a 3-request half before round A's step 3: tranks3-pf2, 23 Oct)
    made = {f.parent.name: _stream_rows(_shorten(f, root / "short" / f.parent.name / "rollout.jsonl", count), 4000,
                                        encode, dests[f.parent.name], split=3 + 2 * max(0, nproc - 2)) for f in srcs}
    check("rollouts", all(m["gameplay"] >= 4 for m in made.values()), {"made": made, "base_share": base_share})
    for g in made:
        shutil.copy(f"{dests[g]}.a", dests[g])
    base_plays = ("base-pf",) if base_share > 0 else ()
    teacher = AutoModelForMultimodalLM.from_pretrained(str(root / "tiny-teacher"), dtype=torch.float32).eval()
    image_id = processor.image_token_id
    k = int(kw.get("k") or 128) if "k" in kw else 128
    top_fn = lambda logits, nxt: (lambda g: {"ids": g["ids"], "logp": g["logp"], "at": g["at"][0]})(  # noqa: E731
        rl1_trainer.vocab_top(logits, k=k, at_ids=(nxt,)))
    fwd = rl1_trainer.stream_forward(teacher, 512, image_id, device="cpu", autocast=False)
    chunks = root / "chunks"

    def pump(t):
        for _ in range(10000):
            t._scan()
            while not t.events.empty():
                t._handle(t.events.get_nowait())
            chain, upto = t._due()
            if chain is None:
                for key in list(t.chains):  # the halves end: every open chain's remainder is scored
                    if not t.chains[key].closed:
                        t._handle(("idle", key, None, None))
                chain, upto = t._due()
                if chain is None:
                    return
            t.score(chain, upto)
        t.saver.shutdown(wait=True)

    os.environ.update(teacher_extra)  # the round's teacher flags (ARC_HEAL_TEACHER_ENV) on the stand-in teacher
    if os.environ.get("RL1_INDEXER", "vectorized") == "vectorized":  # the teacher process's own QSA indexer (c7j: its
        rl1_trainer.vectorized_indexer()  # tie order failed the GPU verify; the HF loop hid it here)
    rl1_trainer.TIES["mode"] = os.environ.get("RL1_QSA_TIES", "topk")
    began = time.time()
    st = heal_stream.StreamTeacher(root / "roll", chunks, encode, fwd, k=k, chunk=PF_CHUNK, image_token_id=image_id, end_id=end_id,
                                   idle=1e9, encoders=2, top_fn=top_fn, base=base_plays, base_share=base_share)
    pump(st)
    st.saver.shutdown(wait=True)
    multi = [c for c in st.chains.values() if len(c.files) >= 2]
    verify = st.verify(multi[0]) if multi else None
    # c7j (24 Oct): every student chain continued at capture offsets, against a fresh pass
    students = [st.verify(c) for c in multi if not st._is_base(c)]
    if verify is not None and students:
        verify = {**verify, "students": len(students), "student_top1_min": min(v["top1_agree"] for v in students),
                  "student_mean_abs": max(v["top_logp_mean_abs"] for v in students)}
    pieces_a = sorted(chunks.glob("s-*.ready"))
    based = sorted(chunks.glob("b-*.ready"))
    # every streamed request rebuilt as its server held it (exact_inputs; the stand-in student has a non-canonical
    # sample, so the counts after it run one over the re-encoding)
    exact_ok = not st.stats.get("inexact")
    check("teacher_stream_a", exact_ok and (not base_share or len(based) >= 2 * nproc) and len(pieces_a) >= 4 * nproc and verify is not None and verify["top1_agree"] >= 0.99
          and verify.get("students", 0) >= 1 and verify["student_top1_min"] >= 0.99 and verify["student_mean_abs"] <= 1e-3
          and verify.get("kl_mean") is not None and verify["kl_mean"] < 1e-3,  # KL path (HEAL_VERIFY_KL), FP16 floor
          {"pieces": len(pieces_a), "status": st.status(), "verify": verify, "seconds": round(time.time() - began)})
    if not res["ok"]:
        return res
    check("teacher_order", *_order_check(root, made, dests, encode, fwd, k, image_id, top_fn, end_id, base_share))
    if not res["ok"]:
        return res
    (root / "general").mkdir()
    if general:
        for f in sorted(Path(general).glob("*.pt"))[:4 + 3 * nproc]:
            shrink(f, root / "general" / f.name, min_tokens=512, replies=0, max_tokens=512)

    def round_args(student, out, resume=None, step0=None, skip="", total=3, routing=None, teacher=""):
        a, i = list(args), 0
        swap = {"--student": student, "--frozen-root": str(root / "tiny-student"), "--teacher": teacher, "--out": out,
                "--general": str(root / "general"), "--batch-tokens": "1", "--seconds": "100000",
                "--total-steps": str(total), "--window": PF_WINDOW}
        # a launch that resumes mid-run (c7d: --step0 6) still starts the tiny rounds at 0, unless a resume says so
        swap["--step0"] = str(step0 if step0 is not None else 0)
        while i < len(a):
            if a[i] in swap:
                a[i + 1] = swap[a[i]]
            if a[i] in ("--resume", "--routing", "--skip", "--heldout", "--heldout-files", "--checkpoint-at"):
                del a[i:i + 2]
                continue
            i += 1
        if "--checkpoint-seconds" in a:
            a[a.index("--checkpoint-seconds") + 1] = "0"
        a += ["--checkpoint-at", "2"] + (["--routing", routing] if routing else [])
        return a + (["--resume", resume] if resume else []) + (["--skip", skip] if skip else [])

    def train(tag, a, extra):
        t0 = time.time()
        try:
            q = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(nproc),
                                "/opt/heal/heal_kd.py", *a], capture_output=True, text=True, env={**env, **extra},
                               cwd="/opt/heal", timeout=6000)  # crash ceiling
        except subprocess.TimeoutExpired:
            q = subprocess.CompletedProcess([], "timeout", "", "")
        ev = root / tag / "events.jsonl"
        events = [json.loads(x) for x in open(ev)] if ev.exists() else []
        return q, events, round(time.time() - t0)

    # routing: the tiny student's ESFT selection from the first pieces (the run takes the starting student's)
    # r0 makes routing.pt only: 1 routing file, no reads (round a runs them; pf2 r0: routing 454 s, reads 651 s)
    noreads = {"HEAL_HELDOUT2": "", "HEAL_MID_READ_EVERY": ""}
    q, ev0, sec = train("r0", round_args(str(root / "tiny-student"), str(root / "r0"), total=1,
                                          teacher=str(chunks)) + ["--no-save", "--route-files", "1"], noreads)
    check("routing", q.returncode == 0 and (root / "r0" / "routing.pt").exists(),
          {"seconds": sec, "stderr": "" if q.returncode == 0 else (q.stderr or "")[-1500:]})
    if not res["ok"]:
        return res
    # the round's own trainer env (modal_rl1.stream_trainer_env): the deltas go to the real arc: bucket, a scratch run
    scratch = f"pfscratch-{(tag or 'pf').replace('/', '-')[-60:]}"
    senv = _round_fn("stream_trainer_env")({k: v for k, v in env.items() if k != "HEAL_DELTA_RUN"}, chunks=chunks, k=k, max_age=kw.get("stream_max_age", 6),
                                           deltas=root / "deltas", run_id=scratch, prune="")
    senv.pop("CUDA_VISIBLE_DEVICES", None)
    senv.update(HEAL_STREAM_STALL="300", HEAL_SAMPLED=env.get("HEAL_SAMPLED") or "1.0")
    senv.update(HEAL_SCHEDULE_STEPS="3")  # the tiny rounds' own count (the launch's is the "schedule" check)
    q, ev_a, sec = train("a", round_args(str(root / "tiny-student"), str(root / "a"),
                                         routing=str(root / "r0" / "routing.pt")) + ["--allow-unfinished-schedule"], senv)
    steps = [e for e in ev_a if e.get("event") == "step"]
    mans = sorted((root / "deltas").glob("v*/manifest.json"))
    ck = sorted((root / "a").glob("ckpt-*"))
    # student stream steps (a general-mix step carries neither stream chunks nor base: g1i-pf, 22 Oct)
    own = [e for e in steps if e.get("stream_chunks") and not e.get("stream_base")]
    ok = (q.returncode == 0 and len(steps) == 3 and all(e.get("finite") for e in steps) and own
          and all(e.get("age_median") is not None for e in own)
          and all((e.get("opd_sampled_positions") or 0) > 0 for e in own)
          and (not base_share or any(e.get("stream_base") for e in steps))
          and [json.loads(m.read_text())["version"] for m in mans] == [1, 2, 3]
          and all(json.loads(m.read_text()).get("sizes") for m in mans) and ck)
    check("round_a", ok, {"seconds": sec, "steps": [{x: e.get(x) for x in ("step", "kd_reply", "age_median", "age_max",
                                                                           "stream_chunks", "opd_sampled_positions", "stream_base")}
                                                    for e in steps], "deltas": [m.parent.name for m in mans],
                          "stderr": "" if ok else _err_tail(q.stderr)})
    if not ok:
        _purge(senv, scratch)
        return res
    check("exporter_keys", *_exporter_keys_check(root / "a" / "events.jsonl", root / "mon", tr))
    try:
        check("delta_bucket", *_delta_bucket_check(senv, scratch, root / "follow", [json.loads(m.read_text()) for m in mans],
                                                   sample=dests[next(iter(made))]))
    except Exception as exc:  # noqa: BLE001 - reported as the failed check
        check("delta_bucket", False, {"error": f"{type(exc).__name__}: {exc}"[:600]})
    if not res["ok"]:
        _purge(senv, scratch)
        return res
    # the play continues; the teacher restarts (the round resumed): only tokens after the restart are scored
    # the restarted teacher reads the play so far, then the play continues (g1i-pf2: the .b half appended before
    # skip_existing was skipped as already scored, 0 new pieces)
    st2 = heal_stream.StreamTeacher(root / "roll", chunks, encode, fwd, k=k, chunk=PF_CHUNK, image_token_id=image_id, end_id=end_id,
                                    idle=1e9, encoders=2, top_fn=top_fn, base=base_plays, base_share=base_share)
    st2.skip_existing()
    for g in made:
        with open(dests[g], "a") as f:
            f.write(Path(f"{dests[g]}.b").read_text())
    first_new = None
    pump(st2)
    st2.saver.shutdown(wait=True)
    new = sorted(set(chunks.glob("s-*.ready")) - set(pieces_a))
    rows0 = [torch.load(str(r).replace(".ready", ".pt"), weights_only=False)["row0"] for r in new]
    check("teacher_restart", len(new) >= nproc and min(rows0, default=0) > 0 and not st2.stats.get("inexact"),
          {"new_pieces": len(new), "skipped_tokens": st2.stats.get("skipped_tokens"), "row0_min": min(rows0, default=None),
           "inexact": st2.stats.get("inexact", 0)})
    step0 = json.loads((ck[-1] / "progress.json").read_text())["step"]
    start = ck[-1] / "student" if (ck[-1] / "student").is_dir() else root / "tiny-student"
    q, ev_b, sec = train("b", round_args(str(start), str(root / "b"), resume=str(ck[-1]), step0=step0,
                                         skip=str(ck[-1] / "progress.json"), total=4,
                                         routing=str(root / "r0" / "routing.pt")) + ["--allow-unfinished-schedule"], senv)
    steps_b = [e for e in ev_b if e.get("event") == "step"]
    mans = sorted((root / "deltas").glob("v*/manifest.json"))
    ok = (q.returncode == 0 and steps_b and steps_b[0]["step"] == step0 + 1 and all(e.get("finite") for e in steps_b)
          and any(json.loads(m.read_text())["version"] == 4 for m in mans)
          and any(e.get("event") == "delta_export" and e.get("start") for e in ev_b))  # the resumed weights first
    check("round_b", ok, {"seconds": sec, "steps": [{x: e.get(x) for x in ("step", "kd_reply", "age_median",
                                                                           "stream_chunks")} for e in steps_b],
                          "stderr": "" if ok else _err_tail(q.stderr)})
    if res["ok"]:
        _stream_live(res, check, root=root, made=made, dests=dests, plays=plays, base_plays=base_plays, encode=encode, end_id=end_id,
                     fwd=fwd, k=k, image_id=image_id, top_fn=top_fn, train=train, round_args=round_args, senv=senv,
                     base_share=base_share)
    _purge(senv, scratch)
    shutil.rmtree(root, ignore_errors=True)
    return res


def _round_fn(name):
    """A function from modal_rl1's own source (with the module constants it reads), as the round runs it."""
    import ast
    tree = ast.parse(Path("/root/modal_rl1.py").read_text())
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == name)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("NEBIUS_BUCKET", "NEBIUS_RCLONE") for t in n.targets))]
    ns = {"os": os}
    exec(compile(ast.Module(keep, []), "modal_rl1_" + name, "exec"), ns)
    return ns[name]


def _purge(env, scratch):
    import subprocess
    runs = _round_fn("stream_trainer_env")({}, chunks="", k=0, max_age=0, deltas="", run_id=scratch)["HEAL_DELTA_REMOTE"]
    runs = runs.rsplit("/", 2)[0]
    for run in (scratch, scratch + "-play"):
        subprocess.run(["rclone", "purge", f"{runs}/{run}"], env=env, capture_output=True, text=True, timeout=600)


def _delta_bucket_check(env, scratch, target, manifests, sample=None):
    """g1l (23 Oct): the first delta upload failed (the trainer env had no arc: remote) and no preflight had run it.
    Round a above uploaded its deltas with the round's own trainer env to the real bucket (a scratch run). Here the
    play side reads them as the play does: follow.json in a scratch play run, the Nebius runner's own follow_source
    and the streamer's copy into the play's folder, then the play bundle's weights_follow.ready_version."""
    import ast
    import importlib.util
    import re
    import shutil as _sh
    import subprocess
    import time
    import types

    remote = env["HEAL_DELTA_REMOTE"]  # arc:<bucket>/runs/validation/<scratch>/deltas
    runs = remote.rsplit("/", 2)[0]  # arc:<bucket>/runs/validation

    def rc(*a, timeout=600, check=False, **kw):  # the runner's rclone flags
        return subprocess.run(["rclone", *a, "--retries", "3", "--low-level-retries", "10", "--stats", "0"], env=env,
                              capture_output=True, text=True, timeout=timeout, **kw)

    listed = rc("lsjson", "--recursive", "--files-only", remote)
    files = json.loads(listed.stdout) if listed.returncode == 0 and listed.stdout.strip() else []
    uploaded = sorted({int(f["Path"].split("/")[0][1:]) for f in files if f["Path"].endswith("manifest.json")})
    tree = ast.parse(Path("/root/nebius_runner.py").read_text())
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runner")
    meth = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "follow_source")
    consts = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("FOLLOW_IN", "REMOTE")
                                                                        for t in n.targets)]
    ns = {"re": re, "json": json, "shutil": _sh, "subprocess": subprocess}
    exec(compile(ast.Module(consts + [meth], []), "nebius_runner_follow", "exec"), ns)
    stub = types.SimpleNamespace(remote_runs=f"{runs}/{scratch}-play", rclone=rc)
    gen = int(time.time())
    put = rc("rcat", f"{runs}/{scratch}-play/stream/follow.json",
             input=json.dumps({"in": f"{scratch}/deltas", "generation": gen, "start_version": 0}) + "\n")
    target.mkdir(parents=True, exist_ok=True)
    source = ns["follow_source"](stub, "unused/deltas", target)
    copied = rc("copy", f"{runs}/{source}", str(target), "--update", "--transfers", "32", "--checkers", "32",
                "--multi-thread-streams", "8")  # the runner's streamer copy
    spec = importlib.util.spec_from_file_location("weights_follow_pf", "/root/weights_follow.py")
    wf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wf)
    got = wf.ready_version(target, 0)
    newest = max((m["version"] for m in manifests), default=None)
    genfile = json.loads((target / "generation").read_text()) if (target / "generation").exists() else {}
    # the round's puller (main-process env, STREAM_PULL_FLAGS) on a rollout in the scratch play's stream
    pulled_ok, pull = True, None
    if sample is not None:
        rc("copyto", str(sample), f"{runs}/{scratch}-play/stream/trials/t/games/pf/rollout.jsonl")
        roll = target.parent / "pull-roll"
        main_env = {**os.environ, **_round_fn("stream_trainer_env")({}, chunks="", k=0, max_age=0, deltas="", run_id="x")}
        main_env = {k2: v for k2, v in main_env.items() if k2.startswith("RCLONE_") or k2 not in ("HEAL_DELTA_REMOTE",)}
        pull = subprocess.run(["rclone", "copy", f"{runs}/{scratch}-play/stream/trials", str(roll / f"{scratch}-play"),
                               *_pull_flags()], env=main_env, capture_output=True, text=True, timeout=300)
        got_file = roll / f"{scratch}-play" / "t" / "games" / "pf" / "rollout.jsonl"
        pulled_ok = pull.returncode == 0 and got_file.exists() and got_file.read_bytes() == Path(sample).read_bytes()
    ok = (listed.returncode == 0 and put.returncode == 0 and copied.returncode == 0 and source == f"{scratch}/deltas"
          and got is not None and got[0] == newest and genfile.get("generation") == gen and newest in uploaded
          and pulled_ok)
    return ok, {"remote": remote, "uploaded_versions": uploaded, "newest_local": newest,
                "ready": None if got is None else {"version": got[0], "files": len(got[1])}, "source": source,
                "pulled": pulled_ok, "errors": [x.stderr[-300:] for x in (listed, put, copied, pull) if x is not None and x.returncode]}


PF_ORDER = (2000, 60.0)  # teacher_order: base captures in the backlog; seconds to the first (student) piece


def _order_check(root, made, dests, encode, fwd, k, image_id, top_fn, end_id, base_share=0.0):
    """g1k (23 Oct): a recorded base backlog in sessions that sort before the student's was encoded and queued first,
    so the teacher scored 0 student pieces in 20 min. Thousands of base captures (copies of a stand-in session with
    the ids stripped, as base play records them) named to sort first, plus the live student session; the round's
    threaded ``run`` path: the first piece must be the student's, within PF_ORDER[1] s."""
    import math
    import shutil
    import time

    import heal_stream

    order = root / "order"
    shutil.rmtree(order, ignore_errors=True)
    games = list(made)
    student, other = games[0], games[-1]
    text = lambda g: Path(f"{dests[g]}.a").read_text() + Path(f"{dests[g]}.b").read_text()  # noqa: E731
    base_rows = "".join(json.dumps({x: v for x, v in json.loads(line).items() if x not in ("tokens", "logprobs")}) + "\n"
                        for line in text(other).splitlines() if line.strip())
    copies = math.ceil(PF_ORDER[0] / max(1, made[other]["gameplay"]))
    for i in range(copies):  # "base-backlog" sorts before "play-pf"
        d = order / "base-backlog" / "t" / "games" / f"c{i:04d}"
        d.mkdir(parents=True)
        (d / "rollout.jsonl").write_text(base_rows)
    d = order / "play-pf" / "t" / "games" / student
    d.mkdir(parents=True)
    (d / "rollout.jsonl").write_text(text(student))
    t = heal_stream.StreamTeacher(order, order / "chunks", encode, fwd, k=k, chunk=PF_CHUNK, image_token_id=image_id,
                                  end_id=end_id, idle=1e9, poll=0.5, encoders=4, top_fn=top_fn, base=("base-backlog",),
                                  base_share=base_share)
    first, began = {}, time.time()
    try:
        t.run(until=lambda t: bool(first) or time.time() - began > 3 * PF_ORDER[1],
              after_piece=lambda t, chain: first.update(key=chain.key, seconds=round(time.time() - began, 1)))
    except RuntimeError:  # the producer's cancelled encodes at the stop
        pass
    shutil.rmtree(order, ignore_errors=True)
    ok = first.get("key", "").startswith("play-pf/") and first["seconds"] <= PF_ORDER[1]
    return ok, {"first": first, "base_captures": copies * made[other]["gameplay"], "captures": t.stats["captures"],
                "inexact": t.stats.get("inexact", 0)}


def _pull_flags():
    """The round's puller flags, read from modal_rl1's own source (one definition for the round and this check)."""
    import ast
    tree = ast.parse(Path("/root/modal_rl1.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "STREAM_PULL_FLAGS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("modal_rl1.STREAM_PULL_FLAGS not found")


def _quack_cache_check(tmp):
    """The round's teacher quack-cache path (modal_rl1 quack_cache_seed / quack_cache_save and the teacher env), run
    from modal_rl1's own source on temporary folders: seed from an empty volume copy, a kernel file written, saved,
    a fresh seed gets it back; the teacher env names the local cache and autotune caching."""
    import ast
    src = Path("/root/modal_rl1.py").read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ("quack_cache_seed", "quack_cache_save"))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("QUACK_CACHE", "QUACK_SAVE_AFTER") for t in n.targets))]
    commits = []
    ns = {"Path": Path, "runs": type("R", (), {"commit": staticmethod(lambda: commits.append(1))})()}
    exec(compile(ast.Module(keep, []), "modal_rl1_quack", "exec"), ns)
    real = ns["QUACK_CACHE"]
    ns["QUACK_CACHE"] = str(Path(tmp) / "vol" / "quack")
    local = ns["quack_cache_seed"](str(Path(tmp) / "local1"))
    (local / "k").mkdir(parents=True, exist_ok=True)
    (local / "k" / "kernel.o").write_bytes(b"o" * 64)
    (local / "k" / "lock").write_text("")
    ns["quack_cache_save"](local)
    again = ns["quack_cache_seed"](str(Path(tmp) / "local2"))
    ok = ((again / "k" / "kernel.o").read_bytes() == b"o" * 64 and not (again / "k" / "lock").exists() and commits
          and real.startswith("/runs/") and '"QUACK_CACHE_DIR": str(local_q)' in src and '"QUACK_CACHE_AUTOTUNING": "1"' in src)
    return bool(ok), {"volume_path": real, "commits": len(commits), "save_after": ns.get("QUACK_SAVE_AFTER")}


MODAL_RL1 = "/root/modal_rl1.py"  # the round's own source, as mounted in the preflight image


def _teacher_cache_check(tmp):
    """modal_rl1's versioned teacher cache (ARC_TEACHER_CACHE=1) from its own source on temporary folders: a key from
    this interpreter, an empty seed, a save after the third logged piece (hub symlink copied as a file), a fresh seed
    gets it back, and a changed version seeds nothing."""
    import ast
    import hashlib
    import shutil
    import threading
    import time

    names = ("teacher_cache_key", "teacher_cache_seed", "teacher_cache_save", "teacher_cache_saver", "teacher_cache_env")
    consts = ("TEACHER_CACHE", "TEACHER_CACHE_DISTS", "QUACK_SAVE_PIECES", "CACHE_SKIP", "CACHE_DIRS")
    tree = ast.parse(Path(MODAL_RL1).read_text())
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in consts for t in n.targets))]
    commits = []
    ns = {"Path": Path, "json": json, "hashlib": hashlib, "time": time,
          "runs": type("R", (), {"commit": staticmethod(lambda: commits.append(1))})()}
    exec(compile(ast.Module(keep, []), "modal_rl1_tcache", "exec"), ns)
    tmp = Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    key, versions = ns["teacher_cache_key"](sys.executable)
    root = tmp / "vol"
    dirs, seeded0 = ns["teacher_cache_seed"](key, versions, local=tmp / "l1", root=root)
    q, hub = dirs["quack"], dirs["hub"]
    (dirs["cute"] / "f.mlir").write_bytes(b"c" * 16)
    (q / "k").mkdir()
    (q / "k" / "kernel.o").write_bytes(b"o" * 64)
    snap = hub / "models--x--y" / "snapshots" / "c0"
    snap.mkdir(parents=True)
    (hub / "models--x--y" / "blobs").mkdir()
    (hub / "models--x--y" / "blobs" / "b").write_bytes(b"s" * 8)
    (snap / "k.so").symlink_to("../../blobs/b")
    log = tmp / "teacher-stream.log"
    log.write_text('{"event": "stream_piece"}\n' * 3)
    done = threading.Event()
    saver = threading.Thread(target=ns["teacher_cache_saver"], kwargs={"poll": 0.05, "done": done}, daemon=True,
                             args=(log, lambda: ns["teacher_cache_save"](key, versions, dirs, root=root)))
    saver.start()
    for _ in range(100):
        if commits:
            break
        time.sleep(0.05)
    done.set()
    saver.join(5)
    dirs2, seeded = ns["teacher_cache_seed"](key, versions, local=tmp / "l2", root=root)
    q2, hub2 = dirs2["quack"], dirs2["hub"]
    _, stale = ns["teacher_cache_seed"](key, {**versions, "torch": "0"}, local=tmp / "l3", root=root)
    env = ns["teacher_cache_env"](dirs2)
    got = hub2 / "models--x--y" / "snapshots" / "c0" / "k.so"
    ok = (not seeded0 and seeded and not stale and len(commits) == 1 and (q2 / "k" / "kernel.o").read_bytes() == b"o" * 64
          and got.is_file() and not got.is_symlink() and not (root / key / "hub" / "models--x--y" / "blobs").exists()
          and (dirs2["cute"] / "f.mlir").read_bytes() == b"c" * 16
          and env == {"QUACK_CACHE_DIR": str(q2), "KERNELS_CACHE": str(hub2), "CUTE_DSL_CACHE_DIR": str(dirs2["cute"])})
    shutil.rmtree(tmp, ignore_errors=True)
    return bool(ok), {"key": key, "torch": versions.get("torch"), "commits": len(commits), "volume_root": ns["TEACHER_CACHE"]}


def _stream_live(res, check, *, root, made, dests, plays, base_plays, encode, fwd, k, image_id, top_fn, train, end_id,
                 round_args, senv, base_share):
    """The stream path end to end and concurrent, with the faults met on 22 Oct (the CPU preflight must
    drive it before a B300 is used): a stub play writes the sessions into a local 'bucket' line by line (half lines
    in flight), the round's own puller command copies them every second, the real teacher loop (``run``: tail
    thread, idle close, stop file) scores them while the trainer consumes the chunks. Injected: a shrunk copy of a
    student session for 6 s (opdS-g1h), a rewritten base session (a new game in the same file), session close by
    idle. Passes when the trainer takes 3 finite steps from the live chunks (a base slot among them), the teacher
    never fails, and the shrink and the rewrite were both seen."""
    import subprocess
    import threading
    import time

    import heal_stream

    bucket, roll, chunks = root / "bucket", root / "roll-live", root / "chunks-live"
    stop, stop_file, pulls = threading.Event(), root / "live-stop", {"ok": 0, "failed": 0}
    flags = _pull_flags()
    sessions = {g: (Path(f"{dests[g]}.a").read_text() + Path(f"{dests[g]}.b").read_text()).splitlines(keepends=True)
                for g in made}
    names = list(sessions)
    where = {}
    for g in names:  # roll/<play>/t/games/<g> -> bucket/<play>/stream/trials/t/games/<g> (the play's upload layout)
        rel = dests[g].parent.relative_to(root / "roll")
        where[g] = bucket / rel.parts[0] / "stream" / "trials" / Path(*rel.parts[1:])
        where[g].mkdir(parents=True, exist_ok=True)
    seen = {"shrink_written": False, "rewrite_written": False, "error": None}

    def play(g, lines, fault):
        try:
            f = where[g] / "rollout.jsonl"
            for i, line in enumerate(lines):
                with open(f, "a") as h:  # half a line in flight, then the rest
                    h.write(line[: len(line) // 2])
                time.sleep(0.2)
                with open(f, "a") as h:
                    h.write(line[len(line) // 2:])
                time.sleep(0.3)
                if fault == "shrink" and i == 2:
                    whole = f.read_text()
                    f.write_text(whole[: len(whole) // 3])
                    seen["shrink_written"] = True
                    time.sleep(6)
                    f.write_text(whole)
            if fault == "rewrite":  # a new game in the same file, longer than the old one: different bytes
                time.sleep(3)
                f.write_text("".join(sessions[names[0]]) + "".join(lines))
                seen["rewrite_written"] = True
        except Exception as exc:  # noqa: BLE001
            seen["error"] = f"{type(exc).__name__}: {exc}"

    def puller():
        while not stop.wait(1.0):
            for one in sorted({p for p in plays}):
                src = bucket / one / "stream" / "trials"
                if not src.is_dir():
                    continue
                r = subprocess.run(["rclone", "copy", str(src), str(roll / one), *flags], capture_output=True, text=True,
                                   timeout=120)
                pulls["ok" if r.returncode == 0 else "failed"] += 1

    roll.mkdir(parents=True, exist_ok=True)
    fill_src = root / "fill-src"  # a recorded base session (ARC_HEAL_STREAM_BASE_FILL), placed by the round's own helper
    (fill_src / "r0").mkdir(parents=True, exist_ok=True)
    (fill_src / "r0" / "rollout.jsonl").write_text("".join(sessions[names[-1]]))
    fill_root = heal_stream.stream_fill(str(fill_src / "r*"), roll)
    st = heal_stream.StreamTeacher(roll, chunks, encode, fwd, k=k, chunk=PF_CHUNK, image_token_id=image_id, end_id=end_id, idle=10.0,
                                   poll=0.5, encoders=2, top_fn=top_fn, stop_file=stop_file,
                                   base=tuple(base_plays) + (fill_root,))
    teacher_err = {}

    def teach():
        try:
            st.run(status_every=1e9)
        except Exception as exc:  # noqa: BLE001
            teacher_err["error"] = f"{type(exc).__name__}: {exc}"[:500]

    faults = {names[0]: "shrink", names[-1]: "rewrite" if base_share > 0 or len(names) > 1 else None}
    threads = [threading.Thread(target=play, args=(g, sessions[g], faults.get(g)), daemon=True) for g in names]
    threads += [threading.Thread(target=puller, daemon=True), threading.Thread(target=teach, daemon=True)]
    for t in threads:
        t.start()
    live = dict(senv, HEAL_STREAM=str(chunks), HEAL_DELTA_DIR=str(root / "deltas-live"), HEAL_STREAM_STALL="600",
                HEAL_HELDOUT2="", HEAL_MID_READ_EVERY="")  # the reads ran in round a
    began = time.time()
    q, ev, sec = train("live", round_args(str(root / "tiny-student"), str(root / "live"),
                                          routing=str(root / "r0" / "routing.pt")) + ["--allow-unfinished-schedule"], live)
    for t in threads[:len(names)]:
        t.join(timeout=120)
    time.sleep(3)
    stop_file.touch()
    stop.set()
    threads[-1].join(timeout=300)
    steps = [e for e in ev if e.get("event") == "step"]
    stats = dict(st.stats)
    ok = (q.returncode == 0 and len(steps) == 3 and all(e.get("finite") for e in steps)
          and (not base_share or any(e.get("stream_base") for e in steps))
          and not teacher_err and st.error is None and seen["error"] is None and not st.stats.get("inexact")
          and seen["shrink_written"] and stats.get("shrinks", 0) >= 1
          and (not faults.get(names[-1]) or (seen["rewrite_written"] and stats.get("rewrites", 0) >= 1))
          and pulls["ok"] > 0 and not pulls["failed"] and any(chunks.glob(f"b-{fill_root}*.ready")))
    check("stream_live", ok, {"seconds": round(time.time() - began), "train_seconds": sec, "pulls": pulls,
                              "teacher": {"error": teacher_err.get("error") or (repr(st.error) if st.error else None),
                                          **{x: stats.get(x) for x in ("pieces", "shrinks", "rewrites", "closed",
                                                                       "breaks", "captures")}},
                              "play": seen, "steps": [{x: e.get(x) for x in ("step", "kd_reply", "stream_base",
                                                                              "age_median")} for e in steps],
                              "stderr": "" if ok else _err_tail(q.stderr)})


def _preflight_score(students, teacher, general):
    """The heal_score path (one process, one card): header checks of the real student folders, then the bundle's
    ``heal_kd.py score`` on the tiny real-vocabulary student (as both 'students', the second a folder path) against
    2 real teacher and real general files; every student must report a finite reply KD."""
    import math
    import subprocess

    res = {"checks": {}, "ok": True}

    def check(name, ok, info):
        res["checks"][name] = {"ok": bool(ok), **info}
        res["ok"] = res["ok"] and bool(ok)

    for name in students:
        folder = name if name.startswith("/") else f"/assets/sft-heal/{name}"
        try:
            check(f"student {name}", *folder_check(folder))
        except Exception as exc:  # noqa: BLE001
            check(f"student {name}", False, {"error": f"{type(exc).__name__}: {exc}"[:300]})
    root = Path("/tmp/pfs")
    env = {**os.environ, "PYTHONPATH": "/opt/heal", "HEAL_NO_FAST_ATTENTION": "1", "HEAL_EXPERTS": "eager"}
    p = subprocess.run([sys.executable, "/opt/heal/heal_kd.py", "preflight-student", "--out", str(root)],
                       capture_output=True, text=True, env=env, cwd="/opt/heal")
    check("tiny_student", p.returncode == 0, {"stderr": p.stderr[-800:] if p.returncode else ""})
    if p.returncode:
        return res
    for name, src in (("teacher", teacher), ("general", general)):
        (root / name).mkdir()
        chosen = sorted(Path(src).glob("*.pt"), key=lambda f: f.stat().st_size)[:2 if name == "teacher" else 4]
        for f in chosen:
            (root / name / f.name).symlink_to(f)
    out = root / "score"
    cmd = [sys.executable, "/opt/heal/heal_kd.py", "score", "--students", "tiny-student", str(root / "tiny-student"),
           "--student-root", str(root), "--teacher", str(root / "teacher"), "--general", str(root / "general"),
           "--compare-files", "2", "--out", str(out), "--window", "32768"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/opt/heal")
    ev = out / "events.jsonl"
    rows = [json.loads(line) for line in open(ev)] if ev.exists() else []
    scored = [r for r in rows if r.get("reply_positions") is not None or "kd_reply" in r or "reply" in r]
    ok = p.returncode == 0 and len(scored) >= 2 and all(
        all(math.isfinite(float(v)) for k, v in r.items() if isinstance(v, (int, float)) and k != "t") for r in scored)
    check("score_run", ok, {"cmd": cmd, "rows": [{k: v for k, v in r.items() if k != "t"} for r in rows][:4],
                            "stderr": "\n".join(p.stderr.splitlines()[-20:]) if not ok else ""})
    return res


def _save(tag, res):
    """The result on the runs volume (/runs/preflight/TAG.json): a dropped client connection (19 Oct, twice) no longer
    loses it; the launcher's ops/pf_fetch.sh reads it."""
    if tag:
        out = Path("/runs/preflight")
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{tag}.json").write_text(json.dumps(res, indent=1, default=str))
        runs.commit()


@app.function(image=image, timeout=21600, cpu=PF_CPU,  # CPU only: gloo ranks
              volumes={"/assets": assets, "/runs": runs})
def preflight(args, kw, data, general, tag=""):
    res = _preflight(args, kw, data, general)
    _save(tag, res)
    return res


@app.function(image=image, timeout=21600, cpu=PF_CPU,  # CPU only, 4 gloo ranks
              volumes={"/assets": assets, "/runs": runs})
def preflight4(args, kw, data, general, tag=""):
    res = _preflight(args, kw, data, general)
    _save(tag, res)
    return res


@app.function(image=image_518, timeout=21600, cpu=PF_CPU,  # CPU only; the trainer image's transformers
              volumes={"/assets": assets, "/runs": runs}, secrets=[modal.Secret.from_name("nebius-s3")])
def stream_preflight(args, kw, segments, general, tag=""):
    res = _stream_preflight(args, kw, segments, general, tag=tag)
    _save(tag, res)
    return res


@app.function(image=image, timeout=3600, cpu=PF_CPU, volumes={"/assets": assets, "/runs": runs})  # CPU only
def preflight_score(students, teacher, general, tag=""):
    res = _preflight_score(students, teacher, general)
    _save(tag, res)
    return res


def wait(call):
    """call.get() that rides out client connection errors (the function itself runs on; launch with --detach).
    GPU fallbacks A10G / L40S (2-card, same cheap class): L4 capacity ran out on 19 Oct (calls queued 15 min)."""
    import time
    while True:
        try:
            return call.get(timeout=60)
        except TimeoutError:
            continue
        except Exception as e:  # only a lost client connection is retried; a failed or stopped call raises
            if type(e).__name__ not in ("ConnectionError", "StreamTerminatedError", "GRPCError", "ClientClosed", "OSError"):
                raise
            print(f"client error ({type(e).__name__}); retrying", flush=True)
            time.sleep(15)


@app.local_entrypoint()
def main(data: str = "", general: str = "", score: str = ""):
    import time
    tag = f"{os.environ.get('RUN', 'preflight')}-{int(time.time())}"
    if score:  # ARC_SCORE-style launch: --score STUDENT,STUDENT --data TEACHER_DIR --general GENERAL_DIR
        call = preflight_score.spawn(score.split(","), data, general, tag)
        print("PREFLIGHT CALL", call.object_id, tag, flush=True)
        res = wait(call)
        res["bundle"] = str(BUNDLE)
        print(json.dumps(res, indent=1, default=str))
        print("PREFLIGHT OK" if res["ok"] else "PREFLIGHT FAILED")
        if not res["ok"]:
            raise SystemExit(1)
        return
    sys.path.insert(0, str(HERE))
    import modal_rl1

    kw = modal_rl1.heal_round_kwargs(os.environ)
    kw["pf_only_smoke"] = os.environ.get("PF_ONLY_SMOKE") == "1"
    first = (kw["teacher_dir"] or kw["extra_teacher"]).split(",")[0]
    data = data or first
    general = general or (str(Path(first).parent / "general") if kw["general_windows"] else "")
    if not data:
        raise SystemExit("no real teacher files: pass --data")
    args = modal_rl1.heal_train_args(kw, teacher="@TEACHER", out="@OUT", general="@GENERAL")
    if kw.get("stream_play"):  # streaming OPD round: the stream path (teacher stream, deltas, restart) on CPU
        call = stream_preflight.spawn(args, kw, kw["segments"], general, tag)
        print("PREFLIGHT CALL", call.object_id, tag, flush=True)
        res = wait(call)
        res["bundle"], res["segments"], res["general"] = str(BUNDLE), kw["segments"], general
        print(json.dumps(res, indent=1, default=str))
        if not res["ok"]:
            raise SystemExit("PREFLIGHT FAILED")
        print("PREFLIGHT OK")
        return
    call = (preflight4 if int(kw.get("cards") or 2) >= 4 else preflight).spawn(args, kw, data, general, tag)
    print("PREFLIGHT CALL", call.object_id, tag, flush=True)
    res = wait(call)
    res["bundle"], res["data"], res["general"] = str(BUNDLE), data, general
    print(json.dumps(res, indent=1, default=str))
    if not res["ok"]:
        raise SystemExit("PREFLIGHT FAILED")
    print("PREFLIGHT OK")
