"""mmrtx KL screen (1 Modal RTX PRO 6000): a candidate checkpoint teacher-forced on the held-out sets (held3 ARC-3
non-excl5 games, held2 hard38 tasks), top-20 prompt logprobs at the reference's assistant-span positions, compared
with the unpruned reference (mmrtx-models:/ref/<ref>/kl.npz, 2 x B200, BF16 KV):
  dnll  = mean NLL of the actual (MiniMax-sampled) tokens, candidate minus reference (exact);
  kl20  = sum over the reference top-20 of p_ref (lp_ref - lp_cand), lp_cand floored at the candidate's 20th logprob
          when a token is outside its top-20 (a lower bound of the truncated KL);
  top1  = agreement of the argmax tokens.
Per-sequence values are saved; a bootstrap over sequences gives the CI. Several candidates/settings run in one container
one after the other (``--jobs`` JSON list), each in its own process.

usage: modal run --detach kl_score.py --run RUN --jobs '[{"name":"p24","model":"minimax-m31f-p24-v1","kv":"fp8"}, ...]'"""
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

try:  # Modal launcher; inside the Nebius docker image there is no modal and the models sit under /w/mmrtx-models
    import modal
    import sys as _sys
    _sys.path[:0] = [str(Path(__file__).resolve().parent), '/root']
    from image import IMAGE  # noqa: E402
    app = modal.App("mmrtx-kl")
    models = modal.Volume.from_name("mmrtx-models")
except ImportError:
    modal = None
    models = types.SimpleNamespace(commit=lambda: None)
ROOT = os.environ.get("MMRTX_ROOT", "/m")


def score_one(job, sets_dir, ref_npz, out_path):
    import gzip
    import numpy as np
    from vllm import LLM, SamplingParams
    rows = [json.loads(line) for name in ("held3", "held2") for line in gzip.open(f"{sets_dir}/{name}.jsonl.gz", "rt")]
    if job.get("stride"):  # a fixed subset (every k-th sequence of each set) for slow (offloaded) candidates
        rows = rows[::int(job["stride"])]
    if job.get("reverse"):
        rows = rows[::-1]  # batch-order variant for the numeric floor
    model = job["model"] if job["model"].startswith("/") else f"{ROOT}/{job['model']}"  # absolute: built at boot
    if job.get("dummy"):
        # boot check: config-only copy cut to N layers, random weights, two rows
        import shutil
        d = Path(f"/tmp/dummy-{job['name']}")
        shutil.copytree(model, d, ignore=shutil.ignore_patterns("*.safetensors", "*.index.json", ".cache"), dirs_exist_ok=True)
        cfg = json.loads((d / "config.json").read_text())
        if int(job["dummy"]) > 0:  # 0 = full depth (memory / offload path check)
            cfg["text_config"]["num_hidden_layers"] = int(job["dummy"])
        (d / "config.json").write_text(json.dumps(cfg))
        model, rows = str(d), rows[:int(job.get("limit", 2))]
    kw = dict(model=model, tensor_parallel_size=1, trust_remote_code=True, max_model_len=33000,
              gpu_memory_utilization=float(job.get("util", 0.92)), enable_prefix_caching=False,
              max_num_batched_tokens=4096, max_num_seqs=int(job.get("seqs", 4)), block_size=128, disable_hybrid_kv_cache_manager=True,
              limit_mm_per_prompt={"image": 0, "video": 0}, enforce_eager=True)
    if job.get("kv", "fp8") != "auto":
        kw.update(kv_cache_dtype=job.get("kv", "fp8"), attention_config={"indexer_kv_dtype": job.get("kv", "fp8")})
    for k in ("offload_group_size", "offload_num_in_group", "offload_prefetch_step", "offload_backend", "cpu_offload_gb"):
        if k in job:
            kw[k] = job[k]
    kw.update(job.get("llm_kw", {}))  # e.g. {"quantization": "humming"} with env VLLM_HUMMING_ONLINE_QUANT_CONFIG
    if job.get("dummy"):
        kw["load_format"] = "dummy"
    t0 = time.time()
    llm = LLM(**kw)
    load_s = time.time() - t0
    ref = np.load(ref_npz)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=20)
    per = {}
    for k in range(0, len(rows), 4):
        batch = rows[k:k + 4]
        outs = llm.generate([{"prompt_token_ids": r["tokens"]} for r in batch], sp, use_tqdm=False)
        for r, o in zip(batch, outs):
            pos = ref[f"{r['id']}|pos"]
            rid, rlp = ref[f"{r['id']}|ids"], ref[f"{r['id']}|lps"].astype(np.float32)
            rtok = ref[f"{r['id']}|tok_lp"]
            kl = nll = agree = 0.0
            n = 0
            for j, p in enumerate(pos):
                d = o.prompt_logprobs[int(p)] or {}
                if not d:
                    continue
                cand = {t: v.logprob for t, v in d.items()}
                vals = sorted(cand.values(), reverse=True)
                floor = vals[min(19, len(vals) - 1)]  # dict = top-20 plus the actual token
                pr = np.exp(rlp[j])
                lc = np.array([cand.get(int(t), floor) for t in rid[j]], np.float32)
                kl += float((pr * (rlp[j] - lc)).sum())
                tok = r["tokens"][int(p)]
                nll += -(cand.get(tok, floor)) + float(rtok[j])
                agree += float(max(cand, key=cand.get) == int(rid[j][0]))
                n += 1
            per[r["id"]] = {"n": n, "kl20": kl, "dnll": nll, "top1": agree}
        with open(Path(out_path).parent / "progress.jsonl", "a") as f:
            f.write(json.dumps({"t": time.time(), "job": job["name"], "seqs": k + len(batch)}) + "\n")
    tot = {k: sum(v[k] for v in per.values()) for k in ("n", "kl20", "dnll", "top1")}
    res = {"job": job, "load_s": round(load_s), "seconds": round(time.time() - t0), "n": tot["n"],
           "kl20": tot["kl20"] / tot["n"], "dnll": tot["dnll"] / tot["n"], "top1": tot["top1"] / tot["n"]}
    for s in ("held3", "held2"):
        sub = {k: v for k, v in per.items() if k.startswith(s)}
        if sub:
            nn = sum(v["n"] for v in sub.values())
            res[s] = {m: round(sum(v[m] for v in sub.values()) / nn, 4) for m in ("kl20", "dnll", "top1")}
    res["per_seq"] = per
    Path(out_path).write_text(json.dumps(res))
    print("SCORE " + json.dumps({k: v for k, v in res.items() if k != "per_seq"}), flush=True)


def sampler(out, halt, every=30):
    """metrics.jsonl rows for tools/ops/watch.py: GPU util / VRAM + kl.progress (scored batches)."""
    last = 0.0
    while not halt.wait(every):
        m = {}
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
            for line in q.strip().splitlines():
                i, u, mem = [x.strip() for x in line.split(",")]
                m[f"infra.gpu_util.{i}"], m[f"infra.vram_gib.{i}"] = float(u), round(float(mem) / 1024, 2)
        except Exception:  # noqa: BLE001
            pass
        p = out / "progress.jsonl"
        m["kl.progress"] = sum(1 for _ in open(p)) if p.exists() else 0
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"t": time.time(), "m": m}) + "\n")
        if time.time() - last > 60:
            try:
                models.commit()
            except Exception:  # noqa: BLE001
                pass
            last = time.time()


def bootstrap(a, b=None, metric="kl20", reps=2000):
    """95% CI of the position-weighted mean (or of a - b on shared sequences) over sequences."""
    import random
    ids = sorted(set(a["per_seq"]) & set(b["per_seq"])) if b else sorted(a["per_seq"])  # paired: shared rows only
    if not ids:
        return None
    rng = random.Random(0)

    def stat(sample):
        num = sum(a["per_seq"][i][metric] - (b["per_seq"][i][metric] if b else 0) for i in sample)
        return num / sum(a["per_seq"][i]["n"] for i in sample)
    vals = sorted(stat([rng.choice(ids) for _ in ids]) for _ in range(reps))
    return round(vals[int(0.025 * reps)], 4), round(vals[int(0.975 * reps)], 4)


def select_layers(job, done, out):
    """Pre-registered rule (mmrtx.md, 25 Oct): rank layer groups by (kl20(swap) - kl20(base)) per layer, both on the
    same stride; add whole groups, cheapest first, until >= min_layers are Humming; predicted kl20 = anchor + sum of
    their deltas. Builds the symlinked variant and returns the job with model and MMRTX_MOE_HUMMING_LAYERS set."""
    sel = job["select"]
    if sel["base"] not in done or any(j not in done for j in sel["group_jobs"].values()):
        return None
    base = done[sel["base"]]["kl20"]
    delta = {g: done[j]["kl20"] - base for g, j in sel["group_jobs"].items()}
    order = sorted(delta, key=lambda g: delta[g] / len(sel["groups"][g]))
    chosen, layers = [], []
    for g in order:
        if len(layers) >= int(sel["min_layers"]):
            break
        chosen.append(g)
        layers += list(sel["groups"][g])
    pred = float(sel["anchor_kl20"]) + sum(delta[g] for g in chosen)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hum_build
    hum_build.variant(sel["src"], sel["hum"], sel["dst"], ",".join(map(str, sorted(layers))))
    info = {"delta": {g: round(v, 4) for g, v in delta.items()}, "order": order, "chosen": chosen,
            "layers": sorted(layers), "n_layers": len(layers), "predicted_kl20": round(pred, 4)}
    (out / "select.json").write_text(json.dumps(info, indent=1))
    print("SELECT " + json.dumps(info), flush=True)
    env = {**job.get("env", {}), "MMRTX_MOE_HUMMING_LAYERS": ",".join(map(str, sorted(layers)))}
    return {**{k: v for k, v in job.items() if k != "select"}, "model": sel["dst"], "env": env, "selection": info}


def run_jobs(out, jobs, ref, sets):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    import threading
    halt = threading.Event()
    threading.Thread(target=sampler, args=(out, halt), daemon=True).start()
    done = {}
    for job in json.loads(jobs):
        path = out / f"{job['name']}.json"
        if "select" in job and not path.exists():
            job = select_layers(job, done, out)
            if job is None:
                print("SELECT FAILED: missing group results", flush=True)
                continue
        if not path.exists():
            rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child", json.dumps(job), f"{ROOT}/{sets}",
                                 f"{ROOT}/ref/{ref}/kl.npz", str(path)]).returncode
            models.commit()
            if rc:
                print(f"JOB FAILED {job['name']} rc={rc}", flush=True)
                if job.get("gate"):
                    print("GATE FAILED: skipping the remaining jobs", flush=True)
                    break
                continue
        if not job.get("dummy"):
            done[job["name"]] = json.loads(path.read_text())
        else:
            print(f"BOOT OK {job['name']}", flush=True)
    summary = {}
    names = list(done)
    for n in names:
        r = done[n]
        summary[n] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("per_seq", "job")}
        summary[n]["kl20_ci"] = bootstrap(r)
        if names and n != names[0]:
            summary[n]["kl20_minus_first_ci"] = bootstrap(r, done[names[0]])
        vs = r["job"].get("vs")
        if vs in done:  # paired difference against a named job (e.g. merge vs drop at the same K)
            summary[n][f"kl20_minus_{vs}_ci"] = bootstrap(r, done[vs])
            summary[n][f"dnll_minus_{vs}_ci"] = bootstrap(r, done[vs], "dnll")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    halt.set()
    models.commit()
    print("SUMMARY " + json.dumps(summary), flush=True)
    print("RUN COMPLETE", flush=True)
    return summary


if modal is not None:
    @app.function(image=IMAGE, gpu="RTX-PRO-6000", timeout=4 * 3600, retries=0, volumes={"/m": models})
    def score(run: str, jobs: str, ref: str = "mmrtx-ref-v2", sets: str = "sets-v2"):
        return run_jobs(Path("/m/kl") / run, jobs, ref, sets)

    @app.local_entrypoint()
    def main(run: str, jobs: str, ref: str = "mmrtx-ref-v2"):
        call = score.spawn(run, jobs, ref)
        print(json.dumps({"call": call.object_id, "app_id": app.app_id}))

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "nebius":
    # python3 kl_score.py nebius JOBS_JSON [REF] [SETS]; results under $ARC_OUT
    run_jobs(Path(os.environ.get("ARC_OUT", "/out")), sys.argv[2], *(sys.argv[3:5]))

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "child":
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    for k, v in json.loads(sys.argv[2]).get("env", {}).items():
        os.environ[k] = str(v)
    score_one(json.loads(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])

