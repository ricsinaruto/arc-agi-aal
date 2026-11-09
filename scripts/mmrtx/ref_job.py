"""mmrtx reference job: the unpruned MiniMax-M3.1-Flash (official QAT weights, BF16 KV) on
Modal 2 x B200, TP2, vLLM's minimax-m31-flash build + mmrtx_hooks.

  build      (CPU)  conversations captured by the fit-data API runs -> token sets on mmrtx-models:/sets
  ref        (2 x B200) phase route: prefill of the fit sets (+ held-out), router mass per expert and MSA block churn per
                    group; phase kl: teacher-forced top-20 logprobs on the held-out reply tokens; phase dspark: DSpark
                    acceptance per draft position on held-out prompts (T 1.0, top_p 0.95). Each phase saves to
                    mmrtx-models:/ref/<run>/ and is skipped on a rerun when its output exists; a stop file
                    (/ref/<run>/STOP) is honoured between groups.
  preflight  (CPU)  the same code paths on a tiny set with a stub engine (--dry): build, phases, resume, stop, summary.

usage: modal run --detach ref_job.py::main_build --run-fit3 R3 --run-fit2 R2
       modal run --detach ref_job.py::main_ref --run RUN       (and ::main_preflight --run RUN-pf beside it)"""
import gzip
import json
import os
import re
import time
from pathlib import Path

import modal

import sys as _sys
_sys.path[:0] = [str(__import__('pathlib').Path(__file__).resolve().parent), '/root']
from image import IMAGE  # noqa: E402

HERE = Path(__file__).resolve().parent
app = modal.App("mmrtx-ref")
models = modal.Volume.from_name("mmrtx-models", create_if_missing=True)
runs3 = modal.Volume.from_name("arc-prolong-runs")
runs2 = modal.Volume.from_name("arc2-apiscreen")
JOB_IMAGE = IMAGE  # ref_job.py itself is mounted at /root by modal run
MODEL = "/m/minimax-m31f"
EXCL5 = ("lp85", "tu93", "vc33", "ar25", "ft09")
FIT_MAX_TOKENS, HELD_MAX_TOKENS = 131072, 32768
DSPARK_K = 7


# ----------------------------------------------------------------------------------------------- build (CPU) ----
def _template(model_dir):
    import jinja2.sandbox
    from tokenizers import Tokenizer
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                       extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)
    env.globals["raise_exception"] = raise_exception
    render = env.from_string(Path(model_dir, "chat_template.jinja").read_text()).render
    return render, Tokenizer.from_file(str(Path(model_dir, "tokenizer.json")))


def to_template_messages(messages, reply=None):
    """OpenRouter-format messages (+ the reply as a last assistant turn) in the form the MiniMax template renders:
    reasoning -> reasoning_content, tool-call arguments parsed, image parts replaced by a text marker."""
    out = []
    for m in list(messages) + ([{"role": "assistant", **reply}] if reply is not None else []):
        m = dict(m)
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    parts.append({"type": "text", "text": "(image omitted)"})
                else:
                    parts.append(p)
            m["content"] = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in parts)
        reasoning = m.pop("reasoning", None)
        if m.get("role") == "assistant" and reasoning and not m.get("reasoning_content"):
            m["reasoning_content"] = reasoning
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                f = dict(c.get("function") or c)
                if isinstance(f.get("arguments"), str):
                    try:
                        f["arguments"] = json.loads(f["arguments"] or "{}")
                    except ValueError:
                        f["arguments"] = {"_raw": f["arguments"]}
                calls.append({"id": c.get("id"), "type": "function", "function": f})
            m["tool_calls"] = calls
        out.append(m)
    return out


def tokenize_conv(render, tok, conv, max_tokens):
    """(token ids, assistant spans) of one captured conversation, cut at max_tokens. Spans come from character
    offsets of the template's own prefixes (each must be a string prefix of the full rendering)."""
    effort = (conv.get("reasoning") or {}).get("effort")
    msgs = to_template_messages(conv["messages"], conv.get("reply") or {})
    tools = conv.get("tools") or None
    full = render(messages=msgs, tools=tools, add_generation_prompt=False, reasoning_effort=effort)
    spans, bad = [], 0
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        a = render(messages=msgs[:i], tools=tools, add_generation_prompt=True, reasoning_effort=effort)
        b = full if i == len(msgs) - 1 else render(messages=msgs[:i + 1], tools=tools, add_generation_prompt=False,
                                                    reasoning_effort=effort)
        if full.startswith(a) and full.startswith(b) and len(b) > len(a):
            spans.append((len(a), len(b)))
        else:
            bad += 1
    enc = tok.encode(full, add_special_tokens=False)
    ids, offs = enc.ids[:max_tokens], enc.offsets[:max_tokens]
    tspans = []
    for cs, ce in spans:
        idx = [k for k, (s, _) in enumerate(offs) if cs <= s < ce]
        if idx:
            tspans.append((idx[0], idx[-1] + 1))
    return ids, tspans, bad


def load_convs(path):
    import hashlib
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    # drop a conversation that another one extends (same messages as its prefix)
    hs = [[hashlib.sha1(json.dumps(m, sort_keys=True, default=str).encode()).hexdigest() for m in r["messages"]]
          for r in rows]
    prefixes = set()
    for h in hs:
        for n in range(1, len(h)):
            prefixes.add(tuple(h[:n]))
    return [r for r, h in zip(rows, hs) if tuple(h) not in prefixes]


def first_assistant_text(conv):
    for m in conv["messages"] + [{"role": "assistant", **(conv.get("reply") or {})}]:
        if m.get("role") == "assistant":
            t = str(m.get("reasoning") or m.get("reasoning_content") or m.get("content") or "")
            if len(t) >= 40:
                return t
    return ""


def label_arc3(convs, trial_dir):
    """Game (source id prefix) of each conversation: its first assistant text found in one game's transcript."""
    roster = {r["game_id"]: r["source_game_id"] for r in json.loads((trial_dir / "public-roster.json").read_text())}
    texts = {}
    for g in sorted((trial_dir / "games").iterdir()):
        f = g / "duck-transcript.txt"
        if f.is_file():
            texts[g.name] = f.read_text(errors="replace")
    import collections
    out = []
    for c in convs:  # vote: 80-char windows from the middle of every assistant reasoning (reply included)
        votes = collections.Counter()
        for m in c["messages"] + [{"role": "assistant", **(c.get("reply") or {})}]:
            r = str(m.get("reasoning") or m.get("reasoning_content") or "") if m.get("role") == "assistant" else ""
            if len(r) >= 200:
                w = r[len(r) // 2:len(r) // 2 + 80]
                votes.update(k for k, t in texts.items() if w in t)
        top = votes.most_common(2)
        ok = top and (len(top) == 1 or top[0][1] >= 3 * top[1][1])
        out.append((roster[top[0][0]][:4], top[0][0]) if ok else (None, None))
    return out


def label_arc2(convs, events_path):
    rows = [json.loads(line) for line in open(events_path) if line.startswith("{")]
    out = []
    for c in convs:
        snip = first_assistant_text(c)[:300]
        hits = {r["task"].split("#")[0] for r in rows if snip and snip in str(r.get("reasoning") or r.get("content") or "")}
        out.append(hits.pop() if len(hits) == 1 else None)
    return out


_W = {}


def _init_worker(model_dir):
    _W["rt"] = _template(model_dir)


def _tok_one(args):
    conv, cap = args
    render, tok = _W["rt"]
    return tokenize_conv(render, tok, conv, cap)


def build_sets(fit3_dir, fit2_dir, model_dir, out_dir, roles2, limit=0):
    """Token sets: fit3/held3 (ARC-3 by game), fit2/held2 (ARC-2 by task role). Returns a summary."""
    import multiprocessing as mp
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, sets = {}, {"fit3": [], "held3": [], "fit2": [], "held2": []}
    c3 = load_convs(Path(fit3_dir) / "traj.jsonl.gz")
    for conv, (game, inst) in zip(c3, label_arc3(c3, Path(fit3_dir) / "arena/trials/duck-transfer")):
        if game is None:
            summary["arc3_unlabelled"] = summary.get("arc3_unlabelled", 0) + 1
            continue
        name = "fit3" if game in EXCL5 else "held3"
        sets[name].append((f"{game}-{inst}", game, conv))
    c2 = load_convs(Path(fit2_dir) / "traj.jsonl.gz")
    for conv, task in zip(c2, label_arc2(c2, Path(fit2_dir) / "events.jsonl")):
        if task is None:
            summary["arc2_unlabelled"] = summary.get("arc2_unlabelled", 0) + 1
            continue
        name = "fit2" if task in roles2["fit"] else "held2" if task in roles2["heldout"] else None
        if name:
            sets[name].append((task, task, conv))
    for name, items in sets.items():
        cap = HELD_MAX_TOKENS if name.startswith("held") else FIT_MAX_TOKENS
        n_tok = n_reply = bad = 0
        todo = items[:limit or None]
        with mp.get_context("fork").Pool(8, initializer=_init_worker, initargs=(model_dir,)) as pool:
            done = pool.map(_tok_one, [(conv, cap) for _, _, conv in todo], chunksize=1)
        with gzip.open(out_dir / f"{name}.jsonl.gz", "wt") as f:
            for k, ((cid, group, conv), (ids, spans, b)) in enumerate(zip(todo, done)):
                bad += b
                n_tok += len(ids)
                n_reply += sum(e - s for s, e in spans)
                f.write(json.dumps({"id": f"{name}-{k:04d}-{cid}", "group": group, "tokens": ids, "spans": spans}) + "\n")
        summary[name] = {"seqs": min(len(items), limit or len(items)), "tokens": n_tok, "reply_tokens": n_reply,
                         "span_mismatch": bad, "groups": sorted({g for _, g, _ in items})}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


@app.function(image=JOB_IMAGE, timeout=3600, cpu=(0.125, 8),
              volumes={"/m": models, "/r3": runs3, "/r2": runs2})
def build(run_fit3: str, run_fit2: str, roles_json: str, out: str = "sets-v1", limit: int = 0):
    s = build_sets(f"/r3/validation/{run_fit3}", f"/r2/{run_fit2}/mm", MODEL, f"/m/{out}", json.loads(roles_json),
                   limit=limit)
    models.commit()
    print(json.dumps(s, indent=1))
    return s


# ----------------------------------------------------------------------------------------------- ref (GPU) ----
PROGRESS = {"path": None}


def progress(**row):
    """One line per finished unit of work (group, batch, phase step): the watcher's ref.progress counter."""
    if PROGRESS["path"]:
        with open(PROGRESS["path"], "a") as f:
            f.write(json.dumps({"t": time.time(), **row}) + "\n")


def sampler(out, stop_event, commit, every=30):
    """Parent-side metrics rows for tools/ops/watch.py: GPU util / VRAM per card + the progress counter."""
    import subprocess
    last_commit = 0.0
    while not stop_event.wait(every):
        m = {}
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
            for line in q.strip().splitlines():
                i, u, mem = [x.strip() for x in line.split(",")]
                m[f"infra.gpu_util.{i}"] = float(u)
                m[f"infra.vram_gib.{i}"] = round(float(mem) / 1024, 2)
        except Exception:  # noqa: BLE001 - row without GPU fields
            pass
        p = out / "progress.jsonl"
        m["ref.progress"] = sum(1 for _ in open(p)) if p.exists() else 0
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"t": time.time(), "m": m}) + "\n")
        if time.time() - last_commit > 60:
            try:
                commit()
            except Exception:  # noqa: BLE001
                pass
            last_commit = time.time()


def read_set(path, limit=0):
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    return rows[:limit] if limit else rows


def coverage(mass):
    """Share of routed mass kept by the top-K experts per layer (K = 16..96), mean / min over MoE layers."""
    import torch
    m = mass[3:].double()
    tot = m.sum(1, keepdim=True).clamp_min(1e-12)
    srt = m.sort(1, descending=True).values.cumsum(1) / tot
    return {k: {"mean": round(float(srt[:, k - 1].mean()), 4), "min": round(float(srt[:, k - 1].min()), 4)}
            for k in (16, 24, 29, 32, 38, 40, 48, 53, 64, 96)}


def churn_summary(acc_list):
    """Per position bin: new blocks per token per head and distinct blocks per token (local heads), mean over the
    sampled layers and ranks."""
    import torch
    bins = ["<8K", "8-32K", "32-64K", "64-128K", ">=128K"]
    new = sum(a["churn_new"].sum(0) for a in acc_list)
    uni = sum(a["churn_union"].sum(0) for a in acc_list)
    n = sum(a["churn_n"].sum(0) for a in acc_list).clamp_min(1)
    return {b: {"new_per_head": round(float(new[i] / n[i]), 3), "distinct": round(float(uni[i] / n[i]), 2),
                "n": int(n[i])} for i, b in enumerate(bins)}


class StubLLM:
    """Preflight engine: vLLM-shaped outputs, no model. Hooks are fed synthetic routing / top-k."""

    def __init__(self, **kw):
        import torch
        import mmrtx_hooks as H
        self.H, self.torch, self.kw = H, torch, kw

    def collective_rpc(self, fn, args=(), kwargs=None):
        return [fn(None, *args, **(kwargs or {}))]

    def generate(self, prompts, sp, use_tqdm=False):
        import types
        torch, H = self.torch, self.H
        outs = []
        for p in prompts:
            ids = p["prompt_token_ids"]
            T = len(ids)
            if H.STATE["tag"] is not None:
                for layer in range(3, 60):
                    H.route_stats(layer, torch.randn(T, 128), torch.zeros(128), H.STATE["tag"])
                for layer in range(0, 60, H.CHURN_EVERY):
                    pos = torch.arange(T)
                    topk = (torch.arange(16)[None, None, :] + (pos // 128)[:, None, None]).expand(T, 2, 16).int()
                    H.churn_stats(layer, pos, topk, H.STATE["tag"])
            pl = None
            if getattr(sp, "prompt_logprobs", None):
                pl = [None] + [{ids[i]: types.SimpleNamespace(logprob=-1.0),
                                **{(ids[i] + j + 1) % 200064: types.SimpleNamespace(logprob=-2.0 - j) for j in range(20)}}
                               for i in range(1, T)]
            outs.append(types.SimpleNamespace(prompt_logprobs=pl, outputs=[types.SimpleNamespace(token_ids=[1] * 4)]))
        return outs

    def get_metrics(self):
        import types
        return [types.SimpleNamespace(name="vllm:spec_decode_num_drafts", value=10),
                types.SimpleNamespace(name="vllm:spec_decode_num_draft_tokens", value=70),
                types.SimpleNamespace(name="vllm:spec_decode_num_accepted_tokens", value=25),
                types.SimpleNamespace(name="vllm:spec_decode_num_accepted_tokens_per_pos", values=[8, 6, 4, 3, 2, 1, 1])]


def make_llm(dry, spec=False):
    if dry:
        return StubLLM()
    from vllm import LLM
    kw = dict(model=MODEL, tensor_parallel_size=2, trust_remote_code=True, max_model_len=FIT_MAX_TOKENS + 16,
              gpu_memory_utilization=0.85, enable_prefix_caching=False, max_num_batched_tokens=8192, max_num_seqs=32,
              block_size=128, disable_hybrid_kv_cache_manager=True, limit_mm_per_prompt={"image": 0, "video": 0},
              disable_log_stats=False)
    if spec:
        kw.update(speculative_config={"method": "dspark", "num_speculative_tokens": DSPARK_K,
                                      "draft_sample_method": "probabilistic"}, max_num_seqs=64)
    else:
        kw.update(enforce_eager=True)
    return LLM(**kw)


def rpc(llm, name, *args, **kwargs):
    import mmrtx_hooks
    return llm.collective_rpc(getattr(mmrtx_hooks, name), args=args, kwargs=kwargs)


def phase_route(llm, sets_dir, out, dry, limit, stop):
    """Prefill of every fit/held sequence (max_tokens 1), one generate call per group; stats per group."""
    import torch
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    stats, t0, ntok = {}, time.time(), 0
    for name in ("fit3", "fit2", "held3", "held2"):
        rows = read_set(Path(sets_dir) / f"{name}.jsonl.gz", limit)
        groups = {}
        for r in rows:
            groups.setdefault(r["group"], []).append(r)
        for g, items in sorted(groups.items()):
            if stop():
                raise SystemExit("stop file")
            tag = f"{name}:{g}"
            rpc(llm, "set_tag", tag=tag)
            llm.generate([{"prompt_token_ids": r["tokens"]} for r in items], sp, use_tqdm=False)
            rpc(llm, "set_tag", tag=None)
            snaps = rpc(llm, "snapshot", reset=True)
            stats[tag] = [s["acc"].get(tag) for s in snaps]
            ntok += sum(len(r["tokens"]) for r in items)
            print(f"route {tag}: {len(items)} seqs, {ntok} tokens so far, {time.time() - t0:.0f} s", flush=True)
            progress(phase="route", tag=tag, tokens=ntok)
    torch.save(stats, out / "route.pt")
    return {"tokens": ntok, "seconds": round(time.time() - t0)}


def phase_kl(llm, sets_dir, out, dry, limit, stop):
    """Teacher-forced top-20 logprobs at every assistant-span position of the held-out sets."""
    import numpy as np
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=20)
    res, t0 = {}, time.time()
    for name in ("held3", "held2"):
        rows = read_set(Path(sets_dir) / f"{name}.jsonl.gz", limit)
        for k in range(0, len(rows), 8):
            if stop():
                raise SystemExit("stop file")
            batch = rows[k:k + 8]
            outs = llm.generate([{"prompt_token_ids": r["tokens"]} for r in batch], sp, use_tqdm=False)
            for r, o in zip(batch, outs):
                pos = [p for s, e in r["spans"] for p in range(max(s, 1), e)]
                ids = np.zeros((len(pos), 20), np.int32)
                lps = np.full((len(pos), 20), -1e4, np.float32)
                tlp = np.zeros(len(pos), np.float32)
                for j, p in enumerate(pos):
                    d = o.prompt_logprobs[p] or {}
                    top = sorted(d.items(), key=lambda kv: -kv[1].logprob)[:20]
                    ids[j, :len(top)] = [t for t, _ in top]
                    lps[j, :len(top)] = [v.logprob for _, v in top]
                    tlp[j] = d[r["tokens"][p]].logprob if r["tokens"][p] in d else -1e4
                res[r["id"]] = {"pos": np.array(pos, np.int32), "ids": ids, "lps": lps.astype(np.float16), "tok_lp": tlp}
            print(f"kl {name}: {k + len(batch)}/{len(rows)} seqs, {time.time() - t0:.0f} s", flush=True)
            progress(phase="kl", set=name, seqs=k + len(batch))
    np.savez_compressed(out / "kl.npz", **{f"{i}|{f}": v[f] for i, v in res.items() for f in v})
    n = sum(len(v["pos"]) for v in res.values())
    nll = -float(sum(v["tok_lp"].sum() for v in res.values())) / max(n, 1)
    return {"seqs": len(res), "positions": n, "nll": round(nll, 4), "seconds": round(time.time() - t0)}


def phase_dspark(llm, sets_dir, out, dry, limit, stop):
    """Acceptance of DSpark drafts (K=7) on held-out prompts cut before an assistant turn, T 1.0 / top_p 0.95."""
    from vllm import SamplingParams
    prompts = []
    for name in ("held3", "held2"):
        for r in read_set(Path(sets_dir) / f"{name}.jsonl.gz", limit):
            for s, _ in r["spans"][-3:]:
                if 2000 < s <= 30000:
                    prompts.append(r["tokens"][:s])
    prompts = prompts[:limit or 96]
    sp = SamplingParams(max_tokens=1024 if not dry else 4, temperature=1.0, top_p=0.95)
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
    gen = sum(len(o.outputs[0].token_ids) for o in outs)
    m = {x.name: (getattr(x, "values", None) or getattr(x, "value", None)) for x in llm.get_metrics()
         if "spec_decode" in x.name}
    drafts = m.get("vllm:spec_decode_num_drafts") or 0
    per_pos = m.get("vllm:spec_decode_num_accepted_tokens_per_pos") or []
    acc = m.get("vllm:spec_decode_num_accepted_tokens") or 0
    res = {"prompts": len(prompts), "generated": gen, "seconds": round(time.time() - t0), "drafts": drafts,
           "accepted": acc, "per_pos_rate": [round(v / drafts, 3) for v in per_pos] if drafts else None,
           "tau_k7": round(1 + acc / drafts, 3) if drafts else None, "metrics": {k: str(v)[:200] for k, v in m.items()}}
    if drafts and per_pos:  # tokens per verify step for a shorter draft: 1 + sum of the first k position rates
        res["tau_by_k"] = {k: round(1 + sum(per_pos[:k]) / drafts, 3) for k in range(1, len(per_pos) + 1)}
    (out / "dspark.json").write_text(json.dumps(res, indent=1))
    return res


def run_ref(run, sets_dir, out_root, dry=False, limit=0, commit=lambda: None, only=None):
    import os
    os.environ["MMRTX_HOOKS"] = "1"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # collective_rpc sends our hook functions and gets tensor dicts back: msgpack refuses both without this
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    import mmrtx_hooks
    if dry:
        mmrtx_hooks.register = mmrtx_hooks.register  # hooks' math is exercised through the stub engine
    out = Path(out_root) / run
    out.mkdir(parents=True, exist_ok=True)
    PROGRESS["path"] = str(out / "progress.jsonl")
    stop = lambda: (out / "STOP").exists()  # noqa: E731
    done = json.loads((out / "phases.json").read_text()) if (out / "phases.json").exists() else {}

    def save(name, info):
        done[name] = info
        (out / "phases.json").write_text(json.dumps(done, indent=1))
        commit()
        print(f"PHASE {name} done: {json.dumps(info)[:400]}", flush=True)

    if only in (None, "main") and ("route" not in done or "kl" not in done):
        progress(phase="load")
        llm = make_llm(dry)
        progress(phase="ready")
        if not dry and not all(r for r in rpc(llm, "set_tag", tag=None)):
            raise RuntimeError("mmrtx_hooks not registered in the workers")
        if "route" not in done:
            save("route", phase_route(llm, sets_dir, out, dry, limit, stop))
        if "kl" not in done:
            save("kl", phase_kl(llm, sets_dir, out, dry, limit, stop))
        del llm
    if only in (None, "dspark") and "dspark" not in done:
        progress(phase="load-dspark")
        llm = make_llm(dry, spec=True)
        progress(phase="ready-dspark")
        save("dspark", phase_dspark(llm, sets_dir, out, dry, limit, stop))
    if only is not None:
        return done
    summary = summarize(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    commit()
    print("SUMMARY " + json.dumps(summary), flush=True)
    print("RUN COMPLETE", flush=True)
    return summary


def summarize(out):
    import torch
    stats = torch.load(out / "route.pt", weights_only=False)
    by_set = {}
    for tag, per_rank in stats.items():
        a = per_rank[0]
        if a is None:
            continue
        by_set.setdefault(tag.split(":")[0], []).append((tag, per_rank))
    res = {}
    fit_mass = sum(pr[0]["mass"] for name in ("fit3", "fit2") for _, pr in by_set.get(name, []))
    if torch.is_tensor(fit_mass):
        res["coverage_fit"] = coverage(fit_mass)
        for name in ("held3", "held2"):
            items = by_set.get(name, [])
            if items:  # held-out mass kept by the experts the fit set ranks top-K
                held = sum(pr[0]["mass"] for _, pr in items)[3:].double()
                order = fit_mass[3:].double().argsort(1, descending=True)
                tot = held.sum(1).clamp_min(1e-12)
                res[f"heldout_kept_{name}"] = {k: round(float((held.gather(1, order[:, :k]).sum(1) / tot).mean()), 4)
                                               for k in (24, 32, 48, 64)}
    res["churn"] = churn_summary([r for items in by_set.values() for _, pr in items for r in pr if r is not None])
    res["phases"] = json.loads((out / "phases.json").read_text())
    return res


@app.function(image=JOB_IMAGE, gpu="B200:2", timeout=3 * 3600, retries=0, volumes={"/m": models})
def ref(run: str, sets: str = "sets-v1", limit: int = 0):
    """Each engine in its own process (route + kl, then DSpark), so the second starts on clean GPUs."""
    import subprocess
    import sys
    import threading
    out = Path("/m/ref") / run
    out.mkdir(parents=True, exist_ok=True)
    halt = threading.Event()
    threading.Thread(target=sampler, args=(out, halt, models.commit), daemon=True).start()
    for part in ("main", "dspark"):
        rc = subprocess.run([sys.executable, "/root/ref_job.py", "child", run, sets, str(limit), part]).returncode
        models.commit()
        if rc:
            raise RuntimeError(f"phase group {part} exited {rc}")
    s = run_ref(run, f"/m/{sets}", "/m/ref", dry=False, limit=limit, commit=models.commit)
    halt.set()
    models.commit()
    return s


@app.function(image=JOB_IMAGE, timeout=3600, cpu=(8, 16), retries=0, volumes={"/m": models})
def preflight(run: str, sets: str = "sets-v1"):
    """CPU preflight of the exact bundle: static hook targets, then run_ref on 2 sequences a set with the stub
    engine: start, every phase, a STOP between groups, resume from phases.json, summary."""
    import shutil
    import os
    os.environ["MMRTX_HOOKS"] = "1"
    import mmrtx_hooks
    mmrtx_hooks.register()
    assert mmrtx_hooks.STATE.get("registered"), "hooks did not register against this vLLM build"
    # the engine-client serializer must carry the rpc payloads both ways (mmrtx-ref-v1 died on this)
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    import torch
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
    enc, dec = MsgpackEncoder(), MsgpackDecoder()
    dec.decode(enc.encode((0, "c", "collective_rpc", (mmrtx_hooks.snapshot, None, (), {"reset": True}))))
    dec.decode(enc.encode({"acc": {"t": {"mass": torch.zeros(60, 128, dtype=torch.float64)}}}))
    out = Path("/m/ref") / run
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    (out / "STOP").write_text("preflight")
    try:
        run_ref(run, f"/m/{sets}", "/m/ref", dry=True, limit=2)
        raise AssertionError("the stop file was not honoured")
    except SystemExit as e:
        assert "stop" in str(e)
    (out / "STOP").unlink()
    run_ref(run, f"/m/{sets}", "/m/ref", dry=True, limit=2, only="main")
    run_ref(run, f"/m/{sets}", "/m/ref", dry=True, limit=2, only="dspark")
    s = run_ref(run, f"/m/{sets}", "/m/ref", dry=True, limit=2)
    assert set(s["phases"]) == {"route", "kl", "dspark"}, s["phases"]
    s2 = run_ref(run, f"/m/{sets}", "/m/ref", dry=True, limit=2)  # resume: nothing re-runs
    assert s2["phases"] == s["phases"]
    import numpy as np
    z = np.load(out / "kl.npz")
    assert len(z.files) > 0
    models.commit()
    print("PREFLIGHT OK " + json.dumps(s)[:600], flush=True)
    return s


@app.local_entrypoint()
def main_build(run_fit3: str, run_fit2: str, out: str = "sets-v1", limit: int = 0):
    roles = json.loads((HERE.parent / "fit/fit-arc2.json").read_text())["mmrtx_roles"]
    call = build.spawn(run_fit3, run_fit2, json.dumps(roles), out, limit)
    print(json.dumps({"call": call.object_id, "app_id": app.app_id}))


@app.local_entrypoint()
def main_ref(run: str, sets: str = "sets-v1"):
    call = ref.spawn(run, sets)
    print(json.dumps({"call": call.object_id, "app_id": app.app_id}))


@app.local_entrypoint()
def main_preflight(run: str, sets: str = "sets-v1"):
    call = preflight.spawn(run, sets)
    print(json.dumps({"call": call.object_id, "app_id": app.app_id}))


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "child":
    import sys
    _, _, run, sets, limit, part = sys.argv
    run_ref(run, f"/m/{sets}", "/m/ref", dry=False, limit=int(limit), commit=lambda: None, only=part)
