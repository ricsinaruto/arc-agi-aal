"""Boot-time fixed-prompt logprob dump (RL redesign, 26 Oct): E4's in-play exactness check and E1's golden.

Probes (``launch/rl/data/kv-dump-probes-v1.jsonl``, built by ``build``): 8 recorded gameplay requests at 8K-100K
context (request k+1 of a v6 try, images inline), each with one fixed 64-token continuation (the first 64 tokens of
reply k, which request k+1 holds). ``dump`` scores the continuation teacher-forced (``prompt_logprobs``, one token
generated; a prompt-logprobs request takes no prefix-cache hit, so every pass recomputes), probes one at a time or
all at once.

Boot step (rl_seq ``mg.redesign.kv_golden``, off by default): after the play adapter loads and before the gate opens,
every play server dumps (one at a time); the reference card (card 1 when there are two or more) dumps again with all
8 at once: its floor. A ``seq_boot`` row (stage kv_dump) carries per-server p99 / mean / max |dlogp| vs the reference
and the floor; pass: p99 <= 1.5 x floor p99 and mean <= 1.1 x floor mean (max is one token of 512: reported only).
The play adapter at boot is zero-delta (defer_trainer's zero adapter, or step 0's B = 0), so a LoRA path (E4) is only
exercised with ``spec["adapter"]``: a trained adapter every server loads as "kv-golden" for the dump.
The first-token watch's zero is taken after it.
"""
import asyncio
import json
import math
import os
import subprocess
import time
from pathlib import Path

ADAPTER_NAME = "kv-golden"
# lora-l1 26 Oct (was max and mean <= 1.5x): H3's E4 fail was one token of 512 (6.93); H4 with E4 off on both cards
# had max 3.67 vs floor 2.07; p99 and mean of both sat on the floor
BOOT_BARS = {"p99_vs_floor": 1.5, "mean_vs_floor": 1.1}


# ---------------------------------------------------------------------------- probes
def rebuild(rows):
    """Rows with their full message lists and tools (rollout rows store a kept prefix length plus new messages)."""
    out, messages, tools = [], [], None
    for r in rows:
        messages = messages[: r["keep"]] + r["messages"]
        tools = r["tools"] if "tools" in r else tools
        out.append({**r, "messages": list(messages), "tools": tools})
    return out


def pairs_of(rows):
    """[(row k, full request k+1)] for gameplay replies whose next request is recorded."""
    full = rebuild(rows)
    return [(rows[k], full[k + 1]) for k in range(len(rows) - 1)
            if rows[k].get("gameplay") and not rows[k].get("scripted") and rows[k].get("tokens")
            and (rows[k].get("usage") or {}).get("prompt_tokens")]


def span_logprobs(entries, start, tokens):
    vals = []
    for i, t in enumerate(tokens):
        e = entries[start + i] if start + i < len(entries) else None
        hit = (e or {}).get(str(t))
        vals.append(float(hit["logprob"] if isinstance(hit, dict) else hit) if hit is not None else float("nan"))
    return vals



def build(rollouts, depths, per_depth=1):
    """Probes from recorded tries: for each target depth (prompt tokens), the ``per_depth`` gameplay requests whose
    prompt is the closest at or below it, one per try, each with its reply span (start, tokens)."""
    cands = []
    for path in rollouts:
        path = Path(path)
        path = path / "rollout.jsonl" if path.is_dir() else path
        rows = [json.loads(line) for line in open(path)]
        for row, nxt in pairs_of(rows):
            cands.append((int(row["usage"]["prompt_tokens"]), str(path.parent), row, nxt, str(path)))
    out, used = [], set()
    for depth in sorted(depths):
        fit = sorted((c for c in cands if c[0] <= depth and (c[1], c[2]["n"]) not in used), key=lambda c: -c[0])
        tries = set()
        for start, label, row, nxt, source in fit:
            if label in tries:
                continue
            tries.add(label)
            used.add((label, row["n"]))
            label = Path(label).name
            body = {"messages": nxt["messages"], "chat_template_kwargs": nxt["template"]}
            if nxt.get("tools") is not None:
                body["tools"] = nxt["tools"]
            out.append({"label": f"{label}-r{row['n']}", "depth": depth, "start": start, "tokens": row["tokens"],
                        "body": body, "rollout": source, "request": row["n"], "engine": row.get("logprobs")})
            if len(tries) >= per_depth:
                break
    return out


def load(path):
    return [json.loads(line) for line in open(path)]


# ---------------------------------------------------------------------------- dump
async def score(post, model, p):
    body = {"model": model, **p["body"], "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0}
    data = await post(body)
    entries = data.get("prompt_logprobs") or ((data.get("choices") or [{}])[0].get("prompt_logprobs")) or []
    return span_logprobs(entries, int(p["start"]), p["tokens"])


async def dump(post, model, probes, *, together=False, log=None):
    """``post(body) -> response json`` (a chat/completions call). One row per probe: the continuation's logprobs
    (``together``: every probe sent at once)."""
    began = time.time()
    if together:
        got = await asyncio.gather(*(score(post, model, p) for p in probes))
    else:
        got = []
        for i, p in enumerate(probes):
            got.append(await score(post, model, p))
            if log:
                log({"event": "kv_golden_probe", "i": i, "label": p["label"], "depth": p["depth"]})
    rows = [{"label": p["label"], "depth": p["depth"], "start": p["start"], "n": len(p["tokens"]), "logprobs": lp}
            for p, lp in zip(probes, got)]
    return {"model": model, "together": together, "probes": rows, "seconds": round(time.time() - began, 1)}


def kernel(server_dir, out, card=None, timeout=600):
    """The int4 page golden on this card with the server's own python (``vllm-server-identity.json`` argv[0])."""
    path = Path(server_dir) / "vllm-server-identity.json"
    if not path.is_file():  # a stub server (CPU preflight) or another serving stack
        return {"rc": None, "out": str(out), "seconds": 0.0, "golden": None, "tail": f"no {path}"}
    identity = json.loads(path.read_text())
    exe = Path(identity["argv"][0])
    python = exe if "python" in exe.name else exe.with_name("python")
    env = {**os.environ, **({"CUDA_VISIBLE_DEVICES": str(card)} if card is not None else {})}
    began = time.time()
    try:
        r = subprocess.run([str(python), "-m", "vllm.arc_kv4", "golden", str(out)], env=env, capture_output=True,
                           text=True, timeout=timeout)
        text, rc = r.stdout + r.stderr, r.returncode
    except (OSError, subprocess.TimeoutExpired) as e:
        text, rc = f"{type(e).__name__}: {e}", None
    line = next((x for x in text.splitlines() if x.startswith("ARC_KV4_GOLDEN ")), None)
    return {"rc": rc, "out": str(out), "seconds": round(time.time() - began, 1),
            "golden": json.loads(line.split(" ", 1)[1]) if line else None, "tail": None if line else text[-600:]}


# ---------------------------------------------------------------------------- compare
def diff(a, b):
    """|dlogp| of two dumps over the probes both hold (pooled tokens); per-probe max for the report."""
    bb = {p["label"]: p for p in b["probes"]}
    d, per, missing = [], {}, []
    for p in a["probes"]:
        q = bb.get(p["label"])
        if q is None or q["n"] != p["n"]:
            missing.append(p["label"])
            continue
        x = [abs(u - v) for u, v in zip(p["logprobs"], q["logprobs"]) if math.isfinite(u) and math.isfinite(v)]
        missing += [p["label"]] if len(x) != p["n"] else []
        d += x
        per[p["label"]] = round(max(x), 5) if x else None
    if not d:
        return {"tokens": 0, "missing": missing}
    s = sorted(d)
    return {"tokens": len(d), "mean_abs": sum(d) / len(d), "max_abs": s[-1], "p99_abs": s[int(0.99 * (len(s) - 1))],
            "missing": missing, "per_probe_max": per}


def boot_verdict(dumps, ref, floor_dump, bars=BOOT_BARS):
    """``dumps``: {server: one-at-a-time dump}; ``ref``: the reference server's name; ``floor_dump``: its all-at-once
    dump. Per server max / mean |dlogp| vs the reference; pass needs every probe scored on both and both bars."""
    keys = ("tokens", "p99_abs", "mean_abs", "max_abs", "missing")
    floor = diff(dumps[ref], floor_dump)
    out = {"ref": ref, "bars": dict(bars), "floor": {k: floor.get(k) for k in keys}, "servers": {}}
    ok = floor.get("tokens", 0) > 0 and not floor["missing"]
    for name, d in sorted(dumps.items()):
        if name == ref:
            continue
        x = diff(d, dumps[ref])
        row = {k: x.get(k) for k in keys}
        row["ok"] = bool(x.get("tokens")) and not x["missing"] and floor.get("tokens", 0) > 0 \
            and len(d["probes"]) == len(dumps[ref]["probes"]) \
            and x["p99_abs"] <= bars["p99_vs_floor"] * floor["p99_abs"] \
            and x["mean_abs"] <= bars["mean_vs_floor"] * floor["mean_abs"]
        ok = ok and row["ok"]
        out["servers"][name] = row
    out["ok"] = bool(ok)
    return out


async def boot_dump(servers, spec, folder, *, model, server_dir=None, log=None):
    """rl_seq (``mg.redesign.kv_golden``): every server's dump one probe at a time (the servers at once), then the
    reference server's all-at-once dump, then (``kernel``) the int4 page golden. Never raises: play starts either way.
    ``spec``: {"probes": path, "ref": server index (default 1 with two or more servers), "kernel": false, "adapter":
    a trained adapter folder, loaded on every server as "kv-golden" for the dump and unloaded after (without it the
    dump runs under ``model``, the zero-delta play adapter: no LoRA path is exercised)}."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    began = time.time()
    names = [getattr(s, "name", None) or f"server{i + 1}" for i, s in enumerate(servers)]
    ref_i = int(spec.get("ref", 1 if len(servers) > 1 else 0))
    adapter = str(spec["adapter"]) if spec.get("adapter") else None
    model = ADAPTER_NAME if adapter else model
    row = {"model": model, "ref": names[ref_i], "adapter": adapter}
    try:
        if adapter:
            await asyncio.gather(*(s.load(ADAPTER_NAME, adapter) for s in servers))
        probes = load(spec["probes"])

        def poster(s):
            return lambda body: s.post("/v1/chat/completions", body, timeout=900)

        got = await asyncio.gather(*(dump(poster(s), model, probes) for s in servers), return_exceptions=True)
        dumps, errors = {}, {}
        for name, g in zip(names, got):
            if isinstance(g, BaseException):
                errors[name] = f"{type(g).__name__}: {g}"[:300]
            else:
                dumps[name] = g
                (folder / f"{name}.json").write_text(json.dumps(g))
        floor_dump = await dump(poster(servers[ref_i]), model, probes, together=True)
        (folder / f"{names[ref_i]}-together.json").write_text(json.dumps(floor_dump))
        row.update(boot_verdict(dumps, names[ref_i], floor_dump) if names[ref_i] in dumps else {"ok": False})
        if errors:
            row.update(errors=errors, ok=False)
    except Exception as e:  # noqa: BLE001 - recorded
        row.update(error=f"{type(e).__name__}: {e}"[:300], ok=False)
    if adapter:  # play goes on under the play adapter (max_loras 1: its first request reloads the slot)
        gone = await asyncio.gather(*(s.unload(ADAPTER_NAME) for s in servers), return_exceptions=True)
        bad = {n: f"{type(g).__name__}: {g}"[:200] for n, g in zip(names, gone) if isinstance(g, BaseException)}
        if bad:
            row["unload_errors"] = bad
    if spec.get("kernel") and server_dir:
        kern = await asyncio.to_thread(kernel, server_dir, folder / "kernel-golden.pt", spec.get("kernel_card"))
        (folder / "kernel.json").write_text(json.dumps(kern))
        row["kernel"] = {k: kern[k] for k in ("rc", "seconds", "golden")}
    row.update(seconds=round(time.time() - began, 1), folder=str(folder))
    (folder / "verdict.json").write_text(json.dumps(row, indent=1, default=str))
    if log:
        log(row)
    return row
