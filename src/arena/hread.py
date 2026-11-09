"""The redesign's held-out read: whole held-out games on the Nebius RTX competition
stack, the base once at step 0, the policy adapter every ``every`` training steps, same bundle and seeds.

- Hand-off (``export_adapter``): rl_seq copies the step's served adapter folder to ``<volume run>/hread/s<step>/`` with
  ``hread-adapter.json`` (size + sha256 per file, ``digest`` over them, ``draft``) and only then emits the feed row
  ``hread_due``. The the ``launch/rl/heldout/hread.py`` mirrors it to the Nebius bucket (manifest in
  nebius_mirror's format, which the VM checks file by file) and queues the read.
- Draft: an adapter with stage D's draft LoRA files is ``draft: "lora"`` (the read server runs ARC_DRAFT_LORA=1 and
  merges it into the draft at activation); without them the server serves the base draft. ``draft_kind`` reads the
  safetensors headers only.
- Scorer (``score``): per try the competition game score of the levels cleared (``game_score``, as rl_loop) as a
  fraction (RHAE), its completion tokens; per game RHAE per 10K tokens (ratio of sums), base vs policy paired by (game,
  pass) (same seeds: vLLM seed = seed_base + pass), delta-method paired SE; the aggregate is the mean over games. Matched
  tokens: the score each try had reached at T tokens, paired at every T both tries reached. ``ref_ratios``: every
  try's levels vs v8.5p's per (game, level) (clear / fail rate, tokens, RHAE/tok as logeq), all games and levels.
- In-run exporter (``ReadMetrics``): the level bench's progress lines -> ``train-metrics.jsonl`` rows (dashboard keys:
  eval.heldout_rhae10k with its base, tok/s, KV, util, VRAM, cost).
"""
import hashlib
import json
import math
import os
import shutil
import statistics
import struct
import time
from pathlib import Path

MANIFEST = "hread-adapter.json"
DROP = ("censored", "error", "replay_mismatch", "recorded")  # left out of every read (as level_bench.HEADLINE_EXCLUDED)
GRID = (25_000, 50_000, 100_000, 150_000, 200_000)


# ---- hand-off ------------------------------------------------------------------------------------------------------

def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def file_manifest(folder, skip=(MANIFEST,)):
    """{relative path: {"size", "sha256"}} of every file under ``folder`` (nebius_mirror's entry format)."""
    folder = Path(folder)
    return {str(p.relative_to(folder)): {"size": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(folder.rglob("*")) if p.is_file() and p.name not in skip}


def digest(files):
    """One sha256 over the sorted "<path> <sha256>" lines (the hand-off's identity, end to end)."""
    return hashlib.sha256("".join(f"{k} {v['sha256']}\n" for k, v in sorted(files.items())).encode()).hexdigest()


def safetensor_names(path):
    """Tensor names of a safetensors file from its header (no tensor data read)."""
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if size > 100 << 20:
            raise ValueError(f"{path}: safetensors header of {size} bytes")
        return [k for k in json.loads(f.read(size)) if k != "__metadata__"]


def draft_kind(folder):
    """"lora" when the adapter carries stage D's draft LoRA (rl1_draft.export: ``draft_lora.json`` + its full
    ``draft_lora.safetensors``, 2 tensors per module; ARC_DRAFT_LORA servers merge it at activation), None without
    draft files (the server serves the base draft). Partial files, or draft weights in any other form (``mtp.`` tensors
    in another file, a ``draft/`` folder), raise: the server would silently serve the base draft."""
    folder = Path(folder)
    meta, tensors = folder / "draft_lora.json", folder / "draft_lora.safetensors"
    if meta.is_file() or tensors.is_file():
        try:
            modules = json.loads(meta.read_text())["modules"]
            count = len(safetensor_names(tensors))
        except (OSError, ValueError, KeyError, struct.error) as exc:
            raise ValueError(f"adapter {folder}: unreadable or partial draft LoRA files ({exc})") from exc
        if not 0 < count == 2 * len(modules):
            raise ValueError(f"adapter {folder}: draft file {count} tensors for {len(modules)} modules")
        return "lora"
    if (folder / "draft").is_dir() and any((folder / "draft").iterdir()):
        raise ValueError(f"adapter {folder}: a draft/ folder is no draft format the server loads")
    for f in sorted(folder.glob("*.safetensors")):
        if any(".mtp." in f".{n}" for n in safetensor_names(f)):
            raise ValueError(f"adapter {folder}: mtp. tensors in {f.name} (the draft LoRA goes in draft_lora.*)")
    return None


def export_adapter(src, dest, meta=None):
    """Copy the adapter folder ``src`` to ``dest`` (a temp folder renamed into place) with its manifest; every copied
    file is read back and checked against the source's hash. Returns the manifest."""
    src, dest = Path(src), Path(dest)
    files = file_manifest(src)
    if "adapter_model.safetensors" not in files or "adapter_config.json" not in files:
        raise ValueError(f"{src}: not a PEFT adapter folder ({sorted(files)[:6]})")
    draft = draft_kind(src)  # refuses partial draft files before anything is copied
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, ignore=shutil.ignore_patterns(MANIFEST))
    got = file_manifest(tmp)
    if got != files:
        bad = sorted(k for k in set(files) | set(got) if files.get(k) != got.get(k))
        shutil.rmtree(tmp, ignore_errors=True)
        raise IOError(f"adapter copy {src} -> {dest}: {len(bad)} files differ, e.g. {bad[:3]}")
    out = {**(meta or {}), "files": files, "total_bytes": sum(v["size"] for v in files.values()),
           "digest": digest(files), "draft": draft, "created_epoch": round(time.time())}
    (tmp / MANIFEST).write_text(json.dumps(out, indent=1) + "\n")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return out


def verify_tree(folder, files):
    """Problems of a copy against a manifest's ``files`` ([] = exact); the hand-off manifest itself counts only when
    listed (the bucket manifest lists it, the trainer's own does not)."""
    got = file_manifest(folder, skip=() if MANIFEST in files else (MANIFEST,))
    out = [f"missing {k}" for k in files if k not in got]
    out += [f"{k}: size {got[k]['size']} != {v['size']}" for k, v in files.items() if k in got and got[k]["size"] != v["size"]]
    out += [f"{k}: sha256 differs" for k, v in files.items() if k in got and got[k]["size"] == v["size"]
            and got[k]["sha256"] != v["sha256"]]
    return out + [f"unexpected {k}" for k in got if k not in files]


def bucket_manifest(files, prefix, source):
    """The bucket manifest (assets-manifests/<slug>.json) as nebius_mirror.build_manifest writes it."""
    return {"prefix": prefix, "source": source, "created_epoch": round(time.time()),
            "total_bytes": sum(v["size"] for v in files.values()), "files": files}


def asset_slug(prefix):
    return prefix.strip("/").replace("/", "--")


def publish_folder(src, prefix, remote, put, ls, source="local", tmp="/tmp"):
    """The hand-off to the bucket: ``src`` checked against its manifest, ``put`` (copy a folder, or one file with
    ``single``) to ``<remote>/assets/<prefix>``, remote sizes from ``ls`` compared, and only then the bucket manifest
    ``<remote>/assets-manifests/<slug>.json`` (the launcher treats it as "mirrored"; the VM checks every file)."""
    src = Path(src)
    man = json.loads((src / MANIFEST).read_text())
    probs = verify_tree(src, man["files"])
    if probs or digest(man["files"]) != man["digest"]:
        raise RuntimeError(f"{src}: does not match its hand-off manifest: {probs[:5] or 'digest'}")
    began = time.time()
    put(str(src), f"{remote}/assets/{prefix}")
    files = {**man["files"], MANIFEST: {"size": (src / MANIFEST).stat().st_size, "sha256": sha256_file(src / MANIFEST)}}
    sizes = ls(f"{remote}/assets/{prefix}")
    wrong = sorted(k for k, v in files.items() if sizes.get(k) != v["size"])
    if wrong:
        raise RuntimeError(f"{prefix}: {len(wrong)} remote files differ in size, e.g. {wrong[:3]}")
    out = bucket_manifest(files, prefix, f"{source}/{src}")
    path = Path(tmp) / f"{asset_slug(prefix)}.json"
    path.write_text(json.dumps(out, indent=1))
    put(str(path), f"{remote}/assets-manifests/{path.name}", single=True)
    return {"prefix": prefix, "digest": man["digest"], "files": len(files), "bytes": out["total_bytes"],
            "draft": man.get("draft"), "step": man.get("step"), "expert_mask_sha256": man.get("expert_mask_sha256"),
            "upload_seconds": round(time.time() - began, 1)}


def mask_problem(adapter_dir, study):
    """None when the adapter was trained under the read server's expert mask, else why not. The adapter's mask is
    the one rl_seq recorded in its hand-off manifest (``expert_mask_sha256``); a mask refit keeps every shape but
    re-maps experts (r15 -> r15b: ~10 per layer), so a mismatch is silent at load and must be refused here."""
    want = ((study.get("server") or {}).get("expert_mask") or {}).get("sha256")
    path = Path(adapter_dir) / MANIFEST
    if not path.is_file():
        return f"{adapter_dir}: no {MANIFEST} (the training mask is unknown)"
    got = json.loads(path.read_text()).get("expert_mask_sha256")
    if got != want:
        return f"{adapter_dir}: trained under expert mask {str(got)[:12]}, the read serves {str(want)[:12]}"
    return None


# ---- scorer --------------------------------------------------------------------------------------------------------

def game_score(cleared, total):
    """The competition's game score (rl_loop.game_score): level k weighted k, a cleared level's score (0-115), capped at
    the scored levels' weight share x 100. ``cleared``: {level: score}."""
    total = max(int(total or 0), max([int(k) for k in cleared] or [0]))
    if total <= 0:
        return 0.0
    weights = sum(range(1, total + 1))
    got = sum(float(sc) * int(lv) for lv, sc in cleared.items())
    scored = sum(int(lv) for lv, sc in cleared.items() if float(sc) > 0)
    return min(got / weights, 100.0 * scored / weights)


def try_rhae(row, at=None):
    """The try's game score as a fraction (RHAE) from the levels it cleared by ``at`` completion tokens (all: None)."""
    cleared = {int(r["level"]): float(r.get("score") or 0.0) for r in row.get("level_rows") or []
               if at is None or int(r.get("at_tokens") or 0) <= at}
    return game_score(cleared, row.get("total_levels")) / 100.0


def reached(row, at):
    """The try's score at ``at`` tokens is known: it played that far, or it won the game before."""
    return int(row.get("completion_tokens") or 0) >= at or row.get("outcome") == "won"


def code(row):
    return str(row["game"])[:4]


def usable(rows, censored):
    """``censored``: "drop" (read A: unfinished tries left out) or "partial" (read B: a censored try counts with the
    levels and tokens it had at the cut). Errors and replay mismatches are always left out."""
    keep = ("censored",) if censored == "partial" else ()
    return [r for r in rows if r.get("outcome") not in DROP or r.get("outcome") in keep]


def ratio_stats(pairs):
    """Per game: base and policy RHAE per 10K tokens (ratio of sums) and their paired delta-method variances.
    ``pairs``: [(base row, policy row)]."""
    n = len(pairs)
    sb = [try_rhae(b) for b, _ in pairs]
    sp = [try_rhae(p) for _, p in pairs]
    tb = [int(b.get("completion_tokens") or 0) for b, _ in pairs]
    tp = [int(p.get("completion_tokens") or 0) for _, p in pairs]
    if not n or not sum(tb) or not sum(tp):
        return None
    rb, rp = sum(sb) / sum(tb), sum(sp) / sum(tp)
    mb, mp = sum(tb) / n, sum(tp) / n
    eb = [(s - rb * t) / mb for s, t in zip(sb, tb)]  # linearized residuals (mean 0)
    ep = [(s - rp * t) / mp for s, t in zip(sp, tp)]
    k = 1.0 / (n * (n - 1)) if n > 1 else None
    var = (lambda xs: k * sum(x * x for x in xs)) if k else (lambda xs: None)
    return {"pairs": n, "base": rb * 1e4, "policy": rp * 1e4, "var_base": None if k is None else var(eb) * 1e8,
            "var_policy": None if k is None else var(ep) * 1e8,
            "var_diff": None if k is None else var([a - b for a, b in zip(ep, eb)]) * 1e8,
            "tokens_base": sum(tb), "tokens_policy": sum(tp), "rhae_base": sum(sb), "rhae_policy": sum(sp)}


def pair_rows(base_rows, policy_rows, censored):
    """{game: [(base, policy)]} matched by (game, pass); a key with more than one row per side is an error."""
    def index(rows, side):
        out = {}
        for r in usable(rows, censored):
            key = (code(r), int(r["pass"]))
            if key in out:
                raise ValueError(f"{side}: two tries for {key}")
            out[key] = r
        return out

    b, p = index(base_rows, "base"), index(policy_rows, "policy")
    games = {}
    for key in sorted(set(b) & set(p)):
        games.setdefault(key[0], []).append((b[key], p[key]))
    return games, sorted(set(b) ^ set(p))


def ratio_ci(p, b, var_p, var_b, var_d):
    """policy / base with a 95% interval (log delta method; the paired covariance from var_d)."""
    if not b or not p or None in (var_p, var_b, var_d):
        return None
    cov = (var_p + var_b - var_d) / 2.0
    v = var_p / p ** 2 + var_b / b ** 2 - 2.0 * cov / (p * b)
    h = 1.96 * math.sqrt(max(0.0, v))
    r = p / b
    return [round(r, 4), round(r * math.exp(-h), 4), round(r * math.exp(h), 4)]


def matched(games, grid):
    """Score at T tokens (game score %, and RHAE per 10K at T) for both arms over the pairs that reached T on both
    sides; game-weighted means and a paired SE."""
    out = []
    for at in grid:
        per = {}
        for g, pairs in games.items():
            ds = [(try_rhae(b, at), try_rhae(p, at)) for b, p in pairs if reached(b, at) and reached(p, at)]
            if ds:
                per[g] = ds
        if not per:
            continue
        mb = statistics.mean(statistics.mean(x for x, _ in v) for v in per.values())
        mp = statistics.mean(statistics.mean(y for _, y in v) for v in per.values())
        var = [statistics.variance([y - x for x, y in v]) / len(v) for v in per.values() if len(v) > 1]
        se = math.sqrt(sum(var)) / len(per) if len(var) == len(per) else None
        out.append({"tokens": at, "games": len(per), "pairs": sum(len(v) for v in per.values()),
                    "score_base": round(100 * mb, 3), "score_policy": round(100 * mp, 3),
                    "rhae10k_base": round(mb / at * 1e4, 4), "rhae10k_policy": round(mp / at * 1e4, 4),
                    "diff_rhae10k": round((mp - mb) / at * 1e4, 4), "se_diff": None if se is None else round(se / at * 1e4, 4)})
    return out


def score(base_rows, policy_rows, censored="drop", grid=GRID, budget=None):
    """The read's verdict, one censoring mode: per game and the game-mean aggregate, paired SE, ratio, matched-token
    table. Bar (pre-registered): policy - base >= 2 SE_diff."""
    games, unpaired = pair_rows(base_rows, policy_rows, censored)
    per = {g: ratio_stats(v) for g, v in games.items()}
    per = {g: s for g, s in per.items() if s is not None}
    if not per:
        return {"censored": censored, "games": {}, "unpaired": unpaired, "aggregate": None}
    G = len(per)
    agg = {"games": G, "pairs": sum(s["pairs"] for s in per.values()),
           "base": sum(s["base"] for s in per.values()) / G, "policy": sum(s["policy"] for s in per.values()) / G}
    full = all(s["var_diff"] is not None for s in per.values())
    for k in ("var_base", "var_policy", "var_diff"):
        agg[k] = sum(s[k] for s in per.values()) / G ** 2 if full else None
    agg["diff"] = agg["policy"] - agg["base"]
    agg["se_diff"] = None if agg["var_diff"] is None else math.sqrt(agg["var_diff"])
    agg["se_base"] = None if agg["var_base"] is None else math.sqrt(agg["var_base"])
    agg["se_policy"] = None if agg["var_policy"] is None else math.sqrt(agg["var_policy"])
    agg["ratio"] = ratio_ci(agg["policy"], agg["base"], agg["var_policy"], agg["var_base"], agg["var_diff"])
    agg["passes_bar"] = agg["se_diff"] is not None and agg["diff"] >= 2 * agg["se_diff"]
    rnd = lambda d: {k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in d.items()}
    grid = [t for t in grid if budget is None or t <= budget]
    return {"censored": censored, "games": {g: rnd(s) for g, s in sorted(per.items())}, "unpaired": unpaired,
            "aggregate": rnd(agg), "matched_tokens": matched(games, grid)}


def same_read(base_bench, policy_bench):
    """Differences between two reads' level-bench.json settings that would break the pairing ([] = same)."""
    keys = ("points", "window", "completion_cap", "reasoning_effort", "temperature")
    out = [k for k in keys if base_bench.get(k) != policy_bench.get(k)]
    sb, sp = dict(base_bench.get("settings") or {}), dict(policy_bench.get("settings") or {})
    for k in set(sb) | set(sp):
        if k != "arms" and sb.get(k) != sp.get(k):
            out.append(f"settings.{k}")
    return sorted(out)


# ---- v8.5p references (every RL metric is a ratio to the same (game, level)'s v8.5p median) ---

def history_refs(rows):
    """{(game, level): clear_rate, median_tokens, median_to_clear, median_actions, n} from rl_backlog history rows.
    v8.5p never fails a level outright (a level not cleared ended with its run, "censored"): counted as not cleared."""
    by = {}
    for r in rows:
        by.setdefault((str(r["game"])[:4], int(r["level"])), []).append(r)
    out = {}
    for k, rs in by.items():
        toks = [int(r["tokens"]) for r in rs if int(r.get("tokens") or 0) > 0]
        clr = [r for r in rs if r.get("cleared") and r.get("tokens") and r.get("actions")]
        out[k] = {"n": len(rs), "clear_rate": sum(bool(r.get("cleared")) for r in rs) / len(rs),
                  "rows": [(int(r["level"]), int(r.get("tokens") or 0), int(r.get("actions") or 0),
                            bool(r.get("cleared")), bool(r.get("censored"))) for r in rs],
                  "median_tokens": statistics.median(toks) if toks else None,
                  "median_to_clear": statistics.median(int(r["tokens"]) for r in clr) if clr else None,
                  "median_actions": statistics.median(int(r["actions"]) for r in clr) if clr else None}
    return out


def level_units(row):
    """(game, level, tokens, actions, cleared) per level a try played: its clears, then the level it stopped on as not
    cleared (rl_backlog.extract_run's rows for v8.5p)."""
    g = code(row)
    lr = sorted(row.get("level_rows") or [], key=lambda r: int(r["level"]))
    out = [(g, int(r["level"]), int(r.get("tokens") or 0), int(r.get("actions") or 0), True) for r in lr]
    nxt = int(lr[-1]["level"]) + 1 if lr else int(row.get("level") or 1)
    total = int(row.get("total_levels") or 0)
    if row.get("outcome") != "won" and (not total or nxt <= total):
        tok = int(row.get("completion_tokens") or 0) - max([int(r.get("at_tokens") or 0) for r in lr] or [0])
        act = int(row.get("total_actions") or row.get("actions") or 0) - sum(u[3] for u in out)
        if tok > 0:
            out.append((g, nxt, tok, max(act, 0), False))
    return out


def unit_logeq(unit, ref, baselines, cap=1.15, clip=8.0, fail=16.0):
    """The trained logeq reward of one level vs its reference (export_rl.logeq): clip(log((R / R_ref) x (T_ref / t)),
    +-log clip), a level not cleared -log fail."""
    _, lvl, tok, act, cleared = unit
    if not cleared or tok <= 0 or act <= 0:
        return -math.log(fail)
    b = baselines[lvl - 1] if 0 < lvl <= len(baselines) and baselines[lvl - 1] else None
    t_b, a_b = ref.get("median_to_clear") or tok, ref.get("median_actions") or act
    rhae = lambda a: min(cap, (b / a) ** 2) if b and a else 1.0
    v = math.log(rhae(act) / rhae(float(a_b))) + math.log(float(t_b) / tok)
    return max(-math.log(clip), min(math.log(clip), v))


def ref_logeq(unit, ref, baselines):
    """The unit's v8.5p mean logeq, its fails included (export_rl.ref_logeq with no group budget: the history's own
    1.25 x p80 of clears cuts; a censored try past it is a fail, one cut before it unknown)."""
    if "_mean" not in ref:
        rows = ref.get("rows") or []
        clears = sorted(t for _, t, _, c, _ in rows if c and t)
        b = None
        if clears:
            pos = (len(clears) - 1) * 0.8
            lo, hi = int(pos), min(int(pos) + 1, len(clears) - 1)
            b = 1.25 * (clears[lo] + (clears[hi] - clears[lo]) * (pos - lo))
        vals = []
        for lvl, tok, act, cleared, censored in rows:
            if cleared and (b is None or tok <= b):
                vals.append(unit_logeq((unit[0], lvl, tok, act, True), ref, baselines))
            elif cleared or b is None or tok >= b or not censored:
                vals.append(unit_logeq((unit[0], lvl, tok, act, False), ref, baselines))
        ref["_mean"] = sum(vals) / len(vals) if vals else None
    return ref["_mean"]


def ref_at(unit, ref, baselines, budget):
    """(v8.5p's mean logeq, its clear rate, rows used) on the unit had each of its tries had ``budget`` tokens for
    the level: a clear within the budget scores, a clear past it or a try still
    playing at it is a fail (-ln16), a run that ended before the budget without a clear is unknown (left out).
    ``budget`` None: no cut (every non-clear a fail)."""
    key = int(budget) if budget else None
    cache = ref.setdefault("_at", {})
    if key not in cache:
        vals, clears = [], 0
        for lvl, tok, act, cleared, censored in ref.get("rows") or []:
            if cleared and (key is None or tok <= key):
                vals.append(unit_logeq((unit[0], lvl, tok, act, True), ref, baselines))
                clears += 1
            elif cleared or key is None or tok >= key or not censored:
                vals.append(-math.log(16.0))
        cache[key] = (sum(vals) / len(vals), clears / len(vals), len(vals)) if vals else (None, None, 0)
    return cache[key]


def unit_budgets(units, budget):
    """Each level unit's own budget: the try's ``budget`` less the tokens its earlier levels took (None: none)."""
    out, used = [], 0
    for u in units:
        out.append(max(1, budget - used) if budget else None)
        used += u[2]
    return out


def ref_ratios(rows, refs, baselines=None, budget=None, budgets=None):
    """A read's tries vs v8.5p per (game, level), over all games and levels (no L1/L3 split): clears and fails
    as observed / expected (the sum of each level's v8.5p clear rate), tokens per level and RHAE/tok (logeq) as geometric
    means of the per-level ratios; per game the same. Levels without a reference are counted, not scored.

    Censoring-matched (the base read's 0.65 was its whole-try 200K budget cutting levels that v8.5p's
    reference leaves out): each unit is compared with v8.5p at the unit's own budget (``ref_at``): ``budget`` the
    whole-try token budget (a read), or ``budgets`` {row label: per-level budget} (training groups)."""
    units, ubud = [], []
    for r in usable(rows, "partial"):
        us = level_units(r)
        units += us
        b = (budgets or {}).get(r.get("label"))
        ubud += [b] * len(us) if b else unit_budgets(us, budget)
    bud = dict(zip(map(id, units), ubud))

    def agg(us):
        cov = [(u, refs[u[:2]]) for u in us if u[:2] in refs]
        m = {"levels": len(us), "covered": len(cov)}
        at = [(u, x, ref_at(u, x, (baselines or {}).get(u[0]) or [], bud[id(u)])) for u, x in cov]
        at = [(u, x, a) for u, x, a in at if a[0] is not None]
        m["matched"] = len(at)
        exp_c, obs = sum(a[1] for _, _, a in at), sum(u[4] for u, _, _ in at)
        m["clears"], m["clears_ref"] = obs, round(exp_c, 3)
        m["clear_rate_vs_ref"] = obs / exp_c if exp_c else None
        m["fail_rate_vs_ref"] = (len(at) - obs) / (len(at) - exp_c) if len(at) - exp_c > 0 else None
        tk = [math.log(u[2] / x["median_tokens"]) for u, x in cov if x.get("median_tokens") and u[2] > 0]
        m["tokens_vs_ref"] = math.exp(sum(tk) / len(tk)) if tk else None
        rel = [unit_logeq(u, x, (baselines or {}).get(u[0]) or []) - a[0] for u, x, a in at]
        m["rhae_tok_vs_ref"] = math.exp(sum(rel) / len(rel)) if rel else None  # vs v8.5p at the unit's own budget
        old = [unit_logeq(u, x, (baselines or {}).get(u[0]) or []) - r for u, x in cov
               if (r := ref_logeq(u, x, (baselines or {}).get(u[0]) or [])) is not None]
        m["rhae_tok_vs_ref_unmatched"] = math.exp(sum(old) / len(old)) if old else None  # the pre-29 Oct score
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}

    games = {}
    for u in units:
        games.setdefault(u[0], []).append(u)
    return {"all": agg(units), "games": {g: agg(us) for g, us in sorted(games.items())},
            "no_ref": sorted({f"{u[0]}-L{u[1]}" for u in units if u[:2] not in refs})}


# ---- in-run exporter -----------------------------------------------------------------------------------------------

def tail_json(path, limit=65536):
    """The last JSON line of a file (None when there is none)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - limit))
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def live_rhae(rows):
    """Game-mean RHAE per 10K over finished usable tries (read B view of what has finished) and its SE."""
    by = {}
    for r in usable(rows, "partial"):
        by.setdefault(code(r), []).append(r)
    vals = []
    for g, rs in by.items():
        t = sum(int(r.get("completion_tokens") or 0) for r in rs)
        if not t:
            continue
        s = [try_rhae(r) for r in rs]
        ratio = sum(s) / t
        se = None
        if len(rs) > 1:
            m = t / len(rs)
            e = [(si - ratio * int(r.get("completion_tokens") or 0)) / m for si, r in zip(s, rs)]
            se = math.sqrt(sum(x * x for x in e) / (len(rs) * (len(rs) - 1))) * 1e4
        vals.append((ratio * 1e4, se))
    if not vals:
        return None, None
    v = sum(x for x, _ in vals) / len(vals)
    se = math.sqrt(sum(s * s for _, s in vals)) / len(vals) if all(s is not None for _, s in vals) else None
    return v, se


class ReadMetrics:
    """``study.hread`` set: every level-bench progress line also becomes one dashboard row in
    ``<trial>/train-metrics.jsonl`` ({"t", "m"}: arc_infra.runops.metrics rows; the watcher mirrors it)."""

    def __init__(self, output, hread, started=None):
        self.path = Path(output) / "train-metrics.jsonl"
        self.output = Path(output)
        self.cfg = dict(hread or {})
        self.started = started or time.time()
        self.prev = None  # (t, generated tokens, prefix queries, prefix hits)

    def row(self, line, rows, engine=None, now=None):
        now = now or time.time()
        m = {"play.done": line.get("done"), "play.running": line.get("running"), "play.errors": line.get("failed"),
             "play.levels": sum(int(r.get("levels") or 0) for r in rows), "infra.gen_tokens": line.get("completion_tokens"),
             "eval.read_step": self.cfg.get("step")}
        gen = line.get("completion_tokens")
        q = h = None
        if engine:
            m["infra.kv_frac"] = engine.get("vllm:kv_cache_usage_perc")
            m["infra.running"] = engine.get("vllm:num_requests_running")
            m["infra.waiting"] = engine.get("vllm:num_requests_waiting")
            m["infra.preemptions"] = engine.get("vllm:num_preemptions_total")
            q, h = engine.get("vllm:prefix_cache_queries_total"), engine.get("vllm:prefix_cache_hits_total")
            acc, drafts = (engine.get("vllm:spec_decode_num_accepted_tokens_total"),
                           engine.get("vllm:spec_decode_num_draft_tokens_total"))
            if acc is not None and drafts:
                m["infra.draft_acceptance"] = acc / drafts
        if self.prev is not None and now > self.prev[0]:
            if gen is not None and self.prev[1] is not None and gen >= self.prev[1]:
                m["infra.gen_tok_s"] = (gen - self.prev[1]) / (now - self.prev[0])
            if None not in (q, h, self.prev[2], self.prev[3]) and q > self.prev[2]:
                m["infra.prefix_hit"] = (h - self.prev[3]) / (q - self.prev[2])
        self.prev = (now, gen, q, h)
        gpu = tail_json(self.output / "vllm-gpu-memory.jsonl")
        for i, g in enumerate((gpu or {}).get("gpus") or []):
            m[f"infra.gpu_util.{i}"] = g.get("utilization.gpu")
            if g.get("memory.used") is not None:
                m[f"infra.vram_gib.{i}"] = g["memory.used"] / 1024
            if i == 0 and g.get("memory.total"):
                m["infra.vram_cap_gib"] = g["memory.total"] / 1024
        if self.cfg.get("usd_per_hour"):  # list price from the bench start plus the boot (records hold the real bound)
            m["infra.cost_usd"] = float(self.cfg["usd_per_hour"]) * (
                now - self.started + float(self.cfg.get("boot_seconds") or 0)) / 3600
        v, se = live_rhae(rows)
        if v is not None:
            m["eval.heldout_rhae10k"], m["eval.heldout_se"] = v, se
        base = self.cfg.get("base") or {}
        if base.get("rhae10k") is not None:
            m["eval.heldout_rhae10k_base"] = base["rhae10k"]
            m["eval.heldout_se_base"] = base.get("se")
            if v is not None and base["rhae10k"]:
                m["eval.heldout_ratio"] = v / base["rhae10k"]
                m["eval.heldout_ratio_target"] = 2.0
        return {"t": now, "m": {k: x for k, x in m.items() if isinstance(x, (int, float)) and not isinstance(x, bool)}}

    def emit(self, line, rows, engine=None, now=None):
        row = self.row(line, rows, engine, now)
        if line.get("event") == "level_bench_final":
            row["note"] = "read end"
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row
