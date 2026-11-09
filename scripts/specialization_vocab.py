#!/usr/bin/env python3
"""CPU-only vocabulary / n-gram / draftability statistics of recorded arena runs (28 Sept).

Reads the per-game `log.jsonl` (model replies: reasoning, content, tool calls) and
`duck-transcript.txt` (system prompt, user prompts, tool results) of recorded Flash arena runs,
tokenizes them with the served tokenizer and writes the tables behind
`docs/validation-20260927/specialization.md`:

- output-token coverage (distinct tokens, K for 99/99.9/99.99/100% of generated tokens);
- held-out out-of-subset rates for restricted output heads (split by run, by game family, by effort);
- input-token coverage (for a restricted input embedding);
- PLE n-gram coverage: the Qwen3.8-Flash-Next PLE rows are hashed bigrams/trigrams of input ids
  (`nvidia/ple_layer.py`, 16 heads x 160 B, FP8), so a GPU-resident subset is sized by the
  distinct bigrams + trigrams, and its hit rate on held-out decode positions;
- a context-lookup (suffix) draft proposer's acceptance on generated tokens.

Usage: tokvenv/bin/python scripts/specialization_vocab.py extract|analyze|union|draft-vocab [--cache DIR]
(draft-vocab writes src/arc_infra/serving_draft_vocab.json, the MTP draft head subset of the serving tweaks)
Needs `tokenizers` and `numpy` (the scratchpad tokvenv has both). Nothing touches a GPU.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter

import numpy as np

RUNS = "./runs"
TOKENIZER = f"{RUNS}/serving-research/qwen-tokenizer/tokenizer.json"
V = 248320  # padded lm_head rows
EOS = 248044

# Primary corpus: Flash (Qwen3.8-Flash-Next-NVFP4) xhigh one-hour runs (c110 + r23, Modal + Kaggle).
PRIMARY_GLOBS = [
    "modal-validation/arena-duck-60m-c110-*",
    "modal-validation/arena-duck-60m-r23-*",
    "kaggle-arena-duck-60m-r23-*",
]
PRIMARY_EXCLUDE = ("effort-medium", "no-think", "ornith", "adaptive-think")
# Shift test sets (never used to build a subset).
SHIFT_RUNS = {
    "effort-medium-60m": ["kaggle-arena-duck-60m-r23-effort-medium-v1"],
    "effort-low-30m": ["kaggle-arena-duck-30m-fixed2x-effort-low-v1"],
    "no-think-60m": ["kaggle-arena-duck-60m-r23-no-think-v1"],
    "adaptive-think-60m": ["modal-validation/arena-duck-60m-r23-adaptive-think-v1-modal-20260927"],
    "competition-harness": [
        "kaggle-arena-flash-competition-fixed2x-v5",
        "kaggle-arena-flash-competition-v1",
    ],
    "30m-xhigh": [
        "modal-validation/arena-duck-30m-fixed2x-notes-v2-modal-20260925",
        "kaggle-arena-duck-30m-flash-control-v4",
        "kaggle-arena-duck-30m-inference-fifo-v1",
    ],
}

SECTION_RE = re.compile(r"^\[([A-Z][A-Z _:a-z0-9]+)\]\s*$")
INPUT_SECTIONS = {"SYSTEM PROMPT", "USER PROMPT", "TOOL RESULT: python"}


def primary_runs() -> list[str]:
    out = []
    for pat in PRIMARY_GLOBS:
        for run in sorted(glob.glob(os.path.join(RUNS, pat))):
            name = os.path.relpath(run, RUNS)
            if any(x in name for x in PRIMARY_EXCLUDE):
                continue
            out.append(name)
    return out


def render_output(msg: dict) -> str:
    """The generated text as the chat template renders it after the '<think>\\n' prompt."""
    reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
    content = (msg.get("content") or "").strip()
    text = reasoning + "\n</think>\n\n" + content
    calls = msg.get("tool_calls") or []
    for i, call in enumerate(calls):
        fn = call.get("function", call)
        if i == 0:
            text += "\n\n" if content else ""
        else:
            text += "\n"
        text += "<tool_call>\n<function=" + str(fn.get("name")) + ">\n"
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        if isinstance(args, dict):
            for k, v in args.items():
                v = v if isinstance(v, str) else json.dumps(v)
                text += "<parameter=" + k + ">\n" + v + "\n</parameter>\n"
        text += "</function>\n</tool_call>"
    return text + "<|im_end|>"


def game_records(game_dir: str):
    outs, usage, effort = [], [], []
    log = os.path.join(game_dir, "log.jsonl")
    for line in open(log, errors="replace"):
        if '"model"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("kind") != "model" or not isinstance(d.get("message"), dict):
            continue
        outs.append(render_output(d["message"]))
        usage.append(int(((d.get("usage") or {}).get("completion_tokens")) or -1))
        effort.append(str(d.get("reasoning_effort")))
    inputs = []
    tr = os.path.join(game_dir, "duck-transcript.txt")
    if os.path.exists(tr):
        seen = set()
        cur, buf = None, []

        def flush():
            if cur in INPUT_SECTIONS and buf:
                text = "".join(buf).strip("\n")
                h = hashlib.sha1(text.encode()).digest()
                if h not in seen:
                    seen.add(h)
                    inputs.append(text)

        for line in open(tr, errors="replace"):
            m = SECTION_RE.match(line)
            if m or line.startswith("--- analysis_step="):
                flush()
                cur, buf = (m.group(1) if m else None), []
                continue
            buf.append(line)
        flush()
    return outs, usage, effort, inputs


def extract(cache: str, runs: list[str]) -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER)
    os.makedirs(cache, exist_ok=True)
    for run in runs:
        dest = os.path.join(cache, run.replace("/", "__") + ".npz")
        if os.path.exists(dest):
            continue
        games = sorted(glob.glob(os.path.join(RUNS, run, "**/games/*/log.jsonl"), recursive=True))
        o_tok, o_off, o_game, o_use, o_eff = [], [0], [], [], []
        i_tok, i_off, i_game = [], [0], []
        names = []
        for gi, lg in enumerate(games):
            gdir = os.path.dirname(lg)
            names.append(os.path.basename(gdir))
            outs, usage, effort, inputs = game_records(gdir)
            for enc, u, e in zip(tok.encode_batch(outs, add_special_tokens=False), usage, effort):
                o_tok.append(np.asarray(enc.ids, dtype=np.int32))
                o_off.append(o_off[-1] + len(enc.ids))
                o_game.append(gi)
                o_use.append(u)
                o_eff.append(e)
            for enc in tok.encode_batch(inputs, add_special_tokens=False):
                i_tok.append(np.asarray(enc.ids, dtype=np.int32))
                i_off.append(i_off[-1] + len(enc.ids))
                i_game.append(gi)
        np.savez_compressed(
            dest,
            out_tokens=np.concatenate(o_tok) if o_tok else np.zeros(0, np.int32),
            out_offsets=np.asarray(o_off, np.int64),
            out_game=np.asarray(o_game, np.int32),
            out_usage=np.asarray(o_use, np.int64),
            out_effort=np.asarray(o_eff),
            in_tokens=np.concatenate(i_tok) if i_tok else np.zeros(0, np.int32),
            in_offsets=np.asarray(i_off, np.int64),
            in_game=np.asarray(i_game, np.int32),
            games=np.asarray(names),
        )
        print(f"{run}: {len(games)} games, {len(o_use)} calls, {o_off[-1]:,} out, {i_off[-1]:,} in", flush=True)


# ----------------------------------------------------------------------------- analysis


def load(cache: str, run: str) -> dict:
    return dict(np.load(os.path.join(cache, run.replace("/", "__") + ".npz")))


def coverage_k(counts: np.ndarray, fracs=(0.99, 0.999, 0.9999, 1.0)) -> dict:
    c = np.sort(counts[counts > 0])[::-1]
    cum = np.cumsum(c) / c.sum()
    return {str(f): int(np.searchsorted(cum, f - 1e-12) + 1) for f in fracs}


def seq_ngrams(tokens: np.ndarray, offsets: np.ndarray):
    """Bigram and trigram ids of each position (within its own sequence; -1 where undefined).

    Mirrors the PLE: the n-gram of position t is (x[t-1], x[t]) / (x[t-2], x[t-1], x[t]); positions
    before the sequence start see EOS, as the PLE pads with EOS.
    """
    t = tokens.astype(np.int64)
    n = len(t)
    idx = np.arange(n)
    seg = np.searchsorted(offsets, idx, side="right") - 1
    pos = idx - offsets[seg]
    p1 = np.where(pos >= 1, np.roll(t, 1), EOS)
    p2 = np.where(pos >= 2, np.roll(t, 2), EOS)
    return p1 * V + t, (p2 * V + p1) * V + t


def isin_sorted(x: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
    if len(sorted_set) == 0:
        return np.zeros(len(x), bool)
    i = np.searchsorted(sorted_set, x)
    i[i >= len(sorted_set)] = len(sorted_set) - 1
    return sorted_set[i] == x


def call_has(mask_bad: np.ndarray, offsets: np.ndarray) -> float:
    """Share of calls with at least one flagged position."""
    csum = np.concatenate([[0], np.cumsum(mask_bad)])
    per = csum[offsets[1:]] - csum[offsets[:-1]]
    return float((per > 0).mean()) if len(per) else 0.0


def _ctx_keys(toks: np.ndarray, offs: np.ndarray, ctx: int) -> tuple[np.ndarray, np.ndarray]:
    """Key of the `ctx` tokens before each position, and whether the call has that many.

    Exact for ctx <= 3 (V**3 < 2**63); a wrapping multiplicative hash for ctx = 4.
    """
    t = toks.astype(np.int64)
    n = len(t)
    key = np.zeros(n, np.int64)
    with np.errstate(over="ignore"):
        for k in range(ctx, 0, -1):
            key = key * (V if ctx <= 3 else np.int64(0x9E3779B97F4A7C15 - 2**64)) + np.roll(t, k)
    idx = np.arange(n)
    seg = np.searchsorted(offs, idx, side="right") - 1
    pos = idx - offs[seg]
    return key, pos >= ctx


def _chain(ok: np.ndarray, offs: np.ndarray, depth: int) -> np.ndarray:
    """Accepted tokens of a `depth`-token chained draft started at each position.

    Once the first j drafted tokens are accepted, the draft's context equals the true context, so
    the chain accepts exactly the leading run of per-position hits (inside the same call).
    """
    n = len(ok)
    idx = np.arange(n)
    seg = np.searchsorted(offs, idx, side="right") - 1
    end = offs[seg + 1]
    acc = np.zeros(n, np.int64)
    run = np.ones(n, bool)
    for d in range(depth):
        j = idx + d
        valid = j < end
        jj = np.minimum(j, n - 1)
        run = run & valid & ok[jj]
        acc += run
    return acc


def draft_eval(train: list[dict], test: list[dict], ctx: int = 3, depth: int = 3) -> dict:
    """Context-lookup (suffix-style) draft proposers on generated tokens, CPU only.

    - corpus: most frequent continuation of the last `ctx` tokens in the train runs' outputs;
    - local: the most recent continuation of the same context earlier in the same game's outputs
      (earlier replies and the current reply so far; strictly causal);
    - hybrid: local when the game has seen the context, else corpus.
    A deterministic draft token d is accepted by rejection sampling with probability p_target(d),
    whose unbiased estimate is the rate at which the recorded sampled token equals d.
    """
    # corpus table
    keys, nxt = [], []
    for d in train:
        k, m = _ctx_keys(d["out_tokens"], d["out_offsets"], ctx)
        keys.append(k[m])
        nxt.append(d["out_tokens"][m].astype(np.int64))
    keys = np.concatenate(keys)
    nxt = np.concatenate(nxt)
    order = np.lexsort((nxt, keys))
    keys, nxt = keys[order], nxt[order]
    change = np.ones(len(keys), bool)
    change[1:] = (keys[1:] != keys[:-1]) | (nxt[1:] != nxt[:-1])
    starts = np.nonzero(change)[0]
    counts = np.diff(np.append(starts, len(keys)))
    pk, pn = keys[starts], nxt[starts]
    # best continuation per key: sort by (key, -count), take first
    o2 = np.lexsort((-counts, pk))
    pk, pn, counts = pk[o2], pn[o2], counts[o2]
    first = np.ones(len(pk), bool)
    first[1:] = pk[1:] != pk[:-1]
    ck, cn = pk[first], pn[first]
    del keys, nxt, order, pk, pn, counts

    res = Counter()
    for d in test:
        toks = d["out_tokens"].astype(np.int64)
        offs = d["out_offsets"]
        games = d["out_game"]
        k, valid = _ctx_keys(d["out_tokens"], offs, ctx)
        n = len(toks)
        # corpus prediction
        i = np.searchsorted(ck, k)
        i[i >= len(ck)] = len(ck) - 1
        c_has = (ck[i] == k) & valid
        c_ok = c_has & (cn[i] == toks)
        # local prediction: previous occurrence of the same key within the same game
        gpos = games[np.searchsorted(offs, np.arange(n), side="right") - 1].astype(np.int64)
        gpos = np.where(valid, gpos, -1 - np.arange(n))  # invalid positions never match
        gkey = k
        o = np.lexsort((np.arange(n), gkey, gpos))
        same = np.zeros(n, bool)
        same[1:] = (gkey[o][1:] == gkey[o][:-1]) & (gpos[o][1:] == gpos[o][:-1])
        prev = np.full(n, -1, np.int64)
        prev[o[1:][same[1:]]] = o[:-1][same[1:]]
        l_has = prev >= 0
        l_ok = l_has & (toks[np.maximum(prev, 0)] == toks)
        h_has = l_has | c_has
        h_ok = np.where(l_has, l_ok, c_ok)
        res["positions"] += int(valid.sum())
        for name, has, ok in (("corpus", c_has, c_ok), ("local", l_has, l_ok), ("hybrid", h_has, h_ok)):
            acc = _chain(ok, offs, depth)
            m = has & valid
            res[name + ":drafts"] += int(m.sum())
            res[name + ":first_ok"] += int(ok[m].sum())
            res[name + ":accepted"] += int(acc[m].sum())
    out = {"ctx": ctx, "depth": depth, "positions": res["positions"], "corpus_keys": int(len(ck))}
    for name in ("corpus", "local", "hybrid"):
        dr = res[name + ":drafts"]
        out[name] = {
            "draft_share": dr / max(1, res["positions"]),
            "first_token_accept": res[name + ":first_ok"] / max(1, dr),
            "accepted_per_draft": res[name + ":accepted"] / max(1, dr),
            "accepted_per_position": res[name + ":accepted"] / max(1, res["positions"]),
        }
    return out


def analyze(cache: str, out_json: str) -> None:
    runs = [r for r in primary_runs() if len(load(cache, r)["out_usage"])]
    data = {r: load(cache, r) for r in runs}
    shift = {k: {r: load(cache, r) for r in v} for k, v in SHIFT_RUNS.items()}
    R: dict = {"runs": runs, "shift_runs": SHIFT_RUNS}

    # --- reconstruction check against the server's completion_tokens
    diffs = []
    for d in data.values():
        n = np.diff(d["out_offsets"])
        u = d["out_usage"]
        ok = u > 0
        diffs.append((n[ok] - u[ok]))
    diffs = np.concatenate(diffs)
    R["reconstruction"] = {
        "calls": int(len(diffs)),
        "mean_tokens_minus_usage": float(diffs.mean()),
        "median_abs": float(np.median(np.abs(diffs))),
        "share_within_2": float((np.abs(diffs) <= 2).mean()),
        "share_within_1pct": float((np.abs(diffs) <= 0.01 * np.maximum(1, np.abs(diffs) + 1)).mean()),
    }

    # --- output coverage (all primary)
    allout = np.concatenate([d["out_tokens"] for d in data.values()])
    allin = np.concatenate([d["in_tokens"] for d in data.values()])
    oc = np.bincount(allout, minlength=V)
    ic = np.bincount(allin, minlength=V)
    R["totals"] = {
        "runs": len(runs),
        "games": int(sum(len(d["games"]) for d in data.values())),
        "calls": int(sum(len(d["out_usage"]) for d in data.values())),
        "output_tokens": int(len(allout)),
        "input_tokens_dedup": int(len(allin)),
    }
    R["output"] = {
        "distinct": int((oc > 0).sum()),
        "coverage_k": coverage_k(oc),
        "top_share": {str(k): float(np.sort(oc)[::-1][:k].sum() / oc.sum()) for k in (256, 1024, 4096, 8192, 16384, 32768)},
    }
    R["input"] = {"distinct": int((ic > 0).sum()), "coverage_k": coverage_k(ic)}
    R["union_distinct"] = int(((oc + ic) > 0).sum())
    ic_only = (ic > 0) & (oc == 0)
    R["input_only_distinct"] = int(ic_only.sum())

    # --- held-out restricted heads
    def subset_eval(train_runs, test_sets: dict, ks=(4096, 8192, 16384, 32768, 65536)):
        tc = np.zeros(V, np.int64)
        tin = np.zeros(V, np.int64)
        for d in train_runs:
            tc += np.bincount(d["out_tokens"], minlength=V)
            tin += np.bincount(d["in_tokens"], minlength=V)
        order = np.argsort(-tc, kind="stable")
        seen = int((tc > 0).sum())
        res = {"train_distinct_out": seen, "train_out_tokens": int(tc.sum())}
        for tname, tdata in test_sets.items():
            toks = np.concatenate([d["out_tokens"] for d in tdata])
            offs_list, base = [], 0
            for d in tdata:
                offs_list.append(d["out_offsets"][:-1] + base)
                base += len(d["out_tokens"])
            offs = np.concatenate(offs_list + [[base]])
            row = {"test_out_tokens": int(len(toks)), "test_calls": int(len(offs) - 1)}
            for k in list(ks) + ["seen", "seen+input"]:
                if k == "seen":
                    member = tc > 0
                    kk = seen
                elif k == "seen+input":
                    member = (tc > 0) | (tin > 0)
                    kk = int(member.sum())
                else:
                    member = np.zeros(V, bool)
                    member[order[:k]] = True
                    kk = k
                bad = ~member[toks]
                row[str(k)] = {
                    "size": kk,
                    "oos_per_token": float(bad.mean()),
                    "oos_per_million": float(bad.mean() * 1e6),
                    "calls_with_oos": call_has(bad, offs),
                }
            # input-side coverage by train (outputs + inputs) for a restricted input embedding
            itoks = np.concatenate([d["in_tokens"] for d in tdata])
            member = (tc > 0) | (tin > 0)
            row["input_oos_per_token_vs_train_union"] = float((~member[itoks]).mean()) if len(itoks) else None
            res[tname] = row
        return res

    runs_a = runs[0::2]
    runs_b = runs[1::2]
    c110 = [r for r in runs if "c110" in r]
    r23 = [r for r in runs if "r23" in r]
    R["heldout"] = {
        "split_alternate_runs": subset_eval([data[r] for r in runs_a], {"other_half": [data[r] for r in runs_b]}),
        "r23_to_c110": subset_eval([data[r] for r in r23], {"c110": [data[r] for r in c110]}),
        "c110_to_r23": subset_eval([data[r] for r in c110], {"r23": [data[r] for r in r23]}),
        "all_primary_to_shift": subset_eval(list(data.values()), {k: list(v.values()) for k, v in shift.items()}),
    }

    # --- by-game split inside r23 (same game ids across runs): half the games -> other half
    r23_games = sorted({str(g).split("-")[0] for r in r23 for g in data[r]["games"]})
    ga = set(r23_games[0::2])

    def game_filter(d, keep):
        gn = np.array([str(g).split("-")[0] in keep for g in d["games"]])
        cm = gn[d["out_game"]]
        toks, offs = [], [0]
        for j in np.nonzero(cm)[0]:
            s = d["out_tokens"][d["out_offsets"][j]:d["out_offsets"][j + 1]]
            toks.append(s)
            offs.append(offs[-1] + len(s))
        im = gn[d["in_game"]] if len(d["in_game"]) else np.zeros(0, bool)
        itoks = [d["in_tokens"][d["in_offsets"][j]:d["in_offsets"][j + 1]] for j in np.nonzero(im)[0]]
        return {
            "out_tokens": np.concatenate(toks) if toks else np.zeros(0, np.int32),
            "out_offsets": np.asarray(offs, np.int64),
            "in_tokens": np.concatenate(itoks) if itoks else np.zeros(0, np.int32),
        }

    ga_data = [game_filter(data[r], ga) for r in r23]
    gb_data = [game_filter(data[r], set(r23_games) - ga) for r in r23]
    R["heldout"]["r23_games_half_to_other_half"] = subset_eval(ga_data, {"other_games": gb_data})
    R["heldout"]["r23_games_half_to_other_half"]["train_games"] = sorted(ga)

    # --- PLE n-gram subset
    def ngram_sets(ds, use_in=True):
        bi, tri = [], []
        for d in ds:
            b, t = seq_ngrams(d["out_tokens"], d["out_offsets"])
            bi.append(np.unique(b))
            tri.append(np.unique(t))
            if use_in and len(d["in_tokens"]):
                b, t = seq_ngrams(d["in_tokens"], d["in_offsets"])
                bi.append(np.unique(b))
                tri.append(np.unique(t))
        return np.unique(np.concatenate(bi)), np.unique(np.concatenate(tri))

    def ple_eval(train, tests: dict):
        bi, tri = ngram_sets(train)
        rows = 8 * (len(bi) + len(tri))
        res = {
            "distinct_bigrams": int(len(bi)),
            "distinct_trigrams": int(len(tri)),
            "rows": int(rows),
            "gib_fp8": rows * 160 / 2**30,
        }
        for tname, tds in tests.items():
            ob, ot, ib, it = [], [], [], []
            call_bad = []
            for d in tds:
                b, t = seq_ngrams(d["out_tokens"], d["out_offsets"])
                hit = isin_sorted(b, bi) & isin_sorted(t, tri)
                ob.append(isin_sorted(b, bi))
                ot.append(isin_sorted(t, tri))
                call_bad.append(call_has(~hit, d["out_offsets"]))
                if len(d["in_tokens"]):
                    b2, t2 = seq_ngrams(d["in_tokens"], d["in_offsets"])
                    ib.append(isin_sorted(b2, bi))
                    it.append(isin_sorted(t2, tri))
            ob, ot = np.concatenate(ob), np.concatenate(ot)
            both = ob & ot
            res[tname] = {
                "decode_bigram_hit": float(ob.mean()),
                "decode_trigram_hit": float(ot.mean()),
                "decode_token_all16_hit": float(both.mean()),
                "prefill_bigram_hit": float(np.concatenate(ib).mean()) if ib else None,
                "prefill_trigram_hit": float(np.concatenate(it).mean()) if it else None,
                "p_iteration_all_hit": {
                    f"{n}_tokens": float(both.mean() ** n) for n in (1, 4, 6, 24)
                },
                "calls_with_a_miss": float(np.mean(call_bad)),
            }
        return res

    R["ple"] = {
        "all_primary": ple_eval(list(data.values()), {}),
        "split_alternate_runs": ple_eval([data[r] for r in runs_a], {"other_half": [data[r] for r in runs_b]}),
        "r23_to_c110": ple_eval([data[r] for r in r23], {"c110": [data[r] for r in c110]}),
        "all_primary_to_shift": ple_eval(list(data.values()), {k: list(v.values()) for k, v in shift.items()}),
    }
    # How the subset grows with corpus size (runs added one at a time, alternating c110/r23)
    growth = []
    order_runs = [r for pair in zip(c110, r23) for r in pair] + r23[len(c110):] + c110[len(r23):]
    acc_b, acc_t = np.zeros(0, np.int64), np.zeros(0, np.int64)
    ntok = 0
    for i, r in enumerate(order_runs):
        b, t = ngram_sets([data[r]])
        acc_b = np.union1d(acc_b, b)
        acc_t = np.union1d(acc_t, t)
        ntok += len(data[r]["out_tokens"]) + len(data[r]["in_tokens"])
        if i in (0, 1, 3, 7, 15, 31) or i == len(order_runs) - 1:
            growth.append({"runs": i + 1, "tokens": ntok, "bigrams": int(len(acc_b)), "trigrams": int(len(acc_t)),
                           "gib": 8 * (len(acc_b) + len(acc_t)) * 160 / 2**30})
    R["ple"]["growth"] = growth

    # --- context-lookup draft proposer (subsample)
    R["draft"] = {
        "ctx3": draft_eval([data[r] for r in runs_a], [data[r] for r in runs_b], ctx=3),
        "ctx4": draft_eval([data[r] for r in runs_a], [data[r] for r in runs_b], ctx=4),
        "ctx2": draft_eval([data[r] for r in runs_a], [data[r] for r in runs_b], ctx=2),
    }

    # --- output composition: reasoning vs content vs tool-call share (tokens), per call
    R["per_call"] = {}
    lens = np.concatenate([np.diff(d["out_offsets"]) for d in data.values()])
    R["per_call"]["mean_tokens"] = float(lens.mean())
    R["per_call"]["median_tokens"] = float(np.median(lens))
    R["per_call"]["p90_tokens"] = float(np.percentile(lens, 90))

    with open(out_json, "w") as f:
        json.dump(R, f, indent=1)
    print(json.dumps(R, indent=1)[:20000])


def union_eval(cache: str, out_json: str) -> None:
    """Restricted target head built as (tokens seen in train) + (token ids < N): the low ids are
    the tokenizer's early merges (English words, code pieces), the high ids mostly multilingual."""
    runs = [r for r in primary_runs() if len(load(cache, r)["out_usage"])]
    data = {r: load(cache, r) for r in runs}
    r23 = [r for r in runs if "r23" in r]
    games = sorted({str(g).split("-")[0] for r in r23 for g in data[r]["games"]})
    ga = set(games[0::2])

    def filt(d, keep):
        gn = np.array([str(g).split("-")[0] in keep for g in d["games"]])
        cm = gn[d["out_game"]]
        toks, offs = [], [0]
        for j in np.nonzero(cm)[0]:
            s = d["out_tokens"][d["out_offsets"][j]:d["out_offsets"][j + 1]]
            toks.append(s); offs.append(offs[-1] + len(s))
        return np.concatenate(toks), np.asarray(offs)

    splits = {
        "alternate_runs": ([data[r]["out_tokens"] for r in runs[0::2]], [(data[r]["out_tokens"], data[r]["out_offsets"]) for r in runs[1::2]]),
        "r23_to_c110": ([data[r]["out_tokens"] for r in r23], [(data[r]["out_tokens"], data[r]["out_offsets"]) for r in runs if "c110" in r]),
        "r23_half_games": ([filt(data[r], ga)[0] for r in r23], [filt(data[r], set(games) - ga) for r in r23]),
    }
    out = {}
    for name, (train, test) in splits.items():
        tc = np.zeros(V, np.int64)
        for t in train:
            tc += np.bincount(t, minlength=V)
        toks = np.concatenate([t for t, _ in test])
        base = 0; offl = []
        for t, o in test:
            offl.append(o[:-1] + base); base += len(t)
        offs = np.concatenate(offl + [[base]])
        row = {}
        for N in (0, 16384, 32768, 65536, 100000, 151643):
            member = (tc > 0) | (np.arange(V) < N)
            bad = ~member[toks]
            row[f"seen+id<{N}"] = {"size": int(member.sum()), "oos_per_million": float(bad.mean() * 1e6), "calls_with_oos": call_has(bad, offs)}
        out[name] = row

    R = json.load(open(out_json))
    R["heldout_seen_plus_low_ids"] = out
    with open(out_json, "w") as f:
        json.dump(R, f, indent=1)
    print(json.dumps(out, indent=1))


DRAFT_VOCAB_SIZE = 16384
DRAFT_VOCAB_OUT = "src/arc_infra/serving_draft_vocab.json"


def byte_level_ids(vocab: dict) -> list[int]:
    """Ids of the 256 single-byte tokens of the byte-level BPE (GPT-2 bytes_to_unicode alphabet)."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1))
    chars, extra = [], 0
    for b in range(256):
        if b in printable:
            chars.append(chr(b))
        else:
            chars.append(chr(256 + extra))
            extra += 1
    ids = [vocab[c] for c in chars]
    if len(set(ids)) != 256:
        raise RuntimeError("the byte-level alphabet did not map to 256 distinct ids")
    return ids


def draft_vocab(cache: str, analysis_json: str, out_json: str, size: int = DRAFT_VOCAB_SIZE) -> None:
    """The MTP draft head's fixed token subset (serving tweaks, 28 Sept; phase-3 Option B).

    Every single-byte token and every special/added token, then the most frequent generated tokens of the
    recorded specialization corpus (the runs listed in ``analysis_json``) until ``size`` ids. Rejection sampling
    keeps the served output exact whatever the subset; only draft acceptance depends on it.
    """
    runs = json.load(open(analysis_json))["runs"]
    counts = np.zeros(V, np.int64)
    for run in runs:
        counts += np.bincount(load(cache, run)["out_tokens"], minlength=V)
    tok = json.load(open(TOKENIZER))
    special = sorted(a["id"] for a in tok["added_tokens"] if a["id"] < V)
    mandatory = sorted(set(byte_level_ids(tok["model"]["vocab"])) | set(special))
    chosen = set(mandatory)
    for token in np.lexsort((np.arange(V), -counts)):  # most frequent first, ties by id
        if len(chosen) >= size:
            break
        if counts[token] > 0:
            chosen.add(int(token))
    if len(chosen) != size:
        raise RuntimeError(f"only {len(chosen)} ids for a {size}-row draft head")
    ids = sorted(chosen)
    covered = float(counts[ids].sum() / counts.sum())
    record = {
        "name": f"arena{size // 1024}k",
        "size": size,
        "vocab_rows": V,
        "corpus": {"analysis": analysis_json, "runs": len(runs), "generated_tokens": int(counts.sum()),
                   "distinct_generated": int((counts > 0).sum())},
        "rule": "all 256 byte tokens and every added/special id, then generated tokens by corpus frequency "
                "(ties by id)",
        "mandatory": {"byte_tokens": 256, "special_tokens": len(special)},
        "corpus_coverage": covered,
        "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "ids": ids,
    }
    with open(out_json, "w") as f:
        json.dump(record, f, separators=(",", ":"))
        f.write("\n")
    print(json.dumps({k: v for k, v in record.items() if k != "ids"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "analyze", "union", "list", "draft-vocab"])
    ap.add_argument("--cache", default=os.environ.get("SPEC_CACHE", "/tmp/spec-cache"))
    ap.add_argument("--out", default="docs/validation-20260927/specialization.json")
    a = ap.parse_args()
    runs = primary_runs() + [r for v in SHIFT_RUNS.values() for r in v]
    if a.cmd == "list":
        print("\n".join(runs))
    elif a.cmd == "extract":
        extract(a.cache, runs)
    elif a.cmd == "union":
        union_eval(a.cache, a.out)
    elif a.cmd == "draft-vocab":
        draft_vocab(a.cache, "docs/validation-20260927/specialization.json", DRAFT_VOCAB_OUT)
    else:
        analyze(a.cache, a.out)


if __name__ == "__main__":
    sys.exit(main())
