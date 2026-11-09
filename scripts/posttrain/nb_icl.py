"""In-context learning-progress check (envs arm): does the served base model's next-board NLL fall with the attempt's
history, and does that fall mark where exploration pays off? A serving-bench gpu-script beside the idle server
(``ARC_BENCH_BASE_URL``, ``ARC_BENCH_MODEL``).

Per attempt (``--rows``: boards, actions, labels) and transition t, scored only at the assistant target (the patch +
<|im_end|>; user/tool positions are degenerate on every stack):
- ``full``: the packed conversation (``arena.nextboard_pack.render``, board every 4 steps) of transitions 0..t-1, then
  turn t with the current board shown;
- ``zero``: turn t alone (intro + current board + action);
- ``rep``: ``zero`` again for every ``--rep-every``-th transition (the server's own repeat noise).
Cold requests (prompt_logprobs reads no prefix cache). Rows append to ``--out``.jsonl as they finish; a restart
skips finished keys."""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nextboard_aux as nb  # noqa: E402
import nextboard_pack as nbp  # noqa: E402


def sequences(att, board_every=4, max_steps=80, rep_every=8):
    """``[(key, kind, t, row)]``: a one-row conversation per scored target (the last turn's target is scored)."""
    a = {"boards": att["boards"], "actions": att["actions"], "level": att.get("level", 0)}
    full = nbp.render(a, board_every=board_every, max_steps=max_steps)[0]["turns"]
    out = []
    for t, turn in enumerate(full):
        if att["labels"][t].get("skip") or not turn["train"]:
            continue
        board = nbp.ascii_board(nbp.grid(att["boards"][t]))
        act = turn["user"].rsplit("\n", 1)[-1]
        last = dict(turn)
        if t and not turn["user"].startswith("Board now:"):
            last["user"] = f"Board now:\n```text\n{board}\n```\n{act}"
        zero = dict(turn, user=f"{nbp.INTRO}\n\nBoard:\n```text\n{board}\n```\n{act}")
        if t:
            out.append((f"{att['attempt']}:{t}:full", "full", t, {"turns": full[:t] + [last]}))
        else:
            out.append((f"{att['attempt']}:{t}:full", "full", t, {"turns": [zero]}))
        out.append((f"{att['attempt']}:{t}:zero", "zero", t, {"turns": [zero]}))
        if t % rep_every == 0:
            out.append((f"{att['attempt']}:{t}:rep", "rep", t, {"turns": [zero]}))
    return out


class Tok:
    """``tokenizer(text, add_special_tokens=False)["input_ids"]`` through the server's /tokenize, cached by text."""

    def __init__(self, client, root, model):
        self.client, self.root, self.model, self.cache = client, root, model, {}

    async def prepare(self, texts):
        todo = [t for t in set(texts) if t not in self.cache]
        sem = asyncio.Semaphore(32)

        async def one(t):
            async with sem:
                r = await self.client.post(self.root + "/tokenize", json={"model": self.model, "prompt": t,
                                                                         "add_special_tokens": False})
                r.raise_for_status()
                self.cache[t] = r.json()["tokens"]

        await asyncio.gather(*(one(t) for t in todo))

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.cache[text]}


def pieces(row):
    out = []
    for turn in row["turns"]:
        out += [nb.USER.format(text=turn["user"]) + nb.OPEN, turn["target"], "<|im_end|>", "\n"]
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--rep-every", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=0)
    a = ap.parse_args()
    import httpx

    url, model = os.environ["ARC_BENCH_BASE_URL"].rstrip("/"), os.environ["ARC_BENCH_MODEL"]
    root = url.removesuffix("/v1")
    out = Path(a.out).with_suffix(".jsonl")
    done = set()
    if out.exists():
        done = {json.loads(l)["key"] for l in out.read_text().splitlines() if l.strip()}
    atts = [json.loads(l) for l in open(a.rows) if l.strip()]
    if a.attempts:
        atts = atts[: a.attempts]
    jobs = [j for att in atts for j in sequences(att, max_steps=a.max_steps, rep_every=a.rep_every) if j[0] not in done]
    jobs.sort(key=lambda j: (j[2], j[0]))  # short contexts first: early rows cover every attempt
    began, stats = time.time(), {"jobs": len(jobs), "done": 0, "tokens": 0, "errors": 0}
    async with httpx.AsyncClient(timeout=1800) as client:
        tok = Tok(client, root, model)
        await tok.prepare([p for j in jobs for p in pieces(j[3])])
        sem = asyncio.Semaphore(a.concurrency)
        lock = asyncio.Lock()
        with open(out, "a") as sink:
            async def one(key, kind, t, row):
                enc = nb.encode(tok, row)
                s, e, _ = enc["spans"][-1]
                body = {"model": model, "prompt": enc["ids"], "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0}
                async with sem:
                    for attempt in range(3):
                        r = await client.post(url + "/completions", json=body)
                        if r.status_code == 200:
                            break
                        await asyncio.sleep(5)
                    else:
                        stats["errors"] += 1
                        return
                entries = r.json()["choices"][0]["prompt_logprobs"]
                lps, top = [], []
                for pos in range(s, e):
                    entry = entries[pos] or {}
                    hit = entry.get(str(enc["ids"][pos])) or {}
                    lps.append(hit.get("logprob"))
                    top.append(hit.get("rank") == 1)
                rec = {"key": key, "kind": kind, "attempt": key.split(":")[0], "t": t, "tokens": len(enc["ids"]),
                       "n": e - s, "lp": lps, "top1": top}
                async with lock:
                    sink.write(json.dumps(rec) + "\n")
                    sink.flush()
                    stats["done"] += 1
                    stats["tokens"] += len(enc["ids"])
                    if stats["done"] % 200 == 0:
                        print(json.dumps({**stats, "seconds": round(time.time() - began)}), flush=True)

            await asyncio.gather(*(one(*j) for j in jobs))
    stats["seconds"] = round(time.time() - began)
    Path(a.out).write_text(json.dumps(stats))
    print(json.dumps(stats), flush=True)
    return 1 if stats["errors"] > 0.01 * max(1, stats["jobs"]) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
