"""KV replay bench (RL 24 Oct): replays recorded hero tries' request pattern against one play server, to read
prefix-cache misses and the host KV offload (ArcOffloadingConnector) without a B300 or a trainer.

Each recorded try is a list of requests [prompt_tokens, completion_tokens, cached_tokens]. Request i sends the
first prompt_tokens of the try's own token stream and generates exactly completion_tokens (ignore_eos), so a
try's next prompt extends its last one (the generated tail is recomputed, as a reply re-sent in a chat would be
when its tokens differ). Tries of one recorded group share the first prompt (a restored start). New tries start
while the engine's KV use is below ``kv_admit`` (one per ``stagger_s``), as the hero's gate admits; running tries
grow past it. A sampler reads /metrics every ``sample_s``; the report's steady part is the second half.

Config (rl_loop.serve_check.kv_replay): see DEFAULT. Report: rl/kv_replay.json."""

import asyncio
import json
import random
import re
import time
from pathlib import Path

DEFAULT = {"trace": "configs/kv_trace.json", "kv_admit": 0.85, "max_live": 128, "seconds": 900, "gap_s": 1.0,
           "stagger_s": 2.0, "adapter": None, "sample_s": 10.0, "seed": 0, "temperature": 0.7}
KEEP = re.compile(r"^vllm:(kv_cache_usage_perc|gpu_cache_usage_perc|num_requests_running|num_requests_waiting|"
                  r"prompt_tokens_total|generation_tokens_total|num_preemptions_total|"
                  r"(external_)?prefix_cache_(queries|hits)_total|.*offload.*|.*kv_transfer.*)$")


def parse_metrics(text):
    """Prometheus text -> {name: summed value} for the kept vLLM families (labels summed)."""
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name, _, rest = line.partition("{") if "{" in line.split(" ")[0] else (line.split(" ")[0], "", line)
        if not KEEP.match(name):
            continue
        try:
            value = float(rest.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + value
    return out


async def metrics(server):
    import httpx

    try:
        async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
            return parse_metrics((await client.get(server.root + "/metrics")).text)
    except Exception:  # noqa: BLE001 - one missed sample does not stop the replay
        return {}


def kv_of(m):
    return m.get("vllm:kv_cache_usage_perc", m.get("vllm:gpu_cache_usage_perc"))


def summary(samples, start_frac=0.5):
    """Rates over the steady part (from ``start_frac`` of the run): gen / prompt / computed-prompt tok/s, GPU and
    external hit rates, mean KV, live tries, preemptions."""
    if len(samples) < 3:
        return {}
    t0, t1 = samples[0]["t"], samples[-1]["t"]
    part = [s for s in samples if s["t"] >= t0 + start_frac * (t1 - t0)]
    a, b = part[0], part[-1]
    dt = max(1e-9, b["t"] - a["t"])
    d = lambda k: b["m"].get(k, 0.0) - a["m"].get(k, 0.0)
    prompt, hits = d("vllm:prompt_tokens_total"), d("vllm:prefix_cache_hits_total")
    ext_q, ext_h = d("vllm:external_prefix_cache_queries_total"), d("vllm:external_prefix_cache_hits_total")
    kvs = [kv_of(s["m"]) for s in part if kv_of(s["m"]) is not None]
    return {"seconds": round(dt, 1), "gen_tok_s": round(d("vllm:generation_tokens_total") / dt, 1),
            "prompt_tok_s": round(prompt / dt, 1), "computed_prompt_tok_s": round(max(0.0, prompt - hits - ext_h) / dt, 1),
            "gpu_hit": round(hits / max(1.0, d("vllm:prefix_cache_queries_total")), 4),
            "external_hit": round(ext_h / ext_q, 4) if ext_q else None, "external_hit_tokens": ext_h,
            "kv_mean": round(sum(kvs) / len(kvs), 3) if kvs else None, "kv_max": max(kvs) if kvs else None,
            "live_mean": round(sum(s["live"] for s in part) / len(part), 1),
            "running_mean": round(sum(s["m"].get("vllm:num_requests_running", 0) for s in part) / len(part), 1),
            "preemptions": d("vllm:num_preemptions_total")}


class Replay:
    def __init__(self, server, model, trace, cfg, feed=None):
        self.server, self.model, self.cfg, self.feed = server, model, cfg, feed
        self.tries = [t for t in trace["tries"] if t["reqs"]]
        self.rng = random.Random(cfg["seed"])
        self.group_prefix, self.live, self.done, self.errors, self.kv = {}, 0, 0, 0, 0.0
        self.samples, self.replies = [], []  # replies: (end time, prompt, cached, completion) from each usage

    def stream(self, t, n):
        """The try's token ids: its group's shared first prompt, then its own tokens."""
        first = int(t["reqs"][0][0])
        shared = self.group_prefix.setdefault(t["group"], [self.rng.randrange(1000, 100000) for _ in range(first)])
        own = random.Random(hash((t["group"], t["try"], self.cfg["seed"])) & 0xFFFFFFFF)
        return shared[:first] + [own.randrange(1000, 100000) for _ in range(max(0, n - first))]

    async def play(self, t):
        self.live += 1
        try:
            ids = self.stream(t, max(int(p) for p, _, _ in t["reqs"]))
            for prompt, completion, _ in t["reqs"]:
                n = max(1, int(completion))
                reply = await self.server.post("/v1/completions", {
                    "model": self.model, "prompt": ids[:int(prompt)], "max_tokens": n, "min_tokens": n,
                    "ignore_eos": True, "temperature": self.cfg["temperature"]}, timeout=3600)
                u = (reply or {}).get("usage") or {}
                self.replies.append((time.time(), int(u.get("prompt_tokens") or 0),
                                     int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                                     int(u.get("completion_tokens") or 0)))
                await asyncio.sleep(float(self.cfg["gap_s"]))
            self.done += 1
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a failed request ends that try; counted
            self.errors += 1
        finally:
            self.live -= 1

    async def sampler(self, end):
        while time.time() < end:
            m = await metrics(self.server)
            if kv_of(m) is not None:
                self.kv = float(kv_of(m))
            self.samples.append({"t": time.time(), "live": self.live, "m": m})
            if self.feed is not None and len(self.samples) % 6 == 0:
                self.feed.emit("kv_replay_sample", live=self.live, kv=self.kv, done=self.done, errors=self.errors)
            await asyncio.sleep(float(self.cfg["sample_s"]))

    async def run(self):
        cfg, end = self.cfg, time.time() + float(self.cfg["seconds"])
        sampler = asyncio.create_task(self.sampler(end))
        tasks, k, last = set(), 0, 0.0
        while time.time() < end:
            now = time.time()
            if (self.live < int(cfg["max_live"]) and self.kv < float(cfg["kv_admit"])
                    and now - last >= float(cfg["stagger_s"])):
                tasks.add(asyncio.create_task(self.play(self.tries[k % len(self.tries)])))
                k, last = k + 1, now
            tasks = {x for x in tasks if not x.done()}
            await asyncio.sleep(0.2)
        await sampler
        for x in tasks:
            x.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        out = summary(self.samples)
        if self.samples and self.replies:  # the engine's own per-request cache accounting (GPU + host hits)
            t0, t1 = self.samples[0]["t"], self.samples[-1]["t"]
            late = [r for r in self.replies if r[0] >= t0 + 0.5 * (t1 - t0)]
            span = max(1e-9, t1 - (t0 + 0.5 * (t1 - t0)))
            p, c = sum(r[1] for r in late), sum(r[2] for r in late)
            out.update({"usage_prompt_tok_s": round(p / span, 1), "usage_uncached_tok_s": round((p - c) / span, 1),
                        "usage_cached_frac": round(c / p, 4) if p else None, "usage_replies": len(late)})
        return {"started": k, "finished": self.done, "errors": self.errors, "summary": out,
                "samples": [{"t": round(s["t"], 1), "live": s["live"], **s["m"]} for s in self.samples]}


async def run(loop, cfg=None):
    cfg = {**DEFAULT, **(cfg or {})}
    env = ((getattr(loop, "trial", None) or {}).get("server_env") or {})
    if env.get("ARC_KV_REPLAY_ADMIT"):  # per trial (one bundle reads several admission targets)
        cfg["kv_admit"] = float(env["ARC_KV_REPLAY_ADMIT"])
    server = loop.server
    model = server.model
    if cfg["adapter"]:
        await server.load("kv-replay", cfg["adapter"])
        model = "kv-replay"
    trace = json.loads((Path(loop.source) / cfg["trace"]).read_text())
    out = await Replay(server, model, trace, cfg, feed=loop.feed).run()
    out["settings"] = cfg
    trial = getattr(loop, "trial", None) or {}
    out["trial"] = {"name": trial.get("name"), "server_env": trial.get("server_env")}
    (Path(loop.output) / "rl").mkdir(parents=True, exist_ok=True)
    (Path(loop.output) / "rl" / "kv_replay.json").write_text(json.dumps(out, indent=1))
    loop.feed.emit("kv_replay", started=out["started"], finished=out["finished"], errors=out["errors"], **out["summary"])
    return {k: out[k] for k in ("started", "finished", "errors", "summary")}
