"""Local replay of every scripted point (real games, real sandbox, no server): does the recording reproduce?
argv: src bundle out games points.json [concurrency] [arm study json]
With an arm study of {"handover": {"mode": "suggest"}} every attempt also writes its path-2 prompts; they are gathered
into out/prompts.json ({point id: {"system", "prompts"}})."""
import asyncio, json, sys, time
from pathlib import Path
import httpx
src, bundle, out, games = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
sys.path.insert(0, str(src))
from arena import level_bench
import arc_infra.metrics
class Quiet:
    def __init__(self, *a): self.start = None
    async def sample(self): return None
    async def run(self): await asyncio.Event().wait()
    def summary(self): return {"valid": True, "preemptions": 0}
arc_infra.metrics.GpuMonitor = Quiet
out.mkdir(parents=True, exist_ok=True)
study = json.loads((bundle / "configs/comparison.json").read_text())
study["level_bench"] = {**study["level_bench"], "capacity_tokens": 400 * 196608, "token_budget": 1, "games": None,
                        "passes": 1, "min_wall_seconds": 5, "launch_cutoff_seconds": 0,
                        "concurrency": int(sys.argv[6]) if len(sys.argv) > 6 else 6, "first_pass": 0,
                        "attempt_wall_seconds": 3000, "start_points": str(Path(sys.argv[5]).resolve()),
                        "arms": [{"name": "replay", "study": json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}}]}
study["wall_seconds"] = 3000
live = []
def handler(request):
    if request.url.path.endswith("/tokenize"):
        return httpx.Response(200, json={"count": len(request.content) // 4, "max_model_len": 262144})
    live.append(1)
    msg = {"role": "assistant", "content": "World model: replay check."}
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": msg}],
                                     "usage": {"completion_tokens": 5, "prompt_tokens": 100}})
lines = []
bench = level_bench.LevelBench(study, {"name": "level-bench", "kind": "level-bench"}, source=bundle, games_dir=games,
                               output=out, server_dir=out, base_url="http://127.0.0.1:9/v1",
                               deadline_epoch=time.time() + 3300, log=lambda l, **k: lines.append(l),
                               transport=httpx.MockTransport(handler))
started = time.time()
rep = asyncio.run(bench.run())
ok = bad = 0
for r in rep["attempts"]:
    s = r.get("script") or {}
    good = bool(s.get("ok"))
    ok += good; bad += not good
    print(r["point"], r["outcome"], "ok" if good else "MISMATCH", s.get("detail"), "ctx", s.get("estimate"), s.get("recorded_estimate"), "msgs", s.get("messages"), s.get("recorded_messages"), "requests", s.get("requests"), "steps", s.get("steps"),
          "levels", s.get("levels"), "wall", r["wall_seconds"], (r.get("error") or "")[:160])
print("replayed", ok, "mismatch", bad, "live requests", len(live), "seconds", round(time.time() - started), rep["status"], rep["task_errors"][:2])
prompts = {}
for r in rep["attempts"]:
    f = out / "attempts" / r["label"] / "suggest_prompts.json"
    if f.is_file():
        prompts[r["point"]] = json.loads(f.read_text())
if prompts:
    (out / "prompts.json").write_text(json.dumps(prompts))
    print("prompts for", len(prompts), "points;", sum(len(v["prompts"]) for v in prompts.values()), "prompts")
(out / "replay.json").write_text(json.dumps({r["point"]: (r.get("script") or {}) for r in rep["attempts"]}, indent=1))
