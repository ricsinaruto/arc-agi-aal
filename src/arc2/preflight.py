"""Launch-gate preflight: the exact RL bundle on a CPU container with a scripted server
and trainer in place of the GPU parts (ARC2_PREFLIGHT=1). The bundle's bootstrap starts ``serve`` instead of the
Flash server and ``stop`` instead of its teardown; arc2.rl_loop starts ``trainer`` instead of rl1_trainer.py.

server: OpenAI-style chat completions (half the replies are a lookup program that is right on the task's train and
test pairs, so groups have spread), /tokenize, /metrics, sleep / wake, LoRA load / unload.
trainer: rl1_trainer's file protocol (alive.json, cmd/NNNN.json -> done/NNNN.json); checkpoint, save and
load_checkpoint write and check real folders, so resume and reload paths are exercised. Tensor mode
(``ARC2_PREFLIGHT_TENSORS``: a python with torch; the snr job): checkpoints, nudge and updates are real tensors in
rl1_trainer's checkpoint format (arc2.snr_math fake: one AdamW step per update), scores read the displacement."""
import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 1234


def answers(bundle):
    """{text of a task's first train input: lookup program source} over every task file in the bundle."""
    from .tasks import text

    out = {}
    for path in sorted(Path(bundle, "data/rl").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            pairs = list(row.get("train") or []) + list(row.get("holdout") or row.get("test") or [])
            if not pairs or "output" not in pairs[-1]:
                continue
            table = {json.dumps(p["input"]): p["output"] for p in pairs if "output" in p}
            program = ("def transform(grid):\n    import json\n    table = " + repr(table)
                       + "\n    return table[json.dumps(grid)]\n")
            out[text(row["train"][0]["input"])] = program
    tasks_dir = os.environ.get("ARC2_PREFLIGHT_TASKS")  # roster bundles: lookup programs for every augmented view
    if tasks_dir:
        from .loop import SYMMETRIES, augment_spec
        from .tasks import pictures_prompt

        for path in sorted(Path(tasks_dir).glob("*.json")):
            task = json.loads(path.read_text())
            for k in range(int(os.environ.get("ARC2_PREFLIGHT_VIEWS") or 4)):
                sym, colours = augment_spec(path.stem, k)
                view = lambda g: [[colours[c] for c in row] for row in SYMMETRIES[sym][0](g)]  # noqa: E731
                pairs = [{"input": view(p["input"]), "output": view(p["output"])} for p in task["train"] + task["test"]]
                table = {json.dumps(p["input"]): p["output"] for p in pairs}
                out[text(pairs[0]["input"])] = ("def transform(grid):\n    import json\n    table = " + repr(table)
                                                + "\n    return table[json.dumps(grid)]\n")
                shown = {"train": pairs[:len(task["train"])], "test": pairs[len(task["train"]):]}
                out.setdefault(pictures_prompt(shown), out[text(pairs[0]["input"])])  # ``pictures``: sizes only
    return out


# ARC2_PREFLIGHT_TOOLS (C1 preflight): the scripted model's `python` calls, by call number. They touch every
# tool_fixes path: a preloaded-name import, a failing snippet whose earlier definitions a later call uses, an empty
# call, output longer than 6,000 characters; the rest are plain.
TOOL_SNIPPETS = ["import np\nprint(np.array(train[0]['input']).shape)",
                 "def helper(g):\n    return len(g)\nn = helper(train[0]['input'])\nraise ValueError('preflight')",
                 "print(helper(train[0]['input']), n)", None, "print('x' * 7000 + 'END')", "print(len(train))"]


def serve_forever(bundle, port):
    table = answers(bundle)
    rng = random.Random(0)
    state = {"asleep": False, "loras": set(), "requests": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body, code=200, kind="application/json"):
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/metrics"):
                return self.reply("vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
                                  "vllm:kv_cache_usage_perc 0.1\n", kind="text/plain")
            if self.path.startswith("/is_sleeping"):
                return self.reply({"is_sleeping": state["asleep"]})
            if self.path.startswith("/v1/models"):
                return self.reply({"data": [{"id": "flash"}] + [{"id": n} for n in sorted(state["loras"])]})
            return self.reply({"ok": True})

        def do_POST(self):
            size = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(size) or b"{}")
            path = self.path.split("?")[0]
            if path == "/sleep":
                state["asleep"] = True
                return self.reply({})
            if path == "/wake_up":
                state["asleep"] = False
                return self.reply({})
            if path == "/v1/load_lora_adapter":
                if not Path(body["lora_path"]).exists():
                    return self.reply({"error": f"no adapter at {body['lora_path']}"}, 400)
                state["loras"].add(body["lora_name"])
                return self.reply({})
            if path == "/v1/unload_lora_adapter":
                state["loras"].discard(body["lora_name"])
                return self.reply({})
            if path == "/reset_prefix_cache":
                return self.reply({"success": True})
            effort = (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
            if path in ("/tokenize", "/v1/chat/completions") and effort not in (None, "xhigh", "medium", "low"):
                # Flash's chat template raises on any other effort (the server answers 400)
                return self.reply({"error": f"Unexpected reasoning effort {effort}. Supported types are xhigh (default), "
                                            "medium, and low."}, 400)
            if path == "/tokenize":
                count = len(json.dumps(body.get("messages") or body.get("prompt") or "")) // 4
                return self.reply({"count": count, "tokens": list(range(count)), "max_model_len": 131072})
            if path == "/v1/chat/completions":
                if state["asleep"]:
                    return self.reply({"error": "server is asleep"}, 503)
                state["requests"] += 1
                prompt = json.dumps(body.get("messages"))
                if body.get("prompt_logprobs") is not None:  # arc2.score: prefill-only likelihood of the prompt
                    n = len(prompt) // 4
                    return self.reply({"choices": [{"message": {"role": "assistant", "content": "x"},
                                                    "finish_reason": "length"}],
                                       "prompt_token_ids": list(range(n)),
                                       "prompt_logprobs": [None] + [{str(i): {"logprob": -0.01 * (i % 7), "rank": 1,
                                                                             "decoded_token": "`"}} for i in range(1, n)],
                                       "usage": {"prompt_tokens": n, "completion_tokens": 1}})
                calls = sum(m.get("role") == "tool" for m in body.get("messages") or [])
                first = json.dumps((body.get("messages") or [{}, {}])[1])
                long_attempt = sum(map(ord, first)) % 3 == 0  # a third of attempts call past the 12-call limit
                if os.environ.get("ARC2_PREFLIGHT_TOOLS") and body.get("tools") and calls < (14 if long_attempt else 4):
                    code = TOOL_SNIPPETS[min(calls, len(TOOL_SNIPPETS) - 1)]
                    arguments = "{}" if code is None else json.dumps({"code": code})
                    time.sleep(float(os.environ.get("ARC2_PREFLIGHT_DELAY") or 0.05))
                    return self.reply({"choices": [{"message": {"role": "assistant", "content": None,
                                                                "reasoning_content": "preflight",
                                                                "tool_calls": [{"id": f"call_{calls}", "type": "function",
                                                                                "function": {"name": "python",
                                                                                             "arguments": arguments}}]},
                                                    "finish_reason": "tool_calls"}],
                                       "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 40,
                                                 "prompt_tokens_details": {"cached_tokens": 0}}})
                program = next((p for key, p in table.items() if json.dumps(key)[1:-1] in prompt), None)
                right = program is not None and rng.random() < 0.5
                code = program if right else "def transform(grid):\n    return grid\n"
                content = "```python\n" + code + "```"
                ids = list(range(11, 11 + max(3, len(content) // 4)))
                entries = [{"token": f"token_id:{t}", "logprob": -0.1,
                            "top_logprobs": [{"token": f"token_id:{t}", "logprob": -0.1}]} for t in ids]
                prompt_tokens = len(prompt) // 4
                time.sleep(float(os.environ.get("ARC2_PREFLIGHT_DELAY") or 0.05))
                return self.reply({"choices": [{"message": {"role": "assistant", "content": content,
                                                            "reasoning_content": "preflight"},
                                                "finish_reason": "stop", "logprobs": {"content": entries},
                                                "token_ids": ids}],
                                   "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(ids),
                                             "prompt_tokens_details": {"cached_tokens": 0}},
                                   "prompt_token_ids": list(range(prompt_tokens))})
            return self.reply({"error": f"unknown path {path}"}, 404)

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def serve(working, bundle, port):
    """Start the scripted server in the background and return once it answers (as the real serve command does)."""
    import urllib.request

    working = Path(working)
    working.mkdir(parents=True, exist_ok=True)
    log = open(working / "vllm-openai-server.log", "a")
    process = subprocess.Popen([sys.executable, "-m", "arc2.preflight", "server", "--bundle", str(bundle),
                                "--port", str(port)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (working / "preflight-server.pid").write_text(str(process.pid))
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            print(json.dumps({"event": "preflight_server_ready", "pid": process.pid}), flush=True)
            return
        except OSError:
            if process.poll() is not None:
                raise SystemExit(f"preflight server exited ({process.returncode})")
            time.sleep(1)
    raise SystemExit("preflight server did not answer")


def stop(working):
    pid_file = Path(working) / "preflight-server.pid"
    if pid_file.exists():
        try:
            os.killpg(int(pid_file.read_text()), signal.SIGTERM)
        except ProcessLookupError:
            pass
        pid_file.unlink()


def trainer(folder):
    folder = Path(folder)
    (folder / "cmd").mkdir(parents=True, exist_ok=True)
    (folder / "done").mkdir(parents=True, exist_ok=True)
    (folder / "alive.json").write_text(json.dumps({"state": "ready", "preflight": True}))
    index, accepted = 0, 0
    # rl1_trainer's probe references (score_windows, op_accept_reference, forget_unaccepted): a probe once scored is
    # tracked; a load forgets every unaccepted score; accept_reference fails while any probe lacks a fresh one (P5b)
    references = {}
    memory = {"allocated_gib": 1.0, "reserved_gib": 1.0, "card_used_gib": 1.0}
    python, script = os.environ.get("ARC2_PREFLIGHT_TENSORS"), Path(__file__).with_name("snr_math.py")
    state = {"current": None, "base": None, "made": 0}

    def tensors(*args):
        done = subprocess.run([python, str(script), "fake", *map(str, args)], capture_output=True, text=True, timeout=600)
        if done.returncode:
            raise RuntimeError("snr_math fake: " + " | ".join(done.stderr.strip().splitlines()[-3:]))
        return json.loads(done.stdout.strip().splitlines()[-1])

    def new_state():
        state["made"] += 1
        return folder / "states" / f"{state['made']:04d}"

    def current():
        if state["current"] is None:
            state["current"] = new_state()
            tensors("init", "--out", state["current"])
        return state["current"]
    while True:
        cmd = folder / "cmd" / f"{index:04d}.json"
        if not cmd.exists():
            time.sleep(0.2)
            continue
        req = json.loads(cmd.read_text())
        op, out = req.pop("op"), {"ok": True}
        try:
            if op in ("checkpoint", "save"):
                path = Path(req["path"])
                path.mkdir(parents=True, exist_ok=True)
                if python and op == "checkpoint" and current() != path:
                    for name in ("master.safetensors", "optimizer.pt"):
                        shutil.copyfile(current() / name, path / name)
                (path / "preflight.json").write_text(json.dumps({"op": op, "accepted": accepted, "t": time.time()}))
                out.update(path=str(path), tensors=744, gib=0.06, seconds=0.1, memory=memory)
            elif op == "load_checkpoint" and python and (Path(req["path"]) / "master.safetensors").exists():
                if os.environ.get("ARC2_PREFLIGHT_BREAK") != "restore":  # test hook: a reload that keeps the old state
                    state["current"] = Path(req["path"])
                for refs in references.values():
                    refs.discard("last")
                out.update(seconds=0.1, **tensors("info", "--state", state["current"]))
            elif op == "load_checkpoint":
                path = Path(req["path"])
                if not (path / "preflight.json").exists():
                    raise FileNotFoundError(f"no checkpoint at {path}")
                accepted = json.loads((path / "preflight.json").read_text())["accepted"]
                for refs in references.values():
                    refs.discard("last")
                out.update(seconds=0.1)
            elif op == "nudge":
                if python:
                    made = new_state()
                    out.update(tensors("nudge", "--state", current(), "--out", made, "--value", req.get("value", 1e-12)))
                    state["current"] = made
                else:
                    out.update(nudged=0, value=req.get("value", 1e-12))
            elif op == "update":
                time.sleep(float(os.environ.get("ARC2_PREFLIGHT_TRAINER_DELAY") or 0))
                attempts = req.get("attempts") or []
                missing = [a["rollout"] for a in attempts if not Path(a["rollout"]).exists()]
                if missing:
                    raise FileNotFoundError(f"rollouts missing: {missing[:2]}")
                # answer credit (arc2 3'): 10 of 50 tokens credited, none for a name in ARC2_PREFLIGHT_NO_ANSWER
                no_answer = os.environ.get("ARC2_PREFLIGHT_NO_ANSWER") or None

                def credited(a):
                    if a.get("credit") != "answer":
                        return None
                    return 0 if no_answer and no_answer in str(a.get("name")) else 10
                out.update(loss=-0.01, grad_norm=0.03, update_norm=0.05, adapter_norm=44.5,
                           trained_tokens=100 * len(attempts), train_seconds=0.5, seconds=0.5, errors=[],
                           memory=memory, windows=[{"name": a.get("name"), "tokens": 100, "targets": 50,
                                                    "trained": 50, "credited": credited(a), "w": a.get("w")} for a in attempts])
                if python:
                    made = new_state()
                    out.update(tensors("update", "--state", current(), "--out", made, "--lr", req["lr"],
                                       "--clip", req.get("clip", 1.0), "--attempts",
                                       json.dumps([{"name": a.get("name"), "w": 0.0 if credited(a) == 0 else a.get("w"),
                                                    "credit": a.get("credit")} for a in attempts])))
                    state["current"] = made
            elif op == "score" and python:  # KL to base 0.003 D^2 (D: the adapter's distance from the base state)
                time.sleep(float(os.environ.get("ARC2_PREFLIGHT_SCORE_DELAY") or 0))  # dose -pfc: the cancel lands mid-run
                first = state["base"] is None
                if first and req.get("label") != "base":
                    raise ValueError("score before a base pass")
                dist = 0.0 if first else tensors("norm", "--state", current(), "--ref", state["base"])["distance"]
                if req.get("label") == "base":
                    state["base"] = current()
                rows = {}
                for w in req.get("windows") or []:
                    kl = 0.003 * dist * dist
                    row = {"tokens": 10, "mismatch": {"tokens": 10, "mean_abs": 0.05, "engine_nll": 0.29,
                                                      "trainer_nll": round(0.3 + 0.5 * kl - (0.001 * dist if w["name"].startswith("held") else 0.0), 6)}}
                    if not first:
                        row.update(kl_base=round(kl, 6), kl_previous=0.0)
                    rows[w["name"]] = row
                out["windows"] = rows
            elif op == "score":
                for w in req.get("windows") or []:
                    refs = references.setdefault(w["name"], set())
                    refs.add("last")
                    if req.get("label") == "base" or not req.get("hold_previous"):
                        refs.add("previous")
                out["windows"] = {w["name"]: {"kl_base": 0.002 * (accepted + 1), "kl_previous": 0.002,
                                              "mismatch": {"tokens": 10, "mean_abs": 0.05, "trainer_nll": 0.3,
                                                           "engine_nll": 0.29}}
                                  for w in req.get("windows") or []}
            elif op == "accept_reference":
                missing = sorted(name for name, refs in references.items() if "last" not in refs)
                if missing:
                    raise ValueError(f"no scored state to accept for {missing}")
                accepted += 1
            elif op == "chunk_check":  # rl1_trainer op_chunk_check's shape: a row per window, a cell per setting
                time.sleep(float(os.environ.get("ARC2_PREFLIGHT_TRAINER_DELAY") or 0))
                attempts = req.get("attempts") or []
                missing = [a["rollout"] for a in attempts if not Path(a["rollout"]).exists()]
                if missing:
                    raise FileNotFoundError(f"rollouts missing: {missing[:2]}")
                names = [s.get("tag") or f"{s.get('select', 'qsa')}@{int(s['chunk'])}" for s in req["settings"]]
                firsts = {}
                for s, key in zip(req["settings"], names):
                    firsts.setdefault(s.get("select", "qsa"), key)
                gap = lambda against: {"against": against, "tokens": 50, "replies": {"log_s": [0.001, -0.002]},
                                       **{f: {"sum": 0.1, "abs_sum": 0.5, "abs_max": 0.2, "beyond_half": 0, "count": 50}
                                          for f in ("actor", "raw")}}
                rows = []
                for a in attempts:
                    cells = {}
                    for s, key in zip(req["settings"], names):
                        cell = {"seconds": 0.1, "prefix_chunk": int(s["chunk"]),
                                "sampler": {"tokens": 50, "mean_abs": 0.05, "mean": 0.01, "beyond_half": 0.01}}
                        if firsts[s.get("select", "qsa")] != key:
                            cell["gap"] = gap(firsts[s.get("select", "qsa")])
                        if names[0] != key:
                            cell["gap_default"] = gap(names[0])
                        cells[key] = cell
                    rows.append({"name": a.get("name"), "tokens": 100, "tail": 60, "cut": 40, "targets": 50, "settings": cells})
                out.update(windows=rows)
            elif op == "park":
                out.update(gib=0.37, seconds=0.1)
            elif op == "profile":
                out.update(skipped="preflight")
            else:
                raise ValueError(f"unknown op {op}")
        except Exception as exc:  # noqa: BLE001 - reported to the caller as the real trainer does
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        tmp = folder / "done" / f"{index:04d}.json.tmp"
        tmp.write_text(json.dumps(out))
        tmp.rename(folder / "done" / f"{index:04d}.json")
        index += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("serve", "server", "stop", "trainer"))
    parser.add_argument("--working")
    parser.add_argument("--bundle", default=os.environ.get("ARC2_PREFLIGHT_BUNDLE", "/opt/arc-validation"))
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--dir")
    args, _ = parser.parse_known_args()
    if args.mode == "serve":
        serve(args.working, args.bundle, args.port)
    elif args.mode == "server":
        serve_forever(args.bundle, args.port)
    elif args.mode == "stop":
        stop(args.working)
    else:
        trainer(args.dir)


if __name__ == "__main__":
    main()
