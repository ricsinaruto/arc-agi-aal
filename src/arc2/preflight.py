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
    state = {"asleep": False, "loras": set(), "requests": 0, "flying": 0, "cut": {}, "cuts": 0, "cut_chat": {}}
    # cache-gate tests (arc2.admission): KV use = ARC2_PREFLIGHT_KV_PER per reply in flight; /metrics down on
    # ARC2_PREFLIGHT_METRICS_DOWN
    kv_per = float(os.environ.get("ARC2_PREFLIGHT_KV_PER") or 0)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body, code=200, kind="application/json"):
            if getattr(self, "capture", None) is not None:  # think_cap: the full reply, held for /v1/completions
                self.capture.append(body)
                return
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/metrics"):
                if os.environ.get("ARC2_PREFLIGHT_METRICS_DOWN"):
                    return self.reply("down", 503, kind="text/plain")
                kv = min(1.0, kv_per * state["flying"]) if kv_per else 0.1
                n = state["requests"]  # prefix counters: 1,000 queried tokens a request, 900 of them hits
                return self.reply("vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
                                  f"vllm:kv_cache_usage_perc {kv}\nvllm:num_preemptions_total 0\n"
                                  f'vllm:prefix_cache_queries_total{{engine="0"}} {1000 * n}\n'
                                  f'vllm:prefix_cache_hits_total{{engine="0"}} {900 * n}\n', kind="text/plain")
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
            if path == "/detokenize":
                return self.reply({"prompt": ""})
            if path == "/v1/completions":  # think_cap's continuation: the rest of the reply cut at the cap
                key = next((t for t in reversed(body.get("prompt") or []) if t >= 900000), None)
                full = state["cut"].pop(key, None)
                if full is None:
                    return self.reply({"error": "unknown cut reply"}, 400)
                return self.reply({"choices": [{"text": full["text"], "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 20,
                                             "prompt_tokens_details": {"cached_tokens": len(body["prompt"]) - 8}}})
            if path == "/tokenize":
                count = len(json.dumps(body.get("messages") or body.get("prompt") or "")) // 4
                return self.reply({"count": count, "tokens": list(range(count)), "max_model_len": 131072})
            if path == "/v1/chat/completions":
                if state["asleep"]:
                    return self.reply({"error": "server is asleep"}, 503)
                state["requests"] += 1
                state["flying"] += 1
                try:
                    return self.chat(body)
                finally:
                    state["flying"] -= 1
            return self.reply({"error": f"unknown path {path}"}, 404)

        def chat(self, body):
            msgs = body.get("messages") or []
            if body.get("continue_final_message") and msgs and msgs[-1].get("role") == "assistant" \
                    and msgs[-1].get("reasoning_content"):  # think_chat's continuation of a cut reply
                full = state["cut_chat"].pop(json.dumps(msgs[:-1], sort_keys=True), None)
                if full is not None:
                    return self.reply({"choices": [{"message": {"role": "assistant", "content": None,
                                                                "reasoning_content": full["text"]},
                                                    "finish_reason": "stop"}],
                                       "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                                                 "prompt_tokens_details": {"cached_tokens": 92}}})
            if body.get("return_token_ids") and int(body.get("max_tokens") or 0) < 40 and not body.get("prompt_logprobs"):
                # think_cap (preflight scale): the reply is cut while thinking; its answer comes from /v1/completions
                self.capture = []
                try:
                    self.chat_full(body)
                finally:
                    full, self.capture = self.capture[0], None
                if "choices" not in full:
                    return self.reply(full)
                message = full["choices"][0]["message"]
                text = message.get("content") or ""
                for call in message.get("tool_calls") or []:
                    args = json.loads(call["function"]["arguments"] or "{}")
                    text += "<tool_call>\n<function=python>\n" + "".join(
                        f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items()) + "</function>\n</tool_call>"
                state["cuts"] += 1
                key = 900000 + state["cuts"]
                state["cut"][key] = {"text": text}
                state["cut_chat"][json.dumps(body.get("messages") or [], sort_keys=True)] = {"text": text}
                return self.reply({"choices": [{"message": {"role": "assistant", "content": None,
                                                            "reasoning_content": "preflight thinking"},
                                                "finish_reason": "length", "token_ids": [11, 12, key]}],
                                   "usage": {"prompt_tokens": full["usage"]["prompt_tokens"], "completion_tokens": 20,
                                             "prompt_tokens_details": {"cached_tokens": 0}},
                                   "prompt_token_ids": list(range(int(full["usage"]["prompt_tokens"])))})
            return self.chat_full(body)

        def chat_full(self, body):
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
            # base13b's refusal loop: a tenth of the prompts never have two fitting functions and keep calling submit
            stuck = os.environ.get("ARC2_PREFLIGHT_SUBMIT") == "twofit" and sum(map(ord, first)) % 10 == 0
            if str(os.environ.get("ARC2_PREFLIGHT_SUBMIT"))[:3] in ("b14", "b15"):
                stuck = True  # every reply a tool call; b14_code decides
            if os.environ.get("ARC2_PREFLIGHT_TOOLS") and body.get("tools") and body.get("tool_choice") != "none" and (stuck or calls < (14 if long_attempt else 4)):
                code = TOOL_SNIPPETS[min(calls, len(TOOL_SNIPPETS) - 1)]
                mode = os.environ.get("ARC2_PREFLIGHT_SUBMIT")
                if mode in ("half", "two") and sum(map(ord, first)) % 2:  # B preflight: half the prompts never submit
                    mode = None
                if mode == "tenth" and sum(map(ord, first)) % 10 == 0:  # b2: a tenth never submit (conv1kv: 45 of
                    mode = None  # 240 open at 9.26M), so the backlog falls below the slots and attempts start
                if mode and calls in (2, 3):  # conversation: two submits,
                    # a wrong one, then the answer table's (when it knows the task): a wrong -> right overwrite
                    known = next((p for key, p in table.items() if json.dumps(key)[1:-1] in prompt), None)
                    # ARC2_PREFLIGHT_SUBMIT=right (copy3 preflight): the right one first
                    right = known is not None and (calls == 3 or mode == "right")
                    code = known + "\nsubmit(transform)" if right else "submit(list(test_inputs))"
                    if mode == "two":  # try_cap: two different guesses; a later try repeats the wrong one (refused),
                        # then a known task's right one with a new second guess (accepted: 4 distinct)
                        code = (known + f"\nsubmit(transform, [[[{calls}]] for _ in test_inputs])" if right
                                else f"submit(list(test_inputs), [[[{calls}]] for _ in test_inputs])")
                if mode == "twofit" and calls >= 2:
                    known = next((p for key, p in table.items() if json.dumps(key)[1:-1] in prompt), None)
                    bare = f"submit(list(test_inputs), [[[{calls}]] for _ in test_inputs])"
                    code = bare if stuck or calls == 2 or known is None else (
                        known + f"\ndef alt(g):\n    return transform(g) if any(g == p['input'] for p in train) else [[{calls}]]\n"
                        "submit(transform, alt)")
                if mode in ("b15c", "b15d", "b15f"):
                    code = b15cd_code(mode, first, body.get("messages") or [], calls, next((p for key, p in table.items() if json.dumps(key)[1:-1] in prompt), None))
                elif str(mode).startswith("b15"):
                    code = b15_code(first, body.get("messages") or [], calls, next((p for key, p in table.items() if json.dumps(key)[1:-1] in prompt), None))
                if str(mode).startswith("b14"):
                    code = b14_code(mode, first, calls, next((p for key, p in table.items() if json.dumps(key)[1:-1] in prompt), None))
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

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def b15cd_code(mode, prompt, messages, calls, known):
    """base15c / d's sim, by prompt. c: a tenth never submit, a tenth repeat at the ask (refused), the rest change. d:
    a tenth never submit, a tenth go idle after a submit (the backstop ends them), a fifth alternate two answers, then
    settle; the rest one revision, then the same answer three times."""
    g = sum(map(ord, prompt)) % 10
    text = " ".join(str(m.get("content")) for m in messages if m.get("role") == "tool")
    done = text.count("alternative second answer" if mode == "b15c" else "Are you sure")
    grid = lambda v: f"submit([[[{v}]] for _ in test_inputs])"  # noqa: E731
    if g == 0 or calls < 2:  # the prompt grows so a small window drains
        return f"y = {calls}\nprint(y, list(range({calls % 97})))"
    if mode == "b15f":  # base15f / 16: a fifth try grids first (refused), then the program three times
        if g < 3 and "Grids alone are refused" not in text:
            return grid(1)
        return f"{known}\nsubmit(transform)" if known is not None else "print('no program')"
    if mode == "b15c":
        if not done:
            return grid(1)
        if g == 1 and "same as the one" not in text:
            return grid(1)
        return f"{known}\nsubmit(transform)" if known is not None else grid(2 + calls % 7)
    if g == 1:
        return grid(1) if not done else "print('ok')"
    if g in (2, 3):
        return grid(1 + done % 2) if done < 6 else grid(3)
    if not done:
        return grid(1)
    return f"{known}\nsubmit(transform)" if known is not None else grid(2)


def b15_code(prompt, messages, calls, known):
    """base15's sim, by prompt: a tenth never submit (tool calls until the window drains, the mark ends them), a tenth
    repeat their first answer at the reopen (refused) and then change it, the rest change it at once."""
    g = sum(map(ord, prompt)) % 10
    at = next((i for i, m in enumerate(messages) if m.get("role") == "user" and "slightly wrong" in str(m.get("content"))), None)
    if at is None:
        if g == 0 or calls < 2:
            return f"y = {calls}\nprint(y, list(range({calls % 97})))"
        return "submit(list(test_inputs))"
    n = sum(m.get("role") == "tool" for m in messages[at:])
    if n == 0 and g == 1:
        return "submit(list(test_inputs))"
    if n == 0:
        return "z = 1"
    if known is not None:
        return f"{known}\nsubmit(transform)"
    return "g = [[[(v + 1) % 10 for v in r] for r in x] for x in test_inputs]\nsubmit(g)"


B14_PASSES = {}


def b14_group(prompt):
    return sum(map(ord, prompt)) % 10


def b14_code(mode, prompt, calls, known):
    """base14's sim, by prompt: an early submitter and a late one agreeing at pass 2, one differing on the last output
    at pass 2 and repeating at pass 3, a pass-1 bad-call loop then the agreeing script, a pass-1 no-progress loop then
    the same; ``b14d`` adds entries that never repeat, so only the mark ends the run. Pass number: this prompt's
    opening requests so far."""
    if calls == 0:
        B14_PASSES[prompt] = B14_PASSES.get(prompt, 0) + 1
        return "x = 1"
    p, g = B14_PASSES.get(prompt, 1), b14_group(prompt)
    if g == 0 and p == 1:
        return ""  # {"code": ""}: 7491f3cf
    if g == 1 and p == 1:
        return "print(1)"  # no progress, same reply: the loop rule
    if g in (2, 3, 4):  # early
        at = 2
    else:
        at = 8
    if calls < at:
        return f"y = {calls}"
    if g in (5, 6):
        value = min(p, 2)  # 1, 2, 2: closes at pass 3
    elif g == 7 and mode == "b14d":
        value = p  # never repeats
    else:
        value = 0
    second = "[[[9]] for _ in test_inputs]"
    if value == 0 and known is not None:
        return f"{known}\nsubmit(transform, {second})"
    return ("g = [[list(r) for r in x] for x in test_inputs]\n"
            f"g[-1][0][0] = (g[-1][0][0] + {value}) % 10\nsubmit(g, {second})")


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
