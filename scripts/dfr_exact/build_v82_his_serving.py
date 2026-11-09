"""v8.2's harness on dfranzen's exact serving: the Nebius bundle.

    python scripts/dfr_exact/build_v82_his_serving.py OUT --v82 <v8.2 bundle> --dfrx <dfrx-c110-100k-v1 bundle>
        [--instances 100] [--streams 10]

The v8.2 bundle is copied as built (its harness, prompts, playbook, frames, drain, budgets, 300K level cap and its
priority scheduler, which is the port of his) with three source adaptations for his engine, listed in ``ADAPTED``.
His server comes from his notebook (the dfrx bundle's notebook, kernel wheels and helpers under ``dfr/``), started
by ``bootstrap_v82sgl.py``. The study runs the arena's ``sglang`` backend at his scheduler's 10 active streams.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP = ("dataset-upload", "experiment.ipynb", "kernel-metadata.json", "MANIFEST.json", "__pycache__", "draft-head")
HIS_ASSETS = ["games", "flash-input/datasets/dfranzen/pennyroyal-v253",
              "flash-input/datasets/dfranzen/taaf-kaggle-source-bundle-copy",
              "flash-input/competitions/arc-prize-2026-arc-agi-3",
              "flash-input/models/dfranzen/albucino-qwen3-8-flash-next-drafter",
              "flash-input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround"]
ADAPTED = {
    "src/arena/sglang.py": "replaced: no sglang_bench (not packaged); TokenBudget as before, the arena's own engine "
                           "monitor, flush isolation",
    "src/arc_infra/metrics.py": "parse_metrics also reads sglang: counters, under the vLLM names the arena uses",
    "src/inference/agent/tool_agent.py": "history reasoning under reasoning_content (his template reads only that key; "
                                         "under v8.2's `reasoning` key SGLang dropped every earlier reasoning block, A v1)",
    "src/arc_infra/model.py": "image_token_count on sglang: the server's prompt_tokens for one image minus its "
                              "template text (his /tokenize does not expand image pads)",
}
METRICS_OLD = '''        match = re.fullmatch(r"((?:vllm|llamacpp):[\\w:]+)(\\{[^}]*\\})?\\s+([-+\\d.eE]+)", line)
        if not match:
            continue
        name, labels, value = match.groups()
'''
METRICS_NEW = '''        match = re.fullmatch(r"((?:vllm|llamacpp|sglang):[\\w:]+)(\\{[^}]*\\})?\\s+([-+\\d.eE]+)", line)
        if not match:
            continue
        name, labels, value = match.groups()
        if name.startswith("sglang:"):
            # dfranzen's SGLang server (12 October): its counters under the vLLM names the arena reads.
            if name in SGLANG_ALIASES:
                values[SGLANG_ALIASES[name]] = values.get(SGLANG_ALIASES[name], 0.0) + float(value)
            values[name] = values.get(name, 0.0) + float(value)
            continue
'''
METRICS_TAIL_OLD = '''        values[name] = values.get(name, 0.0) + float(value)
    return values
'''
METRICS_TAIL_NEW = '''        values[name] = values.get(name, 0.0) + float(value)
    if "sglang:prompt_tokens_total" in values and "sglang:cached_tokens_total" in values:
        cached = values["sglang:cached_tokens_total"]
        values["vllm:prompt_tokens_by_source_total:local_cache_hit"] = cached
        values["vllm:prompt_tokens_by_source_total:local_compute"] = values["sglang:prompt_tokens_total"] - cached
    return values


SGLANG_ALIASES = {"sglang:generation_tokens_total": "vllm:generation_tokens_total",
                  "sglang:num_running_reqs": "vllm:num_requests_running",
                  "sglang:num_queue_reqs": "vllm:num_requests_waiting",
                  "sglang:token_usage": "vllm:kv_cache_usage_perc",
                  "sglang:num_retracted_requests_total": "vllm:num_preemptions_total"}
'''
MODEL_OLD = '''        if self.image_tokens is None:
            self.image_tokens = {}
        if size not in self.image_tokens:
'''
MODEL_NEW = '''        if self.image_tokens is None:
            self.image_tokens = {}
        if size not in self.image_tokens and getattr(self.config, "serving_backend", "vllm") == "sglang":
            # dfranzen's SGLang (12 October): its /tokenize renders one image pad per image, so the count comes from
            # the server's prompt_tokens for the image and for its template text (one token each, temperature 0).
            url = next(p["image_url"]["url"] for m in messages if isinstance(m.get("content"), list)
                       for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url"
                       and png_size(p["image_url"].get("url")) == size)

            async def prompt_tokens(content):
                response = await self.client.post("chat/completions", json={
                    "model": self.config.model, "messages": [{"role": "user", "content": content}], "max_tokens": 1,
                    "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}},
                    timeout=max(1, deadline - time.monotonic()))
                response.raise_for_status()
                return int(response.json()["usage"]["prompt_tokens"])
            image = await prompt_tokens([{"type": "image_url", "image_url": {"url": url}}])
            text = await prompt_tokens([{"type": "text", "text": IMAGE_TEXT}])
            if image - text + 1 < 2:
                raise RuntimeError(f"image token count for {size}: {image} with the image, {text} as text")
            self.image_tokens[size] = image - text + 1
        if size not in self.image_tokens:
'''
# Equal scheduler, as ``priority.equal`` inside the priority policy's state machine: the
# hand-over points stay (context drains/compactions, finished games); the waiting game with the fewest generated tokens
# (clock.completion_tokens; unstarted 0, ties in arrival = roster order) takes a free slot; a game with park_tokens or
# more ranks after every game under it (parked), fewest tokens first among parked games.
PRIORITY_OLD = '''    "level_cap_tokens": 0,
    "level_cap_grant_tokens": 100000,
}
'''
PRIORITY_NEW = '''    "level_cap_tokens": 0,
    "level_cap_grant_tokens": 100000,
    # Equal scheduler: fewest generated tokens first, parked at park_tokens (0 = never).
    "equal": False,
    "park_tokens": 0,
}
'''
SETTINGS_OLD = '''    for key in ("refresh_queue", "tail_fade", "normalize_score", "floor"):'''
SETTINGS_NEW = '''    for key in ("refresh_queue", "tail_fade", "normalize_score", "floor", "equal"):'''
SETTINGS2_OLD = '''    for key, least in (("level_cap_tokens", 0), ("level_cap_grant_tokens", 1)):'''
SETTINGS2_NEW = '''    for key, least in (("level_cap_tokens", 0), ("level_cap_grant_tokens", 1), ("park_tokens", 0)):'''
VALUE_OLD = '''        from .priority import UNSTARTED_BASE, score

        if game.readmissions == 0:'''
VALUE_NEW = '''        from .priority import UNSTARTED_BASE, score

        if self.priority_cfg["equal"]:
            tokens = 0 if game.readmissions == 0 else int(game.clock.completion_tokens())
            return -tokens, {"tokens": tokens, "parked": self._equal_parked(game)}
        if game.readmissions == 0:'''
RANK_OLD = '''        def key(c):
            game = c[0]
            if not game.level_capped:'''
RANK_NEW = '''        def key(c):
            game = c[0]
            if self.priority_cfg["equal"]:
                return (int(self._equal_parked(game)), -c[1], c[3])
            if not game.level_capped:'''
PARKED_OLD = '''    # --- level cap (priority.level_cap_tokens) ---------------------------------------------
'''
PARKED_NEW = '''    def _equal_parked(self, game):
        cap = self.priority_cfg["park_tokens"]
        return bool(cap) and game.readmissions > 0 and game.clock.completion_tokens() >= cap

    # --- level cap (priority.level_cap_tokens) ---------------------------------------------
'''
NOTE_OLD = '''        over = self._level_cap_over(game)
'''
NOTE_NEW = '''        over = self._level_cap_over(game)
        if self.priority_cfg["equal"] and self._equal_parked(game) and not getattr(game, "equal_parked_noted", False):
            game.equal_parked_noted = True
            self.priority_counts["equal_parked"] = self.priority_counts.get("equal_parked", 0) + 1
            self._priority_note("equal_parked", game, tokens=int(game.clock.completion_tokens()),
                                park_tokens=self.priority_cfg["park_tokens"])
'''


# The v8 harness (dfrv8-eq): v8's priority.py / scheduler.py have no level cap, so the same equal
# logic goes in at v8's own anchors (rank key, parked test, parked note and settings identical to the v8.2 patch above).
V8_PRIORITY_OLD = '''    "floor": True,
}
TAIL_LOOKUPS'''
V8_PRIORITY_NEW = '''    "floor": True,
    # Equal scheduler: fewest generated tokens first, parked at park_tokens (0 = never).
    "equal": False,
    "park_tokens": 0,
}
TAIL_LOOKUPS'''
V8_SETTINGS2_OLD = '''    if merged["tail_lookup"] not in TAIL_LOOKUPS:'''
V8_SETTINGS2_NEW = '''    if type(merged["park_tokens"]) is not int or merged["park_tokens"] < 0:
        raise ValueError("priority.park_tokens must be an integer of at least 0")
    if merged["tail_lookup"] not in TAIL_LOOKUPS:'''
V8_RANK_OLD = '''        return sorted(candidates, key=lambda c: (-c[1], c[3]))
'''
V8_RANK_NEW = '''        def key(c):
            game = c[0]
            if self.priority_cfg["equal"]:
                return (int(self._equal_parked(game)), -c[1], c[3])
            return (-c[1], c[3])

        return sorted(candidates, key=key)
'''
V8_PARKED_OLD = '''    def _priority_rank(self, candidates):
'''
V8_PARKED_NEW = PARKED_NEW.split("    # --- level cap")[0] + V8_PARKED_OLD
V8_NOTE_OLD = '''        value, info = self.priority_value(game, fresh=True)
        candidates = self._priority_candidates()
'''
V8_NOTE_NEW = V8_NOTE_OLD + NOTE_NEW.split("\n", 1)[1]


def equal_patch(out, park):
    """The equal scheduler on the bundle's arena sources (v8.2 anchors, or v8's when it has no level cap)."""
    v8 = "level_cap_tokens" not in (out / "src/arena/priority.py").read_text()
    pri = ((V8_PRIORITY_OLD, V8_PRIORITY_NEW), (SETTINGS_OLD, SETTINGS_NEW), (V8_SETTINGS2_OLD, V8_SETTINGS2_NEW)) if v8 \
        else ((PRIORITY_OLD, PRIORITY_NEW), (SETTINGS_OLD, SETTINGS_NEW), (SETTINGS2_OLD, SETTINGS2_NEW))
    sch = ((VALUE_OLD, VALUE_NEW), (V8_RANK_OLD, V8_RANK_NEW), (V8_PARKED_OLD, V8_PARKED_NEW), (V8_NOTE_OLD, V8_NOTE_NEW)) \
        if v8 else ((VALUE_OLD, VALUE_NEW), (RANK_OLD, RANK_NEW), (PARKED_OLD, PARKED_NEW), (NOTE_OLD, NOTE_NEW))
    for old, new in pri:
        patch(out / "src/arena/priority.py", old, new)
    for old, new in sch:
        patch(out / "src/arena/scheduler.py", old, new)
    return v8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"{path}: patch anchor not found once")
    path.write_text(text.replace(old, new))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--v82", required=True)
    ap.add_argument("--dfrx", required=True)
    ap.add_argument("--instances", type=int, default=100)
    ap.add_argument("--streams", type=int, default=10)
    ap.add_argument("--stop-tokens", type=int, help="an instance finishes at this many generated tokens and the trial "
                    "ends when all have (build_v82_ablation's token stop); 300K quality runs")
    ap.add_argument("--equal-park", type=int, help="equal scheduler, parked at this many generated tokens per game")
    ap.add_argument("--serving-from", help="a built his-serving bundle (A v2) whose server block "
                    "and KV-gate density this one takes (the vLLM prefix-cache options stay the source bundle's: unused on SGLang), so only the harness differs")
    ap.add_argument("--replace-src", action="append", default=[], metavar="REL=FILE",
                    help="replace src/REL with FILE (e.g. the tool-latency fix, main 6abfd6f8: duck_transfer/history.py)")
    ap.add_argument("--his-mamba", type=int, help="his CFG MAMBA_CACHE (Mamba state slots), overriding 6 x streams")
    ap.add_argument("--his-spec-steps", type=int, help="his CFG SPEC_STEPS (draft tokens = steps + 1); 3 in his notebook")
    ap.add_argument("--gate-reply-reserve", type=int, help="gate reserves prompt + this many reply tokens instead of the "
                    "completion cap (max_tokens unchanged); measured A v2 p95 7,242 / p99 14,074")
    ap.add_argument("--his-kv", choices=["nvfp4", "fp4_mx_block16"], help="his KV cache dtype (fp8_e4m3); FP4 KV needs "
                    "attention backend triton; the draft's KV stays fp8_e4m3 (no FP4 choice for the draft pool)")
    ap.add_argument("--his-profile", help="minutes after server ready at which to capture torch-profiler traces "
                    "(SGLang /start_profile, 40 steps, CPU+GPU) into WORKING_DIR/profile-<min>, e.g. 25,45")
    ap.add_argument("--his-attn", choices=["triton"], help="attention backend only (his default auto), KV unchanged: "
                    "separates the backend's speed from FP4 KV's")
    ap.add_argument("--his-prune", help="expert prune mask (arc-expert-prune-v1): the server loads a pruned view of his "
                    "checkpoint built at start (kept experts renumbered, router rows sliced; scripts/dfr_exact/prune_view.py)")
    ap.add_argument("--his-int8-dense", action="store_true", help="int8 dense view (scripts/dfr_exact/int8_dense_view.py)")
    ap.add_argument("--his-mrope-async", action="store_true", help="patch his SGLang's blocking M-RoPE delta upload "
                    "(scripts/dfr_exact/mrope_async_patch.py): the per-step host sync under draft_extend")
    ap.add_argument("--his-moe-wna16", action="store_true", help="his auto-round MoE on SGLang's triton W4A16 "
                    "(MoeWNA16) instead of GPTQ-Marlin: a config view with quantization_config backend 'gptq', same weights")
    ap.add_argument("--his-arg", action="append", default=[], metavar="ARG",
                    help="extra server argument appended last (repeat per token; argparse last-wins overrides his)")
    ap.add_argument("--his-env", action="append", default=[], metavar="K=V", help="extra server environment variable")
    ap.add_argument("--his-no-autotune", action="store_true", help="his CFG AUTOTUNE False (--disable-flashinfer-autotune): "
                    "fixed kernel tactics across boots, for deterministic captures")
    ap.add_argument("--his-capture", action="store_true", help="kernel-quality capture instead of play (sgl_capture.py)")
    ap.add_argument("--his-memfrac", type=float, help="his CFG MEMFRAC (0.96): s16 OOMed in a prefill at 0.96, 14 Oct")
    ap.add_argument("--his-streams", type=int, help="his server's request limit (CFG MAXREQ, CUDAGRAPH_MAXBS, decode "
                    "graph sizes, Mamba cache 6x) via whole-line edits of his CFG cell; A v2 base stream sweep")
    ap.add_argument("--wall", type=int, help="run wall seconds for a source bundle built as a 10-minute Kaggle preview "
                    "(v8 submission): wall, preview and notebook seconds as the v8.2 24 h bundle's, live preview score")
    a = ap.parse_args()
    out, v82, dfrx = Path(a.out), Path(a.v82), Path(a.dfrx)
    if out.exists():
        raise SystemExit(f"{out} exists")
    shutil.copytree(v82, out, ignore=shutil.ignore_patterns(*SKIP))
    shutil.copyfile(HERE / "arena_sglang.py", out / "src/arena/sglang.py")
    patch(out / "src/arc_infra/metrics.py", METRICS_OLD, METRICS_NEW)
    patch(out / "src/arc_infra/metrics.py", METRICS_TAIL_OLD, METRICS_TAIL_NEW)
    patch(out / "src/arc_infra/model.py", MODEL_OLD, MODEL_NEW)
    patch(out / "src/inference/agent/tool_agent.py", '                    assistant_message["reasoning"] = self._history_text(',
          '                    assistant_message["reasoning_content"] = self._history_text(')
    adapted = dict(ADAPTED)
    for item in a.replace_src:
        rel, _, src = item.partition("=")
        target = out / "src" / rel
        if not target.is_file() or not Path(src).is_file():
            raise SystemExit(f"--replace-src {item}: no such file")
        shutil.copyfile(src, target)
        adapted[f"src/{rel}"] = f"replaced by {src} (sha256 {sha(target)[:16]})"
    if a.stop_tokens:
        import build_v82_ablation as abl  # lazy: it imports this module
        patch(out / "src/arena/trial.py", abl.TOKEN_STOP_OLD, abl.TOKEN_STOP_NEW)
    if a.equal_park:
        harness_v8 = equal_patch(out, a.equal_park)
        adapted["src/arena/priority.py + scheduler.py"] = (
            "equal scheduler: priority.equal ranks waiting games by fewest generated tokens, "
            f"parked at {a.equal_park}; hand-overs only at drains/compactions and finished games")
    (out / "dfr").mkdir()
    shutil.copytree(dfrx / "wheels", out / "dfr/wheels")
    shutil.copytree(dfrx / "notebook", out / "dfr/notebook")
    for name in ("bootstrap_dfr.py", "run_notebook.py"):
        shutil.copyfile(HERE / name, out / "dfr" / name)
    shutil.copyfile(HERE / "bootstrap_v82sgl.py", out / "bootstrap_v82sgl.py")
    ref = json.loads((dfrx / "configs/comparison.json").read_text())["dfr_exact"]
    study = json.loads((out / "configs/comparison.json").read_text())
    if a.stop_tokens:
        study["token_stop"] = a.stop_tokens
    if len(study["games"]) != 20 or study.get("level_budget_policy") != "priority":
        raise SystemExit("not the v8.2 c110 bundle")
    if a.wall:
        study.update(wall_seconds=a.wall, notebook_seconds=a.wall + 3900)
        study["competition"].pop("no_preview", None)
        study["competition"].update(preview_seconds=a.wall, preview_live_score=True)
    study.update(name=out.name, status="prepared-not-launched", backend="sglang", generation_slots=a.streams,
                 active_limit=a.streams)
    study.pop("b200", None)
    study["server"]["max_num_seqs"] = a.streams
    study["server"].pop("draft_head", None)  # our vLLM draft head: not served here (his NEXTN drafter is)
    if a.equal_park:
        study["priority"].update(equal=True, park_tokens=a.equal_park, tail_fade=False)
        if not harness_v8:
            study["priority"]["level_cap_tokens"] = 0
    if study.get("focus_oversubscription"):
        study["focus_oversubscription"]["focus_slots"] = a.streams
    study["competition"]["preview_instances"] = a.instances
    if a.serving_from:
        ref_study = json.loads((Path(a.serving_from) / "configs/comparison.json").read_text())
        study["server"] = {**ref_study["server"], "max_num_seqs": a.streams}
        study["focus_oversubscription"]["gate_blocks_per_stream"] = ref_study["focus_oversubscription"]["gate_blocks_per_stream"]
    study["dfr_exact"] = {
        "entry": "bootstrap_v82sgl.py", "mode": "v82-harness-his-serving",
        "notebook_file": "notebook/arc-agi-3-milestone-2-solution.json", "notebook_sha256": ref["notebook_sha256"],
        "his_cells_run": [2, 4, 6, 8, 10, 12], "server_start_seconds": 2700,
        # Server start (incl. checkpoint views) + play + 20 min: a hung VM cannot hold a GPU for a day.
        "timeout_seconds": 2700 + int(study["wall_seconds"]) + 1200,
        "play_backstop_seconds": int(study["wall_seconds"]), "assets": HIS_ASSETS, "adapted": adapted,
        "v82_bundle": v82.name, "instances": a.instances, "streams": a.streams}
    if a.his_streams:
        n = a.his_streams
        if n != a.streams:
            raise SystemExit("--his-streams goes with the same --streams")
        study["dfr_exact"]["notebook_edits"] = [
            {"cell": 12, "old": "    MAXREQ=10,", "new": f"    MAXREQ={n},"},
            {"cell": 12, "old": "    CUDAGRAPH_MAXBS=10,", "new": f"    CUDAGRAPH_MAXBS={n},"},
            {"cell": 12, "old": "    MAMBA_CACHE=60,", "new": f"    MAMBA_CACHE={max(60, 6 * n)},"},
            {"cell": 12, "old": 'graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})',
             "new": 'graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, %s, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})'
                    % ", ".join(str(k) for k in range(12, n + 1, 2))}]
    edits = study["dfr_exact"].setdefault("notebook_edits", [])
    if a.his_mamba:
        edits[:] = [e for e in edits if not e["old"].startswith("    MAMBA_CACHE=")]
        edits.append({"cell": 12, "old": "    MAMBA_CACHE=60,", "new": f"    MAMBA_CACHE={a.his_mamba},"})
    if a.his_spec_steps:
        edits.append({"cell": 12, "old": "    SPEC_STEPS=3,", "new": f"    SPEC_STEPS={a.his_spec_steps},"})
    if a.gate_reply_reserve:
        study["gate_reply_reserve"] = a.gate_reply_reserve
        patch(out / "src/arena/trial.py", "slot = GateSlot(gate, game, cap=self.cap, window=window)",
              'slot = GateSlot(gate, game, cap=self.study.get("gate_reply_reserve") or self.cap, window=window)')
    if a.his_kv:
        edits += [{"cell": 12, "old": '    KVDTYPE="fp8_e4m3",', "new": f'    KVDTYPE="{a.his_kv}",'},
                  {"cell": 12, "old": '    "--kv-cache-dtype", CFG["KVDTYPE"], "--mem-fraction-static", str(CFG["MEMFRAC"]),',
                   "new": '    "--kv-cache-dtype", CFG["KVDTYPE"], "--attention-backend", "triton", '
                          '"--mem-fraction-static", str(CFG["MEMFRAC"]),'},
                  {"cell": 12, "old": '        "--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"],',
                   "new": '        "--speculative-draft-kv-cache-dtype", "fp8_e4m3",'}]
    if a.his_attn and not a.his_kv:
        edits.append({"cell": 12, "old": '    "--kv-cache-dtype", CFG["KVDTYPE"], "--mem-fraction-static", str(CFG["MEMFRAC"]),',
                      "new": f'    "--kv-cache-dtype", CFG["KVDTYPE"], "--attention-backend", "{a.his_attn}", '
                             '"--mem-fraction-static", str(CFG["MEMFRAC"]),'})
    if a.his_profile:
        mins = [float(x) for x in a.his_profile.split(",")]
        edits.append({"cell": 12, "old": 'LOG = str(Path(WORKING_DIR) / "serve.log")',
                      "new": 'LOG = str(Path(WORKING_DIR) / "serve.log")\n'
                             "def _arc_profiler(mins=%r):  # arc: step profile (not dfranzen's)\n" % mins
                             + "    import threading, urllib.request, json as _j, time as _t\n"
                             + "    def run():\n"
                             + "        base = f'http://127.0.0.1:{SERVED_MODEL_PORT}'\n"
                             + "        while True:\n"
                             + "            try:\n"
                             + "                if urllib.request.urlopen(base + '/health', timeout=5).status == 200: break\n"
                             + "            except Exception: pass\n"
                             + "            _t.sleep(10)\n"
                             + "        ready = _t.time()\n"
                             + "        for m in mins:\n"
                             + "            _t.sleep(max(0, ready + 60 * m - _t.time()))\n"
                             + "            body = {'output_dir': str(Path(WORKING_DIR) / f'profile-{int(m)}'), 'num_steps': 40, "
                               "'activities': ['CPU', 'GPU'], 'with_stack': True, 'record_shapes': True}\n"
                             + "            try:\n"
                             + "                r = urllib.request.urlopen(urllib.request.Request(base + '/start_profile', "
                               "data=_j.dumps(body).encode(), headers={'Content-Type': 'application/json'}), timeout=600)\n"
                             + "                print('[arc] profile', m, r.status, flush=True)\n"
                             + "            except Exception as e: print('[arc] profile failed', m, e, flush=True)\n"
                             + "    threading.Thread(target=run, daemon=True).start()\n"
                             + "_arc_profiler()"})
        study["his_profile"] = {"minutes_after_ready": mins, "num_steps": 40}
    ARGS_LINE = 'args = [SGLANG, "serve", "--model-path", MODEL_DIR, "--load-format", "safetensors",'
    pre = ""  # code run before the server's args line: checkpoint views, chained (prune, then int8 dense)
    if a.his_prune:
        mask = json.loads(Path(a.his_prune).read_text())
        code = (Path(__file__).parent / "prune_view.py").read_text()
        pre += ("# arc: expert-pruned view of his checkpoint (mask %s)\n" % Path(a.his_prune).name
                + "exec(%r)\n" % code
                + "import pathlib as _arc_pl, time as _arc_time\n"
                + "_arc_m = _arc_pl.Path('/tmp/arc-prune-mask.json'); _arc_m.write_text(%r)\n" % json.dumps(mask)
                + "_arc_t0 = _arc_time.time()\n"
                + "print('[arc] prune view', arc_prune_view(ARC_VIEW, '/tmp/arc-pruned-model', _arc_m), "
                  "round(_arc_time.time() - _arc_t0, 1), 's', flush=True)\n"
                + "ARC_VIEW = '/tmp/arc-pruned-model'\n")
        # The MTP drafter shares the target's expert map (EPLB logical_to_all_physical sized by the target's
        # num_experts): its 512 experts must be cut to the same count. Draft experts only change acceptance, never
        # the output (the target verifies); the last layer's mask stands in (no usage rows for the MTP layer).
        last = str(max(int(k) for k in mask["pruned"]))
        dmask = json.dumps({"experts": mask["experts"], "pruned": {"0": mask["pruned"][last]}})
        edits.append({"cell": 12, "old": "    DRAFT_VIEW = prepare_draft_view(MODEL_DIR, DRAFT_MODEL_DIR, PREFIX)",
                      "new": "    DRAFT_VIEW = prepare_draft_view(MODEL_DIR, DRAFT_MODEL_DIR, PREFIX)\n"
                             + "    exec(%r)\n" % code
                             + "    Path('/tmp/arc-prune-draft-mask.json').write_text(%r)\n" % dmask
                             + "    print('[arc] draft prune view', arc_prune_view(DRAFT_VIEW, '/tmp/arc-pruned-draft', "
                               "'/tmp/arc-prune-draft-mask.json', prefix='mtp.layers.'), flush=True)\n"
                             + "    DRAFT_VIEW = Path('/tmp/arc-pruned-draft')"})
        study["his_prune"] = {"mask": Path(a.his_prune).name, "kept": mask["experts"] - mask["pruned_per_layer"],
                              "draft_mask_layer": int(last)}
    if a.his_mrope_async:
        pre += ("# arc: non-blocking M-RoPE delta upload in his SGLang (scripts/dfr_exact/mrope_async_patch.py)\n"
                + "import pathlib as _arc_pl, subprocess as _arc_sp\n"
                + "_arc_pl.Path('/tmp/arc_mrope_async_patch.py').write_text(%r)\n"
                  % (Path(__file__).parent / "mrope_async_patch.py").read_text()
                + "_arc_sg = _arc_sp.run([PYTHON, '-c', 'import sglang, os; print(os.path.dirname(sglang.__file__))'], "
                  "capture_output=True, text=True, check=True).stdout.strip()\n"
                + "print(_arc_sp.run([PYTHON, '/tmp/arc_mrope_async_patch.py', _arc_sg], capture_output=True, text=True, "
                  "check=True).stdout, flush=True)\n")
        study["his_mrope_async"] = True
    if a.his_kv == "nvfp4":
        pre += ("# arc: NVFP4 KV reads in his QSA backend (scripts/dfr_exact/qsa_kv4_patch.py)\n"
                + "import pathlib as _arc_pl, subprocess as _arc_sp\n"
                + "_arc_pl.Path('/tmp/arc_qsa_kv4_patch.py').write_text(%r)\n"
                  % (Path(__file__).parent / "qsa_kv4_patch.py").read_text()
                + "_arc_sg = _arc_sp.run([PYTHON, '-c', 'import sglang, os; print(os.path.dirname(sglang.__file__))'], "
                  "capture_output=True, text=True, check=True).stdout.strip()\n"
                + "print(_arc_sp.run([PYTHON, '/tmp/arc_qsa_kv4_patch.py', _arc_sg], capture_output=True, text=True, "
                  "check=True).stdout, flush=True)\n")
        study["his_kv4_patch"] = {"recipe": "nvfp4", "global_scale": "first eager write amax x 16 / (6 x 448)",
                                  "dequant_workspace": False, "attention_kv_dtype": "bf16"}
    if a.his_int8_dense:
        code = (Path(__file__).parent / "int8_dense_view.py").read_text()
        pre += ("# arc: int8 dense view (GPTQ W8A16 through his auto-round extra_config), built in his venv\n"
                + "import pathlib as _arc_pl, subprocess as _arc_sp\n"
                + "_arc_pl.Path('/tmp/arc_int8_dense_view.py').write_text(%r)\n" % code
                + "_arc_pl.Path('/tmp/arc_qwen35_gptq_patch.py').write_text(%r)\n"
                  % (Path(__file__).parent / "qwen35_gptq_patch.py").read_text()
                + "_arc_sg = _arc_sp.run([PYTHON, '-c', 'import sglang, os; print(os.path.dirname(sglang.__file__))'], "
                  "capture_output=True, text=True, check=True).stdout.strip()\n"
                + "print(_arc_sp.run([PYTHON, '/tmp/arc_qwen35_gptq_patch.py', _arc_sg + '/srt/models/qwen3_5.py'], "
                  "capture_output=True, text=True, check=True).stdout, flush=True)\n"
                + "_arc_r = _arc_sp.run([PYTHON, '/tmp/arc_int8_dense_view.py', ARC_VIEW, '/tmp/arc-int8-model'], "
                  "capture_output=True, text=True)\n"
                + "print(_arc_r.stdout[-2000:], _arc_r.stderr[-3000:], flush=True)\n"
                + "assert _arc_r.returncode == 0, 'int8 dense view failed'\n"
                + "ARC_VIEW = '/tmp/arc-int8-model'\n")
        study["his_int8_dense"] = {"pattern": "GDN qkv/z/out, attention q/k/v/o, shared experts", "group_size": 128}
    if a.his_moe_wna16:
        pre += ("# arc: MoE on triton W4A16 (MoeWNA16): config view with quantization_config backend 'gptq'\n"
                + "import json as _arc_json, os as _arc_os, pathlib as _arc_pl\n"
                + "_arc_src, _arc_dst = _arc_pl.Path(ARC_VIEW), _arc_pl.Path('/tmp/arc-wna16-model')\n"
                + "_arc_dst.mkdir(parents=True, exist_ok=True)\n"
                + "for _arc_f in _arc_src.iterdir():\n"
                + "    if _arc_f.name not in ('config.json', 'quantization_config.json') and not (_arc_dst / _arc_f.name).exists():\n"
                + "        _arc_os.symlink(_arc_f.resolve(), _arc_dst / _arc_f.name)\n"
                + "for _arc_n in ('config.json', 'quantization_config.json'):\n"
                + "    if (_arc_src / _arc_n).exists():\n"
                + "        _arc_c = _arc_json.loads((_arc_src / _arc_n).read_text())\n"
                + "        _arc_q = _arc_c.get('quantization_config', _arc_c) if _arc_n == 'config.json' else _arc_c\n"
                + "        if _arc_n == 'quantization_config.json' or 'quantization_config' in _arc_c:\n"
                + "            print('[arc] wna16 view', _arc_n, 'backend', _arc_q.get('backend'), '-> gptq', flush=True)\n"
                + "            _arc_q['backend'] = 'gptq'\n"
                + "        (_arc_dst / _arc_n).write_text(_arc_json.dumps(_arc_c, indent=2))\n"
                + "ARC_VIEW = str(_arc_dst)\n")
        study["his_moe_wna16"] = True
    if a.his_arg or a.his_env:
        envs = dict(kv.split("=", 1) for kv in a.his_env)
        edits.append({"cell": 12, "old": "# ---- launch detached and wait for health ----",
                      "new": "args += %r  # arc: --his-arg\nenv.update(%r)  # arc: --his-env\n"
                             "# ---- launch detached and wait for health ----" % (a.his_arg, envs)})
        study["his_args"], study["his_env"] = a.his_arg, envs
    if a.his_capture:
        shutil.copyfile(HERE / "sgl_capture.py", out / "sgl_capture.py")
        kq = out / "configs/kernel-quality"; kq.mkdir(parents=True, exist_ok=True)
        data = HERE.parents[1].parent / "throughput-arm/data/kernel_quality"
        for n in ("prompts.json", "gsm8k.jsonl"):
            shutil.copyfile(data / n, kq / n)
        study["dfr_exact"]["capture"] = True
    if pre:
        edits.append({"cell": 12, "old": ARGS_LINE,
                      "new": "ARC_VIEW = MODEL_DIR\n" + pre + ARGS_LINE.replace("MODEL_DIR", "ARC_VIEW")})
    if a.his_no_autotune:
        study["dfr_exact"].setdefault("notebook_edits", []).append(
            {"cell": 12, "old": "    AUTOTUNE=True,", "new": "    AUTOTUNE=False,"})
    if a.his_memfrac:
        study["dfr_exact"].setdefault("notebook_edits", []).append(
            {"cell": 12, "old": "    MEMFRAC=0.96,", "new": f"    MEMFRAC={a.his_memfrac},"})
    (out / "configs/comparison.json").write_text(json.dumps(study, indent=2) + "\n")
    identity = json.loads((v82 / "MANIFEST.json").read_text())
    identity["files"] = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
                         if p.is_file() and p.suffix != ".ipynb" and "__pycache__" not in p.parts}
    identity.update(dependencies_prepared=True, study=out.name)
    (out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps({"bundle": str(out), "files": len(identity["files"]), "manifest_sha256": sha(out / "MANIFEST.json")}))


if __name__ == "__main__":
    main()
