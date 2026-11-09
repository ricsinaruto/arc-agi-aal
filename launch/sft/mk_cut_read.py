"""Builds a BF16-dense depth-cut held-out read tree (v9b INT8 tree + token_stop port + drop edit): the d16 read, 31 Oct.
The working recipe for a cut student on RTX. Writes into the cwd. usage: mk_cut_read.py d16 STEP"""
import hashlib, json, shutil, sys
from pathlib import Path
DIST = Path("./.claude/worktrees/v84sub/dist")
SRC = DIST / "b2-v9b-p75c613-int8-check-v1"
arm = sys.argv[1]
STEP = sys.argv[2] if len(sys.argv) > 2 else ""  # d16: the checkpoint's step
D16 = [5, 9, 10, 11, 12, 13, 16, 18, 19, 20, 21, 22, 23, 29, 39, 44]
V9 = DIST.parent / "docs/validation-20261009/v9"
SIZES = [3, 6, 9, 18, 33, 48, 60, 66, 72, 81, 90, 96, 105, 114, 129, 144, 162, 177, 192, 210, 228, 240, 255]  # MTP-2: multiples of 3 (kprofR)
ARMS = {f"d16": (f"flash-input/models/arc-heal/p75-d16-c{STEP}", 8, 85, 1),
        "c893": ("flash-input/models/arc-heal/p75-qad-c893", 6, 60, 1),
        "c893fp8": ("flash-input/models/arc-heal/p75-qad-c893", 6, 60, 1),
        "c204cut": ("flash-input/models/arc-heal/p75-cut32x64k6-c204", 6, 60, 4)}  # 60 streams, pairs with c893
model, clones, active, ver = ARMS[arm]
run = f"b2-bf16-p75d16c{STEP}-ho10x{clones}-100k-v{ver}" if arm == "d16" else f"b2-fp8-p75c893-ho10x{clones}-100k-v{ver}" if arm == "c893fp8" else f"b2-int8-p75{arm}-ho10x{clones}-100k-v{ver}"
OUT = Path.cwd() / run
shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT, ignore=shutil.ignore_patterns("__pycache__"))
PORT = '''        stop = EarlyStop(cfg, read_score=read, paths=stop_paths(self.output), log=self.gate_log)
        token_stop = int(study.get("token_stop") or 0)
        if token_stop:
            # ported from the qadv2-v87s read: an instance finishes at token_stop generated tokens (its slot goes to the
            # next game at once); the trial ends when every instance has finished.
            check = stop.check

            def check_tokens(elapsed):
                reason = check(elapsed)
                if reason is not None:
                    return reason
                for game in games:
                    if game.finished_reason is None and game.clock.completion_tokens() >= token_stop:
                        game.note_finished("token_stop")
                        self.gate_log({"event": "token_stop_game", "game": game.game_id,
                                       "tokens": int(game.clock.completion_tokens()), "elapsed": round(elapsed, 1)})
                if all(game.finished_reason is not None for game in games):
                    stop.reason = "token_stop"
                    self.gate_log({"event": "early_stop", "reason": "token_stop", "token_stop": token_stop,
                                   "elapsed": round(elapsed, 1)})
                    return stop.reason
                return None

            stop.check = check_tokens
        return stop
'''
t = OUT / "src/arena/trial.py"; s = t.read_text()
a = '        return EarlyStop(cfg, read_score=read, paths=stop_paths(self.output), log=self.gate_log)\n'
assert s.count(a) == 1; t.write_text(s.replace(a, PORT))
p = OUT / "configs/comparison.json"
d = json.loads(p.read_text().replace(SRC.name, run))
assert d["competition"]["rehearsal"] is True  # local arcade, never the gateway
if arm == "c204cut":  # 32 layers, 64 experts, top-k 6: arc3sub's kprofC-cut edits, on ckpt-204's overlay
    CUT = DIST / "b2-kprofC-cut-v3"
    for f in ("configs/expert-prune/expert-prune-t87-drop16.json", "src/arc_infra/expert_prune_patch.py"):
        (OUT / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CUT / f, OUT / f)
    edits = json.loads((CUT.parents[1] / "docs/validation-20261009/v9/kprofC-cut-edits.json").read_text())
    d["research_checkpoint"].update(drop_layers=edits["research_checkpoint"]["drop_layers"],
                                    text_overrides=edits["research_checkpoint"]["text_overrides"], slim_base=False)
    d["heldout"].update(mask=edits["heldout"]["mask"], heal_overlay=f"/assets/{model}")
    d["server"]["expert_prune"] = edits["server"]["expert_prune"]
if arm == "d16":  # 32 layers, 128 experts, the model's top-k: the cut's drop edit, the bench's mask, BF16 dense, capture list on
    CUT = DIST / "b2-kprofC-cut-v3"
    f = "src/arc_infra/expert_prune_patch.py"
    (OUT / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CUT / f, OUT / f)
    q = V9 / "expert-prune-t75-d16.json"
    (OUT / "configs/expert-prune").mkdir(parents=True, exist_ok=True)
    (OUT / "configs/expert-prune" / q.name).write_bytes(q.read_bytes())
    mask = json.loads(q.read_text())
    assert mask["drop_layers"] == D16
    d["research_checkpoint"].update(drop_layers=D16, slim_base=False)
    d["heldout"].update(mask=q.name, heal_overlay=f"/assets/{model}")
    d["server"]["expert_prune"] = {"mask": f"configs/expert-prune/{q.name}", "sha256": hashlib.sha256(q.read_bytes()).hexdigest()}
    assert d["server"]["dense_kernels"]["int8_weights"]
    d["server"]["dense_kernels"]["int8_weights"] = []
    d["server"].update(cudagraph_capture_sizes=SIZES, max_cudagraph_capture_size=SIZES[-1], max_num_seqs=88)
    d["generation_slots"] = 85
    d["active_limit"] = active; d["focus_oversubscription"]["focus_slots"] = active
if arm in ("c204cut", "d16"):  # v3 died on one empty thinking-off reply: retry that session thinking on, fail only if empty again
    c = OUT / "src/arena/cache_probe.py"; t = c.read_text()
    x = """                    if not reply['choices'][0]['message'].get('content'):
                        raise RuntimeError('Empty serving probe response')
"""
    y = """                    if not reply['choices'][0]['message'].get('content'):
                        await thinking_retry(client, base_url, body, record, index)
"""
    z = "async def run(server_dir, study, base_url):\n"
    h = """async def thinking_retry(client, base_url, body, record, index):
    \"\"\"An empty thinking-off reply: the same session thinking on, its text kept; empty again fails the boot.\"\"\"
    again = {**body, 'max_tokens': 64, 'chat_template_kwargs': {**body['chat_template_kwargs'], 'enable_thinking': True}}
    response = await _post(client, base_url.rstrip('/') + '/chat/completions', json=again)
    response.raise_for_status()
    message = response.json()['choices'][0]['message']
    text = message.get('content') or message.get('reasoning') or message.get('reasoning_content')
    record.setdefault('empty_direct', []).append({'session': index, 'thinking_reply': (text or '')[:300]})
    if not text:
        raise RuntimeError('Empty serving probe response, thinking on as well')


"""
    assert t.count(x) == 1 and t.count(z) == 1
    c.write_text(t.replace(x, y).replace(z, h + z))
for k in ("v86cand", "dn4g8"):  # notes the code never reads (check_play_bundle)
    d.pop(k, None)
d["token_stop"] = 115000  # the anchor's: gameplay passes 100K for >=95% of tries
d["wall_seconds"] = 14400; d["notebook_seconds"] = 18000
d["competition"].update(preview_instances=clones * len(d["games"]), preview_seconds=14400, submission_seconds=18000)
d["research_checkpoint"]["heal_overlay"] = f"/assets/{model}"
if arm in ("c893", "c893fp8"):
    d["heldout"]["heal_overlay"] = f"/assets/{model}"
if arm == "c893fp8":  # paired with the INT8 read: only attn_out, GDN in_proj, QSA q/k/v change; no fit (ARC_FP8_CALIB unset)
    k = d["server"]["dense_kernels"]
    assert sorted(k["int8_weights"]) == sorted(["lm_head", "gdn_in_proj_qkvz", "qsa_qkv", "attn_out"])
    k.update(int8_weights=["lm_head"], fp8_w8a8=["attn_out", "gdn_in_proj_qkvz", "qsa_qkv"])
    assert not any(x.startswith("ARC_FP8_C") or x == "ARC_FP8_BF16" for x in d["trials"][0].get("server_env", {}))
    d["active_limit"] = 53; d["focus_oversubscription"]["focus_slots"] = 53  # the INT8 read's streams; more blocks do not raise them
d["nebius_assets"] = [x for x in d["nebius_assets"] if "arc-heal" not in x] + [model]
d["roster"] = f"held-out read {arm} (trained format): 10 held-out games x {clones}, 100K tokens per try, v9b tree"
d["evidence"]["int8_read"] = "from b2-v9b-p75c613-int8-check-v1: overlay, instances, token_stop 115K (trial.py port), wall"
p.write_text(json.dumps(d, indent=2) + "\n")
m = json.loads((OUT / "MANIFEST.json").read_text().replace(SRC.name, run))
m["files"] = {q.relative_to(OUT).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(OUT.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(OUT, run, d["active_limit"], d["competition"])
