"""V9C 24 h data runs. From the c893 INT3 read's tree (c893 experts INT3 dn4g8 refit on the 15, INT8 dense,
own MTP, qsa_fast + gdn_defer, drain jitter, no CPU tier) with: roster train (15 x 10) or held (10 x 11); density streams min(60,
floor(capacity / 47.9)); level_end_tokens 300,000: a level at the budget ENDS the game (generated gameplay tokens, game.tokens_on_level;
park_tokens 0, level_cap_tokens 0); sft_record as the v9sft / v9bsft runs; carry_live (prefix hit in live_progress, active_limit
knob, level_cap_requests off here); max_actions_per_game 10M; envelope 82,800 s. The trial ends when every entry has ended.
Usage: mkV9c.py train|held"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

V84 = Path(__file__).resolve().parents[3]
D = V84 / "dist"
SRC = D / "b2-int3dn4g8-p75c893-ho10x6-100k-v1"
REF = D / "b2-v9sft-p75c268-kf-all25x5-24h-v2"
ENVELOPE, LEVEL_END = 82800, 300000
side = sys.argv[1]
CLONES = {"train": 10, "held": 11}[side]
GAME_END = 500_000
name = {"train": "b2-v9c-p75c893-kf-tr15x10-24h-v2", "held": "b2-v9c-p75c893-kf-ho10x11-24h-v2"}[side]
out = D / name
shutil.rmtree(out, ignore_errors=True)
shutil.copytree(SRC, out, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "dataset-upload"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def edit(rel, pairs):
    p = out / rel
    text = p.read_text()
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit(f"{rel}: anchor not found once: {old[:80]!r}")
        text = text.replace(old, new)
    p.write_text(text)


# --- SFT recording: verbatim the v9sft / v9bsft runs' ---
shutil.copyfile(V84 / "src/arena/sft_record.py", out / "src/arena/sft_record.py")
assert sha(REF / "src/arena/sft_record.py") == sha(out / "src/arena/sft_record.py"), "sft_record.py differs from the 24 h run's"
edit("src/arena/trial.py", [(
    '            agent.trim = trim_settings(self.study.get("trim"))\n',
    '            agent.trim = trim_settings(self.study.get("trim"))\n'
    '            if self.study.get("sft_record"):  # SFT data run (arena.sft_record)\n'
    '                from .sft_record import SftRecorder\n'
    '                agent.sft_recorder = SftRecorder(self.study, self.output, game.game_id, game.index,\n'
    '                                                 server_dir=self.server_dir)\n')])
edit("src/arena/duck.py", [
    ('                if purpose == "gameplay" and self.sampler is not None and self.sampler.eligible():\n'
     '                    # parallel_samples: k samples of this request, the first with a tool call wins (one reply).\n'
     '                    return await self.sampler.complete(payload, super().request_completion)\n'
     '                return await self.with_server_retries(super().request_completion, payload)\n',
     '                if self.sft_recorder is not None:\n'
     '                    payload = {**payload, "return_token_ids": True}  # arena.sft_record: exact prompt/reply ids\n'
     '                if purpose == "gameplay" and self.sampler is not None and self.sampler.eligible():\n'
     '                    # parallel_samples: k samples of this request, the first with a tool call wins (one reply).\n'
     '                    response = await self.sampler.complete(payload, super().request_completion)\n'
     '                else:\n'
     '                    response = await self.with_server_retries(super().request_completion, payload)\n'
     '                if self.sft_recorder is not None:\n'
     '                    self.sft_recorder.request(self, purpose, payload, response)\n'
     '                return response\n'),
    ('    server_error_retries = 0\n',
     '    server_error_retries = 0\n'
     '    sft_recorder = None  # arena.sft_record (SFT data run)\n\n'
     '    async def run(self):\n'
     '        try:\n'
     '            return await super().run()\n'
     '        finally:\n'
     '            if self.sft_recorder is not None:\n'
     '                self.sft_recorder.finish(self)\n'),
    ('        if self.notes is not None and self.obs.levels > before:\n'
     '            self.notes_due = "level_clear"\n',
     '        if self.notes is not None and self.obs.levels > before:\n'
     '            self.notes_due = "level_clear"\n'
     '        if self.sft_recorder is not None:\n'
     '            self.sft_recorder.action(self, action)\n'),
])
# --- the level budget ends the game, on the token_stop path (ran 60 tries in the read) ---
edit("src/arena/trial.py", [
    ('        token_stop = int(study.get("token_stop") or 0)\n        if token_stop:\n',
     '        token_stop = int(study.get("token_stop") or 0)\n'
     '        level_end = int(study.get("level_end_tokens") or 0)\n'
     '        if token_stop or level_end:\n'),
    ('                    if game.finished_reason is None and game.clock.completion_tokens() >= token_stop:\n',
     '                    if level_end and game.finished_reason is None and game.tokens_on_level() >= level_end:\n'
     '                        game.note_finished("level_budget")\n'
     '                        self.gate_log({"event": "level_budget_end", "game": game.game_id, "levels": game.levels,\n'
     '                                       "tokens": int(game.clock.completion_tokens()), "elapsed": round(elapsed, 1)})\n'
     '                    if token_stop and game.finished_reason is None and game.clock.completion_tokens() >= token_stop:\n'),
    ('                    stop.reason = "token_stop"\n'
     '                    self.gate_log({"event": "early_stop", "reason": "token_stop", "token_stop": token_stop,\n',
     '                    stop.reason = "token_stop" if token_stop else "all_ended"\n'
     '                    self.gate_log({"event": "early_stop", "reason": stop.reason, "token_stop": token_stop,\n'),
])
subprocess.run([sys.executable, str(Path(__file__).parent / "carry_live.py"), str(out)], check=True)
# --- study ---
p = out / "configs/comparison.json"
s = json.loads(p.read_text().replace(SRC.name, name))
held, train = list(s["games"]), list(s["games_excluded"])
assert len(held) == 10 and len(train) == 15 and not set(held) & set(train)
assert sorted(g.split("-")[0] for g in train) == sorted(s["heldout"]["fit_on"])
s["games"], s["games_excluded"] = (train, held) if side == "train" else (held, train)
comp = s["competition"]
assert comp["rehearsal"] is True and comp["preview_live_score"] is True
comp["preview_instances"] = CLONES * len(s["games"])
comp["submission_seconds"], comp["preview_seconds"] = ENVELOPE, ENVELOPE - 600
comp["max_actions_per_game"] = 10_000_000
s["wall_seconds"], s["notebook_seconds"] = 1800, ENVELOPE + 1200
s["b200"]["stop"]["wall_seconds"] = ENVELOPE + 600
s["token_stop"] = GAME_END  # 500K generated tokens on a game ends it (v2)
s["level_end_tokens"] = LEVEL_END
s["active_limit"] = 60
fo = s["focus_oversubscription"]
fo["focus_slots"] = 60
assert fo["gate_blocks_per_stream"] == 47.9 and s["cache_experiment"]["cpu_cache_gib"] == 0
pr = s["priority"]
assert pr["equal"] is True and pr["tail_fade"] is False
pr.update(park_tokens=0, level_cap_tokens=0, level_cap_requests=0)
s["sft_record"] = {"attempt": 1, "module": "arena.sft_record", "format": "arena.sft_record v1",
                   "agreed": "SFT arm a16ace967d8633a89, 29 Oct", "return_token_ids": True}
s["name"] = name
s["roster"] = f"V9C data run: {len(s['games'])} {'train' if side == 'train' else 'held-out'} games x {CLONES} = {comp['preview_instances']}"
s["evidence"]["v9c"] = {"order": "two 24 h V9C data runs, disjoint rosters, a level at 300K generated tokens ends the game",
                        "from": SRC.name, "run_out": "the trial ends when every entry has ended; no second pass"}
p.write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((out / "MANIFEST.json").read_text().replace(SRC.name, name))
m["study"] = name
m["files"] = {q.relative_to(out).as_posix(): sha(q) for q in sorted(out.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts
              and q.name not in ("MANIFEST.json", "build-manifest.json", "kernel-metadata.json")}
(out / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(out, comp["preview_instances"], s["games"][:2])
