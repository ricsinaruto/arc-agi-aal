"""24 h SFT-data run on the v9 v3 stack: the v3 tree (pre-kf: dist/<V3>-sub) with
- roster: all 25 public games (the 10 held-out first, then the 15 fit games) x 5 cyclic clones = 125 instances;
- no caps: priority.park_tokens 300,000 -> 0 (level_cap_tokens 0, hx probe off, tail_fade off already),
  max_actions_per_game 100,000 -> 10,000,000; envelope 32,400 -> ENVELOPE (the Nebius launcher caps its timeout at
  86,400 s from launch: boot + envelope + finalize fit inside it);
- sft_record: v84-sub src/arena/sft_record.py + anchored hooks in trial.py and duck.py (request_completion,
  step, run); the server returns token ids (return_token_ids) on recorded requests.
Then nebius_kaggle_faithful.py --live-score. Usage: mkV9sft.py V3_NAME NAME [--attempt N]"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(".")
V84 = REPO / ".claude/worktrees/v84sub"
D = V84 / "dist"
PY = str(REPO / ".venv/bin/python")
ENVELOPE = 82800
CLONES = 5

ap = argparse.ArgumentParser()
ap.add_argument("v3")
ap.add_argument("name")
ap.add_argument("--attempt", type=int, default=1)
ap.add_argument("--cpu-gib", type=int, default=None, help="override the v3 CPU KV tier (48)")
a = ap.parse_args()

src, sub, out = D / f"{a.v3}-sub", D / f"{a.name}-sub", D / a.name
for p in (sub, out):
    if p.exists():
        shutil.rmtree(p)
shutil.copytree(src, sub, ignore=shutil.ignore_patterns("__pycache__"))
changed = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def edit(rel, pairs):
    p = sub / rel
    text = p.read_text()
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit(f"{rel}: anchor not found once: {old[:80]!r}")
        text = text.replace(old, new)
    p.write_text(text)
    changed.append(rel)


# --- code ---
shutil.copyfile(V84 / "src/arena/sft_record.py", sub / "src/arena/sft_record.py")
changed.append("src/arena/sft_record.py")
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

# --- study ---
p = sub / "configs/comparison.json"
s = json.loads(p.read_text())
public = json.loads((sub / "configs/duck_public_notebook.json").read_text())["public_game_ids"]
held = s["games"]
assert len(held) == 10 and len(s["games_excluded"]) == 15 and set(held + s["games_excluded"]) == set(public)
s["games"] = held + [g for g in public if g not in held]
s["games_excluded"] = []
comp = s["competition"]
assert comp["preview_instances"] == 110 and comp["submission_seconds"] == 32400
comp["preview_instances"] = len(s["games"]) * CLONES
comp["submission_seconds"] = ENVELOPE
assert comp["max_actions_per_game"] == 100000
comp["max_actions_per_game"] = 10_000_000
pr = s["priority"]
assert pr["equal"] is True and pr["park_tokens"] == 300000 and pr["level_cap_tokens"] == 0 and pr["tail_fade"] is False
pr["park_tokens"] = 0
if a.cpu_gib is not None:  # attempt 2 (30 Oct): host RAM headroom for 24 h (v1 anon growth from ~4.5 h)
    assert s["cache_experiment"]["cpu_cache_gib"] == 48
    s["cache_experiment"]["cpu_cache_gib"] = a.cpu_gib
assert s["hx_harness"]["probe"] is False and s["level_budget_policy"] == "priority"
s["roster"] = f"all 25 public games x {CLONES} cyclic clones (10 held-out first, then the 15 fit games); SFT data run"
s["sft_record"] = {"attempt": a.attempt, "module": "arena.sft_record", "format": "arena.sft_record v1",
                   "agreed": "SFT arm a16ace967d8633a89, 29 Oct", "return_token_ids": True}
s["evidence"]["sft_run"] = {"order": "24 h Nebius run for SFT data, no token caps, every instance plays "
                         "until finished; the wall is a crash ceiling",
                "caps_off": ["priority.park_tokens 300000 -> 0", "competition.max_actions_per_game -> 10M"],
                "kept": "per-reply max_tokens (completion_cap 16,384: the harness reply cap the KV gate reserves); "
                        "drains/compactions (context management)",
                "scheduler": "priority.equal: fewest completion tokens first, handover at every drain/compaction; "
                             "unstarted instances first"}
p.write_text(json.dumps(s, indent=2) + "\n")
changed.append("configs/comparison.json")
m = json.loads((sub / "MANIFEST.json").read_text())
for c in changed:
    m["files"][c] = sha(sub / c)
(sub / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
subprocess.run([PY, str(V84 / "docs/validation-20261009/nebius_kaggle_faithful.py"), str(sub), str(out), "--live-score"],
               check=True)
print("built", out, "changed", changed)
