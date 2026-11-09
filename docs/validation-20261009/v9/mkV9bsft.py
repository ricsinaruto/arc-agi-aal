"""24 h SFT-data run on the v9b stack, mixed roster: the v9b kf tree as checked on Nebius
(dist/b2-v9b-p75c613-kf-ho10x11-check-v1: p75 ckpt-613 AR W4A16 + heal overlay, BF16 dense, own MTP, int4 KV,
qsa_fast + gdn_defer, drain jitter, no CPU tier, density streams) with
- roster: data/envs/roster24h-v2.json, 125 slots listed explicitly in study.games (the clone server plays slot i on
  games[i]): 25 public x1, 30 vetted non-public x3, 10 tier variants x1, in the roster's order;
- games: one bundled dir (study.generated_games.dir; bootstrap_arena prefers it to the platform's games) with the 65
  distinct games: public from bucket assets/games (the Nebius reads' --games-dir), non-public from envs' pool-games, variants
  from envs' tiers-b1; every game file's sha256 starts with its id's version hash, every file hashed in the study;
- mini-racer 0.14.1 (py_mini_racer, the PuzzleScript games' worker) wheel + hash in requirements.txt;
- the 300K per-level token cap kept (priority.park_tokens 300,000; data runs from now on), max_actions_per_game
  10M, envelope 82,800 s (as the current 24 h run);
- sft_record exactly as the current 24 h run (mkV9sft.py anchors: trial.py, duck.py; arena.sft_record).
Usage: mkV9bsft.py NAME [--attempt N]"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(".")
V84 = REPO / ".claude/worktrees/v84sub"
D = V84 / "dist"
SRC = D / "b2-v9b-p75c613-kf-ho10x11-check-v1"
ROSTER = REPO / "data/envs/roster24h-v2.json"
SCR = Path("<scratch>/v9b")
# public: the copy every Nebius kf read and the current 24 h run play (execution.json: --games-dir /assets/games; rclone
# copy of bucket assets/games; not the competition copy). Its files carry the API download's header,
# so sha256(game.py)[:8] != the version id there; the non-public files satisfy that rule and are checked by it.
ROOTS = {"public": SCR / "assetgames",
         "nonpublic": Path.home() / "arc-envs-data/mg/pool-games/environment_files",
         "tier_variant": Path.home() / "arc-envs-data/admit-tiers/tiers-b1/environment_files"}
WHEEL = SCR / "mrwheel/mini_racer-0.14.1-py3-none-manylinux_2_27_x86_64.whl"
WHEEL_SHA = "cdf3a088e1363f16a695288f882abf76b3705b8e1df21418208b87ed010037a4"  # PyPI's digest
ENVELOPE = 82800
GAMES_REL = "games/environment_files"

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--attempt", type=int, default=1)
a = ap.parse_args()
out = D / a.name
if out.exists():
    shutil.rmtree(out)
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


# --- SFT recording: verbatim the current 24 h run's (mkV9sft.py) ---
shutil.copyfile(V84 / "src/arena/sft_record.py", out / "src/arena/sft_record.py")
ref = D / "b2-v9sft-p75c268-kf-all25x5-24h-v2"
assert sha(ref / "src/arena/sft_record.py") == sha(out / "src/arena/sft_record.py"), "sft_record.py differs from the live run's"
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

# --- mini-racer wheel ---
assert sha(WHEEL) == WHEEL_SHA
shutil.copyfile(WHEEL, out / "wheels" / WHEEL.name)
req = (out / "requirements.txt").read_text()
assert "mini-racer" not in req
(out / "requirements.txt").write_text(req.rstrip("\n") + f"\nmini-racer==0.14.1 \\\n    --hash=sha256:{WHEEL_SHA}\n")

# --- games ---
roster = json.loads(ROSTER.read_text())
slots, files, distinct = [], {}, {}
for g in roster["games"]:
    gid, ver = g["game"].split("-")
    root = ROOTS["public" if g["source"] == "public" else "tier_variant" if g["source"] == "tier_variant" else "nonpublic"]
    src = root / gid / ver
    if g["game"] not in distinct:
        meta = json.loads((src / "metadata.json").read_text())
        assert meta["game_id"] == g["game"], (meta["game_id"], g["game"])
        py = [p for p in src.iterdir() if p.suffix == ".py"]
        assert len(py) == 1 and (g["source"] == "public" or sha(py[0]).startswith(ver)), (g["game"], [p.name for p in py])
        dst = out / GAMES_REL / gid / ver
        dst.mkdir(parents=True)
        for f in sorted(src.iterdir()):
            if f.is_file() and f.suffix in (".py", ".json"):
                shutil.copyfile(f, dst / f.name)
                files[f"{GAMES_REL}/{gid}/{ver}/{f.name}"] = sha(f)
        distinct[g["game"]] = g["source"]
    slots += [g["game"]] * int(g["clones"])
assert len(slots) == roster["slots"] == 125 and len(distinct) == 65, (len(slots), len(distinct))

# --- study ---
p = out / "configs/comparison.json"
s = json.loads(p.read_text())
s["games"] = slots
s["games_excluded"] = []
s["games_source"] = f"data/envs/roster24h-v2.json (sha256 {sha(ROSTER)})"
s["generated_games"] = {"dir": GAMES_REL, "roster": "data/envs/roster24h-v2.json", "roster_sha256": sha(ROSTER),
                        "distinct": distinct, "files": files,
                        "public_source": "bucket assets/games (= /assets/games of every Nebius kf read)",
                        "nonpublic_source": str(ROOTS["nonpublic"]), "tier_variant_source": str(ROOTS["tier_variant"])}
comp = s["competition"]
assert comp["rehearsal"] is True and comp["preview_live_score"] is True
comp["preview_instances"] = len(slots)
comp["submission_seconds"] = ENVELOPE
comp["preview_seconds"] = ENVELOPE - 600
assert comp["max_actions_per_game"] == 100000
comp["max_actions_per_game"] = 10_000_000
s["notebook_seconds"] = ENVELOPE + 1200
s["b200"]["stop"]["wall_seconds"] = ENVELOPE + 600
pr = s["priority"]
assert pr["equal"] is True and pr["park_tokens"] == 300000 and pr["level_cap_tokens"] == 0 and pr["tail_fade"] is False
assert s["cache_experiment"]["cpu_cache_gib"] == 0 and s["focus_oversubscription"]["gate_blocks_per_stream"] == 47.9
s["roster"] = ("mixed 125 slots (data/envs/roster24h-v2.json): 25 public x1, 30 vetted non-public x3 (9 PuzzleScript, "
               "9 Witness, 6 sonpham-interactive, 6 generated), 10 tier variants x1; SFT data run")
s["sft_record"] = {"attempt": a.attempt, "module": "arena.sft_record", "format": "arena.sft_record v1",
                   "agreed": "SFT arm a16ace967d8633a89, 29 Oct", "return_token_ids": True}
s["name"] = a.name
s.setdefault("evidence", {})["sft_run_v9b"] = {
    "order": "24 h v9b SFT-data run on the 125-slot mixed roster, 300K per-level token cap",
    "stack": "v9b as checked on Nebius (b2-v9b-p75c613-kf-ho10x11-check-v1)",
    "caps": ["priority.park_tokens 300000 (kept: data runs park at 300K tokens on a level)",
             "competition.max_actions_per_game -> 10M"],
    "kept": "per-reply max_tokens (completion_cap 16,384); drains/compactions"}
s["priority"]["level_cap_tokens"] = 300000  # the data runs' per-level cap, beside the request cap
s["evidence"]["sft_run_v9b"]["caps"].append("priority.level_cap_tokens 300000, level_cap_requests 1000, grant 100")
p.write_text(json.dumps(s, indent=2) + "\n")
subprocess.run([sys.executable, str(Path(__file__).parent / "carry_live.py"), str(out)], check=True)

m = json.loads((out / "MANIFEST.json").read_text())
m["study"] = a.name
m["files"] = {q.relative_to(out).as_posix(): sha(q) for q in sorted(out.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts
              and q.name not in ("MANIFEST.json", "build-manifest.json", "kernel-metadata.json")}
(out / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(out, len(slots), "slots", len(distinct), "games", len(files), "game files")
