"""Offline harness evolution (ideas.md point 6; paper-notes.md §2.2): the local model edits the
Duck harness directory between arena rounds, gated by the arena score.

One Modal function runs the whole loop against the bundle's server:

1. Round 0 evaluates the incumbent harness (empty at the start) on the dev roster: one 30-minute
   arena trial (``arena.run``, the fixed-2x scheduler, every Duck game loads the harness).
2. Each round: the proposer (the served model, a tool-calling session over the last round's raw
   traces and the harness) makes one focused edit set to a copy of the incumbent and explains it
   in ``CHANGELOG.md``; the copy is validated (caps, skills compile and import) and smoke-tested
   (a few dev games, a short wall); then evaluated on the dev roster. The candidate replaces the
   incumbent when its score at the round wall beats the incumbent's (ties broken by clears);
   otherwise it is archived and the incumbent stays. Score at the wall is the primary metric by
   the user's decision (25 September); clears, games with a clear and level-1 clear times are
   recorded as co-metrics.
3. After the rounds: the incumbent and the empty baseline are evaluated on the held-out roster.

Frozen by construction: the scheduler, scorer, level detection, action counting, the clocks and
the agent code; the proposer can only write ``prompt.md``, ``playbook.md``, ``skills/*.py`` and
``CHANGELOG.md`` under the candidate directory.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

from arc_infra.model_templates import request_messages, template_kwargs
from inference.utils.openai_compat import build_chat_payload

from .harness import DEFAULT_CAPS, DuckHarness

DEFAULTS = {
    "rounds": 5,
    "round_wall_seconds": 1800,
    "eval_grace_seconds": 480,
    "smoke": {"games": 3, "wall_seconds": 150},
    "proposer": {"tool_turns": 40, "seconds": 1200, "cap_tokens": 8192, "reasoning_effort": "xhigh",
                 "result_chars": 8000, "context_chars": 300000, "temperature": 0.7},
    "holdout": True,
}
EDITABLE = ("prompt.md", "playbook.md", "CHANGELOG.md")
ACCEPTED_STATUS = ("complete", "complete_with_preemptions", "complete_with_errors", "error")
SKILL_NAME = re.compile(r"^skills/([A-Za-z_][A-Za-z0-9_]*)\.py$")

SYSTEM_PROMPT = """You are improving the harness of an agent that plays ARC-AGI-3 games (64x64 colour grids, \
unknown mechanics, levels to clear with as few actions as possible). The agent is a fixed program \
driven by the same model you are; between rounds you may edit only its harness directory:

- harness/prompt.md: text appended to the agent's system prompt (cap {prompt_cap} tokens).
- harness/playbook.md: notes carried as the first user message of every request (cap {playbook_cap} \
tokens): general mechanics patterns, procedures, failure modes to avoid. Never facts about one game.
- harness/skills/<name>.py: files of plain Python functions pre-loaded into the agent's persistent \
interpreter as top-level definitions (cap {skill_cap} chars each, {max_skills} files). The agent's \
system prompt lists each file's first docstring line and its function signatures; the agent calls the \
functions directly. The interpreter is a sandbox: no class statements, no exec/eval/setattr, no numpy; \
imports only from bisect, collections, copy, fractions, functools, heapq, itertools, json, math, \
operator, random, re, statistics, string. Boards are the agent's `current_frame.ascii` strings (rows of \
letter-coded cells) and `current_frame.segmentation`. Test every skill with the `check` tool. Good skills \
are small and general: board diffs, object signatures, click-target enumeration, move-effect tables, \
hypothesis bookkeeping.
- harness/CHANGELOG.md: append one entry per round: Change / Evidence (which traces, what happened) \
/ Expected effect / How the next round can tell.

Rules: make ONE focused change per round (the evaluation is noisy; one cause per round); base it on \
evidence in the traces (read them: the transcripts show the agent's reasoning, tool calls and the \
receipts it got back); prefer removing waste (repeated probes, ignored receipts, long silent \
loops, misread HUDs) over adding advice; keep everything general; never mention game ids, \
coordinates or colours of a specific game; keep the caps. When you are done, call `finish` with \
a one-paragraph summary. You have {tool_turns} tool calls and {seconds} seconds."""


def trace_index(eval_dir):
    """One line per game: what the last evaluation left to read."""
    lines = []
    games = eval_dir / "duck-transfer" / "games"
    for folder in sorted(games.iterdir()) if games.is_dir() else []:
        files = {p.name: p.stat().st_size for p in folder.iterdir() if p.is_file()}
        lines.append(f"traces/{folder.name}/: " + ", ".join(f"{n} ({s:,} B)" for n, s in sorted(files.items())))
    return "\n".join(lines)


def round_table(report):
    rows = []
    for g in sorted(report["games"], key=lambda g: (-g["levels"], -g["score"])):
        t1 = next((round(t["wall_seconds"]) for t in g.get("time_to_level", []) if t["level"] == 1), None)
        rows.append(f"{g['game_id'][:4]}: levels {g['levels']}/{g.get('total_levels')}, score {g['score']:.1f}, "
                    f"actions {g['actions']}, model calls {g.get('model_calls')}, compactions {g.get('arena_compactions', 0)}, "
                    f"level-1 at {t1} s, status {g['status']}" + (f", error {g['error']}" if g.get("error") else ""))
    return "\n".join(rows)


def metrics(report):
    games = report["games"]
    t1 = [next((t["wall_seconds"] for t in g.get("time_to_level", []) if t["level"] == 1), None) for g in games]
    return {
        "score": report["mean_score_equal_weight_25_games"],
        "clears": report["levels"],
        "games_with_clears": sum(1 for g in games if g["levels"] > 0),
        "actions": report["actions"],
        "level1_times": sorted(round(t) for t in t1 if t is not None),
        "errors": sum(1 for g in games if g.get("error")),
        "status": report["status"],
    }


def better(candidate, incumbent):
    if candidate["score"] != incumbent["score"]:
        return candidate["score"] > incumbent["score"]
    return candidate["clears"] > incumbent["clears"]


class Proposer:
    """One tool-calling session of the served model over the traces and a candidate harness."""

    TOOLS = [
        {"type": "function", "function": {"name": "list_dir", "description": "List a directory under the workspace (traces/ or harness/).",
                                          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "read", "description": "Read lines of a workspace file (1-based start, at most 200 lines per call).",
                                          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"}, "lines": {"type": "integer"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "grep", "description": "Regular-expression search in one workspace file or directory (case-insensitive), up to 40 matching lines with line numbers.",
                                          "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern", "path"]}}},
        {"type": "function", "function": {"name": "write", "description": "Write a harness file in full: harness/prompt.md, harness/playbook.md, harness/CHANGELOG.md or harness/skills/<name>.py (empty content deletes a skill).",
                                          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
        {"type": "function", "function": {"name": "check", "description": "Validate the harness: caps, skills compile and import in a fresh interpreter, and `skills.names()`.",
                                          "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "finish", "description": "End the session with a one-paragraph summary of the change and its rationale.",
                                          "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
    ]

    def __init__(self, *, workspace, harness_dir, base_url, model, settings, python, log):
        self.workspace, self.harness_dir = Path(workspace), Path(harness_dir)
        self.base_url, self.model, self.settings, self.python, self.log = base_url.rstrip("/"), model, settings, python, log
        self.writes, self.turns, self.summary = [], 0, None
        self.transcript = (self.workspace / "transcript.jsonl").open("a")

    # --- tools ---------------------------------------------------------------------------

    def resolve(self, path):
        # Containment is checked on the normalised path (the workspace's traces/ and harness/ are
        # symlinks; following them is fine, escaping with ".." is not).
        base = os.path.normpath(str(self.workspace))
        target = os.path.normpath(os.path.join(base, str(path)))
        if target != base and not target.startswith(base + os.sep):
            raise PermissionError("Paths must stay inside the workspace")
        return Path(target)

    def tool_list_dir(self, path="."):
        target = self.resolve(path)
        if not target.is_dir():
            return f"Not a directory: {path}"
        return "\n".join(f"{p.name}{'/' if p.is_dir() else ''}" for p in sorted(target.iterdir())) or "(empty)"

    def tool_read(self, path, start=1, lines=120):
        target = self.resolve(path)
        if not target.is_file():
            return f"No such file: {path}"
        start, lines = max(1, int(start or 1)), max(1, min(200, int(lines or 120)))
        text = target.read_text(errors="replace").splitlines()
        chunk = text[start - 1:start - 1 + lines]
        body = "\n".join(f"{start + i:6d}| {line[:400]}" for i, line in enumerate(chunk))
        return f"{path}: lines {start}-{start + len(chunk) - 1} of {len(text)}\n{body}"

    def tool_grep(self, pattern, path="."):
        target = self.resolve(path)
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            return f"Bad pattern: {exc}"
        files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())[:200]
        out = []
        for file in files:
            try:
                for number, line in enumerate(file.read_text(errors="replace").splitlines(), 1):
                    if regex.search(line):
                        out.append(f"{file.relative_to(self.workspace)}:{number}: {line[:300]}")
                        if len(out) >= 40:
                            return "\n".join(out) + "\n(40-match cap)"
            except OSError:
                continue
        return "\n".join(out) or "(no matches)"

    def tool_write(self, path, content=""):
        rel = str(path).replace("\\", "/")
        rel = rel[len("harness/"):] if rel.startswith("harness/") else rel
        match = SKILL_NAME.match(rel)
        if rel not in EDITABLE and not match:
            return "Refused: only harness/prompt.md, harness/playbook.md, harness/CHANGELOG.md and harness/skills/<name>.py are editable"
        target = self.harness_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if match and not content.strip():
            target.unlink(missing_ok=True)
            self.writes.append({"path": rel, "chars": 0, "deleted": True})
            return f"Deleted {rel}"
        target.write_text(content)
        self.writes.append({"path": rel, "chars": len(content)})
        return f"Wrote {rel} ({len(content)} chars)" + ("\n" + "\n".join(self.tool_check().splitlines()[:6]) if match else "")

    def tool_check(self):
        harness = DuckHarness(self.harness_dir)
        problems = harness.validate()
        probe = harness.skills_preamble() + "\nprint('skills:', skills_names)\n"
        try:
            run = subprocess.run([self.python, "-c", probe], capture_output=True, text=True, timeout=60)
            output = (run.stdout + run.stderr).strip()[-2000:]
        except subprocess.TimeoutExpired:
            output = "import probe timed out (60 s)"
        return "problems: " + (json.dumps(problems) if problems else "none") + "\n" + output

    def tool_finish(self, summary=""):
        self.summary = str(summary)[:4000]
        return "Session ended"

    def call(self, name, arguments):
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            return f"Unknown tool {name}"
        try:
            return str(handler(**arguments))
        except TypeError as exc:
            return f"Bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - reported to the model, never fatal
            return f"{type(exc).__name__}: {str(exc)[:500]}"

    # --- session ---------------------------------------------------------------------------

    def request(self, client, messages):
        cfg = self.settings
        payload = build_chat_payload(provider="vllm", model=self.model, messages=request_messages(self.model, messages),
                                     max_tokens=int(cfg["cap_tokens"]), temperature=float(cfg["temperature"]),
                                     top_p=0.95, top_k=20, thinking=True, tools=self.TOOLS, tool_choice="auto", seed=-1)
        payload["chat_template_kwargs"] = template_kwargs(self.model, {"enable_thinking": True, "reasoning_effort": cfg["reasoning_effort"]})
        started = time.monotonic()
        response = client.post(self.base_url + "/chat/completions", json=payload, timeout=900)
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        return message, data.get("usage", {}), time.monotonic() - started

    def trim(self, messages):
        budget = int(self.settings["context_chars"])
        total = sum(len(json.dumps(m)) for m in messages)
        index = 2
        while total > budget and index < len(messages) - 2:
            m = messages[index]
            if m.get("role") == "tool" and len(m.get("content", "")) > 80:
                total -= len(m["content"]) - 40
                m["content"] = "[earlier tool result dropped to fit the context]"
            index += 1
        return messages

    def run(self, brief):
        cfg = self.settings
        system = SYSTEM_PROMPT.format(prompt_cap=DEFAULT_CAPS["prompt_cap_tokens"], playbook_cap=DEFAULT_CAPS["playbook_cap_tokens"],
                                      skill_cap=DEFAULT_CAPS["skill_cap_chars"], max_skills=DEFAULT_CAPS["max_skills"],
                                      tool_turns=cfg["tool_turns"], seconds=cfg["seconds"])
        messages = [{"role": "system", "content": system}, {"role": "user", "content": brief}]
        started, failures, usage_total = time.monotonic(), 0, {"prompt_tokens": 0, "completion_tokens": 0}
        with httpx.Client() as client:
            while self.summary is None and self.turns < int(cfg["tool_turns"]) and time.monotonic() - started < float(cfg["seconds"]):
                try:
                    message, usage, seconds = self.request(client, self.trim(messages))
                except Exception as exc:  # noqa: BLE001 - one retry, then the session ends
                    failures += 1
                    self.log({"event": "proposer_request_failed", "error": f"{type(exc).__name__}: {str(exc)[:300]}", "failures": failures})
                    if failures >= 2:
                        break
                    continue
                for key in usage_total:
                    usage_total[key] += int(usage.get(key) or 0)
                calls = message.get("tool_calls") or []
                entry = {"role": "assistant", "content": message.get("content") or ""}
                if calls:
                    entry["tool_calls"] = calls
                messages.append(entry)
                self.transcript.write(json.dumps({"turn": self.turns, "seconds": round(seconds, 1), "usage": usage, "message": entry}) + "\n")
                if not calls:
                    # A reply without a tool call: ask once for the finish call, then stop.
                    if any(m.get("role") == "user" and m.get("content", "").startswith("Call `finish`") for m in messages[-3:]):
                        break
                    messages.append({"role": "user", "content": "Call `finish` with your summary when the change is complete, or continue with tool calls."})
                    continue
                for call in calls:
                    self.turns += 1
                    function = call.get("function", {})
                    try:
                        arguments = json.loads(function.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    result = self.call(function.get("name", ""), arguments if isinstance(arguments, dict) else {})
                    result = result[: int(cfg["result_chars"])] + ("\n(truncated)" if len(result) > int(cfg["result_chars"]) else "")
                    messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})
                    self.transcript.write(json.dumps({"turn": self.turns, "tool": function.get("name"), "arguments": arguments, "result": result[:2000]}) + "\n")
                    if self.summary is not None:
                        break
        self.transcript.close()
        return {"turns": self.turns, "seconds": round(time.monotonic() - started, 1), "writes": self.writes,
                "summary": self.summary, "usage": usage_total, "request_failures": failures,
                "ended": "finish" if self.summary is not None else "cap"}


class Evolution:
    def __init__(self, args):
        self.args = args
        self.bundle, self.working, self.output = Path(args.bundle), Path(args.working), Path(args.output)
        self.study = json.loads((self.bundle / "configs/comparison.json").read_text())
        raw = self.study.get("evolution") or {}
        self.cfg = {**DEFAULTS, **raw, "smoke": {**DEFAULTS["smoke"], **raw.get("smoke", {})},
                    "proposer": {**DEFAULTS["proposer"], **raw.get("proposer", {})}}
        if not self.cfg.get("dev_games") or not self.cfg.get("holdout_games"):
            raise ValueError("evolution.dev_games and evolution.holdout_games are required")
        self.root = self.working / "evolve"
        self.root.mkdir(parents=True, exist_ok=True)
        self.incumbent = self.root / "harness"
        self.incumbent.mkdir(exist_ok=True)
        (self.incumbent / "skills").mkdir(exist_ok=True)
        self.baseline = self.root / "baseline-harness"
        self.baseline.mkdir(exist_ok=True)
        self.events = (self.root / "events.jsonl").open("a")
        self.model = self.study.get("served_model") or self.served_model()
        self.rounds = []
        self.started = float(args.started_epoch)
        self.deadline = self.started + float(args.max_seconds)

    def served_model(self):
        from arc_infra.config import Config

        return Config.load(str(self.bundle / "configs/duck_controls.yaml")).model

    def log(self, event):
        entry = {"time": time.time(), "wall_seconds": round(time.time() - self.started, 1), **event}
        self.events.write(json.dumps(entry) + "\n")
        self.events.flush()
        print(json.dumps(entry), flush=True)

    def remaining(self):
        return self.deadline - time.time()

    # --- one arena evaluation ------------------------------------------------------------

    def evaluate(self, name, harness_dir, games, wall):
        folder = self.root / name
        folder.mkdir(parents=True, exist_ok=True)
        study = {**self.study, "games": list(games), "wall_seconds": float(wall), "duck_harness": str(harness_dir),
                 "trials": [{"name": "duck-transfer", "kind": "duck-transfer"}], "name": f"{self.study['name']}-{name}"}
        study_path = folder / "study.json"
        study_path.write_text(json.dumps(study, indent=2))
        envelope = wall + self.cfg["eval_grace_seconds"] + 120
        command = [self.args.python, "-m", "arena.run", "--source", str(self.bundle), "--study", str(study_path),
                   "--output", str(folder), "--trial", "duck-transfer", "--server-dir", str(self.working / "serving"),
                   "--analysis-python", self.args.python, "--base-url", self.args.base_url,
                   "--notebook-start-epoch", str(time.time()), "--notebook-seconds", str(envelope)]
        if self.args.games_dir:
            command += ["--games-dir", self.args.games_dir]
        if self.args.pi_dir:
            command += ["--pi-dir", self.args.pi_dir, "--node", self.args.node]
        self.log({"event": "eval_start", "name": name, "games": len(games), "wall": wall, "harness": str(harness_dir)})
        with (folder / "process.log").open("ab") as stream:
            try:
                completed = subprocess.run(command, check=False, timeout=envelope + 60, stdout=stream, stderr=subprocess.STDOUT)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        report_path = folder / "duck-transfer" / "arena.json"
        # arena.run exits 1 for any status other than "complete" (a game error, a recorded
        # preemption) after writing the full report; evolve v1 discarded such a finished round-0
        # evaluation. A written report is a result; its status and errors are recorded with it.
        if not report_path.is_file():
            self.log({"event": "eval_failed", "name": name, "exit": code})
            return None
        report = json.loads(report_path.read_text())
        if report.get("status") not in ACCEPTED_STATUS:
            self.log({"event": "eval_failed", "name": name, "exit": code, "status": report.get("status")})
            return None
        result = {**metrics(report), "exit": code, "folder": str(folder)}
        self.log({"event": "eval_done", "name": name, **{k: v for k, v in result.items() if k != "level1_times"}})
        return result

    # --- rounds ----------------------------------------------------------------------------

    def brief(self, round_index, incumbent_eval, incumbent_report):
        history = "\n".join(
            f"round {r['round']}: {r['decision']} (candidate {r['candidate']['score']:.3f} vs incumbent {r['incumbent']['score']:.3f}; "
            f"clears {r['candidate']['clears']} vs {r['incumbent']['clears']}) — {r['proposer'].get('summary') or 'no summary'}"
            for r in self.rounds if r.get("candidate")) or "(none yet)"
        harness = DuckHarness(self.incumbent)
        skills = "\n".join(f"- skills/{n}.py: {d or '(no docstring)'}" for n, _, d in harness.skills()) or "(no skills)"
        return (
            f"Round {round_index}. The incumbent harness scored {incumbent_eval['score']:.3f} on the {len(self.cfg['dev_games'])}-game dev roster "
            f"({incumbent_eval['clears']} clears in {incumbent_eval['games_with_clears']} games, {incumbent_eval['actions']} actions, "
            f"level-1 clears at {incumbent_eval['level1_times']} s; score = mean over games of the level-weighted action-efficiency score).\n\n"
            f"Per game in that evaluation:\n{round_table(incumbent_report)}\n\n"
            f"Earlier rounds:\n{history}\n\n"
            f"Current harness/prompt.md:\n{(self.incumbent / 'prompt.md').read_text() if (self.incumbent / 'prompt.md').is_file() else '(empty)'}\n\n"
            f"Current harness/playbook.md:\n{(self.incumbent / 'playbook.md').read_text() if (self.incumbent / 'playbook.md').is_file() else '(empty)'}\n\n"
            f"Current skills:\n{skills}\n\n"
            f"Current harness/CHANGELOG.md:\n{(self.incumbent / 'CHANGELOG.md').read_text() if (self.incumbent / 'CHANGELOG.md').is_file() else '(empty)'}\n\n"
            f"Traces of that evaluation (workspace paths; duck-transcript.txt is the model's own transcript with tool calls and receipts, "
            f"log.jsonl has one JSON event per line: model requests, python tool calls with results, observations, stuck notices, compactions):\n"
            f"{trace_index(Path(incumbent_eval['folder']))}\n\n"
            "Start by reading the transcripts of games that made progress and of games that did not, find the most costly recurring "
            "behaviour, and make one change that addresses it. Write the CHANGELOG entry, run `check`, then `finish`."
        )

    def propose(self, round_index, incumbent_eval):
        folder = self.root / f"round-{round_index:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        candidate = folder / "candidate"
        if candidate.exists():
            shutil.rmtree(candidate)
        shutil.copytree(self.incumbent, candidate)
        workspace = folder / "proposer"
        workspace.mkdir(exist_ok=True)
        traces = workspace / "traces"
        if not traces.exists():
            traces.symlink_to(Path(incumbent_eval["folder"]) / "duck-transfer" / "games", target_is_directory=True)
        link = workspace / "harness"
        if not link.exists():
            link.symlink_to(candidate, target_is_directory=True)
        report = json.loads((Path(incumbent_eval["folder"]) / "duck-transfer" / "arena.json").read_text())
        proposer = Proposer(workspace=workspace, harness_dir=candidate, base_url=self.args.base_url,
                            model=self.model, settings=self.cfg["proposer"],
                            python=self.args.python, log=self.log)
        self.log({"event": "propose_start", "round": round_index})
        outcome = proposer.run(self.brief(round_index, incumbent_eval, report))
        self.log({"event": "propose_done", "round": round_index, **{k: v for k, v in outcome.items() if k != "writes"}, "writes": len(outcome["writes"])})
        (folder / "proposer.json").write_text(json.dumps(outcome, indent=2))
        return candidate, outcome

    def run(self):
        try:
            return self._run()
        finally:
            self.save_evidence()

    def save_evidence(self):
        """Copy the evolution directory into the collected output (it lives in the container's
        working directory, which is not collected)."""
        target = self.output / "evolve"
        try:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(self.root, target, symlinks=True,
                            ignore=shutil.ignore_patterns("recordings", "*.png", "*.npy"))
        except OSError as exc:
            print(json.dumps({"event": "evidence_copy_failed", "error": str(exc)[:300]}), flush=True)

    def _run(self):
        cfg = self.cfg
        wall = float(cfg["round_wall_seconds"])
        per_eval = wall + cfg["eval_grace_seconds"] + 180
        holdout_reserve = (2 * per_eval) if cfg.get("holdout") else 0
        incumbent_eval = self.evaluate("round-00", self.incumbent, cfg["dev_games"], wall)
        if incumbent_eval is None:
            raise RuntimeError("The baseline evaluation failed; inspect evolve/round-00/process.log")
        baseline_eval = incumbent_eval
        for index in range(1, int(cfg["rounds"]) + 1):
            needed = float(cfg["proposer"]["seconds"]) + cfg["smoke"]["wall_seconds"] + 300 + per_eval + holdout_reserve
            if self.remaining() < needed:
                self.log({"event": "round_skipped", "round": index, "remaining": round(self.remaining()), "needed": round(needed)})
                break
            candidate, outcome = self.propose(index, incumbent_eval)
            entry = {"round": index, "proposer": outcome, "incumbent": incumbent_eval, "candidate": None, "decision": None}
            if not outcome["writes"]:
                entry["decision"] = "rejected: no edit"
                self.rounds.append(entry)
                self.log({"event": "round_done", **{k: v for k, v in entry.items() if k in ("round", "decision")}})
                continue
            problems = DuckHarness(candidate).validate()
            if problems:
                entry["decision"] = f"rejected: validation {problems}"
                self.rounds.append(entry)
                self.log({"event": "round_done", "round": index, "decision": entry["decision"]})
                continue
            smoke = self.evaluate(f"round-{index:02d}/smoke", candidate, cfg["dev_games"][: int(cfg["smoke"]["games"])],
                                  float(cfg["smoke"]["wall_seconds"]))
            if smoke is None or smoke["errors"]:
                entry["decision"] = f"rejected: smoke {'failed' if smoke is None else str(smoke['errors']) + ' game errors'}"
                self.rounds.append(entry)
                self.log({"event": "round_done", "round": index, "decision": entry["decision"]})
                continue
            candidate_eval = self.evaluate(f"round-{index:02d}/eval", candidate, cfg["dev_games"], wall)
            if candidate_eval is None:
                entry["decision"] = "rejected: evaluation failed"
                self.rounds.append(entry)
                self.log({"event": "round_done", "round": index, "decision": entry["decision"]})
                continue
            entry["candidate"] = candidate_eval
            if better(candidate_eval, incumbent_eval):
                entry["decision"] = "kept"
                shutil.rmtree(self.incumbent)
                shutil.copytree(candidate, self.incumbent)
                incumbent_eval = candidate_eval
            else:
                entry["decision"] = "reverted"
            self.rounds.append(entry)
            self.log({"event": "round_done", "round": index, "decision": entry["decision"],
                      "candidate_score": candidate_eval["score"], "incumbent_score": incumbent_eval["score"]})
            (self.root / "rounds.json").write_text(json.dumps(self.rounds, indent=2))
        holdout = None
        if cfg.get("holdout") and self.remaining() >= per_eval + 60:
            evolved = self.evaluate("holdout-evolved", self.incumbent, cfg["holdout_games"], wall)
            base = None
            if self.remaining() >= per_eval + 60 and any(r.get("decision") == "kept" for r in self.rounds):
                base = self.evaluate("holdout-baseline", self.baseline, cfg["holdout_games"], wall)
            holdout = {"evolved": evolved, "baseline": base}
        best = self.output / "evolved-harness"
        if best.exists():
            shutil.rmtree(best)
        shutil.copytree(self.incumbent, best)
        summary = {"config": {k: v for k, v in cfg.items()}, "baseline": baseline_eval, "final": incumbent_eval,
                   "rounds": self.rounds, "holdout": holdout, "kept_rounds": [r["round"] for r in self.rounds if r.get("decision") == "kept"],
                   "harness": DuckHarness(self.incumbent).summary(), "seconds": round(time.time() - self.started, 1)}
        (self.root / "evolution.json").write_text(json.dumps(summary, indent=2))
        (self.output / "evolution.json").write_text(json.dumps(summary, indent=2))
        self.log({"event": "evolution_done", "kept_rounds": summary["kept_rounds"], "final_score": incumbent_eval["score"],
                  "baseline_score": baseline_eval["score"]})
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--working", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--games-dir")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pi-dir")
    parser.add_argument("--node", default="node")
    parser.add_argument("--started-epoch", type=float, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Evolution(args).run()


if __name__ == "__main__":
    main()
