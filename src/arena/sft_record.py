"""Per-request SFT records of a Duck arena game (the 24 h v9 data run; format agreed with the SFT arm).

On when the study has ``sft_record``: the trial gives every Duck agent a ``SftRecorder``. The arena's
``request_completion`` asks the server for token ids (``return_token_ids``) and hands each served payload and its
response here; ``step`` hands every environment action. Files, in ``games/<instance>/sft/``:

- ``turns-a<attempt>-<part>.jsonl``: one JSON object per line; a new part at ``PART_BYTES`` (closed parts never change,
  so the runner's 5-minute bucket sync copies each once). Every part starts with a ``meta`` record.
- ``outcome.json``: the instance's outcome, rewritten at most every ``OUTCOME_SECONDS`` and at the end.

Records:
- ``meta``: run, instance, source game, clone, attempt, model, server argv, chat template path + sha256, tokenizer
  file sha256s, the tools (full) and their sha256.
- ``req`` (one per served request, any purpose): ``i``, ``t``, ``purpose``, the game state when sent (``step``,
  ``levels``, ``state``, ``board`` rows as hex strings, else ``board_ref`` = the ``i`` that carried it), ``messages``
  {keep, append, n, sha256}: the full list is the previous ``req``'s list[:keep] + append (keep 0 on the first
  request and every ``SNAPSHOT_EVERY`` requests), sha256 over the canonical JSON of the full list
  (``json.dumps(list, sort_keys=True, separators=(",", ":"), ensure_ascii=False)``); ``prompt_ids`` the same
  delta scheme over the served prompt token ids; ``tools_sha`` (+ ``tools`` when they change); ``params`` (the
  payload less messages and tools); ``response``: choices[0] less token_ids, ``token_ids`` (the generated ids,
  special tokens kept), ``usage``, ``status``, ``parse_error`` (tool-call markup left unparsed); ``actions_before``:
  the environment actions since the previous request.
- ``end``: status, levels, state, won, actions, error, and the actions after the last request.
"""

import hashlib
import json
import time
from array import array
from pathlib import Path

PART_BYTES = 16 << 20
OUTCOME_SECONDS = 120
SNAPSHOT_EVERY = 200
HEX = "0123456789abcdef"
_SERVER_INFO = {}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(text):
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def board_rows(board):
    if board is None:
        return None
    return ["".join(HEX[int(v) & 15] for v in row) for row in board.tolist()]


def common_prefix(a, b):
    """Length of the common prefix of two sequences that support slicing and == (lists, arrays)."""
    lo, hi = 0, min(len(a), len(b))
    if a[:hi] == b[:hi]:
        return hi
    while lo < hi:  # a[:lo] == b[:lo] holds; a[:hi] != b[:hi]
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def server_info(server_dir):
    """argv, chat template and tokenizer hashes of the running server (once per process)."""
    key = str(server_dir)
    if key in _SERVER_INFO:
        return _SERVER_INFO[key]
    info = {}
    try:
        identity = json.loads((Path(server_dir) / "vllm-server-identity.json").read_text())
        argv = identity.get("argv") or []
        info["argv"] = argv
        if "--chat-template" in argv[:-1]:
            path = Path(argv[argv.index("--chat-template") + 1])
            info["chat_template"] = {"path": str(path), "sha256": sha256(path.read_bytes()) if path.is_file() else None}
        model = argv[argv.index("serve") + 1] if "serve" in argv[:-1] else None
        if model and Path(model).is_dir():
            info["model_dir"] = model
            info["model_files"] = {n: sha256((Path(model) / n).read_bytes()) for n in (
                "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "config.json")
                if (Path(model) / n).is_file()}
    except (OSError, ValueError) as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"[:300]
    _SERVER_INFO[key] = info
    return info


class SftRecorder:
    def __init__(self, study, output, game_id, index, server_dir=None):
        cfg = study.get("sft_record") or {}
        self.root = Path(output) / "games" / game_id / "sft"
        self.root.mkdir(parents=True, exist_ok=True)
        self.output, self.server_dir = Path(output), server_dir
        self.attempt = int(cfg.get("attempt", 1))
        self.run, self.game_id, self.index = study.get("name"), game_id, index
        self.games = list(study.get("games") or [])
        self.part, self.file, self.size = 0, None, 0
        self.i = 0
        self.message_digests, self.prompt_ids = [], array("I")
        self.tools_sha = self.board_sha = None
        self.board_i = None
        self.pending = []
        self.outcome_at = 0.0
        self.errors = 0
        self.meta = None
        self.closed = False

    # --- files ---------------------------------------------------------------------------------------
    def source(self):
        try:
            rows = json.loads((self.output / "public-roster.json").read_text())
            by_id = {r["game_id"]: r["source_game_id"] for r in rows}
            if self.game_id in by_id:
                return by_id[self.game_id], [r["game_id"] for r in rows].index(self.game_id)
        except (OSError, ValueError, KeyError):
            pass
        return (self.games[self.index % len(self.games)] if self.games else None), self.index

    def write(self, record):
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        if self.file is None or self.size + len(line) > PART_BYTES and self.size:
            if self.file is not None:
                self.file.close()
            self.part += 1
            self.file = open(self.root / f"turns-a{self.attempt}-{self.part:04d}.jsonl", "a", encoding="utf-8")
            self.size = 0
            if self.meta is not None and record.get("kind") != "meta":
                head = json.dumps(self.meta, separators=(",", ":"), ensure_ascii=False) + "\n"
                self.file.write(head)
                self.size += len(head)
        self.file.write(line)
        self.file.flush()
        self.size += len(line)

    def ensure_meta(self, agent, tools):
        if self.meta is not None:
            return
        source, position = self.source()
        n = len(self.games) or 1
        self.meta = {"kind": "meta", "run": self.run, "instance": self.game_id, "source_game_id": source,
                     "clone": position // n, "roster_position": position, "attempt": self.attempt,
                     "model": getattr(agent.config, "model", None),
                     "server": server_info(self.server_dir) if self.server_dir else None,
                     "tools": tools, "tools_sha": sha256(canonical(tools)) if tools is not None else None,
                     "format": "arena.sft_record v1"}
        self.tools_sha = self.meta["tools_sha"]
        self.write(self.meta)

    def outcome(self, agent, final=False):
        obs = getattr(agent, "obs", None)
        row = {"run": self.run, "instance": self.game_id, "source_game_id": self.meta and self.meta["source_game_id"],
               "clone": self.meta and self.meta["clone"], "attempt": self.attempt, "final": final,
               "requests": self.i, "actions": getattr(agent, "steps", None),
               "levels": None if obs is None else int(obs.levels),
               "state": None if obs is None else str(obs.state), "won": obs is not None and obs.state == "WIN",
               "status": getattr(agent, "status", None), "error": getattr(agent, "error", None),
               "recorder_errors": self.errors, "t": round(time.time(), 3)}
        tmp = self.root / "outcome.json.tmp"
        tmp.write_text(json.dumps(row, indent=1))
        tmp.replace(self.root / "outcome.json")
        self.outcome_at = time.monotonic()
        return row

    # --- hooks -----------------------------------------------------------------------------------------
    def action(self, agent, action):
        obs = agent.obs
        self.pending.append({
            "step": agent.steps, "action": action.name,
            **({"x": action.x, "y": action.y} if action.name == "ACTION6" else {}),
            "levels": int(obs.levels), "state": str(obs.state),
            "board_sha1": hashlib.sha1(obs.board.tobytes()).hexdigest() if obs.board is not None else None})

    def request(self, agent, purpose, payload, response):
        try:
            self._request(agent, purpose, payload, response)
        except Exception as exc:  # noqa: BLE001 - a recorder failure must never end the game
            self.errors += 1
            try:
                self.write({"kind": "recorder_error", "i": self.i, "error": f"{type(exc).__name__}: {exc}"[:500]})
            except Exception:  # noqa: BLE001
                pass

    def _request(self, agent, purpose, payload, response):
        tools = payload.get("tools")
        self.ensure_meta(agent, tools)
        self.i += 1
        messages = payload.get("messages") or []
        texts = [canonical(m) for m in messages]
        digests = [hashlib.sha1(t.encode()).digest() for t in texts]
        keep = 0 if (self.i == 1 or self.i % SNAPSHOT_EVERY == 0) else common_prefix(self.message_digests, digests)
        self.message_digests = digests
        record = {"kind": "req", "i": self.i, "t": round(time.time(), 3), "purpose": purpose}
        obs = agent.obs
        record.update(step=agent.steps, levels=int(obs.levels), state=str(obs.state))
        bsha = hashlib.sha1(obs.board.tobytes()).hexdigest() if obs.board is not None else None
        if bsha != self.board_sha or self.i % SNAPSHOT_EVERY == 0 or self.i == 1:
            record["board"], self.board_sha, self.board_i = board_rows(obs.board), bsha, self.i
        else:
            record["board_ref"] = self.board_i
        record["messages"] = {"keep": keep, "append": [json.loads(t) for t in texts[keep:]], "n": len(messages),
                              "sha256": sha256("[" + ",".join(texts) + "]")}
        tsha = sha256(canonical(tools)) if tools is not None else None
        record["tools_sha"] = tsha
        if tsha != self.tools_sha:
            record["tools"], self.tools_sha = tools, tsha
        record["params"] = {k: v for k, v in payload.items() if k not in ("messages", "tools")}
        record["actions_before"], self.pending = self.pending, []
        status = getattr(response, "status_code", None)
        resp = {"status": status}
        data = None
        if status == 200:
            data = response.json()
        elif response is not None:
            resp["text"] = response.text[:4000]
        if data is not None:
            ids = data.get("prompt_token_ids")
            if ids is not None:
                new = array("I", ids)
                pkeep = 0 if (self.i == 1 or self.i % SNAPSHOT_EVERY == 0) else common_prefix(self.prompt_ids, new)
                record["prompt_ids"] = {"keep": pkeep, "append": new[pkeep:].tolist(), "n": len(new),
                                        "sha256": sha256(new.tobytes())}
                self.prompt_ids = new
            choice = dict((data.get("choices") or [{}])[0])
            resp["token_ids"] = choice.pop("token_ids", None)
            resp["choice"] = choice
            resp["usage"] = data.get("usage")
            resp["id"] = data.get("id")
            message = choice.get("message") or {}
            content = message.get("content") or ""
            if not message.get("tool_calls") and ("<tool_call" in content or "<function" in content):
                resp["parse_error"] = "tool-call markup in content, no parsed tool_calls"
        record["response"] = resp
        self.write(record)
        if time.monotonic() - self.outcome_at >= OUTCOME_SECONDS:
            self.outcome(agent)

    def finish(self, agent):
        if self.closed:
            return
        self.closed = True
        try:
            if self.meta is None:
                self.ensure_meta(agent, None)
            obs = getattr(agent, "obs", None)
            self.write({"kind": "end", "t": round(time.time(), 3), "status": getattr(agent, "status", None),
                        "error": getattr(agent, "error", None), "actions": getattr(agent, "steps", None),
                        "levels": None if obs is None else int(obs.levels),
                        "state": None if obs is None else str(obs.state),
                        "won": obs is not None and obs.state == "WIN", "actions_after": self.pending})
            self.pending = []
            self.outcome(agent, final=True)
        finally:
            if self.file is not None:
                self.file.close()
                self.file = None


def rebuild(paths):
    """Reader (tests and the SFT arm): yield (meta, req, full messages, full prompt ids) per ``req`` record of one
    instance's parts in order, checking both sha256s."""
    messages, prompt_ids, meta = [], array("I"), None
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["kind"] == "meta":
                    meta = rec
                    continue
                if rec["kind"] != "req":
                    continue
                m = rec["messages"]
                messages = messages[: m["keep"]] + m["append"]
                if len(messages) != m["n"] or sha256(canonical(messages)) != m["sha256"]:
                    raise ValueError(f"{path}: messages of req {rec['i']} do not rebuild")
                p = rec.get("prompt_ids")
                if p is not None:
                    prompt_ids = prompt_ids[: p["keep"]] + array("I", p["append"])
                    if len(prompt_ids) != p["n"] or sha256(prompt_ids.tobytes()) != p["sha256"]:
                        raise ValueError(f"{path}: prompt ids of req {rec['i']} do not rebuild")
                yield meta, rec, messages, prompt_ids
