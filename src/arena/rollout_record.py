"""Rollout record for RL (level bench setting ``rollout``): every live chat reply of an attempt is stored with the
request it answered, its sampled token ids and the engine's log-probabilities (``posttrain.rl_rollouts`` reads it).

The request gains ``logprobs``, ``top_logprobs`` and the token-id switches; nothing else changes and the reply the
harness sees is the engine's without those fields. A recording failure is counted and never ends the attempt.
"""

import hashlib
import json
from pathlib import Path

import httpx

DEFAULTS = {"top_logprobs": 20,   # alternatives stored per sampled token (0: the sampled token only)
            "token_ids": True}    # ask for the served prompt's token ids and store their digest


def token_id(key):
    """vLLM token keys: ``token_id:123`` (``return_tokens_as_token_ids``)."""
    return int(str(key).rsplit(":", 1)[-1])


def parse_logprobs(content, top=20):
    """``(tokens, logprobs, alternatives)`` of a chat reply's ``logprobs.content``; ``alternatives[i]`` is
    ``[ids, logprobs]`` of the reported top tokens of position ``i`` (highest first, at most ``top``; the sampled
    token is among them whenever the engine listed it)."""
    tokens, values, alternatives = [], [], []
    for entry in content or []:
        tokens.append(token_id(entry["token"]))
        values.append(round(float(entry["logprob"]), 5))
        if top:
            found = {token_id(t["token"]): float(t["logprob"]) for t in entry.get("top_logprobs") or []}
            order = sorted(found.items(), key=lambda item: (-item[1], item[0]))[:top]
            alternatives.append([[t for t, _ in order], [round(v, 5) for _, v in order]])
    return tokens, values, alternatives


def ids_digest(ids):
    """A digest of a token-id sequence (the served prompt)."""
    return hashlib.sha256(b"".join(int(t).to_bytes(4, "little") for t in ids)).hexdigest()[:32]


def delta(previous, messages):
    """``(keep, new)``: the leading messages shared with the previous request, and the rest."""
    keep = 0
    for old, new in zip(previous or [], messages):
        if old != new:
            break
        keep += 1
    return keep, messages[keep:]


def settings(raw):
    merged = {**DEFAULTS, **(raw or {})}
    if set(merged) - set(DEFAULTS) or type(merged["top_logprobs"]) is not int or not 0 <= merged["top_logprobs"] <= 20:
        raise ValueError("level_bench.rollout: {top_logprobs: 0..20, token_ids: bool}")
    return merged


class RolloutHTTP:
    """The attempt's HTTP session below the fork replay: replayed replies never reach it."""

    def __init__(self, inner, agent, cfg):
        self.inner, self.agent, self.cfg = inner, agent, settings(cfg)
        self.path = Path(agent.workspace.log).parent / "rollout.jsonl"
        self.previous, self.tools, self.level = None, None, 0
        self.stats = {"replies": 0, "gameplay_replies": 0, "tokens": 0, "count_mismatches": 0, "id_mismatches": 0,
                      "without_logprobs": 0, "refusals": 0, "errors": 0, "bytes": 0}
        self.restore, self.restore_at, self.restore_result = None, 0, None  # arena.restore: a scripted prefix

    def restore_post(self, url, payload):
        """A scripted request of a restored attempt: checked against the recorded one, the recorded row written as
        this attempt's own (marked ``scripted``: context for training, never trained), the recorded reply returned."""
        from .restore import compare, marks_of, response

        row = self.restore["rows"][self.restore_at]
        marks = marks_of(payload["messages"])
        keep, _ = delta(self.previous, marks)
        diffs = compare(row, payload, keep, payload["messages"][keep:])
        result = self.restore_result = self.restore_result or {"diffs": [], "requests": 0}
        result["diffs"] += [f"request {self.restore_at}: {d}" for d in diffs]
        result["requests"] += 1
        saved = {k: v for k, v in row.items() if k != "tools_now"}
        line = json.dumps({**saved, "scripted": True}, separators=(",", ":"), ensure_ascii=False) + "\n"
        with self.path.open("a") as file:
            file.write(line)
        self.previous, self.tools = marks, row.get("tools_now")
        self.stats["replies"] += 1
        self.restore_at += 1
        return httpx.Response(200, json=response(row), request=httpx.Request("POST", str(url)))

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def post(self, url, **kwargs):
        if not str(url).rstrip("/").endswith("chat/completions"):
            return await self.inner.post(url, **kwargs)
        payload = kwargs.get("json") or {}
        restore = self.restore
        if restore is not None and self.restore_at < restore["request"]:
            return self.restore_post(url, payload)
        live_check = restore is not None and self.restore_at == restore["request"]
        if live_check:  # the first live request: the recorded request's prompt, then the engine's own reply
            from .restore import compare, marks_of

            row = restore["rows"][self.restore_at]
            marks = marks_of(payload["messages"])
            keep, _ = delta(self.previous, marks)
            result = self.restore_result = self.restore_result or {"diffs": [], "requests": 0}
            result["diffs"] += [f"request {self.restore_at} (live): {d}" for d in compare(row, payload, keep,
                                                                                            payload["messages"][keep:])]
            self.restore_at += 1
            if restore.get("check_only"):
                payload = {**payload, "max_tokens": 1}
        if restore is not None and restore.get("hint"):  # OPD: a hinted try (after the restore check)
            from .restore import hint_messages

            payload = {**payload, "messages": hint_messages(payload["messages"], restore["hint"])}
        while True:
            response = await self.inner.post(url, **{**kwargs, "json": {**payload, **self.extras()}})
            refused = response.status_code == 400 and self.level < 2 and "context length" not in response.text
            if not refused:
                break
            # an engine that refuses the extra fields: ask for less (token ids first), once per attempt
            self.level += 1
            self.stats["refusals"] += 1
            self.agent.workspace.append("rollout_fields_refused", step=self.agent.steps, level=self.level,
                                        detail=response.text[:300])
        if response.status_code != 200:
            return response
        try:
            data = response.json()
            self.record(payload, data)
            if live_check:
                digest = ids_digest(data["prompt_token_ids"]) if data.get("prompt_token_ids") else None
                self.restore_result["prompt_digest"] = {"live": digest, "recorded": row.get("prompt_digest"),
                                                        "same": digest is not None and digest == row.get("prompt_digest"),
                                                        "prompt_tokens": len(data.get("prompt_token_ids") or [])}
            for key in ("prompt_token_ids",):
                data.pop(key, None)
            for choice in data.get("choices") or []:
                choice.pop("logprobs", None)
                choice.pop("token_ids", None)
            return httpx.Response(200, json=data, request=response.request)
        except Exception as exc:  # noqa: BLE001 - the attempt goes on without this reply's record
            self.stats["errors"] += 1
            self.agent.workspace.append("rollout_record_error", step=self.agent.steps,
                                        error=f"{type(exc).__name__}: {exc}"[:300])
            return response

    def extras(self):
        """Request fields added at the current level: 0 everything asked, 1 no token ids, 2 nothing."""
        if self.level == 2:
            return {}
        out = {"logprobs": True, "top_logprobs": self.cfg["top_logprobs"], "return_tokens_as_token_ids": True}
        if self.cfg["token_ids"] and self.level == 0:
            out["return_token_ids"] = True
        return out

    def record(self, payload, data):
        choice, usage = data["choices"][0], data.get("usage") or {}
        tokens, values, alternatives = parse_logprobs((choice.get("logprobs") or {}).get("content"),
                                                      self.cfg["top_logprobs"])
        # compared by content: the harness may rewrite a message it has already sent
        marks = [hashlib.sha1(json.dumps(m, sort_keys=True).encode()).digest() for m in payload["messages"]]
        keep, _ = delta(self.previous, marks)
        new = payload["messages"][keep:]
        prompt_ids, reply_ids = data.get("prompt_token_ids"), choice.get("token_ids")
        if reply_ids and tokens and list(reply_ids) != tokens:
            self.stats["id_mismatches"] += 1
        if reply_ids and not tokens:
            tokens = list(reply_ids)
        row = {"n": self.stats["replies"], "step": self.agent.steps, "gameplay": bool(payload.get("tools")),
               "keep": keep, "messages": new, "template": payload.get("chat_template_kwargs") or {},
               "sampling": {**{k: payload.get(k) for k in ("temperature", "top_p", "top_k", "max_tokens")},
                            **{k: payload[k] for k in ("min_p", "presence_penalty", "frequency_penalty",
                                                       "repetition_penalty") if k in payload}},
               "reply": choice["message"], "finish_reason": choice.get("finish_reason"), "usage": usage,
               "prompt_digest": ids_digest(prompt_ids) if prompt_ids else None,
               "tokens": tokens, "logprobs": values, "top": alternatives}
        if payload.get("tools") != self.tools or not self.stats["replies"]:
            row["tools"] = self.tools = payload.get("tools")
        line = json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"
        with self.path.open("a") as file:
            file.write(line)
        self.previous = marks
        self.stats["replies"] += 1
        self.stats["gameplay_replies"] += row["gameplay"]
        self.stats["tokens"] += len(tokens)
        self.stats["without_logprobs"] += not values
        self.stats["count_mismatches"] += len(tokens) != usage.get("completion_tokens")
        self.stats["bytes"] += len(line.encode())


def totals(rows):
    """The stored replies of a run, summed over its attempts; clears by replay verdict."""
    out = {}
    for row in rows:
        for key, value in (row.get("rollout") or {}).items():
            out[key] = out.get(key, 0) + value
    clears = [row for row in rows if row.get("cleared")]
    out["clears"] = len(clears)
    out["clears_replayed"] = sum((row.get("clear_replay") or {}).get("ok") is True for row in clears)
    return out


def logged_actions(log_path):
    """``[(action text, levels after it)]`` of an attempt's log (text None: an action that cannot be replayed)."""
    from .level_points import encode_action

    out = []
    with open(log_path) as file:
        for line in file:
            if line.startswith('{"kind":"observation"'):
                row = json.loads(line)
                if row.get("action") not in (None, "None"):
                    out.append((encode_action(row["action"]), int(row["levels"])))
    return out


def clear_replay(arc, card, agent, point):
    """A clear counts for training only if the attempt's own action list, replayed on a fresh game, reaches the same
    level count after every action and passes the start level: ``{"ok", "actions", "detail"}``."""
    from arcengine import GameAction

    from .level_points import decode_action

    try:
        agent.workspace.flush()
        actions = logged_actions(agent.workspace.log)
        if any(text is None for text, _ in actions):
            return {"ok": False, "actions": len(actions), "detail": "an action that cannot be replayed"}
        env = arc.make(point["game"], scorecard_id=card, save_recording=False)
        reached = 0
        for index, (text, logged) in enumerate(actions):
            name, x, y = decode_action(text)
            raw = env.reset() if name == "RESET" else env.step(GameAction[name], {"x": x, "y": y} if name == "ACTION6" else {})
            levels = int(raw.levels_completed)
            if levels != logged:
                return {"ok": False, "actions": len(actions), "detail": f"action {index}: levels {levels} != {logged}"}
            reached = max(reached, levels)
        ok = reached > int(point["level_index"])
        return {"ok": ok, "actions": len(actions), "detail": "ok" if ok else f"reaches levels {reached}"}
    except Exception as exc:  # noqa: BLE001 - an unverified clear is reported, the bench goes on
        return {"ok": False, "actions": None, "detail": f"{type(exc).__name__}: {exc}"[:200]}
