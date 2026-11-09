"""The exact served input of every chat request, for SFT reuse: every request's messages as
sent, its non-message fields (model, sampling, tools, chat_template_kwargs) and the server's reply message as
received, written so each request rebuilds byte-exact from the files alone.

Files in the trial's output folder (each under the runner's 32 MB mid-run sync limit):
- ``served-NNN.jsonl``: one row a request, in send order: ``task``, ``seq`` (the task's request number), ``keep``
  (leading messages equal to the task's previous request), ``append`` (the rest, as sent), ``static`` (sha256 of the
  non-message fields), ``response`` (``choices[0].message``), ``finish_reason``, ``usage``, ``t0``/``t1``; with
  ``return_token_ids`` (default on) the reply's ``token_ids`` and the prompt's ids, delta-encoded like the messages
  (``prompt_ids``: ``keep``, ``append``, ``n``, ``sha256``); ``messages_sha256`` over the full message list as sent;
  ``reply``: the engine's reply number (``graded.candidates[].reply`` of the same task).
- ``served-static.jsonl``: ``{"sha": ..., "fields": {...}, "order": [payload keys]}`` once per distinct set.
- ``served-images.jsonl``: ``{"sha": ..., "url": ...}`` once per distinct image; in ``append`` an image's url is
  ``"sha256:<sha>"``.

``rebuild(folder)`` yields ``(row, payload)``: the payload exactly as posted, ``row["prompt_token_ids"]`` restored; a
broken delta chain raises."""
import array
import copy
import hashlib
import json
from pathlib import Path

CHUNK_BYTES = 30 * 2 ** 20
IMAGE_PREFIX = "sha256:"
ROW_KEYS = ("messages", "max_tokens")  # per request; the other payload fields go to served-static.jsonl


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class ServedLog:
    def __init__(self, folder, ids=True):
        self.folder, self.ids = Path(folder), bool(ids)  # ids: the request asks for return_token_ids
        self.folder.mkdir(parents=True, exist_ok=True)
        self.last, self.seq, self.statics, self.images = {}, {}, set(), set()
        self.last_ids = {}  # task -> array('I') of its previous request's prompt ids (delta encoding)
        for name, keep in (("served-static.jsonl", self.statics), ("served-images.jsonl", self.images)):
            path = self.folder / name
            if path.exists():  # a resumed play: append, never write a known entry twice
                keep.update(json.loads(line)["sha"] for line in path.read_text().splitlines() if line.strip())
        self.chunk = len(list(self.folder.glob("served-[0-9]*.jsonl")))
        self.stream = None

    def _images(self, message):
        content = message.get("content")
        if not isinstance(content, list):
            return message
        message = copy.deepcopy(message)
        for part in message["content"]:
            url = (part.get("image_url") or {}).get("url") if isinstance(part, dict) else None
            if isinstance(url, str) and not url.startswith(IMAGE_PREFIX):
                sha = _sha(url)
                if sha not in self.images:
                    self.images.add(sha)
                    self._append("served-images.jsonl", {"sha": sha, "url": url})
                part["image_url"] = {**part["image_url"], "url": IMAGE_PREFIX + sha}
        return message

    def _append(self, name, row):
        with (self.folder / name).open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def _write(self, row):
        line = json.dumps(row) + "\n"
        if self.stream is None or self.stream.tell() + len(line) > CHUNK_BYTES:
            if self.stream is not None:
                self.stream.close()
                self.chunk += 1
            self.stream = (self.folder / f"served-{self.chunk:03d}.jsonl").open("a")
        self.stream.write(line)
        self.stream.flush()

    def _prompt_ids(self, task, ids):
        ids = array.array("I", ids)
        before = self.last_ids.get(task) or array.array("I")
        keep, n = 0, min(len(ids), len(before))
        while keep < n and ids[keep] == before[keep]:
            keep += 1
        self.last_ids[task] = ids
        return {"keep": keep, "append": ids[keep:].tolist(), "n": len(ids), "sha256": _sha(json.dumps(ids.tolist()))}

    def write(self, task, payload, data, t0, t1, reply=None):
        fields = {k: v for k, v in payload.items() if k not in ROW_KEYS}
        order = list(payload)  # key order as posted (byte-exact rebuild)
        static = _sha(json.dumps([fields, order], sort_keys=True))
        if static not in self.statics:
            self.statics.add(static)
            self._append("served-static.jsonl", {"sha": static, "fields": fields, "order": order})
        messages = [self._images(m) for m in payload["messages"]]
        keys = [json.dumps(m, sort_keys=True) for m in messages]
        before = self.last.get(task, [])
        keep = 0
        while keep < min(len(keys), len(before)) and keys[keep] == before[keep]:
            keep += 1
        self.last[task] = keys
        self.seq[task] = self.seq.get(task, 0) + 1
        choice = (data or {}).get("choices", [{}])[0]
        prompt_ids = (data or {}).get("prompt_token_ids")
        ids = {} if prompt_ids is None else {"prompt_ids": self._prompt_ids(task, prompt_ids)}
        if choice.get("token_ids") is not None:
            ids["token_ids"] = choice["token_ids"]
        per = {k: payload[k] for k in ROW_KEYS if k in payload and k != "messages"}
        self._write({**ids, **per, "task": task, "seq": self.seq[task], "reply": reply, "keep": keep,
                     "append": messages[keep:], "messages_sha256": _sha(json.dumps(payload["messages"])),
                     "static": static,
                     "response": choice.get("message"), "finish_reason": choice.get("finish_reason"),
                     "usage": (data or {}).get("usage"), "t0": t0, "t1": t1,
                     **({"think_cut": data["think_cut"]} if (data or {}).get("think_cut") else {})})


def rebuild(folder):
    """``(row, payload)`` per logged request, in send order, the payload as posted (images restored)."""
    folder = Path(folder)
    statics = {r["sha"]: r for r in map(json.loads, (folder / "served-static.jsonl").read_text().splitlines())}
    path = folder / "served-images.jsonl"
    images = {r["sha"]: r["url"] for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}

    def restore(message):
        if isinstance(message.get("content"), list):
            message = copy.deepcopy(message)
            for part in message["content"]:
                url = (part.get("image_url") or {}).get("url") if isinstance(part, dict) else None
                if isinstance(url, str) and url.startswith(IMAGE_PREFIX):
                    part["image_url"]["url"] = images[url[len(IMAGE_PREFIX):]]
        return message

    last, last_ids = {}, {}
    for chunk in sorted(folder.glob("served-[0-9]*.jsonl")):
        for line in chunk.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            messages = last.get(row["task"], [])[:row["keep"]] + row["append"]
            last[row["task"]] = messages
            static = statics[row["static"]]
            values = {**static["fields"], **{k: row[k] for k in ROW_KEYS if k in row and k != "messages"},
                      "messages": [restore(m) for m in messages]}
            payload = {k: values[k] for k in static["order"]}
            if row.get("messages_sha256") and _sha(json.dumps(payload["messages"])) != row["messages_sha256"]:
                raise ValueError(f"served log: broken message chain at {row['task']} request {row['seq']}")
            pid = row.get("prompt_ids")
            if pid is not None:
                ids = last_ids.get(row["task"], [])[:pid["keep"]] + pid["append"]
                last_ids[row["task"]] = ids
                if len(ids) != pid["n"] or _sha(json.dumps(ids)) != pid["sha256"]:
                    raise ValueError(f"served log: broken prompt-id chain at {row['task']} request {row['seq']}")
                row = {**row, "prompt_token_ids": ids}
            yield row, payload
