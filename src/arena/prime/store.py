"""Continual-harness state of one session: entries of four kinds with versions, the refinement record, the edits a
refinement proposes and their rollback. Port of prime-agent v0.3.3 (bc6894c8): ``refinement/refinement.ts`` (proposal
parsing, validation, application, rollback, the planner's overview and history, the prompt block) and
``prime-agent-runtime/src/rlm/harness.py`` (the kernel's create / update / delete semantics). One store per game,
kept by the host; local scope only (a level-bench attempt has no other session to share with)."""

import copy
import json
import re
import time

from . import texts

KINDS = ("prompt", "memory", "skill", "subagent")
ENTRY_LIMIT, REFINEMENT_LIMIT, CONTENT_LIMIT = 6, 5, 180   # the prompt block (DEFAULT_OVERVIEW_*)
FULL_ENTRY_LIMIT, FULL_CONTENT_LIMIT = 12, 4000            # (ours) the block with entries in full
PLAN_ENTRIES, PLAN_CHARS, PLAN_HISTORY = 40, 240, 20        # the planner's overview and history


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def slug(raw, fallback):
    text = re.sub(r"^_+|_+$", "", re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()))[:80]
    return text or fallback


def compact_text(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def skill_reference_error(reference):
    if not isinstance(reference, dict):
        return "skill requires python reference"
    if reference.get("type") != "python":
        return "skill reference.type must be python"
    if not any(isinstance(reference.get(k), str) and reference[k] for k in ("import", "python_import")):
        return "skill requires python import"
    if not any(isinstance(reference.get(k), str) and reference[k] for k in ("callable", "call_pattern")):
        return "skill requires callable or call_pattern"
    return None


def extract_json_object(text):
    """The refiner's JSON object: the whole reply, a fenced block, or the outermost braces."""
    trimmed = str(text or "").strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        return json.loads(trimmed)
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", trimmed)
    if fenced:
        return json.loads(fenced.group(1).strip())
    start, end = trimmed.find("{"), trimmed.rfind("}")
    if start != -1 and end > start:
        return json.loads(trimmed[start:end + 1])
    raise ValueError("Refiner did not return a JSON object")


def parse_proposal(text):
    value = extract_json_object(text)
    if not isinstance(value, dict):
        raise ValueError("Refiner JSON must be an object")

    def text_of(item, key):
        return item[key] if isinstance(item.get(key), str) else None

    def record(item, key):
        return item[key] if isinstance(item.get(key), dict) else None

    edits = value.get("edits") if isinstance(value.get("edits"), list) else []
    return {
        "summary": value["summary"] if isinstance(value.get("summary"), str) else "Refined continual harness state",
        "rationale": value["rationale"] if isinstance(value.get("rationale"), str) else "",
        "expectedOutcome": value["expectedOutcome"] if isinstance(value.get("expectedOutcome"), str) else "",
        "edits": [{"action": e.get("action"), "kind": e.get("kind"), "id": text_of(e, "id"), "title": text_of(e, "title"),
                   "content": text_of(e, "content"), "path": text_of(e, "path"), "reference": record(e, "reference"),
                   "arguments": record(e, "arguments"), "metadata": record(e, "metadata"), "reason": text_of(e, "reason")}
                  for e in edits if isinstance(e, dict)],
    }


def parse_review(text):
    value = extract_json_object(text)
    if not isinstance(value, dict):
        raise ValueError("Auto-refine review JSON must be an object")
    return {"shouldRefine": value.get("shouldRefine") is True,
            "rationale": value["rationale"] if isinstance(value.get("rationale"), str) else "No rationale provided.",
            "instructions": value["instructions"] if isinstance(value.get("instructions"), str) else None}


def validate_edit(edit, computed_id):
    if edit["action"] not in ("create", "update", "delete"):
        return f"unsupported action {edit['action']}"
    if edit["kind"] not in KINDS:
        return f"unsupported kind {edit['kind']}"
    if edit["kind"] == "prompt" and "base_system_prompt" in (edit["id"], computed_id):
        return "base system prompt is not editable"
    if edit["action"] != "create" and not edit["id"]:
        return f"{edit['action']} requires id"
    if edit["action"] != "delete" and (not edit["title"] or not edit["content"]):
        return f"{edit['action']} requires title and content"
    if edit["action"] != "delete" and edit["kind"] == "skill":
        if edit["arguments"] is None:
            return f"{edit['action']} skill requires arguments"
        error = skill_reference_error(edit["reference"])
        if error:
            return f"{edit['action']} {error}"
    return None


class HarnessStore:
    def __init__(self):
        self.entries = {kind: {} for kind in KINDS}
        self.refinements = []   # events shown in the prompt block
        self.history = []       # full results (applied edits with before / after), for the planner and rollback
        self.writes = {"agent": 0, "refine": 0}
        self.serial = 0

    # --- the kernel's CRUD (source "agent") --------------------------------------------------------------------

    def _kind(self, kind):
        if kind not in self.entries:
            raise ValueError(f"unknown harness kind {kind!r}; expected one of {KINDS}")
        return self.entries[kind]

    @staticmethod
    def _bare(entry_id):
        scope, sep, rest = str(entry_id or "").partition(":")
        return rest if sep and rest and scope in ("local", "global") else entry_id

    def _upsert(self, kind, title, content, *, id=None, path=None, reference=None, arguments=None, metadata=None,
                source="agent"):
        records = self._kind(kind)
        if not isinstance(title, str) or not title or not isinstance(content, str):
            raise ValueError(f"{kind} entries require a title and content (strings)")
        entry_id = id or slug(title, kind)
        if kind == "skill" and (reference is not None or entry_id not in records):
            error = skill_reference_error(reference)
            if error:
                raise ValueError(error)
        existing = records.get(entry_id)
        if existing:
            existing.update(title=title, content=content, source=source, updated_at=now(),
                            version=existing["version"] + 1)
            for key, value in (("path", path), ("reference", reference), ("arguments", arguments),
                               ("metadata", metadata)):
                if value is not None:
                    existing[key] = copy.deepcopy(value)
            entry = existing
        else:
            entry = {"id": entry_id, "kind": kind, "title": title, "content": content,
                     "path": path if path is not None else "general", "scope": "local",
                     "reference": copy.deepcopy(reference or {}), "arguments": copy.deepcopy(arguments or {}),
                     "metadata": copy.deepcopy(metadata or {}), "source": source, "created_at": now(),
                     "updated_at": now(), "version": 1}
            records[entry_id] = entry
        self.writes[source if source in self.writes else "agent"] += 1
        return copy.deepcopy(entry)

    def create(self, kind, title, content, *, id=None, path="general", **fields):
        entry_id = self._bare(id) or slug(title, kind)
        if entry_id in self._kind(kind):
            raise ValueError(f"{kind} entry {entry_id!r} already exists")
        return self._upsert(kind, title, content, id=entry_id, path=path, **fields)

    def update(self, kind, id, title, content, **fields):
        entry_id = self._bare(id)
        if entry_id not in self._kind(kind):
            raise ValueError(f"{kind} entry {entry_id!r} does not exist")
        return self._upsert(kind, title, content, id=entry_id, **fields)

    def delete(self, kind, id):
        found = self._kind(kind).pop(self._bare(id), None) is not None
        self.writes["agent"] += found
        return found

    def get(self, kind, id):
        return copy.deepcopy(self._kind(kind).get(self._bare(id)))

    def list(self, kind=None):
        records = [e for k in ([kind] if kind else KINDS) for e in self._kind(k).values()]
        return copy.deepcopy(sorted(records, key=lambda e: (e["kind"], e["path"], e["title"], e["id"])))

    def count(self):
        return sum(len(records) for records in self.entries.values())

    def record_refinement(self, trigger, changes, *, evidence="", outcome="", id=None):
        """The kernel's ``rlm.harness.record_refinement``: an event in the prompt block, no edits."""
        event = {"id": id or self.new_id(), "trigger": str(trigger),
                 "changes": [str(c) for c in changes] if isinstance(changes, (list, tuple)) else [str(changes)],
                 "evidence": str(evidence), "outcome": str(outcome), "created_at": now()}
        self.refinements.append(event)
        self.writes["agent"] += 1
        return copy.deepcopy(event)

    # --- refinement ------------------------------------------------------------------------------------------

    def new_id(self):
        self.serial += 1
        return time.strftime("refine_%Y%m%d%H%M%S", time.gmtime()) + f"{self.serial:03d}"

    def apply(self, proposal, *, refinement_id=None, rollback_of=None):
        """Apply a parsed proposal edit by edit (an invalid edit is recorded as failed and skipped); returns the
        result, which is also appended to the history."""
        refinement_id = refinement_id or self.new_id()
        applied = []
        for edit in proposal["edits"]:
            computed = edit["id"] or (slug(edit["title"] or edit["kind"], str(edit["kind"]))
                                      if edit["action"] == "create" else None)
            entry_id = self._bare(computed) or ""
            error = validate_edit(edit, entry_id)
            if error:
                applied.append({**edit, "id": entry_id, "applied": False, "error": error})
                continue
            records = self.entries[edit["kind"]]
            before = copy.deepcopy(records.get(entry_id))
            if edit["action"] == "delete":
                if before is None:
                    applied.append({**edit, "id": entry_id, "applied": False, "error": "entry not found"})
                    continue
                del records[entry_id]
                applied.append({**edit, "id": entry_id, "before": before, "applied": True})
                continue
            if edit["action"] == "create" and before is not None:
                applied.append({**edit, "id": entry_id, "before": before, "applied": False,
                                "error": "entry already exists"})
                continue
            if edit["action"] == "update" and before is None:
                applied.append({**edit, "id": entry_id, "applied": False, "error": "entry not found"})
                continue
            old = before or {}
            after = {"id": entry_id, "kind": edit["kind"], "title": edit["title"] or old.get("title") or entry_id,
                     "content": edit["content"] or old.get("content") or "",
                     "path": edit["path"] or old.get("path") or "general", "scope": old.get("scope") or "local",
                     "reference": edit["reference"] if edit["reference"] is not None else old.get("reference") or {},
                     "arguments": edit["arguments"] if edit["arguments"] is not None else old.get("arguments") or {},
                     "metadata": edit["metadata"] if edit["metadata"] is not None else old.get("metadata") or {},
                     "source": "refine", "created_at": old.get("created_at") or now(), "updated_at": now(),
                     "version": old.get("version", 0) + 1}
            records[entry_id] = after
            applied.append({**edit, "id": entry_id, "before": before, "after": copy.deepcopy(after), "applied": True})
        changes = [f"{e['action']} {e['kind']}:{e['id']}" for e in applied if e["applied"]]
        self.writes["refine"] += len(changes)
        self.refinements.append({"id": refinement_id, "trigger": proposal["summary"], "changes": changes,
                                 "evidence": proposal["rationale"], "outcome": proposal["expectedOutcome"],
                                 "created_at": now()})
        result = {"id": refinement_id, "summary": proposal["summary"], "rationale": proposal["rationale"],
                  "expectedOutcome": proposal["expectedOutcome"], "appliedEdits": applied, "rollbackOf": rollback_of,
                  "scope": "local"}
        self.history.append(result)
        return result

    def rollback(self, refinement_id):
        """Undo one refinement: its applied edits in reverse, each entry back to its snapshot."""
        target = next((item for item in self.history if item["id"] == refinement_id), None)
        if target is None:
            raise ValueError(f"Refinement {refinement_id} not found")
        edits = []
        blank = {"path": None, "reference": None, "arguments": None, "metadata": None, "title": None, "content": None}
        for edit in reversed(target["appliedEdits"]):
            if not edit["applied"]:
                continue
            before = edit.get("before")
            if before:
                edits.append({"action": "update" if edit.get("after") else "create", "kind": edit["kind"],
                              "id": edit["id"], "title": before["title"], "content": before["content"],
                              "path": before["path"], "reference": before["reference"],
                              "arguments": before["arguments"], "metadata": before["metadata"],
                              "reason": f"Rollback {refinement_id}"})
            elif edit.get("after"):
                edits.append({**blank, "action": "delete", "kind": edit["kind"], "id": edit["id"],
                              "reason": f"Rollback {refinement_id}"})
        proposal = {"summary": f"Rollback refinement {refinement_id}",
                    "rationale": f"Restores continual harness state snapshots from refinement {refinement_id}.",
                    "expectedOutcome": "Faulty refinement edits are reverted.", "edits": edits}
        return self.apply(proposal, rollback_of=refinement_id)

    # --- texts ---------------------------------------------------------------------------------------------------

    def overview_for_plan(self):
        lines = []
        for kind in KINDS:
            entries = list(self.entries[kind].values())
            lines.append(f"{kind}: {len(entries)}")
            for entry in entries[:PLAN_ENTRIES]:
                extra = ""
                if kind == "skill" and entry["reference"]:
                    extra += " ref=" + json.dumps(entry["reference"], separators=(",", ":"))[:PLAN_CHARS]
                if kind == "skill" and entry["arguments"]:
                    extra += " args=" + json.dumps(entry["arguments"], separators=(",", ":"))[:PLAN_CHARS]
                content = " ".join(entry["content"].split())[:PLAN_CHARS]
                lines.append(f"- [{entry['scope']}:{entry['id']}] {entry['title']} ({entry['path']}, "
                             f"v{entry['version']}){extra}: {content}")
            if len(entries) > PLAN_ENTRIES:
                lines.append(f"- +{len(entries) - PLAN_ENTRIES} more {kind} entries")
        return "\n".join(lines)

    def history_for_plan(self):
        if not self.history:
            return "No prior refinement history."
        parts = []
        for item in self.history[-PLAN_HISTORY:]:
            edits = ", ".join(f"{'applied' if e['applied'] else 'failed'} {e['action']} {e['kind']}:{e['id']}"
                              for e in item["appliedEdits"])
            rollback = f" rollbackOf={item['rollbackOf']}" if item["rollbackOf"] else ""
            parts.append(f"[{item['id']}]{rollback} {item['summary']}\n{edits}\nExpected outcome: "
                         f"{item['expectedOutcome']}")
        return "\n\n".join(parts)

    def entries_block(self, full=False):
        """The entry list and the recent refinements of the prompt block (``formatHarnessStateForPrompt`` after its
        fixed head). ``full`` (ours): each entry's whole content, newest 12 per kind."""
        lines, total = [], 0
        entry_limit, content_limit = (FULL_ENTRY_LIMIT, FULL_CONTENT_LIMIT) if full else (ENTRY_LIMIT, CONTENT_LIMIT)
        for kind in KINDS:
            entries = sorted(self.entries[kind].values(), key=lambda e: (e["path"], e["title"], e["id"]))
            total += len(entries)
            lines.append(f"{kind}: {len(entries)}")  # the subagent line's rlm hint needs recursion, which is off
            if full:
                entries = sorted(entries, key=lambda e: e["updated_at"], reverse=True)
            for entry in entries[:entry_limit]:
                extra = ""
                if kind == "skill" and entry["reference"]:
                    extra += " ref=" + compact_text(json.dumps(entry["reference"], separators=(",", ":")), CONTENT_LIMIT)
                if kind == "skill" and entry["arguments"]:
                    extra += " args=" + compact_text(json.dumps(entry["arguments"], separators=(",", ":")),
                                                     CONTENT_LIMIT)
                content = entry["content"].strip() if full else compact_text(entry["content"], CONTENT_LIMIT)
                if len(content) > content_limit:
                    content = content[:content_limit] + "..."
                lines.append(f"- [{entry['scope']}:{entry['id']}] {entry['title']} ({entry['path']}, "
                             f"v{entry['version']}){extra}: {content}")
            if len(entries) > entry_limit:
                lines.append(f"- +{len(entries) - entry_limit} more {kind} entries")
            lines.append("")
        if not total:
            lines += ["No saved harness entries yet.", ""]
        lines.append(f"recent refinements: {len(self.refinements)}")
        for event in self.refinements[-REFINEMENT_LIMIT:]:
            changes = ", ".join(event["changes"]) or "no applied edits"
            outcome = f"; outcome: {compact_text(event['outcome'], CONTENT_LIMIT)}" if event["outcome"] else ""
            lines.append(f"- [{event['id']}] {compact_text(event['trigger'], CONTENT_LIMIT)}: {changes}{outcome}")
        if len(self.refinements) > REFINEMENT_LIMIT:
            lines.append(f"- +{len(self.refinements) - REFINEMENT_LIMIT} older refinement events")
        return "\n".join(lines).strip()

    def prompt_block(self):
        """``formatHarnessStateForPrompt``: the fixed head and the entries."""
        return texts.HARNESS_STATE_HEAD + "\n\n" + self.entries_block()

    def overview(self, max_entries_per_kind=20):
        """The kernel's ``rlm.harness.overview()``."""
        lines = []
        for kind in KINDS:
            entries = sorted(self.entries[kind].values(), key=lambda e: (e["path"], e["title"], e["id"]))
            lines.append(f"{kind}: {len(entries)}")
            lines += [f"- [{e['scope']}:{e['id']}] {e['title']} ({e['path']}, v{e['version']}): "
                      f"{compact_text(e['content'], CONTENT_LIMIT)}" for e in entries[:max_entries_per_kind]]
        lines.append(f"refinements: {len(self.refinements)}")
        return "\n".join(lines)

    def stats(self):
        by = {kind: len(records) for kind, records in self.entries.items()}
        authors = {}
        for records in self.entries.values():
            for entry in records.values():
                authors[entry["source"]] = authors.get(entry["source"], 0) + 1
        return {"entries": by, "by_source": authors, "writes": dict(self.writes), "refinements": len(self.refinements),
                "rollbacks": sum(1 for item in self.history if item["rollbackOf"])}
