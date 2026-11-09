"""Cross-game notes (variant ``fixed-2x-notes``): one notebook of general knowledge per trial.

At every level clear and at every park the agent is asked, in one extra tool-less request
(thinking off, 1,024-token cap), for at most ten lines of general knowledge: mechanics
patterns, python-tool tricks, mistakes to avoid, nothing game-specific. New lines (rough
de-duplication: the same letters and digits) join the shared notebook, which is also written
to ``cross-game-notes.md`` in the trial output and read back from there when a later run
starts in the same folder. From a game's next admission on, its requests carry the newest
notebook lines that fit the injection cap as the first user message after the system prompt,
labelled as notes from other games; the text is frozen for the admission so the prefix
cache survives between requests.
"""

import re
from pathlib import Path

from duck_transfer.trimming import _text_tokens

NOTES_LABEL = (
    "[Shared notes from other games] General knowledge written down by agents playing other "
    "games of this suite (mechanics patterns, python-tool tricks, mistakes to avoid). They are "
    "hypotheses about games in general, not facts about this game: use what the evidence "
    "here supports.\n"
)

NOTES_PROMPT = (
    "Pause for a moment. A shared notebook of general knowledge is kept across the games of "
    "this suite; its current entries are listed below. Append at most {lines} new lines of "
    "GENERAL knowledge that would help an agent playing a different game: mechanics patterns "
    "(how HUDs, timers, counters, goals or level transitions tend to appear), python-tool "
    "tricks that saved time, mistakes to avoid. Nothing specific to this game: no coordinates, "
    "colours, level numbers or object names of this board. Do not call tools and do not repeat "
    "an entry already in the notebook. Reply with only the new lines, one per line starting "
    "with '- ', or the single word NONE.\n\nNotebook so far:\n{notebook}"
)

HEADER = (
    "# Cross-game notes\n\n"
    "General knowledge appended by the arena agents, oldest first (one line per entry).\n\n"
)
MAX_LINE_CHARS = 300
_BULLET = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")


def normalize(line):
    """The rough de-duplication key: letters and digits, single spaces, lower case."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(line).lower()).split())


def parse_notes(text, limit):
    """At most ``limit`` note lines from a reply: bullets stripped, fences, headings and
    ``NONE`` skipped, each line cut at ``MAX_LINE_CHARS``."""
    lines = []
    for raw in str(text or "").splitlines():
        line = _BULLET.sub("", raw).strip()
        if not line or line.startswith("```") or line.upper() == "NONE" or line.endswith(":"):
            continue
        lines.append(line[:MAX_LINE_CHARS])
        if len(lines) >= limit:
            break
    return lines


class CrossGameNotes:
    """The trial's notebook: ordered unique lines, mirrored to ``path`` after every change."""

    SETTINGS = ("lines_per_request", "request_cap_tokens", "inject_cap_tokens", "refresh")
    REFRESH = ("admission", "live")

    def __init__(self, path, *, lines_per_request=10, request_cap_tokens=1024,
                 inject_cap_tokens=2000, refresh="admission"):
        self.path = Path(path)
        self.lines_per_request = int(lines_per_request)
        self.request_cap_tokens = int(request_cap_tokens)
        self.inject_cap_tokens = int(inject_cap_tokens)
        # ``live``: the injected notebook is also refreshed at a
        # game's next level clear and next compaction, so running focus games (which never
        # re-admit) receive the notes; each refresh re-prefills the prompt once.
        if refresh not in self.REFRESH:
            raise ValueError("Notes refresh must be 'admission' or 'live'")
        self.refresh = refresh
        self.lines = []
        self.sources = []
        self._keys = set()
        self.duplicates = 0
        self.carried = 0
        if self.path.is_file():  # an earlier run (or cycle) in the same folder
            for raw in self.path.read_text().splitlines():
                if raw.startswith("- ") and self._append(raw[2:], source="carried"):
                    self.carried += 1

    @classmethod
    def from_settings(cls, path, settings):
        return cls(path, **{key: settings[key] for key in cls.SETTINGS if key in settings})

    def _append(self, line, *, source):
        key = normalize(line)
        if not key or key in self._keys:
            return False
        self._keys.add(key)
        self.lines.append(line)
        self.sources.append(source)
        return True

    def add(self, lines, *, source):
        """Append the new lines (duplicates skipped) and rewrite the file; the number added."""
        added = 0
        for line in lines:
            if self._append(line, source=source):
                added += 1
            else:
                self.duplicates += 1
        if added:
            self.write()
        return added

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(HEADER + "".join(f"- {line}\n" for line in self.lines))

    def text(self):
        return "\n".join(f"- {line}" for line in self.lines)

    def render(self, chars_per_token=3.0):
        """``(message, lines, tokens)``: the newest lines that fit ``inject_cap_tokens`` (oldest
        dropped) as the labelled user message; ``(None, 0, 0)`` when nothing fits."""
        kept = []
        total = _text_tokens(NOTES_LABEL, chars_per_token)
        for line in reversed(self.lines):
            cost = _text_tokens(f"- {line}\n", chars_per_token)
            if total + cost > self.inject_cap_tokens:
                break
            kept.append(line)
            total += cost
        if not kept:
            return None, 0, 0
        kept.reverse()
        message = {"role": "user", "content": NOTES_LABEL + "\n".join(f"- {line}" for line in kept)}
        return message, len(kept), total

    def summary(self):
        return {
            "path": str(self.path),
            "lines": len(self.lines),
            "carried_lines": self.carried,
            "duplicates_skipped": self.duplicates,
            "lines_per_request": self.lines_per_request,
            "request_cap_tokens": self.request_cap_tokens,
            "inject_cap_tokens": self.inject_cap_tokens,
            "refresh": self.refresh,
        }
