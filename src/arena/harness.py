"""The evolvable Duck harness: a directory of model-facing surfaces loaded into every game.

Offline harness evolution (ideas.md point 6, paper-notes.md §2.2) edits a directory, never the
agent code, the scheduler, the scorer or the clocks:

- ``prompt.md``: appended to Duck's system prompt (``prompt_cap_tokens``).
- ``playbook.md``: carried as the first user message of every request, outside the analyzer's
  append-only history, so compaction leaves it in place (``playbook_cap_tokens``).
- ``skills/*.py``: modules pre-loaded into the game's persistent interpreter as
  ``skills.<name>`` whenever the interpreter starts or restarts; ``skills.names()`` lists them
  and the system prompt carries each module's first docstring line.
- ``CHANGELOG.md``: the proposer's rationale per round (not loaded into the game).

Everything is bounded and validated before a game sees it; a skill that does not compile is
skipped and reported, never loaded.
"""

import ast
import json
from pathlib import Path

CHARS_PER_TOKEN = 3.5
DEFAULT_CAPS = {"prompt_cap_tokens": 1500, "playbook_cap_tokens": 2000, "skill_cap_chars": 12000, "max_skills": 12}
PLAYBOOK_LABEL = "[Playbook] Notes carried across games by the evolved harness (general, not about this game):\n"
SKILLS_HEADER = ("\nHelper functions pre-loaded in your persistent interpreter by the harness (call them directly; "
                 "`skills_names` lists the skill files and `skills_source[name]` holds each file's source):\n")
# The Duck sandbox executes plain code with a builtins whitelist: no class statements, no exec,
# no numpy, imports limited to this list (mirrors inference.agent.python_tool_sandbox.SAFE_MODULES).
SANDBOX_MODULES = frozenset({"bisect", "collections", "copy", "fractions", "functools", "heapq", "itertools", "json",
                             "math", "operator", "random", "re", "statistics", "string"})
SANDBOX_FORBIDDEN_CALLS = frozenset({"exec", "eval", "compile", "setattr", "delattr", "globals", "locals", "vars", "open", "__import__"})


def sandbox_violation(tree):
    """The first construct the Duck sandbox would reject, or None."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return f"class statements are not available in the sandbox (line {node.lineno})"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".")[0] for alias in node.names] if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
            for name in names:
                if name not in SANDBOX_MODULES:
                    return f"import of {name!r} is not allowed in the sandbox (line {node.lineno})"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SANDBOX_FORBIDDEN_CALLS:
            return f"{node.func.id}() is not available in the sandbox (line {node.lineno})"
    return None


class DuckHarness:
    def __init__(self, root, **caps):
        self.root = Path(root)
        unknown = set(caps) - set(DEFAULT_CAPS)
        if unknown:
            raise ValueError(f"Unknown harness caps: {sorted(unknown)}")
        self.caps = {**DEFAULT_CAPS, **caps}
        self.problems = []

    # --- files -----------------------------------------------------------------------------

    def _text(self, name, cap_tokens):
        path = self.root / name
        if not path.is_file():
            return ""
        text = path.read_text(errors="replace").strip()
        limit = int(cap_tokens * CHARS_PER_TOKEN)
        if len(text) > limit:
            self.problems.append(f"{name}: {len(text)} chars over the {limit}-char cap; truncated")
            text = text[:limit]
        return text

    def skill_files(self):
        folder = self.root / "skills"
        if not folder.is_dir():
            return []
        return sorted(p for p in folder.glob("*.py") if p.name != "__init__.py" and p.stem.isidentifier())

    def skills(self):
        """``[(name, source, docline)]`` for the skills that compile, within the caps."""
        loaded = []
        for path in self.skill_files():
            source = path.read_text(errors="replace")
            if len(source) > self.caps["skill_cap_chars"]:
                self.problems.append(f"skills/{path.name}: {len(source)} chars over the {self.caps['skill_cap_chars']}-char cap; skipped")
                continue
            try:
                tree = ast.parse(source, str(path))
            except SyntaxError as exc:
                self.problems.append(f"skills/{path.name}: {exc.msg} (line {exc.lineno}); skipped")
                continue
            violation = sandbox_violation(tree)
            if violation:
                self.problems.append(f"skills/{path.name}: {violation}; skipped")
                continue
            if len(loaded) >= self.caps["max_skills"]:
                self.problems.append(f"skills/{path.name}: beyond the {self.caps['max_skills']}-skill cap; skipped")
                continue
            docline = ""
            stripped = source.lstrip()
            for quote in ('"""', "'''"):
                if stripped.startswith(quote):
                    docline = stripped[3:].split(quote, 1)[0].strip().splitlines()[0].strip() if stripped[3:].strip() else ""
                    break
            loaded.append((path.stem, source, docline))
        return loaded

    def skill_signatures(self, source):
        tree = ast.parse(source)
        return [f"{node.name}({', '.join(arg.arg for arg in node.args.args)})"
                for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")]

    # --- model-facing surfaces ---------------------------------------------------------------

    def prompt_text(self):
        """System-prompt addition: ``prompt.md`` plus the skills index (empty when both are)."""
        text = self._text("prompt.md", self.caps["prompt_cap_tokens"])
        skills = self.skills()
        if skills:
            text += SKILLS_HEADER + "".join(
                f"- {name}: {doc or '(no docstring)'} [{', '.join(self.skill_signatures(source)) or 'no public functions'}]\n"
                for name, source, doc in skills)
        return text

    def playbook_message(self):
        text = self._text("playbook.md", self.caps["playbook_cap_tokens"])
        return {"role": "user", "content": PLAYBOOK_LABEL + text} if text else None

    def skills_preamble(self):
        """Code that defines the skills in a fresh interpreter (plain top-level definitions).

        The Duck sandbox runs code under a builtins whitelist (no ``exec``, ``class`` or
        ``setattr``, no numpy), so a skill is a file of functions that becomes part of the
        interpreter's globals; ``skills_names`` and ``skills_source`` describe what was loaded.
        """
        skills = self.skills()
        parts = ["skills_names = " + json.dumps([name for name, _, _ in skills]),
                 "skills_source = " + json.dumps({name: source for name, source, _ in skills})]
        for name, source, _ in skills:
            parts.append(f"# --- harness skill: {name} ---\n{source}")
        return "\n".join(parts) + "\n"

    # --- checks ------------------------------------------------------------------------------

    def validate(self):
        """Problems that would change what a game sees (caps, syntax); empty when clean."""
        self.problems = []
        self.prompt_text()
        self.playbook_message()
        return list(self.problems)

    def summary(self):
        skills = self.skills()
        return {
            "root": str(self.root),
            "prompt_chars": len(self._text("prompt.md", self.caps["prompt_cap_tokens"])),
            "playbook_chars": len(self._text("playbook.md", self.caps["playbook_cap_tokens"])),
            "skills": [name for name, _, _ in skills],
            "skill_chars": sum(len(source) for _, source, _ in skills),
            "problems": sorted(set(self.problems)),
        }
