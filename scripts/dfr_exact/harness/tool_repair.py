"""Malformed tool-name repair (throughput arm, 20 Oct): drain_half's ``drop_malformed`` widened to calls that name no
offered tool or miss a required argument, behind the optional drain_half key ``drop_unknown_tools`` (absent: the
harness is byte-for-byte unchanged).

Why: l45's 150K read lost 2 of 100 games to a malformed first call (name ``python</parameter>\\n<parameter=code``,
arguments ``{}``). Its arguments are valid JSON, so dfranzen's hygiene passed it into history, and the model copied it
on every later call (19/19 broken). The server's parser has already dropped the code, so the call cannot be rewritten;
it is dropped like an invalid-JSON call (reasoning and content stay) and the follow-up names why.

``apply(src)`` patches a bundle's ``src/arena`` (v8.5p source) in place; every anchor must match exactly once.
"""
from pathlib import Path

CD_PATCHES = [
    ('OPTIONAL = {"keep_tokens": None}\n',
     'OPTIONAL = {"keep_tokens": None, "drop_unknown_tools": None}\n'),
    ('        raise ValueError("drain_half keep_tokens is an integer >= 1024")\n',
     '        raise ValueError("drain_half keep_tokens is an integer >= 1024")\n'
     '    if "drop_unknown_tools" in result and type(result["drop_unknown_tools"]) is not bool:\n'
     '        raise ValueError("drain_half drop_unknown_tools is true/false")\n'),
    ('MALFORMED_CALL_FOLLOWUP = (\n',
     'NAME_CALL_FOLLOWUP = (\n'
     '    "Your previous tool call was malformed (it named no available tool or left out its arguments), so nothing "\n'
     '    "was run. Emit exactly one `python` tool call with your code in its `code` argument. "\n'
     ')\n'
     'MALFORMED_CALL_FOLLOWUP = (\n'),
]
CD_APPEND = '''

def unknown_tool_calls(message, tools):
    """One line per tool call of ``message`` that names no tool of ``tools`` (OpenAI tool schemas) or leaves out one
    of that tool's required arguments (drain_half ``drop_unknown_tools``; invalid JSON is ``malformed_calls``')."""
    required = {}
    for tool in tools or []:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        if function.get("name"):
            required[function["name"]] = list((function.get("parameters") or {}).get("required") or [])
    errors = []
    for call in message.get("tool_calls") or []:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        name = str(function.get("name", ""))
        if name not in required:
            errors.append(f"unknown tool name {name[:80]!r}")
            continue
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                continue
        missing = [key for key in required[name] if not isinstance(arguments, dict) or key not in arguments]
        if missing:
            errors.append(f"{name}: missing required arguments {missing}")
    return errors
'''

DUCK_PATCHES = [
    ('            result = self.without_malformed_calls(result)\n',
     '            result = self.without_malformed_calls(result, tools if self.drain.get("drop_unknown_tools") else None)\n'),
    ('    def without_malformed_calls(self, result):\n',
     '    def without_malformed_calls(self, result, tools=None):\n'),
    ('        errors = malformed_calls(result.message)\n',
     '        errors, kind = malformed_calls(result.message), "json"\n'
     '        if not errors and tools:  # drain_half drop_unknown_tools (throughput arm, 20 Oct)\n'
     '            from .context_drain import unknown_tool_calls\n'
     '\n'
     '            errors, kind = unknown_tool_calls(result.message, tools), "name"\n'),
    ('        self.malformed_pending = "length" if str(result.finish_reason or "").strip() == "length" else "json"\n',
     '        self.malformed_pending = "length" if str(result.finish_reason or "").strip() == "length" else kind\n'),
    ('        from .context_drain import MALFORMED_CALL_FOLLOWUP, TRUNCATED_CALL_FOLLOWUP\n',
     '        from .context_drain import MALFORMED_CALL_FOLLOWUP, NAME_CALL_FOLLOWUP, TRUNCATED_CALL_FOLLOWUP\n'),
    ('        prefix = TRUNCATED_CALL_FOLLOWUP if pending == "length" else MALFORMED_CALL_FOLLOWUP\n',
     '        prefix = (TRUNCATED_CALL_FOLLOWUP if pending == "length" else NAME_CALL_FOLLOWUP if pending == "name"\n'
     '                  else MALFORMED_CALL_FOLLOWUP)\n'),
]


def _patch(path, patches, append=""):
    text = path.read_text()
    for old, new in patches:
        if text.count(old) != 1:
            raise SystemExit(f"{path}: anchor found {text.count(old)} times: {old[:80]!r}")
        text = text.replace(old, new)
    path.write_text(text + append)
    compile(path.read_text(), str(path), "exec")


def apply(src):
    """Patch ``src/arena`` of a source tree (a bundle's or a checkout's) in place."""
    arena = Path(src) / "arena"
    if "unknown_tool_calls" in (arena / "context_drain.py").read_text():
        raise SystemExit("already patched")
    _patch(arena / "context_drain.py", CD_PATCHES, CD_APPEND)
    _patch(arena / "duck.py", DUCK_PATCHES)
