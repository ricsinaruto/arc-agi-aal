"""Assemble generated games into the public ``environment_files`` layout.

A catalog module (``catalog/qNNN_<name>.py``) holds one game's rules as ``class Game(GridGame)`` plus
``NAME`` and ``TAGS``; its docstring is the designer's note (the hidden rules). The store writes

    <root>/environment_files/<prefix>/<version>/<prefix>.py   gridbase + game source, no docstrings
    <root>/environment_files/<prefix>/<version>/metadata.json same fields as the public games
    <root>/notes/<prefix>.json                                designer note, tags, solution

so the unchanged ``arc_agi`` local loader (and therefore the Duck harness and the arena) can play
them. The version is the first 8 hex digits of the game file's sha256; the game id is
``<prefix>-<version>`` and the class name is the loader's default (``Q001`` for ``q001``). The game
file carries no designer note, so an agent that could read it would still not see the rules stated.
"""

import ast
import datetime
import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATALOG = HERE / "catalog"
PREFIX_RE = re.compile(r"^q\d{3}$")


def catalog_modules():
    return sorted(p for p in CATALOG.glob("q[0-9][0-9][0-9]_*.py"))


def _strip_docstring(source):
    tree = ast.parse(source)
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        lines = source.splitlines(keepends=True)
        return "".join(lines[body[0].end_lineno:]).lstrip("\n")
    return source


def assemble(module_path, levels=None, prefix=None):
    """(prefix, game source, catalog metadata) for one catalog module. ``levels`` (a list of level
    specs, e.g. from the module's ``generate_level``) replaces the module's own ``LEVELS``; ``prefix``
    (default: the module's own) names the game, e.g. an episode ``q2NN`` of mechanic ``q003``."""
    module_path = Path(module_path)
    own = module_path.name.split("_")[0]
    if not PREFIX_RE.match(own):
        raise ValueError(f"Catalog modules are named qNNN_<name>.py, got {module_path.name}")
    prefix = prefix or own
    if not PREFIX_RE.match(prefix):
        raise ValueError(f"Game prefixes are qNNN, got {prefix}")
    catalog_source = module_path.read_text()
    tree = ast.parse(catalog_source)
    note = ast.get_docstring(tree) or ""
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("NAME", "TAGS"):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    base = _strip_docstring((HERE / "gridbase.py").read_text())
    body = _strip_docstring(catalog_source).replace("  # noqa: F821 - GridGame is inlined by the store", "")
    class_name = prefix[0].upper() + prefix[1:]
    fixed = f"    LEVELS = {levels!r}\n" if levels is not None else "    pass\n"
    source = (f"# Generated ARC-like game {prefix} (posttrain.games.store). MIT License.\n\n{base}\n\n{body}\n\n"
              f"class {class_name}(Game):\n{fixed}")
    return prefix, source, {"name": values.get("NAME", prefix), "tags": values.get("TAGS", []), "note": note}


def catalog_module(module_path):
    """The catalog module itself, with GridGame injected (for ``generate_level`` and tests)."""
    from posttrain.games import gridbase

    module_path = Path(module_path)
    module = types.ModuleType(f"posttrain_catalog_{module_path.stem}")
    module.__dict__["GridGame"] = gridbase.GridGame
    exec(compile(module_path.read_text(), str(module_path), "exec"), module.__dict__)
    return module


def load_class(prefix, source):
    """Import an assembled game source as a fresh module and return its game class."""
    name = f"posttrain_generated_{prefix}_{hashlib.sha256(source.encode()).hexdigest()[:8]}"
    module = types.ModuleType(name)
    module.__file__ = f"<{prefix}>"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.__dict__[prefix[0].upper() + prefix[1:]]


def write_game(root, prefix, source, meta, baseline_actions, extra_notes=None):
    """Write one game; returns its game id. ``baseline_actions`` is one count per level."""
    version = hashlib.sha256(source.encode()).hexdigest()[:8]
    game_id = f"{prefix}-{version}"
    folder = Path(root) / "environment_files" / prefix / version
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{prefix}.py").write_text(source)
    tags = ["keyboard_click" if "click" in meta["tags"] else "keyboard"]
    metadata = {"game_id": game_id, "title": prefix.upper(), "default_fps": 30, "tags": tags,
                "baseline_actions": [int(n) for n in baseline_actions], "local_dir": str(folder),
                "date_downloaded": datetime.datetime.now(datetime.UTC).isoformat()}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    notes = Path(root) / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / f"{prefix}.json").write_text(json.dumps({"game_id": game_id, **meta, **(extra_notes or {})}, indent=1) + "\n")
    return game_id


def spec_from_file(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
