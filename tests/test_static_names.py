"""Names that fail only on branches the CPU tests never take (pth-coloc-v1, 23 Oct: an import inside the CPU branch made
``importlib`` local to the whole function, so the B300 branch raised NameError after the teacher load). Two checks
over the scripts a B300 launch runs: ruff F821/F823, and a function-local import nested in a branch (if/try/with/
for/while) whose name is also bound at module level (it shadows the module's name on every other branch).
``python tests/test_static_names.py FILE...`` checks a launcher outside the repo (exit 1 on a finding)."""
import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

POST = Path(__file__).resolve().parents[1] / "scripts" / "posttrain"
SCRIPTS = [POST / n for n in ("heal_kd.py", "heal_stream.py", "heal_delta.py", "rl1_trainer.py", "rl1_sparse.py",
                              "flash_qlora_job.py", "modal_rl1.py", "rl_job.py", "heal_preflight.py",
                              "modal_heal_tiny.py")]
BRANCHES = (ast.If, ast.Try, ast.With, ast.For, ast.While, ast.AsyncWith, ast.AsyncFor)


def module_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


def shadowed_imports(path):
    """Function-local imports inside a branch whose name is also a module-level name and is read in that function
    (or a nested one) outside the branch: on the other branches the read finds an unbound local or free name."""
    tree = ast.parse(Path(path).read_text())
    top = module_names(tree)
    found = []

    def inside(node, root):
        return any(n is node for n in ast.walk(root))

    for func in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        imports = []  # (name, the outermost branch holding the import, lineno)

        def scan(node, branch):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    continue  # its own scope (scanned as its own function)
                if branch is not None and isinstance(child, (ast.Import, ast.ImportFrom)):
                    for a in child.names:
                        name = (a.asname or a.name).split(".")[0]
                        if name in top:
                            imports.append((name, branch, child.lineno))
                scan(child, branch if branch is not None else (child if isinstance(child, BRANCHES) else None))

        scan(func, None)
        for name, branch, line in imports:
            reads = [n for n in ast.walk(func) if isinstance(n, ast.Name) and n.id == name and not inside(n, branch)]
            if reads:
                found.append(f"{Path(path).name}:{line}: branch-local import of module name {name!r} read at line "
                             f"{reads[0].lineno} outside the branch")
    return found


def ruff_undefined(paths):
    ruff = shutil.which("ruff")
    if ruff is None:
        return None
    r = subprocess.run([ruff, "check", "--select", "F821,F823", "--no-cache", "--output-format", "concise",
                        *map(str, paths)], capture_output=True, text=True)
    return [x for x in r.stdout.splitlines() if ": F8" in x]


def test_no_undefined_names_in_launch_scripts():
    got = ruff_undefined([p for p in SCRIPTS if p.exists()])
    if got is None:
        pytest.skip("ruff not installed")
    assert got == []


def test_no_branch_local_import_shadows_a_module_name():
    found = [f for p in SCRIPTS if p.exists() for f in shadowed_imports(p)]
    assert found == []


def test_the_coloc_v1_pattern_is_caught(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import importlib.util\n\ndef f(stub):\n    if stub:\n        import importlib\n"
                   "    def g():\n        return importlib.util\n    return g()\n")
    assert shadowed_imports(bad) and "importlib" in shadowed_imports(bad)[0]


if __name__ == "__main__":
    files = sys.argv[1:]
    out = [f for p in files for f in shadowed_imports(p)] + (ruff_undefined(files) or [])
    print("\n".join(out) or "clean")
    sys.exit(1 if out else 0)
