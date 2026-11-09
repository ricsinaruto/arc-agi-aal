"""Forked trainer files (trunk, 24 Oct: RL's scripts/posttrain/<f> and PTH's scripts/posttrain/pth/<f>): the functions
and methods present in both copies whose bodies differ, so a fix that should reach both (rank-symmetry clamp, MFU
levers) is visible when it lands in one. A report, never a failure; ci.py stores it with each result.

  fork_report.py [ROOT]      one line per differing function; JSON with --json
"""
import ast
import json
import sys
from pathlib import Path


def defs(path):
    """{qualified name: normalized body} of every function and method in a file."""
    tree = ast.parse(path.read_text())
    out = {}

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = child.body[1:] if (child.body and isinstance(child.body[0], ast.Expr)
                                          and isinstance(getattr(child.body[0], "value", None), ast.Constant)
                                          and isinstance(child.body[0].value.value, str)) else child.body
                out[prefix + child.name] = ast.dump(ast.Module(body=body, type_ignores=[]))
                walk(child, prefix + child.name + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
    walk(tree, "")
    return out


def report(root):
    root = Path(root)
    fork = root / "scripts/posttrain/pth"
    rows = {}
    for theirs in sorted(fork.glob("*.py")) if fork.is_dir() else []:
        mine = root / "scripts/posttrain" / theirs.name
        if not mine.is_file():
            continue
        a, b = defs(mine), defs(theirs)
        both = sorted(set(a) & set(b))
        rows[theirs.name] = {"differ": [n for n in both if a[n] != b[n]], "same": sum(a[n] == b[n] for n in both),
                             "rl_only": len(set(a) - set(b)), "pth_only": len(set(b) - set(a))}
    return rows


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in argv
    argv = [a for a in argv if a != "--json"]
    rows = report(argv[0] if argv else Path(__file__).resolve().parents[2])
    if as_json:
        return print(json.dumps(rows, indent=1))
    for name, r in rows.items():
        print(f"{name}: {len(r['differ'])} shared functions differ, {r['same']} same, {r['rl_only']} RL-only, "
              f"{r['pth_only']} PTH-only" + (": " + ", ".join(r["differ"][:40]) if r["differ"] else ""))
    if not rows:
        print("no forked files")


if __name__ == "__main__":
    main()
