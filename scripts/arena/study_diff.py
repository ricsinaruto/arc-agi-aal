"""Leaf-by-leaf difference of two built arena studies (configs/comparison.json), e.g. a new build against run B's.

    uv run python scripts/arena/study_diff.py dist/<reference bundle> dist/<new bundle> [--ignore name ...]

Prints every changed, added and removed leaf path (lists compared as values). Record-only keys can be listed with
``--ignore`` (a path prefix).
"""

import argparse
import json
from pathlib import Path


def leaves(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(leaves(item, f"{prefix}.{key}" if prefix else str(key)))
        return out or {prefix: {}}
    return {prefix: value}


def diff(before, after, ignore=()):
    a, b = leaves(before), leaves(after)
    rows = []
    for path in sorted(set(a) | set(b)):
        if any(path == p or path.startswith(p + ".") for p in ignore):
            continue
        if path not in b:
            rows.append(("removed", path, a[path], None))
        elif path not in a:
            rows.append(("added", path, None, b[path]))
        elif a[path] != b[path]:
            rows.append(("changed", path, a[path], b[path]))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference")
    parser.add_argument("new")
    parser.add_argument("--ignore", nargs="*", default=[])
    args = parser.parse_args(argv)
    load = lambda p: json.loads((Path(p) / "configs/comparison.json").read_text())  # noqa: E731
    for kind, path, old, new in diff(load(args.reference), load(args.new), args.ignore):
        text = lambda v: json.dumps(v)[:300]  # noqa: E731
        print(f"{kind:8} {path}: {text(old)} -> {text(new)}" if kind == "changed" else f"{kind:8} {path}: "
              f"{text(new if kind == 'added' else old)}")


if __name__ == "__main__":
    main()
