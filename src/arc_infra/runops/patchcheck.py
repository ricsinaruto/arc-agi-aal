"""Runtime-patch applicability on CPU (CPU gate step "patch-check", 24 Oct: v6f died at boot when a hash-guarded vLLM
patch did not match the hero image's file). For every ``*_patch.py`` module of a bundle, its targets are checked
against the pristine pinned runtime tree (the MFU-2 runtime cache's tar parts on the runs volume, or a local site):

  - a module with ``patched_source(original)`` (one-argument form) and ``RELATIVE``: the edit must apply;
  - ``TARGET`` / ``TARGETS`` entries ({"relative", "sha256"}): the file's sha256 must match the guard;
  - ``ORIGINAL_SHA256`` + ``RELATIVE``: the file's sha256 must match.
Two modules editing one file are reported as "stacked": their guards hold only in boot order, which this check cannot
replay; the bundle's own ARC_SERVING_PATCH_CHECK boot (serving setup up to the vLLM start, CPU) covers that order.

    report = check(bundle_src, site)            # {"ok": bool, "patches": {module: {...}}, "stacked": {...}}
"""
import hashlib
import importlib.util
import inspect
import json
import sys
import tarfile
from pathlib import Path


def modules(bundle):
    src = Path(bundle) / "src"
    return sorted(p for d in ("arc_infra", "arena") for p in (src / d).glob("*_patch.py"))


def _load(path, bundle):
    sys.path[:0] = [str(Path(bundle) / "src")]
    try:
        spec = importlib.util.spec_from_file_location(f"{path.parent.name}._pc_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = path.parent.name  # relative imports resolve in the package
        spec.loader.exec_module(mod)
        return mod
    finally:
        del sys.path[0]


def targets(mod):
    """[(relative, expected sha or None)] the module edits."""
    out = []
    for name in ("TARGET", "TARGETS"):
        val = getattr(mod, name, None)
        for t in ([val] if isinstance(val, dict) else list(val or []) if isinstance(val, (list, tuple)) else []):
            if isinstance(t, dict) and t.get("relative"):
                out.append((t["relative"], t.get("sha256")))
    rel = getattr(mod, "RELATIVE", None)
    if isinstance(rel, str):
        out.append((rel, getattr(mod, "ORIGINAL_SHA256", None)))
    return out


def check(bundle, site, read=None):
    """``read(relative) -> bytes | None``: the pristine file (default: ``site / relative``)."""
    read = read or (lambda rel: (Path(site) / rel).read_bytes() if (Path(site) / rel).is_file() else None)
    report, by_file = {}, {}
    for path in modules(bundle):
        name = f"{path.parent.name}/{path.stem}"
        try:
            mod = _load(path, bundle)
        except Exception as e:  # noqa: BLE001
            report[name] = {"ok": False, "why": f"import failed: {e!r}"[:300]}
            continue
        ts = targets(mod)
        if not ts:
            report[name] = {"ok": True, "why": "no declared target (unchecked here)", "unchecked": True}
            continue
        problems, checked = [], []
        for rel, sha in ts:
            by_file.setdefault(rel, []).append(name)
            data = read(rel)
            if data is None:
                problems.append(f"{rel}: missing in the runtime")
                continue
            seen = hashlib.sha256(data).hexdigest()
            if sha and seen != sha:
                problems.append(f"{rel}: sha256 {seen[:12]} != guard {sha[:12]}")
                continue
            ps = getattr(mod, "patched_source", None)
            if ps and len(inspect.signature(ps).parameters) == 1 and getattr(mod, "RELATIVE", None) == rel:
                try:
                    ps(data)
                except Exception as e:  # noqa: BLE001
                    problems.append(f"{rel}: edit does not apply: {e}"[:300])
                    continue
            checked.append(rel)
        report[name] = {"ok": not problems, "why": "; ".join(problems) or f"{len(checked)} target(s) match",
                        "targets": [r for r, _ in ts]}
    stacked = {rel: names for rel, names in by_file.items() if len(names) > 1}
    return {"ok": all(r["ok"] for r in report.values()), "patches": report, "stacked": stacked}


def cache_reader(cache_dir):
    """read(relative) over a runtime cache key dir (part-*.tar, files under usr/local/lib/python3.12/dist-packages)."""
    prefix = "usr/local/lib/python3.12/dist-packages/"
    index = {}
    for part in sorted(Path(cache_dir).glob("part-*.tar")):
        with tarfile.open(part) as t:
            for m in t.getmembers():
                n = m.name.lstrip("./")
                if n.startswith(prefix) and m.isfile():
                    index[n[len(prefix):]] = (part, m.name)

    def read(rel):
        hit = index.get(rel)
        if not hit:
            return None
        with tarfile.open(hit[0]) as t:
            return t.extractfile(hit[1]).read()
    return read


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="runtime patch guards vs the pristine runtime")
    ap.add_argument("bundle")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--site")
    g.add_argument("--cache", help="a runtime cache key dir (part-*.tar)")
    a = ap.parse_args(argv)
    rep = check(a.bundle, a.site, read=cache_reader(a.cache) if a.cache else None)
    for name, r in sorted(rep["patches"].items()):
        print(f"{'ok  ' if r['ok'] else 'FAIL'} {name}: {r['why']}")
    for rel, names in rep["stacked"].items():
        print(f"stacked {rel}: {', '.join(names)} (order-dependent: the bundle's own patch-check boot covers it)")
    print(("ARC_PATCH_CHECK OK " if rep["ok"] else "ARC_PATCH_CHECK FAIL ")
          + json.dumps({k: v["why"] for k, v in rep["patches"].items() if not v["ok"]})[:600])
    return 0 if rep["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
