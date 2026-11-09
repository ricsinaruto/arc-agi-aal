"""The one RL bundle builder: launch/rl/build.py CONFIG [--out DIST] writes DIST/<bundle>/ from the
config's source commit, never from a working tree:
  code     git archive of `source` (src/<packages>, whole) + config.files (trainer files, bootstraps, repo data)
  seeds    data not in git at `source` (wheels, generated games, start points), each checked against its sha256;
           ~/arc-rl-data/<name> seeds are fetched from arc-prolong-assets:rl-seeds/<name> on a miss
  study    configs/comparison.json, written whole from the config ($file values inlined after their hash check)
  MANIFEST sha256 of every file + the source commit + spec_sha256 (the config's bundle part)
The checkout must be clean and `source` pushed and CI-green (launch/bundle_gate.py, ARC_CI_SKIP to override, logged).
DIST defaults to $ARC_RL_DIST or ~/arc-rl-dist (outside every worktree; launch.py, the preflights and export_rl read it)."""
import argparse
import hashlib
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

STORE_PREFIX = "~/arc-rl-data/"  # sha-addressed seeds too big for git (wheels, generated games)
SEED_VOLUME, SEED_REMOTE = "arc-prolong-assets", "rl-seeds"  # their copy, fetched on a miss
DIST = Path(os.environ.get("ARC_RL_DIST") or Path.home() / "arc-rl-dist")
SKIP = ("__pycache__",)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_files(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and not set(p.relative_to(root).parts) & set(SKIP)
                  and p.suffix != ".pyc")


def tree_sha256(path):
    """A file's sha256, or a dir's: sha256 over "<relpath> <sha256>" lines of its files (no __pycache__)."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    lines = "".join(f"{p.relative_to(path).as_posix()} {sha256_file(p)}\n" for p in tree_files(path))
    return sha256_bytes(lines.encode())


def manifest(root):
    """MANIFEST.json files (as build_arena: every file but the manifest, notebooks and Kaggle metadata)."""
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(Path(root).rglob("*"))
            if p.is_file() and p.name != "MANIFEST.json" and p.suffix != ".ipynb"
            and p.name not in ("kernel-metadata.json", "build-manifest.json") and "dataset-upload" not in p.parts}


def git(*a, cwd=C.CHECKOUT, binary=False):
    r = subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode().strip()


def local(path):
    """Seed paths: "~/arc-rl-data/<name>-<sha12>" (the sha-addressed store, $ARC_RL_DATA overrides the root), else a
    checkout-relative or absolute path."""
    if path.startswith(STORE_PREFIX):
        return Path(os.environ.get("ARC_RL_DATA") or os.path.expanduser(STORE_PREFIX)) / path[len(STORE_PREFIX):]
    p = Path(os.path.expanduser(path))
    return p if p.is_absolute() else C.CHECKOUT / p


def spec_sha256(cfg):
    """The hash of what goes into the bundle (source, packages, files, seeds, study): launch, knobs, watcher and record
    fields can change without a rebuild or a new preflight."""
    d = C.launchcfg.to_dict(cfg)
    return sha256_bytes(json.dumps({k: d[k] for k in ("source", "packages", "files", "seeds", "study")},
                                   sort_keys=True).encode())


def check_study(out, study):
    """The study through the bundle's own settings parsers (rl_loop + mg, level bench), as the run will read it."""
    code = ("import json, sys\nfrom pathlib import Path\ns = json.loads(Path(sys.argv[1]).read_text())\n"
            "if 'rl_loop' in s:\n    from arena.rl_loop import settings, check_mg\n    settings(s['rl_loop'])\n"
            "    check_mg(s['rl_loop'].get('mg') or {})\n"
            "if 'level_bench' in s:\n    from arena.level_bench import settings\n    settings(s['level_bench'])\n")
    r = subprocess.run([sys.executable, "-c", code, str(out / "configs/comparison.json")], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": f"{out / 'src'}:{out / 'rl1'}", "PYTHONDONTWRITEBYTECODE": "1"})
    if r.returncode:
        raise SystemExit(f"study refused by the bundle's settings: {(r.stderr or r.stdout).strip().splitlines()[-1]}")


def build(cfg_path, dist=DIST, gate=True):
    cfg_path = Path(cfg_path).resolve()
    cfg = C.load(cfg_path)
    probs = C.checks(cfg)
    if probs:
        raise SystemExit("config: " + "; ".join(probs))
    if gate:
        sys.path.insert(0, str(C.CHECKOUT / "launch"))
        import bundle_gate
        bundle_gate.require(C.CHECKOUT, f"rl:{cfg.bundle}", ref=cfg.source)
    sha = git("rev-parse", f"{cfg.source}^{{commit}}")
    out = Path(dist) / cfg.bundle
    tmp = out.with_name(out.name + ".building")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    tar = git("archive", "--format=tar", sha, "--", *[f"src/{p}" for p in cfg.packages], binary=True)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(tmp, filter="data")
    for dest, rp in cfg.files.items():
        (tmp / dest).parent.mkdir(parents=True, exist_ok=True)
        (tmp / dest).write_bytes(git("show", f"{sha}:{rp}", binary=True))
    for s in cfg.seeds:
        src = fetch(s.path, s.sha256)
        got = tree_sha256(src)
        if got != s.sha256:
            raise SystemExit(f"seed {s.path}: sha256 {got[:12]} != {s.sha256[:12]}")
        dest = tmp / s.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest, ignore=shutil.ignore_patterns(*SKIP, "*.pyc"))
        else:
            shutil.copyfile(src, dest)
    study = inline(C.launchcfg.to_dict(cfg.study))
    (tmp / "configs").mkdir(exist_ok=True)
    (tmp / "configs/comparison.json").write_text(json.dumps(study, indent=2) + "\n")
    check_study(tmp, study)
    identity = {"files": manifest(tmp), "source": f"git {sha}", "git_commit": sha, "dependencies_prepared": True,
                "study": study.get("name"), "config": str(cfg_path.relative_to(C.CHECKOUT)) if cfg_path.is_relative_to(
                    C.CHECKOUT) else str(cfg_path), "spec_sha256": spec_sha256(cfg), "builder": "launch/rl/build.py"}
    (tmp / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    runpy.run_path(str(tmp / "bootstrap.py"))["verify"](tmp)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


def fetch(path, sha256, get=None):
    """A store seed missing here comes from the Modal volume (arc-prolong-assets:rl-seeds/<name>), hash-checked, then
    renamed into place; `launch/rl/build.py --put` uploads one."""
    dest = local(path)
    if dest.exists() or not path.startswith(STORE_PREFIX):
        return dest
    name = path[len(STORE_PREFIX):]
    tmp = dest.with_name(dest.name + ".fetching")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (get or _volume_get)(f"{SEED_REMOTE}/{name}", tmp)
    got = tmp / name if (tmp / name).exists() else tmp
    if tree_sha256(got) != sha256:
        raise SystemExit(f"seed {path}: fetched sha256 {tree_sha256(got)[:12]} != {sha256[:12]}")
    got.rename(dest)
    shutil.rmtree(tmp, ignore_errors=True)
    return dest


def _volume_get(remote, into):
    subprocess.run([str(C.MAIN / ".venv/bin/modal"), "volume", "get", "--force", SEED_VOLUME, remote, str(into)],
                   check=True, capture_output=True)


def put(path):
    """Upload a store seed (~/arc-rl-data/<name>) to the Modal volume, so any checkout can rebuild from it."""
    src = local(path)
    subprocess.run([str(C.MAIN / ".venv/bin/modal"), "volume", "put", "--force", SEED_VOLUME, str(src),
                    f"{SEED_REMOTE}/{src.name}"], check=True)


def inline(value):
    """The study with every {"$file", "sha256"} value replaced by that file's JSON (hash checked)."""
    if isinstance(value, dict):
        if set(value) == {"$file", "sha256"}:
            raw = local(value["$file"]).read_bytes()
            if sha256_bytes(raw) != value["sha256"]:
                raise SystemExit(f"study $file {value['$file']}: sha256 {sha256_bytes(raw)[:12]} != {value['sha256'][:12]}")
            return json.loads(raw)
        return {k: inline(v) for k, v in value.items()}
    if isinstance(value, list):
        return [inline(v) for v in value]
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?")
    ap.add_argument("--out", default=str(DIST))
    ap.add_argument("--put", metavar="SEED", help="upload a store seed (~/arc-rl-data/<name>) to the volume and exit")
    a = ap.parse_args(argv)
    if a.put:
        return put(a.put)
    out = build(a.config, a.out)
    m = json.loads((out / "MANIFEST.json").read_text())
    print(f"built {out} from {m['git_commit'][:10]}: {len(m['files'])} files, MANIFEST {sha256_file(out / 'MANIFEST.json')[:12]}")


if __name__ == "__main__":
    main()
