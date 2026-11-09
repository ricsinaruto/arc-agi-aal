"""Package the Pennyroyal SGLang image's runtime for Kaggle (dataset arc-sglang-runtime).

Pulls ``ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.1`` with Docker, exports ``/opt/pennyroyal``
(fork checkout + uv venv), ``/opt/python`` (the venv's interpreter) and a CUDA 13.3 subset
into a staging tree inside a Linux container (paths and case preserved), prunes what the
server never reads, verifies the package pins, and writes one gzip tar stream split into
``.blob`` parts with a manifest. Nothing is uploaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.1"
INDEX_DIGEST = "sha256:76258532b1c01f7583fbe951f90b80903db931030291db9db4e5a68b28567866"  # the v2.5.1 tag
AMD64_MANIFEST_DIGEST = "sha256:7103728d3d5568e7fe56d684c179632dae5ccd62df48d20807e1cc779867b00c"
DATASET = "richardcsaky/arc-sglang-runtime"
ROOTS = ("opt/pennyroyal", "opt/python", "usr/local/cuda-13.3")
CUDA_SUBSET = ("bin", "nvvm", "include", "targets/x86_64-linux/include",
               "targets/x86_64-linux/lib/libcudadevrt.a", "targets/x86_64-linux/lib/libcudart_static.a")
PRUNE = ("test", "tests", "benchmark", "docs", "examples", "experimental", "sgl-model-gateway", "sgl-router",
         "sgl-kernel", ".github", "3rdparty", "assets", "evidence")
EXPECTED_PACKAGES = {
    "torch": "2.13.0+cu130",
    "torchvision": "0.28.0+cu130",
    "torchaudio": "2.11.0+cu130",
    "flashinfer-python": "0.6.17",
    "flashinfer-jit-cache": "0.6.17+cu130",
    "sglang-kernel": "0.4.6.post1+cu130",
    "triton": "3.7.1",
}
TEMPLATE_SHA256 = "e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2"
TOKEN_MAP_SHA256 = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"
PART_BYTES = 4 * 1024**3
# Not read by the server; terminfo has case-only name pairs that a case-insensitive host merges.
EXPORT_EXCLUDES = ("/share/terminfo/", "/share/man/", "/share/doc/", "/share/info/")


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, capture_output=True, **kwargs).stdout


def image_identity(image=IMAGE) -> dict:
    info = json.loads(run(["docker", "image", "inspect", image]))[0]
    digests = info.get("RepoDigests") or []
    if not any(d.endswith((INDEX_DIGEST, AMD64_MANIFEST_DIGEST)) for d in digests):
        raise RuntimeError(f"Image digest {digests} is not the pinned {INDEX_DIGEST} / {AMD64_MANIFEST_DIGEST}")
    labels = (info.get("Config") or {}).get("Labels") or {}
    return {
        "image": image,
        "index_digest": INDEX_DIGEST,
        "amd64_manifest_digest": AMD64_MANIFEST_DIGEST,
        "repo_digests": digests,
        "config_digest": info.get("Id"),
        "source_commit": labels.get("org.opencontainers.image.revision"),
        "layers": (info.get("RootFS") or {}).get("Layers"),
        "size_bytes": info.get("Size"),
    }


def export_paths(cuda=True) -> list[str]:
    paths = ["/opt/pennyroyal", "/opt/python", "/opt/pennyroyal-container-packages.txt"]
    if cuda:
        paths += [f"/usr/local/cuda-13.3/{sub}" for sub in CUDA_SUBSET]
    return paths


def case_collisions(names) -> list[tuple[str, str]]:
    """Member names that differ only by case (a case-insensitive host would merge them)."""
    seen: dict[str, str] = {}
    collisions = []
    for name in names:
        key = name.lower()
        if key in seen and seen[key] != name:
            collisions.append((seen[key], name))
        seen.setdefault(key, name)
    return collisions


def export_tree(stage: Path, image=IMAGE, *, cuda=True) -> None:
    """Copy the runtime roots out of the image without running it (docker create + cp)."""
    stage.mkdir(parents=True, exist_ok=True)
    container = run(["docker", "create", "--platform", "linux/amd64", image]).strip()
    try:
        for path in export_paths(cuda):
            archive = stage / (path.strip("/").replace("/", "_") + ".tar")
            with archive.open("wb") as out:
                subprocess.run(["docker", "cp", f"{container}:{path}", "-"], stdout=out, check=True)
            with tarfile.open(archive) as tar:
                members = [m for m in tar.getmembers()
                           if not any(x in "/" + m.name + "/" for x in EXPORT_EXCLUDES)]
                collisions = case_collisions(m.name for m in members)
                if collisions:
                    raise RuntimeError(f"Case collisions in {path}: {collisions[:5]}")
                # docker cp names members by the last path element; restore the parent.
                parent = stage / Path(path).parent.relative_to("/")
                parent.mkdir(parents=True, exist_ok=True)
                tar.extractall(parent, members=members, filter="fully_trusted")
            archive.unlink()
    finally:
        subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True)


def prune(stage: Path) -> list[str]:
    removed = []
    repo = stage / "opt/pennyroyal"
    git = repo / ".git"
    if git.exists():
        head = None
        try:
            head = run(["git", "-C", str(repo), "rev-parse", "HEAD"]).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
        (repo / "SOURCE_COMMIT.txt").write_text((head or "unknown") + "\n")
        shutil.rmtree(git)
        removed.append(".git")
    for name in PRUNE:
        target = repo / name
        if target.exists():
            shutil.rmtree(target)
            removed.append(name)
    for cache in list(stage.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)
    return removed


def _dist_version(site: Path, name: str) -> str | None:
    stem = "[-_]".join(re.escape(part) for part in name.split("-"))
    pattern = re.compile(rf"^{stem}-(.+)\.dist-info$", re.IGNORECASE)
    for entry in site.iterdir():
        match = pattern.match(entry.name)
        if match:
            return match.group(1)
    return None


def sha256_file(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify_stage(stage: Path, *, cuda=True) -> dict:
    repo = stage / "opt/pennyroyal"
    site = repo / ".venv/lib/python3.12/site-packages"
    problems = []
    packages = {name: _dist_version(site, name) for name in EXPECTED_PACKAGES}
    for name, expected in EXPECTED_PACKAGES.items():
        if packages[name] != expected:
            problems.append(f"{name}: {packages[name]} != {expected}")
    if not list(site.glob("flashinfer_jit_cache/**/fused_moe_120*")):
        problems.append("flashinfer jit cache lacks the SM120 fused MoE kernel")
    python = repo / ".venv/bin/python"
    if not python.is_symlink() or "/opt/python" not in str(Path(python.parent, python.readlink())):
        problems.append("venv python is not a link into /opt/python")
    cfg = repo / ".venv/pyvenv.cfg"
    if not cfg.is_file() or "home = /opt/python" not in cfg.read_text():
        problems.append("pyvenv.cfg home is not under /opt/python")
    finders = list(site.glob("__editable__*sglang*.py")) + list(site.glob("__editable__*sglang*.pth"))
    if not any("/opt/pennyroyal/python" in f.read_text() for f in finders):
        problems.append("editable sglang finder does not point at /opt/pennyroyal/python")
    pins = {}
    for label, rel, expected in (("template", "configs/pennyroyal/templates/froggeric-v22.5.jinja", TEMPLATE_SHA256),
                                 ("token_map", "configs/pennyroyal/frspec/flash-next-64k.pt", TOKEN_MAP_SHA256)):
        path = repo / rel
        pins[label] = sha256_file(path) if path.is_file() else None
        if pins[label] != expected:
            problems.append(f"{label} sha256 {pins[label]} != {expected}")
    if cuda and not (stage / "usr/local/cuda-13.3/bin/nvcc").is_file():
        problems.append("nvcc missing from the CUDA subset")
    if problems:
        raise RuntimeError("Stage verification failed: " + "; ".join(problems))
    return {"packages": packages, "pins": pins}


class SplitWriter:
    """A file-like sink that rolls to a new part every ``part_bytes``."""

    def __init__(self, directory: Path, prefix: str, part_bytes: int):
        self.directory, self.prefix, self.part_bytes = directory, prefix, part_bytes
        self.parts: list[dict] = []
        self.stream = hashlib.sha256()
        self._file = None
        self._hash = None
        self._written = 0

    def _open(self):
        name = f"{self.prefix}-part-{len(self.parts):02d}.tar.gz.blob"
        self._file = (self.directory / name).open("wb")
        self._hash = hashlib.sha256()
        self._written = 0
        self.parts.append({"file": name})

    def write(self, data: bytes):
        view = memoryview(data)
        while view:
            if self._file is None:
                self._open()
            room = self.part_bytes - self._written
            chunk = view[:room]
            self._file.write(chunk)
            self._hash.update(chunk)
            self.stream.update(chunk)
            self._written += len(chunk)
            view = view[room:]
            if self._written >= self.part_bytes:
                self._close_part()
        return len(data)

    def _close_part(self):
        self._file.close()
        self.parts[-1].update(size=self._written, sha256=self._hash.hexdigest())
        self._file = None

    def close(self):
        if self._file is not None:
            self._close_part()


def write_parts(stage: Path, output: Path, *, prefix: str, part_bytes=PART_BYTES, roots=ROOTS) -> dict:
    import gzip

    output.mkdir(parents=True, exist_ok=True)
    sink = SplitWriter(output, prefix, part_bytes)
    total = 0
    with gzip.GzipFile(fileobj=sink, mode="wb", compresslevel=6, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            for root in roots:
                base = stage / root
                if not base.exists():
                    continue
                entries = sorted(p for p in base.rglob("*") if p.is_file() or p.is_symlink() or p.is_dir())
                for path in [base, *entries]:
                    info = tar.gettarinfo(str(path), arcname=str(path.relative_to(stage)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    if info.isfile():
                        total += info.size
                        with path.open("rb") as file:
                            tar.addfile(info, file)
                    else:
                        tar.addfile(info)
    sink.close()
    return {"parts": sink.parts, "stream_sha256": sink.stream.hexdigest(), "extracted_bytes": total}


def write_manifest(output: Path, *, identity, verification, parts, pruned, cuda) -> dict:
    manifest = {
        "schema": 1,
        **identity,
        "roots": list(ROOTS if cuda else ROOTS[:2]),
        "cuda_toolchain": {"shipped": cuda, "version": "13.3", "subset": list(CUDA_SUBSET)},
        "pins": verification["pins"],
        "packages": verification["packages"],
        "pruned": pruned,
        "packaged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **parts,
    }
    (output / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "dataset-metadata.json").write_text(json.dumps({
        "id": DATASET,
        "title": "ARC SGLang Pennyroyal runtime",
        "licenses": [{"name": "other"}],
        "description": f"Private runtime for the SGLang benchmark: /opt/pennyroyal, /opt/python and a CUDA 13.3 "
        f"subset from {identity['image']} ({identity['index_digest']}); gzip tar stream sha256 "
        f"{parts['stream_sha256']} in {len(parts['parts'])} .blob parts.",
    }, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "dist/sglang-runtime"))
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--part-bytes", type=int, default=PART_BYTES)
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--skip-export", action="store_true", help="Reuse an existing stage directory")
    args = parser.parse_args(argv)
    output = Path(args.output)
    stage = output / "stage"
    upload = output / "dataset-upload"
    cuda = not args.no_cuda
    if not args.skip_pull:
        subprocess.run(["docker", "pull", "--platform", "linux/amd64", IMAGE], check=True)
    identity = image_identity()
    if not args.skip_export:
        if stage.exists():
            shutil.rmtree(stage)
        export_tree(stage, cuda=cuda)
    pruned = prune(stage)
    verification = verify_stage(stage, cuda=cuda)
    if upload.exists():
        shutil.rmtree(upload)
    parts = write_parts(stage, upload, prefix="pennyroyal-v2.5.1", part_bytes=args.part_bytes,
                        roots=ROOTS if cuda else ROOTS[:2])
    manifest = write_manifest(upload, identity=identity, verification=verification, parts=parts, pruned=pruned, cuda=cuda)
    print(json.dumps({k: manifest[k] for k in ("image", "index_digest", "source_commit", "parts", "stream_sha256",
                                                "extracted_bytes", "packages")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
