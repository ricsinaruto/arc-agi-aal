"""prime2 c240 bundle: an arc2 bundle (serving, weights, study; v1-v3 base18, v4 conv1's, whose src has conv1's
prompts) with the roster played by the stock Prime Agent v0.3.3 binary (arc2.prime). Only the code below and the
trial block differ from the base.

usage: python scripts/prime2/build.py OUT --base <arc2 dist>/arc2-rtx-c240-hx-base18-v3b --wall 13000
           --prime-agent prime-agent.tar.gz --node node-v22.19.0-linux-x64.tar.xz --wheelhouse DIR --pip pip-25.2.whl
"""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True  # importing the overlay must not add files to the bundle
OVERLAY = ("src/arc2/prime.py", "src/arc2/prime_sandbox.py")
DISPATCH_OLD = """async def run_arc2(study, trial, *, source, output, server_dir, base_url, deadline_epoch, python=None):
    output = Path(output)
"""
DISPATCH_NEW = """async def run_arc2(study, trial, *, source, output, server_dir, base_url, deadline_epoch, python=None):
    if trial.get("prime") is not None:  # prime2 arm: the stock Prime Agent binary plays the roster (arc2.prime)
        from .prime import run_prime

        return await run_prime(study, trial, source=source, output=output, server_dir=server_dir,
                               base_url=base_url, deadline_epoch=deadline_epoch)
    output = Path(output)
"""
PRIME_AGENT_SHA = "prime-agent v0.3.3 bc6894c8 linux-x64 build (RSI gen 0)"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--base", required=True)
    ap.add_argument("--wall", type=int, default=13000)
    ap.add_argument("--prime-agent", required=True)
    ap.add_argument("--node", required=True)
    ap.add_argument("--wheelhouse", required=True)
    ap.add_argument("--pip", required=True)
    ap.add_argument("--prime", default="{}", help="JSON overrides of arc2.prime.DEFAULTS")
    a = ap.parse_args()
    out, base = Path(a.out), Path(a.base)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    for p in base.rglob("*"):  # hard links: the serving files are base18's bytes
        rel = p.relative_to(base)
        if p.is_dir() or "__pycache__" in rel.parts or rel.as_posix() in ("configs/comparison.json", "MANIFEST.json"):
            continue
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        os.link(p, out / rel)
    for rel in OVERLAY:
        shutil.copyfile(ROOT / rel, out / rel)
    trial_py = out / "src/arc2/trial.py"
    text = trial_py.read_text()
    if text.count(DISPATCH_OLD) != 1:
        raise SystemExit("base trial.py: run_arc2 head not found once")
    trial_py.unlink()  # a hard link into base18: replace, never edit in place
    trial_py.write_text(text.replace(DISPATCH_OLD, DISPATCH_NEW))
    prime = out / "prime"
    (prime / "runtime/wheelhouse").mkdir(parents=True)
    shutil.copyfile(a.prime_agent, prime / "prime-agent.tar.gz")
    shutil.copyfile(a.node, prime / "runtime" / Path(a.node).name)
    shutil.copyfile(a.pip, prime / "runtime" / Path(a.pip).name)
    for wheel in sorted(Path(a.wheelhouse).glob("*.whl")):
        shutil.copyfile(wheel, prime / "runtime/wheelhouse" / wheel.name)
    shutil.copytree(ROOT / "configs/prime2/player", prime / "player")
    study = json.loads((base / "configs/comparison.json").read_text())
    if study["trials"][0]["kind"] != "arc2" or len(study["trials"]) != 1:
        raise SystemExit("base is not a one-trial arc2 bundle")
    sys.path[:0] = [str(out / "src")]
    from arc2 import prime as prime_mod  # the overlay's own defaults

    overrides = json.loads(a.prime)
    unknown = set(overrides) - set(prime_mod.DEFAULTS)
    if unknown:
        raise SystemExit(f"unknown prime keys {sorted(unknown)}")
    study["name"] = out.name
    study["status"] = "prepared-not-launched"
    t = study["trials"][0]
    t["wall_seconds"] = a.wall
    t["prime"] = overrides
    t["arms"] = [{"name": "prime-agent-v0.3.3"}]
    conv = {**prime_mod.DEFAULTS, **overrides}["conv"]
    study["prime2"] = {"player": ("stock Prime Agent v0.3.3 (bc6894c8) rpc, one session per entry; "
                                  + ("conv1's prompts, kernel globals, answer workflow and closing submit (v4)" if conv
                                     else "--autonomous, Prime's prompt (v1-v3)")),
                       "base_bundle": base.name, "prime_agent_sha256": sha(prime / "prime-agent.tar.gz"),
                       "node_sha256": sha(prime / "runtime" / Path(a.node).name),
                       "task_prompt_sha256": sha(prime / "player/task-prompt.txt"),
                       "startup_sha256": sha(prime / "player/startup.py"), "build": PRIME_AGENT_SHA}
    (out / "configs").mkdir(exist_ok=True)
    (out / "configs/comparison.json").write_text(json.dumps(study, indent=2) + "\n")
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
    from build_arena import manifest

    ident = json.loads((base / "MANIFEST.json").read_text())
    ident.update(study=out.name, files=manifest(out))
    (out / "MANIFEST.json").write_text(json.dumps(ident, indent=2) + "\n")
    runpy.run_path(str(out / "bootstrap.py"))["verify"](out)
    print(out, sha(out / "MANIFEST.json"))


if __name__ == "__main__":
    main()
