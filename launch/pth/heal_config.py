"""Typed launch config for PTH's OPD rounds (pilot, infra 25 Oct; runops.launchcfg): every launch_c7k.sh knob is a field
of HealConfig, read from one JSON file with no defaults, checked before anything is reserved, and kept in the record.

    heal_config.py check CFG.json      strict load + PTH checks + launchcfg dry run (anchor, roster vs the play bundle,
                                       data on the volumes, caps gate, open attempt) + diff vs the previous attempt
    heal_config.py env CFG.json        launch_c7k.sh's environment:  ( eval "$(heal_config.py env C)" && bash launch_c7k.sh )
    heal_config.py remember CFG.json   after the round's records.py reserve: the config into its current attempt

cpu_gate.py takes it too: --config CFG.json --config-class launch/pth/heal_config.py:HealConfig (bundle_roster, checks).
Settings the launchers fix themselves (HCARDS, TO) are compared with gpu_count / max_seconds, not passed."""
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

MAIN = Path(os.environ.get("ARC_MAIN_CHECKOUT", "."))
if __name__ == "__main__":  # records live in the main checkout; use its runops
    sys.path.insert(0, str(MAIN / "src"))
from arc_infra.runops import launchcfg  # noqa: E402

WT = Path(os.environ.get("ARC_PTH_WORKTREES", str(MAIN / ".claude/worktrees")))
HEAL, PRUNE = WT / "sft-heal", WT / "sft-prune"


@dataclass(frozen=True)
class HealConfig(launchcfg.LaunchConfig):
    name: str                  # NAME_SET; run must be sft-heal-round-<name>
    dstart: bool               # DSTART=1: from D at step 0; else ck/res_run
    ck: int | None             # CK
    res_run: str | None        # RES_RUN
    hbundle: str               # HBUNDLE_SET, relative to the sft-heal worktree
    pb: str                    # PB_SET, sft-prune dist/<pb>
    plays: tuple[str, ...]     # PLAY_SET: live play first, then version-0 backlogs
    ready: tuple[str, ...] | None  # READY_SET
    hlr: float                 # HLR_SET
    hfwd: float | None         # HFWD_SET
    max_age: int               # MAX_AGE_SET
    vkl: float                 # VKL
    hwin: int | None           # HWIN_SET
    stream_chunk: int | None   # STREAM_CHUNK
    xe: tuple[str, ...]        # XE_SET: extra trainer env pairs
    xt: tuple[str, ...]        # XT_SET: extra teacher env pairs
    new_flags: str | None      # NEW_FLAGS (None: launch_c7k's own)
    pftag: str | None          # PFTAG_SET
    reuse_pf: bool             # REUSE_PF=1
    ack: str                   # ACK (ARC_RESTART_ACK)


def _num(v):  # 3.5e-6, not 3.5e-06: the text the launch line has always carried
    return re.sub(r"e([+-])0*(\d)", lambda m: "e" + m.group(1).replace("+", "") + m.group(2), repr(v)) \
        if isinstance(v, float) else str(v)


def env(cfg):
    """{VAR: value} exactly as launch_c7k.sh reads them (unset fields are left out)."""
    e = {"NAME_SET": cfg.name, "HBUNDLE_SET": cfg.hbundle, "PB_SET": cfg.pb, "PLAY_SET": ",".join(cfg.plays),
         "HLR_SET": _num(cfg.hlr), "MAX_AGE_SET": str(cfg.max_age), "VKL": _num(cfg.vkl),
         "XE_SET": ",".join(cfg.xe), "XT_SET": ",".join(cfg.xt), "ACK": cfg.ack, "AUTH_SET": cfg.authorization,
         "WHAT_SET": cfg.what}
    if cfg.dstart:
        e["DSTART"] = "1"
    opt = {"CK": cfg.ck, "RES_RUN": cfg.res_run, "READY_SET": ",".join(cfg.ready) if cfg.ready else None,
           "HFWD_SET": cfg.hfwd, "HWIN_SET": cfg.hwin, "STREAM_CHUNK": cfg.stream_chunk, "NEW_FLAGS": cfg.new_flags,
           "PFTAG_SET": cfg.pftag, "REUSE_PF": "1" if cfg.reuse_pf else None}
    e.update({k: _num(v) for k, v in opt.items() if v is not None})
    return e


def bundle_roster(cfg):
    """The game ids the play bundle actually plays (its comparison.json `games`, hash suffix dropped)."""
    games = json.loads((PRUNE / "dist" / cfg.pb / "configs/comparison.json").read_text())["games"]
    return [g.split("-")[0] for g in games]


def _modal(*args):
    import subprocess

    return subprocess.run([str(MAIN / ".venv/bin/modal"), "volume", *args], capture_output=True, text=True, timeout=120)


def _ls(spec):
    vol, _, path = spec.partition(":")
    r = _modal("ls", "--json", vol, path)
    rows = json.loads(r.stdout or "[]") if r.returncode == 0 else []
    return [str(i.get("Filename", i.get("filename", ""))).rsplit("/", 1)[-1] for i in rows]


def _index(spec):
    """The dir's model.safetensors.index.json weight_map files."""
    import tempfile

    vol, _, path = spec.partition(":")
    with tempfile.TemporaryDirectory() as t:
        if _modal("get", vol, f"{path}/model.safetensors.index.json", t).returncode:
            return None
        f = next(Path(t).rglob("model.safetensors.index.json"), None)
        return set(json.loads(f.read_text())["weight_map"].values()) if f else None


def shards_missing(cfg, ls=_ls, index=_index):
    """A model dir among the data must hold every shard its index names (or, without an index, every
    model-i-of-N shard): a non-empty dir can be a fetch in progress (25 Oct: expert-bf16 deleted, refetch running)."""
    out = []
    for spec in cfg.data:
        if ":" not in spec or spec.endswith((".pt", ".json")):
            continue
        names = set(ls(spec))
        if "model.safetensors.index.json" in names:
            want = index(spec)
            if want is None:
                out.append(f"data {spec}: model.safetensors.index.json unreadable")
            elif want - names:
                out.append(f"data {spec}: {len(want - names)} of {len(want)} indexed shards missing "
                           f"(e.g. {sorted(want - names)[0]})")
            continue
        shards = {m.group(1): int(m.group(2)) for n in names if (m := re.search(r"-(\d+)-of-(\d+)\.safetensors$", n))}
        if shards and len(shards) != max(shards.values()):
            out.append(f"data {spec}: {len(shards)} of {max(shards.values())} model shards and no index (incomplete copy)")
    return out


def checks(cfg, heal=None, prune=None, ls=_ls, index=_index):
    """PTH-specific problems ([] = fine): naming, start point, bundles, env pairs, launcher-fixed settings and
    complete model copies."""
    heal, prune = Path(heal or HEAL), Path(prune or PRUNE)
    out = shards_missing(cfg, ls, index)
    if cfg.run != f"sft-heal-round-{cfg.name}":
        out.append(f"run {cfg.run} != sft-heal-round-{cfg.name}")
    if cfg.dstart == (cfg.ck is not None):
        out.append("exactly one start point: dstart true, or ck (+ res_run)")
    for what in ("authorization", "ack", "what"):
        if not getattr(cfg, what).strip() or "PENDING" in getattr(cfg, what):
            out.append(f"{what}: not filled in ({getattr(cfg, what)[:40]!r})")
    if not (heal / cfg.hbundle / "rl1").is_dir():
        out.append(f"trainer bundle {heal / cfg.hbundle}: no rl1/")
    if not (prune / "dist" / cfg.pb / "configs/comparison.json").is_file():
        out.append(f"play bundle {prune / 'dist' / cfg.pb}: no configs/comparison.json")
    if not cfg.plays:
        out.append("plays: empty")
    for p in cfg.xe + cfg.xt:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*=[^,\s]*", p):
            out.append(f"env pair {p!r}: KEY=VALUE, no commas or spaces ('+' inside values)")
    c7k, osc = heal / "launch/pth/launch_c7k.sh", heal / "launch/pth/opd_stream_c.sh"
    if c7k.is_file() and osc.is_file():
        cards = re.search(r"\bHCARDS=(\d+)", c7k.read_text())
        to = re.search(r"TO=\$\{TO:-(\d+)\}", osc.read_text())
        if cards and int(cards.group(1)) != cfg.gpu_count:
            out.append(f"gpu_count {cfg.gpu_count} but launch_c7k.sh sets HCARDS={cards.group(1)}")
        if to and int(to.group(1)) != cfg.max_seconds:
            out.append(f"max_seconds {cfg.max_seconds} but opd_stream_c.sh's crash ceiling TO is {to.group(1)}")
    else:
        out.append(f"launchers missing under {heal / 'launch/pth'}")
    return out


def check(cfg, **kw):
    try:
        roster = bundle_roster(cfg)
    except (OSError, KeyError, ValueError) as e:
        roster, probs = None, [f"play bundle roster unreadable: {e!r}"[:200]]
    else:
        probs = []
    return probs + checks(cfg) + launchcfg.dry_run(cfg, roster_ids=roster, **kw)


def main(argv=None):
    a = (argv or sys.argv)[1:]
    if len(a) != 2 or a[0] not in ("check", "env", "remember"):
        sys.exit(__doc__)
    cfg = launchcfg.load(HealConfig, a[1])
    if a[0] == "env":
        return print("\n".join(f"export {k}={shlex.quote(v)}" for k, v in env(cfg).items()))
    if a[0] == "remember":
        launchcfg.remember(cfg)
        return print(f"config stored in {cfg.run}'s current attempt")
    probs = check(cfg, modal_bin=str(MAIN / ".venv/bin/modal"))
    print(launchcfg.report(cfg))
    if probs:
        print("CONFIG REFUSED:\n  " + "\n  ".join(probs))
        sys.exit(1)
    print(f"config {cfg.run} OK")


if __name__ == "__main__":
    main()
