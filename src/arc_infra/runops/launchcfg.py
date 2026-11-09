"""Typed launch configs: every knob of a launch is a field of a frozen dataclass, read from one JSON file
with no silent defaults (a missing or unknown key is an error), checked before anything is reserved, and kept in the
run record per attempt so a relaunch prints its diff against the previous attempt.

    @dataclass(frozen=True)
    class HeroConfig(LaunchConfig):        # an arm's launcher declares its own fields on top of the common ones
        bundle: str
        resume_from: tuple[str, ...]

    cfg = load(HeroConfig, "launch/rl/runs/rl1-loop-hero-v6g.json")
    problems = dry_run(cfg)                # anchor, roster, data paths, gate; [] = launchable
    print(report(cfg))                     # the resolved config and its diff vs the run's previous attempt
    ... reserve, launch ...
    remember(cfg)                          # the config goes into the record's current attempt

Common fields:
  run, arm, platform, gpu, gpu_count, bound_usd, max_seconds, authorization, what,
  anchor: {"run": the base read's run id, or "in-run" (the base is read inside this run on the same bundle),
           "note": text}
  roster: {"name": text, "ids": [game/task ids], "size": n}   (size must equal len(ids))
  data: [paths]  local paths, or "<volume>:<path>" on a Modal volume; each must exist and be non-empty.
"""
import dataclasses
import json
import os
import re
import subprocess
import types
import typing
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Anchor:
    run: str  # the base read's run id, or "in-run"
    note: str


@dataclass(frozen=True)
class Roster:
    name: str
    ids: tuple[str, ...]
    size: int


@dataclass(frozen=True)
class LaunchConfig:
    run: str
    arm: str
    platform: str
    gpu: str
    gpu_count: int
    bound_usd: float
    max_seconds: int
    authorization: str
    what: str
    anchor: Anchor
    roster: Roster
    data: tuple[str, ...]


# --- strict loading ----------------------------------------------------------------------------------------------------
def _coerce(tp, val, where):
    origin = typing.get_origin(tp)
    if tp is typing.Any:
        return val
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if val is None and type(None) in args:
            return None
        errs = []
        for a in args:
            if a is type(None):
                continue
            try:
                return _coerce(a, val, where)
            except ConfigError as e:
                errs.append(str(e))
        raise ConfigError(f"{where}: {val!r} matches none of {tp} ({'; '.join(errs)})")
    if is_dataclass(tp):
        if not isinstance(val, dict):
            raise ConfigError(f"{where}: expected an object for {tp.__name__}, got {type(val).__name__}")
        return _build(tp, val, where)
    if origin is tuple:
        args = typing.get_args(tp)
        if not isinstance(val, list):
            raise ConfigError(f"{where}: expected a list, got {type(val).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, f"{where}[{i}]") for i, v in enumerate(val))
        if len(args) != len(val):
            raise ConfigError(f"{where}: expected {len(args)} items, got {len(val)}")
        return tuple(_coerce(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, val)))
    if origin is dict:
        kt, vt = typing.get_args(tp)
        if not isinstance(val, dict):
            raise ConfigError(f"{where}: expected an object, got {type(val).__name__}")
        return types.MappingProxyType({_coerce(kt, k, f"{where} key"): _coerce(vt, v, f"{where}.{k}")
                                       for k, v in val.items()})
    if tp is float:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {val!r}")
        return float(val)
    if tp in (int, str, bool):
        if type(val) is not tp:
            raise ConfigError(f"{where}: expected {tp.__name__}, got {val!r}")
        return val
    raise ConfigError(f"{where}: unsupported field type {tp}")


def _build(cls, data, where=""):
    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    missing = [n for n in names if n not in data]
    unknown = [k for k in data if k not in names]
    if missing or unknown:
        raise ConfigError(f"{where or cls.__name__}: missing {missing}, unknown {unknown} (no defaults: every field "
                          f"is written out)")
    return cls(**{n: _coerce(hints[n], data[n], f"{where}.{n}" if where else n) for n in names})


def from_dict(cls, data):
    for f in fields(cls):
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
            raise ConfigError(f"{cls.__name__}.{f.name} has a default: launch configs have none")
    cfg = _build(cls, data)
    if cfg.roster.size != len(cfg.roster.ids) or len(set(cfg.roster.ids)) != len(cfg.roster.ids):
        raise ConfigError(f"roster {cfg.roster.name}: size {cfg.roster.size} but {len(cfg.roster.ids)} ids "
                          f"({len(set(cfg.roster.ids))} distinct)")
    return cfg


def load(cls, path):
    return from_dict(cls, json.loads(Path(path).read_text()))


def to_dict(cfg):
    def plain(v):
        if is_dataclass(v):
            return {f.name: plain(getattr(v, f.name)) for f in fields(v)}
        if isinstance(v, (tuple, list)):
            return [plain(x) for x in v]
        if isinstance(v, (dict, types.MappingProxyType)):
            return {k: plain(x) for k, x in v.items()}
        return v
    return plain(cfg)


# --- diff vs the previous attempt --------------------------------------------------------------------------------------
def diff(old, new, prefix=""):
    """['field: old -> new', ...] between two plain configs (dicts)."""
    out = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new), key=str):
            out += diff(old.get(k, "<absent>"), new.get(k, "<absent>"), f"{prefix}.{k}" if prefix else str(k))
    elif old != new:
        out.append(f"{prefix}: {json.dumps(old)[:120]} -> {json.dumps(new)[:120]}")
    return out


def previous(run, root=None):
    """The config of the run's latest attempt that recorded one (records), or None."""
    from . import records

    rec = records.get(run, root) or {}
    for a in reversed(rec.get("attempts") or []):
        if a.get("config"):
            return a["config"]
    return rec.get("config")


def report(cfg, root=None):
    cur = to_dict(cfg)
    prev = previous(cfg.run, root)
    lines = [f"config {cfg.run}:", json.dumps(cur, indent=1)]
    if prev is None:
        lines.append("diff vs previous attempt: (first attempt)")
    else:
        d = diff(prev, cur)
        lines.append("diff vs previous attempt:" + ("".join(f"\n  {x}" for x in d) if d else " none"))
    return "\n".join(lines)


def remember(cfg, root=None):
    """Store the config in the run record's current attempt (after records.reserve)."""
    from . import records

    cur = to_dict(cfg)

    def fn(rec):
        if rec.get("attempts"):
            rec["attempts"][-1]["config"] = cur
        else:
            rec["config"] = cur
    return records.update(cfg.run, fn, root)


# --- dry run: the plan's anchor, roster and data, then the gate --------------------------------------------------------
def _volume_nonempty(spec, modal_bin):
    from . import volfetch

    vol, _, path = spec.partition(":")
    vol = volfetch.route(vol, path)  # validation/<run>/ of a per-run record is on its own volume
    try:
        r = subprocess.run([modal_bin, "volume", "ls", "--json", vol, path], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return f"cannot list {spec}: {e!r}"
    if r.returncode:
        return f"{spec}: not found ({(r.stderr or r.stdout).strip()[-120:]})"
    try:
        items = json.loads(r.stdout or "[]")
    except ValueError:
        return f"{spec}: unreadable listing"
    if not items:
        return f"{spec}: empty"
    size = str(items[0].get("Size", items[0].get("size", 1)))  # the CLI prints "23.5 KiB", "0 B"
    if len(items) == 1 and str(items[0].get("Type", items[0].get("type", ""))).lower() == "file" and \
            re.match(r"\s*0+(\.0*)?(\s|$)", size):
        return f"{spec}: empty file"
    return None


def check_data(paths, modal_bin="modal", volumes=("arc-prolong-runs",)):
    """Problems with the data sources: a local path must exist and be non-empty (a file > 0 bytes, a dir with an
    entry); "<volume>:<path>" is listed on the Modal volume."""
    out = []
    for p in paths:
        vol, sep, _ = p.partition(":")
        if sep and (vol in volumes or not Path(p).exists()) and "/" not in vol:
            err = _volume_nonempty(p, modal_bin)
            if err:
                out.append(f"data {err}")
            continue
        q = Path(os.path.expanduser(p))
        if not q.exists():
            out.append(f"data {p}: missing")
        elif q.is_file() and q.stat().st_size == 0:
            out.append(f"data {p}: empty file")
        elif q.is_dir() and not any(q.iterdir()):
            out.append(f"data {p}: empty dir")
    return out


def check_anchor(anchor, run, root=None):
    """The comparison anchor exists: "in-run", or a run whose record has ended (completed) or is reserved/running."""
    from . import records

    if anchor.run == "in-run":
        return []
    if anchor.run == run:
        return [f"anchor {anchor.run}: is this run itself (use \"in-run\")"]
    rec = records.get(anchor.run, root)
    if not rec or not rec.get("ledger"):
        return [f"anchor {anchor.run}: no run record (launch or schedule the base read first)"]
    st = records.state_of(rec)
    if st == "ended" and rec.get("end_cause") not in (None, "completed"):
        return [f"anchor {anchor.run}: ended {rec.get('end_cause')}, not completed (no usable base read)"]
    return []


def dry_run(cfg, *, root=None, roster_ids=None, modal_bin="modal", gate=True):
    """[] when the launch may go; otherwise every problem found. ``roster_ids``: the ids the bundle/launcher will
    actually run (e.g. read from the bundle's roster file), compared with cfg.roster."""
    from . import records

    probs = []
    if roster_ids is not None and tuple(roster_ids) != tuple(cfg.roster.ids):
        extra = sorted(set(roster_ids) - set(cfg.roster.ids))[:5]
        lack = sorted(set(cfg.roster.ids) - set(roster_ids))[:5]
        probs.append(f"roster {cfg.roster.name}: config {len(cfg.roster.ids)} ids vs bundle {len(roster_ids)} "
                     f"(bundle extra {extra}, missing {lack})")
    probs += check_anchor(cfg.anchor, cfg.run, root)
    probs += check_data(cfg.data, modal_bin=modal_bin)
    if gate:
        try:
            for w in records.gate(cfg.arm, cfg.gpu, cfg.gpu_count, cfg.bound_usd, platform=cfg.platform, root=root,
                                  allowance_exempt=getattr(getattr(cfg, "launch", None), "allowance_exempt", None) or None):
                print(f"GATE WARNING: {w}")
        except records.GateError as e:
            probs.append(f"gate: {e}")
        rec = records.get(cfg.run, root)
        if rec and rec.get("ledger") and records.state_of(rec) != "ended":
            probs.append(f"{cfg.run}: attempt {rec.get('attempt', 1)} is still {records.state_of(rec)}")
    return probs
