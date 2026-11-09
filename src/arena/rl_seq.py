"""RL redesign loop: every GPU plays, then every GPU trains.

Play phase: the backlog (arena.rl_backlog) opens groups of ``group_size`` tries on its best-ranked starts; a try that
clears its level saves the next level's start (harvest snapshot) and frees its stream; new groups open as streams
free. After ``phase_groups`` full groups the phase stops (arena.phase): in-flight replies are aborted and resent after
the train step, harness and game processes stay up, try clocks stop. Train phase: the full groups accumulate (rewards,
selection and GiGPO as v7: Loop.v7_prepare), one optimizer step, checkpoint; the adapter reloads under the same name
with a new id (exact, every resumed try re-prefills under the new weights).

One served name (``adapter``, default "policy") from the start: step 0 serves the trainer's initial LoRA (B = 0, the
base model's outputs), so no request ever names a version. No version book, card-2 Mode B, explorers or in-run reads.

``mg.redesign``: {"backlog": rl_backlog settings, "games" (default rl_backlog.TRAIN), "history": source-relative rows file, "streams": concurrent tries,
"cards": play servers, one per card (default 1: the launcher's; cards 1.. boot here as copies of it), "servers": more
play server base URLs (tests), "max_steps", "warmup" (10), "adapter", "board_check" (true), "hread" ({"every": 8}: the
held-out read hand-off, arena.hread; off when absent), "drivers" (N > 0: groups play in N worker processes,
arena.rl_drivers; 0 / absent: in this process),
"defer_trainer": {"template": adapter dir} (stage H, from mfu2-rlboot's v7 defer_trainer),
"lr_probe": P1 on this loop's own step (``lr_probe``), "freeze_policy" (test legs: only the draft trains),
"init_checkpoint" (a fresh start loads this trainer checkpoint before step 0 is served)}.

Stage D (``mg.draft_weight`` > 0, default 0 = off; ``mg.draft_lr_scale``): the MTP draft head's LoRA trains beside the
policy in every step (rl1_draft; the policy's gradients and loss are those of a draft-off run) and its files ride in
every saved adapter folder; the play servers merge them into the served MTP draft at each adapter load
(ARC_DRAFT_LORA=1, required on every play server). The checkpoint carries the draft's LoRA and Adam state; the seq
state records the draft setting, so a resume keeps it. Each phase row carries the servers' MTP acceptance and gen
tok/s over that phase (``mtp``; also ``train-metrics.jsonl``).

defer_trainer: a fresh start plays its first phase before the trainer has loaded. Step 0 serves an all-zero adapter
(``zero_adapter``: the template's names, shapes and dtypes, every byte 0) under the served name: its LoRA delta is
exactly 0, as the trainer's own initial adapter (B = 0), so the outputs are the same. The trainer process starts at
once with a GPU gate (RL1_GPU_GATE): its CPU part (imports, processor, encoder pool) runs during the first phase; its
card part waits for the first switch, when every server sleeps (an awake server holds ~215 of 268 GiB: the trainer's
~81 GiB does not fit beside it). A resume, or a missing template, takes the usual start (trainer first).
"""
import asyncio
import copy
import json
import os
import re
import shutil
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from . import boot_timeline
from . import harvest as harvest_mod
from . import level_bench, phase, rl_backlog

KEYS = {"backlog", "cards", "games", "history", "streams", "servers", "max_steps", "warmup", "adapter", "board_check",
        "hread", "defer_trainer", "card_env", "whole_segment", "kv_golden", "lr_probe", "score_check", "r3_check", "freeze_policy", "init_checkpoint",
        "replay_groups", "play_only", "admit_log_s", "seed_from", "drivers", "one_call", "recycle", "host_close_gib",
        "minibatches", "control"}
PROBE_KEYS = {"rates", "steps", "probes", "warmup"}
DEFAULTS = {"history": "launch/rl/data/v85p-levels.json", "warmup": 10, "adapter": "policy", "board_check": True,
            "host_close_gib": 190.0}


def settings(mg):
    r = (mg or {}).get("redesign")
    if not r:
        return None
    r = r if isinstance(r, dict) else {}
    unknown = set(r) - KEYS
    if unknown:
        raise ValueError(f"mg.redesign: unknown keys {sorted(unknown)}")
    out = {**DEFAULTS, **r}
    if not out.get("streams"):
        raise ValueError("mg.redesign.streams is required")
    if out.get("max_steps") is None or int(out["max_steps"]) < 0:  # 0: a warm-up leg (boot, cache save, trainer, end)
        raise ValueError("mg.redesign.max_steps is required (>= 0)")
    held = sorted(set(out.get("games") or rl_backlog.TRAIN) & set(rl_backlog.HELD_OUT))
    if held:  # the held-out 10 (5 hardest + the 5 left out) are read in hread only, never trained
        raise ValueError(f"mg.redesign.games: held-out games {held} never train")
    card_env(out)
    probe_settings(out)
    if out.get("drivers") is not None:
        from .rl_drivers import count

        count(out)
    if out.get("freeze_policy") not in (None, False):
        if out["freeze_policy"] is not True:
            raise ValueError("mg.redesign.freeze_policy is true or false")
        if not draft_on(mg):
            raise ValueError("mg.redesign.freeze_policy needs mg.draft_weight > 0 (a frozen policy alone trains nothing)")
        if out.get("lr_probe"):
            raise ValueError("mg.redesign.freeze_policy with lr_probe: the probe's rates train the policy")
    mb = out.get("minibatches")
    if mb is not None and (type(mb) is not int or mb < 1 or (mb > 1 and not out.get("one_call"))):
        raise ValueError("mg.redesign.minibatches is an integer >= 1 (> 1 needs one_call)")
    cg = out.get("control")
    if cg is not None and (not isinstance(cg, dict) or set(cg) - {"every", "tries"}
                           or any(type(cg.get(k, 1)) is not int or cg.get(k, 1) < 1 for k in ("every", "tries"))):
        raise ValueError('mg.redesign.control is {"every": updates, "tries": n} (integers >= 1)')
    rc = out.get("recycle")
    if rc is not None and (not isinstance(rc, dict) or set(rc) - {"every", "host_gib"} or not rc):
        raise ValueError('mg.redesign.recycle is {"every": switches, "host_gib": GiB} (either or both)')
    rg = out.get("replay_groups")
    if rg is not None:
        if (not (isinstance(rg, dict) and rg.get("roots") and int(rg.get("groups") or 0) > 0
                 and int(rg.get("updates", 1)) > 0) or set(rg) - {"roots", "groups", "min_tries", "repeat", "updates", "longest"}):
            raise ValueError("mg.redesign.replay_groups is {roots: [rl dirs], groups: n > 0, min_tries: m, repeat: bool, "
                             "updates: k > 0}")
        if out.get("lr_probe"):
            raise ValueError("mg.redesign.replay_groups with lr_probe: one or the other")
    sf = out.get("seed_from")
    if sf is not None:
        if not (isinstance(sf, dict) and isinstance(sf.get("root"), str) and sf["root"].startswith("/")
                and int(sf.get("groups") or 0) > 0) or set(sf) - {"root", "groups", "min_tries"}:
            raise ValueError("mg.redesign.seed_from is {root: absolute run dir with seq_state.json, groups: n > 0, "
                             "min_tries: m}")
        if out.get("replay_groups") or out.get("lr_probe") or out.get("play_only"):
            raise ValueError("mg.redesign.seed_from with replay_groups / lr_probe / play_only: one or the other")
    ckpt = out.get("init_checkpoint")
    if ckpt is not None and not (isinstance(ckpt, str) and ckpt.startswith("/")):
        raise ValueError("mg.redesign.init_checkpoint is an absolute trainer checkpoint folder (op_checkpoint's)")
    return out


def draft_on(mg):
    """Stage D: the MTP draft's LoRA trains (mg.draft_weight > 0)."""
    return float((mg or {}).get("draft_weight") or 0) > 0


def draft_state(mg):
    """The draft setting a seq state records (None: off)."""
    if not draft_on(mg):
        return None
    return {"weight": float(mg["draft_weight"]), "lr_scale": float(mg.get("draft_lr_scale") or 1.0)}


def draft_serving(envs):
    """server_envs records -> problems: a trained draft is served only by a server that merges it at load
    (ARC_DRAFT_LORA=1). An unreadable record is not a problem here (the caller alerts)."""
    out = []
    for name, rec in envs.items():
        env = rec.get("env")
        if not isinstance(env, dict) or rec.get("error") or env.get("error"):
            continue
        if env.get("ARC_DRAFT_LORA") != "1":
            out.append(f"{name}: ARC_DRAFT_LORA {env.get('ARC_DRAFT_LORA')} (not 1): the trained draft would not be "
                       "served")
    return out


SPEC = ("vllm:generation_tokens_total", "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens_per_pos_total:0")


async def spec_counters(url, get=None):
    """One server's generation and MTP counters ({SPEC name: value}; per-position: position 0), or None."""
    from arc_infra.metrics import parse_metrics

    root = url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        if get is None:
            import httpx

            async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                text = (await client.get(root + "/metrics")).text
        else:
            text = await get(root + "/metrics")
    except Exception:  # noqa: BLE001 - no metrics: no reading
        return None
    values = parse_metrics(text)
    return {k: values[k] for k in SPEC if k in values}


async def spec_snapshot(urls):
    return {"t": time.time(), "servers": list(await asyncio.gather(*(spec_counters(u) for u in urls)))}


def mtp_row(before, after, seconds):
    """A phase's MTP read over every server (counter deltas): accepted / drafted tokens (``accept``, as the held-out
    read's infra.draft_acceptance), position-1 acceptance (``pos1``), tokens per target step (``mean_len``), generated
    tokens and their rate. None without any reading; ``accept`` None when nothing was drafted (MTP off)."""
    tot, seen = dict.fromkeys(SPEC, 0.0), False
    for a, b in zip(before, after):
        if a is None or b is None:
            continue
        seen = True
        for k in SPEC:
            tot[k] += max(0.0, float(b.get(k, 0.0)) - float(a.get(k, 0.0)))
    if not seen:
        return None
    gen, drafts, dtok, acc, pos = (tot[k] for k in SPEC)
    return {"accept": round(acc / dtok, 4) if dtok else None, "pos1": round(pos / drafts, 4) if drafts else None,
            "mean_len": round(1 + acc / drafts, 4) if drafts else None, "drafts": int(drafts),
            "gen_tokens": int(gen), "gen_tok_s": round(gen / seconds, 1) if seconds > 0 else None}


def phase_mtp(loop, seq, before, after, need_draft):
    """The play phase's MTP row (version: the adapter it played), also a train-metrics.jsonl row; a trained draft on
    servers that drafted nothing all phase alerts (MTP off: the draft is never used)."""
    row = mtp_row(before["servers"], after["servers"], after["t"] - before["t"])
    if row is None:
        return None
    row["version"] = seq.version
    if need_draft and not row["drafts"] and row["gen_tokens"]:
        loop.feed.emit("alert", why=f"draft trains but the play servers drafted nothing (MTP off?) at v{seq.version}")
    m = {"infra.draft_acceptance": row["accept"], "infra.draft_pos1": row["pos1"], "infra.gen_tok_s": row["gen_tok_s"],
         "train.policy_version": seq.version}
    try:
        path = Path(loop.output) / "train-metrics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({"t": after["t"], "m": m}) + "\n")
    except OSError:
        pass
    return row


def probe_settings(cfg):
    """mg.redesign.lr_probe (P1 before a run, RL 26 Oct): {"rates": [lr, ...], "steps": optimizer steps per rate,
    "probes": [{"folder": recorded bench folder (absolute, or bundle-relative), "labels": [try label, ...]}],
    "warmup": steps (default: the run's warmup)}. None when off. The trainer settings (windows, CP, tail_ckpt) are
    the run's own: the probe steps exactly as the run does."""
    spec = cfg.get("lr_probe")
    if not spec:
        return None
    if not isinstance(spec, dict) or set(spec) - PROBE_KEYS:
        raise ValueError("mg.redesign.lr_probe is {rates, steps, probes, warmup}")
    rates = spec.get("rates")
    if not isinstance(rates, list) or not rates or not all(
            isinstance(r, (int, float)) and not isinstance(r, bool) and r > 0 for r in rates):
        raise ValueError("mg.redesign.lr_probe.rates is a non-empty list of positive learning rates")
    if type(spec.get("steps")) is not int or spec["steps"] < 1:
        raise ValueError("mg.redesign.lr_probe.steps is a positive integer (optimizer steps per rate)")
    probes = spec.get("probes")
    if not isinstance(probes, list) or not probes or not all(
            isinstance(x, dict) and set(x) == {"folder", "labels"} and x["labels"] for x in probes):
        raise ValueError("mg.redesign.lr_probe.probes is [{folder, labels}] (recorded held-out tries scored after "
                         "every step)")
    if spec.get("warmup") is not None and (type(spec["warmup"]) is not int or spec["warmup"] < 0):
        raise ValueError("mg.redesign.lr_probe.warmup is a non-negative integer")
    return spec


def card_env(cfg):
    """mg.redesign.card_env (RL 26 Oct, a same-host A/B on one N-card run): {"<card>": {KEY: value or null}}, card 0's
    merged into the trial's server_env at the first server's start (scripts/arena/bootstrap.py), card i >= 1's over
    the copy of server 1's environment (rl_loop.card_environment; null removes a key). Returns {int card: env}."""
    raw = cfg.get("card_env") or {}
    if not isinstance(raw, dict):
        raise ValueError("mg.redesign.card_env is {card: {KEY: value or null}}")
    cards, out = int(cfg.get("cards") or 1), {}
    for card, env in raw.items():
        if not str(card).isdigit() or not 0 <= int(card) < cards or not isinstance(env, dict):
            raise ValueError(f"mg.redesign.card_env: card {card!r} outside 0..{cards - 1}, or not a {{KEY: value}} map")
        bad = [k for k, v in env.items() if not isinstance(k, str) or not (v is None or isinstance(v, (str, int, float)))]
        if bad:
            raise ValueError(f"mg.redesign.card_env {card}: values are strings, numbers or null ({bad})")
        out[int(card)] = {k: (None if v is None else str(v)) for k, v in env.items()}
    return out


ZERO_DTYPES = {"BF16", "F16", "F32", "F64"}  # all-zero bytes are 0.0 in these


def zero_adapter(template, dest):
    """``dest``: ``template``'s adapter (config, tensor manifest, safetensors header) with every tensor byte zero.
    Returns the tensor count. Refuses a non-float tensor (zero bytes would not be a zero delta's meaning)."""
    import struct

    template, dest = Path(template), Path(dest)
    with open(template / "adapter_model.safetensors", "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        head = f.read(size)
    header = json.loads(head)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    bad = sorted({v["dtype"] for v in tensors.values()} - ZERO_DTYPES)
    if bad:
        raise ValueError(f"zero adapter: dtypes {bad} in {template}")
    data = max((v["data_offsets"][1] for v in tensors.values()), default=0)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter-tensors.json"):
        if (template / name).is_file():
            shutil.copyfile(template / name, dest / name)
    with open(dest / "adapter_model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", size) + head)
        f.write(b"\0" * data)
    (dest / "zero-adapter.json").write_text(json.dumps({"template": str(template), "tensors": len(tensors)}))
    return len(tensors)


def pending_only(root):
    """A root whose first update died before any state landed (real-v5 27 Oct: stopped with s0001.partial): no
    seq_state.json, a seq_pending.json of step 1."""
    try:
        pend = json.loads((Path(root) / "seq_pending.json").read_text())
    except (OSError, ValueError):
        return False
    return not (Path(root) / "seq_state.json").is_file() and int(pend.get("step", -1)) == 1


def resume_state_exists():
    roots = [r for r in os.environ.get("ARC_RL1_RESUME", "").split(",") if r]
    return any((Path(r) / "seq_state.json").is_file() or pending_only(r) for r in roots)


PLAY_EVERY_S = 600.0  # checklist item 6: a play phase (~100 min) has a restart point <= 10 min old


def saved_folder(root, where):
    """A saved group's folder from its {"rel"} (under the root it was saved in) or {"abs"} (a group a resumed leg
    carried from an earlier root) entry; an /runs/ path gone from this container is looked for under /in/<volume>/."""
    if "rel" in where:
        return Path(root) / where["rel"]
    p = Path(where["abs"])
    if not p.exists() and str(p).startswith("/runs/"):
        p = next((v / p.relative_to("/runs") for v in sorted(Path("/in").glob("*")) if (v / p.relative_to("/runs")).exists()), p)
    return p


def written_root(root):
    """The run root a state read from ``root`` was written under, for states that do not record it (bundles before
    5a42301c7, e.g. real-v11's): a preemption's <root>-lost-<epoch> was <root>; a per-run container reads an earlier
    run at /in/<volume>/<rel>, written as /runs/<rel>."""
    p = re.sub(r"-lost-\d+$", "", str(root).rstrip("/"))
    parts = Path(p).parts
    return str(Path("/runs", *parts[3:])) if parts[1:2] == ("in",) and len(parts) > 3 else p


def remap_root(obj, old, new):
    """``obj`` (JSON) with paths under ``old`` (the run root it was saved in) moved to ``new`` (the root it was read
    from): a preemption restart renames the root to <root>-lost-<epoch> and makes a new, empty one of the old name."""
    if not old or str(old).rstrip("/") == str(new).rstrip("/"):
        return obj
    a, b = json.dumps(str(old).rstrip("/") + "/")[1:-1], json.dumps(str(new).rstrip("/") + "/")[1:-1]
    return json.loads(json.dumps(obj).replace(a, b))


def prespawned_gate(loop):
    """{"gate": path} when the launcher pre-spawned a gated, host-preloading trainer (arena.trainer_prespawn: a resume
    or a start without the deferred first phase): the servers sleep now, so its gate opens and start_trainer adopts it
    (same command and environment); else {} (a plain load)."""
    from . import trainer_prespawn

    if not (Path(loop.work) / "trainer" / trainer_prespawn.RECORD).is_file():
        return {}
    gate = Path(loop.work) / trainer_prespawn.GATE
    gate.write_text(json.dumps({"epoch": time.time()}))
    return {"gate": gate}


def defer_template(loop, cfg):
    """The defer_trainer template adapter dir, or None (no defer). Bundle-relative or absolute; a CPU preflight
    without one writes the stand-in trainer's adapter shape."""
    d = cfg.get("defer_trainer")
    if d is None or d is False:
        return None
    d = d if isinstance(d, dict) else {}
    t = d.get("template")
    if t:
        path = Path(t) if Path(t).is_absolute() else Path(loop.source) / t
        return path if (path / "adapter_model.safetensors").is_file() else None
    if os.environ.get("ARC_PREFLIGHT_FAKE_TRAINER"):
        from .fake_trainer import adapter

        return Path(adapter(loop.work / "adapters" / "zero-template"))
    return None


SWITCHES = ("budget_floor", "budget_guard", "budget_censored")  # live on/off knobs of the curriculum's budget rule
MIX_KNOBS = ("game_cap", "min_update_tokens", "max_update_groups")  # off by default; 0 turns one off live
LIVE_KNOBS = ("streams", "phase_groups", "active_tries", "level_inflight") + SWITCHES  # checklist item 5: {"seq": {...}} in the run's knobs file


def live_knobs(path=None):
    """The ``seq`` section of the live knobs file (ARENA_KNOBS, synced from <run>/arena/live/knobs.json): only
    LIVE_KNOBS, as ints (budget_floor: a bool); {} when absent or unreadable (the run keeps its settings)."""
    path = path or os.environ.get("ARENA_KNOBS")
    try:
        data = json.loads(Path(path).read_text()) if path else {}
    except (OSError, ValueError):
        return {}
    sec = data.get("seq") if isinstance(data, dict) else None
    out = {}
    for k in LIVE_KNOBS:
        v = (sec or {}).get(k)
        if k in SWITCHES:  # a switch: true/false or 1/0
            if isinstance(v, (bool, int)) and v in (0, 1):
                out[k] = bool(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and int(v) > 0:
            out[k] = int(v)
    scale = (sec or {}).get("lr_scale")  # the run's lr x scale, read per update
    if isinstance(scale, (int, float)) and not isinstance(scale, bool) and 0 < scale <= 1:
        out["lr_scale"] = float(scale)
    mb = (sec or {}).get("minibatches")  # (fallback 2): optimizer steps per update, read per update
    if isinstance(mb, int) and not isinstance(mb, bool) and mb >= 1:
        out["minibatches"] = mb
    ep = (sec or {}).get("epochs")  # passes over the update's batch, read per update
    if isinstance(ep, int) and not isinstance(ep, bool) and 1 <= ep <= 4:
        out["epochs"] = ep
    rc = (sec or {}).get("ratio_clip")  # the θ0 trust region of steps after the first: gspo | ppo | off
    if rc in RATIO_CLIPS:
        out["ratio_clip"] = rc
    budget = (sec or {}).get("ckpt_budget_gib")  # the trainer's activation budget (card terms), read per update
    if isinstance(budget, (int, float)) and not isinstance(budget, bool) and 150 <= budget <= 262:
        out["ckpt_budget_gib"] = float(budget)
    for k in MIX_KNOBS:  # the update mix (rl_backlog DEFAULTS): an int >= 0, 0 = off
        v = (sec or {}).get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            out[k] = v
    win = (sec or {}).get("lvl_window")  # the windows' tail cap, the peak of a fully checkpointed window (apply_knobs)
    if isinstance(win, int) and not isinstance(win, bool) and 16384 <= win <= 47104:
        out["lvl_window"] = win
    cap = (sec or {}).get("kv_cap")  # read at every guard poll, not only at a phase start
    if isinstance(cap, (int, float)) and not isinstance(cap, bool) and 0 < cap < 1:
        out["kv_cap"] = float(cap)
    stub = os.environ.get("ARC_PREFLIGHT_STUB") or ""
    if "phase_groups" in out and stub.startswith("{") and json.loads(stub).get("shrink"):
        # rl_loop.shrink_mg's 2 groups: the launch's knob (16) ran real-v12 N1/NK phase 1 to the 45-min ceiling
        out["phase_groups"] = min(out["phase_groups"], 2)
    if stub.startswith("{") and json.loads(stub).get("shrink"):  # a shrink's 2 tiny groups never reach a token floor
        out.pop("min_update_tokens", None)
    return out


def apply_knobs(loop, seq, knobs, pools=None):
    """At a play phase's start: ``active_tries`` = each play server's slot pool (capped by the server's
    max_num_seqs; the pools' KV-fill gate still admits), ``streams`` = the backlog's concurrent tries (capped by
    servers x slots), ``phase_groups`` = full groups per step, ``level_inflight`` = groups in flight per (game, level)
    for saved starts. Returns the changes {knob: [old, new]}."""
    pools = level_bench.SHARED_SLOTS_BY_URL if pools is None else pools
    cap = int((((getattr(loop, "study", None) or {}).get("server") or {}).get("max_num_seqs")) or 0)
    changed = {}
    if "active_tries" in knobs and pools:
        n = min(knobs["active_tries"], cap) if cap else knobs["active_tries"]
        for pool in pools.values():
            if pool.slots != n:
                changed["active_tries"] = [pool.slots, n]
                pool.slots = n
    if "streams" in knobs:
        per = max((p.slots for p in pools.values()), default=0)
        n = min(knobs["streams"], per * len(pools)) if per and pools else knobs["streams"]
        if seq.backlog.streams != n:
            changed["streams"] = [seq.backlog.streams, n]
            seq.backlog.streams = n
    if "phase_groups" in knobs and int(seq.backlog.cfg["phase_groups"]) != knobs["phase_groups"]:
        changed["phase_groups"] = [int(seq.backlog.cfg["phase_groups"]), knobs["phase_groups"]]
        seq.backlog.cfg["phase_groups"] = knobs["phase_groups"]
    if "level_inflight" in knobs and seq.backlog.cfg.get("level_inflight") != knobs["level_inflight"]:
        # (2) a resume's saved backlog cfg would otherwise keep its own value
        changed["level_inflight"] = [seq.backlog.cfg.get("level_inflight"), knobs["level_inflight"]]
        seq.backlog.cfg["level_inflight"] = knobs["level_inflight"]
    if "lvl_window" in knobs and isinstance(getattr(loop, "cfg", None), dict):
        # only down from the launch's value, which the launch's floor check passed (v17 a10: U25 peaked 253.9 on
        # n = 0 windows, out of ckpt_budget_gib's reach, and no other knob could lower it)
        top = int(getattr(seq, "lvl_window_top", 0) or loop.cfg.get("lvl_window") or knobs["lvl_window"])
        seq.lvl_window_top = top
        want = min(knobs["lvl_window"], top)
        if int(loop.cfg.get("lvl_window") or top) != want:
            changed["lvl_window"] = [loop.cfg.get("lvl_window"), want]
            loop.cfg["lvl_window"] = want
    for k in MIX_KNOBS:
        if k in knobs and seq.backlog.cfg.get(k) != (knobs[k] or None):
            changed[k] = [seq.backlog.cfg.get(k), knobs[k] or None]
            seq.backlog.cfg[k] = knobs[k] or None
    for k in SWITCHES:
        if k in knobs and bool(seq.backlog.cur.cfg.get(k)) != bool(knobs[k]):
            changed[k] = [bool(seq.backlog.cur.cfg.get(k)), bool(knobs[k])]
            seq.backlog.cur.cfg[k] = seq.backlog.cfg[k] = bool(knobs[k])
    return changed


BACKOFF = {"kv_peak": 0.85, "step": 20, "floor": 120}  # overnight runs back off on their own


def backoff(knobs, guard, state, slots, cfg=BACKOFF):
    """Phase-start auto back-off: when the last play phase had any preemption or a KV peak over cfg kv_peak, the
    per-server slots go down by cfg step (not below cfg floor); the cut stays until a knob put of active_tries
    (a new base). ``knobs``: this phase's live knobs; ``state``: {"base", "cut"} kept across phases; ``slots``: the
    pools' slots now. Returns (knobs to apply, the back-off row or None)."""
    if "active_tries" in knobs and knobs["active_tries"] != state.get("put"):
        state.update(base=knobs["active_tries"], put=knobs["active_tries"], cut=0)  # a fresh put resets the cut
    state.setdefault("base", slots)
    state.setdefault("cut", 0)
    hit = bool(guard) and (int(guard.get("preemptions") or 0) > 0 or int(guard.get("blind_polls") or 0) > 0
                           or float(guard.get("kv_peak") or 0.0) > cfg["kv_peak"]  # blind: fail closed
                           )
    row = None
    if hit and state["base"] - state["cut"] > cfg["floor"]:
        state["cut"] += cfg["step"]
        row = {"why": guard, "cut": state["cut"]}
    want = max(min(cfg["floor"], state["base"]), state["base"] - state["cut"])
    out = dict(knobs)
    if want != slots or "active_tries" in knobs:
        out["active_tries"] = want
    return out, row


def host_free_gib(meminfo="/proc/meminfo", cgroup="/sys/fs/cgroup"):
    """Host memory a new allocation can take, GiB: the cgroup limit less its anonymous memory, and MemAvailable; the
    smaller (rl1_trainer.host_free_bytes' rule). None when neither is readable."""
    out = []
    try:
        top = Path(cgroup, "memory.max").read_text().strip()
        rows = dict(l.split() for l in Path(cgroup, "memory.stat").read_text().splitlines() if l.count(" ") == 1)
        if top.isdigit():
            out.append((int(top) - int(rows.get("anon", 0))) / 2**30)
    except (OSError, ValueError):
        pass
    try:
        info = {l.split(":")[0]: l.split()[1] for l in Path(meminfo).read_text().splitlines() if len(l.split()) >= 2}
        out.append(int(info["MemAvailable"]) / 2**20)
    except (OSError, ValueError, KeyError):
        pass
    return min(out) if out else None


HOST_EVERY_S = 30.0


async def host_watch(loop, seq, close_gib, every_s=None, free_gib=None):
    """redesign.host_close_gib (real-v7 27 Oct: vLLM's host heaps grew ~20 GiB per M generated tokens and the switch
    adds ~50 GiB; MemAvailable is the gauge): during play, free host below it closes the phase with the groups already
    full (it trains on them), before the switch could run out. Off at 0; none full: it waits."""
    while float(close_gib or 0) > 0:
        await asyncio.sleep(HOST_EVERY_S if every_s is None else every_s)
        free = (free_gib or host_free_gib)()
        if free is not None and free < float(close_gib) and seq.backlog.full:
            loop.feed.emit("seq_host_close", step=seq.step, free_gib=round(free, 1), close_gib=close_gib,
                           groups=len(seq.backlog.full))
            seq.phase_done.set()
            return round(free, 1)


# G-long v4 N1/N2 (27 Oct): under 4 drivers' load the server's /metrics answered after 2-5 s; the guard polled once in
# 240 s and never saw the preemption it backs off on (G-long v3 on GPU: scrape timeouts with one API process too).
# The guard is the one /metrics reader (admit rows use its last read); it polls slower than its timeout, and a server it
# cannot read BLIND_AFTER times in a row is blind: on_blind(url, True) holds admission until a read comes back.
METRICS_TIMEOUT_S, GUARD_EVERY_S, BLIND_AFTER = 15.0, 20.0, 3
KV_HYST = 0.02  # kv_cap: a held server admits again below cap - KV_HYST
LAST_METRICS = {}  # url -> (epoch, parsed /metrics) of the guard's last good read
def mirror_root():
    """The run's volume copy of the trial's rl dir (where the end-of-run collection puts it), None off a run."""
    out = os.environ.get("ARC_VALIDATION_OUTPUT")
    return Path(out) / "arena" / "trials" / (os.environ.get("ARC_RL1_TRIAL") or "rl-loop") / "rl" if out else None


AHEAD_EVERY_S = 20.0  # one_call + early_ahead: the trainer's ahead file rewritten at most this often during play


def kv_want(cap, kv, was):
    """kv_cap's hold: on at kv >= cap, off again below cap - KV_HYST; no cap or no reading: off."""
    return bool(cap) and kv is not None and (kv >= cap or (was and kv > cap - KV_HYST))


async def server_guard(urls, stop, every=GUARD_EVERY_S, get=None, on_blind=None, timeout=METRICS_TIMEOUT_S,
                       on_kv=None):
    """During a play phase: the servers' KV peak (vllm:kv_cache_usage_perc, max over servers and polls) and their
    preemptions over the phase (num_preemptions_total delta), polled every ``every`` s until ``stop`` is set."""
    from arc_infra.metrics import parse_metrics

    if every <= timeout:
        raise ValueError(f"server_guard: poll every {every} s must exceed the read timeout {timeout} s")

    slow = [0.0]  # the slowest good read (s): N3 reads how close the server's /metrics comes to the timeout

    async def read(url):
        root = url.rstrip("/")
        root = root[:-3] if root.endswith("/v1") else root
        began = time.time()
        try:
            if get is None:
                import httpx

                async with httpx.AsyncClient(trust_env=False, timeout=timeout) as client:
                    got = await client.get(root + "/metrics")
                    got.raise_for_status()
                    text = got.text
            else:
                text = await asyncio.wait_for(get(root + "/metrics"), timeout)
        except Exception:  # noqa: BLE001 - a missed poll
            return None
        slow[0] = max(slow[0], time.time() - began)
        return parse_metrics(text)

    first, last, peak, polls, misses, blind, blind_polls = {}, {}, 0.0, 0, dict.fromkeys(urls, 0), set(), 0
    while True:
        rows = await asyncio.gather(*(read(u) for u in urls))
        for u, v in zip(urls, rows):
            if v is None:
                misses[u] += 1
                if misses[u] >= BLIND_AFTER and u not in blind:
                    blind.add(u)
                    if on_blind is not None:
                        on_blind(u, True)
                continue
            misses[u] = 0
            if u in blind:
                blind.discard(u)
                if on_blind is not None:
                    on_blind(u, False)
            LAST_METRICS[u] = (time.time(), v)
            polls += 1
            peak = max(peak, float(v.get("vllm:kv_cache_usage_perc", 0.0)))
            if on_kv is not None:
                on_kv(u, float(v.get("vllm:kv_cache_usage_perc", 0.0)))
            pre = float(v.get("vllm:num_preemptions_total", 0.0))
            first.setdefault(u, pre)
            last[u] = pre
        blind_polls += bool(blind)
        if stop.is_set():
            break
        try:
            await asyncio.wait_for(stop.wait(), every)
        except asyncio.TimeoutError:
            pass
    if on_kv is not None:  # the phase ends: no hold carries into the switch
        for u in urls:
            on_kv(u, None)
    if blind and on_blind is not None:  # the phase ends blind: the hold ends with it (the next phase's back-off cuts);
        for u in list(blind):  # None, not False: no read came back
            on_blind(u, None)
    return {"kv_peak": round(peak, 4), "preemptions": int(sum(last[u] - first[u] for u in last)), "polls": polls,
            "blind_polls": blind_polls, "read_s_max": round(slow[0], 1)}


def base_actions(curriculum, unit):
    """The base model's median actions to clear a set's level, from the history rows (v8.5p plays: the base policy
    with the run's harness); None under 2 clears. The damage probe's reference (rl_loop's mg ref is empty here)."""
    if not unit.get("game") or not unit.get("level"):
        return None
    acts = [int(r["actions"]) for r in curriculum.hist.get((unit["game"], int(unit["level"])), [])
            if r.get("cleared") and r.get("actions")]
    return statistics.median(acts) if len(acts) >= 2 else None


def lr_at(lr, warmup, step):
    """lr of the step after ``step`` optimizer steps: lr * (step + 1) / warmup up to lr, then lr."""
    return lr * min(1.0, (int(step) + 1) / warmup) if warmup > 0 else lr


def schedule_check(rl_loop):
    """CPU gate: ``rl_loop`` (the study's rl_loop block) -> problems ([] = fine). A fresh start
    ramps linearly over ``warmup`` steps to lr and stays there; a resumed Seq at step k (only the restored step: no
    per-container counter) takes the fresh run's lr at k, never a re-warm."""
    cfg = settings(rl_loop.get("mg") or {})
    if cfg is None:
        return []
    lr, warm = float(rl_loop["lr"]), int(cfg["warmup"])
    out = []

    def at(step):
        return Seq.lr(SimpleNamespace(loop=SimpleNamespace(cfg={"lr": lr}), cfg={"warmup": warm}, step=step))

    fresh = [at(k) for k in range(warm + 3)]
    want = [lr * min(1.0, (k + 1) / warm) if warm > 0 else lr for k in range(warm + 3)]
    if any(abs(a - b) > 1e-15 for a, b in zip(fresh, want)):
        out.append(f"lr schedule: fresh steps {fresh[:warm + 1]} != linear warm-up over {warm} to {lr}")
    if warm > 1 and not all(fresh[k] < fresh[k + 1] for k in range(warm - 1)):
        out.append("lr schedule: warm-up is not increasing")
    if any(abs(x - lr) > 1e-15 for x in fresh[max(warm - 1, 0):]):
        out.append(f"lr schedule: not constant {lr} from step {warm}")
    for k in sorted({1, max(warm // 2, 1), warm, warm + 2}):
        resumed = at(k)  # a new container: Seq() then resume() restores step = k, nothing else
        if abs(resumed - want[min(k, len(want) - 1)]) > 1e-15:
            out.append(f"lr schedule: resume at step {k} gives {resumed}, the run's lr there is {want[k]} (re-warm?)")
    return out


class Group:
    """What Loop.v7_prepare reads of a finished group."""

    def __init__(self, gid, unit, version, folder):
        self.gid, self.unit, self.version, self.done = gid, unit, int(version), {"folder": str(folder)}


def unit_of(game, level):
    return f"{game}-L{int(level)}"


def exact_start(row):
    """A try trains only from an exact start: a restored or resumed try whose game state (board,
    steps, levels) or replayed context (every recorded message, template and tool list) differed never trains,
    stricter than rl_loop.restored (16 Oct), which kept a state-exact start with a differing prompt message."""
    return all(not part or bool(part.get("ok")) for part in (row.get("restore"), row.get("resumed")))


def kept_rows(rows):
    """A group's rows its step trains on (exact restored starts only) and the dropped ones."""
    return [r for r in rows if exact_start(r)], [r for r in rows if not exact_start(r)]


def prepare_group(loop, g, rows):
    """A group's trained specs, prepared now and cached per gid (v7_prepare: train_step takes the same specs); None
    when no row has an exact start (train_step then handles the group as before)."""
    kept, _ = kept_rows(rows)
    if not kept:
        return None
    loop.v7 = getattr(loop, "v7", None) or {}
    loop.v7.setdefault("rows", {})[g.gid] = kept
    head, specs = loop.v7_prepare(g)
    loop.v7.setdefault("prepared", {})[g.gid] = (head, specs)
    return specs


def write_ahead(loop, specs, token, gid=None):
    """``ahead-early.json`` in the trainer folder: the trainer expands and encodes these specs at its GPU gate or
    idle poll (a new token: again, the done ones skipped by key). Returns the number of specs (0: none written)."""
    if not specs:
        return 0
    folder = Path(loop.work) / "trainer"
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "ahead-early.json.tmp"
    tmp.write_text(json.dumps({"token": token, "gid": gid, "ahead": specs, **loop.v7_trainer_args()}))
    tmp.rename(folder / "ahead-early.json")
    return len(specs)


def early_ahead(loop, groups, token):
    """early_ahead (D2 27 Oct: the update's first window waited 22.8 s for its encode): the first group
    with rows is prepared now and its specs written to the trainer folder, which the trainer encodes while the
    servers drain and park. Returns the number of specs (0: none written)."""
    for g, rows in groups:
        specs = prepare_group(loop, g, rows)
        if specs is None:
            continue
        return write_ahead(loop, specs, token, g.gid)
    return 0


def cards_used(smi=None):
    """Used GiB per visible card (nvidia-smi order), or None without nvidia-smi."""
    import subprocess

    try:
        out = subprocess.run([smi or os.environ.get("ARC_REAL_NVIDIA_SMI") or "nvidia-smi",
                              "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return [round(float(x) / 1024, 2) for x in out.split()] or None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def split_chunks(items, n):
    """``items`` (the groups with trained specs, in order) into ``n`` disjoint consecutive chunks of whole groups,
    sizes differing by at most one (16 / 4 -> 4 x 4; 15 / 4 -> 4, 4, 4, 3); fewer chunks when there are fewer items."""
    items, n = list(items), max(1, min(int(n), len(items) or 1))
    k, r = divmod(len(items), n)
    out, at = [], 0
    for c in range(n):
        size = k + (c < r)
        out.append(items[at:at + size])
        at += size
    return [c for c in out if c]


def minibatch_row(res, groups, epoch=1):
    """One optimizer step's safety fields (the pre-registered minibatch checks: k3, TIS clip, the θ0 trust region's
    clipped share, update and grad norm, entropy)."""
    k3, rc = res.get("k3") or {}, res.get("ratio_clip") or {}
    return {"groups": groups, "epoch": epoch, "grad_norm": res.get("grad_norm"), "update_norm": res.get("update_norm"),
            "loss": res.get("loss"), "trained_tokens": res.get("trained_tokens"), "entropy": res.get("entropy"),
            "approx_kl": k3.get("approx_kl"), "clip_frac": k3.get("clip_frac"),
            "ratio_clip_frac": rc.get("frac"), "ratio_max_abs_logr": rc.get("max_abs_logr")}


RATIO_CLIPS = ("gspo", "ppo", "off")  # the trainer's presets (rl1_trainer.RATIO_CLIP)


def step_plan(n_chunks, epochs):
    """The optimizer steps of one update: ``[(epoch, chunk, clipped, needs_old)]`` in order. Every step after the
    first reads the update's θ0 log-probabilities: epoch 1's first chunk keeps them as it trains (``keep_old``), its
    later chunks need a forward-only pass at θ0 before the first step (``old_pass``)."""
    out = []
    for e in range(epochs):
        for c in range(n_chunks):
            out.append((e, c, bool(e or c), e == 0 and c > 0))
    return out


def call_health(res):
    """An accumulate call's R3 coverage and encoder errors (v17 a3: U19's encoders raised 5 FileNotFoundErrors the
    trainer re-parsed silently; windows without a served record train on the trainer's own routing, unreported)."""
    served = res.get("served") if isinstance(res.get("served"), dict) else None
    enc = res.get("encode") if isinstance(res.get("encode"), dict) else {}
    out = {"encode_errors": len(enc.get("errors") or [])}
    if served is not None:
        n, miss = int(served.get("windows") or 0) + int(served.get("no_record") or 0), int(served.get("no_record") or 0)
        out.update(served_windows=n, served_no_record=miss, served_share=round(1 - miss / n, 3) if n else None)
    return out


async def train_step(loop, groups, lr, save, background=False, minibatches=None, epochs=None, ratio_clip=None):
    """The one training step (the redesign run's, and the P1 lr probe's): ``groups`` [(Group, try rows)] through
    Loop.v7_prepare (selection, rewards, GiGPO) into the trainer's gradient sums, then one optimizer step saving the
    adapter to ``save``. Returns the step's row (no step, None: no group had a trained try). ``background``
    (mg.async_adapter_save): the step returns after its optimizer; the adapter's files land later (trainer
    ``save_join``, awaited before the servers load it)."""
    loop.v7 = getattr(loop, "v7", None) or {}
    loop.v7.setdefault("rows", {})
    loop.v7.setdefault("prepared", {})
    rows_out, first, dropped, prepared = [], True, {}, []
    redesign = loop.cfg["mg"].get("redesign")
    frozen = {"policy_frozen": True} if isinstance(redesign, dict) and redesign.get("freeze_policy") else {}
    for g, rows in groups:
        bad = [r for r in rows if not exact_start(r)]
        if bad:
            rows = [r for r in rows if exact_start(r)]
            dropped[g.gid] = len(bad)
            loop.feed.emit("seq_restore_drop", gid=g.gid, dropped=len(bad), kept=len(rows),
                           labels=[r.get("label") for r in bad][:20],
                           why=[("state" if not ((r.get("restore") or r.get("resumed") or {}).get("state") or {}).get("ok")
                                 else "context") for r in bad][:20])
        loop.v7["rows"][g.gid] = rows
        prepared.append((g, *loop.v7_prepare(g)))
        loop.v7["rows"].pop(g.gid, None)  # a group prepared early (early_ahead): its cached specs were taken
    # encode-ahead (MFU, D1 26 Oct): each call carries the next group's specs, which the trainer parses and encodes
    # while this call runs (rl1_trainer ahead_start), so the next call's first window is ready at its start
    with_specs = [i for i, (_, _, specs) in enumerate(prepared) if specs]
    mb = int(minibatches or (redesign.get("minibatches") if isinstance(redesign, dict) else 1) or 1)
    chunks = split_chunks(with_specs, mb)
    epochs = max(1, int(epochs or 1))
    multi = len(chunks) > 1 or epochs > 1
    # every step after an update's first is clipped against θ0 (default GSPO's region); "off" only
    # by an explicit knob (v17 a2: 4 unclipped steps at full lr degraded the policy)
    clip = (ratio_clip or "gspo") if multi else None
    clip = None if clip == "off" else clip
    steps = []  # minibatches: the steps before the last, one row each
    if isinstance(redesign, dict) and redesign.get("one_call") and with_specs and multi:
        plan = step_plan(len(chunks), epochs)
        spec_of = [[a for i in idx for a in prepared[i][2]] for idx in chunks]
        for k, (e, c, clipped, _) in enumerate(plan):
            idx, specs_c = chunks[c], spec_of[c]
            nxt = spec_of[plan[k + 1][1]] if k + 1 < len(plan) else None
            res = await loop.trainer.call("accumulate", lr=lr, attempts=specs_c, first=True, **loop.v7_trainer_args(),
                                          **frozen, **({"ahead": nxt} if nxt else {}),
                                          **({"ratio_clip": clip} if clip and clipped else {}),
                                          **({"keep_old": True} if clip and epochs > 1 and k == 0 else {}),
                                          **({"old_reset": True} if k == 0 else {}))
            if not res.get("ok"):
                from .rl_loop import StepFailed

                raise StepFailed(f"accumulate (epoch {e + 1}, minibatch {c + 1}/{len(chunks)}): "
                                 f"{res.get('error') or res.get('stop')}")
            wall = float(res.get("op_wall_seconds") or res.get("seconds") or 0)
            util = await asyncio.to_thread(gpu_util_between, time.time() - wall, time.time()) if wall > 0 else None
            enc = res.get("encode") if isinstance(res.get("encode"), dict) else {}
            hits, total = enc.get("ahead_hits"), enc.get("encoded")
            # the encode stats on every minibatch row (v17 29 Oct: rows without them read as ahead_hit_share 0); the
            # first chunk takes the early-ahead file, the later ones the previous call's ahead
            loop.feed.emit("seq_accumulate", one_call=True, minibatch=c + 1, minibatches=len(chunks), groups=len(idx),
                           epoch=e + 1, epochs=epochs, windows=len(specs_c), call_util=util, ratio_clip=res.get("ratio_clip"),
                           ahead_expected=(Path(loop.work) / "trainer" / "ahead-early.json").is_file() if k == 0 else True,
                           ahead_hits=hits, ahead_windows=total,
                           ahead_hit_share=round(hits / total, 3) if isinstance(hits, int) and isinstance(total, int)
                           and total > 0 else None,
                           **{k: res.get(k) for k in ("trained_tokens", "train_seconds", "card_peak_gib", "setup_seconds",
                                                      "window_wait_seconds", "pre_seconds", "op_wall_seconds")},
                           **call_health(res))
            if k == 0 and clip and len(chunks) > 1:  # θ0 for the later chunks: forward only, before the first step
                old = [a for cc in range(1, len(chunks)) for a in spec_of[cc]]
                got = await loop.trainer.call("old_pass", lr=lr, attempts=old, **loop.v7_trainer_args(), **frozen,
                                              ahead=spec_of[1])
                if not got.get("ok"):
                    from .rl_loop import StepFailed

                    raise StepFailed(f"old_pass: {got.get('error') or got.get('stop')}")
                loop.feed.emit("seq_old_pass", windows=got.get("old_windows"), kept=got.get("old_kept"),
                               scored_tokens=got.get("scored_tokens"), seconds=got.get("train_seconds"),
                               card_peak_gib=got.get("card_peak_gib"), **call_health(got))
            if k + 1 < len(plan):  # every step but the last runs here, unsaved; the last below, saved
                got = await loop.trainer.call("step", lr=lr, **loop.v7_trainer_args(), **frozen)
                if not got.get("ok"):
                    from .rl_loop import StepFailed

                    raise StepFailed(f"step (epoch {e + 1}, minibatch {c + 1}/{len(chunks)}): "
                                     f"{got.get('error') or got.get('stop')}")
                steps.append(minibatch_row(got, len(idx), e + 1))
        first = False
        prepared = [(g, head, []) for g, head, _ in prepared]
    elif isinstance(redesign, dict) and redesign.get("one_call") and with_specs:
        # one accumulate per update (U4: 8 per-group calls ran the 2-card update at 70.7% util: each call
        # paid a first-window encode wait, a cross-rank sync on its unequal windows and ~4 s outside the call); the
        # same windows and sums, one call: the rank split balances over every window of the update
        allspecs = [a for _, _, specs in prepared for a in specs]
        ahead_file = Path(loop.work) / "trainer" / "ahead-early.json"  # written by early_ahead / ahead_pending
        res = await loop.trainer.call("accumulate", lr=lr, attempts=allspecs, first=True, **loop.v7_trainer_args(),
                                      **frozen)
        if not res.get("ok"):
            from .rl_loop import StepFailed

            raise StepFailed(f"accumulate (one call, {len(with_specs)} groups): {res.get('error') or res.get('stop')}")
        first = False
        wall = float(res.get("op_wall_seconds") or res.get("seconds") or 0)
        util = await asyncio.to_thread(gpu_util_between, time.time() - wall, time.time()) if wall > 0 else None
        enc = res.get("encode") if isinstance(res.get("encode"), dict) else {}
        # rank 0's windows through the card path ("encoded", pre-encoded ones included) and those taken pre-encoded
        hits, total = enc.get("ahead_hits"), enc.get("encoded")
        ahead = {"ahead_expected": ahead_file.is_file(), "ahead_hits": hits, "ahead_windows": total,
                 # gate legs 28 Oct: 0 of 192 for every rl_seq run (watch each:train.ahead_hit_share, judge ahead_used)
                 "ahead_hit_share": round(hits / total, 3) if isinstance(hits, int) and isinstance(total, int)
                 and total > 0 else None}
        loop.feed.emit("seq_accumulate", one_call=True, groups=len(with_specs), windows=len(allspecs), call_util=util,
                       **{k: res.get(k) for k in ("trained_tokens", "train_seconds", "card_peak_gib", "setup_seconds",
                                                  "window_wait_seconds", "pre_seconds", "op_wall_seconds", "seconds")},
                       **ahead, **call_health(res))
        prepared = [(g, head, []) for g, head, _ in prepared]  # the rows below: per-group heads, nothing more to send
    for i, (g, head, specs) in enumerate(prepared):
        row = {**head, "restore_dropped": dropped.get(g.gid, 0)}
        if row.get("clear_actions") is not None and row.get("base_actions") is None:  # the damage probe's base (H4: null)
            base = getattr(loop, "seq_base_actions", None)
            row["base_actions"] = base(g.unit) if base else None
        if specs:
            nxt = next((prepared[j][2] for j in with_specs if j > i), None)
            res = await loop.trainer.call("accumulate", lr=lr, attempts=specs, first=first, **loop.v7_trainer_args(),
                                          **frozen, **({"ahead": nxt} if nxt else {}))
            if not res.get("ok"):
                from .rl_loop import StepFailed

                raise StepFailed(f"accumulate {g.gid}: {res.get('error') or res.get('stop')}")
            first = False
            row.update({k: res.get(k) for k in ("trained_tokens", "train_seconds", "card_peak_gib")})
        rows_out.append(row)
        loop.feed.emit("seq_accumulate", **{k: v for k, v in row.items() if not isinstance(v, (list, dict))})
    if first:
        return None
    began = time.time()
    res = await loop.trainer.call("step", lr=lr, save=str(save), **({"save_async": True} if background else {}),
                                  **loop.v7_trainer_args(), **frozen)
    if not res.get("ok"):
        from .rl_loop import StepFailed

        raise StepFailed(f"step: {res.get('error') or res.get('stop')}")
    from .rl_loop import DRAFT_ROW
    ratios = [x["clear_actions"] / x["base_actions"] for x in rows_out if x.get("clear_actions") and x.get("base_actions")]
    mbs = {}
    if steps:
        steps.append(minibatch_row(res, len(chunks[-1]), epochs))
        mbs = {"minibatches": steps, "epochs": epochs, "ratio_clip": clip, "update_norm_sum": round(sum(float(x.get("update_norm") or 0) for x in steps), 6),
               "grad_norm_max": max(float(x.get("grad_norm") or 0) for x in steps)}
    return {"lr": lr, "groups": len(rows_out), **mbs,
            # damage probe (alert at 2x): median over the step's groups of clear actions / base actions
            "actions_per_clear_vs_base": round(statistics.median(ratios), 4) if ratios else None, "tries": sum(int(x.get("tries") or 0) for x in rows_out),
            "trained": sum(int(x.get("trained") or 0) for x in rows_out), "restore_dropped": sum(dropped.values()),
            "step_call_seconds": round(time.time() - began, 1),
            **{k: res.get(k) for k in ("grad_norm", "loss", "entropy", "trained_tokens", "train_seconds", "update_norm",
                                       "torch_peak_gib")},
            # the update's tokens, not its last step's (v17 U24 read 1.01M of 1.82M)
            **({"trained_tokens": sum(int(x.get("trained_tokens") or 0) for x in steps)} if steps else {}),
            **{k: res[k] for k in ("prompt_same_rate", "prompt_checked", "prompt_excluded", "k3", "adv_nonzero_share",
                                   "adv_up_share") if res.get(k) is not None},
            **{k: res[k] for k in DRAFT_ROW if k in res}}


def gpu_util_between(a, b, path=None):
    """Mean GPU util (%, every card) over [a, b] from the run's 1-s gpulog (util/gpulog.txt: per card "i, u %, used
    MiB, total MiB", the epoch last); None without the file or a row (U4b 27 Oct: the update's whole-call util is the
    the training-phase bar, setup and tail included)."""
    import re

    out = os.environ.get("ARC_VALIDATION_OUTPUT")
    path = Path(path) if path else (Path(out) / "util" / "gpulog.txt" if out else None)
    if path is None or not path.is_file():
        return None
    vals = []
    with open(path, errors="replace") as file:
        for line in file:
            parts = line.split()
            try:
                t = float(parts[-1])
            except (ValueError, IndexError):
                continue
            if a <= t <= b:
                vals += [float(u) for u in re.findall(r"(\d+(?:\.\d+)?) %", line)]
    return round(sum(vals) / len(vals), 1) if vals else None


TRY_FIELDS = ("level", "completion_tokens", "cleared", "actions", "level_score", "total_levels", "admitted_requests",
              "gameplay_requests", "noaction_requests", "policy_version")


def try_row(row):
    """A finished try's fields the exporter's try_rows reads (RHAE, tokens, reply stats), for the group_done row."""
    out = {k: row.get(k) for k in TRY_FIELDS if row.get(k) is not None}
    ro = row.get("rollout") or {}
    if isinstance(ro, dict):
        out["rollout"] = {k: ro.get(k) for k in ("live", "capped", "think_tokens", "filler") if ro.get(k) is not None}
    return out


def replay_groups(spec, update=0):
    """mg.redesign.replay_groups {"roots": [rl dirs], "groups": n, "min_tries": m}: up to ``n`` saved groups (folders
    ``<gid>-<game>-L<level>`` with level-bench.json) of at least ``m`` finished tries (restored, with a level, the
    rollout on disk), in folder order across the roots: [(Group, rows)], each a fresh gid. ``repeat``: fewer found
    than ``n`` are cycled to ``n`` (U1 27 Oct: D2 + D1 hold 9 finished groups; the real run trains 16 per update).
    ``updates`` k (U3): k updates replay groups n*u .. n*(u+1)-1 of the cycled list; ``update`` is u."""
    from .rl_loop import level_of, restored

    found = _saved_groups(spec, level_of, restored)
    n = int(spec["groups"])
    if spec.get("repeat") and found:
        found = [found[i % len(found)] for i in range(n * int(spec.get("updates", 1)))]
    found = found[n * update:n * (update + 1)]
    return [(Group(f"r{n * update + i + 1:02d}-{folder.name}", unit, 0, folder), copy.deepcopy(rows))
            for i, (folder, unit, rows) in enumerate(found)]  # a repeat's rows are its own (prepare marks them)


def replay_folders(folders, min_tries=1):
    """seed_from: the named saved group folders (a seed run's first phase, in its order) as replay_groups makes them;
    a folder without enough finished tries is skipped."""
    from .rl_loop import level_of, restored

    out = []
    for i, folder in enumerate(Path(f) for f in folders):
        found = _saved_groups({"roots": [], "groups": 1, "min_tries": min_tries, "folders": [folder]}, level_of, restored)
        out += [(Group(f"s{i + 1:02d}-{folder.name}", unit, 0, folder), copy.deepcopy(rows)) for _, unit, rows in found]
    return out


def replay_sets(loop, replayed):
    """The sets a replayed group's prepare reads (v7_specs' level_norm cap, mg_level_ref), as Seq.spec writes them at
    a group's launch (U1 v2 N1: KeyError mg_sets, nothing had launched); cap = the tries' token budget."""
    sets = loop.cfg.setdefault("mg_sets", {})
    for g, rows in replayed:
        game, level = g.unit.rsplit("-L", 1)
        caps = [int(r["token_budget"]) for r in rows if r.get("token_budget")]
        spec = {"name": g.unit, "game": game, "level": int(level), "role": "train", "fresh": int(level) == 1,
                **({"cap": max(caps)} if caps else {})}
        sets[g.unit] = {**sets.get(g.unit, {}), **spec}


def _saved_groups(spec, level_of, restored):
    out, cap = [], int(spec["groups"]) * int(spec.get("updates", 1))
    benches = [[Path(f) / "level-bench.json" for f in spec["folders"]]] if spec.get("folders") else \
        [sorted(Path(root).glob("*/level-bench.json")) for root in spec["roots"]]
    for found in benches:
        for bench in found:
            try:
                rows = json.loads(bench.read_text()).get("attempts") or []
            except (OSError, ValueError):
                continue
            rows = [r for r in rows if restored(r) and level_of(r) is not None
                    and (bench.parent / "attempts" / r["label"] / "rollout.jsonl").is_file()]
            name = bench.parent.name.split("-", 1)
            if len(rows) < int(spec.get("min_tries") or 1) or len(name) < 2 or "-L" not in name[1]:
                continue
            game, level = name[1].rsplit("-L", 1)
            out.append((bench.parent, unit_of(game, level), rows))
            if len(out) >= cap and not spec.get("longest"):
                return out
    if spec.get("longest"):  # U4: the groups with the longest contexts first (max prompt tokens of any try)
        out.sort(key=lambda g: -max(int(r.get("max_prompt_tokens") or 0) for r in g[2]))
    return out[:cap]


async def lr_probe(loop, seq, ctl, cfg):
    """P1 on the redesign's own step (the probe's step is the run's). ``steps`` play phases on the base
    model (no switch between them; each phase's ``phase_groups`` full groups are one step's data), then one switch to
    train: per rate, the trainer resets to its initial LoRA and a fresh optimizer, the held-out probes are scored on the
    base, and the same groups in the same order go through ``train_step`` (the run's step: exact-start drops,
    v7_prepare selection and rewards, accumulate, step), the probes scored after every step (KL to base). Every rate's
    groups get the same prepared specs (v7_prepare caches per gid; its turn-credit balance would differ on a second
    preparation). Feed rows: lr_probe_rate per rate, lr_probe at the end (modal_rl1's judge). Returns the stop why."""
    import copy

    spec = cfg["lr_probe"]
    steps, warm = int(spec["steps"]), int(spec["warmup"] if spec.get("warmup") is not None else cfg["warmup"])
    collected = []
    for k in range(steps):
        seq.phase_done.clear()
        for grp in seq.backlog.start_phase():
            seq.launch(grp)
        seq.pump()
        t0 = time.time()
        await seq.phase_done.wait()
        if seq.out_of_time:
            return "deadline"
        out = seq.backlog.end_phase()
        groups = []
        for fg in out["full"]:
            grp = fg["group"]
            task = seq.tasks.pop(grp["id"], None)
            if task is not None:
                await task
            folder, rows = seq.rows.pop(grp["id"], (None, []))
            if folder is not None and rows:
                groups.append((Group(grp["id"], unit_of(grp["game"], grp["level"]), grp["version"], folder), rows))
        collected.append(groups)
        loop.feed.emit("seq_probe_play", phase=k + 1, groups=len(groups), tries=sum(len(r) for _, r in groups),
                       play_seconds=round(time.time() - t0, 1))
    stop = await ctl.to_train()
    loop.feed.emit("seq_train", step=0, probe=True, stop=stop, groups=sum(len(g) for g in collected))
    loop.v7 = getattr(loop, "v7", None) or {}
    loop.v7.setdefault("rows", {})
    cache = loop.v7.setdefault("prepared", {})
    prepared = {}
    for groups in collected:  # prepared once: every rate trains on the same specs
        for g, rows in groups:
            loop.v7["rows"][g.gid] = [r for r in rows if exact_start(r)]
            prepared[g.gid] = loop.v7_prepare(g)
    probes = []
    for i, x in enumerate(cfg["lr_probe"]["probes"]):
        folder = Path(x["folder"]) if Path(x["folder"]).is_absolute() else Path(loop.source) / x["folder"]
        probes += loop.windows(folder, [{"label": label} for label in x["labels"]], prefix=f"heldout{i}-")
    mean = lambda d: round(sum(d.values()) / len(d), 6) if d else None
    out = {"rates": {}, "skipped": []}
    for n, lr in enumerate(float(r) for r in spec["rates"]):
        began = time.time()
        await loop.trainer.must("reset", lr=lr)
        await loop.trainer.must("score", windows=probes, label="base")
        rows = []
        for k, groups in enumerate(collected):
            cache.update(copy.deepcopy({g.gid: prepared[g.gid] for g, _ in groups}))
            lr_k = lr * min(1.0, (k + 1) / warm) if warm > 0 else lr
            got = await train_step(loop, groups, lr_k, loop.work / "adapters" / f"probe{n}-s{k + 1:02d}")
            if got is None:  # every group a tie: no step (the rate's row is not ok)
                rows.append({"step": k + 1, "skipped": True})
                continue
            scored = (await loop.trainer.must("score", windows=probes, label=f"{lr}-s{k + 1}"))["windows"]
            pick = lambda key: {name: float(w[key]) for name, w in scored.items() if (w or {}).get(key) is not None}
            rows.append({"step": k + 1, "lr": lr_k, "kl_base": mean(pick("kl_base")),
                         "logratio_base": mean(pick("logratio_base")), "kl_previous": mean(pick("kl_previous")),
                         **{key: got.get(key) for key in ("loss", "grad_norm", "update_norm", "trained_tokens",
                                                          "train_seconds", "trained", "tries", "restore_dropped")}})
        row = {"lr": lr, "ok": len(rows) == steps and not any(r.get("skipped") for r in rows), "steps": rows,
               "warmup": warm, "seconds": round(time.time() - began, 1),
               "trained_tries": sum(int(r.get("trained") or 0) for r in rows)}
        out["rates"][str(lr)] = row
        loop.feed.emit("lr_probe_rate", **row)
    loop.feed.emit("lr_probe", **out)
    return "lr_probe"


class Seq:
    def __init__(self, loop, cfg):
        self.loop, self.cfg = loop, cfg
        self.bcfg = rl_backlog.settings({"redesign": {"backlog": cfg.get("backlog") or {}}})
        rows = rl_backlog.load_history(Path(loop.source) / cfg["history"])
        games = list(cfg.get("games") or rl_backlog.TRAIN)
        curriculum = rl_backlog.Curriculum(rows, self.bcfg)
        loop.seq_base_actions = lambda unit: base_actions(curriculum, (loop.cfg.get("mg_sets") or {}).get(unit) or {})
        self.backlog = rl_backlog.Backlog(games, curriculum, int(cfg["streams"]),
                                          self.bcfg)
        self.step, self.version = 0, 0
        self.play_phases = []  # play_only: each phase's full group folders, in order (seq_state "phases")
        self.seed_folders = None  # seed_from: step 0 replays these (the seed run's first phase)
        self.tasks, self.bench_group, self.finished, self.rows = {}, {}, {}, {}
        self.pending = set()  # harvest snapshots in progress
        self.exports = set()  # held-out read adapter hand-offs in progress
        self.urls, self.on_server = [], {}  # play servers' /v1 URLs; group id -> server index (a group stays on one)
        self.phase_done = asyncio.Event()
        self.out_of_time = False  # the level bench's run deadline reached: no new group starts, the run ends
        self.checks = None
        self.dwell_done = dict.fromkeys(DWELL_KEYS, 0.0)  # finished tries' summed seconds (dwell export)
        self.dwell_done["tries"] = 0
        self.drivers = None  # rl_drivers.DriverPool (mg.redesign.drivers > 0): groups play in worker processes
        self.stop_why = None  # why out_of_time was set, when not the bench deadline
        # one_call + early_ahead: groups prepared as they finish during play (ahead_group), their specs by gid
        self.ahead_specs, self.ahead_tasks, self.ahead_wrote, self.ahead_lock = {}, [], 0.0, asyncio.Lock()
        # each finished group is copied to the volume at once (mirror_group); before the update the
        # phase's groups go to seq_pending.json; a resume trains them from there (resume_groups) instead of replaying
        self.mirror_tasks, self.mirrored, self.resume_groups = [], {}, None
        self.play_wrote = 0.0  # seq_play.json: the phase so far, a mid-play restart point (put_play)
        self.launched_t = {}  # group id -> launch epoch in this container (the mirror prune's mark)
        # control group: every ``every`` updates one untrained twin of a policy group on the base model
        self.controls, self.control_step, self.control_rids = {}, None, set()

    # ---- play -----------------------------------------------------------------------------------------------

    def spec(self, grp):
        game, level = grp["game"], int(grp["level"])
        unit = unit_of(game, level)
        spec = {"name": unit, "game": game, "level": level, "role": "train", "cap": int(grp["budget"]),
                "fresh": bool(grp["start"]["l1"])}
        sets = self.loop.cfg.setdefault("mg_sets", {})
        sets[unit] = {**sets.get(unit, {}), **spec}  # mg_select / mg_level_ref read the set by name
        return spec

    def launch(self, grp):
        self.launched_t.setdefault(grp["id"], time.time())
        spec = self.spec(grp)
        name = f"{grp['id']}-{spec['name']}"
        self.bench_group[name] = grp["id"]
        self.finished[grp["id"]] = 0
        start = grp["start"]
        starts = [] if start["l1"] else [(start["state"]["attempt"], int(start["state"]["request"]))]
        url = None
        if len(self.urls) > 1:  # the server with the fewest unfinished tries of its groups (its own KV gate admits them)
            load = [0] * len(self.urls)
            for gid, i in self.on_server.items():
                live = self.backlog.groups.get(gid)
                if live is not None:
                    load[i] += live["group"]["tries"] - len(live["results"])
            i = min(range(len(self.urls)), key=lambda k: (load[k], k))
            self.on_server[grp["id"]], url = i, self.urls[i]
        driver = None
        if self.drivers is not None:  # the worker with the fewest unfinished tries on this server
            driver = self.drivers.assign(grp["id"], name, grp["tries"], url or self.urls[0])
        self.tasks[grp["id"]] = asyncio.ensure_future(self.play(grp, spec, name, starts, url))
        self.loop.feed.emit("seq_group", gid=grp["id"], set=spec["name"], budget=spec["cap"], tries=grp["tries"],
                            start=start["id"], version=self.version, server=self.on_server.get(grp["id"], 0),
                            **({"driver": driver} if self.drivers is not None else {}))
        self.control(grp, spec, starts, url)

    def control(self, grp, spec, starts, url):
        """the learning metric's control. At every ``every``-th update the first policy group launched on
        a level that already holds 2 finished groups gets a twin: ``tries`` tries from the same restored start under
        the same budget on the base model (version 0, no LoRA slot), never trained or harvested; its tries go out
        as a control_done row (the exporter pairs it with the policy group's group_done)."""
        cg = self.cfg.get("control")
        if not cg or self.control_step == self.step or self.step % int(cg.get("every", 4)):
            return
        if len(self.backlog.cur.fresh.get((grp["game"], int(grp["level"]))) or []) < 2:
            return
        self.control_step = self.step
        gid, tries = f"{grp['id']}c", int(cg.get("tries", 8))
        name = f"{gid}-{spec['name']}"
        if self.drivers is not None and self.drivers.assign(gid, name, tries, url or self.urls[0]) is None:
            self.loop.feed.emit("seq_control_skipped", gid=gid, pair=grp["id"], set=spec["name"], step=self.step,
                                why="no driver slot")
            return
        self.controls[gid] = asyncio.ensure_future(self.control_play(grp, gid, name, spec, starts, url, tries))
        self.loop.feed.emit("seq_control", gid=gid, pair=grp["id"], set=spec["name"], budget=spec["cap"], tries=tries,
                            start=grp["start"]["id"], step=self.step, model=self.loop.server.model)

    async def control_play(self, grp, gid, name, spec, starts, url, tries):
        model = self.loop.server.model
        try:
            if self.drivers is not None:
                folder, rows, play = await self.drivers.play(
                    gid, name, spec, starts, tries, model=model, url=url, harvest=False,
                    mg_set=(self.loop.cfg.get("mg_sets") or {}).get(spec["name"]))
            else:
                folder, rows, play = await self.loop.mg_play(name, spec, starts, tries, model=model, seed=None,
                                                             harvest=False, **({"base_url": url} if url else {}))
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - a lost control is a missing point, never a failed run
            self.loop.feed.emit("seq_control_error", gid=gid, pair=grp["id"], error=repr(error)[:300])
            return None
        finally:
            self.controls.pop(gid, None)
        if folder is not None:  # its R3 captures go at the next prune (served_prune)
            for r in rows:
                rollout_rids(Path(folder) / "attempts" / str(r.get("label")) / "rollout.jsonl", self.control_rids)
        self.loop.feed.emit("control_done", gid=gid, pair=grp["id"], set=spec["name"], step=self.step, version=0,
                            tries=len(rows), clears=sum(1 for r in rows if r.get("cleared")),
                            seconds=(play or {}).get("seconds"), attempts=[try_row(r) for r in rows])
        return play

    async def play(self, grp, spec, name, starts, url=None):
        try:
            if self.drivers is not None:
                folder, rows, play = await self.drivers.play(
                    grp["id"], name, spec, starts, grp["tries"], model=self.cfg["adapter"], url=url,
                    mg_set=(self.loop.cfg.get("mg_sets") or {}).get(spec["name"]))
            else:
                folder, rows, play = await self.loop.mg_play(name, spec, starts, grp["tries"], model=self.cfg["adapter"],
                                                             seed=None, harvest=True, **({"base_url": url} if url else {}))
        except TimeoutError as error:
            if "left for the level bench" not in str(error):
                raise
            # T3 v3 (26 Oct): past the deadline every new bench fails at once; an empty group must not close the phase
            self.out_of_time = True
            self.loop.feed.emit("seq_deadline", gid=grp["id"], error=str(error)[:200])
            self.phase_done.set()
            return None
        except Exception as error:  # noqa: BLE001 - a failed bench ends its group's tries as fails
            self.loop.feed.emit("alert", why="seq: group play failed", gid=grp["id"], error=repr(error)[:300])
            folder, rows, play = None, [], None
            live = self.backlog.groups.get(grp["id"])
            if live is not None and not live["results"]:  # nothing played: no fails in the curriculum, start not reused
                self.backlog.drop_failed(grp["id"])
                self.loop.feed.emit("seq_group_dropped", gid=grp["id"], set=spec["name"], start=grp["start"]["id"],
                                    error=repr(error)[:200])
        self.rows[grp["id"]] = (folder, rows)
        self.on_server.pop(grp["id"], None)
        self.mirror_group(grp, folder)
        self.ahead_group(grp, folder, rows)
        self.loop.feed.emit("group_done", gid=grp["id"], set=spec["name"], version=int(grp["version"]),
                            tries=len(rows), clears=sum(1 for r in rows if r.get("cleared")),
                            seconds=(play or {}).get("seconds"),  # the exporter's row (tools/ops/export_rl.py)
                            attempts=[try_row(r) for r in rows])  # its train.rhae_per_10k (no group folder live)
        live = self.backlog.groups.get(grp["id"])
        if live is not None:  # tries the hook never saw (a crashed bench, censored tries): fails
            for i in range(grp["tries"]):
                if i not in live["results"]:
                    self.backlog.on_try_done(grp["id"], i, cleared=False, tokens=0)
            self.pump()
        return play

    def hook(self, bench, row):
        """level_bench.PROGRESS_HOOK: one finished try."""
        gid = self.bench_group.get(bench)
        if gid is None or gid not in self.backlog.groups:
            return
        from .rl_loop import clear_rhae, clear_tokens

        index = self.finished[gid]
        self.finished[gid] += 1
        dwell_add(self.dwell_done, row)
        t = clear_tokens(row)
        tokens = int(row.get("completion_tokens") or 0)
        actions = int(((row.get("level_rows") or [{}])[0]).get("actions") or row.get("actions") or 0)
        if t and row.get("harvest"):
            task = asyncio.ensure_future(self.harvest(gid, index, bench, row, tokens, actions, clear_rhae(row) or 0.0))
            self.pending.add(task)
            task.add_done_callback(self.pending.discard)
            return
        self.backlog.on_try_done(gid, index, cleared=bool(t), tokens=tokens, actions=actions,
                                 rhae=clear_rhae(row) or 0.0)
        self.pump()

    async def harvest(self, gid, index, bench, row, tokens, actions, rhae):
        """A cleared try's next-level start: snapshot and checks as harvest.collect; a failed check saves nothing."""
        from .restore import load, unsafe_from

        grp = self.backlog.groups[gid]["group"]
        level = int(grp["level"]) + 1
        attempt = self.loop.output / "rl" / bench / "attempts" / row["label"]
        dest = harvest_mod.root(self.loop) / "harvest" / f"s{self.step:04d}-{bench}-{row['label']}"
        mark = row["harvest"]
        why = None
        try:
            if mark.get("state") not in (None, "NOT_FINISHED"):
                why = "game_state"
            else:
                await asyncio.to_thread(harvest_mod.snapshot, attempt, dest)
                request = int(mark["request"])
                first, _ = unsafe_from(dest)
                tspec = {"level": level, "max_prompt": None}
                if first is not None and first <= int(mark.get("step") or 0):
                    why = "unsafe_event"
                elif self.loop.mg_attempt_start(tspec, dest) != request or load(dest, request)["level"] != level:
                    why = "start_mismatch"
                elif self.cfg["board_check"]:
                    if self.checks is None:
                        self.checks = harvest_mod.arcade(self.loop) or False
                    if self.checks:
                        ok, detail = await asyncio.to_thread(harvest_mod.board_check, *self.checks, grp["game"], dest,
                                                             request)
                        why = None if ok else f"board_check: {detail}"
        except (OSError, ValueError, KeyError, IndexError) as error:
            why = f"snapshot_{type(error).__name__}"
        if gid not in self.backlog.groups:
            return
        if why is None:
            self.backlog.on_level_reached(gid, index, state={"attempt": str(dest), "request": int(mark["request"])},
                                          context=None, version=self.version, tokens=tokens, actions=actions, rhae=rhae)
        else:
            shutil.rmtree(dest, ignore_errors=True)
            self.loop.feed.emit("seq_harvest_drop", gid=gid, label=row["label"], why=why)
            self.backlog.on_try_done(gid, index, cleared=True, tokens=tokens, actions=actions, rhae=rhae)
        self.pump()

    def pump(self):
        if self.out_of_time:
            self.phase_done.set()
            return
        while (grp := self.backlog.next_group()) is not None:
            self.launch(grp)
        if self.backlog.phase_done:
            self.phase_done.set()

    # ---- train ----------------------------------------------------------------------------------------------

    def lr(self):
        """The next step's lr: linear warm-up over ``warmup`` steps of the RUN (lr 3e-5, 10-step
        warm-up on a fresh start, none on resume), then constant. Keyed by the run's step (restored by resume), so a
        resumed run takes the lr of its step and never re-warms."""
        return lr_at(float(self.loop.cfg["lr"]), int(self.cfg["warmup"]), self.step)

    def mirror_group(self, grp, folder):
        """The group's folder (tries, level-bench.json) copied to the run's volume tree now, not at the run's end."""
        dest = mirror_root()
        if dest is None or folder is None or not Path(folder).is_dir():
            return
        dest = dest / Path(folder).name
        if dest.resolve() == Path(folder).resolve():
            self.mirrored[grp["id"]] = dest
            self.put_play()
            return

        async def copy():
            try:
                await asyncio.to_thread(shutil.copytree, folder, dest, dirs_exist_ok=True)
                self.mirrored[grp["id"]] = dest
                self.put_play()
            except Exception as error:  # noqa: BLE001 - the end-of-run collection still copies it
                self.loop.feed.emit("alert", why="seq: group mirror failed (alert only)", gid=grp["id"],
                                    error=repr(error)[:300])
        self.mirror_tasks.append(asyncio.ensure_future(copy()))

    async def mirror_join(self):
        tasks, self.mirror_tasks = self.mirror_tasks, []
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def put_pending(self, full):
        """seq_pending.json beside seq_state.json: the update about to run (step + 1) and its groups' volume folders
        (relative to the run root). Written only when every group is mirrored; returns the group count."""
        root = harvest_mod.root(self.loop)
        groups = []
        for fg in full:
            grp = fg["group"]
            where = self.where(grp["id"], root)
            if where is None:
                return 0
            groups.append({"gid": grp["id"], "game": grp["game"], "level": int(grp["level"]),
                           "version": int(grp["version"]), **where})
        path = root / "seq_pending.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"step": self.step + 1, "version": self.version, "groups": groups}))
        tmp.replace(path)
        for g in groups:
            self.mirrored.pop(g["gid"], None)
        return len(groups)

    def where(self, gid, root=None):
        """A mirrored group's saved folder: {"rel"} under this run's root, else {"abs"} (a group restored from an
        earlier leg's root); None when its copy has not landed."""
        dest = self.mirrored.get(gid)
        if dest is None:
            return None
        try:
            return {"rel": str(Path(dest).relative_to(root or harvest_mod.root(self.loop)))}
        except ValueError:
            return {"abs": str(dest)}

    def put_play(self):
        """seq_play.json (checklist item 6, real-v11 attempt 2 lost ~28 min of play to a
        preemption before any phase closed): the backlog and curriculum now, and the saved folders of the full groups
        of the update being played for (step + 1). Written as each group's copy lands and at least every PLAY_EVERY_S
        (play_saver); a resume loads it and trains those groups with the phase's new ones (play_restore)."""
        root = harvest_mod.root(self.loop)
        groups = {}
        for fg in self.backlog.full:
            where = self.where(fg["group"]["id"], root)
            if where is not None:
                groups[fg["group"]["id"]] = where
        state = {"step": self.step + 1, "version": self.version, "root": str(root), "t": time.time(),
                 "groups": groups, "backlog": self.backlog.state_dict()}
        try:
            path = root / "seq_play.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, default=str))
            tmp.replace(path)
        except OSError as error:  # the next write retries; the run never stops for it
            self.loop.feed.emit("alert", why="seq: play state write failed (alert only)", error=repr(error)[:300])
            return None
        self.play_wrote = time.time()
        return len(groups)

    async def play_saver(self, every=None):
        """During a play phase: put_play at least every ``every`` s (the backlog moves between group closes)."""
        every = PLAY_EVERY_S if every is None else every
        while True:
            await asyncio.sleep(max(1.0, every - (time.time() - self.play_wrote)))
            if time.time() - self.play_wrote >= every - 1.0:
                self.put_play()

    async def restore_full(self, root, where):
        """After a resume: the backlog's full groups (this update's, played before the stop) trained from their
        saved folders (``where``: gid -> saved_folder entry), nothing replayed; a group whose tries are not there
        goes (its start back). Returns (kept gids, dropped gids)."""
        from .rl_loop import level_of, restored

        task = getattr(self.loop, "served_restore_task", None)
        if task is not None:  # R3: their routes on disk before any prepare encodes them
            await asyncio.shield(task)
        kept, lost = [], []
        for fg in self.backlog.full:
            grp = fg["group"]
            entry = (where or {}).get(grp["id"])
            folder = saved_folder(root, entry) if entry else None
            found = _saved_groups({"roots": [], "groups": 1, "min_tries": 1, "folders": [folder]}, level_of,
                                  restored) if folder is not None else []
            if not found:
                lost.append(grp["id"])
                continue
            _, _, rows = found[0]
            self.spec(grp)  # its set, as a launch registers it (v7_specs / mg_level_ref read it)
            self.rows[grp["id"]], self.mirrored[grp["id"]] = (folder, rows), folder
            self.ahead_group(grp, folder, rows)
            kept.append(grp["id"])
        self.backlog.drop_full(lost)
        return kept, lost

    async def play_restore(self, roots):
        """The newest seq_play.json among ``roots`` for the update after this step (same policy version): its backlog
        and curriculum replace the loaded ones, in-flight groups go (their starts back), its full groups train from
        their saved folders. Returns the restored group count, 0 when none applies."""
        for root in roots[::-1]:
            path = Path(root) / "seq_play.json"
            try:
                state = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if int(state.get("step", -1)) != self.step + 1 or int(state.get("version", -1)) != self.version:
                continue
            state = remap_root(state, state.get("root") or written_root(root), root)
            self.backlog.load_state(state["backlog"])
            dropped = self.backlog.abandon_inflight()
            kept, lost = await self.restore_full(root, state.get("groups"))
            self.loop.feed.emit("seq_play_resume", root=str(root), step=self.step, groups=len(kept), lost=lost,
                                dropped_groups=dropped, saved_age_s=round(time.time() - float(state.get("t") or 0), 1),
                                gids=kept)
            return len(kept)
        return 0

    def pending_groups(self, root):
        """After a resume at step k: the saved groups of the update k + 1 that the crash cut short, else None."""
        from .rl_loop import level_of, restored

        path = Path(root) / "seq_pending.json"
        if not path.is_file():
            return None
        pend = json.loads(path.read_text())
        if int(pend.get("step", -1)) != self.step + 1 or not pend.get("groups"):
            return None
        out = []
        for g in pend["groups"]:
            folder = saved_folder(root, g)
            found = _saved_groups({"roots": [], "groups": 1, "min_tries": 1, "folders": [folder]}, level_of, restored)
            if not found:
                self.loop.feed.emit("alert", why="seq: a pending group's saved tries are missing: that update replays",
                                    gid=g["gid"], folder=str(folder))
                return None
            _, unit, rows = found[0]
            out.append((Group(g["gid"], unit, int(g["version"]), folder), rows))
        return out

    def ahead_on(self):
        red = self.loop.cfg["mg"].get("redesign")
        return isinstance(red, dict) and bool(red.get("one_call"))

    def ahead_group(self, grp, folder, rows):
        """one_call + early_ahead (U4b 27 Oct: the one call expanded all 192 tries at its start, the cards idle 54 s,
        22% of the call): each group is prepared as it finishes during play (a thread, one at a time) and every spec
        prepared so far handed to the trainer's encoders (at most every AHEAD_EVERY_S; early() writes the rest)."""
        if not (self.ahead_on() and folder is not None and rows):
            return
        g = Group(grp["id"], unit_of(grp["game"], grp["level"]), grp["version"], folder)
        self.ahead_tasks.append(asyncio.ensure_future(self.ahead_prepare(g, rows)))

    async def ahead_prepare(self, g, rows):
        async with self.ahead_lock:
            try:
                specs = await asyncio.to_thread(prepare_group, self.loop, g, rows)
            except Exception as error:  # noqa: BLE001 - the update prepares the group itself
                self.loop.feed.emit("alert", why="seq: ahead prepare failed (alert only)", gid=g.gid,
                                    error=repr(error)[:300])
                return
            if specs:
                self.ahead_specs[g.gid] = specs
            if time.time() - self.ahead_wrote >= AHEAD_EVERY_S:
                self.ahead_write()

    def ahead_write(self):
        specs = [a for v in self.ahead_specs.values() for a in v]
        self.ahead_wrote = time.time()
        return write_ahead(self.loop, specs, f"s{self.step + 1:04d}-a{len(self.ahead_specs)}-{time.time():.3f}")

    def untrained_since(self):
        """The oldest launch among the groups not trained yet (in flight, or full past this update), now when none;
        0 (no mirror prune) while one of them came from an earlier container (its launch time unknown here)."""
        gids = list(self.backlog.groups) + [fg["group"]["id"] for fg in self.backlog.full]
        if any(g not in self.launched_t for g in gids):
            return 0.0
        return min([self.launched_t[g] for g in gids] or [time.time()])

    async def served_prune(self, groups, settle=120.0):
        """R3 capture housekeeping (gate-v1 28 Oct: /tmp/arc-served-route 9.3 GB after one phase, ~1.7 KB per token, on
        the container's rootfs, so host memory): once a step has trained its groups, their tries' route records
        (``<rid>-*``, every request of the try) are deleted; files younger than ``settle`` s are kept (the minute
        mirror to <run>/served-route has not taken them yet). The mirror's passes older than the oldest untrained
        group's launch go too (untrained_since)."""
        from . import served_route

        d, mark = Path(self.loop.served_dir), self.untrained_since()

        def run():
            rids, out = set(), {"files": 0, "bytes": 0, "young": 0}
            for g, rows in groups:
                for r in rows:
                    rollout_rids(Path(g.done["folder"]) / "attempts" / str(r.get("label")) / "rollout.jsonl", rids)
            controls = set(self.control_rids)  # finished control groups' tries: never trained, pruned here
            rids |= controls
            now = time.time()
            for rid in rids:
                for f in d.glob(f"{rid}-*"):
                    try:
                        st = f.stat()
                        if now - st.st_mtime < settle:
                            out["young"] += 1
                            controls.discard(rid)  # kept for the next prune
                            continue
                        f.unlink()
                        out["files"] += 1
                        out["bytes"] += st.st_size
                    except OSError:
                        continue
            out["rids"] = len(rids)
            self.control_rids -= controls
            # leader_first warm-ups (phase.WARM_ID): R3 donors of the groups they warmed, gone once all of those trained
            for f in d.glob("chatcmpl-" + phase.WARM_ID + "*"):
                try:
                    st = f.stat()
                    if now - st.st_mtime >= settle and st.st_mtime < mark:
                        f.unlink()
                        out["warm"] = out.get("warm", 0) + 1
                        out["bytes"] += st.st_size
                except OSError:
                    continue
            dst = getattr(self.loop, "served_mirror_dir", None)
            if dst is not None:  # the volume mirror: passes older than every untrained group's start
                out["mirror"] = served_route.prune_mirror(dst, mark)
            return out

        try:
            got = await asyncio.to_thread(run)
            self.loop.feed.emit("seq_served_prune", step=self.step, **got)
        except Exception as error:  # noqa: BLE001 - housekeeping never ends the run
            self.loop.feed.emit("seq_served_prune", step=self.step, error=repr(error)[:300])

    def recycle_due(self):
        """mg.redesign.recycle {"every": n switches, "host_gib": g} (vLLM host memory grows ~15 GiB
        per M generated tokens and sleep frees none of it): the switch back to play replaces every server's processes
        once ``n`` switches passed since the last recycle, or when the host's free memory is below ``g`` GiB."""
        spec = self.cfg.get("recycle") or {}
        self.since_recycle = getattr(self, "since_recycle", 0) + 1
        if not spec:
            return False
        free = host_free_gib()
        why = ("every" if spec.get("every") and self.since_recycle >= int(spec["every"]) else
               "host" if spec.get("host_gib") and free is not None and free < float(spec["host_gib"]) else None)
        if why is None:
            return False
        self.loop.feed.emit("seq_recycle", step=self.step, why=why, since=self.since_recycle,
                            host_free_gib=None if free is None else round(free, 1))
        self.since_recycle = 0
        return True

    def pending_preview(self):
        """The saved groups a resume will train first, read from the ARC_RL1_RESUME roots before the trainer loads
        (Seq.resume needs the trainer for the checkpoint; the groups need only their folders): [(Group, rows)]."""
        roots = [x for x in os.environ.get("ARC_RL1_RESUME", "").split(",") if x]
        step, keep = None, self.step
        for i, root in enumerate(roots[::-1]):
            if (Path(root) / "seq_state.json").is_file():
                step, at = int(json.loads((Path(root) / "seq_state.json").read_text())["step"]), len(roots) - 1 - i
                break
        try:
            if step is None:  # pending-only: step 0's trainer, the update 1 the stop cut short
                self.step = 0
                for root in roots[::-1]:
                    if pending_only(root):
                        return self.pending_groups(root) or []
                return []
            self.step = step
            for root in [roots[at]] + roots[at + 1:][::-1]:
                if root == roots[at] or not (Path(root) / "seq_state.json").is_file():
                    groups = self.pending_groups(root)
                    if groups:
                        return groups
            return []
        finally:
            self.step = keep

    def ahead_pending(self, groups=None):
        """one_call + early_ahead on a resume with saved groups (real-v6 27 Oct: the pending update's 16 groups were
        prepared at the call, setup 28.7 s with both cards idle): each is prepared from loop start, beside the trainer
        load (``groups``: pending_preview, before it), and the ahead file written once they all are, so the trainer
        encodes their windows as soon as it polls. A group already started is not prepared again."""
        groups = groups if groups is not None else self.resume_groups
        started = getattr(self, "ahead_started", set())
        groups = [(g, rows) for g, rows in (groups or []) if g.gid not in started and g.gid not in self.ahead_specs]
        if not (self.ahead_on() and groups):
            return 0
        self.ahead_started = started | {g.gid for g, _ in groups}
        replay_sets(self.loop, groups)  # their sets, as the update registers them (gate-v1 28 Oct: KeyError mg_sets,
        # 0 of 16 prepared: before the trainer loads no group has launched, so v7_specs' level_norm had no set)

        async def all_then_write(groups):
            for g, rows in groups:
                await self.ahead_prepare(g, rows)
            async with self.ahead_lock:
                if self.ahead_specs:
                    self.ahead_write()
            self.loop.feed.emit("seq_ahead_pending", groups=len(groups), prepared=sum(
                1 for g, _ in groups if g.gid in self.ahead_specs))

        self.ahead_tasks.append(asyncio.ensure_future(all_then_write(groups)))
        return len(groups)

    async def ahead_join(self):
        """Every group prepare started during play, done (before the phase's early() and the update)."""
        tasks, self.ahead_tasks = self.ahead_tasks, []
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def early(self, full, groups=None):
        """The update's first group prepared and handed to the trainer's encoders before the switch
        (rl_seq.early_ahead). ``groups``: given (replay_groups), else the phase's full groups. Returns the spec count."""
        if groups is None:
            groups = []
            for fg in full:
                grp = fg["group"]
                folder, rows = self.rows.get(grp["id"], (None, []))
                if folder is not None and rows:
                    groups.append((Group(grp["id"], unit_of(grp["game"], grp["level"]), grp["version"], folder), rows))
        if self.ahead_on():  # this update's groups in its order; ones prepared during play are in ahead_specs
            during = sum(1 for g, _ in groups if g.gid in self.ahead_specs)
            for g, rows in groups:
                if g.gid not in self.ahead_specs:  # off the event loop (real-v6 N9: 16 groups, loop_lag 184 s)
                    specs = await asyncio.to_thread(prepare_group, self.loop, g, rows)
                    if specs:
                        self.ahead_specs[g.gid] = specs
            mine = [g.gid for g, _ in groups if g.gid in self.ahead_specs]
            specs = [a for gid in mine for a in self.ahead_specs.pop(gid)]  # the rest (a later update's) stay
            n = write_ahead(self.loop, specs, f"s{self.step + 1:04d}-e{len(mine)}-{time.time():.3f}")
            self.ahead_wrote = time.time()
            self.loop.feed.emit("seq_early_ahead", step=self.step + 1, specs=n, groups=len(mine), during_play=during,
                                kept=len(self.ahead_specs))
            return n
        n = await asyncio.to_thread(early_ahead, self.loop, groups, f"s{self.step + 1:04d}-{time.time():.3f}")
        self.loop.feed.emit("seq_early_ahead", step=self.step + 1, specs=n)
        return n

    async def update(self, full, groups=None):
        """The train phase: every full group (or the given ``groups``: replay_groups) into the gradient sums, one
        step; returns the new adapter's path."""
        groups = list(groups) if groups is not None else []
        for fg in full:
            grp = fg["group"]
            folder, rows = self.rows.pop(grp["id"], (None, []))
            if folder is not None and rows:
                groups.append((Group(grp["id"], unit_of(grp["game"], grp["level"]), grp["version"], folder), rows))
        path = self.loop.work / "adapters" / f"v{self.version + 1:04d}"
        kn = live_knobs()  # lr_scale / minibatches: the pre-registered minibatch fallbacks, no restart
        if not (self.loop.cfg["mg"].get("redesign") or {}).get("one_call"):
            kn.pop("minibatches", None)  # > 1 needs one_call (settings); a knob never turns it on without
            kn.pop("epochs", None)
        lr = self.lr() * kn.get("lr_scale", 1.0)
        self.loop.ckpt_budget_gib = kn.get("ckpt_budget_gib")  # None: the settings' value (rl_loop.FAST)
        shown = ("lr_scale", "minibatches", "epochs", "ratio_clip", "ckpt_budget_gib")
        if any(kn.get(k) for k in shown):
            self.loop.feed.emit("seq_train_knobs", step=self.step + 1, lr=lr, **{k: kn[k] for k in shown if k in kn})
        row = await train_step(self.loop, groups, lr, path, background=self.async_save(),
                               minibatches=kn.get("minibatches"), epochs=kn.get("epochs"),
                               ratio_clip=kn.get("ratio_clip"))
        if row is None:  # nothing trained (every group a tie): no step, the adapter stays
            self.loop.feed.emit("seq_step_skip", step=self.step + 1, groups=len(groups))
            return None
        self.step += 1
        self.version += 1
        ckpt = harvest_mod.root(self.loop) / "state" / f"s{self.step:04d}"  # the volume: a new container resumes
        began = time.time()
        background = bool(self.loop.cfg["mg"].get("async_checkpoint"))
        res = await self.loop.trainer.must("checkpoint", path=str(ckpt), **({"background": True} if background else {}))
        self.ckpt_flush()  # the previous background write was joined by this checkpoint call: its state goes first
        state = self.state_dict(ckpt, path)
        if background and (res or {}).get("background"):
            # stage H (T3 v3: 30-43 s per step for 31 GiB): host snapshot now, written by a trainer thread while the
            # next phase plays; seq_state.json follows once the folder lands (renamed from .partial: complete)
            row["checkpoint_background"] = True
            self.ckpt_pending = state
            self.ckpt_task = asyncio.ensure_future(self.ckpt_watch(str(ckpt)))
        else:
            if (res or {}).get("background_refused"):
                row["checkpoint_refused"] = res["background_refused"]
            self.put_state(state=state)
        self.loop.feed.emit("step", step=self.step, version=self.version, checkpoint_seconds=round(time.time() - began, 1),
                            **row)
        if getattr(self.loop, "served_dir", None):  # R3: the trained tries' route records leave the capture directory
            self.prune_task = asyncio.ensure_future(self.served_prune(groups))
        if self.hread_every() and self.step % self.hread_every() == 0:
            # FIFO: each export waits for the one before, so hread_due rows come in step order (CI 5d326b889)
            task = asyncio.ensure_future(self.hread_after(getattr(self, "export_last", None), path, self.step,
                                                          self.version))
            self.export_last = task
            self.exports.add(task)
            task.add_done_callback(self.exports.discard)
        return str(path)

    # ---- held-out read hand-off (arena.hread) ---------------------------------------------------------------

    def async_save(self):
        return bool(self.loop.cfg["mg"].get("async_adapter_save"))

    async def adapter_ready(self, path, need_draft=False):
        """mg.async_adapter_save: the step's adapter files landed (the trainer's background write), checked as the
        synchronous path checks them, before the servers load it. A failed write raises: the run ends and resumes from
        its last landed checkpoint (the adapter is never part of seq_state's resume)."""
        from .rl_loop import draft_files

        done = await self.loop.trainer.must("save_join")
        draft_files(path, need=need_draft)
        self.loop.feed.emit("adapter_saved", step=self.step, path=str(path),
                            **{k: done.get(k) for k in ("write_seconds", "waited", "pending")})
        return done

    def hread_every(self):
        return int((self.cfg.get("hread") or {}).get("every") or 0)

    async def hread_after(self, before, src, step, version):
        if before is not None:
            await asyncio.gather(before, return_exceptions=True)
        return await self.hread_export(src, step, version)

    async def hread_export(self, src, step, version):
        """The step's served adapter -> <volume run>/hread/s<step>/ (copy read back against its hashes), then the
        ``hread_due`` row the launch/rl/heldout/hread.py acts on. Off the loop: play goes on."""
        from . import hread

        dest = harvest_mod.root(self.loop) / "hread" / f"s{int(step):04d}"
        began = time.time()
        try:
            mask = (((getattr(self.loop, "study", None) or {}).get("server") or {}).get("expert_mask") or {}).get("sha256")
            got = await asyncio.to_thread(hread.export_adapter, src, dest, {"step": int(step), "version": int(version),
                                                                              "expert_mask_sha256": mask})
        except Exception as exc:  # the read is lost, never the run
            self.loop.feed.emit("hread_error", step=int(step), error=repr(exc)[:300])
            return None
        rel = str(dest.relative_to("/runs")) if dest.is_relative_to("/runs") else str(dest)
        self.loop.feed.emit("hread_due", step=int(step), version=int(version), path=rel, digest=got["digest"],
                            bytes=got["total_bytes"], files=len(got["files"]), draft=got["draft"],
                            expert_mask_sha256=got.get("expert_mask_sha256"), seconds=round(time.time() - began, 1))
        return got

    # ---- state ----------------------------------------------------------------------------------------------

    def ckpt_flush(self):
        """The deferred state once its checkpoint folder exists; True when written."""
        state = getattr(self, "ckpt_pending", None)
        if not state or not Path(state["checkpoint"]).is_dir():
            return False
        self.ckpt_pending = None
        self.put_state(state=state)
        self.loop.feed.emit("checkpoint_state", step=state["step"])
        return True

    async def ckpt_watch(self, path, every=2.0):
        while (getattr(self, "ckpt_pending", None) or {}).get("checkpoint") == path:
            if self.ckpt_flush():
                return
            await asyncio.sleep(every)

    async def ckpt_join(self):
        """The run's end: the last background write lands and its state is written."""
        if not getattr(self, "ckpt_pending", None):
            return
        done = await self.loop.trainer.must("checkpoint_wait", timeout=3600)
        if not (getattr(self, "ckpt_pending", None) is None or self.ckpt_flush()):
            self.loop.feed.emit("alert", why="seq: background checkpoint did not land", done=done)
        else:
            self.loop.feed.emit("checkpoint_done", **{k: v for k, v in (done or {}).items() if k != "ok"})

    def state_path(self):
        return harvest_mod.root(self.loop) / "seq_state.json"

    def state_dict(self, ckpt, adapter):
        """The loop state at this step (taken now; under async_checkpoint written once the folder lands)."""
        root = harvest_mod.root(self.loop)
        mirrored = {fg["group"]["id"]: self.where(fg["group"]["id"], root) for fg in self.backlog.full}
        return json.loads(json.dumps({"step": self.step, "version": self.version, "checkpoint": str(ckpt),
                                      "adapter": str(adapter), "backlog": self.backlog.state_dict(),
                                      "draft": draft_state(self.loop.cfg["mg"]), "root": str(root),
                                      "mirrored": {k: v for k, v in mirrored.items() if v}}, default=str))

    def put_state(self, ckpt=None, adapter=None, state=None):
        state = state or self.state_dict(ckpt, adapter)
        path = self.state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, default=str))
        tmp.replace(path)

    def play_state(self, full):
        """play_only (G-long): the backlog state after a play phase, plus every phase's full group folders, so a later
        run can start from it (seed_from)."""
        rl = Path(self.loop.output) / "rl"
        self.play_phases.append([str(rl / f"{fg['group']['id']}-{fg['group']['game']}-L{fg['group']['level']}")
                                 for fg in full])
        state = self.state_dict(None, None)
        state.update(play_only=True, phases=self.play_phases)
        self.put_state(state=state)
        return state

    def seed_from(self, sf):
        """seed_from: a fresh run starts from a play_only run's backlog and curriculum (no
        trainer state: its groups were played by the base policy, version 0 like this run's step 0). In-flight groups
        died with that run (their starts go back), its leftover full groups are dropped, starts of games outside this
        roster go; step 0 trains on its first phase's first ``groups`` full groups."""
        state = json.loads((Path(sf["root"]) / "seq_state.json").read_text())
        if not state.get("play_only") or int(state.get("version") or 0) != 0:
            raise ValueError(f"seed_from {sf['root']}: not a play_only base-policy state")
        self.backlog.load_state(state["backlog"])
        self.backlog.full = []
        dropped = self.backlog.abandon_inflight()
        games = {s["game"] for s in self.backlog.starts.values() if s["l1"]}
        roster = set(self.cfg.get("games") or rl_backlog.TRAIN)
        gone = [sid for sid, s in self.backlog.starts.items() if s["game"] not in roster]
        for sid in gone:
            del self.backlog.starts[sid]
        for g in sorted(roster - games):  # an L1 start for a roster game the seed did not play
            self.backlog.starts[f"{g}-L1"] = {"id": f"{g}-L1", "game": g, "level": 1, "state": None, "context": None,
                                              "version": None, "l1": True, "tie": self.backlog.rng.random()}
        self.seed_folders = (state.get("phases") or [[]])[0][:int(sf["groups"])]
        self.loop.feed.emit("seq_seed", root=sf["root"], groups=len(self.seed_folders), dropped_groups=len(dropped),
                            starts=len(self.backlog.starts), outside_roster=len(gone))
        return self.seed_folders

    async def resume(self):
        """The newest seq_state.json among ARC_RL1_RESUME roots: checkpoint, backlog, step. In-flight groups died
        with the earlier container: dropped, their starts back in the backlog."""
        for root in [x for x in os.environ.get("ARC_RL1_RESUME", "").split(",") if x][::-1]:
            path = Path(root) / "seq_state.json"
            if path.is_file():
                state = json.loads(path.read_text())
                state = remap_root(state, state.get("root") or written_root(root), root)  # -lost- / per-run /in/
                draft, now = state.get("draft"), draft_state(self.loop.cfg["mg"])
                if draft and not now:  # its checkpoint holds the draft's LoRA and Adam group: a draft-off trainer refuses
                    raise ValueError(f"resume {root}: its checkpoint carries the draft LoRA ({draft}); mg.draft_weight "
                                     "is off here")
                self.loop.v7_resume_root = Path(root)  # v7_found maps the earlier leg's paths into this container
                await self.loop.trainer.must("load_checkpoint", path=self.loop.v7_found(state["checkpoint"]))
                self.backlog.load_state(state["backlog"])
                dropped = self.backlog.abandon_inflight()
                self.step, self.version = int(state["step"]), int(state["version"])
                self.resume_groups, pending_root = self.pending_groups(root), root
                if not self.resume_groups:  # real-v7 27 Oct: a leg resumed from R, played its update and died with no
                    # state; its groups' seq_pending.json (step + 1) sits in its own, later root
                    roots = [x for x in os.environ.get("ARC_RL1_RESUME", "").split(",") if x]
                    for later in roots[roots.index(root) + 1:][::-1]:
                        if not (Path(later) / "seq_state.json").is_file():
                            groups = self.pending_groups(later)
                            if groups:
                                self.resume_groups, pending_root = groups, later
                                break
                played = carried = 0
                if self.resume_groups:  # the state's carried full groups are in the pending update: trained there
                    pend = {g.gid for g, _ in self.resume_groups}
                    self.backlog.full = [fg for fg in self.backlog.full if fg["group"]["id"] not in pend]
                else:  # the next update's groups played before the stop
                    roots = [x for x in os.environ.get("ARC_RL1_RESUME", "").split(",") if x]
                    played = await self.play_restore(roots[roots.index(root):])
                if not played and self.backlog.full:  # the state's (other) carried full groups, from their folders
                    carried = len((await self.restore_full(root, state.get("mirrored")))[0])
                self.loop.feed.emit("seq_resume", root=root, step=self.step, dropped_groups=dropped, draft=draft,
                                    draft_now=now, pending_groups=len(self.resume_groups or []),
                                    pending_root=pending_root, played_groups=played, carried_groups=carried)
                self.ahead_pending()
                return True
        for root in [x for x in os.environ.get("ARC_RL1_RESUME", "").split(",") if x][::-1]:
            if pending_only(root):  # step 0's trainer (fresh); the update 1 the stop cut short trains its saved tries
                groups = self.pending_groups(root)
                if groups:
                    self.loop.v7_resume_root = Path(root)
                    self.resume_groups = groups
                    self.loop.feed.emit("seq_resume", root=root, step=0, pending_only=True, pending_groups=len(groups))
                    self.ahead_pending()
                    return True
        await self.play_restore([x for x in os.environ.get("ARC_RL1_RESUME", "").split(",") if x])  # step 0's
        return False  # the trainer starts fresh (no checkpoint); any played groups are restored above


async def warm_render(server):
    """One exact count (/tokenize: nothing generated, the first-token watch stays at zero) with an image on a server
    this loop booted. H2 attempt 1 (26 Oct): with ARC_MM_WARMUP_SKIP, server 2's first counts waited 40 s on its cold
    multimodal processor while its card sat at 0% (server 1's cache probe had warmed its own). Seconds, or the error."""
    from .cache_probe import payload

    began = time.time()
    body = payload(0, 8, model=server.model)
    try:
        await server.post("/tokenize", {"model": server.model, "messages": body["messages"],
                                        "chat_template_kwargs": body["chat_template_kwargs"]}, timeout=300)
    except Exception as error:  # noqa: BLE001 - a warm-up only: play still starts
        return f"{type(error).__name__}: {error}"[:200]
    return round(time.time() - began, 1)


def server_envs(loop, extra):
    """Each play server's effective serving environment (rl_loop.server_env_record): server 1's from its process,
    the others' as SecondServer started them. Also written to
    ``<server dir>/server-env.json`` (collected into the run's arena/serving; modal_rl1 copies it to the identity)."""
    from .rl_loop import SecondServer, server_env_record

    out = {}
    if getattr(loop, "server_dir", None) is None:  # no first server (tests' fake loops)
        return out
    try:
        _, env = SecondServer.source(SimpleNamespace(server_dir=Path(loop.server_dir)))
        out["server1"] = {"port": 1234, "card": "0", "env": server_env_record(env)}
    except Exception as exc:  # noqa: BLE001 - a record only
        out["server1"] = {"port": 1234, "card": "0", "error": f"{type(exc).__name__}: {exc}"[:200]}
    for s in extra:
        env = getattr(s, "effective_env", None)
        if env is None:
            try:
                env = server_env_record(s.command()[1])
            except Exception as exc:  # noqa: BLE001
                env = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        name = getattr(getattr(s, "folder", None), "name", None) or f"server{len(out) + 1}"
        out[name] = {"port": getattr(s, "port", None), "card": getattr(s, "card", None), "env": env}
    try:
        (Path(loop.server_dir) / "server-env.json").write_text(json.dumps(out, indent=1))
    except OSError:
        pass
    return out


def contexts_of(live):
    """Live tries' contexts (latest prompt, or the restored start's)."""
    return [max(int(getattr(s, "next_estimate", 0) or 0), int(getattr(s, "floor_tokens", 0) or 0)) for s in live]


def context_spread(live):
    """Live tries' contexts (latest prompt, or the restored start's): quantiles and counts past 40K / 64K / 100K
    (G-long logs the context-length distribution over time)."""
    return spread(contexts_of(live))


def spread(contexts):
    """context_spread over a list of contexts (rl_drivers: every worker's live tries)."""
    ctx = sorted(contexts)
    if not ctx:
        return {"n": 0}
    at = lambda q: ctx[min(len(ctx) - 1, int(q * len(ctx)))]  # noqa: E731
    return {"n": len(ctx), "p10": at(0.1), "p50": at(0.5), "p90": at(0.9), "max": ctx[-1],
            **{f"ge{k}k": sum(c >= k * 1024 for c in ctx) for k in (40, 64, 100)}}


DWELL_KEYS = ("wall", "inference", "tool", "gate_wait", "harness")


def dwell_add(acc, row):
    """A finished try's seconds into ``acc``: wall, inference (requests in flight, client side), tool, KV-gate wait and
    harness (the rest: game, prompt building, driver loop)."""
    wall, inf, tool, gw = (float(row.get(k) or 0.0) for k in ("wall_seconds", "inference_seconds", "tool_seconds",
                                                               "gate_wait_seconds"))
    for k, v in zip(DWELL_KEYS, (wall, inf, tool, gw, max(0.0, wall - inf - tool - gw))):
        acc[k] = acc.get(k, 0.0) + v
    acc["tries"] = acc.get("tries", 0) + 1
    return acc


def dwell_snapshot(pool):
    """(G-long: 272 live, 189 sending, engine 146 running): where each live try is now (sending a
    request, waiting at the KV gate, in a tool, parked, other = harness/game/driver) and the live tries' summed seconds
    per kind so far."""
    now = dict.fromkeys(("sending", "gate_wait", "tool", "parked", "other"), 0)
    acc = dict.fromkeys(DWELL_KEYS, 0.0)
    for slot in list(pool.live):
        game = getattr(slot, "game", None)
        clock = getattr(game, "clock", None)
        waits = getattr(game, "admission_attempts", None) or [{}]
        if id(slot) in pool.sending:
            state = "sending"
        elif waits[-1].get("status") == "waiting":
            state = "gate_wait"
        elif clock is not None and getattr(clock, "_tool_started", None) is not None:
            state = "tool"
        elif clock is not None and getattr(clock, "parked", False):
            state = "parked"
        else:
            state = "other"
        now[state] += 1
        if clock is None or getattr(clock, "start", None) is None:
            continue
        gw = sum(float(r.get("gate_wait_seconds") or 0.0) for r in getattr(game, "requests", None) or [])
        dwell_add(acc, {"wall_seconds": clock.active_seconds(), "inference_seconds": clock.inference_seconds(),
                        "tool_seconds": clock.tool_seconds(), "gate_wait_seconds": gw})
    return {"now": now, "live_s": {k: round(v, 1) for k, v in acc.items()}}


PROC_CPU = {}  # pid -> (cpu seconds, wall) at the last read


def proc_cpu(match=("vllm", "VLLM", "EngineCore", "APIServer")):
    """CPU share (1.0 = one core) since the last call of every process whose cmdline matches, plus this driver; Linux
    /proc only (else {}). Names: the process title or argv[0..2] basename."""
    root, out, tick = Path("/proc"), {}, os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    if not root.is_dir():
        return out
    now = time.time()
    for d in root.iterdir():
        if not d.name.isdigit():
            continue
        pid = int(d.name)
        try:
            cmd = (d / "cmdline").read_bytes().split(b"\0")
            title = " ".join(c.decode(errors="replace") for c in cmd[:3] if c)
            if pid != os.getpid() and not any(m in title for m in match):
                continue
            fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
            cpu = (int(fields[11]) + int(fields[12])) / tick
        except (OSError, IndexError, ValueError):
            continue
        before = PROC_CPU.get(pid)
        PROC_CPU[pid] = (cpu, now)
        if before is None or now <= before[1]:
            continue
        name = "driver" if pid == os.getpid() else title.split()[0].rsplit("/", 1)[-1] if title else str(pid)
        if "serve" in title and name.startswith("python"):
            name = "api_server"
        share = round((cpu - before[0]) / (now - before[1]), 2)
        if share >= 0.02:
            out[f"{name}:{pid}"] = share
    return out


def engine_counts(url, max_age=60.0):
    """The engine's running and waiting requests from the guard's last read of ``url`` (LAST_METRICS), else None."""
    got = LAST_METRICS.get(url)
    if got is None or time.time() - got[0] > max_age:
        return None
    v = got[1]
    return {"running": int(float(v.get("vllm:num_requests_running", 0.0))),
            "waiting": int(float(v.get("vllm:num_requests_waiting", 0.0))), "age_s": round(time.time() - got[0], 1)}


def admit_row(seq):
    """Per server: tries admitted by the backlog (unfinished tries of its groups), the slot pool (slots, tries holding
    one, live, requests in flight, KV / turn waits) and the exact KV gate (requests in flight, waiting, reserved of
    capacity); plus the backlog's streams and parked (admitted vs running over time)."""
    pools = level_bench.SHARED_SLOTS_BY_URL or {}
    drivers = getattr(seq, "drivers", None)  # rl_drivers: the pools live in the workers (their latest state rows)
    servers = []
    for i, url in enumerate(seq.urls):
        row = {"admitted": sum(live["group"]["tries"] - len(live["results"]) for gid, live in seq.backlog.groups.items()
                               if seq.on_server.get(gid, 0) == i)}
        pool = pools.get(url) or pools.get(url.rstrip("/"))
        if drivers is not None:
            row.update(drivers.pool_row(url))
        elif pool is not None:
            row.update(slots=pool.slots, used=pool.used, live=len(pool.live), sending=len(pool.sending),
                       kv_waits=pool.kv_waits, turn_waits=pool.turn_waits, contexts=context_spread(pool.live))
            row["dwell"] = dwell_snapshot(pool)
            if pool.gate is not None:
                snap = pool.gate.snapshot()
                row.update(gate_inflight=snap["inflight"], gate_waiting=snap["waiting"], gate_reserved=snap["reserved"],
                           gate_capacity=snap["capacity"])
        servers.append(row)
    out = {"streams": seq.backlog.streams, "parked": seq.backlog.parked, "servers": servers}
    done = getattr(seq, "dwell_done", None)
    if done is not None:
        out["dwell_done"] = {k: round(v, 1) for k, v in done.items()}
    if drivers is not None:
        out["drivers"] = drivers.driver_rows()
    return out


async def admit_log(loop, seq, every):
    """``seq_admit`` feed rows every ``every`` s (mg.redesign.admit_log_s, default 30), with the dwell export: per
    server the engine's running/waiting beside the tries sending (the gap = requests between client and engine), the
    driver's event-loop lag and the CPU share of the driver and vLLM processes."""
    while True:
        await asyncio.sleep(every)
        try:
            engines = [engine_counts(u) for u in seq.urls]  # the guard's last read (no second /metrics reader)
            row = admit_row(seq)
            for server, eng in zip(row["servers"], engines):
                if eng is not None:
                    server["engine"] = {**eng, "gap": server.get("sending", 0) - eng["running"] - eng["waiting"]}
            lag = getattr(loop, "loop_lag", None)
            if isinstance(lag, dict):
                row["loop_lag_s"] = round(float(lag.get("admit") or 0.0), 2)  # max overshoot of a 1-s sleep
                lag["admit"] = 0.0
            cpu = proc_cpu()
            if cpu:
                row["cpu"] = cpu
            loop.feed.emit("seq_admit", **row)
        except Exception as error:  # noqa: BLE001 - diagnostics never stop play
            loop.feed.emit("seq_admit", error=repr(error)[:200])


def rollout_rids(path, rids):
    """A try's rollout.jsonl request ids added to ``rids`` (every request of the try; unreadable: none)."""
    try:
        with open(path) as file:
            for line in file:
                rid = json.loads(line).get("rid") if line.strip().startswith("{") else None
                if rid:
                    rids.add(str(rid))
    except (OSError, ValueError):
        pass
    return rids


def saved_rids(roots):
    """R3 restore (real-v13 29 Oct: 890 s copying 205 GB of routes for 0 saved groups): the request ids of the saved
    groups a resume from ``roots`` (run roots, resume order) can still train: the newest state's carried full groups
    and the pending and mid-play groups of the update after it. None when no root holds rl_seq state (restore all)."""
    roots = [Path(r) for r in roots]

    def load(path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    names = ("seq_state.json", "seq_pending.json", "seq_play.json")
    if not any((r / n).is_file() for r in roots for n in names):
        return None
    step, folders = 0, []
    for root in roots[::-1]:
        state = load(root / "seq_state.json")
        if state:
            step = int(state.get("step") or 0)
            state = remap_root(state, state.get("root") or written_root(root), root)
            folders += [saved_folder(root, w) for w in (state.get("mirrored") or {}).values() if w]
            break
    for root in roots:
        pend = load(root / "seq_pending.json") or {}
        if int(pend.get("step", -1)) == step + 1:
            folders += [saved_folder(root, g) for g in pend.get("groups") or []]
        play = load(root / "seq_play.json") or {}
        if int(play.get("step", -1)) == step + 1:
            play = remap_root(play, play.get("root") or written_root(root), root)
            folders += [saved_folder(root, w) for w in (play.get("groups") or {}).values() if w]
    rids = set()
    for folder in folders:
        for path in Path(folder).glob("attempts/*/rollout.jsonl"):
            rollout_rids(path, rids)
    return rids


def served_start(loop, cfg):
    """R3 (missing from rl_seq since it began, C-S4 review 28 Oct): with ``mg.served_route`` the trainer
    learns on the play server's routing. Window specs carry ``server.served_route`` (Loop.windows reads
    ``loop.served_dir``), the trainer starts with RL1_ROUTER=served (rl_loop.trainer_command), finished route records
    are mirrored to <run>/served-route every minute, and a resume copies its roots' mirrors back before anything
    encodes. Returns the restore task (None when off)."""
    if not loop.cfg["mg"].get("served_route") or cfg.get("play_only") or getattr(loop, "served_dir", None):
        return None
    from .served_route import restore

    loop.served_dir = ((getattr(loop, "study", None) or {}).get("server") or {}).get("served_route")
    if not loop.served_dir:
        raise ValueError("mg.served_route needs server.served_route (the capture directory)")
    out = os.environ.get("ARC_VALIDATION_OUTPUT")
    runs = [r for r in os.environ.get("ARC_RL1_RESUME", "").split(",") if r]
    roots = [Path(r) / "served-route" for r in runs]

    progress = {}  # restore's tars_done / tars_total / files / bytes, read by the progress rows

    def restore_saved():
        rids = saved_rids(runs)
        progress["rids"] = None if rids is None else len(rids)
        return {**restore(roots, loop.served_dir, rids, progress=progress), "rids": progress["rids"]}

    async def ticking(began, every=30.0):  # a restore's progress, every 30 s
        while True:
            await asyncio.sleep(every)
            loop.feed.emit("served_restore_progress", seconds=round(time.time() - began, 1), **dict(progress))

    async def restoring():
        began = time.time()
        tick = asyncio.ensure_future(ticking(began))
        try:
            got = await asyncio.to_thread(restore_saved) if roots else {"files": 0, "roots": 0}
            loop.feed.emit("served_restore", seconds=round(time.time() - began, 1), **got)
        except Exception as error:  # noqa: BLE001 - a failed restore costs only those routes (the audit shows it)
            loop.feed.emit("served_restore", error=repr(error)[:300])
        finally:
            tick.cancel()

    loop.served_restore_task = asyncio.ensure_future(restoring())
    if out:
        loop.served_mirror_dir = Path(out) / "served-route"
        loop.served_mirror_task = asyncio.ensure_future(served_mirror(loop, loop.served_mirror_dir))
    return loop.served_restore_task


async def served_mirror(loop, dst, every=60.0, status_every=300.0):
    """Finished route records -> ``dst`` (arena.served_route.mirror) every ``every`` s; the capture's state in the
    feed (served_status) every ``status_every`` s."""
    from .served_route import mirror, status

    seen, last = {}, time.time()
    while True:
        await asyncio.sleep(every)
        try:
            got = await asyncio.to_thread(mirror, loop.served_dir, dst, seen)
        except Exception as error:  # noqa: BLE001 - the next pass retries
            loop.feed.emit("served_mirror_failed", error=repr(error)[:300])
            continue
        if time.time() - last >= status_every:
            last = time.time()
            state = await asyncio.to_thread(status, loop.served_dir)
            loop.feed.emit("served_status", mirrored=got, rt=state.get("rt"), bytes=state.get("bytes"),
                           server=state.get("server"))


async def served_on(loop):
    """After the trainer starts: its router confirmed "served" (the route op; a no-op change under RL1_ROUTER)."""
    if not getattr(loop, "served_dir", None):
        return
    loop.feed.emit("served_route", dir=loop.served_dir, **(await loop.trainer.must("route", router="served")))


async def run(loop, gate=None):
    """The redesign loop. Returns the stop reason. ``gate``: tests' PhaseGate (else one over the play servers)."""
    cfg = settings(loop.cfg["mg"])
    if getattr(loop, "mg_screen", None) is None:  # no screen in the redesign: references from the history and sets
        loop.mg_screen = {"sets": {}}  # (v7_specs' level_norm and mg_level_ref read it); pf R3 died without it
    seq = Seq(loop, cfg)
    restoring = served_start(loop, cfg)  # R3: before any window spec is made (F1 prepares at the trainer's load)
    servers = [loop.server]
    from .rl_loop import SecondServer, Server, draft_files

    for url in cfg.get("servers") or []:
        if url.rstrip("/") != loop.server.root and url.rstrip("/") + "/v1" != loop.base_url.rstrip("/"):
            servers.append(Server(url, loop.server.model))
    static = (((getattr(loop, "study", None) or {}).get("server") or {}).get("lora") or {}).get("static") or {}
    static = static.get("name")
    if static:  # server.lora.static (api_server_count > 1): the zero adapter is served from boot, never swapped
        if not cfg.get("play_only") or cfg.get("adapter") != static:
            raise ValueError(f"server.lora.static {static!r}: play_only runs serving that adapter only (no switch)")
    began = time.time()
    per_card = card_env(cfg)
    extra = [SecondServer(loop.server_dir, loop.server.model, card=i, port=1235 + i, gpu_util=None, name=f"server{i + 1}",
                          **({"env": per_card[i]} if per_card.get(i) else {})) for i in range(1, int(cfg.get("cards") or 1))]
    boots = []
    # every server's image processor (study.cache_probe false: server 1 is cold too), joined before play; server 1's
    # while the other cards boot (26 Oct startup cut)
    warm_first = asyncio.ensure_future(warm_render(servers[0]))
    if extra:  # every card's server boots at once, beside the first (the trainer loads after they sleep)
        boots = await asyncio.gather(*(s.boot() for s in extra))
        servers += extra
    warm = asyncio.ensure_future(asyncio.gather(warm_first, *(warm_render(s) for s in servers[1:])))
    for server in servers if static else ():
        server.static_lora = static
    if cfg.get("recycle") and getattr(loop, "server_dir", None) is not None:  # arena.server_recycle's handles
        servers[0].recycle_identity = Path(loop.server_dir) / "vllm-server-identity.json"
        servers[0].recycle_log = Path(loop.server_dir) / "vllm-openai-server.log"
        for server in servers[1:]:
            server.recycle_log = server.folder / "vllm.log"
    # stage H: the launch-to-ready timeline (container start = 0) for the 10-min check, every run (one card too)
    zero = float(os.environ.get("ARC_RL1_STARTED_EPOCH") or 0) or None
    try:
        line = boot_timeline.timeline(loop.server_dir)
        if extra:
            line["extra"] = boot_timeline.extra_phases([s.folder for s in extra], zero)
    except Exception as exc:  # noqa: BLE001 - a report only
        line = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    envs = server_envs(loop, extra)
    loop.feed.emit("seq_boot", servers=len(extra), boots=boots, seconds=round(time.time() - began, 1),
                   ready_t=boot_timeline.since_launch(), timeline=line, server_env=envs)
    need_draft = draft_on(loop.cfg["mg"])  # stage D: its files ride in the adapter folder
    if need_draft:  # a draft no server merges trains for nothing: refused before the trainer loads
        bad = draft_serving(envs)
        if bad:
            warm.cancel()
            for server in extra:
                server.stop()
            raise ValueError(f"mg.draft_weight > 0: {bad}")
        unread = [name for name, rec in envs.items() if not isinstance(rec.get("env"), dict) or rec.get("error")]
        if unread or not envs:
            loop.feed.emit("alert", why=f"draft trains; play server env unreadable ({unread or 'no record'}): "
                                        "ARC_DRAFT_LORA=1 unchecked")
    # groups are pinned to servers (Seq.launch), each server with its own KV-aware slot pool (the submission's gate:
    # hybrid_blocks admission, 0 preemptions): the gate only holds, aborts and resends
    n_drivers = int(cfg.get("drivers") or 0)
    if n_drivers:  # rl_drivers: the workers' gates behind one proxy; the pool starts with play (below)
        from . import rl_drivers

        seq.drivers = rl_drivers.DriverPool(n_drivers, loop, cfg)
        gate = seq.drivers.gate
    mg = loop.cfg["mg"]
    gate = gate or phase.PhaseGate(leader_first=mg.get("leader_first"))
    seq.urls = [s.root + "/v1" for s in servers]
    if mg.get("active_tries"):
        level_bench.SHARED_SLOTS_BY_URL = {
            u: level_bench.SharedSlots(int(mg["active_tries"]), mg.get("active_kv_fill"), project=mg.get("kv_project"),
                                       stagger_s=mg.get("admit_stagger_s") or 0.0, shared_gate=True) for u in seq.urls}
    level_bench.PHASE_GATE = gate if seq.drivers is None else None  # with drivers, no bench plays here
    level_bench.ENGINE_POLL = False  # the guard is the one periodic /metrics reader (G-long v5 N3)
    hook_before = level_bench.PROGRESS_HOOK

    def hook(bench, row):
        if hook_before is not None:
            hook_before(bench, row)
        seq.hook(bench, row)

    level_bench.PROGRESS_HOOK = hook

    deferred = {}  # defer_trainer: the trainer's start task and its GPU gate until the first switch joins them

    async def park():
        res = await loop.trainer.must("park", base=True)
        if (res or {}).get("base_shm"):  # the first park's base snapshot split (rank 0; every rank: util/park-base-shm-r*)
            loop.feed.emit("park_base_shm", **res["base_shm"])
        res = res or {}  # gate leg 4: every park's split (own-state copy, prepin wait, base snapshot)
        loop.feed.emit("seq_park", seconds=res.get("seconds"), gib=res.get("gib"), base_gib=res.get("base_gib"),
                       prepin_wait_s=(res.get("prepin") or {}).get("wait_s"),
                       snapshot_s=(res.get("base_shm") or {}).get("seconds"),
                       **({"release_host": res["release_host"]} if res.get("release_host") else {}))

    async def unpark():
        if deferred.get("task") is not None:  # every server sleeps now: the trainer's card part may load
            began_join = time.time()
            deferred["gate"].write_text(json.dumps({"epoch": began_join}))
            started = await deferred.pop("task")
            loop.feed.emit("seq_trainer_joined", waited_s=round(time.time() - began_join, 1),
                           since_spawn_s=round(time.time() - deferred["spawned"], 1), trainer=started)
            await served_on(loop)
            return
        res = await loop.trainer.must("unpark") or {}
        loop.feed.emit("seq_unpark", **{k: res.get(k) for k in ("seconds", "gib", "base_s", "base_gib", "shm_register",
                                                                 "base_verify") if res.get(k) is not None})

    ctl = phase.PhaseController([gate], servers, park, unpark, adapter=cfg["adapter"],
                                memory=lambda: asyncio.to_thread(cards_used))
    mg.setdefault("trainer_cards", int(cfg.get("cards") or 1))  # one trainer rank per card (rl1_dist, torchrun)
    template = None
    if need_draft and cfg.get("defer_trainer") not in (None, False):  # the zero adapter carries no draft LoRA
        loop.feed.emit("alert", why="defer_trainer off: the draft trains (stage D); the trainer loads first")
    elif cfg.get("init_checkpoint") and cfg.get("defer_trainer") not in (None, False):  # step 0 serves the checkpoint
        loop.feed.emit("alert", why="defer_trainer off: init_checkpoint; the trainer loads first")
    elif cfg.get("defer_trainer") not in (None, False) and getattr(loop, "trainer", None) is None and not resume_state_exists():
        template = defer_template(loop, cfg)
        if template is None:
            loop.feed.emit("alert", why="defer_trainer: no template adapter; the trainer loads first")
    baseline = {}
    marks = {"t0": time.time()}

    async def kv_dump():
        """mg.redesign.kv_golden (E4 A/B, E1; off by default): the fixed-prompt dump on every server under the play
        adapter, after it loads and before the gate opens; a seq_boot row; the first-token zero taken after it."""
        spent = None
        if cfg.get("kv_golden"):
            from . import kv_golden

            spec = {**cfg["kv_golden"], "probes": str(Path(loop.source) / cfg["kv_golden"]["probes"])}  # bundle path
            if spec.get("adapter"):  # a trained adapter: the play adapter is zero-delta here (lora-l1)
                spec["adapter"] = str(Path(loop.source) / spec["adapter"])  # an absolute path stays itself
            row = await kv_golden.boot_dump(servers, spec, loop.output / "kv-golden", model=cfg["adapter"],
                                            server_dir=loop.server_dir)
            loop.feed.emit("seq_boot", stage="kv_dump", kv_dump=row)
            spent = row["seconds"]
        if cfg.get("r3_check"):  # H5 (26 Oct): R3 rows through the host KV tier, A/S/B replay on every server
            from . import r3_offload_check

            spec = {**cfg["r3_check"], "probes": str(Path(loop.source) / cfg["r3_check"]["probes"])}  # bundle path
            dirs = [loop.server_dir] + [s.folder for s in servers if getattr(s, "folder", None)]
            row = await asyncio.to_thread(r3_offload_check.boot_check, servers, spec, loop.output / "r3-check",
                                          model=cfg["adapter"], status_dirs=dirs,
                                          route_dir=((getattr(loop, "study", None) or {}).get("server") or {})
                                          .get("served_route"))
            loop.feed.emit("seq_boot", stage="r3_check", r3_check=row)
            loop.feed.emit("r3_offload", metrics=row["metrics"])  # exporter: infra.r3_offload_bound and refusals
            if not row["ok"]:
                loop.feed.emit("alert", why=row["why"])
            spent = (spent or 0.0) + row["seconds"]
        if spent is not None:
            for url in seq.urls:  # its one-token replies are not play
                baseline[url] = await boot_timeline.generated(url) or 0.0
        return spent
    if template is not None:  # defer_trainer: play on the zero adapter while the trainer's CPU part runs
        path = loop.work / "adapters" / "v0000"
        tensors = zero_adapter(template, path)
        deferred["gate"] = loop.work / "trainer-gpu-gate"
        deferred["gate"].unlink(missing_ok=True)
        deferred["spawned"] = time.time()
        if not cfg.get("play_only"):  # G-long (27 Oct): generation only, no trainer process beside play
            deferred["task"] = asyncio.ensure_future(loop.start_trainer(gate=deferred["gate"]))
        resumed = False
        if cfg.get("seed_from"):
            seq.seed_from(cfg["seed_from"])
        else:  # a step-0 leg stopped mid-play: its played groups and backlog (the trainer starts fresh either way)
            await seq.play_restore([x for x in os.environ.get("ARC_RL1_RESUME", "").split(",") if x])
        for url in seq.urls:
            baseline[url] = await boot_timeline.generated(url) or 0.0
        t1 = time.time()
        await asyncio.gather(*(s.load(cfg["adapter"], str(path)) for s in servers))  # servers awake since boot
        marks["loaded"] = time.time()
        warmed, golden_s = await asyncio.gather(warm, kv_dump())  # /tokenize warm-ups beside the boot checks
        marks["play"] = time.time()
        first = {"load_s": round(marks["loaded"] - t1, 3), "switch_s": round(marks["play"] - t1, 3)}
        phases = {"zero_adapter_tensors": tensors, "deferred_trainer": True, "warm_render_s": warmed,
                  "kv_golden_s": golden_s}
    else:
        warmed = await warm  # before the servers sleep
        await asyncio.gather(*(s.sleep() for s in servers))  # the trainer loads on the free cards
        marks["slept"] = time.time()
        if getattr(loop, "trainer", None) is None:
            if restoring is not None:
                await restoring  # the saved groups' routes on disk before the trainer can encode them
            seq.ahead_pending(seq.pending_preview())  # the saved groups' prepare beside the trainer load (F1)
            started = await loop.start_trainer(**prespawned_gate(loop))
            if isinstance(started, dict) and started.get("prespawn"):  # adopted or not, and why (env/command diff)
                loop.feed.emit("seq_trainer_prespawn", seconds=started.get("seconds"), prespawn=started["prespawn"])
            await served_on(loop)
        marks["trainer"] = time.time()
        ctl.phase = "train"
        resumed = await seq.resume()
        if not resumed and cfg.get("seed_from"):
            seq.seed_from(cfg["seed_from"])
        if not resumed and cfg.get("init_checkpoint"):  # a fresh start from a trained policy (policy-only: fresh draft)
            got = await loop.trainer.must("load_checkpoint", path=cfg["init_checkpoint"])
            loop.feed.emit("seq_init", path=cfg["init_checkpoint"],
                           **{k: v for k, v in (got or {}).items() if k in ("updates", "adapter_norm", "seconds")})
        path = loop.work / "adapters" / f"v{seq.version:04d}"
        await loop.trainer.must("save", path=str(path))  # step 0: the initial LoRA (B = 0) under the served name
        draft_files(path, need=need_draft)
        marks["saved"] = time.time()
        for url in seq.urls:  # tokens generated before play (cache probe, warm-ups): the first-token watch's zero
            baseline[url] = await boot_timeline.generated(url) or 0.0
        dumped = {}

        async def before_open():
            dumped["s"] = await kv_dump()

        first = await ctl.to_play(str(path), before_open=before_open)
        golden_s = dumped.get("s")
        marks["play"] = time.time()
        phases = {k: round(v, 1) for k, v in {
            "sleep_s": marks["slept"] - marks["t0"], "trainer_s": marks["trainer"] - marks["slept"],
            "save_s": marks["saved"] - marks["trainer"], "switch_s": marks["play"] - marks["saved"]}.items()}
        phases["warm_render_s"] = warmed
        phases["kv_golden_s"] = golden_s
    loop.feed.emit("seq_start", resumed=resumed, step=seq.step, servers=[s.root for s in servers],
                   streams=seq.backlog.streams, setup_seconds=round(time.time() - began, 1), switch=first,
                   play_t=boot_timeline.since_launch(), phases=phases)
    if seq.hread_every() and not resumed:  # the base read (no adapter) is due with the run's start
        loop.feed.emit("hread_due", step=0, version=0, path=None, draft=None)

    async def watch_first_token():
        waited = await boot_timeline.first_token(seq.urls, baseline)
        if waited is not None:
            loop.feed.emit("seq_first_token", since_play_s=waited, first_token_t=boot_timeline.since_launch())
            # R3 through the host KV tier: an unbound rows hook recomputes every host hit, silently: a feed alert
            # (the verdict fails) and infra.r3_offload_bound in train-metrics.jsonl (arena.r3_offload_check)
            from . import r3_offload_check

            if r3_offload_check.offload_on(getattr(loop, "study", None), getattr(loop, "trial", None)):
                rows = []  # emitted on the loop, not the thread
                st = await asyncio.to_thread(r3_offload_check.watch, loop.study, loop.trial, [loop.server_dir],
                                             emit=lambda kind, **kw: rows.append((kind, kw)),
                                             metrics_path=Path(loop.output) / "train-metrics.jsonl")
                loop.feed.emit("r3_offload", metrics=r3_offload_check.status_metrics(st))  # exporter: infra.* keys
                for kind, kw in rows:
                    loop.feed.emit(kind, **kw)

    first_token = asyncio.ensure_future(watch_first_token())
    admit_task = asyncio.ensure_future(admit_log(loop, seq, float(cfg.get("admit_log_s") or 30.0)))
    guard, guard_task = None, None  # the play phase's server guard (KV peak, preemptions): the next phase's back-off
    held = {}  # url -> slots before a blind hold

    kv_held = {}

    def kv_hold(url, kv):
        """server_guard's on_kv: live knob seq.kv_cap (KV 0.77-0.79 evicted the cached contexts of tries
        between turns, gen 2.4K -> 1.3-1.8K): at kv >= cap the server's KV gates admit nothing new while anything is
        in flight, until kv < cap - KV_HYST. kv None (phase end) releases."""
        cap = live_knobs().get("kv_cap") if kv is not None else None
        was = kv_held.get(url, False)
        want = kv_want(cap, kv, was)
        if want == was:
            return
        kv_held[url] = want
        if seq.drivers is not None:
            seq.drivers.knobs({"kv_hold": {url: want}})
        else:
            pool = (level_bench.SHARED_SLOTS_BY_URL or {}).get(url)
            if pool is not None and pool.gate is not None:
                pool.gate.hold(want)
        if kv is not None:
            loop.feed.emit("seq_kv_hold", url=url, hold=want, kv=round(kv, 4), cap=cap)

    def blind_hold(url, on):
        """server_guard's on_blind: a server it cannot read takes no new tries (slots down to its live tries) until a
        read comes back (fail closed). on None: the phase ended blind (released, not seen)."""
        pool = (level_bench.SHARED_SLOTS_BY_URL or {}).get(url)
        if pool is None:
            loop.feed.emit("alert", why=f"guard blind: {url} /metrics unread {BLIND_AFTER} polls (no pool to hold)")
            return
        if on and url not in held:
            used = (seq.drivers.pool_row(url).get("used") if seq.drivers is not None else pool.used) or 0
            held[url], pool.slots = pool.slots, max(1, min(pool.slots, int(used)))
            if seq.drivers is not None:  # each worker holds at its own live tries (an even split would leak)
                seq.drivers.knobs({"hold": [url]})
            loop.feed.emit("alert", why=f"guard blind: {url} /metrics unread {BLIND_AFTER} polls; admission held at "
                                        f"{pool.slots} (was {held[url]})", url=url, slots=pool.slots, was=held[url])
        elif not on and url in held:
            pool.slots = held.pop(url)
            if seq.drivers is not None:
                seq.drivers.knobs({"active_tries": pool.slots})
            loop.feed.emit("seq_guard_sees" if on is False else "seq_guard_release", url=url, slots=pool.slots)
    seq.backoff = {}
    why = "max_steps"
    try:
        if seq.drivers is not None:  # its try_done rows go to the same hook; none alive: the run stops

            def all_dead():
                seq.out_of_time, seq.stop_why = True, "drivers_dead"
                seq.phase_done.set()

            await seq.drivers.start(seq.urls, on_try=hook, on_all_dead=all_dead)
        if cfg.get("lr_probe"):  # P1: play the probe's data, then every rate trains on it (no step is served)
            why = await lr_probe(loop, seq, ctl, cfg)
        if cfg.get("score_check"):  # the checkpoint's adapter, served vs trainer; nothing plays or trains
            from . import score_check

            why = await score_check.run(loop, ctl, servers[0], score_check.settings(cfg),
                                        {"name": cfg["adapter"], "path": str(path)}, loop.server.model)
        while not cfg.get("lr_probe") and not cfg.get("score_check") and seq.step < int(cfg["max_steps"]):
            spec0 = await spec_snapshot(seq.urls)  # the phase's MTP counters from here (the adapter just loaded)
            seq.phase_done.clear()
            pools_now = level_bench.SHARED_SLOTS_BY_URL
            slots_now = max((p.slots for p in pools_now.values()), default=0)
            knobs, cut = backoff(live_knobs(), guard, seq.backoff, slots_now) if slots_now else (live_knobs(), None)
            changed = apply_knobs(loop, seq, knobs)  # item 5: live knobs at each play phase's start
            if seq.drivers is not None and "active_tries" in changed:  # each worker's pools: ceil(n / drivers)
                seq.drivers.knobs({"active_tries": changed["active_tries"][1]})
            if changed or cut:
                loop.feed.emit("seq_knobs", step=seq.step, changed=changed, backoff=cut)
            guard_stop = asyncio.Event()
            guard_task = asyncio.ensure_future(server_guard(seq.urls, guard_stop, on_blind=blind_hold,
                                                         on_kv=kv_hold))
            t0, replayed = time.time(), None
            seeded = seq.step == 0 and bool(seq.seed_folders)  # seed_from: the seed run's first phase is step 0's data
            if seq.resume_groups:  # the update a crash cut short trains its saved tries, nothing replays
                replayed, seq.resume_groups = seq.resume_groups, None
                replay_sets(loop, replayed)
                loop.feed.emit("seq_replay", groups=len(replayed), tries=sum(len(r) for _, r in replayed),
                               folders=[g.done["folder"] for g, _ in replayed], resumed_pending=True,
                               gids=[g.gid for g, _ in replayed])
                out = {"full": [], "carried": []}
            elif seeded or (cfg.get("replay_groups") and seq.step < int(cfg["replay_groups"].get("updates", 1))):
                replayed = (replay_folders(seq.seed_folders, int(cfg["seed_from"].get("min_tries") or 1)) if seeded
                            else replay_groups(cfg["replay_groups"], seq.step))  # update leg (RL 27 Oct): no play phase
                replay_sets(loop, replayed)
                loop.feed.emit("seq_replay", groups=len(replayed), tries=sum(len(r) for _, r in replayed),
                               folders=[g.done["folder"] for g, _ in replayed])
                out = {"full": [], "carried": []}
            else:
                if not seq.backlog.phase_done:  # a resume can bring a phase's groups already full (play_restore)
                    for grp in seq.backlog.start_phase():
                        seq.launch(grp)
                seq.pump()
                watch = asyncio.ensure_future(host_watch(loop, seq, cfg["host_close_gib"]))
                saver = asyncio.ensure_future(seq.play_saver())
                try:
                    await seq.phase_done.wait()
                finally:
                    watch.cancel()
                    saver.cancel()
                if seq.out_of_time:
                    why = seq.stop_why or "deadline"
                    break
                out = seq.backlog.end_phase()
                for fg in out["full"]:  # their tries are over: the benches only write their reports
                    task = seq.tasks.pop(fg["group"]["id"], None)
                    if task is not None:
                        await task
            play_s = round(time.time() - t0, 1)
            if cfg.get("play_only"):  # G-long: the phase's groups close, carried tries play on, no switch, no step
                seq.play_state(out["full"])  # seed_from reads it
                guard_stop.set()
                guard = await guard_task
                loop.feed.emit("seq_play_phase", step=seq.step, play_seconds=play_s, groups=len(out["full"]),
                               carried=len(out["carried"]), stats=dict(seq.backlog.stats), guard=guard)
                if os.environ.get("ARC_PREFLIGHT_STUB") and len(seq.play_phases) >= int(cfg["max_steps"]):
                    break  # a preflight: max_steps play phases (G-long N1 played to its 40-min deadline, 27 Oct)
                continue
            loop.feed.emit("seq_train", step=seq.step, play_seconds=play_s,
                           groups=len(replayed if replayed is not None else out["full"]))  # watch: play stops
            await seq.ahead_join()  # one_call: the group prepares started during play
            if replayed is None:  # the phase's tries saved on the volume before the update starts
                await seq.mirror_join()
                n_pending = seq.put_pending(out["full"])
                loop.feed.emit("seq_pending", step=seq.step + 1, groups=n_pending, full=len(out["full"]))
            await seq.early(out["full"], replayed)  # early_ahead: the first group's encodes before the switch
            spec1 = await spec_snapshot(seq.urls)
            mtp = phase_mtp(loop, seq, spec0, spec1, need_draft)
            guard_stop.set()
            guard = await guard_task  # this phase's KV peak and preemptions: the next phase's back-off
            stop = await ctl.to_train()
            adapter = await seq.update(out["full"], replayed)  # a failed step ends the run (the loop's error row)
            if adapter is not None and not seq.async_save():  # async: checked once the files land (before_load)
                draft_files(adapter, need=need_draft)
            # mg.async_adapter_save: the step's background write lands after the park and wake, before the load
            ctl.before_load = ((lambda p: seq.adapter_ready(p, need_draft)) if adapter is not None and seq.async_save()
                               else None)
            start = await ctl.to_play(adapter, recycle=seq.recycle_due())
            seq.backlog.on_update(seq.version, played=replayed is None)  # a pending / replayed update ages no start
            loop.feed.emit("seq_phase", step=seq.step, play_seconds=play_s, stop=stop, start=start,
                           carried=len(out["carried"]), stats=dict(seq.backlog.stats),
                           top=[r[0] for r in seq.backlog.ranking()[:8]], mtp=mtp, guard=guard)
            manual = os.environ.get("ARENA_STOP_REQUEST")
            if manual and Path(manual).exists():
                why = "manual"
                break
    finally:
        first_token.cancel()
        admit_task.cancel()
        if guard_task is not None and not guard_task.done():  # a deadline or error mid-phase
            guard_task.cancel()
        if getattr(seq, "ckpt_task", None) is not None:  # ckpt_join below writes a landed state
            seq.ckpt_task.cancel()
        if deferred.get("task") is not None:  # stopped before the first switch: the loop's teardown ends the process
            deferred["task"].cancel()
        for task in list(seq.tasks.values()) + list(seq.pending) + list(seq.controls.values()):
            task.cancel()
        await asyncio.gather(*seq.tasks.values(), *seq.pending, *seq.controls.values(), return_exceptions=True)
        if seq.drivers is not None:  # stopped and joined: no worker outlives the run
            await seq.drivers.stop()
        level_bench.PROGRESS_HOOK, level_bench.PHASE_GATE, level_bench.SHARED_SLOTS_BY_URL = hook_before, None, {}
        level_bench.ENGINE_POLL = True
        await gate.shutdown()
        for i, server in enumerate(extra, 1):  # every server's log next to the first one's
            server.stop()
            try:
                shutil.copyfile(Path(server.folder) / "vllm.log", Path(loop.output) / f"vllm-openai-server-{i}.log")
            except (OSError, TypeError, AttributeError):
                pass
        if seq.exports:  # a step's adapter hand-off still copying: it finishes (its feed row says so)
            await asyncio.gather(*seq.exports, return_exceptions=True)
    if getattr(loop, "trainer", None) is not None:
        await seq.ckpt_join()  # async_checkpoint: the last write lands and its state is written
    if seq.version and os.environ.get("ARC_VALIDATION_OUTPUT"):
        src = loop.work / "adapters" / f"v{seq.version:04d}"
        if src.is_dir():
            dest = Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "adapters" / f"v{seq.version:04d}"
            await asyncio.to_thread(shutil.copytree, src, dest, dirs_exist_ok=True)
    loop.feed.emit("seq_end", why=why, step=seq.step, version=seq.version)
    return why
