"""The RL loop's host (trial kind ``rl-loop``; ``run_rl_loop``): settings, the play servers, the trainer service
(``scripts/posttrain/rl1_trainer.py``), group selection and rewards (``Loop.v7_prepare``) and the play of one group
(``Loop.mg_play``). The loop itself is ``arena.rl_seq`` (``mg.redesign``). Feed: ``<output>/rl/feed.jsonl`` (one event
per line) and ``<output>/rl/status.json`` (the latest of each kind).
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


DEFAULTS = {  # rl-min C: only the settings rl_seq, its windows and modal_rl1 read; any other key is refused
    "mode": "probe",
    "trainer_python": "/workspace/axolotl-venv/bin/python",
    "trainer_script": "rl1/rl1_trainer.py",             # in the bundle, beside its imports
    "model_dir": "/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1",
    # path_chunk: the trainer's one prefix chunking for pi_old, updates, probes and scores (20 Oct; rl1-gold: 16K vs 32K
    # differ by 0.054 nats/token at the same weights); 16,384 is the probes' and KL floor's chunking
    "path_chunk": 16384,
    "rank": 32, "optimizer": "adamw8bit", "loop_lora": False, "tis_cap": 0.0, "window": 24576, "lr": 6e-5, "clip": 1.0,
    "trainer_env": None,  # trainer_env: extra environment for the trainer (diagnostics)
    "attention": None,  # RL1_ATTENTION
    "mg": None,
    "lvl_window": 32768, "l2_norm": "try", "l2_per_side": 3,  # the windows' length cap, norm and per-side count
    "trainer_timeout": 5400,
}


def adapter_tensors(path):
    """The tensor count of a saved adapter, checked against the manifest its trainer wrote (an
    adapter or overlay that loads fewer tensors than expected refuses)."""
    import struct

    folder = Path(path)
    try:
        manifest = json.loads((folder / "adapter-tensors.json").read_text())
        with open(folder / "adapter_model.safetensors", "rb") as file:
            header = json.loads(file.read(struct.unpack("<Q", file.read(8))[0]))
    except (OSError, ValueError, struct.error) as exc:
        raise ValueError(f"adapter {folder}: no readable tensor manifest or file ({exc})") from exc
    count = sum(1 for k in header if k != "__metadata__")
    if not 0 < count == int(manifest["tensors"]) == int(manifest["expected"]):
        raise ValueError(f"adapter {folder}: file {count} tensors, manifest {manifest}")
    return count


def draft_files(path, need=False):
    """The draft LoRA modules beside a saved adapter (rl1_draft.export; ARC_DRAFT_LORA servers merge them at load):
    0 without draft_lora.json (refused when ``need``); a json without its full tensor file refuses."""
    import struct

    folder = Path(path)
    if not (folder / "draft_lora.json").is_file():
        if need:
            raise ValueError(f"adapter {folder}: the draft trains but no draft_lora.json beside it")
        return 0
    try:
        modules = json.loads((folder / "draft_lora.json").read_text())["modules"]
        with open(folder / "draft_lora.safetensors", "rb") as file:
            header = json.loads(file.read(struct.unpack("<Q", file.read(8))[0]))
    except (OSError, ValueError, KeyError, struct.error) as exc:
        raise ValueError(f"adapter {folder}: unreadable draft LoRA files ({exc})") from exc
    if not 0 < sum(1 for k in header if k != "__metadata__") == 2 * len(modules):
        raise ValueError(f"adapter {folder}: draft file {len(header)} tensors for {len(modules)} modules")
    return len(modules)


# every mg / mg set key the loop accepts (an unknown key refuses, never ignored); rl-min C: only the
# keys rl_seq, Loop's group path, modal_rl1 and the sims read
MG_KEYS = {"active_kv_fill", "active_tries", "admit_stagger_s", "async_adapter_save", "async_checkpoint", "balance",
    "batch_max", "batch_tokens", "cap", "digest_gate", "draft_lr_scale", "draft_weight", "fail_margin", "fail_rule",
    "gate_overcommit", "group_size", "kv_project", "leader_first", "level_norm", "load_digest", "log_clip", "log_fail", "log_rhae_min",
    "placement", "redesign", "ref_fallback", "reload_check", "reply_filter", "reward_ref", "reward_shape", "rhae_cap",
    "served_route", "trainer_cards", "transport_backoff", "turn_credit"}


# The trainer's MFU stages, on every op (prune: mg.trainer_fast and its ~30 flags are gone; these are the
# values real-v5..v9 ran). The kernels' reference paths stay only as the exactness tests' oracles. ckpt_budget_gib 240
# (was 250; v17 29 Oct: card peaks 268.4-268.6 of 268.59 GiB beside the asleep server's 7-10 GiB); live knob seq.ckpt_budget_gib.
FAST = {"prefix_snapshots": True, "prefetch": 4, "encode_procs": 4, "window_cache": 96, "union_attention": True,
        "gdn_fast": True, "chunked_head": True, "fast_dequant": True, "fused_experts": True, "sel_cache": True,
        "hyper_fast": True, "prefix_nograd": True, "ngram_pinned": True, "probe_pinned": True, "feed_stream": True,
        "vision_ahead": True, "fast_topk": True, "fast_topk_wide": "filter", "hyper_xb": True, "cast_once": True,
        "gdn_keep": True, "adaptive_ckpt": True, "ckpt_calibrate": True, "ckpt_budget_gib": 240, "ckpt_gap_gib": 10,
        "whole_segment": True, "fit_tokens": 65536, "tail_ckpt": "offload", "tail_ckpt_min": 98304,
        "early_ahead": True}

def fast(mg):
    """FAST, with mg.redesign.whole_segment overriding its whole_segment (1 card: no CP partner,
    so tries longer than the window split into windows, earlier tokens their no-grad prefix, as before 26 Oct)."""
    red = (mg or {}).get("redesign")
    ws = red.get("whole_segment") if isinstance(red, dict) else None
    return {**FAST, **({"whole_segment": bool(ws)} if ws is not None else {})}


# stage D (mg.draft_weight > 0): the draft's per-step metrics in the step rows (trainer result keys)
DRAFT_ROW = ("draft_kl", "draft_top1", "draft_agree", "draft_paccept", "draft_rows", "draft_grad_norm",
             "draft_update_norm", "draft_skipped")
def check_mg(mg):
    bad = sorted(set(mg) - MG_KEYS)
    if mg.get("redesign"):  # arena.rl_seq's own keys (and its backlog's)
        from . import rl_seq
        rl_seq.settings(mg)
        from . import rl_backlog
        rl_backlog.settings(mg)
    if mg.get("placement") not in (None, "first", "balanced"):
        bad.append("placement: first | balanced")
    if int(mg.get("batch_tokens") or 0) > 0:  # MFU-1 m1g1 (23 Oct): batch_pass gradients disagree with per-window
        bad.append("batch_tokens > 0: batched tails are numerically wrong (grad cos 0.52-0.86 vs per-window), refused until fixed")
    if bad:
        raise ValueError(f"mg keys the loop does not read: {bad}")


def settings(raw, check=True):
    """The study's rl_loop over DEFAULTS, validated. ``check=False``: the merge only (modal_rl1.trainer_key runs it in
    the image's trainer python, which lacks the play stack that mg.redesign's validators import: arcengine)."""
    merged = copy.deepcopy(DEFAULTS)
    for key, value in (raw or {}).items():
        if key not in DEFAULTS:
            raise ValueError(f"Unknown rl_loop setting {key!r}")
        merged[key] = {**merged[key], **value} if isinstance(DEFAULTS[key], dict) else value
    if not check:
        return merged
    if os.environ.get("ARC_RL1_RESUME_ADAPTER") or os.environ.get("ARC_RL1_EVAL_ADAPTERS"):
        raise ValueError("rl_loop: adapter reads (mode final, mg.eval) are gone; ARC_RL1_RESUME_ADAPTER and "
                         "ARC_RL1_EVAL_ADAPTERS are refused")
    if merged["mg"]:
        check_mg(merged["mg"])
    mg = merged["mg"] or {}
    if merged["mode"] != "loop" or not mg.get("redesign"):
        raise ValueError("rl_loop: mode loop with mg.redesign (arena.rl_seq) is the only recipe")
    if mg.get("reward_shape") != "logeq" or (mg.get("turn_credit") or {}).get("mode") != "gigpo":
        raise ValueError("rl_loop: mg.reward_shape logeq with mg.turn_credit mode gigpo is the only reward (rl-min)")
    return merged


# ---- rewards and statistics (pure) ----

# a policy outcome counts at its earned level score (0 when nothing cleared), an infra outcome
# is dropped and logged; any other outcome is treated as infra and named in the drop log.
POLICY_OUTCOMES = ("cleared", "won", "budget", "level_cap", "action_cap", "game_over", "finished",
                   "wall",       # the attempt's own wall (attempt_wall_seconds): slow or looping play
                   "lead_out")   # the lead-in token budget spent without getting through


def outcome_kind(row):
    return "policy" if row.get("outcome") in POLICY_OUTCOMES else "infra"


def usable(row):
    return outcome_kind(row) == "policy"


def drops(rows):
    """The infra drops of one bench: share kept, and each dropped attempt with its outcome."""
    dropped = [{"label": r.get("label"), "game": r.get("game"), "outcome": r.get("outcome")} for r in rows if not usable(r)]
    return {"attempts": len(rows), "usable": len(rows) - len(dropped),
            "usable_share": round((len(rows) - len(dropped)) / len(rows), 4) if rows else None, "dropped": dropped}


class Feed:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.status = {}
        self.started = time.time()

    def emit(self, kind, **data):
        self.last = time.time()
        row = {"t": round(time.time() - self.started, 1), "epoch": round(time.time(), 1), "kind": kind, **data}
        with (self.folder / "feed.jsonl").open("a") as file:
            file.write(json.dumps(row, default=str) + "\n")
        self.status[kind] = row
        tmp = self.folder / "status.json.tmp"
        tmp.write_text(json.dumps(self.status, indent=1, default=str))
        tmp.replace(self.folder / "status.json")
        print(json.dumps({"event": f"rl_{kind}", **{k: v for k, v in row.items() if k not in ("windows", "probes")}},
                         default=str)[:4000], flush=True)
        return row


def coroutine_chain(coro):
    """The await chain of a suspended coroutine: one "file:line function" per frame, then what the innermost awaits."""
    out, seen = [], 0
    while coro is not None and seen < 60:
        seen += 1
        frame = getattr(coro, "cr_frame", None) or getattr(coro, "gi_frame", None)
        if frame is None and not hasattr(coro, "cr_await") and not hasattr(coro, "gi_yieldfrom"):
            out.append(f"awaits {coro!r}"[:400])
            break
        if frame is not None:
            out.append(f"{frame.f_code.co_filename}:{frame.f_lineno} {frame.f_code.co_name}")
        coro = getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None)
    return out


class StallWatch:
    """Stall watchdog (preflight a's 35-min hang): no feed row and no finished model request for
    ``seconds`` -> one dump of every thread and asyncio task of the loop process (await chains), the shared attempt
    pool and gate, the process table and py-spy of every python process when py-spy is installed, written to
    ARC_VALIDATION_OUTPUT; ``kill``: then end the run. Env ARC_RL1_STALL_DUMP=seconds, ARC_RL1_STALL_KILL=1."""

    def __init__(self, feed, loop, seconds, kill, out):
        self.feed, self.loop, self.seconds, self.kill, self.out = feed, loop, float(seconds), bool(kill), Path(out)
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            time.sleep(min(15.0, self.seconds / 4))
            from arc_infra.fleet import ACTIVITY  # replies still arriving: play in progress, not a stall

            silent = time.time() - max(getattr(self.feed, "last", self.feed.started), ACTIVITY["last_reply"])
            if silent >= self.seconds:
                path = self.dump(silent)
                self.feed.emit("stall_dump", path=str(path), silent_seconds=round(silent), kill=self.kill)
                if self.kill:
                    os._exit(3)

    def dump(self, silent):
        import faulthandler

        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / f"stall-dump-{int(time.time())}.txt"
        with open(path, "w") as file:
            from arc_infra.fleet import ACTIVITY

            file.write(f"no feed row or reply for {silent:.0f} s; replies so far {ACTIVITY['replies']}\n\n== threads (faulthandler)\n")
            file.flush()
            faulthandler.dump_traceback(file=file, all_threads=True)
            file.write("\n== asyncio tasks\n")
            tasks = []
            for _ in range(5):
                try:
                    tasks = list(asyncio.all_tasks(self.loop))
                    break
                except RuntimeError:  # the set changed while copying
                    continue
            for task in tasks:
                file.write(f"\n-- {task!r}"[:600] + "\n")
                for line in coroutine_chain(task.get_coro()):
                    file.write(f"   {line}\n")
            try:
                from . import level_bench

                for url, pool in (level_bench.SHARED_SLOTS_BY_URL or {}).items():
                    file.write(f"\n== shared slots {url}: used {pool.used}/{pool.slots} live {len(pool.live)} pending {pool.pending} "
                               f"capacity {pool.capacity} load {pool.load() if pool.capacity else None} kv_waits {pool.kv_waits}\n")
                    if pool.gate is not None:
                        g = pool.gate
                        file.write(f"   gate: capacity {g.capacity} reserved {g.reserved} inflight {g.inflight} queued {len(g.queue)}\n")
            except Exception as exc:  # a diagnostic only
                file.write(f"\n(shared slots: {exc!r})\n")
            table = subprocess.run(["ps", "-eo", "pid,ppid,etime,stat,pcpu,rss,args"], capture_output=True, text=True).stdout
            file.write("\n== processes\n" + table)
            spy = shutil.which("py-spy")
            for line in table.splitlines()[1:]:
                parts = line.split(None, 6)
                if spy and len(parts) == 7 and "python" in parts[6]:
                    try:
                        got = subprocess.run([spy, "dump", "--pid", parts[0]], capture_output=True, text=True, timeout=60)
                        file.write(f"\n== py-spy {parts[0]}: {parts[6][:200]}\n{got.stdout}{got.stderr}")
                    except Exception as exc:
                        file.write(f"\n== py-spy {parts[0]}: {exc!r}\n")
            if not spy:
                file.write("\n(py-spy not installed)\n")
        return path


class GpuSampler:
    """nvidia-smi every 2 s: utilisation and memory per phase."""

    def __init__(self):
        self.rows, self.phase, self.stop = [], "start", threading.Event()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while not self.stop.wait(2.0):
            try:
                out = subprocess.run([os.environ.get("ARC_REAL_NVIDIA_SMI", "nvidia-smi"), "--query-gpu=utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                cards = [[float(v) for v in line.split(",")] for line in out.splitlines() if line.strip()]
                second = cards[1] if len(cards) > 1 else (None, None)  # two-card loop: the trainer's card
                self.rows.append((time.time(), self.phase, cards[0][0], cards[0][1] / 1024, second[0],
                                  None if second[1] is None else second[1] / 1024))
            except Exception:  # a missing sample is not an error
                pass

    def mark(self, phase):
        self.phase = phase

    def summary(self, since):
        rows = [r for r in self.rows if r[0] >= since]
        out = {}
        for phase in sorted({r[1] for r in rows}):
            part = [r for r in rows if r[1] == phase]
            out[phase] = self.stats(part)
        if rows:
            out["all"] = self.stats(rows)
        return out

    @staticmethod
    def stats(part):
        row = {"seconds": round(len(part) * 2.0), "util_mean": round(sum(r[2] for r in part) / len(part), 1),
               "idle_share": round(sum(r[2] < 10 for r in part) / len(part), 3),
               "memory_peak_gib": round(max(r[3] for r in part), 2)}
        second = [r for r in part if r[4] is not None]
        if second:
            row["card1"] = {"util_mean": round(sum(r[4] for r in second) / len(second), 1),
                            "idle_share": round(sum(r[4] < 10 for r in second) / len(second), 3),
                            "memory_peak_gib": round(max(r[5] for r in second), 2)}
        return row


class Trainer:
    def __init__(self, folder, command, env, log_path):
        self.folder = Path(folder)
        (self.folder / "cmd").mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.log = open(log_path, "a")
        from arc_infra.heap_census import install_for

        install_for(command, env)  # HEAP_CENSUS_DIR in trainer_env: the trainer ranks' census (28 Oct: 111 -> 240 GiB)
        self.process = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                        start_new_session=True)

    @classmethod
    def adopt(cls, folder, pid):
        """A trainer the launcher pre-spawned (arena.trainer_prespawn); its output goes to its own log."""
        from .trainer_prespawn import Adopted

        self = cls.__new__(cls)
        self.folder, self.n, self.log = Path(folder), 0, None
        (self.folder / "cmd").mkdir(parents=True, exist_ok=True)
        self.process = Adopted(pid)
        return self

    async def ready(self, timeout):
        end = time.time() + timeout
        alive = self.folder / "alive.json"
        while time.time() < end:
            if self.process.poll() is not None:
                raise RuntimeError(f"trainer exited during load ({self.process.returncode})")
            if alive.exists():
                try:
                    state = json.loads(alive.read_text())
                    if state.get("state") in ("ready", "idle"):
                        return state
                except ValueError:
                    pass
            await asyncio.sleep(5)
        raise TimeoutError("trainer did not load in time")

    async def call(self, op, timeout=3600, **payload):
        index, self.n = self.n, self.n + 1
        tmp = self.folder / "cmd" / f"{index:04d}.json.tmp"
        # ``always``: trainer paths every op runs under (FAST / live knobs: MFU stages on every op, so
        # probes, old logprobs and updates share one numerics)
        tmp.write_text(json.dumps({"op": op, **(getattr(self, "always", None) or {}), **payload}))
        tmp.rename(self.folder / "cmd" / f"{index:04d}.json")
        done = self.folder / "done" / f"{index:04d}.json"
        end = time.time() + timeout
        while time.time() < end:
            if done.exists():
                return json.loads(done.read_text())
            if self.process.poll() is not None:
                raise RuntimeError(f"trainer exited during {op} ({self.process.returncode})")
            await asyncio.sleep(0.02)  # D2 27 Oct: 0.25 s polls were part of a 1-5 s gap between update calls
        raise TimeoutError(f"trainer {op} took over {timeout} s")

    async def must(self, op, timeout=3600, **payload):
        """``call`` that raises with the trainer's own error when the operation failed."""
        result = await self.call(op, timeout=timeout, **payload)
        if not result.get("ok", True):
            raise RuntimeError(f"trainer {op}: {result.get('error')}")
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()


class Server:
    mask_expect = None  # server.expert_mask (hero run): the worker mask's masked entries, checked after start / wake
    mask_sha = None  # and its sha256 (the bundle's server.expert_mask.sha256)

    def __init__(self, base_url, model):
        self.root = base_url.rstrip("/")
        self.root = self.root[:-3] if self.root.endswith("/v1") else self.root
        self.model, self.loaded = model, None

    async def mask_check(self, why):
        """The worker's expert mask is live (arc_infra.expert_mask_runtime status), or raise."""
        if not self.mask_expect:
            return None
        reply = await self.post("/collective_rpc", {"method": "arc_expert_mask", "timeout": 60,
                                                    "args": [json.dumps({"op": "status"})]})
        got = (reply.get("results") or [None])[0] if isinstance(reply, dict) else None
        if (not isinstance(got, dict) or not got.get("ok") or int(got.get("masked_entries", -1)) != int(self.mask_expect)
                or (self.mask_sha and got.get("sha256") != self.mask_sha)):
            raise RuntimeError(f"expert mask not live after {why}: {str(got)[:300]}")
        return got

    async def post(self, path, payload=None, timeout=900):
        import httpx  # here: the trainer test imports this module where httpx is absent

        async with httpx.AsyncClient(trust_env=False, timeout=timeout) as client:
            response = await client.post(self.root + path, json=payload if payload is not None else {})
            if response.status_code >= 400:
                raise RuntimeError(f"{path}: HTTP {response.status_code} {response.text[:800]}")
            try:
                return response.json()
            except ValueError:
                return response.text

    async def sleep(self):
        began = time.time()
        if not getattr(self, "asleep", False):
            await self.post("/sleep?level=1")
            self.asleep = True
        return round(time.time() - began, 1)

    async def wake(self):
        began = time.time()
        await self.post("/wake_up")
        self.asleep = False
        await self.mask_check("wake")
        return round(time.time() - began, 1)

    def recycle_pid(self):
        """The server's root pid: our own child (a recycled or second server), else server 1's identity file."""
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            return int(process.pid)
        return int(json.loads(Path(self.recycle_identity).read_text())["pid"])

    async def recycle(self):
        """mg.redesign.recycle (arena.server_recycle): fresh processes, awake, no adapter loaded."""
        from .server_recycle import recycle

        row = await recycle(self)
        self.asleep, self.loaded = False, None
        self.__dict__["names"], self.__dict__["paths"] = set(), {}
        await self.mask_check("recycle")
        return row

    async def load(self, name, path):
        adapter_tensors(path)
        draft_files(path)
        began = time.time()
        static = getattr(self, "static_lora", None)
        if static is not None:  # served from boot (server.lora.static, a zero adapter): nothing to load at runtime
            if name != static or not (Path(path) / "zero-adapter.json").is_file():
                raise RuntimeError(f"static adapter {static!r} is served from boot; refusing to load {name!r} "
                                   f"from {path} (a trained adapter needs runtime LoRA loading)")
            self.loaded = name
            return 0.0
        names = self.__dict__.setdefault("names", set())
        if self.loaded == name or name in names:
            await self.post("/v1/unload_lora_adapter", {"lora_name": name})
        await self.post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path), "is_3d_lora_weight": True})
        self.loaded = name
        names.add(name)
        return round(time.time() - began, 1)

    async def unload(self, name):
        await self.post("/v1/unload_lora_adapter", {"lora_name": name})
        self.__dict__.setdefault("names", set()).discard(name)
        if self.loaded == name:
            self.loaded = None

    async def tokens(self, text):
        return (await self.post("/tokenize", {"model": self.model, "prompt": text}))["tokens"]

    async def prompt_logprobs(self, model, ids):
        reply = await self.post("/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1,
                                                     "temperature": 0.0, "prompt_logprobs": 0})
        rows = reply["choices"][0]["prompt_logprobs"]
        return [float(rows[i][str(ids[i])]["logprob"]) for i in range(1, len(ids))]


def card_memory_of(index):
    """(used GiB, total GiB) of one card by its PCI-order index (nvidia-smi's own order)."""
    try:
        smi = os.environ.get("ARC_REAL_NVIDIA_SMI") or "nvidia-smi"
        out = subprocess.run([smi, "-i", str(index), "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        used, total = (float(v) / 1024 for v in out.splitlines()[0].split(","))
        return used, total
    except Exception:
        return None, None


def second_server_argv(argv, port, gpu_util=None, kv_scale=None):
    """The live play server's argv for a second server: another port, and a smaller memory share (it wakes beside the
    parked trainer's residue)."""
    out = list(argv)
    def put(flag, value):
        if flag in out:
            out[out.index(flag) + 1] = str(value)
            return True
        return False
    if not put("--port", port):
        out += ["--port", str(port)]
    if gpu_util is not None:
        put("--gpu-memory-utilization", gpu_util)
    if kv_scale is not None and "--kv-cache-memory-bytes" in out:
        i = out.index("--kv-cache-memory-bytes") + 1
        out[i] = str(int(int(out[i]) * float(kv_scale)))
    return out


def card_environment(base, overrides, card="?"):
    """Server 1's environment with this card's overrides (None removes a key). Per-card A/B (RL 26 Oct): a flag
    turned on here must already be on in server 1 (its setup installs the patched files; a process without the flag
    reads them as stock), or carried by ARC_SERVER2_ENV (serving_setup.SERVER2_PATCH_FLAGS). When the ARC_* flags
    differ from server 1's, VLLM_CACHE_ROOT moves to this flag set's own root (boot_cache.vllm_cache_root): server 1's
    AOT graph was traced under its flags (moe_lora_bm16 reads its flag inside traced code), so it is not loaded here."""
    env = dict(base)
    for key, value in (overrides or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    try:
        carried = json.loads(base.get("ARC_SERVER2_ENV") or "{}")
    except ValueError:
        carried = {}
    new = sorted(k for k, v in (overrides or {}).items() if v == "1" and k.startswith("ARC_") and base.get(k) != "1"
                 and carried.get(k) != "1")
    if new:
        raise ValueError(f"card {card} turns on {new}, which server 1's setup did not install: set them on card 0 "
                         "(card_env '0' or the trial's server_env) and null on the cards without them")
    from arc_infra.boot_cache import env_fingerprint, vllm_cache_root

    traced = lambda e: env_fingerprint({k: v for k, v in e.items() if k not in carried})  # per-process flags aside
    if traced(env) != traced(base) and base.get("XDG_CACHE_HOME"):
        env["VLLM_CACHE_ROOT"] = str(vllm_cache_root(base["XDG_CACHE_HOME"], env))
    return env


def server_env_record(env):
    """The serving-relevant part of a server's effective environment (ARC_*, VLLM_*, ARENA_*, CUDA_VISIBLE_DEVICES)."""
    return {k: v for k, v in sorted(env.items())
            if k.startswith(("ARC_", "VLLM_", "ARENA_")) or k == "CUDA_VISIBLE_DEVICES"}


class SecondServer(Server):
    """Time share: a copy of the play server on the trainer's card. It boots asleep-capable beside an
    empty card before the trainer loads, sleeps, and plays only between the trainer's update and the round's end."""

    def __init__(self, server_dir, model, *, card, port=1235, gpu_util=0.88, kv_scale=None, log=None, env=None,
                 name="server2"):
        self.server_dir, self.card, self.port = Path(server_dir), str(card), int(port)
        self.keep_kv = False  # ARC_SLEEP_KEEP_KV in its environment (boot)
        # redesign.card_env: this server only (an in-run A/B); None removes a key server 1 has
        self.extra_env = {str(k): (None if v is None else str(v)) for k, v in (env or {}).items()}
        self.folder = self.server_dir / name  # its own folder: the first server's cache layout glob stays clean
        self.folder.mkdir(parents=True, exist_ok=True)
        super().__init__(f"http://127.0.0.1:{self.port}/v1", model)
        self.gpu_util, self.kv_scale, self.process, self.log = gpu_util, kv_scale, None, log
        self.asleep, self.memory = False, {}

    def source(self):
        identity = json.loads((self.server_dir / "vllm-server-identity.json").read_text())
        pid = int(identity["pid"])
        argv = [a for a in Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0") if a] or identity["argv"]
        env = dict(x.split("=", 1) for x in Path(f"/proc/{pid}/environ").read_bytes().decode().split("\0") if "=" in x)
        return argv, env

    def command(self):
        argv, env = self.source()
        shim = Path(f"/tmp/gpu{self.card}-bin")  # the serving stack's nvidia-smi calls see this card
        real = env.get("ARC_REAL_NVIDIA_SMI") or os.environ.get("ARC_REAL_NVIDIA_SMI")
        if real:
            shim.mkdir(exist_ok=True)
            (shim / "nvidia-smi").write_text('#!/bin/sh\ncase " $* " in *" -i "*|*" --id"*) exec %s "$@";; esac\n'
                                             'exec %s -i %s "$@"\n' % (real, real, self.card))
            (shim / "nvidia-smi").chmod(0o755)
            env["PATH"] = f"{shim}:" + ":".join(x for x in env.get("PATH", "").split(":") if not x.startswith("/tmp/gpu"))
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=self.card)
        return second_server_argv(argv, self.port, self.gpu_util, self.kv_scale), card_environment(
            env, getattr(self, "extra_env", {}), self.card)

    PID = "server.pid"  # parallel boot: written by start(); boot() waits on a live one

    def start(self):
        """Spawn this card's server: from boot(), or at container start beside server 1 (arena.server_start, the
        parallel boot), from the same code, so the process boot() waits on is the one it would have spawned. None when
        the other caller already claimed the card (start.lock: one spawn per card, whichever comes first)."""
        try:
            (self.folder / "start.lock").open("x").close()
        except FileExistsError:
            return None
        argv, env = self.command()
        from arc_infra.heap_census import install_for

        install_for(argv, env)  # HEAP_CENSUS_DIR in its env: the vLLM processes' heap census (real-v7 growth)
        handle = (self.folder / "vllm.log").open("w")
        self.process = subprocess.Popen(argv, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        handle.close()
        (self.folder / "argv.json").write_text(json.dumps(argv))
        (self.folder / self.PID).write_text(json.dumps({"pid": self.process.pid, "epoch": time.time()}))
        return argv, env

    def started(self):
        """(argv, env, spawn epoch) of a live server this card's start() already spawned (parallel boot), else None."""
        try:
            rec = json.loads((self.folder / self.PID).read_text())
            argv = json.loads((self.folder / "argv.json").read_text())
        except (OSError, ValueError):
            return None
        from .trainer_prespawn import Adopted, alive

        if not alive(int(rec["pid"])):
            return None
        self.process = Adopted(rec["pid"])
        return argv, self.command()[1], float(rec["epoch"])

    async def boot(self, timeout=1800):
        import httpx

        began = time.time()
        early = self.started()
        spawned = None if early else self.start()
        if early is None and spawned is None:  # the boot thread claimed the card and is spawning it now
            for _ in range(240):
                await asyncio.sleep(0.5)
                if (early := self.started()) is not None:
                    break
            else:
                raise RuntimeError(f"{self.folder.name}: claimed by the parallel boot, no live server.pid in 120 s")
        argv, env = early[:2] if early else spawned
        self.keep_kv = env.get("ARC_SLEEP_KEEP_KV") == "1"
        if early:
            self.memory["parallel_boot_s"] = round(began - early[2], 1)  # how long it had been booting already
        self.effective_env = server_env_record(env)  # rl_seq's seq_boot row and server-env.json (identity)
        waited = time.time()
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            while time.time() - began < timeout:
                if self.process.poll() is not None:
                    log = (self.folder / "vllm.log").read_text(errors="replace")
                    out = os.environ.get("ARC_VALIDATION_OUTPUT")  # the whole log beside the run (the feed cuts errors)
                    if out:
                        try:
                            Path(out, "util").mkdir(parents=True, exist_ok=True)
                            Path(out, "util", f"{self.folder.name}-vllm.log").write_text(log)
                        except OSError:
                            pass
                    raise RuntimeError(f"second server exited {self.process.returncode}: {log[-1200:]}")
                try:
                    if (await client.get(self.root + "/v1/models")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                from arc_infra.fleet import ACTIVITY  # a live, booting server is progress for the stall watchdog;
                ACTIVITY["last_reply"] = time.time()  # the boot timeout above bounds it (pf p3ac-v1 pf1r)
                await asyncio.sleep(5)
            else:
                raise RuntimeError("second server not ready in time")
        self.memory["awake_gib"] = card_memory_of(self.card)[0]
        self.memory["boot_seconds"] = round(time.time() - began, 1)
        self.memory["wait_seconds"] = round(time.time() - waited, 1)  # what the loop waited
        await self.mask_check("second server boot")
        return dict(self.memory)

    async def sleep(self):
        began = time.time()
        if not self.asleep:
            await self.post("/sleep?level=1")
            self.asleep = True
        self.memory.setdefault("asleep_gib", card_memory_of(self.card)[0])
        return round(time.time() - began, 1)

    async def wake(self):
        began = time.time()
        if self.asleep:
            await self.post("/wake_up")
            self.asleep = False
            await self.mask_check("wake")
        return round(time.time() - began, 1)

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=60)
            except Exception:
                os.killpg(self.process.pid, signal.SIGKILL)


class StepFailed(RuntimeError):
    pass




def game_score(cleared, total):
    """The competition's game score (arc_agi EnvironmentScoreCalculator, as taaf.game._compute_final_score): every
    level of the game weighted by its 1-based index, a cleared level's score (0-115) else 0, capped at the scored
    levels' weight share x 100. ``cleared``: {level: score}."""
    total = max(int(total or 0), max([int(k) for k in cleared] or [0]))
    if total <= 0:
        return 0.0
    weights = sum(range(1, total + 1))
    got = sum(float(sc) * int(lv) for lv, sc in cleared.items())
    scored = sum(int(lv) for lv, sc in cleared.items() if float(sc) > 0)
    return min(got / weights, 100.0 * scored / weights)


def turn_try(att):
    """One try's per-turn inputs for mg_turn_credit (replies, end step), parsed from its log and
    rollout. Module level: a process pool runs it (pfP2 profile, 22 Oct: the JSON decode of 200 tries' rollouts was
    60% of the ~3-min start-up setup, on the event loop)."""
    from .restore import frame_hash, log_rows

    att = Path(att)
    replies, end = [], None
    try:
        obs = {int(o["step"]): (int(o.get("levels") or 0), frame_hash(o["frames"][-1]))
               for o in log_rows(att / "log.jsonl", ("observation",)) if o.get("frames")}
        # the try ends at its clear (a harvested try runs the clearing reply's last actions and one more request on
        # the next level); no clear: the last observation
        clear = log_rows(att / "log.jsonl", ("level_bench_clear",))
        end = int(clear[0]["step"]) if clear else (max(obs) if obs else None)
        with open(att / "rollout.jsonl") as file:
            for line in file:
                r = json.loads(line)
                if r.get("scripted"):
                    continue
                step = int(r.get("step") or 0)
                anchor = obs.get(step) or obs.get(max([s for s in obs if s <= step] or [0]))
                replies.append((int(r["n"]), int((r.get("usage") or {}).get("completion_tokens") or 0), anchor, step))
    except (OSError, ValueError, KeyError):
        replies = []
    return replies, end


TURN_CACHE, TURN_POOL = {}, [None]


def turn_prefetch(atts):
    """Parse every listed try not yet in TURN_CACHE in a process pool (16 workers): {path: turn_try(...)}."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    todo = [str(a) for a in atts if str(a) not in TURN_CACHE]
    if len(todo) < 8:
        return 0
    if TURN_POOL[0] is None:
        TURN_POOL[0] = ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 4),
                                           mp_context=multiprocessing.get_context("forkserver"))
    if len(TURN_CACHE) > 4096:
        TURN_CACHE.clear()
    for att, got in zip(todo, TURN_POOL[0].map(turn_try, todo, chunksize=4)):
        TURN_CACHE[att] = got
    return len(todo)


def restored(row):
    """A try usable as played from its start: no restore, a clean restore, or a restore whose game state matched the
    record (board, steps, levels) though a prompt message differed (a strict-check drop is not a
    broken start); a replay whose game state diverged is not."""
    rs = row.get("restore")
    if not rs or rs.get("ok"):
        return True
    return bool((rs.get("state") or {}).get("ok")) and row.get("outcome") != "replay_mismatch"


def level_of(row):
    """The level a restored try starts on (its point id carries ``-L<n>-``)."""
    return next((int(part[1:]) for part in str(row.get("point", "")).split("-") if part[:1] == "L" and part[1:].isdigit()),
                None)


def clear_tokens(row):
    """Tokens to clear the level for a cleared try, else None."""
    level_rows = row.get("level_rows") or []
    if row.get("outcome") not in ("cleared", "won") or not level_rows:
        return None
    return int(level_rows[0].get("tokens") or row.get("completion_tokens") or 0) or None


def clear_rhae(row):
    """The cleared level's score as a fraction (RHAE, min(1.15, (baseline / actions)^2)), else None."""
    level_rows = row.get("level_rows") or []
    if row.get("outcome") not in ("cleared", "won") or not level_rows or level_rows[0].get("score") is None:
        return None
    return float(level_rows[0]["score"]) / 100.0


def held_metrics(rows):
    toks = [clear_tokens(r) for r in rows]
    clears = [t for t in toks if t]
    rhae = [v for v in (clear_rhae(r) for r in rows) if v is not None]
    actions = [int(r["level_rows"][0].get("actions") or 0) for r in rows if clear_tokens(r)]
    spent = sum(int(r.get("completion_tokens") or 0) for r in rows)
    se = None
    if spent and len(rows) > 1:  # v6x: the ratio estimate's standard error (delta method over tries)
        ratio = sum(rhae) / spent
        dev = [(clear_rhae(r) or 0.0) - ratio * int(r.get("completion_tokens") or 0) for r in rows]
        se = round(math.sqrt(len(rows) / (len(rows) - 1) * sum(d * d for d in dev)) / spent * 1e4, 4)
    return {"tries": len(rows), "clears": len(clears), "clear_rate": round(len(clears) / len(rows), 3) if rows else None,
            "median_to_clear": statistics.median(clears) if clears else None,
            "median_actions": statistics.median(actions) if actions else None,
            "mean_rhae": round(statistics.mean(rhae), 4) if rhae else None,
            "rhae_per_10k": round(sum(rhae) / spent * 1e4, 4) if spent else None, "rhae_10k_se": se,
            "tokens_per_try": round(spent / len(rows), 1) if rows else None}


# ---- the loop ----

class Loop:
    def __init__(self, study, trial, *, source, games_dir, output, server_dir, analysis_python, base_url,
                 deadline_epoch):
        self.study, self.trial, self.cfg = study, trial, settings(study.get("rl_loop"))
        self.source, self.games_dir, self.output = Path(source), games_dir, Path(output)
        self.server_dir, self.analysis_python, self.base_url = server_dir, analysis_python, base_url
        self.deadline = float(deadline_epoch)
        self.feed = Feed(self.output / "rl")
        self.gpu = GpuSampler()
        self.server = Server(base_url, study.get("served_model") or "Qwen/Qwen3.8-Flash-Next-NVFP4")
        self.work = Path("/tmp/rl1")
        self.adapters = self.work / "adapters"
        self.adapters.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.trainer = None

    async def bench(self, name, games, passes, budgets, *, points, model, seed_base, concurrency, reserve, own,
                    base_url=None):
        """One level-bench run on the current adapter: ``games`` x ``passes`` tries of the start-point file ``points``.
        ``own``: study keys of this bench (``policy_version``)."""
        from .level_bench import run_level_bench

        study = copy.deepcopy(self.study)
        study["served_model"] = model
        for key, value in (own or {}).items():
            study[key] = value
        if (self.cfg.get("mg") or {}).get("gate_overcommit"):
            study["gate_overcommit"] = float(self.cfg["mg"]["gate_overcommit"])
        if (self.cfg.get("mg") or {}).get("transport_backoff"):
            study["transport_backoff"] = list(self.cfg["mg"]["transport_backoff"])
        bench = dict(study["level_bench"])
        bench.update(start_points=str(points), games=list(games), passes=int(passes), seed_base=seed_base,
                     concurrency=int(concurrency), token_budget_by_point={f"{g}-game": int(budgets[g]) for g in games},
                     arms=None)
        study["level_bench"] = bench
        folder = self.output / "rl" / name
        self.gpu.mark("play")
        began = time.time()
        report = await run_level_bench(study, {**self.trial, "name": name}, source=self.source,
                                       games_dir=self.games_dir, output=folder, server_dir=self.server_dir,
                                       analysis_python=self.analysis_python, base_url=base_url or self.base_url,
                                       deadline_epoch=self.deadline - reserve)
        engine = report.get("engine") or {}
        rows = [r for r in report["attempts"] if usable(r)]
        self.log_drops(name, report["attempts"])
        tokens = sum(int(r.get("completion_tokens") or 0) for r in report["attempts"])
        play = {"seconds": round(time.time() - began, 1), "attempts": len(report["attempts"]), "usable": len(rows),
                "tokens": tokens, "tok_s": round(tokens / max(1.0, time.time() - began), 1),
                "engine_tok_s": engine.get("tokens_per_second"), "preemptions": engine.get("preemptions")}
        return folder, rows, play

    def log_drops(self, name, attempts):
        info = drops(attempts)
        if info["dropped"]:
            self.feed.emit("drops", bench=name, **info)
        return info

    def game_rows(self, rows, game):
        return [r for r in rows if str(r["game"]).startswith(game)]


    def windows(self, folder, rows, weights=None, prefix=""):
        weights = weights or {}
        served = {"served_route": self.served_dir} if getattr(self, "served_dir", None) else {}
        return [{"rollout": str(Path(folder) / "attempts" / r["label"] / "rollout.jsonl"),
                 "w": float(weights.get(r["label"], 1.0)), "name": prefix + r["label"], "label": r["label"], **served}
                for r in rows]


    def compact_always(self):
        """hero-v6b (24 Oct): a compacted trainer (RL1_EXPERT_COMPACT) refuses any forward off the fused expert path, so
        every op from its start (base reference, park check, probes) runs under FAST, not only those after the loop
        first applies it."""
        if (self.cfg.get("trainer_env") or {}).get("RL1_EXPERT_COMPACT") == "1":
            self.trainer.always = {**(getattr(self.trainer, "always", None) or {}), **fast(self.cfg["mg"])}

    async def start_trainer(self, gate=None):
        from . import trainer_prespawn

        # mg.trainer_overlap (gate): the first CUDA use waits for the gate file
        command, env = trainer_prespawn.command_for(self.cfg, self.source, self.work, gate)
        # start-up (b): the launcher's pre-spawned trainer when it is this exact process, else a fresh one
        pid, pre = await asyncio.to_thread(trainer_prespawn.take, self.work / "trainer", command, env,
                                         self.output / "rl" / "trainer.log")
        if pid is not None:
            self.trainer = Trainer.adopt(self.work / "trainer", pid)
            log = self.output / "rl" / "trainer.log"  # readers (eff watcher, park check) keep the loop's path
            if not log.exists() and not pre["log_equal"]:
                log.parent.mkdir(parents=True, exist_ok=True)
                log.symlink_to(pre["log"])
        else:
            self.trainer = Trainer(self.work / "trainer", command, env, self.output / "rl" / "trainer.log")
        began = time.time()
        self.gpu.mark("trainer_load")
        state = await self.trainer.ready(self.cfg["trainer_timeout"])
        self.expert_mask_trainer(state.get("loaded"))
        self.compact_always()
        out = {"seconds": round(time.time() - began, 1), "loaded": state.get("loaded")}
        if pre is not None:
            out["prespawn"] = pre
        return out


    async def expert_mask_start(self):
        """server.expert_mask (hero run): every server's mask is checked live after its start and each wake."""
        spec = (self.study.get("server") or {}).get("expert_mask")
        if not spec or os.environ.get("ARC_PREFLIGHT_STUB"):  # the replay stub serves no MoE
            return None
        mask = json.loads((self.source / spec["mask"]).read_text())
        Server.mask_expect = sum(len(v) for v in mask["pruned"].values())
        Server.mask_sha = spec["sha256"]
        got = await self.server.mask_check("start")
        self.feed.emit("expert_mask", label=got.get("label"), masked=got.get("masked_entries"), sha256=got.get("sha256"),
                       layers=len(got.get("layers") or []),
                       **{k: got[k] for k in ("skipped_reads", "skipped_expected", "load_profile") if k in got})
        return got

    def expert_mask_trainer(self, loaded):
        """server.expert_mask (hero run): the trainer applied the bundle's mask (RL1_EXPERT_MASK, by sha256), or
        raise; hprobe-v4: a bundle without the trainer's mask code ignored RL1_EXPERT_MASK silently."""
        spec = ((getattr(self, "study", None) or {}).get("server") or {}).get("expert_mask")
        if not spec or os.environ.get("ARC_PREFLIGHT_FAKE_TRAINER"):  # the CPU preflight's stand-in loads no model
            return None
        got = (loaded or {}).get("expert_mask") or {}
        self.feed.emit("expert_mask_trainer", sha256=got.get("sha256"), layers=got.get("layers"), pruned=got.get("pruned"))
        if got.get("sha256") != spec["sha256"]:
            raise RuntimeError(f"trainer expert mask {got.get('sha256')} != the bundle's {spec['sha256']}")
        return got

    async def run(self):
        report = {"status": "error", "settings": self.cfg}
        if os.environ.get("ARC_RL1_STALL_DUMP"):
            StallWatch(self.feed, asyncio.get_running_loop(), os.environ["ARC_RL1_STALL_DUMP"],
                       os.environ.get("ARC_RL1_STALL_KILL"), os.environ.get("ARC_VALIDATION_OUTPUT") or self.output)
        try:
            self.feed.emit("start", mode=self.cfg["mode"], settings=self.cfg)
            host = Path("/tmp/rl1-host-check.json")  # modal_rl1.host_check (boot pinned-copy speed; slow < 0.5 GiB/s)
            if host.is_file():
                try:
                    self.feed.emit("host_check", **json.loads(host.read_text()))
                except ValueError:
                    pass
            await self.expert_mask_start()
            report["stop"] = await self.mg_loop()
            report["status"] = "complete"
        except Exception as error:  # the failure is recorded in the feed and the trial
            import traceback

            report.update(status="error", error=repr(error)[:2000])
            self.feed.emit("error", error=repr(error)[:2000], traceback=traceback.format_exc()[-4000:])
        finally:
            report["gpu"] = self.gpu.summary(self.started)
            self.gpu.stop.set()
            if self.trainer is not None:
                try:
                    await asyncio.wait_for(self.trainer.call("exit", timeout=120), 150)
                except Exception:
                    pass
                self.trainer.close()
            try:
                await self.server.wake()
            except Exception:
                pass
            (self.output / "rl" / "report.json").write_text(json.dumps(report, indent=1, default=str))
        return report


    def v7_found(self, path):
        """A path the earlier leg recorded, as this container sees it: its trial outputs (/tmp/arena/...) under the
        resumed root's arena/, and a /runs/<rel> dir unpacked from a preflight pack under /tmp/<rel>."""
        p = Path(path)
        if p.exists():
            return str(p)
        root = getattr(self, "v7_resume_root", None)
        tries = []
        if root is not None and str(p).startswith("/tmp/arena/"):
            tries.append(root / "arena" / p.relative_to("/tmp/arena"))
        if str(p).startswith("/runs/"):
            tries.append(Path("/tmp") / p.relative_to("/runs"))
            # per-run volumes (28 Oct): an earlier run's leg, mounted read-only at /in/<vol>/
            tries += [v / p.relative_to("/runs") for v in sorted(Path("/in").glob("*"))]
        if root is not None and "state" in p.parts:  # the leg's own state/sNNNN beside its v7_state.json
            tries.append(root / Path(*p.parts[p.parts.index("state"):]))
        return str(next((t for t in tries if t.exists()), p))


    def v7_specs(self, kept):
        """The trainer specs of one group's selected tries."""
        mg, specs = self.cfg["mg"], []
        level_norm = mg.get("level_norm")
        for folder, row, w, share, *rest in kept:
            spec = self.windows(folder, [row], prefix="t-")[0]
            name = rest[0] if rest else None
            extra = {"set": name}
            if level_norm and name:  # the redesign runs no screen: the set's cap
                extra["norm_cap"] = float(self.cfg["mg_sets"][name].get("cap") or mg["cap"])
            specs.append({**spec, "w": float(w), "whole": True, "turn_key": (str(Path(folder)), row["label"]),
                          "window": int(self.cfg["lvl_window"]), "floor": 1024, "margin": 1024, "credited_only": True,
                          "share": share, **extra})
        return specs

    def v7_trainer_args(self):
        mg = self.cfg["mg"]
        return {"signed": True, "share_prefix": True, "prefix_chunk": int(self.cfg["path_chunk"]),
                "norm": self.cfg["l2_norm"], "norm_cap": int(mg["cap"]), "batch_tokens": int(mg.get("batch_tokens") or 0),
                "batch_max": int(mg.get("batch_max") or 8), "max_windows": 100000, "clip": self.cfg["clip"],
                **fast(mg), **({"ckpt_budget_gib": self.ckpt_budget_gib} if getattr(self, "ckpt_budget_gib", None) else {})}


    def v7_prepare(self, g):
        """A group's trained specs and row head (selection, specs, per-group balance; the balance spends the replies'
        turn credit). Cached per gid, so a group prepared early (``ahead``) is accumulated with the same specs."""
        mg, v = self.cfg["mg"], self.v7
        cache = v.setdefault("prepared", {})
        if g.gid in cache:
            return cache.pop(g.gid)
        rows = v["rows"].pop(g.gid, None)
        if rows is None:  # a resumed group: its rows from its folder
            rows = self.mg_rows(g.done["folder"])
        kept, info = self.mg_select(Path(g.done["folder"]), rows, g.unit)
        kept = [(*k, g.unit) if len(k) == 4 else k for k in kept]
        specs = self.v7_specs(kept)
        row = {"gid": g.gid, "set": g.unit, "version": g.version, "tries": len(rows),
               "trained": len(specs), **{k: info.get(k) for k in ("reward_mean", "reward_std", "zero_spread")}}
        acts = [int(r["level_rows"][0].get("actions") or 0) for r in rows if clear_tokens(r)]
        if acts and rows:  # actions per clear vs the level's base (feed)
            ref = self.mg_level_ref(g.unit, str(rows[0].get("point", ""))[:4], int(level_of(rows[0]) or 0))
            row["clear_actions"], row["base_actions"] = statistics.median(acts), ref.get("actions")
        if specs and mg.get("turn_credit") and mg.get("balance"):  # per group (v7): push-down = the group's push-up mass
            turn = getattr(self, "mg_turn", {})
            row["balance"] = self.mg_reply_balance(specs, turn)
            for sp in specs:
                turn.pop(tuple(sp["turn_key"]), None)
        return row, specs


    def mg_point(self, game):
        from .level_bench import load_points

        return load_points(self.source / self.study["level_bench"]["start_points"],
                           {**self.study["level_bench"], "games": [game], "split": "all"})[0]


    @staticmethod
    def mg_attempt_start(spec, attempt, scan=None):
        """The request of ``spec``'s level start in one recorded attempt, or None (never reached, after an unsafe
        step: summary, handoff, interrupt, forced drain; or prompt above max_prompt). ``scan``: the attempt's
        restore.attempt_scan, read once for every level."""
        from .restore import level_starts, unsafe_from

        if scan is not None:
            first, prompts, starts = scan
        else:
            first, _ = unsafe_from(attempt)
            with open(Path(attempt) / "rollout.jsonl") as file:
                prompts = [int((json.loads(line).get("usage") or {}).get("prompt_tokens") or 0) for line in file]
            starts = level_starts(attempt)
        for request, done, step in starts:
            if (done + 1 == int(spec["level"]) and (first is None or step < first)
                    and prompts[int(request)] <= int(spec.get("max_prompt") or 10**9)):
                return int(request)
        return None


    @staticmethod
    def mg_tag(spec, attempt, request):
        import hashlib

        path = str(attempt)
        if path.startswith("/tmp/validation/"):  # a packed preflight leg's unpacked copy: the volume path's tag, so a
            path = "/runs" + path[4:]  # resumed leg's kept tries on its starts match their plan digests (pfB2, 22 Oct)
        elif path.startswith("/in/"):  # per-run volumes (28 Oct): an earlier run's tree read-only at /in/<vol>/ tags as
            path = "/runs/" + path.split("/", 3)[3]  # its /runs path did in that run
        return f"L{spec['level']}-{hashlib.md5(path.encode()).hexdigest()[:6]}-k{int(request):03d}"

    def mg_points(self, spec, starts, name, harvest=False):
        """The points file of one set: a fresh level-1 segment of its game, or its restored starts. ``harvest``: a
        cleared try records the next level's first request (arena.harvest)."""
        from .restore import load

        cap, base = int(spec.get("cap") or self.cfg["mg"]["cap"]), self.mg_point(spec["game"])
        if spec.get("fresh"):
            points = [{**base, "id": f"{base['id']}-L1-{spec['name']}", "through": False, "segment": True, "level": 1,
                       "level_index": 0, "prefix": [], "budget": cap}]
        else:
            points = []
            for attempt, request in starts:
                tag = self.mg_tag(spec, attempt, request)
                points.append({**base, "id": f"{base['id']}-{tag}", "level": load(attempt, request)["level"],
                               "restore": {"attempt": attempt, "request": int(request), "check_only": False, "budget": cap}})
        if harvest:
            points = [{**p, "harvest": True} for p in points]
        path = self.work / f"{name}-points.json"
        path.write_text(json.dumps({"schema": 1, "points": points}))
        return path, points

    async def mg_play(self, name, spec, starts, tries, *, model, seed, harvest=False, base_url=None):
        """``harvest``: training sets whose clears cut the next level's starts. Every try row records its adapter
        version (policy-rNN: NN, else 0)."""
        path, points = self.mg_points(spec, starts, name, harvest=harvest)
        m = re.fullmatch(r"policy-r(\d+)", str(model))
        folder, rows, play = await self.bench(name, [spec["game"]], int(tries),
                                              {spec["game"]: int(spec.get("cap") or self.cfg["mg"]["cap"])},
                                              points=path, model=model, seed_base=seed,
                                              concurrency=len(points) * int(tries), reserve=120, base_url=base_url,
                                              own={"policy_version": int(m.group(1)) if m else 0})
        rows = [r for r in self.game_rows(rows, spec["game"])
                if restored(r) and level_of(r) is not None]
        gated = [r for r in rows if not self.mg_digest_ok(r)]
        if gated:
            rows = [r for r in rows if self.mg_digest_ok(r)]
            self.feed.emit("digest_gate", bench=name, dropped=len(gated), labels=[r.get("label") for r in gated][:20])
        return folder, rows, play

    async def mg_loop_lag(self, every=1.0, alert=5.0):
        """Event-loop lag (failure class 1, 22 Oct: r4's synchronous kept-try copy held all play ~7 min): the overshoot
        of a 1-s sleep. Its max since the last play_progress row and over the round ride on those rows; an overshoot
        over ``alert`` s is a ``loop_lag`` row at once."""
        loop = asyncio.get_event_loop()
        while True:
            began = loop.time()
            await asyncio.sleep(every)
            lag = loop.time() - began - every
            self.loop_lag["tick"], self.loop_lag["round"] = max(self.loop_lag["tick"], lag), max(self.loop_lag["round"], lag)
            self.loop_lag["admit"] = max(self.loop_lag.get("admit", 0.0), lag)  # rl_seq.admit_log's window
            if lag > alert:
                self.feed.emit("loop_lag", seconds=round(lag, 1))


    @staticmethod
    def mg_groups(rows, size=4):
        """Tries grouped by start; a fresh set's tries (one start) in groups of ``size`` by pass."""
        groups = {}
        for r in rows:
            p = int(r.get("pass") or r["label"].rsplit("-p", 1)[-1])
            key = r["point"] if r.get("restore") else f"{r['point']}#g{p // size}"
            groups.setdefault(key, []).append(r)
        return groups

    def mg_select(self, folder, rows, name=None):
        mg = self.cfg["mg"]
        if mg.get("reward_ref") != "human" or mg.get("ref_fallback") != "v0_group" or mg.get("reply_filter"):
            raise ValueError("rl_loop: the reward is mg.reward_ref human with ref_fallback v0_group, no reply_filter (rl-min)")
        if rows and os.environ.get("ARC_TURN_POOL", "1") == "1":  # every try parsed in parallel, once
            turn_prefetch([Path(folder) / "attempts" / r["label"] for r in rows])
        return self.mg_select_rows(folder, rows, name)

    @staticmethod
    def mg_preflight_spread(rewards, info):
        """CPU preflight only (stub spec "spread"): replayed tries repeat their recordings, so a group's rewards tie and
        nothing trains; a tied group gets a fixed rising ladder so the update, save and serve paths run."""
        if len(rewards) < 2 or max(rewards) != min(rewards):
            return rewards
        try:
            on = bool(json.loads(os.environ.get("ARC_PREFLIGHT_STUB") or "{}").get("spread"))
        except ValueError:
            on = False
        if not on:
            return rewards
        info["preflight_spread"] = info.get("preflight_spread", 0) + 1
        top = max(float(max(rewards)), 0.0) + 1.0
        return [top * (1 + 0.1 * i) for i in range(len(rewards))]

    def mg_select_rows(self, folder, rows, name=None):
        """Dynamic sampling and l4's selection: per mixed group (some but not all tries cleared) the per_side largest
        positive and negative advantages of the logeq reward. Returns ([(folder, row, weight, share)], info); each
        picked try's per-reply GiGPO credit goes to self.mg_turn."""
        mg = self.cfg["mg"]
        out, info, side = [], {"groups": 0, "mixed": 0, "trained": 0}, int(self.cfg["l2_per_side"])
        for key, grp in sorted(self.mg_groups(rows, int(mg.get("group_size") or 4)).items()):
            info["groups"] += 1
            self.mg_group_ref = {"name": name, "clears": [(clear_tokens(r), int(r["level_rows"][0].get("actions") or 0))
                                                          for r in grp if clear_tokens(r)
                                                          and int(r["level_rows"][0].get("actions") or 0) > 0]}
            try:
                rewards = self.mg_preflight_spread([self.mg_reward(r, name) for r in grp], info)
            finally:
                self.mg_group_ref = None
            if len(grp) < 2 or max(rewards) == min(rewards):  # all fails, or identical
                info["zero_spread"] = info.get("zero_spread", 0) + 1
                continue
            total = sum(rewards)
            adv = [r - (total - r) / (len(grp) - 1) for r in rewards]
            info["mixed"] += int(any(not clear_tokens(r) for r in grp))
            info.setdefault("_rewards", []).extend(float(x) for x in rewards)
            order = sorted(range(len(grp)), key=lambda i: -adv[i])
            pick = [i for i in order[:side] if adv[i] > 0] + [i for i in order[::-1][:side] if adv[i] < 0]
            out += [(Path(folder), grp[i], adv[i], key) for i in pick]
            extra = self.mg_turn_credit(Path(folder), grp, rewards, name)
            self.mg_turn = getattr(self, "mg_turn", {})
            for i in pick:
                self.mg_turn[(str(Path(folder)), grp[i]["label"])] = extra[i]
        info["trained"] = len(out)
        got = info.pop("_rewards", [])
        if got:  # dashboard: reward spread of the trained groups
            info["reward_mean"] = round(statistics.mean(got), 4)
            info["reward_std"] = round(statistics.pstdev(got), 4)
        toks = sorted(int(r.get("completion_tokens") or 0) for r in rows)
        if toks:
            info["try_tokens_p50"], info["try_tokens_p90"] = toks[len(toks) // 2], toks[min(len(toks) - 1, int(0.9 * len(toks)))]
        return out, info

    def mg_reply_balance(self, specs, turn):
        """mg6: each reply's weight = its try's advantage + its per-turn extra; then push-down mass (|weight| x reply
        tokens / the level's norm constant) is scaled to equal push-up mass at the reply level, so extras and the
        balance agree. A spec keeps its try advantage (scaled when negative) as ``w`` for the trainer's sign filter."""
        up = down = 0.0
        mass, cap = {}, float(self.cfg["mg"]["cap"])
        # GiGPO: A = A_ep + omega x A_step as computed (omega applied in mg_gigpo_steps), no rescale
        ws = [abs(sp["w"]) for sp in specs if sp["w"]]
        ex = [abs(e) for sp in specs for e in (turn.get(tuple(sp.get("turn_key") or ())) or {}).get("extra", {}).values() if e]
        for sp in specs:
            info = turn.get(tuple(sp.get("turn_key") or ())) or {}
            norm = float(sp.get("norm_cap") or cap)
            sp["reply_w"] = {str(n): sp["w"] + info.get("extra", {}).get(n, 0.0) for n in info.get("tokens", {})}
            for n, t in info.get("tokens", {}).items():
                v = sp["reply_w"][str(n)] * t / norm
                up, down = (up + v, down) if v > 0 else (up, down - v)
                mass[sp.get("set")] = mass.get(sp.get("set"), 0.0) + abs(v)
        scale = up / down if down > 0 and up > 0 else 1.0
        flips = 0
        for sp in specs:
            for n, v in sp.get("reply_w", {}).items():
                if v < 0:
                    sp["reply_w"][n] = v * scale
                flips += int((v > 0) != (sp["w"] > 0))
            if sp["w"] < 0:
                sp["w"] *= scale
        total = sum(mass.values()) or 1.0
        return {"up_mass": round(up, 3), "down_mass": round(down, 3), "down_scale": round(scale, 4), "reply_level": True,
                "sign_flips": flips, "w_abs_median": round(statistics.median(ws), 3) if ws else None,
                "extra_abs_median": round(statistics.median(ex), 3) if ex else None,
                "gigpo": self.mg_gigpo_summary(),
                "level_share": {str(k): round(v / total, 3) for k, v in sorted(mass.items(), key=lambda kv: str(kv[0]))}}

    def mg_ref_fallback(self, name):
        """mg.ref_fallback "v0_group": the current group's own clears (mg_select_rows sets mg_group_ref), median tokens
        to clear and actions."""
        g = getattr(self, "mg_group_ref", None) or {}
        got = g.get("clears") or [] if g.get("name") == name else []
        if not got:
            return None
        return {"median_to_clear": statistics.median(t for t, _ in got), "median_actions": statistics.median(a for _, a in got)}


    def mg_reward(self, row, name):
        """The try's logeq reward against the human baseline (mg_human)."""
        return self.mg_human(row, name, clear_tokens(row))

    def mg_human(self, row, name, t):
        """mg.reward_ref "human": RHAE term log RHAE against 1 (the human baseline), RHAE = min(rhae_cap,
        (baseline / actions)^2); token term clip(log(T_g / t), +-log log_clip), T_g the group's median clear tokens.
        A fail scores F, a clear at the group's median actions that used 2 x the level's cap tokens; a clear scores at
        least F. ``mg.log_rhae_min`` (None: off) floors the RHAE term at -log of it. A group without clears: every
        try the fixed -log log_fail (no spread)."""
        mg = self.cfg["mg"]
        g = getattr(self, "mg_group_ref", None) or {}
        if not (g.get("name") == name and g.get("clears")):
            return -math.log(float(mg.get("log_fail", 16.0)))
        lo = math.log(float(mg.get("log_clip", 16.0)))
        level = int(((row.get("level_rows") or [{}])[0]).get("level") or row.get("level") or 0)
        b = self.mg_level_ref(name, str(row.get("point", ""))[:4], level).get("baseline")
        cap_r, floor = float(mg.get("rhae_cap", 1.15)), mg.get("log_rhae_min")

        def rhae_term(a):
            v = math.log(min(cap_r, (b / a) ** 2)) if b and a else 0.0
            return max(v, -math.log(float(floor))) if floor else v

        t_g = statistics.median(x for x, _ in g["clears"])
        a_g = statistics.median(a for _, a in g["clears"])
        token = lambda tok: max(-lo, min(lo, math.log(t_g / tok)))
        if mg.get("fail_rule") == "below_worst":  # a fail scores log(fail_margin) below the group's worst clear
            fail = min(rhae_term(a) + token(x) for x, a in g["clears"]) - math.log(float(mg.get("fail_margin", 2.0)))
        else:
            fail = rhae_term(a_g) + token(2.0 * float(row.get("token_budget") or mg["cap"]))
        actions = int(((row.get("level_rows") or [{}])[0]).get("actions") or 0)
        if not t or actions <= 0:
            return fail
        return max(fail, rhae_term(actions) + token(t))

    def mg_group_floor(self, grp, name):
        """The fail value of a group (GiGPO's return floor): mg_human's F."""
        fails = [r for r in grp if not clear_tokens(r)]
        self.mg_group_ref = {"name": name, "clears": [(clear_tokens(r), int(r["level_rows"][0].get("actions") or 0))
                                                      for r in grp if clear_tokens(r)
                                                      and int(r["level_rows"][0].get("actions") or 0) > 0]}
        try:
            probe = fails[0] if fails else {**grp[0], "level_rows": [], "outcome": "budget"}
            return self.mg_human(probe, name, None)
        finally:
            self.mg_group_ref = None

    def mg_level_ref(self, name, game, level):
        """A set's base reference: human baseline actions (points file), base median tokens and actions (the env
        table's ref, else the group's own clears: mg_ref_fallback). The redesign runs no screen."""
        cache = self.__dict__.setdefault("mg_refs", {})
        m = {}
        tref = (((self.cfg.get("mg_sets") or {}).get(name) or {}).get("ref") or {}) if name else {}
        # order (envs 24 Oct, mg_pool.apply_refs): measured table ref > derived table ref > the fallback
        # per field (envs 25 Oct, mg-refs-v3): a derived ref with actions null keeps its tokens and takes the fallback's
        # actions (ka59-L4: derived 35.4 actions vs real 220.5 saturated the action term on every try)
        part = bool(tref.get("derived") and tref.get("actions") is None)
        fb = self.mg_ref_fallback(name) if name and (part or not (tref.get("measured") or tref.get("derived"))) \
            and self.cfg["mg"].get("ref_fallback") else None
        if fb and part:
            fb = {k: v for k, v in fb.items() if k in ("median_actions", "clears")}
        if fb:  # mg.ref_fallback: a unit the screen never measured takes measured medians, not
            m = fb  # the env table's (324 of v6g's 1,089 sets carry its 64,000-token placeholder: clears clip at +log 8)
        key = (name, game, level, tuple(sorted(fb.items())) if fb else None)
        if key not in cache:
            baselines = (self.mg_point(game).get("baselines") or []) if game else []
            actions = m.get("median_actions")
            ref = (((self.cfg.get("mg_sets") or {}).get(name) or {}).get("ref") or {}) if name else {}
            base_b = float(baselines[level - 1]) if 0 < level <= len(baselines) and baselines[level - 1] else None
            # multi-game (envs 22 Oct): a set no screen measured takes its references from the env table (``ref``:
            # {tokens, actions, baseline}); without any, the reward's token and action terms are both 0 for every try
            cache[key] = {"baseline": base_b or ref.get("baseline"),
                          "tokens": m.get("median_to_clear") or ref.get("tokens"),
                          "actions": actions or ref.get("actions")}
        return cache[key]

    def mg_turn_credit(self, folder, grp, rewards, name=None):
        """Per-reply GiGPO credit per try (mg_gigpo_steps): {"tokens": {n: reply tokens}, "extra": {n: extra},
        "filtered": {}}."""
        tries = []
        for row in grp:
            att = str(folder / "attempts" / row["label"])
            tries.append(TURN_CACHE.get(att) or turn_try(att))
        return self.mg_gigpo_steps(grp, rewards, tries, self.mg_group_floor(grp, name))

    def mg_gigpo_steps(self, grp, rewards, tries, floor):
        """mg.turn_credit mode "gigpo" (v6x; Feng+ 2025 GiGPO in full): each gameplay reply is a step;
        step groups are the replies of the group's tries (every visit, revisits in the same try included) at one
        anchor state (level, frame hash of the board the reply saw); a step's return is the discounted return from it,
        gamma^(replies left to the try's last scored reply) x (the try's reward - the fail floor) (sparse terminal
        reward, a fail 0); A_step = (G - mean) / std over its anchor group (0 for a group of one or no spread); the
        reply's extra is omega x A_step, added to the try's leave-one-out advantage (mg_reply_balance, no clip, no
        rescale). Replies after the try's end (a harvested try's next-level request) get none."""
        tc = self.cfg["mg"]["turn_credit"]
        gamma, omega = float(tc.get("gamma", 0.95)), float(tc.get("omega", 1.0))
        if tc.get("step_scale") == "episode_std":  # GiGPO normalizes both terms per group; our episode term is the raw
            # leave-one-out reward gap, so the unit-variance step term is put on the same scale: x the group's reward
            # std (low-spread groups were step-dominated: hero-v6g ka59-L2, 30 sign flips at w_abs_median 0.22)
            omega *= statistics.pstdev([float(r) for r in rewards]) if len(rewards) > 1 else 0.0
        visits = {}  # anchor -> [(try, n, G)]
        for i, ((replies, end), reward) in enumerate(zip(tries, rewards)):
            ret = max(0.0, float(reward) - floor)
            idx = [k for k, (_, _, _, step) in enumerate(replies) if end is None or step <= end]
            last = idx[-1] if idx else -1
            for k in idx:
                n, _, anchor, _ = replies[k]
                if anchor is not None:
                    visits.setdefault(anchor, []).append((i, n, ret * gamma ** (last - k)))
        extra, stats = [dict() for _ in grp], self.__dict__.setdefault("mg_gigpo_stats", {})
        for anchor, vs in visits.items():
            stats["steps"] = stats.get("steps", 0) + len(vs)
            if len(vs) < 2:
                continue
            stats["in_group"] = stats.get("in_group", 0) + len(vs)
            if len({v[0] for v in vs}) >= 2:
                stats["cross_try"] = stats.get("cross_try", 0) + len(vs)
            g = [v[2] for v in vs]
            mu, sd = statistics.mean(g), statistics.pstdev(g)
            for i, n, G in vs:
                a = (G - mu) / sd if sd > 1e-9 else 0.0
                extra[i][n] = omega * a
                stats["abs_sum"] = stats.get("abs_sum", 0.0) + abs(a)
                stats["nonzero"] = stats.get("nonzero", 0) + int(a != 0.0)
            stats.setdefault("sizes", []).append(len(vs))
        stats["no_anchor"] = stats.get("no_anchor", 0) + sum(1 for replies, _ in tries for r in replies if r[2] is None)
        return [{"tokens": {n: t for n, t, _, _ in replies}, "extra": e, "filtered": {}} for (replies, _), e in zip(tries, extra)]

    def mg_gigpo_summary(self):
        """This update's step-group stats: share of steps in an anchor group of size >= 2 (any, and
        across >= 2 tries), mean |A_step| over those steps and over all steps, median group size; then reset."""
        st = self.__dict__.pop("mg_gigpo_stats", None) or {}
        steps = st.get("steps", 0)
        if not steps:
            return None
        sizes = sorted(st.get("sizes") or [])
        return {"steps": steps, "share_group2": round(st.get("in_group", 0) / steps, 4),
                "share_cross_try": round(st.get("cross_try", 0) / steps, 4),
                "mean_abs_step_in_group": round(st.get("abs_sum", 0.0) / st["in_group"], 4) if st.get("in_group") else None,
                "mean_abs_step": round(st.get("abs_sum", 0.0) / steps, 4), "nonzero": st.get("nonzero", 0),
                "group_size_median": sizes[len(sizes) // 2] if sizes else None, "groups": len(sizes),
                "no_anchor": st.get("no_anchor", 0)}

    def mg_rows(self, folder):
        return [r for r in json.loads((Path(folder) / "level-bench.json").read_text())["attempts"] if usable(r)
                and restored(r) and level_of(r) is not None]


    def mg_digest_ok(self, row):
        """mg.digest_gate: a try restored from an in-run snapshot (harvest or branch cut) trains
        only when its first live prompt's token ids hash as the source's recorded request did (byte-identical)."""
        rs = row.get("restore")
        if not self.cfg["mg"].get("digest_gate") or not rs:
            return True
        src = str(rs.get("source") or "")
        if not re.search(r"/(harvest|branch|branch-from-tar)/", src):
            return True
        return bool((rs.get("prompt_digest") or {}).get("same"))


    async def mg_loop(self):
        """The redesign loop (arena.rl_seq, mg.redesign): the only recipe."""
        mg = self.cfg["mg"]
        stub = os.environ.get("ARC_PREFLIGHT_STUB") or ""
        if stub.startswith("{") and json.loads(stub).get("shrink"):
            # fast CPU preflight: every path of the round at a fraction of its tries
            self.shrink_mg(mg, int(json.loads(stub).get("max_steps") or 2))
            red = mg.get("redesign") if isinstance(mg.get("redesign"), dict) else {}
            self.feed.emit("preflight_shrink", max_steps=red.get("max_steps"),
                           phase_groups=(red.get("backlog") or {}).get("phase_groups"))
        self.loop_lag = {"tick": 0.0, "round": 0.0}
        self.mg_lag_task = asyncio.ensure_future(self.mg_loop_lag())
        if not mg.get("redesign"):
            raise ValueError("rl_loop: mg.redesign (arena.rl_seq) is the only loop; the v7 async loop is gone")
        from . import rl_seq

        return await rl_seq.run(self)


    @staticmethod
    def shrink_mg(mg, steps=2):
        """The CPU preflight's shrink of an mg dict (in place): the real run's 64 steps x 16 groups ran the CPU leg past
        its 45-min ceiling (26 Oct); at most 2 steps (``steps``: the NK leg's 4) x 2 groups, never raising a smaller
        setting."""
        red = mg.get("redesign")
        if isinstance(red, dict):
            red["max_steps"] = min(int(red.get("max_steps") or steps), int(steps))
            red["backlog"] = {**(red.get("backlog") or {}),
                              "phase_groups": min(int((red.get("backlog") or {}).get("phase_groups") or 16), 2)}
        return mg


def trainer_command(cfg, source, work):
    script = Path(source) / cfg["trainer_script"]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHONPATH", "VIRTUAL_ENV"))}
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", AXOLOTL_SONICMOE_NVFP4_BACKEND="dequant",
               PYTHONPATH=str(script.parent),
               HF_HUB_OFFLINE="0", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")  # as the pilot; v3: 16 GiB fragmented at 32K
    env.update({str(k): str(v) for k, v in (cfg.get("trainer_env") or {}).items()})
    if cfg.get("attention"):  # RL1_ATTENTION: masked (trainer default), selected or triton (rl1_sparse)
        env["RL1_ATTENTION"] = str(cfg["attention"])
    env.update({str(k): str(v) for k, v in (cfg.get("trainer_env") or {}).items()})  # e.g. RL1_SWITCH_INTERVAL, RL1_SPLIT_GEMM
    if (cfg.get("mg") or {}).get("served_route") and not (cfg["mg"].get("redesign") or {}).get("play_only"):
        env["RL1_ROUTER"] = "served"  # R3 from the first window: rl_seq's F1 encodes saved groups during the load
    command = [cfg["trainer_python"], "-u", str(script), "--dir", str(Path(work) / "trainer"),
               "--model", cfg["model_dir"], "--rank", str(cfg["rank"]), "--lr", str(cfg["lr"]),
               "--window", str(cfg["window"]), "--optimizer", cfg["optimizer"]]
    command += (["--loop-lora"] if cfg["loop_lora"] else []) + (["--tis-cap", str(cfg["tis_cap"])] if cfg["tis_cap"] else [])
    command += ["--prefix-chunk", str(int(cfg["path_chunk"]))]
    if os.environ.get("ARC_TRAINER_CACHE"):  # stage H: the trainer's Inductor / Triton caches (autotune results too)
        cache = Path(os.environ["ARC_TRAINER_CACHE"])  # in a folder the launcher restores from and saves to the volume
        env.update(TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"),
                   TRITON_CACHE_AUTOTUNING="1")
    mg = cfg.get("mg") or {}
    if float(mg.get("draft_weight") or 0) > 0:  # stage D (rl1_draft): the MTP draft's LoRA trained beside the policy
        env["RL1_DRAFT_WEIGHT"] = str(float(mg["draft_weight"]))
        env["RL1_DRAFT_LR_SCALE"] = str(float(mg.get("draft_lr_scale") or 1.0))
    if mg.get("load_digest"):  # H3: the loaded model's layer-3 digest (preload_host vs direct load, bitwise)
        env["RL1_LOAD_DIGEST"] = "1"
    env["RL1_ENCODE_PROCS"] = str(FAST["encode_procs"])  # cold start (RL C2 4b): encoders load beside the model
    # mg.trainer_cards (RL C2): one trainer process per card via torchrun, RL1_DIST=1; rank 0 serves the cmd dir
    cards = int((cfg.get("mg") or {}).get("trainer_cards") or 1)
    if cards > 1:
        env["RL1_DIST"] = "1"
        command = [command[0], "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={cards}",
                   *command[2:]]
    if os.environ.get("ARC_PREFLIGHT_FAKE_TRAINER"):  # CPU launch-gate preflight: the loop's paths without a model
        env["PYTHONPATH"] = str(Path(source) / "src")
        # one interpreter for the launcher's pre-spawn and the loop (arena.trainer_prespawn): the image's python
        command = [os.environ.get("ARC_FAKE_TRAINER_PYTHON") or sys.executable, "-u", "-m", "arena.fake_trainer",
                   "--dir", str(Path(work) / "trainer"),
                   "--prefix-chunk", str(int(cfg["path_chunk"]))]
    return command, env


async def run_rl_loop(study, trial, **kwargs):
    return await Loop(study, trial, **kwargs).run()


