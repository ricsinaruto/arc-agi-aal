"""Stand-in trainer for CPU launch-gate preflights: the loop's file protocol (``rl_loop.Trainer``) with no
model. Every update attempt's rollout file is read (the data path the real trainer takes) and adapters and checkpoints
are written as folders the replay stub and a resume accept. KL values are fixed small numbers: the preflight tests the
loop's control flow, not learning.

    python -m arena.fake_trainer --dir <work>/trainer
"""

import argparse
import json
import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path


def reply_tokens(path):
    tokens = replies = 0
    with open(path) as file:
        for line in file:
            row = json.loads(line)
            if row.get("scripted"):
                continue
            replies += 1
            tokens += int((row.get("usage") or {}).get("completion_tokens") or 0)
    return tokens, replies


def adapter(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 16, "fake": True}))
    names = [f"base_model.model.layers.{i}.mlp.lora_{ab}.weight" for i in range(2) for ab in "AB"]
    header = {n: {"dtype": "BF16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]} for i, n in enumerate(names)}
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    (folder / "adapter_model.safetensors").write_bytes(len(head).to_bytes(8, "little") + head + b"\0" * 2 * len(names))
    (folder / "adapter-tensors.json").write_text(json.dumps({"tensors": len(names), "expected": len(names)}))
    if draft_weight() > 0:  # stage D: rl1_draft.export's files beside it (rl_loop.draft_files checks them)
        modules = ["layers.0.self_attn.qkv_proj", "layers.0.mlp.shared_expert.down_proj"]
        dh = json.dumps({f"base_model.model.mtp.{m}.lora_{ab}.weight": {"dtype": "BF16", "shape": [1],
                                                                        "data_offsets": [2 * i, 2 * i + 2]}
                         for i, (m, ab) in enumerate((m, ab) for m in modules for ab in "AB")}).encode()
        dh += b" " * (-len(dh) % 8)
        (folder / "draft_lora.safetensors").write_bytes(len(dh).to_bytes(8, "little") + dh + b"\0" * 4 * len(modules))
        (folder / "draft_lora.json").write_text(json.dumps({"format": "fake", "modules": {m: {} for m in modules}}))
    return str(folder)


def draft_weight():
    return float(os.environ.get("RL1_DRAFT_WEIGHT") or 0.0)


def probes_of(windows, kl=0.02, based=None):
    """``based``: the names holding a base reference; as the real trainer, the others carry no kl_base/logratio_base."""
    out = {}
    for w in windows or []:
        name = w.get("name") or Path(w["rollout"]).parent.name
        row = {"kl_base": kl, "kl_previous": kl / 4, "logratio_base": 0.0, "tokens": 1000}
        if based is not None and name not in based:
            row = {"kl_previous": kl / 4, "tokens": 1000}
        out[name] = row
    return out


PARKED_MIB, RESIDENT_MIB = 3072, 250560


def card_used(mib):
    """The preflight's stand-in nvidia-smi (modal_rl1.FAKE_SMI) reads the trainer card's memory.used from here, so a
    park frees the card as on a B300 (mg.timeshare.lazy_boot boots server 2 only into a freed card)."""
    folder = os.environ.get("ARC_FAKE_SMI_DIR")
    if folder and Path(folder).is_dir():
        card = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
        (Path(folder) / f"used-{card}").write_text(str(int(mib)))


# the real trainer's ops the tiny shadow runs (ARC_PREFLIGHT_TINY_TRAINER) and the fields of its answer kept
TINY_OPS = ("accumulate", "old_pass", "step", "checkpoint", "checkpoint_wait", "load_checkpoint", "save_join", "route",
            "exit")
TINY_TIMEOUT_S = 900
TRAINER_PYTHON = "/workspace/axolotl-venv/bin/python"  # rl_loop DEFAULTS' trainer_python
TINY_KEYS = ("trained_tokens", "grad_norm", "loss", "update_norm", "old_windows", "served", "ratio_clip", "updates",
             "adaptive_ckpt")


class TinyProc:
    """rl1_tiny's serve loop in a child on the trainer's python (the stand-in's python has no torch): one JSON line per
    op each way. A dead or silent child is a failed op, never a hang (TINY_TIMEOUT_S)."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        out = os.environ.get("ARC_VALIDATION_OUTPUT")  # in the run's output: a leg's pack keeps it
        self.log = Path(out) / "tiny-serve.log" if out else self.root / "serve.log"
        base = Path(__file__).resolve().parents[2]
        self.src = base / "src"
        self.script = next((d / "rl1_tiny.py" for d in (base / "rl1", base / "scripts" / "posttrain")
                            if (d / "rl1_tiny.py").is_file()), None)
        self.proc = None

    def start(self):
        python = os.environ.get("ARC_TINY_PYTHON")
        if not python:
            python = TRAINER_PYTHON if Path(TRAINER_PYTHON).is_file() else sys.executable
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHONPATH", "VIRTUAL_ENV"))}
        env.update(PYTHONPATH=os.pathsep.join([str(self.script.parent), str(self.src)]), PYTHONUNBUFFERED="1",
                   CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false")
        with self.log.open("ab") as err:
            self.proc = subprocess.Popen([python, "-u", str(self.script), "serve", str(self.root)], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=err, env=env)

    def tail(self):
        try:
            return self.log.read_text(errors="replace")[-3000:]
        except OSError:
            return ""

    def fail(self, why):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None
        return {"ok": False, "error": why, "traceback": self.tail()}

    def call(self, op, args):
        if self.script is None:
            return {"ok": False, "error": "rl1_tiny.py not found beside the source"}
        if self.proc is None:
            self.start()
        try:
            self.proc.stdin.write((json.dumps({"op": op, "args": args}) + "\n").encode())
            self.proc.stdin.flush()
        except OSError as error:
            return self.fail(f"tiny trainer gone: {error!r}")
        ready, _, _ = select.select([self.proc.stdout], [], [], TINY_TIMEOUT_S)
        if not ready:
            return self.fail(f"tiny trainer silent {TINY_TIMEOUT_S} s on {op}")
        line = self.proc.stdout.readline()
        if not line:
            return self.fail(f"tiny trainer exited on {op} (code {self.proc.wait()})")
        return json.loads(line)


class Fake:
    def __init__(self, path_chunk=16384, folder=None):
        self.parked, self.updates, self.lr = False, 0, None
        self.path_chunk = int(path_chunk)
        self.served = {"windows": 0, "no_record": 0}  # served-route coverage of the windows seen (rl1_trainer's audit)
        self.ckpt_bg = None  # (thread, wake event, folder) of a background checkpoint write
        self.dir = Path(folder) if folder else None

    def shadow(self, cmd):
        """ARC_PREFLIGHT_TINY_TRAINER=1 ((1)): the loop's training commands also run on the real
        rl1_trainer (rl1_tiny: the tiny model, CPU, synthetic windows by spec name, served routes when the loop turned
        R3 on), in a TinyProc child on the trainer's python, with the run's own arguments (Triton kernels off); its
        paths (checkpoints, adapters) under <trainer dir>/tiny. A failure there fails the command. None: off, or an
        op it does not take."""
        if os.environ.get("ARC_PREFLIGHT_TINY_TRAINER") != "1" or cmd.get("op") not in TINY_OPS:
            return None
        if cmd["op"] == "route" and cmd.get("router") not in ("served", "off"):
            return None
        if getattr(self, "tiny", None) is None:
            self.tiny = TinyProc((self.dir or Path(".")) / "tiny")
        args = {k: v for k, v in cmd.items() if k != "op"}
        for key in ("path", "save"):  # its own files: the stub and the loop read the stand-in's
            if args.get(key):
                args[key] = str(self.tiny.root / "paths" / Path(args[key]).parent.name / Path(args[key]).name)
        if cmd["op"] == "load_checkpoint" and not Path(args.get("path") or "").exists():
            return {"ok": True, "skipped": "no tiny checkpoint: an earlier container's"}  # a resume's first load
        got = self.tiny.call(cmd["op"], args)
        self.tally("ok" if got.get("ok") else "failed", cmd["op"])
        return got

    def tally(self, how, op):
        """<output>/tiny_ops.json: the shadow's op counts, for the leg's verdict (tiny_trainer_ran)."""
        self.tiny_ops = getattr(self, "tiny_ops", {"ok": {}, "failed": {}})
        self.tiny_ops[how][op] = self.tiny_ops[how].get(op, 0) + 1
        out = os.environ.get("ARC_VALIDATION_OUTPUT")
        if out:
            (Path(out) / "tiny_ops.json").write_text(json.dumps(self.tiny_ops))

    def ahead_encode(self, cmd):
        """The real trainer's ahead keys against this call's (gate legs 28 Oct: 0 of 192 ahead windows hit in every
        rl_seq run, unseen on CPU): rl1_trainer's ahead_start / ahead_specs / expand_cmd and rl1_encode.spec_key run on
        the loop's ahead-early.json and this accumulate. {"ahead_hits", "encoded"}; None without an ahead file."""
        import sys
        from types import SimpleNamespace

        path = self.dir / "ahead-early.json" if self.dir is not None else None
        if path is None or not path.is_file() or not cmd.get("attempts"):
            return None
        root = Path(__file__).resolve().parents[2]
        where = next((d for d in (root / "rl1", root / "scripts" / "posttrain") if (d / "rl1_trainer.py").is_file()), None)
        if where is None:
            return {"error": "rl1_trainer.py not found beside the source"}
        if str(where) not in sys.path:
            sys.path.insert(0, str(where))
        import rl1_encode
        import rl1_trainer

        class Done:
            def __init__(self, fn, args):
                self.value, self.error = None, None
                try:
                    self.value = fn(*args)
                except Exception as error:  # noqa: BLE001 - raised at result(), as a pool future
                    self.error = error

            def done(self):
                return True

            def result(self):
                if self.error is not None:
                    raise self.error
                return self.value

        class Sync:
            def submit(self, fn, *args):
                return Done(fn, args)

        t = rl1_trainer.Trainer.__new__(rl1_trainer.Trainer)
        t.args = SimpleNamespace(window=int(os.environ.get("ARC_FAKE_TRAINER_WINDOW") or 47104))
        t.encoder_pool = lambda procs: Sync()
        ahead_cmd = json.loads(path.read_text())
        t.ahead_start({**ahead_cmd, "encode_procs": max(1, int(ahead_cmd.get("encode_procs") or 1))})
        keys = {rl1_encode.spec_key(s) for s in (t.ahead_specs() or [])}
        call = [a for a in t.expand_cmd({**cmd, "encode_procs": 0}) if a["w"] > 0 or (cmd.get("signed") and a["w"] < 0)]
        hits = sum(rl1_encode.spec_key(a) in keys for a in call)
        return {"ahead_hits": hits, "encoded": len(call)}  # as rl1_trainer: every card-path window, pre-encoded or not

    def ckpt_join(self):
        """The pending background checkpoint lands now (as rl1_trainer.ckpt_join); its result, None when none."""
        if self.ckpt_bg is None:
            return None
        t, done, folder = self.ckpt_bg
        self.ckpt_bg = None
        done.set()
        t.join()
        return {"path": str(folder), "write_seconds": 0.0}

    def served_count(self, spec):
        """A window has served routes when its end request's record is in the capture directory (rl1_trainer
        served_routing's first lookup)."""
        if not spec.get("served_route"):
            return
        rid = None
        try:
            with open(spec["rollout"]) as file:
                for line in file:
                    rid = json.loads(line).get("rid") or rid
        except (OSError, ValueError):
            pass
        hit = bool(rid) and any(Path(spec["served_route"]).glob(f"{rid}-*.rt"))
        self.served["windows" if hit else "no_record"] += 1

    # forward ops a compacted real trainer refuses off the fused expert path (rl1_experts.moe_forward)
    FORWARD_OPS = ("accumulate", "score", "old_pass")

    def run(self, op, cmd):
        if os.environ.get("RL1_EXPERT_COMPACT") == "1" and op in self.FORWARD_OPS and not cmd.get("fused_experts"):
            raise RuntimeError("expert compaction needs the fused expert path (off)")  # as the real trainer
        if op == "update" and cmd.get("prefix_chunk") is not None \
                and int(cmd["prefix_chunk"]) != self.path_chunk:  # as the trainer's ``pinned``
            raise ValueError(f"prefix chunk pinned at {self.path_chunk}; this call asks for {int(cmd['prefix_chunk'])}")
        if op in ("score",):
            based = self.__dict__.setdefault("based", set())
            if cmd.get("label") == "base":
                based.update(w.get("name") or Path(w["rollout"]).parent.name for w in cmd.get("windows") or [])
            if cmd.get("label") in ("resume", "rollback") and getattr(self, "real_state", False):
                return {"windows": probes_of(cmd.get("windows"), based=based), "stand_in": True}  # a real run's state
            if cmd.get("label") in ("resume", "rollback", "kl_stop") or str(cmd.get("label")).startswith("reload"):
                # a restored state (the reload check's pre-reload repeats included), or the state a k3
                # early stop kept) scores as its update's probes
                return {"windows": probes_of(cmd.get("windows"), 0.02 + 0.002 * self.updates, based=based)}
            rows = probes_of(cmd.get("windows"), 0.0 if cmd.get("label") == "base" else 0.02, based=based)
            if cmd.get("dump"):  # score_check: a reply's raw log-probabilities, stood in
                for w in cmd.get("windows") or []:
                    rows[w.get("name") or Path(w["rollout"]).parent.name]["replies"] = [
                        {"request": w.get("end_request"), "start": 0, "raw": [-0.1] * 8}]
            return {"windows": rows}
        if op == "old_pass":  # rl1_trainer.op_old_pass: θ0's windows forward only, nothing accumulated or stepped
            if cmd.get("old_reset") or getattr(self, "old_names", None) is None:
                self.old_names = set()
            windows, errors, total = [], [], 0
            for spec in cmd.get("attempts") or []:
                self.served_count(spec)
                try:
                    tokens, _ = reply_tokens(spec["rollout"])
                except (OSError, KeyError, ValueError) as error:
                    errors.append({"name": spec.get("name"), "error": repr(error)[:200]})
                    continue
                total += tokens
                windows.append({"name": spec.get("name"), "targets": tokens})
                self.old_names.add(spec.get("name"))
            return {"ok": True, "old_pass": True, "old_windows": len(windows), "old_kept": len(self.old_names),
                    "scored_tokens": total, "windows": windows, "errors": errors, "train_seconds": 0.1}
        if op == "accumulate":  # one group's windows read and counted; the step comes with op "step"
            if cmd.get("first") or getattr(self, "acc", None) is None:
                self.acc = {"calls": 0, "tokens": 0}
            if cmd.get("old_reset") or getattr(self, "old_names", None) is None:
                self.old_names = set()
            names = [a.get("name") for a in cmd.get("attempts") or []]
            if cmd.get("keep_old"):
                self.old_names.update(names)
            clip = None
            if cmd.get("ratio_clip"):  # the trainer's θ0 tally: a clipped window without θ0 trains unclipped
                hit = sum(n in self.old_names for n in names)
                clip = {"preset": cmd["ratio_clip"], "attached": hit, "missing": len(names) - hit}
            out = self.run("update", {**cmd, "save": None, "probes": None, "kl_stop": -1.0})
            self.acc["calls"] += 1
            self.acc["tokens"] += int(out.get("trained_tokens") or 0)
            try:
                enc = self.ahead_encode(cmd)
            except Exception as error:  # noqa: BLE001 - the judge fails the leg on the missing share, with this reason
                enc = {"error": repr(error)[:300]}
            return {"ok": True, "accumulated": True, "windows": out["windows"], "errors": out["errors"],
                    "trained_tokens": out.get("trained_tokens"), "acc_calls": self.acc["calls"], "train_seconds": 0.1,
                    **({"encode": enc} if enc is not None else {}), **({"ratio_clip": clip} if clip else {})}
        if op == "step":
            if not getattr(self, "acc", None) or not self.acc["calls"]:
                return {"ok": False, "error": "nothing accumulated"}
            die = (json.loads(os.environ.get("ARC_PREFLIGHT_STUB") or "{}") or {}).get("die_step")
            self.steps_seen = getattr(self, "steps_seen", 0) + 1
            if die and self.steps_seen == int(die):  # the kill-in-update leg: a bug in the update
                return {"ok": False, "error": f"preflight die_step {int(die)}: the update fails"}
            tokens, self.acc = self.acc["tokens"], None
            out = self.run("update", {**cmd, "attempts": []})
            return {**out, "ok": True, "trained_tokens": tokens, "applied": True}
        if op == "update":
            if cmd.get("policy_frozen") and draft_weight() <= 0:  # rl1_trainer.update_once refuses it
                raise ValueError("policy_frozen without a draft LoRA: the step would train nothing")
            # standard dosing's k3 early stop (rl1_trainer op_update): a fixed stand-in approx KL of 1e-5 per reply
            kl = {"approx_kl": 1e-5, "approx_kl_tok": 1e-7, "clip_frac": 0.0}
            stop = cmd.get("kl_stop") is not None and kl["approx_kl"] > float(cmd["kl_stop"])
            if self.parked:
                card_used(RESIDENT_MIB)
            self.parked, self.updates = False, self.updates + (0 if stop else 1)
            windows, errors, total = [], [], 0
            for spec in cmd.get("attempts") or []:
                self.served_count(spec)
                try:
                    tokens, replies = reply_tokens(spec["rollout"])
                except (OSError, KeyError, ValueError) as error:
                    errors.append({"name": spec.get("name"), "error": repr(error)[:200]})
                    continue
                total += tokens
                windows.append({"name": spec.get("name"), "tokens": tokens, "targets": tokens, "trained": tokens,
                                "w": spec.get("w"), "tis": {"mean": 1.0, "capped": 0.0},
                                "signed": {"up": max(0.0, float(spec.get("w") or 0)), "down": max(0.0, -float(spec.get("w") or 0)),
                                           "gspo": {"replies": replies, "clipped": 0}} if cmd.get("signed") else None})
            if stop:  # the step is not taken (no save, no probes)
                return {**kl, "ok": True, "applied": False, "kl_stopped": True, "kl_stop": float(cmd["kl_stop"]),
                        "windows": windows, "errors": errors, "loss": 0.0, "trained_tokens": total, "train_seconds": 0.1,
                        "lr": cmd.get("lr")}
            out = {**kl, "windows": windows, "errors": errors, "loss": 0.0, "grad_norm": 0.01,
                   "update_norm": 0.0 if cmd.get("policy_frozen") else 0.1,
                   "trained_tokens": total, "train_seconds": 0.1, "probe_seconds": 0.0, "lr": cmd.get("lr")}
            if draft_weight() > 0:  # rl1_trainer.draft_finish's step fields (rl_loop.DRAFT_ROW)
                out.update(draft_kl=0.5, draft_top1=0.7, draft_agree=0.7, draft_paccept=0.75, draft_rows=total,
                           draft_grad_norm=0.01, draft_update_norm=0.01, draft_skipped=None)
            if cmd.get("probes"):
                out["probes"] = probes_of(cmd["probes"], 0.02 + 0.002 * self.updates)
            if cmd.get("save"):
                adapter(cmd["save"])
            return out
        if op in ("save",):
            return {"path": adapter(cmd["path"])}
        if op == "save_join":  # mg.async_adapter_save (rl1_trainer op_save_join): the fake writes in the step itself
            return {"pending": False}
        if op == "checkpoint":  # ``background`` (rl1_trainer op_checkpoint, MFU-1 E): written into .partial by a
            prior = self.ckpt_join()  # thread, renamed after RL1_FAKE_CKPT_SECONDS; one write in flight at a time
            folder = Path(cmd["path"])
            blob = json.dumps({"updates": self.updates, "lr": self.lr})  # the host snapshot: taken now
            if cmd.get("background"):
                import threading

                part = folder.with_name(folder.name + ".partial")
                part.mkdir(parents=True, exist_ok=True)
                (part / "state.json").write_text(blob)
                done = threading.Event()

                def write():
                    done.wait(3.0)
                    if folder.exists():
                        shutil.rmtree(folder)
                    part.rename(folder)

                t = threading.Thread(target=write, daemon=True)
                t.start()
                self.ckpt_bg = (t, done, folder)
                return {"path": str(folder), "background": True, "snapshot_seconds": 0.0, "seconds": 0.0, "gib": 0.0,
                        **({"prior": prior} if prior else {})}
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "state.json").write_text(blob)
            return {"path": str(folder), "seconds": 0.1, "gib": 0.0, **({"prior": prior} if prior else {})}
        if op == "checkpoint_wait":
            done = self.ckpt_join()
            return {"pending": done is not None, **(done or {})}
        if op == "load_checkpoint":
            self.ckpt_join()  # a background write of this folder lands first
            folder = Path(cmd["path"])
            if not (folder / "state.json").exists() and (folder / "master.safetensors").exists() \
                    and (folder / "optimizer.pt").exists():  # a real rl1 state (mg.init_state): fresh optimizer
                self.updates, self.real_state = 0, True
                return {"path": cmd["path"], "updates": 0, "real_state": True}
            self.real_state = False
            state = json.loads((folder / "state.json").read_text())
            self.updates = int(state["updates"])
            return {"path": cmd["path"], "updates": self.updates}
        if op == "park":
            already, self.parked = self.parked, True
            card_used(PARKED_MIB)
            acc = getattr(self, "acc", None)
            kept = None
            if acc and acc["calls"]:  # as rl1_trainer.op_park: gradients survive a mid-batch park only with keep_grads
                if cmd.get("keep_grads"):
                    kept = {"n": 4, "sq": float(acc["tokens"])}
                    self.kept_grads = kept
                else:
                    acc["calls"], acc["tokens"] = 0, 0  # .grad = None: the batch is gone
            return {"parked": True, "already": already, "seconds": 0.1, "grads_kept": kept}
        if op == "unpark":  # rl1_trainer.op_unpark
            if not self.parked:
                return {"already": True}
            self.parked = False
            card_used(RESIDENT_MIB)
            out = {"unparked": True, "seconds": 0.1}
            if getattr(self, "kept_grads", None):
                out["grads_restored"], self.kept_grads = dict(self.kept_grads), None
            return out
        if op == "reset":
            self.updates, self.lr = 0, cmd.get("lr")
            return {"reset": True}
        if op == "route" and cmd.get("router") is None and cmd.get("qsa") is None:  # the update's coverage read
            got = {"rows": self.served["windows"], "agree": self.served["windows"], "unserved_calls": 0,
                   "windows": dict(self.served)}
            if cmd.get("reset_counts"):
                self.served = {"windows": 0, "no_record": 0}
            return {"served": got}
        if op == "route":  # rl1-route: modes checked as the trainer checks them
            for key in ("router", "qsa"):
                if cmd.get(key) not in (None, "off", "capture", "replay") and not (key == "router" and cmd[key] == "served"):
                    raise ValueError(f"route {key}: {cmd.get(key)}")
            return {"router": cmd.get("router"), "qsa": cmd.get("qsa"), "recorded": 0, "qsa_recorded": 0,
                    "served": {"rows": 0, "agree": 0, "unserved_calls": 0, "windows": {}}}
        if op == "exit":
            self.ckpt_join()
            return {}
        raise ValueError(f"fake trainer: unknown op {op}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--prefix-chunk", type=int, default=16384)
    args, _ = parser.parse_known_args()
    folder = Path(args.dir)
    (folder / "cmd").mkdir(parents=True, exist_ok=True)
    (folder / "done").mkdir(parents=True, exist_ok=True)
    loaded, gate = {"fake": True}, os.environ.get("RL1_GPU_GATE")
    if os.environ.get("RL1_LOAD_DIGEST") == "1":  # mg.load_digest: as flash_qlora_job's loaded.phases.digest
        loaded["phases"] = {"digest": {"fake": True}}
    if gate:  # mg.trainer_overlap: the model's card part waits for the loop's gate file, as rl1_trainer.gpu_gate
        began = time.time()
        while not os.path.exists(gate):
            time.sleep(0.1)
        loaded["gate_waited_s"] = round(time.time() - began, 2)
    (folder / "alive.json").write_text(json.dumps({"state": "ready", "loaded": loaded, "time": time.time()}))
    fake, seen = Fake(args.prefix_chunk, folder), set()
    while True:
        for path in sorted((folder / "cmd").glob("*.json")):
            if path.name in seen:
                continue
            seen.add(path.name)
            cmd = json.loads(path.read_text())
            try:
                shadow = fake.shadow(cmd)
                if shadow is not None and not shadow.get("ok"):  # the real update path failed: the leg's answer
                    out = {"ok": False, "error": f"tiny trainer ({cmd['op']}): {shadow.get('error') or shadow.get('stop')}\n"
                                                 f"{(shadow.get('traceback') or '')[-2500:]}",  # the run's error row
                           "traceback": shadow.get("traceback"), "tiny": True}
                else:
                    out = {"ok": True, **fake.run(cmd["op"], cmd)}
                    if shadow is not None:
                        out["tiny"] = {k: shadow.get(k) for k in TINY_KEYS if shadow.get(k) is not None}
            except Exception as error:  # the loop sees the failure as the real trainer reports one
                out = {"ok": False, "error": repr(error)[:500]}
            tmp = folder / "done" / (path.name + ".tmp")
            tmp.write_text(json.dumps(out))
            tmp.rename(folder / "done" / path.name)
            print(json.dumps({"op": cmd["op"], "ok": out["ok"]}), flush=True)
            if cmd["op"] == "exit":
                return
        time.sleep(0.2)


if __name__ == "__main__":
    main()
