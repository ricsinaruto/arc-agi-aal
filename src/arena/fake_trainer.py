"""Stand-in trainer for CPU launch-gate preflights: the loop's file protocol (``rl_loop.Trainer``) with no
model. Every update attempt's rollout file is read (the data path the real trainer takes) and adapters and checkpoints
are written as folders the replay stub and a resume accept. KL values are fixed small numbers: the preflight tests the
loop's control flow, not learning.

    python -m arena.fake_trainer --dir <work>/trainer
"""

import argparse
import json
import os
import shutil
import subprocess
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
    return str(folder)


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


def nb_fake(path, windows):
    """mg.nextboard stand-in: every row's turns must hold a user text and a target; stats are fixed."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows or any(not r.get("turns") or any(not t.get("user") or not t.get("target") for t in r["turns"]) for r in rows):
        raise ValueError(f"nextboard file {path}: empty or malformed")
    picked = rows[:windows] if windows else rows
    steps = sum(len(r["turns"]) for r in picked)
    return {"windows": len(picked), "steps": steps, "targets": sum(len(t["target"]) // 3 + 1 for r in picked for t in r["turns"]),
            "em": 0.1, "body": 0.1, "token_acc": 0.9, "nll": 0.3, "fake": True}


PARKED_MIB, RESIDENT_MIB = 3072, 250560


def card_used(mib):
    """The preflight's stand-in nvidia-smi (modal_rl1.FAKE_SMI) reads the trainer card's memory.used from here, so a
    park frees the card as on a B300 (mg.timeshare.lazy_boot boots server 2 only into a freed card)."""
    folder = os.environ.get("ARC_FAKE_SMI_DIR")
    if folder and Path(folder).is_dir():
        card = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
        (Path(folder) / f"used-{card}").write_text(str(int(mib)))


class Fake:
    def __init__(self, path_chunk=16384, folder=None):
        self.parked, self.updates, self.lr, self.kept = False, 0, None, {}
        self.path_chunk = int(path_chunk)
        # split-half SNR stand-in (RL1_SNR_FAKE_PYTHON): checkpoints in rl1_trainer's format (master.safetensors,
        # optimizer.pt) and one AdamW step per update, by snr_math's ``fake`` commands, as arc2's preflight trainer
        self.snr_python = os.environ.get("RL1_SNR_FAKE_PYTHON")
        self.folder, self.current, self.made = Path(folder or "."), None, 0
        self.served = {"windows": 0, "no_record": 0}  # served-route coverage of the windows seen (rl1_trainer's audit)
        self.ckpt_bg = None  # (thread, wake event, folder) of a background checkpoint write

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

    def tensors(self, *args):
        script = Path(__file__).with_name("snr_math.py")
        done = subprocess.run([self.snr_python, str(script), "fake", *map(str, args)], capture_output=True, text=True,
                              timeout=600)
        if done.returncode:
            raise RuntimeError("snr_math fake: " + " | ".join(done.stderr.strip().splitlines()[-3:]))
        return json.loads(done.stdout.strip().splitlines()[-1])

    def state(self):
        if self.current is None:
            self.current = self.new_state()
            self.tensors("init", "--out", self.current)
        return self.current

    def new_state(self):
        self.made += 1
        return self.folder / "states" / f"{self.made:04d}"

    def snr_run(self, op, cmd):
        """The SNR stand-in's ops, or None for the plain fake's."""
        if op == "nudge":
            made = self.new_state()
            out = self.tensors("nudge", "--state", self.state(), "--out", made, "--value", cmd.get("value", 1e-12))
            self.current = made
            return out
        if op == "checkpoint":
            folder = Path(cmd["path"])
            folder.mkdir(parents=True, exist_ok=True)
            for name in ("master.safetensors", "optimizer.pt"):
                shutil.copyfile(self.state() / name, folder / name)
            return {"path": str(folder), "seconds": 0.1, "gib": 0.0}
        if op == "load_checkpoint" and (Path(cmd["path"]) / "master.safetensors").exists():
            self.current = Path(cmd["path"])
            return {"path": cmd["path"], **self.tensors("info", "--state", self.current)}
        if op == "update":
            for spec in cmd.get("attempts") or []:
                reply_tokens(spec["rollout"])  # the data path: the rollout must be readable
            made = self.new_state()
            out = self.tensors("update", "--state", self.state(), "--out", made, "--lr", cmd["lr"], "--clip",
                               cmd.get("clip", 1.0), "--attempts",
                               json.dumps([{"name": a.get("name"), "w": a.get("w")} for a in cmd.get("attempts") or []]))
            self.current = made
            return {**out, "errors": [], "windows": [{"name": a.get("name"), "w": a.get("w")} for a in cmd.get("attempts") or []]}
        return None

    # forward ops a compacted real trainer refuses off the fused expert path (rl1_experts.moe_forward)
    FORWARD_OPS = ("fit", "k3_score", "accumulate", "score_ids", "score", "old_logprobs", "batch_check", "update",
                   "park_check", "chunk_check", "served_check", "r3_check", "profile")

    def run(self, op, cmd):
        if os.environ.get("RL1_EXPERT_COMPACT") == "1" and op in self.FORWARD_OPS and not cmd.get("fused_experts"):
            raise RuntimeError("expert compaction needs the fused expert path (off)")  # as the real trainer
        if self.snr_python:
            out = self.snr_run(op, cmd)
            if out is not None:
                return out
        if op in ("update", "old_logprobs") and cmd.get("prefix_chunk") is not None \
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
            return {"windows": probes_of(cmd.get("windows"), 0.0 if cmd.get("label") == "base" else 0.02, based=based)}
        if op == "nb_score":
            return nb_fake(cmd["file"], None)
        if op == "accumulate":  # v7 (rl_async): one group's windows read and counted; the step comes with op "step"
            if cmd.get("first") or getattr(self, "acc", None) is None:
                self.acc = {"calls": 0, "tokens": 0, "kept": set()}
            out = self.run("update", {**cmd, "save": None, "probes": None, "kl_stop": -1.0})
            self.acc["kept"] |= set(cmd.get("keep_old") or ())  # as rl1_trainer: kept pi_old gathers over the step's accumulate calls
            self.acc["calls"] += 1
            self.acc["tokens"] += int(out.get("trained_tokens") or 0)
            return {"ok": True, "accumulated": True, "windows": out["windows"], "errors": out["errors"],
                    "trained_tokens": out.get("trained_tokens"), "acc_calls": self.acc["calls"], "train_seconds": 0.1}
        if op == "step":
            if not getattr(self, "acc", None) or not self.acc["calls"]:
                return {"ok": False, "error": "nothing accumulated"}
            tokens, kept, self.acc = self.acc["tokens"], self.acc["kept"], None
            out = self.run("update", {**cmd, "attempts": []})
            self.kept_old = kept  # k3_score after the step reads them
            return {**out, "ok": True, "trained_tokens": tokens, "applied": True}
        if op == "acc_reset":
            self.acc = None
            return {"ok": True}
        if op == "update":
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
            out = {**kl, "windows": windows, "errors": errors, "loss": 0.0, "grad_norm": 0.01, "update_norm": 0.1,
                   "trained_tokens": total, "train_seconds": 0.1, "probe_seconds": 0.0, "lr": cmd.get("lr")}
            if cmd.get("probes"):
                out["probes"] = probes_of(cmd["probes"], 0.02 + 0.002 * self.updates)
            if cmd.get("aux"):  # mg.nextboard: the packed file read as the trainer reads it
                out["aux"] = nb_fake(cmd["aux"]["file"], int(cmd["aux"].get("windows") or 2))
            if cmd.get("save"):
                adapter(cmd["save"])
            self.kept_old = set(cmd.get("keep_old") or ())  # mg.k3_measure: pi_old of these tries, from this step
            return out
        if op == "k3_score":  # forward-only k3 vs the kept pi_old (rl1_trainer op_k3_score); drops the kept copy
            names = [spec.get("name") or Path(spec["rollout"]).parent.name for spec in cmd.get("attempts") or []]
            kept, self.kept_old = getattr(self, "kept_old", set()), set()
            missing = [n for n in names if n not in kept]
            for spec in cmd.get("attempts") or []:
                reply_tokens(spec["rollout"])  # the windows' data path, as the trainer reads it
            return {"ok": bool(names) and not missing, "approx_kl": 1e-5 if names and not missing else None,
                    "approx_kl_tok": 1e-7, "clip_frac": 0.0, "replies": len(names), "windows": len(names),
                    "missing": len(missing), "kept": len(kept), "seconds": 0.1}
        if op == "targets":
            return {"targets": {w.get("name"): 1000 for w in cmd.get("windows") or []}}
        if op in ("save",):
            return {"path": adapter(cmd["path"])}
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
                    done.wait(float(os.environ.get("RL1_FAKE_CKPT_SECONDS", "3")))
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
        if op == "compact_experts":  # rl1_trainer.op_compact_experts (MFU-1 expert compaction read)
            if self.parked:
                raise RuntimeError("compact_experts: unpark first")
            return {"ok": True, "layers": 47, "kept": 320, "freed_gib": 21.0, "calls": 0}
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
        if op == "old_logprobs":  # rl1-gold: kept copies by label; a compare needs an earlier keep
            names = []
            for spec in cmd.get("attempts") or []:
                self.served_count(spec)
                reply_tokens(spec["rollout"])  # the data path: the rollout must be readable
                names.append(spec.get("name"))
            if cmd.get("compare") and cmd["compare"] not in self.kept:
                raise ValueError(f"no kept old log-probs {cmd['compare']!r}")
            if cmd.get("keep"):
                self.kept[cmd["keep"]] = names
            out = {"windows": len(names), "seconds": 0.1, "prefix_reused": 0, "window_wait_seconds": 0.0,
                   "engine_gap": [[n, 100, 5.7, 1] for n in names]}  # rl1_trainer.engine_gap rows (0.057 floor)
            if cmd.get("compare"):
                out["compare"] = {"against": cmd["compare"], "replies": len(names), "clipped_share": 0.0, "log_s_mean": 0.0}
            return out
        if op == "old_use":
            return {"windows": len(self.kept[cmd["label"]]), "label": cmd["label"]}
        if op == "old_compare":
            return {"compare": {"a": cmd["a"], "b": cmd["b"], "replies": len(self.kept[cmd["a"]]) + 0 * len(self.kept[cmd["b"]])}}
        if op == "chunk_check":
            for spec in cmd.get("attempts") or []:
                reply_tokens(spec["rollout"])
            return {"windows": len(cmd.get("attempts") or []),
                    "settings": [{"setting": f"{st.get('select', 'qsa')}@{st['chunk']}", "sampler_gap": {}, "gap": {}}
                                 for st in cmd["settings"]]}
        if op in ("carry_check", "causal_check"):  # an arm / config with fixes must change a path the op runs with
            for arm in (cmd.get("arms") if op == "carry_check" else cmd.get("configs")) or []:
                fixes = arm.get("fixes") if isinstance(arm, dict) else None
                if fixes and all(cmd.get(k) == v for k, v in fixes.items()):
                    return {"ok": False, "error": f"{arm.get('name')}: fixes {fixes} change nothing the op runs with"}
                if isinstance(arm, dict) and arm.get("checkpoint") and not Path(arm["checkpoint"]).exists():
                    return {"ok": False, "error": f"{arm.get('name')}: no checkpoint at {arm['checkpoint']}"}
        if op == "causal_check":
            for spec in cmd.get("attempts") or []:
                reply_tokens(spec["rollout"])
            return {"length": cmd.get("length", 8704), "configs": {c.get("name", "as_is"): {"first_over_bar": None}
                                                                  for c in cmd.get("configs") or [{}]}}
        if op == "carry_check":  # rl1_trainer.op_carry_check (MFU-1 C read): every window's rollout readable
            for spec in cmd.get("attempts") or []:
                reply_tokens(spec["rollout"])
            arms = cmd.get("arms") or ["base", "half_chunk", "repeat", "carry", "carry_margin"]
            return {"chunk": 16384, "arms": {(a["name"] if isinstance(a, dict) else a): {"ok": True, "windows": len(cmd.get("attempts") or [])}
                                              for a in arms}}
        if op == "adapter_scale":
            return {"factor": float(cmd["factor"]), "adapter_norm": 0.1 * float(cmd["factor"])}
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
        if op == "r3_check":  # the data path: every window's rollout readable
            for spec in cmd.get("attempts") or []:
                reply_tokens(spec["rollout"])
            row = {"windows": len(cmd.get("attempts") or []), "tokens": 0, "mean": 0.0, "mean_abs": 0.0, "rms": 0.0,
                   "tis_hi_share": 0.0, "tis_lo_share": 0.0}
            return {"off": row, "served": {**row, "route": {"rows": 0, "agree": 0}}, "grad_cos": 1.0, "seconds": 0.1}
        if op == "experts":
            return {"path": cmd.get("path", "native"), "stats": {}}
        if op == "expert_ref":
            return {"layer": int(cmd.get("layer", 24)), "delta": {"cos": 1.0}, "grads": {}}
        if op == "served_check":
            for spec in cmd.get("attempts") or []:
                reply_tokens(spec["rollout"])
            return {"windows": len(cmd.get("attempts") or []), "parts": {t: {"slope": 1.0} for t in cmd.get("zero") or ("all",)}}
        if op == "prepin":  # start-up cut (d): the real trainer allocates the park's pinned buffers on a thread
            return {"started": True, "tensors": 0, "gib": 0.0}
        if op == "base_export":  # start-up cut (c): the probes' base references under the loop's key
            path = Path(cmd["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"key": cmd["key"], "names": sorted(cmd["names"])}))
            return {"windows": len(cmd["names"]), "path": str(path)}
        if op == "base_drop":
            return {"dropped": len(cmd.get("names") or [])}
        if op == "base_import":
            blob = json.loads(Path(cmd["path"]).read_text())
            if blob["key"] != cmd["key"] or blob["names"] != sorted(cmd["names"]):
                raise ValueError(f"base cache mismatch: {blob}")
            self.__dict__.setdefault("based", set()).update(blob["names"])
            return {"windows": len(blob["names"]), "names": blob["names"]}
        if op == "exit":
            self.ckpt_join()
            return {}
        if op in ("accept_reference", "nudge", "park_check"):
            return {"identical": True} if op == "park_check" else {}
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
                out = {"ok": True, **fake.run(cmd["op"], cmd)}
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
