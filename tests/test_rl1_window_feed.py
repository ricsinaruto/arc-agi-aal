"""MFU stage W (22 Oct): the GSPO old pass and the update steps read their windows through ``Trainer.window_feed``;
encoder processes (``encode_procs``) and the old pass's host windows (``window_cache``) must hand the card the same
tensors as the default thread path, and the update the same gradients."""
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import rl1_encode  # noqa: E402
import rl1_trainer  # noqa: E402

SYS, USER, ASSIST, END, THINK_END = 1, 2, 3, 4, 5
SAMPLING = {"temperature": 0.7, "top_p": 0.95, "top_k": 20}


def text_ids(text):
    return [100 + ord(c) % 200 for c in text]


def message_ids(m):
    return [{"system": SYS, "user": USER, "assistant": ASSIST}[m["role"]]] + text_ids(m["content"]) + [END]


def stub_encoded(capture, pair=False):
    """A deterministic stand-in for the processor's encode: a reply's tokens sit in later prompts token for token."""
    messages = capture["payload"]["messages"]
    prompt = [t for m in messages for t in message_ids(m)] + [ASSIST]
    ids = torch.tensor([prompt + text_ids(capture["reply"]["content"]) + [END]])
    labels = ids.clone()
    labels[:, :len(prompt)] = -100
    pixels = torch.randn(len(messages), 16, generator=torch.Generator().manual_seed(len(prompt)))
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels, "pixel_values": pixels}
    return (inputs, len(prompt)) if pair else inputs


def stub_loader(model, window):  # rl1_encode.init in an encoder process
    return stub_encoded, THINK_END


def write_rollout(folder, replies, seed):
    folder.mkdir(parents=True)
    rows, messages = [], []
    for k in range(replies):
        new = ([{"role": "system", "content": f"game {seed}"}] if k == 0 else
               [{"role": "assistant", "content": rows[-1]["reply"]["content"]}]) + [{"role": "user", "content": f"frame {seed}.{k} " * (3 + k % 4)}]
        reply = f"move {seed}-{k} " * (4 + (seed + k) % 5)
        keep = len(messages)
        messages = messages + new
        prompt = sum(len(message_ids(m)) for m in messages) + 1
        tokens = text_ids(reply) + [END]
        logprobs = [-0.05 - 0.01 * (i % 7) for i in range(len(tokens))]
        rows.append({"n": k, "step": k, "gameplay": True, "keep": keep, "messages": new, "template": {}, "sampling": SAMPLING,
                     "reply": {"content": reply}, "finish_reason": "stop",
                     "usage": {"prompt_tokens": prompt, "completion_tokens": len(tokens)}, "prompt_digest": None,
                     "tokens": tokens, "logprobs": logprobs,
                     "top": [[[t, t + 1], [v, v - 2.0]] for t, v in zip(tokens, logprobs)]})
    (folder / "rollout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(folder / "rollout.jsonl")


def bits(x):
    """Bit-level identity: tensors by dtype, shape and bytes; floats by their IEEE bytes (NaN == NaN)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().contiguous()
        return ("t", str(x.dtype), tuple(x.shape), x.view(torch.uint8).numpy().tobytes() if x.numel() else b"")
    if isinstance(x, dict):
        return {k: bits(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [bits(v) for v in x]
    if isinstance(x, float):
        return ("f", struct.pack("<d", x))
    return x


@pytest.fixture
def tries(tmp_path):
    base = {"window": 600, "floor": 64, "margin": 32, "credited_only": True, "whole": True}
    out = []
    for i, (w, share) in enumerate([(1.0, "s0"), (-0.5, "s0"), (0.7, "s1"), (-1.0, "s1")]):
        out.append({**base, "rollout": write_rollout(tmp_path / f"t{i}", 12, i), "w": w, "share": share, "name": f"t{i}"})
    return out


def make_trainer(monkeypatch, log):
    monkeypatch.setattr(rl1_trainer, "CARD", "cpu")
    t = rl1_trainer.Trainer.__new__(rl1_trainer.Trainer)
    t.args = SimpleNamespace(window=600, model="stub", tis_cap=0.0)
    t.path_chunk = t.prefix_chunk = 16384
    t.encoded = t.encoded_host = stub_encoded
    t.think_end = THINK_END
    t.window_cache, t.encoders, t.encoder_loader = rl1_encode.Cache(), None, stub_loader
    p = torch.nn.Parameter(torch.zeros(()))
    t.params = [("model.layers.0.self_attn.q_proj.lora_B", p)]
    t.optimizer = torch.optim.SGD([p], lr=0.0)
    t.updates, t.references = 0, {}
    t.mismatch = lambda made, out: {}
    t.adapter_norm = lambda: 0.0
    t.window_why = lambda spec: {}
    t.save_adapter = lambda folder, only=None: {"path": folder}

    def window_pass(made, *, scale=None, vocab=None, shared=None):
        arrays, ids = made["arrays"], made["inputs"]["input_ids"][0]
        log.append({"grad": scale is not None, "name": made["name"], "old": "old" in arrays,
                    "made": bits({"inputs": made["inputs"], "arrays": {k: v for k, v in arrays.items() if k != "old"},
                                  **{k: v for k, v in made.items() if k not in ("inputs", "arrays")}}),
                    "old_bits": bits(arrays.get("old"))})
        # the policy moves with each applied step (s != 1 for the later steps), reads every input tensor
        base = -(ids[arrays["positions"]].float() % 7) / 10 - 1e-3 * made["inputs"]["pixel_values"].sum() + 0.01 * t.updates
        actor = base + t.params[0][1]
        valid = torch.ones_like(base, dtype=torch.bool)
        loss = signed = None
        if scale is not None:
            loss, _, signed = t.weighted_loss(arrays, actor, valid, scale)
            loss.backward()
        return {"actor": actor.detach().float(), "raw": base.detach(), "valid": valid, "entropy": None, "signed": signed,
                "opd": None, "tis": None, "loss": None if loss is None else float(loss.detach()), "prefix_seconds": 0.0,
                "forward_seconds": 0.0, "backward_seconds": 0.0, "prefix_reused": False, "cut": 0, "tokens": int(ids.numel()),
                "tail": made["tail"], "targets": int(arrays["positions"].numel())}

    t.window_pass = window_pass
    return t


def one_update(t, batches, fast):
    """As ``rl_loop.dosed_update`` with GSPO: pi_old of the later steps' tries, then one update per batch."""
    common = {"share_prefix": True, "prefix_chunk": 16384, "signed": True, **fast}
    gspo = {"policy_loss": "gspo", "gspo_clip": [3e-4, 4e-4]}
    later = [s for b in batches[1:] for s in b]
    old = t.op_old_logprobs({"attempts": later, **common})
    steps = [t.op_update({"attempts": b, "lr": 1.0, "max_windows": 100, "digest": True, **common, **gspo,
                          **({"save": "adapter"} if i == len(batches) - 1 else {})}) for i, b in enumerate(batches)]
    return old, steps


FAST = {"encode_procs": 2, "window_cache": 1.0, "prefetch": 2}


def test_stage_w_feeds_the_same_windows_and_gradients(monkeypatch, tries):
    batches = [tries[:1], tries[1:]]
    runs = {}
    for label, fast in (("default", {}), ("procs+cache", FAST), ("cache", {"window_cache": 1.0}), ("procs", {"encode_procs": 2})):
        log = []
        t = make_trainer(monkeypatch, log)
        try:
            old, steps = one_update(t, batches, fast)
        finally:
            if t.encoders is not None:
                t.encoders[1].shutdown()
        runs[label] = (log, {k: bits(v) for k, v in t.old.items()}, [s["grad_digest"] for s in steps], old, steps)
    log, old_bits, digests, old, steps = runs["default"]
    later = [e for e in log if not e["grad"]]
    assert old["windows"] == len(later) > 3 and "encode" not in old
    step1 = [e for e in log if e["grad"] and e["name"].split("#")[0] == "t0"]
    step2 = [e for e in log if e["grad"] and e["name"].split("#")[0] != "t0"]
    # (b) step 1's tries are not in the old pass and train at s = 1 (no pi_old); every later window has pi_old
    assert step1 and not any(e["old"] for e in step1) and not any(e["name"].startswith("t0") for e in later)
    assert step2 and all(e["old"] for e in step2) and {e["name"] for e in step2} == {e["name"] for e in later}
    assert sum(w["signed"]["gspo"]["clipped"] for w in steps[1]["windows"]) > 0  # pi_old moved s off 1
    for label in ("procs+cache", "cache", "procs"):
        other_log, other_old, other_digests, other_old_row, other_steps = runs[label]
        assert other_log == log, label  # same tensors into every pass, same pi_old into the update
        assert other_old == old_bits and other_digests == digests, label
        assert [s["loss"] for s in other_steps] == [s["loss"] for s in steps], label
    fast_old, fast_steps = runs["procs+cache"][3], runs["procs+cache"][4]
    assert fast_old["encode"]["cache"]["kept"] == len(later) and fast_old["encode"]["errors"] == []
    assert fast_steps[1]["encode"]["cache_hits"] == len(step2) and fast_steps[1]["encode"]["encoded"] == 0
    assert fast_steps[0]["encode"]["cache_hits"] == 0  # step 1's tries were never in the old pass
    assert fast_steps[1]["encode"]["cache"]["gib"] == 0.0 and fast_steps[1]["encode"]["cache"]["windows"] == 0


def test_cache_cap_and_misses_reencode(monkeypatch, tries):
    log = []
    t = make_trainer(monkeypatch, log)
    one = rl1_encode.nbytes(t.attempt_window({**tries[2], "whole": False, "end": None, "name": "x"}))
    cap = 2.5 * one / 2**30
    old, steps = one_update(t, [tries[:1], tries[1:]], {"window_cache": cap})
    cache = old["encode"]["cache"]
    assert 1 <= cache["kept"] < old["windows"] and cache["kept"] + cache["over_cap"] == old["windows"]
    assert cache["peak_gib"] <= cache["cap_gib"]
    assert steps[1]["encode"]["cache_hits"] == cache["kept"] and steps[1]["encode"]["encoded"] == cache["over_cap"]
    ref = []
    plain = make_trainer(monkeypatch, ref)
    _, plain_steps = one_update(plain, [tries[:1], tries[1:]], {})
    assert log == ref and [s["grad_digest"] for s in steps] == [s["grad_digest"] for s in plain_steps]


def test_rows_cache_and_pack_round_trip(tries):
    spec = {**tries[1], "whole": False, "end": None}
    fresh = rl1_encode.read_rows(spec)
    rl1_encode.ROWS.clear()
    first, again = rl1_encode.read_rows(spec, cached=True), rl1_encode.read_rows({**spec, "w": 2.0}, cached=True)
    assert bits(first) == bits(fresh) and [r["w"] for r in again] == [2.0] * len(fresh)
    assert bits([{k: v for k, v in r.items() if k != "w"} for r in again]) == bits([{k: v for k, v in r.items() if k != "w"} for r in fresh])
    made, _ = rl1_encode.encode_window(stub_encoded, THINK_END, 600, spec, 0, device="cpu")
    import pickle

    assert bits(rl1_encode.unpack(pickle.loads(pickle.dumps(rl1_encode.pack(made))))) == bits(made)


def test_one_step_token_update_then_k3_measure(monkeypatch, tries):
    """1-step default, as rl_loop.dosed_update sends it: one token-level step on every try (applied)
    keeping the subsample's pi_old (keep_old), then the forward-only k3_score on the subsample (no gradient)."""
    log = []
    t = make_trainer(monkeypatch, log)
    t.args.tis_cap = 2.0
    common = {"share_prefix": True, "prefix_chunk": 16384, "signed": True}
    sub = [tries[0], tries[3]]
    step = t.op_update({"attempts": tries, "lr": 1.0, "max_windows": 100, **common, "save": "adapter",
                        "keep_old": [s["name"] for s in sub]})
    kept = len(t.old)
    n = len(log)
    measure = t.op_k3_score({"attempts": sub, **common, "gspo_clip": [0.015, 0.019]})
    assert step["ok"] and step.get("applied", True) and t.updates == 1 and kept > 0
    assert not any(e["old"] for e in log[:n] if e["grad"])  # the token step ignores pi_old
    assert measure["ok"] and measure["windows"] == kept and measure["approx_kl"] > 0 and t.updates == 1
    assert not any(e["grad"] for e in log[n:]) and t.old == {}  # forward only; kept pi_old dropped

def served_store(tmp_path, specs):
    """R3 records for each spec's window: the end request's own record covers the window's tokens (start 0), routes a
    fixed function of position and layer; rows get their request ids (``rid``)."""
    import numpy as np

    d = tmp_path / "served"
    d.mkdir()
    (d / "layers.json").write_text(json.dumps({"layers": [0, 1, 2], "k": 2}))
    for i, spec in enumerate(specs):
        made, rows = rl1_encode.encode_window(stub_encoded, THINK_END, 600, spec, 0, device="cpu")
        end = made["replies"][-1]["request"]
        lines = [json.loads(line) for line in open(spec["rollout"])]
        for row in lines:
            row["rid"] = f"chatcmpl-t{i}n{row['n']}"
        Path(spec["rollout"]).write_text("".join(json.dumps(r) + "\n" for r in lines))
        target = made["inputs"]["input_ids"][0].numpy().astype(np.int32)
        pos = np.arange(len(target))[:, None, None]
        ids = ((pos * 7 + np.arange(3)[None, :, None] * 3 + np.arange(2)[None, None, :] + i) % 512).astype(np.int16)
        with open(d / f"chatcmpl-t{i}n{end}-abcdefgh.rt", "wb") as f:
            f.write(np.array([0, len(target), 3, 2], dtype=np.int32).tobytes() + target.tobytes() + ids.tobytes())
    return str(d)


def test_r3_routes_resolved_in_encoder_processes_equal_the_trainer_side(monkeypatch, tries, tmp_path):
    """MFU-1 m1h1 fix: with R3 on and encoder processes, each window's served-route table is resolved in the encoder
    process (no rollout re-read, no GIL work in the feed thread) and reaches the pass bit for bit as the trainer-side
    resolve gives it; router off strips the host table."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "arena"))  # served_route, as rl1/ has it
    specs = [{**s, "whole": False, "end": None} for s in tries]
    route = served_store(tmp_path, specs)
    specs = [{**s, "served_route": route} for s in specs]
    rl1_encode.ROWS.clear()
    monkeypatch.setitem(rl1_trainer.ROUTE, "layers", [0, 1, 2])
    monkeypatch.setattr(rl1_trainer, "route_columns", lambda model, layers: rl1_trainer.ROUTE.update(layers=list(layers)))
    calls = []
    real = rl1_trainer.Trainer.served_routing
    monkeypatch.setattr(rl1_trainer.Trainer, "served_routing", lambda self, *a: calls.append(1) or real(self, *a))
    common = {"share_prefix": True, "prefix_chunk": 16384, "signed": True}
    runs = {}
    for router in ("served", "off"):
        monkeypatch.setitem(rl1_trainer.ROUTE, "router", router)
        for label, fast in (("thread", {}), ("procs", {"encode_procs": 2, "prefetch": 2})):
            log, seen = [], []
            t = make_trainer(monkeypatch, log)
            inner = t.window_pass
            t.window_pass = lambda made, **kw: seen.append(bits({k: made.get(k) for k in ("served", "served_audit")}))\
                or inner(made, **kw)
            rl1_trainer.SERVED.clear()
            rl1_trainer.SERVED_AUDIT.clear()
            n = len(calls)
            try:
                step = t.op_update({"attempts": specs, "lr": 1.0, "max_windows": 100, "digest": True, **common, **fast})
            finally:
                if t.encoders is not None:
                    t.encoders[1].shutdown()
            runs[router, label] = (log, seen, step["grad_digest"], dict(rl1_trainer.SERVED_AUDIT), len(calls) - n)
    log, seen, digest, audit, trainer_calls = runs["served", "thread"]
    assert trainer_calls == len(specs) and audit["windows"] == len(specs) and "no_record" not in audit
    assert all(s["served"] is not None and s["served"][0] == "t" and s["served"][1] == "torch.int16" for s in seen)
    p_log, p_seen, p_digest, p_audit, p_calls = runs["served", "procs"]
    assert p_calls == 0  # every table came from an encoder process
    assert p_seen == seen and p_log == log and p_digest == digest and p_audit == audit
    for label in ("thread", "procs"):
        o_log, o_seen, _, o_audit, o_calls = runs["off", label]
        assert o_calls == 0 and o_audit == {} and all(s["served"] is None for s in o_seen)
        assert o_log == runs["off", "thread"][0]  # no host table leaks into the made dict
