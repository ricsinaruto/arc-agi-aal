"""Sliding-window cache measurement: the probe overlay, the selection edit and the trial flow on a mock server."""

import hashlib
import json
import math
import random
import time

import httpx
import numpy as np
import pytest

from arena import serving, swkv_kl, swkv_probe_patch, swkv_probe_runtime


def test_overlay_appends_the_install_and_accepts_only_itself(tmp_path, monkeypatch):
    site, target = tmp_path / "site", tmp_path / "site" / swkv_probe_patch.RELATIVE
    target.parent.mkdir(parents=True)
    target.write_text("class QSAKeyStateCache:\n    pass\n")
    monkeypatch.setattr(swkv_probe_patch, "SHA256", hashlib.sha256(target.read_bytes()).hexdigest())
    (site / "vllm").mkdir(exist_ok=True)
    identity = swkv_probe_patch.apply(site, tmp_path / "out")
    assert target.read_text().endswith("_arc_swkv_probe.install(QSAKeyStateCache)\n")
    assert (site / swkv_probe_patch.HELPER).read_bytes() == (
        swkv_probe_patch.Path(swkv_probe_patch.__file__).with_name("swkv_probe_runtime.py").read_bytes())
    assert swkv_probe_patch.apply(site, tmp_path / "out") == identity  # a restart in place
    target.write_text(target.read_text() + "# edited\n")
    with pytest.raises(RuntimeError):
        swkv_probe_patch.apply(site, tmp_path / "out")


def test_selection_edit_needs_both_anchors_once():
    source = ("def qsa_select_paged_tokens(out):\n    for row_slice in ():\n"
              + swkv_probe_runtime.MASK_ANCHOR + "        expand(\n" + swkv_probe_runtime.CHECK_ANCHOR
              + "    return out\n")
    edited = swkv_probe_runtime.patched_select_source(source)
    assert edited.index(swkv_probe_runtime.MASK_INSERT) < edited.index(swkv_probe_runtime.MASK_ANCHOR)
    assert edited.index(swkv_probe_runtime.CHECK_ANCHOR) < edited.index(swkv_probe_runtime.CHECK_INSERT)
    compile(edited, "edited", "exec")
    with pytest.raises(RuntimeError):
        swkv_probe_runtime.patched_select_source(source + swkv_probe_runtime.MASK_ANCHOR)


def test_lab_switch_needs_an_eager_server():
    assert serving.lab_overrides({"ARENA_SWKV_PROBE": "1"}) == {"swkv_probe": True}
    with pytest.raises(ValueError):
        serving.lab_overrides({"ARENA_SWKV_PROBE": "yes"})
    with pytest.raises(ValueError, match="enforce_eager"):
        serving.configure(None, None, swkv_probe=True)


def _mock_server(log):
    state = {"cut": None, "shift": 0, "count_only": False, "cached": {}}

    def logprobs(ids, k, noise):
        out = []
        for position, token in enumerate(ids):
            rng = random.Random(hash((tuple(ids[max(0, position - 3):position]), position)) % 10**9)
            logits = [rng.gauss(0, 2) + random.Random(position * 31 + int(noise * 1000)).gauss(0, noise)
                      for _ in range(30)]
            z = math.log(sum(math.exp(v) for v in logits) + 50)
            ranked = sorted(((tok, v - z) for tok, v in enumerate(logits)), key=lambda kv: -kv[1])
            entry = {f"token_id:{tok}": {"logprob": v} for tok, v in ranked[:k]}
            entry.setdefault(f"token_id:{token}", {"logprob": dict(ranked).get(token, -12.0)})
            out.append(entry)
        return out

    def handler(request):
        body = json.loads(request.content)
        if request.url.path.endswith("/collective_rpc"):
            assert body["method"] == "arc_swkv_probe"
            cmd = json.loads(body["args"][0])
            before = [5, 10, 0 if not state["count_only"] else 4] if state["cut"] is not None else None
            if cmd["op"] == "set":
                state.update(cut=cmd.get("cut"), shift=cmd.get("shift", 0), count_only=bool(cmd.get("count_only")))
            elif cmd["op"] == "clear":
                state.update(cut=None, shift=0, count_only=False)
            log.append((cmd["op"], cmd.get("cut"), cmd.get("shift", 0)))
            return httpx.Response(200, json={"results": [{"ok": True, "op": cmd["op"], "before": before,
                                                          "select_calls": 0, "forward_calls": 0}]})
        ids, salt = body["prompt"], body["cache_salt"]
        if "prompt_logprobs" not in body:  # a primer
            state["cached"][salt] = len(ids)
            return httpx.Response(200, json={"choices": [{}], "usage": {}})
        cached = state["cached"].pop(salt)
        assert body["vllm_xargs"] == {"arc_read_prefix_cache": 1} and body["prompt_logprobs"] == 20
        masked = state["cut"] is not None and not state["count_only"]
        noise = 0.4 * masked + 0.2 * bool(state["shift"]) + (0.3 if len(ids) < 150 else 0.0)
        entries = logprobs(ids, 20, noise)
        return httpx.Response(200, json={"choices": [{"prompt_logprobs": [None] + entries[cached + 1:]}],
                                         "usage": {"prompt_tokens_details": {"cached_tokens": cached}}})

    return httpx.MockTransport(handler)


async def test_trial_runs_the_passes_against_a_mock_server(tmp_path, monkeypatch):
    async def idle(*args, **kwargs):
        return {}

    monkeypatch.setattr(swkv_kl, "isolate_trial", idle)
    rng = random.Random(2)
    contexts = []
    for index in range(3):
        system = [rng.randrange(30) for _ in range(8)]
        dropped = [rng.randrange(30) for _ in range(60)]
        kept = [rng.randrange(30) for _ in range(50)]
        note = [rng.randrange(30) for _ in range(6)]
        tail = [rng.randrange(30) for _ in range(40)]
        contexts.append({
            "index": index, "run": "r", "game": f"k{index}", "family": "f", "meta": {},
            "seqs": {"A0": system + note + kept + tail, "A": system + kept + note + tail,
                     "C": system + dropped + kept + note + tail},
            "primer": {"A0": 64, "A": 58, "C": 118}, "start": {"A0": 64, "A": 64, "C": 124},
            "spans": [[2, 20], [25, 40]],
            "masks": {"ragged": {"cut": 118, "ranges": [[8, 64]]}, "aligned": {"cut": 118, "ranges": [[8, 68]]}}})
    path = tmp_path / "src" / swkv_kl.INPUTS
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(contexts))
    calls = []
    result = await swkv_kl.run_swkv_kl({"served_model": "m"}, {"name": "swkv"}, source=tmp_path / "src",
                                       output=tmp_path / "out", server_dir=None, base_url="http://x/v1",
                                       deadline_epoch=time.time() + 600, transport=_mock_server(calls),
                                       log=lambda *a, **k: None)
    assert result["status"] == "complete" and not result["skipped"], result["errors"]
    assert len(result["arms"]) == 3 * len(swkv_kl.ARMS)
    pairs = result["summary"]["pairs"]
    assert pairs["A|A2"]["mean_kl"] == pytest.approx(0.0) and pairs["A|A2"]["positions"] == 3 * 33
    assert pairs["C|C2"]["mean_kl"] == pytest.approx(0.0)
    assert pairs["C|Br"]["mean_kl"] > 0 and pairs["A|S500"]["mean_kl"] > 0 and pairs["C|A"]["mean_kl"] > 0
    assert result["summary"]["decisions"]["KL(C||Br) - KL(C||A)"]["contexts"] == 3
    assert ("set", 118, 0) in calls and ("set", None, 900_000) in calls and calls[-1][0] == "clear"
    counted = [a for a in result["arms"] if a["arm"] == "C"][0]
    assert counted["counters"] == [5, 10, 4] and counted["cached"] == 118
    stored = np.load(tmp_path / "out/arrays-0.npz")
    assert stored["Br_ids"].shape == (39, 20) and stored["A_actual"].shape == (39,)
    assert json.loads((tmp_path / "out/swkv-kl.json").read_text())["status"] == "complete"


async def test_the_deadline_skips_arms_and_keeps_what_ran(tmp_path, monkeypatch):
    async def idle(*args, **kwargs):
        return {}

    monkeypatch.setattr(swkv_kl, "isolate_trial", idle)
    ids = list(range(20)) * 4
    context = {"index": 0, "run": "r", "game": "k", "family": "f", "meta": {}, "seqs": {"A": ids, "A0": ids, "C": ids},
               "primer": {"A": 40, "A0": 40, "C": 40}, "start": {"A": 44, "A0": 44, "C": 44}, "spans": [[2, 30]],
               "masks": {"ragged": {"cut": 40, "ranges": [[8, 16]]}, "aligned": {"cut": 40, "ranges": [[8, 20]]}}}
    path = tmp_path / "src" / swkv_kl.INPUTS
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([context]))
    result = await swkv_kl.run_swkv_kl({"served_model": "m"}, {"name": "swkv"}, source=tmp_path / "src",
                                       output=tmp_path / "out", server_dir=None, base_url="http://x/v1",
                                       deadline_epoch=time.time() + 5, transport=_mock_server([]),
                                       log=lambda *a, **k: None)
    assert not result["arms"] and len(result["skipped"]) == len(swkv_kl.ARMS) and result["status"] == "error"
