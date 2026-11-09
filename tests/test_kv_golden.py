"""arena.kv_golden: probes from recorded tries, dumps on fake servers, the boot verdict."""
import asyncio
import json


from arena import kv_golden


def rollout(tmp_path, name, starts):
    rows, d = [], tmp_path / name
    for n, start in enumerate(starts):
        rows.append({"n": n, "keep": n * 2, "messages": [{"role": "user", "content": f"{name}{n}"}],
                     "template": {"enable_thinking": True}, "tools": [], "gameplay": True, "tokens": [10 + n, 20 + n],
                     "logprobs": [-0.1, -0.2], "usage": {"prompt_tokens": start}})
    d.mkdir()
    (d / "rollout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def entries_for(probe, shift=0.0):
    out = [None] * (probe["start"] + len(probe["tokens"]))
    for i, t in enumerate(probe["tokens"]):
        out[probe["start"] + i] = {str(t): {"logprob": -0.5 - 0.1 * i + shift}}
    return out


def fake_dump(probes, shift=0.0, per=None):
    rows = []
    for p in probes:
        lp = [-0.5 - 0.1 * i + (per or {}).get(p["label"], shift) for i in range(len(p["tokens"]))]
        rows.append({"label": p["label"], "depth": p["depth"], "start": p["start"], "n": len(p["tokens"]),
                     "logprobs": lp, "repeat": lp})
    return {"probes": rows}


def test_build_picks_one_request_per_try_per_depth(tmp_path):
    a, b = rollout(tmp_path, "a", [100, 900, 5000, 7000]), rollout(tmp_path, "b", [120, 4000, 9000])
    probes = kv_golden.build([a, b], [1000, 6000], per_depth=2)
    got = [(p["label"], p["depth"], p["start"]) for p in probes]
    assert got == [("a-r1", 1000, 900), ("b-r0", 1000, 120), ("a-r2", 6000, 5000), ("b-r1", 6000, 4000)]
    assert probes[0]["body"]["messages"][-1]["content"] == "a2"  # request k+1 holds reply k
    assert probes[0]["body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_dump_scores_the_span(tmp_path):
    probes = kv_golden.build([rollout(tmp_path, "a", [5, 9, 30])], [10, 40])
    by_label = {p["label"]: p for p in probes}
    seen = []

    async def post(body):
        seen.append(body)
        assert body["prompt_logprobs"] == 0 and body["max_tokens"] == 1 and body["model"] == "base"
        label = next(lbl for lbl, p in by_label.items() if p["body"]["messages"] == body["messages"])
        return {"choices": [{"prompt_logprobs": entries_for(by_label[label])}]}

    got = asyncio.run(kv_golden.dump(post, "base", probes))
    assert len(seen) == 2 and [p["logprobs"] for p in got["probes"]] == [[-0.5, -0.6]] * 2


def noisy_dump(probes, seed, scale=0.2, spike=None):
    """|dlogp| noise shaped like the B300 server floor (heavy-tailed: most tokens ~0.05, a few > 1)."""
    import random

    rng = random.Random(seed)
    out = fake_dump(probes)
    for i, p in enumerate(out["probes"]):
        p["logprobs"] = [v + rng.expovariate(1 / scale) * rng.choice((-1, 1)) for v in p["logprobs"]]
        if spike is not None and i == 0:
            p["logprobs"][0] += spike  # one token of the pool: H3's 6.93
    return out


def test_boot_verdict_bars_p99_and_mean_vs_floor(tmp_path):
    """lora-l1 (26 Oct): p99 <= 1.5 x floor p99, mean <= 1.1 x floor mean; max reported only (H3 failed E4 on one
    token of 512 while its p99 and mean sat on the floor)."""
    probes = [{"label": f"p{i}", "depth": 8000, "start": 100, "tokens": list(range(64))} for i in range(8)]  # 512
    ref, floor = noisy_dump(probes, 1), noisy_dump(probes, 2)
    spike = kv_golden.boot_verdict({"server1": noisy_dump(probes, 3, spike=6.9), "server2": ref}, "server2", floor)
    row = spike["servers"]["server1"]
    assert spike["ok"] and row["max_abs"] > 3 * spike["floor"]["max_abs"]  # one token: max alone no longer fails it
    assert set(spike["floor"]) >= {"p99_abs", "mean_abs", "max_abs"} and spike["bars"] == kv_golden.BOOT_BARS
    worse = kv_golden.boot_verdict({"server1": noisy_dump(probes, 3, scale=0.26), "server2": ref}, "server2", floor)
    assert not worse["ok"] and worse["servers"]["server1"]["mean_abs"] > 1.1 * worse["floor"]["mean_abs"]
    shifted = fake_dump(probes, shift=0.0)
    for p in shifted["probes"]:  # a LoRA error on a few % of tokens: mean barely moves, p99 does
        p["logprobs"] = [v + (3.0 if j % 25 == 0 else 0.0) for j, v in enumerate(p["logprobs"])]
    tail = kv_golden.boot_verdict({"server1": shifted, "server2": fake_dump(probes)}, "server2",
                                  noisy_dump(probes, 4, scale=0.05))
    assert not tail["ok"]


def test_boot_verdict_bars_at_one_and_a_half_floor(tmp_path):
    probes = kv_golden.build([rollout(tmp_path, "a", [5, 9, 30]), rollout(tmp_path, "b", [7, 20])], [10, 40], 2)
    ref, floor = fake_dump(probes), fake_dump(probes, shift=0.02)  # card 1 alone vs all at once: 0.02
    ok = kv_golden.boot_verdict({"server1": fake_dump(probes, 0.022), "server2": ref}, "server2", floor)
    assert ok["ok"] and ok["servers"]["server1"]["ok"] and abs(ok["floor"]["max_abs"] - 0.02) < 1e-9
    worse = kv_golden.boot_verdict({"server1": fake_dump(probes, 0.023), "server2": ref}, "server2", floor)
    assert not worse["ok"]  # 0.023 > 1.1 x 0.02 (mean)
    short = fake_dump(probes)
    short["probes"] = short["probes"][1:]
    assert not kv_golden.boot_verdict({"server1": short, "server2": ref}, "server2", floor)["ok"]
    assert not kv_golden.boot_verdict({"server1": fake_dump(probes, 0.001), "server2": ref}, "server2", ref)["ok"]
    exact = kv_golden.boot_verdict({"server1": ref, "server2": ref}, "server2", floor)
    assert exact["ok"] and exact["servers"]["server1"]["max_abs"] == 0


class FakeCard:
    """A play server answering chat prompt_logprobs; ``shift`` moves its values, ``together`` its concurrent ones."""

    def __init__(self, probes, shift=0.0, fail=False):
        self.probes, self.shift, self.fail, self.inflight, self.peak, self.model = probes, shift, fail, 0, 0, "base"

    async def post(self, route, body, timeout=900):
        assert route == "/v1/chat/completions" and body["prompt_logprobs"] == 0 and body["model"] == "policy"
        if self.fail:
            raise RuntimeError("boom")
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        await asyncio.sleep(0.001)
        p = next(p for p in self.probes if p["body"]["messages"] == body["messages"])
        out = entries_for(p, self.shift + (0.01 if self.inflight > 1 else 0.0))
        self.inflight -= 1
        return {"prompt_logprobs": out}


def test_boot_dump_reference_floor_and_errors(tmp_path):
    probes = kv_golden.build([rollout(tmp_path, "a", [5, 9, 30, 50]), rollout(tmp_path, "b", [7, 20])], [10, 40], 2)
    path = tmp_path / "probes.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in probes))
    cards = [FakeCard(probes, 0.007), FakeCard(probes)]  # floor mean 0.0067 (some together passes overlap)
    rows = []
    got = asyncio.run(kv_golden.boot_dump(cards, {"probes": str(path)}, tmp_path / "out", model="policy",
                                          log=rows.append))
    assert got["ref"] == "server2" and cards[1].peak == len(probes) and cards[0].peak == 1  # floor: all at once
    assert abs(got["floor"]["max_abs"] - 0.01) < 1e-9 and got["servers"]["server1"]["ok"] and got["ok"]
    assert (tmp_path / "out/server2-together.json").is_file() and rows[0] is got
    bad = asyncio.run(kv_golden.boot_dump([FakeCard(probes, fail=True), FakeCard(probes)], {"probes": str(path)},
                                          tmp_path / "out2", model="policy"))
    assert not bad["ok"] and "boom" in bad["errors"]["server1"]
    one = asyncio.run(kv_golden.boot_dump([FakeCard(probes)], {"probes": str(path)}, tmp_path / "out3",
                                          model="policy"))
    assert one["ref"] == "server1" and one["servers"] == {} and one["ok"]  # one card: the floor only


class AdapterCard(FakeCard):
    """FakeCard serving LoRA adapters by name: the zero-delta play adapter scores like the base, "kv-golden" adds
    ``lora`` (a bad LoRA path adds ``lora`` x ``error``)."""

    def __init__(self, probes, log, name, error=1.0):
        super().__init__(probes)
        self.log, self.tag, self.error, self.adapters = log, name, error, set()

    async def load(self, name, path):
        self.log.append(f"{self.tag}:load:{name}:{path}")
        self.adapters.add(name)

    async def unload(self, name):
        self.log.append(f"{self.tag}:unload:{name}")
        self.adapters.discard(name)

    async def post(self, route, body, timeout=900):
        assert body["model"] in self.adapters | {"policy"}
        self.shift = 0.3 * self.error if body["model"] == "kv-golden" else 0.0
        return await FakeCard.post(self, route, {**body, "model": "policy"}, timeout)


def test_boot_dump_scores_a_trained_adapter_when_given(tmp_path):
    """lora-l1 (26 Oct): the play adapter at boot is zero-delta, so E4 vs vLLM's LoRA path differed by nothing in
    H3/H4. With spec adapter every server loads it as "kv-golden", the dump scores under it, and it is unloaded after;
    a LoRA path error shows."""
    probes = kv_golden.build([rollout(tmp_path, "a", [5, 9, 30, 50]), rollout(tmp_path, "b", [7, 20])], [10, 40], 2)
    path = tmp_path / "probes.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in probes))
    log = []
    good = [AdapterCard(probes, log, "c0"), AdapterCard(probes, log, "c1")]
    got = asyncio.run(kv_golden.boot_dump(good, {"probes": str(path), "adapter": "/a/s0017"}, tmp_path / "o1",
                                          model="policy"))
    assert got["model"] == "kv-golden" and got["adapter"] == "/a/s0017" and got["ok"]
    assert log[:2] == ["c0:load:kv-golden:/a/s0017", "c1:load:kv-golden:/a/s0017"]
    assert log[-2:] == ["c0:unload:kv-golden", "c1:unload:kv-golden"] and not good[0].adapters
    bad = [AdapterCard(probes, [], "c0", error=1.2), AdapterCard(probes, [], "c1")]
    got = asyncio.run(kv_golden.boot_dump(bad, {"probes": str(path), "adapter": "/a/s0017"}, tmp_path / "o2",
                                          model="policy"))
    assert not got["ok"] and got["servers"]["server1"]["mean_abs"] > 0.05
    zero = asyncio.run(kv_golden.boot_dump([AdapterCard(probes, [], "c0", error=1.2), AdapterCard(probes, [], "c1")],
                                           {"probes": str(path)}, tmp_path / "o3", model="policy"))
    assert zero["ok"] and zero["adapter"] is None  # the zero-delta play adapter cannot see the LoRA error
