"""MTP exactness screen on CPU: the residual has mean 0 at every draft offset under exact rejection sampling and
moves (or leaves the support) under the classic truncation bugs; chunk labels and the vLLM top-k/top-p rule."""
import math

import pytest

from arc2 import mtp_screen as m


def test_play_distribution_matches_vllm_rule():
    top = [(i, math.log(p)) for i, p in enumerate([0.5, 0.3, 0.15, 0.04, 0.01])]
    q = m.play_distribution(top, temperature=1.0, k=20, p=0.95)
    # ascending cumulative 0.01, 0.05 <= 0.05 dropped; 0.20 kept onward
    assert set(q) == {0, 1, 2} and abs(sum(math.exp(v) for v in q.values()) - 1) < 1e-12
    assert set(m.play_distribution(top, temperature=1.0, k=2, p=1.0)) == {0, 1}
    assert set(m.play_distribution([(7, -5.0)], temperature=0.7)) == {7}  # the largest is always kept


def test_labels_offsets():
    chunks = [[1], [2, 3, 4, 5], [6, 7], [8], [9, 10]]
    labels = m.label_chunks(chunks, drafts=3, finished_by_length=False)
    assert labels == [(0, "first"), (1, "accepted"), (2, "accepted"), (3, "accepted"), (4, "bonus"),
                      (1, "accepted"), (2, "resampled"), (1, "resampled"), (1, "end"), (2, "end")]


@pytest.mark.parametrize("drafts", [0, 1, 3])
def test_exact_sampling_has_zero_mean_residual_at_every_offset(drafts):
    out = m.summarize(m.simulate(drafts, steps=30000, seed=drafts))
    for key, g in out.items():
        if key.startswith("offset") or key == "all":
            assert g["outside"] == 0 and abs(g["mean_r"]) < 4 * g["se"] + 1e-4, (key, g)


@pytest.mark.parametrize("bug", ["untruncated", "accept_untruncated"])
def test_truncation_bugs_are_caught(bug):
    out = m.summarize(m.simulate(3, steps=30000, seed=11, bug=bug))
    flagged = [k for k, g in out.items() if k.startswith("offset") and (
        g["outside"] > 0 or abs(g["mean_r"]) > 4 * g["se"])]
    assert flagged, out


def test_screen_against_a_scripted_server(tmp_path):
    """The whole client path on a mock server streaming exact samples in vLLM's completions format (3 drafts)."""
    import asyncio
    import json
    import random
    import time

    import httpx

    rng = random.Random(3)
    vocab = 30

    def step_tokens():
        top = [(t, rng.gauss(0, 2.5)) for t in range(vocab)]
        z = math.log(sum(math.exp(v) for _, v in top))
        raw = sorted([(t, v - z) for t, v in top], key=lambda x: -x[1])[:20]
        q = m.play_distribution(raw)
        u, c = rng.random(), 0.0
        for tok, lq in q.items():
            c += math.exp(lq)
            if u <= c:
                break
        return tok, dict(raw)[tok], raw

    def stream():
        sizes = [1] + [rng.choice([1, 2, 3, 4]) for _ in range(40)]
        lines = []
        for k, n in enumerate(sizes):
            toks = [step_tokens() for _ in range(n)]
            body = {"choices": [{"index": 0, "text": "x", "finish_reason": "length" if k == len(sizes) - 1 else None,
                                 "logprobs": {"tokens": [f"token_id:{t}" for t, _, _ in toks],
                                              "token_logprobs": [v for _, v, _ in toks],
                                              "top_logprobs": [{f"token_id:{a}": b for a, b in top} for _, _, top in toks]}}]}
            lines.append("data: " + json.dumps(body) + "\n\n")
        return "".join(lines) + "data: [DONE]\n\n"

    def handler(request):
        path = request.url.path
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "flash"}]})
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2, 3], "count": 3})
        if path == "/metrics":
            return httpx.Response(200, text="vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"} 5\n")
        payload = json.loads(request.content)
        if payload.get("prompt_logprobs"):
            n = len(payload["prompt"])
            rows = [None] * n
            for i in range(3, n):
                _, _, raw = step_tokens()
                rows[i] = {str(t): {"logprob": v} for t, v in raw}
            return httpx.Response(200, json={"choices": [{"prompt_logprobs": rows}]})
        return httpx.Response(200, text=stream(), headers={"content-type": "text/event-stream"})

    real = httpx.AsyncClient

    def client(**kw):
        return real(**{**kw, "transport": httpx.MockTransport(handler)})

    import arc2.mtp_screen as screen
    screen.httpx.AsyncClient = client
    try:
        data = tmp_path / "data/evaluation"
        data.mkdir(parents=True)
        for t in ("a", "b"):
            (data / f"{t}.json").write_text(json.dumps({"train": [{"input": [[1]], "output": [[2]]}],
                                                        "test": [{"input": [[1]], "output": [[2]]}]}))
        report = asyncio.run(screen.run_mtp_screen(
            {"server": {"mtp_tokens": 3}, "arc2": {"data": "data"}},
            {"tasks": ["a", "b"], "split": "evaluation", "rescore": 1, "concurrency": 2},
            source=tmp_path, output=tmp_path / "out", base_url="http://x/v1", deadline_epoch=time.time() + 3600))
    finally:
        screen.httpx.AsyncClient = real
    assert report["status"] == "complete" and not report["errors"], report["errors"]
    assert report["samples"] == 2 and report["summary"]["all"]["outside"] == 0
    assert set(report["accepted_per_position"]["chunks"]) == {0, 1, 2} and report["rescore"]["tokens"] > 0
    assert (tmp_path / "out/mtp-screen.json").is_file()
