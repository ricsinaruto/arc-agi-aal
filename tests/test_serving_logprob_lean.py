"""serving_fast.logprob_lean: on recorded real-v5 replies, everything our client and trainer read (rollout_record.
parse_logprobs: sampled ids, their logprobs, the alternatives and their order) equals the pinned pydantic path's,
which runs here verbatim (logprob_fast's model_construct version)."""
import json
import sys
import types
from pathlib import Path

import pytest

pydantic = pytest.importorskip("pydantic")
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from arena.rollout_record import parse_logprobs  # noqa: E402

REPLIES = Path(__file__).parent / "golden/data/logprob_lean_replies.json"
PINNED = Path("~/arc-shared/pinned-vllm")


class _Response:  # starlette.responses.Response stand-in (not installed in the test env)
    def __init__(self, content=None, media_type=None, headers=None):
        self.body, self.media_type, self.headers = content, media_type, headers


sys.modules.setdefault("starlette", types.ModuleType("starlette"))
_responses = types.ModuleType("starlette.responses")
_responses.Response = _Response
sys.modules["starlette.responses"] = _responses

from arc_infra import logprob_lean_runtime as L  # noqa: E402


class Base(BaseModel):  # vllm OpenAIBaseModel
    model_config = ConfigDict(extra="allow")


class ChatCompletionLogProb(Base):
    token: str
    logprob: float = -9999.0
    bytes: list[int] | None = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    top_logprobs: list[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(Base):
    content: list[ChatCompletionLogProbsContent] | None = None


class Choice(Base):
    index: int
    logprobs: ChatCompletionLogProbs | None = None
    finish_reason: str | None = "stop"


class Reply(Base):
    id: str = "chatcmpl-1"
    choices: list[Choice]


class Logprob:
    def __init__(self, logprob, rank=None, decoded_token=None):
        self.logprob, self.rank, self.decoded_token = logprob, rank, decoded_token


def placeholder(token_id):
    return f"token_id:{token_id}"


class Pinned:
    """The pinned _create_chat_logprobs / _get_top_logprobs (vllm chat serving, sha a2440b6b), logprob_fast applied."""
    return_tokens_as_token_ids = False

    def _get_decoded_token(self, logprob, token_id, tokenizer, return_as_token_id=False):
        if return_as_token_id:
            return placeholder(token_id)
        return logprob.decoded_token

    def _get_top_logprobs(self, logprobs, top_logprobs, tokenizer, should_return_as_token_id, return_all=False):
        return [
            ChatCompletionLogProb.model_construct(
                token=(token := self._get_decoded_token(p[1], p[0], tokenizer,
                                                        return_as_token_id=should_return_as_token_id)),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")),
            )
            for i, p in enumerate(logprobs.items())
            if return_all or top_logprobs == -1 or (top_logprobs is not None and i < top_logprobs)
        ]

    def _create_chat_logprobs(self, token_ids, top_logprobs, tokenizer, num_output_top_logprobs=None,
                              logprob_token_ids=None, return_as_token_id=None):
        logprobs_content = []
        should = return_as_token_id if return_as_token_id is not None else self.return_tokens_as_token_ids
        for i, token_id in enumerate(token_ids):
            step = top_logprobs[i]
            if step is None or step.get(token_id) is None:
                token = placeholder(token_id)
                logprobs_content.append(ChatCompletionLogProbsContent(
                    token=token, bytes=list(token.encode("utf-8", errors="replace"))))
            else:
                st = step[token_id]
                logprobs_content.append(ChatCompletionLogProbsContent(
                    token=self._get_decoded_token(st, token_id, tokenizer, should),
                    logprob=max(st.logprob, -9999.0),
                    bytes=None if st.decoded_token is None else list(st.decoded_token.encode("utf-8", errors="replace")),
                    top_logprobs=self._get_top_logprobs(step, num_output_top_logprobs, tokenizer, should,
                                                        return_all=bool(logprob_token_ids))))
        return ChatCompletionLogProbs(content=logprobs_content)


def steps_of(reply):
    """Engine-side per-position dicts as vLLM builds them: the sampled token first, then the ranked top-k."""
    out = []
    for tid, value, (ids, values) in zip(reply["tokens"], reply["logprobs"], reply["top"]):
        step = {tid: Logprob(value, decoded_token="x")}
        for t, v in zip(ids, values):
            step.setdefault(t, Logprob(v, decoded_token=None))
        out.append(step)
    return out


def both(token_ids, steps, n=20, return_all=False):
    pinned = Pinned()._create_chat_logprobs(token_ids, steps, None, n, [1] if return_all else None, True)
    old = json.loads(json.dumps(Reply(choices=[Choice(index=0, logprobs=pinned)]).model_dump(),
                                ensure_ascii=False, separators=(",", ":")))
    lean = L.lean_logprobs(ChatCompletionLogProbs, token_ids, steps, n, return_all)
    resp = L.lean_response(Reply(choices=[Choice(index=0, logprobs=lean)]), headers={"h": "1"})
    assert resp is not None and resp.media_type == "application/json" and resp.headers == {"h": "1"}
    return old, json.loads(resp.body)


def test_recorded_replies_read_the_same():
    replies = json.loads(REPLIES.read_text())
    assert len(replies) >= 3
    for reply in replies:
        old, new = both(reply["tokens"], steps_of(reply))
        a = old["choices"][0]["logprobs"]["content"]
        b = new["choices"][0]["logprobs"]["content"]
        assert parse_logprobs(a) == parse_logprobs(b)
        assert [(e["token"], e["logprob"], [(t["token"], t["logprob"]) for t in e["top_logprobs"]]) for e in a] == \
               [(e["token"], e["logprob"], [(t["token"], t["logprob"]) for t in e["top_logprobs"]]) for e in b]
        assert all("bytes" not in e and all("bytes" not in t for t in e["top_logprobs"]) for e in b)
        old["choices"][0]["logprobs"], new["choices"][0]["logprobs"] = None, None
        assert old == new  # the rest of the reply is unchanged
    assert not L.RAW


@pytest.mark.parametrize("n,return_all", [(None, False), (0, False), (3, False), (-1, False), (2, True)])
def test_edge_cases(n, return_all):
    steps = [None, {7: Logprob(-0.1)}, {5: Logprob(-0.2), 9: Logprob(-float("inf"))},
             {1: Logprob(-1.0), 2: Logprob(-2.0), 3: Logprob(-3.0)}]
    old, new = both([4, 8, 5, 1], steps, n, return_all)  # 4: no step; 8: not in its step; -inf clamps to -9999
    a, b = old["choices"][0]["logprobs"]["content"], new["choices"][0]["logprobs"]["content"]
    strip = lambda c: [{k: v for k, v in e.items() if k != "bytes"} | {"top_logprobs": [  # noqa: E731
        {k: v for k, v in t.items() if k != "bytes"} for t in e["top_logprobs"]]} for e in c]
    assert strip(a) == strip(b)


def test_unmarked_reply_keeps_the_pinned_path():
    assert L.lean_response(Reply(choices=[Choice(index=0, logprobs=None)])) is None
    stray = ChatCompletionLogProbs.model_construct(content=None)
    assert L.lean_response(Reply(choices=[Choice(index=0, logprobs=stray)])) is None


def test_raw_is_bounded():
    for _ in range(L.LIMIT + 5):
        L.lean_logprobs(ChatCompletionLogProbs, [1], [{1: Logprob(-0.5)}], 20, False)
    assert len(L.RAW) == L.LIMIT
    L.RAW.clear()


def test_switch_needs_logprob_fast():
    from arc_infra import serving_fast_patch as F

    with pytest.raises(ValueError):
        F.validate({"logprob_fast": False, "image_cache": False, "logprob_lean": True})
    on = F.validate({"logprob_fast": True, "image_cache": False, "logprob_lean": True})
    pins = [p["relative"] for p, _, _ in F.plan(on)]
    assert F.ROUTER["relative"] in pins and pins.count(F.CHAT["relative"]) == 1
    assert "logprob_lean" not in F.validate({"logprob_fast": True, "image_cache": False})


@pytest.mark.skipif(not PINNED.is_dir(), reason="pinned vLLM source not on this machine (the DB dry boot applies it)")
def test_applies_to_the_pinned_files(tmp_path):
    import shutil

    from arc_infra import serving_fast_patch as F

    site = tmp_path / "site"
    for pin in (F.LOGPROBS, F.CHAT, F.ROUTER):
        (site / pin["relative"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PINNED / pin["relative"], site / pin["relative"])
    spec = {"logprob_fast": True, "image_cache": False, "logprob_lean": True}
    ident = F.apply(site, tmp_path / "out", spec)
    assert ident["lean_helper_sha256"] and (site / F.LEAN_HELPER).read_bytes() == \
        Path(L.__file__).read_bytes()
    assert F.apply(site, tmp_path / "out", spec) == ident  # a restart accepts its own overlay
