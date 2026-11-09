"""serving_fast.logprob_fast on top of prefix_cache.reply_ids in the one chat-serving file (RL play, 26 Oct): the
composed file runs each patch's code exactly as that patch alone does, on the same request."""
import hashlib
import sys
import types

import pytest

from arc_infra import serving_fast_patch as F
from arena import prefix_cache_patch as P

# A stand-in for the pinned chat-serving module holding every anchor both patches edit (their text is checked by the
# patches themselves on the real pinned file at boot).
SOURCE = '''from vllm.utils.serial_utils import numpy2base64


class ChatCompletionLogProb:
    def __init__(self, **kw):
        self.kw, self.how = kw, "validated"

    @classmethod
    def model_construct(cls, **kw):
        made = cls(**kw)
        made.how = "constructed"
        return made


class Serving:
    def render(self, request, tokenizer, result):
        conversation, engine_inputs = result
        return engine_inputs

    def params(self, request, max_tokens):
        if True:
            if request.beam:
                sampling_params = None
            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.default_sampling_params,
                )
        return sampling_params

    def _create_chat_logprobs(self, ids, tokenizer):
        return [
            ChatCompletionLogProb(
                token=(
                    token := self._get_decoded_token(
                        i, tokenizer)
                ),
                logprob=-0.5 * i,
            )
            for i in ids
        ]

    def full(self, final_res, tokenizer):
        choices: list[ChatCompletionResponseChoice] = []
        return choices
'''


@pytest.fixture
def pinned(monkeypatch):
    sha = hashlib.sha256(SOURCE.encode()).hexdigest()
    monkeypatch.setitem(F.CHAT, "sha256", sha)
    monkeypatch.setattr(P, "CHAT_SERVING_SHA256", sha)
    calls = []
    reply = types.ModuleType("arc_reply_ids")
    reply.substitute = lambda inputs, tok, max_len, cap: inputs.update(prompt_token_ids=[9, 9] + inputs["ids"])
    reply.record = lambda res, tok: calls.append(("record", res))
    mods = {"vllm": types.ModuleType("vllm"), "vllm.utils": types.ModuleType("vllm.utils"),
            "vllm.utils.serial_utils": types.ModuleType("vllm.utils.serial_utils"),
            "vllm.entrypoints": types.ModuleType("vllm.entrypoints"),
            "vllm.entrypoints.openai": types.ModuleType("vllm.entrypoints.openai"),
            "vllm.entrypoints.openai.chat_completion": types.ModuleType("vllm.entrypoints.openai.chat_completion")}
    mods["vllm.utils.serial_utils"].numpy2base64 = None
    mods["vllm.entrypoints.openai.chat_completion"].arc_reply_ids = reply
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    return calls


def load(text):
    ns = {"ChatCompletionResponseChoice": object}
    exec(compile(text, "chat_serving.py", "exec"), ns)
    s = ns["Serving"]()
    s.model_config = types.SimpleNamespace(max_model_len=1000)
    s.default_sampling_params = {}
    s.return_tokens_as_token_ids = True
    s._get_decoded_token = lambda i, tok: f"token_id:{i}"
    return s


def request():
    params = types.SimpleNamespace(extra_args=None)
    return types.SimpleNamespace(beam=False, logprobs=True, return_tokens_as_token_ids=True, max_completion_tokens=64,
                                 max_tokens=None, to_sampling_params=lambda m, d: params)


def run(s, calls):
    """Everything both patches touch, on one request: the engine prompt ids, the sampling params' marks, the logprob
    entries (token text, logprob, how built) and the reply record."""
    del calls[:]
    inputs = s.render(request(), None, (None, {"ids": [1, 2, 3]}))
    params = s.params(request(), 64)
    lps = [(e.kw["token"], e.kw["logprob"], e.how) for e in s._create_chat_logprobs([4, 5], None)]
    s.full("final", None)
    return {"prompt_token_ids": inputs.get("prompt_token_ids"), "extra_args": params.extra_args, "logprobs": lps,
            "records": list(calls)}


def test_composed_chat_patch_equals_each_patch_alone(pinned):
    reply = P.patched_chat_serving(SOURCE.encode())
    fast = F.edited(SOURCE.encode(), F.CHAT, F.CHAT_EDITS)
    both = F.edited(reply, F.CHAT, F.CHAT_EDITS)  # boot order: prefix_cache first, then serving_fast
    assert F.base_of(reply, F.CHAT) == (reply, ("reply_ids",))
    r, f, b = (run(load(t.decode()), pinned) for t in (reply, fast, both))
    assert b["prompt_token_ids"] == r["prompt_token_ids"] == [9, 9, 1, 2, 3] and b["records"] == r["records"]
    assert b["extra_args"] == f["extra_args"] == {"arc_sampled_only": True}
    assert b["logprobs"] == f["logprobs"] and {h for *_, h in b["logprobs"]} == {"constructed"}
    # the other order gives the same file: the edits are disjoint
    other = fast.decode()
    for anchor, insert, after in ((P.REPLY_IMPORT_ANCHOR, P.REPLY_IMPORT_INSERT, True),
                                  (P.REPLY_RENDER_ANCHOR, P.REPLY_RENDER_INSERT, True),
                                  (P.REPLY_FINISH_ANCHOR, P.REPLY_FINISH_INSERT, False)):
        other = other.replace(anchor, anchor + insert if after else insert + anchor)
    assert other.encode() == both


def test_anything_but_reply_ids_on_the_pinned_file_is_refused(pinned):
    reply = P.patched_chat_serving(SOURCE.encode())
    for bad in (reply + b"# x\n", reply.replace(P.REPLY_FINISH_INSERT.encode(), b""), SOURCE.encode() + b"\n"):
        assert F.base_of(bad, F.CHAT) is None
        with pytest.raises(RuntimeError, match="unrecognized"):
            F.edited(bad, F.CHAT, F.CHAT_EDITS)
    assert F.base_of(reply, F.IMAGE) is None  # only the chat file may carry reply_ids
