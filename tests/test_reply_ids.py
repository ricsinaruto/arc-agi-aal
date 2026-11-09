"""reply_ids (arena.reply_ids_runtime): a re-sent reply keeps the token ids it was generated with."""
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from arena import prefix_cache_patch, reply_ids_runtime
from arena.prefix_cache_patch import validate
from arena.reply_ids_runtime import ReplyIds

PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))
TOKENIZER_JSON = Path(os.environ.get("ARC_TOKENIZER_JSON", "/nonexistent"))
BASE = {"mamba_align_fix": True, "mamba_keep_state": True, "mamba_demote_superseded": True}
END, START = 9, 8
VOCAB = {1: "un", 2: "covers", 3: "unc", 4: "overs", 5: " the", 6: " key", 7: "user\n", START: "<s>", END: "<e>",
         10: "assistant\n", 11: "<img>", 12: " door", 13: "uncovers"}
CANONICAL, GENERATED = [13, 5, 6], [3, 4, 5, 6]  # one text, two splits


class Tokenizer:
    all_special_ids = [START, END]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(VOCAB[i] for i in ids)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return [VOCAB[i] for i in ids]


@dataclass(frozen=True)
class Placeholder:
    offset: int
    length: int


def finished(prompt, generated):
    return SimpleNamespace(prompt_token_ids=list(prompt), outputs=[SimpleNamespace(token_ids=list(generated))])


def tokens(ids, **extra):
    return {"type": "token", "prompt_token_ids": list(ids), **extra}


PROMPT = [START, 7, 5, 6, END, START, 10]
NEXT_TURN = [START, 7, 12, END, START, 10]


def test_option_is_off_by_default_and_needs_the_request_end_entry():
    tail = {**BASE, "request_end_tail": True}
    assert "reply_ids" not in validate(tail) and "reply_ids" not in validate({**tail, "reply_ids": False})
    assert validate({**tail, "reply_ids": True})["reply_ids"] is True
    with pytest.raises(ValueError):
        validate({**BASE, "reply_ids": True})


def test_a_diverged_reply_gets_its_generated_ids_back_and_later_placeholders_move():
    cache = ReplyIds()
    cache.record(finished(PROMPT, GENERATED + [END]), Tokenizer())
    before, after = Placeholder(2, 1), Placeholder(len(PROMPT) + len(CANONICAL) + 3, 1)
    sent = tokens(PROMPT + CANONICAL + [END] + NEXT_TURN, type="multimodal",
                  mm_placeholders={"image": [before, after]})
    assert cache.substitute(sent, Tokenizer(), 10_000) == 1
    assert sent["prompt_token_ids"] == PROMPT + GENERATED + [END] + NEXT_TURN
    assert sent["mm_placeholders"] == {"image": [before, Placeholder(after.offset + 1, 1)]}
    assert cache.stats["replaced_spans"] == 1 and cache.stats["replaced_tokens"] == len(GENERATED)
    # The record stays: every later prompt of the game re-sends this reply.
    again = tokens(PROMPT + CANONICAL + [END] + NEXT_TURN + [12, END])
    assert cache.substitute(again, Tokenizer(), 10_000) == 1
    assert again["prompt_token_ids"] == PROMPT + GENERATED + [END] + NEXT_TURN + [12, END]


def test_the_same_text_from_another_source_is_tokenized_as_before():
    cache = ReplyIds()
    cache.record(finished(PROMPT, GENERATED + [END]), Tokenizer())
    # The reply's text inside a user turn (or a tool output) of another prompt, and of this game's own later prompt.
    elsewhere = tokens([START, 7] + CANONICAL + [END, START, 10])
    assert cache.substitute(elsewhere, Tokenizer(), 10_000) == 0
    assert elsewhere["prompt_token_ids"] == [START, 7] + CANONICAL + [END, START, 10]
    quoted = PROMPT + CANONICAL + [END, START, 7] + CANONICAL + [END, START, 10]
    own = tokens(quoted)
    assert cache.substitute(own, Tokenizer(), 10_000) == 1
    assert own["prompt_token_ids"] == PROMPT + GENERATED + [END, START, 7] + CANONICAL + [END, START, 10]
    # Behind a prompt that differs in one token the reply is not replaced either.
    other = tokens([START, 7, 5, 12, END, START, 10] + CANONICAL + [END] + NEXT_TURN)
    assert cache.substitute(other, Tokenizer(), 10_000) == 0


def test_identical_ids_drop_the_record_and_another_text_is_left_alone():
    cache = ReplyIds()
    cache.record(finished(PROMPT, CANONICAL + [END]), Tokenizer())
    sent = tokens(PROMPT + CANONICAL + [END] + NEXT_TURN)
    assert cache.substitute(sent, Tokenizer(), 10_000) == 0 and cache.count == 0 and cache.ids == 0
    assert cache.stats["resent_identical"] == 1 and not cache.lengths
    cache.record(finished(PROMPT, GENERATED + [END]), Tokenizer())
    edited = tokens(PROMPT + [13, 5, 12, END] + NEXT_TURN)  # the harness re-sent another text
    assert cache.substitute(edited, Tokenizer(), 10_000) == 0
    assert edited["prompt_token_ids"] == PROMPT + [13, 5, 12, END] + NEXT_TURN
    assert cache.stats["resent_text_differs"] == 1 and cache.count == 0  # dropped: bounded work per request


def test_a_chain_of_diverged_replies_is_rebuilt_from_the_prompts_as_sent():
    cache, tok = ReplyIds(), Tokenizer()
    cache.record(finished(PROMPT, GENERATED + [END]), tok)
    second = tokens(PROMPT + CANONICAL + [END] + NEXT_TURN)
    cache.substitute(second, tok, 10_000)
    cache.record(finished(second["prompt_token_ids"], [1, 2, END]), tok)  # "uncovers" as [un][covers]
    third = tokens(PROMPT + CANONICAL + [END] + NEXT_TURN + [13, END] + NEXT_TURN)
    assert cache.substitute(third, tok, 10_000) == 2
    assert third["prompt_token_ids"] == PROMPT + GENERATED + [END] + NEXT_TURN + [1, 2, END] + NEXT_TURN


def test_clones_with_one_prompt_keep_their_own_replies():
    cache, tok = ReplyIds(), Tokenizer()
    cache.record(finished(PROMPT, GENERATED + [END]), tok)
    cache.record(finished(PROMPT, [1, 2, END]), tok)
    a, b = tokens(PROMPT + CANONICAL + [END] + NEXT_TURN), tokens(PROMPT + [13, END] + NEXT_TURN)
    assert cache.substitute(a, tok, 10_000) == 1 and cache.substitute(b, tok, 10_000) == 1
    assert a["prompt_token_ids"] == PROMPT + GENERATED + [END] + NEXT_TURN
    assert b["prompt_token_ids"] == PROMPT + [1, 2, END] + NEXT_TURN


def test_the_cache_is_bounded_by_ids_and_reports_its_memory():
    cache, tok = ReplyIds(max_ids=10), Tokenizer()
    for i in range(6):
        cache.record(finished(PROMPT + [5] * i, GENERATED + [END]), tok)
    assert cache.ids == 8 and cache.count == 2 and cache.stats["evicted"] == 4
    assert cache.memory_bytes() == 4 * 8 + 2 * reply_ids_runtime.RECORD_OVERHEAD
    assert sorted(cache.lengths) == [len(PROMPT) + 4, len(PROMPT) + 5]
    cache.record(finished(PROMPT, [5, 6]), tok)  # no turn end (finished by length): not recorded
    cache.record(finished(PROMPT, GENERATED * 4 + [END]), tok)  # longer than the bound
    assert cache.count == 2 and cache.stats["record_skip_no_turn_end"] == 2


def test_no_substitution_past_the_length_limit_or_with_positional_fields():
    cache, tok = ReplyIds(), Tokenizer()
    cache.record(finished(PROMPT, GENERATED + [END]), tok)
    ids = PROMPT + CANONICAL + [END] + NEXT_TURN
    sent = tokens(ids)
    assert cache.substitute(sent, tok, len(ids)) == 0 and sent["prompt_token_ids"] == ids
    assert cache.stats["skip_too_long"] == 1
    masked = tokens(ids, assistant_tokens_mask=[0] * len(ids))
    assert cache.substitute(masked, tok, 10_000) == 0 and masked["prompt_token_ids"] == ids
    inside = tokens(ids, type="multimodal", mm_placeholders={"image": [Placeholder(len(PROMPT) + 1, 1)]})
    assert cache.substitute(inside, tok, 10_000) == 0 and inside["prompt_token_ids"] == ids


def test_serving_hooks_never_raise(monkeypatch, capsys):
    monkeypatch.setattr(reply_ids_runtime, "_CACHE", ReplyIds())
    monkeypatch.setattr(reply_ids_runtime, "LOG_EVERY", 1)
    reply_ids_runtime.record(object(), Tokenizer())
    reply_ids_runtime.record(finished(PROMPT, GENERATED + [END]), Tokenizer())
    sent = tokens(PROMPT + CANONICAL + [END] + NEXT_TURN)
    reply_ids_runtime.substitute([sent], object(), 10_000, 100)  # a tokenizer without decode
    assert sent["prompt_token_ids"] == PROMPT + CANONICAL + [END] + NEXT_TURN
    reply_ids_runtime.substitute([sent], Tokenizer(), 10_000, None)
    assert sent["prompt_token_ids"] == PROMPT + GENERATED + [END] + NEXT_TURN
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("ARC_REPLY_IDS_STATS ") and '"replaced_spans": 1' in line and '"memory_bytes"' in line


def test_real_tokenizer_non_canonical_split_round_trip():
    tokenizers = pytest.importorskip("tokenizers")
    if not TOKENIZER_JSON.exists():
        pytest.skip("Optional tokenizer (ARC_TOKENIZER_JSON)")
    raw = tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))

    class Real:
        all_special_ids = [raw.token_to_id("<|im_end|>"), raw.token_to_id("<|im_start|>")]

        def decode(self, ids, skip_special_tokens=False):
            return raw.decode(list(ids), skip_special_tokens=skip_special_tokens)

        def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
            return [raw.id_to_token(i) for i in ids]

    def enc(text):
        return raw.encode(text, add_special_tokens=False).ids

    end = Real.all_special_ids[0]
    prompt = enc("<|im_start|>user\nWhat does the lever do? It uncovers the exit.<|im_end|>\n<|im_start|>assistant\n")
    reply = "<think>\nThe lever uncovers the exit.\n</think>\n\nIt uncovers the exit."
    canonical = enc(reply)
    generated = enc("<think>\nThe lever un") + enc("covers the exit.\n</think>\n\nIt unc") + enc("overs the exit.")
    assert generated != canonical and raw.decode(generated, skip_special_tokens=False) == reply
    rest = enc("\n<|im_start|>user\nGo on.<|im_end|>\n<|im_start|>assistant\n")
    cache = ReplyIds()
    cache.record(finished(prompt, generated + [end]), Real())
    sent = tokens(prompt + canonical + [end] + rest)
    assert cache.substitute(sent, Real(), 100_000) == 1
    assert sent["prompt_token_ids"] == prompt + generated + [end] + rest
    assert raw.decode(sent["prompt_token_ids"], skip_special_tokens=False) == raw.decode(
        prompt + canonical + [end] + rest, skip_special_tokens=False)


def test_patch_on_the_pinned_chat_serving(tmp_path):
    source = PINNED_SITE / prefix_cache_patch.CHAT_SERVING_RELATIVE
    if not source.exists():
        pytest.skip("Optional pinned runtime (ARC_PINNED_SITE)")
    patched = prefix_cache_patch.patched_chat_serving(source.read_bytes()).decode()
    assert patched.count("_arc_reply_ids.substitute(") == 1 and patched.count("_arc_reply_ids.record(final_res") == 1
    # Substitution precedes the prompt length and max_tokens computation; the record follows the last engine output.
    assert patched.index("_arc_reply_ids.substitute(") < patched.index("max_tokens = get_max_tokens(")
    assert patched.index("final_res = res") < patched.index("_arc_reply_ids.record(") < patched.index(
        "for output in final_res.outputs:")
    with pytest.raises(RuntimeError):
        prefix_cache_patch.patched_chat_serving(patched.encode())
    helper = Path(reply_ids_runtime.__file__).read_bytes()
    assert b"import vllm" not in helper and b"from vllm" not in helper
    assert hashlib.sha256(source.read_bytes()).hexdigest() == prefix_cache_patch.CHAT_SERVING_SHA256
