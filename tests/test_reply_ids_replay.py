"""RL1_REPLY_IDS (29 Oct): the trainer replays the play server's reply-id cache from the rollout (rl_rollouts.substitute),
so a window holds the ids the server saw; the served prompt digest is the exactness gate."""

from posttrain import rl_rollouts as rr

END = 99  # the turn end (special)


class Tok:
    """Ids are strings; a split differs from another by its pieces only."""

    vocab = {1: "<s>", 2: "sys", 3: "user", 4: "assistant", 10: " Circular", 11: " C", 12: "ircular", 13: " ok",
             14: " Circ", 15: "ular", 20: " Square", END: "<|im_end|>"}
    all_special_ids = [END]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab[i] for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [self.vocab[i] for i in ids]


def row(prompt, tokens):
    return {"usage": {"prompt_tokens": len(prompt)}, "prompt_digest": rr.ids_digest(prompt), "tokens": tokens}


def test_a_regenerated_split_is_put_back_and_the_served_digest_matches():
    p1 = [1, 2, 3, 4]
    gen1 = [11, 12, 13, END]  # sampled " C" "ircular": the canonical split is " Circular"
    served2 = p1 + gen1 + [3, 4]  # the server re-sent the reply with its sampled ids
    canon2 = p1 + [10, 13, END] + [3, 4]
    out, splices = rr.substitute(canon2, rr.reply_records([row(p1, gen1)]), Tok())
    assert out == served2 and splices == [(4, 6, [11, 12, 13])]
    assert rr.ids_digest(out) == rr.ids_digest(served2)
    # a second reply keyed on the substituted prefix (the server hashes its output sequence)
    gen2 = [14, 15, END]
    served3 = served2 + gen2 + [3, 4]
    canon3 = canon2 + [10, END] + [3, 4]
    records = rr.reply_records([row(p1, gen1), row(served2, gen2)])
    out3, splices3 = rr.substitute(canon3, records, Tok())
    assert out3 == served3 and len(splices3) == 2


def test_a_one_token_text_change_is_not_substituted_and_fails_the_digest():
    p1 = [1, 2, 3, 4]
    gen1 = [11, 12, 13, END]
    served2 = p1 + gen1 + [3, 4]
    edited = p1 + [20, 13, END] + [3, 4]  # the re-sent reply's text differs (" Square"): never the sampled ids
    out, splices = rr.substitute(edited, rr.reply_records([row(p1, gen1)]), Tok())
    assert out == edited and splices == []
    assert rr.ids_digest(out) != rr.ids_digest(served2)


def test_canonical_replies_and_no_records_leave_ids_alone():
    p1 = [1, 2, 3, 4]
    gen1 = [10, 13, END]  # sampled the canonical split
    canon2 = p1 + gen1 + [3, 4]
    assert rr.substitute(canon2, rr.reply_records([row(p1, gen1)]), Tok()) == (canon2, [])
    assert rr.substitute(canon2, [], Tok()) == (canon2, [])


def test_no_substitution_across_an_image_or_without_a_turn_end():
    p1 = [1, 2, 3, 4]
    gen1 = [11, 12, 13, END]
    canon2 = p1 + [10, 13, END] + [3, 4]
    assert rr.substitute(canon2, rr.reply_records([row(p1, gen1)]), Tok(), images=[(5, 6)])[1] == []
    cut = [11, 12, 13]  # a length-cut reply: no turn end, the server records nothing
    assert rr.substitute(canon2, rr.reply_records([row(p1, cut)]), Tok())[1] == []


def test_captures_carry_the_records_of_every_earlier_request():
    def r(n, gameplay, prompt, tokens):
        return {"n": n, "gameplay": gameplay, "keep": 0, "messages": [], "template": {}, "reply": {}, "usage":
                {"prompt_tokens": len(prompt)}, "prompt_digest": rr.ids_digest(prompt), "tokens": tokens}

    rows = [r(0, True, [1, 2], [11, END]), r(1, False, [1, 2, 3], [13, END]), r(2, True, [1, 2, 3, 4], [14, END])]
    caps = rr.captures(rows)
    assert [len(c["records"]) for c in caps] == [0, 2]  # the non-gameplay request's reply counts too
