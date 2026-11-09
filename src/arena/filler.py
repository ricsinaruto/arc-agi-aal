"""Filler rate of sampled replies (pilot 1 drift guard; pilot c's loops: 'hmm' runs before its collapse): 'hmm' tokens per
1,000 thinking tokens (tokens before ``</think>``), the definition of rl-inv-replies (20 Oct). The Flash tokenizer's
classes are constants here; ``check`` recomputes them from a tokenizer.json (byte-level BPE, no ``tokenizers``)."""
import json
import re

HMM = re.compile(r"h+m+[.,!?:;]*")
HMM_IDS = frozenset({33165, 43899, 47701, 50397, 77264, 84485, 85152})  # hm HM Ġhm ĠHM Hmm Ġhmm ĠHmm
THINK_END = 248069


def byte_decoder():
    """GPT-2 byte-level alphabet: printable char -> byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for c, b in zip(cs, bs)}


def classes(path):
    """(hmm ids, </think> id) of a tokenizer.json."""
    data = json.loads(open(path, encoding="utf-8").read())
    dec = byte_decoder()
    hmm = set()
    for piece, i in data["model"]["vocab"].items():
        try:
            text = bytes(dec[c] for c in piece).decode("utf-8", errors="ignore")
        except KeyError:
            continue
        if HMM.fullmatch(text.strip().lower()):
            hmm.add(int(i))
    end = next((int(t["id"]) for t in data.get("added_tokens") or [] if t.get("content") == "</think>"), None)
    return frozenset(hmm), end


def check(path):
    """The constants against the served model's tokenizer: {"same": bool, ...}."""
    hmm, end = classes(path)
    return {"same": hmm == HMM_IDS and end == THINK_END, "hmm": len(hmm), "end": end}


def reply_filler(tokens, hmm=HMM_IDS, end=THINK_END):
    """(hmm count, thinking tokens) of one reply's sampled token ids."""
    think = tokens.index(end) if end in tokens else len(tokens)
    return sum(1 for t in tokens[:think] if t in hmm), think
