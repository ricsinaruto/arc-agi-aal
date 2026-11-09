"""RSI gen 0: the 16K draft vocabulary refit on the 10 fixed selection games only. Same rule as
specialization_vocab.draft_vocab (all byte tokens and added ids, then generated tokens by frequency, ties by id); the
counts are the generated tokens (assistant spans after the generation prompt) of a fixed-10 usage corpus.
usage: draft_vocab_fixed10.py <usage.json> <out.json>"""
import hashlib, json, sys
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))
from specialization_vocab import TOKENIZER, V, byte_level_ids  # noqa: E402

rows = json.load(open(sys.argv[1]))
counts = np.zeros(V, np.int64)
for r in rows:
    ids = r["ids"]
    for a, b in r["assistant_spans"]:
        counts += np.bincount(np.asarray(ids[a + 3:b], dtype=np.int64), minlength=V)
tok = json.load(open(TOKENIZER))
special = sorted(a["id"] for a in tok["added_tokens"] if a["id"] < V)
chosen = set(byte_level_ids(tok["model"]["vocab"])) | set(special)
for t in np.lexsort((np.arange(V), -counts)):
    if len(chosen) >= 16384:
        break
    if counts[t] > 0:
        chosen.add(int(t))
ids = sorted(chosen)
rec = {"name": "fixed10-16k", "size": 16384, "vocab_rows": V,
       "corpus": {"usage": sys.argv[1], "sha256": hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest(),
                  "families": sorted({r["family"] for r in rows}), "generated_tokens": int(counts.sum()),
                  "distinct_generated": int((counts > 0).sum())},
       "rule": "all 256 byte tokens and every added/special id, then generated tokens by corpus frequency (ties by id)",
       "corpus_coverage": float(counts[ids].sum() / counts.sum()),
       "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(), "ids": ids}
json.dump(rec, open(sys.argv[2], "w"), separators=(",", ":"))
print(json.dumps({k: v for k, v in rec.items() if k != "ids"}))
