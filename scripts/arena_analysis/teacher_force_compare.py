"""Cross-server comparison of the spec bench's ``teacher_force`` results (throughput arm, 3 October).

Two bench.json files of servers that teacher-forced the same fixed sequences (``spec_sweep.teacher_force``): per
position KL(reference || candidate) on the coarse top-k partition, argmax agreement and |delta logprob| of the next
recorded token (arena.kernel_gate.compare_positions), overall and per sequence.

    uv run --extra dev python scripts/arena_analysis/teacher_force_compare.py <reference bench.json> <candidate bench.json>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from arena.kernel_gate import compare_positions  # noqa: E402


def rows(path):
    data = json.loads(Path(path).read_text())
    tf = data["spec_sweep"]["teacher_force"]
    return {r["id"]: r for r in tf["sequences"]}


def compare(reference, candidate):
    ref, cand = rows(reference), rows(candidate)
    pairs, per = [], []
    for key, r in ref.items():
        c = cand.get(key)
        if c is None or c["ids"] != r["ids"]:
            continue
        mine = []
        # top[i] is the distribution at position i of the kept tail; the recorded next token is ids[i] (vLLM's prompt
        # logprobs at position i score token i given the tokens before it).
        for p, q, chosen in zip(r["top"], c["top"], r["ids"]):
            p = {int(k): v for k, v in p.items()}
            q = {int(k): v for k, v in q.items()}
            mine.append((p, q, chosen))
        pairs.extend(mine)
        per.append({"id": key, "tokens": r["tokens"], **compare_positions(mine)})
    return {"summary": compare_positions(pairs), "sequences": per}


if __name__ == "__main__":
    out = compare(sys.argv[1], sys.argv[2])
    print(json.dumps(out["summary"], indent=1))
    for row in out["sequences"]:
        print(row["id"], row["tokens"], round(row["mean_kl"] or 0, 4), round(row["top1_agreement"] or 0, 3),
              round(row["mean_abs_delta_logprob"] or 0, 4))
