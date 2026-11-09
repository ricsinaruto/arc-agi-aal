"""serving_fast.logprob_lean (installed as vllm/arc_lean.py): a full (non-stream) chat reply that shows tokens as
``token_id:N`` builds its logprob entries as plain dicts ({token, logprob, top_logprobs: [{token, logprob}]}, no
``bytes``) and is written by one json.dumps (real-v5 27 Oct: pydantic construct/dump + bytes lists were ~half of each
API process's core at top_logprobs 20). Token ids, values, alternatives and their order are the pinned path's."""

import json
from itertools import islice

from starlette.responses import Response

RAW = {}  # id(the reply's logprobs object) -> (object, entries); popped when the reply is written
LIMIT = 4096  # oldest dropped past this (a reply that failed between build and write)


def lean_logprobs(cls, token_ids, top_logprobs, num_output_top_logprobs, return_all):
    content = []
    for i, token_id in enumerate(token_ids):
        step = top_logprobs[i]
        sampled = None if step is None else step.get(token_id)
        if sampled is None:
            content.append({"token": f"token_id:{token_id}", "logprob": -9999.0, "top_logprobs": []})
            continue
        if return_all or num_output_top_logprobs == -1:
            items = step.items()
        elif num_output_top_logprobs is None:
            items = ()
        else:
            items = islice(step.items(), max(num_output_top_logprobs, 0))
        content.append({"token": f"token_id:{token_id}", "logprob": max(sampled.logprob, -9999.0),
                        "top_logprobs": [{"token": f"token_id:{t}", "logprob": max(p.logprob, -9999.0)}
                                         for t, p in items]})
    made = cls.model_construct(content=None)
    while len(RAW) >= LIMIT:
        RAW.pop(next(iter(RAW)))
    RAW[id(made)] = (made, content)
    return made


def lean_response(response, headers=None):
    """The reply as JSON with the lean entries in place, or None when no choice carries them."""
    found = {}
    for i, choice in enumerate(response.choices):
        lp = getattr(choice, "logprobs", None)
        row = RAW.pop(id(lp), None) if lp is not None else None
        if row is not None and row[0] is lp:
            found[i] = row[1]
    if not found:
        return None
    out = response.model_dump()
    for i, entries in found.items():
        out["choices"][i]["logprobs"]["content"] = entries
    body = json.dumps(out, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")).encode("utf-8")
    return Response(content=body, media_type="application/json", headers=headers)
