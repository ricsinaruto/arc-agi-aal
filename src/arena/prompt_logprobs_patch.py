"""Opt-in teacher forcing on a *warm* request for the pinned vLLM runtime (29 September; prefix-unit-int4.md).

Pinned vLLM makes every request with ``prompt_logprobs`` skip the prefix-cache lookup
(``SamplingParams.__post_init__``: ``skip_reading_prefix_cache = prompt_logprobs is not None``), because the logprobs
of cached positions are never computed. A partial-hit correctness test needs the opposite: resume from a cached
state and teacher-force every position after it. The worker already supports that shape: it allocates one row per
prompt position and fills rows from ``request.num_computed_tokens`` on (``GPUModelRunner._get_prompt_logprobs_dict``),
so the rows after the hit are exact and the rows of the cached prefix are left unwritten.

Two hash-guarded edits, active only on requests that ask for it:

- ``vllm/sampling_params.py``: ``extra_args["arc_read_prefix_cache"]`` (``vllm_xargs`` in the OpenAI API) sets
  ``skip_reading_prefix_cache = False`` before the default rule; every other request is unchanged.
- ``vllm/v1/outputs.py``: ``LogprobsTensors.empty_cpu`` allocates zeros instead of uninitialised memory, so the
  unwritten rows of a warm request carry token 0 (a valid id for the detokenizer) instead of garbage ids. Rows that
  are written (every row of any cold request) are unchanged.

The client knows the hit (``usage.prompt_tokens_details.cached_tokens``) and scores only positions after it.

Measured (prefix-unit-int4 v1, 29 September): the served runtime runs the V2 GPU model runner
(``v1/worker/gpu/sample/prompt_logprob.py``), which returns rows for the scheduled tokens only, so a warm request's
``prompt_logprobs`` list is ``[None]`` + the positions after the hit (length = prompt - cached) and the zero-row edit
is never exercised; ``arena.partial_tf`` maps entry j to position cached + j.
"""
import hashlib
import json
from pathlib import Path

SAMPLING_RELATIVE = 'vllm/sampling_params.py'
SAMPLING_SHA256 = '73e5bf250cb13803e4ce390407ae059b497898c8b81f9985d2b5c055d4db8f7d'
SAMPLING_ANCHOR = '        if self.skip_reading_prefix_cache is None:\n            # If prefix caching is enabled,\n'
SAMPLING_REPLACEMENT = ('        # ARC warm prompt logprobs: an explicit request (extra_args) reads the prefix cache.\n'
                        '        if (self.skip_reading_prefix_cache is None and self.extra_args\n'
                        '                and self.extra_args.get("arc_read_prefix_cache")):\n'
                        '            self.skip_reading_prefix_cache = False\n'
                        + SAMPLING_ANCHOR)
OUTPUTS_RELATIVE = 'vllm/v1/outputs.py'
OUTPUTS_SHA256 = '1d255ca30fa2db85ce80033e0d38ce7faecbff794a42899bc84bd420d2a68f39'
OUTPUTS_ANCHOR = ('        logprob_token_ids = torch.empty(\n'
                  '            (num_positions, num_tokens_per_position), dtype=torch.int32, device="cpu"\n'
                  '        )\n'
                  '        logprobs = torch.empty_like(logprob_token_ids, dtype=torch.float32)\n'
                  '        selected_token_ranks = torch.empty(\n')
OUTPUTS_REPLACEMENT = ('        # ARC warm prompt logprobs: zero rows (a warm request leaves its cached rows unwritten).\n'
                       '        logprob_token_ids = torch.zeros(\n'
                       '            (num_positions, num_tokens_per_position), dtype=torch.int32, device="cpu"\n'
                       '        )\n'
                       '        logprobs = torch.zeros_like(logprob_token_ids, dtype=torch.float32)\n'
                       '        selected_token_ranks = torch.zeros(\n')
MARKER = 'prompt-logprobs-patch.json'
TARGETS = ((SAMPLING_RELATIVE, SAMPLING_SHA256, SAMPLING_ANCHOR, SAMPLING_REPLACEMENT),
           (OUTPUTS_RELATIVE, OUTPUTS_SHA256, OUTPUTS_ANCHOR, OUTPUTS_REPLACEMENT))


def patched(original, sha256, anchor, replacement, relative):
    if hashlib.sha256(original).hexdigest() != sha256:
        raise RuntimeError(f'Warm prompt-logprobs patch rejected an unrecognized pinned {relative}')
    text = original.decode()
    if text.count(anchor) != 1:
        raise RuntimeError(f'Warm prompt-logprobs patch needs exactly one anchor in {relative}')
    text = text.replace(anchor, replacement)
    compile(text, relative, 'exec')
    return text.encode()


def apply(site, output):
    """Patch the extracted runtime (idempotent; re-applied after a restart re-extracts it)."""
    site, marker = Path(site), Path(output) / MARKER
    previous = json.loads(marker.read_text()) if marker.exists() else {}
    identity = {}
    for relative, sha256, anchor, replacement in TARGETS:
        target = site / relative
        current = hashlib.sha256(target.read_bytes()).hexdigest()
        done = previous.get(relative)
        if done and current == done['patched_sha256']:
            identity[relative] = done
            continue
        text = patched(target.read_bytes(), sha256, anchor, replacement, relative)
        target.write_bytes(text)
        identity[relative] = {'original_sha256': current, 'patched_sha256': hashlib.sha256(text).hexdigest()}
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    return identity
