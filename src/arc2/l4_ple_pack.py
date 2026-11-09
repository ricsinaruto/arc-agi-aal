"""Four L4s: the PLE layer's prefill convolution in groups of bounded padding (``ple_pack``).

The pinned ``_short_conv_dilated_prefill_batched`` pads every prefilling row of a step to the step's longest row:
rows x longest x 10,240 channels in BF16, several such tensors alive at its peak. Every allocation that ended an
engine on the L4s was one of them (arc2-l4-score-v4, package rehearsals 3 and 5: 4 to 10 rows x one 3,136-token
chunk, 246 to 614 MiB a tensor), while the step's real tokens never pass 4,096. The throughput arm met the same
call in its 8,192-token chunk test and wrote the fix (``ple_pack_bound``, branch ``v82-chunk`` d394e896,
``arc_infra.serving_kernels_patch``): rows run in consecutive groups, outputs and states equal to the pinned call
(its CPU test, here ``tests/test_arc2.py``). That option rides the ``ple_uva`` overlay, which the L4s do not use
(their table is in the CPU offload process), so this module applies the same edit on its own, with a tighter
bound: a group's padding is at most ``SLACK`` slots, so a step's largest call holds its tokens plus ``SLACK``.

Standard library only. ``apply`` edits the extracted runtime (disposable), once, by its exact anchor.
"""

import hashlib
import json
from pathlib import Path

RELATIVE = "vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
ANCHOR = "    def _short_conv_dilated_prefill_batched(\n"
MARK = "ARC2 L4 ple_pack"
SLACK = 512
REPLACEMENT = '''    def _short_conv_dilated_prefill_batched(
        self,
        x_p,
        metadata,
        conv_state,
        conv_weights,
        state_indices_tensor_p,
        num_prefills,
        num_decode_tokens,
        num_prefill_tokens,
    ):
        # ARC2 L4 ple_pack: prefill rows in consecutive groups whose padding is at most __SLACK__ slots (the pinned
        # call pads every row to the step's longest).
        packed = self._arc_short_conv_prefill_packed
        starts = metadata.non_spec_query_start_loc
        initial = metadata.has_initial_states_p
        if (
            num_prefills < 2
            or starts is None
            or initial is None
            or num_prefills * metadata.max_prefill_query_len <= num_prefill_tokens + __SLACK__
            or (x_p.is_cuda and torch.cuda.is_current_stream_capturing())
        ):
            return packed(x_p, metadata, conv_state, conv_weights, state_indices_tensor_p, num_prefills,
                          num_decode_tokens, num_prefill_tokens)
        import dataclasses as _arc_dataclasses

        starts = starts[-num_prefills - 1 :] - num_decode_tokens
        bounds = starts.tolist()
        output = torch.empty_like(x_p)
        first = 0
        while first < num_prefills:
            last, longest = first + 1, bounds[first + 1] - bounds[first]
            while last < num_prefills:
                wider = max(longest, bounds[last + 1] - bounds[last])
                if (last + 1 - first) * wider > bounds[last + 1] - bounds[first] + __SLACK__:
                    break
                last, longest = last + 1, wider
            low, high = bounds[first], bounds[last]
            output[low:high] = packed(
                x_p[low:high],
                _arc_dataclasses.replace(
                    metadata,
                    non_spec_query_start_loc=starts[first : last + 1] - low,
                    has_initial_states_p=initial[first:last],
                    max_prefill_query_len=longest,
                ),
                conv_state,
                conv_weights,
                state_indices_tensor_p[first:last],
                last - first,
                0,
                high - low,
            )
            first = last
        return output

    def _arc_short_conv_prefill_packed(
'''


def patched(text, slack=SLACK):
    """The file's text with the grouped call in front of the pinned one (unchanged if it is already there)."""
    if MARK in text:
        return text
    if text.count(ANCHOR) != 1 or "_arc_short_conv_prefill_packed" in text:
        raise RuntimeError("ple_pack: the pinned prefill convolution is not where it was; re-read arc2.l4_ple_pack")
    return text.replace(ANCHOR, REPLACEMENT.replace("__SLACK__", str(int(slack))))


def apply(site_packages, working, slack=SLACK):
    """Edit the runtime's PLE layer and leave a record in ``working``."""
    path = Path(site_packages) / RELATIVE
    before = path.read_text()
    after = patched(before, slack)
    if after != before:
        path.write_text(after)
    record = {"relative": RELATIVE, "slack": int(slack), "changed": after != before,
              "original_sha256": hashlib.sha256(before.encode()).hexdigest(),
              "patched_sha256": hashlib.sha256(after.encode()).hexdigest()}
    (Path(working) / "ple-pack-patch.json").write_text(json.dumps(record, indent=1))
    return record
