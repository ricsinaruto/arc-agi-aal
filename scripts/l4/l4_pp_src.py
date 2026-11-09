"""PP 2 + TP 2 + EP on four L4s, MTP 0..3 (l4pp, MTP 2 Nov): stages 0-23 and 24-47 of two cards each.

``apply`` edits the extracted runtime (disposable), once, by exact anchors:
- ``model_state.py`` refuses PP because later stages get no raw ids; PLE is layer index 1, so it runs on the first
  stage, and the guard now refuses only a partition that moves it off. Later stages read ``ngram_context`` from their
  own ``all_token_ids`` (filled by the PP broadcast) and no layer uses it.
- ``model.py``'s loader raises on the final mixer's weights on every stage but the last (first boot, l4pp2).
- ``gpu_worker.py`` refuses PP with PLE offload and registers every worker; the offload process expects dp x tp
  registrations, so only the first stage registers.
- ``ple_offload/worker.py``: the offload process builds the model at pp_size 1 and inherits the server's
  ``VLLM_PP_LAYER_PARTITION`` (``--pp-layers``), which vLLM refuses there; it drops it first.
With a draft (MTP k, V2 runner: the speculator is built on the last stage only, PPHandler sends the sampled tokens):
- ``mtp.py``: the draft's forward takes the embedding branch only on the first stage; it lives whole on the last
  stage, so it always takes it (it never gets intermediate tensors).
- ``pp_utils.py`` + ``model_runner.py``: the earlier stages build their inputs from their own ``draft_tokens``, which
  only the last stage fills. The last stage now sends its proposal after each broadcast of sampled tokens, on the same
  group and condition; the earlier stages receive it with the sampled tokens and write it into ``draft_tokens`` with
  them, ``pp_size`` steps later.

Standard library only at import.
"""

import hashlib
import json
from pathlib import Path

MARK = "ARC2 L4 pp"
GUARD = '''        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )
'''
GUARD_NEW = f'''        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if pp_size > 1:  # {MARK}: only the first stage needs raw ids, and PLE must stay there
            from vllm.distributed.utils import get_pp_indices

            first_end = get_pp_indices(config.num_hidden_layers, 0, pp_size)[1]
            if max(int(layer) for layer in config.ple_layer_ids) >= first_end:
                raise RuntimeError("PLE layers must sit on the first pipeline stage")
'''
VALIDATE = '''        if parallel_config.pipeline_parallel_size != 1:
            unsupported.append(f"PP={parallel_config.pipeline_parallel_size}")
'''
VALIDATE_NEW = f"        # {MARK}: PP allowed, the first stage alone registers with the offload process\n"
REGISTER = '''        if self._ple_offload_enabled:
            self.model_runner._setup_ple_offload(
'''
REGISTER_NEW = f'''        from vllm.distributed import get_pp_group  # {MARK}

        if self._ple_offload_enabled and get_pp_group().is_first_rank:
            self.model_runner._setup_ple_offload(
'''
MIXER = '''            "hyper_connection_mixer.block_inject_weight",
        ]
        loader = AutoWeightsLoader(
'''
MIXER_NEW = f'''            "hyper_connection_mixer.block_inject_weight",
        ]
        if self.hyper_connection_mixer is None:  # {MARK}: the mixer lives on the last stage
            skip_substrs.append("hyper_connection_mixer.")
        loader = AutoWeightsLoader(
'''
PLE_RUNNER = """            with set_current_vllm_config(vllm_config):
                runner = PleOffloadRunner(vllm_config)
"""
PLE_RUNNER_NEW = f"""            import os as _arc_os  # {MARK}: this process builds the model at pp_size 1; the stage split
            _arc_os.environ.pop("VLLM_PP_LAYER_PARTITION", None)  # (inherited) is the GPU workers' only (l4kd41)
            getattr(envs.__getattr__, "cache_clear", lambda: None)()
            with set_current_vllm_config(vllm_config):
                runner = PleOffloadRunner(vllm_config)
"""
FILES = {"vllm/models/qwen3_8_flash_next/nvidia/model.py": ((MIXER, MIXER_NEW),),
         "vllm/v1/ple_offload/worker.py": ((PLE_RUNNER, PLE_RUNNER_NEW),),
         "vllm/models/qwen3_8_flash_next/nvidia/model_state.py": ((GUARD, GUARD_NEW),),
         "vllm/v1/worker/gpu_worker.py": ((VALIDATE, VALIDATE_NEW), (REGISTER, REGISTER_NEW))}
MTP_FIRST = """        if get_pp_group().is_first_rank:
            assert hidden_states is not None
            if inputs_embeds is None:
"""
MTP_FIRST_NEW = f"""        if True:  # {MARK}: the draft lives whole on the last stage
            assert hidden_states is not None
            if inputs_embeds is None:
"""
PP_INIT = """        self.max_sample_len = num_speculative_steps + 1
"""
PP_INIT_NEW = f"""        self.max_sample_len = num_speculative_steps + 1
        self.drafts_due = False  # {MARK}: set by a sent broadcast, cleared by its drafts
"""
PP_SLOT = """    gen_at_receive_np: np.ndarray  # [num_reqs]
"""
PP_SLOT_NEW = f"""    gen_at_receive_np: np.ndarray  # [num_reqs]
    draft_tokens: torch.Tensor | None = None  # {MARK}: [num_reqs, max_sample_len - 1]
"""
PP_PREV = """        self.main_stream.wait_event(slot.event)
        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
        )
"""
PP_PREV_NEW = f"""        self.main_stream.wait_event(slot.event)
        drafts = None
        if slot.draft_tokens is not None:  # {MARK}: (rows, request indices) of the kept entries
            if exclude_mask.any():
                keep = np.flatnonzero(~exclude_mask)
                drafts = (slot.draft_tokens[async_copy_to_gpu(keep, device=self.device)],
                          async_copy_to_gpu(slot.idx_mapping_np[keep], device=self.device))
            else:
                drafts = (slot.draft_tokens, slot.idx_mapping)
        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
            draft_tokens=drafts,
        )
"""
PP_RECV = """            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            event = self.broadcast_stream.record_event()
"""
PP_RECV_NEW = f"""            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            draft_tokens = None
            if self.max_sample_len > 1:  # {MARK}: the last stage's proposal, sent after its sample
                draft_tokens = torch.empty(
                    num_reqs, self.max_sample_len - 1, dtype=torch.int64, device=self.device
                )
                torch.distributed.broadcast(
                    draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
                draft_tokens.record_stream(self.main_stream)
            event = self.broadcast_stream.record_event()
"""
PP_QUEUE = """            gen_at_receive_np,
        )
        return bool(need_sampled_mask.all())
"""
PP_QUEUE_NEW = f"""            gen_at_receive_np,
            draft_tokens,  # {MARK}
        )
        return bool(need_sampled_mask.all())
"""
PP_SEND = """            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
"""
PP_SEND_NEW = f"""            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
        self.drafts_due = self.max_sample_len > 1  # {MARK}: broadcast_drafts follows

    def broadcast_drafts(self, draft_tokens: torch.Tensor, input_batch: InputBatch) -> None:
        \"\"\"{MARK}: this step's proposal to the earlier stages, after its sampled tokens (one per sent broadcast).\"\"\"
        assert self.is_last_rank
        if not self.drafts_due:
            return
        self.drafts_due = False
        drafts = draft_tokens[: input_batch.num_reqs].to(torch.int64).contiguous()
        if drafts.shape != (input_batch.num_reqs, self.max_sample_len - 1):
            raise RuntimeError(f"pp drafts {{tuple(drafts.shape)}} for {{input_batch.num_reqs}} requests")
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(drafts, src=self.last_rank, group=self.broadcast_group)
            drafts.record_stream(self.broadcast_stream)
"""
RUN_SEND = """            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
"""
RUN_SEND_NEW = f"""            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
            if self.pp_handler is not None:  # {MARK}
                self.pp_handler.broadcast_drafts(draft_tokens, input_batch)
"""
RUN_RECV = """            outputs = self.pp_handler.get_prev_sampled_outputs()
            if outputs is not None:
                self.postprocess_sampled(**outputs)
"""
RUN_RECV_NEW = f"""            outputs = self.pp_handler.get_prev_sampled_outputs()
            if outputs is not None:
                drafts = outputs.pop("draft_tokens", None)
                self.postprocess_sampled(**outputs)
                if drafts is not None:  # {MARK}: the last stage's proposal for these requests
                    self.req_states.draft_tokens[drafts[1]] = drafts[0]
"""
DRAFT_FILES = {"vllm/models/qwen3_8_flash_next/nvidia/mtp.py": ((MTP_FIRST, MTP_FIRST_NEW),),
               "vllm/v1/worker/gpu/pp_utils.py": ((PP_INIT, PP_INIT_NEW), (PP_SLOT, PP_SLOT_NEW), (PP_PREV, PP_PREV_NEW),
                                                  (PP_RECV, PP_RECV_NEW), (PP_QUEUE, PP_QUEUE_NEW),
                                                  (PP_SEND, PP_SEND_NEW)),
               "vllm/v1/worker/gpu/model_runner.py": ((RUN_SEND, RUN_SEND_NEW), (RUN_RECV, RUN_RECV_NEW))}


def patched(text, edits):
    """The file's text with ``edits`` applied (unchanged if it already carries them)."""
    if MARK in text:
        return text
    for old, new in edits:
        if text.count(old) != 1:
            raise RuntimeError(f"pp: anchor not found once: {old[:60]!r}; re-read scripts/l4/l4_pp_src.py")
        text = text.replace(old, new)
    return text


def apply(site_packages, working, draft=0):
    """Edit the runtime and leave a record in ``working``; with a draft, the draft files too."""
    record = {}
    for relative, edits in {**FILES, **(DRAFT_FILES if draft else {})}.items():
        path = Path(site_packages) / relative
        before = path.read_text()
        after = patched(before, edits)
        if after != before:
            path.write_text(after)
        record[relative] = {"changed": after != before, "original_sha256": hashlib.sha256(before.encode()).hexdigest(),
                            "patched_sha256": hashlib.sha256(after.encode()).hexdigest()}
    (Path(working) / "pp-patch.json").write_text(json.dumps(record, indent=1))
    return record
