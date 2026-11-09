"""The n-gram (PLE) rows of a whole example looked up once on the host, pinned, and sliced per model call with an
async copy (sft-heal's HEAL_PLE_TABLE, 21 Oct S4). The model's own lookup, with its table in host RAM, does a
GPU -> host id copy, a host gather and a pageable copy at every forward and every checkpoint replay, each a sync that
leaves the card idle. Exact: the rows are the module's own forward on a host copy of it (the table shared), and a
sequence's rows equal its windows' (the cache keeps the last ngram_size - 1 ids; the eos rule looks back no further).
``ROWS["now"]``: {layer_idx: [1, n, D] on the device} for the current call (and its replay), else the original."""
import copy
import types
import weakref

import torch
import torch.nn.functional as F

ROWS = {"now": None, "proxies": {}}


def modules(model):
    core = (model.get_base_model() if hasattr(model, "get_base_model") else model).model.language_model
    return [layer.ple.ple_embedding for layer in core.layers if getattr(layer, "ple", None) is not None]


def install(model):
    """Each n-gram embedding's forward: with rows set for the call, the original's cache bookkeeping (the window's
    last ids for the next call) and the rows; else the original."""
    for mod in modules(model):
        if getattr(mod, "_sft_original", None) is not None:
            continue
        original = mod.forward
        mod._sft_original = original

        def forward(self, input_ids, past_key_values, original=original):
            rows = ROWS["now"]
            got = rows.get(self.layer_idx) if rows is not None else None
            if got is None:
                return original(input_ids, past_key_values)
            if got.shape[1] != input_ids.shape[1]:
                raise RuntimeError(f"PLE rows for {got.shape[1]} tokens, call of {input_ids.shape[1]}")
            if past_key_values is not None:  # Qwen4ExpTextNGramEmbedding.forward's cache update, unchanged
                ids = input_ids.long()
                if (not past_key_values.has_previous_state(self.layer_idx, state_idx=2)
                        and ids.shape[1] < self.context_len):
                    ids = F.pad(ids, (self.context_len - ids.shape[1], 0), value=self.eos_token_id)
                past_key_values.update_conv_state(ids, self.layer_idx, state_idx=2, conv_kernel_size=self.context_len)
            return got

        mod.forward = types.MethodType(forward, mod)


@torch.no_grad()
def rows(model, ids, pin=False):
    """Every PLE layer's rows of a whole sequence (``ids`` [T] or [1, T] on the host)."""
    out = {}
    for mod in modules(model):
        got = ROWS["proxies"].get(id(mod))
        proxy = got[1] if got is not None and got[0]() is mod else None
        if proxy is None:
            proxy = copy.copy(mod)
            object.__setattr__(proxy, "_buffers", {k: (None if v is None else v.cpu()) for k, v in mod._buffers.items()})
            ROWS["proxies"][id(mod)] = (weakref.ref(mod), proxy)
        got = type(mod).forward(proxy, ids.reshape(1, -1).long().cpu(), None)
        out[mod.layer_idx] = got.pin_memory() if pin else got
    return out


def piece(table, start, stop, device):
    """The rows of [start, stop) on ``device`` (async from pinned host memory)."""
    return None if table is None else {i: t[:, start:stop].to(device, non_blocking=True) for i, t in table.items()}
