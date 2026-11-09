"""Train the Qwen3.8-Flash-Next MTP draft head on target hidden states dumped from the served model.

Draft-head fine-tune (docs/validation-20260927/draft-head-finetune.md). Runs in the
serving runtime's Python (torch) after the vLLM server has stopped, on the dumps written by
``arena.draft_head`` (``<dump>/<split>/<index>.target.bin``: the target's pre-final-mixer multi-stream
hidden state, BF16 [L, hc_count * hidden], one row per position of the sequence).

The module below is a training re-implementation of the pinned vLLM draft
(``vllm/models/qwen3_8_flash_next/nvidia/mtp.py`` + ``model.py`` decoder layer + ``qsa.py`` /
``indexer_qsa.py`` + ``hyperconnection.py``), in plain torch with checkpoint tensor names:

- input fusion (``residual_linear_shared``): pre-norm + ``fc_embedding`` of the next token's embedding,
  pre-norm over the whole multi stream + ``fc_hidden`` per HC branch, embedding added to every branch;
- one decoder layer: gated HC mix -> QSA attention (GQA 24/2 heads x 256, sigmoid output gate, partial
  RoPE 64 of 256, theta 1e7) whose keys are selected per query by the weight-free QSA indexer (4 heads
  x 128, ReLU scores over mean-pooled 4-token key blocks, top 512 blocks = 2,048 tokens + the <= 3 tail
  tokens) -> HC combine/mix -> MoE (512 BF16 experts top-10 softmax renormalised, sigmoid-gated shared
  expert) -> the final HC mixer (sampled single stream for the shared LM head, multi stream for the next
  step);
- draft steps 2..k (index sharing off, as served): the input is the previous step's multi stream and
  the next token; the query sits one position later, attends to the step-1 keys it selects with its own
  indexer query, the <= 3 step-1 tail tokens and the chain keys the earlier draft steps wrote at the
  positions after the anchor. (Served, the block that a chain key completes competes in the top-512
  instead of being always attended; identical whenever <= 512 blocks are visible.)

Teacher forcing: step k at anchor i consumes the recorded token x[i+k] and predicts x[i+k+1]. Only
generated positions (the recorded model outputs) are anchors.

Loss: soft cross-entropy of each draft step against the target's own next-token distribution (T=1) at
the verified position, weights ``--step-weights``.

Acceptance proxy (greedy drafts, as served: ``draft_sample_method`` greedy, standard rejection): with
p^T the target's sampling distribution (T 0.7, top-k 20, top-p 0.95) and d_k the draft argmax,
P(accept 1) = E[p^T_{i+1}(d1)], P(accept 1..2) = E[1{x_{i+2}=d1} p^T_{i+2}(d2)],
P(accept 1..3) = E[1{x_{i+2}=d1} 1{x_{i+3}=d2} p^T_{i+3}(d3)] (unbiased because x ~ p^T); their sum is
the expected accepted tokens per 3-token draft, comparable with vLLM's per-position acceptance
counters (75.5 / 57.7 / 44.6% measured in the arena hours). Top-1 match rates are reported as well.

Trainable: every ``mtp.*`` tensor except the routed experts and the (weight-free-selection) indexer
projections. The shared embedding and LM head stay frozen (the target uses them).

Locked-model run (7 October): ``--experts nvfp4`` serves the frozen routed experts through an emulation of vLLM's
online NvFp4 MoE quantizer (the served draft's experts), ``--draft-vocab`` restricts the draft argmax to the served
16K subset (metrics ``v16k`` beside ``full``), ``--select-split`` names the split of the probe and of the state
selection (a DEV split; the report splits are never used for selection), ``--draft-splits`` compares the served
draft's own step-1 rows with this implementation on every dumped sequence, ``--init-overlay`` loads an exported
head (evaluation of a trained head on other dumps), and checkpoints are written every ``--checkpoint-every`` steps.

Export: the trained tensors as an ``arena.weights_overlay`` (BF16, checkpoint names and shapes); the
served draft re-quantizes its experts itself (the phase-4 NvFp4 draft quantizer runs after loading),
and the experts are unchanged here anyway.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from arena import weights_overlay as wo  # noqa: E402

EPS = 1e-6
THETA = 1.0e7
TEMPERATURE, TOP_K, TOP_P = 0.7, 20, 0.95
TOPK_EXPERTS = 10
TOKEN_BUDGET, RATIO = 2048, 4
BLOCK_TOPK = TOKEN_BUDGET // RATIO
EMBED = "model.language_model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"
TARGET_MIXER = "model.language_model.hyper_connection_mixer."
FROZEN_MTP = ("mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj",
              "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
              "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
              "mtp.layers.0.self_attn.indexer.k_layernorm.weight")
BF16 = torch.bfloat16


def log(event, **fields):
    print(json.dumps({"event": f"draft_head_{event}", "t": round(time.time(), 1), **fields}), flush=True)


# --- checkpoint ------------------------------------------------------------------------------------


def read_tensors(model_dir, names, device):
    """Stdlib safetensors reads (no safetensors dependency) of ``names`` from the checkpoint."""
    model_dir = Path(model_dir)
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
    by_shard, out = {}, {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)
    for shard, shard_names in sorted(by_shard.items()):
        header = wo.read_header(model_dir / shard)
        for name in shard_names:
            spec = header[1][name]
            raw = bytearray(wo.read_tensor(model_dir / shard, name, header))
            tensor = torch.frombuffer(raw, dtype=dtypes[spec["dtype"]]).reshape(spec["shape"])
            out[name] = tensor.to(device)
    return out


def mtp_names(model_dir):
    weight_map = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    return sorted(n for n in weight_map if n.startswith("mtp."))


# --- math helpers ----------------------------------------------------------------------------------


def gemma_rms(x, weight, groups=1):
    """vLLM GemmaRMSNorm (x * rsqrt(mean(x^2) + eps) * (1 + w), in fp32, cast back); ``groups`` splits
    the last dimension into equal groups normalised separately (HC per-branch norm)."""
    dtype = x.dtype
    xf = x.float()
    if groups > 1:
        xf = xf.unflatten(-1, (groups, xf.shape[-1] // groups))
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    if groups > 1:
        xf = xf.flatten(-2)
    return (xf * (1.0 + weight.float())).to(dtype)


class Rope:
    """Neox rotary on the first ``rot`` dims (partial rotary 0.25 of 256 -> 64); the text-only MRoPE
    (equal t/h/w positions) reduces to this."""

    def __init__(self, rot, device, max_pos=1 << 18):
        inv = 1.0 / (THETA ** (torch.arange(0, rot, 2, dtype=torch.float32, device=device) / rot))
        freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32, device=device), inv)
        self.cos, self.sin = freqs.cos().to(BF16), freqs.sin().to(BF16)
        self.rot = rot

    def __call__(self, x, pos):  # x [N, heads, D], pos [N]
        half = self.rot // 2
        cos = self.cos[pos].float()[:, None, :]
        sin = self.sin[pos].float()[:, None, :]
        x1, x2 = x[..., :half].float(), x[..., half:self.rot].float()
        rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)
        return torch.cat((rotated, x[..., self.rot:]), dim=-1)


def sampling_probs(logits):
    """The target's sampling distribution (temperature, top-k, top-p as served), as (ids [N,K], probs)."""
    top, ids = (logits.float() / TEMPERATURE).topk(TOP_K, dim=-1)
    probs = top.softmax(-1)
    cum = probs.cumsum(-1)
    keep = (cum - probs) < TOP_P
    keep[:, 0] = True
    probs = torch.where(keep, probs, torch.zeros_like(probs))
    return ids, probs / probs.sum(-1, keepdim=True)


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def fake_nvfp4(weight, block=16):
    """vLLM's online NvFp4 MoE quantizer (``quantization/online/nvfp4.py``, as recorded in
    scripts/kernel_bench_phase4.py: per-expert global scale 6 * 448 / amax, ``scaled_fp4_quant``: 16-wide blocks
    along the last dimension with an E4M3 scale amax / 6 and E2M1 values), dequantized. [E, N, K] in and out."""
    levels = torch.tensor(E2M1, dtype=torch.float32, device=weight.device)
    mids = (levels[1:] + levels[:-1]) * 0.5
    out = torch.empty_like(weight)
    for e in range(weight.shape[0]):
        x = weight[e].float()
        scale = (6.0 * 448.0) / x.abs().amax().clamp_min(1e-8)
        blocks = (x * scale).unflatten(-1, (-1, block))
        sf = (blocks.abs().amax(-1, keepdim=True) / 6.0).to(torch.float8_e4m3fn).float()
        scaled = torch.where(sf > 0, blocks / sf.clamp_min(1e-30), torch.zeros_like(blocks))
        q = levels[torch.bucketize(scaled.abs().clamp(max=6.0), mids)] * scaled.sign()
        out[e] = (q * sf / scale).flatten(-2).to(weight.dtype)
    return out


# --- the draft module ------------------------------------------------------------------------------


class Draft(torch.nn.Module):
    def __init__(self, tensors, trainable, *, hc_count, heads, kv_heads, head_dim, idx_heads, idx_dim, rot):
        super().__init__()
        self.names = sorted(n for n in tensors if n.startswith("mtp."))
        self.trainable = [n for n in self.names if n in trainable]
        self.params = torch.nn.ParameterDict()
        self.frozen = {}
        for name in self.names:
            if name in trainable:
                self.params[name.replace(".", "__")] = torch.nn.Parameter(tensors[name].float())
            else:
                self.frozen[name] = tensors[name]
        self.embed, self.lm_head = tensors[EMBED], tensors[LM_HEAD]
        self.target_mixer = {k[len(TARGET_MIXER):]: v for k, v in tensors.items() if k.startswith(TARGET_MIXER)}
        self.hc, self.heads, self.kv_heads, self.head_dim = hc_count, heads, kv_heads, head_dim
        self.idx_heads, self.idx_dim = idx_heads, idx_dim
        self.hidden = self.embed.shape[1]
        self.rope = Rope(rot, self.embed.device)
        self.expert_sets = {"bf16": (self.frozen["mtp.layers.0.mlp.experts.gate_up_proj"],
                                     self.frozen["mtp.layers.0.mlp.experts.down_proj"])}
        self.experts = "bf16"
        self.vocab = None  # the served draft's token subset (ids), or None

    def use_experts(self, kind):
        """``bf16`` (checkpoint) or ``nvfp4`` (the served draft's online-quantized experts, emulated)."""
        if kind not in self.expert_sets:
            if kind != "nvfp4":
                raise ValueError(f"unknown experts {kind!r}")
            with torch.no_grad():
                self.expert_sets[kind] = tuple(fake_nvfp4(w) for w in self.expert_sets["bf16"])
        self.experts = kind

    def load_tensors(self, tensors):
        """Replace trainable tensors (checkpoint names) in place; every given tensor must be trainable."""
        with torch.no_grad():
            for name, tensor in tensors.items():
                param = self.params[name.replace(".", "__")]
                if tuple(param.shape) != tuple(tensor.shape):
                    raise ValueError(f"{name}: {tuple(tensor.shape)} for {tuple(param.shape)}")
                param.copy_(tensor.to(param.device).float())

    def w(self, name):
        name = "mtp." + name
        if name in self.frozen:
            return self.frozen[name]
        return self.params[name.replace(".", "__")].to(BF16)

    def export(self):
        return {n: self.params[n.replace(".", "__")].detach().to(BF16).cpu() for n in self.trainable}

    # HyperConnection (GatedResidual): mix -> (block input, injection logits or None)
    def hc_mix(self, prefix, x, weights=None):
        get = weights.__getitem__ if weights is not None else (lambda n: self.w(f"{prefix}.{n}"))
        xn = gemma_rms(x, get("hc_norm.weight"), groups=self.hc)
        down = F.linear(xn, get("input_mix_weight_down.weight"))
        injection = None
        if weights is None and "mtp." + f"{prefix}.block_inject_weight.weight" in self.names:
            injection = F.linear(xn, get("block_inject_weight.weight"))
        lora = down.float() / self.hc
        lora = (lora * torch.sigmoid(lora)).to(x.dtype)
        gate = F.linear(lora, get("input_mix_weight_up.weight"))
        mixed = (torch.sigmoid(gate.float()) * xn.float()).unflatten(-1, (self.hc, self.hidden)).mean(-2)
        return mixed.to(x.dtype), injection

    def hc_combine(self, x, block, injection):
        scale = 2.0 * torch.sigmoid(injection.float() / self.hc)
        out = x.float().unflatten(-1, (self.hc, self.hidden)) + block.float()[:, None, :] * scale[..., None]
        return out.flatten(-2).to(x.dtype)

    def target_sample(self, multi):
        return self.hc_mix(None, multi, weights=self.target_mixer)[0]

    def step_input(self, prev_multi, tokens):
        emb = self.embed[tokens]
        emb = F.linear(gemma_rms(emb, self.w("pre_fc_norm_embedding.weight")), self.w("fc_embedding.weight"))
        hidden = gemma_rms(prev_multi, self.w("pre_fc_norm_hidden.weight"))
        hidden = F.linear(hidden.unflatten(-1, (self.hc, self.hidden)), self.w("fc_hidden.weight"))
        return (hidden + emb[:, None, :]).flatten(-2)

    def qkv(self, block, pos):
        n = block.shape[0]
        qg = F.linear(block, self.w("layers.0.self_attn.q_proj.weight")).view(n, self.heads, 2 * self.head_dim)
        q, gate = qg[..., :self.head_dim], qg[..., self.head_dim:].reshape(n, -1)
        k = F.linear(block, self.w("layers.0.self_attn.k_proj.weight")).view(n, self.kv_heads, self.head_dim)
        v = F.linear(block, self.w("layers.0.self_attn.v_proj.weight")).view(n, self.kv_heads, self.head_dim)
        q = self.rope(gemma_rms(q, self.w("layers.0.self_attn.q_norm.weight")), pos)
        k = self.rope(gemma_rms(k, self.w("layers.0.self_attn.k_norm.weight")), pos)
        return q, gate, k, v

    def index_qk(self, block, pos):
        projected = F.linear(block, self.w("layers.0.self_attn.indexer.index_qk_proj.weight"))
        q = projected[:, :self.idx_heads * self.idx_dim].view(-1, self.idx_heads, self.idx_dim)
        q = self.rope(gemma_rms(q, self.w("layers.0.self_attn.indexer.q_layernorm.weight")), pos)
        return q, projected[:, self.idx_heads * self.idx_dim:]

    def compress(self, raw_keys):
        """Compressed QSA keys of the complete 4-token blocks: RMSNorm(mean) with RoPE at the block start."""
        blocks = raw_keys.shape[0] // RATIO
        pooled = raw_keys[:blocks * RATIO].float().view(blocks, RATIO, -1).mean(1).to(raw_keys.dtype)
        keys = gemma_rms(pooled, self.w("layers.0.self_attn.indexer.k_layernorm.weight"))
        pos = torch.arange(blocks, device=raw_keys.device) * RATIO
        return self.rope(keys[:, None, :], pos)[:, 0, :]

    def moe(self, x):
        n = x.shape[0]
        router = F.linear(x, self.w("layers.0.mlp.gate.weight")).float().softmax(-1)
        weights, ids = router.topk(TOPK_EXPERTS, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        flat = ids.flatten()
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=router.shape[1]).tolist()
        token = order // TOPK_EXPERTS
        xs = x[token]
        w13, w2 = self.expert_sets[self.experts]
        inter = w2.shape[2]
        outs, start = [], 0
        for expert, count in enumerate(counts):
            if count == 0:
                continue
            h = F.linear(xs[start:start + count], w13[expert])
            act = (F.silu(h[:, :inter].float()) * h[:, inter:].float()).to(x.dtype)
            outs.append(F.linear(act, w2[expert]))
            start += count
        routed = torch.cat(outs).float() * weights.flatten()[order][:, None]
        out = torch.zeros(n, x.shape[1], dtype=torch.float32, device=x.device).index_add(0, token, routed)
        g = F.linear(x, self.w("layers.0.mlp.shared_expert.gate_proj.weight"))
        u = F.linear(x, self.w("layers.0.mlp.shared_expert.up_proj.weight"))
        shared = F.linear((F.silu(g.float()) * u.float()).to(x.dtype), self.w("layers.0.mlp.shared_expert.down_proj.weight"))
        shared = shared.float() * torch.sigmoid(F.linear(x, self.w("layers.0.mlp.shared_expert_gate.weight")).float())
        return (out + shared).to(x.dtype)

    def select(self, q_index, compressed, anchor):
        """Step-1 key indices a query anchored at ``anchor`` attends to: top-512 of the complete blocks
        before it (by its own indexer query) plus the <= 3 tail tokens. ([C, S] indices, [C, S] valid)."""
        visible = (anchor + 1) // RATIO
        blocks = compressed.shape[0]
        width = min(BLOCK_TOPK, blocks)
        pieces_idx, pieces_ok = [], []
        if width:
            scores = torch.einsum("chd,nd->chn", q_index.float(), compressed.float()).clamp_min(0).sum(1)
            scores = scores / math.sqrt(self.idx_dim)
            scores.masked_fill_(torch.arange(blocks, device=anchor.device)[None, :] >= visible[:, None], -math.inf)
            top, chosen = scores.topk(width, dim=-1)
            tokens = (chosen[:, :, None] * RATIO + torch.arange(RATIO, device=anchor.device)).flatten(1)
            pieces_idx.append(tokens)
            pieces_ok.append((top > -math.inf)[:, :, None].expand(-1, -1, RATIO).flatten(1))
        tail = visible[:, None] * RATIO + torch.arange(RATIO - 1, device=anchor.device)
        pieces_idx.append(tail)
        pieces_ok.append(tail <= anchor[:, None])
        return torch.cat(pieces_idx, 1), torch.cat(pieces_ok, 1)

    def attend(self, q, keys, values, idx, ok, extra_k, extra_v):
        """Sparse GQA over gathered step-1 keys plus per-row chain keys; q [C,H,D]."""
        c = q.shape[0]
        safe = idx.clamp(0, keys.shape[0] - 1)
        kg, vg = keys[safe], values[safe]  # [C,S,KV,D]
        if extra_k is not None:
            kg, vg = torch.cat((kg, extra_k), 1), torch.cat((vg, extra_v), 1)
            ok = torch.cat((ok, torch.ones(c, extra_k.shape[1], dtype=torch.bool, device=ok.device)), 1)
        group = self.heads // self.kv_heads
        qh = q.view(c, self.kv_heads, group, self.head_dim)
        scores = torch.einsum("cgqd,csgd->cgqs", qh.float(), kg.float()) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~ok[:, None, None, :], -math.inf)
        probs = scores.softmax(-1)
        out = torch.einsum("cgqs,csgd->cgqd", probs, vg.float())
        return out.reshape(c, self.heads * self.head_dim).to(q.dtype)

    def layer_rest(self, x, block, injection, attn_flat, gate):
        attn = F.linear(attn_flat * torch.sigmoid(gate.float()).to(attn_flat.dtype), self.w("layers.0.self_attn.o_proj.weight"))
        hidden = self.hc_combine(x, attn, injection)
        block2, injection2 = self.hc_mix("layers.0.mlp_hyper_connection", hidden)
        multi = self.hc_combine(hidden, self.moe(block2), injection2)
        sample, _ = self.hc_mix("hyper_connection_mixer", multi)
        return sample, multi

    def forward(self, h, tokens, anchors, steps, chunk=512, grad_ckpt=True):
        """Draft steps 1..``steps`` at ``anchors`` (step k consumes x[i+k]); h [L, hc*hidden] target rows.
        Returns per-step sampled hidden [len(anchors), hidden] (k-th for x[i+k+1])."""
        length = h.shape[0]
        pos_all = torch.arange(length - 1, device=h.device)
        x1 = self.step_input(h[:-1], tokens[1:])
        block1, injection1 = self.hc_mix("layers.0.attn_hyper_connection", x1)
        _, _, keys, values = self.qkv(block1, pos_all)
        with torch.no_grad():
            _, raw = self.index_qk(block1, pos_all)
            compressed = self.compress(raw)
        samples, prev_multi, chain_k, chain_v = [], None, [], []
        for step in range(1, steps + 1):
            query_pos = anchors + (step - 1)
            if step == 1:
                x, block, injection = x1[anchors], block1[anchors], injection1[anchors]
            else:
                x = self.step_input(prev_multi, tokens[(anchors + step).clamp(max=length - 1)])
                block, injection = self.hc_mix("layers.0.attn_hyper_connection", x)
            q, gate, k_own, v_own = self.qkv(block, query_pos)
            with torch.no_grad():
                q_index, _ = self.index_qk(block, query_pos)
            if step > 1:
                chain_k.append(k_own[:, None])
                chain_v.append(v_own[:, None])
            extra_k = torch.cat(chain_k, 1) if chain_k else None
            extra_v = torch.cat(chain_v, 1) if chain_v else None
            outs = []
            for a in range(0, anchors.shape[0], chunk):
                sl = slice(a, a + chunk)
                with torch.no_grad():
                    idx, ok = self.select(q_index[sl], compressed, anchors[sl])
                args = (q[sl], keys, values, idx, ok,
                        None if extra_k is None else extra_k[sl], None if extra_v is None else extra_v[sl])
                if grad_ckpt and torch.is_grad_enabled():
                    outs.append(checkpoint(self.attend, *args, use_reentrant=False))
                else:
                    outs.append(self.attend(*args))
            sample, prev_multi = self.layer_rest(x, block, injection, torch.cat(outs), gate)
            samples.append(sample)
        return samples


# --- data ------------------------------------------------------------------------------------------


class Split:
    def __init__(self, data_dir, dump_dir, split):
        data = np.load(Path(data_dir) / f"{split}.npz")
        self.tokens, self.loss, self.offsets = data["tokens"], data["loss"], data["offsets"]
        self.dump = Path(dump_dir) / split
        done = json.loads((self.dump / "index.json").read_text()) if (self.dump / "index.json").is_file() else {}
        self.items = sorted(int(k) for k, v in done.items() if v.get("ok"))

    def sequence(self, index, width, device):
        a, b = int(self.offsets[index]), int(self.offsets[index + 1])
        tokens = torch.from_numpy(self.tokens[a:b].astype(np.int64)).to(device)
        loss = torch.from_numpy(self.loss[a:b].astype(np.bool_)).to(device)
        raw = np.fromfile(self.dump / f"{index:05d}.target.bin", dtype=np.int16)
        h = torch.from_numpy(raw).to(device).view(BF16).view(b - a, width)
        return tokens, loss, h

    def draft_rows(self, index, width, device):
        path = self.dump / f"{index:05d}.draft.bin"
        if not path.is_file():
            return None
        pos = np.fromfile(self.dump / f"{index:05d}.draft.pos", dtype=np.int32)
        rows = torch.from_numpy(np.fromfile(path, dtype=np.int16)).to(device).view(BF16).view(len(pos), width)
        return torch.from_numpy(pos.astype(np.int64)).to(device), rows


def anchors_for(loss, steps):
    """Anchor i with x[i+1..i+steps+1] all generated (the draft chain stays inside one model output)."""
    length = loss.shape[0]
    ok = torch.ones(length, dtype=torch.bool, device=loss.device)
    for j in range(1, steps + 2):
        shifted = torch.zeros_like(ok)
        shifted[:length - j] = loss[j:]
        ok &= shifted
    return torch.nonzero(ok).flatten()


# --- loss and metrics ------------------------------------------------------------------------------


def soft_ce(draft_hidden, target_hidden, lm_head):
    d = F.linear(draft_hidden, lm_head).float().log_softmax(-1)
    with torch.no_grad():
        t = F.linear(target_hidden, lm_head).float().softmax(-1)
    return -(t * d).sum(-1)


def step_loss(draft_hidden, target_hidden, lm_head, chunk=1024):
    total = draft_hidden.new_zeros((), dtype=torch.float32)
    for a in range(0, draft_hidden.shape[0], chunk):
        total = total + checkpoint(soft_ce, draft_hidden[a:a + chunk], target_hidden[a:a + chunk], lm_head,
                                   use_reentrant=False).sum()
    return total / max(1, draft_hidden.shape[0])


def draft_ids(model, hidden):
    """Greedy draft tokens of sampled hidden rows: over the full head and over the served subset (or None)."""
    logits = F.linear(hidden, model.lm_head).float()
    full = logits.argmax(-1)
    sub = model.vocab[logits[:, model.vocab].argmax(-1)] if model.vocab is not None else None
    return full, sub


@torch.no_grad()
def draft_metrics(model, h, tokens, anchors, samples, chunk=2048):
    """Per-anchor greedy draft d_k, match flags and p^T of the verifying target position, per vocabulary
    (``full``; ``v16k`` with ``model.vocab``)."""
    steps = len(samples)
    names = ["full"] + (["v16k"] if model.vocab is not None else [])
    rows = {name: {"p": [], "match": []} for name in names}
    for k, sample in enumerate(samples, start=1):
        parts = {name: ([], []) for name in names}
        for a in range(0, anchors.shape[0], chunk):
            sl = slice(a, a + chunk)
            verify = anchors[sl] + k
            ids, probs = sampling_probs(F.linear(model.target_sample(h[verify]), model.lm_head))
            for name, d in zip(names, draft_ids(model, sample[sl])):
                parts[name][0].append((probs * (ids == d[:, None])).sum(-1))
                parts[name][1].append(d == tokens[verify + 1])
        for name in names:
            rows[name]["p"].append(torch.cat(parts[name][0]))
            rows[name]["match"].append(torch.cat(parts[name][1]))
    out = {}
    for name in names:
        p = torch.stack(rows[name]["p"])  # [steps, N]
        match = torch.stack(rows[name]["match"]).float()
        cumulative = [p[0]]
        for k in range(1, steps):
            cumulative.append(match[:k].prod(0) * p[k])
        out[name] = {"n": int(anchors.shape[0]), "p": p.sum(1).tolist(), "match": match.sum(1).tolist(),
                     "cum": [float(c.sum()) for c in cumulative], "first": match[0].bool()}
    return out


def summarize(acc):
    n = max(1, acc["n"])
    cum = [c / n for c in acc["cum"]]
    return {"anchors": acc["n"], "p_accept_step": [round(p / n, 4) for p in acc["p"]],
            "top1_match_step": [round(m / n, 4) for m in acc["match"]],
            "cumulative_accept": [round(c, 4) for c in cum], "accepted_per_draft": round(sum(cum), 4),
            "tokens_per_iteration": [round(1 + sum(cum[:k + 1]), 4) for k in range(len(cum))],
            "loss": round(acc["loss"] / max(1, acc["loss_n"]), 5) if acc.get("loss_n") else None}


def merge(acc, m):
    m = {k: v for k, v in m.items() if k != "first"}
    if not acc:
        return {**m, "loss": 0.0, "loss_n": 0}
    acc["n"] += m["n"]
    for key in ("p", "match", "cum"):
        acc[key] = [a + b for a, b in zip(acc[key], m[key])]
    return acc


@torch.no_grad()
def evaluate(model, split, items, args, device, width, fake_fp8=False, collect=None):
    """``{vocabulary: summary}`` over ``items``; ``primary`` repeats the served vocabulary's (v16k when set).
    ``collect`` (a uint8 array over the split's tokens, 255 = no draft) receives, at each drafted token's
    position (anchor + 2), whether the served vocabulary's step-1 draft equals the recorded token."""
    accs = {}
    for index in items:
        tokens, loss_mask, h = split.sequence(index, width, device)
        if fake_fp8:
            h = fp8_roundtrip(h, model.hc)
        anchors = anchors_for(loss_mask, args.steps)
        if anchors.numel() == 0:
            continue
        try:
            samples = model(h, tokens, anchors, args.steps, grad_ckpt=False)
            metrics = draft_metrics(model, h, tokens, anchors, samples)
            losses = [step_loss(s, model.target_sample(h[anchors + k + 1]), model.lm_head)
                      for k, s in enumerate(samples)]
        except torch.cuda.OutOfMemoryError:
            log("oom", phase="eval", index=index, tokens=int(tokens.numel()))
            torch.cuda.empty_cache()
            continue
        if collect is not None:
            first = metrics["v16k" if "v16k" in metrics else "full"]["first"]
            collect[int(split.offsets[index]) + 2 + anchors.cpu().numpy()] = first.cpu().numpy()
        weighted = float(sum(w * x for w, x in zip(args.step_weights, losses))) * anchors.numel()
        for name, m in metrics.items():
            accs[name] = merge(accs.get(name), m)
            accs[name]["loss"] += weighted
            accs[name]["loss_n"] += int(anchors.numel())
    if not accs:
        return {}
    out = {name: summarize(acc) for name, acc in accs.items()}
    return {**out, "primary": out["v16k" if "v16k" in out else "full"]}


@torch.no_grad()
def served_step1(model, split, items, args, device, width):
    """The served draft's own step-1 rows (dumped during prefill: its served experts and attention) against
    this implementation with the current weights, on the evaluation anchors of every sequence that has them:
    P(accept 1) of both, greedy-token agreement and the hidden rows' distance."""
    names = ["full"] + (["v16k"] if model.vocab is not None else [])
    total = {"anchors": 0, "sequences": 0, "cos": 0.0, "rel": 0.0, "beyond_2k": 0,
             **{f"{who}_{name}": 0.0 for who in ("served", "ours", "agree", "agree_beyond_2k") for name in names}}
    for index in items:
        found = split.draft_rows(index, model.hidden, device)
        if found is None:
            continue
        pos, rows = found
        tokens, loss_mask, h = split.sequence(index, width, device)
        where, row = np.unique(pos.cpu().numpy(), return_index=True)  # the first row seen per position
        inside = (where >= 0) & (where < h.shape[0])
        first = torch.full((h.shape[0],), -1, dtype=torch.long, device=device)
        first[torch.from_numpy(where[inside]).to(device)] = torch.from_numpy(row[inside]).to(device)
        anchors = anchors_for(loss_mask, args.steps)
        anchors = anchors[first[anchors] >= 0]
        if anchors.numel() == 0:
            continue
        served = rows[first[anchors]]
        ours = torch.cat([model(h, tokens, anchors[a:a + 4096], 1, grad_ckpt=False)[0]
                          for a in range(0, anchors.numel(), 4096)])
        total["anchors"] += int(anchors.numel())
        total["sequences"] += 1
        total["cos"] += float(F.cosine_similarity(ours.float(), served.float(), dim=-1).sum())
        total["rel"] += float(((ours.float() - served.float()).norm(dim=-1)
                               / served.float().norm(dim=-1).clamp_min(1e-6)).sum())
        long = anchors >= TOKEN_BUDGET + RATIO
        total["beyond_2k"] += int(long.sum())
        for a in range(0, anchors.numel(), 2048):
            sl = slice(a, a + 2048)
            ids, probs = sampling_probs(F.linear(model.target_sample(h[anchors[sl] + 1]), model.lm_head))
            for name, ds, do in zip(names, draft_ids(model, served[sl]), draft_ids(model, ours[sl])):
                total[f"served_{name}"] += float((probs * (ids == ds[:, None])).sum())
                total[f"ours_{name}"] += float((probs * (ids == do[:, None])).sum())
                total[f"agree_{name}"] += float((ds == do).sum())
                total[f"agree_beyond_2k_{name}"] += float(((ds == do) & long[sl]).sum())
    n = max(1, total["anchors"])
    out = {"anchors": total["anchors"], "sequences": total["sequences"], "experts": model.experts,
           "cos_mean": round(total["cos"] / n, 5), "rel_err_mean": round(total["rel"] / n, 5)}
    for name in names:
        out[name] = {"p_accept1_served": round(total[f"served_{name}"] / n, 4),
                     "p_accept1_ours": round(total[f"ours_{name}"] / n, 4),
                     "argmax_agree": round(total[f"agree_{name}"] / n, 4),
                     "argmax_agree_beyond_2k": round(total[f"agree_beyond_2k_{name}"] / max(1, total["beyond_2k"]), 4)}
    return out


def read_overlay(path):
    """``{name: BF16 tensor}`` of an exported head (``arena.weights_overlay`` directory)."""
    path = Path(path)
    manifest = json.loads((path / "overlay-manifest.json").read_text())
    file = path / manifest["overlay"]["file"]
    header = wo.read_header(file)
    out = {}
    for name in manifest["tensors"]:
        spec = header[1][name]
        if spec["dtype"] != "BF16":
            raise ValueError(f"{name}: {spec['dtype']}")
        out[name] = torch.frombuffer(bytearray(wo.read_tensor(file, name, header)), dtype=BF16).reshape(spec["shape"])
    return out, manifest["overlay"]["sha256"]


def fp8_roundtrip(h, hc):
    """E4M3 with one scale per row and HC branch (the storage option for the full run's dumps)."""
    x = h.float().unflatten(-1, (hc, -1))
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
    return (x / scale).to(torch.float8_e4m3fn).float().mul(scale).flatten(-2).to(h.dtype)


# --- main ------------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3, help="draft depth trained and evaluated")
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-anchors", type=int, default=6144, help="anchors per optimizer step (random subset)")
    ap.add_argument("--step-weights", type=float, nargs="+", default=[1.0, 0.8, 0.6])
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-subset", type=int, default=8, help="select-split sequences in the probe")
    ap.add_argument("--deadline-epoch", type=float, default=0.0)
    ap.add_argument("--reserve-seconds", type=float, default=420.0, help="kept for the final eval and export")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--base-identity", default="", help="JSON {repo, revision} for the overlay manifest")
    ap.add_argument("--experts", choices=("bf16", "nvfp4"), default="bf16",
                    help="frozen routed experts in training and evaluation (nvfp4: the served draft's, emulated)")
    ap.add_argument("--alt-experts", choices=("bf16", "nvfp4"), default=None,
                    help="also evaluate the probe sequences with these experts, before and after")
    ap.add_argument("--draft-vocab", default=None, help="JSON with the served draft's token ids (metrics v16k)")
    ap.add_argument("--select-split", default="eval", help="split of the probe and of the state selection")
    ap.add_argument("--report-splits", nargs="+", default=["eval"], help="splits evaluated in full before and after")
    ap.add_argument("--curve-stride", type=int, default=0,
                    help="every n-th sequence of the first report split joins the probe (reported, never selected on)")
    ap.add_argument("--draft-splits", nargs="*", default=["eval"],
                    help="splits whose dumped served-draft rows are compared with this implementation")
    ap.add_argument("--served-head", choices=("before", "after"), default="before",
                    help="which weights the dump server's draft had (after: the --init-overlay head)")
    ap.add_argument("--init-overlay", default=None, help="an exported head loaded after the 'before' evaluation")
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--fp8-check", action="store_true", help="the pilot's FP8-dump storage check")
    ap.add_argument("--save-accept", action="store_true",
                    help="write accept-<split>-<before|after>.npy: the step-1 draft's match per token position")
    args = ap.parse_args(argv)
    started = time.time()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    persist = os.environ.get("ARC_VALIDATION_OUTPUT")  # the Modal run volume: survives a failed job
    persist = Path(persist) / "draft-head" if persist else None
    report = {"args": vars(args), "started_epoch": started}

    def save():
        (out / "draft-head-train.json").write_text(json.dumps(report, indent=1))
        if persist is not None:
            try:
                persist.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(out / "draft-head-train.json", persist / "draft-head-train.json")
            except OSError as exc:
                log("persist_error", error=str(exc)[:200])

    config = json.loads((Path(args.model_dir) / "config.json").read_text())
    text = config.get("text_config", config)
    names = mtp_names(args.model_dir)
    tensors = read_tensors(args.model_dir, [*names, EMBED, LM_HEAD,
                                            *(TARGET_MIXER + s for s in ("hc_norm.weight",
                                                                         "input_mix_weight_down.weight",
                                                                         "input_mix_weight_up.weight"))], device)
    trainable = [n for n in names if n not in FROZEN_MTP]
    model = Draft(tensors, set(trainable), hc_count=text["hc_count"], heads=text["num_attention_heads"],
                  kv_heads=text["num_key_value_heads"], head_dim=text["head_dim"],
                  idx_heads=text["indexer_n_heads"], idx_dim=text["indexer_head_dim"],
                  rot=int(text["head_dim"] * text.get("partial_rotary_factor", 0.25)))
    width = text["hc_count"] * text["hidden_size"]
    if args.draft_vocab:
        ids = json.loads(Path(args.draft_vocab).read_text())["ids"]
        model.vocab = torch.tensor(ids, dtype=torch.long, device=device)
    report["trainable_parameters"] = int(sum(p.numel() for p in model.params.values()))
    report["trainable_tensors"] = trainable
    report["load_seconds"] = round(time.time() - started, 1)
    log("loaded", seconds=report["load_seconds"], trainable=report["trainable_parameters"])
    split_names = list(dict.fromkeys([args.select_split, *args.report_splits, *args.draft_splits]))
    splits = {name: Split(args.data_dir, args.dump_dir, name) for name in split_names}
    train = Split(args.data_dir, args.dump_dir, "train") if (Path(args.data_dir) / "train.npz").is_file() else None
    report["sequences"] = {"train": len(train.items) if train else 0, **{k: len(v.items) for k, v in splits.items()}}
    if not splits[args.select_split].items or not splits[args.report_splits[0]].items:
        raise RuntimeError("no held-out sequences were dumped")
    probe_items = {args.select_split: splits[args.select_split].items[:args.eval_subset]}
    if args.curve_stride:
        probe_items["curve"] = splits[args.report_splits[0]].items[::args.curve_stride]

    def probe():
        """The select split's probe and (reported only) the curve sequences of the first report split."""
        rows = {args.select_split: evaluate(model, splits[args.select_split], probe_items[args.select_split], args,
                                            device, width)}
        if "curve" in probe_items:
            rows["curve"] = evaluate(model, splits[args.report_splits[0]], probe_items["curve"], args, device, width)
        return rows

    def full(tag):
        rows = {}
        for name in args.report_splits:
            bits = np.full(len(splits[name].tokens), 255, np.uint8) if args.save_accept else None
            rows[name] = evaluate(model, splits[name], splits[name].items, args, device, width, collect=bits)
            if bits is not None:
                np.save(out / f"accept-{name}-{tag}.npy", bits)
                if persist is not None:
                    try:
                        persist.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(out / f"accept-{name}-{tag}.npy", persist / f"accept-{name}-{tag}.npy")
                    except OSError as exc:
                        log("persist_error", error=str(exc)[:200])
        if args.alt_experts and args.alt_experts != args.experts:
            model.use_experts(args.alt_experts)
            rows[f"probe_{args.alt_experts}"] = probe()
            model.use_experts(args.experts)
        log(f"eval_{tag}", **{name: row.get("primary") or row for name, row in rows.items()})
        return rows

    def served(tag):
        if args.served_head != tag or not any((splits[name].dump / f"{i:05d}.draft.bin").is_file()
                                              for name in args.draft_splits for i in splits[name].items):
            return
        rows = {}
        for name in args.draft_splits:
            for kind in dict.fromkeys(["bf16", "nvfp4", args.experts]):
                model.use_experts(kind)
                rows.setdefault(name, {})[kind] = served_step1(model, splits[name], splits[name].items, args, device,
                                                               width)
        model.use_experts(args.experts)
        report["served_step1"] = {"head": tag, "splits": rows}
        log("served_step1", head=tag, **rows)

    t0 = time.time()
    model.use_experts(args.experts)
    report["experts_seconds"] = round(time.time() - t0, 1)
    served("before")
    save()
    report["evals"] = {"before": full("before")}
    report["eval_before"] = report["evals"]["before"][args.report_splits[0]]["primary"]
    report["probe_before"] = probe()
    if args.fp8_check:
        report["eval_before_fp8_dumps"] = evaluate(model, splits[args.select_split], probe_items[args.select_split],
                                                   args, device, width, fake_fp8=True)
    save()
    if args.init_overlay:
        loaded, digest = read_overlay(args.init_overlay)
        model.load_tensors(loaded)
        report["init_overlay"] = {"path": args.init_overlay, "tensors": len(loaded), "overlay_sha256": digest}
        log("init_overlay", **report["init_overlay"])
    if args.train_steps and (train is None or not train.items):
        raise RuntimeError("no training sequences were dumped")

    params = list(model.params.values())
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
    total = args.train_steps

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - args.warmup) / max(1, total - args.warmup))))

    def selected(rows):
        return rows[args.select_split].get("primary", {}).get("accepted_per_draft", 0.0)

    def checkpoint_state(step):
        folder = out / "checkpoints"
        folder.mkdir(exist_ok=True)
        path = folder / f"step-{step:05d}.pt"
        torch.save(model.export(), path)
        if persist is not None:
            try:
                (persist / "checkpoints").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, persist / "checkpoints" / path.name)
            except OSError as exc:
                log("persist_error", error=str(exc)[:200])
        log("checkpoint", step=step, path=str(path))

    best = {"accepted_per_draft": selected(report["probe_before"]), "step": 0,
            "state": {k: v.detach().clone() for k, v in model.params.items()}}
    curve = [{"step": 0, **report["probe_before"]}]
    history, order, step, step_times = [], [], 0, []
    while step < total:
        if not order:
            order = list(train.items)
            random.shuffle(order)
        index = order.pop()
        t0 = time.time()
        if args.deadline_epoch and step_times:
            left = args.deadline_epoch - args.reserve_seconds - time.time()
            if left < 2 * float(np.mean(step_times[-10:])):
                report["stopped"] = f"deadline after {step} steps"
                break
        tokens, loss_mask, h = train.sequence(index, width, device)
        anchors = anchors_for(loss_mask, args.steps)
        if anchors.numel() == 0:
            continue
        if anchors.numel() > args.max_anchors:
            anchors = anchors[torch.randperm(anchors.numel(), device=device)[:args.max_anchors]].sort().values
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        try:
            samples = model(h, tokens, anchors, args.steps)
            losses = [step_loss(s, model.target_sample(h[anchors + k + 1]).detach(), model.lm_head)
                      for k, s in enumerate(samples)]
            loss = sum(w * x for w, x in zip(args.step_weights, losses))
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            log("oom", phase="train", index=index, tokens=int(tokens.numel()), anchors=int(anchors.numel()))
            samples = losses = loss = None
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            report.setdefault("oom_skips", []).append(index)
            continue
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        step += 1
        step_times.append(time.time() - t0)
        history.append({"step": step, "seq": index, "anchors": int(anchors.numel()), "loss": round(float(loss.detach()), 5),
                        "step_losses": [round(float(x), 5) for x in losses], "grad_norm": round(float(norm), 4),
                        "lr": lr_at(step - 1), "seconds": round(step_times[-1], 2)})
        if step % 10 == 0 or step == 1:
            log("step", **history[-1])
        if step % args.eval_every == 0 and step < total:
            model.eval()
            rows = probe()
            curve.append({"step": step, **rows})
            log("eval_subset", step=step, **{k: v.get("primary") for k, v in rows.items()})
            if selected(rows) > best["accepted_per_draft"]:
                best = {"accepted_per_draft": selected(rows), "step": step,
                        "state": {k: v.detach().clone() for k, v in model.params.items()}}
            model.train()
        if args.checkpoint_every and step % args.checkpoint_every == 0 and step < total:
            checkpoint_state(step)
        report["history"], report["curve"] = history, curve
        if step % 25 == 0:
            save()
    report["train_steps_done"] = step
    report["train_seconds"] = round(sum(step_times), 1)
    report["seconds_per_step"] = round(float(np.mean(step_times)), 3) if step_times else None
    final = probe()
    curve.append({"step": step, **final})
    report["curve"] = curve
    if step and args.checkpoint_every:
        checkpoint_state(step)
    if selected(final) >= best["accepted_per_draft"] or (args.init_overlay and not step):
        best = {"accepted_per_draft": selected(final), "step": step, "state": None}
    if best["state"] is not None:
        with torch.no_grad():
            for k, v in best["state"].items():
                model.params[k].copy_(v)
    report["selected_step"] = best["step"]
    report["evals"]["after"] = full("after")
    report["eval_after"] = report["evals"]["after"][args.report_splits[0]]["primary"]
    served("after")
    save()
    if not args.no_export and (step or not args.init_overlay):
        manifest_path = Path(args.model_dir) / "MODEL_MANIFEST.json"
        tensors_out = {}
        for name, tensor in model.export().items():
            tensors_out[name] = ("BF16", list(tensor.shape), tensor.contiguous().view(torch.int16).numpy().tobytes())
        if manifest_path.is_file():
            inventory = json.loads(manifest_path.read_text())
            base_files = {row["path"]: {"bytes": int(row["size"]), "sha256": row["sha256"]}
                          for row in inventory["files"]}
            identity = {"repo": inventory.get("repo"), "revision": inventory.get("revision")}
        else:
            base_files = {p.name: {"bytes": p.stat().st_size, "sha256": wo.sha256_file(p)}
                          for p in Path(args.model_dir).iterdir() if p.is_file()}
            identity = json.loads(args.base_identity or '{"repo": "local", "revision": "local"}')
        t0 = time.time()
        manifest = wo.write_overlay(out / "overlay", args.model_dir, tensors_out, base_identity=identity,
                                    base_files=base_files, log=lambda m: log("export", message=m),
                                    extra={"producer": "arena.draft_head_train", "selected_step": best["step"],
                                           "eval_before": report["eval_before"], "eval_after": report["eval_after"],
                                           "experts": args.experts,
                                           "served_as": "server.draft_head.overlay (loader-level, MTP tensors only)"})
        report["export"] = {"seconds": round(time.time() - t0, 1), "tensors": len(tensors_out),
                            "overlay_sha256": manifest["overlay"]["sha256"], "bytes": manifest["overlay"]["bytes"],
                            "identical_to_base": manifest["identical_to_base"]}
        if persist is not None:
            try:
                shutil.copytree(out / "overlay", persist / "overlay", dirs_exist_ok=True)
            except OSError as exc:
                log("persist_error", error=str(exc)[:200])
        log("exported", **report["export"])
    report["elapsed_seconds"] = round(time.time() - started, 1)
    save()


if __name__ == "__main__":
    main()
