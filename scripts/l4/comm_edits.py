"""Options ``custom_ar`` (arc2.l4_comm: vLLM's custom all-reduce over PCIe peer access) and ``nccl_p2p_level``
(NCCL_P2P_LEVEL) for arc2/l4_serving.py, as text edits that apply to the v49 bundle's file and to the branch's.
usage: comm_edits.py PATH (edits in place; each anchor must occur exactly once)"""
import sys
from pathlib import Path

EDITS = [
    ("warm_rht=False, context=None", "warm_rht=False, context=None, custom_ar=False, nccl_p2p_level=None"),
    ("    if logits_topk and not 32 <= int(logits_topk) <= 512:",
     "    if nccl_p2p_level not in (None, \"PIX\", \"PXB\", \"PHB\", \"NODE\", \"SYS\"):\n"
     "        raise ValueError(\"nccl_p2p_level is one of PIX, PXB, PHB, NODE, SYS\")\n"
     "    if logits_topk and not 32 <= int(logits_topk) <= 512:"),
    ('"warmup_dump": warmup_dump or None, "warm_rht": bool(warm_rht),',
     '"warmup_dump": warmup_dump or None, "warm_rht": bool(warm_rht),\n'
     '                            "custom_ar": bool(custom_ar), "nccl_p2p_level": nccl_p2p_level,'),
    ('"warm_rht", "context"', '"warm_rht", "context", "custom_ar", "nccl_p2p_level"'),
    ('           "kv4_guard": "arc2_l4_finite = arc2.l4_finite:register"}',
     '           "custom_ar": "arc2_l4_comm = arc2.l4_comm:register",  # l4 arm: all-reduce over peer access\n'
     '           "kv4_guard": "arc2_l4_finite = arc2.l4_finite:register"}'),
    ('        if options.get("kv4_guard"):\n',
     '        if options.get("nccl_p2p_level"):  # l4 arm: Kaggle\'s L4 pairs are PHB with peer access; NCCL used SHM\n'
     '            env["NCCL_P2P_LEVEL"] = str(options["nccl_p2p_level"])\n'
     '        if options.get("custom_ar"):  # arc2.l4_comm (plugin): vLLM\'s custom all-reduce on four PCIe cards\n'
     '            env["ARC2_L4_CUSTOM_AR"] = "1"\n'
     '        if options.get("kv4_guard"):\n'),
]


def apply(path):
    path = Path(path)
    text = path.read_text()
    for old, new in EDITS:
        if text.count(old) != 1:
            raise SystemExit(f"{path}: anchor found {text.count(old)} times: {old[:60]!r}")
        text = text.replace(old, new)
    compile(text, str(path), "exec")
    path.write_text(text)


if __name__ == "__main__":
    apply(sys.argv[1])
