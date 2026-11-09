"""The GPU-path digest of a heal bundle (rule of 21 Oct: the real-trainer gate runs only when a change
touches the GPU path: trainer forward/backward, kernels, memory). Hashes the source of the GPU-path functions and
classes of heal_kd.py (AST, so comments and the read/report code outside them do not count) and whole kernel and
model modules. The launcher runs the gate when the digest is not in its passed list.

    python gpu_digest.py BUNDLE_RL1_DIR      -> prints the digest and the hashed symbols
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

# heal_kd.py symbols on the GPU path: the model and its forward/backward, the losses, the step loop and its memory,
# the optimizer and the checkpoint/resume state
HEAL_SYMBOLS = (
    "Runner", "Held", "Capture", "GradSync", "RowAdam", "MtpHead",
    "_kd_chunk", "kd_loss", "bucket_kl", "sampled_reverse_terms", "sampler_logprob",
    "train_sequence", "dp_sequence", "dp_dummy", "slot_plan", "span_windows", "runner_extend", "commit_captured",
    "capture_on", "recompute_policy", "comm_flags", "shard", "load_student", "load_mtp", "restore_fp32",
    "save_student", "local_params", "load_local_params", "wrap_adam_state", "checkpoint", "checkpoint_async",
    "_cpu_copy", "wait_checkpoint", "run_train", "lr_at", "wsd_lr", "cosine_lr", "install_ple", "ple_rows",
    "ple_slice", "load_teacher", "load_teacher_cpu", "teacher_to", "truncate_loops", "loop_mask",
)
WHOLE_FILES = ("flash_qlora_job.py", "rl1_sparse.py", "rl1_sparse2.py", "rl1_gdn.py", "native_experts.py",
               "flash_config.json")


def symbols(source, names):
    tree = ast.parse(source)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            found[node.name] = ast.dump(node, include_attributes=False)
    return found


def digest(rl1_dir):
    rl1 = Path(rl1_dir)
    h = hashlib.sha256()
    found = symbols((rl1 / "heal_kd.py").read_text(), set(HEAL_SYMBOLS))
    missing = sorted(set(HEAL_SYMBOLS) - set(found))
    for name in sorted(found):
        h.update(name.encode() + b"\0" + found[name].encode() + b"\0")
    files = []
    for name in WHOLE_FILES:
        f = rl1 / name
        if f.exists():
            h.update(name.encode() + b"\0" + f.read_bytes() + b"\0")
            files.append(name)
    return h.hexdigest()[:16], {"symbols": len(found), "missing": missing, "files": files}


if __name__ == "__main__":
    d, info = digest(sys.argv[1])
    print(d)
    print(json.dumps(info), file=sys.stderr)
