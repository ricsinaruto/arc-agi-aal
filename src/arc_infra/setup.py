"""Bound the existing reviewed serving setup without importing its game harness."""

import argparse
import os
import re
import signal
from pathlib import Path

from .serving import NATIVE_CONTEXT, SERVER_CONTEXT, SUPPORTED_CONTEXTS, load_recipe
from .serving_fork import GPU_MEMORY_UTILIZATION_ENV, fork_identity, stage_fork_bundle

UTILIZATION = re.compile(r"^0\.\d{1,3}$")


def parse(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--mtp-tokens", type=int, choices=(1, 3), default=3)
    parser.add_argument("--generation-slots", type=int, choices=(8, 12, 16), default=8)
    parser.add_argument(
        "--max-model-len", type=int, choices=SUPPORTED_CONTEXTS, default=SERVER_CONTEXT
    )
    parser.add_argument(
        "--kv-cache-gib",
        type=int,
        default=5,
        help="Explicit KV cache size; 0 lets vLLM size it from its utilization fraction",
    )
    parser.add_argument(
        "--prefix-caching",
        action="store_true",
        help="Enable vLLM prefix caching (the pinned Duck profile keeps it off)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        default=None,
        help=(
            "Run the ARC fork of the pinned script at this utilization with a vLLM-profiled "
            "KV cache (requires --kv-cache-gib 0); omitted = the unmodified pinned script"
        ),
    )
    args = parser.parse_args(argv)
    if args.kv_cache_gib < 0:
        parser.error("--kv-cache-gib must be non-negative")
    if args.gpu_memory_utilization is not None:
        value = args.gpu_memory_utilization.strip()
        if not UTILIZATION.match(value) or not 0.5 <= float(value) <= 0.99:
            parser.error("--gpu-memory-utilization must be a fraction between 0.5 and 0.99")
        if args.kv_cache_gib:
            parser.error("--gpu-memory-utilization needs --kv-cache-gib 0 (profiled allocation)")
    elif args.max_model_len == NATIVE_CONTEXT:
        parser.error("--max-model-len 262144 needs the fork's profiled cache (--gpu-memory-utilization)")
        args.gpu_memory_utilization = value
    return args


def configure(recipe, max_model_len):
    """Override the pinned script's context constant (read at call time) before ``main``."""
    if max_model_len not in SUPPORTED_CONTEXTS:
        raise ValueError("Unsupported server context")
    recipe.ANALYZER_CONTEXT = max_model_len
    return recipe


def recipe_for(args, input_root=Path("/kaggle/input")):
    """The pinned script, or (with a utilization) the staged ARC fork at that utilization."""
    options = {
        "mtp_tokens": args.mtp_tokens,
        "generation_slots": args.generation_slots,
        "kv_cache_memory_bytes": args.kv_cache_gib * 1024**3,
        "prefix_caching": args.prefix_caching,
    }
    if args.gpu_memory_utilization is not None:
        os.environ[GPU_MEMORY_UTILIZATION_ENV] = args.gpu_memory_utilization
        options["bundle_dir"] = stage_fork_bundle(input_root, Path(args.output) / "fork-bundle")
        options["setup_sha256"] = fork_identity()["fork_sha256"]
    return load_recipe(args.output, input_root, **options)


def main():
    args = parse()
    recipe = configure(recipe_for(args), args.max_model_len)
    signal.signal(signal.SIGALRM, recipe._setup_timeout_handler)
    signal.alarm(recipe.SETUP_TOTAL_TIMEOUT)
    try:
        recipe.main()
    except BaseException:
        recipe.stop_owned_server()
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
