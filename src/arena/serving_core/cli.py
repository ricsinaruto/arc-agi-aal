"""Explicit serving commands: ``serve-native`` (the pinned ARC fork at 262,144) and ``preflight``."""

import argparse
import asyncio
import json
from pathlib import Path

from .config import Config
from .serving import preflight, start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve-native")
    serve.add_argument("--working", required=True)
    serve.add_argument("--input-root", default="/kaggle/input")
    serve.add_argument("--kv-cache-memory-bytes", type=int, default=0,
                       help="Explicit BF16 KV limit; 0 uses GPU memory profiling")
    check = sub.add_parser("preflight")
    check.add_argument("--base-url", default=Config().base_url)
    args = parser.parse_args()
    if args.command == "serve-native":
        start(args.working, Path(args.input_root), kv_cache_memory_bytes=args.kv_cache_memory_bytes)
    else:
        print(json.dumps(asyncio.run(preflight(Config(base_url=args.base_url))), indent=2))


if __name__ == "__main__":
    main()
