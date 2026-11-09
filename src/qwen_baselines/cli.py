"""Explicit commands only. Preparing artifacts never submits GPU work."""

import argparse
import asyncio
import json
from pathlib import Path

from .config import GAMES, Config
from .runner import comparison
from .serving import preflight, start


def prolong_profile():
    from prolong_qwen.profile import P_DEFAULTS

    return P_DEFAULTS


def retrodict_profile():
    from retrodict_qwen.profile import R_DEFAULTS

    return R_DEFAULTS


def schema_profile():
    from schema_qwen.profile import S_DEFAULTS

    return S_DEFAULTS


PROFILES = {"P": prolong_profile, "R": retrodict_profile, "S": schema_profile}


def config_from_args(args, settings):
    """The run Config: profile settings win over the A/B/C command-line defaults.

    The P/R/S profiles carry their own values for keys the parser also exposes (for
    example ``actions_per_turn``); passing both to ``Config`` raised a TypeError at the
    start of the Retrodict v5 kernel (18 September).
    """
    explicit = {
        "base_url": args.base_url,
        "gameplay_seconds": args.seconds,
        "text_grid": args.text_grid,
        "image_pixels": args.image_pixels,
        "actions_per_turn": args.actions_per_turn or None,
        "history_tokens": args.history_tokens or None,
        "preserve_thinking": args.preserve_thinking,
    }
    return Config(**{**explicit, **settings})


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
    run = sub.add_parser("run")
    run.add_argument(
        "--baseline", nargs="+", choices=["A", "B", "C", "P", "R", "S", "all"], required=True
    )
    run.add_argument("--games", nargs="+", default=list(GAMES))
    run.add_argument("--games-dir", required=True)
    run.add_argument("--seconds", type=float, default=600)
    run.add_argument("--output", required=True)
    run.add_argument("--base-url", default=Config().base_url)
    run.add_argument("--pi-dir", default=str(Path(__file__).resolve().parents[2] / "pi"))
    run.add_argument("--node", default="node")
    # Observation/loop settings; defaults are the current decisions (see Config).
    run.add_argument("--text-grid", action="store_true", help="Attach the 64-row text grid")
    run.add_argument("--image-pixels", type=int, default=Config().image_pixels)
    run.add_argument(
        "--actions-per-turn",
        type=int,
        default=Config().actions_per_turn,
        help="Baseline A game actions executed per response; 0 removes the cap",
    )
    run.add_argument(
        "--history-tokens",
        type=int,
        default=Config().history_tokens,
        help="A/B prompt-token cap before oldest exchanges are trimmed; 0 uses the window",
    )
    run.add_argument("--preserve-thinking", action="store_true")
    # Coding-agent settings; P/R/S defaults come from their profile modules.
    run.add_argument("--compaction-tokens", type=int, default=None)
    run.add_argument("--bash-seconds", type=float, default=None)
    run.add_argument("--bash-max-seconds", type=float, default=None)
    run.add_argument("--scientific", action="store_true", help="Expose numpy to sandbox Python")
    run.add_argument("--grid-mode", choices=["ascii", "hex", "num"], default=None)
    run.add_argument("--max-actions", type=int, default=None)
    run.add_argument("--plan-cap", type=int, default=None)
    run.add_argument("--call-turns", type=int, default=None)
    args = parser.parse_args()
    if args.command == "serve-native":
        start(args.working, Path(args.input_root), kv_cache_memory_bytes=args.kv_cache_memory_bytes)
    elif args.command == "preflight":
        print(json.dumps(asyncio.run(preflight(Config(base_url=args.base_url))), indent=2))
    else:
        baselines = list("ABC") if args.baseline == ["all"] else args.baseline
        settings = {}
        if len(baselines) == 1 and baselines[0] in PROFILES:
            settings.update(PROFILES[baselines[0]]())
        elif any(b in PROFILES for b in baselines):
            raise SystemExit("Run P, R and S separately: each carries its own defaults")
        for name in (
            "compaction_tokens",
            "bash_seconds",
            "bash_max_seconds",
            "grid_mode",
            "max_actions",
            "plan_cap",
            "call_turns",
        ):
            if getattr(args, name) is not None:
                settings[name] = getattr(args, name)
        if args.scientific:
            settings["scientific_python"] = True
        cfg = config_from_args(args, settings)
        asyncio.run(preflight(cfg))
        asyncio.run(
            comparison(
                cfg,
                baselines,
                args.games,
                args.games_dir,
                args.output,
                pi_dir=args.pi_dir,
                node=args.node,
            )
        )


if __name__ == "__main__":
    main()
