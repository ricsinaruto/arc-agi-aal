"""Run the existing selected-game Duck-transfer fleet against SGLang."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

from arc_infra.config import Config
from arc_infra.fleet import run_replicas
from arc_infra.metrics import GpuMonitor
from arc_infra.model import ModelClient

from .metrics import parse_sglang_metrics, summary


class FleetMonitor(GpuMonitor):
    """Keep the fleet's sampling boundaries and GPU telemetry, using SGLang counters."""

    async def sample(self):
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.get(self.metrics_url, timeout=3)
                response.raise_for_status()
            item = {"elapsed_seconds": time.monotonic() - self.start,
                    **parse_sglang_metrics(response.text)}
            self.samples.append(item)
            with (self.output / "telemetry.jsonl").open("a") as file:
                file.write(json.dumps(item) + "\n")
            (self.output / "sglang-metrics.txt").write_text(response.text)
        except httpx.HTTPError as exc:
            self.errors.append({"elapsed_seconds": time.monotonic() - self.start,
                                "error": type(exc).__name__})

    def summary(self):
        return summary(self.samples, errors=self.errors)


async def run(source, folder, games_dir, games, seconds, base_url, model, config_path, backend="sglang"):
    from duck_transfer import loader

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(str(Path(source) / config_path))
    cfg.base_url, cfg.model, cfg.gameplay_seconds = base_url, model, seconds
    profile = loader.configure(cfg)
    # Fail before gameplay if the backend cannot count the actual tool/template schema.
    client = ModelClient(cfg)
    try:
        count = await client.count_tokens(
            [{"role": "user", "content": "Inspect the board."}], time.monotonic() + 30,
            tools=[{"type": "function", "function": {"name": "python", "description": "Python",
                    "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                                   "required": ["code"]}}}],
            template_kwargs={"enable_thinking": True, "reasoning_effort": cfg.reasoning_effort},
        )
        (folder / "api-preflight.json").write_text(json.dumps({"token_count": count}))
    finally:
        await client.close()
    from duck_transfer.transfer import TransferDuckAgent

    report = await run_replicas(
        cfg, games_dir, folder, sys.executable, (FleetMonitor if backend == "sglang" else GpuMonitor)(folder, cfg.base_url),
        agent_class=TransferDuckAgent, agent_profile="duck-transfer", profile_source=profile,
        sessions=25, games=games, clock_sessions=25,
    )
    report["status"] = "error" if report["failed_sessions"] or report["task_errors"] else "complete"
    report["selected_games"] = list(games)
    report["timing_normalization"] = "competition-equivalent clock of a 25-session fleet"
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "output", "games-dir", "base-url", "model", "config"):
        p.add_argument("--" + field, required=True)
    p.add_argument("--backend", choices=("sglang", "vllm"), default="sglang")
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--game", action="append", required=True)
    args = p.parse_args()
    result = asyncio.run(run(args.source, args.output, args.games_dir, args.game, args.seconds,
                             args.base_url, args.model, args.config, args.backend))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
