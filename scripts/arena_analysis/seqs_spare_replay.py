"""Exact-gate replay at max_num_seqs = streams + k on another served pool (29 September; prefix-unit-int4.md §4).

The closed-loop stream replay of ``gate_replay.py`` (phase-2 tenures of a recorded run, holds scaled by a decode
step model), with three changes:
- the spare sequences ``k`` above the focus streams is a parameter (the run: 2);
- the served layout (cost table and 0.9 gate) comes from a ``cache-layout.json`` whose ``num_blocks`` may be
  overridden (the int4 pools after larger CUDA-graph captures);
- the slowdown reference is the recorded run's own focus-stream count.

    PYTHONPATH=src uv run python scripts/arena_analysis/seqs_spare_replay.py <trial dir> --layout <cache-layout.json> \
        --blocks 1269 --reference 23 --streams 36 38 40 42 44 --spare 2 4 --iteration 14.1 0.417 --out result.json
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gate_replay", HERE / "gate_replay.py")
gr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial_dir")
    parser.add_argument("--layout", required=True, help="cache-layout.json of the served pool (costs, groups)")
    parser.add_argument("--blocks", type=int, default=None, help="override num_blocks (the gate is margin x it)")
    parser.add_argument("--margin", type=float, default=0.9, help="the KV gate's share of the served blocks")
    parser.add_argument("--reference", type=int, required=True, help="the recorded run's focus streams")
    parser.add_argument("--streams", type=int, nargs="+", required=True)
    parser.add_argument("--spare", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--iteration", type=float, nargs=2, default=(14.1, 0.417), metavar=("A_MS", "B_MS"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--pools", nargs="+", default=["all", "late"])
    parser.add_argument("--wall", type=float, default=None, help="the recorded run's wall (default: its last event)")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    from arena.cache import BlockBudget

    gr.ITERATION_MS = tuple(args.iteration)
    reference = args.reference
    gr.slowdown = lambda streams, reference=reference: ((gr.ITERATION_MS[0] + gr.ITERATION_MS[1] * streams)
                                                         / (gr.ITERATION_MS[0] + gr.ITERATION_MS[1] * reference))
    report, _ = gr.load_run(args.trial_dir)
    layout = copy.deepcopy(json.loads(Path(args.layout).read_text()))
    if args.blocks:
        layout["num_blocks"] = args.blocks
    budget = BlockBudget(layout, args.margin)
    switch = report["scheduler"]["switch_wall_seconds"]
    wall = args.wall or min(7200.0, report["scheduler"]["events"][-1]["wall_seconds"])
    pools = {"all": gr.phase2_tenures(report, budget)}
    if "late" in args.pools:
        pools["late"] = gr.phase2_tenures(report, budget, since=switch + (wall - switch) / 2)
    result = {"trial_dir": args.trial_dir, "layout": args.layout, "num_blocks": layout["num_blocks"],
              "capacity_blocks": budget.capacity, "margin": args.margin, "iteration_ms": list(gr.ITERATION_MS), "reference_streams": reference,
              "switch": switch, "wall": wall, "tenures": {k: len(v) for k, v in pools.items()}, "rows": []}
    for name in args.pools:
        for spare in args.spare:
            for n in args.streams:
                row = gr.replay_streams(pools[name], budget.capacity, n, slots=n + spare, seeds=args.seeds,
                                        start=switch, duration=wall - switch)
                row = {"pool": name, "spare": spare, **{k: row[k] for k in (
                    "streams", "max_num_seqs", "hold_factor", "time_share_waiting", "time_share_waiting_on_blocks",
                    "waits", "stream_time_lost_to_waits", "mean_running", "reserved_blocks", "final_length_blocks")}}
                result["rows"].append(row)
                print(json.dumps({k: row[k] for k in ("pool", "spare", "streams", "max_num_seqs",
                                                      "stream_time_lost_to_waits", "time_share_waiting_on_blocks",
                                                      "mean_running")}
                                 | {"mean_wait": row["waits"]["mean_seconds"], "p90": row["waits"]["p90_seconds"]}),
                      flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
