"""Build the approved nine-hour Modal studies without launching them."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from arena.campaign import ARMS, INT3_ARMS, MTP_ARMS, WALL, components
from build_arena import build, kaggle_files


def served_blocks(arm):
    """The int8-block pool the build projects the startup on (int4 blocks are these bytes re-cut): unpruned-hero
    1,335 -> 1,270; prune25 1,755 -> 1,669; prune37 2,132 -> 2,028 (the estimate in arena.campaign blocks2-prune37);
    prune25 MTP-1 1,638 -> 1,438 blocks of 13 QSA-layer pages (the measured arena-spec-sweep-p25-mtp1-v3 pool)."""
    if arm == "blocks2-plans-prune50-int3":
        # The prune25 int8 pool + the 6,144 further pruned experts (6,144 x 2,764,800 B / 21,196,800 B per int8 block =
        # 801) + the 12,288 kept experts' int3 savings (12,288 x 614,400 B = 356): 2,912 -> 2,770 int4 blocks
        # (projection; the arm's floor comes from the measured 1,647 + 762 + 339 = 2,748).
        return 1755 + 801 + 356
    if arm == "blocks2-plans-prune37-int3":
        # The prune37 int8 projection + the 15,360 kept experts' int3 savings (15,360 x 614,400 B / 21,196,800 B per
        # int8 block = 445): 2,577 -> 2,451 int4 blocks (as the Kaggle blocks2-p37-int3 build).
        return 2132 + 445
    if arm in INT3_ARMS and arm not in MTP_ARMS:
        # MTP-0: the prune25 int8 pool + the int3 experts' 534 int8 blocks: 2,289 -> 2,177 int4 blocks (projection;
        # the arm's floor comes from the measured 1,647 + 508).
        return 1755 + 534
    if arm in INT3_ARMS:
        # + the int3 experts' saved bytes (18,432 served experts x 614,400 B = 11,324,620,800 B / 21,196,800 B per int8
        # block = 534.3): 2,172 -> 1,907 MTP-1 int4 blocks (the estimate in arena.campaign's int3 arm).
        return 1638 + 534
    if arm in MTP_ARMS:
        return 1638
    return 2132 if "prune37" in arm else 1755 if "prune25" in arm else 1335


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", choices=ARMS, nargs="+")
    ap.add_argument("--dependencies", action="store_true")
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()
    for arm in args.arms:
        name = f"arena-campaign-{arm}-9h-{args.version}"
        output, study = build(name, paired_30m=True, dependencies=args.dependencies,
                              arena_variant="fixed-2x-mtp0-wide-kv8-hero",
                              hero_v2=components(arm), hero_v2_served_blocks=served_blocks(arm),
                              exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"],
                              opening_burst_requests=3, wall_seconds=WALL, clone_instances=110,
                              campaign_arm=arm)
        digest = kaggle_files(output, study, smoke=False,
                              runtime_dataset="richardcsaky/arc-arena-experiments-runtime")
        print(json.dumps({"arm": arm, "output": str(output), "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
