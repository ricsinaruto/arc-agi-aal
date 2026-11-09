"""Built-preview timing gate (30 Oct: prev9h v2 skipped its trial at 485 s: notebook --max-seconds 32,400 left
31,795 s of bootstrap budget for a 31,800 s wall + 300 s margin). For a BUILT Kaggle preview bundle, evaluates every
gate between notebook start and trial start with the bundle's own code and the notebook's own --max-seconds, at
setup times up to SETUP_MAX seconds (the v2 preview reached the trial at ~485 s):
1. bootstrap_arena.py: budget = max_seconds - end_seconds(120) - setup; skip if budget < trial_wall + margin(300);
2. arena.run: hard_end = start + max_seconds; non-competition trials need hard_end - now >= wall + 480;
3. arena.competition.gameplay_seconds (public preview): available = hard_end - now - cleanup >= preview_seconds;
4. the bootstrap's subprocess timeout min(budget, trial_wall + TRIAL_GRACE) covers the wall.
The formulas are asserted verbatim in the bundle's source, so a code change fails this check instead of passing it.
Usage: prevcheck.py BUNDLE [SETUP_MAX]; exit 1 on failure. Also imported by tests/test_prevcheck.py."""
import json
import re
import sys
from pathlib import Path

SETUP_MAX = 1200.0
FORMULAS = ("budget = started + args.max_seconds - float(reserves.get('end_seconds', 120)) - time.time()",
            "margin = float(reserves.get('trial_seconds', 300))",
            "skip = budget < trial_wall + margin or trial_wall <= 0")


def notebook_max_seconds(bundle):
    book = json.loads((bundle / "experiment.ipynb").read_text())
    text = "".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in book["cells"])
    found = re.findall(r"'--max-seconds', '(\d+)'", text)
    if len(found) != 1:
        raise AssertionError(f"notebook --max-seconds found {found}")
    return float(found[0])


def check(bundle, setup_max=SETUP_MAX):
    bundle = Path(bundle)
    boot = (bundle / "bootstrap_arena.py").read_text()
    for f in FORMULAS:
        assert f in boot, f"bootstrap formula changed: {f}"
    assert "and hard_end - time.time() < wall + 480):" in (bundle / "src/arena/run.py").read_text()
    grace = float(re.search(r"^TRIAL_GRACE = (\d+)", boot, re.M).group(1))
    study = json.loads((bundle / "configs/comparison.json").read_text())
    max_s = notebook_max_seconds(bundle)
    reserves = study.get("bootstrap_reserves") or {}
    end_s, margin = float(reserves.get("end_seconds", 120)), float(reserves.get("trial_seconds", 300))
    sys.path.insert(0, str(bundle / "src"))
    try:
        from arena.competition import gameplay_seconds
    finally:
        sys.path.pop(0)
    rows = []
    for trial in study["trials"]:
        trial_wall = float(trial.get("wall_seconds", study["wall_seconds"]))
        comp = study.get("competition") if trial.get("competition", True) else None
        for setup in (300.0, 600.0, float(setup_max)):
            budget = max_s - end_s - setup
            assert not (budget < trial_wall + margin or trial_wall <= 0), (
                f"{trial['name']}: bootstrap skips at setup {setup:.0f} s (budget {budget:.0f} < wall {trial_wall:.0f} + {margin:.0f})")
            assert min(budget, trial_wall + grace) >= trial_wall, f"{trial['name']}: subprocess timeout below the wall"
            if comp:
                got = gameplay_seconds({**study, "competition": comp}, hard_end=max_s, now=setup)
                assert got == comp["preview_seconds"], (trial["name"], got)
            else:
                assert max_s - setup >= study["wall_seconds"] + 480, f"{trial['name']}: run.py time gate at {setup}"
            rows.append({"trial": trial["name"], "setup": setup, "budget": budget, "wall": trial_wall})
    # startup line (30 Oct, preview v2 load 655 s): the setup time at which the bootstrap skips the trial
    limit = min(max_s - end_s - r["wall"] - margin for r in rows)
    return {"max_seconds": max_s, "rows": rows, "slack_at_setup_max": rows[-1]["budget"] - rows[-1]["wall"] - margin,
            "trial_start_limit": limit}


if __name__ == "__main__":
    try:
        r = check(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else SETUP_MAX)
    except AssertionError as exc:
        print(f"PREVCHECK FAILED {sys.argv[1]}: {exc}")
        sys.exit(1)
    print(f"PREVCHECK OK {Path(sys.argv[1]).name}: max_seconds {r['max_seconds']:.0f}, slack at setup "
          f"{SETUP_MAX:.0f} s = {r['slack_at_setup_max']:.0f} s")
    est = float(sys.argv[3]) if len(sys.argv) > 3 else None  # measured/predicted setup (notebook start -> trial start)
    print(f"STARTUP trial-start limit {r['trial_start_limit']:.0f} s"
          + (f"; expected setup {est:.0f} s, margin {r['trial_start_limit'] - est:.0f} s" if est is not None else ""))
    if est is not None and est > r["trial_start_limit"]:
        sys.exit(1)
