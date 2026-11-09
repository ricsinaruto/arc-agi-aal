"""Summaries of the c110-path sweeps (28 September): paired differences and the robustness checks.

    python scripts/arena_analysis/c110_report.py [--dir docs/validation-20260927/c110-path] <what>

``paired NAME A B [PER]``: mean paired difference A - B over common seeds; with PER (runs per bootstrap roster) also
the share of rosters where A beats B. ``boot``: per profile and efficiency, the bootstrap-roster means and how often
each conclusion holds. ``heldout``: in-sample vs held-out Gittins constants. ``sens``: rankings under scaled clear
rates. ``ninehour``: MTP-0 six slots against MTP-3 two slots at 9 hours.
"""

import argparse
import json
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parents[2] / "docs" / "validation-20260927" / "c110-path"


def load(name, d=DIR):
    return json.loads(Path(d, f"{name}.result.json").read_text())


def scores(r, label):
    return np.array(r[label]["scores"])


def paired(r, a, b, per=None):
    d = scores(r, a) - scores(r, b)
    out = {"diff": round(float(d.mean()), 3), "se": round(float(d.std() / np.sqrt(len(d))), 3),
           "p_run": round(float(np.mean(d > 0)), 2)}
    if per:
        blocks = d.reshape(-1, per).mean(axis=1)
        out["p_roster"] = round(float(np.mean(blocks > 0)), 2)
        out["rosters"] = len(blocks)
    return out


def boot_summary(d=DIR):
    rows = []
    for i in range(5):
        path = Path(d, f"boot-p{i}.result.json")
        if not path.exists():
            continue
        r = load(f"boot-p{i}", d)
        per = 6
        for eff in ("1", "2"):
            g, f, c = f"gittins-c110 x{eff}", f"fixed-2x x{eff}", f"gittins-c110 cap600 x{eff}"
            roster_means = {k: scores(r, k).reshape(-1, per).mean(axis=1) for k in (g, f, c)}
            rows.append({
                "profile": i, "eff": eff,
                "gittins": round(float(scores(r, g).mean()), 2), "fixed-2x": round(float(scores(r, f).mean()), 2),
                "cap600": round(float(scores(r, c).mean()), 2),
                "gittins roster p10/p50/p90": [round(float(x), 2) for x in np.percentile(roster_means[g], [10, 50, 90])],
                "rosters gittins > fixed-2x": paired(r, g, f, per)["p_roster"],
                "rosters cap1200 >= cap600": float(np.mean(roster_means[g] >= roster_means[c])),
                "rosters gittins mean >= 3": float(np.mean(roster_means[g] >= 3.0)),
                "runs gittins >= 3": round(float(np.mean(scores(r, g) >= 3.0)), 2),
            })
    return rows


def heldout_summary(d=DIR):
    out = []
    for s in range(8):
        path = Path(d, f"heldout-s{s}.result.json")
        if not path.exists():
            continue
        r = load(f"heldout-s{s}", d)
        cfg = json.loads(Path(d, f"heldout-s{s}.config.json").read_text())
        for half in (0, 1):
            other = 1 - half
            # Constants fitted without this half (fit-excl-H{half}); hysteresis chosen on the other half's roster with
            # the constants fitted without the other half, i.e. tuned without seeing this half.
            tune = {h: float(np.mean(r[f"R{other} gittins fit-excl-H{other} h{h}"]["scores"])) for h in ("1", "4", "8")}
            h_best = max(tune, key=tune.get)
            held = r[f"R{half} gittins fit-excl-H{half} h{h_best}"]["mean"]
            insample = r[f"R{half} gittins all-data h4"]["mean"]
            fixed = r[f"R{half} fixed-2x"]["mean"]
            out.append({"split": s, "half": cfg["halves"][half], "fixed-2x": fixed, "gittins in-sample": insample,
                        "gittins held-out": held, "held-out h": h_best,
                        "held-out h4": r[f"R{half} gittins fit-excl-H{half} h4"]["mean"],
                        "gap in - held": round(insample - held, 3), "held-out - fixed-2x": round(held - fixed, 3)})
    return out


def sens_summary(d=DIR):
    out = []
    for i in range(3):
        path = Path(d, f"sens-p{i}.result.json")
        if not path.exists():
            continue
        r = load(f"sens-p{i}", d)
        for eff in ("0.5", "0.75", "1.33"):
            keys = [f"fixed-2x x{eff}", f"gittins-c110 x{eff}", f"gittins-c110 cap600 x{eff}", f"known-rate x{eff}"]
            out.append({"profile": i, "eff": eff, **{k.rsplit(" ", 1)[0]: r[k]["mean"] for k in keys},
                        "gittins - fixed-2x": paired(r, keys[1], keys[0]),
                        "cap1200 - cap600": paired(r, keys[1], keys[2])})
    return out


def ninehour_summary(d=DIR):
    out = {}
    for name in ("ninehour-distinct", "ninehour-population", "ninehour-boot23", "ninehour-x0.5", "ninehour-x1.33"):
        path = Path(d, f"{name}.result.json")
        if not path.exists():
            continue
        r = load(name, d)
        per = 2 if name == "ninehour-boot23" else None
        res = {k: v["mean"] for k, v in r.items()}
        res["fixed-2x MTP-0 - MTP-3"] = paired(r, "fixed-2x MTP-0 6 slots", "fixed-2x MTP-3 2 slots", per)
        res["gittins-c110 MTP-0 - fixed-2x MTP-3"] = paired(r, "gittins-c110 MTP-0 6 slots", "fixed-2x MTP-3 2 slots",
                                                            per)
        res["gittins-c110 MTP-0 - fixed-2x MTP-0"] = paired(r, "gittins-c110 MTP-0 6 slots", "fixed-2x MTP-0 6 slots",
                                                            per)
        out[name] = res
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(DIR))
    parser.add_argument("what")
    parser.add_argument("args", nargs="*")
    a = parser.parse_args()
    if a.what == "paired":
        per = int(a.args[3]) if len(a.args) > 3 else None
        print(json.dumps(paired(load(a.args[0], a.dir), a.args[1], a.args[2], per)))
        return
    fn = {"boot": boot_summary, "heldout": heldout_summary, "sens": sens_summary, "ninehour": ninehour_summary}[a.what]
    res = fn(a.dir)
    print(json.dumps(res, indent=1))
    Path(a.dir, f"summary-{a.what}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
