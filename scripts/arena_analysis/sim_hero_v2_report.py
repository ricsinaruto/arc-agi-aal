"""Markdown tables for sim-hero-v2.md from the ``sim_hero_v2.py`` result files (29 September)."""

import json
from pathlib import Path

import numpy as np
import sim_hero_v2 as hv


def _load(d, name):
    p = Path(d, f"{name}.result.json")
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v):
    return f"{v['mean']:.2f} ({v['p10']:.1f} / {v['p50']:.1f} / {v['p90']:.1f})"


def pd(r, a, b, per=None):
    x = hv.paired(r, a, b, per)
    share = x.get("p_roster", x["p_run"])
    return f"{x['diff']:+.2f} ± {x['se']:.2f} ({int(round(100 * share))}%)", x


def tuning(d, summary):
    sel = hv.selected(d)
    print("### Selected configurations\n")
    print("| Chosen on | Configuration |\n| --- | --- |")
    for k, c in sel.items():
        print(f"| {k} | `{hv.tag(c)}` |")
    summary["selected"] = {k: hv.tag(c) for k, c in sel.items()}
    print("\n### Held-out folds (tune2 seeds): the configuration chosen on one fold, played on the other\n")
    print("| Horizon | Chosen on | Played on | Cell | rank | fixed-2x sw6 | fixed-2x sw2 | fair q20K | rank − best fixed-2x |")
    print("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    out = {}
    for hz in ("hour", "nine"):
        for f_own, f_other in (("F0", "F1"), ("F1", "F0")):
            cfg = sel[f"{f_own}-{hz}"]
            for played in (f_own, f_other):
                r = _load(d, f"tune2-{played}-{hz}")
                for cell in hv.TUNE_CELLS:
                    a = f"{cell} | {hv.tag(cfg)}"
                    fx = max(("fixed-2x sw6", "fixed-2x sw2"), key=lambda k: r[f"{cell} | {k}"]["mean"])
                    txt, x = pd(r, a, f"{cell} | {fx}")
                    print(f"| {hz} | {f_own} | {played}{' (held out)' if played != f_own else ''} | {cell} | "
                          f"{r[a]['mean']:.2f} | {r[f'{cell} | fixed-2x sw6']['mean']:.2f} | "
                          f"{r[f'{cell} | fixed-2x sw2']['mean']:.2f} | {r[f'{cell} | fair q20K']['mean']:.2f} | {txt} |")
                    out[f"{hz} | {f_own} -> {played} | {cell}"] = x
    summary["heldout_folds"] = out
    # how flat is the optimum: the spread of the top-10 on tune2 (own fold)
    print("\n### tune2: the ten best configurations per fold and horizon (mean over the two tuning cells)\n")
    top = hv.top_configs(d)
    for hz in ("hour", "nine"):
        for f in hv.folds():
            r = _load(d, f"tune2-{f}-{hz}")
            union = sorted({hv.tag(c) for v in top.values() for c in v}, key=lambda t: -hv.objective(r, t))
            fx = max(np.mean([r[f"{c} | {k}"]["mean"] for c in hv.TUNE_CELLS]) for k in ("fixed-2x sw6", "fixed-2x sw2"))
            print(f"- {hz} {f} (best fixed-2x {fx:.2f}): " + "; ".join(
                f"`{t}` {hv.objective(r, t):.2f}" for t in union[:5]))
    print()


def final(d, summary):
    for hz, title in (("hour", "One hour, c110"), ("nine", "Nine hours, 110 distinct games")):
        r = _load(d, f"final-{hz}")
        if r is None:
            continue
        cells, arms = [], []
        for label in r:
            c, a = label.split(" | ")
            cells += [c] if c not in cells else []
            arms += [a] if a not in arms else []
        print(f"### {title} (mean (p10 / p50 / p90))\n")
        short = [a.split(" ")[0] if a.startswith("rank[") else a for a in arms]
        print("| Cell | " + " | ".join(short) + " | rank[nine] − best fixed-2x | rank[nine] − fair |")
        print("| --- | " + " | ".join("---:" for _ in arms) + " | ---: | ---: |")
        rk = next(a for a in arms if a.startswith("rank[nine]"))
        for c in cells:
            fx = max(("fixed-2x sw6", "fixed-2x sw2"), key=lambda k: r[f"{c} | {k}"]["mean"])
            t1, x1 = pd(r, f"{c} | {rk}", f"{c} | {fx}")
            fr = max(("fair q20K", "fair q40K"), key=lambda k: r[f"{c} | {k}"]["mean"])
            t2, x2 = pd(r, f"{c} | {rk}", f"{c} | {fr}")
            print(f"| {c} | " + " | ".join(fmt(r[f"{c} | {a}"]) for a in arms) + f" | {t1} ({fx[-3:]}) | {t2} |")
            summary[f"final-{hz} | {c}"] = {a: {k: r[f"{c} | {a}"][k] for k in ("mean", "p10", "p50", "p90", "se")}
                                           for a in arms}
            summary[f"final-{hz} | {c}"]["rank_vs_fixed"] = x1
        print()
        # windows at max streams
        print(f"**{title}: 64K / 48K / 32K at max streams (best arm per cell; paired difference against the 64K cell "
              f"of the same pool and max_num_seqs, same arm)**\n")
        print("| Pool, max_num_seqs | Arm | 64K | 48K | 32K | 48K − 64K | 32K − 64K |")
        print("| --- | --- | --- | --- | --- | ---: | ---: |")
        for pool in ("p12", "p25"):
            for cap in (25, 48):
                by_w = {w: next(c for c in cells if c.startswith(f"{pool} cap{cap} ") and c.endswith(f"x{w}K"))
                        for w in (64, 48, 32)}
                for arm in ("fixed-2x sw6", "fixed-2x sw2", rk):
                    row = [f"{by_w[w].split(' ')[2]} {r[f'{by_w[w]} | {arm}']['mean']:.2f}" for w in (64, 48, 32)]
                    dd = [pd(r, f"{by_w[w]} | {arm}", f"{by_w[64]} | {arm}")[0] for w in (48, 32)]
                    print(f"| {pool}, {cap} | {arm.split(' ')[0] if arm.startswith('rank') else arm} | "
                          + " | ".join(row) + " | " + " | ".join(dd) + " |")
        print()


def robust(d, summary):
    print("### Robustness: rank[nine] / rank[hour] − the better fixed-2x switch, paired (share of rosters or runs)\n")
    checks = [("1 h bootstrap", "robust-hour-boot", hv.cp.BOOT_PER), ("1 h x0.5", "robust-hour-x0.5", None),
              ("1 h x1.33", "robust-hour-x1.33", None), ("1 h tail 40K x0.5", "robust-hour-tail", None),
              ("9 h bootstrap 23", "robust-nine-boot", 2), ("9 h x0.5", "robust-nine-x0.5", None),
              ("9 h x1.33", "robust-nine-x1.33", None), ("9 h tail 40K x0.5", "robust-nine-tail", None)]
    print("| Check | " + " | ".join(hv.ROBUST_CELLS) + " |\n| --- | " + " | ".join("---" for _ in hv.ROBUST_CELLS) + " |")
    rob = {}
    for title, name, per in checks:
        r = _load(d, name)
        if r is None:
            continue
        for arm in ("rank[nine]", "rank[hour]"):
            row = []
            for c in hv.ROBUST_CELLS:
                fx = max(("fixed-2x sw6", "fixed-2x sw2"), key=lambda k: r[f"{c} | {k}"]["mean"])
                txt, x = pd(r, f"{c} | {arm}", f"{c} | {fx}", per)
                row.append(f"{r[f'{c} | {fx}']['mean']:.2f} → {txt}")
                rob[f"{name} | {c} | {arm}"] = x
            print(f"| {title} {arm} | " + " | ".join(row) + " |")
    summary["robust"] = rob
    print()
    for hz in ("hour", "nine"):
        r = _load(d, f"robust-{hz}-serving")
        if r is None:
            continue
        print(f"### Serving sensitivity ({hz})\n\n| Cell variant | fixed-2x sw6 | fixed-2x sw2 | rank[nine] |\n| --- | ---: | ---: | ---: |")
        rows = {}
        for label, v in r.items():
            c, a = label.split(" | ")
            rows.setdefault(c, {})[a] = v["mean"]
        for c, v in rows.items():
            print(f"| {c} | " + " | ".join(f"{v[a]:.2f}" if a in v else "" for a in ("fixed-2x sw6", "fixed-2x sw2",
                                                                                      "rank[nine]")) +
                  ("" if "rank[nine]" in v else " " + ", ".join(f"{a} {x:.2f}" for a, x in v.items())) + " |")
        summary[f"serving-{hz}"] = rows
        print()


def ticks(d, summary):
    """§8: every tick arm against the 60 s tick (paired), with swaps, re-admissions, re-prefill and swap delay."""
    per = {"hour": {"base": None, "boot": hv.cp.BOOT_PER}, "nine": {"base": None, "boot": 2}}
    hours = {"hour": 1.0, "nine": 9.0}
    out = {}
    for hz in ("hour", "nine"):
        for kind in ("base", "boot"):
            r = _load(d, f"tick-{hz}-{kind}")
            if r is None:
                continue
            print(f"\n### ticks {hz} {kind}\n")
            print("| Cell | Arm | Mean (p10 / p90) | − 60 s | Swaps / h | Re-admissions / h | Re-prefilled tokens / h | "
                  "Re-prefill share | Swap delay mean / p90 (s) | Outgoing tokens since clear |")
            print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
            for label, v in r.items():
                cell, arm = label.split(" | ")
                s, h = v.get("stats", {}), hours[hz]
                if arm == "rank tick 60s":
                    dd = "-"
                else:
                    dd, x = pd(r, label, f"{cell} | rank tick 60s", per[hz][kind])
                    out[f"{hz}-{kind} | {label}"] = x
                scale = 3.0 if "x3" in arm else 0.0 if "x0" in arm else 1.0
                reprefill = s.get("readmits", 0) * hv.TICK_CELLS[cell]["readmit"][1] * hv.PREFILL_RATE * scale
                print(f"| {cell} | {arm} | {v['mean']:.2f} ({v['p10']:.2f} / {v['p90']:.2f}) | {dd} | "
                      f"{s.get('swaps', float('nan')) / h:.1f} | {s.get('readmits', 0) / h:.1f} | "
                      f"{reprefill / h / 1e3:,.0f}K | {100 * s.get('reprefill_share', 0):.1f}% | "
                      f"{s.get('swap_delay_mean', float('nan')):.0f} / {s.get('swap_delay_p90', float('nan')):.0f} | "
                      f"{s.get('out_tokens_since_clear', float('nan')) / 1e3:.1f}K |")
    if out:
        summary["ticks"] = out
        Path(d, "tick-summary.json").write_text(json.dumps(out, indent=1))


def switches(d, summary):
    """§9: fixed-2x on the token clock under every opening rule, paired against the six-clear switch."""
    per = {"hour": {"base": None, "boot": hv.cp.BOOT_PER}, "nine": {"base": None, "boot": 2}}
    out = {}
    for hz in ("hour", "nine"):
        for kind in ("base", "boot"):
            r = _load(d, f"switch-{hz}-{kind}")
            if r is None:
                continue
            print(f"\n### switch {hz} {kind}\n")
            print("| Cell | Opening | Mean (p10 / p90) | − six-clear | Switch (s) | Cleared at switch | Focus slots: kept / "
                  "backlog | Games with a level-1 clear (end) | Levels |")
            print("| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |")
            for label, v in r.items():
                cell, arm = label.split(" | ")
                s = v.get("stats", {})
                if arm == "sw6":
                    dd = "-"
                else:
                    dd, x = pd(r, label, f"{cell} | sw6", per[hz][kind])
                    out[f"{hz}-{kind} | {label}"] = x
                c1 = v.get("cleared1", {})
                print(f"| {cell} | {arm} | {v['mean']:.2f} ({v['p10']:.2f} / {v['p90']:.2f}) | {dd} | "
                      f"{s.get('switch_seconds', float('nan')):.0f} | {s.get('cleared_at_switch', float('nan')):.1f} | "
                      f"{s.get('kept_at_switch', float('nan')):.1f} / {s.get('backlog_at_switch', float('nan')):.1f} | "
                      f"{c1.get('mean', float('nan')):.1f} | {v['levels']:.1f} |")
    if out:
        summary["switch"] = out
        Path(d, "switch-summary.json").write_text(json.dumps(out, indent=1))


def report(d):
    summary = {}
    for fn in (tuning, final, robust, ticks, switches):
        try:
            fn(d, summary)
        except SystemExit as e:
            print(f"({fn.__name__}: {e})")
    Path(d, "summary.json").write_text(json.dumps(summary, indent=1))
