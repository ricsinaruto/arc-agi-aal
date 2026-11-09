"""prelaunch_check.py: the round's fast checks on the exact launcher values, before any record or app (c7d/c7e, 24 Oct:
two run ids burned on a refused trainer env flag and checkpoint steps below the resume step). Env: E (ARC_HEAL_ENV
pairs), CKAT, STEP0, TOTAL; WIN (the loss window): the mask and memory sizing at the longest unit (E's HEAL_MAX_UNIT,
c7m 24 Oct). Exit 1 with the reason."""
import re
import ast
import os
import sys

SRC = "./.claude/worktrees/sft-heal/scripts/posttrain/modal_rl1.py"
tree = ast.parse(open(SRC).read())
ns = {}
exec(compile(ast.Module([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "env_pairs"], []), SRC, "exec"), ns)
bad, env = [], {}
try:
    env = ns["env_pairs"](os.environ.get("E", ""))
except ValueError as e:
    bad.append(f"trainer env: {e}")
step0, total = int(os.environ.get("STEP0") or 0), int(os.environ.get("TOTAL") or 0)
ck = [int(x) for x in (os.environ.get("CKAT") or "").split(",") if x.strip()]
late = [s for s in ck if s <= step0 or (total and s > total)]
if late:
    bad.append(f"checkpoint-at {late} outside steps {step0 + 1}..{total}")
# c7m (24 Oct): the MTP forward builds a [queries <= WIN, keys <= unit] bool mask; the plain-causal check once summed it
# whole as int64 (49.4 GiB at a 120,556-token unit). Units are bounded by HEAL_MAX_UNIT (the play's context window;
# longer pieces are dropped), the mask by MASK_MAX_GIB, the check's row-chunk temporaries by 4 GiB, and the trainer
# peak predicted at the longest unit: base + per-token x unit + mask + check temporaries, >= 5% of the card free.
# Base: c7n steps 1-3 (per-step peaks per rank 155-237 GiB at units 72-115K; ranks differ by up to 78 GiB at one
# length, so the peak is not unit-driven): the largest, 236.75 GiB at 114,793 tokens, minus its mask and temporaries.
G, sizing = 2 ** 30, ""
win = int(os.environ.get("WIN") or 0)
if win:
    unit = int(env.get("HEAL_MAX_UNIT") or 0)
    rl1 = open(SRC.replace("modal_rl1.py", "rl1_trainer.py")).read()
    whole = [f for f in ("rl1_trainer.py", "heal_kd.py")
             if re.search(r"\[0, 0\]\.sum\(-1\)", open(SRC.replace("modal_rl1.py", f)).read())]
    rows = int(re.search(r"^PLAIN_CAUSAL_ROWS = (\d+)", rl1, re.M).group(1))
    mask, temp = win * unit, rows * unit * 9
    static, per_tok = float(os.environ.get("MEM_STATIC_GIB", 229.5)), float(os.environ.get("MEM_PER_TOKEN_GIB", 0))
    card = float(os.environ.get("CARD_GIB", 267.68))
    pred = static + per_tok * unit + (mask + temp) / G
    if not unit:
        bad.append("HEAL_MAX_UNIT unset: unit length, MTP mask and memory unbounded")
    if whole:
        bad.append(f"whole-mask sum in {whole}")
    if mask > float(os.environ.get("MASK_MAX_GIB", 12)) * G:
        bad.append(f"MTP mask {win} x {unit} = {mask / G:.1f} GiB > MASK_MAX_GIB")
    if temp > 4 * G:
        bad.append(f"mask check temporaries {rows} x {unit} x 9 B = {temp / G:.1f} GiB > 4")
    if pred > 0.95 * card:
        bad.append(f"trainer peak at a {unit}-token unit predicted {pred:.0f} GiB > 0.95 x {card} GiB")
    sizing = f"; unit <= {unit}, MTP mask {mask / G:.1f} GiB, check temps {temp / G:.2f} GiB, peak ~{pred:.0f}/{card} GiB"
if bad:
    print("PRELAUNCH REFUSED: " + "; ".join(bad))
    sys.exit(1)
print(f"prelaunch OK: env pairs, {len(ck)} checkpoint steps in {step0 + 1}..{total}{sizing}")
