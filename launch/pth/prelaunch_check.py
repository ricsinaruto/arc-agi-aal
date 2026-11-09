"""prelaunch_check.py: the round's fast checks on the exact launcher values, before any record or app (c7d/c7e, 24 Oct:
two run ids burned on a refused trainer env flag and checkpoint steps below the resume step). Env: E (ARC_HEAL_ENV
pairs), CKAT, STEP0, TOTAL. Exit 1 with the reason."""
import ast
import os
import sys

SRC = "./.claude/worktrees/sft-heal/scripts/posttrain/modal_rl1.py"
tree = ast.parse(open(SRC).read())
ns = {}
exec(compile(ast.Module([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "env_pairs"], []), SRC, "exec"), ns)
bad = []
try:
    ns["env_pairs"](os.environ.get("E", ""))
except ValueError as e:
    bad.append(f"trainer env: {e}")
step0, total = int(os.environ.get("STEP0") or 0), int(os.environ.get("TOTAL") or 0)
ck = [int(x) for x in (os.environ.get("CKAT") or "").split(",") if x.strip()]
late = [s for s in ck if s <= step0 or (total and s > total)]
if late:
    bad.append(f"checkpoint-at {late} outside steps {step0 + 1}..{total}")
if bad:
    print("PRELAUNCH REFUSED: " + "; ".join(bad))
    sys.exit(1)
print(f"prelaunch OK: env pairs, {len(ck)} checkpoint steps in {step0 + 1}..{total}")
