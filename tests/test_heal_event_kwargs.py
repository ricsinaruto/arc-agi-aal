"""heal_kd event(...) calls: no keyword given twice (explicit kwargs vs the keys of a spread **reader(...) result).
Static (AST), no torch: v30's fixed_before passed files= next to **heldout_kl(...), which returns "files" too, and
crashed 4 B300 arms before step 1 (21 Oct)."""
import ast
import os
from pathlib import Path
import sys as _sys  # trunk 24 Oct: PTH's trainer copies (scripts/posttrain/pth) shadow RL's for this module
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "pth"))
for _m in ("rl1_trainer", "flash_qlora_job", "rl1_sparse"):
    _sys.modules.pop(_m, None)

# HEAL_KD_SRC: a bundle's copy (launch_heal.sh checks the bundle it launches)
SRC = Path(os.environ.get("HEAL_KD_SRC") or Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "heal_kd.py")


def _dict_keys(node):
    return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def return_keys(fn):
    """String keys a function can return: returned dict literals, plus a returned name's dict-literal assignments and
    name["key"] = ... stores."""
    keys, names = set(), set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and node.value is not None:
            if isinstance(node.value, ast.Dict):
                keys |= _dict_keys(node.value)
            elif isinstance(node.value, ast.Name):
                names.add(node.value.id)
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names and isinstance(node.value, ast.Dict):
                    keys |= _dict_keys(node.value)
                if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in names
                        and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str)):
                    keys.add(t.slice.value)
    return keys


def event_calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "event":
            yield node


def test_event_kwargs_do_not_collide():
    tree = ast.parse(SRC.read_text())
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    event_params = {a.arg for a in funcs["event"].args.args}  # positional params ("out") also collide
    readers = {name: return_keys(fn) for name, fn in funcs.items()}
    for name in list(readers):  # a checking wrapper (heldout_kl -> _heldout_kl) returns its inner function's keys
        readers[name] |= readers.get("_" + name, set())
    assert {"files", "kd_reply"} <= readers["heldout_kl"]  # the extractor sees heldout_kl's keys
    checked, bad = 0, []
    for call in event_calls(tree):
        explicit = [k.arg for k in call.keywords if k.arg is not None]
        seen = set(event_params) | set()
        for name in explicit:
            if name in seen:
                bad.append((call.lineno, name, "explicit"))
            seen.add(name)
        for k in call.keywords:
            if k.arg is None and isinstance(k.value, ast.Call) and isinstance(k.value.func, ast.Name):
                spread = readers.get(k.value.func.id, set())
                clash = spread & seen
                if clash:
                    bad.append((call.lineno, sorted(clash), k.value.func.id))
                seen |= spread
                checked += 1
    assert checked >= 6, checked  # held-out / probe / fixed-batch reads are spread into events
    assert not bad, f"event() keyword given twice: {bad}"


def test_extractor_catches_the_v30_bug():
    src = '''
def event(out, **row):
    return row
def heldout_kl(folder):
    out = {"files": 1, "kd": 0.1}
    out["kd_reply"] = 0.2
    return out
def run(log, queue):
    event(log, event="fixed_before", files=[q for q in queue], **heldout_kl("x"))
'''
    tree = ast.parse(src)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    keys = return_keys(funcs["heldout_kl"])
    call = next(event_calls(tree))
    explicit = {k.arg for k in call.keywords if k.arg}
    assert keys == {"files", "kd", "kd_reply"} and explicit & keys == {"files"}
