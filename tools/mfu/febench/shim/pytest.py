"""Minimal pytest stand-in for running a test file where pytest is not installed (febench GPU self-test)."""
import importlib
import itertools
import os
import sys
import traceback


class _Skip(Exception):
    pass


def importorskip(name):
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise _Skip(str(e))


class _Mark:
    def parametrize(self, names, values):
        def deco(f):
            f.__dict__.setdefault("_params", []).insert(0, ([n.strip() for n in names.split(",")], values))
            return f
        return deco

    def skipif(self, cond, reason=""):
        def deco(f):
            if cond:
                f._skip = reason
            return f
        return deco

    def __getattr__(self, name):
        return lambda *a, **k: (lambda f: f)


mark = _Mark()


class raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        import re
        if kind is None:
            raise AssertionError(f"DID NOT RAISE {self.exc}")
        if not issubclass(kind, self.exc):
            return False
        if self.match is not None and not re.search(self.match, str(value)):
            raise AssertionError(f"{value!r} does not match {self.match!r}")
        return True


class _Capsys:
    def __init__(self):
        import io
        self.buf, self.old = io.StringIO(), sys.stdout
        sys.stdout = self.buf

    def readouterr(self):
        out = self.buf.getvalue()
        self.buf.seek(0)
        self.buf.truncate()
        self.old.write(out)
        return type("Captured", (), {"out": out, "err": ""})()

    def close(self):
        self.old.write(self.buf.getvalue())
        sys.stdout = self.old


class MonkeyPatch:
    def __init__(self):
        self.undo = []

    def setenv(self, k, v):
        self.undo.append((os.environ.get(k), k)); os.environ[k] = v

    def delenv(self, k, raising=True):
        self.undo.append((os.environ.get(k), k)); os.environ.pop(k, None)

    def setitem(self, d, k, v):
        old = d.get(k, _Skip); self.undo.append(("item", d, k, old)); d[k] = v

    def setattr(self, obj, name, value):
        self.undo.append(("attr", obj, name, getattr(obj, name))); setattr(obj, name, value)

    def close(self):
        for u in reversed(self.undo):
            if u[0] == "item":
                _, d, k, old = u
                if old is _Skip:
                    d.pop(k, None)
                else:
                    d[k] = old
            elif u[0] == "attr":
                setattr(u[1], u[2], u[3])
            else:
                old, k = u
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old


def run_file(path):
    import inspect
    sys.path.insert(0, os.path.dirname(path))
    mod = importlib.import_module(os.path.basename(path)[:-3])
    passed = failed = skipped = 0
    for name, f in sorted(vars(mod).items()):
        if not (name.startswith("test_") and callable(f)):
            continue
        if getattr(mod, "pytestmark", None) is not None and getattr(mod.pytestmark, "_skip", None):
            skipped += 1; continue
        if getattr(f, "_skip", None):
            skipped += 1; continue
        params = getattr(f, "_params", [])
        combos = [dict()]
        for names, values in params:
            combos = [dict(c, **dict(zip(names, v if len(names) > 1 else (v,)))) for c in combos for v in values]
        for c in combos:
            mp = MonkeyPatch()
            kw = dict(c)
            sig = inspect.signature(f).parameters
            if "monkeypatch" in sig:
                kw["monkeypatch"] = mp
            if "tmp_path" in sig:
                skipped += 1; continue
            cap = kw["capsys"] = _Capsys() if "capsys" in sig else None
            if cap is None:
                kw.pop("capsys")
            try:
                f(**kw); passed += 1
            except _Skip:
                skipped += 1
            except Exception:
                failed += 1; print(f"FAILED {name} {c}"); traceback.print_exc()
            finally:
                if cap is not None:
                    cap.close()
                mp.close()
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    return failed == 0 and passed > 0


if __name__ == "__main__":
    sys.exit(0 if run_file(sys.argv[1]) else 1)
