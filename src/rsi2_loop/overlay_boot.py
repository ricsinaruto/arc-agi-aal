"""``sitecustomize`` of a proposer-built serving stack (bundle ``rsi/serving-src/rsi_boot/``, first on PYTHONPATH of
the server and of every process it starts). With ``RSI_OVERLAY`` set (the candidate's ``serving/overlay``), any module
under it replaces that one module, wherever the original lives: ``overlay/vllm/v1/core/sched/scheduler.py`` stands in
for ``vllm.v1.core.sched.scheduler`` and the rest of vLLM still comes from the installed runtime. A package's
``__init__.py`` in the overlay replaces that package's init; its other modules are found in the overlay first, then in
the original package. Then the next ``sitecustomize`` on the path (the runtime's own, if any) runs as usual."""
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


class OverlayFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root):
        self.root = root

    def _original(self, fullname, path, target):
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                return spec
        return None

    def find_spec(self, fullname, path=None, target=None):
        rel = os.path.join(self.root, *fullname.split("."))
        init = os.path.join(rel, "__init__.py")
        if os.path.isfile(init):
            original = self._original(fullname, path, target)
            locations = [rel] + list((original.submodule_search_locations or []) if original else [])
            return importlib.util.spec_from_file_location(fullname, init, submodule_search_locations=locations)
        if os.path.isfile(rel + ".py"):
            return importlib.util.spec_from_file_location(fullname, rel + ".py")
        return None


def install():
    root = os.environ.get("RSI_OVERLAY")
    if root and os.path.isdir(root) and not any(isinstance(f, OverlayFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, OverlayFinder(os.path.abspath(root)))


def chain():
    """The next sitecustomize on sys.path (this one shadows it)."""
    for entry in sys.path:
        folder = os.path.abspath(entry or ".")
        if folder == HERE:
            continue
        candidate = os.path.join(folder, "sitecustomize.py")
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("_rsi_next_sitecustomize", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return


install()
try:
    chain()
except Exception as exc:  # noqa: BLE001 - the runtime's own customisation failing must not stop the server
    print(f"rsi_boot: next sitecustomize failed: {type(exc).__name__}: {exc}", file=sys.stderr)
