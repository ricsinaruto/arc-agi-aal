"""RL1_HUB_PREFETCH=1: the hub kernels (``kernels`` library) a Flash forward resolves lazily, fetched and imported
before the GPU gate (D2 27 Oct: the first accumulate fetched kernels-community/fla v1 (293 files), mamba-ssm v3 (42)
and a 7-file fla set, ~8 s of its 13.6 s cold prefix). The repos come from transformers' own kernel mapping for the
hub-kernel layers the model's modeling module declares (cuda, training mode), so a transformers bump moves with it.
After the prefetch rl1_firstcall counts each later ``snapshot_download`` (``hub_fetch`` in first_calls): the first
forward must add none that downloads (bar: hub seconds after the gate <= 0.5)."""

import importlib
import inspect
import re
import sys
import time

DECLARED = re.compile(r"use_kernel_(?:forward_from_hub|func_from_hub_with_fallback|func_from_hub)\(\s*[\"']([\w.-]+)[\"']")


def declared(model):
    """Hub-kernel layer names declared by the model's transformers modeling modules (decorators in their source)
    plus any module's ``kernel_layer_name``."""
    names, seen = set(), set()
    for mod in model.modules():
        name = getattr(type(mod), "kernel_layer_name", None)
        if isinstance(name, str):
            names.add(name)
        where = type(mod).__module__
        if where.startswith("transformers.models.") and where not in seen:
            seen.add(where)
            try:
                names.update(DECLARED.findall(inspect.getsource(sys.modules[where])))
            except (OSError, TypeError, KeyError):
                pass
    return names


def repos(names, mapping, device="cuda", mode="TRAINING"):
    """(repo_id, version, revision) for each name's ``device`` entry in ``mapping`` (transformers'
    get_kernel_mapping_transformers): a repository object, or a {Mode: repository} dict (``mode`` by name)."""
    out = []
    for name in sorted(names):
        entry = (mapping.get(name) or {}).get(device)
        if isinstance(entry, dict):  # mode None: every mode's repository (U1: a 7-file repo fetched after the gate)
            entries = [v for k, v in entry.items() if mode is None or getattr(k, "name", str(k)).upper().endswith(mode)]
        else:
            entries = [entry]
        for entry in entries:
            repo = getattr(entry, "repo_id", None) or getattr(entry, "_repo_id", None)
            if repo is None:
                continue
            key = (repo, getattr(entry, "version", None) or getattr(entry, "_version", None),
                   getattr(entry, "revision", None) or getattr(entry, "_revision", None))
            if key not in out:
                out.append(key)
    return out


def prefetch(model, get=None, mapping=None):
    """Fetch and import each repo before the gate. Returns the row for ``loaded`` (names, repos, seconds, errors);
    never raises (a failed prefetch leaves the lazy path as it was)."""
    began, row = time.time(), {"names": [], "repos": [], "errors": []}
    first = sys.modules.get("rl1_firstcall")
    if first is not None and first.INSTALLED:  # kernels may have been imported after install
        first.wrap_hub()
    try:
        if mapping is None:
            hub = importlib.import_module("transformers.integrations.hub_kernels")
            mapping = hub.get_kernel_mapping_transformers()
        if get is None:
            get = importlib.import_module("kernels").get_kernel
        names = declared(model)
        row["names"] = sorted(names)
        for repo, version, revision in repos(names, mapping, mode=None):
            t0 = time.time()
            try:
                kw = {"version": version} if version is not None else {"revision": revision} if revision else {}
                get(repo, **kw)
                row["repos"].append({"repo": repo, "version": version, "revision": revision,
                                     "seconds": round(time.time() - t0, 2)})
            except Exception as error:  # noqa: BLE001 - the lazy path still runs
                row["errors"].append(f"{repo}: {error!r}"[:300])
    except Exception as error:  # noqa: BLE001
        row["errors"].append(repr(error)[:300])
    row["seconds"] = round(time.time() - began, 2)
    if first is not None:  # from here on, hub_fetch counts only fetches after the prefetch
        first.STATS.pop("hub_fetch", None)
    return row
