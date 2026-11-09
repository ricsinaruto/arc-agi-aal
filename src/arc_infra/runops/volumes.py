"""Per-run Modal volumes (one volume per experiment, not everything on arc-prolong-runs).

A run's record names its volume (records.volume_of: "arc-run-<run>", a v2 volume, or the shared arc-prolong-runs for
older runs). In the container the run's own volume is mounted at /runs, so /runs/validation/<out_id>/ is unchanged.
Inputs written by other runs (a corpus, a serving dir, a resume source) are mounted read-only at /in/<volume>/ and
their paths rewritten, so a run never writes to another run's volume.

    mounts, path = volumes.plan(run, inputs=["/runs/validation/rl1-screen-seed-v2/arena/trials/rl-loop/rl"])
    @app.function(volumes=volumes.modal_mounts(mounts), ...)
    ... use path(p) for every input path in the job spec
"""
import re

from arc_infra.runops import records as R

RUNS = "/runs"
INPUTS = "/in"
CACHE, CACHE_VOLUME = "/cache", "arc-cache"  # shared v2, read-write: compile/boot/trainer/runtime caches of per-run jobs
CACHE_DIRS = ("compile-cache", "boot-cache", "trainer-cache", "runtime-cache", "runtime-cache-preflight")


def _source_run(path):
    """The run whose output a /runs/validation/<folder>/... path is (<folder> may be <run>-attempt<N>), else None."""
    m = re.match(r"^/?runs/validation/([^/]+)/", str(path))
    if not m or m.group(1) == "preflight":
        return None
    return re.sub(r"-attempt\d+$", "", m.group(1))


def _cache_dir(path):
    parts = str(path).lstrip("/").split("/")
    return len(parts) > 1 and parts[0] == "runs" and parts[1] in CACHE_DIRS


def plan(run, inputs=(), root=None):
    """({mount point: volume name}, rewrite): the run's own volume at /runs; each input on another volume read-only at
    /in/<volume>; for a per-run volume the shared caches at /cache (/runs/<cache dir>/... -> /cache/<cache dir>/...).
    ``rewrite(p)`` maps a path to where the container sees it (paths on the run's own volume and non-/runs paths are
    returned unchanged). A run on the shared volume gets {"/runs": arc-prolong-runs} and an identity rewrite."""
    own = R.volume_of(run, root)
    mounts, moved = {RUNS: own}, {}
    if own == R.SHARED_VOLUME:
        return mounts, lambda p: p
    mounts[CACHE] = CACHE_VOLUME
    for p in inputs:
        if _cache_dir(p):
            continue
        src = _source_run(p) if str(p).startswith(("/runs/", "runs/")) else None
        vol = R.volume_of(src, root) if src else (R.SHARED_VOLUME if str(p).startswith(("/runs/", "runs/")) else None)
        if vol and vol != own:
            mounts[f"{INPUTS}/{vol}"] = vol
            moved[str(p)] = vol

    def rewrite(p):
        if _cache_dir(p):
            return CACHE + "/" + str(p).lstrip("/").split("/", 1)[1]
        vol = moved.get(str(p))
        if vol is None:
            src = _source_run(str(p)) if str(p).startswith(("/runs/", "runs/")) else None
            vol = R.volume_of(src, root) if src else None
            if vol is None or vol == own or f"{INPUTS}/{vol}" not in mounts:
                return p
        return f"{INPUTS}/{vol}/" + str(p).lstrip("/").split("/", 1)[1]
    return mounts, rewrite


def rewrite_env(env, rewrite):
    """The env with every /runs/... value (comma lists too) passed through ``rewrite``."""
    def one(v):
        return ",".join(rewrite(x) if x.startswith("/runs/") else x for x in v.split(","))
    return {k: one(v) if isinstance(v, str) and "/runs/" in v else v for k, v in env.items()}


def cache_root(mounts):
    """Where the caches are in the container: /cache on a per-run volume, else /runs (arc-prolong-runs)."""
    return CACHE if CACHE in mounts else RUNS


def modal_mounts(mounts):
    """{mount point: modal.Volume}: the run's own volume read-write (created as v2 on first use when it is per-run),
    every input volume read-only."""
    import modal

    out = {}
    for point, name in mounts.items():
        if point in (RUNS, CACHE):
            out[point] = (modal.Volume.from_name(name, create_if_missing=True, version=2) if name != R.SHARED_VOLUME
                          else modal.Volume.from_name(name))
        else:
            out[point] = modal.Volume.from_name(name).read_only()
    return out
