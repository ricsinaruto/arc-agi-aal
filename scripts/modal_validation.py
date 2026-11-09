"""Launch plan of the held-out read on Nebius (launch/rl/launch.py backend "nebius": tools/ops/nlaunch.sh passes this
file to the nebius branch's scripts/nebius_validation.py as ARC_MODAL_VALIDATION).

nebius_validation never imports this file: it parses it and runs its plain top-level functions (with json, Path, os,
time, TEST and KIND in scope) to get the run's timeout and bootstrap arguments. Keep these two functions plain (no
decorator, no imports, names from that scope only). The read is test "b200-run", kind "arena", on one RTX PRO 6000:
the bundle's notebook seconds + 300 s, and the trials of the study's chain (nebius_validation replaces the arguments
below with ``--trial <name>`` per chain trial).
"""
# ruff: noqa: F821  (json, Path and os come from nebius_validation's scope)


def study():
    return json.loads((Path(os.environ["ARC_VALIDATION_BUNDLE"]) / "configs/comparison.json").read_text())


def validation_timeout(kind, test):
    if (kind, test) != ("arena", "b200-run"):
        raise KeyError(f"{kind}/{test}: only arena b200-run (the held-out read) is planned here")
    return min(86400, int(study()["notebook_seconds"]) + 300)


def arena_arguments(test):
    if test != "b200-run":
        raise KeyError(f"{test}: only b200-run (the held-out read) is planned here")
    return ["--trial", "duck-transfer"] if study()["b200"]["harness"] == "arena" else []
