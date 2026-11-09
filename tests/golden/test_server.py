"""Golden (redesign): what each live config hands the play/read server: the serving-kernel flags the boot builds
(serving_kernels_patch.validate + server_flags), the study's server block, each trial's server_env and the redesign's
per-card env. Pinned through the trunk/rl cleanup (26 Oct)."""
from common import B, check, config

LIVE = ("rl1-loop-seq-real-v8", "rl1-loop-seq-glong-v5", "b2-rl-hread-rd-v1")


def test_live_server_inputs():
    from arc_infra import serving_kernels_patch as skp

    got = {}
    for run in LIVE:
        study = B.inline(config(run)["study"])
        server = study.get("server") or {}
        spec = skp.validate(server["serving_kernels"]) if server.get("serving_kernels") else None
        red = ((study.get("rl_loop") or {}).get("mg") or {}).get("redesign") or {}
        got[run] = {"flags": skp.server_flags(spec, int(server.get("mtp_tokens") or 0)) if spec else None,
                    "server": server, "server_env": {t.get("name"): t.get("server_env") for t in study.get("trials") or []},
                    "card_env": red.get("card_env")}
    check("server", got)
