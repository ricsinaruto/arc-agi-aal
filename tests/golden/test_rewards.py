"""Golden (c): rewards, advantages (logeq vs the group ref), dynamic-sampling weights and per-reply turn credit of the
hero recipe on a recorded mixed group (R leg g00158-wa30-L3: 7/12 cleared)."""
from common import check, fixture, loop, sha

HERO = "rl1-loop-seq-real-v8"


def test_group_rewards_weights_and_turn_credit(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_TURN_POOL", "0")  # turn parsing inline (the pool only prefetches the same parse)
    folder = fixture("g00158-wa30-L3")
    lp = loop(HERO, tmp_path)
    rows = lp.mg_rows(folder)
    kept, info = lp.mg_select(folder, rows, folder.name)
    info.pop("_shadow", None)
    check("rewards", {"rows": [r["label"] for r in rows], "info": info,
                      "kept": [[r["label"], round(w, 9), share] for _, r, w, share, *_ in kept],
                      "turn": {str(k): sha(v) for k, v in sorted((getattr(lp, "mg_turn", None) or {}).items(), key=str)}})
