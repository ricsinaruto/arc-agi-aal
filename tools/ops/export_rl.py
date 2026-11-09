"""RL v7 (hero) exporter -> <scratch>/train-metrics/<run>.jsonl every 120 s, on arc_infra.runops.metrics (reference
migration, infra stage 3; was scratch rl-arm/hero/hero-export.py + .rl-tools/gate/metrics_export.py).
Infra rows from the RL watcher's pass files in --passes (<run>-watch-passes.jsonl: vLLM metrics of both servers;
<run>-card-util.jsonl: card util/VRAM); learning rows from the run's feed on the volume (step, accumulate, group_done,
read, wread, envs). Usage: export_rl.py RUN --passes DIR [--base=RUN0] [--once]"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402

sys.path.insert(0, str(paths.ROOT_PATH / "src"))
from arc_infra.runops import metrics  # noqa: E402

MODAL = str(paths.ROOT_PATH / ".venv/bin/modal")
CAP_GIB = 268.6  # B300 usable (nvidia-smi total)
FEED = "validation/{run}/arena/trials/rl-loop/rl/feed.jsonl"


def tail_new(path, key, st):
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text().splitlines()
    i = st.get(key, 0)
    st[key] = len(lines)
    out = []
    for line in lines[i:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def pooled(sets, names=None):
    ok = [m for n, m in sets.items() if (names is None or n in names) and m.get("rhae_per_10k") is not None
          and m.get("tries") and m.get("tokens_per_try")]
    W = sum(m["tries"] * m["tokens_per_try"] for m in ok)
    if not W:
        return None, None
    w = [m["tries"] * m["tokens_per_try"] / W for m in ok]
    return (sum(a * m["rhae_per_10k"] for a, m in zip(w, ok)),
            sum((a * float(m.get("rhae_10k_se") or 0)) ** 2 for a, m in zip(w, ok)) ** 0.5)


def ratio(v, se, v0, se0):
    """v / v0 with a 95% CI (delta method on independent reads), or None."""
    if not v or not v0:
        return None
    r = v / v0
    d = 1.96 * r * ((se / v) ** 2 + (se0 / v0) ** 2) ** 0.5
    return r, max(0.0, r - d), r + d


def feed_metrics(e, st):
    k, m = e.get("kind"), {}
    if k == "step":
        m.update({"train.step": e.get("step"), "train.lr": e.get("lr"), "train.grad_norm": e.get("grad_norm"),
                  "train.policy_loss": e.get("loss"), "train.entropy": e.get("entropy"),
                  "train.tokens": e.get("trained_tokens"), "train.tries": e.get("trained"),
                  "train.update_norm_step": e.get("update_norm"), "train.approx_kl": (e.get("k3") or {}).get("approx_kl"),
                  "train.clip_frac": (e.get("k3") or {}).get("clip_frac"),
                  "time.checkpoint_s": e.get("checkpoint_seconds")})
        lags = e.get("lag") or {}
        if lags:
            n = sum(lags.values())
            m["train.age_median"] = sorted(int(a) for a, c in lags.items() for _ in range(c))[n // 2]
            m["train.age_max"] = max(int(a) for a in lags)
        if st.get("last_step_epoch"):
            m["time.step_s"] = e["epoch"] - st["last_step_epoch"]
        st["last_step_epoch"] = e["epoch"]
        acc = st.pop("acc", None)
        if acc:
            m["time.train_s"] = acc["train_s"]
            m["infra.peak_alloc_gib"] = acc["peak"]
            m["infra.gpu_util.trainer_mean"] = acc["util"]
        grp = st.pop("grp", None)
        if grp and grp["tries"]:
            m["play.clear_rate"] = grp["clears"] / grp["tries"]
            m["play.band_frac"] = grp["band"] / max(1, grp["groups"])
            m["play.groups"] = grp["groups"]
    elif k == "accumulate":
        a = st.setdefault("acc", {"train_s": 0.0, "peak": None, "util": None})
        a["train_s"] += float(e.get("train_seconds") or 0)
        a["groups"] = a.get("groups", 0) + 1
        m["play.live_groups_accumulated"] = a["groups"]  # toward batch_groups (8 x 12 = 96 tries per step)
        if e.get("card_peak_gib") is not None:
            a["peak"] = max(a["peak"] or 0, float(e["card_peak_gib"]))
        if e.get("gpu_util_mean") is not None:
            a["util"] = e["gpu_util_mean"]
    elif k == "group_done":
        g = st.setdefault("grp", {"tries": 0, "clears": 0, "groups": 0, "band": 0})
        t, c = int(e.get("tries") or 0), int(e.get("clears") or 0)
        g["tries"] += t
        g["clears"] += c
        g["groups"] += 1
        g["band"] += int(0 < c < t)
        # live play between steps (a slow step stays visible)
        m.update({"play.live_groups_done": g["groups"], "play.live_tries_done": g["tries"],
                  "play.live_band_frac": g["band"] / g["groups"], "play.live_clear_rate": g["clears"] / max(1, g["tries"])})
    elif k == "read":
        sets = e.get("sets") or {}
        extra = set(e.get("extra_sets") or [])
        v, se = pooled(sets, [n for n in sets if n not in extra])
        m["eval.heldout_rhae10k"], m["eval.heldout_se"] = v, se
        m["eval.read_step"] = e.get("step")
        if int(e.get("step") or 0) == 0 and v:
            st["base"] = [v, se]
        r = ratio(v, se, *(st.get("base") or [None, None]))
        if r:  # target: 2x held-out RHAE/tok vs the base (version 0)
            m["eval.heldout_ratio"], m["eval.heldout_ratio_lo"], m["eval.heldout_ratio_hi"] = r
            m["eval.heldout_ratio_target"] = 2.0
        if extra:
            m["eval.heldout_ext_rhae10k"] = pooled(sets)[0]
    elif k == "wread":
        m["eval.wgame_score"], m["eval.wgame_score_per_mtok"] = e.get("score"), e.get("score_per_mtok")
        r = e.get("ratio_score")
        if r:
            m["eval.wgame_ratio"], m["eval.wgame_ratio_lo"], m["eval.wgame_ratio_hi"] = r
            m["eval.wgame_ratio_target"] = 2.0
        m["eval.wgame_ratio_per_mtok"] = e.get("ratio_per_mtok")
    elif k == "envs":
        m["play.pool"] = e.get("pool")
        for lvl, n in (e.get("fresh_by_level") or {}).items():
            m[f"play.fresh_L{lvl}"] = n
        for s, n in (e.get("start_source") or {}).items():
            m[f"play.src_{s}"] = n
    return m


def volume_text(path):
    try:
        return subprocess.run([MODAL, "volume", "get", "--force", "arc-prolong-runs", path, "-"], capture_output=True,
                              text=True, timeout=120).stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def once(run, passes, st, out):
    for p in tail_new(passes / f"{run}-watch-passes.jsonl", "passes", st):
        s2 = p.get("s2") or {}
        m = {"infra.kv_frac": p.get("kv"),  # fraction: train_dash shows *_frac x 100
             "infra.running": (p.get("running") or 0) + (s2.get("num_requests_running") or 0),
             "infra.waiting": (p.get("waiting") or 0) + (s2.get("num_requests_waiting") or 0),
             "infra.gen_tok_s": p.get("gen_tok_s")}
        g2 = s2.get("generation_tokens_total")
        if g2 is not None and st.get("g2") is not None and p["t"] > st["g2t"] and g2 >= st["g2"]:
            m["infra.gen_tok_s"] = (m["infra.gen_tok_s"] or 0) + (g2 - st["g2"]) / (p["t"] - st["g2t"])
        if g2 is not None:
            st["g2"], st["g2t"] = g2, p["t"]
        out.emit(m, t=p["t"])
    for u in tail_new(passes / f"{run}-card-util.jsonl", "util", st):
        m = {"infra.vram_cap_gib": CAP_GIB}
        for i, c in (u.get("util5") or {}).items():
            m[f"infra.gpu_util.{i}"] = c.get("util")
            if c.get("vram") is not None:
                m[f"infra.vram_gib.{i}"] = c["vram"] * CAP_GIB
        out.emit(m, t=u["t"])
    lines = [x for x in volume_text(FEED.format(run=run)).splitlines() if x.strip()]
    for line in lines[st.get("feed", 0):]:
        try:
            e = json.loads(line)
        except ValueError:
            break  # a torn last line: read again next pass
        st["feed"] = st.get("feed", 0) + 1
        m = feed_metrics(e, st)
        alert = e.get("why") if e.get("kind") == "alert" else None
        if m or alert:
            out.emit(m, t=e.get("epoch") or time.time(), alert=alert)


def queue_row(run):
    q = paths.read_json(paths.ROOT_PATH / "docs/modal-queue.json")
    return next((r for r in q.get("live", []) if r.get("run") == run), {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--passes", required=True, type=Path, help="the RL watcher's pass-file folder")
    ap.add_argument("--base", help="a resumed leg: the run whose step-0 read anchors the ratio")
    ap.add_argument("--continues", help="the leg this one resumes (default --base); written into the meta every start")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    row = queue_row(a.run)
    n = str(row.get("gpu") or "").split("x")[0].strip()
    state = a.passes / f"{a.run}-export-state.json"
    st = json.loads(state.read_text()) if state.exists() else {}
    parent = a.continues or a.base
    st.setdefault("until", int(time.time()))  # the branch point: this leg's first exporter start (~its launch)
    metrics.meta(a.run, arm="rl", what=row.get("what"), cards=int(n) if n.isdigit() else None,
                 continues=[{"run": parent, "until": st["until"]}] if parent else None)
    if a.base and not st.get("base"):
        for line in volume_text(FEED.format(run=a.base)).splitlines():
            if line.strip().startswith("{"):
                e = json.loads(line)
                if e.get("kind") == "read" and int(e.get("step") or 0) == 0:
                    feed_metrics(e, st)
    out = metrics.writer(a.run)
    while True:
        try:
            once(a.run, a.passes, st, out)
            state.write_text(json.dumps(st))
        except Exception as error:  # noqa: BLE001 - one bad pass never ends the exporter
            print(f"export pass failed: {error!r}", flush=True)
        if a.once:
            return
        time.sleep(120)


if __name__ == "__main__":
    main()
