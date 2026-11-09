"""trunk/sft data selection: which recorded v8.5p-harness plays the SFT trainer may learn from -> manifest JSON.

    .venv/bin/python scripts/sft/pick.py --runs launch/sft/data2-runs.txt --out manifest.json [--threshold 0.9]
    .venv/bin/python scripts/sft/pick.py --rethreshold manifest.json --threshold T --out manifest-T.json   # no scan

Candidates: bucket runs (runs/validation/<run>) with token-exact recordings (arena/working/trials/*/games/*/rollout.jsonl).
Training-15 games only. Harness: a run is v8.5p-harness iff ``harness(arena.json)`` equals the reference's (agent profile,
harness flags/fixes, loop rule and every non-serving arena config key). Per run x game: instance curves (cumulative
completion tokens of ``kind == "model"`` log rows vs levels of ``kind == "observation"`` rows) scored with the
scorecard's level scores; game curve = mean over the run's instances; mark m = min(run, ref) per-instance mean tokens;
kept iff the run's mean score over [0, m] (area under its score-vs-tokens curve / m) >= threshold x the reference's;
the old one-mark rule S_run(m) >= S_ref(m) stays in each row as kept_mark. A pair without recordings is a row with
recorded False, never kept. Run as a script: the scan is a serialized Modal function (a module named ``select``
cannot be imported by name in the container; the stdlib's wins)."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

REF = "b2-sabl-v85p-c100-300k-v1-nebius-20261019"
# a game REF does not play (the excl5 games ar25 ft09 lp85 tu93 vc33) is scored against v8.5p's own excl5 recording when
# its harness equals REF's (rows name their reference); that run is also a held-out source
REF_EXCL5 = "b2-sft4x-ex5ref-v1"
TRAIN = ("tr87", "sb26", "m0r0", "r11l", "re86", "cn04", "sc25", "cd82", "tn36", "ka59", "dc22", "ls20", "su15", "g50t",
         "s5i5")  # RL's training 15 (26 Oct)
HELDOUT = ("sp80", "wa30", "sk48", "lf52", "bp35", "ft09", "vc33", "lp85", "tu93", "ar25")  # the other 10 public games
# not the v8.5p served stack: student plays, the int3-requant held-out play, and any run served
# from the Intel W4A16-AutoRound weights (no t37 overlay or prune: the s1 recorded plays); excluded from rows and held-out
STUDENT = ("b2-sft4x-c1s-v1a", "b2-sft4x-c1s-v1b", "b2-sft4x-opd1-g1-v1", "b2-sft4x-opd0-dh10-v1r")
W4A16 = "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround"


def not_stock(run):
    """Why a scanned run is not on the v8.5p served stack (None: it is)."""
    if run["run"] in STUDENT:
        return "student or requant play"
    if any(W4A16 in a for a in run.get("assets") or []):
        return "served from the W4A16-AutoRound weights, not the v8.5p stack"
    return None
def candidates(every, on_policy=False):
    """Scanned runs that may give training rows. ``on_policy``: the student's own plays are the point,
    so student / W4A16-served runs stay in (the harness rule still applies; use --threshold 0 for no score rule)."""
    return list(every) if on_policy else [r for r in every if not not_stock(r)]


THRESHOLD = 0.9
BUCKET, ENDPOINT, REGION = "arc-runs-bucket", "https://storage.eu-south1.nebius.cloud", "eu-south1"
SERVING = {"model", "model_revision", "base_url", "serving_backend", "gameplay_seconds", "setup_seconds", "parallel_games",
           "max_num_seqs", "max_batched_tokens", "gpu_memory_utilization", "speculative_tokens", "seed"}

app = modal.App("sft-select", include_source=False)
image = modal.Image.debian_slim(python_version="3.12").apt_install("rclone")


# ---- pure parts (unit-tested) ---------------------------------------------------------------------------------------

def harness(arena):
    """The play harness of one trial's arena.json: what the model sees and may do, not how it is served."""
    games = arena.get("games") or [{}]
    per_game = {json.dumps({"profile": g.get("agent_profile"), "flags": (g.get("dfr_harness") or {}).get("flags"),
                            "fixes": sorted((g.get("harness_fixes") or {}).get("fixes") or []),
                            "loop": (g.get("loop_rule") or {}).get("action"),
                            "hx": sorted(g.get("hx_harness") or {}) != []}, sort_keys=True) for g in games}
    return {"config": {k: v for k, v in sorted((arena.get("config") or {}).items()) if k not in SERVING},
            "games": sorted(per_game)}


def diff(a, b):
    out = [f"config.{k}: {a['config'].get(k)!r} vs {b['config'].get(k)!r}"
           for k in sorted(set(a["config"]) | set(b["config"])) if a["config"].get(k) != b["config"].get(k)]
    return out + ([f"games: {a['games']} vs {b['games']}"] if a["games"] != b["games"] else [])


def curve(rows):
    """(points, tokens) of one instance log: points [(cumulative completion tokens, levels completed)], levels
    monotone (the running max)."""
    tokens, levels, points = 0, 0, [(0, 0)]
    for row in rows:
        if row.get("kind") == "model":
            tokens += (row.get("usage") or {}).get("completion_tokens") or 0
        elif row.get("kind") == "observation" and (row.get("levels") or 0) > levels:
            levels = row["levels"]
            points.append((tokens, levels))
    return points, tokens


def score_at(points, level_scores, level_count, mark):
    """Game score after ``mark`` completion tokens (arc_agi EnvironmentScoreCalculator.to_score): sum_i (i+1) *
    level_score_i over the levels done by then, capped at 100 x their weights, over sum_i (i+1) for i < level_count."""
    done = max(lv for t, lv in points if t <= mark)
    raw = sum((i + 1) * level_scores[i] for i in range(done))
    return min(raw, 100 * done * (done + 1) / 2) / sum(range(1, level_count + 1))


def game_curve(instances, mark):
    """Mean over a game's instances of their score at ``mark``."""
    return sum(score_at(i["points"], i["level_scores"], i["level_count"], mark) for i in instances) / len(instances)


def area(points, level_scores, level_count, mark):
    """Integral of score_at over [0, mark] (a step function: exact)."""
    total, t0 = 0.0, 0
    s0 = score_at(points, level_scores, level_count, 0)
    for t in sorted({t for t, _ in points if 0 < t < mark}):
        total += s0 * (t - t0)
        t0, s0 = t, score_at(points, level_scores, level_count, t)
    return total + s0 * max(mark - t0, 0)


def game_mean_score(instances, mark):
    """The game curve's mean score over [0, mark]: mean over instances of area / mark."""
    if mark <= 0:
        return game_curve(instances, 0)
    return sum(area(i["points"], i["level_scores"], i["level_count"], mark) for i in instances) / len(instances) / mark


def keep(row, threshold):
    """Curve rule: recorded and mean score over [0, m] >= threshold x the reference's (both zero: kept, as before)."""
    return row.get("recorded", True) and "area_ref" in row and row["area_run"] >= threshold * row["area_ref"]


def select(runs, ref, extra=None, threshold=THRESHOLD):
    """Manifest rows for every candidate run x training game, plus the run-level exclusions. ``extra``: a second
    reference for the games REF does not play, used only if its harness equals REF's."""
    ref_trial = next(iter(ref["trials"].values()))
    ref_games = {g: (ref["run"], i) for g, i in by_game(ref_trial).items()}
    for t in (extra or {}).get("trials", {}).values():
        if not t.get("error") and not diff(t["harness"], ref_trial["harness"]):
            ref_games.update({g: (extra["run"], i) for g, i in by_game(t).items() if g not in ref_games})
    rows, excluded = [], []
    for run in runs:
        for trial, t in run["trials"].items():
            name = run["run"] if len(run["trials"]) == 1 else f"{run['run']}:{trial}"
            if t.get("error"):
                excluded.append({"run": name, "reason": t["error"]})
                continue
            d = diff(t["harness"], ref_trial["harness"])
            if d:
                excluded.append({"run": name, "reason": "harness: " + "; ".join(d)})
                continue
            for game, inst in sorted(by_game(t).items()):
                if game not in TRAIN:
                    continue
                if game not in ref_games:
                    rows.append({"run": name, "bucket_run": run["run"], "trial": trial, "game": game,
                                 "instances": sorted(i["id"] for i in inst), "kept": False, "both_zero": False,
                                 "recorded": any(i["reply_tokens"] for i in inst),
                                 "reply_tokens": sum(i["reply_tokens"] for i in inst),
                                 "instance_reply": {i["id"]: i["reply_tokens"] for i in inst},
                                 "reason": "no reference play of this game"})
                    continue
                ref_name, ref_inst = ref_games[game]
                run_tok = sum(i["tokens"] for i in inst) / len(inst)
                ref_tok = sum(i["tokens"] for i in ref_inst) / len(ref_inst)
                mark = min(run_tok, ref_tok)
                s_run, s_ref = game_curve(inst, mark), game_curve(ref_inst, mark)
                a_run, a_ref = game_mean_score(inst, mark), game_mean_score(ref_inst, mark)
                end_run = sum(i["card_score"] for i in inst) / len(inst)
                end_ref = sum(i["card_score"] for i in ref_inst) / len(ref_inst)
                reply = sum(i["reply_tokens"] for i in inst)
                rows.append({"run": name, "bucket_run": run["run"], "trial": trial, "game": game, "ref": ref_name,
                             "instances": sorted(i["id"] for i in inst), "mark": round(mark),
                             "score_run": round(s_run, 4), "score_ref": round(s_ref, 4), "kept_mark": s_run >= s_ref,
                             "area_run": round(a_run, 6), "area_ref": round(a_ref, 6),
                             "curve_ratio": round(a_run / a_ref, 4) if a_ref else None,
                             "score_per_mtok_run": round(end_run / max(run_tok, 1) * 1e6, 3),
                             "score_per_mtok_ref": round(end_ref / max(ref_tok, 1) * 1e6, 3),
                             "tokens_run": round(run_tok), "tokens_ref": round(ref_tok),
                             "log_tokens": sum(i["tokens"] for i in inst), "recorded": reply > 0, "reply_tokens": reply,
                             "instance_reply": {i["id"]: i["reply_tokens"] for i in inst},
                             "both_zero": a_run == 0 and a_ref == 0})
    return rethreshold(rows, threshold), excluded


def rethreshold(rows, threshold):
    """``kept`` and ``reason`` of every scored row at ``threshold``."""
    for r in rows:
        if "area_ref" not in r:
            continue
        r["kept"] = keep(r, threshold)
        op = ">=" if r["area_run"] >= threshold * r["area_ref"] else "<"
        r["reason"] = (f"mean score over [0, m] {r['area_run']:.3f} {op} {threshold} x {r['area_ref']:.3f}"
                       + ("" if r["recorded"] else "; no token-exact recording"))
    return rows


def heldout(runs, ref, extra=None):
    """Held-out-10 recordings of the harness-matched pool runs and the excl5 reference (no score filter; student plays
    left out)."""
    ref_h = next(iter(ref["trials"].values()))["harness"]
    out = []
    for run in runs + [extra] * bool(extra):
        if not_stock(run):
            continue
        for trial, t in run["trials"].items():
            if t.get("error") or diff(t["harness"], ref_h):
                continue
            for game, inst in sorted(by_game(t).items()):
                if game in HELDOUT and any(i["reply_tokens"] for i in inst):
                    out.append({"bucket_run": run["run"], "trial": trial, "game": game,
                                "instance_reply": {i["id"]: i["reply_tokens"] for i in inst if i["reply_tokens"]},
                                "reply_tokens": sum(i["reply_tokens"] for i in inst)})
    return out


def find_key(obj, key):
    """The first value of ``key`` anywhere in a JSON tree (depth first), or None."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            got = find_key(v, key)
            if got is not None:
                return got
    return None


def kd_pool(runs, ref):
    """KD pool: every train-15 (run, game) pair whose harness equals REF's up to the sampling temperature,
    no score rule, unfinished trials included (their harness unchecked: no arena.json). ``t1``: played at temperature
    1.0, a separate column."""
    ref_h = next(iter(ref["trials"].values()))["harness"]
    rows, skipped = [], []
    for run in runs:
        stack = not_stock(run) or "v8.5p"
        for trial, t in run["trials"].items():
            finished = not t.get("error")
            if finished:
                d = [x for x in diff(t["harness"], ref_h) if not x.startswith("config.temperature:")]
                if d:
                    skipped.append({"run": run["run"], "trial": trial, "reason": "harness: " + "; ".join(d)})
                    continue
                inst = {k: i for k, i in t["instances"].items()}
            else:
                inst = t.get("partial") or {}
                if not inst:
                    skipped.append({"run": run["run"], "trial": trial, "reason": t["error"]})
                    continue
            games = {}
            for k, i in inst.items():
                if i["game"] in TRAIN:
                    games.setdefault(i["game"], {})[k] = i["reply_tokens"]
            for game, reply in sorted(games.items()):
                rows.append({"bucket_run": run["run"], "trial": trial, "game": game, "instance_reply": reply,
                             "reply_tokens": sum(reply.values()), "temperature": t.get("temperature"),
                             "t1": t.get("temperature") == 1.0, "finished": finished, "stack": stack,
                             "source": "recorded"})
    return {"rows": rows, "skipped": skipped}


def kd_table(pool):
    """Reply tokens per game: temperature-1.0 runs apart; and per served stack."""
    out = {"per_game": {}, "per_stack": {}}
    for r in pool["rows"]:
        g = out["per_game"].setdefault(r["game"], {"base": 0, "t1": 0})
        g["t1" if r["t1"] else "base"] += r["reply_tokens"]
        s = out["per_stack"].setdefault(r["stack"], {"base": 0, "t1": 0})
        s["t1" if r["t1"] else "base"] += r["reply_tokens"]
    out["per_game"] = dict(sorted(out["per_game"].items()))
    out["total"] = {c: sum(v[c] for v in out["per_game"].values()) for c in ("base", "t1")}
    out["pairs"] = sum(r["reply_tokens"] > 0 for r in pool["rows"])
    out["unfinished_reply_tokens"] = sum(r["reply_tokens"] for r in pool["rows"] if not r["finished"])
    return out


def by_game(trial):
    out = {}
    for k, inst in trial["instances"].items():
        out.setdefault(inst["game"], []).append({**inst, "id": k})
    return out


def apply_rebuilt(manifest, results):
    """Rows and held-out entries of the runs rebuild.py replayed: an instance counts when its replay passed every check
    (``checks.ok``); its reply tokens are the rebuilt gameplay replies'. Rows of those runs get ``source: rebuilt``."""
    ok = {}
    for r in results:
        j = r["job"]
        if r["checks"].get("ok"):
            ok[(j["run"], j["trial"], j["k"])] = (j["game_id"], int(r.get("reply_tokens") or 0))
    runs = {(j["run"], j["trial"]) for j in (r["job"] for r in results)}
    for row in manifest["rows"] + (manifest.get("kd_pool") or {}).get("rows", []):
        if (row["bucket_run"], row["trial"]) in runs:
            row["instance_reply"] = {k: ok.get((row["bucket_run"], row["trial"], k), (None, 0))[1]
                                     for k in row["instance_reply"]}
            row["reply_tokens"] = sum(row["instance_reply"].values())
            row["recorded"], row["source"] = row["reply_tokens"] > 0, "rebuilt"
    if manifest.get("kd_pool"):
        manifest["kd_pool"]["table"] = kd_table(manifest["kd_pool"])
    held = {}
    for (run, trial, k), (game, tokens) in ok.items():
        if game in HELDOUT and tokens:
            held.setdefault((run, trial, game), {})[k] = tokens
    manifest["heldout"] = [h for h in manifest["heldout"] if (h["bucket_run"], h["trial"]) not in runs] + [
        {"bucket_run": run, "trial": trial, "game": game, "instance_reply": inst, "reply_tokens": sum(inst.values()),
         "source": "rebuilt"} for (run, trial, game), inst in sorted(held.items())]
    manifest["rebuild"] = {"instances": len(results), "ok": len(ok),
                           "failed": sorted(f"{r['job']['run']}:{r['job']['k']}" for r in results
                                            if not r["checks"].get("ok"))}
    return manifest


def tally(rows, key="reply_tokens"):
    per_game = {}
    for r in rows:
        per_game[r["game"]] = per_game.get(r["game"], 0) + r[key]
    return {"pairs": len(rows), "runs": len({r["run"] for r in rows}), "games": len(per_game),
            "reply_tokens": sum(per_game.values()), "per_game": dict(sorted(per_game.items()))}


def table(rows, thresholds=(1.0, 0.9, 0.8, 0.7)):
    """Kept (recorded) pairs / runs / games / reply tokens per curve threshold and for the old one-mark rule, plus the
    unrecorded pairs that pass the same rule (their log completion tokens as ``reply_tokens``)."""
    scored = [r for r in rows if "area_ref" in r]
    rules = {f"curve>={t}": (lambda r, t=t: r["area_run"] >= t * r["area_ref"]) for t in thresholds}
    rules["mark rule"] = lambda r: r["kept_mark"]
    return {name: {**tally([r for r in scored if r["recorded"] and ok(r)]),
                   "unrecorded": tally([r for r in scored if not r["recorded"] and ok(r)], "log_tokens")}
            for name, ok in rules.items()}


def cap_preview(rows, mult=2.0):
    """Per game kept reply tokens before and after build.py's cap (mult x the median game), estimated on the manifest's
    reply tokens (build caps whole built examples, after its encoding drops)."""
    import statistics

    before = tally([r for r in rows if r["kept"]])["per_game"]
    if not before:
        return {}
    median = statistics.median(before.values())
    return {"median": median, "cap": mult * median, "before": before,
            "after": {g: min(v, mult * median) for g, v in before.items()}}


def report(manifest):
    lines = [f"{'rule':<12} {'pairs':>5} {'runs':>4} {'games':>5} {'reply M':>8}   unrecorded: pairs runs log-tok M"]
    for k, v in manifest["thresholds"].items():
        u = v["unrecorded"]
        lines.append(f"{k:<12} {v['pairs']:>5} {v['runs']:>4} {v['games']:>5} {v['reply_tokens'] / 1e6:>8.2f}"
                     f"   {u['pairs']:>5} {u['runs']:>4} {u['reply_tokens'] / 1e6:.1f}")
    c = manifest["cap_preview"]
    if c:
        lines.append(f"threshold {manifest['threshold']}: reply M per game before -> after the cap ({c['cap'] / 1e6:.2f})")
        lines += [f"  {g} {c['before'][g] / 1e6:.2f} -> {c['after'][g] / 1e6:.2f}" for g in c["before"]]
    return "\n".join(lines)


def summary(rows, excluded, held):
    kept = [r for r in rows if r["kept"]]
    per_game = tally(kept)["per_game"]
    return {"candidate_pairs": len(rows), "kept_pairs": len(kept), "kept_runs": len({r["run"] for r in kept}),
            "kept_games": len(per_game), "reply_tokens_kept": sum(per_game.values()),
            "reply_tokens_per_game": dict(sorted(per_game.items())),
            "games_under_100k": sorted(g for g, v in per_game.items() if v < 100_000),
            "kept_both_zero": sum(r["both_zero"] for r in kept), "excluded_runs": len(excluded),
            "kept_vs_excl5_ref": sum(r.get("ref") == REF_EXCL5 for r in kept),
            "heldout_reply_tokens": sum(h["reply_tokens"] for h in held),
            "heldout_reply_per_game": {g: sum(h["reply_tokens"] for h in held if h["game"] == g) for g in HELDOUT}}


# ---- bucket scan (Modal CPU) ---------------------------------------------------------------------------------------

def rclone(*args):
    return subprocess.run(["rclone", *args], check=True, capture_output=True, text=True).stdout


def rollout_reply(path):
    """Played gameplay reply tokens of one rollout.jsonl (0: none)."""
    reply = 0
    if Path(path).exists():
        for line in open(path, "rb"):
            r = json.loads(line)
            if r.get("gameplay", True) and not r.get("scripted"):
                reply += (r.get("usage") or {}).get("completion_tokens") or 0
    return reply


@app.function(image=image, timeout=2700, retries=0, cpu=(0.125, 8), secrets=[modal.Secret.from_name("nebius-s3")],
              serialized=True, max_containers=60)
def scan(run: str) -> dict:
    """One run's trials: roster, harness, served assets and per-instance curves, scorecard scores, reply tokens."""
    import shutil

    os.environ.update(RCLONE_CONFIG_ARC_TYPE="s3", RCLONE_CONFIG_ARC_PROVIDER="Other", RCLONE_CONFIG_ARC_ENV_AUTH="true",
                      RCLONE_CONFIG_ARC_ENDPOINT=ENDPOINT, RCLONE_CONFIG_ARC_REGION=REGION,
                      RCLONE_CONFIG_ARC_NO_CHECK_BUCKET="true")
    src, dst = f"arc:{BUCKET}/runs/validation/{run}", Path("/tmp") / run
    inc = ["execution.json", "arena/trials/*/public-roster.json", "arena/working/trials/*/public-roster.json", "arena/trials/*/scorecard.json",
           "arena/trials/*/arena.json", "arena/working/trials/*/games/*/log.jsonl",
           "arena/working/trials/*/games/*/rollout.jsonl", "arena/trials/study.json"]
    rclone("copy", src, str(dst), *sum((["--include", i] for i in inc), []), "--transfers", "32", "--retries", "5")
    assets = [a["prefix"] for a in json.loads((dst / "execution.json").read_text()).get("assets", [])] \
        if (dst / "execution.json").exists() else []
    study = dst / "arena/trials/study.json"
    temperature = find_key(json.loads(study.read_text()), "temperature") if study.exists() else None
    trials = {}
    for work in sorted((dst / "arena/working/trials").glob("*/")):
        trial, final = work.name, dst / "arena/trials" / work.name
        if not (final / "scorecard.json").exists() or not (final / "arena.json").exists():
            # unfinished: no scores or arena.json; its recorded sessions still count for the KD pool
            roster_file = work / "public-roster.json"
            roster = ({r["game_id"]: r["source_game_id"].split("-")[0] for r in json.loads(roster_file.read_text())}
                      if roster_file.exists() else {})
            partial = {}
            for k, game in sorted(roster.items()):
                log, roll = work / "games" / k / "log.jsonl", work / "games" / k / "rollout.jsonl"
                if log.exists():
                    partial[k] = {"game": game, "tokens": curve([json.loads(x) for x in open(log)])[1],
                                  "reply_tokens": rollout_reply(roll)}
            trials[trial] = {"error": "no scorecard.json / arena.json (run did not finish)", "partial": partial,
                             "temperature": temperature}
            continue
        arena = json.loads((final / "arena.json").read_text())
        roster_file = next((f for f in (final / "public-roster.json", work / "public-roster.json") if f.exists()), None)
        if roster_file is None:
            trials[trial] = {"error": "no public-roster.json"}
            continue
        roster = {r["game_id"]: r["source_game_id"].split("-")[0] for r in json.loads(roster_file.read_text())}
        card = {e["id"]: e for e in json.loads((final / "scorecard.json").read_text())["environments"]}
        instances, mismatches = {}, []
        for k, game in sorted(roster.items()):
            log, roll, env = work / "games" / k / "log.jsonl", work / "games" / k / "rollout.jsonl", card.get(k)
            if env is None:
                continue
            rows = [json.loads(line) for line in open(log)] if log.exists() else []
            points, tokens = curve(rows)
            run0 = env["runs"][0] if env.get("runs") else {}
            ls = list(run0.get("level_scores") or [0.0] * env["level_count"])
            if env["levels_completed"] > points[-1][1]:  # a clear by the last reply's tool call, after the log's last
                points.append((tokens, env["levels_completed"]))  # observation (the run stopped mid-call)
                mismatches.append(f"{k}: {env['levels_completed'] - points[-2][1]} final clear(s) not in the log")
            reply = rollout_reply(roll)
            inst ={"game": game, "tokens": tokens, "points": points, "level_scores": ls,
                    "level_count": env["level_count"], "card_score": env["score"], "reply_tokens": reply}
            end = score_at(points, ls, env["level_count"], tokens)
            if abs(end - env["score"]) > 1e-6 * max(1.0, env["score"]):
                mismatches.append(f"{k}: curve end {end:.4f} vs scorecard {env['score']:.4f}")
            instances[k] = inst
        trials[trial] = {"harness": harness(arena), "model": (arena.get("config") or {}).get("model"),
                         "instances": instances, "curve_mismatches": mismatches,
                         "temperature": (arena.get("config") or {}).get("temperature", temperature)}
    shutil.rmtree(dst, ignore_errors=True)
    return {"run": run, "assets": assets, "trials": trials}


def register(app_id):
    """CPU-only Modal app -> run records (the reconcile flags an unregistered app)."""
    root = Path(__file__).resolve().parents[2]
    common = subprocess.run(["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True).stdout.strip()
    for tool in (root / "tools/ops/records.py", Path(common or root).parent / "tools/ops/records.py"):
        if tool.exists():  # trunk/sft has no tools/: the main checkout's
            subprocess.run([sys.executable, str(tool), "cpu-app", app_id], check=False)
            return
    print(f"WARNING: no tools/ops/records.py; register {app_id} by hand", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="bucket runs, one per line (# comments)")
    ap.add_argument("--rethreshold", help="an earlier manifest: re-decide kept at --threshold, no scan")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--rebuilt", action="append", default=[], help="rebuild.py results (with --rethreshold): replayed "
                    "instances of unrecorded runs become rows' token-exact sessions")
    ap.add_argument("--out", default="manifest.json")
    ap.add_argument("--on-policy", action="store_true", help="keep student / W4A16-served runs (on-policy pool)")
    a = ap.parse_args()
    if a.rethreshold:
        manifest = json.loads(Path(a.rethreshold).read_text())
        if a.rebuilt:
            manifest = apply_rebuilt(manifest, [r for f in a.rebuilt for r in json.loads(Path(f).read_text())])
        rows = rethreshold(manifest["rows"], a.threshold)
        manifest.update(threshold=a.threshold, summary=summary(rows, manifest["excluded"], manifest["heldout"]),
                        thresholds=table(rows), cap_preview=cap_preview(rows))
        Path(a.out).write_text(json.dumps(manifest, indent=1))
        print(report(manifest))
        return
    names = [line.strip() for line in open(a.runs) if line.strip() and not line.startswith("#")]
    todo = list(dict.fromkeys([REF, REF_EXCL5] + names))
    with modal.enable_output(), app.run():
        register(app.app_id)
        got = list(scan.map(todo, return_exceptions=True))
    scanned = {r["run"]: r for r in got if isinstance(r, dict)}
    failed = {n: repr(r)[:300] for n, r in zip(todo, got) if not isinstance(r, dict)}
    ref, extra = scanned[REF], scanned.get(REF_EXCL5)
    every = [scanned[n] for n in dict.fromkeys(names) if n in scanned]
    cands = candidates(every, a.on_policy)
    rows, excluded = select(cands, ref, extra, a.threshold)
    excluded += [{"run": r["run"], "reason": not_stock(r)} for r in every if r not in cands]
    excluded += [{"run": n, "reason": f"scan failed: {e}"} for n, e in failed.items()]
    held = heldout(cands, ref, extra if extra and REF_EXCL5 not in names else None)
    mism = {f"{r['run']}:{t}": v["curve_mismatches"] for r in [ref] + [extra] * bool(extra) + cands for t, v in r["trials"].items()
            if v.get("curve_mismatches")}
    served = {r["run"]: [x for x in r["assets"] if x != "games"] for r in every}
    ref_h = next(iter(ref["trials"].values()))["harness"]
    extra_ok = bool(extra) and any(not t.get("error") and not diff(t["harness"], ref_h) for t in extra["trials"].values())
    manifest = {"reference": REF, "reference_excl5": REF_EXCL5 if extra_ok else None, "train_games": list(TRAIN),
                "heldout_games": list(HELDOUT), "threshold": a.threshold, "summary": summary(rows, excluded, held),
                "thresholds": table(rows), "cap_preview": cap_preview(rows),
                "rows": rows, "excluded": excluded, "heldout": held, "curve_mismatches": mism, "served": served}
    manifest["kd_pool"] = kd_pool(every, ref)
    manifest["kd_pool"]["table"] = kd_table(manifest["kd_pool"])
    Path(a.out).write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["summary"], indent=1))
    print(report(manifest))


if __name__ == "__main__":
    main()
