"""RL step 1 driver (13 October, ``docs/rl-plan.md``): play rounds through the level bench on the running server,
rewards and advantages, updates by the trainer service (``scripts/posttrain/rl1_trainer.py``) on the same card, the
adapter reloaded into the server, evaluations, the live feed and the stop rules. Trial kind ``rl-loop``.

One card: the trainer stays resident (weights, adapter, optimizer); the server sleeps (level 1) while it trains and
wakes for play. The adapter is served unmerged under the name ``policy``; every agent request names it.

Modes (``study["rl_loop"]["mode"]``):
- ``probe``: fit check (checkpointing off / alternate / full at 32K), play = train parity on a perturbed adapter,
  round 0 (sb26 x 16 + base eval: lp85 and seeded sb26), then the learning-rate probe on round 0: per rate the
  adapter from its initial state, ``probe_updates`` updates on the training attempts, the held-out attempts and the
  lp85 probes scored after each.
- ``loop``: round 0 (or ``round0_from``: a probe job's round-0 folder), then rounds until a stop rule.
- ``final``: the adapter of ``resume_state`` (one card: the trainer converts it beside the sleeping server and
  exits), the served-logprob equality check on ``equality_source``, then the final evaluation.

Feed: ``<output>/rl/feed.jsonl`` (one event per line) and ``<output>/rl/status.json`` (the latest of each kind).
"""

import asyncio
import copy
import json
import math
import os
import sys
import shutil
import signal
import statistics
import subprocess
import threading
import time
from pathlib import Path

from .rl_ladder import LadderMixin
from .rl_sft import SftMixin
from .rl_single import MultiMixin, OpdMixin, SingleLevelMixin, TransferMixin
from .rl_async import AsyncMixin, ModeBMixin
from .v7_envs import EnvsV7Mixin


DEFAULTS = {
    "mode": "probe",
    "trainer_python": "/workspace/axolotl-venv/bin/python",
    "trainer_script": "rl1/rl1_trainer.py",             # in the bundle, beside its imports
    "model_dir": "/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1",
    # path_chunk: the trainer's one prefix chunking for pi_old, updates, probes and scores (20 Oct; rl1-gold: 16K vs 32K
    # differ by 0.054 nats/token at the same weights); 16,384 is the probes' and KL floor's chunking
    "path_chunk": 16384,
    "rank": 32, "optimizer": "adamw8bit", "loop_lora": False, "tis_cap": 0.0, "window": 24576, "lr": 6e-5, "clip": 1.0, "max_windows": 8,
    "train_game": "sb26", "train_attempts": 32, "train_budget": 30000,
    "eval_game": "lp85", "eval_attempts": 16, "eval_budget": 40000, "eval_seeded": 16, "eval_seed_base": 1000,
    "eval_rounds": [8, 16, 24], "eval_attempts_mid": 8, "rounds": 25,
    "steps_per_round": 4,  # every attempt with A > 0, in about 4 optimizer steps per round
    # "signed" = one step per round on the per_side largest |A| windows each side
    # at a train_tail tail, push-down weight balanced to push-up on the served trained tokens (TOPR negatives)
    "update_mode": "positive", "per_side": 12, "train_tail": None,
    "replay": None, "resume_from": None,  # recorded round folders whose updates rebuild a lost adapter, then the next
    "lp85_every": 0, "lp85_attempts": 8,  # a seeded lp85 guard inside every n-th round's play bench
    "checkpoint_every": 0,
    # two cards, the server on GPU 0 and the trainer on GPU 1; a round's play overlaps the update
    # on the previous round's attempts (one step of lag, corrected by TIS against the recorded engine logprobs)
    "dual": False, "trainer_device": "1", "trainer_env": None,  # trainer_env: extra environment for the trainer (diagnostics)
    "resume_state": None, "recorded": None,  # a state checkpoint (op_checkpoint) and the rounds it trained through
    # F-small restore check (A): every level start of the first restore_attempts attempts of a recorded round,
    # replayed to that request (a one-token live request), checked against the recording; no trainer
    "restore_source": None, "restore_attempts": 8,
    # resuming an F-small run (15 Oct, f1 crash): the state's round number, the source run's later trend points
    # ({"full": [[x, y]], "segments": [[x, y]]}), the full games the next states come from, and whether the restore
    # gate runs again (it gates update 1, which the source run already passed)
    "resume_round": None, "trend_seed": None, "f_states_from": None, "f_gate": True,
    # mode "final" (15 Oct): the final eval of a saved state; first the served logprobs on this server (adapter loaded)
    # against the engine log-probabilities recorded on the default serving path with the same adapter
    "equality_source": None, "equality_points": 24, "equality_factor": 1.25,  # bar: max(0.03, factor x repeat noise)
    # mode "final" (16 Oct): resume_adapter, a served adapter folder (a state converted on CPU) loaded into the awake
    # server with no trainer; transfer {"sources", "levels", "starts", "tries", "seed", "pick_seed", "cap",
    # "max_prompt"}: paired restored tries (adapter and base, same starts and seeds) on levels it never trained on
    "resume_adapter": None, "transfer": None, "attention": None, "trainer_env": None,
    # multi-game baseline: mode "screen" (base tries, one card) or a loop with mg.screen
    "mg": None,
    # paired re-evaluation: {"attempts": n, "seed_base": s, "games": [...]}: the adapter and the
    # base model on the same seeds, both benches at once
    "paired": None,
    # F v2: segment reward "tokens" = cleared - live tokens / f_budget (no cap); f_steps optimizer
    # steps per round over every informative state (states split across the steps; probes after the last unless
    # f_probe_each); whole-game rounds every f_whole_every rounds (played beside the segments) with one extra signed
    # step on f_whole_per_side games each side in the next update; f_pick "adaptive": contested levels (clear rate
    # 0.1-0.9 from the segment history, seeded by f_level_seed {level: [cleared, total]}) weighted p(1-p), half the
    # states from where full games stopped; the gradient tail of a segment covers its live part only
    "f_reward": "score", "f_steps": 1, "f_probe_each": True, "f_whole_every": 0, "f_whole_per_side": 6,
    "f_level_seed": None, "f_live_tail": False,
    # level ladder: per round lvl_games whole games (pinned adapter "game",
    # up to lvl_game_budget tokens) in the background, and per level 1..lvl_levels lvl_tries restored tries from one
    # on-policy level start within the level's budget B_l; passes kept (positive only), rank-weighted by tokens and
    # actions; every token of a kept try trained (whole-segment windows) over lvl_passes passes in batches of
    # lvl_batch; then B_l x lvl_shrink (held while a level has fewer than lvl_hold_below passes). Round 0: base
    # games (B_l measured on this harness where >= 6 clears). Stops: NaN/inf, restore failures above
    # lvl_restore_fail, usable share below lvl_usable; KL and lp85 are report-only.
    "lvl": False, "lvl_games": 8, "lvl_game_budget": 150000, "lvl_tries": 16, "lvl_levels": 8, "lvl_budgets": None,
    "lvl_round0_games": 16, "lvl_measure": True, "lvl_shrink": 0.7, "lvl_hold_below": 2, "lvl_passes": 2,
    "lvl_batch": 16, "lvl_window": 32768, "lvl_lp85_every": 3, "lvl_lp85_games": 8, "lvl_final_games": 16,
    "lvl_final_seed": 7000, "lvl_restore_fail": 0.1, "lvl_usable": 0.75, "lvl_seg_concurrency": 48,
    "lvl_batches": 2, "lvl_final_hours": 1.5, "lvl_keep_max": 8, "lvl_game_wall": 6000, "lvl_norm": "tokens", "lvl_resume": None,
    # OPD: mode "opd_gate" (teacher gate, server only) or opd True in a loop (iterated hinted teacher)
    "opd": False, "opd_clip": 5.0, "opd_gate_tries": 4, "opd_gate_seed": 7000, "opd_gate_other_levels": [3, 4, 5, 6, 7, 8],
    # next single-level run: l2_record {"games", "budget", "seed"}: the starts come from base
    # games recorded at the start of the loop (from level 1, stopped by the budget soon after level l2_level begins)
    # instead of lvl_resume's games-00; l2_norm "cap": each try's summed log-likelihood over l2_cap (Dr. GRPO)
    "l2_record": None, "l2_norm": "try", "l2_reward": "tokens", "l2_starts_per_step": 1, "l2_kl_stop": None,
    # early stop: held-out clear rate <= stop_rate, or held-out median > stop_median x the base
    # round's, in stop_rounds consecutive policy-played rounds
    "l2_stop_rate": None, "l2_stop_median": None, "l2_stop_rounds": 2,
    "l2": False, "l2_level": 2, "l2_held": 4, "l2_seed": 0, "l2_starts": 3, "l2_tries": 16, "l2_eval_tries": 4,
    "l2_eval_seed": 5000, "l2_cap": 32000, "l2_per_side": 3, "l2_passes": 1, "l2_final_tries": 8, "l2_final_seed": 6000,
    "l2_final_games": 4, "l2_final_hours": 1.1,
    # F-small: per round f_states level starts of the latest full attempts, each restored and
    # played live f_continuations times until that level clears or f_budget tokens; reward = the level's score,
    # baseline = the state's other continuations; f_refresh_attempts full games every f_refresh_every rounds
    "f_small": False, "f_states": 16, "f_continuations": 8, "f_budget": 16384, "f_refresh_every": 2,
    "f_refresh_attempts": 16,
    # "spread": round robin over the levels reached; "contested" (proposed 15 Oct): first the level each full game
    # failed on, then levels whose clear rate in those games lies in [0.1, 0.9], then round robin
    "f_pick": "spread",
    "probe_rates": [2e-5, 6e-5, 2e-4], "probe_updates": 4, "probe_held": 4,
    "fit": {"window": 32768, "total": 40960, "modes": ["off", "alternate", "full"], "margin_gib": 10, "lr": 1e-3,
            "trials": None, "first_fit": False, "select_window": None},
    "concurrency": 32,  # 32 attempts per round
    "limits": {"mismatch_mean_abs": 0.05, "mismatch_mean": 0.02, "kl_previous": 0.05, "kl_base_untrained": 0.02,
               "collapse_drop": 1.0, "eval_se": 2.0, "util_stop": 50.0, "util_alert": 70.0, "length_ratio": 2.0,
               "capped_points": 0.05, "pass_reward": 4.5, "parity_corr": 0.9, "parity_base_abs": 0.08,
               "elapsed_hours": 15.6, "awake_gib": 148.0, "asleep_gib": 3.6,
               # "engine": the plan's rule on trainer vs sampling-time engine; "split": stop when the fresh server and
               # the trainer disagree at round 0 (> split_fresh_abs), the engine gap only alerts above round 0 + rise
               "mismatch_rule": "engine", "split_fresh_abs": 0.03, "split_engine_rise": 0.02,
               "split_bar": "flat", "split_noise_factor": 1.25, "usable_share": 0.9,
               "kl_floor_relative": False, "kl_untrained_alert": False, "kl_untrained_backstop": 0.05,
               "loop_hours": None},  # awake: serving test at 32 streams, 32 GiB KV
    "round0_from": None, "round0_play_from": None,
    "trainer_timeout": 5400, "update_timeout": 3600,
    # cheapest-clear SFT (SFT arm, 16 Oct; arena.rl_sft): mode "sft" with these settings (rl_sft.SFT_DEFAULTS)
    "sft": None,
}
OLD_PASS_SKIP = ("aux", "hold_previous", "norm", "norm_cap", "batch_tokens", "batch_max")  # update-only keys


def adapter_tensors(path):
    """The tensor count of a saved adapter, checked against the manifest its trainer wrote (an
    adapter or overlay that loads fewer tensors than expected refuses)."""
    import struct

    folder = Path(path)
    try:
        manifest = json.loads((folder / "adapter-tensors.json").read_text())
        with open(folder / "adapter_model.safetensors", "rb") as file:
            header = json.loads(file.read(struct.unpack("<Q", file.read(8))[0]))
    except (OSError, ValueError, struct.error) as exc:
        raise ValueError(f"adapter {folder}: no readable tensor manifest or file ({exc})") from exc
    count = sum(1 for k in header if k != "__metadata__")
    if not 0 < count == int(manifest["tensors"]) == int(manifest["expected"]):
        raise ValueError(f"adapter {folder}: file {count} tensors, manifest {manifest}")
    return count


# every mg / mg set / record / screen key the loop reads (an unknown key refuses, never ignored;
# tests/test_mg_control.py keeps this in step with the code)
MG_KEYS = {"turn_auto",  # v6l: server 1 turn cap follows engine KV
    "action_backstop", "action_floor", "breakage", "kl_damage", "action_hi", "action_pow", "active_kv_fill", "active_tries", "kv_project", "seed_scan", "admit_stagger_s", "admit_stagger2_s", "at_caps", "balance",
    "base_adapter", "base_ref", "batch_max", "batch_tokens", "behaviour_guard", "bench", "cap", "carry", "lr_dose", "checkpoint", "eval",
    "early_play", "engine_admission", "epochs", "exclude_starts", "fill_tries", "final", "final_concurrency", "final_ref", "fixed_fresh", "fixed_sets", "fixed_tries", "fresh_tries",
    "gate_overcommit", "gpu_util", "group_size", "harvest", "curriculum", "train_summary", "nextboard", "gspo_clip", "gspo_norm", "half_split", "held_starts", "init_state", "kl_control", "kl_floor",
    "async", "v7", "read_alert", "branch", "ceiling_handoff", "dose_probe", "dosing", "target_kl", "minibatches", "pack_rounds", "try_keep", "try_snap_s", "knob_gate_kl", "keep_states", "kl_k_predict", "resume_check_tol", "kl_ramp", "kl_step", "kl_target", "kl_trips_stop", "kl_stop_auto", "k3_measure", "tis_out_alert", "length_guard", "level_norm", "log_clip", "log_fail", "log_rhae_floor", "lr_max",
    "lr_probe", "lr_schedule", "max_groups", "max_rounds", "min_usable", "no_repeat", "nudge_base", "nudge_value", "park_base", "park_prepin", "r1_first", "base_cache", "base_cache_tol", "host_load", "host_load_threads",
    "pick_seed", "policy_loss", "prescreen", "probe_every", "probe_sets", "record", "ref_fallback", "ref_round1", "reward_ref", "resume",
    "read_final", "read_norm", "reads_overlap", "fixed_every", "fixed_base_tries", "reload_check", "reload_check_once", "async_checkpoint", "window_guard", "resume_skip", "resume_train_pending", "trainer_fast", "reply_filter", "r3_check", "served_route", "reuse_from", "reward_log", "reward_shape", "rhae_cap", "rhae_hi", "rollback", "rule_a_updates", "rule_f_pooled",
    "rule_f_ratio", "rule_f_updates", "screen", "screen_root", "screens", "seed", "seed_from", "seed_sampling",
    "seed_u1", "select", "sets", "placement", "explore", "rotation", "running_caps", "seed_starts", "unseen_read", "heldout_alternate", "heldout_first", "digest_gate", "lookahead_below", "lookahead_lag", "single_card", "step_rollbacks_stop", "stratify", "stream_poll", "stream_starts",
    "success_clear_drop", "success_ratio", "success_rounds", "success_updates", "tail_fill", "tight_action_hi", "tight_action_pow",
    "timeshare", "token_hi", "trainer_overlap", "train_sampling", "train_starts", "train_tries", "transport_backoff", "turn_credit", "updates", "warn", "wread"}
SET_KEYS = {"name", "game", "level", "role", "fresh", "tries", "cap", "final_cap", "max_prompt", "primary", "round_tries",
            "sources", "start_half", "starts", "train_starts", "train_tries", "untrained_game", "seed", "final_seed",
            "source", "family", "ref", "prior", "read", "pool_max", "reserve"}  # multi-game (envs 22 Oct): the mix source, env-table refs/prior
RECORD_KEYS = {"game", "games", "budget", "seed"}
BREAK_KEYS = {"reply_ratio", "ratio", "abs", "cap_abs", "clear_drop", "actions_rise", "rollback_first", "filler_ratio"}
WARN_KEYS = {"ratio", "reply_ratio", "actions"}
SCREEN_KEYS = {"screen", "root", "map"}
# mg.nextboard (envs arm): aux next-board loss on the round's own tries (arena.nextboard_pack, rl1 nextboard_aux)
NEXTBOARD_KEYS = {"weight", "windows", "attempts", "board_every", "max_steps", "max_tokens", "read", "read_every"}
READ_NORM_KEYS = {"lam", "n_boot", "base"}  # mg.read_norm (envs): per-game normalised held-out read


PLACEHOLDER_TOKENS = (64000,)  # the env table's no-data cap written as ref.tokens before 24 Oct (envs mg_pool)


def check_mg(mg):
    bad = sorted(set(mg) - MG_KEYS)
    from .curriculum import DEFAULTS as CURRICULUM_KEYS
    from .harvest import KEYS as HARVEST_KEYS
    from .v7_envs import WREAD_KEYS

    for group, known in (("breakage", BREAK_KEYS), ("warn", WARN_KEYS), ("harvest", set(HARVEST_KEYS)),
                         ("curriculum", set(CURRICULUM_KEYS) | {"shares", "seed", "allow_no_ref"}),
                         ("nextboard", NEXTBOARD_KEYS), ("rotation", {"keep", "min_explored", "every", "retire_p", "park_p", "min_tries"}),
                         ("read_norm", READ_NORM_KEYS), ("wread", set(WREAD_KEYS))):
        if group == "read_norm" and not isinstance(mg.get(group), dict):  # read_norm: true takes the defaults
            continue
        bad += [f"{group}.{k}" for k in sorted(set(mg.get(group) or {}) - known)]
    for group, known in (("sets", SET_KEYS), ("record", RECORD_KEYS), ("screens", SCREEN_KEYS)):
        for i, item in enumerate(mg.get(group) or []):
            bad += [f"{group}[{i}].{k}" for k in sorted(set(item) - known)]
    held = [x["name"] for x in mg.get("sets") or [] if x.get("role", "train") == "train" and isinstance(x.get("ref"), dict)
            and x["ref"].get("tokens") in PLACEHOLDER_TOKENS and not (x["ref"].get("measured") or x["ref"].get("derived"))]
    if held:  # envs 24 Oct (mg_pool.placeholder_refs): a no-data cap as the reward ref flattens the token term
        bad.append(f"placeholder reward refs (tokens {PLACEHOLDER_TOKENS}, not measured/derived): {held[:5]} of {len(held)}")
    sampling = mg.get("train_sampling")
    if sampling is not None and (not isinstance(sampling, dict) or not sampling or set(sampling) - {"temperature", "top_p", "top_k"}
                                 or any(type(v) not in (int, float) or isinstance(v, bool) for v in sampling.values())):
        bad.append("train_sampling: {temperature, top_p, top_k: number}")
    if mg.get("placement") not in (None, "first", "balanced"):
        bad.append("placement: first | balanced")
    if mg.get("lookahead_below") is not None:  # continuous play V1 (MFU-2): versioned adapters across two cards
        below, lag = mg["lookahead_below"], mg.get("lookahead_lag", 2)
        if type(below) is not int or below < 1 or type(lag) is not int or not 1 <= lag <= 2:
            bad.append("lookahead_below: int >= 1; lookahead_lag: int 1..2")
        bad += [f"lookahead_below with {k}" for k in ("single_card", "curriculum", "bench", "carry") if mg.get(k)]
    if int(mg.get("batch_tokens") or 0) > 0:  # MFU-1 m1g1 (23 Oct): batch_pass gradients disagree with per-window
        bad.append("batch_tokens > 0: batched tails are numerically wrong (grad cos 0.52-0.86 vs per-window), refused until fixed")
    if bad:
        raise ValueError(f"mg keys the loop does not read: {bad}")


def settings(raw):
    merged = copy.deepcopy(DEFAULTS)
    raw = {k: v for k, v in (raw or {}).items() if k != "lvl_prefix_chunk"}  # replaced by path_chunk (20 Oct)
    for key, value in raw.items():
        if key not in DEFAULTS:
            raise ValueError(f"Unknown rl_loop setting {key!r}")
        merged[key] = {**merged[key], **value} if isinstance(DEFAULTS[key], dict) else value
    if os.environ.get("ARC_RL1_RESUME_ADAPTER"):  # launch-time read target (main-run reads: one bundle, many points)
        merged["resume_adapter"] = os.environ["ARC_RL1_RESUME_ADAPTER"]
    if os.environ.get("ARC_RL1_EVAL_ADAPTERS") and (merged["mg"] or {}).get("eval"):
        merged["mg"]["eval"]["adapters"] = json.loads(os.environ["ARC_RL1_EVAL_ADAPTERS"])
    if merged["mg"]:
        check_mg(merged["mg"])
    if merged["mode"] not in ("probe", "loop", "restore", "final", "opd_gate", "screen", "sft"):
        raise ValueError("rl_loop.mode is probe, loop or restore")
    return merged


# ---- rewards and statistics (pure) ----

def attempt_reward(row):
    """Sum of the level scores min(1.15, (baseline / actions)^2) over the levels the attempt cleared."""
    return round(sum(min(115.0, float(level.get("score") or 0.0)) for level in row.get("level_rows") or []) / 100.0, 4)


# a policy outcome counts at its earned level score (0 when nothing cleared), an infra outcome
# is dropped and logged; any other outcome is treated as infra and named in the drop log.
POLICY_OUTCOMES = ("cleared", "won", "budget", "level_cap", "action_cap", "game_over", "finished",
                   "wall",       # the attempt's own wall (attempt_wall_seconds): slow or looping play
                   "lead_out")   # the lead-in token budget spent without getting through
INFRA_OUTCOMES = ("replay_mismatch",  # the recorded replay diverged (harness/engine)
                  "error",            # a harness or server exception
                  "censored",         # still running at the bench's own deadline (the run's clock)
                  "recorded")         # a scripted replay, no live play


def outcome_kind(row):
    return "policy" if row.get("outcome") in POLICY_OUTCOMES else "infra"


def usable(row):
    return outcome_kind(row) == "policy"


def drops(rows):
    """The infra drops of one bench: share kept, and each dropped attempt with its outcome."""
    dropped = [{"label": r.get("label"), "game": r.get("game"), "outcome": r.get("outcome")} for r in rows if not usable(r)]
    return {"attempts": len(rows), "usable": len(rows) - len(dropped),
            "usable_share": round((len(rows) - len(dropped)) / len(rows), 4) if rows else None, "dropped": dropped}


def leave_one_out(rewards):
    """Each reward minus the mean of the others."""
    n = len(rewards)
    total = sum(rewards)
    return [r - (total - r) / (n - 1) if n > 1 else 0.0 for r in rewards]


def mean_se(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return round(mean, 4), (round(se, 4) if se is not None else None)


def reply_stats(path, cap):
    """From an attempt's rollout.jsonl: replies, mean generated tokens per reply, share at the reply cap, the
    engine's mean sampled-token NLL (entropy estimate)."""
    replies = tokens = capped = 0
    nll, count = 0.0, 0
    try:
        with open(path) as file:
            for line in file:
                row = json.loads(line)
                if not row.get("gameplay"):
                    continue
                replies += 1
                used = int((row.get("usage") or {}).get("completion_tokens") or len(row.get("tokens") or []))
                tokens += used
                capped += row.get("finish_reason") == "length" or used >= cap
                values = [v for v in row.get("logprobs") or [] if v is not None and v > -9000]
                nll -= sum(values)
                count += len(values)
    except OSError:
        return {}
    return {"replies": replies, "tokens_per_reply": round(tokens / replies, 1) if replies else None,
            "capped_share": round(capped / replies, 4) if replies else None,
            "sampled_nll": round(nll / count, 4) if count else None}


def round_stats(rows, rewards, advantages, folder, cap, base_mean=None):
    replies = [reply_stats(Path(folder) / "attempts" / r["label"] / "rollout.jsonl", cap) for r in rows]
    levels = [len(r.get("level_rows") or []) for r in rows]
    actions = [int(r.get("total_actions") or r.get("actions") or 0) for r in rows]
    level_actions = [a["actions"] for r in rows for a in r.get("level_rows") or []]

    def avg(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "attempts": len(rows), "reward_mean": avg(rewards),
        "reward_sd": round(statistics.stdev(rewards), 4) if len(rewards) > 1 else None,
        "reward_min": min(rewards) if rewards else None, "reward_max": max(rewards) if rewards else None,
        "above_round0": (round(sum(r > base_mean for r in rewards) / len(rewards), 4)
                         if base_mean is not None and rewards else None),
        "advantage_mean": avg(advantages),
        "advantage_sd": round(statistics.stdev(advantages), 4) if len(advantages) > 1 else None,
        "advantage_positive_share": round(sum(a > 0 for a in advantages) / len(advantages), 4) if advantages else None,
        "levels_mean": avg(levels), "tokens_mean": avg([r.get("completion_tokens") for r in rows]),
        "tokens_per_reply": avg([s.get("tokens_per_reply") for s in replies]),
        "capped_share": avg([s.get("capped_share") for s in replies]),
        "sampled_nll_engine": avg([s.get("sampled_nll") for s in replies]),
        "actions_mean": avg(actions), "actions_per_level": avg(level_actions),
        "replies_total": sum(s.get("replies") or 0 for s in replies),
    }


def parity(server_base, server_adapter, trainer_base, trainer_adapter):
    """Play = train on plain sequences: base log-probabilities (server minus trainer) and the adapter's shift on
    each side (server shift against trainer shift: correlation, slope)."""
    base = [s - t for sb, tb in zip(server_base, trainer_base) for s, t in zip(sb, tb)]
    shift_s = [a - b for sa, sb in zip(server_adapter, server_base) for a, b in zip(sa, sb)]
    shift_t = [a - b for ta, tb in zip(trainer_adapter, trainer_base) for a, b in zip(ta, tb)]
    n = min(len(shift_s), len(shift_t))
    shift_s, shift_t = shift_s[:n], shift_t[:n]
    ms, mt = sum(shift_s) / n, sum(shift_t) / n
    cov = sum((a - ms) * (b - mt) for a, b in zip(shift_s, shift_t))
    vs, vt = sum((a - ms) ** 2 for a in shift_s), sum((b - mt) ** 2 for b in shift_t)
    return {"tokens": n, "base_mean_abs": round(sum(abs(d) for d in base) / len(base), 5),
            "base_mean": round(sum(base) / len(base), 5),
            "shift_server_mean_abs": round(sum(abs(a) for a in shift_s) / n, 5),
            "shift_trainer_mean_abs": round(sum(abs(b) for b in shift_t) / n, 5),
            "shift_corr": round(cov / math.sqrt(vs * vt), 4) if vs > 0 and vt > 0 else None,
            "shift_slope": round(cov / vt, 4) if vt > 0 else None}


# ---- I/O ----

def card_memory():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        used, total = (float(v) / 1024 for v in out.splitlines()[0].split(","))
        return used, total
    except Exception:  # no reading: no check
        return None, None


class Feed:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.status = {}
        self.started = time.time()

    def emit(self, kind, **data):
        self.last = time.time()
        row = {"t": round(time.time() - self.started, 1), "epoch": round(time.time(), 1), "kind": kind, **data}
        with (self.folder / "feed.jsonl").open("a") as file:
            file.write(json.dumps(row, default=str) + "\n")
        self.status[kind] = row
        tmp = self.folder / "status.json.tmp"
        tmp.write_text(json.dumps(self.status, indent=1, default=str))
        tmp.replace(self.folder / "status.json")
        print(json.dumps({"event": f"rl_{kind}", **{k: v for k, v in row.items() if k not in ("windows", "probes")}},
                         default=str)[:4000], flush=True)
        return row


def coroutine_chain(coro):
    """The await chain of a suspended coroutine: one "file:line function" per frame, then what the innermost awaits."""
    out, seen = [], 0
    while coro is not None and seen < 60:
        seen += 1
        frame = getattr(coro, "cr_frame", None) or getattr(coro, "gi_frame", None)
        if frame is None and not hasattr(coro, "cr_await") and not hasattr(coro, "gi_yieldfrom"):
            out.append(f"awaits {coro!r}"[:400])
            break
        if frame is not None:
            out.append(f"{frame.f_code.co_filename}:{frame.f_lineno} {frame.f_code.co_name}")
        coro = getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None)
    return out


class StallWatch:
    """Stall watchdog (preflight a's 35-min hang): no feed row and no finished model request for
    ``seconds`` -> one dump of every thread and asyncio task of the loop process (await chains), the shared attempt
    pool and gate, the process table and py-spy of every python process when py-spy is installed, written to
    ARC_VALIDATION_OUTPUT; ``kill``: then end the run. Env ARC_RL1_STALL_DUMP=seconds, ARC_RL1_STALL_KILL=1."""

    def __init__(self, feed, loop, seconds, kill, out):
        self.feed, self.loop, self.seconds, self.kill, self.out = feed, loop, float(seconds), bool(kill), Path(out)
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            time.sleep(min(15.0, self.seconds / 4))
            from arc_infra.fleet import ACTIVITY  # replies still arriving: play in progress, not a stall

            silent = time.time() - max(getattr(self.feed, "last", self.feed.started), ACTIVITY["last_reply"])
            if silent >= self.seconds:
                path = self.dump(silent)
                self.feed.emit("stall_dump", path=str(path), silent_seconds=round(silent), kill=self.kill)
                if self.kill:
                    os._exit(3)

    def dump(self, silent):
        import faulthandler

        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / f"stall-dump-{int(time.time())}.txt"
        with open(path, "w") as file:
            from arc_infra.fleet import ACTIVITY

            file.write(f"no feed row or reply for {silent:.0f} s; replies so far {ACTIVITY['replies']}\n\n== threads (faulthandler)\n")
            file.flush()
            faulthandler.dump_traceback(file=file, all_threads=True)
            file.write("\n== asyncio tasks\n")
            tasks = []
            for _ in range(5):
                try:
                    tasks = list(asyncio.all_tasks(self.loop))
                    break
                except RuntimeError:  # the set changed while copying
                    continue
            for task in tasks:
                file.write(f"\n-- {task!r}"[:600] + "\n")
                for line in coroutine_chain(task.get_coro()):
                    file.write(f"   {line}\n")
            try:
                from . import level_bench

                pool = level_bench.SHARED_SLOTS
                if pool is not None:
                    file.write(f"\n== shared slots: used {pool.used}/{pool.slots} live {len(pool.live)} pending {pool.pending} "
                               f"capacity {pool.capacity} load {pool.load() if pool.capacity else None} kv_waits {pool.kv_waits}\n")
                    if pool.gate is not None:
                        g = pool.gate
                        file.write(f"   gate: capacity {g.capacity} reserved {g.reserved} inflight {g.inflight} queued {len(g.queue)}\n")
            except Exception as exc:  # a diagnostic only
                file.write(f"\n(shared slots: {exc!r})\n")
            table = subprocess.run(["ps", "-eo", "pid,ppid,etime,stat,pcpu,rss,args"], capture_output=True, text=True).stdout
            file.write("\n== processes\n" + table)
            spy = shutil.which("py-spy")
            for line in table.splitlines()[1:]:
                parts = line.split(None, 6)
                if spy and len(parts) == 7 and "python" in parts[6]:
                    try:
                        got = subprocess.run([spy, "dump", "--pid", parts[0]], capture_output=True, text=True, timeout=60)
                        file.write(f"\n== py-spy {parts[0]}: {parts[6][:200]}\n{got.stdout}{got.stderr}")
                    except Exception as exc:
                        file.write(f"\n== py-spy {parts[0]}: {exc!r}\n")
            if not spy:
                file.write("\n(py-spy not installed)\n")
        return path


class GpuSampler:
    """nvidia-smi every 2 s: utilisation and memory per phase."""

    def __init__(self):
        self.rows, self.phase, self.stop = [], "start", threading.Event()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while not self.stop.wait(2.0):
            try:
                out = subprocess.run([os.environ.get("ARC_REAL_NVIDIA_SMI", "nvidia-smi"), "--query-gpu=utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                cards = [[float(v) for v in line.split(",")] for line in out.splitlines() if line.strip()]
                second = cards[1] if len(cards) > 1 else (None, None)  # two-card loop: the trainer's card
                self.rows.append((time.time(), self.phase, cards[0][0], cards[0][1] / 1024, second[0],
                                  None if second[1] is None else second[1] / 1024))
            except Exception:  # a missing sample is not an error
                pass

    def mark(self, phase):
        self.phase = phase

    def summary(self, since):
        rows = [r for r in self.rows if r[0] >= since]
        out = {}
        for phase in sorted({r[1] for r in rows}):
            part = [r for r in rows if r[1] == phase]
            out[phase] = self.stats(part)
        if rows:
            out["all"] = self.stats(rows)
        return out

    @staticmethod
    def stats(part):
        row = {"seconds": round(len(part) * 2.0), "util_mean": round(sum(r[2] for r in part) / len(part), 1),
               "idle_share": round(sum(r[2] < 10 for r in part) / len(part), 3),
               "memory_peak_gib": round(max(r[3] for r in part), 2)}
        second = [r for r in part if r[4] is not None]
        if second:
            row["card1"] = {"util_mean": round(sum(r[4] for r in second) / len(second), 1),
                            "idle_share": round(sum(r[4] < 10 for r in second) / len(second), 3),
                            "memory_peak_gib": round(max(r[5] for r in second), 2)}
        return row


class Trainer:
    def __init__(self, folder, command, env, log_path):
        self.folder = Path(folder)
        (self.folder / "cmd").mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.log = open(log_path, "a")
        self.process = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                        start_new_session=True)

    @classmethod
    def adopt(cls, folder, pid):
        """A trainer the launcher pre-spawned (arena.trainer_prespawn); its output goes to its own log."""
        from .trainer_prespawn import Adopted

        self = cls.__new__(cls)
        self.folder, self.n, self.log = Path(folder), 0, None
        (self.folder / "cmd").mkdir(parents=True, exist_ok=True)
        self.process = Adopted(pid)
        return self

    async def ready(self, timeout):
        end = time.time() + timeout
        alive = self.folder / "alive.json"
        while time.time() < end:
            if self.process.poll() is not None:
                raise RuntimeError(f"trainer exited during load ({self.process.returncode})")
            if alive.exists():
                try:
                    state = json.loads(alive.read_text())
                    if state.get("state") in ("ready", "idle"):
                        return state
                except ValueError:
                    pass
            await asyncio.sleep(5)
        raise TimeoutError("trainer did not load in time")

    async def call(self, op, timeout=3600, **payload):
        index, self.n = self.n, self.n + 1
        tmp = self.folder / "cmd" / f"{index:04d}.json.tmp"
        # ``always``: trainer paths every op runs under (mg.trainer_fast / live knobs: MFU stages on every op, so
        # probes, old logprobs and updates share one numerics)
        tmp.write_text(json.dumps({"op": op, **(getattr(self, "always", None) or {}), **payload}))
        tmp.rename(self.folder / "cmd" / f"{index:04d}.json")
        done = self.folder / "done" / f"{index:04d}.json"
        end = time.time() + timeout
        while time.time() < end:
            if done.exists():
                return json.loads(done.read_text())
            if self.process.poll() is not None:
                raise RuntimeError(f"trainer exited during {op} ({self.process.returncode})")
            await asyncio.sleep(0.25)
        raise TimeoutError(f"trainer {op} took over {timeout} s")

    async def must(self, op, timeout=3600, **payload):
        """``call`` that raises with the trainer's own error when the operation failed."""
        result = await self.call(op, timeout=timeout, **payload)
        if not result.get("ok", True):
            raise RuntimeError(f"trainer {op}: {result.get('error')}")
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()


class Server:
    mask_expect = None  # server.expert_mask (hero run): the worker mask's masked entries, checked after start / wake
    mask_sha = None  # and its sha256 (the bundle's server.expert_mask.sha256)

    def __init__(self, base_url, model):
        self.root = base_url.rstrip("/")
        self.root = self.root[:-3] if self.root.endswith("/v1") else self.root
        self.model, self.loaded = model, None

    async def mask_check(self, why):
        """The worker's expert mask is live (arc_infra.expert_mask_runtime status), or raise."""
        if not self.mask_expect:
            return None
        reply = await self.post("/collective_rpc", {"method": "arc_expert_mask", "timeout": 60,
                                                    "args": [json.dumps({"op": "status"})]})
        got = (reply.get("results") or [None])[0] if isinstance(reply, dict) else None
        if (not isinstance(got, dict) or not got.get("ok") or int(got.get("masked_entries", -1)) != int(self.mask_expect)
                or (self.mask_sha and got.get("sha256") != self.mask_sha)):
            raise RuntimeError(f"expert mask not live after {why}: {str(got)[:300]}")
        return got

    async def post(self, path, payload=None, timeout=900):
        import httpx  # here: the trainer test imports this module where httpx is absent

        async with httpx.AsyncClient(trust_env=False, timeout=timeout) as client:
            response = await client.post(self.root + path, json=payload if payload is not None else {})
            if response.status_code >= 400:
                raise RuntimeError(f"{path}: HTTP {response.status_code} {response.text[:800]}")
            try:
                return response.json()
            except ValueError:
                return response.text

    async def sleep(self):
        began = time.time()
        if getattr(self, "dual", False):  # the trainer has its own card
            return 0.0
        if not getattr(self, "asleep", False):
            await self.post("/sleep?level=1")
            self.asleep = True
        return round(time.time() - began, 1)

    async def wake(self):
        began = time.time()
        if getattr(self, "dual", False):
            return 0.0
        await self.post("/wake_up")
        self.asleep = False
        await self.mask_check("wake")
        return round(time.time() - began, 1)

    async def load(self, name, path):
        adapter_tensors(path)
        began = time.time()
        names = self.__dict__.setdefault("names", set())
        if self.loaded == name or name in names:
            await self.post("/v1/unload_lora_adapter", {"lora_name": name})
        await self.post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path), "is_3d_lora_weight": True})
        self.loaded = name
        names.add(name)
        return round(time.time() - began, 1)

    async def unload(self, name):
        await self.post("/v1/unload_lora_adapter", {"lora_name": name})
        self.__dict__.setdefault("names", set()).discard(name)
        if self.loaded == name:
            self.loaded = None

    async def tokens(self, text):
        return (await self.post("/tokenize", {"model": self.model, "prompt": text}))["tokens"]

    async def prompt_logprobs(self, model, ids):
        reply = await self.post("/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1,
                                                     "temperature": 0.0, "prompt_logprobs": 0})
        rows = reply["choices"][0]["prompt_logprobs"]
        return [float(rows[i][str(ids[i])]["logprob"]) for i in range(1, len(ids))]


def align_prompt_logprobs(entries, expected):
    """A ``prompt_logprobs`` list (entry 0 None, then ``{"<id>": {"logprob": x}}`` or ``{"<id>": x}``) as one value
    per position of ``expected``; raises unless the server scored exactly these tokens."""
    if len(entries) != len(expected):
        raise RuntimeError(f"prompt_logprobs: {len(entries)} entries for {len(expected)} expected tokens")
    out = [float("nan")]
    for i in range(1, len(expected)):
        hit = (entries[i] or {}).get(str(expected[i]))
        if hit is None:
            raise RuntimeError(f"prompt_logprobs: position {i} is not token {expected[i]}")
        out.append(float(hit["logprob"] if isinstance(hit, dict) else hit))
    return out


def dose_batches(specs, weights, steps):
    """every attempt with a positive weight, in ``steps`` minibatches mixed by advantage rank
    (rank i goes to batch i mod steps, so each batch spans the range). ``specs``: the round's window specs."""
    ranked = sorted([{**s, "w": float(weights[s["label"]])} for s in specs if weights[s["label"]] > 0],
                    key=lambda s: -s["w"])
    k = max(1, min(int(steps), len(ranked)))
    return [ranked[i::k] for i in range(k)] if ranked else []


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201,
       12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 22: 2.074,
       25: 2.060, 30: 2.042}


def reward_trend(points):
    """Pre-registered primary read: OLS slope of round-mean reward on round index, SE from the
    residuals, 95% CI (t, n - 2 degrees of freedom). ``points``: [(round, mean reward)], round 0 included."""
    n = len(points)
    if n < 3:
        return {"n": n, "slope": None}
    xs, ys = [float(x) for x, _ in points], [float(y) for _, y in points]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    se = math.sqrt(sum(r * r for r in resid) / (n - 2) / sxx)
    df = n - 2
    t95 = T95.get(df) or T95[max(k for k in T95 if k <= df)] if df <= 30 else 1.96
    return {"n": n, "slope": round(slope, 4), "se": round(se, 4), "ci95": [round(slope - t95 * se, 4), round(slope + t95 * se, 4)],
            "intercept": round(my - slope * mx, 4)}


def split_verdict(split, limits):
    """The split rule per window: fresh server vs trainer mean abs within the bar. ``split_bar``: "flat" (the bar is
    ``split_fresh_abs``) or "noise" (``max(split_fresh_abs, split_noise_factor x fresh-vs-fresh)``)."""
    rows = []
    for name, w in (split or {}).items():
        fresh, repeat = w["fresh_trainer"], w["fresh_repeat"]
        if not fresh.get("tokens"):
            continue
        bar = limits["split_fresh_abs"]
        if limits["split_bar"] == "noise" and repeat.get("tokens"):
            bar = max(bar, limits["split_noise_factor"] * repeat["mean_abs"])
        rows.append({"window": name, "fresh_trainer": fresh["mean_abs"], "fresh_repeat": repeat.get("mean_abs"),
                     "bar": round(bar, 5), "ok": fresh["mean_abs"] <= bar})
    return {"bar": limits["split_bar"], "windows": rows, "ok": bool(rows) and all(r["ok"] for r in rows)}


def compare_logprobs(a, b):
    d = [x - y for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if not d:
        return {"tokens": 0}
    return {"tokens": len(d), "mean_abs": round(sum(abs(x) for x in d) / len(d), 5), "mean": round(sum(d) / len(d), 5),
            "beyond_half": round(sum(abs(x) > 0.5 for x in d) / len(d), 5)}


def make_text(index, chars):
    import random

    rng, lines, size, row = random.Random(1000 + index), [f"log {index:03d}"], 8, 0
    words = ("move", "left", "right", "up", "down", "wall", "door", "key", "blue", "red", "green", "grid", "cell")
    while size < chars:
        line = f"{row:06d} {rng.choice(words)} {rng.choice(words)} " + " ".join(str(rng.randrange(10)) for _ in range(24))
        lines.append(line)
        size += len(line) + 1
        row += 1
    return "\n".join(lines)[:chars]


def card_memory_of(index):
    """(used GiB, total GiB) of one card by its PCI-order index (nvidia-smi's own order)."""
    try:
        smi = os.environ.get("ARC_REAL_NVIDIA_SMI") or "nvidia-smi"
        out = subprocess.run([smi, "-i", str(index), "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        used, total = (float(v) / 1024 for v in out.splitlines()[0].split(","))
        return used, total
    except Exception:
        return None, None


def second_server_argv(argv, port, gpu_util=None, kv_scale=None):
    """The live play server's argv for a second server: another port, and a smaller memory share (it wakes beside the
    parked trainer's residue)."""
    out = list(argv)
    def put(flag, value):
        if flag in out:
            out[out.index(flag) + 1] = str(value)
            return True
        return False
    if not put("--port", port):
        out += ["--port", str(port)]
    if gpu_util is not None:
        put("--gpu-memory-utilization", gpu_util)
    if kv_scale is not None and "--kv-cache-memory-bytes" in out:
        i = out.index("--kv-cache-memory-bytes") + 1
        out[i] = str(int(int(out[i]) * float(kv_scale)))
    return out


class SecondServer(Server):
    """Time share: a copy of the play server on the trainer's card. It boots asleep-capable beside an
    empty card before the trainer loads, sleeps, and plays only between the trainer's update and the round's end."""

    def __init__(self, server_dir, model, *, card, port=1235, gpu_util=0.88, kv_scale=None, log=None, env=None,
                 keep_lora=False):
        self.server_dir, self.card, self.port = Path(server_dir), str(card), int(port)
        # mg.timeshare.keep_lora (MFU-2 25 Oct): under ARC_SLEEP_KEEP_KV the LoRA slots come back with the weights
        # (sleepkv5 W2: no reload after wake, outputs at the no-sleep floor), so an adapter already loaded from the
        # same folder is not reloaded at block open (11-17 s per load, ~3 per step)
        self.keep_lora, self.keep_kv, self.paths, self.skipped_loads = bool(keep_lora), False, {}, 0
        self.extra_env = {str(k): str(v) for k, v in (env or {}).items()}  # timeshare.env: server 2 only (an in-run A/B)
        self.folder = self.server_dir / "server2"  # its own folder: the first server's cache layout glob stays clean
        self.folder.mkdir(parents=True, exist_ok=True)
        super().__init__(f"http://127.0.0.1:{self.port}/v1", model)
        self.gpu_util, self.kv_scale, self.process, self.log = gpu_util, kv_scale, None, log
        self.asleep, self.adapter, self.memory = False, None, {}

    def source(self):
        identity = json.loads((self.server_dir / "vllm-server-identity.json").read_text())
        pid = int(identity["pid"])
        argv = [a for a in Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0") if a] or identity["argv"]
        env = dict(x.split("=", 1) for x in Path(f"/proc/{pid}/environ").read_bytes().decode().split("\0") if "=" in x)
        return argv, env

    def command(self):
        argv, env = self.source()
        shim = Path(f"/tmp/gpu{self.card}-bin")  # the serving stack's nvidia-smi calls see this card
        real = env.get("ARC_REAL_NVIDIA_SMI") or os.environ.get("ARC_REAL_NVIDIA_SMI")
        if real:
            shim.mkdir(exist_ok=True)
            (shim / "nvidia-smi").write_text('#!/bin/sh\ncase " $* " in *" -i "*|*" --id"*) exec %s "$@";; esac\n'
                                             'exec %s -i %s "$@"\n' % (real, real, self.card))
            (shim / "nvidia-smi").chmod(0o755)
            env["PATH"] = f"{shim}:" + ":".join(x for x in env.get("PATH", "").split(":") if not x.startswith("/tmp/gpu"))
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=self.card)
        env.update(getattr(self, "extra_env", {}))
        return second_server_argv(argv, self.port, self.gpu_util, self.kv_scale), env

    async def boot(self, timeout=1800):
        import httpx

        argv, env = self.command()
        self.keep_kv = env.get("ARC_SLEEP_KEEP_KV") == "1"
        began = time.time()
        handle = (self.folder / "vllm.log").open("w")
        self.process = subprocess.Popen(argv, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        handle.close()
        (self.folder / "argv.json").write_text(json.dumps(argv))
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            while time.time() - began < timeout:
                if self.process.poll() is not None:
                    tail = (self.folder / "vllm.log").read_text(errors="replace")[-3000:]
                    raise RuntimeError(f"second server exited {self.process.returncode}: {tail}")
                try:
                    if (await client.get(self.root + "/v1/models")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                from arc_infra.fleet import ACTIVITY  # a live, booting server is progress for the stall watchdog;
                ACTIVITY["last_reply"] = time.time()  # the boot timeout above bounds it (pf p3ac-v1 pf1r)
                await asyncio.sleep(5)
            else:
                raise RuntimeError("second server not ready in time")
        self.memory["awake_gib"] = card_memory_of(self.card)[0]
        self.memory["boot_seconds"] = round(time.time() - began, 1)
        await self.mask_check("second server boot")
        return dict(self.memory)

    async def sleep(self):
        began = time.time()
        if not self.asleep:
            await self.post("/sleep?level=1")
            self.asleep = True
        if not self.keeps_lora():
            self.adapter = None  # reloaded on the next wake: no reliance on LoRA slots surviving sleep
            self.paths = {}
        self.memory.setdefault("asleep_gib", card_memory_of(self.card)[0])
        return round(time.time() - began, 1)

    async def wake(self):
        began = time.time()
        if self.asleep:
            await self.post("/wake_up")
            self.asleep = False
            await self.mask_check("wake")
        return round(time.time() - began, 1)

    def keeps_lora(self):
        return bool(getattr(self, "keep_lora", False) and getattr(self, "keep_kv", False))

    async def load(self, name, path):
        """keep_lora: no reload of a name already loaded from this folder (the slots survive a keep-KV sleep)."""
        paths = self.__dict__.setdefault("paths", {})
        if self.keeps_lora() and name in self.__dict__.get("names", set()) and paths.get(name) == str(path):
            self.skipped_loads = getattr(self, "skipped_loads", 0) + 1
            return 0.0
        took = await super().load(name, path)
        paths[name] = str(path)
        return took

    async def unload(self, name):
        await super().unload(name)
        self.__dict__.setdefault("paths", {}).pop(name, None)

    async def serve(self, name, path):
        """The first server's adapter under the same name (a reload only when the folder changed)."""
        if path is None:
            return 0.0
        if self.adapter == (name, str(path)):
            return 0.0
        try:
            took = await self.load(name, path)
        except RuntimeError:  # the name did not survive sleep: a plain load
            self.__dict__.setdefault("names", set()).discard(name)
            self.loaded = None
            took = await self.load(name, path)
        for old in sorted(n for n in self.__dict__.get("names", set()) if n != name and n.startswith("policy-r")
                          and not self.keeps_lora()):  # keep_lora: the block prunes after its loads (prune)
            try:  # mg.lookahead_below: one version per round; earlier names leave the LoRA slots
                await self.unload(old)
            except RuntimeError:  # already gone with the sleep
                self.names.discard(old)
        self.adapter = (name, str(path))
        return took

    async def prune(self, keep):
        """keep_lora: unload the policy names the block did not ask for (serve() keeps them for a skip-free reload)."""
        gone = []
        for old in sorted(n for n in self.__dict__.get("names", set()) if n not in keep and n.startswith("policy-r")):
            try:
                await self.unload(old)
            except RuntimeError:
                self.names.discard(old)
                self.paths.pop(old, None)
            gone.append(old)
        return gone

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=60)
            except Exception:
                os.killpg(self.process.pid, signal.SIGKILL)


# ---- the loop ----

class Loop(AsyncMixin, ModeBMixin, EnvsV7Mixin, LadderMixin, SingleLevelMixin, OpdMixin, TransferMixin, MultiMixin,
           SftMixin):
    @property
    def serving_adapter(self):
        return self.__dict__.get("_serving_adapter")

    @serving_adapter.setter
    def serving_adapter(self, value):
        # each newly served adapter is a new policy for finished-try reuse (mg_try_keep); a resume or in-place rollback
        # then restores the uid recorded with the state it loaded (same weights)
        import uuid

        self.__dict__["_serving_adapter"] = value
        self.__dict__["policy_uid"] = uuid.uuid4().hex[:12] if value is not None else "base"

    def __init__(self, study, trial, *, source, games_dir, output, server_dir, analysis_python, base_url,
                 deadline_epoch):
        self.study, self.trial, self.cfg = study, trial, settings(study.get("rl_loop"))
        self.source, self.games_dir, self.output = Path(source), games_dir, Path(output)
        self.server_dir, self.analysis_python, self.base_url = server_dir, analysis_python, base_url
        self.deadline = float(deadline_epoch)
        self.feed = Feed(self.output / "rl")
        self.gpu = GpuSampler()
        self.server = Server(base_url, study.get("served_model") or "Qwen/Qwen3.8-Flash-Next-NVFP4")
        self.work = Path("/tmp/rl1")
        self.adapters = self.work / "adapters"
        self.adapters.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.history = []          # round rows
        self.drop_alerts = []      # benches whose usable share fell under limits["usable_share"] (since the last round)
        self.kl_floor = {}         # probe KL between two scorings of the same weights (loop start)
        self.base = {}             # round-0 references: reward mean, eval means
        self.trainer = None
        self.last_wake = None
        self.server.dual = bool(self.cfg["dual"]) or bool(study.get("no_server"))  # no server: sleep/wake are no-ops

    async def wake(self):
        """Wake the play server only when the card has room for it beside the resident trainer (rl1-ttest-v1: the
        awake server's ~146 GiB did not fit); otherwise stop with the numbers."""
        if self.cfg["dual"]:
            return 0.0
        park = await self.trainer.must("park") if self.trainer is not None else None
        used, total = card_memory()
        need = self.cfg["limits"]["awake_gib"] - self.cfg["limits"]["asleep_gib"]
        self.last_wake = {"park": park, "free_gib": None if used is None else round(total - used, 2), "needed_gib": need}
        self.feed.emit("wake", **self.last_wake)
        if used is not None and total - used < need:
            raise StopLoop("play_memory", {**self.last_wake, "card_used_gib": round(used, 2)})
        return await self.server.wake()

    def left(self):
        return self.deadline - time.time()

    def elapsed_hours(self):
        return (time.time() - self.started) / 3600

    async def bench(self, name, games, passes, budgets, *, seed_base=None, arms=None, model="policy", points=None,
                    concurrency=None, reserve=600, wall=None, sampling=None, tail=None, own=None):
        """One level-bench run on the current adapter: ``games`` x ``passes`` fresh games from level 1 (``points``: a
        start-point file of its own, e.g. restored attempts). ``own`` (mg.lookahead_below): the bench's own
        ``try_keep`` roots, ``policy_version`` and ``server1_only`` (level_bench.LevelBench)."""
        from .level_bench import run_level_bench

        study = copy.deepcopy(self.study)
        study["served_model"] = model
        for key, value in (own or {}).items():
            study[key] = value
        if (self.cfg.get("mg") or {}).get("gate_overcommit"):
            study["gate_overcommit"] = float(self.cfg["mg"]["gate_overcommit"])
        if (self.cfg.get("mg") or {}).get("placement"):  # mg.placement "balanced": screens; RL time share (server 2 awake)
            study["two_server_placement"] = str(self.cfg["mg"]["placement"])
        if (self.cfg.get("mg") or {}).get("transport_backoff"):
            study["transport_backoff"] = list(self.cfg["mg"]["transport_backoff"])
        if tail is not None:  # mg.tail_fill: {"expected": {point id: tokens}, "last": held-out} (level_bench.tail_order)
            study["tail_fill"] = tail
        bench = dict(study["level_bench"])
        if points is not None:
            bench["start_points"] = str(points)
        bench.update(games=list(games), passes=int(passes), seed_base=seed_base,
                     concurrency=int(concurrency or min(self.cfg["concurrency"], int(passes) * len(games))),
                     token_budget_by_point={f"{g}-game": int(budgets[g]) for g in games}, arms=arms)
        if wall:
            bench["attempt_wall_seconds"] = int(wall)
        if sampling:  # every request of these tries (level_bench.sampling); recorded per reply for the trainer's actor
            bench["sampling"] = {**(bench.get("sampling") or {}), **sampling}
        study["level_bench"] = bench
        folder = self.output / "rl" / name
        self.gpu.mark("play")
        began = time.time()
        report = await run_level_bench(study, {**self.trial, "name": name}, source=self.source,
                                       games_dir=self.games_dir, output=folder, server_dir=self.server_dir,
                                       analysis_python=self.analysis_python, base_url=self.base_url,
                                       deadline_epoch=self.deadline - reserve)
        engine = report.get("engine") or {}
        rows = [r for r in report["attempts"] if usable(r)]
        self.log_drops(name, report["attempts"])
        tokens = sum(int(r.get("completion_tokens") or 0) for r in report["attempts"])
        play = {"seconds": round(time.time() - began, 1), "attempts": len(report["attempts"]), "usable": len(rows),
                "tokens": tokens, "tok_s": round(tokens / max(1.0, time.time() - began), 1),
                "engine_tok_s": engine.get("tokens_per_second"), "preemptions": engine.get("preemptions")}
        return folder, rows, play

    def log_drops(self, name, attempts):
        if name == "restore-check" or name.startswith("prescreen-"):  # check-only: "restored" by design, not play
            return None
        info = drops(attempts)
        if info["dropped"]:
            self.feed.emit("drops", bench=name, **info)
        if info["usable_share"] is not None and info["usable_share"] < self.cfg["limits"]["usable_share"]:
            self.drop_alerts.append(f"{name}: usable share {info['usable_share']} < {self.cfg['limits']['usable_share']}")
        return info

    def game_rows(self, rows, game):
        return [r for r in rows if str(r["game"]).startswith(game)]

    async def swap_in(self, path):
        timings = {}
        timings["wake_seconds"] = await self.wake()
        timings["load_seconds"] = await self.server.load("policy", path)
        return timings

    async def evaluate(self, label, attempts):
        g, e = self.cfg["train_game"], self.cfg["eval_game"]
        folder, rows, play = await self.bench(f"eval-{label}", [g, e], attempts,
                                              {g: self.cfg["train_budget"], e: self.cfg["eval_budget"]},
                                              seed_base=self.cfg["eval_seed_base"])
        return folder, rows, {"label": label, "play": play, **self.eval_stats(rows)}

    def eval_stats(self, rows):
        out = {}
        for game in (self.cfg["train_game"], self.cfg["eval_game"]):
            values = [attempt_reward(r) for r in self.game_rows(rows, game)]
            out[game] = dict(zip(("mean", "se"), mean_se(values)), n=len(values))
        return out

    def windows(self, folder, rows, weights=None, prefix=""):
        weights = weights or {}
        served = {"served_route": self.served_dir} if getattr(self, "served_dir", None) else {}
        return [{"rollout": str(Path(folder) / "attempts" / r["label"] / "rollout.jsonl"),
                 "w": float(weights.get(r["label"], 1.0)), "name": prefix + r["label"], "label": r["label"], **served}
                for r in rows]

    def check_stop(self, row):
        """The plan's stop rules on one round's row; returns (stop reason or None, alerts)."""
        limits, alerts = self.cfg["limits"], list(self.drop_alerts)
        self.drop_alerts.clear()
        update = row.get("update") or {}
        if update and (update.get("stop") == "nan_or_inf"):
            return "nan_or_inf", alerts
        mismatch = [w["mismatch"] for w in update.get("windows", []) if (w.get("mismatch") or {}).get("tokens")]
        if mismatch:
            mean_abs = sum(m["mean_abs"] * m["tokens"] for m in mismatch) / sum(m["tokens"] for m in mismatch)
            mean = sum(m["mean"] * m["tokens"] for m in mismatch) / sum(m["tokens"] for m in mismatch)
            row["mismatch"] = {"mean_abs": round(mean_abs, 5), "mean": round(mean, 5)}
            if limits["mismatch_rule"] == "split":
                self.engine_gap = self.engine_gap if getattr(self, "engine_gap", None) is not None else mean_abs
                if mean_abs > self.engine_gap + limits["split_engine_rise"]:
                    alerts.append(f"engine mismatch {mean_abs:.4f} above round 1 ({self.engine_gap:.4f}) + "
                                  f"{limits['split_engine_rise']}")
            elif mean_abs > limits["mismatch_mean_abs"] or abs(mean) > limits["mismatch_mean"]:
                return "mismatch", alerts
        for name, probe in (update.get("probes") or {}).items():
            floor = self.kl_floor.get(name, 0.0) if limits["kl_floor_relative"] else 0.0
            if (probe.get("kl_previous") or 0) > floor + limits["kl_previous"]:
                return f"kl_previous:{name}", alerts
            if name.startswith("untrained") and limits["kl_untrained_alert"]:
                if (probe.get("kl_base") or 0) > floor + limits["kl_untrained_backstop"]:
                    return f"kl_base_untrained_backstop:{name}", alerts
                if (probe.get("kl_base") or 0) > floor + limits["kl_base_untrained"]:
                    alerts.append(f"lp85 KL {name} {probe.get('kl_base')} > floor + {limits['kl_base_untrained']}")
            elif name.startswith("untrained") and (probe.get("kl_base") or 0) > floor + limits["kl_base_untrained"]:
                return f"kl_base_untrained:{name}", alerts
        rounds = [r for r in self.history if r.get("stats")]
        base = self.base.get("reward_mean")
        if base is not None and len(rounds) >= 2:
            last = [r["stats"]["reward_mean"] for r in rounds[-2:]]
            if sum(last) / 2 < base - limits["collapse_drop"]:
                return "reward_collapse", alerts
            if sum(last) / 2 >= limits["pass_reward"]:
                return "pass_reached", alerts
        evaluation = row.get("eval") or {}
        game = self.cfg["eval_game"]
        if evaluation and self.base.get("eval"):
            now, then = evaluation.get(game) or {}, self.base["eval"].get(game) or {}
            if now.get("mean") is not None and then.get("mean") is not None and now.get("se") and then.get("se"):
                se = math.sqrt(now["se"] ** 2 + then["se"] ** 2)
                if now["mean"] < then["mean"] - limits["eval_se"] * se:
                    return "untrained_eval_down", alerts
        util = (row.get("gpu") or {}).get("all", {}).get("util_mean")
        if util is not None:
            if util < limits["util_alert"]:
                alerts.append(f"gpu_util {util}")
            lows = [r for r in self.history[-1:] + [row] if (r.get("gpu") or {}).get("all", {}).get("util_mean", 100)
                    < limits["util_stop"]]
            if len(self.history) >= 1 and len(lows) == 2:
                return "gpu_util_low", alerts
        stats, first = row.get("stats") or {}, (self.history[0].get("stats") if self.history else None) or {}
        for key in ("tokens_per_reply", "actions_per_level"):
            if stats.get(key) and first.get(key) and stats[key] > limits["length_ratio"] * first[key]:
                alerts.append(f"{key} {stats[key]} vs {first[key]}")
        if stats.get("capped_share") is not None and first.get("capped_share") is not None and (
                stats["capped_share"] > first["capped_share"] + limits["capped_points"]):
            alerts.append(f"capped_share {stats['capped_share']} vs {first['capped_share']}")
        if self.elapsed_hours() > limits["elapsed_hours"]:
            return "elapsed", alerts
        round_s = (row.get("timing") or {}).get("round_s") or 0
        if limits["loop_hours"] and time.time() - self.loop_started + round_s > 3600 * float(limits["loop_hours"]):
            return "loop_hours", alerts  # another round would end past the loop's wall
        return None, alerts

    def advantages(self, rows):
        rewards = [attempt_reward(r) for r in rows]
        adv = leave_one_out(rewards)
        weights = {r["label"]: max(0.0, a) for r, a in zip(rows, adv)}
        return rewards, adv, weights

    def spawn_trainer(self, gate=None, extra_env=None):
        """The trainer process now (its ready wait comes later, start_trainer(spawned=...)): mg.host_load spawns it at
        round 1's start, its load on the host until the gate."""
        from . import trainer_prespawn

        command, env = trainer_prespawn.command_for(self.cfg, self.source, self.work, gate)
        env.update(extra_env or {})
        return Trainer(self.work / "trainer", command, env, self.output / "rl" / "trainer.log")

    def compact_always(self):
        """hero-v6b (24 Oct): a compacted trainer (RL1_EXPERT_COMPACT) refuses any forward off the fused expert path, so
        every op from its start (base reference, park check, probes) runs under mg.trainer_fast, not only those after
        the loop first applies it."""
        fast = (self.cfg.get("mg") or {}).get("trainer_fast")
        if (self.cfg.get("trainer_env") or {}).get("RL1_EXPERT_COMPACT") == "1" and fast:
            self.trainer.always = {**(getattr(self.trainer, "always", None) or {}), **fast}

    async def start_trainer(self, gate=None, extra_env=None, spawned=None):
        from . import trainer_prespawn

        if spawned is not None:
            self.trainer = spawned
            began = time.time()
            self.gpu.mark("trainer_load")
            state = await self.trainer.ready(self.cfg["trainer_timeout"])
            self.expert_mask_trainer(state.get("loaded"))
            self.compact_always()
            return {"seconds": round(time.time() - began, 1), "loaded": state.get("loaded"), "spawned_early": True}
        # mg.trainer_overlap (gate): the first CUDA use waits for the gate file
        command, env = trainer_prespawn.command_for(self.cfg, self.source, self.work, gate)
        env.update(extra_env or {})
        # start-up (b): the launcher's pre-spawned trainer when it is this exact process, else a fresh one
        pid, pre = await asyncio.to_thread(trainer_prespawn.take, self.work / "trainer", command, env,
                                         self.output / "rl" / "trainer.log")
        if pid is not None:
            self.trainer = Trainer.adopt(self.work / "trainer", pid)
            log = self.output / "rl" / "trainer.log"  # readers (eff watcher, park check) keep the loop's path
            if not log.exists() and not pre["log_equal"]:
                log.parent.mkdir(parents=True, exist_ok=True)
                log.symlink_to(pre["log"])
        else:
            self.trainer = Trainer(self.work / "trainer", command, env, self.output / "rl" / "trainer.log")
        began = time.time()
        self.gpu.mark("trainer_load")
        state = await self.trainer.ready(self.cfg["trainer_timeout"])
        self.expert_mask_trainer(state.get("loaded"))
        self.compact_always()
        out = {"seconds": round(time.time() - began, 1), "loaded": state.get("loaded")}
        if pre is not None:
            out["prespawn"] = pre
        return out

    async def fit_and_parity(self):
        self.gpu.mark("fit")
        sleep = await self.server.sleep()
        fit = await self.trainer.call("fit", timeout=3600, save=str(self.adapters / "perturbed"),
                                      **{k: v for k, v in self.cfg["fit"].items() if v is not None})
        self.feed.emit("fit", sleep_seconds=sleep, **{k: v for k, v in fit.items() if k != "traceback"})
        if not fit.get("ok"):
            raise StopLoop("no_native_fit", fit)
        # the dense part alone (attention, GDN, shared expert): its own shift on both sides
        await self.trainer.must("save", path=str(self.adapters / "perturbed-dense"), only="dense")
        # server side first, then the trainer beside the sleeping server: the awake server (~134 GiB) and the
        # resident trainer (~106 GiB) leave no room for scoring (rl1-probe-v3)
        wake = await self.wake()
        sequences = [(await self.server.tokens(make_text(i, 9000)))[:4096] for i in (1, 2)]
        server_base = [await self.server.prompt_logprobs(self.server.model, s) for s in sequences]
        repeat = [await self.server.prompt_logprobs(self.server.model, s) for s in sequences]
        noise = [abs(a - b) for x, y in zip(server_base, repeat) for a, b in zip(x, y)]
        noise = sum(noise) / len(noise)
        load = await self.server.load("policy", self.adapters / "perturbed")
        server_adapter = [await self.server.prompt_logprobs("policy", s) for s in sequences]
        await self.server.load("policy", self.adapters / "perturbed-dense")
        server_dense = [await self.server.prompt_logprobs("policy", s) for s in sequences]
        await self.server.unload("policy")
        await self.server.sleep()
        trainer_adapter = (await self.trainer.must("score_ids", sequences=sequences, adapter="on"))["logprobs"]
        trainer_base = (await self.trainer.must("score_ids", sequences=sequences, adapter="off"))["logprobs"]
        check = parity(server_base, server_adapter, trainer_base, trainer_adapter)
        trainer_dense = (await self.trainer.must("score_ids", sequences=sequences, adapter="on",
                                                 zero_experts=True))["logprobs"]
        check["dense"] = parity(server_base, server_dense, trainer_base, trainer_dense)
        limits = self.cfg["limits"]
        dense = check["dense"]
        # the dense shift can sit near the server's own repeat noise: then its correlation cannot be judged
        dense["judged"] = dense["shift_trainer_mean_abs"] >= 3 * noise
        check.update(server_repeat_noise=round(noise, 5),
                     ok=(check["shift_corr"] is not None and check["shift_corr"] >= limits["parity_corr"]
                         and check["base_mean_abs"] <= limits["parity_base_abs"]
                         and (not dense["judged"] or (dense["shift_corr"] or 0) >= limits["parity_corr"])))
        self.feed.emit("parity", wake_seconds=wake, load_seconds=load, **check)
        if not check["ok"]:
            raise StopLoop("parity", check)
        await self.trainer.must("reset", lr=self.cfg["lr"])
        saved = await self.trainer.must("save", path=str(self.adapters / "r00"))
        await self.wake()
        await self.server.load("policy", self.adapters / "r00")
        return saved

    async def round_zero(self):
        g = self.cfg["train_game"]
        if self.cfg["round0_from"]:
            source = Path(self.cfg["round0_from"])
            saved = json.loads((source / "round0.json").read_text())
            played = Path((saved.get("play") or {}).get("reused") or source)  # a probe that reused another's play
            rows = json.loads((played / "round-00" / "level-bench.json").read_text())["attempts"]
            self.log_drops("round-00", rows)
            folder, rows, play = played / "round-00", [r for r in rows if usable(r)], {"reused": str(played)}
            eval_all = json.loads((source / "eval-r00" / "level-bench.json").read_text())["attempts"]
            self.log_drops("eval-r00", eval_all)
            eval_folder, eval_rows = source / "eval-r00", [r for r in eval_all if usable(r)]
            evaluation = {**saved["eval"], **self.eval_stats(eval_rows)}  # recomputed under the outcome map
        elif self.cfg["round0_play_from"]:  # a finished round-0 play of a stopped run (rl1-probe-v7), eval fresh
            source = Path(self.cfg["round0_play_from"])
            rows = json.loads((source / "round-00" / "level-bench.json").read_text())["attempts"]
            self.log_drops("round-00", rows)
            folder, rows, play = source / "round-00", [r for r in rows if usable(r)], {"reused": str(source)}
            eval_folder, eval_rows, evaluation = await self.evaluate("r00", self.cfg["eval_attempts"])
        else:
            folder, rows, play = await self.bench("round-00", [g], self.cfg["train_attempts"],
                                                  {g: self.cfg["train_budget"]})
            eval_folder, eval_rows, evaluation = await self.evaluate("r00", self.cfg["eval_attempts"])
        rows = self.game_rows(rows, g)
        rewards, adv, weights = self.advantages(rows)
        stats = round_stats(rows, rewards, adv, folder, int(self.study["completion_cap"]))
        self.base.update(reward_mean=stats["reward_mean"], eval=evaluation)
        row = {"round": 0, "stats": stats, "play": play, "eval": evaluation, "rewards": rewards}
        self.history.append(row)
        (self.output / "rl" / "round0.json").write_text(json.dumps(row, indent=1, default=str))
        self.feed.emit("round", **row)
        self.feed.emit("eval", **evaluation)
        # probe windows (fixed for the run), never trained (audit, 14 Oct): 2 sb26 and 2 lp85 attempts of the seeded
        # round-0 eval; the lp85 drift stop reads the "untrained-" ones
        probes = self.windows(eval_folder, self.game_rows(eval_rows, g)[:2], prefix="heldout-")
        probes += self.windows(eval_folder, self.game_rows(eval_rows, self.cfg["eval_game"])[:2], prefix="untrained-")
        return folder, rows, weights, probes

    async def probe_rates(self, folder, rows, weights, probes):
        held = [r for i, r in enumerate(sorted(rows, key=lambda r: r["label"])) if i % 4 == 3][: self.cfg["probe_held"]]
        train = [r for r in rows if r not in held]
        rewards = {r["label"]: attempt_reward(r) for r in rows}
        mean = sum(rewards.values()) / len(rewards)
        held_specs = [{**w, "name": ("held-up-" if rewards[r["label"]] > mean else "held-down-") + r["label"]}
                      for w, r in zip(self.windows(folder, held), held)]
        _, _, train_weights = self.advantages(train)
        results = []
        self.gpu.mark("train")
        await self.server.sleep()
        for rate in self.cfg["probe_rates"]:
            await self.trainer.must("reset", lr=rate)
            base = await self.trainer.must("score", windows=held_specs + probes, label="base")
            rows_rate = []
            for update in range(1, self.cfg["probe_updates"] + 1):
                result = await self.trainer.call(
                    "update", timeout=self.cfg["update_timeout"], lr=rate, clip=self.cfg["clip"],
                    max_windows=self.cfg["max_windows"], probes=held_specs + probes,
                    attempts=[{**w, "w": train_weights[r["label"]]} for w, r in zip(self.windows(folder, train), train)])
                probe = result.get("probes") or {}
                summary = {
                    "rate": rate, "update": update, "ok": result.get("ok"), "loss": result.get("loss"),
                    "grad_norm": result.get("grad_norm"), "update_norm": result.get("update_norm"),
                    "adapter_norm": result.get("adapter_norm"), "train_seconds": result.get("train_seconds"),
                    "torch_peak_gib": result.get("torch_peak_gib"), "card_peak_gib": result.get("card_peak_gib"),
                    "held_up_gain": avg(-p["logratio_base"] for n, p in probe.items() if n.startswith("held-up") and "logratio_base" in p),
                    "held_down_gain": avg(-p["logratio_base"] for n, p in probe.items() if n.startswith("held-down") and "logratio_base" in p),
                    "untrained_kl_base": avg(p["kl_base"] for n, p in probe.items() if n.startswith("untrained") and "kl_base" in p),
                    "kl_previous_max": max((p.get("kl_previous") or 0 for p in probe.values()), default=None),
                    "mismatch": [w.get("mismatch") for w in result.get("windows", [])][:3],
                    "update_wall_seconds": result.get("seconds"),
                    "unpark_seconds": (result.get("unparked") or {}).get("seconds")}
                rows_rate.append(summary)
                self.feed.emit("probe_update", **summary)
                if not result.get("ok"):
                    break
            results.append({"rate": rate, "base": base.get("windows"), "updates": rows_rate})
        limits = self.cfg["limits"]
        good = [r["rate"] for r in results if r["updates"] and r["updates"][-1].get("held_up_gain") is not None
                and r["updates"][-1]["held_up_gain"] > 0
                and (r["updates"][-1].get("untrained_kl_base") or 0) <= limits["kl_base_untrained"]
                and max((u.get("kl_previous_max") or 0) for u in r["updates"]) <= limits["kl_previous"]]
        pick = {"rates": [r["rate"] for r in results], "chosen": max(good) if good else None}
        (self.output / "rl" / "lr-probe.json").write_text(json.dumps({"results": results, "pick": pick}, indent=1,
                                                                    default=str))
        self.feed.emit("lr_pick", **pick)
        return pick

    async def fresh_logprobs(self, payload, expected):
        """The server's cold log-probability of each prompt token for a served request (its own messages, images and
        template settings), checked token for token against ``expected``: index i is token i (0: NaN)."""
        body = {**payload, "model": self.server.model, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0}
        reply = await self.server.post("/v1/chat/completions", body, timeout=1200)
        entries = reply.get("prompt_logprobs") or ((reply.get("choices") or [{}])[0].get("prompt_logprobs")) or []
        return align_prompt_logprobs(entries, expected)

    async def mismatch_split(self, folder, rows):
        """Round-0 play = train split: on the earlier reply tokens of two trained windows, the engine's sampling-time
        log-probabilities, two cold server passes of the last reply's served request (images included) and the
        trainer's image-aware window pass (rl1-loop-v2's version scored bare ids on both sides: invalid)."""
        await self.server.sleep()
        made = (await self.trainer.must("split_windows", windows=self.windows(folder, rows[:2])))["windows"]
        await self.wake()
        out = {}
        for w in made:
            first = await self.fresh_logprobs(w["payload"], w["prompt_ids"])
            second = await self.fresh_logprobs(w["payload"], w["prompt_ids"])
            fresh, repeat = [first[p] for p in w["positions"]], [second[p] for p in w["positions"]]
            out[w["name"]] = {"request": w["request"], "prompt_tokens": len(w["prompt_ids"]),
                              "image_tokens": w["image_tokens"], "prompt_same": w["prompt_same"],
                              "engine_trainer": compare_logprobs(w["engine"], w["trainer"]),
                              "engine_fresh": compare_logprobs(w["engine"], fresh),
                              "fresh_trainer": compare_logprobs(fresh, w["trainer"]),
                              "fresh_repeat": compare_logprobs(fresh, repeat)}
        await self.server.sleep()
        self.feed.emit("mismatch_split", windows=out)
        return out

    async def signed_update(self, number, folder, rows, adv, probes):
        """one optimizer step over the ``per_side`` largest positive and negative advantages
        (TOPR: negatives with the ratio clipped to [0, 1] in the trainer), negatives scaled so push-down weight x
        served trained tokens equals push-up; returns ``dosed_update``'s merged result with the balance."""
        k, tail = int(self.cfg["per_side"]), self.cfg["train_tail"]
        specs = [{**s, **({"window": int(tail)} if tail else {})} for s in self.windows(folder, rows)]
        a = {r["label"]: float(v) for r, v in zip(rows, adv)}
        up = sorted([s for s in specs if a[s["label"]] > 0], key=lambda s: -a[s["label"]])[:k]
        down = sorted([s for s in specs if a[s["label"]] < 0], key=lambda s: a[s["label"]])[:k]
        targets = (await self.trainer.must("targets", windows=up + down))["targets"]
        mass_up = sum(a[s["label"]] * targets.get(s["name"], 0) for s in up)
        mass_down = sum(-a[s["label"]] * targets.get(s["name"], 0) for s in down)
        scale = mass_up / mass_down if mass_down > 0 else 0.0
        batch = [{**s, "w": a[s["label"]]} for s in up] + [{**s, "w": a[s["label"]] * scale} for s in down if scale > 0]
        balance = {"up_attempts": len(up), "down_attempts": len(down), "down_scale": round(scale, 4),
                   "served_up": round(mass_up, 1), "served_down": round(mass_down, 1),
                   "up_tokens": sum(targets.get(s["name"], 0) for s in up),
                   "down_tokens": sum(targets.get(s["name"], 0) for s in down)}
        merged, path = await self.dosed_update(number, folder, rows, None, probes, batches=[batch] if batch else [],
                                               signed=True, extra=balance)
        return merged, path

    def step_lr(self, index):
        """(lr, multiplier) of step ``index`` of this update. ``mg.lr_schedule`` (pilot 1, RL pipeline rule: warm-up
        on every start and resume, decay inside the staged data): global step k (accepted updates' steps + 1 +
        index) decays linearly from 1 after ``warmup`` steps to ``end`` at step ``total``; the j-th step this process
        sends (start or resume) is scaled by a ramp ``start`` .. 1 over ``warmup`` steps. ``cfg["lr"]`` is the
        peak (the dose rule steers it)."""
        sched = (self.cfg.get("mg") or {}).get("lr_schedule")
        if not sched:
            return float(self.cfg["lr"]), 1.0
        warm, total, end = int(sched.get("warmup", 4)), int(sched["total"]), float(sched.get("end", 0.5))
        start = float(sched.get("start", 0.25))
        k = int(getattr(self, "mg_steps_done", 0)) + index + 1
        j = int(getattr(self, "proc_steps", 0)) + 1
        decay = 1.0 - (1.0 - end) * min(1.0, max(0.0, (k - warm) / max(1, total - warm)))
        ramp = 1.0 if warm <= 1 or j >= warm else start + (1.0 - start) * (j - 1) / (warm - 1)
        mult = round(decay * ramp, 6)
        return float(self.cfg["lr"]) * mult, mult

    def window_guard(self, number, step):
        """mg.window_guard {"peak_gib", "fallback"} (MFU-1 D, 48K tails ship with an in-run memory
        check): a step whose torch peak exceeds ``peak_gib`` (the prediction + 5%) sets lvl_window back to
        ``fallback`` from the next update on (the specs are built per update)."""
        guard = (self.cfg.get("mg") or {}).get("window_guard")
        if not guard or int(self.cfg["lvl_window"]) <= int(guard["fallback"]):
            return False
        oom = step.get("why") == "oom"  # the trainer's out-of-memory retry (``oom_fallback_window``)
        peak = step.get("torch_peak_gib")
        if not oom and (peak is None or float(peak) <= float(guard["peak_gib"])):
            return False
        was, self.cfg["lvl_window"] = int(self.cfg["lvl_window"]), int(guard["fallback"])
        why = "oom" if oom else "peak"
        self.feed.emit("window_fallback", round=number, step=step.get("step"), why=why,
                       peak_gib=None if peak is None else float(peak), limit_gib=float(guard["peak_gib"]), window=was,
                       to=self.cfg["lvl_window"], **({"error": step.get("error")} if oom else {}))
        self.feed.emit("alert", why=f"window guard ({why}): lvl_window falls back to {self.cfg['lvl_window']} (alert only)",
                       round=number, peak_gib=None if peak is None else float(peak), to=self.cfg["lvl_window"])
        return True

    async def dosed_update(self, number, folder, rows, weights, probes, *, batches=None, signed=False, probe_last=False,
                           extra=None, max_windows=None, cmd_extra=None, early=None):
        """One round's update as ``steps_per_round`` optimizer steps (``dose_batches``); the probes are scored after
        every step (KL to base and to the previous step, in the feed: ``update_step``); the adapter is saved after
        the last. Returns the merged result (``ok``, ``windows`` of every step, ``probes``: the last step's, with
        ``kl_previous`` the largest step value per probe) and the adapter folder to serve.
        ``early`` (mg5e; PPO-style KL early stopping, the trainer holding "previous" at the served state): probes
        every ``probe_every`` steps and after the last; mean KL-to-previous above ``floor`` + ``rollback_at`` ends the
        update as ``early.action`` "rollback" (nothing saved), above ``floor`` + ``stop_at`` (or a quadratic forecast
        for the next probe above ``rollback_at``) keeps the steps so far ("stop"); the adapter is saved after the
        last step taken."""
        path = self.adapters / f"r{number:02d}"
        if hasattr(self, "served_restored"):  # (c): a resume's restored route records are in place first
            await self.served_restored()
        if batches is None:
            batches = dose_batches(self.windows(folder, rows), weights, self.cfg["steps_per_round"])
        if not batches:
            self.feed.emit("update_step", round=number, step=0, of=0, skipped="no attempt with A > 0")
            if self.serving_adapter is None:
                raise StopLoop("no_positive_advantage", {"round": number})
            return {"ok": True, "skipped": True, "windows": [], "probes": None, "trained_tokens": 0}, self.serving_adapter
        merged, steps, kl_prev = {"ok": True, "windows": [], "steps": []}, [], {}
        mg = self.cfg.get("mg") or {}
        sent = []  # the lr of every step sent (mg.lr_schedule)
        measure = None  # mg.k3_measure: the step's dose, measured after it
        if len(batches) == 1 and int(mg.get("k3_measure") or 0) > 0:
            whole = batches[0]
            k = min(len(whole), int(mg["k3_measure"]))
            measure = [whole[round(i * (len(whole) - 1) / max(1, k - 1))] for i in range(k)] if k > 1 else whole[:1]
            # pi_old = the step's own forward on the subsample (keep_old); forward-only k3 after the step (23 Oct:
            # the old pass + lr-0 update took ~10 min of ovf1's U1)
        policy = {}
        gspo = {"policy_loss": "gspo", "gspo_clip": list(mg.get("gspo_clip") or [3e-4, 4e-4]),
                **({"gspo_norm": mg["gspo_norm"]} if mg.get("gspo_norm") else {})}
        if mg.get("policy_loss") == "gspo":  # pi_old = the served state, scored at the start (GSPO arm)
            policy = gspo
            seen = set()  # mg.epochs: a window recurs in later epochs; scored once (step 1's included then)
            later = [spec for batch in batches[1:] for spec in batch if id(spec) not in seen and not seen.add(id(spec))]
            if later:  # step 1 runs at the served state itself (s = 1); later steps need its log-probabilities
                # the update's window path (prefix sharing and chunk, trainer_fast: prefetch, snapshots, stage W's
                # encoder processes and window cache); the trainer's ``always`` carries trainer_fast as well
                old = await self.trainer.must("old_logprobs", timeout=self.cfg["update_timeout"], attempts=later,
                                              signed=signed, **{k: v for k, v in (cmd_extra or {}).items()
                                                                if k not in OLD_PASS_SKIP})
                rows = old.pop("engine_gap", None)
                if rows is not None:  # live engine-vs-pi_old per lag (22 Oct standing check; MTP exactness bar)
                    old["engine_gap"] = engine_gap_summary(rows, getattr(self, "mg_lag_now", None) or {})
                self.feed.emit("gspo_old", round=number, **old)
                merged["gspo_old_seconds"] = old.get("seconds")
        if mg.get("r3_check") and not getattr(self, "r3_checked", False):  # R3 check: before the first update
            self.r3_checked = True
            check = await self.trainer.must("r3_check", timeout=self.cfg["update_timeout"], attempts=batches[0],
                                            max_windows=max_windows or len(batches[0]), signed=signed,
                                            **(cmd_extra or {}), **policy)
            self.feed.emit("r3_check", round=number, **check)
            if (mg["r3_check"] or {}).get("stop", True):
                raise StopLoop("r3_check_done", {"round": number})
        fell_back = 0  # mg.window_guard: the window an out-of-memory retry trained at
        for index, batch in enumerate(batches):
            last = index == len(batches) - 1
            if early:
                probed = last or (index + 1) % int(early.get("probe_every") or 1) == 0
            else:
                probed = last or not probe_last
            lr, mult = self.step_lr(index)
            sent.append(lr)
            # standard dosing: PPO/SB3 target_kl on the step's k3 approx-KL to pi_old. an
            # alert only (the maintainer or coordinator decides a stop); mg.kl_stop_auto restores the automatic stop
            kl_stop = (1.5 * float(mg["target_kl"]) if mg.get("dosing") == "standard" and mg.get("target_kl")
                       and policy and index > 0 and mg.get("kl_stop_auto") else None)
            result = await self.trainer.call(
                "update", timeout=self.cfg["update_timeout"], lr=lr, clip=self.cfg["clip"],
                max_windows=max_windows or len(batch), probes=probes if probed else None,
                save=str(path) if last and not early else None, attempts=batch, signed=signed, **(cmd_extra or {}), **policy,
                **({"kl_stop": kl_stop} if kl_stop is not None else {}),
                **({"keep_old": [m.get("name") or Path(m["rollout"]).parent.name for m in measure]}
                   if measure is not None else {}),
                **({"oom_fallback_window": int(guard["fallback"])}
                   if (guard := mg.get("window_guard")) and int(self.cfg["lvl_window"]) > int(guard["fallback"]) else {}))
            if result.get("window_fallback"):  # the trainer ran out of memory and redid this update at the fallback
                self.window_guard(number, {"step": index + 1, **result["window_fallback"]})
                fell_back = int(result["window_fallback"]["to"])
            probe = result.get("probes") or {}
            for name, p in probe.items():  # early: KL to the served state (held), the latest value counts
                kl_prev[name] = (p.get("kl_previous") or 0.0) if early else max(kl_prev.get(name, 0.0), p.get("kl_previous") or 0.0)
            self.proc_steps = getattr(self, "proc_steps", 0) + 1
            per = max(1, len(batches) // max(1, int(mg.get("epochs") or 1)))
            step = {"step": index + 1, "of": len(batches), "pass": index // per + 1, "attempts": len(batch), "ok": result.get("ok"),
                    **({"k3_target": float(mg["target_kl"]), "k3_stop": 1.5 * float(mg["target_kl"])}
                       if mg.get("dosing") == "standard" and mg.get("target_kl") and policy else {}),
                    "lr": lr, "lr_mult": mult,
                    "windows": len(result.get("windows") or []), "prefix_reused": result.get("prefix_reused"),
                    "prefix_seconds": round(sum(float(w.get("prefix_seconds") or 0) for w in result.get("windows") or []), 1),
                    # profile (trainer efficiency): forward / backward over the windows, optimizer,
                    # waits for the encoder thread, card util and peak
                    **{k: round(sum(float(w.get(k) or 0) for w in result.get("windows") or []), 1)
                       for k in ("forward_seconds", "backward_seconds")},
                    **{k: result.get(k) for k in ("step_seconds", "window_wait_seconds", "gpu_util_mean", "torch_peak_gib")},
                    "w_sum": round(sum(s["w"] for s in batch), 4), "loss": result.get("loss"),
                    "grad_norm": result.get("grad_norm"), "update_norm": result.get("update_norm"),
                    "entropy": result.get("entropy"),  # mean policy token entropy on trained tokens (trainer, nats)
                    # k3 approx-KL to pi_old on this step's batch before its optimizer step (per reply on s_i; per
                    # token, diagnostic) and GSPO's clipped share
                    **{k: result.get(k) for k in ("approx_kl", "approx_kl_tok", "clip_frac", "log_s_q") if result.get(k) is not None},
                    **({"kl_stopped": True, "applied": False, "kl_stop": result.get("kl_stop")} if result.get("kl_stopped") else {}),
                    "trained_tokens": result.get("trained_tokens"), "train_seconds": result.get("train_seconds"),
                    "probe_seconds": result.get("probe_seconds"),
                    "kl_base": {n: p.get("kl_base") for n, p in probe.items()},
                    **({"kl_base_pinned": {n: p.get("kl_base_pinned") for n, p in probe.items()}}
                       if any(p.get("kl_base_pinned") is not None for p in probe.values()) else {}),
                    "kl_previous": {n: p.get("kl_previous") for n, p in probe.items()},
                    "logratio_base": {n: p.get("logratio_base") for n, p in probe.items() if p.get("logratio_base") is not None},
                    **({"gspo": {**{k: sum(int(((w.get("signed") or {}).get("gspo") or {}).get(k) or 0)
                                           for w in result.get("windows") or []) for k in ("replies", "clipped")},
                                 "clipped_alt": {b: sum(int((((w.get("signed") or {}).get("gspo") or {}).get("clipped_alt")
                                                              or {}).get(b) or 0) for w in result.get("windows") or [])
                                                 for b in ("0.004", "0.004_0.005")},
                                 # pi_old over the sampler outside [0.5, 2]: replies (the TIS unit) and tokens (23 Oct)
                                 "tis_out": {u: sum(int((((w.get("signed") or {}).get("gspo") or {}).get("tis_out")
                                                          or {}).get(u) or 0) for w in result.get("windows") or [])
                                             for u in ("replies", "tokens")}}}
                       if policy else {}),
                    "tis": [w.get("tis") for w in result.get("windows", [])],
                    **({"tis_out": {u: sum(int(((w.get("signed") or {}).get("tis_out") or {}).get(u) or 0)
                                           for w in result.get("windows") or []) for u in ("tokens", "of")}}
                       if not policy and any((w.get("signed") or {}).get("tis_out") for w in result.get("windows") or [])
                       else {}),
                    **({"aux": result["aux"]} if result.get("aux") else {}),  # mg.nextboard: the aux loss's step stats
                    "errors": result.get("errors") or []}  # windows the trainer could not make (mg5d: were silent)
            if signed:
                parts = [w.get("signed") or {} for w in result.get("windows", [])]
                up, down = sum(x.get("up", 0) for x in parts), sum(x.get("down", 0) for x in parts)
                tokens = sum(x.get("down_tokens", 0) for x in parts)
                raw_down = sum(int(x.get("down_tokens", 0)) for x in parts)  # trained tokens with a negative weight
                step.update(extra or {}, up_weight=round(up, 3), down_weight=round(down, 3), raw_down_tokens=raw_down,
                            raw_up_tokens=int(result.get("trained_tokens") or 0) - raw_down,
                            up_share=round(up / (up + down), 4) if up + down else None,
                            down_clipped=round(sum(x.get("down_clipped", 0) * x.get("down_tokens", 0) for x in parts)
                                               / tokens, 5) if tokens else None)
            if getattr(self, "served_dir", None):  # R3: served-route coverage and train/serve routing agreement
                step["served"] = (await self.trainer.must("route", reset_counts=True)).get("served")
                acc = self.__dict__.setdefault("mg_served_acc", {})  # per update, for the round row
                win = (step["served"] or {}).get("windows") or {}
                for key in ("windows", "no_record", "tokens", "unresolved"):
                    acc[key] = acc.get(key, 0) + int(win.get(key) or 0)
            self.feed.emit("update_step", round=number, **step)
            self.window_guard(number, step)
            out = step.get("tis_out") or {}
            if out.get("of") and out["tokens"] / out["of"] > float(mg.get("tis_out_alert", 0.05)):
                self.feed.emit("alert", why="TIS: token share outside [0.5, 2] over the alert line", round=number,
                               step=index + 1, share=round(out["tokens"] / out["of"], 4), **out)
            if step.get("k3_stop") is not None and result.get("approx_kl") is not None and not result.get("kl_stopped"):
                over = "stop" if float(result["approx_kl"]) > step["k3_stop"] else (
                    "target" if float(result["approx_kl"]) > step["k3_target"] else None)
                if over:  # alert-only dosing: the step was taken
                    self.feed.emit("alert", why=f"k3 over {over} (alert only, step taken)", round=number, step=index + 1,
                                   of=len(batches), approx_kl=result["approx_kl"], k3_target=step["k3_target"],
                                   k3_stop=step["k3_stop"], clip_frac=result.get("clip_frac"), lr=lr)
            if result.get("kl_stopped"):  # the step was not taken: the update ends at the previous step
                merged["early"] = {"action": "kl_stop", "steps": index, "of": len(batches), "approx_kl": result.get("approx_kl"),
                                   "limit": result.get("kl_stop")}
                merged["kl_stopped"] = True
                break
            steps.append(step)
            merged["windows"] += result.get("windows", [])
            for key in ("stop", "unparked") if index == 0 else ("stop",):
                if result.get(key) is not None:
                    merged[key] = result[key]
            if not result.get("ok"):
                merged.update(ok=False, stop=result.get("stop") or "update_failed")
                break
            if probe:
                merged["probes"] = {n: {**p, "kl_previous": kl_prev[n]} for n, p in probe.items()}
            for key in ("grad_norm_after_clip", "adapter_norm", "torch_peak_gib", "card_peak_gib", "gpu_util_mean", "lr"):
                merged[key] = result.get(key)
            if early and probe:
                vals = [p.get("kl_previous") for p in probe.values() if p.get("kl_previous") is not None]
                if not vals:
                    raise RuntimeError("early stopping needs kl_previous on every probe (trainer reference missing)")
                excess = sum(vals) / len(vals) - float(early["floor"])
                merged["early"] = {"action": "full", "steps": index + 1, "of": len(batches), "excess": round(excess, 5)}
                if excess > float(early["rollback_at"]):
                    merged["early"]["action"] = "rollback"
                    break
                # KL grows about quadratically with the steps taken (second-order trust-region model, TRPO): stop now
                # when the next probe would land past the rollback line
                ahead = min(index + 1 + int(early.get("probe_every") or 1), len(batches))
                predicted = max(excess, 0.0) * (ahead / (index + 1)) ** 2
                merged["early"]["predicted"] = round(predicted, 5)
                if (excess > float(early["stop_at"]) or predicted > float(early["rollback_at"])) and not last:
                    merged["early"]["action"] = "stop"
                    break
        if measure is not None and steps and merged.get("ok"):  # the step's dose: k3 of the new state vs the kept
            # pi_old, forward only (GSPO's per-reply k3 on s_i, whatever the loss)
            if fell_back:  # the step's windows were the OOM retry's: the kept pi_old keys are at that size
                measure = [{**m, "window": fell_back} if m.get("whole") and int(m.get("window") or 0) > fell_back else m
                           for m in measure]
            result = await self.trainer.call("k3_score", timeout=self.cfg["update_timeout"], attempts=measure,
                                             signed=signed, gspo_clip=gspo["gspo_clip"],
                                             **{k: v for k, v in (cmd_extra or {}).items() if k not in OLD_PASS_SKIP})
            row = {"round": number, "windows": result.get("windows"), "approx_kl": result.get("approx_kl"),
                   **{k: result.get(k) for k in ("approx_kl_tok", "clip_frac", "replies", "missing", "seconds")},
                   **({"k3_target": float(mg["target_kl"]), "k3_stop": 1.5 * float(mg["target_kl"])}
                      if mg.get("target_kl") else {}),
                   "ok": bool(result.get("ok")) and result.get("approx_kl") is not None}
            self.feed.emit("k3_measure", **row)
            merged["k3_measure"] = row
            kl = result.get("approx_kl")
            if row.get("k3_stop") is not None and kl is not None:
                over = "stop" if kl > row["k3_stop"] else ("target" if kl > row["k3_target"] else None)
                if over:
                    self.feed.emit("alert", why=f"k3 over {over} (alert only, step taken)", round=number, step=1,
                                   of=1, approx_kl=kl, k3_target=row["k3_target"], k3_stop=row["k3_stop"],
                                   clip_frac=result.get("clip_frac"), lr=sent[0] if sent else None)
            if not row["ok"]:
                self.feed.emit("alert", why="k3 measure failed (step kept)", round=number,
                               error=str(result.get("stop") or result.get("error") or result.get("missing"))[:300])
        if early and merged.get("ok") and (merged.get("early") or {}).get("action") in ("full", "stop"):
            merged["adapter"] = await self.trainer.must("save", path=str(path))
        if merged.get("kl_stopped") and merged.get("ok"):  # saved by no step call: save now, probes as diagnostics
            merged["adapter"] = await self.trainer.must("save", path=str(path))
            if probes:
                scored = await self.trainer.must("score", windows=probes, label="kl_stop",
                                                 hold_previous=bool((cmd_extra or {}).get("hold_previous")))
                if isinstance(scored.get("windows"), dict):
                    merged["probes"] = scored["windows"]
        merged["steps"] = steps
        if sent:
            merged["lr_effective"] = sum(sent[:len(steps)]) / max(1, len(steps))
        if signed and steps:
            merged["balance"] = {k: steps[-1].get(k) for k in ("up_weight", "down_weight", "up_share", "down_clipped",
                                                                 "down_scale", "up_tokens", "down_tokens")}
            # the whole round's token-weighted push (sum over trained tokens of the signed weight, TIS included):
            # what a constant normaliser trains; the per-try share is the round row's train balance
            up, down = (sum(float(s.get(key) or 0) for s in steps) for key in ("up_weight", "down_weight"))
            merged["token_balance"] = {"up": round(up, 1), "down": round(down, 1),
                                       "up_share": round(up / (up + down), 4) if up + down else None}
        for key in ("train_seconds", "probe_seconds"):
            merged[key] = round(sum(float(s.get(key) or 0) for s in steps), 1)
        merged["trained_tokens"] = int(sum(int(s.get("trained_tokens") or 0) for s in steps))
        merged["loss"] = avg(s["loss"] for s in steps if s.get("loss") is not None)
        merged["grad_norm"] = max((s["grad_norm"] for s in steps if s.get("grad_norm") is not None), default=None)
        merged["update_norm"] = round(sum(float(s.get("update_norm") or 0) for s in steps), 6)
        return merged, path

    async def replay(self, probes):
        """Rebuild a lost adapter (rl1-loop-v7's r03 lived in the container): the recorded rounds' updates again, as
        that run made them (positive-only, steps_per_round steps, probes after each round's last step, compared with
        its feed), then the next recorded round's play as the data of the first new update."""
        g = self.cfg["train_game"]
        for number, path in enumerate(self.cfg["replay"], start=1):
            source = Path(path)
            rows = self.game_rows([r for r in json.loads((source / "level-bench.json").read_text())["attempts"]
                                   if usable(r)], g)
            rewards, adv, weights = self.advantages(rows)
            update, serve = await self.dosed_update(number, source, rows, weights, probes, probe_last=True)
            if not update.get("ok"):
                raise StopLoop("replay_failed", {"round": number, "stop": update.get("stop")})
            self.serving_adapter = serve
            self.feed.emit("replay", round=number, source=str(source), probes=update.get("probes"),
                           trained_tokens=update.get("trained_tokens"), loss=update.get("loss"))
        resume = Path(self.cfg["resume_from"])
        rows = self.game_rows([r for r in json.loads((resume / "level-bench.json").read_text())["attempts"]
                               if usable(r)], g)
        recorded = [Path(p) for p in self.cfg["replay"][1:]] + [resume]
        for number, source in enumerate(recorded, start=1):  # the recorded rounds' rewards: the trend's first points
            played = self.game_rows([r for r in json.loads((source / "level-bench.json").read_text())["attempts"]
                                     if usable(r)], g)
            values = [attempt_reward(r) for r in played]
            self.trend_points.append((number, sum(values) / len(values)))
            self.history.append({"round": number, "stats": {"reward_mean": sum(values) / len(values)}, "recorded": str(source)})
        rewards, adv, weights = self.advantages(rows)
        self.last_adv = adv
        self.checkpoint(len(self.cfg["replay"]), force=True)
        await self.save_checkpoint()
        self.feed.emit("resume", rounds=len(self.cfg["replay"]), trend=reward_trend(self.trend_points), source=str(resume))
        return resume, rows, weights, len(self.cfg["replay"]) + 1

    def checkpoint(self, number, force=False):
        """The trainer's fp32 master and optimizer state on the volume (last one only) every ``checkpoint_every``
        rounds: a later loop can resume from it (rl1-loop-v7's adapter was lost with its container)."""
        every, root = int(self.cfg["checkpoint_every"] or 0), os.environ.get("ARC_VALIDATION_OUTPUT")
        if not root or not (force or (every and number % every == 0)):
            return None
        self.pending_checkpoint = (number, Path(root) / "state")

    async def save_checkpoint(self, background=False):
        """``background`` (mg.async_checkpoint, MFU-1 E): the trainer writes the state from host copies while the
        server plays (rNN.partial, renamed when complete); the last complete state stays until then. Readers of the
        folder call ``checkpoint_landed`` first."""
        pending = getattr(self, "pending_checkpoint", None)
        if not pending:
            return
        number, folder = pending
        self.pending_checkpoint = None
        await self.checkpoint_landed()  # the previous background write (its deferred mg_state) first
        if getattr(self, "mg_own_copy", None) is not None:  # S2 resume_skip: the restored state's copy lands first
            await self.mg_own_copy
        early = (self.cfg.get("mg") or {}).get("early_play")
        background = background and not early  # the adapter beside the state needs the folder written
        result = await self.trainer.must("checkpoint", path=str(folder / f"r{number:02d}"),
                                         **({"background": True} if background else {}))
        background = background and bool(result.get("background"))  # refused (host memory): written synchronously
        if early:  # S2: the state's adapter beside it (a resume plays it early)
            result["adapter"] = await self.trainer.must("save", path=str(folder / f"r{number:02d}" / "adapter"))
        if (result.get("prior") or {}).get("path"):  # the previous background write, landed at this call
            self.ckpt_landed = Path(result["prior"]["path"]).name
        keep = (f"r{number:02d}", getattr(self, "keep_checkpoint", None), *(n for n, _ in getattr(self, "mg_served", [])),
                *((getattr(self, "ckpt_landed", None),) if background else ()))  # complete until this one lands
        for old in folder.iterdir() if folder.exists() else []:
            if old.name not in keep and not (background and old.name == f"r{number:02d}.partial"):
                shutil.rmtree(old, ignore_errors=True)
        self.ckpt_pending = f"r{number:02d}" if background else None
        if background:
            self.ckpt_watch = asyncio.ensure_future(self.ckpt_watch_folder(self.ckpt_pending, folder / self.ckpt_pending))
        else:
            self.ckpt_landed = f"r{number:02d}"
        self.feed.emit("checkpoint", round=number, **result)

    async def checkpoint_landed(self):
        """Waits for a background checkpoint (no unpark: the trainer may stay parked); feed ``checkpoint_done``."""
        if not getattr(self, "ckpt_pending", None):
            return
        name, self.ckpt_pending = self.ckpt_pending, None
        watch = getattr(self, "ckpt_watch", None)
        if watch is not None and not watch.done():
            watch.cancel()
        done = await self.trainer.must("checkpoint_wait", timeout=3600)
        self.ckpt_landed = name
        root = os.environ.get("ARC_VALIDATION_OUTPUT")  # a load already joined the write: no path, the folder is there
        if done.get("path") or root:
            self.ckpt_flush(name, Path(done["path"]) if done.get("path") else Path(root) / "state" / name)
        self.feed.emit("checkpoint_done", **{k: v for k, v in done.items() if k != "ok"})

    def ckpt_flush(self, name, folder):
        """The mg_state.json deferred while state ``name`` was written: the root file (atomic replace) and the state's
        own copy, once its folder exists (renamed from .partial = complete)."""
        deferred = getattr(self, "ckpt_deferred", None)
        if not (deferred and deferred[0] == name and folder.is_dir()):
            return False
        if not (folder / "mg_state.json").exists():
            (folder / "mg_state.json").write_text(deferred[1])
        root = Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "mg_state.json"
        tmp = root.with_name(root.name + ".tmp")
        tmp.write_text(deferred[1])
        os.replace(tmp, root)
        self.ckpt_deferred = None
        self.feed.emit("checkpoint_state", state=name)
        return True

    async def ckpt_watch_folder(self, name, folder, every=5.0):
        """mg.async_checkpoint: the root mg_state.json follows the state as soon as its folder lands (no trainer op;
        the trainer may be parked), not at the next checkpoint."""
        while getattr(self, "ckpt_pending", None) == name:
            if folder.is_dir() and getattr(self, "ckpt_deferred", None) and self.ckpt_deferred[0] == name:
                self.ckpt_flush(name, folder)
                return
            await asyncio.sleep(every)

    async def loop(self, folder, rows, weights, probes):
        g = self.cfg["train_game"]
        self.loop_started = time.time()
        self.gpu.mark("train")
        split = None
        try:
            split = await self.mismatch_split(folder, rows)
        except Exception as error:  # a diagnostic under the plan's rule; required under the split rule
            self.feed.emit("mismatch_split", error=repr(error)[:500])
        if self.cfg["limits"]["mismatch_rule"] == "split":
            verdict = split_verdict(split, self.cfg["limits"])
            self.feed.emit("split_verdict", **verdict)
            if not verdict["ok"]:
                raise StopLoop("mismatch_fresh", verdict)
        await self.server.sleep()
        await self.trainer.must("score", windows=probes, label="base")
        # the KL floor: the same probes scored again with the same weights (pass-to-pass trainer noise); with
        # limits["kl_floor_relative"] the KL stops sit this far above it
        repeat = (await self.trainer.must("score", windows=probes))["windows"]
        self.kl_floor = {name: max(0.0, float(row.get("kl_base") or 0.0)) for name, row in repeat.items()}
        self.feed.emit("kl_floor", floor=self.kl_floor, relative=self.cfg["limits"]["kl_floor_relative"],
                       logratio={name: row.get("logratio_base") for name, row in repeat.items()})
        self.trend_points = [(0, self.base.get("reward_mean"))]
        self.serving_adapter = None
        stop = None
        start = 1
        self.last_adv = self.advantages(rows)[1]
        if self.cfg["replay"]:
            folder, rows, weights, start = await self.replay(probes)
        elif self.cfg["resume_state"]:
            folder, rows, weights, start = await self.resume_state(probes)
        if self.cfg["opd"]:  # one level, on-policy distillation from the hinted current policy
            stop = await self.opd_rounds(start, probes)
            return await self.l2_finish(stop)
        if self.cfg["l2"]:  # one level, continuous token reward
            stop = await self.l2_rounds(start, probes)
            return await self.l2_finish(stop)
        if self.cfg["lvl"]:
            stop = await self.lvl_rounds(start, probes)
            return await self.lvl_finish(stop)
        if self.cfg["f_small"]:
            if self.cfg["restore_source"] and self.cfg["f_gate"]:  # check (B) gates update 1 (A passed on CPU, 46 of 46)
                check = await self.restore_check()
                if not (check["pass"] and (check.get("teacher_forced") or {}).get("pass")):
                    raise StopLoop("restore_check", {k: check.get(k) for k in ("points", "ok", "failures")}
                                   | {"teacher_forced": {k: v for k, v in (check.get("teacher_forced") or {}).items()
                                                         if k != "rows"}})
            stop = await self.f_rounds(start, folder, probes)
            return await self.finish(stop)
        if self.cfg["dual"]:
            stop = await self.overlap_rounds(start, folder, rows, weights, probes)
            return await self.finish(stop)
        for number in range(start, self.cfg["rounds"] + 1):
            since = time.time()
            generated = sum(int(r.get("completion_tokens") or 0) for r in rows)
            self.gpu.mark("train")
            began_update = time.time()
            if self.cfg["update_mode"] == "signed":
                update, serve = await self.signed_update(number, folder, rows, self.last_adv, probes)
            else:
                update, serve = await self.dosed_update(number, folder, rows, weights, probes)
            update["seconds"] = round(time.time() - began_update, 1)
            row = {"round": number, "update": {k: v for k, v in update.items() if k != "traceback"}}
            trained_share = round(update.get("trained_tokens", 0) / max(1, generated), 4)
            if not update.get("ok"):
                stop = update.get("stop") or "update_failed"
                row["stop"] = stop
                self.feed.emit("round", **row)
                break
            self.checkpoint(number)
            await self.save_checkpoint()  # beside the sleeping server (the trainer comes back to the card for it)
            self.gpu.mark("swap")
            row["swap"] = await self.swap_in(serve)
            self.serving_adapter = serve
            row["swap"]["memory"] = self.last_wake  # bytes moved to host, seconds, free card memory at wake
            old = self.adapters / f"r{number - 2:02d}"
            if number >= 2 and old.exists() and old != serve:
                shutil.rmtree(old, ignore_errors=True)
            guard = bool(self.cfg["lp85_every"]) and (number - start + 1) % int(self.cfg["lp85_every"]) == 0
            e = self.cfg["eval_game"]
            if guard:  # a seeded lp85 play guard inside the round's bench (same seeds as eval-r00)
                arms = [{"name": "train", "games": [g]},
                        {"name": "guard", "games": [e], "passes": int(self.cfg["lp85_attempts"]),
                         "seed_base": int(self.cfg["eval_seed_base"])}]
                folder, all_rows, play = await self.bench(f"round-{number:02d}", [g, e], self.cfg["train_attempts"],
                                                          {g: self.cfg["train_budget"], e: self.cfg["eval_budget"]},
                                                          arms=arms)
            else:
                folder, all_rows, play = await self.bench(f"round-{number:02d}", [g], self.cfg["train_attempts"],
                                                          {g: self.cfg["train_budget"]})
            rows = self.game_rows(all_rows, g)
            rewards, adv, weights = self.advantages(rows)
            self.last_adv = adv
            if guard:
                values = [attempt_reward(r) for r in self.game_rows(all_rows, e)]
                row["eval"] = {"label": f"guard-r{number:02d}", e: dict(zip(("mean", "se"), mean_se(values)), n=len(values))}
                self.feed.emit("eval", **row["eval"])
            row.update(play=play, rewards=rewards,
                       stats=round_stats(rows, rewards, adv, folder, int(self.study["completion_cap"]),
                                         self.base.get("reward_mean")))
            self.trend_points.append((number, row["stats"]["reward_mean"]))
            row["trend"] = reward_trend(self.trend_points)
            if number in self.cfg["eval_rounds"]:
                _, _, row["eval"] = await self.evaluate(f"r{number:02d}", self.cfg["eval_attempts_mid"])
                self.feed.emit("eval", **row["eval"])
            row["gpu"] = self.gpu.summary(since)
            row["swap"]["sleep_seconds"] = await self.server.sleep()
            # where a round's wall goes (would a separate play GPU pay?)
            park = (row["swap"].get("memory") or {}).get("park") or {}
            row["timing"] = {"update_s": update.get("seconds"), "unpark_s": (update.get("unparked") or {}).get("seconds"),
                             "park_s": park.get("seconds"), "wake_s": row["swap"].get("wake_seconds"),
                             "load_s": row["swap"].get("load_seconds"), "sleep_s": row["swap"]["sleep_seconds"],
                             "play_s": play.get("seconds"), "eval_s": (row.get("eval") or {}).get("play", {}).get("seconds"),
                             "round_s": round(time.time() - since, 1)}
            stop, row["alerts"] = self.check_stop(row)
            row["stop"] = stop
            self.history.append(row)
            self.feed.emit("round", **{k: v for k, v in row.items() if k != "update"},
                           update_summary={k: row["update"].get(k) for k in (
                               "loss", "grad_norm", "grad_norm_after_clip", "lr", "update_norm", "adapter_norm",
                               "trained_tokens", "train_seconds", "torch_peak_gib", "card_peak_gib", "probe_seconds",
                               "gpu_util_mean", "unparked", "window_wait_seconds", "step_seconds")},
                           probes=row["update"].get("probes"), trained_share=trained_share,
                           steps=len(row["update"].get("steps") or []),
                           sampled_nll_trainer=avg(w.get("sampled_nll_trainer") for w in row["update"].get("windows", [])))
            manual = os.environ.get("ARENA_STOP_REQUEST")
            if not stop and manual and Path(manual).exists():
                stop = "manual"
            if stop or self.left() < 3600:
                stop = stop or "wall"
                break
        return await self.finish(stop)

    async def overlap_rounds(self, start, folder, rows, weights, probes):
        """Two cards: round n's update (trainer card) runs while round n's attempts play on the
        adapter of the update before (server card); the next update trains on those attempts one step after the
        adapter that played them, TIS against the recorded engine logprobs. Same stop rules, guard and checkpoints."""
        g, e = self.cfg["train_game"], self.cfg["eval_game"]
        stop, played_by, data_by = None, start - 1, start - 1  # adapters that play this round / played its data
        for number in range(start, self.cfg["rounds"] + 1):
            since = time.time()
            generated = sum(int(r.get("completion_tokens") or 0) for r in rows)
            self.gpu.mark("overlap")
            guard = bool(self.cfg["lp85_every"]) and (number - start + 1) % int(self.cfg["lp85_every"]) == 0

            async def update(number=number, folder=folder, rows=rows, weights=weights, adv=self.last_adv):
                began = time.time()
                if self.cfg["update_mode"] == "signed":
                    result, serve = await self.signed_update(number, folder, rows, adv, probes)
                else:
                    result, serve = await self.dosed_update(number, folder, rows, weights, probes)
                result["seconds"] = round(time.time() - began, 1)
                if result.get("ok"):
                    self.checkpoint(number)
                    await self.save_checkpoint()
                return result, serve

            async def play(number=number, guard=guard):
                if guard:
                    arms = [{"name": "train", "games": [g]},
                            {"name": "guard", "games": [e], "passes": int(self.cfg["lp85_attempts"]),
                             "seed_base": int(self.cfg["eval_seed_base"])}]
                    return await self.bench(f"round-{number:02d}", [g, e], self.cfg["train_attempts"],
                                            {g: self.cfg["train_budget"], e: self.cfg["eval_budget"]}, arms=arms)
                return await self.bench(f"round-{number:02d}", [g], self.cfg["train_attempts"],
                                        {g: self.cfg["train_budget"]})

            tasks = [asyncio.create_task(update()), asyncio.create_task(play())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            failed = next((t for t in done if t.exception() is not None), None)
            if failed is not None:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise failed.exception()
            (update_row, serve), (folder, all_rows, play_row) = tasks[0].result(), tasks[1].result()
            row = {"round": number, "played_by": played_by, "lag": number - 1 - data_by,
                   "update": {k: v for k, v in update_row.items() if k != "traceback"}}
            trained_share = round(update_row.get("trained_tokens", 0) / max(1, generated), 4)
            if not update_row.get("ok"):
                stop = row["stop"] = update_row.get("stop") or "update_failed"
                self.feed.emit("round", **row)
                break
            row["swap"] = {"load_seconds": await self.server.load("policy", serve)}
            old = self.adapters / f"r{number - 2:02d}"
            if old.exists() and old not in (serve, self.serving_adapter):
                shutil.rmtree(old, ignore_errors=True)
            self.serving_adapter = serve
            rows = self.game_rows(all_rows, g)
            rewards, adv, weights = self.advantages(rows)
            self.last_adv = adv
            if guard:
                values = [attempt_reward(r) for r in self.game_rows(all_rows, e)]
                row["eval"] = {"label": f"guard-r{number:02d}", "played_by": played_by,
                               e: dict(zip(("mean", "se"), mean_se(values)), n=len(values))}
                self.feed.emit("eval", **row["eval"])
            row.update(play=play_row, rewards=rewards,
                       stats=round_stats(rows, rewards, adv, folder, int(self.study["completion_cap"]),
                                         self.base.get("reward_mean")))
            self.trend_points.append((played_by, row["stats"]["reward_mean"]))  # x = the adapter that played
            row["trend"] = reward_trend(self.trend_points)
            row["gpu"] = self.gpu.summary(since)
            row["timing"] = {"update_s": update_row.get("seconds"), "play_s": play_row.get("seconds"),
                             "load_s": row["swap"]["load_seconds"], "round_s": round(time.time() - since, 1)}
            data_by, played_by = played_by, number
            stop, row["alerts"] = self.check_stop(row)
            row["stop"] = stop
            self.history.append(row)
            self.feed.emit("round", **{k: v for k, v in row.items() if k != "update"},
                           update_summary={k: row["update"].get(k) for k in (
                               "loss", "grad_norm", "grad_norm_after_clip", "lr", "update_norm", "adapter_norm",
                               "trained_tokens", "train_seconds", "torch_peak_gib", "card_peak_gib", "probe_seconds",
                               "gpu_util_mean", "window_wait_seconds", "step_seconds")},
                           probes=row["update"].get("probes"), trained_share=trained_share,
                           steps=len(row["update"].get("steps") or []),
                           sampled_nll_trainer=avg(w.get("sampled_nll_trainer") for w in row["update"].get("windows", [])))
            manual = os.environ.get("ARENA_STOP_REQUEST")
            if not stop and manual and Path(manual).exists():
                stop = "manual"
            if stop or self.left() < 3600:
                stop = stop or "wall"
                break
        return stop

    def pick_states(self, folder, count, seed):
        """``count`` level starts of a round's full attempts, spread over the levels (round robin over the levels
        reached, a seeded attempt within each): ``[(attempt folder, request, tag, False)]``."""
        from .restore import level_starts

        rows = self.game_rows([r for r in json.loads((Path(folder) / "level-bench.json").read_text())["attempts"]
                               if usable(r)], self.cfg["train_game"])
        by_level = {}
        for row in rows:
            attempt = Path(folder) / "attempts" / row["label"]
            for request, done, step in level_starts(attempt):
                by_level.setdefault(done + 1, []).append((attempt, request))
        import random

        rng = random.Random(seed)
        for options in by_level.values():
            rng.shuffle(options)
        out, levels = [], sorted(by_level)
        if self.cfg["f_pick"] == "adaptive":
            return self.pick_adaptive(rows, folder, by_level, count, rng)
        if self.cfg["f_pick"] == "contested":
            failed = []  # the level each game was on when it ended without clearing it
            for row in rows:
                attempt = Path(folder) / "attempts" / row["label"]
                starts = level_starts(attempt)
                if starts and row.get("outcome") not in ("won",):
                    failed.append((attempt, starts[-1][0], starts[-1][1] + 1))
            rng.shuffle(failed)
            reached = {lv: len(v) for lv, v in by_level.items()}
            rate = {lv: (reached.get(lv + 1, 0) / reached[lv]) if reached[lv] else 0 for lv in reached}
            contested = [lv for lv in levels if 0.1 <= rate[lv] <= 0.9]
            taken = set()
            for attempt, request, level in failed[: count // 2]:  # at most half: the rest stay spread over levels
                if len(out) < count and (attempt, request) not in taken:
                    taken.add((attempt, request))
                    out.append((attempt, request, f"s{len(out):02d}-L{level}-{attempt.name[:4]}-k{request:03d}", False))
            for level in contested * count:
                options = [o for o in by_level[level] if o not in taken]
                if len(out) >= count or not options:
                    continue
                attempt, request = options[0]
                taken.add((attempt, request))
                out.append((attempt, request, f"s{len(out):02d}-L{level}-{attempt.name[:4]}-k{request:03d}", False))
            for level in levels:
                by_level[level] = [o for o in by_level[level] if o not in taken]
        while len(out) < count and any(by_level.values()):
            for level in levels:
                if by_level[level] and len(out) < count:
                    attempt, request = by_level[level].pop()
                    out.append((attempt, request, f"s{len(out):02d}-L{level}-{attempt.name[:4]}-k{request:03d}", False))
        return out

    def level_rate(self, level, full_rate):
        """A level's clear rate: the segment history (seed plus played segments) once it holds 8, else full games."""
        cleared, total = self.level_counts().get(level, (0, 0))
        return cleared / total if total >= 8 else full_rate

    def level_counts(self):
        seed = {int(k): tuple(v) for k, v in (self.cfg["f_level_seed"] or {}).items()}
        hist = getattr(self, "level_history", {})
        return {lv: (seed.get(lv, (0, 0))[0] + sum(hist.get(lv, [])), seed.get(lv, (0, 0))[1] + len(hist.get(lv, [])))
                for lv in set(seed) | set(hist)}

    def pick_adaptive(self, rows, folder, by_level, count, rng):
        """Half the states where full games stopped on a level not yet mastered (clear rate <= 0.9), the rest over
        the contested levels (0.1-0.9) in proportion to p(1-p), largest remainder; no contested level: the lowest."""
        from .restore import level_starts

        reached = {lv: len(v) for lv, v in by_level.items()}
        full = {lv: reached.get(lv + 1, 0) / reached[lv] for lv in reached if reached[lv]}
        rate = {lv: self.level_rate(lv, full[lv]) for lv in full}
        out, taken = [], set()

        def add(attempt, request, level):
            taken.add((attempt, request))
            out.append((attempt, request, f"s{len(out):02d}-L{level}-{attempt.name[:4]}-k{request:03d}", False))

        failed = []
        for row in rows:
            attempt = Path(folder) / "attempts" / row["label"]
            starts = level_starts(attempt)
            if starts and row.get("outcome") != "won" and rate.get(starts[-1][1] + 1, 0.5) <= 0.9:
                failed.append((attempt, starts[-1][0], starts[-1][1] + 1))
        rng.shuffle(failed)
        for attempt, request, level in failed[: count // 2]:
            add(attempt, request, level)
        weights = {lv: p * (1 - p) for lv, p in rate.items() if 0.1 <= p <= 0.9}
        if not weights:
            weights = {lv: 1.0 for lv in sorted(rate, key=lambda lv: rate[lv])[:2]}
        need, total = count - len(out), sum(weights.values())
        share = {lv: need * w / total for lv, w in weights.items()}
        alloc = {lv: int(s) for lv, s in share.items()}
        for lv in sorted(share, key=lambda lv: -(share[lv] - alloc[lv]))[: need - sum(alloc.values())]:
            alloc[lv] += 1
        for lv, n in sorted(alloc.items()):
            for attempt, request in [o for o in by_level[lv] if o not in taken][:n]:
                add(attempt, request, lv)
        self.feed.emit("pick", rates={lv: round(p, 3) for lv, p in sorted(rate.items())}, alloc=alloc,
                       failed=len(failed), states=len(out))
        return out

    def segment_reward(self, row):
        """A continuation's reward: the level score (capped at 1.15), or with f_reward "tokens" the clear (1/0) less
        its live tokens over the segment budget (success plus a linear token cost, no cap)."""
        if self.cfg["f_reward"] == "tokens":
            cleared = row.get("outcome") in ("cleared", "won")
            return round((1.0 if cleared else 0.0) - int(row.get("completion_tokens") or 0) / float(self.cfg["f_budget"]), 5)
        return attempt_reward(row)

    def segment_advantages(self, rows):
        """Per state (start point) the leave-one-out advantage of each continuation's reward; a state whose
        continuations all score the same gives none."""
        groups = {}
        for row in rows:
            groups.setdefault(row["point"], []).append(row)
        adv, used = [], []
        for point, group in sorted(groups.items()):
            rewards = [self.segment_reward(r) for r in group]
            if len(group) < 2 or max(rewards) == min(rewards):
                continue
            for row, a in zip(group, leave_one_out(rewards)):
                used.append(row)
                adv.append(a)
        return used, adv, {"states": len(groups), "states_with_signal": len({r["point"] for r in used})}

    def live_tail(self, spec):
        """A segment's gradient tail: its live part (after the scripted context) plus a margin, at most train_tail."""
        rows = [json.loads(line) for line in open(spec["rollout"])]
        scripted = [r for r in rows if r.get("scripted") and r.get("usage")]
        live = [r for r in rows if not r.get("scripted") and r.get("usage")]
        cap = int(self.cfg["train_tail"] or self.cfg["window"])
        if not scripted or not live:
            return cap
        end = live[-1]["usage"]["prompt_tokens"] + live[-1]["usage"]["completion_tokens"]
        start = scripted[-1]["usage"]["prompt_tokens"] + scripted[-1]["usage"]["completion_tokens"]
        return int(min(cap, max(2048, math.ceil((end - start + 1024) / 1024) * 1024)))

    async def f_update(self, number, seg, games, probes):
        """F v2 update: the informative states split over f_steps signed steps (states kept whole, negatives scaled
        per step so push-down x served tokens equals push-up), plus one signed step on the previous whole games
        (f_whole_per_side each side); probes after every step (f_probe_each) or after the last."""
        batches, balance = [], {"steps": []}

        async def signed(specs, a):
            targets = (await self.trainer.must("targets", windows=specs))["targets"]
            up = [s for s in specs if a[s["name"]] > 0]
            down = [s for s in specs if a[s["name"]] < 0]
            mass_up = sum(a[s["name"]] * targets.get(s["name"], 0) for s in up)
            mass_down = sum(-a[s["name"]] * targets.get(s["name"], 0) for s in down)
            scale = mass_up / mass_down if mass_down > 0 else 0.0
            balance["steps"].append({"up": len(up), "down": len(down), "down_scale": round(scale, 4),
                                     "up_tokens": sum(targets.get(s["name"], 0) for s in up),
                                     "down_tokens": sum(targets.get(s["name"], 0) for s in down)})
            return [{**s, "w": a[s["name"]]} for s in up] + [{**s, "w": a[s["name"]] * scale} for s in down if scale > 0]

        if games is not None:
            folder, rows, adv = games
            k, tail = int(self.cfg["f_whole_per_side"]), self.cfg["train_tail"]
            specs = [{**s, **({"window": int(tail)} if tail else {})} for s in self.windows(folder, rows, prefix="w-")]
            a = {s["name"]: float(v) for s, v in zip(specs, adv)}
            pick = (sorted([s for s in specs if a[s["name"]] > 0], key=lambda s: -a[s["name"]])[:k]
                    + sorted([s for s in specs if a[s["name"]] < 0], key=lambda s: a[s["name"]])[:k])
            if pick:
                batches.append(await signed(pick, a))
        if seg is not None:
            folder, rows, adv = seg
            specs = self.windows(folder, rows, prefix="s-")
            for s in specs:
                s["window"] = self.live_tail(s) if self.cfg["f_live_tail"] else int(self.cfg["train_tail"] or self.cfg["window"])
            a = {s["name"]: float(v) for s, v in zip(specs, adv)}
            states = sorted({r["point"] for r in rows})
            n = max(1, min(int(self.cfg["f_steps"]), len(states)))
            point = {s["name"]: r["point"] for s, r in zip(specs, rows)}
            for i in range(n):
                part = [s for s in specs if states.index(point[s["name"]]) % n == i]
                if part:
                    batches.append(await signed(part, a))
        merged, path = await self.dosed_update(number, None, None, None, probes, batches=batches, signed=True,
                                               probe_last=not self.cfg["f_probe_each"], extra=None)
        merged["balance"] = balance
        return merged, path

    async def f_rounds(self, start, source, probes):
        """F-small on two cards: round n plays its segments (and every f_refresh_every-th round the full refresh
        games with the lp85 guard) on the adapter of the update before, while the trainer takes the update on round
        n-1's segments; the first round's segments gate update 1 (every restore exact, every first live prompt's ids
        equal to the recording's)."""
        g, e = self.cfg["train_game"], self.cfg["eval_game"]
        states_from, segments, stop = Path(self.cfg["f_states_from"] or source), None, None
        whole = None  # the previous round's whole games, trained in the next update (f_whole_every)
        played_by, data_by = start - 1, None
        budget, k = int(self.cfg["f_budget"]), int(self.cfg["f_continuations"])
        for number in range(start, self.cfg["rounds"] + 1):
            since = time.time()
            self.gpu.mark("overlap")
            refresh = bool(self.cfg["f_refresh_every"]) and (number - start + 1) % int(self.cfg["f_refresh_every"]) == 0

            async def update(number=number, data=segments, games=whole):
                if data is None and games is None:
                    return None, None
                began = time.time()
                if self.cfg["f_steps"] > 1 or self.cfg["f_whole_every"]:
                    result, serve = await self.f_update(number, data, games, probes)
                else:
                    result, serve = await self.signed_update(number, data[0], data[1], data[2], probes)
                result["seconds"] = round(time.time() - began, 1)
                if result.get("ok"):
                    self.checkpoint(number)
                    await self.save_checkpoint()
                return result, serve

            async def play(number=number, refresh=refresh):
                states = self.pick_states(states_from, int(self.cfg["f_states"]), seed=number)
                path, points = self.restore_points(states, f"segments-{number:02d}", budget=budget)
                seg_play = self.bench(f"segments-{number:02d}", [g], k, {g: budget}, points=path,
                                      concurrency=min(len(points) * k, int(self.cfg["concurrency"])))
                full = None
                if refresh:
                    arms = [{"name": "train", "games": [g]},
                            {"name": "guard", "games": [e], "passes": int(self.cfg["lp85_attempts"]),
                             "seed_base": int(self.cfg["eval_seed_base"])}]
                    full_play = self.bench(f"round-{number:02d}", [g, e], int(self.cfg["f_refresh_attempts"]),
                                           {g: self.cfg["train_budget"], e: self.cfg["eval_budget"]}, arms=arms,
                                           concurrency=int(self.cfg["f_refresh_attempts"]) + int(self.cfg["lp85_attempts"]))
                    if self.cfg["f_whole_every"]:  # beside the segments (the server has room for both)
                        seg, full = await asyncio.gather(seg_play, full_play)
                        return seg, full
                    seg = await seg_play
                    full = await full_play
                    return seg, full
                return await seg_play, full

            tasks = [asyncio.create_task(update()), asyncio.create_task(play())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            failed = next((t for t in done if t.exception() is not None), None)
            if failed is not None:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise failed.exception()
            (update_row, serve), ((seg_folder, seg_rows, seg_play), full) = tasks[0].result(), tasks[1].result()
            row = {"round": number, "played_by": played_by, "lag": None if data_by is None else number - 1 - data_by}
            if update_row is not None:
                row["update"] = {k_: v for k_, v in update_row.items() if k_ != "traceback"}
                if not update_row.get("ok"):
                    stop = row["stop"] = update_row.get("stop") or "update_failed"
                    self.feed.emit("round", **row)
                    break
                row["swap"] = {"load_seconds": await self.server.load("policy", serve)}
                old = self.adapters / f"r{number - 2:02d}"
                if old.exists() and old not in (serve, self.serving_adapter):
                    shutil.rmtree(old, ignore_errors=True)
                self.serving_adapter = serve
            # the segments: restore checks, rewards, advantages
            seg_all = json.loads((seg_folder / "level-bench.json").read_text())["attempts"]
            checks = [r.get("restore") or {} for r in seg_all]
            exact = sum(bool(c.get("ok")) and bool((c.get("prompt_digest") or {}).get("same")) for c in checks)
            row["restore"] = {"attempts": len(seg_all), "exact": exact,
                              "failures": [{"label": r["label"], "outcome": r["outcome"],
                                            "diffs": (r.get("restore") or {}).get("diffs"),
                                            "state": (r.get("restore") or {}).get("state")}
                                           for r in seg_all if not ((r.get("restore") or {}).get("ok"))][:5]}
            if segments is None and self.cfg["f_gate"] and exact < len(seg_all):  # gate (B) before update 1: every restore exact
                stop = row["stop"] = "restore_gate"
                self.feed.emit("round", **row)
                raise StopLoop("restore_gate", row["restore"])
            seg_ok = [r for r in self.game_rows(seg_rows, g) if (r.get("restore") or {}).get("ok")]
            used, adv, info = self.segment_advantages(seg_ok)
            rewards = [attempt_reward(r) for r in seg_ok]
            for r in seg_ok:  # the per-level clear history behind the adaptive state choice
                level = next((int(part[1:]) for part in str(r.get("point", "")).split("-")
                              if part[:1] == "L" and part[1:].isdigit()), None)
                if level is not None:
                    self.level_history = getattr(self, "level_history", {})
                    self.level_history.setdefault(level, []).append(int(r["outcome"] in ("cleared", "won")))
            row["segments"] = {"play": seg_play, **info, "reward_mean": avg(rewards),
                               "f_reward_mean": avg(self.segment_reward(r) for r in seg_ok),
                               "clear_share": round(sum(r["outcome"] in ("cleared", "won") for r in seg_ok) / max(1, len(seg_ok)), 4),
                               "tokens_mean": avg(int(r.get("completion_tokens") or 0) for r in seg_ok)}
            self.segment_points = getattr(self, "segment_points", []) + [(played_by, row["segments"]["reward_mean"])]
            row["segment_trend"] = reward_trend(self.segment_points)
            segments = (seg_folder, used, adv) if used else None
            whole = None
            data_by = played_by
            if full is not None:
                folder, all_rows, play_row = full
                rows = self.game_rows(all_rows, g)
                full_rewards, full_adv, _ = self.advantages(rows)
                values = [attempt_reward(r) for r in self.game_rows(all_rows, e)]
                row["eval"] = {"label": f"guard-r{number:02d}", "played_by": played_by,
                               e: dict(zip(("mean", "se"), mean_se(values)), n=len(values))}
                self.feed.emit("eval", **row["eval"])
                row.update(play=play_row, rewards=full_rewards,
                           stats=round_stats(rows, full_rewards, full_adv, folder, int(self.study["completion_cap"]),
                                             self.base.get("reward_mean")))
                self.trend_points.append((played_by, row["stats"]["reward_mean"]))
                row["trend"] = reward_trend(self.trend_points)
                states_from = folder  # the next states come from the newest full games
                if self.cfg["f_whole_every"]:
                    whole = (folder, rows, full_adv)
            row["gpu"] = self.gpu.summary(since)
            row["timing"] = {"update_s": (update_row or {}).get("seconds"), "segments_s": seg_play.get("seconds"),
                             "refresh_s": (full[2].get("seconds") if full else None), "round_s": round(time.time() - since, 1)}
            if update_row is not None:
                played_by = number
            stop, row["alerts"] = self.check_stop(row)
            row["stop"] = stop
            self.history.append(row)
            update_part = row.get("update") or {}
            self.feed.emit("round", **{k_: v for k_, v in row.items() if k_ != "update"},
                           update_summary={k_: update_part.get(k_) for k_ in (
                               "loss", "grad_norm", "update_norm", "adapter_norm", "trained_tokens", "train_seconds",
                               "torch_peak_gib", "card_peak_gib", "probe_seconds", "gpu_util_mean")},
                           probes=update_part.get("probes"), balance=update_part.get("balance"))
            manual = os.environ.get("ARENA_STOP_REQUEST")
            if not stop and manual and Path(manual).exists():
                stop = "manual"
            if stop or self.left() < 3600:
                stop = stop or "wall"
                break
        return stop

    async def finish(self, stop):
        """The final play test on the last adapter."""
        if stop not in ("nan_or_inf", "mismatch", "update_failed"):
            await self.wake()
            if self.cfg["paired"]:  # the adapter and the base on the same seeds
                final = await self.paired_eval(self.cfg["paired"])
                self.feed.emit("final", stop=stop, eval=final, base=self.base)
                return stop
            _, _, final = await self.evaluate("final", self.cfg["eval_attempts"])
            self.feed.emit("eval", **final)
            self.feed.emit("final", stop=stop, eval=final, base=self.base)
        return stop

    def recorded_rounds(self, folders):
        """Rounds another run played (in order, rounds 1..n): the trend's first points and the stop rules' history;
        returns the last round's usable sb26 rows."""
        g, played = self.cfg["train_game"], []
        for number, source in enumerate(Path(p) for p in folders):
            played = self.game_rows([r for r in json.loads((source / "level-bench.json").read_text())["attempts"]
                                     if usable(r)], g)
            values = [attempt_reward(r) for r in played]
            self.trend_points.append((number + 1, sum(values) / len(values)))
            self.history.append({"round": number + 1, "stats": {"reward_mean": sum(values) / len(values)},
                                 "recorded": str(source)})
        return played

    async def resume_state(self, probes):
        """continue another run from its state checkpoint (adapter + optimizer, after its last
        update) with the round that adapter played as the first update's data; the probes are scored on the loaded
        state (KL to this run's base: compare with the source's last update_step)."""
        loaded = await self.trainer.must("load_checkpoint", path=str(self.cfg["resume_state"]))
        recorded = list(self.cfg["recorded"] or [])
        rows = self.recorded_rounds(recorded)
        number = int(self.cfg["resume_round"] or len(recorded))
        seed = self.cfg["trend_seed"] or {}
        self.trend_points += [tuple(x) for x in seed.get("full", [])]
        self.segment_points = [tuple(x) for x in seed.get("segments", [])]
        serve = self.adapters / f"r{number:02d}"
        await self.trainer.must("save", path=str(serve))
        scored = (await self.trainer.must("score", windows=probes))["windows"]
        self.feed.emit("resume_state", loaded=loaded, rounds=number, source=recorded[-1],
                       kl_base={name: row.get("kl_base") for name, row in scored.items()},
                       trend=reward_trend(self.trend_points))
        swap = await self.swap_in(serve)
        self.serving_adapter = serve
        _, adv, weights = self.advantages(rows)
        self.last_adv = adv
        self.feed.emit("resume", rounds=number, swap=swap, source=recorded[-1])
        return Path(recorded[-1]), rows, weights, number + 1

    def train_point(self):
        """The study's play-through start point of the train game (the sb26 game from level 1)."""
        from .level_bench import load_points

        cfg = {**self.study["level_bench"], "games": [self.cfg["train_game"]]}
        return load_points(self.source / cfg["start_points"], cfg)[0]

    def restore_points(self, specs, name, budget=None, hint=None, pin=False):
        """A start-point file of restored attempts: ``specs`` [(attempt folder, request, tag, check_only)]."""
        from .restore import load

        base = self.train_point()
        points = []
        for spec in specs:
            folder, request, tag, check = spec[:4]
            own = spec[4] if len(spec) > 4 else budget  # a per-start budget (level ladder)
            level = load(folder, request)["level"]
            points.append({**base, "id": f"{base['id']}-{tag}", "level": level,
                           "restore": {"attempt": str(folder), "request": int(request), "check_only": bool(check),
                                       **({"budget": int(own)} if own else {}), **({"hint": hint} if hint else {}),
                                       **({"pin_actions": True} if pin else {})},
                           **({"priority": int(own)} if len(spec) > 4 and own else {})})  # longest budgets start first
        path = self.work / f"{name}-points.json"
        path.write_text(json.dumps({"schema": 1, "points": points}))
        return path, points

    async def restore_check(self):
        """Check (A), every level start of recorded attempts restored by re-execution; each
        scripted request must equal the recorded one (messages, template, tools), the game, the action list and the
        counters must equal the recording at the first live request, and that request's served prompt ids must hash
        to the recorded digest."""
        from .restore import level_starts

        source = Path(self.cfg["restore_source"])
        g = self.cfg["train_game"]
        played = sorted(self.game_rows([r for r in json.loads((source / "level-bench.json").read_text())["attempts"]
                                        if usable(r)], g), key=lambda r: r["label"])[: int(self.cfg["restore_attempts"])]
        specs = []
        for row in played:
            folder = source / "attempts" / row["label"]
            for request, done, step in level_starts(folder):
                if request:
                    specs.append((folder, request, f"{row['label'][:4]}-k{request:03d}", True))
        path, points = self.restore_points(specs, "restore-check")
        self.feed.emit("restore_points", points=len(points), attempts=len(played))
        folder, _, play = await self.bench("restore-check", [g], 1, {g: self.cfg["train_budget"]},
                                           model=self.server.model, points=path,
                                           concurrency=min(len(points), int(self.cfg["concurrency"])))
        rows = json.loads((folder / "level-bench.json").read_text())["attempts"]
        results = [{"label": r["label"], "outcome": r["outcome"], **(r.get("restore") or {})} for r in rows]
        ok = [r for r in results if r.get("ok") and r["outcome"] == "restored" and (r.get("prompt_digest") or {}).get("same")]
        summary = {"points": len(points), "attempts": len(rows), "ok": len(ok), "pass": len(ok) == len(points) == len(rows),
                   "failures": [{k: r.get(k) for k in ("label", "outcome", "diffs", "state", "prompt_digest")}
                                for r in results if r not in ok][:20], "play": play}
        summary["teacher_forced"] = await self.teacher_forced([r for r in rows if r.get("restore")])
        self.feed.emit("restore_check", **summary)
        return summary

    async def teacher_forced(self, rows, model=None):
        """Check (B) on the restore points: the recorded next request (the restored prompt plus the original reply)
        scored cold twice; on the reply's tokens, the original engine log-probabilities against the cold pass, beside
        the cold pass against itself (the engine noise floor; bar max(0.03, 1.25 x that))."""
        def rebuild(rows):  # posttrain.rl_rollouts.restore (not shipped with the arena source)
            out, messages, tools = [], [], None
            for r in rows:
                messages = messages[: r["keep"]] + r["messages"]
                tools = r["tools"] if "tools" in r else tools
                out.append({**r, "messages": list(messages), "tools": tools})
            return out

        out, engine_fresh, floor = [], [], []
        for row in rows:
            info = row["restore"]
            k, length = info["request"], (info.get("prompt_digest") or {}).get("prompt_tokens")
            recorded = [json.loads(line) for line in open(Path(info["source"]) / "rollout.jsonl")]
            if not length or k + 1 >= len(recorded) or not recorded[k + 1]["gameplay"]:
                continue
            nxt = rebuild(recorded[: k + 2])[-1]
            body = {"messages": nxt["messages"], "chat_template_kwargs": nxt["template"],
                    "model": model or self.server.model, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0}
            if nxt.get("tools") is not None:
                body["tools"] = nxt["tools"]
            passes = []
            for _ in range(2):
                reply = await self.server.post("/v1/chat/completions", body, timeout=1200)
                entries = reply.get("prompt_logprobs") or ((reply.get("choices") or [{}])[0].get("prompt_logprobs")) or []
                passes.append(entries)
            tokens, engine = recorded[k]["tokens"] or [], recorded[k]["logprobs"] or []
            span = [passes[0][length + i] if length + i < len(passes[0]) else None for i in range(len(tokens))]
            ids_ok = all(e is not None and str(t) in e for e, t in zip(span, tokens))
            def values(entries):
                vals = []
                for i, t in enumerate(tokens):
                    e = entries[length + i] if length + i < len(entries) else None
                    hit = (e or {}).get(str(t))
                    vals.append(float(hit["logprob"] if isinstance(hit, dict) else hit) if hit is not None else float("nan"))
                return vals
            first, second = values(passes[0]), values(passes[1])
            ef, rep = compare_logprobs(engine, first), compare_logprobs(first, second)
            out.append({"label": row["label"], "request": k, "reply_tokens": len(tokens), "ids_match": ids_ok,
                        "engine_fresh": ef, "fresh_repeat": rep})
            if ef.get("tokens"):
                engine_fresh.append((ef["mean_abs"], ef["tokens"]))
            if rep.get("tokens"):
                floor.append((rep["mean_abs"], rep["tokens"]))
        mean = lambda pairs: round(sum(v * n for v, n in pairs) / sum(n for _, n in pairs), 5) if pairs else None
        gap, noise = mean(engine_fresh), mean(floor)
        bar = None if noise is None else round(max(0.03, float(self.cfg["equality_factor"]) * noise), 5)
        return {"points": len(out), "ids_match": sum(o["ids_match"] for o in out), "engine_fresh_mean_abs": gap,
                "fresh_repeat_mean_abs": noise, "bar": bar,
                "pass": bool(out) and all(o["ids_match"] for o in out) and gap is not None and bar is not None and gap <= bar,
                "rows": out}

    def equality_rows(self):
        """Points for the equality check: recorded gameplay replies whose next request is gameplay too (the reply
        sits in its prompt), spread over each attempt, attempts in label order; a restored try's scripted replies
        (the source game's) are skipped."""
        source, per, out = Path(self.cfg["equality_source"]), 3, []
        for folder in sorted(p for p in (source / "attempts").iterdir() if (p / "rollout.jsonl").exists()):
            rows = [json.loads(line) for line in open(folder / "rollout.jsonl")]
            ks = [k for k in range(len(rows) - 1) if rows[k].get("gameplay") and rows[k + 1].get("gameplay")
                  and not rows[k].get("scripted")
                  and rows[k].get("tokens") and (rows[k].get("usage") or {}).get("prompt_tokens")]
            for k in sorted({ks[round(i * (len(ks) - 1) / (per - 1))] for i in range(per)} if len(ks) >= per else set(ks)):
                out.append({"label": f"{folder.name}-k{k:03d}", "restore": {
                    "request": k, "source": str(folder), "prompt_digest": {"prompt_tokens": rows[k]["usage"]["prompt_tokens"]}}})
        return out[: int(self.cfg["equality_points"])]

    async def final_only(self):
        """Mode final: convert the state to an adapter beside the sleeping server (one card), check the served
        logprobs against the default path's recording, then the final evaluation."""
        if self.cfg["resume_adapter"]:  # converted on CPU: no trainer, the server stays awake
            serve = Path(self.cfg["resume_adapter"])
            swap = {"load_seconds": await self.server.load("policy", serve)}
            self.feed.emit("resume_adapter", source=self.cfg["resume_adapter"], path=str(serve), swap=swap)
        else:
            await self.server.sleep()
            trainer = await self.start_trainer()
            loaded = await self.trainer.must("load_checkpoint", path=str(self.cfg["resume_state"]))
            serve = self.adapters / f"r{int(self.cfg['resume_round'] or 0):02d}"
            saved = await self.trainer.must("save", path=str(serve))
            await self.trainer.call("exit", timeout=120)
            self.trainer.close()
            self.trainer = None
            swap = await self.swap_in(serve)
            self.feed.emit("resume_state", trainer=trainer, loaded=loaded, saved=saved, swap=swap)
        # no equality source: the loop's own saved adapter folder, served as the loop served it
        check = {"skipped": True}
        if self.cfg["equality_source"]:
            rows = self.equality_rows()
            check = await self.teacher_forced(rows, model="policy")
            self.feed.emit("equality", source=self.cfg["equality_source"], **check)
            if not check["pass"]:
                raise StopLoop("equality", {k: v for k, v in check.items() if k != "rows"})
        if (self.cfg["mg"] or {}).get("final"):
            final = await self.mg_final()
            self.feed.emit("final", stop=None, eval=final, base=None)
            return {"equality": {k: v for k, v in check.items() if k != "rows"}, "eval": final}
        if self.cfg["transfer"]:
            final = await self.transfer_eval(self.cfg["transfer"])
            self.feed.emit("final", stop=None, eval=final, base=None)
            return {"equality": {k: v for k, v in check.items() if k != "rows"}, "eval": final}
        if self.cfg["paired"]:
            final = await self.paired_eval(self.cfg["paired"])
            self.feed.emit("final", stop=None, eval=final, base=None)
            return {"equality": {k: v for k, v in check.items() if k != "rows"}, "eval": final}
        _, _, final = await self.evaluate("final", self.cfg["eval_attempts"])
        self.feed.emit("eval", **final)
        self.feed.emit("final", stop=None, eval=final, base=None)
        return {"equality": {k: v for k, v in check.items() if k != "rows"}, "eval": final}

    async def paired_eval(self, spec):
        """The adapter and the base on the same seeds (seed_base + pass), both benches at once; per game the two
        means and the paired difference (policy - base) over seeds both arms finished."""
        games, n, seed = list(spec["games"]), int(spec["attempts"]), int(spec["seed_base"])
        budgets = {g: self.cfg["train_budget"] if g == self.cfg["train_game"] else self.cfg["eval_budget"] for g in games}
        names = {"policy": "policy", "base": self.server.model}
        results = await asyncio.gather(*(self.bench(f"eval-paired-{tag}", games, n, budgets, seed_base=seed,
                                                    model=model, concurrency=n * len(games), reserve=120)
                                         for tag, model in names.items()))
        out = {"seed_base": seed, "attempts": n}
        for (tag, _), (_, rows, play) in zip(names.items(), results):
            out[f"{tag}_play"] = play
            out[tag] = {g: dict(zip(("mean", "se"), mean_se([attempt_reward(r) for r in self.game_rows(rows, g)])),
                            n=len(self.game_rows(rows, g))) for g in games}
        for g in games:
            by = [{(r.get("pass"), r["label"].split("-p")[-1]): attempt_reward(r) for r in self.game_rows(rows, g)}
                  for _, rows, _ in results]
            keys = sorted(set(by[0]) & set(by[1]), key=str)
            diffs = [by[0][k] - by[1][k] for k in keys]
            mean, se = mean_se(diffs)
            out.setdefault("paired", {})[g] = {"mean_diff": mean, "se": se, "n": len(diffs),
                                               "policy_wins": sum(d > 0 for d in diffs), "ties": sum(d == 0 for d in diffs)}
        self.feed.emit("eval", label="paired", **out)
        return out

    async def expert_mask_start(self):
        """server.expert_mask (hero run): every server's mask is checked live after its start and each wake."""
        spec = (self.study.get("server") or {}).get("expert_mask")
        if not spec or os.environ.get("ARC_PREFLIGHT_STUB"):  # the replay stub serves no MoE
            return None
        mask = json.loads((self.source / spec["mask"]).read_text())
        Server.mask_expect = sum(len(v) for v in mask["pruned"].values())
        Server.mask_sha = spec["sha256"]
        got = await self.server.mask_check("start")
        self.feed.emit("expert_mask", label=got.get("label"), masked=got.get("masked_entries"), sha256=got.get("sha256"),
                       layers=len(got.get("layers") or []),
                       **{k: got[k] for k in ("skipped_reads", "skipped_expected", "load_profile") if k in got})
        return got

    def expert_mask_trainer(self, loaded):
        """server.expert_mask (hero run): the trainer applied the bundle's mask (RL1_EXPERT_MASK, by sha256), or
        raise; hprobe-v4: a bundle without the trainer's mask code ignored RL1_EXPERT_MASK silently."""
        spec = ((getattr(self, "study", None) or {}).get("server") or {}).get("expert_mask")
        if not spec or os.environ.get("ARC_PREFLIGHT_FAKE_TRAINER"):  # the CPU preflight's stand-in loads no model
            return None
        got = (loaded or {}).get("expert_mask") or {}
        self.feed.emit("expert_mask_trainer", sha256=got.get("sha256"), layers=got.get("layers"), pruned=got.get("pruned"))
        if got.get("sha256") != spec["sha256"]:
            raise RuntimeError(f"trainer expert mask {got.get('sha256')} != the bundle's {spec['sha256']}")
        return got

    async def run(self):
        report = {"status": "error", "settings": self.cfg}
        if os.environ.get("ARC_RL1_STALL_DUMP"):
            StallWatch(self.feed, asyncio.get_running_loop(), os.environ["ARC_RL1_STALL_DUMP"],
                       os.environ.get("ARC_RL1_STALL_KILL"), os.environ.get("ARC_VALIDATION_OUTPUT") or self.output)
        try:
            self.feed.emit("start", mode=self.cfg["mode"], settings=self.cfg)
            await self.expert_mask_start()
            if self.cfg["mode"] == "restore":
                report["restore"] = await self.restore_check()
                report["status"] = "complete"
                return report
            if self.cfg["mode"] == "opd_gate":
                report["gate"] = await self.opd_gate()
                report["status"] = "complete"
                return report
            if self.cfg["mode"] == "final":
                report["final"] = await self.final_only()
                report["status"] = "complete"
                return report
            if self.cfg["mode"] == "sft":
                report["sft"] = await self.sft_run()
                report["status"] = "complete"
                return report
            if self.cfg["mode"] == "screen":
                report["screen"] = await self.mg_screen()
                report["status"] = "complete"
                return report
            if self.cfg["mode"] == "loop" and (self.cfg["mg"] or {}).get("screen"):
                report["stop"] = await self.mg_loop()
                report["status"] = "complete"
                return report
            await self.server.sleep()  # the trainer loads beside a sleeping server (unknown load transient)
            report["trainer"] = await self.start_trainer()
            self.feed.emit("trainer", **report["trainer"])
            await self.fit_and_parity()
            folder, rows, weights, probes = await self.round_zero()
            if self.cfg["mode"] == "probe":
                report["pick"] = await self.probe_rates(folder, rows, weights, probes)
            else:
                report["stop"] = await self.loop(folder, rows, weights, probes)
            report["status"] = "complete"
        except StopLoop as stop:
            report.update(status="complete_with_errors", stop=stop.reason, detail=stop.detail)
            self.feed.emit("stop", reason=stop.reason, detail=stop.detail)
        except Exception as error:  # the failure is recorded in the feed and the trial
            import traceback

            report.update(status="error", error=repr(error)[:2000])
            self.feed.emit("error", error=repr(error)[:2000], traceback=traceback.format_exc()[-4000:])
        finally:
            report["gpu"] = self.gpu.summary(self.started)
            self.gpu.stop.set()
            if getattr(self, "mg_explorers", None) is not None:  # envs 22 Oct: explorer plays end with the run
                try:
                    await asyncio.wait_for(self.mg_explorers.stop(), 300)
                except Exception as error:  # noqa: BLE001 - teardown goes on
                    self.feed.emit("alert", why="explorer stop failed", error=repr(error)[:300])
            if getattr(self, "server2", None) is not None:
                self.server2.stop()
            if self.trainer is not None:
                try:  # a background checkpoint lands before the trainer exits (its own join has 120 s)
                    await asyncio.wait_for(self.checkpoint_landed(), 900)
                except Exception as error:  # noqa: BLE001 - teardown goes on
                    self.feed.emit("alert", why="background checkpoint did not land", error=repr(error)[:300])
                try:
                    await asyncio.wait_for(self.trainer.call("exit", timeout=120), 150)
                except Exception:
                    pass
                self.trainer.close()
            try:
                await self.server.wake()
            except Exception:
                pass
            (self.output / "rl" / "report.json").write_text(json.dumps(report, indent=1, default=str))
        return report


def avg(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 6) if values else None


class StopLoop(Exception):
    def __init__(self, reason, detail=None):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def trainer_command(cfg, source, work):
    script = Path(source) / cfg["trainer_script"]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHONPATH", "VIRTUAL_ENV"))}
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", AXOLOTL_SONICMOE_NVFP4_BACKEND="dequant",
               PYTHONPATH=str(script.parent),
               HF_HUB_OFFLINE="0", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")  # as the pilot; v3: 16 GiB fragmented at 32K
    env.update({str(k): str(v) for k, v in (cfg.get("trainer_env") or {}).items()})
    if cfg.get("attention"):  # RL1_ATTENTION: masked (trainer default), selected or triton (rl1_sparse)
        env["RL1_ATTENTION"] = str(cfg["attention"])
    env.update({str(k): str(v) for k, v in (cfg.get("trainer_env") or {}).items()})  # e.g. RL1_SWITCH_INTERVAL, RL1_SPLIT_GEMM
    command = [cfg["trainer_python"], "-u", str(script), "--dir", str(Path(work) / "trainer"),
               "--model", cfg["model_dir"], "--rank", str(cfg["rank"]), "--lr", str(cfg["lr"]),
               "--window", str(cfg["window"]), "--optimizer", cfg["optimizer"]]
    command += (["--loop-lora"] if cfg["loop_lora"] else []) + (["--tis-cap", str(cfg["tis_cap"])] if cfg["tis_cap"] else [])
    command += ["--prefix-chunk", str(int(cfg["path_chunk"]))]
    if os.environ.get("ARC_PREFLIGHT_FAKE_TRAINER"):  # CPU launch-gate preflight: the loop's paths without a model
        env["PYTHONPATH"] = str(Path(source) / "src")
        # one interpreter for the launcher's pre-spawn and the loop (arena.trainer_prespawn): the image's python
        command = [os.environ.get("ARC_FAKE_TRAINER_PYTHON") or sys.executable, "-u", "-m", "arena.fake_trainer",
                   "--dir", str(Path(work) / "trainer"),
                   "--prefix-chunk", str(int(cfg["path_chunk"]))]
    return command, env


HOLDER = """
import json, sys, time, torch
control, ack, held, target = sys.argv[1], sys.argv[2], None, None
while True:
    try:
        want = float(open(control).read())
    except (OSError, ValueError):
        want = target
    if want != target:
        held, target = None, want
        torch.cuda.empty_cache()
        try:
            held = torch.empty(int(want * 2**30), dtype=torch.uint8, device="cuda")
            state = {"gib": want, "ok": True}
        except torch.OutOfMemoryError as error:
            state = {"gib": want, "ok": False, "error": str(error)[:300]}
        open(ack, "w").write(json.dumps(state))
    time.sleep(0.5)
"""


def engine_gap_summary(rows, lag_of):
    """Pooled |engine - trainer pi_old| over sampled tokens (mean, share beyond 0.5 nats, delta-method SE over windows),
    for all windows and per policy lag (window names start with their try's name; no lag map: all lag 0)."""
    import math

    def lag(name):
        hits = [k for k in lag_of if str(name).startswith(k)]
        return lag_of[max(hits, key=len)] if hits else (0 if not lag_of else "unknown")

    def pool(part):
        part = [r for r in part if r[1] > 0]
        n, s, b = sum(r[1] for r in part), sum(r[2] for r in part), sum(r[3] for r in part)
        if not n:
            return {"windows": 0, "tokens": 0}
        mean = s / n
        se = (math.sqrt(len(part) / (len(part) - 1) * sum((r[2] - mean * r[1]) ** 2 for r in part)) / n
              if len(part) > 1 else None)
        return {"windows": len(part), "tokens": n, "mean_abs": round(mean, 5), "se": None if se is None else round(se, 5),
                "beyond_half": round(b / n, 5)}

    by = {}
    for r in rows:
        by.setdefault(str(lag(r[0])), []).append(r)
    return {"all": pool(rows), **{f"lag{k}": pool(v) for k, v in sorted(by.items())}}


async def run_rl_loop(study, trial, **kwargs):
    return await Loop(study, trial, **kwargs).run()


