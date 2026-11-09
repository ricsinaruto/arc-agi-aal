"""The llama.cpp arena backend for model screens (27 September): server command, fixed slot capacity,
token counts, metric aliases, template controls, the serve check's pieces, the bundle and the budget."""

import ast
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from arc_infra.metrics import Monitor, parse_metrics
from arc_infra.model import ModelClient
from arc_infra.model_templates import (
    DOTS3_MODEL,
    GLM53_FLASH_MODEL,
    allowed_efforts,
    request_messages,
    template_kwargs,
    text_only,
)
from arena.llamacpp import SlotCapacity, overlap_rate, server_argv, study_profile, tool_round_trip
from arena.llamacpp_profiles import PROFILES, profile

ROOT = Path(__file__).resolve().parents[1]

METRICS = """# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 1200
llamacpp:prompt_tokens_cached_total 34000
llamacpp:tokens_predicted_total {predicted}
llamacpp:requests_processing {running}
llamacpp:requests_deferred 0
"""


def study(slots=4, window=49152):
    return {"generation_slots": slots, "windows": [window], "llamacpp": {"profile": "glm53-flash-aj-iq2xxs"}}


def test_server_command_gives_every_slot_a_fixed_window_and_the_model_card_sampling():
    prof = profile("glm53-flash-aj-iq2xxs")
    argv = server_argv(prof, binary="/opt/llama-server", weights_dir="/weights/glm", slots=4, window=49152)
    assert argv[:3] == ["/opt/llama-server", "-m", "/weights/glm/" + prof["model_file"]]
    pairs = dict(zip(argv, argv[1:], strict=False))
    assert pairs["-c"] == str(4 * 49152) and pairs["-np"] == "4" and "--no-kv-unified" in argv
    assert pairs["-ctk"] == pairs["-ctv"] == "q8_0" and pairs["-fa"] == "on" and pairs["-ngl"] == "99"
    assert "--jinja" in argv and pairs["--reasoning-format"] == "deepseek" and "--metrics" in argv
    assert pairs["-a"] == GLM53_FLASH_MODEL and pairs["--port"] == "1234"
    assert (pairs["--temp"], pairs["--top-p"], pairs["--top-k"], pairs["--min-p"]) == ("1.0", "0.95", "0", "0.0")
    # The MTP block was pruned at quantization (--prune-layers 45); nothing to disable at serve time.
    assert "--prune-layers" not in argv and "--spec-type" not in argv


def test_profiles_pin_weights_and_llama_cpp_source():
    for name, prof in PROFILES.items():
        assert prof["model_file"] in prof["files"] and all(size > 0 for size in prof["files"].values())
        assert len(prof["llama_cpp"]["commit"]) == 40
        assert prof["text_only"] or prof.get("mmproj") in prof["files"]
        assert prof["window"] <= 262144 and prof["slots"] > 0, name
    glm = profile("glm53-flash-aj-iq2xxs")
    assert glm["llama_cpp"]["ref"] == "pull/27752/head" and sum(glm["files"].values()) == 87346006560
    assert sum(profile("dots3-note-prev-iq2xs")["files"].values()) == 84256657856
    with pytest.raises(ValueError, match="Unknown llama.cpp profile"):
        profile("mimo")
    assert study_profile({"llamacpp": {"profile": "glm53-flash-aj-iq2xxs",
                                       "overrides": {"model_file": "tiny.gguf"}}})["model_file"] == "tiny.gguf"


def test_flash_iq3_preserves_37_windows_and_vision_projector():
    prof = profile("flash-next-gsq-iq3s")
    argv = server_argv(prof, binary="server", weights_dir="/weights", slots=37, window=93184)
    assert argv[argv.index('-c') + 1] == str(37 * 93184)
    assert argv[argv.index('--mmproj') + 1] == '/weights/' + prof['mmproj']
    assert argv[argv.index('-ctk') + 1] == argv[argv.index('-ctv') + 1] == 'q4_0'


def test_llama_pilot_budget_preserves_scheduler_horizon():
    from arena.llamacpp_run import effective_trial_wall
    study = {'wall_seconds': 32400, 'early_stop': {'always': True, 'at_seconds': 1800}}
    assert effective_trial_wall(study) == 1800
    assert study['wall_seconds'] == 32400
    study['early_stop'] = {'at_seconds': 1800, 'score_above': 1}
    assert effective_trial_wall(study) == 32400  # conditional stops cannot justify underbudgeting


def test_slot_capacity_is_a_plain_slot_gate_checked_against_the_server(tmp_path):
    props = {"total_slots": 4, "default_generation_settings": {"n_ctx": 49152}}
    (tmp_path / "props.json").write_text(json.dumps(props))
    choice = SlotCapacity.load(tmp_path, study()).choice()
    assert choice["window"] == 49152 and choice["capacity"] == 4 * 49152 and choice["slots"] == 4
    for bad in ({"total_slots": 2, "default_generation_settings": {"n_ctx": 49152}},
                {"total_slots": 4, "default_generation_settings": {"n_ctx": 32768}},
                {"total_slots": 4}):
        with pytest.raises(RuntimeError, match="llama-server has"):
            SlotCapacity(bad, study())


def test_llamacpp_metrics_read_through_the_vllm_names(tmp_path):
    values = parse_metrics(METRICS.format(predicted=500, running=3))
    assert values["vllm:generation_tokens_total"] == 500 and values["vllm:num_requests_running"] == 3
    assert values["vllm:num_requests_waiting"] == 0 and values["llamacpp:tokens_predicted_total"] == 500
    assert values["vllm:prompt_tokens_by_source_total:local_cache_hit"] == 34000
    monitor = Monitor(tmp_path)
    monitor.samples = [{"elapsed_seconds": 0, "metrics": parse_metrics(METRICS.format(predicted=500, running=0))},
                       {"elapsed_seconds": 100, "metrics": parse_metrics(METRICS.format(predicted=4500, running=4))}]
    summary = monitor.summary()
    assert summary["valid"] and summary["accepted_generation_tokens"] == 4000
    assert summary["accepted_generation_tokens_per_second"] == 40 and "preemptions" not in summary
    assert summary["peak_running_requests"] == 4
    assert summary["generation_counter_updates"] == "on_slot_reset"
    assert summary["generation_counter_includes_inflight"] is False
    # A flat completion counter with running requests must not be read as an idle decoder.
    from arena.live_progress import engine_snapshot
    live = engine_snapshot([monitor.samples[-1], {**monitor.samples[-1], "elapsed_seconds": 200}])
    assert live["tokens_per_second_recent"] == 0
    assert live["generation_counter_includes_inflight"] is False
    vllm = engine_snapshot([{"elapsed_seconds": 0, "metrics": {"vllm:generation_tokens_total": 10}},
                            {"elapsed_seconds": 10, "metrics": {"vllm:generation_tokens_total": 30}}])
    assert vllm["tokens_per_second_recent"] == 2
    assert "generation_counter_updates" not in vllm
    # vLLM lines parse exactly as before.
    assert parse_metrics('vllm:num_preemptions_total{engine="0"} 2\n') == {"vllm:num_preemptions_total": 2.0}


@pytest.mark.parametrize("count,expected,status", [(None, None, "complete"), (0, True, "complete"),
                                                   (2, False, "complete_with_preemptions")])
def test_missing_preemption_counter_is_unknown(monkeypatch, count, expected, status):
    from arena import report
    monkeypatch.setattr(report, "fleet_summary", lambda rows, seconds: {
        "failed_sessions": 0, "mean_score_equal_weight_25_games": 0})
    monkeypatch.setattr(report, "audit_scheduler", lambda result: {"valid": True})
    engine = {"valid": True, **({"preemptions": count} if count is not None else {})}
    result = report.trial_report({}, rows=[], elapsed=10, choice={}, cap=100, config={},
                                 scheduler={}, gate={}, engine=engine, task_errors=[])
    assert result["preemption_free"] is expected
    assert result["status"] == status


def test_model_client_counts_through_llama_server_input_tokens():
    seen = []

    def handler(request):
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"input_tokens": 1234, "object": "response.input_tokens"})

    config = SimpleNamespace(base_url="http://127.0.0.1:1234/v1", parallel_games=1, model=GLM53_FLASH_MODEL,
                             preserve_reasoning=False, reasoning_effort="max", max_model_len=49152,
                             serving_backend="llamacpp")
    client = ModelClient(config)
    client.client = httpx.AsyncClient(base_url="http://127.0.0.1:1234/v1/", transport=httpx.MockTransport(handler))
    tools = [{"type": "function", "function": {"name": "python"}}]
    count = asyncio.run(client.count_tokens([{"role": "user", "content": "hi"}], 1e12, tools=tools,
                                            template_kwargs={"reasoning_effort": "max"}))
    assert count == 1234
    path, body = seen[0]
    assert path == "/v1/chat/completions/input_tokens"
    assert body["tools"] == tools and body["chat_template_kwargs"] == {"reasoning_effort": "max"}
    assert body["messages"] == [{"role": "user", "content": "hi"}] and body["model"] == GLM53_FLASH_MODEL


def test_glm_and_dots3_template_controls_and_text_boards():
    assert text_only(GLM53_FLASH_MODEL) and text_only(DOTS3_MODEL)
    assert allowed_efforts(GLM53_FLASH_MODEL) == ("max", "high", "low") and allowed_efforts(DOTS3_MODEL) == ("native",)
    assert template_kwargs(GLM53_FLASH_MODEL, {"enable_thinking": True, "reasoning_effort": "max"}) == {
        "reasoning_effort": "max"}
    assert template_kwargs(GLM53_FLASH_MODEL, {"reasoning_effort": "xhigh"}) == {"reasoning_effort": "max"}
    # The template has no off switch: thinking-off asides (compaction summaries) get the shortest effort.
    assert template_kwargs(GLM53_FLASH_MODEL, {"enable_thinking": False, "reasoning_effort": "max"}) == {
        "reasoning_effort": "low"}
    assert template_kwargs(DOTS3_MODEL, {"enable_thinking": False, "reasoning_effort": "native"}) == {
        "enable_thinking": False}
    messages = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    assert request_messages(GLM53_FLASH_MODEL, messages) == [{"role": "user", "content": "a\nb"}]


def test_overlap_rate_counts_tokens_only_while_every_stream_decodes():
    one = [0.0, 1.0, 2.0, 3.0, 4.0]
    two = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    # Window (2, 4]: stream one has 2 tokens, stream two 4 (2.5, 3, 3.5, 4) -> 6 tokens / 2 s.
    assert overlap_rate([one, two]) == pytest.approx(3.0)
    assert overlap_rate([one]) == pytest.approx(1.0)
    assert overlap_rate([[0.0, 1.0], [5.0, 6.0]]) is None and overlap_rate([[0.0]]) is None


def replies(first_message, second_text):
    responses = iter([
        {"choices": [{"message": first_message, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 6100},
         "timings": {"prompt_n": 6100}},
        {"choices": [{"message": {"role": "assistant", "content": second_text}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 6300}, "timings": {"prompt_n": 250}},
    ])

    def handler(request):
        if request.url.path == "/tokenize":
            text = json.loads(request.content)["content"]
            return httpx.Response(200, json={"tokens": [0] * len(text.split())})
        return httpx.Response(200, json=next(responses))

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def run_round_trip(client):
    return asyncio.run(tool_round_trip(client, "http://x/v1", GLM53_FLASH_MODEL, template_kwargs={"reasoning_effort": "max"},
                                       sampling={"temperature": 1.0, "top_p": 0.95}, system_tokens=50))


def test_tool_round_trip_passes_on_a_python_call_and_an_answer_that_uses_its_result():
    call = {"id": "c1", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": "print(1)"})}}
    good = run_round_trip(replies({"role": "assistant", "content": "", "reasoning_content": "think",
                                   "tool_calls": [call]}, "It is 121,932,631,112,635,269."))
    assert good["passed"] and good["tool_call_ok"] and good["answer_uses_result"] and good["reasoning_chars"] == 5
    assert good["second_turn_reused_fraction"] == pytest.approx(1 - 250 / 6300, abs=1e-3)
    no_call = run_round_trip(replies({"role": "assistant", "content": "121932631112635269"}, "unused"))
    assert not no_call["passed"] and not no_call["tool_call_ok"]


def test_llamacpp_probe_bundle_keeps_the_probe_and_replaces_the_serving_keys(tmp_path):
    spec = importlib.util.spec_from_file_location("build_arena", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    games = ["ar25", "sb26", "tu93", "vc33", "lf52", "r11l", "re86", "cn04"]
    _, flash = builder.build("flash", paired_30m=True, output=tmp_path / "flash", arena_variant="reclaim-fixed-2x",
                             opening_burst_requests=3, wall_seconds=900, probe_games=games)
    _, glm = builder.build("glm", paired_30m=True, output=tmp_path / "glm", arena_variant="reclaim-fixed-2x",
                           opening_burst_requests=3, wall_seconds=3600, probe_games=games,
                           llamacpp="glm53-flash-aj-iq2xxs")
    assert glm["backend"] == "llamacpp" and glm["llamacpp"]["profile"] == "glm53-flash-aj-iq2xxs"
    assert glm["served_model"] == GLM53_FLASH_MODEL and glm["windows"] == [49152] and glm["generation_slots"] == 4
    assert glm["reasoning_effort"] == "max" and glm["temperature"] == 1.0 and glm["sampling"]["top_k"] == 0
    assert not {"cache_admission", "cache_experiment", "isolate_trial_cache"} & set(glm)
    same = ("games", "probe", "completion_cap", "opening_switch_level1_games", "level_budget_policy", "budget_clock",
            "backlog_policy", "opening_burst_requests", "trials", "active_limit")
    assert {k: glm[k] for k in same} == {k: flash[k] for k in same}
    assert glm["llamacpp"]["kv_unified"] is False and "serve_check_extra" not in glm["llamacpp"]
    # GLM follow-up: one slot or a unified cache, and a trimmed serve check for a probe.
    _, one = builder.build("glm1", paired_30m=True, output=tmp_path / "glm1", arena_variant="reclaim-fixed-2x",
                           opening_burst_requests=3, wall_seconds=2700, probe_games=games,
                           llamacpp="glm53-flash-aj-iq2xxs", llamacpp_slots=1, llamacpp_kv_unified=True,
                           llamacpp_serve_check={"depths": [8192], "streams": [1]})
    assert one["generation_slots"] == 1 and one["server"]["slots"] == 1 and one["server"]["context"] == 49152
    assert one["llamacpp"]["kv_unified"] and one["llamacpp"]["serve_check"] == {"depths": [8192], "streams": [1]}
    _, dots = builder.build("dots", paired_30m=True, output=tmp_path / "dots", arena_variant="reclaim-fixed-2x",
                            opening_burst_requests=3, wall_seconds=3600, probe_games=games,
                            llamacpp="dots3-note-prev-iq2xs")
    assert dots["llamacpp"]["server_args"][:6] == ["-ngl", "99", "-fa", "auto", "-ub", "128"]
    assert "-ctk" not in dots["llamacpp"]["server_args"] and "serve_check_extra" not in dots["llamacpp"]
    argv = server_argv(profile("dots3-note-prev-iq2xs"), binary="b", weights_dir="w", slots=8, window=49152,
                       kv_unified=True)
    assert "--kv-unified" in argv and "--no-kv-unified" not in argv and argv[argv.index("-c") + 1] == str(8 * 49152)
    with pytest.raises(ValueError, match="--llamacpp"):
        builder.build("bad", paired_30m=True, output=tmp_path / "bad", arena_variant="reclaim-fixed-2x",
                      wall_seconds=3600, llamacpp="glm53-flash-aj-iq2xxs")


def test_trial_duck_config_carries_the_backend_and_the_loader_takes_the_card_sampling():
    from arena.trial import duck_config
    from duck_transfer import loader

    base = {"reasoning_effort": "max", "temperature": 1.0, "generation_slots": 4, "server": {"mtp_tokens": 0},
            "served_model": GLM53_FLASH_MODEL, "backend": "llamacpp"}
    cfg = duck_config(base, ROOT, window=49152, cap=16384, wall=3600, base_url="http://127.0.0.1:1234/v1")
    assert cfg.serving_backend == "llamacpp" and cfg.model == GLM53_FLASH_MODEL and cfg.speculative_tokens == 0
    before = dict(loader.DUCK_ENV)
    try:
        loader.configure(cfg, sampling={"top_p": 0.95, "top_k": 0})
        assert loader.DUCK_ENV["LOCAL_ANALYZER_TOP_K"] == "0" and loader.DUCK_ENV["LOCAL_ANALYZER_TOP_P"] == "0.95"
    finally:
        loader.DUCK_ENV.clear()
        loader.DUCK_ENV.update(before)


def test_live_progress_line_reads_the_trial_and_the_server_counter(tmp_path):
    from arena.llamacpp import progress_line

    (tmp_path / "progress.json").write_text(json.dumps({"elapsed_seconds": 60.2, "levels": 2, "active": ["a", "b"]}))
    line = progress_line(tmp_path, METRICS.format(predicted=6500, running=4), {"epoch": 0, "tokens": 500})
    assert line["gameplay_seconds"] == 60.2 and line["levels"] == 2 and line["active"] == 2
    assert line["completion_tokens"] == 6000 and line["running"] == 4
    empty = progress_line(tmp_path / "missing", "", {"epoch": 0, "tokens": None})
    assert empty["levels"] is None and "completion_tokens" not in empty


def test_flag_check_fails_when_the_binary_cannot_print_its_help(tmp_path):
    from arena.llamacpp import check_flags

    fake = tmp_path / "llama-server"
    fake.write_text("#!/bin/sh\necho 'error while loading shared libraries: libcuda.so.1' >&2\nexit 127\n")
    fake.chmod(0o755)
    with pytest.raises(RuntimeError, match="--help failed"):
        check_flags(fake, ["--metrics"])
    fake.write_text("#!/bin/sh\necho '--port PORT  --metrics  --jinja'\n")
    assert check_flags(fake, ["--metrics", "--jinja", "-c", "10"])
    with pytest.raises(RuntimeError, match="does not know"):
        check_flags(fake, ["--cache-ram"])


def load_reserve(ledger):
    source = ROOT / "scripts/modal_llamacpp.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reserve")
    ns = {"LEDGER": ledger, "TEST": "probe-60m", "PROFILE_NAME": "glm53-flash-aj-iq2xxs", "json": json, "Path": Path,
          "GPU_RATE": 0.000842, "CPU_CEILING": 8, "MEMORY_CEILING_GIB": 96}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    return ns["reserve"]


def test_screen_budget_reserves_in_the_ledger_and_refuses_duplicates_and_the_cap(tmp_path):
    ledger = tmp_path / "spend.json"
    ledger.write_text(json.dumps({"infrastructure_reserve_usd": 2, "runs": [{"run_id": "x", "charged_bound_usd": 1}],
                                  "cap_usd": 20}))
    reserve = load_reserve(ledger)
    bound = reserve("glm-probe", 7200)
    assert bound == pytest.approx((7200 + 60) * (0.000842 + 8 * 0.0000131 + 96 * 0.00000222) * 1.15)
    entry = json.loads(ledger.read_text())["runs"][-1]
    assert entry["run_id"] == "glm-probe" and entry["kind"] == "llamacpp-screen" and entry["state"] == "reserved"
    before = ledger.read_text()
    with pytest.raises(ValueError, match="already reserved"):
        reserve("glm-probe", 60)
    with pytest.raises(RuntimeError, match="Spend cap"):
        reserve("glm-probe-2", 7200)
    assert ledger.read_text() == before
