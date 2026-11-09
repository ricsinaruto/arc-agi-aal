import pytest

from arena.prime import rewrite_code, settings, with_prime
from arena.prime.session import ANCHOR, PrimeSession
from arena.prime.store import HarnessStore, parse_proposal, parse_review


def test_rewrite_code():
    assert rewrite_code("await refine.run()") == ("_prime_display((_prime_await( refine.run())))", "await")
    assert rewrite_code("x = 1;")[0] == "x = 1;"
    assert rewrite_code("a, b")[0] == "_prime_display((a, b))"
    out, note = rewrite_code("async def f():\n    return await g()\ny = (await (f()))['k']")
    assert note == "await" and "await " not in out.replace("_prime_await", "")
    compile(out, "<cell>", "exec")
    assert rewrite_code("%%bash\nls")[1] == "magic"
    assert rewrite_code("def f(:\n  pass") == ("def f(:\n  pass", None)


def test_store_apply_and_rollback():
    store = HarnessStore()
    store.create("memory", "Lever", "opens the gate")
    with pytest.raises(ValueError):
        store.create("skill", "s", "c")  # no python reference
    result = store.apply(parse_proposal(
        'text {"summary": "s", "rationale": "r", "expectedOutcome": "o", "edits": ['
        '{"action": "update", "kind": "memory", "id": "local:lever", "title": "Lever", "content": "is at (3,5)"},'
        '{"action": "create", "kind": "skill", "title": "x", "content": "y"},'
        '{"action": "delete", "kind": "memory", "id": "none"}]}'))
    assert [e["applied"] for e in result["appliedEdits"]] == [True, False, False]
    assert store.get("memory", "lever")["version"] == 2 and "is at (3,5)" in store.entries_block()
    store.rollback(result["id"])
    assert store.get("memory", "lever")["content"] == "opens the gate"
    assert parse_review('{"shouldRefine": true}')["shouldRefine"] is True


def test_settings():
    assert settings(None) is None and settings({"arm": "F"})["keep_recent_tokens"] == 20000
    with pytest.raises(ValueError):
        settings({"arm": "X"})
    with pytest.raises(ValueError):
        settings({"arm": "F", "compaction": "native"})


@pytest.mark.parametrize("arm", ["N", "F"])
def test_kernel_block(arm):
    sent = []
    fake = ("def main(_send, _recv, _json_safe, runtime_globals, action_results, initial=None):\n"
            "    def _refresh_state(payload):\n        runtime_globals['history'] = []\n"
            + ANCHOR + "        break\n    return _refresh_state\n")
    scope = {}
    exec(compile(with_prime(fake, settings({"arm": arm})), "<bootstrap>", "exec"), scope)
    globals_ = {"__builtins__": {"__import__": __import__}, "action": lambda actions: {"executed": True}}
    replies = iter([{"type": "animation_result", "animation": {"ok": True, "value": {"id": "lever"}}},
                    {"type": "animation_result", "animation": {"ok": False, "error": "no"}}])
    refresh = scope["main"](sent.append, lambda: next(replies), lambda value: value, globals_, [])
    assert globals_["rlm"].harness.create_memory("Lever", "opens")["id"] == "lever"
    assert sent[0]["request"] == {"prime": "harness.create", "args": {
        "kind": "memory", "title": "Lever", "content": "opens", "path": "general"}}
    with pytest.raises(RuntimeError, match="no"):
        globals_["refine"].run()
    with pytest.raises(RuntimeError):
        globals_["rlm"]("sub-task")
    assert globals_["__builtins__"]["__import__"]("compact") is globals_["compact"]
    refresh({})
    assert ("action" in globals_, "history" in globals_) == ((True, True) if arm == "N" else (False, False))
    assert ("fixed_broker_client" in globals_) == (arm == "F")


def test_cut_index():
    session = PrimeSession.__new__(PrimeSession)
    session.cfg = settings({"arm": "N", "keep_recent_tokens": 25})
    session.message_tokens = lambda message: 10
    roles = ["system", "user", "assistant", "tool", "user", "assistant", "tool", "assistant", "tool"]
    messages = [{"role": r} for r in roles]
    assert session.cut_index(messages, 1) == 7    # 30 tokens kept from index 6 (a tool result): the next cut point
    session.message_tokens = lambda message: 30 if message is messages[-1] else 10
    assert session.cut_index(messages, 1) == 7    # the newest tool result alone exceeds the keep: the call before it
    assert session.cut_index(messages[:2], 1) == 1


def test_takeover_pieces():
    assert settings({"arm": "N", "takeover": True})["takeover"]
    for bad in ({"arm": "F", "takeover": True}, {"arm": "N", "compaction": "native", "takeover": True}):
        with pytest.raises(ValueError):
            settings(bad)
    out, note = rewrite_code("x = await f()\nx", b"_prime_takeover();")
    assert out.startswith("_prime_takeover();x = _prime_await( f())") and out.endswith("_prime_display((x))")
    assert rewrite_code("y = 1;", b"_prime_takeover();")[0] == "_prime_takeover();y = 1;"
    assert rewrite_code("for i in range(2):\n    pass", b"_prime_takeover();")[1] == "rewrite_failed"
    session = PrimeSession.__new__(PrimeSession)
    session.store = HarnessStore()
    assert session.seed("Goal: reach the gate\n- UP moves\n\nFailed:\n- cycling") == 2
    assert session.store.get("memory", "playbook_02")["content"] == "Failed:\n- cycling"
    assert session.seed("") == 0
    # The kernel after the takeover: playbook() raises, also after the next state refresh re-installs it.
    fake = ("def main(_send, _recv, _json_safe, runtime_globals, action_results, initial=None):\n"
            "    def _refresh_state(payload):\n        runtime_globals['playbook'] = lambda text=None: 'kept'\n"
            + ANCHOR + "        break\n    return _refresh_state\n")
    scope, globals_ = {}, {"__builtins__": {"__import__": __import__}}
    exec(compile(with_prime(fake, settings({"arm": "N", "takeover": True})), "<bootstrap>", "exec"), scope)
    refresh = scope["main"](None, None, None, globals_, [])
    refresh({})
    assert globals_["playbook"]() == "kept"
    globals_["_prime_takeover"]()
    refresh({})
    with pytest.raises(RuntimeError, match="retired"):
        globals_["playbook"]("x")


def test_run2_options():
    block = settings({"arm": "N", "refine_at": "level_clear", "notices": False, "digest_full": True, "inventory": True})
    assert block["keep_recent_tokens"] == 20000 and not block["notices"]
    with pytest.raises(ValueError):
        settings({"arm": "N", "refine_at": "never"})
    store = HarnessStore()
    store.create("memory", "Rules", "x" * 500)
    assert "x" * 500 in store.entries_block(full=True) and "x" * 200 not in store.entries_block()
    session = PrimeSession.__new__(PrimeSession)
    session.cfg, session.requested_refine, session.levels_refined = block, None, 0
    session.owner = type("Owner", (), {"obs": type("Obs", (), {"levels": 0})()})()
    assert session.refine_due() is None
    session.owner.obs.levels = 2
    assert session.refine_due() == "level_clear"
