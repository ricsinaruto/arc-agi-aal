import json, threading, time

from arena import replay_stub as rs


def test_kv_layout_reads_full_attention_group():
    line = 'x ARC_CACHE_LAYOUT ' + json.dumps({"num_blocks": 6990, "groups": [{"type": "MambaSpec", "block_size": 1},
                                                                             {"type": "FullAttentionSpec", "block_size": 800}]})
    assert rs.kv_layout("a\n" + line) == (6990, 800)
    assert rs.kv_layout("no layout") == (0, 0)


def test_kv_mode_waits_and_logs_when_full(capsys):
    s = rs.State(None, 0, kv_blocks=10, block_size=100)
    a = s.admit(500)  # 5 + 1 state block
    assert a == 6 and abs(s.kv() - 0.6) < 1e-9
    got = []
    t = threading.Thread(target=lambda: got.append(s.admit(500)))
    t.start(); time.sleep(0.3)
    assert not got and s.waiting == 1 and s.failures == 1
    s.release(a); t.join(2)
    assert got == [6] and s.waiting == 0 and s.kv_peak == 0.6
    assert capsys.readouterr().out.count("ARC_CACHE_ALLOCATION_FAILURE") == 1


def test_default_mode_unchanged():
    s = rs.State(None, 0)
    s.running = 64
    assert s.admit(10 ** 6) == 0 and s.kv() == 0.5
