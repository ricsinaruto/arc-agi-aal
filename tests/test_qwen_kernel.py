import sys
from dataclasses import replace

import pytest
from test_qwen_core import bridge_at

from qwen_baselines.config import Config
from qwen_baselines.kernel import Kernel
from qwen_baselines.serving import worker_probe

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Requires Linux Landlock/seccomp CPU test container"
)


async def test_persistent_state_errors_snapshots_archive_and_restart(tmp_path):
    bridge = await bridge_at(tmp_path)
    kernel = Kernel(bridge, Config(python_seconds=0.3))
    await kernel.start()
    try:
        first = await kernel.execute(
            "def value(board): return int(board[0,0])\nsaved = grid\na = 7\nstep('ACTION7')\nraise ValueError('ordinary')"
        )
        assert first["status"] == "error" and not first["namespace_lost"]
        result = await kernel.execute(
            "print(a, value(saved), value(observe()['grid']), len(history), len(transitions))"
        )
        assert "7 0 1 2 1" in result["output"]
        result = await kernel.execute("saved.setflags(write=True)")
        assert result["status"] == "error"
        result = await kernel.execute("while True: pass")
        assert result["namespace_lost"] and "not replayed" in result["error"]
        # The deliberately tiny timeout tests interruption, not the speed of
        # rebuilding a worker on a shared cloud CPU. Restore the real allowance.
        kernel.config = replace(kernel.config, python_seconds=30)
        result = await kernel.execute(
            "print(len(history), transcript[0]['code'], history[0]['grid'][0,0])"
        )
        assert "def value(board)" in result["output"] and result["status"] == "complete"
        result = await kernel.execute("print(a)")
        assert "NameError" in result["error"]
        assert bridge.actions == 1
    finally:
        await kernel.close()


async def test_cross_level_persistence_and_full_history_after_eviction(tmp_path):
    bridge = await bridge_at(tmp_path, clear_at=2)
    kernel = Kernel(bridge, Config())
    await kernel.start()
    try:
        result = await kernel.execute(
            "helper = lambda: observe()['levels_completed']\nstep('ACTION1')\nstep('ACTION1')\nprint(helper())"
        )
        assert "1" in result["output"]
        await bridge.begin_turn()
        result = await kernel.execute("print(helper()); step('RESET')")
        assert not result["namespace_lost"]
        await bridge.begin_turn()
        result = await kernel.execute("print(helper(),len(transitions[:]),len(list(history)))")
        assert "0 3 4" in result["output"]
    finally:
        await kernel.close()


async def test_full_lazy_archive_after_cache_eviction_and_worker_replacement(tmp_path):
    bridge = await bridge_at(tmp_path)
    kernel = Kernel(bridge, Config())
    await kernel.start()
    try:
        code = "for i in range(40):\n    step('ACTION7')\nprint('archived evidence')"
        result = await kernel.execute(code)
        assert result["status"] == "complete" and bridge.actions == 40
        assert len(bridge.archive.cache) <= 32
        await kernel.close()
        result = await kernel.execute(
            "print(history[0]['grid'][0,0], len(list(transitions)), transcript[0]['output']); print([o['index'] for o in history[-2:]])"
        )
        assert "0 40 archived evidence" in result["output"]
        assert "[39, 40]" in result["output"]
        assert bridge.archive.transcript(0)["code"] == code
    finally:
        await kernel.close()


async def test_segment_binding_in_the_worker(tmp_path):
    bridge = await bridge_at(tmp_path)
    kernel = Kernel(bridge, Config())
    await kernel.start()
    try:
        result = await kernel.execute(
            "s = segment(grid); f = segment(observe()['frames'][0]); "
            "print(len(s['nodes']), s['nodes'][0]['bbox'], s['nodes'][0]['color'], len(f['nodes']))"
        )
        assert result["status"] == "complete" and "1 [0, 0, 64, 64] 0 1" in result["output"]
        result = await kernel.execute("import segmentation")
        assert "ImportError" in result["error"]
    finally:
        await kernel.close()


async def test_transition_helpers_in_the_worker(tmp_path):
    bridge = await bridge_at(tmp_path)
    bridge.objects = True  # runner enables it for A/B/C
    kernel = Kernel(bridge, Config())
    await kernel.start()
    try:
        result = await kernel.execute(
            "print(changes())\nr = step('ACTION1')\nc = changes()\n"
            "print(c['changed_cells'], c['change_bbox'], r['transition']['changed_cells'], "
            "previous()['index'], observe()['index'])"
        )
        assert result["status"] == "complete", result
        assert result["output"].startswith("None\n")  # objects were enabled after initialize
        assert "4096 [0, 0, 64, 64] 4096 0 1" in result["output"]
    finally:
        await kernel.close()


def test_serving_preflight_worker_probe_reaches_ready(tmp_path):
    # The same probe serve-native runs before GPU startup; it must stage the helper itself.
    probe = worker_probe(tmp_path)
    assert probe.returncode == 0 and '"type": "ready"' in probe.stdout, probe.stderr
