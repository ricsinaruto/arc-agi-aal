import json
import sys
from pathlib import Path

import pytest

from arena import rl_loop

ROOT = Path(__file__).resolve().parents[1]


def _modal_rl1():
    sys.path.insert(0, str(ROOT / "scripts/posttrain"))
    return pytest.importorskip("modal_rl1")


def test_settings_merge_only_skips_the_redesign_validators(monkeypatch):
    # the image's trainer python lacks arcengine, which rl_seq's validators (check_mg) import (T3/T4 trainer_key_error)
    def no_play_stack(mg):
        raise ModuleNotFoundError("arcengine")

    monkeypatch.setattr(rl_loop, "check_mg", no_play_stack)
    raw = {"rank": 8, "mg": {"redesign": {"cards": 1}}}
    merged = rl_loop.settings(raw, check=False)
    assert merged["rank"] == 8 and merged["mg"]["redesign"] == {"cards": 1}
    assert merged["window"] == rl_loop.DEFAULTS["window"]


def test_boot_cache_in_skips_the_legacy_compile_restore():
    m = _modal_rl1()
    assert m.boot_cache_in({"trials": [{"server_env": {"ARC_BOOT_CACHE_IN": "/runs/boot-cache/hero"}}]})
    assert not m.boot_cache_in({"trials": [{"server_env": {"ARC_BOOT_CACHE_SAVE": "1"}}, {}]})
    assert not m.boot_cache_in(None)


def test_host_check_records_and_flags_slow(tmp_path):
    m = _modal_rl1()
    fake = tmp_path / "py"
    fake.write_text('#!/bin/sh\necho \'{"host": "h", "gib": 4.0, "pin_alloc_s": 1.2, "d2h_gib_s": 0.22, "h2d_gib_s": 0.5}\'\n')
    fake.chmod(0o755)
    identity = {}
    row = m.host_check(identity, path=tmp_path / "hc.json", python=str(fake))
    assert row["slow"] is True and identity["host_check"] == row and row["pin_alloc_gib_s"] == 3.33
    # T4r-v2's host: a fast copy (52 GiB/s) is not enough; a slow pinned allocation flags it
    fake.write_text('#!/bin/sh\necho \'{"host": "h", "gib": 4.0, "pin_alloc_s": 8.0, "d2h_gib_s": 52.2}\'\n')
    assert m.host_check({}, path=tmp_path / "hc3.json", python=str(fake))["slow"] is True
    fake.write_text('#!/bin/sh\necho \'{"host": "h", "gib": 4.0, "pin_alloc_s": 1.21, "d2h_gib_s": 52.2}\'\n')
    assert m.host_check({}, path=tmp_path / "hc4.json", python=str(fake))["slow"] is False
    assert json.loads((tmp_path / "hc.json").read_text())["d2h_gib_s"] == 0.22
    # no card here: the real probe records why it skipped, the run goes on
    row = m.host_check({}, path=tmp_path / "hc2.json")
    assert "skipped" in row or "error" in row or row.get("d2h_gib_s") is not None
    assert set(row["memory"]) == {"mem_total_gib", "mem_available_gib", "cgroup_max_gib", "cgroup_current_gib"}


def test_memory_limits_reads_meminfo_and_cgroup(tmp_path):
    m = _modal_rl1()
    (tmp_path / "meminfo").write_text(f"MemTotal: {1000 * 2 ** 20} kB\nMemFree: 1 kB\nMemAvailable: {200 * 2 ** 20} kB\n")
    (tmp_path / "memory.max").write_text(str(900 * 2 ** 30))
    (tmp_path / "memory.current").write_text(str(300 * 2 ** 30))
    assert m.memory_limits(tmp_path / "meminfo", tmp_path) == {
        "mem_total_gib": 1000.0, "mem_available_gib": 200.0, "cgroup_max_gib": 900.0, "cgroup_current_gib": 300.0}
    (tmp_path / "memory.max").write_text("max\n")
    assert m.memory_limits(tmp_path / "meminfo", tmp_path)["cgroup_max_gib"] is None
    assert m.memory_limits(tmp_path / "none", tmp_path / "none")["mem_total_gib"] is None


def test_procmem_row_groups_rss_anon(tmp_path):
    m = _modal_rl1()
    procs = {101: ("python\0-u\0/opt/rl1/rl1_trainer.py\0--dir\0x\0", "python", 200),
             102: ("python\0-u\0/opt/rl1/rl1_trainer.py\0", "python", 100),
             103: ("VLLM::Worker\0", "VLLM::Worker", 50),
             104: ("/tmp/v/bin/python\0-m\0vllm.entrypoints.cli.main\0serve\0", "python", 20),
             105: ("python\0-c\0from multiprocessing.spawn import spawn_main\0", "python", 4),
             106: ("", "kworker", 0)}
    for pid, (cmd, comm, gib) in procs.items():
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "cmdline").write_text(cmd)
        (d / "comm").write_text(comm + "\n")
        (d / "status").write_text(f"Name: {comm}\nRssAnon:\t{gib * 2 ** 20} kB\n")
    (tmp_path / "self").mkdir()
    (tmp_path / "meminfo").write_text(f"AnonPages: {500 * 2 ** 20} kB\nShmem: {178 * 2 ** 20} kB\n")
    row = m.procmem_row(tmp_path, top=2)
    assert row["anon_gib"] == 374.0 and row["basis"] == ["RssAnon"]
    assert row["anonpages_gib"] == 500.0 and row["shmem_gib"] == 178.0
    assert row["groups"]["python rl1_trainer.py"] == [2, 300.0]
    assert row["groups"]["VLLM::Worker"] == [1, 50.0]
    assert row["groups"]["python -m vllm.entrypoints.cli.main"] == [1, 20.0]
    assert "kworker" not in row["groups"]
    assert row["top"] == [[101, "python rl1_trainer.py", 200.0], [102, "python rl1_trainer.py", 100.0]]
    (tmp_path / "101" / "status").write_text(f"Name: python\nVmRSS:\t{7 * 2 ** 20} kB\n")  # gVisor: no RssAnon
    row = m.procmem_row(tmp_path, top=1)
    assert row["groups"]["python rl1_trainer.py"] == [2, 107.0] and row["basis"] == ["RssAnon", "VmRSS"]
