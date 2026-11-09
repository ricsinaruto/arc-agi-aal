import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from sglang_bench import launch, loading_metrics


def process(proc, pid, parent, group, *, rss=2, major=3):
    directory = proc / str(pid)
    directory.mkdir()
    fields = ['0'] * 22
    for index, value in {0: 'S', 1: parent, 2: group, 7: 17, 9: major,
                         11: 23, 12: 5, 19: 12345, 21: rss}.items():
        fields[index] = str(value)
    (directory / 'stat').write_text(f'{pid} (name (with) spaces) ' + ' '.join(fields))
    (directory / 'io').write_text('read_bytes: 4096\nwrite_bytes: 42\nrchar: 9000\nwchar: 30\n')
    return directory


def test_resource_sample_selects_owned_tree_and_group_without_process_arguments(tmp_path):
    process(tmp_path, 10, 1, 10)
    process(tmp_path, 11, 10, 10, rss=3, major=4)
    process(tmp_path, 12, 11, 12)  # Descendant in a different process group.
    process(tmp_path, 13, 1, 10)  # Reparented worker retains the server process group.
    process(tmp_path, 99, 1, 99, rss=99999, major=99999)
    (tmp_path / 'meminfo').write_text('MemAvailable: 100 kB\nMemTotal: 200 kB\nCached: 10 kB\n')
    row = loading_metrics.sample(10, proc=tmp_path)
    assert [p['pid'] for p in row['processes']] == [10, 11, 12, 13]
    assert row['summed_rss_bytes'] == 9 * loading_metrics.os.sysconf('SC_PAGE_SIZE')
    assert row['host']['MemAvailable_bytes'] == 102400
    assert row['processes'][1]['major_faults'] == 4
    assert row['processes'][0]['user_ticks'] == 23
    assert row['processes'][0]['start_ticks'] == 12345
    assert row['processes'][0]['io']['read_bytes'] == 4096
    assert 'cmdline' not in json.dumps(row) and 'environ' not in json.dumps(row)


def test_exiting_process_and_missing_io_are_not_treated_as_zero_counters(tmp_path):
    directory = process(tmp_path, 10, 1, 10)
    (directory / 'io').unlink()
    (tmp_path / '11').mkdir()  # Exited between /proc enumeration and stat read.
    row = loading_metrics.sample(10, proc=tmp_path)
    assert row['unreadable_stat_count'] == 1
    assert row['processes'][0]['io_error'] == 'FileNotFoundError'
    assert 'io' not in row['processes'][0]


def test_unsupported_platform_and_output_errors_do_not_stop_serving(tmp_path, monkeypatch):
    def unavailable(_):
        raise FileNotFoundError('/proc')

    monkeypatch.setattr(loading_metrics, 'sample', unavailable)
    path = tmp_path / 'samples.jsonl'
    loading_metrics.append_sample(10, path)
    row = json.loads(path.read_text())
    assert row['error'] == 'FileNotFoundError' and row['sample_seconds'] >= 0
    loading_metrics.append_sample(10, tmp_path / 'does-not-exist' / 'out')


@pytest.mark.parametrize('outcome', ['ready', 'exit', 'timeout'])
def test_wait_ready_samples_before_poll_and_on_all_exit_paths(tmp_path, monkeypatch, outcome):
    captured = []
    monkeypatch.setattr(loading_metrics, 'append_sample', lambda pid, path: captured.append((pid, path)))
    def endpoint(request):
        if request.url.path == '/health':
            return httpx.Response(200)
        return httpx.Response(200, json={'data': [{'id': 'test-model'}]})

    client = httpx.Client(transport=httpx.MockTransport(endpoint))
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: client)
    log = tmp_path / 'sglang-server.log'
    log.write_text('Load weight end\n')
    server = SimpleNamespace(process=SimpleNamespace(pid=10, poll=lambda: 1 if outcome == 'exit' else None),
                             profile=SimpleNamespace(served_model_name='test-model'),
                             base_url='http://localhost/v1', log_path=log)
    if outcome == 'ready':
        assert launch.wait_ready(server, timeout=1)['models'] == ['test-model']
    else:
        with pytest.raises(RuntimeError if outcome == 'exit' else TimeoutError):
            launch.wait_ready(server, timeout=0 if outcome == 'timeout' else 1)
    assert len(captured) == (1 if outcome == 'timeout' else 2)
    assert all(path == tmp_path / 'loading-resources.jsonl' for _, path in captured)
    assert client.is_closed
