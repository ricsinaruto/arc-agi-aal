"""Linux startup resource snapshots without reading process arguments or environments."""

import json
import os
import time
from pathlib import Path


def sample(root_pid, *, proc=Path('/proc')):
    proc = Path(proc)
    rows = {}
    skipped = 0
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            # comm can contain spaces and parentheses; the final ')' ends it.
            raw = (directory / 'stat').read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            rows[int(directory.name)] = {
                'pid': int(directory.name), 'ppid': int(fields[1]),
                'pgid': int(fields[2]), 'state': fields[0],
                'start_ticks': int(fields[19]),
                'minor_faults': int(fields[7]), 'major_faults': int(fields[9]),
                'user_ticks': int(fields[11]), 'system_ticks': int(fields[12]),
                'rss_bytes': int(fields[21]) * os.sysconf('SC_PAGE_SIZE'),
            }
        except (OSError, ValueError, IndexError):
            skipped += 1  # Includes processes that exited during this sample.
    owned = {int(root_pid)}
    while True:
        expanded = owned | {pid for pid, r in rows.items()
                            if r['ppid'] in owned or r['pgid'] == root_pid}
        if expanded == owned:
            break
        owned = expanded
    processes = []
    for pid in sorted(owned & rows.keys()):
        row = rows[pid]
        try:
            io = {}
            for line in (proc / str(pid) / 'io').read_text().splitlines():
                key, value = line.split(':', 1)
                if key in ('read_bytes', 'write_bytes', 'rchar', 'wchar'):
                    io[key] = int(value)
            row['io'] = io
        except (OSError, ValueError) as exc:
            row['io_error'] = type(exc).__name__
        processes.append(row)
    host = {}
    try:
        for line in (proc / 'meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable', 'Cached', 'SwapTotal', 'SwapFree'):
                host[key + '_bytes'] = int(value.split()[0]) * 1024
    except (OSError, ValueError) as exc:
        host['error'] = type(exc).__name__
    return {
        'epoch': time.time(), 'root_pid': root_pid,
        'root_present': root_pid in rows, 'clock_ticks_per_second': os.sysconf('SC_CLK_TCK'),
        'processes': processes, 'host': host, 'unreadable_stat_count': skipped,
        # RSS sums include shared pages more than once; this is not unique RAM usage.
        'summed_rss_bytes': sum(r['rss_bytes'] for r in processes),
    }


def append_sample(root_pid, output):
    """Best-effort diagnostics must not turn a ready server into a startup failure."""
    started = time.monotonic()
    try:
        record = sample(root_pid)
    except Exception as exc:
        record = {'epoch': time.time(), 'root_pid': root_pid, 'error': type(exc).__name__}
    record['sample_seconds'] = time.monotonic() - started
    try:
        with Path(output).open('a') as stream:
            stream.write(json.dumps(record, separators=(',', ':')) + '\n')
    except OSError:
        pass
