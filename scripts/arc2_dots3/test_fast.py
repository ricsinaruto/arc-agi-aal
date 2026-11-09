import os, sys, socket, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dots3_screen import fast_args, pool_log, boot, LOCAL

def test_args():
    d = fast_args('dp4', 8000, 16)
    assert d[d.index('--tensor-parallel-size') + 1] == '1' and d[d.index('--data-parallel-size') + 1] == '4'
    assert d[d.index('--max-num-seqs') + 1] == '16' and d[d.index('--max-num-batched-tokens') + 1] == '16384'
    assert '--enable-expert-parallel' in d and '--enable-prefix-caching' in d and d[2] == LOCAL and '262144' in d
    t = fast_args('tp4', 8000, 48)
    assert t[t.index('--tensor-parallel-size') + 1] == '4' and '--data-parallel-size' not in t

def test_pool():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d, 'v.log'); log.write_text('GPU KV cache size: 5 tokens\n' + ''.join(
            f'(EngineCore_DP{k}) GPU KV cache size: {v:,} tokens\n' for k, v in enumerate((10, 20, 30, 40))))
        assert pool_log(log, 4, Path(d, 'p'))[0] == 100 and Path(d, 'p').read_text() == 'GPU KV cache size: 100 tokens\n'
        log.write_text('GPU KV cache size: 1,131,347 tokens\n')
        assert pool_log(log, 4, Path(d, 'p'))[0] == 4 * 1131347
        assert pool_log(log, 1, Path(d, 'p'))[0] == 1131347

def test_boot():
    with tempfile.TemporaryDirectory() as d:
        Path(d, 'v1').mkdir(); Path(d, 'v1', 'models').write_text('{}')
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
        url = f'http://127.0.0.1:{port}/v1/models'
        name, proc, _ = boot([('dp4', [sys.executable, '-c', 'import sys; sys.exit(3)'], dict(os.environ), Path(d, 'a'), url),
                              ('tp4', [sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1', '--directory', d],
                               dict(os.environ), Path(d, 'b'), url)], print, poll=0.5, ready_timeout=30)
        assert name == 'tp4'; proc.kill()
        name, proc, _ = boot([('dp4', [sys.executable, '-c', 'import time; time.sleep(30)'], dict(os.environ), Path(d, 'c'), url)],
                             print, poll=0.5, ready_timeout=2)
        assert name is None

if __name__ == '__main__':
    test_args(); test_pool(); test_boot(); print('fast tests pass')
