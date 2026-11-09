"""Remote job failures must not be mistaken for transient polling failures."""
import runpy
import socket
from pathlib import Path

import pytest

modal = pytest.importorskip("modal")
call_state = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                               "scripts/arena/watch_live.py"))["call_state"]


class Call:
    def __init__(self, error=None):
        self.error = error

    def get(self, *, timeout):
        assert timeout == 0
        if self.error is not None:
            raise self.error


@pytest.mark.parametrize("error", [TimeoutError(), ConnectionError(), socket.gaierror(),
                                   modal.exception.ConnectionError("transport")])
def test_transient_poll_errors(error):
    assert call_state(Call(error)) == "running"


@pytest.mark.parametrize("error", [modal.exception.FunctionTimeoutError("Function timed out"),
                                   RuntimeError("Connection to model failed"),
                                   RuntimeError("worker timed out")])
def test_execution_failures_are_terminal(error):
    assert call_state(Call(error)).startswith("error:")


def test_completion():
    assert call_state(Call()) == "done"


def test_campaign_final_report_uses_closed_score(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts/arena"))
    module = runpy.run_path(str(root / "scripts/arena/watch_campaign.py"))
    final = module['completed_progress']({
        'status': 'complete', 'wall_seconds': 1800, 'levels': 3,
        'mean_score_equal_weight_games': .75, 'actions': 100,
        'engine': {'accepted_generation_tokens_per_second': 999, 'preemptions': 0},
        'games': [{'compactions': 2, 'compaction_seconds': 50}],
        'end_reason': 'stop_at', 'scheduler_audit': {'valid': True}})
    assert final['live']['score'] == {'score': .75, 'at_elapsed_seconds': 1800}
    assert final['live']['compactions'] == 2 and final['phase'] == 'final'
    assert 'gameplay_tokens' not in final['live']  # Do not invent selected-token accounting.
    text = module['render']([{'run_id': 'example', 'state': 'done', 'progress': final}], 'now')
    assert '0.75' in text and 'Final status=complete; end=stop_at; scheduler audit valid=True' in text
