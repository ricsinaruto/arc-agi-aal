"""Host-side plumbing for the bridged Duck analyzer: in-memory state and one log per turn.

The public ``analyze`` re-reads ``tool_runtime_state.json`` every turn and twice per python
call, rewrites a ~100 KB ``prompt.log`` per model request and re-serialises the whole
history into the sandbox on every call (v3/v4 runs: the file reached megabytes and every
action rewrote it). This mixin serves the runtime state from the memory, records
the prompt-log snapshot arguments during the turn and writes them once when it ends, and
passes the v5 receipt keys (``transition``, ``tried_no_effect_here``) through the public
receipt compaction.
"""

import json
from types import MethodType

from .libraries import with_globals

# "gameplay_changed" .. "view_shift" are present only under the arena's harness v2 (effect-aware receipts);
# "gameplay_changed", "animation" and the last four only under the arena's dfr_harness (the batch no-op stop).
RECEIPT_KEYS = ("transition", "tried_no_effect_here", "skipped_repeats", "gameplay_changed", "clock_only",
                "gameplay_cells", "gameplay_bbox", "animation", "view_shift", "no_op_action", "no_op_action_index",
                "skipped_actions", "action_trace")


class PlumbingMixin:
    _owner = None

    def analyze(self, state_path, action_num, **kwargs):
        owner = self._owner
        module = owner.module
        snapshot = {}

        def stash(*args, **data):
            snapshot.clear()
            snapshot.update(args=args, data=data)

        run = MethodType(
            with_globals(
                module.ToolAgent.analyze,
                load_runtime_state=owner.runtime_state,
                _write_prompt_log_snapshot=stash,
            ),
            self,
        )
        self._token_cache = {}
        self._turn_summary = None
        try:
            return run(state_path, action_num, **kwargs)
        finally:
            if snapshot:
                module._write_prompt_log_snapshot(*snapshot["args"], **snapshot["data"])
                owner.prompt_log_writes += 1

    def _run_python_tool(self, state_path, arguments):
        owner = self._owner
        self._python_timeout = max(0.01, min(30, owner.remaining()))
        owner.sandbox.cpu_seconds = int(owner.remaining()) + 60
        execute = MethodType(
            with_globals(
                owner.module.ToolAgent._run_python_tool,
                load_runtime_state=owner.runtime_state,
                _ascii_history_view_payload=owner.ascii_history,
                run_sandboxed_python=owner.sandbox_runner,
            ),
            self,
        )
        output = owner.run_python_tool(execute, state_path, arguments)
        result = json.loads(output.content)
        if output.step_executed and self._last_step_summary:
            self._turn_summary = dict(self._last_step_summary)
        owner.workspace.append(
            "duck_python",
            step=owner.steps,
            code=arguments.get("code"),
            result=result,
            sandbox_calls=owner.sandbox.calls,
            restarts=owner.sandbox.restarts,
            history_synced=owner.sandbox.synced,
        )
        return output

    def _compact_action_result(self, payload):
        compact = super()._compact_action_result(payload)
        for key in RECEIPT_KEYS:
            if key in payload:
                compact[key] = payload[key]
        return compact
