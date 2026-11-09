"""hx5 ``simcheck`` (8 October): the harness checks the agent's own model of a stuck level against what happened.

Why (docs/harness-arm.md 5.2; hx5 LOG): the agent already writes abstract-state simulators and search on its own (a
``step(state, action)`` function in 68 of 120 baseline DEV games, BFS in 80), but nothing checks them against the
recorded play, and 16 of 33 stuck levels kept a mechanic read off one observation. With the winning sequence in the
prompt it clears 86% of never-cleared levels: it executes plans, it does not find them.

Two opt-in flags, both silent before the stuck signal (``MODE_TOKENS`` generated gameplay tokens on a level without a
clear):
- ``dfr_harness.evidence``: one line per user turn with the ledger's counts per action (tries / changed / no-op, a
  no-op seen on one board only flagged, valid actions never tried, same board + action with different results).
- ``dfr_harness.simcheck``: model mode, i.e. one notice with the contract, then one model-check line per user turn.

Mechanism:
- Ledger: every executed transition of the level (board before, action, board after), rebuilt from the host's history
  on every python call (the model cannot alter it).
- Contract: ``MODEL.register(encode, step)`` with ``encode(frame) -> state`` and ``step(state, action) -> state`` in the
  agent's own abstraction. After every python call the child replays the level's ledger through them (time-limited,
  cached per transition) and reports reproduced / wrong / missed (the board interior changed, the state did not: a
  constant ``encode`` cannot fit) with the earliest and the latest mismatch.
- Lever: in model mode an action list is cut after ``CAP`` actions unless a registered model reproduces ``FIT`` of the
  ledger; with one, lists of any length run step by step, each step compared with the model's prediction, and stop at
  the first mismatch (which is new evidence in the ledger). No action is ever refused.
- Search: ``MODEL.solve(goal, ...)`` (breadth-first over ``step``), ``MODEL.frontier(...)`` (shortest predicted path
  to a state that never occurred).
Generic: no per-game information; constants below.
"""

MODE_TOKENS = 30000   # generated gameplay tokens on a level without a clear before model mode starts
CAP = 3               # actions of one list executed without a fitting model, in model mode
FIT = 0.8             # share of the level's ledger a model must reproduce for unlimited checked lists
MIN_LEDGER = 5        # ... on at least this many transitions (fewer: any registered model counts as fitting)
SECONDS = 1.5         # the per-call budget of the ledger replay
BORDER = 4            # the outer ring the interior comparison ignores (the dfr guards' ring)
STATE_CHARS = 140
# simcarry (10 October): the model is written at each clear, scored on the finished level and carried.
CARRY_FIT = 0.9       # share of the finished level's ledger the model must reproduce to end the model-first hold
BUILD_TOKENS = 12000  # generated gameplay tokens after a clear at which the hold ends regardless
PLAN = (60, 100000, 2.0)  # the harness's own search over a fitting model: depth, nodes, seconds per call
PLAN_SHOWN = 40       # actions of an offered plan written into the turn

NOTICE = (
    "[Model check, by the harness] {tokens:,} generated tokens have gone into level {level} without clearing it. "
    "From now on the harness checks your understanding of this level against what actually happened.\n"
    "- The harness keeps a ledger of every transition of this level (board before, action, board after): "
    "`MODEL.ledger()` lists it.\n"
    "- Register your model of the level: `MODEL.register(encode, step)`, with `encode(frame) -> state` (any hashable "
    "description of the board that you plan with: positions, tuples, frozensets; `frame` is like `current_frame`) and "
    "`step(state, action) -> state` (`action` as you pass it to `action(...)`: a name or a MOUSE dict). If you "
    "already have such functions, register them.\n"
    "- After every turn the harness replays the whole ledger through them and tells you how many transitions they "
    "reproduce and where it goes wrong (the earliest and the latest case). A transition where the board changed but "
    "your state did not counts as missed.\n"
    "- `MODEL.solve(goal, actions=None, max_depth=60)` searches breadth-first over your `step` from the current board "
    "for the shortest action list with `goal(state) == True`. `MODEL.frontier()` returns the shortest action list to "
    "a state that has never occurred: use it when the goal is unknown or the model is unproven.\n"
    "- While a registered model reproduces at least {fit}% of the ledger, `action([...])` runs lists of any length, "
    "checks every step against the model and stops at the first step that does not match. That mismatch is new "
    "evidence: fix the model before planning again. Without such a model an action list is cut after {cap} actions."
)
REMINDER = (
    "[Model check, by the harness] Level {level} is in model mode (the earlier notice left the context): "
    "`MODEL.register(encode, step)` with `encode(frame) -> state` and `step(state, action) -> state`; `MODEL.check()` "
    "(fit on the ledger), `MODEL.ledger()`, `MODEL.solve(goal, actions=None, max_depth=60)`, `MODEL.frontier()`. "
    "While a registered model reproduces at least {fit}% of the ledger, action lists run in full and stop at the "
    "first step it did not predict; without one they are cut after {cap} actions."
)
CAP_DETAIL = (
    "Model check: {requested} actions were requested and the first {cap} were executed. On this level a longer list "
    "needs a registered model that reproduces the ledger (`MODEL.register(encode, step)`; `MODEL.check()` shows the "
    "fit). The remaining {skipped} action(s) were NOT executed."
)
MISMATCH_DETAIL = (
    "Model check: action {index} of {requested} ({action}) did not do what your registered model predicted. "
    "Predicted state{where}: {predicted}. Actual state{where}: {actual}. The remaining {skipped} action(s) were NOT "
    "executed. The "
    "transition is in the ledger: fix `step`/`encode` so that `MODEL.check()` reproduces it, then plan again."
)

CHILD_BLOCK = r'''
    import time as _sc_time

    _SC = {"mode": None, "table": None, "model": None, "cache": {}, "events": [], "registered": 0, "solves": 0,
           "frontiers": 0, "checked_lists": 0, "fit_ok": False, "ledgers": {}, "inside": 0,
           "carry": False, "build": None, "build_open": None,  # build: the NUMBER of the level whose model is owed
           "win": None, "acts": None, "wins": {},
           "plan": None, "plan_key": None, "plans": 0, "plan_runs": 0, "left": None}
    _SC_CAP, _SC_FIT, _SC_MIN, _SC_SECONDS, _SC_BORDER, _SC_CHARS = __SC_CONSTANTS__
    _SC_CARRY_FIT, _SC_PLAN, _SC_HELD_DETAIL = __SC_CARRY__

    class ModelFirstActionError(Exception):
        """Raised INSTEAD of executing an action while the model of the level just cleared is still owed."""
    _SC_CAP_DETAIL = __SC_CAP_DETAIL__
    _SC_MISMATCH_DETAIL = __SC_MISMATCH_DETAIL__

    def _sc_action_obj(text):
        """The ledger's action label as the agent passes it to action(): a name, or a MOUSE dict."""
        text = str(text).strip()
        if text.upper().startswith("MOUSE(") and text.endswith(")"):
            fields = {}
            for part in text[6:-1].split(","):
                key, _, value = part.partition("=")
                try:
                    fields[key.strip().lower()] = int(value)
                except ValueError:
                    pass
            return {"action": "MOUSE", "row": fields.get("row"), "col": fields.get("col")}
        return text.upper()

    def _sc_label(item):
        if isinstance(item, dict):
            name = str(item.get("action", "")).strip().upper()
            if name == "MOUSE":
                return "MOUSE(row=%s, col=%s)" % (item.get("row"), item.get("col"))
            return name
        return str(item).strip().upper()

    def _sc_items(actions):
        if isinstance(actions, (str, dict)):
            return [actions]
        try:
            return list(actions)
        except TypeError:
            return []

    def _sc_board(frame):
        """The board inside the outer ring as bytes (kept on the frame: frames persist across calls)."""
        board = getattr(frame, "_sc_board", None)
        if board is None:
            grid, b = frame._grid, _SC_BORDER
            board = bytes(bytearray(int(v) & 255 for row in grid[b:len(grid) - b] for v in row[b:len(row) - b]))
            try:
                frame._sc_board = board
            except Exception:
                pass
        return board

    def _sc_level():
        return int(getattr(runtime_globals.get("current_frame"), "level", 0) or 0)

    def _sc_ledger(level=None):
        """One level's transitions (default: the current level; RESETs left out): dicts with step, action, label,
        before, after, board (the before board inside the outer ring), changed (that board changed), won (the action
        completed the level), repeat (same board and action seen before) and conflict (... with a different board
        after)."""
        transitions = runtime_globals.get("transitions") or []
        level = _sc_level() if level is None else int(level)
        key = (len(transitions), getattr(runtime_globals.get("current_frame"), "step", None))
        cached = _SC["ledgers"].get(level)
        if cached is not None and cached[0] == key:
            return cached[1]
        rows, seen = [], {}
        for t in transitions:
            label = str(t.action).strip()
            before, after = t.before_frame, t.after_frame
            if label.upper() == "RESET" or before is None or after is None or int(before.level) != level:
                continue
            won = int(after.level) > level
            board, following = _sc_board(before), _sc_board(after)
            row = {"step": after.step, "action": _sc_action_obj(label), "label": _sc_label(_sc_action_obj(label)),
                   "before": before, "after": after, "won": won, "board": board, "changed": won or board != following}
            first = seen.setdefault((board, row["label"]), row)
            row["repeat"] = first is not row
            row["conflict"] = (first is not row and not won and not first["won"]
                               and _sc_board(first["after"]) != following)
            rows.append(row)
        for old in [k for k in _SC["ledgers"] if k < level - 1]:
            del _SC["ledgers"][old]
        _SC["ledgers"][level] = (key, rows)
        return rows

    def _sc_short(value):
        try:
            text = repr(value)
        except Exception as exc:
            text = "<repr failed: %s>" % type(exc).__name__
        return text if len(text) <= _SC_CHARS else text[:_SC_CHARS - 12] + " [truncated]"

    def _sc_diff(predicted, actual, where="", depth=0):
        """Where two states part: (path into the state, predicted there, actual there). Sequences and dicts are
        walked to the first differing item; sets show the items only one side has; long strings the differing part."""
        if depth < 6 and type(predicted) is type(actual):
            if isinstance(predicted, (tuple, list)) and len(predicted) == len(actual):
                for index, (left, right) in enumerate(zip(predicted, actual)):
                    if left != right:
                        return _sc_diff(left, right, where + "[%d]" % index, depth + 1)
            elif isinstance(predicted, dict) and set(predicted) == set(actual):
                for key in predicted:
                    if predicted[key] != actual[key]:
                        return _sc_diff(predicted[key], actual[key], where + "[%s]" % _sc_short(key), depth + 1)
            elif isinstance(predicted, (set, frozenset)):
                try:
                    return (where + " (items only one side has)", _sc_short(sorted(predicted - actual)),
                            _sc_short(sorted(actual - predicted)))
                except Exception:
                    pass
            elif isinstance(predicted, str) and len(predicted) == len(actual) and len(predicted) > 40:
                first = next(i for i, (left, right) in enumerate(zip(predicted, actual)) if left != right)
                last = max(i for i, (left, right) in enumerate(zip(predicted, actual)) if left != right)
                last = min(last + 1, first + 60)
                return (where + "[%d:%d]" % (first, last), repr(predicted[first:last]), repr(actual[first:last]))
        return (where, _sc_short(predicted), _sc_short(actual))

    def _sc_evidence(rows):
        """Counts from the ledger alone. Per action name [tries, changed, no-op, boards tried on]; ``single``: names
        that never changed anything and were tried on one board only; clicks also per cell; valid actions never
        tried; same board + action that gave different results."""
        table, boards, cells = {}, {}, {}
        for row in rows:
            click = isinstance(row["action"], dict)
            name = "MOUSE" if click else row["action"]
            entry = table.setdefault(name, [0, 0, 0])
            entry[0] += 1
            entry[1 if row["changed"] else 2] += 1
            boards.setdefault(name, set()).add(row["board"])
            if click:
                cell = cells.setdefault((row["action"].get("row"), row["action"].get("col")), [0, set()])
                cell[0] += 1 if row["changed"] else 0
                cell[1].add(row["board"])
        valid = [str(a).upper() for a in (runtime_globals.get("valid_actions") or [])]
        return {"transitions": len(rows), "wins": sum(1 for r in rows if r["won"]),
                "actions": {name: table[name] + [len(boards[name])] for name in sorted(table)},
                "single": sorted(n for n, e in table.items() if n != "MOUSE" and e[1] == 0 and len(boards[n]) == 1),
                "cells": len(cells),
                "single_cells": sum(1 for changed, seen in cells.values() if not changed and len(seen) == 1),
                "untried": [a for a in valid if a not in table and a not in ("RESET", "UNDO")],
                "conflicts": [r["step"] for r in rows if r["conflict"]][:4]}

    def _sc_win(state, expected):
        """The registered win rule on one state: "hit" / "miss" for the state after the winning action, "false" when
        it holds on another recorded board, None when it is right there or no rule is registered."""
        if _SC["win"] is None:
            return None
        try:
            holds = bool(_SC["win"](state))
        except Exception:
            return "error"
        if expected:
            return "hit" if holds else "miss"
        return "false" if holds else None

    def _sc_verdict(row, encode, step, key=None):
        _SC["inside"] += 1  # the model's functions run: a real action() inside them is refused
        try:
            before = encode(row["before"])
            if row["won"]:
                # A winning action must not break the model; the win rule must hold on the state it predicts.
                _SC["wins"][key] = _sc_win(step(before, row["action"]), True)
                return ("won",)
            actual = encode(row["after"])
            _SC["wins"][key] = _sc_win(actual, False)
            same = before == actual
            predicted = step(before, row["action"])
        except Exception as exc:
            return ("wrong", "%s: %s" % (type(exc).__name__, str(exc)[:100]), "", "")
        finally:
            _SC["inside"] -= 1
        if predicted == actual:
            return ("missed",) if row["changed"] and same else ("exact",)
        where, left, right = _sc_diff(predicted, actual)
        return ("wrong", left, right, where)

    def _sc_fit(level=None):
        """Replay one level's ledger (default: the current level) through the registered model (cached per
        transition; ``_SC_SECONDS`` per call). ``first``: the earliest and the latest transition the model gets wrong
        or misses. ``win``: the registered win rule on the same record."""
        encode, step = _SC["model"]
        current = _sc_level()
        level = current if level is None else int(level)
        rows, cache, begin = _sc_ledger(level), _SC["cache"], _sc_time.monotonic()
        counts = {"exact": 0, "wrong": 0, "missed": 0, "won": 0}
        wins = {"hit": 0, "miss": 0, "false": 0, "error": 0}
        first, checked, pending = [], 0, 0
        for row in rows:
            key = (level, row["step"])
            verdict = cache.get(key)
            if verdict is None:
                if _sc_time.monotonic() - begin > _SC_SECONDS:
                    pending += 1
                    continue
                verdict = cache[key] = _sc_verdict(row, encode, step, key)
            checked += 1
            counts[verdict[0]] += 1
            if _SC["wins"].get(key):
                wins[_SC["wins"][key]] += 1
            if verdict[0] == "wrong":
                first.append({"step": row["step"], "action": row["label"], "where": verdict[3],
                              "predicted": verdict[1], "actual": verdict[2]})
            elif verdict[0] == "missed":
                first.append({"step": row["step"], "action": row["label"], "missed": True})
        first = first[:1] + first[-1:] if len(first) > 2 else first  # the earliest and the latest
        scored = counts["exact"] + counts["wrong"] + counts["missed"]
        share = counts["exact"] / float(scored) if scored else None
        if level == current:
            _SC["fit_ok"] = bool(scored < _SC_MIN or share >= _SC_FIT) if not pending else _SC["fit_ok"]
            ok = _SC["fit_ok"]
        else:
            ok = bool(not pending and (scored < _SC_MIN or share >= _SC_FIT))
        return {"checked": checked, "scored": scored, "pending": pending, "share": share, "ok": ok, "level": level,
                "carried": bool(not pending and (scored < _SC_MIN or share >= _SC_CARRY_FIT)),
                "win": dict(wins) if _SC["win"] is not None else None, "first": first, **counts}

    def _sc_plan():
        """simcarry: the harness's own search of a fitting model with a registered win rule, from the current board
        (once per board and registration). ``MODEL.plan`` holds the action list."""
        if not _SC["carry"] or _SC["model"] is None or _SC["win"] is None or not _SC["fit_ok"]:
            _SC["plan"] = None
            return None
        frame = runtime_globals.get("current_frame")
        key = (_sc_level(), getattr(frame, "step", None), _SC["registered"])
        if _SC["plan_key"] == key:
            return _SC["plan"]
        depth, nodes, seconds = _SC_PLAN
        try:
            _SC["inside"] += 1
            try:
                start = _SC["model"][0](frame)
            finally:
                _SC["inside"] -= 1
            found = _sc_search(start, _SC["win"], _SC["acts"], depth, nodes, seconds)
        except Exception as exc:
            found = {"path": None, "nodes": 0, "reason": "%s: %s" % (type(exc).__name__, str(exc)[:100])}
        path = found.get("path")
        _SC["plans"] += 1
        _SC["plan_key"] = key
        _SC["plan"] = {"length": len(path) if path is not None else None, "nodes": found.get("nodes"),
                       "reason": found.get("reason"), "path": [_sc_label(i) for i in path] if path else path,
                       "items": list(path) if path else path}
        _SC["events"].append({"kind": "plan", "level": _sc_level(), "found": bool(path),
                              "length": len(path) if path is not None else None, "nodes": found.get("nodes")})
        return _SC["plan"]

    def _sc_search(start, goal, actions, max_depth, max_nodes, seconds, known=None):
        """Breadth-first over the registered ``step``: the shortest action list to ``goal(state)`` (or, with ``known``,
        to a state outside it)."""
        _SC["inside"] += 1  # a real action() inside step / goal / actions is refused
        try:
            return _sc_search_inner(start, goal, actions, max_depth, max_nodes, seconds, known)
        finally:
            _SC["inside"] -= 1

    def _sc_search_inner(start, goal, actions, max_depth, max_nodes, seconds, known):
        from collections import deque as _deque

        encode, step = _SC["model"]
        begin = _sc_time.monotonic()
        if actions is None:
            actions = [a for a in (runtime_globals.get("valid_actions") or []) if a not in ("MOUSE", "RESET")]
        options = actions if callable(actions) else (lambda state, _fixed=list(actions): _fixed)

        def key_of(state):
            try:
                hash(state)
                return state
            except TypeError:
                return repr(state)

        def done(state):
            return key_of(state) not in known if known is not None else bool(goal(state))

        if known is None and done(start):
            return {"path": [], "nodes": 1, "reason": "the start state already satisfies the goal"}
        previous, queue, nodes = {key_of(start): None}, _deque([(start, 0)]), 0
        while queue:
            state, depth = queue.popleft()
            if depth >= int(max_depth):
                continue
            for item in options(state):
                nodes += 1
                if nodes > int(max_nodes) or _sc_time.monotonic() - begin > float(seconds):
                    return {"path": None, "nodes": nodes, "reason": "search budget exhausted (raise max_nodes/seconds "
                            "or use a smaller state)"}
                try:
                    following = step(state, item)
                except Exception as exc:
                    return {"path": None, "nodes": nodes, "reason": "step failed: %s: %s" % (type(exc).__name__, exc)}
                key = key_of(following)
                if key in previous:
                    continue
                previous[key] = (key_of(state), item)
                if done(following):
                    path, cursor = [], key
                    while previous[cursor] is not None:
                        cursor, item = previous[cursor]
                        path.append(item)
                    return {"path": path[::-1], "nodes": nodes, "reason": "found"}
                queue.append((following, depth + 1))
        return {"path": None, "nodes": nodes, "reason": "no path within max_depth %d (%d states reached)"
                % (int(max_depth), len(previous))}

    class _SCModel(object):
        """The harness's model check (see the notice in your user turn)."""

        def register(self, encode, step, win=None, actions=None):
            """Register ``encode(frame) -> state`` and ``step(state, action) -> state``; optionally ``win(state) ->
            bool`` and ``actions`` (a list, or ``actions(state) -> list``). Returns the ledger fit."""
            if not callable(encode) or not callable(step):
                raise TypeError("MODEL.register(encode, step) takes two functions")
            if win is not None and not callable(win):
                raise TypeError("win must be a function win(state) -> bool")
            _SC["model"], _SC["cache"], _SC["fit_ok"] = (encode, step), {}, False
            _SC["win"], _SC["acts"], _SC["wins"], _SC["plan_key"] = win, actions, {}, None
            _SC["registered"] += 1
            report = _sc_fit()
            event = {"kind": "register", "level": _sc_level(), "scored": report["scored"], "exact": report["exact"],
                     "win": win is not None}
            summary = self._summary(report)
            if _SC["build"] is not None:
                finished = _sc_fit(_SC["build"])
                event.update(finished_level=_SC["build"], finished_scored=finished["scored"],
                             finished_exact=finished["exact"])
                summary = {"finished_level": self._summary(finished), "current_level": summary}
            _SC["events"].append(event)
            return summary

        def _summary(self, report):
            out = {"transitions": report["scored"], "reproduced": report["exact"], "wrong": report["wrong"],
                   "missed_changes": report["missed"], "not_checked_yet": report["pending"],
                   "first_mismatches": report["first"], "lists_unlimited": report["ok"]}
            if report.get("win") is not None:
                win = report["win"]
                out["win_rule"] = {"true_after_the_winning_action": win["hit"], "false_after_it": win["miss"],
                                   "true_on_other_recorded_boards": win["false"], "errors": win["error"]}
            return out

        def check(self, level=None):
            """The registered model replayed on the level's ledger (``level``: a level number; default the current)."""
            if _SC["model"] is None:
                return {"error": "no model registered: MODEL.register(encode, step)"}
            return self._summary(_sc_fit(level))

        def ledger(self, last=None, level=None):
            """A level's recorded transitions (``level``: a level number; default the current): dicts with step,
            action, before, after (frames), changed, won."""
            rows = [{k: r[k] for k in ("step", "action", "before", "after", "changed", "won")}
                    for r in _sc_ledger(level)]
            return rows[-int(last):] if last else rows

        @property
        def plan(self):
            """simcarry: the action list of the harness's last search (None without one)."""
            found = _SC["plan"]
            return list(found["items"]) if found and found.get("items") else None

        def solve(self, goal=None, actions=None, max_depth=60, max_nodes=200000, seconds=10, start=None):
            """Breadth-first search over your ``step`` from the current board (or ``start``, a state) for the shortest
            action list with ``goal(state) == True`` (default: the registered ``win``). ``actions``: a list, or
            ``actions(state) -> list``; default: the registered ones, else the valid non-MOUSE actions. Returns
            {'path', 'nodes', 'reason'}."""
            if _SC["model"] is None:
                return {"path": None, "nodes": 0, "reason": "no model registered: MODEL.register(encode, step)"}
            goal = _SC["win"] if goal is None else goal
            actions = _SC["acts"] if actions is None else actions
            if goal is None:
                return {"path": None, "nodes": 0, "reason": "no goal given and no win rule registered"}
            _SC["solves"] += 1
            begin = _SC["model"][0](runtime_globals.get("current_frame")) if start is None else start
            result = _sc_search(begin, goal, actions, max_depth, max_nodes, seconds)
            _SC["events"].append({"kind": "solve", "level": _sc_level(), "nodes": result["nodes"],
                                  "found": result["path"] is not None,
                                  "length": len(result["path"]) if result["path"] is not None else None})
            return result

        def frontier(self, actions=None, max_depth=40, max_nodes=200000, seconds=10):
            """The shortest action list your model predicts leads to a state that never occurred on this level."""
            if _SC["model"] is None:
                return {"path": None, "nodes": 0, "reason": "no model registered: MODEL.register(encode, step)"}
            _SC["frontiers"] += 1
            actions = _SC["acts"] if actions is None else actions
            encode = _SC["model"][0]
            known = set()
            for row in _sc_ledger():
                for frame in (row["before"], row["after"]):
                    try:
                        state = encode(frame)
                        hash(state)
                    except TypeError:
                        state = repr(state)
                    except Exception:
                        continue
                    known.add(state)
            begin = encode(runtime_globals.get("current_frame"))
            result = _sc_search(begin, None, actions, max_depth, max_nodes, seconds, known=known)
            _SC["events"].append({"kind": "frontier", "level": _sc_level(), "nodes": result["nodes"],
                                  "found": result["path"] is not None})
            return result

    _sc_model_object = _SCModel()
    _sc_inner_action = runtime_globals["action"]

    def action(actions):
        if _SC["inside"]:
            raise RuntimeError("action() was called inside encode/step/goal: a model predicts, it must not act")
        build = _SC["build"]
        if build is not None and _SC["build_open"] != build:
            carried = False
            if _SC["model"] is not None:
                try:
                    carried = _sc_fit(build)["carried"]
                except Exception:
                    carried = False
            if not carried:
                _SC["events"].append({"kind": "held", "level": build})
                raise ModelFirstActionError(_SC_HELD_DETAIL.format(level=build, left=max(int(_SC["left"] or 0), 0)))
            _SC["build_open"] = build
            _SC["events"].append({"kind": "build_open", "level": build})
        if not _SC["mode"]:
            return _sc_inner_action(actions)
        items = _sc_items(actions)
        if _SC["plan"] and _SC["plan"].get("path") and [_sc_label(i) for i in items] == _SC["plan"]["path"]:
            _SC["plan_runs"] += 1
            _SC["events"].append({"kind": "plan_run", "level": _sc_level(), "length": len(items)})
        if _SC["model"] is None or not _SC["fit_ok"]:
            if _SC["carry"] or len(items) <= _SC_CAP:
                return _sc_inner_action(actions)
            result = _sc_inner_action(items[:_SC_CAP])
            _SC["events"].append({"kind": "cap", "level": _sc_level(), "requested": len(items)})
            if isinstance(result, dict) and not result.get("stopped_early"):
                result = dict(result)
                result.update(requested_count=len(items), stopped_early=True, stop_reason="model_required",
                              stop_detail=_SC_CAP_DETAIL.format(requested=len(items), cap=_SC_CAP,
                                                                skipped=len(items) - _SC_CAP))
            return result
        if len(items) <= 1:
            return _sc_inner_action(actions)
        encode, step = _SC["model"]
        _SC["checked_lists"] += 1
        executed, result, mismatch = [], None, None
        for index, item in enumerate(items):
            _SC["inside"] += 1
            try:
                predicted, failed = step(encode(runtime_globals.get("current_frame")), item), None
            except Exception as exc:
                predicted, failed = None, "%s: %s" % (type(exc).__name__, str(exc)[:100])
            finally:
                _SC["inside"] -= 1
            result = _sc_inner_action([item])
            if not isinstance(result, dict) or not result.get("executed"):
                break
            executed.append(_sc_label(item))
            if any(result.get(flag) for flag in ("level_completed", "game_over", "done", "run_complete")):
                break
            try:
                actual = encode(runtime_globals.get("current_frame"))
            except Exception as exc:
                actual, failed = None, failed or "%s: %s" % (type(exc).__name__, str(exc)[:100])
            if failed:
                mismatch = {"index": index + 1, "action": _sc_label(item), "where": "", "predicted": failed,
                            "actual": _sc_short(actual)}
                break
            if predicted != actual:
                where, left, right = _sc_diff(predicted, actual)
                mismatch = {"index": index + 1, "action": _sc_label(item), "where": where, "predicted": left,
                            "actual": right}
                break
            if result.get("gameplay_changed") is False:
                break  # a no-op ends the list, predicted or not (the stale-state guard would refuse the next action)
        if isinstance(result, dict):
            result = dict(result)
            result.update(executed_actions=executed, requested_count=len(items), executed_count=len(executed),
                          stopped_early=len(executed) < len(items) or bool(mismatch))
            if mismatch:
                _SC["fit_ok"] = False
                skipped = len(items) - len(executed)
                result.update(stop_reason="model_mismatch", model_mismatch=mismatch,
                              stop_detail=_SC_MISMATCH_DETAIL.format(requested=len(items), skipped=skipped, **mismatch))
                _SC["events"].append({"kind": "mismatch", "level": _sc_level(), "index": mismatch["index"],
                                      "requested": len(items)})
            else:
                _SC["events"].append({"kind": "checked", "level": _sc_level(), "executed": len(executed),
                                      "requested": len(items)})
        return result

    runtime_globals["action"] = action
    runtime_globals["ModelFirstActionError"] = ModelFirstActionError
    _sc_refresh_inner = _refresh_state

    def _refresh_state(state_payload):
        _sc_refresh_inner(state_payload)
        mode = state_payload.get("simcheck") if isinstance(state_payload, dict) else None
        if isinstance(mode, dict):
            before = _SC["mode"]
            _SC["mode"] = mode.get("level") if mode.get("mode") else None
            _SC["table"] = mode.get("level") if mode.get("table") else None
            _SC["carry"] = bool(mode.get("carry"))
            _SC["build"] = mode.get("build")
            _SC["left"] = mode.get("left")
            if _SC["mode"] and _SC["mode"] != before and _SC["model"] is not None:
                try:  # a model registered on an earlier level: its fit on this level's ledger decides the cap
                    _sc_fit()
                except Exception:
                    _SC["fit_ok"] = False
        if _SC["mode"] or _SC["carry"]:
            runtime_globals["MODEL"] = _sc_model_object

    def _sc_take():
        """The call's report for the host: events, and past the stuck signal the ledger counts and the fit."""
        out = {"mode": _SC["mode"], "table": _SC["table"], "events": [],
               "registered": _SC["registered"], "solves": _SC["solves"], "frontiers": _SC["frontiers"],
               "checked_lists": _SC["checked_lists"]}
        if _SC["carry"]:
            out.update(carry=True, win=_SC["win"] is not None, level=_sc_level(), build_open=_SC["build_open"])
            try:  # the level just left: what the hold and the carried-model record are read from
                if _SC["model"] is not None and _sc_level() > 1:
                    out["finished"] = _sc_fit(_sc_level() - 1)
                else:
                    out["finished"] = {"level": _sc_level() - 1, "scored": len(_sc_ledger(_sc_level() - 1)),
                                       "none": True} if _sc_level() > 1 else None
            except Exception as exc:
                out["finished"] = {"level": _sc_level() - 1, "error": "%s: %s" % (type(exc).__name__, str(exc)[:120])}
        if not _SC["mode"] and not _SC["table"]:
            out["events"] = list(_SC["events"])
            del _SC["events"][:]
            if _SC["carry"]:
                out.update(plans=_SC["plans"], plan_runs=_SC["plan_runs"])
            return out
        try:
            out["ledger"] = _sc_evidence(_sc_ledger())
            if _SC["mode"]:
                out["fit"] = _sc_fit() if _SC["model"] is not None else None
                if _SC["carry"]:
                    plan = _sc_plan()
                    out["plan"] = {k: v for k, v in plan.items() if k != "items"} if plan else None
        except Exception as exc:  # a broken model must never break the call
            out["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:120])
        out["events"] = list(_SC["events"])
        del _SC["events"][:]
        if _SC["carry"]:
            out.update(plans=_SC["plans"], plan_runs=_SC["plan_runs"])
        return out

'''
ANCHOR = "    request = initial\n    while True:\n"
RESULT_FIELD = '"action_results": _json_safe(action_results),'
RUNTIME_NAMES = '"last_action_result", "action", "result"'
HELD_DETAIL = (
    "Model first: no action was executed. Level {level} is cleared and its model is still owed: "
    "`MODEL.register(encode, step, win)` must reproduce the recorded transitions of level {level} (`MODEL.check("
    "level={level})` shows the fit and the first mismatches; `MODEL.ledger(level={level})` lists the record). Play "
    "continues as soon as it does, or when the modelling allowance is spent (about {left:,} generated tokens left "
    "at the start of this call)."
)


def with_simcheck(bootstrap):
    """``bootstrap`` with the model-check child code (after the dfr_harness code: its action wrapper wraps theirs, so
    the stale-state and terminal latches still apply to every single step) and the ``simcheck`` report field."""
    import json

    if "_SC = {" in bootstrap:
        raise ValueError("sim_check: the sandbox code is already installed")
    if bootstrap.count(ANCHOR) != 1 or bootstrap.count(RESULT_FIELD) != 2:
        raise ValueError("sim_check: pinned Duck sandbox loop changed")
    child = (CHILD_BLOCK.replace("__SC_CONSTANTS__", repr((CAP, FIT, MIN_LEDGER, SECONDS, BORDER, STATE_CHARS)))
             .replace("__SC_CARRY__", repr((CARRY_FIT, PLAN, HELD_DETAIL)))
             .replace("__SC_CAP_DETAIL__", json.dumps(CAP_DETAIL))
             .replace("__SC_MISMATCH_DETAIL__", json.dumps(MISMATCH_DETAIL)))
    text = bootstrap.replace(ANCHOR, child.lstrip("\n") + ANCHOR, 1)
    text = text.replace(RESULT_FIELD, RESULT_FIELD + ' "simcheck": _sc_take(),')
    if text.count(RUNTIME_NAMES) == 1:  # the fixes loop's inventory skips runtime names
        text = text.replace(RUNTIME_NAMES, RUNTIME_NAMES + ', "MODEL", "ModelFirstActionError"', 1)
    return text


def notice(level, tokens=None):
    """The contract at the stuck signal; without ``tokens`` the short form after a context drain or compaction."""
    if tokens is None:
        return REMINDER.format(level=level, fit=int(round(FIT * 100)), cap=CAP)
    return NOTICE.format(level=level, tokens=int(tokens), fit=int(round(FIT * 100)), cap=CAP)


CARRY_NOTICE = (
    "[Model first, by the harness] Level {level} is cleared. Before you play on, write what you learned as code that "
    "the harness can check, so that it is still there on the next levels.\n"
    "- `MODEL.register(encode, step, win, actions=None)`: `encode(frame) -> state` (any hashable description of the "
    "board that you plan with: positions, tuples, frozensets; `frame` is like `current_frame`), `step(state, action) "
    "-> state` (`action` as you pass it to `action(...)`: a name or a MOUSE dict), `win(state) -> bool` (True exactly "
    "when the level is completed) and, for games played with MOUSE, `actions(state) -> list` (the actions worth trying "
    "from a state). If you already have such functions, register them.\n"
    "- The harness replays every recorded transition of level {level} ({transitions} of them; `MODEL.ledger(level="
    "{level})` lists them) through your functions: `MODEL.check(level={level})` shows how many they reproduce and the "
    "first mismatches. `win` must be True after the action that completed the level and False on every other "
    "recorded board.\n"
    "- `action(...)` is held until the model reproduces at least {fit}% of level {level}, or until {allowance:,} "
    "generated tokens have gone into it; then play continues either way.\n"
    "- The model is carried. On the following levels the harness checks it against every action you take and shows "
    "the first mismatch; action lists run step by step under that check and stop at the first step it did not "
    "predict. When it fits and `win` is registered, the harness searches it for a winning action list and offers it "
    "(`MODEL.plan`); `MODEL.solve(goal=None, actions=None, max_depth=60)` and `MODEL.frontier()` search it for you. "
    "Write it in terms of objects and rules, not of level {level}'s coordinates, so that it still holds on the next "
    "level; extend it there when something new appears."
)
CARRY_NEXT = (
    "[Model first, by the harness] Level {level} is cleared. {status} `action(...)` is held until the registered "
    "model reproduces at least {fit}% of level {level}'s {transitions} recorded transitions (`MODEL.check(level="
    "{level})`), or until {allowance:,} generated tokens have gone into it."
)
CARRY_REMINDER = (
    "[Model check, by the harness] (the earlier notice left the context) `MODEL.register(encode, step, win, "
    "actions=None)` with `encode(frame) -> state`, `step(state, action) -> state`, `win(state) -> bool`; "
    "`MODEL.check(level=None)`, `MODEL.ledger(level=None)`, `MODEL.solve(goal=None, actions=None, max_depth=60)`, "
    "`MODEL.frontier()`, `MODEL.plan`. The harness checks the registered model against every action; lists run under "
    "the per-step check and stop at the first step it did not predict."
)


def _fit_text(fit, level, what="your registered model"):
    """"... reproduces X of Y transitions of level N (...)" plus the mismatch and win-rule parts."""
    text = f"{what} reproduces {fit['exact']} of {fit['scored']} transitions of level {level}"
    extra = [f"{fit['wrong']} wrong" if fit.get("wrong") else "", f"{fit['missed']} missed changes"
             if fit.get("missed") else "", f"{fit['pending']} not checked yet (time limit)" if fit.get("pending")
             else ""]
    extra = [e for e in extra if e]
    parts = [text + (f" ({', '.join(extra)})" if extra else "")]
    for item in fit.get("first") or []:
        if item.get("missed"):
            parts.append(f"missed: step {item['step']} {item['action']} changed the board but not your state")
        else:
            where = f" state{item['where']}:" if item.get("where") else ""
            parts.append(f"wrong: step {item['step']} {item['action']}:{where} predicted {item['predicted']}, "
                         f"actual {item['actual']}")
    win = fit.get("win")
    if win is not None:
        faults = []
        if win.get("miss"):
            faults.append("`win` is False after the action that completed the level")
        if win.get("false"):
            faults.append(f"`win` is True on {win['false']} recorded board(s) where the level was not completed")
        if win.get("error"):
            faults.append(f"`win` raised on {win['error']} board(s)")
        if faults:
            parts.append("win rule: " + "; ".join(faults))
        elif win.get("hit"):
            parts.append("win rule: holds on the record")
    return parts


def carry_notice(level, transitions, first, fit=None):
    """simcarry, at a clear whose model is still owed. ``first``: the full contract; later the short form."""
    if first:
        return CARRY_NOTICE.format(level=level, transitions=transitions, fit=int(round(CARRY_FIT * 100)),
                                   allowance=BUILD_TOKENS)
    status = ("No model is registered." if not fit or fit.get("none") or "exact" not in fit else
              ". ".join(_fit_text(fit, level, "Your registered model")) + ".")
    return CARRY_NEXT.format(level=level, transitions=transitions, status=status, fit=int(round(CARRY_FIT * 100)),
                             allowance=BUILD_TOKENS)


def carry_passed(level, fit):
    """simcarry, at a clear the registered model already reproduces: no hold."""
    return ("[Model check, by the harness] Level " + str(level) + " is cleared. " +
            ". ".join(_fit_text(fit, level, "Your registered model")) + ": carried to the next level.")


def hold_line(level, fit, left):
    """simcarry: the per-turn line while the finished level's model is owed."""
    if not fit or fit.get("none") or "exact" not in fit:
        body = f"no model registered for level {level} yet"
    else:
        body = ". ".join(_fit_text(fit, level))
    return (f"Model first: {body}. `action(...)` is held until it reproduces at least "
            f"{int(round(CARRY_FIT * 100))}% (about {max(int(left), 0):,} tokens of the allowance left).")


def hold_end_line(level, fit, reason):
    """simcarry: the line of the turn after the hold ended."""
    if reason == "fit":
        body = ". ".join(_fit_text(fit, level)) if fit and "exact" in fit else f"the model of level {level} fits"
        return ("Model first: " + body + ". Play continues; the model is carried and checked against every action "
                "from here.")
    return (f"Model first: the modelling allowance for level {level} is spent. Play continues. Whatever is "
            "registered stays and is checked against every action; fix it as the mismatches show.")


def carry_line(report, level):
    """simcarry: the per-turn line on a level played with a carried model (model check, win rule, offered plan)."""
    if not report or not report.get("mode"):
        return None
    if report.get("error"):
        return f"Model check: the harness could not run the check ({report['error']})."
    fit = report.get("fit")
    if fit is None:
        return (f"Model check: no model registered (`MODEL.register(encode, step, win)`); the ledger holds "
                f"{(report.get('ledger') or {}).get('transitions', 0)} transitions of level {level}.")
    parts = _fit_text(fit, level)
    plan = report.get("plan")
    if plan and plan.get("path"):
        shown = ", ".join(plan["path"][:PLAN_SHOWN]) + (" ..." if len(plan["path"]) > PLAN_SHOWN else "")
        parts.append(f"your model and win rule predict a win in {plan['length']} action(s) from this board: [{shown}]. "
                     "`action(MODEL.plan)` runs the list under the per-step check and stops at the first step the "
                     "model did not predict")
    elif plan:
        parts.append(f"the harness's search of your model found no winning action list ({plan.get('reason')})")
    elif fit.get("ok") and not report.get("win"):
        parts.append("no `win` registered, so the harness cannot search it for you")
    return "Model check: " + ". ".join(parts) + "."


def evidence_line(report, level):
    """The user-turn line of the ``evidence`` flag: what the level's ledger says per action (None without a ledger)."""
    ledger = (report or {}).get("ledger") or {}
    if not report or not report.get("table") or not ledger.get("transitions"):
        return None
    parts = []
    for name, (tries, changed, noop, boards) in (ledger.get("actions") or {}).items():
        text = f"{name} {changed}/{tries}"
        if name == "MOUSE":
            text += f" ({ledger.get('cells', 0)} cells clicked"
            if ledger.get("single_cells"):
                text += f"; {ledger['single_cells']} did nothing on the one board they were tried on"
            text += ")"
        elif name in (ledger.get("single") or []):
            text += " (tried on ONE board only: a single no-op is not a rule, try it on a different board)"
        parts.append(text)
    line = (f"Evidence (the harness's record of level {level}, {ledger['transitions']} actions; actions that changed "
            "the board / tries): " + ", ".join(parts) + ".")
    if ledger.get("untried"):
        line += " Never tried on this level: " + ", ".join(ledger["untried"]) + "."
    if ledger.get("conflicts"):
        line += (" The same board and action gave different results at step(s) "
                 + ", ".join(str(s) for s in ledger["conflicts"]) + ": something outside the board decides it.")
    return line


def check_line(report, level):
    """The user-turn line of model mode for the child's last report (None without one)."""
    if not report or not report.get("mode"):
        return None
    if report.get("error"):
        return f"Model check: the harness could not run the check ({report['error']})."
    ledger = report.get("ledger") or {}
    parts = []
    fit = report.get("fit")
    if fit is None:
        parts.append(f"Model check: no model registered for level {level} (action lists are cut after {CAP} actions)")
        if "transitions" in ledger:
            parts.append(f"The ledger holds {ledger['transitions']} transitions of this level")
    else:
        text = (f"Model check: your registered model reproduces {fit['exact']} of {fit['scored']} transitions of "
                f"level {level}")
        extra = [f"{fit['wrong']} wrong" if fit["wrong"] else "", f"{fit['missed']} missed changes" if fit["missed"]
                 else "", f"{fit['pending']} not checked yet (time limit)" if fit["pending"] else ""]
        extra = [e for e in extra if e]
        text += f" ({', '.join(extra)})" if extra else ""
        text += ("; action lists run in full under the per-step check" if fit["ok"] else
                 f"; below {int(round(FIT * 100))}%: action lists are cut after {CAP} actions")
        parts.append(text)
        for item in fit.get("first") or []:
            if item.get("missed"):
                parts.append(f"missed: step {item['step']} {item['action']} changed the board but not your state")
            else:
                where = f" state{item['where']}:" if item.get("where") else ""
                parts.append(f"wrong: step {item['step']} {item['action']}:{where} predicted {item['predicted']}, "
                             f"actual {item['actual']}")
    return ". ".join(parts) + "."


def new_stats():
    """``levels``: one row per level that reached the stuck signal (tokens at the signal, tokens generated after it,
    of those the tokens before a fitting model existed, whether one ever did, whether the level was cleared)."""
    return {"mode_levels": 0, "mode_tokens": 0, "unfit_tokens": 0, "registered": 0, "solves": 0, "frontiers": 0,
            "checked_lists": 0, "caps": 0, "mismatches": 0, "notice_chars": 0, "line_chars": 0, "table_chars": 0,
            "fit_last": None, "levels": [],
            # simcarry: one row per clear (the finished level's model) and totals of the model-first holds
            "carry": {"clears": 0, "carried_at_clear": 0, "holds": 0, "hold_tokens": 0, "hold_turns": 0,
                      "hold_fit": 0, "hold_allowance": 0, "held_actions": 0, "plans": 0, "plans_found": 0,
                      "plan_runs": 0, "win_registered": 0, "rows": []}}
