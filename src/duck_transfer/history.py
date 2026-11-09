"""Retain per-action outcomes through Duck's original observation serialization."""


def preserve_transition_results(bootstrap):
    replacements = {
        "    return items\n\n\ndef _transitions_from_history":
            "        items[-1].result = dict(entry.get('result') or {})\n"
            "    return items\n\n\ndef _transitions_from_history",
        "                result={},":
            "                result=dict(getattr(entry, 'result', {})),",
        "        transitions[-1].result = dict(last_action_result)":
            "        if not transitions[-1].result:\n"
            "            transitions[-1].result = dict(last_action_result)",
    }
    for old, new in replacements.items():
        if bootstrap.count(old) != 1:
            raise ValueError("Pinned Duck transition serializer changed")
        bootstrap = bootstrap.replace(old, new, 1)
    return incremental_transitions(bootstrap)


# Tool latency (throughput arm, 13 Oct): ``_refresh_state`` rebuilt ``transitions`` from the whole history after every
# action and at every call (one TransitionView and a dict copy per entry): 2 ms per action at 100 entries, 95 ms plus
# GC pauses at 20K. The incremental builder keeps the views of the retained history and builds only new entries. What
# the model sees is unchanged: a new list per refresh, views equal to the rebuilt ones, and a view or a result the
# model changes (attribute set or deleted, result mutated) is rebuilt at the next refresh, as every view was before.
# A changed history entry (attribute set on it) rebuilds all views. One difference is left: an in-place mutation of a
# history entry's ``result`` dict is no longer copied into the existing views.
INCREMENTAL_CALL = "transitions = _transitions_from_history(history, action_result)"
INCREMENTAL_ANCHOR = "\n\n\ndef _json_safe(value):"
INCREMENTAL_CODE = '''


_TV_CACHE = {"first": None, "last": None, "n": 0, "items": [], "src": [], "hist": {}}
_TV_TRACK = {}     # id(cached view) -> its index in the cache
_TV_DIRTY = set()  # cache indices to rebuild; -1: a history entry changed, rebuild all


class _TrackedResult(dict):
    """A cached view's ``result``: a mutation marks the view for a rebuild at the next refresh."""

    __slots__ = ("_tv_index",)

    def _tv_mark(self):
        _TV_DIRTY.add(getattr(self, "_tv_index", -1))

    def __setitem__(self, key, value):
        self._tv_mark()
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        self._tv_mark()
        dict.__delitem__(self, key)

    def __ior__(self, other):
        self._tv_mark()
        return dict.__ior__(self, other)

    def pop(self, *args):
        self._tv_mark()
        return dict.pop(self, *args)

    def popitem(self):
        self._tv_mark()
        return dict.popitem(self)

    def clear(self):
        self._tv_mark()
        dict.clear(self)

    def update(self, *args, **kwargs):
        self._tv_mark()
        dict.update(self, *args, **kwargs)

    def setdefault(self, key, default=None):
        self._tv_mark()
        return dict.setdefault(self, key, default)


def _tv_setattr(self, name, value):
    index = _TV_TRACK.get(id(self))
    if index is not None:
        _TV_DIRTY.add(index)
    object.__setattr__(self, name, value)


def _tv_delattr(self, name):
    index = _TV_TRACK.get(id(self))
    if index is not None:
        _TV_DIRTY.add(index)
    object.__delattr__(self, name)


def _he_setattr(self, name, value):
    if id(self) in _TV_CACHE["hist"]:
        _TV_DIRTY.add(-1)
    object.__setattr__(self, name, value)


def _he_delattr(self, name):
    if id(self) in _TV_CACHE["hist"]:
        _TV_DIRTY.add(-1)
    object.__delattr__(self, name)


TransitionView.__setattr__, TransitionView.__delattr__ = _tv_setattr, _tv_delattr
HistoryEntryView.__setattr__, HistoryEntryView.__delattr__ = _he_setattr, _he_delattr


def _tv_build(index, history, position):
    entry = history[position]
    view = TransitionView(action=str(getattr(entry, "action", "") or "").strip(),
                          before_frame=history[position - 1].frame if position > 0 else None,
                          after_frame=entry.frame, result={})
    result = _TrackedResult(dict(getattr(entry, "result", {})))
    result._tv_index = index
    object.__setattr__(view, "result", result)
    _TV_TRACK[id(view)] = index
    return view


def _tv_reset(history):
    cache = _TV_CACHE
    cache.update(first=history[0] if history else None, last=None, n=0, items=[], src=[], hist={})
    _TV_TRACK.clear()
    _TV_DIRTY.clear()


def _transitions_incremental(history, last_action_result):
    """``_transitions_from_history`` with the views of the retained history kept between refreshes."""
    cache = _TV_CACHE
    count = len(history)
    if (not history or cache["first"] is not history[0] or cache["n"] > count or -1 in _TV_DIRTY
            or (cache["n"] and history[cache["n"] - 1] is not cache["last"])):
        _tv_reset(history)
    items, src = cache["items"], cache["src"]
    for index in sorted(_TV_DIRTY):
        if 0 <= index < len(items):
            _TV_TRACK.pop(id(items[index]), None)
            items[index] = _tv_build(index, history, src[index])
    _TV_DIRTY.clear()
    for position in range(cache["n"], count):
        entry = history[position]
        cache["hist"][id(entry)] = position
        if str(getattr(entry, "action", "") or "").strip():
            src.append(position)
            items.append(_tv_build(len(items), history, position))
    cache["n"], cache["last"] = count, history[-1] if history else None
    transitions = list(items)
    if transitions and isinstance(last_action_result, dict) and not transitions[-1].result:
        last = transitions[-1]  # a fresh view, as before: the override does not stick to the cached one
        transitions[-1] = TransitionView(action=last.action, before_frame=last.before_frame,
                                         after_frame=last.after_frame, result=dict(last_action_result))
    return transitions'''


def incremental_transitions(bootstrap):
    """``bootstrap`` (with the transition-result patch) whose ``_refresh_state`` builds ``transitions`` incrementally."""
    if bootstrap.count(INCREMENTAL_CALL) != 1 or bootstrap.count(INCREMENTAL_ANCHOR) != 1:
        raise ValueError("Pinned Duck transition builder changed")
    bootstrap = bootstrap.replace(INCREMENTAL_ANCHOR, INCREMENTAL_CODE + INCREMENTAL_ANCHOR, 1)
    return bootstrap.replace(INCREMENTAL_CALL,
                             "transitions = _transitions_incremental(history, action_result)", 1)
