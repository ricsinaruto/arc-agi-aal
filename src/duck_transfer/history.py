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
    return bootstrap
