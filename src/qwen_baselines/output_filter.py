"""Keep whole-board dumps out of the conversation.

Duck caps tool output near 1K tokens and never prints frames; C printed grids and B
dumped arrays in v3. Runs of grid-like rows (64 symbols, or 48+ single-symbol tokens as
in numpy/list reprs) are replaced by one notice, and very long outputs are truncated.
The archive keeps the full text.
"""

import re

ROW_TOKENS = re.compile(r"[\s,]+")
STRIP = str.maketrans("", "", "[]()'\"")


def is_grid_row(line, *, min_symbols=48):
    text = line.translate(STRIP).strip()
    if not text:
        return False
    if " " not in text and "," not in text:
        return len(text) >= min_symbols and len(set(text)) <= 16 and text.isalnum()
    tokens = [t for t in ROW_TOKENS.split(text) if t]
    return len(tokens) >= min_symbols and all(len(t) <= 2 and t.isalnum() for t in tokens)


def suppress_dump(text, *, min_rows=24, max_chars=6000, keep_chars=4000):
    """``(filtered_text, info)``; ``info`` is None when nothing was changed."""
    lines = text.split("\n")
    kept, suppressed, run = [], 0, []

    def flush():
        nonlocal suppressed
        if len(run) >= min_rows:
            suppressed += len(run)
            kept.append(f"[grid dump suppressed: {len(run)} rows; read small crops instead]")
        else:
            kept.extend(run)
        run.clear()

    for line in lines:
        if is_grid_row(line):
            run.append(line)
        else:
            flush()
            kept.append(line)
    flush()
    result = "\n".join(kept)
    info = {"suppressed_rows": suppressed} if suppressed else None
    if len(result) > max_chars:
        total = len(result)
        result = (
            result[:keep_chars]
            + f"\n[output truncated: {total} chars; keep tool output short and read "
            "small crops instead]"
        )
        info = {**(info or {}), "truncated_chars": total}
    return result, info
