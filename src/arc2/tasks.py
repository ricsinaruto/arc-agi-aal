"""ARC tasks as text: prompt, exact comparison and the mismatch feedback. Pure functions."""

import base64
import json
import os
import struct
import zlib
from pathlib import Path

SYSTEM = """You solve ARC puzzles by writing a Python program.

A puzzle has training pairs (input grid -> output grid) that all follow one transformation rule, and test inputs. A \
grid is a rectangle of digits 0-9 (colours), at most 30x30. Rows are lines; cells are separated by spaces.

Work out the rule from the training pairs, then give one Python code block that defines `transform(grid)`. It \
receives a grid as a list of lists of ints and returns the output grid as a list of lists of ints. Write the general \
rule: it has to hold for every training pair and for the test inputs. Never look up the examples.

Your program is run on every training input and compared with the expected output cell by cell. When a pair fails \
you get the exact differences and can send a corrected program.

- numpy (`np`) and scipy can be imported; there is no file or network access.
- A run on one grid has a few seconds.
- What your program prints is shown back to you (first 1,500 characters); use it to test a hypothesis.
- End every reply with the full program in one ```python block."""

HELPERS = """- Helpers preloaded in the tool and in your program (no import needed; grids may be lists or arrays, grids \
returned are numpy int arrays):
  `background(g)` most common colour; `colours(g)` {colour: count}; `bbox(g, bg=None)` (r0, c0, r1, c1) inclusive \
of the non-bg cells, or None; `crop(g, box)`;
  `objects(g, bg=None, diag=False, multicolour=False)` connected components as dicts with `colour` (most common), \
`cells` [(r, c)], `size`, `box` (r0, c0, r1, c1), `grid` (cropped, bg elsewhere), `mask` (cropped bool);
  `panels(g)` sub-grids between full rows/columns of one colour (separators), as (r0, c0, grid) row by row; \
`split(g, rows, cols)` the grid in equal blocks;
  `symmetries(g)` names among identity, rot90, rot180, rot270, flipud, fliplr, transpose, antitranspose that leave g \
unchanged; `transform8(g, name)` applies one;
  `period(g, unknown=None)` smallest (row period, col period) that tiles g, cells equal to `unknown` ignored; \
`complete(g, unknown)` fills `unknown` cells from that tiling or from the grid's symmetries;
  `enclosed(g, bg=None)` bool mask of bg cells not connected to the border (holes); `paste(g, sub, r, c, \
transparent=None)`; `recolour(g, mapping)`; `line(g, r0, c0, r1, c1, colour)` (straight or diagonal);
  `upscale(g, k)`, `downscale(g, k)` (k x k blocks to one cell, by majority); `diff(a, b)` [(r, c, a, b)] of cells that \
differ (sizes must match)."""

# The tool arm (stage 3): the ARC-AGI-3 agent's persistent `python` tool on a puzzle. Wording after the Duck
# prompt's (duck_transfer.sandbox.PROMPT_REPLACEMENTS and the tool's description), with the puzzle's globals.
SYSTEM_TOOL = SYSTEM.replace(
    "- End every reply with the full program in one ```python block.",
    "- Before you answer you can call the `python` tool with one `code` string, as often as you need: it runs a "
    "snippet with `train` (the training pairs: dicts with `input` and `output`, grids as lists of lists of ints), "
    "`test_inputs` and `np` preloaded. Use it to inspect the grids and to test a partial idea on the training pairs "
    "instead of working it out by hand.\n"
    "- Python state persists across `python` calls within this conversation: imports, functions and variables you "
    "define stay available. A snippet that raises leaves nothing behind.\n"
    "- Your answer is a reply without a tool call that ends with the full program in one ```python block. The "
    "program is run on its own: it must contain everything it needs.")
PYTHON_TOOL = {"type": "function", "function": {
    "name": "python",
    "description": ("Run a Python snippet in this puzzle's persistent interpreter against the preloaded puzzle. "
                    "Available globals: `train` (list of training pairs, each a dict with `input` and `output` grids "
                    "as lists of lists of ints), `test_inputs` (list of grids) and `np` (numpy). "
                    "Use `print(...)` for compact output or assign final data to `result`."),
    "parameters": {"type": "object", "required": ["code"], "properties": {"code": {
        "type": "string",
        "description": "Python code to run in this puzzle's persistent session; definitions persist across tool "
                       "calls."}}}}}
# setting ``conversation``: one long conversation per puzzle, the aim stated, one persistent
# interpreter, the answer by submit().
CONV_SYSTEM = """You solve one ARC puzzle. It has training pairs (input grid -> output grid) that all follow one \
transformation rule, and one or more test inputs. A grid is a rectangle of digits 0-9 (colours), at most 30x30. Rows \
are lines; cells are separated by spaces.

Your goal is the correct output grid for each test input. You have two guesses per test input; a guess counts only if \
every cell matches.

- The `python` tool runs code in one persistent interpreter that lives for this whole puzzle: `train` (the training \
pairs: dicts with `input` and `output`, grids as lists of lists of ints), `test_inputs` and `np` are preloaded; \
imports, functions and variables you define stay available. numpy, scipy and the standard library; no files; 30 s a \
call. Call it as often as you need: inspect the grids, test ideas on the training pairs.
- To check a rule, reply without a tool call with one ```python block that defines `transform(grid)` (a list of lists \
of ints in and out). It is run on every training input and compared cell by cell; you get the exact differences. Each \
program gets a number N and is then callable in the interpreter as `program_N(grid)`. Write the general rule: it has \
to hold for the test inputs too, which can contain cases no training pair shows.
- After each program you see the distinct test outputs of the programs that pass every training pair. Do not spend \
effort re-deriving a reading you already have; a second guess is useful only if it is a genuinely different reading.
- When you are ready, call `submit(first, second)` in the interpreter: each guess is the output grid for the test \
input (a list with one grid per test input when there are several); `second` may be None. Submitting ends the \
puzzle."""
CONV_TOOL = {"type": "function", "function": {
    "name": "python",
    "description": ("Run Python in this puzzle's persistent interpreter (one process for the whole puzzle). Globals: "
                    "`train`, `test_inputs`, `np`, `program_N` for each program you sent, and `submit(first, "
                    "second=None)` to give your answer and end the puzzle. Print what you want to see, or assign it "
                    "to `result`."),
    "parameters": {"type": "object", "required": ["code"], "properties": {"code": {
        "type": "string", "description": "Python code; definitions persist across calls."}}}}}
CONV_NOTHING = ("Your reply had neither a `python` call nor a program. Call the tool, reply with one ```python block "
                "that defines `transform(grid)`, or call `submit(first, second)` in python to answer.")
CONV_CUT = ("Your reply reached the length limit. Go on from where you were; keep each reply shorter (use the "
            "`python` tool to test ideas rather than working through grids by hand).")
CONV_COMPACT = ("The conversation has reached its length limit. Write a summary for yourself to continue from: the "
                "rule hypotheses you have tried and why each failed, the programs (by number) and their results, the "
                "distinct test outputs you have, what you still suspect, and what to try next. Plain text, no tool "
                "call, at most about 1,500 words.")
CONV_RESUMED = ("[Earlier part of this conversation, summarised by you when the context filled up. The interpreter "
                "still holds everything you defined.]\n{summary}")


def tool_spec(settings):
    return CONV_TOOL if settings.get("conversation") else PYTHON_TOOL


NO_CODE_TOOL = ("Your reply had neither a `python` tool call nor a program. Call the tool with one `code` string, or "
                "reply with the full program in one ```python block.")
TOOL_LIMIT = "No more `python` calls in this conversation. Reply now with the full program in one ```python block."
# Setting ``tool_fixes`` (C1, audit 20 Oct): the tool's output as start and end up to TOOL_CHARS, no `tools` after the
# last call (TOOL_LAST closes it), preloaded names import, a failing snippet keeps what ran before it, empty calls free.
TOOL_CHARS = 6000
TOOL_LAST = "That was the last `python` call of this attempt. Reply now with the full program in one ```python block."
TOOL_FIXES = (
    ("- numpy (`np`) and scipy can be imported;", "- numpy is preloaded as `np`; scipy can be imported;"),
    ("`test_inputs` and `np` preloaded.", "`test_inputs` and `np` preloaded; a call shows up to 6,000 characters of "
     "what it prints (the start and the end)."),
    ("A snippet that raises leaves nothing behind.", "A snippet that raises keeps what ran before the failing statement."))
NO_CODE = ("Your reply had no ```python block that defines `transform(grid)`. Reply with the full program in one "
           "```python block.")
# ``truncated_continue``: a reply cut at the cap goes on from its reasoning (kept in the context) instead of being cut short
CONTINUE = "Your reply reached the length limit. Continue from where you stopped."
TRUNCATED = ("Your reply reached the length limit before a program. Give your best `transform(grid)` now in one "
             "```python block and keep the reasoning short.")
REVIEW = """Your program reproduces every training output. Before it is submitted, check what it gives for the test \
input{s}, which can contain cases that no training pair shows.

{outputs}

Compare each test input with the training inputs and name what is new in it (sizes, counts, colours, positions, \
orientations, objects that touch or overlap, more or fewer objects). At those places, check that the output follows \
the rule the training pairs show and is not an accident of how the program was written.

If the test output{s} is right, reply with exactly KEEP and no program. If not, reply with the full corrected program \
in one ```python block; it must still reproduce every training output."""
# The first replay (10 October): asked to judge, the model kept weighing two readings until the reply cap on half
# of the wrong programs. Two answers are allowed, so this one asks for the other reading instead of a verdict.
REVIEW_SECOND = """Your program reproduces every training output. Its test output{s} will be submitted as the first \
of two allowed answers. Before that, look at what it gives for the test input{s}, which can contain cases that no \
training pair shows.

{outputs}

Compare each test input with the training inputs and name what is new in it (sizes, counts, colours, positions, \
orientations, objects that touch or overlap, more or fewer objects). Where the training pairs do not settle how \
such a case is handled, do not try to decide which reading is right: the second answer is free.

- If the test output{s} can only be read one way, reply with exactly KEEP and no program.
- Otherwise reply with the full program for the most plausible other reading in one ```python block. It must still \
reproduce every training output and should give a different test output. It becomes the second answer next to the \
first, so nothing is lost if it is wrong.

Decide within a few paragraphs."""
REVIEWS = {"check": REVIEW, "second": REVIEW_SECOND}
# ``prompt`` "arc2" (arm setting): what ARC-AGI-2 puzzles tend to need, after the system prompt.
ARC2_HINT = """

About these puzzles: a rule often combines several steps, and objects (connected regions of one colour) can carry \
a meaning the grid itself defines, such as a legend, a key colour or a marker that says what to do where. A step can \
depend on context: which object, its size, position or neighbours. Check every training pair on its own and say what \
changes in it; a rule that fits most pairs but not all is usually missing a condition. Prefer rules about objects, \
their counts, sizes, positions and relations over cell-by-cell patterns."""

# ``prompt`` "arc2b" (25 Oct): a process hint (per-pair change, what differs between pairs, rule as conditions ->
# actions, test-only cases), a different wording from ARC2_HINT.
ARC2_HINT_B = """

How to work: for each training pair, say in one line what changes from input to output, object by object. Then say \
what differs between the pairs and which detail of the input decides it; the rule is a set of conditions and the \
action each one triggers. Write the program from that rule. Before answering, look at the test input{s} for anything \
the training pairs never show (a new colour, size, count or position) and make sure the rule still says what to do \
there."""

# ``prompt`` "rotate" (25 Oct): attempt a (1-based) gets LENSES[(a - 1) % 4] after the system prompt, so attempts on one task look
# for different kinds of rule (class B misses: another rule also fits training and the intended one is never found).
LENSES = ("", """

For this attempt, look first at objects: connected regions of one colour, their shapes, sizes and positions, and \
what each one becomes in the output.""", """

For this attempt, look first at geometry: reflections, rotations, translations, scaling, tiling, symmetry to \
complete, lines to draw or extend between cells.""", """

For this attempt, look first at counting and selection: how many of each thing there are, which object is the \
largest, rarest or odd one out, and what condition decides which cells change.""")

# Carry-forward (arm setting ``carry``): an attempt that ends without a verified program writes a short note, and the
# next attempt starts from the task, the earlier notes and the best program so far with its differences.
NOTE = """This attempt ends here. In at most 120 words, for whoever tries this puzzle next: which rules did the \
training pairs rule out, and what would you try instead? Plain text, no program, no tool call; keep the reasoning \
short."""
CARRY = """Earlier attempts on this puzzle did not find a rule that fits every training pair.{notes}{best}

You can build on that program or start over; either way end with the full program in one ```python block."""
CARRY_NOTES = "\n\nTheir notes:\n{notes}"
CARRY_BEST = """\n\nThe best program so far passes {passed} of {pairs} training pairs:
```python
{code}
```
What it got wrong:
{feedback}"""
# ``pool`` (arm setting): earlier programs of the task, pooled into a fresh attempt (Berman / Poetiq-style revision).
POOL = """Earlier attempts on this puzzle wrote the programs below. None is known to be right. A program that \
reproduces every training pair can still be wrong on the test input: such programs often fit the pairs with special \
cases, and earlier verified programs disagree on the test output.{programs}

Work out the rule the training pairs really show. Keep what these programs get right, find where their rules break \
(a special case that only fits the pairs, a case the test input has and the pairs do not), and write one program \
with the general rule; a different rule from all of them is fine. End with the full program in one ```python block."""
POOL_ITEM = """\n\nProgram {letter} ({status}):\n```python\n{code}\n```"""
POOL_CODE_CHARS = 4000


def pool_prompt(task, programs):
    """The first message of a pooled attempt: the task, then the earlier programs (``Engine.pick_pool``)."""
    items = []
    for n, p in enumerate(programs):
        status = ("reproduces every training pair" if p["verified"]
                  else f"reproduces {p['passed']} of {p['pairs']} training pairs, {100 * p['accuracy']:.0f}% of cells")
        code = p["code"] if len(p["code"]) <= POOL_CODE_CHARS else p["code"][:POOL_CODE_CHARS] + "\n# ... (cut)"
        items.append(POOL_ITEM.format(letter="ABCDEFGH"[n], status=status, code=code))
    return task_prompt(task) + "\n\n" + POOL.format(programs="".join(items))


# ``pool_style`` "check" (item 2, 21 Oct): most misses on hard tasks are verified programs that are wrong on the
# test input (64% of the near ones have a test input larger than every training input). The attempt sees one
# earlier verified program with its test output and checks it where the test input is new.
CHECK = """An earlier attempt wrote the program below. It reproduces every training pair, but that does not make it \
right: the test input{s} can contain cases that no training pair shows, and programs that fit the pairs are often \
wrong there.

```python
{code}
```

For the test input{s} it gives:

{outputs}

Compare each test input with the training inputs and name what is new in it (sizes, counts, colours, positions, \
orientations, objects that touch or overlap, more or fewer objects). At those places, work out how the rule the \
training pairs show should apply, and whether this output follows it or is an accident of how the program was \
written. Then write the program for the most plausible reading of the rule: a different test output from the one \
above unless you are sure that one is right. It must reproduce every training pair. End with the full program in one \
```python block."""


def check_prompt(task, code, tests):
    """The first message of a ``pool_style`` "check" attempt: the task, one verified program and its test outputs."""
    shown = "\n\n".join(f"Test input {i} ({shape(item['input'])}) -> ({shape(out)}):\n{text(out)}"
                        for i, (item, out) in enumerate(zip(task["test"], tests), 1))
    code = code if len(code) <= POOL_CODE_CHARS else code[:POOL_CODE_CHARS] + "\n# ... (cut)"
    return task_prompt(task) + "\n\n" + CHECK.format(s="s" if len(tests) > 1 else "", code=code, outputs=shown)


# ``pool_style`` "judge" (item 3, 21 Oct): on hard tasks the right reading is often among earlier verified programs
# but loses the vote (past runs, 8 verified attempts a task: right present 57%, in the top 2 by votes 37%).
JUDGE = """Earlier attempts on this puzzle wrote {k} programs that reproduce every training pair. They disagree on \
the test input{s}: below are their {n} different test outputs, each with one program that gives it and how many of \
the attempts gave it. At most one reading is right, and the most frequent one is often wrong: programs that fit the \
pairs with special cases tend to agree with each other.{readings}

Work out the rule the training pairs show and how it applies where the test input{s} go beyond the training \
inputs. Rank the readings by how likely each is right. Two answers are submitted per test input, so the second choice \
counts too. Before the program write one line: RANKING: X, Y (the letters of your first and second choice). End \
with the full program for your first choice in one ```python block: copy it, or correct it if no reading is right. \
It must reproduce every training pair."""
JUDGE_ITEM = """\n\nReading {letter} ({n} of {k} attempts):\n```python\n{code}\n```\n{outputs}"""


def judge_prompt(task, shown):
    """The first message of a ``pool_style`` "judge" attempt: the task, then each reading's program, test outputs
    and attempt count (``Engine.judge_pool``)."""
    k, s = sum(n for _, _, n in shown), "s" if len(task["test"]) > 1 else ""
    items = []
    for i, (code, outs, n) in enumerate(shown):
        code = code if len(code) <= POOL_CODE_CHARS else code[:POOL_CODE_CHARS] + "\n# ... (cut)"
        outputs = "\n\n".join(f"Test input {j} -> ({shape(out)}):\n{text(out)}" for j, out in enumerate(outs, 1))
        items.append(JUDGE_ITEM.format(letter="ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i], n=n, k=k, code=code, outputs=outputs))
    return task_prompt(task) + "\n\n" + JUDGE.format(k=k, n=len(shown), s=s, readings="".join(items))


# ``confirm_style`` "other": the attempt after a verified program asks for the most plausible other reading.
OTHER_READING = """A program that reproduces every training pair has already been found. For the test input{s} it \
gives:

{outputs}

Two answers are allowed per test input, so the second one is free. Look for the most plausible other rule that also \
reproduces every training pair but treats the test input{s} differently (cases no training pair shows: sizes, counts, \
colours, positions, overlaps). If you are sure only one reading fits, write the program for that rule."""

# ``lessons`` (arm setting; ARC-AGI-3's cross-game playbook): after a task's first verified program one short reply
# states a general lesson; new attempts on other tasks see the latest ones (never the same task's other copy).
LESSON = """Your program reproduces every training pair. In at most 60 words, state one general lesson from this \
puzzle that could help on other ARC puzzles (a kind of rule, a trap, a check worth making). Do not describe this \
puzzle's grids. Plain text, no program, no tool call; keep the reasoning short."""
LESSONS = """Lessons noted on other puzzles solved earlier (they may not apply here):
{lessons}

"""
# ``minds`` 2: every second attempt works in a second style, for diversity between independent attempts.
MIND_B = """

Work in this style: first write the rule in plain words. Then check it against each training pair one at a time, \
naming what changes in that pair and how the rule produces it. If any pair does not follow, revise the rule before \
writing code. Write the program only for a rule that explains every pair."""

DETAILED_PAIRS = 3
DIFF_CELLS = 30
STDOUT_CHARS = 1500


def chat_request(model, messages, max_tokens, *, temperature, effort, tools=None, template=None):
    """The body of one chat completion as the trial sends it: text messages only, non-streaming, top_p 0.95,
    top_k 20, thinking on at ``effort``. ``tools``: offered with the model free to call one or answer.
    ``template``: further chat-template switches (a candidate model's, e.g. preserve_thinking)."""
    return {"model": model, "messages": messages, "stream": False, "max_tokens": int(max_tokens),
            "temperature": temperature, "top_p": 0.95, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": effort, **(template or {})},
            **({"tools": tools, "tool_choice": "auto"} if tools else {})}


def tool_call(message):
    """The first `python` call of a reply as ``(the call as it goes back into the conversation, code)``; code is
    None when the arguments are not a JSON object with a string ``code``. None without a call."""
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") != "python":
            continue
        raw = function.get("arguments")
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else raw
            code = arguments.get("code") if isinstance(arguments, dict) else None
        except ValueError:
            code = None
        kept = {"id": str(call.get("id") or "call_0"), "type": "function",
                "function": {"name": "python", "arguments": raw if isinstance(raw, str) else json.dumps(raw or {})}}
        return kept, code if isinstance(code, str) and code.strip() else None
    return None


def clip(text, limit):
    """All of a short text; else its start and its end around a marker (arc2.worker.clip)."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return f"{text[:head]}\n... [{len(text) - limit} characters not shown] ...\n{text[len(text) - (limit - head):]}"


def tool_text(result, chars=0):
    """What a `python` call shows the model: its prints and ``result``, or the error. ``chars`` (tool_fixes): start
    and end up to that many characters (the worker has cut them so already) in place of the first STDOUT_CHARS."""
    cut = (lambda text: clip(text, chars + 120)) if chars else (lambda text: text[:STDOUT_CHARS])
    parts = [cut(result.get("stdout") or "").rstrip()]
    if result.get("error"):
        parts.append("Error: " + result["error"])
    elif result.get("result") is not None:
        parts.append("result = " + cut(str(result["result"])))
    text = "\n".join(p for p in parts if p)
    return text or "(no output: print what you want to see, or assign it to `result`)"


def write_json(path, data):
    """Whole or not at all: a reader (or a notebook cut at its limit) never sees half a file."""
    path = Path(path)
    path.with_name(path.name + ".part").write_text(json.dumps(data))
    os.replace(path.with_name(path.name + ".part"), path)


def load(folder, ids):
    return {i: json.loads((Path(folder) / f"{i}.json").read_text()) for i in ids}


def text(grid):
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def shape(grid):
    return f"{len(grid)}x{len(grid[0])}"


def valid(grid):
    """A rectangular grid of ints 0-9, 1 to 30 cells a side."""
    return (isinstance(grid, list) and 1 <= len(grid) <= 30 and all(isinstance(r, list) for r in grid)
            and 1 <= len(grid[0]) <= 30 and all(len(r) == len(grid[0]) for r in grid)
            and all(type(c) is int and 0 <= c <= 9 for r in grid for c in r))


DIRECT_SYSTEM = """You solve ARC puzzles. A puzzle has training pairs (input grid -> output grid) that all follow one \
transformation rule, and test inputs. A grid is a rectangle of digits 0-9 (colours), at most 30x30. Rows are lines; \
cells are separated by spaces.

Work out the rule from the training pairs and apply it to each test input yourself: no program this time. End your \
reply with one fenced block per test input, in order, each holding only the output grid in the same format (rows \
as lines, cells separated by spaces), like:
```
0 1
1 0
```"""


DIRECT_AFTER = ("No program so far reproduces every training pair. Skip the program now: apply the rule as you "
                "understand it to each test input yourself and give the output grids. Keep the reasoning short and end "
                "with one fenced block per test input, in order, holding only the grid (rows as lines, cells separated "
                "by spaces). Do not call the tool.")


def direct_grids(content, count):
    """The last ``count`` fenced blocks of ``content`` as grids (None where a block is not a grid); None without
    ``count`` blocks."""
    blocks, inside, lines = [], False, []
    for line in (content or "").splitlines():
        if line.strip().startswith("```"):
            if inside:
                blocks.append(lines)
            inside, lines = not inside, []
        elif inside:
            lines.append(line)
    if len(blocks) < count:
        return None
    grids = []
    for block in blocks[-count:]:
        try:
            grid = [[int(c) for c in row.replace(",", " ").split()] for row in block if row.strip()]
        except ValueError:
            grid = None
        grids.append(grid if grid is not None and valid(grid) else None)
    return grids


# ``images`` (arm setting): the puzzle also as pictures after its text, one per training pair (input left, output
# right) and one per test input; ARC palette, 1 px grey lines between cells. On ARC-AGI-3 the same model scored
# 2.58 with vision off vs about 3.6-4.0 with it.
PALETTE = [(0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0), (170, 170, 170), (240, 18, 190),
           (255, 133, 27), (127, 219, 255), (135, 12, 37)]
IMAGE_SIDE = 448  # px for the larger side of one grid; cells 14-28 px
IMAGES_NOTE = ("The same puzzle as pictures (standard ARC colours: 0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 grey, "
               "6 magenta, 7 orange, 8 light blue, 9 maroon). Each training picture shows the input on the left and the "
               "output on the right. The text grids above are exact; use the pictures for shapes and layout.")


def png_url(pixels):
    """A base64 PNG data URL of rows of (r, g, b)."""
    height, width = len(pixels), len(pixels[0])
    raw = b"".join(b"\x00" + bytes(v for px in row for v in px) for row in pixels)

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def grid_pixels(grids, gap_cells=1, image_side=IMAGE_SIDE):
    """Grids side by side on a white background, top-aligned, one cell size for all (``image_side`` // 32 to // 16 px)."""
    side = max(max(len(g), len(g[0])) for g in grids)
    cell = max(image_side // 32, min(image_side // 16, image_side // side))
    gap = cell * gap_cells
    width = sum(len(g[0]) * cell + 1 for g in grids) + gap * (len(grids) - 1)
    height = max(len(g) for g in grids) * cell + 1
    line, white = (85, 85, 85), (255, 255, 255)
    pixels = [[white] * width for _ in range(height)]
    x0 = 0
    for g in grids:
        h, w = len(g), len(g[0])
        for y in range(h * cell + 1):
            row = pixels[y]
            for x in range(w * cell + 1):
                if y % cell == 0 or x % cell == 0:
                    row[x0 + x] = line
                else:
                    row[x0 + x] = PALETTE[g[y // cell][x // cell]]
        x0 += w * cell + 1 + gap
    return pixels


CHANGES_NOTE = (" Where input and output have the same size, a third panel shows only the cells that change (their "
                "output colour; unchanged cells grey).")


def change_grid(a, b):
    """b's colour where it differs from a, grey (5) elsewhere; None unless a and b have the same shape."""
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return None
    return [[y if x != y else 5 for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def image_parts(task, image_side=IMAGE_SIDE, changes=False, note=None):
    """Content parts: the note, then one picture per training pair and per test input, each after a short label.
    ``changes`` (arm setting ``image_changes``): a same-shape pair's picture gets a third panel of changed cells."""
    parts = [{"type": "text", "text": "\n\n" + (note or IMAGES_NOTE) + (CHANGES_NOTE if changes else "")}]
    for i, pair in enumerate(task["train"], 1):
        grids = [pair["input"], pair["output"]]
        diff = change_grid(*grids) if changes else None
        parts += [{"type": "text", "text": f"\nTraining pair {i}:"},
                  {"type": "image_url", "image_url": {"url": png_url(grid_pixels(grids + ([diff] if diff else []), image_side=image_side))}}]
    for i, item in enumerate(task["test"], 1):
        parts += [{"type": "text", "text": f"\nTest input {i}:"},
                  {"type": "image_url", "image_url": {"url": png_url(grid_pixels([item["input"]], image_side=image_side))}}]
    return parts


# ``image_feedback`` (arm setting, with ``images``): after a wrong program, a picture per failing pair (up to
# DETAILED_PAIRS) of input | your output | expected output.
def feedback_parts(task, verdict, image_side=IMAGE_SIDE):
    parts = []
    for i, (pair, out, (ok, _, _)) in enumerate(zip(task["train"], verdict["outputs"], verdict["rows"]), 1):
        if ok or not valid(out):
            continue
        parts += [{"type": "text", "text": f"\nPair {i}: input, your output, expected output:"},
                  {"type": "image_url", "image_url": {"url": png_url(grid_pixels(
                      [pair["input"], out, pair["output"]], image_side=image_side))}}]
        if len(parts) >= 2 * DETAILED_PAIRS:
            break
    return parts


# ``review`` with ``images``: the review message also shows each test input beside the program's output.
def review_parts(task, tests, image_side=IMAGE_SIDE):
    parts = []
    for i, (item, out) in enumerate(zip(task["test"], tests), 1):
        if valid(out):
            parts += [{"type": "text", "text": f"\nTest input {i} and your program's output:"},
                      {"type": "image_url", "image_url": {"url": png_url(grid_pixels([item["input"], out],
                                                                                     image_side=image_side))}}]
    return parts


SHOW_NOTE = ("\n- In the `python` tool, `show(*grids)` puts up to 4 grids side by side in a picture that you see after "
             "the tool result (up to 4 pictures a call), e.g. `show(train[0]['input'], my_output, train[0]['output'])`.")


def shown_parts(shown):
    """The user message after a `python` call that called show(): one picture per call, grids side by side."""
    parts = [{"type": "text", "text": "Pictures from show():"}]
    for grids in shown:
        grids = [g for g in grids if valid(g)]
        if grids:
            parts.append({"type": "image_url", "image_url": {"url": png_url(grid_pixels(grids))}})
    return parts if len(parts) > 1 else None


# ``examples`` (arm setting): worked examples from a library of programs verified on ARC-AGI-2 *training* tasks
# (data/arc2/library.json, never an evaluation task), the k nearest by ``task_features``, shown before the puzzle.
EXAMPLES_HEAD = ("Solved examples from other ARC puzzles, each with a program that reproduces all of its pairs. They "
                 "may or may not resemble this puzzle; use an idea only if it fits.")
EXAMPLE_CHARS = 6000  # an example's pairs as text, at most (pairs beyond are left out)


def task_features(task):
    """A few numbers that describe a task's pairs: shape relation, sizes, palettes, changed share, colours added."""
    pairs = task["train"]
    def cols(g): return {v for r in g for v in r}
    same = all(len(p["input"]) == len(p["output"]) and len(p["input"][0]) == len(p["output"][0]) for p in pairs)
    area = [len(p["output"]) * len(p["output"][0]) / (len(p["input"]) * len(p["input"][0])) for p in pairs]
    ratio = sum(area) / len(area)
    changed = (sum(sum(a != b for ra, rb in zip(p["input"], p["output"]) for a, b in zip(ra, rb))
                   / (len(p["input"]) * len(p["input"][0])) for p in pairs) / len(pairs)) if same else 1.0
    size = sum(len(p["input"]) * len(p["input"][0]) for p in pairs) / len(pairs)
    return [1.0 if same else 0.0, min(ratio, 4.0) / 4, len(set().union(*[cols(p["input"]) for p in pairs])) / 10,
            len(set().union(*[cols(p["output"]) for p in pairs])) / 10, changed,
            sum(bool(cols(p["output"]) - cols(p["input"])) for p in pairs) / len(pairs), min(size, 900) / 900]


def pick_examples(task, library, k):
    """The k library entries nearest to ``task`` (L1 over task_features; the shape relation weighs double)."""
    want = task_features(task)
    def dist(entry):
        have = entry["features"]
        return 2 * abs(want[0] - have[0]) + sum(abs(a - b) for a, b in zip(want[1:], have[1:]))
    return sorted(library, key=lambda e: (dist(e), e["id"]))[:k]


def examples_prompt(entries):
    parts = [EXAMPLES_HEAD]
    for n, entry in enumerate(entries, 1):
        shown, size = [], 0
        for i, pair in enumerate(entry["train"], 1):
            block = (f"Pair {i}\nInput ({shape(pair['input'])}):\n{text(pair['input'])}\n"
                     f"Output ({shape(pair['output'])}):\n{text(pair['output'])}")
            if shown and size + len(block) > EXAMPLE_CHARS:
                break
            shown.append(block); size += len(block)
        parts.append(f"Example {n}\n" + "\n".join(shown) + f"\nProgram:\n```python\n{entry['program'].strip()}\n```")
    return "\n\n".join(parts) + "\n\nNow the puzzle to solve.\n\n"


VIEWS_NOTE = ("\n- After each grid, a computed summary lists its objects (4-connected cells of one colour other than the "
              "background) as colour@(top,left)-(bottom,right) xsize; after a same-shape pair, the cells that change. "
              "They are exact but partial; the grids are the ground truth.")
VIEW_OBJECTS = 12  # objects listed a grid; the rest are counted by colour
VIEW_CHANGES = 40  # changed cells listed one by one up to this many


def objects_view(grid):
    from .arckit import background, objects
    bg = background(grid)
    found = objects(grid, bg=bg)
    shown = " ".join(f"{o['colour']}@({o['box'][0]},{o['box'][1]})-({o['box'][2]},{o['box'][3]})x{o['size']}"
                     for o in found[:VIEW_OBJECTS])
    rest = {}
    for o in found[VIEW_OBJECTS:]:
        rest[o["colour"]] = rest.get(o["colour"], 0) + 1
    more = (" +" + ", ".join(f"{n} more of colour {c}" for c, n in sorted(rest.items()))) if rest else ""
    return f"Objects (background {bg}, {len(found)}): {shown or 'none'}{more}"


def changes_view(a, b):
    if shape(a) != shape(b):
        return None
    cells = [(r, c, x, y) for r, (ra, rb) in enumerate(zip(a, b)) for c, (x, y) in enumerate(zip(ra, rb)) if x != y]
    kinds = {}
    for _, _, x, y in cells:
        kinds[(x, y)] = kinds.get((x, y), 0) + 1
    line = f"Changes: {len(cells)} of {len(a) * len(a[0])} cells" + (
        "; " + ", ".join(f"{x}->{y} x{n}" for (x, y), n in sorted(kinds.items(), key=lambda kv: -kv[1])) if cells else "")
    if 0 < len(cells) <= VIEW_CHANGES:
        line += "\nChanged cells (row,column): " + " ".join(f"({r},{c})" for r, c, _, _ in cells)
    return line


VIEW_MATCHES = 10  # object matches listed a pair
VIEW_MATCH_SIZE = 3  # objects of fewer cells are left out of the matching


def matches_view(a, b):
    """Input objects found again in the output with the same shape: in place, moved and/or recoloured; output
    objects with no such input object are new."""
    from .arckit import background, objects
    src = [o for o in objects(a, bg=background(a)) if o["size"] >= VIEW_MATCH_SIZE]  # small ones match anything
    dst = [o for o in objects(b, bg=background(b)) if o["size"] >= VIEW_MATCH_SIZE]
    key = lambda o: o["mask"].tobytes() + bytes(o["mask"].shape)
    free = list(range(len(dst))); kept = moved = 0; items = []
    for o in src:
        cands = [j for j in free if key(dst[j]) == key(o)]
        if not cands:
            continue
        j = min(cands, key=lambda j: (dst[j]["colour"] != o["colour"], abs(dst[j]["box"][0] - o["box"][0]) + abs(dst[j]["box"][1] - o["box"][1])))
        free.remove(j); d = dst[j]
        dr, dc = d["box"][0] - o["box"][0], d["box"][1] - o["box"][1]
        if (dr, dc) == (0, 0) and d["colour"] == o["colour"]:
            kept += 1
            continue
        moved += 1
        what = (f" moved ({dr:+d},{dc:+d})" if (dr, dc) != (0, 0) else "") + (
            f" recoloured {o['colour']}->{d['colour']}" if d["colour"] != o["colour"] else "")
        items.append(f"{o['colour']}@({o['box'][0]},{o['box'][1]})x{o['size']}{what}")
    new = len(free)
    head = f"Object matches (same shape, {VIEW_MATCH_SIZE}+ cells): {kept} in place, {moved} moved or recoloured, {len(src) - kept - moved} input objects not found, {new} new in the output"
    return head + (": " + "; ".join(items[:VIEW_MATCHES]) + (f"; +{len(items) - VIEW_MATCHES} more" if len(items) > VIEW_MATCHES else "") if items else "")


# Arm setting ``pictures`` (with ``images`` and ``tool``): the first message carries no text grids, only each grid's
# size; the puzzle is in the pictures, its exact values in the `python` tool's globals.
PICTURES_HEAD = ("The puzzle is given as pictures below, not as text. The exact grids are in the `python` tool: "
                 "`train[i]['input']`, `train[i]['output']` (pair i + 1) and `test_inputs[k]`, as lists of lists of "
                 "ints; print them there when you need exact cell values.")
PICTURES_NOTE = ("Pictures in the standard ARC colours (0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 grey, 6 magenta, "
                 "7 orange, 8 light blue, 9 maroon); grey lines mark the cells. Each training picture shows the input on "
                 "the left and the output on the right.")


def pictures_prompt(task):
    """``pictures``: the text part of the first message, sizes only."""
    lines = [PICTURES_HEAD, ""]
    for i, pair in enumerate(task["train"], 1):
        lines.append(f"Training pair {i}: input {shape(pair['input'])} -> output {shape(pair['output'])}")
    for i, item in enumerate(task["test"], 1):
        lines.append(f"Test input {i}: {shape(item['input'])}")
    return "\n".join(lines)


def task_prompt(task, views=False, columns=False):
    """The puzzle as text; ``views``: each grid followed by its object summary, each same-shape pair by its changes;
    ``columns``: each grid also column by column (the transposed text), so vertical structure reads as lines."""
    v = (lambda g: "\n" + objects_view(g)) if views else (lambda g: "")
    if columns:
        v0 = v
        v = lambda g: v0(g) + "\nThe same grid column by column (line k = column k, top to bottom):\n" + text(
            [list(r) for r in zip(*g)])
    parts = []
    for i, pair in enumerate(task["train"], 1):
        change = changes_view(pair["input"], pair["output"]) if views else None
        if views == 2:  # views 2: also which input objects reappear in the output
            change = (change + "\n" if change else "") + matches_view(pair["input"], pair["output"])
        parts.append(f"Training pair {i}\nInput ({shape(pair['input'])}):\n{text(pair['input'])}{v(pair['input'])}\n"
                     f"Output ({shape(pair['output'])}):\n{text(pair['output'])}{v(pair['output'])}"
                     + (f"\n{change}" if change else ""))
    for i, item in enumerate(task["test"], 1):
        parts.append(f"Test input {i} ({shape(item['input'])}):\n{text(item['input'])}{v(item['input'])}")
    return "\n\n".join(parts)


def review_prompt(task, tests, style="check"):
    """The one message after the task's first verified program (arm setting ``review``): its test outputs.
    ``style`` "check" asks for a verdict, "second" for the other reading as the second answer."""
    shown = "\n\n".join(f"Test input {i} ({shape(item['input'])}) -> your program's output ({shape(out)}):\n{text(out)}"
                        for i, (item, out) in enumerate(zip(task["test"], tests), 1))
    return REVIEWS[style].format(s="s" if len(tests) > 1 else "", outputs=shown)


def extract_codes(content):
    """Every fenced block that defines ``transform``, in order."""
    blocks, inside, lines = [], False, []
    for line in (content or "").splitlines():
        if line.strip().startswith("```"):
            if inside:
                blocks.append("\n".join(lines))
            inside, lines = not inside, []
        elif inside:
            lines.append(line)
    return [b for b in blocks if "def transform" in b]


def extract_code(content):
    """The last fenced block that defines ``transform``; None without one."""
    blocks, inside, lines = [], False, []
    for line in (content or "").splitlines():
        if line.strip().startswith("```"):
            if inside:
                blocks.append("\n".join(lines))
            inside, lines = not inside, []
        elif inside:
            lines.append(line)
    found = [b for b in blocks if "def transform" in b]
    return found[-1] if found else None


def compare(expected, got):
    """``(correct, cell accuracy, differing cells)``; a wrong shape or an invalid grid scores 0."""
    if not valid(got) or len(got) != len(expected) or len(got[0]) != len(expected[0]):
        return False, 0.0, None
    diffs = [(r, c, e, g) for r, (erow, grow) in enumerate(zip(expected, got))
             for c, (e, g) in enumerate(zip(erow, grow)) if e != g]
    cells = len(expected) * len(expected[0])
    return not diffs, 1 - len(diffs) / cells, diffs


def judge(task, run):
    """A program run (``sandbox.run`` on the training inputs, then the test inputs) against the task:
    training pairs passed, mean cell accuracy, the test outputs (None where invalid) and ``verified``."""
    n = len(task["train"])
    outputs = run.get("outputs") or [{"error": run.get("error", "no result")}] * (n + len(task["test"]))
    rows = [compare(pair["output"], out) for pair, out in zip(task["train"], outputs[:n])]
    tests = [out if valid(out) else None for out in outputs[n:]]
    passed = sum(r[0] for r in rows)
    return {"passed": passed, "pairs": n, "accuracy": sum(r[1] for r in rows) / n, "tests": tests,
            "verified": passed == n and all(t is not None for t in tests), "rows": rows, "outputs": outputs}


def describe(out):
    if isinstance(out, dict):
        return "error: " + str(out.get("error"))[:400]
    return "not a valid grid (a rectangular list of lists of ints 0-9, at most 30x30)"


def feedback(task, run, verdict):
    """The user message after a program that is not verified."""
    if run.get("error"):
        return f"Your program did not run: {str(run['error'])[:1200]}\n\nFix it and reply with the full program."
    n, lines, detailed = verdict["pairs"], [], 0
    lines.append(f"Your program is correct on {verdict['passed']} of {n} training pairs.")
    for i, (pair, out, (ok, _, diffs)) in enumerate(zip(task["train"], verdict["outputs"], verdict["rows"]), 1):
        if ok:
            continue
        if not valid(out):
            lines.append(f"Pair {i}: {describe(out)}")
        elif diffs is None:
            lines.append(f"Pair {i}: wrong shape, {shape(out)} instead of {shape(pair['output'])}.")
        elif detailed < DETAILED_PAIRS:
            detailed += 1
            cells = "; ".join(f"({r},{c}): {e} not {g}" for r, c, e, g in diffs[:DIFF_CELLS])
            more = f"; and {len(diffs) - DIFF_CELLS} more" if len(diffs) > DIFF_CELLS else ""
            lines.append(f"Pair {i}: {len(diffs)} of {len(out) * len(out[0])} cells differ. Your output:\n{text(out)}\n"
                         f"Differences as (row,column): expected not yours, counted from 0: {cells}{more}")
        else:
            lines.append(f"Pair {i}: {len(diffs)} of {len(out) * len(out[0])} cells differ.")
    for i, out in enumerate(verdict["outputs"][n:], 1):
        if not valid(out):
            lines.append(f"Test input {i}: {describe(out)}")
    if verdict["passed"] == n:
        lines.append("Every training pair passes, but the program must also return a valid grid for every test input.")
    if run.get("stdout"):
        lines.append("Printed by your program:\n" + run["stdout"][:STDOUT_CHARS])
    lines.append(FIX_LINE)
    return "\n".join(lines)


FIX_LINE = "Fix the rule and reply with the full corrected program."
# ``trace`` (arm setting; ABPR, arXiv 2603.20334): the failing program's steps on its first failing training pair.
TRACE = """Your program's steps on training pair {i} (each line with the variables it set or changed):
{trace}
Find the first step where the rule the training pairs show would give something else.
"""


def carry_prompt(task, notes, best):
    """The first message of a carried attempt: the task, then the earlier notes and the best program so far."""
    text = CARRY.format(notes=CARRY_NOTES.format(notes="\n".join(f"- {n}" for n in notes)) if notes else "",
                        best=CARRY_BEST.format(**best) if best else "")
    return task_prompt(task) + "\n\n" + text


# ``repair`` (arm setting): after a task's first verified program, while its verified test outputs are one reading,
# one reply gives the test grids itself with that output in view; a different grid is the second answer. Of the
# verified-but-wrong outputs at 17M tokens (images run), 37 of 91 were within 5% of the right cells.
REPAIR = """A program that reproduces every training pair gives the test output{s} below. Two answers are allowed per \
test input and this is the first. Programs that fit every training pair are still often wrong on the test input at a \
case no training pair shows (sizes, counts, colours, positions, objects that touch or overlap), often in a few cells.

{outputs}

Check the output{s} cell by cell against the rule the training pairs show, above all where the test input is new. No \
program this time. End your reply with one fenced block per test input, in order, holding only the output grid you \
think is right (rows as lines, cells separated by spaces); give the same grid if it is already right."""
# ``anchor`` (arm setting): every attempt first predicts the test grids, then writes the program; a prediction is a
# candidate, and among verified outputs one that matches a prediction ranks first after copy agreement.
ANCHOR = """

Before any code: say what is new in the test input{s} compared with the training inputs, then predict the test \
output{s} yourself, each in a fenced block opened with ```predict (rows as lines, cells separated by spaces). Then \
write the program as usual; it must reproduce every training pair, and should agree with your prediction unless you \
find the prediction wrong."""


def repair_prompt(task, tests):
    shown = "\n\n".join(f"Test input {i} ({shape(item['input'])}) -> the program's output ({shape(out)}):\n{text(out)}"
                        for i, (item, out) in enumerate(zip(task["test"], tests), 1))
    return task_prompt(task) + "\n\n" + REPAIR.format(s="s" if len(tests) > 1 else "", outputs=shown)


def predicted_grids(content, count):
    """Grids in fenced blocks opened with ```predict (setting ``anchor``), the first ``count``; None without any."""
    blocks, inside, lines = [], False, []
    for line in (content or "").splitlines():
        s = line.strip()
        if s.startswith("```"):
            if inside:
                blocks.append(lines)
                inside = False
            elif s[3:].strip().lower() == "predict":
                inside, lines = True, []
            continue
        if inside:
            lines.append(line)
    if not blocks:
        return None
    grids = []
    for block in blocks[:count]:
        try:
            grid = [[int(c) for c in row.replace(",", " ").split()] for row in block if row.strip()]
        except ValueError:
            grid = None
        grids.append(grid if grid is not None and valid(grid) else None)
    return grids + [None] * (count - len(grids))


# ``holdout`` (arm setting): on tasks with 3+ training pairs each attempt is shown all but one; a program verifies only
# if it also reproduces the hidden pair. At 17M tokens 91 of 115 lost outputs had a verified but wrong program (48 of
# them two different wrong readings): rules that fit the shown pairs but not a case they do not show.
HOLDOUT_NOTE = """

One training pair of this puzzle is kept back: your program is also checked on it, so the rule has to hold beyond \
the pairs shown, as it must for the test input."""
# Arm setting ``reject_known`` (item 7): a verified program whose test output{s} repeat a reading earlier attempts
# already found stays a candidate; the attempt is asked for a different rule that also fits every pair.
KNOWN_READING = ("Your program is correct on every training pair, but its output for the test input{s} is a reading "
                 "earlier attempts on this puzzle already found, and it is kept as one answer. Look for a different rule "
                 "that also fits every training pair and gives a different test output{s}, for example one that handles "
                 "what is new in the test input{s} differently. Reply with the full program.")


HOLDOUT_FAIL = """Your program is correct on all {n} training pairs shown, but wrong on the training pair that is kept \
back. Like the test input, that pair holds a case the shown pairs do not settle: your rule is too specific to the pairs \
you see. Find a more general reading that explains the shown pairs without special cases, and reply with the full \
program."""
# ``hypotheses`` (arm setting): several candidate rules before any code.
# ``two_readings`` (21 Oct): two answers are allowed, and on hard tasks the right reading is often a close second
# (judge test: right among 8 attempts' readings 58%, top by votes 25%). Each reply may carry both readings.
TWO_READINGS = """

Two answers are allowed per test input. Look for the two most plausible readings of the rule: both reproduce every \
training pair but treat the test input{s} differently. End your reply with two ```python blocks, each a full program \
with ``transform``: first the reading you think most likely, then the other one. If only one reading fits, write one \
block."""
HYPOTHESES = """

Before any code, write down at least three different rules that could explain the training pairs, say how each \
treats the test input{s} (above all what is new there), and pick the one that explains every pair with the fewest \
special cases. Then write the program for it."""
# ``scouts`` (Engine.scout_rules): short independent rule statements, a pick among them, the pick in the attempt.
SCOUT_SYSTEM = """You study ARC puzzles. A puzzle has training pairs (input grid -> output grid) that all follow one \
transformation rule, and test inputs. A grid is a rectangle of digits 0-9 (colours), at most 30x30. Rows are lines; \
cells are separated by spaces. You do not write code."""
SCOUT = """

Do not write code. In at most 120 words, state the rule that turns each training input into its output, precisely \
enough to apply it by hand, and say how it applies to the test input{s}, above all to anything there that the \
training inputs do not show. Start with RULE:"""
SCOUT_CHARS = 1500
SCOUT_UNFINISHED = "(an analysis cut off while thinking; its last lines) "
SCOUT_JUDGE = """

Independent analyses of this puzzle proposed the rules below. At most one is right, and the right one may be in the \
minority: rules that fit the training pairs only through special cases are often wrong.{rules}

Check each rule against every training pair and work out what it does with the test input{s}. End with one line: \
PICK: X (the letter of the most plausible rule)."""
SCOUT_PICK = """

A separate analysis of this puzzle judged this the most plausible rule:
{rule}
Check it against the training pairs first. If it holds, write the program for it; if it does not, find the rule \
that does."""


def scout_judge_prompt(task, rules):
    s = "s" if len(task["test"]) > 1 else ""
    items = "".join(f"\n\nRule {'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]}:\n{r}" for i, r in enumerate(rules))
    return task_prompt(task) + SCOUT_JUDGE.format(rules=items, s=s)


# ``confirm_style`` "diverse": the attempt after a verified program lists every distinct verified reading so far.
DIVERSE = """Programs that reproduce every training pair have already been found. Their test output{s}, one reading \
each:

{outputs}

Two answers are allowed per test input. The readings above can all be wrong where the test input holds a case no \
training pair shows. Find a rule that is clearly different from these readings, also reproduces every training pair, \
and gives a different test output: name what is new in the test input and how else the training pairs allow it to be \
handled. Do not write a variant of a reading above."""


def diverse_prompt(task, readings):
    blocks = []
    for r, tests in enumerate(readings, 1):
        blocks.append(f"Reading {r}:\n" + "\n\n".join(
            f"Test input {i} ({shape(item['input'])}) -> ({shape(out)}):\n{text(out)}"
            for i, (item, out) in enumerate(zip(task["test"], tests), 1)))
    return task_prompt(task) + "\n\n" + DIVERSE.format(s="s" if len(task["test"]) > 1 else "", outputs="\n\n".join(blocks))


def other_prompt(task, tests):
    """The first message of an attempt after a verified program (``confirm_style`` "other")."""
    shown = "\n\n".join(f"Test input {i} ({shape(item['input'])}) -> ({shape(out)}):\n{text(out)}"
                        for i, (item, out) in enumerate(zip(task["test"], tests), 1))
    return task_prompt(task) + "\n\n" + OTHER_READING.format(s="s" if len(tests) > 1 else "", outputs=shown)
