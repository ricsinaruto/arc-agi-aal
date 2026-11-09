"""PRO-LONG persistent memory block (variant ``fixed-2x-memory``).

Every K actions the agent rewrites a structured world-model block (goal, verified mechanics,
current level and state, plan, pitfalls, reusable code names) in one extra tool-less request
(thinking off, 2,048-token cap) from its recent transcript. The block is the first user
message after the system prompt of every later request and survives a compaction verbatim:
the compaction summary then covers only what the block does not already hold. The block is
kept outside the analyzer's append-only history, so the history itself is unchanged.
"""

MEMORY_LABEL = (
    "[Persistent memory block] Your own memory block, rewritten every few actions and kept "
    "verbatim across context compaction; trust it over older messages, revise it when the "
    "board contradicts it:\n"
)

MEMORY_PROMPT = (
    "Rewrite your persistent memory block. It is the first message of every later request and "
    "the only part of this transcript guaranteed to survive context compaction, so it must be "
    "complete on its own. Do not call tools. Use exactly these headings, each followed by short "
    "factual lines (no narrative):\n"
    "GOAL: what completes a level, with the evidence.\n"
    "VERIFIED MECHANICS: what every action does (per level if it differs), no-ops included; "
    "mark hypotheses as such.\n"
    "CURRENT LEVEL AND STATE: level number, step number, the board's configuration in brief "
    "with coordinates that matter, the last actions and their results.\n"
    "PLAN: the next actions and the expectations to check.\n"
    "PITFALLS: expectations that failed, HUD or timer cells, actions tried on this board "
    "without effect, sequences to avoid.\n"
    "REUSABLE CODE: names and purpose of helper functions or variables still defined in the "
    "python tool.\n"
    "Keep it under 400 words: copy forward everything from the previous block that is still "
    "true, drop what was refuted, add what was learned since."
)

MEMORY_COMPACTION_NOTE = (
    "\nYour persistent memory block (the first user message) is kept verbatim after this "
    "compaction: do not repeat what it already holds; summarise only what it lacks."
)


def memory_message(block):
    return {"role": "user", "content": MEMORY_LABEL + str(block).strip()}
