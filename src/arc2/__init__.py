"""ARC-AGI-2 arm (9 October): a verify-and-refine loop on the arena's serving stack.

``tasks``: grids as text, exact comparison, mismatch feedback. ``sandbox`` / ``worker``: one short-lived child per
program run. ``loop``: one task's attempts, candidates and guesses, and the score-against-tokens read-out.
``trial``: the arena trial kind ``arc2`` (the arena's KV gate and server as they are, tiers of token budgets).
"""
