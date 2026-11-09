"""arc2 RTX design bundles: arc2's own v3b serving with the v87b KV mechanism only.
Base = the RO bundle (arc2-rtx-c240-hx-convro-v3b: runtime, weights, roster 120 x 2, data). Changes:
- harness: src/arc2 at arc2 HEAD; the design's arm settings (DESIGNS);
- KV / prefix (v87b's mechanism, check_rtx_stack.py): the exact reserving KVGate (no arc2.admission: every request
  reserves prompt + 16K cap, total <= kv_margin x blocks, so no preemption can happen); streams by v87b's formula,
  slots = gate capacity / measured reserved blocks a stream (2,092 / 28.5 -> 74), max_num_seqs = slots + 4;
  the L4-match caps gone (gate_match 863 blocks, prefill throttle);
- memory: v87b's expert_prune_runtime.py (= v3b's + the PLE prefill bound, the real OOM fix) and HEAD's
  serving_setup.py (vLLM argv-wait fix);
- no time limit: marks to 20M, wall 43,200 s (crash ceiling), notebook wall + 4,800 s.
NAME-cpu32 (the tier is part of the v87b recipe): + v87b's 32 GiB CPU prefix tier exactly
(arc_offload_runtime.py byte-equal, ARC_OFFLOAD_EAGLE_TAIL / NO_EAGLE_POP, fixed engine_id), and HEAD's cache_probe.py
(GPU-only reset before the host-load revisit: the 6.1M-token pool never evicts the probe's working set).
usage: build_rtx_designs.py NAME [NAME ...] (p4 f4 r4 s4, p4-cpu32 ...)"""
import json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path('./.claude/worktrees/arc2'); DIST = ROOT / 'dist'
V87B = Path('./.claude/worktrees/v84sub/dist/arena-flash-competition-v87b-sub')
BASE = DIST / 'arc2-rtx-c240-hx-convro-v3b'
SLOTS, WALL = 74, 43200
# arc2 RTX baked boot image (bake run arc2-rtx-c120-hx-bake-cpu32-p1, 30 Oct): referenced by every design built after it
IMAGE = os.environ.get('ARC2_RTX_IMAGE') or None  # computeimage-e07fn6jhq0qr1wq35q measured no faster (cap1, t10r: 31-32 min)
COMMON = {'churn': True}  # the churn stop rule in every design
ROUNDS = {'open_order': 'rounds', 'reopen_order': 'rounds', 'swap': 'budget', 'budget_floor': True,
          'compact_on_swap': True, 'compact_over': 65536, 'same_answer_swap': True, 'force_submit_after': 0}
# (P4/R4 stopped: every later round ended at its first resubmit): the identical-answer swap scoped to the
# round, a later round's budget >= its resumed context, a compacted never-submitted task first; built with -cpu32
ROUNDS2 = {**ROUNDS, 'same_answer_scope': 'round', 'budget_context': True, 'compacted_first': True}
# later rounds must iterate, never resubmit: submit() refuses a repeat, a firm round prompt with the
# earlier answers and the budget; the short read q6 runs it (P-based, -cpu32) on the 61 tasks P4 / R4 got wrong or
# never submitted on some copy (mixed 40 + never right 21), both copies, to an 8M-token stop
ROUNDS3 = {**ROUNDS2, 'no_repeat': True, 'round_prompt': True, 'same_answer_swap': False,  # dropped
           'tail_clip': True}  # the kept-tail fix (656c7ee16; the base arm had conv1's tail)
SUBSET = {'q6': (['0934a4d8', '142ca369', '16b78196', '16de56c4', '195c6913', '20a9e565', '21897d95', '247ef758', '271d71e2', '28a6681f', '2b83f449', '2d0172a1', '31f7f899', '35ab12c3', '3dc255db', '446ef5d2', '4c3d4a41', '4c416de3', '4c7dc4dd', '4e34c42c', '53fb4810', '5545f144', '581f7754', '62593bfd', '64efde09', '6e453dd6', '7491f3cf', '78332cb0', '7b5033c1', '7b80bb43', '800d221b', '80a900e0', '8698868d', '88bcf3b4', '88e364bc', '898e7135', '8b7bacbf', '8b9c3697', '9385bd28', '9bbf930d', 'a32d8b75', 'a6f40cea', 'abc82100', 'b0039139', 'b6f77b65', 'b9e38dc0', 'c7f57c3e', 'cbebaa4b', 'd35bdbdc', 'da515329', 'dbff022c', 'dd6b8c4b', 'de809cff', 'dfadab01', 'e12f9a14', 'e3721c99', 'e376de54', 'e8686506', 'eee78d87', 'f560132c', 'faa9f03d'], 8_000_000)}
# the conv1-analysis items on conv1's harness (conv1kv's arm on HEAD
# code: close mode, conv1's tail and bad-call text), v87b gate + 32 GiB tier, no ROUNDS / Q6 settings; anchor conv1kv;
# they end by themselves at roster close (marks to 20M)
CONV1 = Path('./.claude/worktrees/arc2/dist/arc2-rtx-c240-hx-conv1kv-v3b/configs/comparison.json')
FORK_KEEP = {'midpass_first': False, 'summary_prefill': None, 'ns_first_gate': False}
CONV1_ARM = {'tail_clip': False, 'bad_call_fix': False, '_base': 'conv1kv'}
# naive11's fixes are the default base of new arc2 designs (conv1's harness + bad_call_fix,
# tail_clip, compacted_first; midpass_first and ns_first_gate are loop defaults)
BASE11 = {**CONV1_ARM, 'bad_call_fix': True, 'tail_clip': True, 'compacted_first': True}
# the default base of new designs = naive12 (v8.5p drain, playbook, step-back) + bad_call_fix
BASE12 = {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'naive', 'keep_first': True,
          'reopen_order': 'first_submit', 'context_mode': 'drain', 'drop_malformed': True, 'loop_rule': True,
          'tool_chars': 3072 * 4, 'drain': {'turn_cap': 0}, 'memo': True, 'step_back': True, 'bad_call_fix': True}
AUG_FULL2 = ['rot90', 'rot180', 'rot270', 'flipud', 'fliplr', 'transpose', 'antitranspose', 'color1', 'color2']
DESIGNS = {  # the ROUNDS design (conv.py) in every run
    'cut1': {**CONV1_ARM, 'cut_mode': 'code_now', 'short_turns': True},
    # (targets.md: context cap, reply cap; arc2-conv1-analysis.md items), conv1's harness, anchor conv1kv
    'cap24': {**CONV1_ARM, 'cap': 24576},  # truncation: 20% of conv1's tokens were replies cut at 16K
    'win64': {**CONV1_ARM, '_window': 65536},  # context cap 64K: earlier compaction, shorter prefills
    'ctx192': {**CONV1_ARM, '_window': 196608},  # targets.md 192K (256K: 211 vs 217 at 9.26M)
    'ctx256': {**CONV1_ARM, '_window': 262144},  # targets.md 192K / 256K context cap; 74 slots (median stream unchanged)
    'views': {**CONV1_ARM, 'copy_views': True},  # copy 2 on an augmented view: decorrelated copies
    'chk1': {**CONV1_ARM, 'submit_check': True, 'allow_force': True},  # the harness runs the submitted program on the train pairs
    # submit() needs two functions that both fit every training pair and differ on the test;
    # after 3 refusals or the entry's tokens-so-far budget, chk1's rule
    'chk2': {**CONV1_ARM, 'submit_check': True, 'allow_force': True, 'submit_check2': True},
    # G1 B: once every entry has started, one fresh independent second attempt (spawned from
    # the token budget, augmented view 1) for entries with ttfs > 60K or 60K without a submit; agree -> keep,
    # differ -> verified first, else the entry's first guess first and the second attempt's as guess 2
    'b1': {**CONV1_ARM, 'second_attempt': 60000, 'second_view': True},
    # G1 b2 (arc2.multi, shared with the 4xL4 port): submit_check; once every entry has started, free
    # slots start fresh attempts (attempt k on view k - 1) for the backlog entry ranked first (never-submitted by
    # tokens spent, then the longest tokens to first submit of its latest attempt); re-entry after each attempt
    # submits; guesses = per output the distinct answers of all attempts by (verified, agreeing attempts, earliest)
    'b2': {**CONV1_ARM, 'submit_check': True, 'allow_force': True, 'multi_attempt': True, 'multi_view': True},
    # G1 naive (targets.md "Naive design"): every submit ends the pass; when its turn comes (backlog
    # empty, fewest passes first) the task reopens fresh, shown only its last program and submitted answer;
    # keep_first pins submission 1's first guess as guess 2
    'naive1': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'naive', 'keep_first': True},
    # naive1 with no cross-copy signal in the reopen order (fewest passes, then longest first submit)
    # targets.md naive designs ("just doubling"): every pass after a submit is a completely fresh attempt, no cross-copy order
    'naive3': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'blank', 'keep_first': True,
               'reopen_order': 'first_submit'},
    # naive1 + 24K reply cap (naive1: 300 replies cut at 16K = 4.9M of 20.6M tokens). Launched 29 Oct with 58 slots
    # (x1.256 from cap24's reservations): KV 0.69 at steady state; re-derived from that measurement: 58 x 0.8 / 0.686
    # = 67 slots (x1.095 a stream) for KV 0.8
    'naive4': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'naive', 'keep_first': True,
               'reopen_order': 'first_submit', 'cap': 24576, '_slots': 67},
    # naive3 + each blank pass on augmented view k (decorrelates the passes; b2's views without the merge)
    'naive5': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'blank', 'fresh_view': True,
               'keep_first': True, 'reopen_order': 'first_submit'},
    # naive2 + the ARC2 prompt hint (HINT: +10 at 9.26M on conv1, p 0.29; the largest of the G2 reads)
    'naive6': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'naive', 'keep_first': True,
               'reopen_order': 'first_submit', 'prompt': 'arc2'},
    # fork reads: reopen designs from naive1's saved first passes (236 submitted, 4 keep their base
    # result; first-pass tokens 10.44M); a new interpreter each pass. fk-naive1 (naive1's own settings) validates the
    # method against naive1's 280 at 20M and anchors the others
    # fork reads keep fk-naive1 p2's harness (built at b9cf68644): midpass_first and summary_prefill off
    'fk-naive1': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'naive', 'keep_first': True,
                  '_fork': 'naive1'},
    'fk-naive2': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'naive', 'keep_first': True,
                  'reopen_order': 'first_submit', '_fork': 'naive1', **FORK_KEEP},
    'fk-naive3': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'blank', 'keep_first': True,
                  'reopen_order': 'first_submit', '_fork': 'naive1', **FORK_KEEP},
    # b2 on the fork: submitted entries end at their first submit; fresh
    # independent attempts (task only, new interpreter) on the backlog by tokens to first submit, longest first;
    # never-submitted entries first (ns rule); submit_check; merge by (verified, agreeing attempts, earliest)
    'fk-b2': {**CONV1_ARM, 'submit_check': True, 'allow_force': True, 'multi_attempt': True, 'multi_view': False, 'multi_ns_first': True,
              '_fork': 'naive1', **FORK_KEEP},
    # targets.md "Backlog rounds design" on the fork: ROUNDS3 (P-mode rounds: rounds order,
    # budget max(ttfs, context) floor one reply, swap at budget, compact > 64K, repeats refused, round prompt, no
    # identical-answer swap, compacted never-submitted first); round 2 opens a new conversation (no saved context);
    # never-submitted entries play on (fork_ns_play); keep_first as fk-naive1
    'fk-rounds': {**CONV1_ARM, 'submit_mode': 'provisional', 'firm': True, **ROUNDS3, 'keep_first': True,
                  'fork_ns_play': True, '_fork': 'naive1', **FORK_KEEP},
    # the targets.md backlog rounds design as a full run on BASE11 (later rounds keep their context);
    # anchor naive11
    'rounds1': {**BASE11, 'submit_mode': 'provisional', 'firm': True, **ROUNDS3, 'keep_first': True},
    # / ~13:15: submit_check; check2 only for first submits past 40K tokens or a failed augmentation check
    # (the submitted function under the transforms, a soft trigger); chk4's check2 mechanics and exits; on BASE12
    # hyp1 (29 Oct; naive12 analysis): ttfs <= 30K closes at the first submit (99 right / 7 wrong in naive12), every
    # other entry reopens in the hypothesis-comparison mode (CONV_HYP). Bar >= +30 vs naive12 279 at 20M.
    'hyp1': {**BASE12, 'fresh_text': 'hyp', 'triage_close': 30000},
    # pass1 (29 Oct; naive12: > 60K to the first submit 52% all-right): a pass reaching 60K tokens without a submit
    # submits its best program output and the entry comes back in a fresh pass. Bar >= +30 vs naive12 279 at 20M.
    'pass1': {**BASE12, 'pass_budget': 60000},
    # chk6: chk5 without the aug check (replay: fires on 72% of right vs 81% of wrong functions)
    # chk6b: chk6 with the force path removed (no-force default: only a function that
    # reproduces every training pair is accepted; grids alone never)
    # v52 (owner via l4, docs/v52-submission.md): BASE12 + naive6's hint; v9's drain jitter (range proposed by l4:
    # T in [78K, 114K] (window 131,072 budget 114,432), 56K kept; awaiting the range)
    'v52': {**BASE12, 'prompt': 'arc2'},
    'v52j': {**BASE12, 'prompt': 'arc2',
             'drain': {'turn_cap': 0, 'keep_tokens': 56000, 'jitter': {'low': 78000, 'high': 114000, 'seed': 0}}},
    # v52r: RTX twin of the v52 submission harness: 152K context (window 151,040 as v9), drain
    # trigger U[90K, 134K] per drain, 64K kept, ARC2 hint
    'v52r': {**BASE12, 'prompt': 'arc2', '_window': 151040,
             'drain': {'turn_cap': 0, 'keep_tokens': 64000, 'jitter': {'low': 90000, 'high': 134000, 'seed': 0}}},
    # try-cap: the context window is the try's cap (no drain, no compaction); a submit needs two
    # different guesses and ends the try; a later try needs 4 distinct guesses; 2 accepted submits end the task; c120.
    # Streams: naive12's time-weighted reserve 26.1 blocks at 128K, 23.3 under a 64K window -> 28.5 x 23.3 / 26.1
    # base13b: every submit two different guesses; first submit past 20K: both functions that fit, on all
    # the entry's submits, no exit; no drain: a pass ends at an accepted submit or at the 128K context cap, and every
    # pass is a plain repeat of pass 1 (new interpreter, numbers restart); no-submit entries first in the reopen queue;
    # vote across all guesses; a never-accepted entry scored on its last refused pair. Memo off: its text is false here.
    # The 20K rule is not in the prompt: only a refusal of an entry past it says what passes.
    'base13b': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'keep_first': False, 'pick': 'vote_all',
                'fresh_text': 'blank', 'fresh_interp': True, 'pass_clean': True, 'submit_two': True, 'check2_late': 20000,
                'score_refused': True, 'context_mode': 'cap', 'drain': None, 'reopen_order': 'never_first', 'memo': False},
    # base14: base13b without verification, the first-guess line and
    # score_refused; every entry gets pass 2, later passes only until two passes share all first guesses (agree1's
    # agree_close); a loop-rule fire or 20 bad calls in a row end a pass.
    'base14': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'keep_first': False, 'pick': 'vote_all',
               'fresh_text': 'blank', 'fresh_interp': True, 'pass_clean': True, 'submit_two': True, 'no_first_line': True,
               'agree_close': True, 'loop_exit': 20, 'context_mode': 'cap', 'drain': None, 'reopen_order': 'never_first',
               'memo': False},
    'tc128': {**BASE12, 'context_mode': 'cap', 'try_cap': True, 'keep_first': False, 'reopen_order': 'try_cap',
              'drain': None, '_copies': 1},
    'tc64': {**BASE12, 'context_mode': 'cap', 'try_cap': True, 'keep_first': False, 'reopen_order': 'try_cap',
             'drain': None, '_copies': 1, '_window': 65536, '_bps': 25.44},
    'base12': {**BASE12, '_copies': 1},  # the c120 anchor: BASE12 unchanged
    # 30 Oct designs (c120, 10M, anchor base12 138). agree1: a pass repeating an earlier pass's first guess closes the
    # task (replay base12c: 0.91M freed, 138 -> 137), the freed tokens go to the rest. cap24c: 24K reply cap (26% of
    # naive12 tokens are replies cut at 16K; cap24 on conv1 +7 at 9.26M), 67 streams by the formula.
    'agree1': {**BASE12, '_copies': 1, 'agree_close': True},
    'cap24c': {**BASE12, '_copies': 1, 'cap': 24576, '_slots': 67},
    # altR / altP (30 Oct, fit probe: 32 of 46 naive12 misses at 20M had a train-fitting program with a wrong test
    # output): ask for a different rule that also fits every pair, at the next pass (altR) or in the pass at its
    # first train-fit (altP).
    'altR': {**BASE12, '_copies': 1, 'alt_fit': 'reopen'},
    'altP': {**BASE12, '_copies': 1, 'alt_fit': 'pass'},
    # solved detector read: Flash's log-prob of every submitted reading of base12c / naive12
    'score1': {**BASE12, '_copies': 1, '_score': 'cands-base12c-naive12'},
    # re-solves (pass 2+) at reasoning effort medium (Flash's template has xhigh, medium, low only), pass 1 xhigh
    'lean': {**BASE12, '_copies': 1, 'resolve_effort': 'medium'},
    # (lean: re-solve value is depth): cap24c's pass 1 (24K reply cap: 120 vs 117 rights)
    # with base re-solves (16K: cap24c's pass 2 lost 15 vs 20); cap24c's 67 slots for the 24K reservation
    'cap1': {**BASE12, '_copies': 1, 'cap': 24576, 'resolve_cap': 16384, '_slots': 67},
    # option 2: re-solves at temperature 1.0 (more independent at the same depth; pass 1 0.7)
    't10r': {**BASE12, '_copies': 1, 'resolve_temperature': 1.0},
    # (failure modes): 23 of base12c's 50 pass-1 misses are within 20 cells of
    # the truth (right rule, test-input edge case). rep1: the pass after the first submit is a repair of that answer's
    # train-fitting function with a test-vs-train novelty report; rep2: the same check inside pass 1 at its first submit
    'rep1': {**BASE12, '_copies': 1, 'repair': 'fresh'},
    'rep2': {**BASE12, '_copies': 1, 'repair': 'inpass'},
    # A (fresh2) every later try is a completely fresh attempt (blank head, new interpreter, no
    # playbook); final guesses = most-submitted first guess (ties: latest) + the most-submitted different one
    'fresh2': {**BASE12, '_copies': 1, 'fresh_text': 'blank', 'fresh_interp': True, 'keep_first': False, 'pick': 'vote'},
    # B (notes1, ARC-AGI-3 harness port): each later try in a new interpreter opens with the earlier tries' playbook
    # (v8.5p memo_style 2 notes), never their programs or answers; a try ends only once its notes are rewritten
    'notes1': {**BASE12, '_copies': 1, 'try_notes': True},
    # vcheck: at the first submit, the equivariance verifier on the submitting function; a
    # training symmetry broken on the test input holds the pass for one check (answer kept as guess 2) naming the
    # transform and cells; + tool_fix (`import train` / `np` bind the session's names; an empty call is explained)
    'vcheck': {**BASE12, '_copies': 1, 'vcheck': True, 'tool_fix': True},
    # think1c (think1's clean read): a reply still thinking at 10K goes on in the same turn as a chat
    # continuation (images kept; think1's raw-id path dropped them) + vcheck's tool fixes
    'think1c': {**BASE12, '_copies': 1, 'think_cap': 10000, 'think_chat': True, 'tool_fix': True},
    # c120 to 20M, tool_fix in both. ind20 = one naive reopen, then independent tries (fresh2's, vote pick),
    # ordered by conv.try_class; ctl20 = its control
    'ctl20': {**BASE12, '_copies': 1, 'tool_fix': True, '_stop20': True},
    'ind20': {**BASE12, '_copies': 1, 'tool_fix': True, '_stop20': True, 'indep_after': 1, 'reopen_order': 'indep'},
    # BASE12 without step-back, with the ARC2 hint (v52/v55's); tool_fix off, pairs with base12
    'base13': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2'},
    # base12v: v52r on c120 without step-back: 152K context, drain keep 64K, jittered trigger; to 15M
    'base12v': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', '_window': 151040, '_stop15': True,
                'drain': {'turn_cap': 0, 'keep_tokens': 64000, 'jitter': {'low': 90000, 'high': 134000, 'seed': 0}}},
    # base15: base13 with full context across two passes: a submit parks the entry; it reopens
    # by tokens to first submit, ascending, in the same conversation, told its answer was slightly wrong; a second
    # submit must differ and ends the entry; both first guesses score. Slots: full-context sim (scratchpad da/fm/s15.py)
    # raises reserve per stream 22.4 -> 31.2 blocks; 64 gives KV ~.8 in pass 1, lowered live in pass 2.
    'base15': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'reopen', 'fresh_each': False,
               'reopen_order': 'first_submit_asc',
               'reopen_wrong': True, 'second_differs': True, '_bps': 32.7},
    # base15b: base15 with submit(answer), one answer only, and no served text naming two
    'base15b': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'reopen', 'fresh_each': False,
                'reopen_order': 'first_submit_asc', 'reopen_wrong': True, 'second_differs': True, '_bps': 32.7,
                'one_answer': True},
    # base15e: base15b, one line of the system text; 15M a crash ceiling, both passes run out
    'base15e': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'reopen', 'fresh_each': False,
                'reopen_order': 'first_submit_asc', 'reopen_wrong': True, 'second_differs': True, '_bps': 32.7,
                'one_answer': True, 'submit_final': True, '_stop15': True},
    # base15c: base15b as one conversation: a first submit stays in its slot and is asked for a different
    # second answer, the second accepted submit ends it; every drain parks; backlog by fewest tokens; c120 to 15M.
    # idle_end: the backstop. Slots: scratchpad da/fm/s15c.py.
    'base15c': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'continue', 'fresh_each': False,
                'second_differs': True, 'one_answer': True, 'park_drain': True, 'open_order': 'fewest', 'idle_end': 30,
                'idle_tokens': 128, '_bps': 43.6, '_stop15': True},
    # base15d: base15c, repeats allowed, each submit asked to re-check, ends at the same answer three
    # submits in a row; final slots are the first and the last submit only.
    'base15d': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'continue', 'fresh_each': False,
                'recheck': True, 'pick': 'first_last', 'one_answer': True, 'park_drain': True, 'open_order': 'fewest',
                'idle_end': 30, 'idle_tokens': 128, '_bps': 43.6, '_stop15': True},
    # base15f: base15d, only programs that fit every training pair are recorded, the 100% line; every
    # accepted submit told it is slightly wrong. base16: one pair never served, each submit must pass it, pass/fail only
    'base15f': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'continue', 'fresh_each': False,
                'recheck': 'wrong', 'submit_check': True, 'submit_final': True,
                'idle_any': True, 'fallback_last': 'last', 'pick': 'first_last', 'one_answer': True, 'park_drain': True, 'open_order': 'fewest',
                'idle_end': 30, 'idle_tokens': 128, '_bps': 32.7, '_seqs': 88, '_stop15': True},
    'base16': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'continue', 'fresh_each': False,
                'recheck': 'wrong', 'submit_check': True, 'submit_final': True,
                'idle_any': True, 'fallback_last': 'passing', 'hold_one': 20261101, 'pick': 'first_last', 'one_answer': True, 'park_drain': True, 'open_order': 'fewest',
                'idle_end': 30, 'idle_tokens': 128, '_bps': 32.7, '_seqs': 88, '_stop15': True},
    # base15c2: base15c, the 100% line, every accepted submit told it was wrong (base15b's text) and it
    # stays in its slot; each submit differs from the first only; no end at 2; final slots first and last. churn off:
    # an entry ends only at the idle end or 15M. Start at the recipe's 64, ceiling 88.
    'base15c2': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'continue', 'fresh_each': False,
                 'second_differs': True, 'differs_first': True, 'recheck': 'reopen', 'submit_final': True, 'pick': 'first_last',
                 'churn': False, 'one_answer': True, 'park_drain': True, 'open_order': 'fewest', 'idle_end': 30,
                 'idle_tokens': 128, '_bps': 32.7, '_seqs': 88, '_stop15': True},
    # base16b: base15c (the second accepted submit ends it, it must differ, both score), the 100% line, the
    # first accepted submit told it was wrong (base15b's text), programs only, base16's held pair (same seed, same pairs)
    'base16b': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'continue', 'fresh_each': False,
                'second_differs': True, 'reopen_wrong': True, 'submit_check': True, 'submit_final': True, 'idle_any': True,
                'fallback_last': 'passing', 'hold_one': 20261101, 'one_answer': True, 'park_drain': True,
                'open_order': 'fewest', 'idle_end': 30, 'idle_tokens': 128, '_bps': 32.7, '_seqs': 88, '_stop15': True},
    # base17: base15e's submits and line, base15c's conversation: two submits end a try, not the task; every
    # try after it is blank (new conversation and interpreter) through the fewest-tokens backlog, until 15M. Final slots:
    # the two grids most submits agree on, as of the mark; a tie goes to a try's first submit, then the oldest.
    'base17': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'continue', 'fresh_each': False,
               'fresh_tries': True, 'fresh_text': 'blank', 'fresh_interp': True, 'pass_clean': True, 'keep_first': False,
               'second_differs': True, 'differs_first': True, 'reopen_wrong': True, 'one_answer': True, 'submit_final': True,
               'pick': 'vote_all', 'park_drain': True, 'open_order': 'fewest', 'idle_end': 30, 'idle_tokens': 128,
               'idle_any': True, '_bps': 32.7, '_seqs': 88, '_stop15': True},
    # base17r: base17 rebuilt on the fewest order by try; streams 56: base17 held KV .75 at 52 (2180 x .96 / 56)
    'base17r': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'continue', 'fresh_each': False,
                'fresh_tries': True, 'fresh_text': 'blank', 'fresh_interp': True, 'pass_clean': True, 'keep_first': False,
                'second_differs': True, 'differs_first': True, 'reopen_wrong': True, 'one_answer': True, 'submit_final': True,
                'pick': 'vote_all', 'park_drain': True, 'open_order': 'fewest', 'idle_end': 30, 'idle_tokens': 128,
                'idle_any': True, '_bps': 37.4, '_seqs': 88, '_stop15': True},
    # base17b: base17 with submit(first, second), the two different on every output, one accepted submit
    # ends the try; votes over all guesses of all tries, a tie to a try's first guess
    'base17b': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'continue', 'fresh_each': False,
                'fresh_tries': True, 'fresh_text': 'blank', 'fresh_interp': True, 'pass_clean': True, 'keep_first': False,
                'submit_two': True, 'no_first_line': True, 'submit_final': True, 'pick': 'vote_all', 'park_drain': True,
                'open_order': 'fewest', 'idle_end': 30, 'idle_tokens': 128, 'idle_any': True, '_bps': 32.7, '_seqs': 88,
                '_stop15': True},
    # v56: base15c's conversation (second accepted submit ends it, both score), base15e's line and text,
    # v52's context. base16b without programs or the held pair
    'v56': {**BASE12, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'continue', 'fresh_each': False, '_window': 151040,
            'drain': {'turn_cap': 0, 'keep_tokens': 64000, 'jitter': {'low': 90000, 'high': 134000, 'seed': 0}},
            'second_differs': True, 'reopen_wrong': True, 'one_answer': True, 'submit_final': True, 'park_drain': True,
            'open_order': 'fewest', 'idle_end': 30, 'idle_tokens': 128, '_bps': 32.7},
    # the same on c120 (10M mark, v56 with p50 c268 weights, built by build_p75_read.py v56c268)
    'v56c120': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'continue', 'fresh_each': False,
                '_window': 151040, 'drain': {'turn_cap': 0, 'keep_tokens': 64000, 'jitter': {'low': 90000, 'high': 134000, 'seed': 0}},
                'second_differs': True, 'reopen_wrong': True, 'one_answer': True, 'submit_final': True, 'park_drain': True,
                'open_order': 'fewest', 'idle_end': 30, 'idle_tokens': 128, '_bps': 32.7},
    # v56r: v56 (context, drain) + base17r's fresh rounds, no token stop of its own: the wall ends play
    'v56r': {**BASE12, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'continue', 'fresh_each': False, '_window': 151040,
             'drain': {'turn_cap': 0, 'keep_tokens': 64000, 'jitter': {'low': 90000, 'high': 134000, 'seed': 0}},
             'second_differs': True, 'reopen_wrong': True, 'one_answer': True, 'submit_final': True, 'park_drain': True,
             'open_order': 'fewest', 'idle_end': 30, 'idle_tokens': 128, '_bps': 32.7, 'fresh_tries': True,
             'fresh_text': 'blank', 'fresh_interp': True, 'pass_clean': True, 'keep_first': False, 'differs_first': True,
             'pick': 'vote_all', 'idle_any': True},
    # the same on c120
    'v56rc120': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'continue', 'fresh_each': False,
                 '_window': 151040, 'drain': {'turn_cap': 0, 'keep_tokens': 64000, 'jitter': {'low': 90000, 'high': 134000, 'seed': 0}},
                 'second_differs': True, 'reopen_wrong': True, 'one_answer': True, 'submit_final': True, 'park_drain': True,
                 'open_order': 'fewest', 'idle_end': 30, 'idle_tokens': 128, '_bps': 32.7, 'fresh_tries': True,
                 'fresh_text': 'blank', 'fresh_interp': True, 'pass_clean': True, 'keep_first': False, 'differs_first': True,
                 'pick': 'vote_all', 'idle_any': True},
    # base15bf: base15b + the idle backstop on the full BF16 model, unpruned, on 2 B300 (tp-bf16play's
    # serving: TP2, MTP off, BF16 KV); 15M is the crash ceiling. Slots: 7.7M-token pool / 4.3M RTX pool x 48.
    'base15bf': {**BASE12, '_copies': 1, 'step_back': False, 'prompt': 'arc2', 'submit_mode': 'reopen', 'fresh_each': False,
                 'reopen_order': 'first_submit_asc', 'reopen_wrong': True, 'second_differs': True, 'one_answer': True,
                 'idle_end': 30, 'idle_tokens': 128, '_slots': 84, '_stop15': True, '_bf16x2': True},
    'bake': {**BASE12, '_copies': 1, '_bake': True},  # the assets onto a disk for the boot image (no play)
    # (score1 weak: likelihood AUC 0.57, pass-1 length AUC 0.79): lean re-solves only for
    # entries whose first submit came under 32K tokens (likely solved), full-effort re-solves for the rest
    'leanS': {**BASE12, '_copies': 1, 'resolve_effort': 'medium', 'resolve_effort_under': 32000},
    'chk6b': {**BASE12, 'submit_check': True, 'submit_check2': True, 'check2_late': 40000, 'check2_max_refusals': 10,
              'check2_token_exit': None},
    'chk6': {**BASE12, 'submit_check': True, 'allow_force': True, 'submit_check2': True, 'check2_late': 40000, 'check2_max_refusals': 10,
             'check2_token_exit': None},
    # think1 (29 Oct; naive12: 331 replies cut at 16K inside thinking = 26% of tokens; 0 producing replies of the 286
    # right outputs ran past 10K): a reply still thinking at 10K ends its think block in the same turn and acts
    'think1': {**BASE12, 'think_cap': 10000},
    'chk5': {**BASE12, 'submit_check': True, 'allow_force': True, 'submit_check2': True, 'check2_late': 40000, 'aug_check': AUG_FULL2,
             'check2_max_refusals': 10, 'check2_token_exit': None},
    'fk-naive5': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'blank', 'fresh_view': True,
                  'keep_first': True, 'reopen_order': 'first_submit', '_fork': 'naive1', **FORK_KEEP},
    # first-pass levers (the run-to-run spread is in the first pass), on naive2 (no copy
    # signal); bars sized to the ~+-20 noise: paired vs naive2 >= +30 at 20M and > 277, guard >= 209
    'naive7': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'naive', 'keep_first': True,
               'reopen_order': 'first_submit', 'effort': 'low'},  # Flash's template has xhigh / medium / low only
    'naive8': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'naive', 'keep_first': True,
               'reopen_order': 'first_submit', 'effort': 'medium'},
    'naive9': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'naive', 'keep_first': True,
               'reopen_order': 'first_submit', 'submit_check': True, 'allow_force': True},
    # 1.0 vs 0.7 (27 Oct): verified programs right 72% vs 85%; 0.5 extends that trend
    'naive10': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'naive', 'keep_first': True,
                'reopen_order': 'first_submit', 'temperature': 0.5},
    # conv1-analysis blocker (bad-call loops) + empty-tail clip + backlog rule (never-submitted compacted task first)
    'naive11': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'naive', 'keep_first': True,
                'reopen_order': 'first_submit', 'bad_call_fix': True, 'tail_clip': True, 'compacted_first': True},
    # chk2's rule on naive2 to 20M (every design not run to 20M runs to 20M)
    'chk3': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'naive', 'keep_first': True,
             'reopen_order': 'first_submit', 'submit_check': True, 'allow_force': True, 'submit_check2': True},
    # v8.5p's context management on naive2: drain_half (no summary), drain note + inventory, turn cap,
    # 3,072-token tool output, malformed calls dropped, loop rule
    'naive12': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'naive', 'keep_first': True,
                'reopen_order': 'first_submit', 'context_mode': 'drain', 'drop_malformed': True, 'loop_rule': True,
                'tool_chars': 3072 * 4, 'drain': {'turn_cap': 0}, 'memo': True, 'step_back': True},
    # chk2's rule, the only exit after 10 refusals (no token exit)
    'chk4': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'naive', 'keep_first': True,
             'reopen_order': 'first_submit', 'submit_check': True, 'allow_force': True, 'submit_check2': True, 'check2_max_refusals': 10,
             'check2_token_exit': None},
    'naive2': {**CONV1_ARM, 'submit_mode': 'fresh', 'fresh_each': True, 'fresh_text': 'naive', 'keep_first': True,
               'reopen_order': 'first_submit'},
    'hint1': {**CONV1_ARM, 'prompt': 'arc2'},
    'q6': {'submit_mode': 'provisional', 'firm': True, **ROUNDS3},
    # bar (> 257 at 20M with the tokens left after every task submitted): conv1's harness, Q6's rounds
    # after round 1, and keep_first (Q6's 3 right->wrong were two-guess later submits dropping the old first guess)
    'p7': {**CONV1_ARM, 'submit_mode': 'provisional', 'firm': True, **ROUNDS3, 'keep_first': True},
    # p7 rerun with keep_first pinning submission 1's first guess (arc2 335346c6f), all else as p7
    'p7b': {**CONV1_ARM, 'submit_mode': 'provisional', 'firm': True, **ROUNDS3, 'keep_first': True},
    # (p7: rounds >= 2 took 58-84% of the tokens after 10M with ~25 entries unsubmitted):
    # p7b with later rounds held until every playing entry has a first submit
    'p7c': {**CONV1_ARM, 'submit_mode': 'provisional', 'firm': True, **ROUNDS3, 'keep_first': True,
            'rounds_after_all_submitted': True},
    'p6': {'submit_mode': 'provisional', 'firm': True, **ROUNDS3},
    'r6': {'submit_mode': 'reopen', 'reopen_passes': 0, 'partner_text': False, **ROUNDS3},
    'p5': {'submit_mode': 'provisional', 'firm': True, **ROUNDS2},
    'r5': {'submit_mode': 'reopen', 'reopen_passes': 0, 'partner_text': False, **ROUNDS2},
    'f5': {'submit_mode': 'fresh', **ROUNDS2},
    's5': {'submit_mode': 'fresh', 'submit_check': True, 'allow_force': True, **ROUNDS2},
    # superseded ~12:00 (round-1 submit counted by the identical-answer swap):
    'p4': {'submit_mode': 'provisional', 'firm': True, **ROUNDS},
    'f4': {'submit_mode': 'fresh', **ROUNDS},
    'r4': {'submit_mode': 'reopen', 'reopen_passes': 0, 'partner_text': False, **ROUNDS},
    's4': {'submit_mode': 'fresh', 'submit_check': True, 'allow_force': True, **ROUNDS},
    # superseded 09:00 (balanced order):
    #
    'p3': {'submit_mode': 'provisional', 'firm': True, 'open_order': 'balance'},
    'f3': {'submit_mode': 'fresh', 'reopen_order': 'balance', 'open_order': 'balance'},
    'r3': {'submit_mode': 'reopen', 'reopen_passes': 0, 'reopen_order': 'balance', 'open_order': 'balance', 'partner_text': False},
    's20': {'submit_mode': 'fresh', 'reopen_order': 'balance', 'open_order': 'balance', 'submit_check': True, 'allow_force': True},
    # superseded 28 Oct ~08:25 (longest-first order): p2, p2t, f, r20
    'p2': {'submit_mode': 'provisional', 'firm': True, 'open_order': 'first_submit'},
    'p2t': {'submit_mode': 'provisional', 'firm': True, 'open_order': 'tokens'},
    'f': {'submit_mode': 'fresh', 'reopen_order': 'first_submit'},
    'r20': {'submit_mode': 'reopen', 'reopen_passes': 0, 'reopen_order': 'first_submit', 'partner_text': False},
}
# c240 to 30M: base15e and base17r as their c120 reads (their bundles' own src/arc2), copies 2, 5M marks
for _d in ('base15e', 'base17r'):
    DESIGNS[f'{_d}240'] = {**{k: v for k, v in DESIGNS[_d].items() if k not in ('_copies', '_stop15')}, '_stop30': True,
                           '_src_of': f'arc2-rtx-c120-hx-{_d}-cpu32-v3b'}
# base18: base15e + a third answer asked as the second, strict pass levels, a
# selector call per output picks guess 2 from answers 2 and 3 (guess 1 = answer 1; no selector result: answer 2)
DESIGNS['base18'] = {**DESIGNS['base15e'], 'third': True}
# base19: base15e with a much stronger second-answer ask, then "are you 100% sure" after every later
# answer (repeats accepted), 5 identical submits in a row close the task; guess 1 answer 1, guess 2 the latest
DESIGNS['base19'] = {**DESIGNS['base15e'], 'sure': True}
DESIGNS['v56240'] = {**DESIGNS['v56'], '_stop30': True, '_src_of': 'arc2-rtx-c240-hx-v56-cpu32-v3b'}  # v56 at 59e323cf8
# base15e256: base15e with a 256K window (max_model_len is 262,144 already); the drain's depth scales with
# it (56,960 -> 113,920 below the budget); slots stay base15e's (ctx256: median stream unchanged), the guard holds KV
DESIGNS['base15e256'] = {**DESIGNS['base15e'], '_window': 262144, 'drain': {'turn_cap': 0, 'drain_tokens': 113920},
                         '_src_of': 'arc2-rtx-c120-hx-base15e-cpu32-v3b'}
# base19w256: base15e256 with base19's stronger ask for the second answer only; 52 slots (base15e256: 134
# preemptions at 5-7.5M on 64)
DESIGNS['base19w256'] = {**{k: v for k, v in DESIGNS['base15e256'].items() if k != '_src_of'}, 'strong_ask': True,
                         '_bps': 40.25}  # 2180 x .96 / 40.25 = 52: a 256K stream reserves more
# base20: base15e256 + sandbox fixes (imports of preloaded names, getsource, `result`, salvaged call
# bodies), check(fn) in place of the program paragraphs, 896 px pictures, base19's strong ask with a picture of each test
# input beside the submitted answer, longest first submit reopened first, 24K replies; 2180 x .96 / (40.25 x 1.095) = 47
DESIGNS['base20'] = {**{k: v for k, v in DESIGNS['base15e256'].items() if k != '_src_of'}, 'sandbox_fix': True,
                     'check_fn': True, 'image_side': 896, 'strong_ask': True, 'reopen_picture': True,
                     'reopen_order': 'first_submit', 'cap': 24576, '_bps': 44.07}
# base21: base15e exactly + restore_names, base20's sandbox_fix and check_fn; no stall backstop
DESIGNS['base21'] = {**DESIGNS['base15e'], 'sandbox_fix': True, 'check_fn': True, 'restore_names': True, 'stall_end': 0}
# base22: base15e256 + the answer picture after base15e's reopen text (224 px), longest first submit
# reopened first, sandbox_fix and restore_names; HEAD src (base15e256 ran base15e's); no stall backstop, no check(fn)
DESIGNS['base22'] = {**{k: v for k, v in DESIGNS['base15e256'].items() if k != '_src_of'}, 'reopen_picture': True,
                     'reopen_order': 'first_submit', 'sandbox_fix': True, 'restore_names': True, 'stall_end': 0}
# v57240: the Kaggle v57 harness on c240 to 30M: base15e256's arm + reopen_picture, sandbox_fix,
# restore_names, stall_end 0 (reopen_order first_submit_asc kept), src/arc2 of the base22 bundle (d8d30e317) as l4 took it
DESIGNS['v57240'] = {**{k: v for k, v in DESIGNS['base15e256'].items() if k not in ('_src_of', '_copies', '_stop15')},
                     'reopen_picture': True, 'sandbox_fix': True, 'restore_names': True, 'stall_end': 0, '_stop30': True,
                     '_src_of': 'arc2-rtx-c120-hx-base22-cpu32-v3b'}
HEAD_FILES = ['src/arc_infra/serving_setup.py'] + sorted(
    p.relative_to(ROOT).as_posix() for p in (ROOT / 'src/arc2').rglob('*.py') if '__pycache__' not in p.parts)
PLE = 'src/arc_infra/expert_prune_runtime.py'
OFFLOAD = 'src/arc_infra/arc_offload_runtime.py'
TIER_FILES = ['src/arena/serving.py', 'src/arena/cache_probe.py']  # HEAD's (engine_id; probe reset)
TIER_GIB, TIER_ENV = 32, {'ARC_OFFLOAD_EAGLE_TAIL': '1', 'ARC_OFFLOAD_NO_EAGLE_POP': '1'}  # v87b's
dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', *HEAD_FILES, *TIER_FILES], capture_output=True, text=True)
assert dirty.stdout == '', 'commit the harness files first: ' + dirty.stdout
commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'scripts/arena')]
from build_arena import manifest

for design in sys.argv[1:]:
    roster = 'c120' if DESIGNS[design.removesuffix('-cpu32')].get('_copies') == 1 else 'c240'
    name = f'arc2-rtx-{roster}-hx-{design}-v3b'
    design, tier = (design[:-6], True) if design.endswith('-cpu32') else (design, False)
    heads = HEAD_FILES + (TIER_FILES if tier else [])
    out = DIST / name; shutil.rmtree(out, ignore_errors=True); out.mkdir()
    for p in BASE.rglob('*'):
        rel = p.relative_to(BASE).as_posix()
        if p.is_dir() or '__pycache__' in p.parts or rel in ('configs/comparison.json', 'MANIFEST.json', PLE) \
                or rel in heads or (tier and rel == OFFLOAD):
            continue
        (out / rel).parent.mkdir(parents=True, exist_ok=True); os.link(p, out / rel)
    for rel in heads:
        (out / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / rel, out / rel)
    shutil.copyfile(V87B / PLE, out / PLE)
    if tier:
        shutil.copyfile(V87B / OFFLOAD, out / OFFLOAD)
    study = json.loads((BASE / 'configs/comparison.json').read_text())
    study['name'] = name
    for key in ('gate_match', 'l4_match', 'prefill', 'admission', 'rtx_of'):
        study['arc2'].pop(key, None)
    bps = DESIGNS[design].get('_bps')
    slots = DESIGNS[design].get('_slots') or (round(2180 * 0.96 / bps) if bps else SLOTS)  # formula streams
    if bps:
        study['arc2']['blocks_per_stream'] = bps
    study['arc2']['kv'] = ('v87b mechanism, arc2 serving (scripts/arc2_weights/build_rtx_designs.py, arc2 %s): exact '
                           'reserving gate, %d slots by the formula' % (commit[:8], slots))
    study['generation_slots'] = study['active_limit'] = slots
    # workers up to max_num_seqs (slots + 4, the recipe's own headroom): control.json "slots" can raise the gate live
    seqs = DESIGNS[design].get('_seqs') or slots + 4  # _seqs: room for a live raise past the recipe's headroom
    if '_seqs' in DESIGNS[design]:
        study['arc2']['live_seqs'] = seqs
    study['arc2']['live_slots_max'] = seqs
    study['server']['max_num_seqs'] = seqs
    need = min(256, seqs * (1 + int(study['server'].get('mtp_tokens') or 0)))  # graphs cover every batch
    top = max(study['server']['max_cudagraph_capture_size'], -(-need // 8) * 8)
    sizes = study['server'].get('cudagraph_capture_sizes')
    if sizes and sizes[-1] < top:  # arena.serving: the list must end at max_cudagraph_capture_size
        sizes += list(range(sizes[-1] + 8, top + 1, 8))
    study['server']['max_cudagraph_capture_size'] = top
    trial = study['trials'][0]
    if tier:
        study['cache_experiment']['cpu_cache_gib'] = TIER_GIB
        trial['server_env'] = {**(trial.get('server_env') or {}), **TIER_ENV}
    spec = dict(DESIGNS[design])
    window = spec.pop('_window', None)
    base_fork = spec.pop('_fork', None)
    if base_fork:  # arc2.fork: the variant starts from this base run's first passes (data/arc2/fork/<base>.json)
        rel = f'data/arc2/fork/{base_fork}.json'
        (out / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / rel, out / rel)
        trial['fork'] = rel
    if spec.pop('_bake', None):  # nebius bake (30 Oct): the VM fetches + verifies the assets, powers off, keeps its disk
        study['nebius_bake'] = True
    elif IMAGE:  # boot from the baked image: the runner skips the asset fetch (sizes still checked)
        study['nebius_image_id'] = IMAGE
    score = spec.pop('_score', None)
    if score:  # arc2.score (no play): the served model's log-prob of each recorded reading, 4 augmented views
        rel = f'data/arc2/score/{score}.json'
        (out / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / rel, out / rel)
        trial.update(score_items=rel, score_views=4, score_concurrency=64)
    spec.pop('_slots', None)
    spec.pop('_bps', None)
    spec.pop('_seqs', None)
    copies, stop20 = spec.pop('_copies', None), spec.pop('_stop20', False)
    stop15 = spec.pop('_stop15', False)
    stop30, src_of = spec.pop('_stop30', False), spec.pop('_src_of', None)
    bf16x2 = spec.pop('_bf16x2', False)
    if bf16x2:  # arc2.b300_serving: no overlay, mask, draft or kernel overlay; the model's own BF16 cache, no CPU tier
        keep = ('max_model_len', 'prefix_caching', 'prompt_tokens_details', 'prefix_match_unit', 'cudagraph_capture_sizes',
                'max_cudagraph_capture_size', 'max_num_seqs', 'renderer_num_workers', 'mm_processor_cache_gb')
        study['server'] = {**{k: study['server'][k] for k in keep}, 'mtp_tokens': 0, 'max_num_batched_tokens': 4096,
                           'gpu_memory_utilization': '0.94'}
        rl = json.loads((ROOT.parent / 'rl/dist/rl1-loop-ovf2/configs/comparison.json').read_text())['cache_experiment']
        study['cache_experiment'] = {**rl, 'cpu_cache_gib': 0, 'kv_cache_dtype': 'auto'}  # RL's B300 prefix recipe (MTP 0)
        study['serving_profile'], study['model_datasets'] = None, []
        trial['server_env'] = {'ARENA_GPU_MEMORY_UTILIZATION': '0.94', 'ARENA_READY_GUARD': '300,3000',
                               'ARENA_SETUP_TIMEOUTS': '2700,3600'}
        study['arc2'].update(bf16x2={'cards': 2, 'model': 'expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540'},
                             cache_layout='smallest')
        study['arc2'].pop('weights', None)
        for key in ('draft-head',):
            shutil.rmtree(out / key, ignore_errors=True)
        (out / 'src/arc2/b300_serving.py').parent.mkdir(exist_ok=True)
        for rel in ('src/arc_infra/server.py', 'src/arena/serving_core/serving.py'):
            shutil.copyfile(ROOT / rel, out / rel)
    if window:
        study['windows'] = [window]
    if spec.pop('_base', None) == 'conv1kv':  # conv1's harness: conv1kv's arm, no ROUNDS COMMON settings
        arm = {**json.loads(CONV1.read_text())['trials'][0]['arms'][0], **spec}
        arm['name'] = f'{design}-' + '-'.join(f'{k}{v}' for k, v in spec.items())
    else:
        arm = {**trial['arms'][0], **COMMON, **spec}
        arm['name'] = f'{design}-' + '-'.join(f'{k}{v}' for k, v in {**COMMON, **spec}.items())
    trial.update(arms=[arm], wall_seconds=WALL,
                 marks={'v22-5k': 2300000, '4.6M': 4600000, 'v13-12h': 9260000, '10M': 10000000, '11M': 11000000,
                        '14M': 14000000, '15M': 15000000, '18M': 18000000, '20M': 20000000})
    if copies == 1:  # c120: each task once; reads end at 10M
        trial['copies'] = 1
        trial['marks'] = {'2.5M': 2500000, '5M': 5000000, '7.5M': 7500000, '10M': 10000000}
    if stop20:
        trial['marks'].update({'12.5M': 12500000, '15M': 15000000, '17.5M': 17500000, '20M': 20000000})
    if stop15:
        trial['marks'].update({'12.5M': 12500000, '15M': 15000000})
    if stop30:  # c240 to 30M: 5M marks only, the largest is the stop
        trial['marks'] = {f'{m}M': m * 1_000_000 for m in range(5, 31, 5)}
    if src_of:  # the harness exactly as the named bundle ran it
        shutil.rmtree(out / 'src/arc2')
        shutil.copytree(ROOT / 'dist' / src_of / 'src/arc2', out / 'src/arc2', ignore=shutil.ignore_patterns('__pycache__'))
    if design in SUBSET:  # a short mechanism read (coordinator-ordered): task subset and its own token stop
        ids, stop = SUBSET[design]
        assert set(ids) <= set(trial['tasks']), 'subset outside the roster'
        trial['tasks'] = ids
        trial['marks'] = {f'{m}M': m * 1_000_000 for m in range(1, stop // 1_000_000 + 1)}
    study['wall_seconds'], study['notebook_seconds'] = WALL, WALL + 4800
    study['arc2']['timeout_cap_seconds'] = WALL + 4800 + 300  # the launcher's timeout and VM deadline follow it
    if stop30:  # 30M at the stack check's 500 tok/s floor needs 60,000 s: the crash ceiling, not a time limit
        trial['wall_seconds'] = study['wall_seconds'] = 64800
        study['notebook_seconds'] = 64800 + 4800
        study['arc2']['timeout_cap_seconds'] = 64800 + 4800 + 300
    if bf16x2:  # a crash ceiling inside the launcher's 4 h timeout (bootstrap_arena skips a trial whose wall does not fit)
        study['wall_seconds'], trial['wall_seconds'] = 9600, 9600 # crash ceiling inside a bound
        study['notebook_seconds'] = study['arc2']['timeout_cap_seconds'] = 11940
    (out / 'configs/comparison.json').write_text(json.dumps(study, indent=2) + '\n')
    idn = json.loads((BASE / 'MANIFEST.json').read_text())
    idn.update(study=name, files=manifest(out),
               source=f'arc2-rtx-c240-hx-convro-v3b + arc2 {commit[:8]} harness, design {design}'
                      f'{" + v87b 32 GiB CPU tier" if tier else ""} '
                      f'(scripts/arc2_weights/build_rtx_designs.py)')
    (out / 'MANIFEST.json').write_text(json.dumps(idn, indent=2) + '\n')
    __import__('runpy').run_path(str(out / 'bootstrap.py'))['verify'](out)
    if bf16x2:  # not the RTX stack: check_rtx_stack does not apply
        print(out)
        continue
    check = subprocess.run([sys.executable, str(ROOT / 'scripts/arc2_weights/check_rtx_stack.py'), str(out)]
                           + ([f'--stop-tokens={SUBSET[design][1]}'] if design in SUBSET else [])
                           + (['--stop-tokens=20000000'] if stop20 else [])
                           + (['--stop-tokens=15000000'] if stop15 else [])
                           + (['--stop-tokens=30000000'] if stop30 else []))
    assert check.returncode == 0, 'stack check failed'
    print(out)
