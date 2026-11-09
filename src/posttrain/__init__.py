"""Post-training data arm (ideas.md point 4, paper-notes 2.7; 2026).

Target: a 35B-A3B MoE (Ornith-1.5 or Qwen3.6-35B-A3B). No training compute yet; this package builds
data on CPU. Training data comes only from the 14 dev games of the evolve-v4 split
(``posttrain.split``); the 9 holdout games stay clean for evaluation.
"""
