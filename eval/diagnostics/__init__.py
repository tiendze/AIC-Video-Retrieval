"""Diagnostics — measurements that explain WHY a channel scores what it
scores, as opposed to `eval/run.py` which measures WHAT it scores.

Deliberately separate from `eval/run.py` (CLAUDE.md §6: "eval/run.py là entry
point duy nhất của evaluation"): nothing in here emits Recall/MRR/AIC Final
Score or any number that could be quoted as a retrieval-quality claim. These
modules answer a different question — "is the embedding space degenerate?" —
and their output belongs in `reports/diagnostics/`, never in a metrics report.
"""
