# Task: Stage 1 protocol follow-up (S1-1..S1-4) on fix/stage-1-protocol
- branch: fix/stage-1-protocol
- status: active
- created: 2026-11-18 14:21
- updated: 2026-11-18 14:30

## Plan
- [x] 1. Loader contract: label_granularity and dtype checks
- [x] 2. HDFS loader: granularity, int8 labels, stable timestamp sort
- [x] 3. HDFS loader: log raw lines read and extra rows

## Next
all steps done: final vf passed, ready to merge into fix/stage-1-3-protocol when user asks

## Files
(none)

## Log
- 2026-11-18 14:22 · base.py: label_granularity required + dtype checks; hdfs.py: granularity, int8, stable sort (pulled from step 2 so code stays importable)
- 2026-11-18 14:29 · S1-1..S1-4 tests for HDFS loader; fixed duplicated docstring lines in base.py
- 2026-11-18 14:30 · hdfs.py logs raw lines read, rows and extra multi-block rows; base.py docstring alignment
