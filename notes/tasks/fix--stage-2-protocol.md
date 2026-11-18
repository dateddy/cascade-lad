# Task: Stage 2 protocol follow-up (Drain fit_on=stream|train, S2-1..S2-6)
- branch: fix/stage-2-protocol
- status: active
- created: 2026-11-18 14:59
- updated: 2026-11-18 16:09

## Plan
- [x] 1. DrainParser.add (online ids) and fit_on config
- [x] 2. parse_lines with fit_on modes and per-line cache
- [x] 3. build_windows keeps extra per-line columns
- [x] 4. prepare uses fit_on, line_pos, template columns, loader key
- [x] 5. HDFS_2k loader config and stream prefix check (S2-3)
- [x] 6. stream vs train comparison (S2-6) and script
- [x] 7. Run on HDFS_2k and record numbers
- [x] 8. Reword invariant 2

## Next
merge fix/stage-2-protocol into fix/stage-1-3-protocol

## Files
(none)

## Log
- 2026-11-18 15:02 · DrainParser.add (online ids+template at that moment), n_clusters, fit_on in drain.yaml
- 2026-11-18 15:05 · parse_lines stream/train + parquet cache, parser_check and prepare call updated
- 2026-11-18 15:12 · build_windows extra_cols
- 2026-11-18 15:16 · prepare: loader key, line_pos, template col, parse all rows, fit_on in [data]
- 2026-11-18 15:18 · hdfs_2k loader config, prefix_mismatches (S2-3)
- 2026-11-18 15:21 · compare_fit_on, run_compare, script, fit_on_hdfs_2k experiment
- 2026-11-18 15:44 · HDFS_2k: grouping_accuracy 0.9975 (stream), S2-3 mismatches 0, S2-6 joined 442/442, unk_train_invocab_stream 0, 2nd run processed cache hit
- 2026-11-18 16:09 · invariant 2 reworded in AGENTS.md; parsing.py docstring
