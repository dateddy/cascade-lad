# Task: Stage 2 protocol follow-up (Drain fit_on=stream|train, S2-1..S2-6)
- branch: fix/stage-2-protocol
- status: active
- created: 2026-11-18 14:59
- updated: 2026-11-18 15:05

## Plan
- [x] 1. DrainParser.add (online ids) and fit_on config
- [x] 2. parse_lines with fit_on modes and per-line cache
- [ ] 3. build_windows keeps extra per-line columns
- [ ] 4. prepare uses fit_on, line_pos, template columns, loader key
- [ ] 5. HDFS_2k loader config and stream prefix check (S2-3)
- [ ] 6. stream vs train comparison (S2-6) and script
- [ ] 7. Run on HDFS_2k and record numbers
- [ ] 8. Reword invariant 2

## Next
step 3: build_windows extra_cols

## Files
(none)

## Log
- 2026-11-18 15:02 · DrainParser.add (online ids+template at that moment), n_clusters, fit_on in drain.yaml
- 2026-11-18 15:05 · parse_lines stream/train + parquet cache, parser_check and prepare call updated
