# Task: Stage 3 — HDFS blocks grouped, split chronologically, encoded with a train-only vocab, written to data/processed/<hash>/
- branch: feat/stage-3-grouping-split-vocab
- status: done
- created: 2026-11-04 09:44
- updated: 2026-11-20 00:14

## Plan
- [x] 1. Dependency and configs
- [x] 2. Grouping (tests first)
- [x] 3. Chronological split with purge (tests first; diff reviewed by user)
- [x] 4. Vocab (tests first; diff reviewed by user)
- [x] 5. Prepare pipeline and script
- [x] 6. Real run on HDFS (100k lines)

## Next
Merge Stage 3 into main (gk preflight merge)

## Files
(none)

## Log
- 2026-11-04 09:46 · pyarrow added; split/vocab/prepare_hdfs configs; vf PASS
- 2026-11-04 09:47 · grouping.py + 6 tests; vf PASS
- 2026-11-04 09:49 · splitting.py + 9 tests; user reviewed; vf PASS
- 2026-11-04 09:50 · vocab.py + 10 tests; user reviewed; vf PASS
- 2026-11-04 09:52 · prepare.py + scripts/prepare_data.py + 3 tests + README; vf PASS
- 2026-11-04 09:53 · real HDFS 100k run: miss then hit; 3 decisions recorded
- 2026-11-06 15:06 · Fast-forwarded fix/cache-keys-include-data-code-raw-fingerp (0a06b2d, 7a671bd: D-2e83, D-e14d, D-23c3) into Stage 3; full vf PASS
- 2026-11-20 00:14 · auto-closed (branch deleted)
