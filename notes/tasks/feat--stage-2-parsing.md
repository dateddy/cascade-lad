# Task: Drain log parser with train-only fitting and a hash-keyed cache, checked on HDFS_2k (grouping accuracy ≥ 0.95)
- branch: feat/stage-2-parsing
- status: active
- created: 2026-11-04 09:25
- updated: 2026-11-04 09:29

## Plan
- [x] 1. Parser config, DrainParser, grouping accuracy
- [x] 2. Hash-keyed parse cache
- [x] 3. HDFS_2k download (plain files, pinned commit + md5)
- [x] 4. Parser check entry point
- [x] 5. Real-data check and stage close-out

## Next
merge feat/stage-2-parsing after user review

## Files
(none)

## Log
- 2026-11-04 09:26 · DrainParser fit/transform(UNK)/templates + grouping_accuracy, drain.yaml, drain3 0.9.11; 7 tests pass, vf PASS
- 2026-11-04 09:27 · cache_key + parse_cached (tmp dir + rename), hit skips fit; 5 cache tests pass, vf PASS
- 2026-11-04 09:27 · fetch_files + hdfs_2k.yaml (loghub sha 374038fa pinned, md5s); 3 tests, real download ok, vf PASS
- 2026-11-04 09:29 · parser_check.run_check + scripts/check_parser.py + parser: null default; 2 tests, vf PASS
- 2026-11-04 09:29 · HDFS_2k grouping_accuracy=0.9975 (16 pred vs 14 true templates, 0 UNK); rerun cache hit; vf PASS
