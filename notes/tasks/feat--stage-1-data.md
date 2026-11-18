# Task: Stage 1: download HDFS_v1 and load it into the standard `timestamp, content, group_id, label` DataFrame
- branch: feat/stage-1-data
- status: active
- created: 2026-11-04 09:00
- updated: 2026-11-04 09:07

## Plan
- [x] 1. Loader interface and registry
- [x] 2. HDFS loader and dataset config
- [x] 3. Zenodo download with pinned md5
- [x] 4. Real-data check and stage close-out

## Next
merge feat/stage-1-data after user review

## Files
(none)

## Log
- 2026-11-04 09:00 · LogLoader base, registry, check_standard; 5 tests pass, vf PASS
- 2026-11-04 09:02 · HDFSLoader + configs/dataset/hdfs.yaml; 6 tests pass, vf PASS
- 2026-11-04 09:04 · download.py + scripts/download_data.py, md5 76a24b4d pinned from Zenodo API; 4 tests pass, vf PASS
- 2026-11-04 09:07 · Real download md5 ok, rerun no-op; 100k slice cols exact, label mean 0.0305; blk_-1001553972418305662/blk_-1010952805175971965 Normal, blk_-1018108268208665701/blk_-1188841150645918532 Anomaly match CSV; vf PASS
