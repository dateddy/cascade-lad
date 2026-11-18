# Task: processed and parse cache keys track data code, raw data and drain3 version; a hit logs real state (D-2e83)
- branch: fix/cache-keys-include-data-code-raw-fingerp
- status: active
- created: 2026-11-05 16:39
- updated: 2026-11-06 14:59

## Plan
- [x] 1. Parse key: data-code hash + drain3 version
- [x] 2. Processed key: code hash, raw fingerprint, drain3 version
- [x] 3. Real [data] logging on a processed hit

## Next
Merge fix branch into feat/stage-3, then Stage 3 into main (gk preflight merge)

## Files
(none)

## Log
- 2026-11-05 16:40 · parsing.py: source_hash, drain3_version in cache_key + 3 tests; vf PASS
- 2026-11-05 16:41 · prepare.py: raw_fingerprint, _key_parts (code, raw, drain3) + 6 tests; vf PASS
- 2026-11-05 16:42 · prepare.py: _hit_stats from cached files + 2 tests; full vf PASS
- 2026-11-06 14:54 · Review fixes: no size under max_lines, build counts on hit; full vf PASS
- 2026-11-06 14:59 · e2e on 7a671bd (runs/e2e2_A..F): fresh miss, rerun hit with (from build) counts, data code edit -> both miss, revert -> hit same key, append past max_lines -> hit; stats identical
