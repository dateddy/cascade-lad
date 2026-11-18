# Plan: processed and parse cache keys track data code, raw data and drain3 version; a hit logs real state (D-2e83)
Constraints: edit only src/cascadelad/data/prepare.py, src/cascadelad/data/parsing.py and their tests. No change to splitting, vocab, grouping or loader logic. No new dependencies. Show the diff before each commit and wait for a yes.
Done when: changing any .py file under src/cascadelad/data/, the raw log or label file, or the drain3 version changes processed_key; the code and drain3 changes also change parsing.cache_key; on a processed hit, [data] logs real status and stats recomputed from the cached files, not meta.json stats; each case has a test; full vf exits 0.
Out of scope: loader API changes (e.g. a raw_files() method), BGL fingerprint rules, Stage 3 merge review.
Decisions: D-2e83

Context: on 2026-11-05 we confirmed that editing splitting code or truncating HDFS.log left processed_key unchanged, so stale processed data came back as a cache hit. The hit path also replays the stats meta.json saved at build time (including that build's "parse cache" status), and that misleads.
User choices: new branch off stage-3; on a hit, recompute stats from the cached files; the parse key gets no raw fingerprint (the fit and apply contents already cover raw changes).

## 1. Parse key: data-code hash + drain3 version
Files: src/cascadelad/data/parsing.py, tests/test_parsing_cache.py
Do:
- Add `DATA_DIR = Path(__file__).resolve().parent`.
- Add `source_hash(root: Path = DATA_DIR) -> str`: sha256 over every `*.py` under root (rglob, so loaders/ is included), sorted by POSIX relative path. For each file it hashes `relpath\0`, the byte length and the bytes. Returns 16 hex chars.
- Add `drain3_version() -> str` that returns `importlib.metadata.version("drain3")`. Also use it in the existing meta dict.
- In `cache_key`, add `"code": source_hash()` and `"drain3": drain3_version()` to the hashed json (cfg_part stays without parsed_dir). Write both into the parse meta.json as `code_hash` and `drain3_version`.
Tests (new):
- `test_source_hash_tracks_nested_py_files`: build a tmp tree with a.py and sub/b.py. Editing b.py changes the hash, and adding a .txt file does not.
- `test_key_changes_with_code_hash` and `test_key_changes_with_drain3_version`: monkeypatch `parsing.source_hash` / `parsing.drain3_version` to return a different value, and the key changes.
Verify: python .agentkit/skills/verifying-changes/scripts/verify.py --affected
Done: the 3 new tests pass, and the existing parse-cache tests still pass unchanged

## 2. Processed key: code hash, raw fingerprint, drain3 version
Files: src/cascadelad/data/prepare.py, tests/test_prepare.py
Do:
- Import `source_hash, drain3_version` from parsing.
- Add `raw_fingerprint(dataset_cfg: Mapping) -> dict`. For `raw_dir/log_file` it records `{"size": file size in bytes, "md5": md5 of the first max_lines lines read in binary mode via islice (whole file in 1 MiB chunks when max_lines is null)}`. Binary lines never cover less than what the text-mode loader reads, so the fingerprint stays safe. If `label_file` is set, it also records size + md5 of the whole label file. Keys: `log_file`, `label_file`.
- `processed_key(cfg)` adds `"code": source_hash()`, `"raw": raw_fingerprint(cfg["dataset"])` and `"drain3": drain3_version()` to `parts`. Refactor: a private `_key_parts(cfg)` builds the dict. prepare() calls it once and stores it as `meta["key_parts"]`.
Tests (new, using the existing make_cfg/write_raw):
- `test_key_changes_with_data_code`: monkeypatch `prepare_mod.source_hash`, and the key changes.
- `test_key_changes_with_drain3_version`: monkeypatch `prepare_mod.drain3_version`, and the key changes.
- `test_key_changes_when_raw_log_edited_or_truncated`: changing one char in a line inside max_lines changes the key, and truncating the last line changes it too.
- `test_key_changes_when_label_file_edited`: flipping one label changes the key.
- `test_key_ignores_lines_past_max_lines_but_not_size`: set max_lines=2 and change a char after line 2 at the same size, and the key stays the same. Append a line, and the key changes.
- `test_stale_cache_not_reused_after_raw_change`: prepare, then edit raw, then prepare. The result is a new dir and the log shows `processed cache: miss`.
Verify: python .agentkit/skills/verifying-changes/scripts/verify.py --affected
Done: 6 new tests pass, and test_key_ignores_parsed_dir_but_not_split still passes

## 3. Real [data] logging on a processed hit
Files: src/cascadelad/data/prepare.py, tests/test_prepare.py
Do: replace the meta.json stats replay with `_hit_stats(cfg, key, out_dir) -> dict`. It reads the 3 parquet files and vocab.json and returns `raw_dir`, `processed key`, `processed dir`, `processed cache: hit`, `parse cache: not run`, `groups <s>` (row count), `anomaly % <s>` (label mean, same format as the miss path), `vocab size`, and `test unk rate` (share of test event_ids equal to vocab.unk_id, 4 decimals; this matches Vocab.unk_rate because encode maps every unseen token to unk_id). It leaves out purged / dropped counts because those exist only at build time. The miss path is unchanged except for the new `processed dir` line.
Tests:
- Extend `test_second_run_is_cache_hit_without_loading`: the hit log contains `parse cache: not run` and `groups train: 5`, and its `test unk rate` value equals the miss run's value.
- New `test_hit_stats_come_from_files_not_meta`: overwrite meta.json stats with junk (e.g. `"vocab size": 999`, `"parse cache": "miss"`), rerun, and the log shows the real vocab size, no 999, and `parse cache: not run`.
Verify: python .agentkit/skills/verifying-changes/scripts/verify.py
Done: full vf VERDICT PASS. Show `git diff` for steps 1–3 and commit only after the user says yes.

## End-to-end check (after the user says yes to the commits)
Run `python scripts/prepare_data.py` twice on the real HDFS 100k slice: first a miss (new key, because the code hash changed), then a hit whose [data] block shows `parse cache: not run` and real stats. A one-char edit to a test copy of the raw dir must produce a miss. data/raw is never touched: the check runs on a copy, using a dataset.raw_dir override.
