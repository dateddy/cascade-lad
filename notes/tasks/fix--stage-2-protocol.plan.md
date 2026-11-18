# Plan: Stage 2 protocol follow-up (Drain fit_on=stream|train, S2-1..S2-6)

## Context
Stage 2 is merged, but `docs/plan.md` lines 65-80 add a protocol follow-up on `fix/stage-1-3-protocol`: Drain gets `fit_on: stream` (default, online ids, D-fc1a) and `fit_on: train` (RQ4 ablation). Today `parsing.py` only does fit-on-train with dedup, prepare parses only split lines, and processed outputs drop line positions, so the RQ4 stream-vs-train join is impossible. Stage 1 follow-up was done on `fix/stage-1-protocol` and merged into `fix/stage-1-3-protocol`; this task follows the same pattern on `fix/stage-2-protocol`.

Constraints: research invariants 1, 3-7 untouched; vocab stays train-only; leakage tests stay green; `content` never modified; no new dependency (pyarrow already used by prepare).
Done when: S2-1..S2-6 in docs/plan.md pass; `vf` exits 0; on HDFS_2k the compare script prints the S2-3 mismatch count (0) and the S2-6 join coverage + `<UNK>`-train/in-vocab-stream count.
Out of scope: A1/D5 NaN check (G1), "windows flagged only under train" (needs a detector, Stage 5+), full-HDFS run.
Decisions: D-fc1a, D-6f30, D-2982

User answers (2026-11-18), to record with `nt decide` right after acceptance (topic parsing / data):
- Dataset configs get a `loader:` key; HDFS_2k loads through the hdfs loader with the full HDFS label file.
- Under stream, a line's `template` text is the cluster template returned when that line was added (never the final one).
- S2-6 comparison lives in `src/cascadelad/data/fit_on_compare.py` + `scripts/compare_fit_on.py`.
- AGENTS.md invariant 2 reworded (text in Step 6).

Assumptions:
- Stream parses every row (no dedup, per plan text); train mode keeps dedup for fit and match (results identical, faster).
- "Line position" = `line_pos`, the row index of the loader output (after multi-block explosion and stable time sort); (group_id, line_pos) is unique.
- Under train, `<UNK>` lines get `template = unk_token`.
- Both modes parse all loaded rows (before purging), so a stream run is independent of the split.

### Step 1: DrainParser.add (online ids) and fit_on config
- Files: src/cascadelad/data/parsing.py, configs/parser/drain.yaml, tests/test_parsing.py
- Contract:
  - `drain.yaml`: add `fit_on: stream`.
  - `DrainParser.add(contents) -> list[tuple[str, str]]`: per line `add_log_message`; returns (`f"E{cluster_id}"`, `template_mined`) at that moment; no dedup.
  - Add `n_clusters` property. `fit`/`transform`/`templates`/`parse_cached` unchanged.
- Tests: test_stream_ids_fixed_when_added, test_stream_template_is_template_at_that_moment, test_stream_prefix_stable, test_template_ids_are_E_strings, test_train_cluster_count_unchanged_after_apply (S2-4), test_train_unseen_is_unk_string
- Out of scope: caching, prepare.py, parser_check.py.

### Step 2: parse_lines with fit_on modes and per-line cache
- Files: src/cascadelad/data/parsing.py, src/cascadelad/data/parser_check.py, tests/test_parsing_cache.py, tests/test_parser_check.py, README.md
- Contract:
  - `parse_lines(contents, cfg, fit_mask=None) -> (DataFrame[template_id, template], cache_hit)` replaces `parse_cached`; rows align with `contents`.
  - `stream`: `fit_mask` must be None (ValueError otherwise); one `add()` pass.
  - `train`: `fit_mask` required, same length, one True or more (ValueError otherwise); fit on unique masked contents, `transform` unique contents, map back; `<UNK>` template = `unk_token`; meta records both cluster counts. Other `fit_on` -> ValueError.
  - `cache_key(cfg, contents, fit_mask)`: cfg minus parsed_dir, code hash, drain3 version, all contents; under `train` also the packed mask bytes (D-6f30); under `stream` no mask.
  - Cache files `lines.parquet`, `templates.json`, `meta.json`, atomic write as today.
  - `run_check` uses `parse_lines`; README line 41 updated.
- Tests: test_fit_mask_rules, test_key_changes_with_fit_on_and_each_drain_param, test_train_key_changes_with_fit_mask, test_stream_key_ignores_fit_mask, test_miss_then_hit_without_fitting, test_na_and_null_round_trip_as_strings, test_perfect_grouping_then_cache_hit, test_content_mismatch_raises
- Out of scope: prepare.py, grouping.py.

### Step 3: build_windows keeps extra per-line columns
- Files: src/cascadelad/data/grouping.py, tests/test_grouping.py
- Contract: `build_windows(lines, token_col, extra_cols: Sequence[str] = ())` adds one list column per name in `extra_cols`, same row order as `templates`. Default output unchanged.
- Tests: test_extra_cols_are_aligned_lists
- Out of scope: session_bounds, ordering rules.

### Step 4: prepare uses fit_on, line_pos, template columns, loader key
- Files: src/cascadelad/data/prepare.py, configs/dataset/hdfs.yaml, tests/test_prepare.py
- Contract:
  - `hdfs.yaml`: `loader: hdfs`; prepare calls `get_loader(cfg["dataset"]["loader"], ...)`.
  - After load: `line_pos` = `range(len(lines))` (int64). Split as today.
  - Parse all loaded rows: stream -> `parse_lines(contents, parser_cfg)`; train -> `fit_mask = group_id in split.train`. Assign `template_id`, `template` (`content` untouched), then keep split groups.
  - `build_windows(..., "template_id", extra_cols=("line_pos",))`; `OUT_COLUMNS` adds `line_pos`.
  - `[data]` stats (miss and hit) add `fit_on` (S2-2).
- Tests: test_parsing_keeps_content_byte_identical (S2-1), test_data_log_prints_fit_on, test_stream_ratio_change_is_parse_cache_hit, test_train_ratio_change_is_parse_cache_miss (S2-5), test_line_pos_unique_with_group_id; existing prepare tests get the `loader` key and stay green.
- Out of scope: splitting.py, vocab.py logic.

### Step 5: HDFS_2k loader config and stream prefix check (S2-3)
- Files: configs/dataset/hdfs_2k.yaml, src/cascadelad/data/fit_on_compare.py (new), tests/test_fit_on_compare.py (new)
- Contract:
  - `hdfs_2k.yaml` adds `loader: hdfs`, `label_file: ../hdfs/preprocessed/anomaly_label.csv`, `max_lines: null`, `timestamp_format`, `block_regex`, `label_column`, `block_column`, `label_map` (same values as hdfs.yaml); existing keys stay.
  - `prefix_mismatches(contents, cfg, train_ratio) -> int`: stream ids of the first `int(n*ratio)` contents parsed alone vs within the full stream (`DrainParser.add`, no cache).
- Tests: test_prefix_mismatches_zero_on_stream, test_hdfs_2k_cfg_matches_hdfs_regexes
- Out of scope: compare_fit_on, scripts.

### Step 6: stream vs train comparison (S2-6) and script
- Files: src/cascadelad/data/fit_on_compare.py (new), scripts/compare_fit_on.py (new), configs/experiment/fit_on_hdfs_2k.yaml (new), tests/test_fit_on_compare.py (new)
- Contract:
  - `compare_fit_on(stream_dir, train_dir, split="test") -> dict`: explode `event_ids` and `line_pos` of both parquet files, join on (group_id, line_pos); raise if group sets differ or the join is not one-to-one; returns `lines`, `joined`, `unk_stream`, `unk_train`, `unk_train_invocab_stream` (each dir uses its own vocab.json unk id).
  - Script: Hydra, `start_run`; runs `prepare` with `parser.fit_on=stream`, then `train`; `prefix_mismatches` with `compare.prefix_ratio: 0.7`; `compare_fit_on`; logs a `[compare]` section, writes metrics.json.
- Tests: test_compare_counts_unk_train_invocab_stream, test_compare_raises_on_different_groups, test_compare_raises_on_duplicate_keys
- Out of scope: detector-based "windows flagged only under train".

### Step 7: Run on HDFS_2k and record numbers
- Files: docs/plan.md
- Contract: `python scripts/check_parser.py dataset=hdfs_2k parser=drain run_name=<d>_hdfs2k_parsecheck`; `python scripts/compare_fit_on.py experiment=fit_on_hdfs_2k run_name=<d>_hdfs2k_fit_on` twice (second run: processed cache hit). Report S2-3 mismatches (expect 0), S2-6 coverage (joined == lines) and the count.
- Tests: none new.
- Out of scope: code changes; docs/plan.md is edited only if a criterion needs wording fixed.
- Verify: `vf`

### Step 8: Reword invariant 2
- Files: AGENTS.md, src/cascadelad/data/parsing.py
- Contract: AGENTS.md invariant 2 -> "Vocab is built on train only; unseen template ids map to `<UNK>`. Templates come from Drain `fit_on=stream` (online, each line's id fixed when it is read; default) or `fit_on=train` (fit on normal train lines, frozen; ablation)." Module docstring says the same.
- Tests: none.
- Out of scope: any other invariant; parsing.py code (module docstring only).

## Verification
- Per step: `vf --affected`, then full `vf` (VERDICT PASS) before each commit.
- End to end: Step 7 commands; second compare run logs `processed cache: hit`; changing only `split.train_ratio` with stream logs `parse cache: hit`, with `parser.fit_on=train` logs `miss`.
- After all steps: merge `fix/stage-2-protocol` into `fix/stage-1-3-protocol` (same style as Stage 1, check `git log --first-parent --merges -5`).
