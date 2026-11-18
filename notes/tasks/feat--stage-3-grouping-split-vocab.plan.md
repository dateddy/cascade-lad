# Plan: Stage 3 — HDFS blocks grouped, split chronologically, encoded with a train-only vocab, written to data/processed/<hash>/
Constraints: Invariants 1–3 and 5 (chronological split, templates/vocab fit on normal train only, train has no anomalous block, labels only carried for later threshold/eval). Reuse `parse_cached` (src/cascadelad/data/parsing.py) and the loader registry; no magic numbers (all params in YAML); `scripts/` stays thin. splitting.py and vocab.py diffs are shown and explained to the user before commit (line-by-line review). Window columns follow D-2c8d: `group_id, event_ids, timestamp` (+ `label`, stripped by callers before fit/score).
Done when: `uv run python scripts/prepare_data.py experiment=prepare_hdfs run_name=<name>` creates `data/processed/<hash>/{train,val,test}.parquet`, `vocab.json`, `meta.json`; tests/test_splitting.py and tests/test_grouping.py pass (test line timestamps > val > train, vocab from train only, train has no anomalous block); run.log `[data]` prints split sizes, purged count, anomaly %, vocab size, test `<UNK>` rate; a second run reports a processed-cache hit; full `vf` PASS.
Out of scope: sliding/time windows (BGL), random-split comparison, detectors, evaluation module (Stage 4), changing the Drain config.
Decisions (settled with the user, to record with `nt decide` after acceptance): boundary-straddling blocks are purged; Drain + vocab are fit on normal train blocks only; pyarrow is added for parquet.

### Step 1: Dependency and configs
- Mode: solo
- Files: pyproject.toml, uv.lock, configs/config.yaml, configs/split/chrono.yaml (new), configs/vocab/default.yaml (new), configs/experiment/prepare_hdfs.yaml (new)
- Contract: `uv add pyarrow`. `configs/split/chrono.yaml` (`# @package split`): `name: chrono`, `train_ratio: 0.7`, `val_ratio: 0.1`, `test_ratio: 0.2`. `configs/vocab/default.yaml` (`# @package vocab`): `pad_token: '<PAD>'`, `unk_token: ${parser.unk_token}`, `min_count: 1`. configs/config.yaml: `split: null` and `vocab: null` in defaults, `paths.processed_dir: data/processed`. `configs/experiment/prepare_hdfs.yaml` (`# @package _global_`): defaults `override /dataset: hdfs`, `override /parser: drain`, `override /split: chrono`, `override /vocab: default`.
- Tests: test_train_creates_run_dir, test_perfect_grouping_then_cache_hit (existing; must still pass)
- Out of scope: any code under src/; Drain config
- Verify: uv run python -c "import pyarrow" ; python .agentkit/skills/verifying-changes/scripts/verify.py

### Step 2: Grouping (tests first)
- Mode: solo
- Files: tests/test_grouping.py (new), src/cascadelad/data/grouping.py (new)
- Contract: `session_bounds(lines: pd.DataFrame) -> pd.DataFrame` with columns `group_id, start, end, label` (start/end = min/max line timestamp; label = the group's label; ValueError if a group has more than one label). `build_windows(lines: pd.DataFrame, token_col: str) -> pd.DataFrame` with columns `group_id, templates (list[str]), timestamp (= start), label`; events keep original row order; groups ordered by (timestamp, group_id); empty input raises.
- Tests: test_session_bounds_start_end_label, test_mixed_label_group_raises, test_windows_keep_row_order_within_group, test_windows_columns_timestamp_and_order, test_ties_broken_by_group_id_and_deterministic, test_empty_input_raises
- Out of scope: splitting, vocab, loaders
- Verify: uv run pytest tests/test_grouping.py -q

### Step 3: Chronological split with purge (tests first; diff reviewed by user)
- Mode: solo
- Files: tests/test_splitting.py (new), src/cascadelad/data/splitting.py (new)
- Contract: `chronological_split(bounds: pd.DataFrame, cfg: Mapping) -> SplitResult` (dataclass: `train, val, test: list[str]` group ids, `n_purged: dict[str,int]`, `n_train_anomalous_dropped: int`). Ratios sum to 1, else ValueError. Sort by (start, group_id); cut by count into train/val/test candidates. `t1` = start of first val candidate, `t2` = start of first test candidate. Keep train only if `end < t1`, val only if `end < t2`; test kept as is; dropped ones counted in `n_purged`. Then drop anomalous blocks from train (invariant 3) and count them. Val and test keep both labels. No random option. Diff shown and explained to the user before commit.
- Tests: test_split_membership_and_counts, test_no_temporal_leakage, test_train_has_no_anomalous_block_but_val_test_keep_both_labels, test_groups_are_disjoint_and_cover_only_input, test_deterministic_and_independent_of_input_order, test_start_ties_broken_by_group_id, test_ratios_must_sum_to_one, test_too_few_blocks_for_three_splits_raises
- Out of scope: random-split comparison, grouping, vocab
- Verify: uv run pytest tests/test_splitting.py -q

### Step 4: Vocab (tests first; diff reviewed by user)
- Mode: solo
- Files: tests/test_vocab.py (new), src/cascadelad/data/vocab.py (new)
- Contract: class `Vocab`: `build(sequences: Iterable[list[str]], cfg: Mapping) -> Vocab` (classmethod; ids pad=0, unk=1, then tokens with count ≥ min_count sorted by (-count, token); the parser's `<UNK>` never gets its own id beyond unk), `encode(tokens: list[str]) -> list[int]` (unseen → unk id), `unk_rate(sequences) -> float` (share of encoded events equal to unk id; empty raises), `size`, `to_json(path)`, `from_json(path)`. Diff shown and explained to the user before commit.
- Tests: test_ids_pad_unk_then_by_frequency, test_count_ties_broken_by_token, test_unseen_token_maps_to_unk, test_parser_unk_token_never_gets_its_own_id, test_built_only_from_given_sequences, test_min_count_filters_rare_tokens, test_deterministic, test_json_round_trip, test_unk_rate_hand_made, test_unk_rate_empty_raises
- Out of scope: splitting, grouping, parser
- Verify: uv run pytest tests/test_vocab.py -q

### Step 5: Prepare pipeline and script
- Mode: pipeline
- Files: src/cascadelad/data/prepare.py (new), scripts/prepare_data.py (new), tests/test_prepare.py (new), README.md
- Contract: `processed_key(cfg) -> str`: 16-hex sha256 of sorted JSON of dataset, parser (minus `parsed_dir`), split, vocab cfgs. `prepare(cfg: Mapping, logger) -> Path`: all outputs in `<processed_dir>/<key>/` → log `processed cache: hit`, return. Else loader → `session_bounds` → `chronological_split` → `parse_cached(fit=normal train, apply=all kept)` → `build_windows` → `Vocab.build(train)` → `event_ids`; write `{train,val,test}.parquet` (`group_id, event_ids, timestamp, label`), `vocab.json`, `meta.json` via a `.tmp` dir renamed last. `[data]` log: raw dir, key, parse hit, groups/purged/anomaly % per split, train anomalous dropped, vocab size, test `<UNK>` rate. `scripts/prepare_data.py`: Hydra → `start_run` → `prepare(...)`, no logic. README: prepare command.
- Tests: test_outputs_leakage_and_vocab, test_second_run_is_cache_hit_without_loading, test_key_ignores_parsed_dir_but_not_split
- Out of scope: grouping, splitting and vocab logic; Drain config; detectors
- Verify: uv run pytest tests/test_prepare.py -q ; python .agentkit/skills/verifying-changes/scripts/verify.py

### Step 6: Real run on HDFS (100k lines)
- Mode: solo
- Files: notes/decisions.md
- Contract: `uv run python scripts/prepare_data.py experiment=prepare_hdfs run_name=2026-11-04_hdfs_prepare_100k`, then the same with `run_name=2026-11-04_hdfs_prepare_100k_rerun`. First run.log shows `processed cache: miss` and the full `[data]` section; second shows `processed cache: hit`; `data/processed/<hash>/` holds the 5 files. Split sizes, anomaly %, vocab size and test `<UNK>` rate reported to the user. The three decisions recorded with `nt decide`.
- Tests: none new (real-data run); the run.log checks in Contract are the done criterion
- Out of scope: any code change; data/raw (immutable)
- Verify: python .agentkit/skills/verifying-changes/scripts/verify.py
