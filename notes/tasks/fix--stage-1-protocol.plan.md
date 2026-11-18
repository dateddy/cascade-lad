# Plan: Stage 1 protocol follow-up (S1-1..S1-4) on fix/stage-1-protocol
Constraints: data/raw immutable; tests/test_splitting.py and tests/test_grouping.py stay green and unchanged; no new dependency; content is never masked or templated in the loader.
Done when: tests for S1-1..S1-4 (docs/plan.md S1-1..S1-4) pass and the full vf (python .agentkit/skills/verifying-changes/scripts/verify.py) ends with VERDICT: PASS.
Out of scope: BGL/Thunderbird loaders, Stage 2/3 follow-ups, merging into fix/stage-1-3-protocol.

### Step 1: Loader contract: label_granularity and dtype checks
- Files: src/cascadelad/data/loaders/base.py, tests/test_loader_base.py
- Contract: add LABEL_GRANULARITIES = ("line", "group") and a required class attribute LogLoader.label_granularity (no default); register_loader raises ValueError (message contains "label_granularity") if the class lacks it or it is not in LABEL_GRANULARITIES. check_standard, after the column check, raises ValueError naming the column when timestamp is not datetime64, content or group_id is not a string dtype, or label dtype is not int8; the {0,1} value check stays. Update the module docstring: label is the per-line label, or the group label broadcast to every line when label_granularity is "group"; content is the raw message with parameters.
- Tests: test_missing_label_granularity_raises, test_invalid_label_granularity_raises, test_wrong_label_dtype_raises, test_wrong_timestamp_dtype_raises (existing dummy fixtures updated to int8 labels and label_granularity = "group", no assertion removed)
- Out of scope: src/cascadelad/data/loaders/hdfs.py
- Verify: uv run pytest tests/test_loader_base.py -q

### Step 2: HDFS loader: granularity, int8 labels, stable timestamp sort
- Files: src/cascadelad/data/loaders/hdfs.py, tests/test_hdfs_loader.py
- Contract: HDFSLoader.label_granularity = "group". The label column is built as int8. The frame is sorted with sort_values("timestamp", kind="stable") then reset_index(drop=True), so equal timestamps keep file order. The fixture LOG gains a line with an earlier timestamp and an equal-timestamp pair; existing expected counts are updated to match.
- Tests: test_s1_1_dtypes, test_s1_2_group_granularity_broadcast, test_s1_3_content_keeps_parameters, test_s1_4_sorted_stable
- Out of scope: grouping, splitting and prepare code
- Verify: uv run pytest tests/test_hdfs_loader.py tests/test_prepare.py -q

### Step 3: HDFS loader: log raw lines read and extra rows
- Files: src/cascadelad/data/loaders/hdfs.py, tests/test_hdfs_loader.py
- Contract: count raw lines read (n_read, at most max_lines because islice stays) and log at INFO "read %d raw lines, %d rows (%d extra from multi-block lines)" where extra = len(df) - (n_read - no_block). len(df) may exceed max_lines.
- Tests: test_s1_4_max_lines_and_extra_rows_logged
- Out of scope: changing which lines are read or dropped
- Verify: uv run pytest tests/test_hdfs_loader.py -q
