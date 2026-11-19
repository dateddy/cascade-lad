# Decisions archive

Verbatim entries moved out of decisions.md on 2026-11-19 (D-4a1e). Not read by notes.py; grep it.
The complete log before the cleanup is decisions.md at commit b53a4de.

| Entry | Now in |
|---|---|
| D-7a93 | D-2c8d |
| D-26b9 | D-5b72 |
| D-8314 | D-fc1a |
| D-2e83 | D-6c03 |
| D-b52f | D-c83f |
| D-bf45 | backlog Stage 6-7 |
| D-149b | backlog Stage 6-7 |
| D-9417 | D-cbb3 + backlog |
| D-bb3b | dropped (D-f150) |
| D-bd28 | backlog G1 |
| D-3ee7 | backlog G1 |
| D-eac4 | backlog G1 |
| D-109c | backlog G1 |
| D-346d | D-7e94 + backlog G3 |
| D-c9f0 | backlog: before Stage 5 |
| D-cd49 | backlog G3 |
| D-c333 | backlog G2 |
| D-2be2 | D-6c03 |
| D-d7e1 | backlog (groups) |
| D-5634 | backlog (order) |
| D-e14d | D-6c03 + backlog G1 |
| D-23c3 | D-6c03 |
| D-376b | done (AGENTS.md invariant 2) |
| D-a0f3 | scope reason, fix/stage-2-protocol merged |
| D-a0b1 | done (AGENTS.md invariants 2-3) |
| D-0d06 | D-7e94 + backlog G3 |
| D-f150 | done |
| D-eb1d | scope reason, fix/stage-3-protocol merged |
| D-6db5 | scope reason, fix/stage-3-protocol merged |
| D-0037 | scope reason, fix/stage-3-protocol merged |

## D-7a93 · 2026-11-03 · detector-interface · feat/stage-0-skeleton
Detector.fit(train: DataFrame, val: DataFrame|None) / score(windows: DataFrame) -> 1-D float ndarray, higher = more anomalous; window columns group_id, event_ids, timestamp; no label read, no thresholding in score
Why: Fixed in Stage 0 so DeepLog can be written in parallel; val needed for early stopping on val loss (invariant 4)

## D-26b9 · 2026-11-04 · parsing · feat/stage-2-parsing
Drain via drain3; parser API is fit(train contents) + transform(any contents, match only, unmatched -> <UNK>)
Why: Invariant 2: templates fit on train only; drain3 match() does not update the tree

## D-8314 · 2026-11-04 · parsing · feat/stage-3-grouping-split-vocab
Drain parser and vocab are fit on normal train blocks only (anomalous train blocks dropped before fit)
Why: Invariants 2 and 3: fit only on what the detector trains on

## D-2e83 · 2026-11-05 · caching · feat/stage-3-grouping-split-vocab
processed_key (and parse cache key) must include: hash of src/cascadelad/data/ source files, raw file fingerprint (size + md5 of the part read), drain3 version. On a cache hit, [data] must log the real state of this run, not replay meta.json stats. Implementation waits until the user finishes setup; then plan it.
Why: Verified 2026-11-05: changing splitting code or truncating HDFS.log kept the same key and silently returned stale processed data (cache hit, git_commit.txt pointing at the new commit).

## D-b52f · 2026-11-05 · data · feat/stage-3-grouping-split-vocab
The max_lines slice (default 100k) is only a smoke test for pipeline and baseline runs (data flows end to end). Its metrics are never main results or interpreted. Slice size may be changed per baseline model. Main results use full data.
Why: On 100k lines train spans ~9 min, <UNK> comes from lifecycle phase (first Deleting block at 20:45:24 > t_val), val has 18 anomalies, test tail blocks are truncated but keep full-block labels.

## D-bf45 · 2026-11-05 · ablation · feat/stage-3-grouping-split-vocab
Deferred until PCA and DeepLog run on full HDFS (Stage 6-7): contamination ablation with a YAML flag (e.g. split.train_normal_only) training on all train blocks without label filtering. Whether to run it stays open. Touches splitting and invariant 3: needs explicit user yes.
Why: Supports or refutes the unsupervised claim; not needed before baselines exist.

## D-149b · 2026-11-05 · evaluation · feat/stage-3-grouping-split-vocab
Deferred to Stage 7, before any RQ4 work: on full HDFS check correlation between <UNK> in a test session and its label; if present report F1 split by with/without <UNK>. RQ4 needs real log evolution (BGL per spec, or synthetic template changes in LogRobust style); HDFS_v1 is one software version, so its <UNK> is not evidence of evolution.
Why: On 100k slice <UNK> came from lifecycle phase (Deleting block), not code change; a UNK-label correlation would inflate detector F1.

## D-9417 · 2026-11-05 · evaluation · feat/stage-3-grouping-split-vocab
Open, proposed defaults (not decided): unit of evaluation = session (block); threshold = max F1 on val, same procedure for every method (invariant 5); PCA input = per-session template count vector; DeepLog session score = max over windows of -log p(actual next key). Decide each at its stage (4, 5, 6) after explanation.
Why: User wants to decide when building models with enough background; recorded so defaults are not adopted silently.

## D-bb3b · 2026-11-05 · data-validation · fix/cache-keys-include-data-code-raw-fingerp
Planned fix A1+D5 (not implemented): right after the template_id map in prepare.py, raise ValueError if any template_id is NaN (count + 3 example contents). No NaN handling (drop/UNK/keep): NaN means a code bug. Test: monkeypatch parse_cached to omit one content, expect the raise. Optional guard in Vocab.build rejecting non-str tokens (user reviews vocab diff).
Why: Verified 2026-11-05: a missing content silently becomes NaN; in train it gets its own vocab id (fake template, no symptom); only in val/test it counts as <UNK> and biases RQ4.

## D-bd28 · 2026-11-05 · data-validation · fix/cache-keys-include-data-code-raw-fingerp
Planned test D1 (not implemented): add to the test_hdfs_loader fixture a line naming the same block twice (Deleting block blk_300 file /mnt/blk_300) and assert it yields exactly 1 row. Keep the distinct-blocks test as a guard for other datasets. No code change.
Why: HDFS_v1 has 0 lines with two distinct blocks but 1,402,056 lines naming the same block twice (81% of 575,061 blocks). Removing dict.fromkeys in hdfs.py passes all 16 current tests; the new test fails (2 rows != 1).

## D-3ee7 · 2026-11-05 · data-validation · fix/cache-keys-include-data-code-raw-fingerp
Planned fix A2+A3 (not implemented, low priority, do with A1): right after reading the label file in hdfs.py, raise on duplicate BlockId (3 examples) and on labels outside label_map (list the values); drop the misleading strict=True. One small test per case.
Why: dict() keeps the last duplicate silently; unmapped labels crash later as int(NaN). Verified 2026-11-05: HDFS_v1 anomaly_label.csv has 575,061 unique BlockIds, only Normal/Anomaly, no nulls, so this is a guard only.

## D-eac4 · 2026-11-05 · data-validation · fix/cache-keys-include-data-code-raw-fingerp
A4 (decided, not implemented): keep errors="replace" in hdfs.py and parser_check.py, but count lines containing replaced bytes and report "decode errors: N" in [data]. D3 (planned test): write a CRLF fixture with write_bytes and assert no content ends with \r, so CRLF is tested on every OS.
Why: User chose count-and-report over strict so a future dataset (BGL) is not blocked. Verified 2026-11-05: HDFS.log has 0 decode errors, 0 non-ASCII bytes, and CRLF on 100% of lines; current tests hit CRLF only on Windows via write_text.

## D-109c · 2026-11-06 · data-validation · fix/cache-keys-include-data-code-raw-fingerp
A5 (planned, with A4): count lines dropped for having no group id and report "dropped lines (no group id): N" in [data] instead of a loader INFO line only.
Why: Verified 2026-11-05: 0 such lines in full HDFS_v1; guard for datasets with session-less lines.

## D-346d · 2026-11-06 · data · fix/cache-keys-include-data-code-raw-fingerp
B1 (decided, not implemented): hdfs.yaml max_lines defaults to null; a dev experiment config (e.g. prepare_hdfs_dev.yaml) sets 100000. Also print max_lines in [data] and write data.max_lines to metrics.json; the Stage 7 aggregation script rejects runs with max_lines != null.
Why: Today every config inherits 100k and [data] does not show it, so a smoke-slice number could enter results/ unnoticed (D-b52f relied on memory only).

## D-c9f0 · 2026-11-06 · data · fix/cache-keys-include-data-code-raw-fingerp
D2 (decided, not implemented): add a reader in src/cascadelad/data/ (e.g. load_split(dir, name)) that reads a processed parquet and converts event_ids to list[int], with a test. Every consumer of processed data goes through it. Detector contract stays list[int].
Why: Verified 2026-11-05: read_parquet returns numpy arrays; arr + [0,0] adds elementwise instead of padding (silent), .count fails. Cheap to fix before detectors exist.

## D-cd49 · 2026-11-06 · data · fix/cache-keys-include-data-code-raw-fingerp
D6 (planned): add a loader test with max_lines None reading every fixture line, and run the full-HDFS prepare once early (via ex, machine otherwise idle) to record time and peak memory, before Stage 7. Run only when the user says go.
Why: Full data path was never run or tested. Measured 2026-11-05: 1M lines load peaks 0.47 GB, extrapolated ~5.3 GB for 11.2M lines on a 7.7 GB machine: OOM or swap is likely.

## D-c333 · 2026-11-06 · data · fix/cache-keys-include-data-code-raw-fingerp
A6 (planned): after a successful zip extraction, write data/raw/<ds>/.manifest.json with size and md5 of each extracted file. On "already present", check sizes against the manifest (md5 optional); a missing manifest means the earlier extraction did not finish: raise a clear error.
Why: md5 is checked only on the zip, which is deleted; extractall writes in place, so an interrupted extraction leaves a truncated HDFS.log that later runs accept silently.

## D-2be2 · 2026-11-06 · caching · fix/cache-keys-include-data-code-raw-fingerp
D-2e83 is implemented in commit 0a06b2d on branch fix/cache-keys-include-data-code-raw-fingerp (not merged yet): both cache keys hash src/cascadelad/data sources and drain3 version; processed_key also hashes the raw log (first max_lines lines) and label file; a processed hit logs stats recomputed from cached files.
Why: Status update: the earlier entry said implementation waits for user setup.
Supersedes: D-2e83

## D-d7e1 · 2026-11-06 · audit-plan · fix/cache-keys-include-data-code-raw-fingerp
Audit fixes grouped into tasks. G1 loader/prepare checks: A1-A5, D1, D3. G2 shared hash + atomic cache-write helpers: A6, A9, C. G3 max_lines path: B1, D6, B6. G4 run setup: A7, A8, B2, B3. G5 splitting (user reviews line by line): B4, D7. G6 Drain config: B5, D8, D9. D2 (load_split) stands alone.
Why: Low-priority items share a file or a fix with a higher one, so each group is one task instead of 20.

## D-5634 · 2026-11-06 · audit-plan · fix/cache-keys-include-data-code-raw-fingerp
Order: 0) finish D-2e83 branch (review, e2e check, merge) and merge Stage 3; 1) G3, then the full-HDFS run of D6 when the user says go; 2) D2 before Stage 5; 3) G1; 4) G2; 5) G4, G5, G6 when convenient or with the stage that touches those files.
Why: G3 decides which data every result uses and the full run may force a loader rewrite (OOM risk), so its unknown is the largest; D2 is cheap only before detectors exist.

## D-e14d · 2026-11-06 · caching · fix/cache-keys-include-data-code-raw-fingerp
Review of 0a06b2d: (1) any edit to src/cascadelad/data/*.py, comments included, invalidates both caches: accepted. (3) md5 of the whole raw log on every prepare call when max_lines=null: accepted. (5) ZeroDivisionError in _hit_stats on an empty test split: deferred to G1. Open: size in raw fingerprint under max_lines; build-time counts on hit.
Why: User review 2026-11-06: safety over recompute cost; empty test split cannot happen on HDFS.

## D-23c3 · 2026-11-06 · caching · fix/cache-keys-include-data-code-raw-fingerp
Closes D-e14d open points. (2) With max_lines, the raw log fingerprint is md5 of the first max_lines lines only, no size; whole-file hashing keeps size+md5. (4) On a processed hit, purged train/val and train anomalous dropped come from meta.json, logged as '<name> (from build)'; other [data] stats stay recomputed from files.
Why: Lines past max_lines are never read, so size only caused useless rebuilds; truncation detection belongs to A6 (D-c333). meta.json is written atomically with the outputs; hit and miss logs keep the same fields.

## D-376b · 2026-11-18 · parsing · fix/stage-2-protocol
AGENTS.md invariant 2 is reworded: vocab on train only, unseen ids -> <UNK>; templates from Drain fit_on=stream (default) or fit_on=train (ablation).
Why: User 2026-11-18: D-fc1a required the rewording; done in step 8 of fix/stage-2-protocol.

## D-a0f3 · 2026-11-18 · scope · fix/stage-2-protocol
src/cascadelad/data/prepare.py: only the parse call changed (parse_cached -> parse_lines, all loaded rows, fit mask under fit_on=train) so the suite stays importable after step 2; line_pos, template column, fit_on log and loader key stay in step 4.
Why: User 2026-11-18 chose to keep it: one-line bridge, step 4 rewrites that block. Test assertion changes in tests/test_parsing_cache.py approved (signature and cache-format change, equivalents kept).

## D-a0b1 · 2026-11-19 · splitting · fix/stage-3-protocol
AGENTS.md invariant 3 names the dirty-train exception (split.drop_train_anomalies=false, D-fb5d); invariant 2 says fit_on=train fits on the train lines the detector sees (D-3c69).
Why: User 2026-11-19: AGENTS.md must match plan.md, spec.md and recorded decisions.

## D-0d06 · 2026-11-19 · data · fix/stage-3-protocol
Done in 0695b41: hdfs.yaml max_lines null, configs/experiment/prepare_hdfs_dev.yaml sets 100000, [data] prints max_lines. Still to do: data.max_lines in metrics.json (when metrics.json exists, Stage 4/5) and the Stage 7 aggregation rejecting runs with max_lines != null.
Why: Status update 2026-11-19.
Supersedes: D-346d

## D-f150 · 2026-11-19 · data-validation · fix/stage-3-protocol
NaN template_id guard dropped: prepare now indexes parse_lines output by line position (line_pos), not by content, so a missing content cannot produce NaN.
Why: Verified 2026-11-19 in prepare.py: template_id = parsed[...].to_numpy()[pos], rows aligned with contents by parse_lines.
Supersedes: D-bb3b

## D-eb1d · 2026-11-19 · scope · fix/stage-3-protocol
README.md: max_lines default text updated to null + prepare_hdfs_dev (D-346d, commit 0695b41)
Why: User 2026-11-19 asked to implement D-346d; README documented the old 100000 default.

## D-6db5 · 2026-11-19 · scope · fix/stage-3-protocol
docs/plan.md: Stage 1 files line now says max_lines default null and names prepare_hdfs_dev (D-346d, commit 0695b41)
Why: User 2026-11-19 asked to implement D-346d; plan.md still said default 100000.

## D-0037 · 2026-11-19 · scope · fix/stage-3-protocol
docs/hdfs_setup_comparison.md: data-scope bullet updated for max_lines null default and prepare_hdfs_dev (D-346d, commit 0695b41)
Why: User 2026-11-19 asked to implement D-346d; the note told readers to set null in hdfs.yaml, now the default.
