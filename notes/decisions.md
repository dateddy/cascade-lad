# Decisions

Append-only. Never edit or delete an entry; to change a decision, add a new one
with `Supersedes: D-xxx`. Written by notes.py decide.

## D-7a93 · 2026-11-03 · detector-interface · feat/stage-0-skeleton
Detector.fit(train: DataFrame, val: DataFrame|None) / score(windows: DataFrame) -> 1-D float ndarray, higher = more anomalous; window columns group_id, event_ids, timestamp; no label read, no thresholding in score
Why: Fixed in Stage 0 so DeepLog can be written in parallel; val needed for early stopping on val loss (invariant 4)

## D-2c8d · 2026-11-03 · detector-interface · feat/stage-0-skeleton
Detector base class enforces the window contract on every subclass: fit/score reject a 'label' column and duplicate group_id; score returns a 1-D float array in row order (scores[i] belongs to group_id.iloc[i]), shape-checked. Callers strip labels before fit/score. Signatures unchanged.
Why: Invariant 5: labels never reach detectors; enforcement must not depend on each subclass calling check_windows
Supersedes: D-7a93

## D-7362 · 2026-11-03 · tooling · feat/stage-0-skeleton
No Makefile and no .pre-commit-config.yaml; commands documented in README as uv run ...; agent-kit git hooks plus vf are the gate
Why: Kit hooks and vf already cover lint/format/tests; make is not available on the Windows dev machine

## D-26b9 · 2026-11-04 · parsing · feat/stage-2-parsing
Drain via drain3; parser API is fit(train contents) + transform(any contents, match only, unmatched -> <UNK>)
Why: Invariant 2: templates fit on train only; drain3 match() does not update the tree

## D-7d73 · 2026-11-04 · splitting · feat/stage-3-grouping-split-vocab
Blocks straddling a split boundary are purged (train kept iff end < first val start; val kept iff end < first test start); split cut by block count on start time
Why: Invariant 1 strict: no train/val line later than any val/test start

## D-8314 · 2026-11-04 · parsing · feat/stage-3-grouping-split-vocab
Drain parser and vocab are fit on normal train blocks only (anomalous train blocks dropped before fit)
Why: Invariants 2 and 3: fit only on what the detector trains on

## D-ea3b · 2026-11-04 · tooling · feat/stage-3-grouping-split-vocab
pyarrow added as a dependency for parquet outputs
Why: docs/plan.md and AGENTS.md specify parquet for processed data and scores

## D-2e83 · 2026-11-05 · caching · feat/stage-3-grouping-split-vocab
processed_key (and parse cache key) must include: hash of src/cascadelad/data/ source files, raw file fingerprint (size + md5 of the part read), drain3 version. On a cache hit, [data] must log the real state of this run, not replay meta.json stats. Implementation waits until the user finishes setup; then plan it.
Why: Verified 2026-11-05: changing splitting code or truncating HDFS.log kept the same key and silently returned stale processed data (cache hit, git_commit.txt pointing at the new commit).

## D-b52f · 2026-11-05 · data · feat/stage-3-grouping-split-vocab
The max_lines slice (default 100k) is only a smoke test for pipeline and baseline runs (data flows end to end). Its metrics are never main results or interpreted. Slice size may be changed per baseline model. Main results use full data.
Why: On 100k lines train spans ~9 min, <UNK> comes from lifecycle phase (first Deleting block at 20:45:24 > t_val), val has 18 anomalies, test tail blocks are truncated but keep full-block labels.

## D-093f · 2026-11-05 · terminology · feat/stage-3-grouping-split-vocab
Keep the term unsupervised in spec for now (spec 1.2, RQ3). Revisit after the model architecture is decided. Known tension: train is filtered to normal blocks with labels (invariant 3, D-8314), which the literature calls semi-supervised; retrieval pool drawn from filtered train would leak labels into RQ3.
Why: User has not decided the model architecture yet; wording follows architecture.

## D-bf45 · 2026-11-05 · ablation · feat/stage-3-grouping-split-vocab
Deferred until PCA and DeepLog run on full HDFS (Stage 6-7): contamination ablation with a YAML flag (e.g. split.train_normal_only) training on all train blocks without label filtering. Whether to run it stays open. Touches splitting and invariant 3: needs explicit user yes.
Why: Supports or refutes the unsupervised claim; not needed before baselines exist.

## D-149b · 2026-11-05 · evaluation · feat/stage-3-grouping-split-vocab
Deferred to Stage 7, before any RQ4 work: on full HDFS check correlation between <UNK> in a test session and its label; if present report F1 split by with/without <UNK>. RQ4 needs real log evolution (BGL per spec, or synthetic template changes in LogRobust style); HDFS_v1 is one software version, so its <UNK> is not evidence of evolution.
Why: On 100k slice <UNK> came from lifecycle phase (Deleting block), not code change; a UNK-label correlation would inflate detector F1.

## D-9417 · 2026-11-05 · evaluation · feat/stage-3-grouping-split-vocab
Open, proposed defaults (not decided): unit of evaluation = session (block); threshold = max F1 on val, same procedure for every method (invariant 5); PCA input = per-session template count vector; DeepLog session score = max over windows of -log p(actual next key). Decide each at its stage (4, 5, 6) after explanation.
Why: User wants to decide when building models with enough background; recorded so defaults are not adopted silently.
