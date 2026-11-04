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
