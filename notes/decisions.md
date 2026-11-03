# Decisions

Append-only. Never edit or delete an entry; to change a decision, add a new one
with `Supersedes: D-xxx`. Written by notes.py decide.

## D-7a93 · 2026-11-03 · detector-interface · feat/stage-0-skeleton
Detector.fit(train: DataFrame, val: DataFrame|None) / score(windows: DataFrame) -> 1-D float ndarray, higher = more anomalous; window columns group_id, event_ids, timestamp; no label read, no thresholding in score
Why: Fixed in Stage 0 so DeepLog can be written in parallel; val needed for early stopping on val loss (invariant 4)
