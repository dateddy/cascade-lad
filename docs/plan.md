# Implementation plan

One branch per stage (`feat/stage-N-<slug>`). Merge into `main` only when every done criterion for the stage passes and `vf` reports PASS. Do not start a stage before the previous one is merged.

Order rule: follow the data flow, and get one thin end-to-end slice working (PCA) before adding DeepLog or a second dataset.

Stages 1–3 are already merged. Their protocol changes go on one branch, `fix/stage-1-3-protocol`, listed under **Follow-up** in each stage. Merge that branch before starting Stage 5.

---

## Evaluation protocol (applies to every stage)

These decisions come from the literature review (Le & Zhang 2022, LogLLM 2025, RAPID 2023, LogPrompt 2023, LogGPT 2023). They keep our numbers comparable with published tables and keep the "no labels" claim defensible.

| Topic | Decision |
|---|---|
| Datasets | HDFS_v1 (session), BGL (main benchmark), Thunderbird first 10M lines (optional third) |
| Grouping | HDFS: session by `block_id`. BGL / Thunderbird: fixed 100-line, non-overlapping windows |
| Window label | Anomalous if it contains at least one anomalous line; per-line labels are kept |
| Split | Chronological train / val / test. Test = last 20%, so BGL test matches Le & Zhang and LogLLM (≈ 9,427 windows, 817 anomalies). HDFS also runs in `random` mode for comparison with published numbers |
| Train | Anomalous sequences dropped by default (`drop_train_anomalies: true`); `false` is the dirty-train ablation |
| Val | Normal-only by default |
| Thresholds | Never read val or test labels: `val_percentile` (default), `pot`, or `flag_rate` |
| Parser | Drain with `fit_on: stream` (default, template ids assigned online) or `train` (ablation) |
| Metrics | TP/FP/FN/TN, precision, recall, F1, specificity, MCC, AU-PR, flag rate, recall at flag rate 10% / 20%, bootstrap 95% CI |
| Seeds | 3 per configuration, reported as mean ± std |
| Baselines | always-anomaly, unseen-sequence, PCA, DeepLog. Published numbers are added as rows marked "reported", with source |

---

## Stage 0: Repo skeleton and run infrastructure
**Files:** `pyproject.toml` (uv), `.gitignore` (`data/`, `runs/`, `cache/`), `README.md` (`uv run` commands for test, lint, format, data), `src/cascadelad/__init__.py`, `src/cascadelad/utils/seed.py`, `src/cascadelad/utils/run.py` (create run dir, save config snapshot + git sha with clean/dirty flag, set up logging to console + `run.log`), `src/cascadelad/detectors/base.py` (interface only), `configs/` Hydra skeleton, `scripts/train.py` (minimal).

**Done when:**
- `python scripts/train.py experiment=dummy run_name=dummy` creates `runs/dummy/` with `config.yaml`, `git_commit.txt`, `run.log`, and the log prints the `[setup]` section.
- `vf` passes (ruff + an empty pytest suite).
- `detectors/base.py` interface is fixed so the DeepLog model can be written in parallel.

## Stage 1: Download and load data
**Files:** `scripts/download_data.py` (Zenodo record 8196385, checksum, unzip to `data/raw/<dataset>/`), `src/cascadelad/data/loaders/base.py` (interface + registry), `data/loaders/hdfs.py`, `configs/dataset/hdfs.yaml` (includes `max_lines`, default `null` = all lines; the 100000-line development slice is `configs/experiment/prepare_hdfs_dev.yaml`, D-346d). `[data]` prints `max_lines`; slice metrics are never main results (D-c83f).

**Dataset:** HDFS_v1 only.

**Done when:**
- The loader returns a DataFrame with exactly `timestamp, content, group_id, label`.
- Block labels from `anomaly_label.csv` are joined correctly (test with a few known block ids).

**Follow-up (`fix/stage-1-3-protocol`):**
- Each loader declares `label_granularity: "line" | "group"`. `label` holds the per-line label where the dataset provides one (BGL, Thunderbird) and the block label broadcast to every line otherwise (HDFS).
- `content` keeps the raw message with parameter values; no templating or masking happens in the loader.

**Follow-up done when:**
- **S1-1** HDFS loader output: columns exactly `timestamp, content, group_id, label`; dtypes `datetime64`, `str`, `str`, `int8` with values in {0, 1}.
- **S1-2** `HDFSLoader.label_granularity == "group"`; every line of a block carries that block's label (checked on a sample of blocks).
- **S1-3** Fixture line `Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106` → `content` still contains the block id and the IP.
- **S1-4** Output is sorted by `timestamp` (stable, so lines with equal timestamps keep file order); the number of raw file lines read is `<= max_lines`. A line with several block ids gives one row per block id, so `len(df)` can exceed `max_lines`; the number of extra rows is logged.

## Stage 2: Log parsing
**Files:** `src/cascadelad/data/parsing.py` (Drain wrapper, cache to `data/parsed/<hash>/`), `configs/parser/drain.yaml`.

**Done when:**
- On `HDFS_2k.log` (loghub repo), output templates agree with `HDFS_2k.log_structured.csv` for a high share of lines (report the share).
- A second run reports a cache hit and does not re-parse.

**Follow-up (`fix/stage-1-3-protocol`):**
- `configs/parser/drain.yaml` gets `fit_on: stream | train`.
  - `stream` (default, D-fc1a): one Drain instance reads all lines in time order, without deduplicating first. Each line's `template_id` is the cluster id returned by `add_log_message()` for that line, recorded at that moment and never recomputed, so later lines cannot change the ids of earlier ones.
  - `train` (RQ4 parser ablation, D-fc1a): Drain learns on the normal train lines only, then is frozen; val/test lines are matched without changing the tree (`match()` in drain3), and unmatched lines get `<UNK>`.
  - The vocab is built from train only in both modes, so a template first seen in val/test maps to the vocab `<UNK>` either way.
- Parsing adds `template_id` and `template` columns; `content` stays unchanged.
- The choice of `fit_on` is printed in the `[data]` log section.
- Processed outputs keep `group_id` and each line's position in the stream, so a `stream` run and a `train` run on the same split can be joined line by line. RQ4 uses this to count test lines that are `<UNK>` under `train` but in-vocab under `stream` (anomalies stream absorbs into known templates), and windows flagged only under `train` (extra windows sent to the LLM).

**Follow-up done when:**
- **S2-1** Parsing adds `template_id` and `template`; `content` is byte-identical before and after parsing.
- **S2-2** The `[data]` log prints `fit_on`.
- **S2-3** `fit_on=stream`: on `HDFS_2k` split 70/30 in time order, the `template_id` of every train line is the same whether Drain parses the train part alone or the full stream.
- **S2-4** `fit_on=train`: the number of clusters after parsing val/test equals the number after train; unmatched lines get `template_id = "<UNK>"` (string, the `unk_token`). Template ids stay strings (`"E<cluster_id>"`), separate from the integer vocab ids.
- **S2-5** The cache key includes `fit_on` and the Drain parameters; with `fit_on=train` it also includes the split boundary (D-6f30). Changing any key part gives a cache miss; with `fit_on=stream`, changing only the split ratios gives a parse cache hit.
- **S2-6** For the same split, the `stream` and `train` outputs join one-to-one on (`group_id`, line position); on `HDFS_2k` the join covers every test line and the count of lines `<UNK>` under `train` but in-vocab under `stream` is printed.

## Stage 3: Grouping, splitting, vocab
**Files:** `data/grouping.py`, `data/splitting.py`, `data/vocab.py`, `scripts/prepare_data.py`, `tests/test_splitting.py`, `tests/test_grouping.py`. Write the tests first.

**Done when:**
- `data/processed/<hash>/{train,val,test}.parquet` and `vocab.json` exist.
- Leakage tests pass: every test timestamp is later than every train timestamp; vocab is built from train only; train contains no anomalous sequence.
- The `[data]` log section prints split sizes, anomaly %, vocab size and test `<UNK>` rate.

**Follow-up (`fix/stage-1-3-protocol`):**
- `split.mode: chronological | random`. For session data (HDFS), chronological order uses each session's start time, and the leakage test checks `max(train session start) < min(test session start)` instead of comparing every timestamp. `random` exists only to compare HDFS against published numbers.
- `split.drop_train_anomalies: true` (default). With `false`, train keeps anomalous sequences; the `[data]` section reports how many.
- Val is the chronological slice right after train, with anomalies dropped by default (normal-only). Thresholds never read val or test labels.
- Window datasets (BGL, Thunderbird) use fixed 100-line, non-overlapping windows (step 100). A window is anomalous if it contains at least one anomalous line; per-line labels are stored with each window.
- The `[data]` section prints anomaly counts as well as percentages for every split.

**Follow-up done when:**
- **S3-1** `split.mode=chronological`: for window data, every val/test line timestamp ≥ the last train timestamp; for session data, `max(train session start) < min(val session start)` and the same for val → test.
- **S3-2** `split.mode=random`: the same seed gives the same split; a different seed gives a different split.
- **S3-3** `drop_train_anomalies=true` → train has 0 anomalous sequences; `false` → no rows are removed and the `[data]` section reports the train anomaly count.
- **S3-4** Val is normal-only by default (0 anomalies); test keeps its natural anomaly ratio.
- **S3-5** Windows: every window has exactly 100 lines (the incomplete last window is dropped and its count is logged); window label = max of its line labels; the stored per-line label list has length 100; no window crosses a split boundary.
- **S3-6** Vocab is built from train only; ids for `<PAD>` and `<UNK>` are reserved; the test `<UNK>` rate printed in `[data]` matches a direct recount.
- **S3-7** The `[data]` section prints absolute counts and percentages for every split.

## Stage 4: Evaluation module
**Files:** `evaluation/metrics.py`, `evaluation/report.py` (shared metric table), `evaluation/cost.py` (timers; token counting later), `tests/test_metrics.py`.

**Metrics:** TP/FP/FN/TN, precision, recall, F1, specificity, MCC, AU-PR (from scores), flag rate, recall at flag rate 10% and 20%, bootstrap 95% CI for F1 and recall (1,000 resamples, fixed seed).

**Done when:**
- **S4-1** On hand-made arrays, TP/FP/FN/TN, precision, recall, F1, specificity and MCC match hand-computed values to 1e-9; TP+FP+FN+TN = n.
- **S4-2** Edge cases do not raise: no positive predictions → precision = 0, recall = 0; all-positive predictions → specificity = 0.
- **S4-3** AU-PR equals `sklearn.metrics.average_precision_score` on random scores.
- **S4-4** `flag_rate` = predicted positives / n; `recall_at_flag_rate(0.1)` flags exactly the top 10% of scores.
- **S4-5** Bootstrap CI with a fixed seed is deterministic and satisfies lower ≤ point estimate ≤ upper.
- **S4-6** The metric table matches a stored snapshot and prints in the agreed format.

**Decisions (Stage 4):** AU-PR in numpy, sklearn dev-only (D-f043); table format (D-c59c); parameters in `configs/evaluation/default.yaml` (D-9318); percentile bootstrap CI without clamping (D-bd43); `recall_at_flag_rate` flags `ceil(rate * n)` windows, ties by row order (D-a7b6); the paired bootstrap of the F1 difference (spec 4.4) is implemented here in `metrics.py` (D-e726); multi-seed std uses ddof=1 (D-e411).

## Stage 5: Detector interface + PCA (first number)
**Files:** `detectors/pca.py`, `detectors/always_anomaly.py`, `detectors/unseen_sequence.py`, `methods/detector_only.py` (score, then threshold), `evaluation/thresholds.py`, `configs/threshold/{val_percentile,pot,flag_rate}.yaml`, `scripts/evaluate.py`, `configs/model/pca.yaml`, `configs/experiment/baseline_hdfs_pca.yaml`.

**Threshold policy:** `val_percentile` (default, on normal-only val), `pot`, or `flag_rate`. Threshold functions see scores only, never labels. The method and value are saved in `metrics.json`.

**Trivial baselines:** `always_anomaly` flags every window. `unseen_sequence` flags a window if its template sequence never occurs in train. Both appear in the same metric table as PCA.

**Note:** PCA trained on anomaly-free train is reported as normal-only, not fully unsupervised.

**Done when:**
- **S5-1** Threshold functions accept only scores and config; a test asserts that their signature has no label argument.
- **S5-2** `val_percentile=99` flags 1% ± 0.2% of val on synthetic scores; `flag_rate=0.1` flags 10% ± 1% of a test set drawn from the same distribution; `pot` returns a finite threshold above the val median.
- **S5-3** `always_anomaly`: recall = 1 and specificity = 0 on any test set.
- **S5-4** `unseen_sequence`: a window identical to a train window scores 0; a window with an unseen template sequence scores 1 (fixture).
- **S5-5** PCA scores are identical across two runs.
- **S5-6** One `ex run …` command runs the whole pipeline on the 100k-line slice and writes `metrics.json` (with threshold method and value), `scores.parquet`, the metric table at the end of `run.log`, and appends exactly one line to `experiments/runs.jsonl`.

## Stage 6: DeepLog
**Files:** `detectors/deeplog.py` (model + continuous score from the probability of the actual next key), `training/trainer.py` (loop, early stopping on val loss, `best.pt` / `last.pt`, `history.csv`, resume), `configs/model/deeplog.yaml` (defaults from the paper: h=10, g=9, 2 layers, 64 units).

**Window score:** `deeplog.window_score: min_prob | count_outside_topg`. The binary decision goes through the threshold policy from Stage 5, not a hard-coded top-g, so the later ρ sweep can reuse it.

**Diagnostics:** `scores.parquet` stores, per window, how many keys fell outside top-g (`n_outside_topg`) and how many were `<UNK>` (`n_unk`). This is needed later to count windows flagged only because of unseen templates.

**Done when:**
- **S6-1** Training overfits a tiny batch (loss falls below 0.05 within 200 steps).
- **S6-2** Early stopping stops after `patience` epochs without val-loss improvement.
- **S6-3** Resuming from `last.pt` gives the same next-epoch val loss as an uninterrupted seeded run, within 1e-4.
- **S6-4** Both `window_score` modes give a higher score to a window with one key replaced by `<UNK>` than to its unmodified copy.
- **S6-5** `<UNK>` is never inside top-g; `scores.parquet` has `n_outside_topg` and `n_unk`, with `n_unk <= n_outside_topg`.
- **S6-6** (Recorded check, not a unit test) Full HDFS with `split.mode=random`: F1 within ±0.05 of the published 0.908 (P 0.835, R 0.994; LogLLM table). The chronological result is logged next to it.
- Val top-g accuracy on normal sequences is close to what the DeepLog paper reports.
- The user has reviewed the model and scoring code line by line.

## Stage 7: Scale up
- Full HDFS_v1 in both split modes; record parse and train time.
- BGL: only `data/loaders/bgl.py` + `configs/dataset/bgl.yaml`, using fixed 100-line windows. If anything outside the loader and config must change, revisit stages 1–3. Reference: DeepLog P 0.166 / R 0.988 / F1 0.285 (LogLLM), F1 0.426 (Le & Zhang).
- Thunderbird (optional, if time allows): first 10M lines, same windowing; only `data/loaders/thunderbird.py` + config.
- 3 seeds per configuration; `scripts/aggregate_results.py` writes mean ± std tables to `results/tables/`.
- Baseline table rows: always-anomaly, unseen-sequence, PCA, DeepLog. Published numbers (LogLLM, NeuralLog, LogRobust, RAPID) are added as rows marked "reported", with source.

**Done when:**
- **S7-1** BGL loader: `label_granularity == "line"`; lines whose first field is `-` get label 0, others 1 (checked on fixture lines).
- **S7-2** BGL test split ≈ 9,427 windows and ≈ 817 anomalies (±2%), with actual values logged. The tolerance is there because the papers do not say how they handle the last incomplete window.
- **S7-3** `aggregate_results.py` writes mean ± std and fails if any configuration has fewer than 3 seeds.
- **S7-4** "Reported" rows carry a source column and are never averaged with our own runs.
- **S7-5** (If Thunderbird is added) Same checks as S7-1; the test anomaly count is printed and the table shows a bootstrap CI.
- A baseline table (always-anomaly, unseen-sequence, PCA, DeepLog × HDFS, BGL, chronological split, 3 seeds) exists in `results/tables/`. Target: end of November 2026.

---

## Later (not planned in detail yet)
1. **LLM verifier:** input = content with parameters (timestamp and header removed), one window per prompt, JSON output `{verdict, reason, evidence_lines}`, temperature 0, pinned open-weight model. Use the `crafting-prompts` skill for the prompt.
2. **Cheap filter + cache:** normalize parameters RAPID-style, dedupe windows, skip windows that exactly match a known normal window.
3. **Cascade with ρ sweep.** Stage-2 metrics: specificity on stage-1 false positives, share of true positives kept, format-error rate, HitRate@k against BGL line labels.
4. **Retrieval:** 1–few nearest normal windows in the prompt; ablation with and without.
5. **Baselines under our protocol:** LogPrompt; RAPID with a label-free threshold, or cite its reported 0.548 on BGL.
6. **Robustness:** LogRobust-style injection at 5–20% (word edits; deleted, duplicated or shuffled events) plus meaning-flip edits ("succeeded" → "failed").
7. **Contamination check:** half-line completion on Loghub lines.
8. **LogBERT:** cite from LogLLM instead of running, unless time remains.
9. **Ablations:** Drain `fit_on=train`, dirty train (`drop_train_anomalies=false`), second parser (Spell or IPLoM).
10. **Demo.**