# Implementation plan

One branch per stage (`feat/stage-N-<slug>`). Merge into `main` only when every done criterion for the stage passes and `vf` reports PASS. Do not start a stage before the previous one is merged.

Order rule: follow the data flow, and get one thin end-to-end slice working (PCA) before adding DeepLog or a second dataset.

---

## Stage 0: Repo skeleton and run infrastructure
**Files:** `pyproject.toml` (uv), `.gitignore` (`data/`, `runs/`, `cache/`), `.pre-commit-config.yaml` (ruff), `Makefile` (`data`, `test`, `lint`), `src/cascadelad/__init__.py`, `src/cascadelad/utils/seed.py`, `src/cascadelad/utils/run.py` (create run dir, save config snapshot + git sha with clean/dirty flag, set up logging to console + `run.log`), `src/cascadelad/detectors/base.py` (interface only), `configs/` Hydra skeleton, `scripts/train.py` (minimal).

**Done when:**
- `python scripts/train.py experiment=dummy run_name=dummy` creates `runs/dummy/` with `config.yaml`, `git_commit.txt`, `run.log`, and the log prints the `[setup]` section.
- `vf` passes (ruff + an empty pytest suite).
- `detectors/base.py` interface is fixed so the DeepLog model can be written in parallel.

## Stage 1: Download and load data
**Files:** `scripts/download_data.py` (Zenodo record 8196385, checksum, unzip to `data/raw/<dataset>/`), `src/cascadelad/data/loaders/base.py` (interface + registry), `data/loaders/hdfs.py`, `configs/dataset/hdfs.yaml` (includes `max_lines`, default 100000 for development).

**Dataset:** HDFS_v1 only.

**Done when:**
- The loader returns a DataFrame with exactly `timestamp, content, group_id, label`.
- Block labels from `anomaly_label.csv` are joined correctly (test with a few known block ids).

## Stage 2: Log parsing
**Files:** `src/cascadelad/data/parsing.py` (Drain wrapper, cache to `data/parsed/<hash>/`), `configs/parser/drain.yaml`.

**Done when:**
- On `HDFS_2k.log` (loghub repo), output templates agree with `HDFS_2k.log_structured.csv` for a high share of lines (report the share).
- A second run reports a cache hit and does not re-parse.

## Stage 3: Grouping, splitting, vocab
**Files:** `data/grouping.py`, `data/splitting.py`, `data/vocab.py`, `scripts/prepare_data.py`, `tests/test_splitting.py`, `tests/test_grouping.py`. Write the tests first.

**Done when:**
- `data/processed/<hash>/{train,val,test}.parquet` and `vocab.json` exist.
- Leakage tests pass: every test timestamp is later than every train timestamp; vocab is built from train only; train contains no anomalous sequence.
- The `[data]` log section prints split sizes, anomaly %, vocab size and test `<UNK>` rate.

## Stage 4: Evaluation module
**Files:** `evaluation/metrics.py`, `evaluation/report.py` (shared metric table), `evaluation/cost.py` (timers; token counting later), `tests/test_metrics.py`.

**Done when:** on hand-made label/prediction arrays, P/R/F1/FP/FN match values computed by hand, and the table prints in the agreed format.

## Stage 5: Detector interface + PCA (first number)
**Files:** `detectors/pca.py`, `methods/detector_only.py` (score, then threshold chosen on val), `scripts/evaluate.py`, `configs/model/pca.yaml`, `configs/experiment/baseline_hdfs_pca.yaml`.

**Done when:** one `ex run …` command runs the whole pipeline on the 100k-line slice and produces `metrics.json`, `scores.parquet`, the metric table at the end of `run.log`, and a record in `experiments/runs.jsonl`.

## Stage 6: DeepLog
**Files:** `detectors/deeplog.py` (model + continuous score from the probability of the actual next key), `training/trainer.py` (loop, early stopping on val loss, `best.pt` / `last.pt`, `history.csv`, resume), `configs/model/deeplog.yaml` (defaults from the paper: h=10, g=9, 2 layers, 64 units).

**Done when:**
- Val top-g accuracy on normal sequences is close to what the DeepLog paper reports.
- Results are in the same range as `deep-loglizer` under the same data setup.
- Resuming from `last.pt` works.
- The user has reviewed the model and scoring code line by line.

## Stage 7: Scale up
- Full HDFS_v1; record parse and train time.
- BGL: only `data/loaders/bgl.py` + `configs/dataset/bgl.yaml` (time-window grouping). If anything outside the loader and config must change, revisit stages 1–3.
- 3 seeds per configuration; `scripts/aggregate_results.py` writes mean ± std tables to `results/tables/`.

**Done when:** a baseline table (PCA, DeepLog × HDFS, BGL, chronological split, 3 seeds) exists in `results/tables/`. Target: end of November 2026.

---

## Later (not planned in detail yet)
LogBERT → LLM verifier + cache (`crafting-prompts` skill for the prompt) → cascade with ρ sweep → retrieval → demo.