# cascade-lad

Unsupervised log anomaly detection: a lightweight detector scores every window, and the top-ρ% most suspicious windows go to an LLM verifier. Spec: [docs/spec.md](docs/spec.md). Stage plan: [docs/plan.md](docs/plan.md).

## Commands

All commands run through uv, from the repo root.

```
uv sync                                  # create .venv and install dependencies
uv run pytest -q                         # tests
uv run ruff check .                      # lint
uv run ruff format .                     # format (use --check to only verify)
uv run python scripts/train.py experiment=dummy run_name=dummy   # smoke run, writes runs/dummy/
uv run python scripts/download_data.py dataset=hdfs run_name=download   # fetch HDFS_v1 into data/raw/hdfs/ (md5-checked)
uv run python scripts/download_data.py dataset=hdfs_2k run_name=download   # HDFS_2k sample + Loghub ground truth
uv run python scripts/check_parser.py dataset=hdfs_2k parser=drain run_name=parser-check   # Drain grouping accuracy on HDFS_2k
uv run python scripts/prepare_data.py experiment=prepare_hdfs run_name=hdfs-prepare   # group, split chronologically, build vocab -> data/processed/<hash>/
```


Before committing, run the full gate: `python .agentkit/skills/verifying-changes/scripts/verify.py`.

## Data

`uv run python scripts/download_data.py dataset=hdfs run_name=download` downloads HDFS_v1 from Zenodo
(record 8196385), checks the pinned md5 and extracts it. Existing files are never overwritten; a second
run does nothing. Layout:

```
data/raw/hdfs/
  HDFS.log
  preprocessed/anomaly_label.csv   # BlockId,Label (Normal/Anomaly)
```

`get_loader("hdfs", cfg.dataset).load()` returns `timestamp, content, group_id, label`, one row per
(log line, block id), reading the first `max_lines` lines (default `null` = all; `experiment=prepare_hdfs_dev` sets the 100000-line smoke slice, D-346d). `[data]` prints `max_lines`.

## Parsing

`DrainParser` (`src/cascadelad/data/parsing.py`) runs Drain in one of two modes, set by `parser.fit_on`. `stream` (default): `add` reads lines in time order and each line's `template_id` and `template` are fixed when it is read. `train`: `fit` learns on the selected (normal train) lines, then `transform` only matches; unseen lines map to `<UNK>`. `parse_lines(contents, cfg, fit_mask)` returns one `template_id, template` row per line and caches them in `data/parsed/<hash>/` (`lines.parquet`, `templates.json`, `meta.json`); the hash covers the parser config incl. `fit_on`, the contents and, under `train`, the fit mask, so a rerun is a cache hit and, under `stream`, a new split over the same stream is too.
