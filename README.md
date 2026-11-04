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
(log line, block id), reading the first `max_lines` lines (default 100000, `null` = all).

## Parsing

`DrainParser` (`src/cascadelad/data/parsing.py`) learns templates in `fit` and only matches in `transform`; unseen templates map to `<UNK>`. `parse_cached` stores results in `data/parsed/<hash>/` (`assignments.csv`, `templates.json`, `meta.json`); the hash covers the parser config and the fit/apply contents, so a rerun is a cache hit.
