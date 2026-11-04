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
```


Before committing, run the full gate: `python .agentkit/skills/verifying-changes/scripts/verify.py`.
