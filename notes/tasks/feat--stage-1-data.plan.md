# Plan: Stage 1: download HDFS_v1 and load it into the standard `timestamp, content, group_id, label` DataFrame

Context: Stage 0 (skeleton, run infra, Detector interface) is on `main`. Stage 1 in `docs/plan.md` adds the first data path: fetch HDFS_v1 from Zenodo record 8196385 into `data/raw/hdfs/`, and a loader registry with an HDFS loader that Stage 2 (Drain parsing) will consume. Branch: `feat/stage-1-data`.

Constraints:
- `data/raw/` is immutable. The download step never overwrites existing files.
- Every parameter lives in `configs/dataset/hdfs.yaml`, including URL, md5, file names, `max_lines` (100000) and the line regex. No magic numbers in code.
- No new dependencies. Use the stdlib (`urllib.request`, `hashlib`, `zipfile`) plus pandas.
- `scripts/download_data.py` is a thin Hydra entry point. The logic lives in `src/cascadelad/data/`.
- Tests use synthetic fixtures in `tmp_path` and a `file://` URL. No network calls and no `pytest.skip`.

User decisions (Frame):
- `content` holds the message after `Component: `. The date, time, pid, level and component are not included.
- `max_lines` keeps the first N lines in file order. Partial blocks are not filtered here; Stage 3 handles them.
- The zip md5 is pinned in YAML. A mismatch deletes the partial download and raises.

Plan-level choices (flag if you disagree):
- `timestamp` comes from the first two fields, parsed as `%y%m%d %H%M%S`.
- A line with several distinct `blk_` ids becomes one row per block (loglizer convention). A line with no block id is dropped, and the loader logs the count.
- `label` is an int: `Anomaly` → 1, `Normal` → 0. A block id missing from `anomaly_label.csv` raises `ValueError`.
- A line that does not match the regex raises `ValueError` with its line number.

Done when:
- `uv run python scripts/download_data.py dataset=hdfs` produces `data/raw/hdfs/HDFS.log` and `anomaly_label.csv`, with the md5 verified.
- `get_loader("hdfs", cfg.dataset).load()` returns a DataFrame with exactly `["timestamp", "content", "group_id", "label"]` on the 100k slice.
- Known block ids from the real `anomaly_label.csv` (at least 2 Normal and 2 Anomaly) carry the correct label.
- `vf` returns PASS.

Out of scope: Drain parsing (Stage 2), grouping into windows, splitting and vocab (Stage 3), BGL and other datasets, and filtering partial blocks.

## 1. Loader interface and registry
Files: src/cascadelad/data/__init__.py (new), src/cascadelad/data/loaders/__init__.py (new), src/cascadelad/data/loaders/base.py (new), tests/test_loader_base.py (new)
Do:
- In `base.py`, define `STANDARD_COLUMNS: tuple[str, ...] = ("timestamp", "content", "group_id", "label")`.
- Define `class LogLoader(ABC)`: `__init__(self, cfg: Mapping)` stores the cfg; abstract `_load(self) -> pd.DataFrame`; public `load(self) -> pd.DataFrame` calls `_load` and then `check_standard(df)`.
- `check_standard` raises `ValueError` unless the columns are exactly `STANDARD_COLUMNS` in that order and `label` holds only {0, 1}.
- Registry: a `_REGISTRY: dict[str, type[LogLoader]]`, a decorator `register_loader(name: str)` that raises on a duplicate name, and `get_loader(name: str, cfg: Mapping) -> LogLoader` that raises `KeyError` listing the known names.
- `loaders/__init__.py` imports `hdfs` so registration happens. Add that import in step 2.
- Tests cover: a dummy registered loader round-trips; a duplicate name raises; an unknown name raises `KeyError`; wrong or extra columns raise; a label of 2 raises.
Verify: uv run pytest tests/test_loader_base.py -q
Done: the 5 tests in tests/test_loader_base.py pass.

## 2. HDFS loader and dataset config
Files: src/cascadelad/data/loaders/hdfs.py (new), src/cascadelad/data/loaders/__init__.py (new), configs/dataset/hdfs.yaml (new), tests/test_hdfs_loader.py (new)
Do:
- `configs/dataset/hdfs.yaml` (`# @package dataset`) holds: `name: hdfs`, `raw_dir: data/raw/hdfs`, `log_file: HDFS.log`, `label_file: anomaly_label.csv` (path relative to `raw_dir`; fix it to the real zip layout in step 3), `max_lines: 100000`.
- It also holds `line_regex: '^(?P<date>\d{6}) (?P<time>\d{6}) (?P<pid>\d+) (?P<level>\w+) (?P<component>[^:]+): (?P<content>.*)$'`, `timestamp_format: '%y%m%d %H%M%S'`, `block_regex: 'blk_-?\d+'`, and `label_map: {Normal: 0, Anomaly: 1}`.
- Leave `zenodo` (url, filename, md5) for step 3.
- `@register_loader("hdfs") class HDFSLoader(LogLoader)`: read the first `max_lines` lines with `itertools.islice` (`null` = all lines), using `encoding="utf-8", errors="replace"`. Match `line_regex` per line and raise `ValueError(f"line {i}: …")` when a line does not match.
- Build `timestamp` with `pd.to_datetime(date + " " + time, format=timestamp_format)`.
- Find block ids with `block_regex`, take the unique ones in order of appearance, and emit one row per id. Drop lines with no id and log the count via `logging.getLogger(__name__)`.
- Join labels from `label_file` (columns `BlockId, Label`) through `label_map`. Raise if any block is unlabeled. Return the standard columns with the dtypes datetime64, str, str, int64.
- Tests write a synthetic `HDFS.log` (about 6 lines: 3 blocks, one line holding 2 blocks, one line with no block) and an `anomaly_label.csv` into `tmp_path`. They assert:
  - the columns are exact
  - the content excludes the header
  - the timestamp is parsed
  - the 2-block line yields 2 rows
  - the no-block line is dropped
  - known block ids map to 0 and 1
  - `max_lines=3` returns rows only from the first 3 lines
  - an unlabeled block raises
  - a malformed line raises
Verify: uv run pytest tests/test_hdfs_loader.py -q
Done: all tests in tests/test_hdfs_loader.py pass, and `get_loader("hdfs", cfg)` works on the fixture.

## 3. Zenodo download with pinned md5
Files: src/cascadelad/data/download.py (new), scripts/download_data.py (new), configs/config.yaml, configs/dataset/hdfs.yaml (new), tests/test_download.py (new), README.md
Do:
- Look up the real values once with `curl -s https://zenodo.org/api/records/8196385`. Find the `HDFS_v1.zip` entry, take its `checksum` (`md5:…`) and download link, and add them to `hdfs.yaml` as `zenodo: {record: 8196385, filename: HDFS_v1.zip, url: <link>, md5: <hex>}`.
- `download.py` defines `md5sum(path: Path) -> str`, which hashes in chunks with the chunk size from cfg `download.chunk_bytes` in hdfs.yaml.
- It also defines `fetch_dataset(cfg: Mapping) -> Path`:
  - If `raw_dir/log_file` and `raw_dir/label_file` already exist, log "already present" and return without touching anything.
  - Otherwise stream `url` with `urllib.request.urlopen` to `raw_dir/<filename>.part`, check the md5, and delete the `.part` file and raise `ValueError` on a mismatch.
  - On a match, rename the file and `zipfile` extract it into `raw_dir`. Extraction refuses to overwrite existing files and rejects absolute or `..` member paths.
  - Return `raw_dir`.
- After the first real download, set `label_file` in hdfs.yaml to the actual path inside the zip (expected `preprocessed/anomaly_label.csv`).
- In `configs/config.yaml`, add `- dataset: null` to the defaults before `experiment`.
- `scripts/download_data.py` is a `@hydra.main` that calls `fetch_dataset(cfg.dataset)` and nothing else. It runs under `hydra.run.dir` like train.py; pass `run_name=download` on the CLI or give it a default in the script's overrides. Document the command in README.md.
- Tests build a small zip in `tmp_path` and point `url` at `Path.as_uri()`. They assert:
  - a correct md5 extracts the files
  - a wrong md5 raises and leaves no `.part` and no extracted files
  - a second call with the files present is a no-op (mtime unchanged)
  - a zip member `../evil` raises
Verify: uv run pytest tests/test_download.py tests/test_train_script.py -q
Done: the tests in test_download.py pass, the existing train-script test still passes, and hdfs.yaml has a real 32-hex md5.

## 4. Real-data check and stage close-out
Files: README.md
Do:
- Run `uv run python scripts/download_data.py dataset=hdfs run_name=download`. Confirm the md5 passes and the files exist.
- Load the slice: `uv run python -c "…get_loader('hdfs', cfg).load()…"` composed via `hydra.compose(config_name='config', overrides=['dataset=hdfs','run_name=x'])`. Print `df.columns.tolist()`, `len(df)`, `df.label.mean()`, and the rows for 2 Normal and 2 Anomaly block ids picked from `anomaly_label.csv` that appear in the first 100k lines.
- Record the chosen ids and the observed output in the step note (`nt task --done`).
- Add a "Data" section to README with the download command and the expected `data/raw/hdfs/` layout.
- Finish with the full `vf`.
Verify: python .agentkit/skills/verifying-changes/scripts/verify.py
Done: the printed columns equal `['timestamp', 'content', 'group_id', 'label']`, the 4 known block ids show their CSV labels, and the `vf` VERDICT is PASS.

## Verification (end to end)
1. Run `uv run pytest -q`. All tests pass, including test_loader_base, test_hdfs_loader and test_download.
2. Run the download command once on real data, then run it again. The second run logs "already present".
3. Run the step 4 load snippet and check the columns and the known block labels.
4. Run `vf` and confirm PASS. Merge to `main` only after that.

Note: `nt brief` still shows `feat/stage-0-skeleton` as active (4/4), but its commits are already on `main`. Close that note when the branch for this stage is created.
