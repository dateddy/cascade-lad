# Plan: Drain log parser with train-only fitting and a hash-keyed cache, checked on HDFS_2k (grouping accuracy ≥ 0.95)

## Context
Stage 2 of `docs/plan.md`: turn header-stripped log `content` strings into template ids, the input Stage 3 needs for grouping, splitting and vocab. Stage 1 is merged into local `main` (e151286, main is 5 commits ahead of origin, not pushed). User decisions so far: use **drain3** (new dependency, approved); accuracy threshold is **grouping accuracy ≥ 0.95** on HDFS_2k.

Invariant 2 (templates fit on train only, unseen → `<UNK>`) shapes the API: the parser has separate `fit` (learns templates) and `transform` (matches only, never learns). Stage 2 fits and transforms the same 2k lines (as the Loghub benchmark does). Stage 3 will call `fit(train)` and then `transform(all)`.

Constraints: no logic in `scripts/`; every parameter lives in YAML; `data/raw/` is never overwritten; leakage invariants untouched; pyarrow is not added (cache uses CSV/JSON).
Done when: `scripts/check_parser.py` on HDFS_2k logs `grouping_accuracy >= 0.95` (exact value reported), a second run logs `cache hit` and does not call `fit`, and `vf` exits 0.
Out of scope: train-only fitting on real splits (Stage 3), parquet/pyarrow, BGL parsing, parse-time benchmarks on full HDFS (Stage 7), Loghub-2.0 corrected labels.

## 1. Parser config, DrainParser, grouping accuracy
Files: pyproject.toml, uv.lock, configs/parser/drain.yaml (new), configs/parser/.gitkeep, src/cascadelad/data/parsing.py (new), tests/test_parsing.py (new)
Do:
- `uv add drain3`; `git rm configs/parser/.gitkeep`.
- `configs/parser/drain.yaml` (`# @package parser`): `name: drain`, `depth: 4`, `sim_th: 0.5`, `max_children: 100`, `max_clusters: null`, `parametrize_numeric_tokens: true`, `extra_delimiters: []`, `masking: [{regex: 'blk_-?\d+', mask_with: BLK}, {regex: '(\d+\.){3}\d+(:\d+)?', mask_with: IP}]`, `match_strategy: fallback`, `unk_token: '<UNK>'`, `parsed_dir: data/parsed`.
- `parsing.py`: `class DrainParser` with `__init__(self, cfg: Mapping)` building a drain3 `TemplateMiner` (no persistence handler) from a `TemplateMinerConfig` filled from cfg (depth, sim_th, max_children, max_clusters, parametrize_numeric_tokens, extra_delimiters, masking as `MaskingInstruction`s); `fit(self, contents: Iterable[str]) -> "DrainParser"` calls `add_log_message` once per unique content in first-seen order (`dict.fromkeys`); `transform(self, contents: Iterable[str]) -> list[str]` calls `miner.match(c, full_search_strategy=cfg["match_strategy"])` and returns `f"E{cluster.cluster_id}"` or `cfg["unk_token"]` when `None`; `transform` raises `RuntimeError` if called before `fit`; `templates(self) -> dict[str, str]` maps id → template text. `grouping_accuracy(pred: Sequence[str], truth: Sequence[str]) -> float`: Loghub definition (a line is correct iff the set of lines sharing its predicted id equals the set sharing its true id); raise `ValueError` on length mismatch or empty input.
- Tests: two lines differing only in a block id/IP/number get the same id; a content never seen in `fit` → `<UNK>`; `transform` does not add clusters (`len(templates())` unchanged); `transform` before `fit` raises; `grouping_accuracy` on hand-made arrays (perfect = 1.0; one merged cluster = hand-computed fraction; relabelled ids still 1.0; length mismatch raises).
Verify: python .agentkit/skills/verifying-changes/scripts/verify.py --affected
Done: `uv run pytest tests/test_parsing.py -q` passes; `drain3` is in pyproject dependencies.

## 2. Hash-keyed parse cache
Files: src/cascadelad/data/parsing.py (new), tests/test_parsing_cache.py (new)
Do:
- `cache_key(cfg: Mapping, fit_contents: Sequence[str], apply_contents: Sequence[str]) -> str`: sha256 over canonical JSON of cfg without `parsed_dir`, then the fit contents and apply contents joined with `\n` (each section prefixed by its length so the boundary is unambiguous); return the first 16 hex chars.
- `parse_cached(fit_contents, apply_contents, cfg) -> tuple[pd.DataFrame, dict[str, str], bool]`: returns (`assignments` with columns `content, template_id` for unique apply contents in first-seen order, `templates`, `cache_hit`). Dir `Path(cfg["parsed_dir"]) / key`. On hit: read `assignments.csv` (`keep_default_na=False`, `dtype=str`) and `templates.json`, log `cache hit: <dir>`, never construct a parser. On miss: log `cache miss: fitting on N unique lines`, fit + transform, write `assignments.csv`, `templates.json`, `meta.json` (`n_fit`, `n_apply`, `n_templates`, `n_unk`, drain3 version, cfg) into `<key>.tmp/`, then `rename` to `<key>/` (crashed run leaves no half cache; remove a stale `.tmp` first).
- Tests (tmp `parsed_dir`): first call `cache_hit is False` and the dir has the 3 files; second call `cache_hit is True` with identical outputs while `DrainParser.fit` is monkeypatched to raise; changing `sim_th` or the fit contents gives a different key; changing only `parsed_dir` gives the same key; a content such as `"NA"` or `"null"` round-trips as a string.
Verify: python .agentkit/skills/verifying-changes/scripts/verify.py --affected
Done: `uv run pytest tests/test_parsing_cache.py -q` passes.

## 3. HDFS_2k download (plain files, pinned commit + md5)
Files: src/cascadelad/data/download.py, configs/dataset/hdfs_2k.yaml (new), tests/test_download.py
Do:
- Get the latest logpai/loghub commit sha that touches `HDFS/`: `gh api "repos/logpai/loghub/commits?path=HDFS/HDFS_2k.log_structured.csv&per_page=1" --jq ".[0].sha"`. Pin it in the raw URLs.
- `configs/dataset/hdfs_2k.yaml` (`# @package dataset`): `name: hdfs_2k`, `raw_dir: data/raw/hdfs_2k`, `log_file: HDFS_2k.log`, `structured_file: HDFS_2k.log_structured.csv`, `line_regex` copied from `hdfs.yaml` (comment: must match hdfs.yaml), `content_column: Content`, `event_column: EventId`, `files:` list of `{filename, url, md5}` for both files at `https://raw.githubusercontent.com/logpai/loghub/<sha>/HDFS/<file>`, `download: {chunk_bytes: 1048576}`. Compute each md5 by downloading once to the scratchpad (`curl -sL <url> | md5sum`) and pin it.
- `download.py`: new `fetch_files(cfg: Mapping) -> Path` → for each entry: skip if `raw_dir/filename` exists (log `already present`), else download to `.part`, check md5 with `md5sum`, raise `ValueError` on mismatch (deleting `.part`), rename into place. `fetch_dataset` dispatches to `fetch_files` first when `"files" in cfg`; the zenodo path is unchanged.
- Tests (file:// URIs as in the existing tests): both files land in `raw_dir`; md5 mismatch raises and leaves no file and no `.part`; a second call with files present does not open the URL (point url at a missing path, must still succeed).
Verify: python .agentkit/skills/verifying-changes/scripts/verify.py --affected
Done: `uv run pytest tests/test_download.py -q` passes; `uv run python scripts/download_data.py dataset=hdfs_2k run_name=download` puts both files in `data/raw/hdfs_2k/`.

## 4. Parser check entry point
Files: configs/config.yaml, src/cascadelad/data/parser_check.py (new), scripts/check_parser.py (new), tests/test_parser_check.py (new), README.md
Do:
- `configs/config.yaml`: add `- parser: null` to defaults after `dataset`.
- `parser_check.py`: `run_check(dataset_cfg: Mapping, parser_cfg: Mapping, logger: logging.Logger) -> dict[str, float | int | bool]`: read `structured_file` (`dtype=str`, `keep_default_na=False`); read `log_file`, strip headers with `line_regex`, and raise `ValueError` if the extracted contents differ from the `content_column` (reports first differing line); call `parse_cached(contents, contents, parser_cfg)`; map each line to its template id; compute `grouping_accuracy` against `event_column`. Log a `[parse]` section: `lines`, `predicted templates`, `true templates`, `n_unk`, `grouping_accuracy` (4 decimals), `cache hit|miss` + dir. Return those values.
- `scripts/check_parser.py`: Hydra main like `scripts/train.py`; `start_run(cfg, "configs/config.yaml")`, call `run_check` with `OmegaConf.to_container(..., resolve=True)` of `cfg.dataset` and `cfg.parser`, write the returned dict to `run_dir/metrics.json`. No other logic.
- Test: tiny fixture (≈6 HDFS-style lines + matching structured CSV with 2 EventIds) in tmp_path; `run_check` returns `grouping_accuracy == 1.0` and `cache_hit False`, second call `cache_hit True`; a content mismatch between log and CSV raises.
- README: add the `check_parser.py` and `download_data.py dataset=hdfs_2k` commands, and a `data/parsed/<hash>/` line under Data.
Verify: python .agentkit/skills/verifying-changes/scripts/verify.py --affected
Done: `uv run pytest tests/test_parser_check.py -q` passes.

## 5. Real-data check and stage close-out
Files: README.md
Do:
- `uv run python scripts/check_parser.py dataset=hdfs_2k parser=drain run_name=parser-check-hdfs2k-1` → read `grouping_accuracy` and `cache miss` from `runs/parser-check-hdfs2k-1/run.log`.
- Same command with `run_name=parser-check-hdfs2k-2` → `cache hit`, same accuracy.
- If accuracy < 0.95: stop, open `dbg open --layer data …`, report to the user; do not tune `sim_th`/masking without their yes.
- Full `vf`; `nt task --check 5 --done "HDFS_2k grouping_accuracy=<x>, rerun cache hit; vf PASS" --next "merge feat/stage-2-parsing after user review"`.
Verify: python .agentkit/skills/verifying-changes/scripts/verify.py
Done: run 1 log shows `grouping_accuracy` ≥ 0.9500 and `cache miss`; run 2 log shows `cache hit`; `vf` VERDICT PASS.
