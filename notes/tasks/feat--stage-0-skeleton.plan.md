# Plan: Stage 0 â€” repo skeleton, run infrastructure, fixed detector interface
Constraints: Python >=3.10 via uv; only deps approved by the user (runtime: hydra-core, numpy, pandas; dev: pytest, ruff, pre-commit); every parameter in YAML; logs are plain text to console + run.log; do not touch the agent-kit git hooks in .git/hooks; detectors never threshold inside score().
Done when: `uv run python scripts/train.py experiment=dummy run_name=dummy` creates runs/dummy/{config.yaml,git_commit.txt,run.log} and run.log contains a `[setup]` section; full `vf` exits 0; detectors/base.py interface recorded with `nt decide`.
Out of scope: data loading (Stage 1), torch, `pre-commit install` (it would replace the agent-kit pre-commit hook; config file only), README.

Context: docs/plan.md Stage 0. User decisions (this session): detector API takes DataFrame windows; minimal dependency set; keep Makefile (calls `uv run â€¦`) although `make` is not installed on this Windows machine.

## 1. Project tooling and package stub
Files: pyproject.toml (new), .gitignore, .pre-commit-config.yaml (new), Makefile (new), .agentkit.json, src/cascadelad/__init__.py (new), src/cascadelad/utils/__init__.py (new), src/cascadelad/detectors/__init__.py (new), tests/test_smoke.py (new)
Do:
- pyproject.toml: `[project] name="cascadelad"`, `version="0.0.1"`, `requires-python=">=3.10"`, build backend hatchling, `[tool.hatch.build.targets.wheel] packages=["src/cascadelad"]`. Add deps with `uv add hydra-core numpy pandas` and `uv add --dev pytest ruff pre-commit` (this writes uv.lock; commit it, never print it). `[tool.ruff] line-length=100, target-version="py310"`, `[tool.ruff.lint] select=["E","F","I","UP","B"]`. `[tool.pytest.ini_options] testpaths=["tests"]`.
- .gitignore: append `data/`, `runs/`, `cache/`, `outputs/`, `multirun/`, `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `.ruff_cache/`. Keep the existing `/notes/tasks/` line.
- .pre-commit-config.yaml: repo `https://github.com/astral-sh/ruff-pre-commit`, `rev` = `v` + version printed by `uv run ruff --version`, hooks `ruff` (args `[--fix]`) and `ruff-format`. Do NOT run `pre-commit install`.
- Makefile: `.PHONY: data test lint format`; `test: uv run pytest -q`; `lint: uv run ruff check . && uv run ruff format --check .`; `format: uv run ruff format . && uv run ruff check --fix .`; `data:` echo "Stage 1: scripts/download_data.py not written yet" and exit 0.
- .agentkit.json `verify`: `"format": "uv run ruff format --check ."`, `"lint": "uv run ruff check ."`, `"test": "uv run pytest -q"` (uses the project venv's ruff, not the global miniconda one).
- src/cascadelad/__init__.py: `__version__ = "0.0.1"`. Empty `__init__.py` for utils/ and detectors/.
- tests/test_smoke.py: `test_import()` asserts `cascadelad.__version__ == "0.0.1"` (an empty suite makes pytest exit 5, so vf would fail).
Verify: python .agentkit/skills/verifying-changes/scripts/verify.py
Done: `uv sync` succeeds; VERDICT PASS (exit 0) with 1 test passed.

## 2. Seed and run utilities
Files: src/cascadelad/utils/seed.py (new), src/cascadelad/utils/run.py (new), tests/test_run.py (new)
Do:
- seed.py: `set_seed(seed: int) -> None` sets `random.seed`, `np.random.seed`, `os.environ["PYTHONHASHSEED"]`; if `importlib.util.find_spec("torch")` then also `torch.manual_seed` and `torch.cuda.manual_seed_all` (no torch import otherwise).
- run.py, all public functions typed:
  - `git_state(cwd: Path | None = None) -> tuple[str, bool]`: sha from `git rev-parse HEAD`, dirty = `git status --porcelain` non-empty; on any git error return `("unknown", True)`.
  - `create_run_dir(runs_dir: Path, run_name: str, overwrite: bool = False) -> Path`: mkdir parents; if dir exists and contains any file and not overwrite â†’ raise `FileExistsError` (an empty existing dir is allowed, Hydra may create it first).
  - `write_git_commit(run_dir: Path, sha: str, dirty: bool) -> None`: writes `"<sha> clean\n"` or `"<sha> dirty\n"` to git_commit.txt.
  - `save_config(cfg: DictConfig, run_dir: Path) -> None`: `OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)`.
  - `setup_logging(run_dir: Path, name: str = "cascadelad") -> logging.Logger`: level INFO, clears existing handlers on that logger, `propagate=False`, StreamHandler(stdout) + FileHandler(run_dir/"run.log", encoding="utf-8"), format `"%(asctime)s %(levelname)s %(message)s"` (no colors).
  - `log_section(logger: logging.Logger, section: str, items: Mapping[str, object]) -> None`: logs `[<section>]` then one line `  <key>: <value>` per item, in order.
  - `start_run(cfg: DictConfig, config_path: str) -> tuple[Path, logging.Logger]`: create_run_dir(Path(cfg.paths.runs_dir), cfg.run_name, cfg.overwrite) â†’ set_seed(cfg.seed) â†’ git_state â†’ write_git_commit â†’ save_config â†’ setup_logging â†’ `log_section(logger, "setup", {"git_sha": sha, "git_tree": "clean"/"dirty", "seed": cfg.seed, "device": cfg.device, "config": config_path})`.
- tests/test_run.py (use tmp_path; build cfg with `OmegaConf.create`): files config.yaml/git_commit.txt/run.log exist after start_run; run.log contains `[setup]` and `seed: 0`; git_commit.txt matches `^(\w+|unknown) (clean|dirty)$`; create_run_dir on a non-empty dir raises FileExistsError and succeeds with overwrite=True; set_seed(1) twice gives identical `np.random.rand()`. Close the logger's FileHandlers at the end of each test (Windows file locks on tmp_path).
Verify: uv run pytest tests/test_run.py -q
Done: all tests in tests/test_run.py pass; `vf` PASS.

## 3. Detector interface (fixed for parallel DeepLog work)
Files: src/cascadelad/detectors/base.py (new), tests/test_detector_base.py (new)
Do:
- base.py: module docstring stating the window contract: a window DataFrame has columns `group_id` (str), `event_ids` (list[int], vocab ids, `<UNK>` already mapped), `timestamp` (first event time); detectors must not read a `label` column. Constant `WINDOW_COLUMNS: tuple[str, ...] = ("group_id", "event_ids", "timestamp")`.
- `class Detector(ABC)`:
  - `@abstractmethod fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None` â€” train holds normal windows only; val (normal only) is for early stopping on val loss.
  - `@abstractmethod score(self, windows: pd.DataFrame) -> np.ndarray` â€” 1-D float array, `len == len(windows)`, same row order, higher = more anomalous; no thresholding.
  - concrete `check_windows(windows: pd.DataFrame) -> None` (staticmethod): raises `ValueError` naming missing columns from WINDOW_COLUMNS.
- tests/test_detector_base.py: `Detector()` raises TypeError; a tiny subclass (score = length of event_ids) returns float array of correct length in row order; check_windows raises ValueError on a frame missing `event_ids`.
- After tests pass: `nt decide --topic detector-interface --text "Detector.fit(train: DataFrame, val: DataFrame|None) / score(windows: DataFrame) -> 1-D float ndarray, higher = more anomalous; window columns group_id, event_ids, timestamp; no label read, no thresholding in score" --why "Fixed in Stage 0 so DeepLog can be written in parallel; val needed for early stopping on val loss (invariant 4)"`.
Verify: uv run pytest tests/test_detector_base.py -q
Done: tests pass; new entry in notes/decisions.md.

## 4. Hydra config skeleton and scripts/train.py
Files: configs/config.yaml (new), configs/experiment/dummy.yaml (new), configs/dataset/.gitkeep (new), configs/parser/.gitkeep (new), configs/model/.gitkeep (new), configs/llm/.gitkeep (new), configs/cascade/.gitkeep (new), scripts/train.py (new), tests/test_train_script.py (new)
Do:
- configs/config.yaml:
  ```yaml
  defaults:
    - _self_
    - experiment: null
    - override hydra/job_logging: none
    - override hydra/hydra_logging: none
  run_name: <Hydra mandatory-missing marker: three question marks, no quotes>
  seed: 0
  device: cpu
  overwrite: false
  paths:
    runs_dir: runs
  hydra:
    run:
      dir: ${paths.runs_dir}/${run_name}
    output_subdir: null
    job:
      chdir: false
  ```
- configs/experiment/dummy.yaml: `# @package _global_` then `note: dummy`.
- scripts/train.py (thin, no logic): `@hydra.main(version_base="1.3", config_path="../configs", config_name="config")` `main(cfg)`: `exp = HydraConfig.get().runtime.choices.get("experiment")`; `config_path = f"configs/experiment/{exp}.yaml" if exp else "configs/config.yaml"`; `run_dir, logger = start_run(cfg, config_path)`; `logger.info("[outputs]")`; `logger.info("  run_dir: %s", run_dir)`. Nothing else.
- tests/test_train_script.py: subprocess `[sys.executable, "scripts/train.py", "experiment=dummy", "run_name=t", f"paths.runs_dir={tmp_path.as_posix()}"]`, cwd = repo root, check=True; assert the 3 files exist under tmp_path/"t", run.log contains `[setup]` and `config: configs/experiment/dummy.yaml`; and no `outputs/` dir appears at repo root.
Verify: uv run python scripts/train.py experiment=dummy run_name=dummy  (then check runs/dummy/ has the 3 files and `[setup]` in run.log; delete runs/dummy afterwards only with the user's yes)
Done: the command exits 0 and creates runs/dummy/{config.yaml,git_commit.txt,run.log} with `[setup]`; full `vf` VERDICT PASS (exit 0).

## Verification (whole stage)
- `python .agentkit/skills/verifying-changes/scripts/verify.py` â†’ VERDICT PASS (exit 0), not piped.
- `uv run python scripts/train.py experiment=dummy run_name=dummy` â†’ runs/dummy/ with config.yaml, git_commit.txt (`<sha> clean|dirty`), run.log with `[setup]` section (git_sha, git_tree, seed, device, config).
- Then merge feat/stage-0-skeleton into main per docs/plan.md (via managing-git, after user OK).

## Executor checklist (after acceptance)
1. `gk branch-name` â†’ `feat/stage-0-skeleton`; 2. save this plan to `notes/tasks/<branch>.plan.md` (`pl path`); 3. `pl check`; 4. `pl apply`; 5. per step: `pl step N` â†’ implement â†’ Verify â†’ `vf` â†’ `nt task --check N --done â€¦ --next â€¦` â†’ commit (scope `infra`).

