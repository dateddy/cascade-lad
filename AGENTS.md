# Agent rules

Skills live in `.agentkit/skills/<name>/` — read a skill's SKILL.md before the work it covers.
Short names used below (run with `python`; paths relative to `.agentkit/skills/`):
`nt` keeping-notes/scripts/notes.py · `gk` managing-git/scripts/gitkit.py · `pl` writing-plans/scripts/plan.py
`vf` verifying-changes/scripts/verify.py · `run` economizing-context/scripts/run.py · `rmap` economizing-context/scripts/repomap.py
`ex` tracking-experiments/scripts/exp.py · `dbg` debugging-systematically/scripts/dbg.py

## Mode: question or task
- Message starts with `?` → **question mode**: read-only. Read, search and run read-only commands; do not edit files, commit, or touch task notes. Answer concisely, then stop.
- Starts with `/task` or `task:` → follow **writing-plans** (Frame first).
- No prefix and the message asks for a change (add, fix, implement, refactor, remove, rename, commit…) → **task mode**.
- No prefix and unclear → treat as a question; end with one line offering to do it.

## Task lifecycle
1. Start of session: `nt brief`. Trust it over re-exploring; read only what it points to.
2. Non-trivial task → plan first (writing-plans). ≤2 files and no design decision → lite path: no plan file. New public interface/schema, ≥3 modules or an open design choice → suggest `/grill` (grilling-designs) first; never auto-run it. After the plan is accepted: branch (`gk branch-name`), save plan, `pl check`, `pl apply`.
3. Per step: `pl step N` → implement → `vf` → `nt task --check N --done "…" --next "…"` → commit (managing-git). Before editing a module: `rmap --zone <file>`. Inner loop: `vf --affected`; the full `vf` is still required before commit.
4. Done only when `vf` exits 0 after the last edit (never pipe it; the VERDICT line is authoritative). If the guards are installed (`install.py --guards`), their `verify.sh` must exit 0 too. Never weaken tests to pass.
5. Decision settled by the user → `nt decide --topic … --text … --why …`.
6. Training/eval runs → `ex run … -- <cmd>`, never bare.
7. Something fails → debugging-systematically: `dbg open --layer code|data|llm …`, no guess-and-edit.

## Failure log
- Every mistake of yours that the user corrects gets an entry in `FAILURES.md` (date, what happened, root cause, control). The entry must name a control; prefer a script or hook that always runs over a new instruction line.

## Scripts decide, you write
- Exit codes: 0 OK · 2 WARN → show the user and wait for yes · 1 BLOCK → fix and re-run. Never bypass a BLOCK.
- Never use `--no-verify` (or any other way around a git hook). A hook BLOCK is fixed, not bypassed.
- Destructive or irreversible commands (reset --hard, force push, deleting branches/files, dropping stashes): quote the exact command and wait for an explicit yes.

## Token discipline
- Locate before reading: `rmap`, search (rg/grep), `rmap --symbols`, then read the needed line range.
- Wrap commands that may print >60 lines in `run`. Grep saved logs instead of re-running.
- Never print lockfiles, datasets, notebook outputs, generated files or checkpoints.
- Replies: lead with the result. No recap of steps, no restating code you just wrote, no filler.
- Reply in the user's language.

<!-- Project-specific rules below this line -->

## Project: cascade-lad

Unsupervised log anomaly detection on large-scale system logs. A lightweight detector (PCA, DeepLog, LogBERT) scores every window; the top-ρ% most suspicious windows go to an LLM verifier, optionally with context retrieved from unlabeled training windows.

- Spec (problem, RQs, hypotheses): `docs/spec.md`
- Stage plan with done criteria: `docs/plan.md`. Work one stage at a time. A stage is done only when every done criterion for it in `docs/plan.md` passes.

### Layout
- `configs/` (Hydra): `dataset/`, `parser/`, `model/`, `llm/`, `cascade/`; `experiment/` composes them. Every parameter lives in YAML. No magic numbers in code.
- `src/cascadelad/`
  - `data/`: `loaders/` (one class per dataset, all return the standard columns `timestamp, content, group_id, label`), `parsing.py`, `grouping.py`, `splitting.py`, `vocab.py`
  - `detectors/`: `base.py` defines `fit(train)` and `score(windows) -> continuous score`. Thresholding (incl. top-g) is applied after scoring, never inside `score`.
  - `verifiers/`: LLM client with cache, prompts, retriever
  - `methods/`: `detector_only.py`, `llm_only.py` (= cascade with ρ = 100%), `cascade.py` (proposed method)
  - `training/`, `evaluation/`, `utils/`
- `scripts/`: thin entry points. Read config, call `src/`. No logic.
- `data/raw/` is immutable: never edit or overwrite. `data/parsed/` and `data/processed/` use directories named by config hash. None of `data/` is in git.
- `runs/<date>_<dataset>_<method>_<note>_seed<k>/`: `config.yaml`, `git_commit.txt`, `run.log`, `history.csv`, `checkpoints/{best,last}.pt`, `metrics.json`, `scores.parquet`. Not in git.
- `experiments/runs.jsonl`: run index kept by `ex` (committed). `results/`: final tables and figures for the thesis (committed). `cache/llm/`: not in git.

### Research invariants: never change without the user's explicit yes
1. Splits are chronological. A random split is allowed only as a clearly labeled comparison.
2. Templates and vocab are fit on train only. Unseen templates map to `<UNK>`.
3. Detectors train on normal train sequences only.
4. Model selection (`best.pt`, early stopping) uses val loss on normal sequences. Never F1 or any metric that uses anomaly labels.
5. Anomaly labels are used only for (a) threshold / g selection on val, with the same procedure for every method, and (b) test evaluation, run once per run after training.
6. Verifier retrieval draws only from unlabeled train windows.
7. Every method is evaluated through `evaluation/` (same metric code, same table format).

If a task seems to require breaking one of these, stop and ask.

### Experiments
- Run through `ex`, passing `run_name` explicitly so the metrics path is known in advance:
  `ex run --name <dataset>-<method> --config configs/experiment/<x>.yaml --metrics runs/<run_name>/metrics.json --param run_dir=runs/<run_name> -- python scripts/train.py experiment=<x> run_name=<run_name>`
- Numbers that go into `results/` need 3 seeds per configuration and a clean git tree.
- `metrics.json` keys: `test.precision`, `test.recall`, `test.f1`, `test.fp`, `test.fn`, `threshold`, `threshold_split`, `infer_ms_per_window`, plus `cost.*` for LLM runs.
- Every LLM call goes through the cached client (key = hash of prompt, model, params) and logs tokens and latency.

### Code
- Python ≥ 3.10, uv, ruff (format + lint), pytest. Type hints on public functions.
- Leakage tests (`tests/test_splitting.py`, `tests/test_grouping.py`) must stay green. Never weaken them.
- Per-run log sections, in order: setup (git sha + clean/dirty, seed, device, config path), data (input paths, processed hash, split sizes, anomaly %, vocab size, test `<UNK>` rate), model (params), one line per epoch (train_loss, val_loss, top1, top-g, time), eval table, outputs. Logging goes to console and `run.log` (plain text, no color codes).
- F1 / FP / FN are computed only after training, never per epoch.
- Ask before adding a dependency.
- Changes to splitting, vocab, DeepLog scoring, thresholding or metrics: show the diff and explain the logic. The user reviews these line by line before commit.