# Decisions

Rules in force only (D-4a1e). Append with notes.py decide; to change one, add a new entry
with `Supersedes: D-xxx` and move the old one to decisions-archive.md when the branch merges.
History: decisions-archive.md (verbatim). Open work: backlog.md.

## D-2c8d · 2026-11-03 · detector-interface · feat/stage-0-skeleton
Detector base class enforces the window contract on every subclass: fit/score reject a 'label' column and duplicate group_id; score returns a 1-D float array in row order (scores[i] belongs to group_id.iloc[i]), shape-checked. Callers strip labels before fit/score. Signatures unchanged.
Why: Invariant 5: labels never reach detectors; enforcement must not depend on each subclass calling check_windows
Supersedes: D-7a93

## D-7362 · 2026-11-03 · tooling · feat/stage-0-skeleton
No Makefile and no .pre-commit-config.yaml; commands documented in README as uv run ...; agent-kit git hooks plus vf are the gate
Why: Kit hooks and vf already cover lint/format/tests; make is not available on the Windows dev machine

## D-ea3b · 2026-11-04 · tooling · feat/stage-3-grouping-split-vocab
pyarrow added as a dependency for parquet outputs
Why: docs/plan.md and AGENTS.md specify parquet for processed data and scores

## D-dac0 · 2026-11-19 · tooling · fix/stage-3-protocol
Commit scope notes is allowed. A decision that explains a code or doc change is committed with that change under its scope; a commit touching only notes/ (status updates, scope reasons, run results) uses docs(notes).
Why: User 2026-11-19: keeps notes-only commits apart from docs/ edits and silences the scope WARN.

## D-7d73 · 2026-11-04 · splitting · feat/stage-3-grouping-split-vocab
Blocks straddling a split boundary are purged (train kept iff end < first val start; val kept iff end < first test start); split cut by block count on start time
Why: Invariant 1 strict: no train/val line later than any val/test start

## D-fb5d · 2026-11-18 · splitting · fix/stage-1-3-protocol
split.drop_train_anomalies: true (default) | false. false is the dirty-train ablation: train keeps anomalous sequences and detectors train on them, an allowed, labeled exception to invariant 3.
Why: User 2026-11-18 approved the exception to invariant 3 for the Stage 3 follow-up; the default and the existing leakage test stay.

## D-fc1a · 2026-11-13 · parsing · main
Drain fit_on=stream is the default: one instance reads all lines in time order; a line's template_id is the cluster id returned when it is added, never recomputed. fit_on=train (fit on normal train lines, frozen, match only, unmatched -> <UNK>) stays as the RQ4 parser ablation. Vocab stays train-only. AGENTS.md invariant 2 must be reworded.
Why: User 2026-11-13: stream = online deployment (spec A5); the ablation measures anomalies stream hides vs extra windows train sends to the LLM.
Supersedes: D-8314

## D-6f30 · 2026-11-13 · parsing · main
Parse cache key includes the split boundary only when fit_on=train; under fit_on=stream it does not (plan S2-5 to be reworded).
Why: User 2026-11-13: stream parsing does not depend on the split, so keying on it would only force needless re-parses.

## D-2982 · 2026-11-13 · parsing · main
template_id stays a string: "E<cluster_id>", unmatched -> "<UNK>" (unk_token); plan S2-4 changed from -1.
Why: User 2026-11-13: models read integer vocab ids, not template_id; strings keep the two id spaces apart and vocab.json round-trips (JSON keys are strings).

## D-d10d · 2026-11-18 · parsing · fix/stage-2-protocol
Under fit_on=stream a line's template text is the cluster template returned when that line was added (never the final template).
Why: User 2026-11-18: matches D-fc1a 'recorded then, never recomputed'; no future lines leak into a line's template.

## D-3c69 · 2026-11-18 · parsing · fix/stage-1-3-protocol
With drop_train_anomalies=false and fit_on=train, Drain fits on every train line, anomalous ones included (the same train set the detector sees).
Why: User 2026-11-18: filtering anomalies for the parser would read labels, which defeats the dirty-train ablation.

## D-858f · 2026-11-18 · parsing · fix/stage-2-protocol
S2-6 stream-vs-train comparison lives in src/cascadelad/data/fit_on_compare.py plus scripts/compare_fit_on.py.
Why: User 2026-11-18: reusable for RQ4 on full HDFS/BGL; keeps parser_check about Loghub accuracy only.

## D-3ffe · 2026-11-18 · data · fix/stage-2-protocol
Dataset configs get a loader: key; prepare calls get_loader(cfg.dataset.loader). HDFS_2k loads through the hdfs loader with the full HDFS anomaly_label.csv.
Why: User 2026-11-18: S2-3/S2-6 must run on HDFS_2k through prepare; all 2200 HDFS_2k block ids have labels.

## D-cfbf · 2026-11-18 · data · fix/stage-1-3-protocol
Fixed windows (S3-5): dataset config key grouping: {mode: session|fixed, size, step}; build_fixed_windows plus synthetic tests now; BGL/Thunderbird loader is a later task. Processed parquet gains a line_labels column for every grouping mode.
Why: User 2026-11-18: no BGL loader exists yet; one schema for all datasets keeps consumers simple.

## D-c83f · 2026-11-06 · data · fix/cache-keys-include-data-code-raw-fingerp
A max_lines slice (100k for HDFS) is only a smoke test for pipeline and baseline runs; its metrics are never main results. The slice lives in a dev experiment config, not in the dataset default (D-346d). Slice size may differ per baseline model. Main results use full data.
Why: Same intent as D-b52f; the default moves to null so forgetting an override gives correct (slow) results, not smoke numbers.
Supersedes: D-b52f

## D-cbb3 · 2026-11-19 · evaluation · fix/stage-3-protocol
Thresholds and DeepLog top-g never read val or test labels: val_percentile (default, 99th percentile of val scores), pot, or flag_rate; same procedure for every method (plan.md protocol, spec 4.4). AGENTS.md invariant 5 reworded: labels only drop anomalies from train/val and score test. Still open from D-9417: evaluation unit = session, PCA input = per-session count vector, DeepLog session score = max over windows of -log p; decide at Stages 4-6.
Why: User 2026-11-19: align AGENTS.md with plan.md/spec.md; val is normal-only by default, so max F1 on val cannot be computed anyway.
Supersedes: D-9417

## D-093f · 2026-11-05 · terminology · feat/stage-3-grouping-split-vocab
Keep the term unsupervised in spec for now (spec 1.2, RQ3). Revisit after the model architecture is decided. Known tension: train is filtered to normal blocks with labels (invariant 3, D-8314), which the literature calls semi-supervised; retrieval pool drawn from filtered train would leak labels into RQ3.
Why: User has not decided the model architecture yet; wording follows architecture.

## D-4a1e · 2026-11-19 · tooling · fix/stage-1-3-protocol
Decision log layout: notes/decisions.md holds only rules in force. Superseded, done, status and scope-reason entries move verbatim to notes/decisions-archive.md (grep it; nt show does not read it). Decided-but-not-built work lives in notes/backlog.md as checkboxes; finishing one ticks the box, no new decision. Scope-reason entries move to the archive when their branch merges. A new ID must not reuse an archived one.
Why: User 2026-11-19 (option D): 47 mixed entries (rules, todos, status, scope logs) could not be followed.

## D-5b72 · 2026-11-19 · parsing · fix/stage-1-3-protocol
Drain via drain3, wrapped by DrainParser: add() for fit_on=stream (online, in order, no dedup); fit() then transform() for fit_on=train (transform only matches, unmatched -> unk_token).
Why: Replaces D-26b9 (fit/transform-only API) after D-fc1a made stream the default.

## D-6c03 · 2026-11-19 · caching · fix/stage-1-3-protocol
Parse key: parser config (minus parsed_dir), hash of src/cascadelad/data sources, drain3 version, contents, plus the fit mask under fit_on=train (D-6f30). processed_key adds the raw log (whole file: size+md5; with max_lines: md5 of the first max_lines lines) and the label file. Any edit under src/cascadelad/data, comments included, invalidates both. A processed hit logs stats recomputed from the cached files; build-time counts come from meta.json, labeled "(from build)".
Why: Merges D-2e83, D-2be2, D-e14d, D-23c3 (stale cache verified 2026-11-05; safety over recompute cost).

## D-7e94 · 2026-11-19 · data · fix/stage-1-3-protocol
max_lines defaults to null (all lines) in dataset configs; the 100k HDFS smoke slice is configs/experiment/prepare_hdfs_dev.yaml (D-c83f). [data] prints max_lines (a number or "all").
Why: Merges D-346d and D-0d06 (done in 0695b41); remaining items are in backlog.md.

## D-f043 · 2026-11-20 · evaluation · main
AU-PR is implemented in numpy in src/; scikit-learn is a dev-only dependency used to cross-check it in tests.
Why: User 2026-11-20: runtime code stays free of sklearn; the test still checks against the reference implementation.

## D-c59c · 2026-11-20 · evaluation · main
Metric table is Markdown with 4 decimals: header line dataset/split/seeds/threshold, columns method|P|R|F1|Spec|MCC|AU-PR|flag|R@10%|R@20%|TP|FP|FN|TN; multi-seed rows show mean ± std.
Why: User 2026-11-20: pastes directly into results/ and run.log; spec requires every table to state dataset, split, seeds, threshold.

## D-9318 · 2026-11-20 · evaluation · main
Evaluation parameters (bootstrap n_resamples, seed, CI level, flag rates) live in a new Hydra group configs/evaluation/default.yaml.
Why: User 2026-11-20: no magic numbers in code.

## D-bd43 · 2026-11-20 · evaluation · feat/stage-4-evaluation
Bootstrap CI uses the plain percentile method (2.5/97.5 quantiles of resampled metrics), point estimate from the full test set, no clamping.
Why: User 2026-11-20: simplest honest method; S4-5 is checked on a fixture.

## D-a7b6 · 2026-11-20 · evaluation · feat/stage-4-evaluation
recall_at_flag_rate flags k=ceil(rate*n) windows; ties broken by stable sort (earlier row wins), so exactly k are flagged.
Why: User 2026-11-20: deterministic and keeps S4-4 exact when scores tie (0/1 scores).

## D-e726 · 2026-11-20 · evaluation · feat/stage-4-evaluation
Paired bootstrap of the F1 difference between two systems is implemented in Stage 4 (metrics.py), not deferred.
Why: User 2026-11-20: spec 4.4 needs it for the minimum bar; avoids reopening metrics.py later.

## D-e411 · 2026-11-20 · evaluation · feat/stage-4-evaluation
Multi-seed tables use sample std (ddof=1); a single seed prints the value without +-.
Why: User 2026-11-20: standard for 3 seeds; config key evaluation.std_ddof.

## D-8671 · 2026-11-20 · scope · feat/stage-4-evaluation
.gitattributes: add 'tests/snapshots/** text eol=lf' to Stage 4 (outside the plan's Files list).
Why: core.autocrlf=true would check the metric-table snapshot out as CRLF, and test_table_matches_snapshot compares bytes (S4-6), so it would fail on a fresh checkout.

## D-09bd · 2026-11-20 · evaluation · feat/stage-4-evaluation
recall_at_flag_rate uses tie-aware expected recall: k=ceil(rate*n); windows scoring above the k-th score count fully, the remaining slots are filled from the tied group at its anomaly share. Deterministic and independent of row order.
Why: User 2026-11-20: row-order tie-breaking biased discrete-score baselines; on full HDFS test always-anomaly R@10% was 0.2113 by row order vs 0.1000 expected.
Supersedes: D-a7b6

## D-b6c5 · 2026-11-20 · evaluation · feat/stage-4-evaluation
Bootstrap CI and paired bootstrap use the circular block bootstrap over test rows in stored (chronological) order; block_length null = ceil(n^(1/3)), an integer overrides, 1 = iid. The block length used is returned with the CI.
Why: User 2026-11-20: adjacent windows are correlated (bursts), so iid resampling gives too narrow intervals and makes 'paired interval excludes 0' too easy.

## D-5a19 · 2026-11-20 · evaluation · feat/stage-4-evaluation
evaluate() returns f1/recall CI bounds (f1_ci_low, f1_ci_high, recall_ci_low, recall_ci_high, ci_block_length); the CI is shown in a separate table (format_ci_table, one row per method x seed); the main metric table and its snapshot are unchanged.
Why: User 2026-11-20: avoids aggregating CIs across seeds; keeps D-c59c table format.

## D-6141 · 2026-11-21 · evaluation · feat/stage-5-pca
Evaluation unit is the group: session for HDFS, fixed window for BGL/Thunderbird.
Why: User 2026-11-21: matches the processed data unit; closes backlog 'Unit of evaluation'.

## D-47fc · 2026-11-21 · detectors · feat/stage-5-pca
Trivial baselines (always_anomaly, unseen_sequence) output 0/1 scores and decide by a fixed rule score > model.rule_threshold (0.5), threshold_method=rule; PCA and later detectors use the threshold policy.
Why: User 2026-11-21: val_percentile with s > tau flags nothing for a constant score and is meaningless for 0/1 scores; the rule reads no labels and spec 4.3 defines these baselines by the rule.

## D-58a7 · 2026-11-21 · detectors · feat/stage-5-pca
PCA (Xu 2009/loglizer): per-group template count vector, TF-IDF with idf=log((N+1)/(df+1)) on train, zero-mean on train mean, components up to 95% explained variance, score = SPE.
Why: User 2026-11-21: comparable with loglizer-based published tables; smoothed idf stays >= 0 and gives <UNK> the largest weight; closes backlog 'PCA input'.

## D-a3c4 · 2026-11-21 · evaluation · feat/stage-5-pca
POT threshold fits the GPD to val excesses by method of moments in numpy (init_level, risk, min_excesses in configs/threshold/pot.yaml); invalid fits raise.
Why: User 2026-11-21: no new runtime dependency (scipy).

## D-d181 · 2026-11-21 · experiments · feat/stage-5-pca
One Stage-5 ex run evaluates PCA (primary, metrics.json test.*) plus the trivial baselines (baselines.<name>.test.*) in one metric table.
Why: User 2026-11-21: S5-6 needs exactly one runs.jsonl line and plan wants baselines in the same table.
