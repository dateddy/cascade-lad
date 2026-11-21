# Research Spec v0.2: Unsupervised Anomaly Detection in Large-Scale System Event Sequences

*Last updated: 2026-11-10. Sections 1–12 written; section 0 to be written last.*

## 0. Summary

*(Write last: problem, gap, method, evaluation, expected result, in 5 sentences.)*

## 1. Problem and motivation

Automated log anomaly detection faces a trade-off: lightweight models are fast but semantically blind, while LLMs read log semantics but are too expensive to process the full log stream and, used alone, raise too many false alarms. This work exploits the complementarity between the two.

### 1.1 Context

Large-scale software systems (cloud platforms, distributed systems, microservices) produce millions to billions of log lines per day. Logs are the main record of how a system actually behaves at runtime. Detecting anomalies in logs early shortens the time to detect and recover from incidents (MTTD, MTTR) and reduces the cost of service disruptions [to cite: He et al., 2021].

### 1.2 Current practice and why it falls short

In practice, log monitoring relies on rules, keywords and alert thresholds, deployed through tools such as the ELK Stack or Splunk. This is insufficient for four reasons:

1. **Volume:** log volume far exceeds what operators can inspect manually.
2. **Anomalies without error keywords:** many incidents show up only as an abnormal order or count of events, while each individual log line looks normal [Xu et al., 2009]. Conversely, words such as "error" or "exception" also appear in normal logs, so keyword rules raise false alarms.
3. **New failures:** rules only catch failure types that are already known.
4. **No labels:** incidents are rare, labeling logs requires experts, and new failure types have no labels. Supervised methods are therefore hard to apply in practice, which motivates methods that do not need anomaly labels.

### 1.3 Existing automated approaches and their tension

**Lightweight models (statistical and deep learning)** such as PCA [Xu et al., 2009], DeepLog [Du et al., 2017] and LogBERT [Guo et al., 2021] handle large volumes at low cost. They have three limitations:

- Logs are represented as template IDs or template counts, so the semantics of the log content, including parameter values, are lost.
- They are sensitive to **log evolution**: when code updates introduce new templates, performance degrades [Zhang et al., 2019].
- They produce many false positives, causing alert fatigue for operators. DeepLog itself, on BGL without online updates and with 1% of the log used for training, reaches a precision of only 0.16; its false positive rate falls from 40.1% to 1.7% only when a human labels false positives at runtime [Du et al., 2017, Fig. 10]. Under a chronological split with 100-line windows, a later re-evaluation finds the same precision, 0.166 at a recall of 0.988 [Guan et al., 2025; see also Le & Zhang, 2022].

**LLMs** read log semantics, can handle unseen templates and can explain alerts in natural language. They have three limitations:

- **Cost and latency** make them unsuitable for processing the full log stream; a recent review notes that no LLM approach yet handles logs in real time [De la Cruz Cabello et al., 2025].
- **Used alone, they over-flag.** Zero-shot LogPrompt reaches a precision of 0.249 on BGL [Liu et al., 2023], and zero-shot ChatGPT labels nearly every window as anomalous on Spirit [Qi et al., 2023].
- **They can assign wrong labels with confidence.** In this work, "hallucination" means an incorrect label decision by the LLM.

Approaches that reach high accuracy with LLMs do so by fine-tuning on labeled anomalies, which brings back the labeling cost and long training times [Guan et al., 2025].

### 1.4 Motivation and gap

The two families fail in complementary ways. Lightweight detectors have high recall but low precision; LLMs read meaning but are costly and, without context, biased toward "anomalous". This suggests a division of labor: a lightweight detector screens the full log stream, and an LLM verifies only the small fraction of windows the detector flags, given similar normal windows as context. The LLM then takes the role the human reviewer plays in DeepLog's online update, without anomaly labels. The detector itself follows the standard assumption of normal-only training data; Section 3 states this assumption and an ablation without it.

Existing LLM approaches either prompt an LLM on every log (LogPrompt, LogGPT), fine-tune it with labels (LogLLM), or replace generation with retrieval of similar normal logs (RAPID) [Liu et al., 2023; Qi et al., 2023; Guan et al., 2025; No et al., 2023]. Among the works reviewed, none lets a lightweight detector decide which windows reach the LLM. The open question is therefore how much this cascade improves on either component alone, at what cost, and whether it is robust to log evolution.

## 2. Research questions and hypotheses

The work answers four questions. The thresholds Δp, Δr, ρ\*, k, Δ₃ and Δ₄ in the hypotheses will be fixed after running the baselines (November 2026 milestone), before building the main method, and recorded in the decision log.

Notation: ρ is the fraction of windows sent to the LLM, selected as the top-ρ% by the lightweight detector's anomaly score (Section 3.5). At the same ρ, detector-only labels every flagged window anomalous; LLM-only sends every window to the LLM, equivalent to ρ = 100%.

Datasets: HDFS and BGL, plus Thunderbird if time allows.

### RQ1: Effectiveness of LLM verification

Compared with the lightweight model alone and the LLM alone, how does the cascade pipeline change precision, recall and F1?

- **H1:** Compared with the lightweight model, the cascade increases precision by at least Δp points while recall drops by no more than Δr points, resulting in a higher F1.
- **Answered by:** a table of precision, recall, F1, specificity and MCC for the trivial baselines, the lightweight model, LLM-only and the cascade, plus two counts for the LLM: stage-1 false positives correctly removed and true anomalies wrongly removed.
- **Cost note:** LLM-only runs on the full BGL test set (≈ 9,427 windows) and on a stratified sample of HDFS test sessions; the sample size is fixed in Section 4.

### RQ2: Accuracy–cost trade-off

As ρ varies, how do F1 and cost (tokens and latency per 1,000 windows) change?

- **H2:** There exists ρ ≤ ρ\* at which the cascade's F1 is at least that of both detector-only and LLM-only, at a cost at least k times lower than LLM-only.
- **Answered by:** F1-versus-cost curves as ρ varies, on each dataset, with the detector-only curve over the same ρ values and LLM-only as the ρ = 100% endpoint. Cost is reported with and without the window cache.

### RQ3: Role of retrieved context

Does giving the LLM similar historical normal windows, retrieved from the training slice, make verification more accurate?

- **H3:** At the same ρ, adding retrieved context increases the cascade's F1 by at least Δ₃ points over zero-shot LLM verification.
- **Constraint:** retrieval draws only from the training slice, which is filtered exactly as the detector's training data; it never uses labels of retrieved windows. Retrieved examples are normal windows only, following evidence that normal examples help and anomalous ones hurt [Qi et al., 2023; No et al., 2023].
- **Answered by:** an ablation table of zero-shot versus retrieval-augmented verification, for 1, 3 and 5 retrieved windows.

### RQ4: Robustness to log evolution

When the test set contains templates absent from the training set, how much does the F1 of the lightweight model and of the cascade degrade?

- **H4:** The cascade's F1 degradation is at least Δ₄ points smaller than the lightweight model's.
- **Definition:** a window is "new-template" if it contains a template id absent from the training vocabulary (Drain `fit_on=stream`).
- **Setup:** chronological split; results reported separately for windows with only known templates and windows with new templates. BGL is preferred because its logs change naturally over time. Because natural new templates may be few, a controlled complement injects LogRobust-style perturbations at 5–20% (word edits; deleted, duplicated or shuffled events) plus meaning-flip edits such as "succeeded" → "failed" [Zhang et al., 2019].
- **Answered by:** an F1 table on known-template versus new-template windows, and F1 versus injection rate.

### Scope notes

- Hallucination, in the sense of wrong label decisions, is measured in RQ1 (false positives and false negatives introduced by the LLM).
- Explanation quality is measured quantitatively as a secondary metric: on datasets with per-line labels (BGL, Thunderbird), HitRate@k checks whether the lines the LLM cites as evidence are truly anomalous. It is also shown qualitatively in the demo.
- RQ4 is secondary. If time runs short, it is reduced to the natural-evolution analysis on BGL only.

## 3. Problem definition

Given a time-ordered log stream, the task is to label each sequence of log lines as normal or anomalous, and to justify each anomalous label with evidence lines, without using anomaly labels anywhere in the method.

### 3.1 Input

- A log stream L = (l₁, …, lₙ), ordered by time. Each line lᵢ = (tᵢ, cᵢ) has a timestamp tᵢ and a content string cᵢ: the message without timestamp and header fields, with parameter values kept.
- A log parser (Drain) maps each content string to a template id eᵢ ∈ E ∪ {UNK}, where E is the set of templates seen in training and UNK marks a template absent from it.
- Ground truth, used for evaluation only: a per-line label yᵢ ∈ {0, 1} where the dataset provides one (BGL, Thunderbird), or a per-session label otherwise (HDFS).

### 3.2 Sequences

Lines are grouped into sequences S, the unit that receives a label.

| Grouping | Datasets | Definition |
|---|---|---|
| Session | HDFS | All lines sharing a `block_id`, in time order |
| Fixed window | BGL, Thunderbird | 100 consecutive lines, non-overlapping (step 100); an incomplete last window is dropped |

A sequence is anomalous if it contains at least one anomalous line: y(S) = maxᵢ∈S yᵢ. For HDFS, y(S) is the session label. The 100-line window matches the setting of Le & Zhang (2022) and Guan et al. (2025), so results can be placed next to their published tables.

### 3.3 Splits

Sequences are ordered by their start time (first line for windows, first line of the session for HDFS) and split chronologically into train D_tr, validation D_val and test D_te, with D_te the last 20%. No sequence crosses a split boundary. HDFS is additionally run with a random split, only to compare against published numbers.

### 3.4 Output

For every test sequence S:

- a label ŷ(S) ∈ {0, 1};
- for every sequence the LLM labels anomalous, a short reason r(S) and a set of evidence lines Ev(S) ⊆ S.

### 3.5 The cascade, formally

1. **Detector.** A lightweight model f, trained on D_tr, gives each sequence an anomaly score s(S) = f(S), higher meaning more anomalous.
2. **Flagging.** For a target flag rate ρ, the threshold τ_ρ is the (1 − ρ) quantile of s over D_val. A sequence is flagged if s(S) > τ_ρ. The threshold uses scores only, never labels; the flag rate actually reached on D_te is reported.
3. **Filter and cache (optional).** Let n(S) be S with parameter values replaced by type tokens. If n(S) equals a normalized training sequence, S is labeled normal without calling the LLM. If n(S) was already verified, the cached verdict is reused.
4. **Retrieval (RQ3).** C_k(S) is the set of k training sequences most similar to S (similarity defined in Section 6); k = 0 means zero-shot.
5. **Verifier.** A frozen LLM g reads the content of S and C_k(S) and returns (v, r, Ev) with verdict v ∈ {0, 1}.
6. **Decision.** ŷ(S) = 0 if S is not flagged; otherwise ŷ(S) = v.

Reference systems at the same ρ:

- **Detector-only** labels every flagged sequence anomalous: ŷ(S) = 1 if s(S) > τ_ρ.
- **LLM-only** sends every sequence to the verifier, equivalent to ρ = 100%.

### 3.6 Assumptions

- **A1 (time order):** training data precede validation data, which precede test data.
- **A2 (normal-only training, main setting):** anomalous sequences are removed from D_tr and D_val. This is the standard assumption of DeepLog, LogBERT and RAPID, often called semi-supervised: removing them uses labels once, before training, as an operator would by curating a clean period of logs.
- **A2′ (unfiltered training, ablation):** D_tr and D_val keep their anomalous sequences. Results under A2′ test whether the method still works when no labels are used at all.
- **A3 (label use):** apart from the filtering in A2, labels are used only for evaluation. They are never used to set thresholds, build prompts, choose retrieved examples or tune the verifier.
- **A4 (LLM):** the LLM is open-weight, run locally, frozen (no fine-tuning) and queried at temperature 0. It sees only sequence content and retrieved training sequences.
- **A5 (parser):** Drain runs online over the stream in time order (`fit_on=stream`); each line's template id is fixed when the line is processed, so later lines cannot change earlier ids.

### 3.7 Terminology

In the title, "unsupervised" means that the method uses no anomaly labels. Under A2 the literature usually calls the setting semi-supervised; this work reports results under both A2 and A2′ and uses that distinction explicitly when comparing with other methods.

## 4. Evaluation

Every system is evaluated on the same chronological test split (Section 3.3), with label-free thresholds (Section 3.5), and judged against a minimum bar fixed now and hypothesis thresholds fixed after the baselines.

### 4.1 Datasets and splits

| Dataset | Grouping | Split | Role |
|---|---|---|---|
| BGL | 100-line windows | Chronological, test = last 20% (≈ 9,427 windows, ≈ 817 anomalous) | Main benchmark; per-line labels for HitRate |
| HDFS | Session (`block_id`) | Chronological by session start; random split as a second run | Comparison with the literature; known to be near saturation |
| Thunderbird (if time allows) | 100-line windows, first 10M lines | Chronological, test = last 20% | Extreme imbalance (≈ 29 anomalous test windows); reported with confidence intervals only |

Dataset statistics and known issues are in Section 5.

### 4.2 Metrics

**Detection (every system).**
- Counts TP, FP, FN, TN, always reported next to percentages.
- Precision, recall, F1, specificity (TN / (TN + FP)) and MCC.
- AU-PR for systems that output a continuous score (detectors).
- Bootstrap 95% confidence intervals for F1 and recall (1,000 resamples of test sequences, fixed seed). Adjacent test sequences are correlated (anomalies come in bursts), so resampling is a circular block bootstrap over the chronological test order [Politis & Romano, 1992], a variant of the moving block bootstrap [Künsch, 1989; Liu & Singh, 1992]; an iid bootstrap [Efron, 1979] would give intervals that are too narrow. The interval is the percentile interval [Efron & Tibshirani, 1993]. Block length is L = ⌈n^(1/3)⌉: the n^(1/3) rate is the optimal order for variance estimation [Hall, Horowitz & Jing, 1995], but the constant 1 is our convention, not taken from the literature; data-driven selection [Politis & White, 2004; Patton, Politis & White, 2009] is the alternative if L is questioned. L is reported with every interval (decision D-b6c5).

**Stage 1 (detector).**
- Realized flag rate on the test set.
- Recall at flag rate 10% and 20%: the share of anomalous sequences the detector would send to the LLM.

**Stage 2 (verifier), computed on flagged sequences only.**
- False positives removed: count and share of stage-1 false positives that the LLM labels normal.
- True positives lost: count and share of stage-1 true positives that the LLM labels normal.
- Format-error rate: share of LLM outputs that cannot be parsed into the required JSON.

**Explanation (BGL and Thunderbird, which have per-line labels).**
- Evidence precision: the share of cited evidence lines that are truly anomalous, averaged over sequences the LLM labels anomalous.
- Hit@k: the share of true-positive sequences in which at least one of the first k cited lines is anomalous (k = 1, 3).

**Cost.**
- LLM calls per 1,000 test sequences, with and without the cache.
- Input and output tokens per 1,000 test sequences.
- Latency per verified sequence (median and 95th percentile) and end-to-end wall-clock time per 1,000 sequences, with the hardware stated.
- Detector training time.

**Robustness (RQ4).**
- F1 on known-template versus new-template sequences.
- F1 as a function of the injection rate (5%, 10%, 15%, 20%).

Point-adjusted F1 is not used: it lets random scores outperform real detectors [Zamanzadeh Darban et al., 2024].

### 4.3 Baselines

| Group | Method | Source | Purpose |
|---|---|---|---|
| Trivial | Always-anomaly | Run | Shows what "flag everything" scores |
| Trivial | Unseen-sequence (anomalous if the template sequence never occurs in training) | Run | Shows how much of a benchmark is solved by novelty alone |
| Normal-only detectors | PCA [Xu et al., 2009] | Run | Classical baseline |
| Normal-only detectors | DeepLog [Du et al., 2017] | Run | Stage-1 detector on its own (detector-only) |
| LLM-only | Our verifier at ρ = 100% | Run | Upper end of the cost curve |
| LLM-only | LogPrompt [Liu et al., 2023] | Run under our protocol, if time allows | Published zero-shot LLM method |
| Training-free retrieval | RAPID [No et al., 2023] | Reported (BGL F1 0.548 [Guan et al., 2025]), or run with a label-free threshold | Strongest method without anomaly labels |
| Supervised (reference) | LogRobust, NeuralLog, LogLLM | Reported [Guan et al., 2025] | Upper bound when labels are available |

Rules for reported numbers:
- They are used only where the protocol matches: BGL and Thunderbird with chronological 100-line windows. Published HDFS numbers use a random split and are compared only with our random-split run.
- They are marked "reported" in every table, with their source, and never averaged with our runs.

### 4.4 Protocol details

- **Thresholds.** Detector-only is reported at the default threshold (99th percentile of validation scores) and across the ρ grid. No threshold reads validation or test labels.
- **ρ grid.** 1, 2, 5, 10, 15, 20, 30, 50 and 100%.
- **Retrieval.** k = 0 (zero-shot), 1, 3 and 5.
- **Seeds.** Trained detectors run with 3 seeds; tables show mean ± std.
- **LLM determinism.** The LLM runs once at temperature 0. A 500-sequence subset is run a second time and the label agreement between the two runs is reported.
- **LLM-only on HDFS.** It runs on a stratified sample of 10,000 test sessions that keeps the test anomaly ratio. The cascade is reported on the same sample as well as on the full test set. The sample size may shrink if the cost estimate from BGL requires it; any change is logged.
- **Data contamination.** For 200 lines per dataset, the LLM is given the first half of a line and asked to complete it; the exact-match rate is reported. A high rate is discussed as a threat to validity.
- **Comparing two systems.** A difference in F1 counts as real if the 95% paired bootstrap interval of the difference, computed on the same test sequences, excludes 0. The paired bootstrap follows Koehn (2004); here both systems are resampled with the same circular blocks (Section 4.2).

### 4.5 Ablations

| Ablation | Values | Question | RQ |
|---|---|---|---|
| Flag rate ρ | Grid in 4.4 | How do F1 and cost trade off? | RQ2 |
| Retrieved context | k = 0, 1, 3, 5 | Does similar normal context help the LLM? | RQ3 |
| Verifier input | Raw line / content / template | Do parameter values help the LLM? | RQ1 |
| Reasoning | Verdict only / reason before verdict | Does asking for a reason change decisions? | RQ1 |
| Filter and cache | Off / on | How much LLM cost do they save, and do they change F1? | RQ2 |
| LLM size | 2–3 open-weight models of different sizes | Is a small model enough? | RQ2 |
| Parser | Drain `stream` / Drain `train` | How sensitive is stage 1 to how the parser is fitted? | RQ4 |
| Training data | A2 (filtered) / A2′ (unfiltered) | Does the method work with no labels at all? | All |

### 4.6 Success criteria

**Minimum bar (fixed now).** On BGL, at its best ρ, the cascade must:
1. beat both trivial baselines and detector-only at the same ρ in F1, with a paired bootstrap interval excluding 0;
2. remove more stage-1 false positives than it loses true positives.

If either condition fails, the result is reported as a negative finding with an analysis of why (Section 9).

**Hypothesis thresholds (fixed after the baselines).** Δp, Δr, ρ\*, k, Δ₃ and Δ₄ in H1–H4 are set once detector-only and LLM-only results exist on BGL. They are recorded in the decision log with the date, before any cascade run.

### 4.7 Reporting

- Every table states the dataset, split, seeds and threshold method.
- Percentages are always shown next to counts.
- Thunderbird results carry confidence intervals and are not used for the main conclusions.

## 5. Data

The work uses three public Loghub datasets: BGL as the main benchmark, HDFS for comparison with the literature, and Thunderbird as an extremely imbalanced third case. All three are known to have weaknesses, listed in 5.3, and the evaluation is designed around them.

### 5.1 Source and handling

- **Source:** Loghub [Zhu et al., 2023], downloaded from Zenodo record 8196385 with checksum verification.
- **Storage:** raw files stay in `data/raw/<dataset>/` and are never committed. Parsed and processed data are cached under a hash of the configuration that produced them, so every result can be traced to its exact preprocessing.
- **Development slice:** the first 100,000 lines of each dataset, used only to develop and test code. No reported result comes from it.

### 5.2 Dataset statistics

Figures below come from the cited papers. Our own counts are printed by the pipeline's `[data]` log section and replace these once Stage 3 and Stage 7 runs exist.

| | HDFS_v1 | BGL | Thunderbird (first 10M lines) |
|---|---|---|---|
| System | Hadoop file system, 203 EC2 nodes [Xu et al., 2009] | BlueGene/L supercomputer, LLNL | Thunderbird supercomputer, Sandia |
| Lines | 11,175,629 | 4,747,963 | 10,000,000 |
| Anomalous lines | – (labels per session) | 348,460 (7.34%) | 4,937 (0.049%) |
| Label level | Session (`block_id`), from `anomaly_label.csv` | Line: first field `-` = normal, otherwise an alert category | Line, same convention as BGL |
| Grouping | 575,061 sessions | 100-line windows: 47,135 | 100-line windows: 99,997 |
| Anomalous sequences | 16,838 (2.93%) | Train part 4,009 (10.63%); test 817 (8.67%) | Train part 837 (1.05%); test 29 (0.15%) |
| Test sequences (last 20%) | ≈ 115,000 | 9,427 | 20,000 |

Sources: Guan et al. (2025) for line counts and window statistics; Xu et al. (2009) for HDFS system details.

### 5.3 Known issues and how the evaluation handles them

| Dataset | Issue | Consequence | Handling |
|---|---|---|---|
| HDFS | 575K sessions reduce to about 680 distinct event-count vectors in the original data, and labels were assigned per distinct vector [Xu et al., 2009] | Near-perfect scores are easy, especially under a random split, where identical sequences fall in both train and test | HDFS is secondary; we report a run with sequences deduplicated between train and test |
| HDFS | Sessions span time, so a session can start before the split point and end after it | A strict "every test timestamp after every train timestamp" split is impossible | Split by session start time (Section 3.3) |
| HDFS | Published numbers almost all use a random split | Our chronological results are not comparable to them | A second, random-split run is kept for that comparison only |
| BGL | Labels mark alert categories, which are largely tied to specific templates | Novelty alone may solve much of the benchmark | The unseen-sequence baseline (Section 4.3) measures this directly |
| BGL, Thunderbird | A window is anomalous if any one of its lines is | Larger windows inflate the anomaly ratio (for example 18% on BGL with 1-hour windows [Le & Zhang, 2022]) | Fixed 100-line windows, the same as the reference papers |
| Thunderbird | Only about 29 anomalous windows in the test split | One wrong window changes recall by about 3 points | Counts and bootstrap intervals are reported; no main conclusion rests on Thunderbird |
| Thunderbird | The anomaly ratio falls from 1.05% in the training part to 0.15% in the test split | Thresholds and detectors see a different distribution at test time | Stated as a threat to validity |
| All | The datasets have been public for years | An LLM may have seen them during pre-training | Contamination check (Section 4.4) |

### 5.4 Preprocessing

1. **Content extraction:** each dataset's header fields (timestamp, node, level, component) are separated from the message content. Parameter values stay in the content.
2. **Parsing:** Drain runs online over the stream in time order (`fit_on=stream`, assumption A5). Template ids are fixed when each line is processed.
3. **Grouping and splitting:** as in Sections 3.2 and 3.3. The proportions are 70% train, 10% validation and 20% test, all chronological. The test split is therefore identical to the reference papers, and validation is taken from what those papers used as training data.
4. **Filtering:** anomalous sequences are removed from train and validation under assumption A2 and kept under A2′.
5. **Vocabulary:** built from training templates only; any other template maps to `<UNK>`.
6. **Normalization for the cache and retrieval:** parameter values are replaced by type tokens (IP, path, number, block id), following RAPID [No et al., 2023]. The verifier still receives the original content.

## 6. Method

The method is a two-stage cascade: DeepLog scores every sequence, a cheap filter and cache remove sequences that need no LLM call, and a frozen open-weight LLM verifies the rest, optionally with retrieved normal sequences as context. Section 3.5 gives the formal pipeline; this section fixes each component and explains why it was chosen over the alternative.

### 6.1 Stage 1: lightweight detector

- **Model:** DeepLog [Du et al., 2017]: a 2-layer LSTM with 64 units that predicts the next template id from the previous h = 10. Defaults follow the original paper (g = 9).
- **Sequence score:** the score of a sequence is 1 − min p(eᵢ | previous h events) over its positions, so one very unlikely event is enough to raise it (`window_score=min_prob`). The count of events outside the top-g predictions (`count_outside_topg`) is kept as an alternative. A template unseen in training (`<UNK>`) has probability 0 and always counts as outside the top-g.
- **Threshold:** the flag-rate quantile on validation scores (Section 3.5), never a fixed top-g rule.

Why:
- **DeepLog rather than LogBERT.** On BGL under the reference protocol, both reach the same F1 (0.285 versus 0.283), but DeepLog trains about 6 times faster and tests about 13 times faster [Guan et al., 2025, Table III]. It also raises alarms earliest [Le & Zhang, 2022].
- **DeepLog rather than PCA.** PCA uses event counts and ignores order, so it cannot see missing or reordered events. PCA remains a baseline.
- **A continuous score rather than the original top-g yes/no rule.** The cascade needs to rank sequences to send exactly the top ρ% to the LLM, and the yes/no rule gives no ranking.
- **A flag-rate threshold rather than one tuned for F1.** Tuning for F1 needs labels. A flag rate also fixes the LLM cost in advance.

### 6.2 Filter and cache

- **Normalization:** parameter values are replaced by type tokens (IP, path, number, block id), following RAPID [No et al., 2023].
- **Filter:** a flagged sequence whose normalized form exactly equals a normalized training sequence is labeled normal without calling the LLM. The filter runs only under assumption A2, where training sequences are known to be normal; under A2′ it is switched off.
- **Cache:** the verdict for each normalized sequence is stored, and identical normalized sequences reuse it.

Why:
- **An exact match rather than a similarity threshold.** A threshold would have to be tuned, and tuning it well needs labels. An exact match to a known-normal sequence needs no tuning.
- **A cache.** Normalized windows repeat often in system logs, and at temperature 0 the LLM would return the same verdict anyway. The cache changes cost, not accuracy, which the filter-and-cache ablation (Section 4.5) checks.

### 6.3 Stage 2: LLM verifier

**Model**
- An open-weight, instruction-tuned LLM, served locally, frozen and queried at temperature 0 (assumption A4).
- The main model has 7–14B parameters; the size ablation adds one smaller and one larger model.
- Concrete models are chosen after measuring prompt length on BGL (below). Requirements: open weights, a context window large enough for the target sequence plus five retrieved sequences, and reliable JSON output.

**Input**
- The sequence's content lines, numbered so the LLM can cite them, with timestamps and header fields removed and parameter values kept.
- Consecutive identical lines are collapsed into one line with a repeat count (for example `[×37]`) to shorten the prompt.
- Under RQ3, the k retrieved normal sequences, labeled as examples of normal behavior of the same system.

**Prompt**
- The task is framed as reviewing a window that an automatic detector flagged, and deciding whether it shows a real problem.
- The prompt states that words such as "error" or "exception" can appear during normal operation.
- The LLM writes its reasoning first and its verdict last.
- Output is JSON: `{"reason": …, "evidence_lines": […], "verdict": "normal" | "anomalous"}`.

**Prompt development**
- The prompt is written from domain knowledge and checked only on the development slice (Section 5.1).
- Changes are limited to format and clear failure cases; no accuracy-driven tuning on labeled data is allowed.
- Every prompt version is logged, and the final version is frozen before any test run.

**Failure handling**
- An output that does not parse is retried once with a format reminder.
- If it still fails, the sequence keeps the detector's label (anomalous) and counts as a format error.

Why:
- **Content rather than templates.** Templates drop parameter values, which carry information: on BGL, LogLLM reaches a precision of 0.943 with raw text versus 0.861 with masked parameters [Guan et al., 2025, Table IV], and the original HDFS study found its signal in variable values [Xu et al., 2009].
- **Content rather than full raw lines.** Timestamps and node ids add noise; ChatGPT performed better on content than on raw lines [Qi et al., 2023].
- **A whole window rather than one template at a time.** LogPrompt labels each template separately and cannot see order or missing events [Liu et al., 2023].
- **Reasoning before the verdict.** Asking for a reason raised LogPrompt's session F1 on BGL from 0.189 to 0.307 [Liu et al., 2023]. The cited lines also make Hit@k measurable (Section 4.2).
- **A frozen model rather than fine-tuning.** Fine-tuning needs labeled anomalies and long training: LogLLM needs about 18 hours on a 48 GB GPU [Guan et al., 2025].
- **An open-weight local model rather than a commercial API.** Logs can contain sensitive data, pinned weights make results reproducible, and there is no per-call fee [De la Cruz Cabello et al., 2025]. The cost is that small open models may be weaker than the largest commercial ones; the size ablation measures this.
- **Keeping the detector's label on a format failure.** It never loses recall because of an output error, and format errors are reported separately.

### 6.4 Retrieval (RQ3)

- **Index:** normalized training sequences, filtered under A2 and unfiltered under A2′. Labels of retrieved sequences are never read.
- **Candidates:** the 50 nearest training sequences by cosine similarity of sentence embeddings from a general-purpose pretrained encoder.
- **Reranking:** token-level maxSim, as in RAPID [No et al., 2023]; the top k distinct sequences are kept (k = 1, 3, 5).
- **Exclusion:** exact normalized matches are not retrieved, since the filter (6.2) already handles them.

Why:
- **Normal examples only.** In-context normal examples raised ChatGPT's F1 on Spirit, while anomalous examples made it collapse to labeling everything normal [Qi et al., 2023]. Normal examples also need no anomaly labels.
- **A few nearest sequences rather than many.** Comparing with the single nearest normal sequence clearly beat averaging over several in RAPID (F1 0.9999 versus 0.4659 on BGL in their setting), because further neighbors belong to unrelated log types [No et al., 2023].
- **Embeddings with token-level reranking rather than keyword search alone.** Log lines mix identifiers with natural-language words; the token-level step keeps small differences, such as one changed word, visible.

### 6.5 Summary of design choices

| Component | Chosen | Main alternative | Deciding reason |
|---|---|---|---|
| Detector | DeepLog | LogBERT | Same F1 on BGL, much cheaper |
| Sequence score | Continuous (min probability) | Top-g yes/no | Ranking is needed to control ρ |
| Threshold | Flag-rate quantile on validation | Best F1 | No labels, fixed cost |
| Filter | Exact normalized match | Similarity threshold | No tuning, no labels |
| Verifier input | Content with parameters, whole window | Templates, one at a time | Keeps parameter values and order |
| Reasoning | Reason before verdict | Verdict only | Better decisions, measurable evidence |
| LLM | Open-weight, frozen, local | Fine-tuned, or commercial API | No labels, privacy, reproducibility |
| Retrieved context | 1–5 nearest normal sequences | Labeled or many examples | Normal examples help; distant ones add noise |

## 7. Related work and gap

Prior work either detects cheaply without semantics, uses semantics only with labels, or uses an LLM on every log; none of the works reviewed lets a lightweight detector decide which windows an LLM verifies, measures the resulting accuracy–cost trade-off, or checks LLM explanations against line labels.

### 7.1 Five groups of methods

**Lightweight detectors without anomaly labels.** PCA [Xu et al., 2009] flags sessions whose event-count vectors fall outside the normal subspace. DeepLog [Du et al., 2017], LogAnomaly [Meng et al., 2019] and LogBERT [Guo et al., 2021] learn normal event sequences and flag unlikely events; LogGPT [Han et al., 2023] replaces the LSTM with a small GPT-2 tuned by reinforcement learning. All of them work on template ids, are cheap to run, and under a chronological split reach high recall but low precision on BGL. DeepLog's own remedy for false positives is a human who labels them at runtime [Du et al., 2017].

**Supervised detectors.** LogRobust [Zhang et al., 2019] adds semantic template vectors and attention to resist log evolution. NeuralLog [Le & Zhang, 2021] and LogLLM [Guan et al., 2025] read log text with pretrained language models; LogLLM couples BERT with a fine-tuned Llama-3-8B and reaches the best published BGL F1. All need labeled anomalies; 1% wrong labels cuts LogRobust's F1 by about 37% [Le & Zhang, 2022], and LogLLM needs about 18 hours of training on a 48 GB GPU.

**LLM prompting.** LogPrompt [Liu et al., 2023] prompts ChatGPT zero-shot on each template and adds natural-language explanations, rated by human experts. Qi et al. (2023) prompt ChatGPT on whole windows. RAGLog [Pan et al., 2023] retrieves normal log entries for an LLM to compare against. These methods need no training data but run the LLM on every log, and used alone they over-flag (Section 1.3).

**Training-free retrieval.** RAPID [No et al., 2023] compares each test log with its nearest normal log in a pretrained encoder's embedding space, with no LLM and no training. Under the reference protocol its precision is high but its recall is low.

**Evaluation studies and reviews.** Le & Zhang (2022) show that random splits, large windows and missing specificity inflate published results; their protocol is the one this work follows. Zamanzadeh Darban et al. (2024) review time-series anomaly detection and its metrics, including thresholding without labels. De la Cruz Cabello et al. (2025) review LLMs for log anomaly detection and note that none yet handles logs in real time.

**Outside logs.** In industrial visual inspection, Myriad uses a pretrained anomaly detector as an expert that guides a multimodal LLM, a structural precedent for detector-plus-LLM pipelines [cited via Li et al., 2025; primary source to be read].

### 7.2 Comparison under the reference protocol

BGL, chronological 80/20 split, 100-line windows [Guan et al., 2025, Table II], unless noted.

| Method | Group | Anomaly labels | Reads log text | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| DeepLog | Lightweight | No (normal-only training) | No | 0.166 | 0.988 | 0.285 |
| LogAnomaly | Lightweight | No (normal-only training) | Template vectors | 0.176 | 0.985 | 0.299 |
| LogBERT | Lightweight | No (normal-only training) | No | 0.165 | 0.989 | 0.283 |
| RAPID | Training-free retrieval | No (normal reference set) | Yes | 0.874 | 0.399 | 0.548 |
| LogRobust | Supervised | Yes | Template vectors | 0.696 | 0.968 | 0.810 |
| NeuralLog | Supervised | Yes | Yes | 0.792 | 0.884 | 0.835 |
| LogLLM | Supervised | Yes | Yes | 0.861 | 0.979 | 0.916 |
| LogPrompt* | LLM prompting | No | Yes (per template) | 0.249 | 0.834 | 0.384 |

\* Different protocol: baselines trained on the first 4,000 lines only [Liu et al., 2023]; shown for scale, not for direct comparison.

### 7.3 Closest works and how this work differs

| Closest work | What it does | Difference here |
|---|---|---|
| DeepLog with online update [Du et al., 2017] | A human labels the detector's false positives | An LLM replaces the human; no labels at runtime |
| LogPrompt [Liu et al., 2023], Qi et al. (2023) | An LLM labels every log or window | The LLM sees only flagged windows, plus normal context |
| RAPID [No et al., 2023] | Nearest-normal comparison decides alone | Nearest normal windows become context for an LLM decision |
| LogLLM [Guan et al., 2025] | Fine-tuned LLM with labels | Frozen LLM, no labels, a fraction of the compute |

### 7.4 Gap

1. **No detector-gated LLM verification.** Among the works reviewed, no log anomaly detector decides which windows an LLM verifies (RQ1).
2. **No accuracy–cost curve.** LLM methods report accuracy at one operating point; none reports how accuracy changes with the share of logs sent to the LLM (RQ2).
3. **Context for verification untested in a cascade.** Retrieval has been used as a detector (RAPID) or as LLM context on every log (RAGLog), not as context for verifying flagged windows (RQ3).
4. **Explanations judged by impression.** LogPrompt's explanations were rated for usefulness and readability, not checked against which lines were truly anomalous (Section 4.2).
5. **Robustness of LLM-based methods to log evolution.** Injection tests exist for supervised models [Zhang et al., 2019] but not for LLM verification (RQ4).

The literature search so far covers the works listed in the references. Before submission, it will be extended with a systematic search for LLM-based log analysis and for model cascades or routing in other domains, and this section will be updated.

## 8. Scope

The thesis builds and evaluates one cascade (DeepLog plus a frozen open-weight LLM) on three public datasets, offline; it does not aim to beat supervised methods or to build a production monitoring system.

### 8.1 In scope

- **Pipeline:** DeepLog as the stage-1 detector, the filter and cache, a frozen open-weight LLM verifier, and retrieval of normal sequences as context (Section 6).
- **Data:** HDFS, BGL and, if time allows, Thunderbird, under the protocol in Section 4.
- **Experiments:** the four research questions, the baselines in Section 4.3 and the ablations in Section 4.5.
- **Evaluation of explanations:** evidence precision and Hit@k on datasets with per-line labels.
- **Demo:** a replay of BGL test windows showing detector scores, LLM verdicts, reasons and evidence lines (Section 11).
- **Code:** a reproducible repository with configuration files, fixed seeds and logged runs.

### 8.2 Out of scope

| Item | Why it is excluded |
|---|---|
| Fine-tuning the LLM | Needs labeled anomalies and large compute; contradicts the no-label setting (Section 6.3) |
| Online or incremental learning of the detector | The detector is trained once per run; DeepLog's online update needs human labels |
| Other stage-1 detectors as main results (LogBERT, LogGPT, PCA) | DeepLog is enough to answer the research questions; other detectors appear only as baselines or as reported numbers |
| Log parsing research | Drain is used as-is; a second parser appears only as an ablation |
| Root-cause analysis and repair suggestions | The task stops at detecting and justifying an anomaly |
| Commercial LLM APIs | Privacy, reproducibility and cost (Section 6.3) |
| New or private datasets | Only public Loghub datasets, so results can be reproduced |
| Real-time streaming deployment | Experiments run offline on recorded logs; latency is measured per sequence instead |
| Multi-modal signals (metrics, traces) | Only log text is used |

### 8.3 Non-goals

- **Beating supervised methods in F1.** LogLLM (BGL F1 0.916) uses labels and hours of fine-tuning; the aim is to move a label-free pipeline from the normal-only range (F1 ≈ 0.28–0.30) toward the supervised range at a controlled cost.
- **Finding the best LLM.** The size ablation answers whether a small open model is enough, not which model is best overall.
- **Optimizing prompts for accuracy.** Prompt changes are limited to format and clear failures (Section 6.3).
- **Proving the method generalizes to all systems.** Results hold for the three datasets used; Section 9 discusses what may not transfer.

### 8.4 Reductions if time runs short

In this order:

1. Drop Thunderbird; keep HDFS and BGL.
2. Run LogPrompt only on BGL, or report it as a published number with its protocol noted.
3. Reduce RQ4 to the natural-evolution analysis on BGL, without injection.
4. Reduce the LLM size ablation to two models.
5. Run retrieval with k = 1 and 3 only.

RQ1 and RQ2 on BGL, the minimum bar in Section 4.6, and the demo are never reduced.

## 9. Risks and fallbacks

The largest risk is that the LLM does not remove enough false positives, because zero-shot LLMs over-flag logs (Section 1.3); every other risk has a cheaper fallback, and a negative result on the main risk still yields a complete thesis (9.2).

### 9.1 Risks

| ID | Risk | Early signal | Mitigation | Fallback |
|---|---|---|---|---|
| R1 | The LLM labels most flagged windows anomalous, removing few false positives | The LLM's share of "anomalous" verdicts on flagged BGL windows is close to 100% in the first cascade run | Retrieved normal context (RQ3); prompt framing that normal logs contain error words; a larger model | Report as a negative result with error analysis (9.2) |
| R2 | The LLM removes true anomalies, so recall falls more than precision rises | True positives lost exceed false positives removed (minimum bar, Section 4.6) | Add an "uncertain" verdict mapped to anomalous; keep reasoning before the verdict | Report the trade-off curve; recommend the cascade only at the ρ values where the bar holds |
| R3 | Prompts are too long or the LLM is too slow | Prompt-length measurement on BGL before choosing the model (Section 6.3); latency per sequence | Collapse repeated lines; cache; lower ρ; send only lines around the detector's least likely events | Smaller model; k ≤ 3; LLM-only on a sample only |
| R4 | Not enough GPU memory or time | The chosen model does not fit or runs below about 1 sequence per second | 4-bit quantization; a 7B model; batching | Drop the largest model from the size ablation |
| R5 | DeepLog does not reproduce the reference numbers | Stage 6–7 results outside the ranges in `plan.md` | Compare with deep-loglizer and LogADEmpirical under the same split | Use the LogADEmpirical implementation as stage 1 and document the gap |
| R6 | Novelty alone solves most of BGL | The unseen-sequence baseline reaches an F1 close to DeepLog or higher | Report results separately for windows with only known templates; give more weight to RQ4 | Reframe the contribution around cost and verified explanations |
| R7 | The LLM has seen the datasets in pre-training | High exact-match rate in the contamination check (Section 4.4) | Rename identifiers and perturb parameter values, then re-run a subset | Report as a threat to validity with the measured rate |
| R8 | The LLM is not deterministic at temperature 0 | Label agreement below 95% on the repeated 500-sequence subset | Majority vote over 3 runs for disagreeing sequences | Report agreement and the F1 range across runs |
| R9 | Many outputs fail to parse | Format-error rate above 2% | Constrained JSON decoding in the serving library | Detector's label is kept (Section 6.3); format errors reported |
| R10 | The schedule slips | A milestone in Section 10 missed by more than one week | Reductions in Section 8.4, in order | Kill criteria in Section 10 |

H3 or H4 being false is not a risk to the thesis: either is a valid answer to its research question and is reported as such.

### 9.2 If the main hypothesis fails

If the cascade fails the minimum bar in Section 4.6, the thesis still delivers:

1. **A strict re-evaluation** of trivial baselines, PCA and DeepLog under a chronological, label-free protocol on HDFS, BGL and Thunderbird.
2. **Accuracy–cost curves** for detector-only, LLM-only and the cascade, showing where the LLM helps and where it does not.
3. **An error analysis of the LLM's decisions** on flagged windows, grouping failures by cause (for example error-word bias, missing context, unseen templates) with examples.
4. **Guidance** on when LLM verification of log alerts is not worth its cost.

### 9.3 Threats to validity

**Internal validity**
- Assumption A2 uses labels once to filter training data. The A2′ ablation measures how much results depend on it.
- The prompt is checked on a labeled development slice. Changes are limited to format and clear failures, and every version is logged (Section 6.3).
- Implementation errors are guarded by the tests in each stage of `plan.md` and by reproducing published baseline numbers.

**Construct validity**
- A window is labeled anomalous if any one line is, which ties the difficulty of the task to the window size.
- BGL labels mark alert categories, which are close to template identity, so they may reward novelty more than understanding (R6).
- Hit@k measures whether the LLM points at anomalous lines, not whether its reasoning is correct or useful.
- Cost depends on the hardware and serving library; both are reported with every cost figure.

**External validity**
- The three datasets come from supercomputers and a Hadoop cluster recorded years ago. Results may not transfer to modern microservice logs, whose formats, volumes and failure types differ.
- One detector and a few open-weight LLMs are tested; other combinations may behave differently.
- Experiments replay recorded logs offline; a live system adds concerns, such as bursts and back-pressure, that are not measured.

**Conclusion validity**
- Thunderbird has about 29 anomalous test windows, so its results carry wide intervals and support no main conclusion.
- Trained detectors use 3 seeds; differences are judged by paired bootstrap intervals, not by point estimates alone.
- Possible data contamination of the LLM (R7) may inflate LLM results; the measured rate is reported.

## 10. Plan, milestones and kill criteria

Work runs in five milestones from November 2026 to March 2027, ahead of the defense in April 2027; each milestone ends with a dated decision, and results are frozen on 2027-03-15. Implementation stages and their tests are in `docs/plan.md`. A milestone that finishes early lets the next one start early.

### 10.1 Milestones

| Milestone | Target date | Deliverables | Done when |
|---|---|---|---|
| M0. Data pipeline | Done (2026-11-10) | Plan stages 0–3 merged | Leakage tests pass; follow-up branch `fix/stage-1-3-protocol` merged before Stage 5 |
| M1. Baselines | 2026-11-30 | Plan stages 4–7: trivial baselines, PCA and DeepLog on HDFS and BGL, 3 seeds; Thunderbird if time allows | Baseline table in `results/tables/`; DeepLog within the reference ranges; hypothesis thresholds (Section 4.6) fixed in the decision log |
| M2. Verifier | 2026-12-31 | Prompt-length measurement; LLM chosen; prompt frozen; LLM-only on BGL test and the HDFS sample; contamination and determinism checks; first cascade run on BGL at ρ = 10% and 20% | LLM-only and first cascade results logged with cost figures |
| M3. Cascade (RQ1, RQ2) | 2027-01-31 | Filter and cache; full ρ sweep on HDFS and BGL; stage-2 and cost metrics; input and reasoning ablations | RQ1 and RQ2 tables and F1-versus-cost curves; minimum bar checked |
| M4. Context and robustness (RQ3, RQ4) | 2027-02-28 | Retrieval with k = 1, 3, 5; RQ4 natural-evolution split and injection; parser, A2′ and LLM-size ablations; Thunderbird runs if not done | RQ3 and RQ4 tables; all ablation rows filled |
| M5. Write-up and demo | 2027-03-31 | Thesis draft; demo (Section 11); results frozen on 2027-03-15 | Full draft sent to the advisor; demo runs end to end on BGL |
| Defense | April 2027 | Slides and rehearsal | – |

Rough compute check for M2: at 1 sequence per second, LLM-only takes about 2.6 hours on the BGL test set (9,427 windows) and about 2.8 hours on the 10,000-session HDFS sample.

### 10.2 Kill criteria and decision points

| ID | Date | Condition | Decision |
|---|---|---|---|
| K1 | 2026-11-30 | DeepLog is still outside the reference ranges | Switch stage 1 to the LogADEmpirical implementation (R5), allow at most one extra week, document the gap |
| K2 | 2026-12-15 | No candidate LLM fits the available GPU at about 1 sequence per second or faster | Use the smallest candidate with 4-bit quantization; run LLM-only on samples only (R3, R4) |
| K3 | 2027-01-15 | On BGL at ρ = 10% and 20%, the cascade removes fewer stage-1 false positives than it loses true positives, after trying the frozen prompt, the "uncertain" verdict and retrieval with k = 1 | Stop improving the method; follow the negative-result path in Section 9.2 and use the remaining time for analysis, RQ2 curves and the error study |
| K4 | 2027-02-28 | M4 is not complete | Apply the reductions in Section 8.4, in order, until it fits |
| K5 | 2027-03-15 | Results freeze | No new experiments; only writing, figures and the demo |

### 10.3 Working rules

- Every decision that changes the protocol, a threshold or the scope is recorded in `notes/decisions.md` with its date and reason.
- Every reported number comes from a logged run in `experiments/runs.jsonl`; no number is typed into the thesis by hand.
- The spec is updated when a decision changes it, and its version and date are bumped.

## 11. Demo

The demo replays BGL test windows through the cascade and shows, for each flagged window, what the detector saw, what the LLM decided and which lines it cited, next to the ground truth. It runs offline on a laptop from frozen results, so it does not depend on a GPU or network at the defense.

### 11.1 Purpose

- Show the committee how the cascade works on real logs, one window at a time.
- Make the accuracy–cost trade-off (RQ2) concrete.
- Show explanations next to the true anomalous lines, including cases where the LLM is wrong.

### 11.2 Views

1. **Stream view.** BGL test windows play in time order. Each window shows its detector score and its path through the cascade: not flagged; flagged and filtered; flagged and taken from the cache; or flagged and verified by the LLM. Running counters show windows processed, LLM calls made and alerts raised by detector-only versus the cascade.
2. **Window view.** For one window: the numbered content lines; the lines DeepLog found least likely; the LLM's verdict, reason and evidence lines; the retrieved normal windows, if any; and the ground-truth anomalous lines highlighted for comparison.
3. **Trade-off view.** A ρ slider that shows precision, recall, F1, LLM calls and latency per 1,000 windows for detector-only and the cascade at that ρ, read from the frozen results of the ρ sweep.

### 11.3 Cases shown

Four kinds of window are prepared, each picked by a fixed rule (a seeded random draw within its category) rather than by hand, so the demo does not cherry-pick:

| Case | What it shows |
|---|---|
| True anomaly kept | The LLM confirms a real anomaly and cites the right lines |
| False positive removed | The LLM clears a window DeepLog flagged, for example one with error words or a new template |
| True anomaly lost | The LLM wrongly clears a real anomaly, shown openly with its reason |
| False positive kept | The LLM agrees with a wrong detector alert |

### 11.4 Data and modes

- **Replay mode (default):** reads the frozen run artifacts (`scores.parquet`, LLM outputs, metrics) produced by the final experiments. No model runs during the demo.
- **Live mode (optional):** runs the verifier on 3–5 chosen windows, if a GPU is available at the defense, to show that the replay is not scripted.

### 11.5 Implementation

- A small web app built with a Python UI framework, such as Streamlit, reading files from a results directory.
- One command starts it, for example `uv run demo --run <run_id>`.
- No new results are computed for the demo; every number shown is traceable to a logged run.

### 11.6 Done when

- The demo starts with one command on a laptop with no GPU and no network.
- All three views work on the frozen BGL results.
- Each of the four cases in 11.3 has at least one example, picked by the fixed rule.
- A 5-minute walkthrough script exists and has been rehearsed once.

## 12. Open questions

Fourteen questions remain open. Each has a date or milestone by which it must be answered. Once answered, a question is removed from this list and recorded in `notes/decisions.md` with its date, the decision, the alternatives considered and the reason.

| # | Question | Why it matters | Decide by |
|---|---|---|---|
| Q1 | What GPU and how many GPU hours are available? | Sets which LLMs are feasible and how large the LLM-only runs can be | 2026-11-30 (M1) |
| Q2 | Who owns which milestone in the two-person team? | Lets milestones run in parallel | 2026-11-30 (M1) |
| Q3 | Is the 70 / 10 / 20 chronological split final? | Validation size affects thresholds; train size affects DeepLog | Before plan Stage 5 |
| Q4 | Are the default detector threshold (99th percentile of validation scores) and the ρ grid final? | They fix the operating points of every result | After validation score distributions are seen in M1 |
| Q5 | What values do Δp, Δr, ρ\*, k, Δ₃ and Δ₄ take? | They turn H1–H4 into testable claims | 2026-11-30 (M1), before any cascade run |
| Q6 | Which LLMs are the main model and the size-ablation models? | Every stage-2 result depends on them | 2026-12-15 (K2), after measuring prompt length on BGL |
| Q7 | Does the verifier output include an "uncertain" verdict from the start, or only as a mitigation for R2? | Changes the output format and the recall of the cascade | 2026-12-15, before the prompt is frozen |
| Q8 | Is the LLM-only sample of 10,000 HDFS test sessions affordable? | Sets the cost of RQ1 and RQ2 on HDFS | After LLM-only on BGL (M2) |
| Q9 | Which sentence encoder is used for retrieval? | Sets retrieval quality and speed for RQ3 | Before M4 work starts; at the latest 2027-01-15 for the k = 1 trial in K3 |
| Q10 | Are the risk thresholds right: 1 sequence per second (R4), 95% agreement (R8), 2% format errors (R9)? | They trigger fallbacks | 2026-12-15 (M2) |
| Q11 | Is Thunderbird run or dropped? | First item in the reductions of Section 8.4 | 2027-02-28 (K4) |
| Q12 | Is LogPrompt run under our protocol, or only reported with its protocol noted? | Adds the strongest published zero-shot LLM baseline | 2027-01-31 (M3) |
| Q13 | Does a systematic search find prior model cascades or LLM routing for log analysis or other tasks? | May change gap 1 in Section 7.4 from "not done" to "not done for log anomaly detection" | Before the thesis draft, 2027-03-15 |
| Q14 | Which framework runs the demo (Streamlit or another)? | Affects the demo build in M5 | 2027-02-28 |

## References

- Xu et al. (2009). Detecting Large-Scale System Problems by Mining Console Logs. SOSP.
- Du et al. (2017). DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning. CCS.
- Meng et al. (2019). LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies in Unstructured Logs. IJCAI.
- Zhang et al. (2019). Robust Log-Based Anomaly Detection on Unstable Log Data (LogRobust). ESEC/FSE.
- Le & Zhang (2021). Log-based Anomaly Detection Without Log Parsing (NeuralLog). ASE.
- Guo et al. (2021). LogBERT: Log Anomaly Detection via BERT. IJCNN.
- He et al. (2021). A Survey on Automated Log Analysis for Reliability Engineering. ACM Computing Surveys.
- Le & Zhang (2022). Log-based Anomaly Detection with Deep Learning: How Far Are We? ICSE.
- Liu et al. (2023). LogPrompt: Prompt Engineering Towards Zero-Shot and Interpretable Log Analysis. arXiv:2308.07610 (official version: ICSE 2024 Companion / ICPC 2024).
- Qi et al. (2023). LogGPT: Exploring ChatGPT for Log-Based Anomaly Detection. arXiv:2309.01189.
- Han, Yuan & Trabelsi (2023). LogGPT: Log Anomaly Detection via GPT. IEEE BigData. arXiv:2309.14482.
- Pan, Wong & Yuan (2023). RAGLog: Log Anomaly Detection using Retrieval Augmented Generation. arXiv:2311.05261.
- No et al. (2023). RAPID: Training-free Retrieval-based Log Anomaly Detection with PLM considering Token-level information. arXiv:2311.05160.
- Zhu et al. (2023). Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics. ISSRE.
- Jiang et al. (2024). A Large-scale Evaluation for Log Parsing Techniques: How Far are We? (Loghub-2.0). ISSTA.
- Guan et al. (2025). LogLLM: Log-based Anomaly Detection Using Large Language Models. arXiv:2411.08561.
- Zamanzadeh Darban et al. (2024). Deep Learning for Time Series Anomaly Detection: A Survey. ACM Computing Surveys.
- Li et al. (2025). A survey of deep learning for industrial visual anomaly detection. Artificial Intelligence Review.
- De la Cruz Cabello, Prince Sales & Machado (2025). AIOps for log anomaly detection in the era of LLMs: A systematic literature review. Intelligent Systems with Applications.

**Statistics (bootstrap, Section 4.2 and 4.4).**
- Efron (1979). Bootstrap Methods: Another Look at the Jackknife. The Annals of Statistics 7(1): 1–26. doi:10.1214/aos/1176344552.
- Efron & Tibshirani (1993). An Introduction to the Bootstrap. Chapman & Hall.
- Künsch (1989). The Jackknife and the Bootstrap for General Stationary Observations. The Annals of Statistics 17(3): 1217–1241. doi:10.1214/aos/1176347265.
- Liu & Singh (1992). Moving Blocks Jackknife and Bootstrap Capture Weak Dependence. In LePage & Billard (eds.), Exploring the Limits of Bootstrap, Wiley.
- Politis & Romano (1992). A Circular Block-Resampling Procedure for Stationary Data. In LePage & Billard (eds.), Exploring the Limits of Bootstrap, Wiley.
- Hall, Horowitz & Jing (1995). On Blocking Rules for the Bootstrap with Dependent Data. Biometrika 82(3): 561–574. doi:10.1093/biomet/82.3.561.
- Politis & White (2004). Automatic Block-Length Selection for the Dependent Bootstrap. Econometric Reviews 23(1): 53–70.
- Patton, Politis & White (2009). Correction to "Automatic Block-Length Selection for the Dependent Bootstrap". Econometric Reviews 28(4): 372–375. doi:10.1080/07474930802459016.
- Koehn (2004). Statistical Significance Tests for Machine Translation Evaluation. EMNLP, pp. 388–395 (ACL Anthology W04-3250).

*Verified against the source PDF: Xu 2009, Du 2017 (Fig. 10 on BGL), Zhang 2019, Le & Zhang 2022, Liu 2023, Qi 2023, No 2023, Guan 2025, Han 2023, Zamanzadeh Darban 2024, Li 2025, De la Cruz Cabello 2025, Loghub. RAGLog checked on its arXiv page only. Meng 2019, Le & Zhang 2021, Guo 2021, He 2021 and Jiang 2024 are from memory or secondary sources; verify before submission. Myriad is cited through Li et al. (2025); read the primary source.*

*Statistics references, checked 2026-11-20 by web search (bibliographic records only, no full text read). Confirmed via Project Euclid / publisher / ACL Anthology records: Efron 1979, Künsch 1989, Hall, Horowitz & Jing 1995 (venue, volume, pages, DOI), Politis & White 2004 and Patton, Politis & White 2009, Koehn 2004. To check: (a) Politis & Romano 1992 and Liu & Singh 1992 are attributed to the LePage & Billard volume by secondary sources and memory; the search confirmed the book (Wiley, 1992, ISBN 9780471536314) but not the chapter titles or pages; (b) the n^(1/3) claim for Hall et al. 1995 comes from citing reviews and the R package `blocklength`, not the paper itself; (c) Efron & Tibshirani 1993 was not searched; (d) no log anomaly detection paper using a block bootstrap for F1/recall intervals was found; nearby uses: blockwise bootstrap for ASR (arXiv:1912.09508) and paired bootstrap F1 intervals in time-series anomaly detection (arXiv:2608.24113). Claim "first in log anomaly detection" only after a systematic search.*