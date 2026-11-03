# Research Spec v0.1: Unsupervised Anomaly Detection in Large-Scale System Event Sequences

*Last updated: 2026-11-03. Sections 1–2 drafted; sections 3–12 to be written.*

## 0. Summary

*(Write last: problem, gap, method, evaluation, expected result, in 5 sentences.)*

## 1. Problem and motivation

Automated log anomaly detection faces a trade-off: lightweight models are fast but semantically blind, while LLMs understand log semantics but are too expensive to process the full log stream. This work exploits the complementarity between the two.

### 1.1 Context

Large-scale software systems (cloud platforms, distributed systems, microservices) produce millions to billions of log lines per day. Logs are the main record of how a system actually behaves at runtime. Detecting anomalies in logs early shortens the time to detect and recover from incidents (MTTD, MTTR) and reduces the cost of service disruptions [to cite: He et al., 2021].

### 1.2 Current practice and why it falls short

In practice, log monitoring relies on rules, keywords and alert thresholds, deployed through tools such as the ELK Stack or Splunk. This is insufficient for four reasons:

1. **Volume:** log volume far exceeds what operators can inspect manually.
2. **Anomalies without error keywords:** many incidents show up only as an abnormal order or count of events, while each individual log line looks normal.
3. **New failures:** rules only catch failure types that are already known.
4. **No labels:** incidents are rare, labeling logs requires experts, and new failure types have no labels. Supervised methods are therefore hard to apply in practice, which motivates unsupervised methods.

### 1.3 Existing automated approaches and their tension

**Lightweight models (statistical and deep learning)** such as PCA [Xu et al., 2009], DeepLog [Du et al., 2017] and LogBERT [Guo et al., 2021] handle large volumes at low cost. They have three limitations:

- Logs are represented as template IDs, so the semantics of the log content are lost.
- They are sensitive to **log evolution**: when code updates introduce new templates, performance degrades [Zhang et al., 2019].
- They produce many false positives, causing alert fatigue for operators. DeepLog itself, on BGL without online updates, reaches a precision of only 0.16; it recovers (false positive rate from 40.1% to 1.7%) only when a human labels false positives at runtime [Du et al., 2017, Fig. 10].

**LLMs** understand semantics, handle unseen templates and can explain alerts in natural language. However, their cost and latency make them unsuitable for processing the full log stream. LLMs can also assign wrong labels with confidence; in this work, "hallucination" means an incorrect label decision by the LLM.

### 1.4 Motivation

The two limitations are complementary. Lightweight models suit screening the full log stream; LLMs suit verifying a small fraction of suspicious windows, taking the role the human reviewer plays in DeepLog's online update, without requiring labels. A combined (cascade) pipeline can therefore potentially keep the scalability of lightweight models and the semantic accuracy of LLMs. The open question, not yet answered systematically, is how much benefit this combination brings, at what cost, and whether it is robust to log evolution.

## 2. Research questions and hypotheses

The work answers four questions. The thresholds N, M, P, Q, R in the hypotheses will be fixed after running the baselines (November 2026 milestone), before building the main method.

Notation: ρ is the fraction of windows sent to the LLM, selected as the top-ρ% by the lightweight detector's anomaly score.

### RQ1: Effectiveness of LLM verification

Compared with the lightweight model alone and the LLM alone, how does the cascade pipeline change precision, recall and F1?

- **H1:** Compared with the lightweight model, the cascade increases precision by at least N points while recall drops by no more than M points, resulting in a higher F1.
- **Answered by:** a P/R/F1 table for the lightweight model, LLM-only and the cascade on HDFS and BGL, plus the number of false positives correctly removed and true anomalies wrongly removed by the LLM.

### RQ2: Accuracy–cost trade-off

As ρ varies, how do F1 and cost (tokens and latency per 1,000 windows) change?

- **H2:** There exists ρ ≤ N% such that the cascade reaches at least M% of the F1 of LLM-only, at a cost at least P times lower.
- **Answered by:** F1-versus-cost curves as ρ varies, on HDFS and BGL. Detector-only (ρ = 0%) and LLM-only (ρ = 100%) are the two endpoints of the curve.

### RQ3: Role of retrieved context

Does giving the LLM similar historical windows, retrieved from **unlabeled** training data, make verification more accurate?

- **H3:** At the same ρ, adding retrieved context increases the cascade's F1 by at least Q points over zero-shot LLM verification.
- **Constraint:** retrieval draws only from training logs and uses no anomaly labels, so the whole pipeline remains unsupervised.
- **Answered by:** an ablation table of zero-shot versus retrieval-augmented verification, for several numbers of retrieved windows.

### RQ4: Robustness to log evolution

When the test set contains templates absent from the training set, how much does the F1 of the lightweight model and of the cascade degrade?

- **H4:** The cascade's F1 degradation is at least R points smaller than the lightweight model's.
- **Setup:** chronological split; results reported separately for windows containing new templates. BGL is preferred because its logs change naturally over time.
- **Answered by:** an F1 table on windows with only known templates versus windows with new templates.

### Scope notes

- Hallucination, in the sense of wrong label decisions, is measured in RQ1 (false positives and false negatives introduced by the LLM).
- The quality of the LLM's explanations is outside the quantitative evaluation; it is shown qualitatively in the demo.
- RQ4 is secondary. If time runs short, it is reduced to an analysis on BGL only.

## 3. Problem definition

*(To write: input, output, formal notation, assumptions about training data, sequence grouping.)*

## 4. Evaluation

*(To write before section 6: metrics, baselines, data split, ablations, success criteria.)*

## 5. Data

## 6. Method

*(Include "why A instead of B" for each design choice.)*

## 7. Related work and gap

## 8. Scope

*(In-scope / out-of-scope / non-goals.)*

## 9. Risks and fallbacks

## 10. Plan, milestones and kill criteria

*(See `docs/plan.md` for the implementation stages.)*

## 11. Demo

## 12. Open questions and decision log

*(Decisions are recorded in `notes/decisions.md`.)*

## References

- Xu et al. (2009). Detecting Large-Scale System Problems by Mining Console Logs. SOSP.
- Du et al. (2017). DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning. CCS.
- Zhang et al. (2019). Robust Log-Based Anomaly Detection on Unstable Log Data (LogRobust). ESEC/FSE.
- Guo et al. (2021). LogBERT: Log Anomaly Detection via BERT. IJCNN.
- He et al. (2021). A Survey on Automated Log Analysis for Reliability Engineering. ACM Computing Surveys.
- Zhu et al. (2023). Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics. ISSRE.
- Jiang et al. (2024). A Large-scale Evaluation for Log Parsing Techniques: How Far are We? (Loghub-2.0). ISSTA.

*References except DeepLog and Loghub are from memory; verify each source before submission.*