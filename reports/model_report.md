# Model Report — FinOps Process Intelligence Engine

All figures below are from an actual run of the code in this repo against the synthetic datasets in `data/synthetic/`, regenerated and re-verified before this report was written. Nothing here is a target range or a placeholder.

---

## 1. Documentation Gap Scorer

### 1a. Classifier comparison

| Approach | Accuracy | Notes |
|---|---|---|
| TF-IDF cosine similarity to one canonical "full" description per category | **100.0%** (36/36 rows) | No training — a pure similarity lookup |
| TF-IDF + LogisticRegression, trained/tested with a random split | **54.5%** (6/11 held-out rows correct) | Trained model, genuinely worse |

**The baseline wins, and that's the real finding.** With only 36 documented steps spread across 10 taxonomy categories (because this dataset deliberately mirrors "largely undocumented" — most category x entity combinations have zero or one example), several categories have fewer than 2 documented examples:

- Categories with fewer than 2 rows cannot be meaningfully split into train/test, so the LogisticRegression classifier is effectively trying to learn 10-way text classification from roughly 3-4 examples per class on average. A simple cosine-similarity match against one well-written reference description per category sidesteps that entirely — it doesn't need many examples, it needs one good one.
- This is not a reason to prefer the baseline forever. It's a reason to lead with the baseline *now*, and note explicitly that the trained classifier would be expected to overtake it once real documentation volume grows — which is itself the measurable argument for why this project (and this role) matters: more real SOPs written → more training data → a genuinely learnable classification problem, not just a lookup table.

### 1b. Coverage report (the actual deliverable)

| Category | Full | Partial | Undocumented | Coverage % | Fully Documented % |
|---|---|---|---|---|---|
| Revenue Cutoff | 1 | 2 | 2 | 60% | 20% |
| Accrual Review | 1 | 1 | 3 | 40% | 20% |
| Prepaid Amortization | 2 | 3 | 0 | 100% | 40% |
| Bank Reconciliation | 0 | 2 | 3 | 40% | 0% |
| Intercompany Elimination | 4 | 1 | 0 | 100% | 80% |
| Fixed Asset Roll-forward | 0 | 4 | 1 | 80% | 0% |
| Journal Entry Approval | 3 | 2 | 0 | 100% | 60% |
| Account Reconciliation Sign-off | 3 | 1 | 1 | 80% | 60% |
| Flux Variance Analysis | 1 | 2 | 2 | 60% | 20% |
| Cash Receipts Matching | 1 | 2 | 2 | 60% | 20% |

**Overall: 72.0% of expected steps have *some* documentation; only 32.0% are fully SOP-quality.** The gap between those two numbers is the real story — a coverage dashboard that only reported "72% documented" would understate the risk. Bank Reconciliation and Fixed Asset Roll-forward are the two highest-risk categories: Bank Reconciliation is 0% fully documented across all 5 entities despite having some partial coverage, and Fixed Asset Roll-forward is in the same position. Both are exactly the kind of category where "partial, informal documentation" creates false confidence — someone believes it's handled because *a* description exists, when it isn't actually SOP-quality.

**Practical recommendation**: prioritize Bank Reconciliation and Fixed Asset Roll-forward for full SOP write-ups first — not because they're undocumented (several categories are worse on that axis), but because they're the ones where partial documentation is most likely to be mistaken for complete documentation.

---

## 2. Close-Task Risk Classifier

Trained on months 0-29 (300 rows), tested on months 30-35 (60 rows), time-based split (not random — this is operational monthly data).

- Train missed/rework rate: 25.7%
- Test missed/rework rate: 30.0%
- Test accuracy: **88%**
- Missed/rework class: precision **0.79**, recall **0.83**, f1 **0.81**
- On-time class: precision 0.93, recall 0.90, f1 0.92

### Feature importances

| Feature | Importance |
|---|---|
| complexity_score | 0.325 |
| owner_tenure_months | 0.287 |
| upstream_dependencies | 0.147 |
| sop_exists | 0.127 |
| days_to_deadline | 0.114 |

**This did not confirm the project's original hypothesis, and that's reported honestly rather than adjusted after the fact.** Going in, the expectation was that `sop_exists` would dominate — "no SOP" as the primary driver of missed deadlines and rework. The actual trained model ranks it 4th of 5 features. Task complexity and owner tenure matter more.

**What this actually means practically**: documentation alone is necessary but not sufficient. A complex task performed by a short-tenured owner is high-risk whether or not an SOP exists, because an SOP reduces ambiguity but doesn't reduce inherent task difficulty. The more complete recommendation, supported by this data rather than assumed: pair SOP-writing with task simplification/decomposition for the highest-complexity tasks, and prioritize SOP coverage specifically for tasks currently staffed by newer owners, where the tenure-complexity combination is riskiest. This is a better, more specific recommendation than "write more SOPs" — and it's one you can only make credibly after actually running the model and looking at what it found, not what you expected it to find.

---

## 3. Cash Receipts Anomaly Detector

Unsupervised IsolationForest, contamination parameter set to 0.07 (the real-world expectation), **not tuned against the held-out label** — tuning contamination to match the known anomaly rate would be leaking the label into model selection, which defeats the point of testing an unsupervised approach honestly.

- Flagged: 42 of 600 receipts (7.0%)
- Post-hoc check against the held-out synthetic label (31 true anomalies, never used in training):
  - **Recall: 100%** — every synthetic anomaly was flagged
  - **Precision: 73.8%** — of the 42 flagged, 31 were true anomalies, 11 were false positives

**This is a believable, not-too-good-to-be-true result given how the synthetic anomalies were constructed** (amount mismatches of 30-70% off expected, combined with 15-60 day payment delays and low reference-match quality — a genuinely distinct cluster from normal receipts, which cluster tightly around 98-100% of expected amount, 0-5 day delay, and high match quality). A real cash-receipts dataset would likely show messier, more continuous anomaly patterns and a correspondingly lower precision/recall; this result should be read as "the approach works when anomalies are genuinely distinguishable," not as a production-ready precision number.

---

## Future Improvements

1. Replace the synthetic taxonomy-coverage dataset with a real inventory of the client's actual close checklist once on the job — the gap scorer's trained classifier should improve materially as real documentation volume grows past the current 36-row ceiling.
2. Extend the close-task risk model with a true task-complexity decomposition (number of sub-steps, number of systems touched) rather than a single 1-5 proxy score.
3. Build a feedback loop where an analyst confirms or rejects each flagged documentation gap and anomaly, and retrain periodically on confirmed labels rather than purely synthetic ones.
4. Extend the taxonomy itself to cover cash receipts and AP sub-categories once the role's scope expands past month-end close, per the JD's stated phase 2.
