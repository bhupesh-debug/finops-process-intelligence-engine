# finops-process-intelligence-engine

**Automated gap analysis, close-task risk scoring, and cash-receipts anomaly detection for a Financial Operations team formalizing an undocumented close process.**

Built as a portfolio project targeting the *Financial Operations Analyst* contract role (Staffing Technologies client, Financial District NYC) — a role explicitly about documenting an undocumented close process, building SOPs and process maps, and performing gap analysis, later expanding into cash receipts.

## Problem

The JD's own framing is the problem statement: *"a growing, largely undocumented environment"* where process knowledge lives in people's heads rather than in SOPs. Gap analysis done by hand means scanning scattered checklists and tribal knowledge and hoping nothing's missed. This project turns that into a repeatable, auditable process:

1. **Documentation gap scoring** — classify free-text process-step descriptions against a 10-category close-control taxonomy, and report exactly which categories are undocumented, partially documented, or fully SOP-quality — across every entity/department that should have one.
2. **Close-task risk scoring** — predict which recurring close tasks are at elevated risk of missing deadline or needing rework, using task metadata including whether a documented SOP exists for that task.
3. **Cash receipts anomaly detection** — flag unusual cash receipt transactions for manual review, built ahead of the role's stated expansion into cash receipts.

## What's Inside

| Module | What it does | Business value |
|---|---|---|
| `src/data_pipeline.py` | Generates synthetic close-control taxonomy coverage data, a 36-month close-task log, and cash receipt transactions | Mirrors a real, incompletely-documented close environment rather than a tidy textbook dataset |
| `src/gap_scorer.py` | TF-IDF similarity baseline + trained classifier for routing text to taxonomy categories, plus the real deliverable: a full coverage report | Turns "we think we're mostly documented" into a measured, category-by-category gap list |
| `src/close_risk_model.py` | RandomForestClassifier predicting missed-deadline/rework risk per close task | Tests whether "write the SOP" actually reduces risk, rather than assuming it |
| `src/cash_anomaly_model.py` | IsolationForest anomaly detection on cash receipts | Flags receipts for review before they become a reconciliation backlog |
| `src/dashboard.py` | Streamlit app — coverage heatmap, task risk, flagged receipts queue | One view for a FinOps analyst to triage gaps and risk daily |
| `tests/test_models.py` | 8 pytest tests against the data generation and coverage logic | CI-enforced correctness, not just a working demo |

## Quick Start

```bash
pip install -r requirements.txt
python src/data_pipeline.py          # generate synthetic taxonomy/task/receipt data
python src/gap_scorer.py             # documentation gap classifier + coverage report
python src/close_risk_model.py       # close-task risk classifier
python src/cash_anomaly_model.py     # cash receipts anomaly detection
pytest tests/ -v                     # 8 tests, data + coverage-logic correctness
streamlit run src/dashboard.py       # interactive coverage/risk/anomaly dashboard
```

## Results

| Component | Metric | Result |
|---|---|---|
| Gap scorer — TF-IDF cosine baseline | Accuracy (all 36 documented rows) | **100%** |
| Gap scorer — trained LogisticRegression | Accuracy (11-row held-out test) | **54.5%** |
| Documentation coverage (overall) | Any documentation vs. fully SOP-quality | **72% have any, only 32% are fully SOP-quality** |
| Close-task risk classifier | Missed/rework recall | **0.83** (precision 0.79), 88% accuracy |
| Cash receipts anomaly detector | Recall / Precision (post-hoc, vs. held-out synthetic label) | **100% / 73.8%**, flagging 7.0% of receipts |

**The baseline beat the trained model on the gap scorer, and that's reported as-is, not hidden.** With only 36 labeled examples spread across 10 categories, a simple cosine-similarity lookup against one canonical description per category outperforms a trained LogisticRegression classifier, which has too little data per class to generalize. This is itself a finding, not a failure: it's a direct, measurable symptom of "largely undocumented environment" — there isn't yet enough written material to train a model on, which is exactly the gap this role exists to close. See `reports/model_report.md` for the full breakdown, including which taxonomy categories have fewer than 2 documented examples.

**The close-task risk model's top feature wasn't SOP existence — it was task complexity**, followed by owner tenure. `sop_exists` ranked 4th of 5 features. That's a more interesting and more honest finding than the hypothesis going in (see `reports/model_report.md`), and it changes the practical recommendation: documentation alone won't fix the riskiest tasks — those are inherently complex, and documentation needs to be paired with simplifying or decomposing the task itself.

## Repo Structure

```
finops-process-intelligence-engine/
├── README.md
├── requirements.txt
├── .github/workflows/test.yml
├── data/synthetic/
│   ├── close_steps_documented.csv
│   ├── close_task_log.csv
│   └── cash_receipts.csv
├── src/
│   ├── data_pipeline.py
│   ├── gap_scorer.py
│   ├── close_risk_model.py
│   ├── cash_anomaly_model.py
│   └── dashboard.py
├── tests/
│   └── test_models.py
├── reports/
│   └── model_report.md
└── slides/
    └── finops_process_intelligence_deck.pptx
```

## Tech Stack

Python 3.11, pandas, scikit-learn (TF-IDF, LogisticRegression, RandomForestClassifier, IsolationForest), Streamlit, pytest, GitHub Actions CI.

## About

Built by Bhupesh Chandra Dimri as a portfolio project for a Financial Operations Analyst role. The goal wasn't a generic ML demo — it was building the actual tool the job description describes doing by hand.
