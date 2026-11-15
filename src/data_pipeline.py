"""
data_pipeline.py — generates three synthetic datasets for the FinOps
Process Risk & Documentation Gap Intelligence project:

1. close_steps_documented.csv — free-text descriptions of close process steps
   as they currently exist (at varying levels of completeness), labeled
   against a reference close-control taxonomy. Backs the NLP gap scorer.
2. close_task_log.csv — a 36-month recurring close-task log with task
   metadata, used to predict missed-deadline/rework risk.
3. cash_receipts.csv — synthetic cash receipt transactions, used for
   unsupervised anomaly detection ahead of the role's stated expansion
   into cash receipts.

All three are synthetic and intentionally built to mirror the JD's own
language ("largely undocumented environment," "month-end close," "cash
receipts") rather than a generic dataset bolted on afterward.
"""
import numpy as np
import pandas as pd

RNG_SEED = 7

# Reference close-control taxonomy — grounded in standard month-end close
# control categories (not proprietary to any single company).
TAXONOMY = [
    "Revenue Cutoff",
    "Accrual Review",
    "Prepaid Amortization",
    "Bank Reconciliation",
    "Intercompany Elimination",
    "Fixed Asset Roll-forward",
    "Journal Entry Approval",
    "Account Reconciliation Sign-off",
    "Flux Variance Analysis",
    "Cash Receipts Matching",
]

# Phrase banks per taxonomy category, at varying "documentation completeness."
# full = a real SOP-style description; partial = an informal one-liner;
# undocumented = the category has no written step at all (handled below).
PHRASE_BANK = {
    "Revenue Cutoff": {
        "full": "Review all revenue transactions booked in the final 3 business days of the period, confirm shipping/delivery terms support the recognition date, and document any cutoff adjustments with supporting evidence before journal entry posting.",
        "partial": "Check revenue near period end for cutoff issues.",
    },
    "Accrual Review": {
        "full": "Compile outstanding accrual requests from department owners by the 2nd business day, validate each against open PO and invoice data, and post accrual journal entries with a documented basis for each estimate.",
        "partial": "Collect accrual estimates from departments and book them.",
    },
    "Prepaid Amortization": {
        "full": "Run the prepaid amortization schedule, reconcile the current-month amortization expense to the schedule total, and investigate any variance greater than $5,000 before posting.",
        "partial": "Amortize prepaids per the schedule.",
    },
    "Bank Reconciliation": {
        "full": "Download the bank statement, match each transaction to the GL cash account line by line, document all reconciling items with explanations, and obtain sign-off from the controller before closing the period.",
        "partial": "Review bank statement, match to GL, note differences.",
    },
    "Intercompany Elimination": {
        "full": "Pull intercompany balances from each entity, confirm they net to zero before elimination, document any out-of-balance amounts with a resolution plan, and post the elimination entry with supporting schedule attached.",
        "partial": "Eliminate intercompany balances.",
    },
    "Fixed Asset Roll-forward": {
        "full": "Update the fixed asset roll-forward for additions, disposals, and depreciation for the period, reconcile the ending balance to the GL, and document the basis for any manual adjustments.",
        "partial": "Update fixed asset schedule.",
    },
    "Journal Entry Approval": {
        "full": "Route all manual journal entries above the materiality threshold through the documented two-person approval workflow, retain supporting documentation for each entry, and log approver sign-off before posting.",
        "partial": "Get journal entries approved before posting.",
    },
    "Account Reconciliation Sign-off": {
        "full": "Complete account reconciliations for all balance sheet accounts per the close calendar, attach supporting schedules, and route each reconciliation for manager sign-off before the account is considered closed.",
        "partial": "Reconcile balance sheet accounts.",
    },
    "Flux Variance Analysis": {
        "full": "Generate the month-over-month and budget-to-actual variance report, investigate any line item variance exceeding 10% or $25,000, and document the driver and business explanation for each flagged variance.",
        "partial": "Review variances vs. prior month.",
    },
    "Cash Receipts Matching": {
        "full": "Match each incoming cash receipt to its corresponding invoice or expected payment using reference number and amount, flag unmatched or partial-amount receipts for manual review, and clear the matched items from the open-items report.",
        "partial": "Match cash receipts to invoices where possible.",
    },
}

CASH_RECEIPT_PAYERS = [f"Customer-{i:03d}" for i in range(1, 41)]

# Each taxonomy category is checked separately across these sub-entities
# (departments/subsidiaries), mirroring how a real close actually repeats
# the same control across multiple books. This gives the gap scorer enough
# labeled text to train on, and gives the coverage heatmap a real
# denominator (10 categories x 5 entities = 50 expected steps) instead of
# one yes/no per category.
ENTITIES = ["Corporate", "Subsidiary-A", "Subsidiary-B", "Marketing BU", "Field Ops BU"]


def _paraphrase(text, entity):
    """Light templated variation so the same category doesn't produce
    byte-identical rows across entities — still synthetic, but enough
    lexical variety that TF-IDF has to learn the category's real
    vocabulary instead of memorizing one fixed string."""
    return f"[{entity}] {text}"


def generate_close_steps_documented(seed=RNG_SEED):
    """For each (taxonomy category x entity) pair, independently sample a
    documentation state: full (real SOP-style text), partial (a vague
    one-liner — still a gap risk), or undocumented (no step exists at all
    — these pairs are OMITTED from the output, which is the point: an
    undocumented step leaves no trace, and gap detection has to come from
    comparing what's present against the full taxonomy x entity grid, not
    from a labeled row saying 'this is missing')."""
    rng = np.random.default_rng(seed)
    rows = []
    step_id = 1
    expected_total = len(TAXONOMY) * len(ENTITIES)
    for category in TAXONOMY:
        for entity in ENTITIES:
            # ~30% full, ~40% partial, ~30% undocumented — mirrors a
            # "largely undocumented environment" rather than a tidy one.
            state = rng.choice(["full", "partial", "undocumented"], p=[0.3, 0.4, 0.3])
            if state == "undocumented":
                continue
            text = _paraphrase(PHRASE_BANK[category][state], entity)
            rows.append({
                "step_id": f"STEP-{step_id:03d}",
                "entity": entity,
                "raw_text": text,
                "documented_completeness": state,
                "taxonomy_label": category,
            })
            step_id += 1
    df = pd.DataFrame(rows)
    coverage = (
        df.groupby("taxonomy_label").size().reindex(TAXONOMY, fill_value=0) / len(ENTITIES)
    )
    return df, coverage, expected_total


def generate_close_task_log(seed=RNG_SEED, n_months=36):
    rng = np.random.default_rng(seed + 1)
    task_names = [f"{cat} — Monthly" for cat in TAXONOMY]
    # Each task gets a fixed sop_exists flag for realism (an SOP, once
    # written, tends to stay written), but complexity/tenure/dependencies
    # vary month to month.
    sop_status = {t: rng.choice([0, 1], p=[0.55, 0.45]) for t in task_names}
    base_complexity = {t: rng.integers(1, 6) for t in task_names}

    rows = []
    for month_idx in range(n_months):
        for task in task_names:
            owner_tenure = max(3, int(rng.normal(14, 9)))
            complexity = int(np.clip(base_complexity[task] + rng.integers(-1, 2), 1, 5))
            days_to_deadline = int(rng.integers(1, 6))
            sop_exists = sop_status[task]
            upstream_deps = int(rng.integers(0, 5))

            # Risk model: no SOP + high complexity + low tenure + many deps
            # + tight deadline genuinely raises miss/rework probability.
            # Weighted and thresholded to land near a realistic ~20-25%
            # base rate rather than a coin flip, so the classifier has a
            # real minority class to find instead of a near-even split.
            risk_score = (
                (1 - sop_exists) * 0.30
                + (complexity / 5) * 0.18
                + (3 / owner_tenure) * 0.14
                + (upstream_deps / 4) * 0.12
                + (1 / days_to_deadline) * 0.08
            )
            risk_score += rng.normal(0, 0.07)
            missed_or_rework = int(risk_score > 0.62)

            rows.append({
                "month_index": month_idx,
                "task_name": task,
                "owner_tenure_months": owner_tenure,
                "days_to_deadline": days_to_deadline,
                "complexity_score": complexity,
                "sop_exists": sop_exists,
                "upstream_dependencies": upstream_deps,
                "missed_or_rework": missed_or_rework,
            })
    return pd.DataFrame(rows)


def generate_cash_receipts(seed=RNG_SEED, n=600):
    rng = np.random.default_rng(seed + 2)
    rows = []
    for i in range(n):
        payer = rng.choice(CASH_RECEIPT_PAYERS)
        expected_amount = float(rng.uniform(500, 45_000))
        # Most receipts match cleanly; a minority are anomalous
        is_anomalous = rng.random() < 0.07
        if is_anomalous:
            amount = expected_amount * rng.uniform(0.3, 0.7) if rng.random() < 0.5 else expected_amount * rng.uniform(1.3, 2.2)
            days_to_expected = int(rng.integers(15, 60))
            ref_match_quality = float(rng.uniform(0, 0.4))
        else:
            amount = expected_amount * rng.uniform(0.98, 1.0)
            days_to_expected = int(rng.integers(0, 5))
            ref_match_quality = float(rng.uniform(0.85, 1.0))

        rows.append({
            "receipt_id": f"CR-{i+1:04d}",
            "payer": payer,
            "amount": round(amount, 2),
            "expected_amount": round(expected_amount, 2),
            "days_to_expected_date": days_to_expected,
            "reference_match_quality": round(ref_match_quality, 3),
            "is_synthetic_anomaly_label": int(is_anomalous),  # held out for eval only, not used in training
        })
    return pd.DataFrame(rows)


def main():
    steps_df, coverage, expected_total = generate_close_steps_documented()
    task_log_df = generate_close_task_log()
    cash_df = generate_cash_receipts()

    steps_df.to_csv("data/synthetic/close_steps_documented.csv", index=False)
    task_log_df.to_csv("data/synthetic/close_task_log.csv", index=False)
    cash_df.to_csv("data/synthetic/cash_receipts.csv", index=False)

    print(f"close_steps_documented.csv: {len(steps_df)} documented steps out of {expected_total} expected (category x entity)")
    print("Per-category coverage (documented fraction of 5 entities):")
    print(coverage.round(2).to_string())
    print(f"\nclose_task_log.csv: {len(task_log_df)} rows, {task_log_df['missed_or_rework'].mean():.1%} missed/rework rate")
    print(f"cash_receipts.csv: {len(cash_df)} rows, {cash_df['is_synthetic_anomaly_label'].mean():.1%} synthetic anomaly rate")


if __name__ == "__main__":
    main()
