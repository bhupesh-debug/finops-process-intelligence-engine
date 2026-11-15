"""
cash_anomaly_model.py — unsupervised anomaly detection on cash receipt
transactions, built ahead of the JD's stated expansion into cash receipts
after the initial close-documentation phase.

Deliberately unsupervised (IsolationForest), not a trained classifier: the
dataset carries an `is_synthetic_anomaly_label` column, but that label is
held out and used ONLY for evaluating the unsupervised model after the
fact, never fed into training. A real cash-receipts dataset won't have
fraud/error labels either, so pretending to have them here would overstate
what this approach can actually do on day one.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder


def load_receipts(path="data/synthetic/cash_receipts.csv"):
    return pd.read_csv(path)


def main():
    df = load_receipts()

    df["amount_vs_expected_ratio"] = df["amount"] / df["expected_amount"]
    le = LabelEncoder()
    df["payer_enc"] = le.fit_transform(df["payer"])

    features = ["amount_vs_expected_ratio", "days_to_expected_date", "reference_match_quality"]
    X = df[features]

    # contamination set to the real-world expectation (~7%), not tuned to
    # the held-out label — tuning to the label would be leaking it.
    model = IsolationForest(contamination=0.07, random_state=42)
    df["anomaly_flag"] = (model.fit_predict(X) == -1).astype(int)

    flagged = df[df["anomaly_flag"] == 1]
    print(f"Flagged {len(flagged)} of {len(df)} receipts ({len(flagged)/len(df):.1%}) for manual review")

    # Evaluate against the held-out synthetic label — honest post-hoc check,
    # not part of training.
    true_anomalies = df["is_synthetic_anomaly_label"].sum()
    overlap = ((df["anomaly_flag"] == 1) & (df["is_synthetic_anomaly_label"] == 1)).sum()
    recall = overlap / true_anomalies if true_anomalies else float("nan")
    precision = overlap / len(flagged) if len(flagged) else float("nan")

    print(f"\nPost-hoc check against synthetic ground truth (NOT used in training):")
    print(f"  True synthetic anomalies in dataset: {true_anomalies}")
    print(f"  Recall (of true anomalies, % flagged): {recall:.1%}")
    print(f"  Precision (of flagged, % actually anomalous): {precision:.1%}")

    print("\nTop 5 flagged receipts by lowest reference-match quality:")
    print(flagged.sort_values("reference_match_quality").head(5)[
        ["receipt_id", "payer", "amount", "expected_amount", "days_to_expected_date", "reference_match_quality"]
    ].to_string(index=False))

    return {"flagged_pct": len(flagged) / len(df), "recall": recall, "precision": precision}


if __name__ == "__main__":
    main()
