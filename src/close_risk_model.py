"""
close_risk_model.py — predicts whether a given month-end close task will
miss its deadline or require rework, using task metadata.

Maps to the JD's "identify gaps and recommend improvements" responsibility,
turned forward-looking: instead of only analyzing documentation gaps after
the fact, this flags which upcoming close tasks are at elevated risk before
the close happens, using the same sop_exists signal as a feature — directly
testing whether "write the SOP" would actually reduce risk, rather than
assuming it.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def load_task_log(path="data/synthetic/close_task_log.csv"):
    return pd.read_csv(path)


def time_based_split(df, train_months=30):
    """Time-based split, not random — this is operational monthly data
    with potential drift, so training on the past and testing on the most
    recent months is the honest evaluation, not k-fold shuffling."""
    train = df[df["month_index"] < train_months]
    test = df[df["month_index"] >= train_months]
    return train, test


def main():
    df = load_task_log()
    train, test = time_based_split(df)

    features = ["owner_tenure_months", "days_to_deadline", "complexity_score",
                "sop_exists", "upstream_dependencies"]
    X_train, y_train = train[features], train["missed_or_rework"]
    X_test, y_test = test[features], test["missed_or_rework"]

    print(f"Train: {len(train)} rows (months 0-29) | Test: {len(test)} rows (months 30-35)")
    print(f"Train missed/rework rate: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("\nClassification report (0=on_time, 1=missed_or_rework):")
    print(classification_report(y_test, preds, target_names=["on_time", "missed_or_rework"], zero_division=0))

    importances = dict(zip(features, clf.feature_importances_))
    print("Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {feat}: {imp:.3f}")

    return {"importances": importances}


if __name__ == "__main__":
    main()
