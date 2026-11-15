"""
gap_scorer.py — the centerpiece model of this project.

Two connected pieces:

1. A TF-IDF + LogisticRegression multi-class classifier that reads a free-text
   process-step description and predicts which close-control taxonomy
   category it belongs to, compared against a pure TF-IDF cosine-similarity
   baseline. This proves a written step CAN be automatically routed to its
   control category, which is the mechanical first half of "gap analysis."
2. A coverage report that compares what's actually documented (by category
   x entity) against the full expected taxonomy grid, and flags every gap —
   both "undocumented" (missing entirely) and "partial" (documented but
   below SOP-quality completeness). This is the actual gap-analysis output
   a FinOps analyst would use, and it does NOT require the classifier to be
   perfect — it's a direct, interpretable comparison against the taxonomy.

The classifier is the "can this be automated" proof; the coverage report is
the real deliverable.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from data_pipeline import TAXONOMY, ENTITIES, PHRASE_BANK


def load_steps(path="data/synthetic/close_steps_documented.csv"):
    return pd.read_csv(path)


def baseline_cosine_similarity(df):
    """Baseline: TF-IDF vectorize both the documented steps and a single
    canonical 'full' description per taxonomy category, then assign each
    step to its nearest category by cosine similarity. No training —
    purely a similarity lookup, which is the honest floor to beat."""
    reference_texts = [PHRASE_BANK[cat]["full"] for cat in TAXONOMY]
    vectorizer = TfidfVectorizer(stop_words="english")
    all_text = list(df["raw_text"]) + reference_texts
    tfidf = vectorizer.fit_transform(all_text)
    step_vectors = tfidf[: len(df)]
    ref_vectors = tfidf[len(df):]

    sims = cosine_similarity(step_vectors, ref_vectors)
    preds = [TAXONOMY[i] for i in sims.argmax(axis=1)]
    acc = accuracy_score(df["taxonomy_label"], preds)
    return acc, preds


def trained_classifier(df):
    """LogisticRegression on TF-IDF features, trained on the documented
    steps themselves. With only 36 rows across 10 classes this is a small
    dataset — reported honestly, including which classes have too few
    examples to evaluate meaningfully, rather than hiding that behind an
    aggregate accuracy number."""
    X_text = df["raw_text"]
    y = df["taxonomy_label"]

    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()

    vectorizer = TfidfVectorizer(stop_words="english", max_features=200)
    X = vectorizer.fit_transform(X_text)

    # Can't stratify on classes with <2 members; fall back to a plain
    # random split and report the limitation rather than crash or fudge it.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc, rare_classes, len(y_test)


def coverage_report(df):
    """The real deliverable: for every (taxonomy category x entity) pair,
    report documented / partial / undocumented, and roll up to a
    per-category coverage percentage for the heatmap."""
    full_grid = pd.MultiIndex.from_product([TAXONOMY, ENTITIES], names=["taxonomy_label", "entity"]).to_frame(index=False)
    merged = full_grid.merge(df[["taxonomy_label", "entity", "documented_completeness"]],
                              on=["taxonomy_label", "entity"], how="left")
    merged["documented_completeness"] = merged["documented_completeness"].fillna("undocumented")

    pivot = merged.pivot_table(index="taxonomy_label", columns="documented_completeness",
                                aggfunc="size", fill_value=0).reindex(TAXONOMY)
    for col in ["full", "partial", "undocumented"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["coverage_pct"] = (pivot["full"] + pivot["partial"]) / len(ENTITIES)
    pivot["fully_documented_pct"] = pivot["full"] / len(ENTITIES)
    return pivot[["full", "partial", "undocumented", "coverage_pct", "fully_documented_pct"]]


def main():
    df = load_steps()

    baseline_acc, _ = baseline_cosine_similarity(df)
    clf_acc, rare_classes, n_test = trained_classifier(df)

    print(f"Baseline (TF-IDF cosine similarity to canonical category text): {baseline_acc:.1%} accuracy on all {len(df)} rows")
    print(f"Trained classifier (TF-IDF + LogisticRegression), {n_test}-row held-out test: {clf_acc:.1%} accuracy")
    if rare_classes:
        print(f"  NOTE: {len(rare_classes)} categories have <2 documented examples ({rare_classes}) — "
              f"too few to train/evaluate reliably on their own; this is a direct, honest symptom "
              f"of the 'largely undocumented environment' the dataset was built to represent.")

    print("\nCoverage report (by taxonomy category, across 5 entities):")
    report = coverage_report(df)
    print(report.to_string())

    overall_coverage = report["coverage_pct"].mean()
    overall_full = report["fully_documented_pct"].mean()
    print(f"\nOverall: {overall_coverage:.1%} of expected steps have ANY documentation, "
          f"only {overall_full:.1%} are fully SOP-quality documented.")

    return {"baseline_acc": baseline_acc, "clf_acc": clf_acc, "coverage": report}


if __name__ == "__main__":
    main()
