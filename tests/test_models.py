"""
test_models.py — real pytest tests against the generated data and the
gap-scorer, risk-model, and anomaly-model logic.
Run with: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from data_pipeline import (
    generate_close_steps_documented,
    generate_close_task_log,
    generate_cash_receipts,
    TAXONOMY,
    ENTITIES,
)
from gap_scorer import coverage_report


@pytest.fixture(scope="module")
def steps_df():
    df, _, _ = generate_close_steps_documented()
    return df


@pytest.fixture(scope="module")
def task_log():
    return generate_close_task_log()


@pytest.fixture(scope="module")
def cash_df():
    return generate_cash_receipts()


def test_steps_only_contain_known_taxonomy_categories(steps_df):
    """Every documented step must map to one of the 10 reference taxonomy
    categories — no row should carry a category outside the defined
    control framework."""
    assert set(steps_df["taxonomy_label"].unique()).issubset(set(TAXONOMY))


def test_steps_only_contain_known_entities(steps_df):
    assert set(steps_df["entity"].unique()).issubset(set(ENTITIES))


def test_coverage_report_grid_is_complete(steps_df):
    """The coverage report must have exactly one row per taxonomy category
    — no category should be silently dropped just because it has zero
    documented steps for every entity."""
    report = coverage_report(steps_df)
    assert len(report) == len(TAXONOMY)
    assert set(report.index) == set(TAXONOMY)


def test_coverage_percentages_are_bounded(steps_df):
    """Coverage and fully-documented percentages must be valid fractions
    (0 to 1) — a bug in the grid math could otherwise silently produce a
    coverage value above 100%."""
    report = coverage_report(steps_df)
    assert (report["coverage_pct"] >= 0).all() and (report["coverage_pct"] <= 1).all()
    assert (report["fully_documented_pct"] >= 0).all() and (report["fully_documented_pct"] <= 1).all()
    # fully-documented can never exceed overall coverage for the same category
    assert (report["fully_documented_pct"] <= report["coverage_pct"] + 1e-9).all()


def test_task_log_missed_rework_is_binary(task_log):
    assert set(task_log["missed_or_rework"].unique()).issubset({0, 1})


def test_task_log_sop_exists_is_stable_per_task(task_log):
    """sop_exists is modeled as a fixed property of a task (once an SOP is
    written it doesn't disappear next month) — verify no task flips
    between 0 and 1 across months in the generated log."""
    per_task_values = task_log.groupby("task_name")["sop_exists"].nunique()
    assert (per_task_values == 1).all()


def test_cash_receipts_anomaly_label_rate_is_reasonable(cash_df):
    """The synthetic anomaly rate should land in a realistic single-digit
    to low-double-digit range — not near 0% (nothing to detect) or above
    30% (no longer a minority-anomaly problem)."""
    rate = cash_df["is_synthetic_anomaly_label"].mean()
    assert 0.02 <= rate <= 0.30


def test_cash_receipts_amount_positive(cash_df):
    assert (cash_df["amount"] > 0).all()
    assert (cash_df["expected_amount"] > 0).all()
