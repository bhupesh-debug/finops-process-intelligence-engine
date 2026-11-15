"""
dashboard.py — Streamlit app for reviewing documentation coverage, close-task
risk, and flagged cash receipts in one place.

Run with: streamlit run src/dashboard.py
"""
import pandas as pd
import streamlit as st

from gap_scorer import load_steps, coverage_report
from data_pipeline import TAXONOMY

st.set_page_config(page_title="FinOps Process Intelligence", layout="wide")


@st.cache_data
def load_all():
    steps_df = load_steps()
    coverage = coverage_report(steps_df)
    task_log = pd.read_csv("data/synthetic/close_task_log.csv")
    cash = pd.read_csv("data/synthetic/cash_receipts.csv")
    return steps_df, coverage, task_log, cash


def main():
    st.title("FinOps Process Risk & Documentation Gap Intelligence")

    steps_df, coverage, task_log, cash = load_all()

    st.subheader("Documentation Coverage by Control Category")
    st.bar_chart(coverage[["coverage_pct", "fully_documented_pct"]])
    st.dataframe(coverage, use_container_width=True)

    st.subheader("Close-Task Risk — Missed/Rework Rate by Task")
    risk_by_task = task_log.groupby("task_name")["missed_or_rework"].mean().sort_values(ascending=False)
    st.bar_chart(risk_by_task)

    st.subheader("Flagged Cash Receipts (Open Review Queue)")
    st.dataframe(
        cash.sort_values("reference_match_quality").head(20)[
            ["receipt_id", "payer", "amount", "expected_amount", "days_to_expected_date", "reference_match_quality"]
        ],
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
