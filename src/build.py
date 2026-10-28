"""Ingest the real BPI Challenge 2013 incident log, apply the privacy gate, build case metrics.

Usage: python src/parse_xes.py <07_bpic2013_incidents.xes.gz> data/parsed
       python src/build.py
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Columns that describe individual people or narrow to them. Dropped before any analysis.
PERSON_COLUMNS = ["org:resource", "resource country", "org:role"]


def privacy_gate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop person-level columns; keep only distinct-contributor counts per organization."""
    contributors = raw.groupby("organization involved")["org:resource"].nunique().rename("distinct_contributors")
    ev = raw.drop(columns=[c for c in PERSON_COLUMNS if c in raw.columns])
    ev = ev.rename(columns={
        "case:concept:name": "case_id", "time:timestamp": "ts", "concept:name": "status",
        "lifecycle:transition": "substatus", "org:group": "grp", "organization involved": "org",
        "organization country": "org_country",
    })
    ev["seq"] = range(len(ev))  # original file order breaks timestamp ties
    assert not set(PERSON_COLUMNS) & set(ev.columns)
    return ev, contributors


def quality_checks(raw: pd.DataFrame, ev: pd.DataFrame, cases: pd.DataFrame) -> dict:
    g = ev.groupby("case_id")
    last = g.ts.max()
    return {
        "events": int(len(ev)), "cases": int(ev.case_id.nunique()),
        "duplicate_events": int(raw.duplicated().sum()),
        "out_of_order_timestamps": int(g.ts.apply(lambda s: (s.diff().dt.total_seconds() < 0).sum()).sum()),
        "cases_never_resolved": int(cases.resolved_at.isna().sum()),
        "unmatched_status_events": int((ev.status == "Unmatched").sum()),
        "impact_changes_within_case": int((g.impact.nunique() > 1).sum()),
        "product_changes_within_case": int((g["product"].nunique() > 1).sum()),
        "missing_group": int(ev.grp.isna().sum()),
        "earliest_last_event": str(last.min()), "latest_last_event": str(last.max()),
        "earliest_open": str(g.ts.min().min()),
        "share_opened_2012_04_or_later": float((g.ts.min() >= "2012-04-01").mean()),
        "max_abs_component_sum_error_h": float(
            (cases[["queue_h", "active_status_h", "wait_user_h", "wait_other_h", "other_h"]].sum(axis=1)
             - cases.resolution_h).abs().max()),
    }


def main() -> None:
    raw = pd.read_parquet(DATA / "parsed" / "events.parquet")
    ev, contributors = privacy_gate(raw)
    con = duckdb.connect()
    con.register("ev_df", ev)
    con.execute("CREATE TABLE ev AS SELECT * FROM ev_df")
    con.execute((ROOT / "sql" / "case_metrics.sql").read_text())
    cases = con.execute("SELECT * FROM case_metrics").df()
    dq = quality_checks(raw, ev, cases)
    cases.to_parquet(DATA / "case_metrics.parquet", index=False)
    con.execute("SELECT * FROM spells").df().to_parquet(DATA / "spells.parquet", index=False)
    contributors.to_frame().to_csv(DATA / "org_contributors.csv")
    (DATA / "data_quality.json").write_text(json.dumps(dq, indent=2))
    print(json.dumps(dq, indent=1))


if __name__ == "__main__":
    main()
