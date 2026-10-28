"""Checks: privacy, accounting, cohort validity, SQL cross-checks. Run build.py and analysis.py first."""
import json
import re
from pathlib import Path

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT, DATA = ROOT / "outputs", ROOT / "data"


@pytest.fixture(scope="module")
def r():
    return json.loads((OUT / "results.json").read_text())


def test_no_person_names_in_outputs():
    names = set(pd.read_parquet(DATA / "parsed" / "events.parquet", columns=["org:resource"])["org:resource"].dropna())
    names = {n for n in names if len(n) >= 4 and n.isalpha()}
    text = " ".join(f.read_text() for f in OUT.rglob("*") if f.suffix in {".csv", ".json"})
    tokens = set(re.findall(r"[A-Za-z]+", text))
    assert not (names & tokens), sorted(names & tokens)[:10]


def test_person_columns_dropped_before_analysis():
    for f in ("case_metrics.parquet", "spells.parquet"):
        cols = set(pd.read_parquet(DATA / f).columns)
        assert not cols & {"org:resource", "resource country", "org:role"}


def test_reportable_orgs_meet_floors(r):
    contrib = pd.read_csv(DATA / "org_contributors.csv", index_col=0).distinct_contributors
    for o in r["privacy"]["reportable_orgs"]:
        assert contrib[o] >= r["privacy"]["min_contributors"]


def test_published_cells_suppressed():
    for f in OUT.glob("*.csv"):
        t = pd.read_csv(f)
        if "n_cases" in t:
            assert (t.n_cases.dropna() >= 20).all(), f.name


def test_elapsed_time_components_add_up(r):
    assert r["data_quality"]["max_abs_component_sum_error_h"] < 1e-6
    assert abs(sum(x["share_of_elapsed"] for x in r["time_decomposition"]) - 1) < 1e-9


def test_cohort_has_full_follow_up(r):
    c = pd.read_parquet(DATA / "case_metrics.parquet")
    o = pd.to_datetime(c.opened_at, utc=True)
    coh = c[(o >= "2012-05-01") & (o < "2012-05-05") & c.resolved_at.notna()]
    assert len(coh) == r["population"]["analysis_cohort"]
    end = pd.to_datetime(c.last_event_at, utc=True).max()
    assert (end - pd.Timestamp("2012-05-05", tz="UTC")).days >= 7  # 7-day outcome fully observable


def test_sql_transfers_match_pandas():
    sp = pd.read_parquet(DATA / "spells.parquet").sort_values(["case_id", "step_no"])
    pre = sp[sp.pre_resolution]
    chg = pre.grp.ne(pre.groupby("case_id").grp.shift()) & pre.groupby("case_id").cumcount().gt(0)
    pd_transfers = chg.groupby(pre.case_id).sum()
    c = pd.read_parquet(DATA / "case_metrics.parquet").set_index("case_id")
    assert (c.loc[pd_transfers.index, "transfers"] == pd_transfers).all()


def test_ping_pong_definition_on_toy_log():
    ev = pd.DataFrame({
        "case_id": ["a"] * 4 + ["b"] * 3,
        "ts": pd.to_datetime(["2012-05-01 00:00", "2012-05-01 01:00", "2012-05-01 02:00", "2012-05-01 03:00",
                              "2012-05-01 00:00", "2012-05-01 01:00", "2012-05-01 02:00"], utc=True),
        "status": ["Accepted", "Accepted", "Accepted", "Completed", "Accepted", "Accepted", "Completed"],
        "substatus": ["In Progress"] * 3 + ["Resolved", "In Progress", "In Progress", "Resolved"],
        "grp": ["G1", "G2", "G1", "G1", "G1", "G2", "G2"],
        "org": "X", "impact": "Low", "product": "P", "seq": range(7)})
    con = duckdb.connect()
    con.register("ev", ev)
    con.execute((ROOT / "sql" / "case_metrics.sql").read_text())
    m = con.execute("SELECT case_id, transfers, ping_pong, resolution_h FROM case_metrics ORDER BY case_id").df()
    assert m.ping_pong.tolist() == [1, 0] and m.transfers.tolist() == [2, 1]
    assert m.resolution_h.tolist() == [3.0, 2.0]
