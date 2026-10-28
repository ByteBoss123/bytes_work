"""Responsible workflow analytics on the real BPI Challenge 2013 (Volvo IT) incident log.

Decision question: which process change would most reduce how long customers wait for
incident resolution: faster assignment out of queues, fewer team-to-team transfers,
or tighter rules for the "Wait - User" status?

Every published number is aggregate. Person-level columns are dropped in build.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))

MIN_CONTRIBUTORS = 5
MIN_CELL_N = 20
SLOW_H = 24             # primary outcome: "not resolved within 24 hours"
SLOW_7D_H = 168         # secondary outcome
LANDMARK_H = 4          # exposure window for the landmark analysis
# The extract contains incidents opened up to 2012-05-06 whose last event falls on/after 2012-04-30.
# Incidents opened in April are therefore included mainly when they ran long (see selection_diagnostic).
# Incidents opened 1-4 May are included regardless of duration and followed to 22 May (>= 18 days).
COHORT_START, COHORT_END = "2012-05-01", "2012-05-05"
CONTROLS = "C(impact) + C(product_g) + C(org_g) + C(open_day)"


# ------------------------------------------------------------------ prep
def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    c = pd.read_parquet(DATA / "case_metrics.parquet")
    sp = pd.read_parquet(DATA / "spells.parquet")
    contrib = pd.read_csv(DATA / "org_contributors.csv", index_col=0).distinct_contributors
    return c, sp, contrib


def prepare(c: pd.DataFrame, contrib: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    c = c[c.resolved_at.notna()].copy()
    c["opened_at"] = pd.to_datetime(c.opened_at, utc=True)
    c = c[(c.opened_at >= pd.Timestamp(COHORT_START, tz="UTC")) & (c.opened_at < pd.Timestamp(COHORT_END, tz="UTC"))].copy()
    c["slow"] = (c.resolution_h > SLOW_H).astype(int)
    c["slow_7d"] = (c.resolution_h > SLOW_7D_H).astype(int)
    c["log_res"] = np.log(c.resolution_h + 1)
    c["excl_wait_h"] = c.resolution_h - c.wait_user_h - c.wait_other_h
    c["open_day"] = c.opened_at.dt.strftime("%Y-%m-%d")
    pc = c["product"].value_counts()
    c["product_g"] = np.where(c["product"].map(pc) >= 30, c["product"], "other_products")
    oc = c.owning_org.value_counts()
    reportable = sorted(o for o in oc.index if oc[o] >= MIN_CELL_N and contrib.get(o, 0) >= MIN_CONTRIBUTORS)
    c["org_g"] = np.where(c.owning_org.isin(reportable), c.owning_org, "smaller orgs (pooled)")
    c["transfers_bin"] = pd.cut(c.transfers, [-1, 0, 1, 2, 1e9], labels=["0", "1", "2", "3+"]).astype(str)
    return c, reportable


def rr_poisson(formula: str, d: pd.DataFrame, term: str, cluster: str = "product_g") -> dict:
    """Modified Poisson (robust, product-clustered SE) -> risk ratio for a 0/1 outcome."""
    m = smf.glm(formula, data=d, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    b, (lo, hi) = m.params[term], m.conf_int().loc[term]
    return dict(rr=float(np.exp(b)), ci_low=float(np.exp(lo)), ci_high=float(np.exp(hi)),
                p_value=float(m.pvalues[term]), n=int(m.nobs))


def e_value(rr: float, lo: float) -> dict:
    """VanderWeele & Ding E-value for a risk ratio > 1 and for its CI lower bound."""
    f = lambda r: r + np.sqrt(r * (r - 1)) if r > 1 else 1.0
    return {"point": float(f(rr)), "ci_bound": float(f(lo))}  # 1.0 when the CI includes the null


# ---------------------------------------------------- 1. where the time goes
def decomposition(c: pd.DataFrame) -> pd.DataFrame:
    parts = {"queue_h": "waiting in a queue for assignment", "wait_user_h": "status 'Wait - User'",
             "wait_other_h": "other wait statuses (vendor, implementation, ...)",
             "active_status_h": "in an active status (In Progress / Assigned)"}
    tot = c.resolution_h.sum()
    return pd.DataFrame([dict(component=v, share_of_elapsed=c[k].sum() / tot,
                              share_of_cases_with_any=(c[k] > 0).mean(), median_h_if_any=c.loc[c[k] > 0, k].median())
                         for k, v in parts.items()])


# --------------------------------------------------- 2. transfers and delay
def transfers_analysis(c: pd.DataFrame) -> dict:
    raw = rr_poisson("slow ~ ping_pong", c, "ping_pong")
    adj = rr_poisson(f"slow ~ ping_pong + {CONTROLS}", c, "ping_pong")

    # Landmark: exposure measured in the first 24h, among incidents still open at 24h.
    # Removes the competing explanation that long incidents simply have more time to collect transfers.
    sp = pd.read_parquet(DATA / "spells.parquet", columns=["case_id", "ts", "grp", "is_transfer", "step_no"])
    sp = sp.merge(c[["case_id", "opened_at"]], on="case_id")
    sp["ts"] = pd.to_datetime(sp.ts, utc=True)
    early = sp[sp.ts <= sp.opened_at + pd.Timedelta(hours=LANDMARK_H)]
    visits = early[early.is_transfer | (early.step_no == 1)].groupby(["case_id", "grp"]).size()
    early_pp = (visits > 1).groupby("case_id").any().astype(int).rename("early_ping_pong")
    early_tr = early.groupby("case_id").is_transfer.sum().rename("early_transfers")
    lm = c[c.resolution_h > LANDMARK_H].join(early_pp, on="case_id").join(early_tr, on="case_id")
    lm[["early_ping_pong", "early_transfers"]] = lm[["early_ping_pong", "early_transfers"]].fillna(0)
    lm["early_2plus"] = (lm.early_transfers >= 2).astype(int)
    lm_adj = rr_poisson(f"slow ~ early_ping_pong + {CONTROLS}", lm, "early_ping_pong")
    lm_raw = rr_poisson("slow ~ early_ping_pong", lm, "early_ping_pong")

    # Replication in two halves split at the median opening time
    med = lm.opened_at.median()
    halves = {name: rr_poisson(f"slow ~ early_ping_pong + C(impact) + C(product_g) + C(org_g)", d, "early_ping_pong")
              for name, d in [("opened_first_half", lm[lm.opened_at <= med]),
                              ("opened_second_half", lm[lm.opened_at > med])]}
    adj_7d = rr_poisson(f"slow_7d ~ ping_pong + {CONTROLS}", c, "ping_pong")

    # Mechanism check: how much queue time does each transfer bring?
    q = smf.ols(f"queue_h ~ transfers + {CONTROLS}", data=c).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(c.product_g)[0]})
    rates = (c.groupby("transfers_bin").agg(n_cases=("case_id", "size"), slow_rate=("slow", "mean"),
                                             median_resolution_h=("resolution_h", "median"),
                                             median_queue_h=("queue_h", "median")).reset_index())
    return {
        "ping_pong_share": float(c.ping_pong.mean()),
        "rr_ping_pong_raw": raw, "rr_ping_pong_adjusted": adj, "rr_ping_pong_adjusted_7d_outcome": adj_7d,
        "landmark": {"population": f"incidents still open at {LANDMARK_H}h", "n": int(len(lm)),
                     "early_ping_pong_share": float(lm.early_ping_pong.mean()),
                     "slow_rate_early_pp": float(lm.loc[lm.early_ping_pong == 1, "slow"].mean()),
                     "slow_rate_no_early_pp": float(lm.loc[lm.early_ping_pong == 0, "slow"].mean()),
                     "rr_raw": lm_raw, "rr_adjusted": lm_adj,
                     "e_value": e_value(lm_adj["rr"], lm_adj["ci_low"]), "replication_halves": halves},
        "queue_h_per_transfer": {"estimate": float(q.params["transfers"]),
                                 "ci": [float(x) for x in q.conf_int().loc["transfers"]]},
        "by_transfers": rates,
        "_landmark_frame": lm,
    }


# ------------------------------------------ 3. "Wait - User": measurement validity
def wait_user_analysis(c: pd.DataFrame, sp: pd.DataFrame, reportable: list[str]) -> dict:
    fam = sm.families.Binomial()
    m0 = smf.glm("used_wait_user ~ C(impact) + C(product_g) + C(open_day)", data=c, family=fam).fit()
    m1 = smf.glm(f"used_wait_user ~ {CONTROLS}", data=c, family=fam).fit()
    lr = 2 * (m1.llf - m0.llf)
    df = m1.df_model - m0.df_model  # org terms
    # adjusted (marginal) usage by org: predict every incident as if owned by that org
    adj = {o: float(m1.predict(c.assign(org_g=o)).mean()) for o in reportable}

    rows = []
    for o in reportable:
        d = c[c.org_g == o]
        rows.append(dict(org=o, n_cases=len(d), used_wait_user_raw=d.used_wait_user.mean(),
                         used_wait_user_adjusted=adj[o],
                         median_elapsed_h=d.resolution_h.median(), median_excl_wait_h=d.excl_wait_h.median(),
                         slow_rate_elapsed=d.slow.mean(), slow_rate_excl_wait=(d.excl_wait_h > SLOW_H).mean()))
    t = pd.DataFrame(rows)
    rho = stats.spearmanr(t.slow_rate_elapsed, t.slow_rate_excl_wait)

    # what happens right after a Wait - User spell (pre-resolution)
    sp = sp.sort_values(["case_id", "step_no"])
    sp["next_status"] = sp.groupby("case_id").status.shift(-1)
    sp["next_sub"] = sp.groupby("case_id").substatus.shift(-1)
    w = sp[(sp.spell_type == "wait_user") & sp.pre_resolution & sp.case_id.isin(c.case_id)]
    nxt = np.where(w.next_status == "Completed", "resolved next",
                   np.where(w.next_sub.isin(["In Progress", "Assigned"]), "back to active work",
                            np.where(w.next_status == "Queued", "re-queued", "another wait / other")))
    after = pd.Series(nxt).value_counts(normalize=True).to_dict()
    return {"lr_test_org_terms": {"chi2": float(lr), "df": int(df), "p_value": float(stats.chi2.sf(lr, df))},
            "overall_usage": float(c.used_wait_user.mean()),
            "share_of_elapsed_excluded_by_wait_clock": float((c.wait_user_h + c.wait_other_h).sum() / c.resolution_h.sum()),
            "org_table": t, "spearman_slow_rate_elapsed_vs_excl_wait": {"rho": float(rho.statistic),
                                                                         "p_value": float(rho.pvalue), "n_orgs": len(t)},
            "after_wait_user_spell": after, "n_wait_user_spells": int(len(w)),
            "median_wait_user_spell_h": float(w.dur_h.median())}


# ----------------------------------------------------------- 4. rework (reopen)
def reopen_analysis(c: pd.DataFrame) -> dict:
    c = c.copy()
    c["res_bin"] = pd.cut(c.resolution_h, [-0.01, 1, 24, 168, 1e9],
                          labels=["<1h", "1-24h", "1-7d", ">7d"]).astype(str)
    tab = c.groupby("res_bin").agg(n_cases=("case_id", "size"), reopen_rate=("reopened", "mean")).reindex(
        ["<1h", "1-24h", "1-7d", ">7d"]).reset_index()
    c["fast_close"] = (c.resolution_h < 1).astype(int)
    adj = rr_poisson(f"reopened ~ fast_close + {CONTROLS}", c, "fast_close")
    return {"overall_reopen_rate": float(c.reopened.mean()), "by_resolution_time": tab, "rr_fast_close_adjusted": adj}


# ------------------------------------------------- 5. pilot: what could we detect?
def pilot_mde(c: pd.DataFrame) -> dict:
    """Smallest reduction in the 24h-slow rate among incidents that enter 'Wait - User' that a
    case-randomized pilot could detect (alpha 0.05, power 0.8). No effect size is assumed."""
    p0 = float(c.loc[c.used_wait_user == 1, "slow"].mean())
    per_day = float(c.loc[c.used_wait_user == 1].groupby("open_day").size().mean())
    out = {"baseline_slow_rate_wait_user_cases": p0, "wait_user_incidents_per_weekday_in_cohort": per_day, "designs": []}
    for n in (500, 1000, 2000, 4000):
        es = NormalIndPower().solve_power(nobs1=n, alpha=0.05, power=0.8, ratio=1.0)
        # invert Cohen's h for a reduction from p0
        p1 = np.sin(np.arcsin(np.sqrt(p0)) - es / 2) ** 2
        out["designs"].append(dict(incidents_per_arm=n, detectable_p_treatment=float(p1),
                                   detectable_reduction_pp=float((p0 - p1) * 100),
                                   weekdays_of_volume_needed=float(2 * n / per_day)))
    return out


def selection_diagnostic(c0: pd.DataFrame) -> pd.DataFrame:
    o = pd.to_datetime(c0.opened_at, utc=True)
    band = np.select([o < "2012-04-01", o < "2012-05-01", o < "2012-05-05"],
                     ["opened before April", "opened 1-30 April", "opened 1-4 May (analysis cohort)"], "opened 5-6 May")
    d = c0.assign(band=band, slow_7d=(c0.resolution_h > SLOW_7D_H))
    return (d.groupby("band").agg(n_cases=("case_id", "size"), median_resolution_h=("resolution_h", "median"),
                                  share_not_resolved_7d=("slow_7d", "mean"), reopen_rate=("reopened", "mean"))
            .reset_index())



# ---------------------------------------- 6. assignment queues: when and how full
BUSINESS_UTC = (6, 16)   # weekdays 08:00-18:00 Brussels summer time (UTC+2), fixed before looking at results


def queue_analysis(c: pd.DataFrame, sp: pd.DataFrame) -> dict:
    """Queue waits by arrival time and by backlog of the same support group at arrival."""
    import duckdb
    sp = sp.copy()
    sp["ts"] = pd.to_datetime(sp.ts, utc=True)
    sp["next_ts"] = pd.to_datetime(sp.next_ts, utc=True)
    allq = sp[(sp.spell_type == "queue") & sp.next_ts.notna()][["case_id", "grp", "ts", "next_ts", "dur_h", "pre_resolution", "step_no"]]
    con = duckdb.connect()
    con.register("q", allq)
    # backlog = other incidents already waiting in the same group's queue when this one arrives (whole log)
    bl = con.execute("""SELECT a.case_id, a.step_no, COUNT(b.case_id) AS backlog
                        FROM q a LEFT JOIN q b ON a.grp = b.grp AND b.ts < a.ts AND b.next_ts > a.ts
                        GROUP BY a.case_id, a.step_no""").df()
    q = allq.merge(bl, on=["case_id", "step_no"])
    q = q[q.pre_resolution & q.case_id.isin(c.case_id)].merge(
        c[["case_id", "impact", "product_g", "org_g", "open_day"]], on="case_id")
    h, dow = q.ts.dt.hour, q.ts.dt.dayofweek
    q["off_hours"] = (~((dow < 5) & (h >= BUSINESS_UTC[0]) & (h < BUSINESS_UTC[1]))).astype(int)
    q["backlog_bin"] = pd.cut(q.backlog, [-1, 0, 5, 1e9], labels=["0", "1-5", "6+"]).astype(str)
    q["wait_1h"] = (q.dur_h > 1).astype(int)
    ctrl = "C(impact) + C(product_g) + C(org_g) + C(open_day)"
    cl = {"groups": pd.factorize(q.product_g)[0]}
    m = smf.glm(f"dur_h ~ off_hours + C(backlog_bin) + {ctrl}", data=q, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds=cl)
    ratio = lambda k: dict(ratio=float(np.exp(m.params[k])), ci_low=float(np.exp(m.conf_int().loc[k, 0])),
                           ci_high=float(np.exp(m.conf_int().loc[k, 1])), p_value=float(m.pvalues[k]))
    rr_off = rr_poisson(f"wait_1h ~ off_hours + C(backlog_bin) + {ctrl}", q, "off_hours")
    off = ratio("off_hours")
    # what-if: off-hours arrivals wait like comparable business-hours arrivals (adjusted ratio).
    off_h = q.loc[q.off_hours == 1, "dur_h"].sum()
    saved = lambda r: off_h * (1 - 1 / r)
    n_inc = c.case_id.nunique()
    by = (q.groupby(["off_hours", "backlog_bin"]).agg(n_cases=("case_id", "size"), median_wait_h=("dur_h", "median"),
                                                      share_wait_over_1h=("wait_1h", "mean")).reset_index())
    return {"queue_entries": int(len(q)), "share_entries_off_hours": float(q.off_hours.mean()),
            "share_queue_time_from_off_hours_entries": float(off_h / q.dur_h.sum()),
            "mean_wait_ratio_off_hours": off,
            "mean_wait_ratio_backlog_1_5": ratio("C(backlog_bin)[T.1-5]"),
            "mean_wait_ratio_backlog_6plus": ratio("C(backlog_bin)[T.6+]"),
            "rr_wait_over_1h_off_hours": rr_off,
            "what_if_off_hours_like_business_hours": {
                "queue_hours_saved_per_1000_incidents": float(saved(off["ratio"]) / n_inc * 1000),
                "range_from_ratio_ci": [float(saved(off["ci_low"]) / n_inc * 1000), float(saved(off["ci_high"]) / n_inc * 1000)],
                "share_of_all_elapsed_time": float(saved(off["ratio"]) / c.resolution_h.sum())},
            "table": by}


def landmark_power(lm: pd.DataFrame) -> dict:
    """Smallest risk ratio the landmark comparison could detect (alpha 0.05, power 0.8)."""
    n1 = int(lm.early_ping_pong.sum()); n0 = int(len(lm) - n1)
    p0 = float(lm.loc[lm.early_ping_pong == 0, "slow"].mean())
    es = NormalIndPower().solve_power(nobs1=n1, ratio=n0 / n1, alpha=0.05, power=0.8)
    p1 = np.sin(np.arcsin(np.sqrt(p0)) + es / 2) ** 2
    return {"exposed": n1, "unexposed": n0, "baseline": p0, "min_detectable_rr": float(p1 / p0)}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


# ---------------------------------------------------------------- publishing
def suppress(t: pd.DataFrame, keys: list[str], n_col: str = "n_cases") -> pd.DataFrame:
    t = t.copy()
    small = t[n_col] < MIN_CELL_N
    for col in t.columns:
        if col not in keys:
            t[col] = t[col].astype("float64")
            t.loc[small, col] = np.nan
    t["suppressed"] = small.values
    return t


def figures(c, dec, tr, wu) -> None:
    fd = OUT / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    blue, grey = "#1f5f8b", "#9aa4b1"

    fig, ax = plt.subplots(figsize=(7, 3.2))
    d = dec.sort_values("share_of_elapsed")
    ax.barh(d.component, d.share_of_elapsed * 100, color=blue)
    ax.set_xlabel("% of total elapsed time to resolution")
    ax.set_title("Where incidents' elapsed time goes (elapsed status time, not effort)")
    fig.tight_layout(); fig.savefig(fd / "time_decomposition.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.4))
    lm = tr["landmark"]
    rows = [("All resolved incidents: raw", tr["rr_ping_pong_raw"], grey),
            ("All resolved incidents: adjusted", tr["rr_ping_pong_adjusted"], grey),
            (f"Still open at {LANDMARK_H}h, ping-pong in first {LANDMARK_H}h: raw", lm["rr_raw"], blue),
            (f"Still open at {LANDMARK_H}h, ping-pong in first {LANDMARK_H}h: adjusted", lm["rr_adjusted"], blue)]
    for i, (lab, r, col) in enumerate(rows):
        ax.errorbar(r["rr"], i, xerr=[[r["rr"] - r["ci_low"]], [r["ci_high"] - r["rr"]]], fmt="o", color=col, capsize=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis()
    ax.axvline(1, color="k", lw=0.8); ax.set_xlabel("Risk ratio: not resolved within 24 hours (95% CI)")
    ax.set_title("Ping-pong between groups and slow resolution")
    fig.tight_layout(); fig.savefig(fd / "ping_pong_rr.png", dpi=150); plt.close(fig)

    t = wu["org_table"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(t))
    ax.scatter(t.used_wait_user_raw * 100, y, color=grey, label="raw", zorder=3)
    ax.scatter(t.used_wait_user_adjusted * 100, y, color=blue, label="adjusted for impact, product, day", zorder=3)
    ax.set_yticks(y, t.org); ax.invert_yaxis()
    ax.set_xlabel("% of incidents put in 'Wait - User' at least once")
    ax.set_title("'Wait - User' use differs by organization beyond case mix\n(ordered by name, not ranked)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout(); fig.savefig(fd / "wait_user_by_org.png", dpi=150); plt.close(fig)


def main() -> dict:
    OUT.mkdir(exist_ok=True)
    c0, sp, contrib = load()
    c, reportable = prepare(c0, contrib)
    dq = json.loads((DATA / "data_quality.json").read_text())

    dec = decomposition(c)
    tr = transfers_analysis(c)
    lm = tr.pop("_landmark_frame")
    wu = wait_user_analysis(c, sp, reportable)
    ro = reopen_analysis(c)
    pl = pilot_mde(c)
    qa = queue_analysis(c, sp)
    lp = landmark_power(lm)
    sel = selection_diagnostic(c0)

    # published tables (aggregate, suppressed)
    org_tab = suppress(wu["org_table"], ["org"])
    by_tr = suppress(tr["by_transfers"], ["transfers_bin"])
    reo = suppress(ro["by_resolution_time"], ["res_bin"])
    org_impact = suppress(c[c.org_g.isin(reportable)].groupby(["org_g", "impact"]).agg(
        n_cases=("case_id", "size"), median_resolution_h=("resolution_h", "median"),
        slow_rate=("slow", "mean"), ping_pong_rate=("ping_pong", "mean")).reset_index(), ["org_g", "impact"])
    for name, t in {"org_wait_user": org_tab, "transfers": by_tr, "reopen_by_resolution_time": reo,
                    "org_by_impact": org_impact, "time_decomposition": dec, "selection_diagnostic": sel,
                    "queue_waits": suppress(qa["table"], ["off_hours", "backlog_bin"])}.items():
        t.to_csv(OUT / f"{name}.csv", index=False)
    figures(c, dec, tr, wu)

    n_pre = int(len(c0) - c0.resolved_at.isna().sum())
    results = {
        "data_quality": dq,
        "selection_diagnostic": sel.to_dict("records"),
        "population": {"resolved_incidents": n_pre, "analysis_cohort": int(len(c)),
                       "cohort_rule": f"resolved, opened {COHORT_START} to before {COHORT_END}",
                       "slow_definition_h": SLOW_H, "slow_rate": float(c.slow.mean()),
                       "slow_7d_rate": float(c.slow_7d.mean()), "unresolved_in_cohort_excluded": int(
                           ((pd.to_datetime(c0.opened_at, utc=True) >= COHORT_START) &
                            (pd.to_datetime(c0.opened_at, utc=True) < COHORT_END) & c0.resolved_at.isna()).sum()),
                       "median_resolution_h": float(c.resolution_h.median())},
        "privacy": {"dropped_columns": ["org:resource", "resource country", "org:role"],
                    "min_contributors": MIN_CONTRIBUTORS, "min_cell_n": MIN_CELL_N,
                    "reportable_orgs": reportable,
                    "suppressed_cells": {"org_by_impact": int(org_impact.suppressed.sum()),
                                         "transfers": int(by_tr.suppressed.sum()),
                                         "reopen_by_resolution_time": int(reo.suppressed.sum())},
                    "total_cells": {"org_by_impact": int(len(org_impact))}},
        "time_decomposition": dec.to_dict("records"),
        "transfers": {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v) for k, v in tr.items()},
        "wait_user": {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v) for k, v in wu.items()},
        "reopen": {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v) for k, v in ro.items()},
        "pilot_minimum_detectable_effect": pl,
        "queues": {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v) for k, v in qa.items()},
        "landmark_power": lp,
        "holm_adjusted_p": holm({
            "ping_pong_adjusted": tr["rr_ping_pong_adjusted"]["p_value"],
            "landmark_early_ping_pong": tr["landmark"]["rr_adjusted"]["p_value"],
            "wait_user_org_variation": wu["lr_test_org_terms"]["p_value"],
            "reopen_fast_close": ro["rr_fast_close_adjusted"]["p_value"],
            "queue_off_hours": qa["mean_wait_ratio_off_hours"]["p_value"],
            "queue_backlog_6plus": qa["mean_wait_ratio_backlog_6plus"]["p_value"]}),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ("population", "privacy")}, indent=1))
