# Responsible Workflow Analytics on a Real IT Incident Log

Process-mining and statistical analysis of the **real BPI Challenge 2013 incident-management event log** (Volvo IT Belgium; 7,554 incidents, 65,533 status-change events). The log was published by Ward Steeman on 4TU.ResearchData; this repo uses the copy bundled in the open-source `pm4py` test suite (`07_bpic2013_incidents.xes.gz`, SHA-256 `6ec2136c…d522`).

**Decision question:** which process change would most reduce how long customers wait for incidents to be resolved?
1. faster assignment out of queues (off-hours coverage, backlog control),
2. fewer team-to-team transfers ("ping-pong"), or
3. tighter rules for the `Wait - User` status?

**Ground rules:**
- Every duration is **elapsed calendar time in a status**, not effort or productivity.
- Results are reported for organizations and processes, never for individuals.

## 1. Data relevance and quality (findings that shaped the design)

| Check | Result |
|---|---|
| Duplicate events / out-of-order timestamps | 0 / 0 |
| Incidents never resolved | 8 of 7,554 |
| Impact or product changing within an incident | 0 |
| Pre-resolution time components summing to total | max error < 1e-11 h |

The most important finding is that **the extract is selected on its outcome.** It holds incidents opened up to 6 May 2012 whose *last* event falls between 30 April and 22 May. An incident opened in April therefore appears mostly when it ran long:

| Opened | Incidents | Median resolution | Not resolved within 7 days |
|---|---|---|---|
| Before April 2012 | 326 | 1,459 h | 89.9% |
| 1–30 April | 2,039 | 215 h | 62.3% |
| **1–4 May (analysis cohort)** | **5,168** | **0.58 h** | **3.4%** |

Pooling all incidents would overstate delays and distort every comparison. The analysis therefore uses the **1–4 May opening cohort** (5,162 resolved incidents; 6 unresolved excluded). These incidents are included whatever their duration and are followed for at least 18 days, so 24-hour and 7-day outcomes are fully observed.

## 2. Privacy and responsible use

- Person-level columns are dropped at ingestion, before any metric is computed: `org:resource` (1,440 individual names), `resource country`, and `org:role`.
- Only a count of distinct contributors per organization is kept, and only to apply the reporting floor.
- An organization is reported only if it has ≥ 5 distinct contributors and ≥ 20 incidents in the cohort. Six organizations qualify; the rest are pooled.
- Published table cells with fewer than 20 incidents are suppressed (6 of 19 organization × impact cells).
- Support groups (649 of them, some very small) are never reported. Organizations are listed by name, never ranked.
- A test scans every published CSV and JSON file for any of the 1,440 names.
- **Limitation:** there is no complementary suppression, so these rules reduce re-identification risk without guaranteeing privacy.

## 3. Findings (cohort: 5,162 incidents; 20.1% not resolved within 24 h)

**Where the elapsed time goes.** Shares are of total elapsed time to resolution, summed over all incidents.

| Component | Share of elapsed | Incidents with any | Median when present |
|---|---|---|---|
| `Wait - User` status | 37.8% | 18.3% | 20.9 h |
| Assignment queues | 25.2% | 46.3% | 0.8 h |
| Active status (In Progress / Assigned) | 21.2% | 99.8% | 0.2 h |
| Other waits (vendor, implementation, …) | 15.7% | 12.3% | 3.4 h |

**Ping-pong between groups: a competing explanation tested.** About 8.3% of incidents leave a support group and later return to it.
- Across the whole cohort, ping-pong incidents are 3.24× as likely to miss 24 h after adjusting for impact, product, owning organization and opening day (95% CI 2.75–3.80). The ratio is 3.07× for the 7-day outcome.
- **But long incidents have more time to collect transfers**, so part of this association could run in the reverse direction.
- A landmark analysis removes that problem: it measures ping-pong only within the first 4 hours, among the 1,684 incidents still open at 4 h. There the adjusted risk ratio is **1.11 (0.92–1.35)**.
- The ratio is null in both halves of the cohort (1.22, CI 0.78–1.91; and 0.99, CI 0.88–1.11).
- **Conclusion:** the large raw association is mostly not evidence that early ping-pong causes delay. A moderate effect (up to about 1.35×) cannot be ruled out, and later transfers are not tested.
- The landmark test had 80% power to detect a risk ratio of about 1.21 or larger (104 incidents exposed vs 1,580 not).
- Each transfer is associated with **6.1 more hours in assignment queues** (4.7–7.5), which points to queue time as the mechanism worth watching.

**`Wait - User`: workaround, legitimate pause, or measurement problem?**
- Use of the status differs by organization beyond case mix. After adjustment, the share of incidents ever placed in `Wait - User` ranges from **1.6% to 38.7%** (likelihood-ratio test on organization terms: χ² = 91.7, df = 6, p < 1e-17).
- An SLA clock that pauses during waits leaves out **53.5% of the elapsed time customers experience**. For the largest organization, it roughly halves the share of incidents missing 24 h (18.1% → 8.7%).
- Ranking the six organizations by either metric gives the same order (Spearman ρ = 1.0), but the levels change a lot.
- **44.9% of `Wait - User` periods end directly in resolution**, with no further active work. The log cannot show whether the user's reply solved the problem or the status was used to park the incident. That question needs **interviews with agents and customers**, not more log data.

**Assignment queues: when work arrives and how full the queue is.** This section looks at 4,505 queue entries.
- **Off-hours** is defined in advance, before looking at results: outside weekdays 08:00–18:00 Brussels time.
- Off-hours arrivals are 24.0% of queue entries but account for **35.9% of all queue time**.
- After adjusting for impact, product, owning organization, opening day and backlog, off-hours arrivals wait **1.82× as long on average** (95% CI 1.55–2.13). They are 1.61× as likely to wait more than 1 hour (1.53–1.70).
- **Backlog** is the number of incidents already waiting in the same group's queue on arrival. With 6 or more waiting, average wait is **2.18×** longer than with none (1.73–2.76); with 1–5 waiting, 1.83× (1.56–2.15).

Median queue wait (hours), by arrival time and backlog:

| Arrival | Empty queue | 1–5 waiting | 6+ waiting |
|---|---|---|---|
| Business hours | 0.14 | 0.40 | 0.67 |
| Off-hours | 0.39 | 1.07 | 5.02 |

- **What if off-hours arrivals waited like comparable business-hours arrivals?** That would remove about **966 queue hours per 1,000 incidents** (range 764–1,138 from the CI), or about **4.1% of all elapsed time**.
- **Caveat:** the remaining difference can't be confirmed as a staffing effect. Off-hours incidents could differ in ways the log doesn't record. Backlog is also counted only from incidents in this extract, so it's a lower bound.

**Rework (reopened incidents).**
- 3.4% of incidents are reopened after resolution.
- Incidents closed within 1 hour are reopened more often: adjusted risk ratio 1.38 (1.06–1.80), p = 0.018. After Holm correction across the six main tests, p = 0.036. It's still the weakest result, so treat it as a lead to investigate, not a finding.

## 4. Decision options

| Option | Evidence | Strength | Next step |
|---|---|---|---|
| Govern `Wait - User` (clear entry criteria, automatic follow-up) | Largest share of elapsed time (37.8%); usage varies by organization beyond case mix | Descriptive; whether it is a workaround is unknown | Interviews, then a case-randomized pilot (below) |
| **Off-hours queue coverage** (staggered shifts or an evening triage rota for high-backlog groups) | Off-hours arrivals wait 1.82× as long (1.55–2.13) after adjustment; about 966 queue hours per 1,000 incidents (764–1,138), about 4.1% of elapsed time | Adjusted association with a clear mechanism; staffing data not available | Get rota data; pilot a shift extension in the groups with the largest backlogs and compare against similar groups |
| Prevent queue backlog (overflow routing when 6+ incidents are waiting) | Waits are 2.18× as long with 6+ waiting | Adjusted association | Set an alert threshold; evaluate with an interrupted time series |
| Reduce ping-pong through routing rules | Raw risk ratio 3.2 shrinks to 1.1 (CI includes 1) once time-at-risk is handled | Weak for early ping-pong | Do not prioritize on current evidence |

**Pilot design for a `Wait - User` follow-up rule.**
- Baseline: 59.0% of incidents that enter `Wait - User` miss 24 h. This cohort averaged about 237 such incidents per weekday.
- The pilot needs 80% power at α = 0.05. No effect size is assumed; the table shows what each design could detect.

| Incidents per arm | Smallest detectable reduction | Weekdays of volume needed |
|---|---|---|
| 1,000 | 6.2 pp | ≈ 8.5 |
| 2,000 | 4.4 pp | ≈ 17 |

## 5. Limitations

- Group names without a "2nd"/"3rd" suffix are assumed to be first-line support; off-hours uses Brussels time for every organization, though some organizations may work in other time zones.
- Observational data. The only counterfactual method used is an adjusted landmark comparison, and unmeasured complexity can still confound it. The E-value for the landmark estimate is 1.46 at the point estimate and 1.0 at the CI bound.
- The analysis cohort covers four days in May 2012 at one organization, and the extraction rules are inferred from the data itself.
- No qualitative data was available. The `Wait - User` interpretation is posed as a question to answer, not as a conclusion.

## Run

```bash
pip install -r requirements.txt
python -I src/parse_xes.py data/raw/07_bpic2013_incidents.xes.gz data/parsed
python src/build.py       # privacy gate + DuckDB SQL (sql/case_metrics.sql): status spells, transfers, ping-pong
python src/analysis.py    # cohort, models, landmark analysis, suppressed tables, figures, results.json
pytest -q                 # 8 tests
```

**Tests cover:**
- no person names in any published output
- person columns removed before analysis
- reporting floors
- cell suppression
- time components adding up
- cohort follow-up
- SQL transfer counts matching an independent pandas computation
- a toy log for the ping-pong definition

**Stack:** Python, pandas, DuckDB SQL (window functions), statsmodels (modified Poisson with product-clustered errors, GLM), SciPy, matplotlib.
