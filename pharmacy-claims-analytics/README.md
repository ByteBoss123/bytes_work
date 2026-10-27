# Pharmacy Benefit Operations Analytics

**Real public data only.** A prior-authorization (PA) forecast trained on CMS Part D formulary files, an HTTP scoring service with drift, failure-rate and latency alerts, and two real healthcare-data analyses (de-identified MIMIC-IV medication records and CMS Part D spending aggregates). Python, SQL, scikit-learn.

This is a portfolio project. It is not professional PBM experience and is not affiliated with CVS Health or Caremark. No claims, rejections, member records or patient-level PBM data are used anywhere: those are not public.

## The operational question

Each fall, Part D plans publish next year's formulary rules. A prior-authorization requirement means a pharmacy claim for that drug will reject until the PA is approved, so PA changes drive rejection volume, call volume and member outreach.

**For each (formulary, drug) pair a plan lists next year, will it require prior authorization, and which pairs are *newly* gaining PA?** Every feature comes from the prior contract year's public file; the label is the next year's PA flag.

## Data

CMS Quarterly Prescription Drug Plan Formulary, Pharmacy Network, and Pricing Information files ([data.cms.gov](https://catalog.data.gov/dataset/quarterly-prescription-drug-plan-formulary-pharmacy-network-and-pricing-information)). Exact URLs and SHA-256 hashes of each download are in [`reports/pa_metrics.json`](reports/pa_metrics.json). Raw files are not committed.

| Contract year | Formulary-drug pairs | Formularies | Drugs (RXCUI) | Pairs with PA |
|---|---:|---:|---:|---:|
| 2024 | 1,490,827 | 421 | 6,522 | 26.6% |
| 2025 | 1,298,449 | 378 | 6,092 | 27.7% |
| 2026 | 1,121,429 | 331 | 5,994 | 28.1% |

Unit: formulary × RXCUI (PA = any NDC under the RXCUI flagged). Formularies get new IDs every year, so each 2026 formulary is linked to its 2025 predecessor through the continuing plans (contract-plan-segment) that use it; 90.8% of 2026 pairs belong to a linked formulary.

## Design

- **Train:** 2024 features → 2025 labels (1,298,449 pairs). **Test:** 2025 features → 2026 labels (1,121,429 pairs), evaluated once.
- **Model selection:** 75 held-out training-year formularies (255,908 pairs). Histogram gradient boosting beat logistic regression on average precision (0.992 vs 0.987), then was refit on the full training year.
- **Features (16):** last year's PA/step-therapy/quantity-limit flags and tier for the same pair; the drug's prior-year PA, step-therapy and quantity-limit rates across all formularies, breadth and mean tier; the predecessor formulary's PA share and size; plan type (PDP vs MA) and special-needs status. No year-Y rule enters the features.
- **Baselines an analyst could run without a model:** (1) carry forward last year's PA flag, falling back to the drug's prior-year PA rate for new pairs; (2) the drug's prior-year PA rate alone.
- **Uncertainty:** 200-replicate bootstrap that resamples whole formularies.

## Results (CY2026 test year)

| | Model | Carry-forward rule | Drug PA rate only |
|---|---:|---:|---:|
| ROC-AUC, all 1.12M pairs | **0.995** | 0.986 | 0.982 |
| Average precision | **0.990** | 0.969 | 0.960 |
| Brier score (lower is better) | 0.0166 | **0.0163** | 0.0458 |
| Precision when flagging the top 314,740 pairs (the true PA count) | 96.2% | **96.4%** | 88.0% |

**What this means**
- The model ranks pairs better overall (AUC +0.009, 95% CI +0.008 to +0.010), but **at the operating point it does not beat the carry-forward rule**: precision is 96.2% vs 96.4%, and calibration is no better. Last year's PA flag explains most of next year's.
- **The hard part is change.** 985,605 pairs appeared in both years; only 6,242 gained PA (0.87%) and 4,528 lost it. For flagging the pairs that gain PA, the model (AUC 0.840, AP 0.034) is slightly **worse** than simply ranking by the drug's prior-year PA rate (AUC 0.852, AP 0.040). Carry-forward cannot rank these at all (AUC 0.5).
- **Pairs new to a formulary** (135,824; 33.8% PA): model AUC 0.979 vs 0.978 for the drug-rate fallback; the difference is not significant (CI −0.0003 to +0.0016).
- **Drugs new to Part D** (7,912 pairs; 46.5% PA): AUC 0.556. Prior-year formulary data carries almost no signal for them.
- PA rate by last year's tier: 78.6% at tier 5 (specialty) vs 8.5% at tier 2 ([`pa_rate_by_prior_tier.csv`](reports/pa_rate_by_prior_tier.csv)).
- Score drift between the training-year validation formularies and the 2026 test year: PSI 0.041, under the 0.20 alert threshold.

**Recommendation from this run:** for next-year PA planning, start from the carry-forward list and rank non-PA pairs by the drug's PA rate across other plans; the gradient-boosted model adds ranking polish, not a better operating list. The open problem is predicting PA additions and new-to-market drugs, which needs signals this file does not have (launch dates, list price, clinical class).

## Scoring service and monitoring

[`pbm/service.py`](pbm/service.py) serves the CI-trained model over HTTP (`/predict`, `/health`, `/monitoring`, `/metrics`). It refuses to start unless the model's SHA-256 matches the monitoring reference, rejects labels, identifiers and out-of-range values, and tracks:

- request failure rate over the last 100 requests (alert above 5%)
- p95 processing latency (alert above 2,000 ms)
- PSI of the last 1,000 scores against the validation-score reference (alert above 0.20)

The loopback integration test ([`reports/service_validation.json`](reports/service_validation.json)) sends 145 HTTP requests replaying real 2026 feature rows. A shift phase replays only pairs that required PA last year (real rows, unaltered values), which raised the drift alert; invalid payloads raised the failure-rate alert; both cleared on recovery.

**AWS:** [`aws/`](aws/README.md) deploys the service as a container on Lambda with CloudWatch alarms. The earlier deployment served the removed synthetic model, so its evidence was deleted; the real-data model has **not yet been redeployed**.

## Real healthcare-data analyses

- [MIMIC-IV medication workflow](MIMIC_WORKFLOW.md): 18,087 prescription-component rows from 100 de-identified patients; join validation and timestamp/documentation quality checks.
- [CMS Part D spending](REAL_DATA.md): 5,000 provider-drug aggregate rows covering $36.7M in recorded drug cost.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
python -m pbm.formulary --out reports      # downloads ~8 GB of CMS archives
PYTHONPATH=. python tests/service_integration.py
python -m pbm.service --reports reports
```

The real-data run executes in GitHub Actions ([`.github/workflows/pa-formulary.yml`](../.github/workflows/pa-formulary.yml)), which commits the metrics, model and monitoring reference back to this branch. Log: [`reports/formulary_run.log`](reports/formulary_run.log).

## Limitations

- Plan-design records only: no claims, rejection counts, member enrollment or utilization, so impact on rejection volume is not measured.
- One training year pair and one test year pair. The CY2027 file (expected January 2027) would allow a prospective test of the frozen model.
- PA is collapsed to RXCUI level; NDC-specific rules are lost.
- 9.2% of 2026 pairs sit in formularies with no linked predecessor and get no pair history.

## Supported resume wording

> Forecast 2026 prior-authorization requirements for 1.12M real CMS Part D formulary-drug pairs (331 formularies) from prior-year rules; gradient boosting reached 0.995 ROC-AUC but matched a carry-forward rule at the operating cutoff (96.2% vs 96.4% precision), so recommended the simpler rule.

Keep "formulary" or "plan-design" data. Do not describe this as claims data, rejection data or production PBM work.
