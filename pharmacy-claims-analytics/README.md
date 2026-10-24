# Pharmacy Claims Operations Analytics

**A PBM portfolio project combining synthetic claims modeling, real public CMS aggregate-data analysis, and a locally tested HTTP monitoring service.** Python + SQL + scikit-learn, temporal model validation, batch scoring, an offline interactive dashboard, and simulated monitoring.

## The operational question

At the moment a synthetic pharmacy claim is initially rejected, which cases should an operations team review first because no paid resolution is likely within seven days?

This project demonstrates a workflow relevant to Pharmacy Benefit Manager (PBM) operations. It is **not professional PBM experience**, not a production deployment, not clinical decision support, and not affiliated with CVS Health or Caremark. The claims model uses simulated records and relationships. A separate CMS analysis uses real public provider-drug aggregates; these are not patient-level claims. Rejection is not automatically a final coverage denial, abandonment, or harm to a member.

## Added: real healthcare data and measured service telemetry

- [CMS spending analysis](REAL_DATA.md): 5,000 real public aggregate rows covering $36.7M in recorded drug spending. Separate from synthetic model training.
- [HTTP scoring and monitoring service](SERVICE.md): actual request telemetry, model fingerprints, rolling latency/failure/PSI metrics, persisted alerts, and Prometheus-compatible output. Verified using 145 HTTP requests and 7,000 synthetic episode scores.
- Live integration evidence: [service_validation.json](reports/service_validation.json). Local test deployment only; no real-user production or patient-data deployment is claimed.
- [Cloud Run deployment kit](deploy/README.md): Cloud Build image, IAM-authenticated Cloud Run service, log-based Cloud Monitoring alert policies, and a remote validation script. Evidence of a hosted run lives in `reports/cloud_run_validation.json` once `deploy/validate.sh` has been run.

## Measured result — seed 42

| Held-out test measure | Selected logistic regression | Rejection-reason baseline |
|---|---:|---:|
| ROC-AUC | 0.665 | 0.642 |
| Average precision | 0.513 | 0.456 |
| Brier score (lower is better) | 0.210 | 0.214 |
| Recall at 20% review capacity | 32.4% | 29.7% |
| Cases captured in 592 reviews | 337 / 1,041 | 309 / 1,041 |

**60,000 synthetic episodes; 2,958 test rejected episodes.** These are measured simulator results, not evidence of real-world generalization or improved patient outcomes. The model captures 28 more unresolved cases than the reason-only baseline at the same cohort-level review capacity; it does not resolve or prevent them. See [metrics.json](reports/metrics.json) for all partitions, validation results, confidence intervals, and denominators.

## Run locally (Python 3.11 or 3.12)

From this project directory:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m pbm.pipeline --n 60000 --seed 42 --out reports
python -m pbm.score --model reports/model.joblib --input reports/sample_synthetic_claims.csv --output reports/scored_sample.csv
```

Open `reports/dashboard.html` in a browser. It is self-contained, works offline, and supports rejection-reason selection and review-capacity comparison. GitHub displays its source; download it to interact. No API key, cloud account, or private patient data is required.

The committed dashboard and metrics let reviewers inspect the measured run without installing Python. Large generated SQLite databases, fitted models, and full predictions are deliberately excluded from Git. Regenerate them with the pipeline command. Never load an untrusted joblib model.

## Architecture

1. **Generate:** seed-controlled initial episodes, submission-time features, and separate future resolution timestamps.
2. **Validate:** reject duplicate episode IDs, missing fields, invalid categories/ranges, and impossible timestamp order.
3. **Curate with SQLite:** enforce claim grain; derive seven-day labels only for mature episodes.
4. **Train and validate:** training-only preprocessing; logistic regression versus histogram gradient boosting; select by validation average precision.
5. **Evaluate:** frozen model on a later test period; prevalence and training-derived rejection-reason baselines; capacity curves, Brier score, subgroup diagnostics, and bootstrap ROC-AUC interval.
6. **Score:** persist locally and run batch inference on initially rejected episodes; fail on unexpected categories.
7. **Monitor:** compare validation/test score distributions and an explicitly injected stress scenario using PSI.

## Leakage and censoring controls

- Prediction time is immediately after **initial rejection**.
- `resolved_at` is used only to construct the outcome; it is never a model input.
- Only episodes with seven days of follow-up before the fixed snapshot enter training/evaluation.
- Training: Jan 1–May 10; validation: May 18–Jun 22; test: Jun 30–Aug 20, 2025. Seven-day gaps permit earlier labels to mature.
- IDs and submission dates are excluded from modeling. One episode per member avoids cross-split member overlap in this simplified simulator.
- No tuning or model selection uses test metrics. Validation chooses between two fixed candidates; neither is refit on test data.
- The generator's relationships are deliberately known. Learning them proves implementation works, not that actual PBM workflows follow them.

## Repository guide

| File | Purpose |
|---|---|
| `pbm/pipeline.py` | Generator, validation, SQL execution, evaluation, monitoring |
| `pbm/score.py` | Batch inference CLI |
| `pbm/dashboard.py` | Offline interactive dashboard renderer |
| `sql/curate.sql` | Mature outcomes and submission-time modeling dataset |
| `sql/operations.sql` | Workflow metrics with explicit denominators |
| `tests/test_pipeline.py` | Grain, chronology, embargo, censoring, ranking, PSI tests |
| `reports/metrics.json` | Complete measured evaluation |
| `reports/dashboard.html` | Downloadable interactive dashboard |
| `reports/manifest.json` | Versions, seed, and source hashes |
| `MODEL_CARD.md` | Intended use, limitations, and monitoring semantics |
| `DATA_DICTIONARY.md` | Field definitions and simulation assumptions |

## What to investigate next

The simulator produces measurable differences across rejection categories. This supports demonstrating workflow analysis, not recommending changes to actual benefit policy. A real implementation would need longitudinal transactions, resubmission/reversal matching, clinical urgency review, expert-validated outcomes, data authorization, subgroup checks, and prospective evaluation. The current capacity comparison ranks a whole held-out cohort and does not simulate staffing arrivals or daily backlogs.

## Resume wording supported by this run

> Implemented a Python/SQL pharmacy claims analytics project on 60,000 synthetic episodes; achieved 0.665 test ROC-AUC and captured 32.4% of seven-day unresolved cases at 20% review capacity versus 29.7% for a reason-based baseline, with an interactive dashboard and simulated drift alerts.

Use under **Projects**, with “synthetic” and “simulated” retained. Do not claim production deployment, healthcare employment, actual cost savings, or clinical impact from this project.

## Domain and methodology references

These sources explain context; **none supplied the generated data or simulation parameters**.

- [CMS: Guide to requests for Part D Prescription Drug Event data](https://www.cms.gov/PrescriptionDrugCovGenIn/Downloads/GuidePartDDataRequests.pdf). PDE records are summary extracts, not the full sequence of individual claim transactions. This project does not claim to reproduce CMS PDE or NCPDP schemas.
- [CMS: Prescription Drug Event Data Guidance](https://www.cms.gov/medicare/prescription-drug-coverage/drugcoverageclaimsdata/index).
- [scikit-learn: time-series split and gaps](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). This project implements explicit calendar cutoffs and gaps rather than calling TimeSeriesSplit.

## Verification

The full pipeline, batch scoring, and **17 Python tests passed** (10 pipeline, 7 service). Optional dashboard interaction tests (`npm install && npm test`, Node 22+) passed in jsdom. Browser visual verification was blocked by a failed Chromium download; see [verification record](reports/VALIDATION.md). This is a local runnable project, not a hosted application.
