# Verification record (Oct 26, 2026)

- **Real-data run:** GitHub Actions run 40436413306 downloaded the CY2024, CY2025 and CY2026 CMS formulary archives (SHA-256 in `pa_metrics.json`), passed the 5 pipeline logic tests, and produced `pa_metrics.json`, `model.joblib` and `monitoring_reference.json`. Log: `formulary_run.log`.
- **Model fingerprint:** `model.joblib` SHA-256 `529e5e44…` matches `monitoring_reference.json` and `manifest.json`.
- **Local tests (pinned requirements):** 14 passed (5 pipeline logic on hand-built fixtures, 9 service checks against the CI-trained model).
- **Loopback integration:** 145 HTTP requests, 7,000 real reference pairs scored; failure-rate and drift alerts raised and cleared (`service_validation.json`).
- **Not done:** AWS redeployment of this model; GCP deploy kit never executed.
- **CMS spending and MIMIC analyses:** unchanged; see `cms_analysis.json` and `mimic_workflow.json`.
