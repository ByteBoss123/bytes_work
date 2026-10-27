# Verification record (Oct 26, 2026)

- **Real-data run:** GitHub Actions run 40436413306 downloaded the CY2024, CY2025 and CY2026 CMS formulary archives (SHA-256 in `pa_metrics.json`), passed the 5 pipeline logic tests, and produced `pa_metrics.json`, `model.joblib` and `monitoring_reference.json`. Log: `formulary_run.log`.
- **Model fingerprint:** `model.joblib` SHA-256 `529e5e44…` matches `monitoring_reference.json` and `manifest.json`.
- **Local tests (pinned requirements):** 14 passed (5 pipeline logic on hand-built fixtures, 9 service checks against the CI-trained model).
- **Loopback integration:** 145 HTTP requests, 7,000 real reference pairs scored; failure-rate and drift alerts raised and cleared (`service_validation.json`).
- **AWS Lambda (Oct 27):** real-data model deployed; 150-request validation passed and both CloudWatch alarms fired (`aws_lambda_validation.json`).
- **Dashboard:** rendered in headless Chromium on GitHub Actions in desktop, dark and phone views, with no console errors or overflow (`dashboard_screenshot.json`).
- **Not done:** GCP deploy kit never executed.
- **CMS spending and MIMIC analyses:** unchanged; see `cms_analysis.json` and `mimic_workflow.json`.
- **Clinical-notes extraction:** GitHub Actions run 40448264497 (6 logic tests passed; LLM run on all 1,081 test sentences). Formulary link recomputed in run 40454539321 after fixing the RxNav query encoding.
- **Drug-label RAG:** AWS CodeBuild build #8 (commit e9eb6da): 14 unit tests passed; 470 Bedrock prompts answered; results copied into `drug_rag.json` (outputs in S3 `drug-rag/8/`). Build #7 was discarded: it predates the RxNorm label fallback and prose-answer parsing.
