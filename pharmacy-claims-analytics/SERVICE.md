# Local inference and request monitoring

This is a **running local HTTP service implementation**, tested through actual HTTP calls. Request telemetry is measured, not fabricated. Test traffic is synthetic. It is not a real-user production deployment and does not establish production employment experience.

## Start

From the project directory after installing requirements:

```bash
python -m pbm.pipeline
python -m pbm.service --reports reports --port 8765
```

The server binds only to `127.0.0.1`. It has no external ingress, credentials, TLS, container orchestration, or hosted persistence. Python's built-in HTTP server is suitable for this controlled demonstration, not public production serving.

- GET `/health`: model readiness and model SHA-256.
- POST `/predict`: 1–100 rejected episodes, submission-time features only.
- GET `/monitoring`: rolling metrics, thresholds, sample sizes, and current alert states.
- GET `/metrics`: Prometheus-compatible text exposition. A Prometheus server is not deployed.

Example payload:

```json
{"episodes":[{"days_supply":30,"submitted_cost":120.0,"prior_rejections_90d":1,"member_tenure_days":300,"weekend":0,"reject_reason":"coverage_check","channel":"retail","drug_type":"standard"}]}
```

There are no patient/member IDs in the service schema. Identifiers, outcome timestamps, extra fields, unseen categories, nonfinite/negative numeric fields, and oversized batches are rejected. Payloads are not logged. SQLite telemetry stores request status, processing latency, model fingerprint, and probabilities only. Latency excludes response transfer and telemetry persistence; it is **server processing latency**, not end-to-end client latency.

## Monitoring logic

- Failure rate: last 100 prediction requests; alert above 5% after at least 20 requests. Counts 4xx validation failures and 5xx internal errors together; this is request failure rate, not server-only error rate.
- Processing latency: last 100 prediction requests; p95 alert above 2,000 ms after 20 requests.
- Score drift: last 1,000 predictions compared with validation risk scores; PSI alert above 0.20 after 200 scores.
- Model/reference fingerprint mismatch prevents startup.
- Alert changes persist to SQLite; active states are available over HTTP. No Slack/email/PagerDuty notification is configured.
- Records are filtered by model fingerprint. Restarts with the same model preserve counters and history.
- Labels are unavailable: no claim of live ROC-AUC, calibration monitoring, concept-drift detection, or clinical quality monitoring.

Thresholds are demonstration defaults. A real service needs workload-based SLOs, retention controls, authentication, infrastructure health monitoring, incident ownership, delayed-label evaluation, and independent validation. Do not expose this server publicly.

## Reproduce the live integration test

```bash
# macOS/Linux
PYTHONPATH=. python tests/service_integration.py
# Windows PowerShell
$env:PYTHONPATH = "."
python tests/service_integration.py
```

The test starts an ephemeral loopback service, sends 145 actual HTTP requests, scores 7,000 synthetic episodes, deliberately injects invalid payloads and shifted features, verifies alerts and recovery, checks Prometheus output, and shuts down the test server. Results are in `reports/service_validation.json`. This is an integration test, not a load test or a sustained uptime measurement.
