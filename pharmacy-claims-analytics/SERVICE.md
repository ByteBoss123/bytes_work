# Local inference and request monitoring

This is a **running local HTTP service implementation**, tested through actual HTTP calls. Request telemetry is measured, not fabricated. Test traffic replays real CMS formulary feature rows (public plan-design data). It is not a real-user production deployment and does not establish production employment experience.

## Start

From the project directory after installing requirements (the model and monitoring reference are committed by the CI run):

```bash
python -m pbm.service --reports reports --port 8765
```

By default the server binds only to `127.0.0.1`. It has no credentials, TLS, or hosted persistence of its own. For a hosted run, `deploy/` builds a container that binds `0.0.0.0` behind Cloud Run's IAM-authenticated HTTPS ingress, and emits alert transitions as structured JSON logs for Cloud Monitoring. See [aws/README.md](aws/README.md) for the AWS Lambda deployment that was run, and [deploy/README.md](deploy/README.md) for the unexecuted Cloud Run kit. Python's built-in HTTP server is suitable for this controlled demonstration, not public production serving.

- GET `/health`: model readiness and model SHA-256.
- POST `/predict`: 1–100 formulary-drug pairs, prior-year features only (see `reports/monitoring_reference.json` for the 16 feature names; `null` is allowed for missing history).
- GET `/monitoring`: rolling metrics, thresholds, sample sizes, and current alert states.
- GET `/metrics`: Prometheus-compatible text exposition. A Prometheus server is not deployed.

Example payload: one row of `reports/reference_pairs.json` wrapped as `{"pairs": [ ... ]}`.

There are no plan, formulary, drug or member identifiers in the schema. Labels, identifiers, extra fields, booleans, nonfinite or negative values, out-of-range indicators and rates, and oversized batches are rejected. Payloads are not logged. SQLite telemetry stores request status, processing latency, model fingerprint, and probabilities only. Latency excludes response transfer and telemetry persistence; it is **server processing latency**, not end-to-end client latency.

## Monitoring logic

- Failure rate: last 100 prediction requests; alert above 5% after at least 20 requests. Counts 4xx validation failures and 5xx internal errors together; this is request failure rate, not server-only error rate.
- Processing latency: last 100 prediction requests; p95 alert above 2,000 ms after 20 requests.
- Score drift: last 1,000 predictions compared with validation risk scores; PSI alert above 0.20 after 200 scores.
- Model/reference fingerprint mismatch prevents startup.
- Alert changes persist to SQLite; active states are available over HTTP. No Slack/email/PagerDuty notification is configured.
- Records are filtered by model fingerprint. Restarts with the same model preserve counters and history.
- Labels arrive only with the next contract-year file: no live ROC-AUC or calibration monitoring.

Thresholds are demonstration defaults. A real service needs workload-based SLOs, retention controls, authentication, infrastructure health monitoring, incident ownership, delayed-label evaluation, and independent validation. Do not expose this server publicly.

## Reproduce the live integration test

```bash
# macOS/Linux
PYTHONPATH=. python tests/service_integration.py
# Windows PowerShell
$env:PYTHONPATH = "."
python tests/service_integration.py
```

The test starts an ephemeral loopback service, sends 145 actual HTTP requests, scores 7,000 real reference pairs, injects invalid payloads, replays a real subpopulation shift (pairs that required PA last year), verifies alerts and recovery, checks Prometheus output, and shuts down the test server. Results are in `reports/service_validation.json`. This is an integration test, not a load test or a sustained uptime measurement.
