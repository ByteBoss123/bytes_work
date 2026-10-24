# Cloud Run deployment with Cloud Monitoring alerts

Deploys the scoring service to **Cloud Run** behind IAM authentication, turns its own
alert transitions into **Cloud Monitoring alert policies**, and validates the live
endpoint with synthetic traffic.

Run everything in **Google Cloud Shell** (gcloud, git, Python 3 and curl are already there).

## Steps

```bash
git clone -b pharmacy-claims-analytics https://github.com/ByteBoss123/bytes_work
cd bytes_work/pharmacy-claims-analytics
gcloud config set project YOUR_PROJECT_ID

bash deploy/deploy.sh                                   # build + deploy, prints /health
NOTIFY_EMAIL=you@example.com bash deploy/alerts.sh      # metrics + policies (+ email)
bash deploy/validate.sh 30                              # phases + 30-minute soak
```

`validate.sh` writes `reports/cloud_run_validation.json`. Commit that file: it is the
evidence for any resume claim about the deployment. Optional: `UPTIME_CHECK=1 bash deploy/alerts.sh`
adds an authenticated `/health` uptime check. Remove everything with `bash deploy/teardown.sh`.

## What gets created

| Piece | Detail |
|---|---|
| Cloud Run service `pbm-claims-scoring` | `us-east4`, `--no-allow-unauthenticated`, max 1 instance, 1 vCPU / 1 GiB |
| Image | Built by Cloud Build into Artifact Registry repo `pbm`; trains the seed-42 model at build time |
| Log-based metrics | `pbm_score_drift_alert`, `pbm_request_failure_rate_alert`, `pbm_latency_alert`: count of `pbm_alert_transition` log lines with `active=true` |
| Alert policies (5) | The 3 service rules above, Cloud Run p95 latency > 2 s for 5 min, more than 2 HTTP 5xx in 5 min |
| Optional | Email notification channel (`NOTIFY_EMAIL`), `/health` uptime check (`UPTIME_CHECK=1`) |

The service's thresholds are unchanged from `SERVICE.md`: PSI > 0.20 over the last 1,000
scores, request failure rate > 5% over the last 100 requests, p95 processing latency > 2,000 ms.

## What validate.sh exercises

1. 20 normal requests (validation-period episodes): no drift alert expected.
2. 10 invalid payloads: request-failure-rate alert expected.
3. 20 shifted-feature requests (1,000 scores): drift alert expected.
4. 100 normal requests: both alerts expected to clear.
5. Optional soak: one normal 50-episode request every 2 seconds for N minutes.

Phases 2 and 3 should open two incidents in Cloud Monitoring (and send email if a channel
is set). Check **Monitoring > Alerting > Incidents** a few minutes after the run; log-based
metrics lag by about a minute.

## Limits (keep these in any description of the work)

- **Traffic is synthetic**, sent by you. No real users, real claims or production workload.
- **The model is trained on simulated claims.** Deployment does not change that.
- **Telemetry is per-instance.** It lives in `/tmp` SQLite on the single instance and resets
  when Cloud Run replaces the instance (scale to zero, redeploy). Durable telemetry would need
  Cloud SQL, Firestore or BigQuery.
- **No live performance monitoring.** Outcome labels never arrive, so there is no live AUC or
  calibration tracking; drift alerts are on score distributions only.
- Thresholds are demonstration defaults, not workload-based SLOs.

## Troubleshooting

- **`gcloud builds submit` permission errors** (common in projects created after 2024, where
  Cloud Build runs as the Compute Engine default service account): grant that account
  `roles/artifactregistry.writer`, `roles/logging.logWriter` and `roles/storage.objectViewer`,
  then rerun `deploy.sh`.
- **`403` from `/health`:** your account needs `roles/run.invoker` on the service (project
  Owners already have it).
- **Policy creation retries in alerts.sh:** normal right after the log-based metrics are created.
