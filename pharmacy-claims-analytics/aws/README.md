# AWS Lambda deployment with CloudWatch alarms

`deploy_aws.py` runs the scoring service as a **container-image AWS Lambda function** (us-east-1) behind an **IAM-authenticated Function URL**. The service's own alert transitions feed **CloudWatch alarms**, and an optional **EventBridge Scheduler canary** sends one request per minute.

**Status (Oct 27, 2026):** deployed. Image `pa-f21ee17` (digest `sha256:2624016d…`) serves the CI-trained model `529e5e44…`. The 150-request validation replayed real formulary rows: every status was as expected, both service alerts raised and cleared, and the drift and failure-rate CloudWatch alarms went OK → ALARM 48 s and 73 s after the service raised them, each executing its SNS action. Evidence: [`../reports/aws_lambda_validation.json`](../reports/aws_lambda_validation.json).

The separate `pbm-claims-durable-monitor` function and its `pbm-claims-canary` schedule (created Oct 24 in another session, not in this repo) sent synthetic-claims payloads, so the schedule is **paused**. Its `pbm-durable-*` alarms are idle.

## Architecture

| Piece | Detail |
|---|---|
| Image | `aws/Dockerfile.lambda`: Python 3.12 slim (ECR Public) + AWS Lambda Web Adapter 0.8.4; packages `reports/model.joblib` and `reports/monitoring_reference.json` from the CI run |
| Build | CodeBuild `pbm-claims-build` → ECR `pbm-claims-scoring`; uploads `reports/reference_pairs.json` to S3 for validation traffic |
| Function | `pbm-claims-scoring`, 2,048 MB, 30 s timeout; Function URL with `AWS_IAM` auth (not public) |
| Service alarms | 3 log metric filters on `pbm_alert_transition` JSON logs with `active=true` → alarms `pbm-scoring-score_drift`, `-request_failure_rate`, `-latency` |
| Platform alarms | Lambda `Errors` > 0, `Throttles` > 0, p95 `Duration` > 2,000 ms (5-minute windows) |
| Notifications | SNS topic `pbm-claims-scoring-alerts` |
| Canary | Scheduler `pbm-claims-canary`: `rate(1 minute)`, one 100-pair `/predict` call (reference pairs 900–999, PSI 0.151 vs. validation when repeated, under the 0.20 threshold), auto-ends after 6 hours |

## Run

```bash
pip install boto3 numpy
python aws/deploy_aws.py --email you@example.com   # build, deploy, alarms, 6-hour canary
python aws/validate_lambda.py                      # normal -> invalid -> shift -> recovery
python aws/canary_report.py
python aws/teardown_aws.py                         # delete everything
```

Validation traffic replays real 2026 formulary feature rows; the shift phase replays only rows whose pair required PA the prior year (values unaltered).

## Limits

- Telemetry is per execution environment (SQLite in `/tmp`) and resets when Lambda recycles it.
- No reserved concurrency; concurrent traffic could split monitoring windows across environments.
- No live accuracy monitoring: labels arrive only with the next contract-year file.
- Thresholds are demonstration defaults, not workload-based SLOs.
