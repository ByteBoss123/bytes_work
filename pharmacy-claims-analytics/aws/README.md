# AWS Lambda deployment with CloudWatch alarms

The scoring service runs as a **container-image AWS Lambda function** (us-east-1) behind an
**IAM-authenticated Function URL**. Its own alert transitions feed **CloudWatch alarms**, and an
**EventBridge Scheduler canary** sends one synthetic request per minute.

Evidence: [`reports/aws_lambda_validation.json`](../reports/aws_lambda_validation.json).

## Architecture

| Piece | Detail |
|---|---|
| Image | `aws/Dockerfile.lambda`: Python 3.12 slim (ECR Public) + AWS Lambda Web Adapter 0.8.4; trains the seed-42 model at build time |
| Build | CodeBuild `pbm-claims-build` → ECR `pbm-claims-scoring` |
| Function | `pbm-claims-scoring`, 2,048 MB, 30 s timeout; Function URL with `AWS_IAM` auth (not public) |
| Service alarms | 3 log metric filters on `pbm_alert_transition` JSON logs with `active=true` → `PBMScoring/*AlertRaised` → alarms `pbm-scoring-score_drift`, `-request_failure_rate`, `-latency` |
| Platform alarms | Lambda `Errors` > 0, `Throttles` > 0, p95 `Duration` > 2,000 ms (5-minute windows) |
| Notifications | SNS topic `pbm-claims-scoring-alerts` (alarm and OK actions); add an email subscriber to receive them |
| Canary | Scheduler `pbm-claims-canary`: `rate(1 minute)`, one 100-episode `/predict` call, auto-ends after 6 hours |

Service thresholds are unchanged from `SERVICE.md`: PSI > 0.20 over the last 1,000 scores,
request failure rate > 5% over the last 100 requests, p95 processing latency > 2,000 ms.

## Measured results (2026-10-24)

- **Model fingerprint** in the deployed image matches the committed run (`cd7d8298…`).
- **Complete validation run** (fresh environment, 150 requests): every request returned the
  expected status (140 × 200, 10 × 400 for the invalid-payload phase, 0 × 5xx). Failure-rate and
  drift alerts fired and cleared; PSI and alert values match the local loopback test exactly.
- **CloudWatch alarms** after a controlled injection: request-failure-rate alarm OK → ALARM
  51 s after the service raised it; drift alarm OK → ALARM 88 s after. Both executed their SNS action.
- **Duration alarm** also fired once on cold starts (first invoke ≈ 15.6 s) and recovered.
- **Bug found by deploying:** Function URL events forward `/health?` even with no query string;
  exact path matching returned 404. Fixed with query-insensitive routing plus a regression test.

## Reproduce

```bash
pip install boto3 numpy
python aws/deploy_aws.py --email you@example.com   # build, deploy, alarms, 6-hour canary
python aws/validate_lambda.py                      # 150-request phases (+ --soak-minutes N)
python aws/canary_report.py                        # canary invocations, errors, durations, alarms
python aws/teardown_aws.py                         # delete everything
```

`deploy_aws.py` packages the local source to S3 for CodeBuild. The live deployment above was
built by an equivalent buildspec that cloned GitHub commit `3f609e0` and overlaid the two changed
files, verified by SHA-256 in the build log. The scripts have not been run end to end as written.

## Limits (keep these in any description)

- **Synthetic model, synthetic traffic.** No real claims, users or production load.
- **Telemetry is per execution environment** (SQLite in `/tmp`); it resets when Lambda recycles
  the environment, so windows are not durable across cold starts.
- **No reserved concurrency.** The account limit of 10 leaves no room to reserve. Sequential
  traffic reused one environment, but concurrent traffic could split windows across environments.
- **No live performance monitoring.** No outcome labels, so no live AUC or calibration tracking.
- Thresholds are demonstration defaults, not workload-based SLOs.
