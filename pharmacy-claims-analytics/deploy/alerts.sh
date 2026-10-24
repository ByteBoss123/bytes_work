#!/usr/bin/env bash
# Cloud Monitoring alerting for the deployed scoring service.
#   1-3. Log-based metrics on the service's own alert transitions (PSI drift,
#        request failure rate, processing latency), each with an alert policy.
#   4.   Cloud Run p95 end-to-end request latency > 2,000 ms for 5 minutes.
#   5.   More than 2 HTTP 5xx responses in 5 minutes.
#   6.   Optional (UPTIME_CHECK=1): authenticated /health uptime check + policy.
# Re-runnable: existing metrics and policies with the same names are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/config.sh
gcloud config set project "$PROJECT_ID" >/dev/null
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
RUN_FILTER="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$SERVICE\""
RUN_FILTER_JSON="${RUN_FILTER//\"/\\\"}"   # same filter, quotes escaped for JSON strings

# --- Notification channel (optional) -------------------------------------------------
CHANNELS="[]"
if [[ -n "$NOTIFY_EMAIL" ]]; then
  TOKEN="$(gcloud auth print-access-token)"
  API="https://monitoring.googleapis.com/v3/projects/$PROJECT_ID/notificationChannels"
  CHANNEL="$(curl -fsS -H "Authorization: Bearer $TOKEN" "$API" | python3 -c '
import json,sys
email=sys.argv[1]
for c in json.load(sys.stdin).get("notificationChannels",[]):
    if c.get("type")=="email" and c.get("labels",{}).get("email_address")==email:
        print(c["name"]); break' "$NOTIFY_EMAIL")"
  if [[ -z "$CHANNEL" ]]; then
    CHANNEL="$(curl -fsS -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
      -d "{\"type\":\"email\",\"displayName\":\"$POLICY_PREFIX alerts\",\"labels\":{\"email_address\":\"$NOTIFY_EMAIL\"}}" \
      "$API" | python3 -c 'import json,sys; print(json.load(sys.stdin)["name"])')"
  fi
  echo "Notification channel: $CHANNEL"
  CHANNELS="[\"$CHANNEL\"]"
else
  echo "NOTIFY_EMAIL not set: policies will open incidents in the console without email."
fi

create_policy() {  # $1 = policy JSON file
  local name
  name="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["displayName"])' "$1")"
  if [[ -n "$(gcloud monitoring policies list --filter="displayName=\"$name\"" --format='value(name)')" ]]; then
    echo "Policy exists, skipped: $name"
  elif gcloud monitoring policies create --policy-from-file="$1" --format='value(name)' >/dev/null; then
    echo "Policy created: $name"
  else
    return 1
  fi
}

# --- 1-3: service alert transitions -> log-based metrics -> policies -----------------
for ALERT in score_drift request_failure_rate latency; do
  METRIC="pbm_${ALERT}_alert"
  if ! gcloud logging metrics describe "$METRIC" >/dev/null 2>&1; then
    gcloud logging metrics create "$METRIC" \
      --description="Scoring service raised its $ALERT alert (jsonPayload.active=true)" \
      --log-filter="$RUN_FILTER AND jsonPayload.event=\"pbm_alert_transition\" AND jsonPayload.alert=\"$ALERT\" AND jsonPayload.active=true"
    echo "Log metric created: $METRIC"
  else
    echo "Log metric exists, skipped: $METRIC"
  fi
  case "$ALERT" in
    score_drift)          RULE="PSI of the last 1,000 scores vs. validation scores > 0.20 (min 200 scores)";;
    request_failure_rate) RULE="4xx+5xx share of the last 100 /predict requests > 5% (min 20 requests)";;
    latency)              RULE="p95 server processing latency of the last 100 /predict requests > 2,000 ms";;
  esac
  cat > "$TMP/$ALERT.json" <<JSON
{
  "displayName": "$POLICY_PREFIX: $ALERT alert raised",
  "combiner": "OR",
  "severity": "WARNING",
  "conditions": [{
    "displayName": "$ALERT alert transition logged",
    "conditionThreshold": {
      "filter": "metric.type=\"logging.googleapis.com/user/$METRIC\" AND resource.type=\"cloud_run_revision\"",
      "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_SUM", "crossSeriesReducer": "REDUCE_SUM"}],
      "comparison": "COMPARISON_GT",
      "thresholdValue": 0,
      "duration": "0s",
      "trigger": {"count": 1}
    }
  }],
  "alertStrategy": {"autoClose": "1800s"},
  "notificationChannels": $CHANNELS,
  "documentation": {"mimeType": "text/markdown",
    "content": "Service rule: $RULE. Check GET /monitoring on $SERVICE. Synthetic portfolio service; no real-user traffic."}
}
JSON
done

# Log-based metric descriptors can take a minute to appear before policies can reference them.
for ALERT in score_drift request_failure_rate latency; do
  for attempt in 1 2 3 4 5 6 7; do
    if create_policy "$TMP/$ALERT.json"; then break; fi
    [[ $attempt == 7 ]] && { echo "Could not create policy for $ALERT" >&2; exit 1; }
    echo "Waiting for metric descriptor (attempt $attempt)..."; sleep 20
  done
done

# --- 4: Cloud Run p95 request latency --------------------------------------------------
cat > "$TMP/run_latency.json" <<JSON
{
  "displayName": "$POLICY_PREFIX: Cloud Run p95 latency > 2s",
  "combiner": "OR",
  "severity": "WARNING",
  "conditions": [{
    "displayName": "p95 request latency above 2,000 ms for 5 min",
    "conditionThreshold": {
      "filter": "metric.type=\"run.googleapis.com/request_latencies\" AND $RUN_FILTER_JSON",
      "aggregations": [{"alignmentPeriod": "300s", "perSeriesAligner": "ALIGN_PERCENTILE_95", "crossSeriesReducer": "REDUCE_MAX"}],
      "comparison": "COMPARISON_GT",
      "thresholdValue": 2000,
      "duration": "300s",
      "trigger": {"count": 1}
    }
  }],
  "alertStrategy": {"autoClose": "1800s"},
  "notificationChannels": $CHANNELS,
  "documentation": {"mimeType": "text/markdown",
    "content": "Cloud Run end-to-end latency, including cold starts. The service's own processing-latency rule is the separate latency alert."}
}
JSON
create_policy "$TMP/run_latency.json"

# --- 5: HTTP 5xx count -------------------------------------------------------------------
cat > "$TMP/run_5xx.json" <<JSON
{
  "displayName": "$POLICY_PREFIX: 5xx responses",
  "combiner": "OR",
  "severity": "ERROR",
  "conditions": [{
    "displayName": "More than 2 5xx responses in 5 min",
    "conditionThreshold": {
      "filter": "metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class=\"5xx\" AND $RUN_FILTER_JSON",
      "aggregations": [{"alignmentPeriod": "300s", "perSeriesAligner": "ALIGN_SUM", "crossSeriesReducer": "REDUCE_SUM"}],
      "comparison": "COMPARISON_GT",
      "thresholdValue": 2,
      "duration": "0s",
      "trigger": {"count": 1}
    }
  }],
  "alertStrategy": {"autoClose": "1800s"},
  "notificationChannels": $CHANNELS,
  "documentation": {"mimeType": "text/markdown", "content": "Internal scoring failures or platform errors on $SERVICE."}
}
JSON
create_policy "$TMP/run_5xx.json"

# --- 6: optional authenticated uptime check -------------------------------------------
if [[ "$UPTIME_CHECK" == "1" ]]; then
  URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')"
  HOST="${URL#https://}"
  NUMBER="$(gcloud projects describe "$PROJECT_ID" --format 'value(projectNumber)')"
  # Lets the Monitoring service agent call the IAM-protected service with an OIDC token.
  gcloud run services add-iam-policy-binding "$SERVICE" --region "$REGION" \
    --member="serviceAccount:service-$NUMBER@gcp-sa-monitoring-notification.iam.gserviceaccount.com" \
    --role="roles/run.invoker" >/dev/null
  CHECK="$(gcloud monitoring uptime list-configs --filter="displayName=\"$POLICY_PREFIX health\"" --format='value(name)')"
  if [[ -z "$CHECK" ]]; then
    CHECK="$(gcloud monitoring uptime create "$POLICY_PREFIX health" --resource-type=uptime-url \
      --resource-labels="host=$HOST,project_id=$PROJECT_ID" --protocol=https --path=/health \
      --period=5 --service-agent-auth=oidc-token --format='value(name)')"
  fi
  CHECK_ID="${CHECK##*/}"
  cat > "$TMP/uptime.json" <<JSON
{
  "displayName": "$POLICY_PREFIX: /health uptime failing",
  "combiner": "OR",
  "severity": "ERROR",
  "conditions": [{
    "displayName": "Uptime check failing from more than 1 region",
    "conditionThreshold": {
      "filter": "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" AND resource.type=\"uptime_url\" AND metric.labels.check_id=\"$CHECK_ID\"",
      "aggregations": [{"alignmentPeriod": "600s", "perSeriesAligner": "ALIGN_NEXT_OLDER", "crossSeriesReducer": "REDUCE_COUNT_FALSE", "groupByFields": ["resource.label.*"]}],
      "comparison": "COMPARISON_GT",
      "thresholdValue": 1,
      "duration": "600s",
      "trigger": {"count": 1}
    }
  }],
  "notificationChannels": $CHANNELS
}
JSON
  create_policy "$TMP/uptime.json"
fi

echo "Done. Policies: https://console.cloud.google.com/monitoring/alerting?project=$PROJECT_ID"
