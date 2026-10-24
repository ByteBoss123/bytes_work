#!/usr/bin/env bash
# Remove everything deploy.sh and alerts.sh created (the image repository is kept
# unless DELETE_IMAGES=1).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source deploy/config.sh
gcloud config set project "$PROJECT_ID" >/dev/null

for name in $(gcloud monitoring policies list --filter="displayName:\"$POLICY_PREFIX\"" --format='value(name)'); do
  gcloud monitoring policies delete "$name" --quiet && echo "Deleted policy $name"
done
for name in $(gcloud monitoring uptime list-configs --filter="displayName=\"$POLICY_PREFIX health\"" --format='value(name)'); do
  gcloud monitoring uptime delete "$name" --quiet && echo "Deleted uptime check $name"
done
for metric in pbm_score_drift_alert pbm_request_failure_rate_alert pbm_latency_alert; do
  gcloud logging metrics delete "$metric" --quiet 2>/dev/null && echo "Deleted log metric $metric"
done
gcloud run services delete "$SERVICE" --region "$REGION" --quiet && echo "Deleted service $SERVICE"
if [[ "${DELETE_IMAGES:-0}" == "1" ]]; then
  gcloud artifacts repositories delete "$REPO" --location "$REGION" --quiet && echo "Deleted repository $REPO"
fi
echo "Email notification channels are left in place; remove them in the console if unused."
