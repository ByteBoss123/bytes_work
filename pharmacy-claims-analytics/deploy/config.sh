# Shared settings for deploy/*.sh. Override any value with an environment variable.
# shellcheck shell=bash
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-east4}"                 # Northern Virginia
SERVICE="${SERVICE:-pbm-claims-scoring}"
REPO="${REPO:-pbm}"                          # Artifact Registry repository
NOTIFY_EMAIL="${NOTIFY_EMAIL:-}"             # optional: email for alert notifications
UPTIME_CHECK="${UPTIME_CHECK:-0}"            # 1 = also create an authenticated /health uptime check
# shellcheck disable=SC2034  # used by the scripts that source this file
POLICY_PREFIX="PBM scoring"

if [[ -z "$PROJECT_ID" ]]; then
  echo "Set PROJECT_ID or run: gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi
