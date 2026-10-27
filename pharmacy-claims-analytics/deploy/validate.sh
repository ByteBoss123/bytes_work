#!/usr/bin/env bash
# Run the validation phases (replaying real CMS formulary rows) (+ optional soak) against the deployed service.
# Usage: deploy/validate.sh [SOAK_MINUTES]   (default 30)
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/config.sh
SOAK="${1:-30}"

[[ -d .venv ]] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
[[ -f reports/reference_pairs.json ]] || { echo 'reports/reference_pairs.json missing (produced by the CI run)'; exit 1; }

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT_ID" --format 'value(status.url)')"
.venv/bin/python tests/remote_validation.py --url "$URL" --soak-minutes "$SOAK"
echo "Results: reports/cloud_run_validation.json"
