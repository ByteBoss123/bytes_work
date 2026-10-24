#!/usr/bin/env bash
# Build the image with Cloud Build and deploy it to Cloud Run (IAM-authenticated).
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/config.sh

echo "Project: $PROJECT_ID  Region: $REGION  Service: $SERVICE"
gcloud config set project "$PROJECT_ID" >/dev/null

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com monitoring.googleapis.com logging.googleapis.com

if ! gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION" \
    --description="Pharmacy claims scoring images"
fi

TAG="$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$SERVICE:$TAG"
gcloud builds submit --tag "$IMAGE" .

# max-instances=1: rolling monitoring windows live in one instance's SQLite file.
# More instances would split the windows. See deploy/README.md, Limits.
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --no-allow-unauthenticated \
  --max-instances 1 \
  --concurrency 20 \
  --cpu 1 --memory 1Gi \
  --timeout 60

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')"
echo "Service URL: $URL"
echo "Health check:"
curl -fsS -H "Authorization: Bearer $(gcloud auth print-identity-token)" "$URL/health"
echo
