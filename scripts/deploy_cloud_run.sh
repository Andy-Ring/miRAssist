#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-PROJECT_ID}"
REGION="${REGION:-REGION}"
SERVICE_NAME="${SERVICE_NAME:-SERVICE_NAME}"
IMAGE_NAME="${IMAGE_NAME:-IMAGE_NAME}"
IMAGE_URI="${IMAGE_URI:-gcr.io/${PROJECT_ID}/${IMAGE_NAME}}"

if [[ "${PROJECT_ID}" == "PROJECT_ID" || "${REGION}" == "REGION" || "${SERVICE_NAME}" == "SERVICE_NAME" || "${IMAGE_NAME}" == "IMAGE_NAME" ]]; then
  echo "Set PROJECT_ID, REGION, SERVICE_NAME, and IMAGE_NAME before running this script."
  exit 1
fi

gcloud builds submit --project "${PROJECT_ID}" --tag "${IMAGE_URI}" .

gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --image "${IMAGE_URI}" \
  --set-env-vars "WORKER_MODE=inline,JOBSTORE_BACKEND=postgres,EVIDENCE_BACKEND=postgres,EVIDENCE_TABLE=mirassist_evidence_variant_a_rf_v1,MIRASSIST_LEARNED_SCORE_COLUMN=mirassist_model_score,MIRASSIST_USE_LEARNED_SCORE=1"

echo "Set secrets like DATABASE_URL and MIRASSIST_VLLM_HTTP_URL separately with Cloud Run environment variables or Secret Manager."
