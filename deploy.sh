#!/usr/bin/env bash
# Deploy Manual AI RAG to Google Cloud Run (lucrative.queef@gmail.com)
set -euo pipefail

ACCOUNT="${GCP_ACCOUNT:-lucrative.queef@gmail.com}"
PROJECT_ID="${GCP_PROJECT:-gen-lang-client-0149343298}"
REGION="${GCP_REGION:-asia-south1}"
SERVICE="${CLOUD_RUN_SERVICE:-manual-ai-rag}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE}:latest"

echo "==> Using account: ${ACCOUNT}"
gcloud config set account "${ACCOUNT}"
gcloud config set project "${PROJECT_ID}"

echo "==> Enable required APIs"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project="${PROJECT_ID}"

echo "==> Build & push image via Cloud Build"
gcloud builds submit --tag "${IMAGE}" --project="${PROJECT_ID}"

echo "==> Deploy Cloud Run service: ${SERVICE}"
# Secrets must already exist as env vars / Secret Manager — never bake .env into the image.
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 3 \
  --set-env-vars "OPENROUTER_LLM_MODEL=google/gemini-2.5-flash,OPENROUTER_EMBEDDING_MODEL=qwen/qwen3-embedding-8b,OPENROUTER_EMBEDDING_DIMENSIONS=1536" \
  --update-secrets "OPENROUTER_API_KEY=OPENROUTER_API_KEY:latest,CLUSTER_ENDPOINT=CLUSTER_ENDPOINT:latest,QDRANT_API_KEY=QDRANT_API_KEY:latest" \
  --project="${PROJECT_ID}"

echo "==> Service URL"
gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)'
echo "UI path:  <SERVICE_URL>/ui/"
echo "Chat API: <SERVICE_URL>/chat"
