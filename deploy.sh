#!/usr/bin/env bash
# Deploy Manual AI RAG to Google Cloud Run (lucrative.queef@gmail.com)
set -euo pipefail

ACCOUNT="${GCP_ACCOUNT:-lucrative.queef@gmail.com}"
PROJECT_ID="${GCP_PROJECT:-project-0ff01dde-a087-4c96-b04}"
# Match Qdrant Cloud Frankfurt (eu-central-1) → GCP europe-west3
REGION="${GCP_REGION:-europe-west3}"
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
  secretmanager.googleapis.com \
  --project="${PROJECT_ID}"

REPO="cloud-run-source-deploy"
echo "==> Ensure Artifact Registry repo exists: ${REPO} (${REGION})"
if ! gcloud artifacts repositories describe "${REPO}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Cloud Run images for Manual AI RAG" \
    --project="${PROJECT_ID}"
fi

# Allow Cloud Build SA to push images
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
CB_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
gcloud artifacts repositories add-iam-policy-binding "${REPO}" \
  --location="${REGION}" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/artifactregistry.writer" \
  --project="${PROJECT_ID}" \
  >/dev/null || true

echo "==> Build & push image via Cloud Build"
gcloud builds submit --tag "${IMAGE}" --project="${PROJECT_ID}"

SERVICE_URL_HINT="https://${SERVICE}-nszha5soya-ey.a.run.app"
# Commas in CORS_ORIGINS break --set-env-vars; use a YAML env file instead.
ENV_FILE="$(mktemp /tmp/manual-ai-env.XXXXXX.yaml)"
cat > "${ENV_FILE}" <<EOF
OPENROUTER_LLM_MODEL: "google/gemini-2.5-flash"
OPENROUTER_EMBEDDING_MODEL: "qwen/qwen3-embedding-8b"
OPENROUTER_EMBEDDING_DIMENSIONS: "1536"
CORS_ORIGINS: "http://127.0.0.1:8000,http://localhost:8000,${SERVICE_URL_HINT}"
EOF
trap 'rm -f "${ENV_FILE}"' EXIT

echo "==> Deploy Cloud Run service: ${SERVICE} (no traffic yet)"
# Secrets must already exist in Secret Manager — never bake .env into the image.
# 8Gi + concurrency 4: retest still saw miss-path 503s at 4 concurrent under 4Gi/8.
# Keep one warm instance so cold-start + model load does not eat the request budget.
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 2 \
  --concurrency 4 \
  --cpu-boost \
  --timeout 300 \
  --max-instances 5 \
  --min-instances "${MIN_INSTANCES:-1}" \
  --startup-probe="httpGet.path=/ready,initialDelaySeconds=10,timeoutSeconds=10,periodSeconds=10,failureThreshold=18" \
  --env-vars-file "${ENV_FILE}" \
  --update-secrets "OPENROUTER_API_KEY=OPENROUTER_API_KEY:latest,CLUSTER_ENDPOINT=CLUSTER_ENDPOINT:latest,QDRANT_API_KEY=QDRANT_API_KEY:latest" \
  --no-traffic \
  --tag=candidate \
  --project="${PROJECT_ID}"

# Prefer the tagged candidate URL (never fall back to the live URL — that
# would eval the old revision). Cloud Run prints it as:
#   https://candidate---SERVICE-xxxxx.REGION.run.app
CANDIDATE_URL="$(gcloud run services describe "${SERVICE}" \
  --region "${REGION}" \
  --project="${PROJECT_ID}" \
  --format='yaml(status.traffic)' 2>/dev/null \
  | awk '/tag: candidate/{f=1} f && /url:/{print $2; exit}')"
if [[ -z "${CANDIDATE_URL}" ]]; then
  LIVE_URL="$(gcloud run services describe "${SERVICE}" \
    --region "${REGION}" \
    --project="${PROJECT_ID}" \
    --format='value(status.url)')"
  # https://SERVICE-xxx.a.run.app → https://candidate---SERVICE-xxx.a.run.app
  CANDIDATE_URL="${LIVE_URL/https:\/\//https:\/\/candidate---}"
fi
if [[ -z "${CANDIDATE_URL}" || "${CANDIDATE_URL}" != *candidate* ]]; then
  echo "ERROR: could not resolve candidate URL (refusing to eval live)." >&2
  exit 1
fi
echo "==> Candidate URL: ${CANDIDATE_URL}"

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  echo "==> Running injection eval gate against candidate"
  if .venv/bin/python scripts/run_eval.py --base "${CANDIDATE_URL}" --skip-gold; then
    echo "==> Eval passed — shifting 100% traffic to new revision"
    gcloud run services update-traffic "${SERVICE}" \
      --region "${REGION}" \
      --to-latest \
      --project="${PROJECT_ID}"
  else
    echo "==> Eval FAILED — traffic NOT shifted. Inspect candidate tag and fix."
    exit 1
  fi
else
  echo "==> SKIP_EVAL=1 — shifting traffic without eval"
  gcloud run services update-traffic "${SERVICE}" \
    --region "${REGION}" \
    --to-latest \
    --project="${PROJECT_ID}"
fi

echo "==> Service URL"
gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)'
echo "UI path:  <SERVICE_URL>/ui/"
echo "Chat API: <SERVICE_URL>/chat"
echo "Tip: set a monthly credit limit on your OpenRouter key in the OpenRouter dashboard."
