# Deployment Guide (Google Cloud Platform)

This guide walks you through the current architecture and deployment process in **Singapore (`asia-southeast1`)**.

## 🏗️ Architecture Overview
- **Region**: `asia-southeast1` (Singapore)
- **Engine**: Google Cloud Run
- **Storage**: Google Cloud Storage (for ML models)
- **CI/CD**: Google Cloud Build

## 🚀 Infrastructure Setup

### 1. Artifact Registry
Images are stored in a regional Docker repository:
`asia-southeast1-docker.pkg.dev/[PROJECT_ID]/price-verification-repo/`

### 2. ML Models (GCS Fuse)
To keep Docker images small, models are stored in a GCS bucket and mounted at runtime.
- **Bucket**: `gs://price-verification-models`
- **Mount Path**: `/app/price_models`

If you add new models, upload them to the bucket:
```bash
gcloud storage cp -r new_models/*.pkl gs://price-verification-models/
```

### 3. Secret Manager
Sensitive API keys are stored in Secret Manager:
- **Secret Name**: `api-service-api-keys`
- **Env Var**: `API_KEYS` (mounted to `api-service`)

## 🛠️ Deployment Process

Every push to the `main` branch triggers an automated build via `cloudbuild.yaml`.

### Build Steps:
1. **Pull Cache**: Pulls previous images to speed up build times.
2. **Docker Build**: Builds `Dockerfile.api` and `Dockerfile.web`.
3. **Artifact Push**: Pushes images to Artifact Registry.
4. **Cloud Run Deploy**:
   - Both services are deployed to `asia-southeast1`.
   - **Session Affinity** is enabled for the Web Service.
   - **GCS Fuse Volume** is mounted to both services.

## 🔗 Custom Domains
Managed via Cloud Run Domain Mappings:
- **API**: `price-api.phenomroman.com` -> `ghs.googlehosted.com.`
- **Web**: `price-app.phenomroman.com` -> `ghs.googlehosted.com.`

## 🧹 Maintenance & Cleanup
To delete old revisions or images:
```bash
# Delete all but latest 5 revisions
gcloud run revisions list --service [SERVICE_NAME] --region asia-southeast1 ...
```
