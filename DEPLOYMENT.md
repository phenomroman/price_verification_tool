
# Deployment Guide (Google Cloud Platform)

This guide walks you through setting up automated CI/CD deployment to Google Cloud Run.

## Prerequisites
- A Google Cloud Platform (GCP) Project (you have `price-verification-tool`).
- Google Cloud SDK (`gcloud` CLI) installed locally OR use Cloud Shell.
- GitHub repository connected to your GCP project.

## Initial Setup (One-Time)

### 1. Enable Required APIs
Run these commands in your local terminal or Cloud Shell:
```bash
gcloud services enable run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com
```

### 2. Create Artifact Registry Repository
Create a repository to store your Docker images:
```bash
gcloud artifacts repositories create price-verification-repo \
    --repository-format=docker \
    --location=asia-south1 \
    --description="Docker repository for Price Verification Tool"
```

### 3. Connect GitHub to Cloud Build
1.  Go to the [Cloud Build Triggers page](https://console.cloud.google.com/cloud-build/triggers).
2.  Click **Create Trigger**.
3.  **Name**: `price-verification-ci`
4.  **Region**: `asia-south1`
5.  **Source**: Select your GitHub repository.
6.  **Event**: Push to a branch (select `main` or your production branch).
7.  **Configuration**: Select **Cloud Build configuration file (yaml or json)**.
8.  **Location**: `cloudbuild.yaml` (default).
9.  Click **Create**.

### 4. Grant Permissions
Cloud Build needs permission to deploy to Cloud Run.
1.  Go to [IAM & Admin](https://console.cloud.google.com/iam-admin/iam).
2.  Find the service account ending in `@cloudbuild.gserviceaccount.com`.
3.  Edit the principal and add these roles:
    - **Cloud Run Admin**
    - **Service Account User**

## Environment Variables (First Deployment)

After your first successful build/deploy (triggered by a push), you need to configure the environment variables for your services in the Cloud Console.

### API Service (`api-service`)
1.  Go to [Cloud Run](https://console.cloud.google.com/run).
2.  Select `api-service` -> **Edit & Deploy New Revision**.
3.  **Variables**: Add `API_KEYS` (e.g., `["your-secret-key"]`).
4.  Click **Deploy**.

### Web Service (`web-service`)
1.  Go to Cloud Run -> Select `web-service` -> **Edit & Deploy New Revision**.
2.  **Variables**:
    - `API_URL`: The URL of your `api-service` (found at the top of the api-service details page).
    - `API_KEY`: One of the keys you set in the API service.
3.  Click **Deploy**.

## Continuous Deployment
Once set up, every time you push code to the `main` branch on GitHub, Cloud Build will automatically:
1.  Build new Docker images.
2.  Push them to Artifact Registry.
3.  Update the Cloud Run services with the new code.

## Troubleshooting & Optimization

### Web Service Performance (Streamlit)
If your Streamlit app disconnects frequently or loads slowly, it's likely due to missing **Session Affinity**. Streamlit relies on persistent WebSocket connections, but Cloud Run defaults to load balancing requests across multiple instances.

**Fix:** Enable session affinity for the web service.
```bash
gcloud run services update web-service --session-affinity --region asia-south1
```
This ensures a user stays connected to the same container instance during their session.

## Custom Domains & URLs
- **Web App:** [https://price-app.phenomroman.com](https://price-app.phenomroman.com)
    - Hosted via Firebase Hosting (proxy to `web-service`).
- **API:** [https://price-api.phenomroman.com](https://price-api.phenomroman.com)
    - Hosted via Cloud Run Domain Mapping (`asia-southeast1`).
    - Docs: [/docs](https://price-api.phenomroman.com/docs)
