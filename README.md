# price_verification_tool
Use this tool to assess potential under/over-invoicing based on historical import data.
## Overview
The web application predicts unit prices for goods based on historical import data. It attempts to call an external API for predictions, and automatically falls back to local model inference if the API is unavailable.

## Configuration

### External API URL
The web application requires an external API endpoint that implements a `/predict` endpoint compatible with the prediction model.

Set the `EXTERNAL_API_URL` environment variable to point to your external API server:

```bash
export EXTERNAL_API_URL="https://api.example.com"
```

If `EXTERNAL_API_URL` is not set, the application will try `API_URL`, and if that is also not set, it defaults to `http://localhost:8000`.

### Fallback Behavior
If the external API is unreachable or returns an error, the web application automatically falls back to the local inference engine (`core.models.inference_engine.predict`). API failures are displayed as toast warnings (non-blocking notifications) rather than blocking error messages.

## Running the Application

### Using Docker Compose
```bash
docker-compose up web
```

The web application will be available at `http://localhost:8501`.

To use an external API, add the environment variable to the docker-compose.yml:
```yaml
services:
  web:
    environment:
      - PYTHONPATH=/app
      - EXTERNAL_API_URL=https://api.example.com
```
## Deployment
The application can be deployed using Docker Compose as shown above. For production deployment to an external server, make sure to set the `EXTERNAL_API_URL` environment variable to point to your external API server.