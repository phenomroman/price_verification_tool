# Price Verification Tool 🏷️

An AI-powered tool to assess potential under/over-invoicing based on historical import data. It provides automated price verification for various goods using machine learning models.

## 🚀 Live Access
- **Web Interface**: [https://price-app.phenomroman.com](https://price-app.phenomroman.com)
- **API Documentation**: [https://price-api.phenomroman.com/docs](https://price-api.phenomroman.com/docs)

## 🏗️ Architecture
The project consist of two primary services co-located in **Singapore (`asia-southeast1`)**:
1.  **API Service**: A FastAPI backend that handles model inference and batch processing.
2.  **Web Service**: A Streamlit frontend that provides an interactive calculator and domain-specific visualizations.

### Key Features
- **Scalable Hosting**: Deployed on Google Cloud Run with direct domain mapping.
- **External ML Models**: Heavy model files (.pkl) are externalized to Google Cloud Storage and mounted via **GCS Fuse** to keep Docker images lightweight.
- **CI/CD**: Fully automated deployment pipeline using Google Cloud Build.
- **Security**: API key validation via Cloud Secret Manager.

## 🛠️ Local Development

### Prerequisites
- Python 3.10+
- The `price_models/` folder (Download from GCS bucket if not present).

### Running Locally
1. **Clone the repo.**
2. **Install dependencies**:
   ```bash
   pip install -r requirements-core.txt
   pip install -r api/requirements.txt
   pip install -r web/requirements.txt
   ```
3. **Run the API**:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
4. **Run the Web App**:
   ```bash
   streamlit run web/app.py
   ```

## 📄 License & Privacy
This project is governed by its [Privacy Policy](PRIVACY_POLICY.md), [Terms of Service](TERMS_OF_SERVICE.md), and [Security Policy](SECURITY.md).
Licensed under the [MIT License](LICENSE).