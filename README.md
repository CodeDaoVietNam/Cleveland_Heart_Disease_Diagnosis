# Cleveland Heart Disease Diagnosis System

An end-to-end clinical decision support system for cardiovascular risk assessment. This project combines Machine Learning, Explainable AI (XAI), Generative AI, and Computer Vision under a production-grade Microservices architecture.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running with Docker](#running-with-docker)
- [API Reference](#api-reference)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Data Pipeline](#data-pipeline)

## Overview

The system predicts the probability of heart disease using the UCI Cleveland Heart Disease dataset (303 patients, 13 clinical features). It is designed around a **Frontend-Backend separation** pattern: the Streamlit dashboard (UI) communicates with a FastAPI server (AI Core) through REST APIs, enabling independent scaling and maintenance.

## System Architecture

```
                          +---------------------+
                          |   Streamlit (UI)    |
                          |     Port 8501       |
                          +----------+----------+
                                     |
                            HTTP / JSON Requests
                                     |
                          +----------v----------+
                          |   FastAPI (Backend)  |
                          |     Port 8000        |
                          |                      |
                          |  /predict            |
                          |  /explain  (SHAP)    |
                          |  /batch_predict      |
                          |  /importance         |
                          +----------+-----------+
                                     |
                    +----------------+----------------+
                    |                |                 |
             +------v------+  +-----v------+  +------v-------+
             | XGBoost .pkl|  | SHAP Engine|  | Gemini LLM   |
             | (Inference) |  | (XAI)      |  | (GenAI API)  |
             +-------------+  +------------+  +--------------+
```

## Key Features

| Feature | Description | Technology |
|---|---|---|
| Predictive Analytics | Binary classification of heart disease risk | XGBoost, Scikit-learn |
| Explainable AI (XAI) | SHAP Waterfall charts decomposing each prediction | SHAP, Matplotlib |
| Virtual Doctor Assistant | LLM-generated medical advice and personalized care plans | Google Gemini API |
| Medical Vision (OCR) | Auto-extract Cholesterol, Blood Pressure, FBS from uploaded lab reports | EasyOCR |
| PDF Report Export | Downloadable clinical assessment documents | ReportLab |
| Patient History Database | Persistent storage of all diagnosis sessions | SQLite |
| Batch Prediction | Upload CSV for high-throughput multi-patient evaluation | Pandas, FastAPI |
| Feature Importance Dashboard | Visual breakdown of model decision weights | XGBoost, Plotly |

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly, CSS (Inter font) |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Machine Learning | XGBoost, Scikit-learn, SHAP |
| Generative AI | Google Gemini (`google-genai`) |
| Computer Vision | EasyOCR |
| Database | SQLite3 |
| PDF Generation | ReportLab |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions (Flake8 + Pytest) |
| Logging | Python `logging` with RotatingFileHandler |

## Project Structure

```
.
├── api/
│   └── main.py                  # FastAPI backend server (endpoints: /predict, /explain, /batch_predict, /importance)
├── app.py                       # Streamlit frontend dashboard (communicates with backend via HTTP)
├── pages/
│   ├── 2_Tong_Quan_Mo_Hinh.py   # Model performance metrics and feature importance visualization
│   ├── 3_Phan_Tich_Du_Lieu.py   # Exploratory Data Analysis (EDA) dashboard
│   ├── 4_Nhap_Lieu_Hang_Loat.py # Batch prediction interface (CSV upload)
│   └── 5_Lich_Su_Kham_Benh.py   # Patient diagnosis history viewer
├── src/
│   ├── _Prepare_Data.py         # Data cleaning, feature engineering, train/test splitting
│   ├── _XGBoost_Diagnosis.py    # XGBoost model training and hyperparameter tuning
│   ├── _AdaBoost_Diagnosis.py   # AdaBoost model training
│   ├── _GradientBoosting_Diagnosis.py
│   └── _RandomForest_Diagnosis.py
├── utils/
│   ├── db.py                    # SQLite database initialization and CRUD operations
│   ├── llm_assistant.py         # Gemini LLM prompt engineering and response handlers
│   ├── ocr_helper.py            # EasyOCR medical report text extraction
│   ├── pdf_gen.py               # ReportLab PDF document generation
│   └── logger.py                # Centralized logging with file rotation
├── models/                      # Serialized ML models (.pkl)
├── data/
│   └── cleveland.csv            # UCI Cleveland Heart Disease dataset
├── splits/                      # Train/validation/test CSV splits (raw + feature-engineered)
├── figures/                     # Model evaluation charts (feature importance PDFs)
├── tests/
│   └── test_api.py              # Pytest: API health, prediction validation, schema enforcement
├── logs/                        # Runtime log files (auto-generated, rotated at 5MB)
├── Dockerfile.api               # Container image for FastAPI backend
├── Dockerfile.ui                # Container image for Streamlit frontend
├── docker-compose.yml           # Multi-container orchestration
├── .github/workflows/main.yml   # GitHub Actions CI/CD pipeline
├── requirements.txt             # Python dependency manifest
├── .env                         # Environment variables (GEMINI_API_KEY)
└── .gitignore                   # Git exclusion rules
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Docker and Docker Compose

### Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/Cleveland_Heart_Disease_Diagnosis.git
cd Cleveland_Heart_Disease_Diagnosis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
# Create a .env file in the project root
echo 'GEMINI_API_KEY="your_google_gemini_api_key"' > .env
```
You can obtain a free API key from [Google AI Studio](https://aistudio.google.com/).

### Running the Application

The system requires **two terminals** running simultaneously:

**Terminal 1 -- Start the Backend API:**
```bash
uvicorn api.main:app --port 8000 --reload
```

**Terminal 2 -- Start the Frontend UI:**
```bash
streamlit run app.py
```

Access the application at `http://localhost:8501`. The backend API documentation is available at `http://localhost:8000/docs`.

## Running with Docker

Build and start all services with a single command:

```bash
docker-compose up --build
```

This will start:
- Backend API on port `8000`
- Frontend UI on port `8501`

To stop all services:
```bash
docker-compose down
```

## API Reference

All endpoints are served by FastAPI at `http://localhost:8000`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check (returns server status and model load state) |
| `POST` | `/predict` | Single patient prediction (returns prediction + probability) |
| `POST` | `/explain` | SHAP Waterfall plot as PNG image for a single patient |
| `POST` | `/batch_predict` | Batch prediction for a list of patients (JSON array) |
| `GET` | `/importance` | Feature importance chart as PNG image |

Interactive API documentation is auto-generated at `/docs` (Swagger UI).

## Testing

Run the automated test suite:

```bash
pytest tests/ -v
```

The test suite covers:
- API server health verification
- Prediction endpoint with valid clinical input
- Schema validation enforcement (missing fields return HTTP 422)

## CI/CD Pipeline

Every `push` or `pull_request` to the `main` branch triggers a GitHub Actions workflow that:

1. Sets up Python 3.11 on a clean Ubuntu runner
2. Installs all project dependencies from `requirements.txt`
3. Runs **Flake8** linter to catch syntax errors and code complexity issues
4. Runs **Pytest** to validate API behavior and data contracts

The pipeline configuration is defined in `.github/workflows/main.yml`.

## Data Pipeline

The ML workflow follows a structured sequence:

1. **Raw Data** (`data/cleveland.csv`) -- 303 patients, 13 clinical attributes
2. **Preprocessing** (`src/_Prepare_Data.py`) -- Missing value handling, One-Hot Encoding, feature engineering (`hr_ratio`), train/val/test splitting
3. **Model Training** (`src/_XGBoost_Diagnosis.py`) -- Hyperparameter tuning, cross-validation (K-Fold = 5), model serialization to `.pkl`
4. **Inference** (`api/main.py`) -- Deserialization, prediction, probability estimation
5. **Explainability** (`api/main.py /explain`) -- SHAP TreeExplainer decomposition
6. **Reporting** (`utils/pdf_gen.py`) -- Automated PDF clinical report generation
