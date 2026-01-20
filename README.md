# Fraud Detection API

A **FastAPI-based machine learning service** for transaction fraud detection. The application supports batch predictions from CSV or JSON input, background processing, task tracking, and evaluation metrics such as confusion matrices and classification reports.

This project is designed as a **simulation of a remote fraud-detection service**, similar to what a financial institution might deploy behind an API.

**Disclaimer:**
This project is still being developed, so some features might not work. I will be providing a full usage section later.


## Features

* REST API built with **FastAPI**
* Batch prediction from **CSV or JSON** inputs
* Background task execution (non-blocking requests)
* Multiple ML models supported (XGBoost, Ensemble, etc.)
* Centralized preprocessing and validation pipeline
* Task tracking with persistent JSON storage
* Fraud statistics (total records, frauds detected, legitimate entries)
* Optional evaluation metrics (confusion matrix, precision, recall, F1)
* Frontend-ready responses (HTML-friendly summaries)


## Machine Learning

The application loads **pre-trained models** (e.g. XGBoost) from disk and applies them to incoming transaction data at the moment.
I'm thinking about adding a full model training pipeline later, where the user can make their own model using a simple interface.

Supported capabilities:

* Feature validation (required columns & types)
* Consistent preprocessing across all models
* Model selection at request time
* Binary fraud classification (`is_fraud`)


## Screenshots
<img width="1072" height="675" alt="Opera Pillanatfelvétel_2026-01-21_004341_localhost" src="https://github.com/user-attachments/assets/e5d3e5b6-0a75-4c8c-8374-bd43e9d46345" />

--

<img width="1145" height="761" alt="Opera Pillanatfelvétel_2026-01-21_004353_localhost" src="https://github.com/user-attachments/assets/3753434e-463a-4b8f-9dea-24d8c2c8da97" />

--

