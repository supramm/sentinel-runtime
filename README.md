
<div align="center">

# SentinelAI

### Industrial Predictive Maintenance Intelligence Platform

A deployed industrial AI system for anomaly detection, temporal fault analysis, and Remaining Useful Life prediction.

<br>

<p align="center">

<a href="https://sentinel-frontend-delta.vercel.app/">
  <img src="https://img.shields.io/badge/Live%20Deployment-SentinelAI%20Frontend-00C2FF?style=for-the-badge&logo=vercel&logoColor=white" alt="Live Deployment">
</a>

<a href="https://react.dev/">
  <img src="https://img.shields.io/badge/React-Frontend-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React">
</a>

<a href="https://vitejs.dev/">
  <img src="https://img.shields.io/badge/Vite-Build%20System-646CFF?style=for-the-badge&logo=vite&logoColor=FFD62E" alt="Vite">
</a>

<a href="https://www.typescriptlang.org/">
  <img src="https://img.shields.io/badge/TypeScript-App-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript">
</a>

<a href="https://streamlit.io/">
  <img src="https://img.shields.io/badge/Streamlit-Runtime-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
</a>

<a href="https://pytorch.org/">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
</a>

<a href="https://scikit-learn.org/">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit Learn">
</a>

</p>

<br>

## [Launch SentinelAI Platform](https://sentinel-frontend-delta.vercel.app/)

Production deployment featuring:
Isolation Forest Anomaly Detection • LSTM Autoencoder Diagnostics • Remaining Useful Life Prediction

</div>

---

## Deployment Preview

<p align="center">
  <a href="https://sentinel-frontend-delta.vercel.app/">
    <img width="1920" height="1080" alt="SentinelAI Frontend Preview" src="https://github.com/user-attachments/assets/c8b5d7c7-b35b-4bc0-a862-35fadb29e5ef" />
  </a>
</p>

<p align="center">
  <strong>Main Deployment:</strong>
  <br>
  <a href="https://sentinel-frontend-delta.vercel.app/">
    https://sentinel-frontend-delta.vercel.app/
  </a>
</p>

---

## Live Runtime Engines

The frontend integrates three independently deployed Streamlit runtime engines.

| Engine | Live App | Purpose |
|---|---|---|
| Isolation Forest Engine | [Open IF Engine](https://sentinel-if-engine.streamlit.app/) | Statistical anomaly detection |
| LSTM Autoencoder Engine | [Open LSTM Engine](https://sentinel-lstm-engine.streamlit.app/) | Temporal reconstruction anomaly detection |
| RUL Prediction Engine | [Open RUL Engine](https://sentinel-rul-engine.streamlit.app/) | Remaining Useful Life estimation |

---

## Project Overview

SentinelAI is a predictive maintenance platform built to analyze industrial telemetry and detect early signs of machine degradation.

The system combines:

- a React/Vite frontend deployed on Vercel
- three Streamlit runtime engines deployed independently
- trained ML and deep learning models
- structured runtime assets for inference
- dashboard-level visualization for operational monitoring

The frontend acts as the main control surface, while the runtime engines handle model-specific inference workflows.

---

## System Architecture

```text
                          ┌────────────────────────────┐
                          │   SentinelAI Frontend       │
                          │   React + Vite + Vercel     │
                          └──────────────┬─────────────┘
                                         │
                                         │ iframe integration
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
              ▼                          ▼                          ▼
┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
│ Isolation Forest Engine│   │ LSTM Autoencoder Engine│   │ RUL Prediction Engine  │
│ Streamlit + Scikit-Learn│  │ Streamlit + PyTorch    │   │ Streamlit + PyTorch    │
└────────────────────────┘   └────────────────────────┘   └────────────────────────┘
````

---

## Model Engines

### Isolation Forest Engine

The Isolation Forest runtime provides lightweight anomaly detection over processed telemetry features.

Key functions:

* anomaly score generation
* threshold-based detection
* feature-level fault indicators
* fast runtime inference

---

### LSTM Autoencoder Engine

The LSTM Autoencoder runtime detects temporal anomalies through reconstruction error.

Key functions:

* sequence reconstruction
* reconstruction-error scoring
* temporal degradation analysis
* feature attribution visualization

---

### RUL Prediction Engine

The RUL runtime estimates Remaining Useful Life using a deep temporal model.

Key functions:

* degradation trajectory analysis
* CNN + BiLSTM prediction
* RUL estimation
* model output comparison against true values

---

## Screenshots

### Frontend Dashboard

<p align="center">
  <img width="1920" height="1080" alt="Frontend Dashboard" src="https://github.com/user-attachments/assets/c8b5d7c7-b35b-4bc0-a862-35fadb29e5ef" />
</p>

---

### Isolation Forest Engine

<p align="center">
  <img width="1920" height="1080" alt="Isolation Forest Engine" src="https://github.com/user-attachments/assets/e5301beb-a80c-431d-bbfb-15e57b650328" />
</p>

---

### LSTM Autoencoder Engine

<p align="center">
  <img width="1920" height="1080" alt="LSTM Autoencoder Engine" src="https://github.com/user-attachments/assets/8d303d0f-c2f0-455e-a097-b1776aaa38e9" />
</p>

---

### RUL Prediction Engine

<p align="center">
  <img width="1920" height="1080" alt="RUL Prediction Engine" src="https://github.com/user-attachments/assets/00781219-1c4d-4c8c-86bf-a54d710752e0" />
</p>

---

## Repository Structure

```text
sentinel-runtime/
│
├── apps/
│   ├── if_app.py
│   ├── lstm_app.py
│   └── rul_app.py
│
├── models/
│   ├── if/
│   ├── lstm_ae/
│   └── rul/
│
├── data/
│   └── processed/
│
├── outputs/
│   ├── results.json
│   └── scenarios.json
│
├── utils/
│   └── path_manager.py
│
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Tech Stack

| Layer         | Tools                           |
| ------------- | ------------------------------- |
| Frontend      | React, Vite, TypeScript         |
| Runtime UI    | Streamlit                       |
| ML Models     | Scikit-Learn, PyTorch           |
| Data Handling | NumPy, Pandas, Joblib           |
| Visualization | Plotly, Matplotlib              |
| Deployment    | Vercel, Streamlit Cloud, GitHub |

---

## Deployment Design

SentinelAI is deployed as a modular system.

The frontend and model runtimes are separated intentionally:

```text
Frontend Repository
        │
        ▼
Vercel Deployment
        │
        ▼
User Interface + Embedded Runtime Apps


Runtime Repository
        │
        ▼
Streamlit Cloud Deployments
        │
        ├── IF Engine
        ├── LSTM AE Engine
        └── RUL Engine
```

This makes the project easier to maintain because each model runtime can be updated independently without breaking the main frontend.

---

## Local Runtime Setup

Clone the runtime repository:

```bash
git clone https://github.com/supramm/sentinel-runtime.git
cd sentinel-runtime
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run individual apps:

```bash
streamlit run apps/if_app.py
streamlit run apps/lstm_app.py
streamlit run apps/rul_app.py
```

---

## Deployment Notes

Important production decisions:

* runtime apps are deployed separately on Streamlit Cloud
* frontend is deployed separately on Vercel
* Streamlit iframe links should use embed mode inside the frontend
* model files and runtime assets are kept inside the runtime repository
* local absolute paths were removed from deployment-critical files
* broad Vercel rewrites were avoided to prevent static asset MIME errors

---

## Current Status

| Component                 | Status   |
| ------------------------- | -------- |
| Frontend deployment       | Live     |
| IF Engine                 | Live     |
| LSTM Autoencoder Engine   | Live     |
| RUL Engine                | Live     |
| GitHub runtime repository | Complete |
| Streamlit deployment      | Complete |
| Vercel deployment         | Complete |

---

## Links

| Resource            | Link                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------ |
| Main Frontend       | [sentinel-frontend-delta.vercel.app](https://sentinel-frontend-delta.vercel.app/)    |
| IF Engine           | [sentinel-if-engine.streamlit.app](https://sentinel-if-engine.streamlit.app/)        |
| LSTM Engine         | [sentinel-lstm-engine.streamlit.app](https://sentinel-lstm-engine.streamlit.app/)    |
| RUL Engine          | [sentinel-rul-engine.streamlit.app](https://sentinel-rul-engine.streamlit.app/)      |
| Runtime Repository  | [github.com/supramm/sentinel-runtime](https://github.com/supramm/sentinel-runtime)   |
| Frontend Repository | [github.com/supramm/sentinel-frontend](https://github.com/supramm/sentinel-frontend) |

---

<div align="center">

## SentinelAI

Industrial telemetry intelligence for predictive maintenance.

<br>

Built by **Supram Kumar**

</div>

