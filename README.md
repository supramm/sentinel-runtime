
<div align="center">

# SentinelAI Runtime

High-performance predictive maintenance runtime infrastructure for anomaly detection and Remaining Useful Life estimation.

Built as a fully deployable ML inference stack using Streamlit, PyTorch, and Scikit-Learn.

<br>

[Isolation Forest Engine](https://sentinel-if-engine.streamlit.app/) •
[LSTM Autoencoder Engine](https://sentinel-lstm-engine.streamlit.app/) •
[RUL Prediction Engine](https://sentinel-rul-engine.streamlit.app/)

</div>

---

## Overview

SentinelAI Runtime is the production inference layer powering a distributed predictive maintenance platform designed for industrial telemetry analysis.

The system consists of three independently deployed ML engines:

| Engine | Purpose | Core Model |
|---|---|---|
| IF Engine | Fast statistical anomaly detection | Isolation Forest |
| LSTM Engine | Temporal anomaly reconstruction | LSTM Autoencoder |
| RUL Engine | Remaining Useful Life prediction | CNN + BiLSTM |

The repository is structured as a fully isolated runtime environment optimized for:
- cloud deployment
- reproducible inference
- model portability
- frontend integration
- independent service scaling

---

## Live Deployments

| Service | URL |
|---|---|
| IF Engine | https://sentinel-if-engine.streamlit.app/ |
| LSTM Engine | https://sentinel-lstm-engine.streamlit.app/ |
| RUL Engine | https://sentinel-rul-engine.streamlit.app/ |

---

## System Architecture

```text
GitHub Repository
        │
        ▼
Streamlit Cloud Runtime Layer
        │
        ├── IF Anomaly Engine
        ├── LSTM Reconstruction Engine
        └── RUL Prediction Engine
                │
                ▼
Frontend Dashboard Integration
````

---

## Runtime Structure

```text
Runtime_Clean/
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
├── outputs/
├── utils/
│
├── requirements.txt
└── requirements-dev.txt
```

---

## Core Features

### Isolation Forest Engine

* statistical anomaly scoring
* engineered telemetry analysis
* threshold-based fault detection
* lightweight inference runtime

### LSTM Autoencoder Engine

* sequence reconstruction analysis
* temporal degradation tracking
* reconstruction-error visualization
* deep anomaly representation learning

### RUL Prediction Engine

* Remaining Useful Life estimation
* CNN + BiLSTM temporal modeling
* degradation trajectory analysis
* predictive maintenance forecasting

---

## Technical Stack

| Layer         | Technologies            |
| ------------- | ----------------------- |
| Runtime       | Python, Streamlit       |
| Deep Learning | PyTorch                 |
| ML Models     | Scikit-Learn            |
| Visualization | Plotly                  |
| Serialization | Joblib, NumPy           |
| Deployment    | GitHub, Streamlit Cloud |

---

## Deployment Design

The runtime layer was intentionally separated from:

* training pipelines
* notebooks
* experimental research code
* frontend systems

to create a production-oriented deployment architecture with:

* isolated inference services
* centralized path management
* portable runtime environments
* GitHub-driven deployment workflows

---

## Local Development

### Clone Repository

```bash
git clone https://github.com/supramm/sentinel-runtime.git
cd sentinel-runtime
```

### Create Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Applications

### IF Engine

```bash
streamlit run apps/if_app.py
```

### LSTM Engine

```bash
streamlit run apps/lstm_app.py
```

### RUL Engine

```bash
streamlit run apps/rul_app.py
```

---

## Screenshots

### IF Engine

<!-- Insert screenshot -->

<br>

### LSTM Engine

<!-- Insert screenshot -->

<br>

### RUL Engine

<!-- Insert screenshot -->

---

## Deployment Notes

* fully self-contained runtime architecture
* centralized pathlib-based path management
* Streamlit Cloud compatible
* cross-platform deployment support
* Python 3.11 recommended
* GitHub-integrated redeployment workflow

---

## Future Expansion

* React/Vite monitoring dashboard
* unified telemetry visualization layer
* real-time inference streaming
* multi-machine monitoring
* cloud persistence layer
* alerting and notification pipeline
* distributed inference orchestration

---

<div align="center">

Supram Kumar
AI/ML Systems Engineering • Predictive Maintenance • Deep Learning Infrastructure

</div>
