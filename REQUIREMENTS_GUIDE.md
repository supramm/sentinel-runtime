# Quick Reference: Which Requirements File to Use?

## TL;DR

| Scenario | File | Command |
|----------|------|---------|
| **Deploy to Streamlit Cloud** | `requirements.txt` | `pip install -r requirements.txt` |
| **Deploy locally (Windows/Linux/macOS)** | `requirements.txt` | `pip install -r requirements.txt` |
| **Local development (EDA, training, notebooks)** | `requirements-dev.txt` | `pip install -r requirements-dev.txt` |
| **Docker container (production)** | `requirements.txt` | Use in Dockerfile |

---

## Files Overview

### 📋 `requirements.txt` (5 packages, ~250 MB)

**Use this for:** ✅ Production deployment, Streamlit Cloud, Docker

**What's included:**
- streamlit (UI framework)
- numpy (numerical ops)
- torch (deep learning inference)
- joblib (model loading)
- plotly (visualizations)

**What's NOT included:**
- Jupyter (notebooks)
- pandas (data manipulation)
- sklearn (training)
- matplotlib/seaborn (static plots)
- Training datasets

**Installation:**
```bash
pip install -r requirements.txt
```

**For Streamlit Cloud (CPU-only torch):**
```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall
```

---

### 📋 `requirements-dev.txt` (25+ packages, ~1-2 GB)

**Use this for:** ✅ Local development, notebook exploration, model training

**What's included:**
- ✓ All of requirements.txt (via `-r requirements.txt`)
- ✓ Jupyter + ipykernel (notebooks)
- ✓ pandas + scipy (data manipulation)
- ✓ scikit-learn (training)
- ✓ matplotlib + seaborn (plotting)
- ✓ datasets + ucimlrepo (data loading)
- ✓ pytest, black, flake8, mypy (code quality)

**Installation:**
```bash
pip install -r requirements-dev.txt
```

**What can you do:**
- Run Jupyter notebooks
- Train/retrain models
- Explore data with pandas
- Lint and test code
- Download datasets

---

## Comparison

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ Criterion           │ requirements.txt     │ requirements-dev.txt │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Size                │ ~250 MB              │ ~1-2 GB              │
│ Install time        │ 2-5 min              │ 10-20 min            │
│ Jupyter support     │ ❌ No                │ ✅ Yes               │
│ Training tools      │ ❌ No                │ ✅ Yes               │
│ Data exploration    │ ❌ No                │ ✅ Yes               │
│ Streamlit apps      │ ✅ Yes               │ ✅ Yes               │
│ Code quality tools  │ ❌ No                │ ✅ Yes               │
│ Deployment ready    │ ✅ Yes               │ ⚠️  No (too large)   │
└─────────────────────┴──────────────────────┴──────────────────────┘
```

---

## Scenario-Based Guide

### Scenario 1: Deploy to Streamlit Cloud

**Goal:** Get if_app.py, lstm_app.py, rul_app.py running on streamlit.io

**Steps:**
1. Create `streamlit_app.py` at project root
2. Specify `requirements.txt` in Streamlit Cloud settings
3. Deploy

**Or using Streamlit CLI:**
```bash
streamlit deploy --logger.level=debug
```

**Note:** If torch download fails, Streamlit Cloud will use CPU wheels automatically.

---

### Scenario 2: Deploy Locally (Development Machine)

**Goal:** Run apps locally to test before cloud deployment

**Steps:**
```bash
cd Runtime_Clean
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run apps/if_app.py
```

**Open browser:** http://localhost:8501

---

### Scenario 3: Local Development (Notebooks + Training)

**Goal:** Explore data, train models, run experiments

**Steps:**
```bash
cd SentinalAI
python -m venv venv-dev
source venv-dev/bin/activate

pip install -r Runtime_Clean/requirements-dev.txt

# Start Jupyter
jupyter notebook

# In notebook:
# - Run predictive_maintenance/notebooks/*.ipynb
# - Explore data with pandas
# - Train models with sklearn/torch
# - Generate plots with matplotlib/seaborn
```

---

### Scenario 4: Docker Container (Production)

**Goal:** Deploy to production using Docker

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY Runtime_Clean /app

RUN pip install -r requirements.txt && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8501

CMD ["streamlit", "run", "apps/if_app.py"]
```

**Build & Run:**
```bash
docker build -t sentinelai-runtime .
docker run -p 8501:8501 sentinelai-runtime
```

---

### Scenario 5: Linux Server (Production)

**Goal:** Deploy to Linux server without Docker

**Prerequisites:** Python 3.11+

**Steps:**
```bash
# Install system dependencies (optional)
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv

# Create virtual environment
python3.11 -m venv /opt/sentinelai/venv
source /opt/sentinelai/venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Optional: Use CPU torch
pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall

# Run with systemd or supervisor for auto-restart
```

**systemd service example:**
```ini
[Unit]
Description=SentinelAI Streamlit App
After=network.target

[Service]
Type=simple
User=streamlit
WorkingDirectory=/opt/sentinelai
ExecStart=/opt/sentinelai/venv/bin/streamlit run /opt/sentinelai/apps/if_app.py --server.headless true
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Troubleshooting

### Issue: "torch wheel not found"

**Symptom:** Installation fails on Windows

**Solution:**
```bash
# Use pre-compiled CPU wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

### Issue: "numpy has no attribute..."

**Symptom:** Runtime error after updating numpy

**Cause:** numpy 2.0 breaking changes

**Solution:**
```bash
pip install "numpy<2.0"
pip install -r requirements.txt
```

---

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Symptom:** Streamlit app crashes

**Cause:** torch not installed or wrong environment

**Solution:**
```bash
# Check environment
which python
pip list | grep torch

# Reinstall if needed
pip install -r requirements.txt --force-reinstall
```

---

### Issue: App slow on Streamlit Cloud

**Symptom:** Inference takes 30+ seconds

**Cause:** CPU-only torch on cloud server

**Solution:** Normal behavior. Acceptable for interactive demos.

---

## Package Details

### streamlit
- **Purpose:** Web framework
- **Minimum version:** 1.28.0
- **Current version:** 1.35.0+
- **Size:** ~15 MB
- **Installation time:** 1-2 min

### numpy
- **Purpose:** Array operations
- **Constraint:** Must be <2.0 for torch compatibility
- **Size:** ~30 MB
- **Installation time:** <1 min

### torch
- **Purpose:** Deep learning inference
- **For Streamlit Cloud:** Must use CPU wheels
- **Size:** 150-200 MB
- **Installation time:** 2-3 min

### joblib
- **Purpose:** Load sklearn models
- **Size:** ~1 MB
- **Installation time:** <1 min

### plotly
- **Purpose:** Interactive visualizations
- **Size:** ~10 MB
- **Installation time:** <1 min

---

## Advanced: Custom Installation

### Production: Minimal Size (only essential)

```bash
# Install only what's needed
pip install streamlit numpy torch joblib plotly
```

---

### Development: Full Stack

```bash
pip install -r requirements-dev.txt
```

---

### Testing: Different Python Versions

```bash
# Test on Python 3.10
python3.10 -m venv test-3.10
source test-3.10/bin/activate
pip install -r requirements.txt
streamlit run apps/if_app.py --logger.level=debug
```

---

## Environment Variables

**None required** for Runtime_Clean.

All configuration is:
- Embedded in apps (st.set_page_config, etc.)
- Resolved via PathManager (no hardcoded paths)
- Loaded from model files (no env dependencies)

---

## FAQ

**Q: Can I use Anaconda instead of pip/venv?**  
A: Yes. `conda install streamlit numpy torch joblib plotly` works too.

**Q: Do I need GPU for this?**  
A: No. All packages work CPU-only for inference.

**Q: Will models trained on GPU work on CPU?**  
A: Yes, with `map_location="cpu"` in torch.load() (already in code).

**Q: How do I downgrade torch if there are issues?**  
A: `pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu`

**Q: Can I deploy to Heroku?**  
A: Yes, Heroku supports Streamlit. Use `requirements.txt` in Procfile.

**Q: Do I need to update requirements regularly?**  
A: Only for security patches. New major versions (numpy 2.0, torch 3.0) may need testing.

---

## Summary

- **For production:** Use `requirements.txt` (5 packages, 250 MB)
- **For development:** Use `requirements-dev.txt` (25+ packages, 1-2 GB)
- **For Streamlit Cloud:** Use `requirements.txt` + CPU torch wheels
- **For local testing:** Use `requirements.txt` (same as production)
- **For Docker:** Use `requirements.txt` in Dockerfile

**That's it!** 🎉

Generated: May 11, 2026
