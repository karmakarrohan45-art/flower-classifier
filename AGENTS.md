# AGENTS.md

Commands to run this project locally.

## Prerequisites
- Python 3.11+ with pip
- Node.js 20+ with npm
- On Windows: `C:\Windows\System32` must be on PATH (for TensorFlow DLLs)

## Quick Start (model already trained)

Run the web app immediately — the trained TF.js model ships in `frontend/public/model/`:

```bash
cd frontend
npm install
npm run dev
# → open http://localhost:3000
```

## Full Pipeline (re-train + convert)

```bash
# 1. Install Python packages
pip install -r requirements.txt

# 2. Train model + evaluate + convert to TF.js (one step)
python train_model.py

# 3. Run the web app
cd frontend
npm install
npm run dev
```

## Individual Steps

```bash
# Train only (saves Keras model + evaluation artifacts to model/)
python train_model.py

# Evaluate an existing model + convert to TF.js
python evaluate_and_convert.py

# Label correction utility
python annotate.py --limit 50
```

## Linting

```bash
# Frontend lint
cd frontend
npm run lint

# Frontend type-check (via Vite build)
cd frontend
npm run build
```

## File Map

- `train_model.py` — End-to-end training pipeline
- `evaluate_and_convert.py` — Evaluate + convert an existing Keras model
- `annotate.py` — Interactive label correction tool
- `frontend/` — React + TypeScript web application
- `model/` — Trained model artifacts
- `frontend/public/model/` — TensorFlow.js model (served to the browser)
- `frontend/public/flowers.json` — Flower metadata from flowers.csv
- `flowers.csv` — Original tabular flower data (72 entries)
