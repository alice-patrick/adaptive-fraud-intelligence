# Adaptive Fraud Intelligence

Decision-centric fraud detection system with adaptive alert selection, cost-aware ranking, operational monitoring, and real-time simulation support.

---

## Overview

This project explores a decision-centric approach to fraud detection, focusing not only on prediction quality but also on operational cost optimization, adaptive alert prioritization, investigation-capacity constraints, and real-time monitoring.

The system combines machine learning fraud scoring with cost-aware decision logic and adaptive alert budgeting to simulate more realistic fraud operations workflows.

---

## Features

### Machine Learning
- Fraud probability scoring
- Threshold evaluation
- Model-based transaction risk estimation

### Decision Intelligence
- Adaptive alert budgeting
- Cost-aware fraud ranking
- Static threshold vs adaptive decision comparison
- Investigation-capacity-aware alert selection
- Risk-zone transaction prioritization

### Monitoring & Simulation
- Sequential fraud simulation
- Operating curve evaluation
- Monitoring metrics
- Real-time simulation support
- Analyst queue prioritization

### Interfaces
- FastAPI backend
- Streamlit dashboard

---

## Main Result

Compared to the static threshold baseline, the adaptive decision system achieved:

| System | Recall | Precision | Alerts | Total Cost |
|---|---|---|---|---|
| Static Threshold | 0.625 | 0.033 | 303 | 600,060 |
| Decision System | 0.875 | 0.038 | 372 | 552,563 |

The adaptive decision engine improved fraud recall while simultaneously reducing total operational cost under operational alert constraints.

---

## System Architecture

The project includes:

- Fraud scoring pipeline
- Decision engine
- Adaptive thresholding logic
- Cost optimization module
- Operational monitoring layer
- FastAPI backend
- Streamlit dashboard
- Sequential simulation workflows

---

## Project Structure

```text
app/
    api/
    core/
    model/
    monitoring/
    tests/

decisioning/
    cost_logic.py
    decision_engine.py
    strategies.py
    suppression.py
    thresholding.py

scripts/
    evaluate_decision_strategies.py
    run_sequential_simulation.py
    run_realtime_demo.py
    plot_operating_curve.py

config/
models/
notebooks/
```

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Streamlit
- Matplotlib
- Joblib

---

## How To Run

### Install dependencies

```bash
py -m pip install -r requirements.txt
```

### Add dataset

Place dataset here:

```text
data/raw/AIML Dataset.csv
```

### Run FastAPI backend

```bash
py -m uvicorn app.api.main:app --reload
```

### Run Streamlit dashboard

```bash
py -m streamlit run app/dashboard.py
```

---

## Dataset

This project uses the PaySim synthetic financial transaction dataset.

The dataset is not included in the repository due to size limitations.

Expected dataset path:

```text
data/raw/AIML Dataset.csv
```

---

## Future Improvements

- Probability calibration improvements
- Drift detection
- Online learning
- Analyst feedback loops
- Queue-aware investigation allocation
- Dynamic risk adaptation
- Real streaming integration

---

## Thesis Context

This repository was developed as part of a Master's thesis focused on decision-centric fraud detection, adaptive transaction prioritization, cost optimization, and monitoring for real-time capable fraud systems.
