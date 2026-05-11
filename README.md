# Fraud Decision System

Decision-centric fraud detection system with adaptive alert selection, cost-aware ranking, operational monitoring, and real-time simulation support.

---

## Features

- FastAPI backend
- Streamlit dashboard
- Static threshold vs decision system comparison
- Cost-aware fraud ranking
- Operating curve evaluation
- Sequential simulation
- Analyst queue prioritization
- Monitoring metrics

---

## Tech Stack

- Python
- FastAPI
- Streamlit
- Scikit-learn
- Pandas
- NumPy

---

## Project Structure

```text
app/
decisioning/
scripts/
config/
models/
notebooks/
```

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
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

## Main Result

The decision system improved fraud recall and reduced total operational cost compared with the static threshold baseline while maintaining operationally manageable alert volumes.