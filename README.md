# TensorFlow Data Validation + Wide & Deep Keras Model (Churn Lab)

> **What this lab demonstrates:** a realistic, end-to-end workflow for **tabular ML**:
> **(1)** validate data with **TFDV**, **(2)** build a robust input pipeline, **(3)** train a **Wide & Deep** model in **Keras**, and **(4)** export artifacts for reproducibility.

---

## 📦 What’s inside (deliverables)

### Data
- `data/telco_churn_synthetic.csv` — full dataset (synthetic, generated locally)
- `data/train.csv`, `data/val.csv`, `data/test.csv` — splits
- `data/test_anomalous.csv` — intentionally corrupted set to prove TFDV anomaly detection

### Code
- `src/data_validation.py` — CSV→TFRecord + TFDV stats/schema/anomaly checks + drift/skew report
- `src/train_model.py` — Keras **Wide & Deep** model with preprocessing layers + evaluation + model export
- `tests/test_smoke.py` — quick “does it run?” tests (optional)

### Outputs (generated after running)
- `artifacts/tfdv/` — stats, schema, anomaly text reports
- `artifacts/model/` — saved model, metrics json, training logs

---

## ✅ Lab checklist (what I did)

### Part A — Data Validation (TFDV)
- [x] Generated dataset with **mixed feature types** (numeric + categorical)
- [x] Computed **train/test statistics**
- [x] Inferred **schema** from training statistics
- [x] Validated clean test data vs schema
- [x] Validated an intentionally **anomalous** dataset and captured anomalies
- [x] Produced a basic **drift/skew** comparison report (train vs test)

### Part B — Modeling (TensorFlow / Keras)
- [x] Built a `tf.data` pipeline reading directly from CSV
- [x] Used **Keras preprocessing layers** (Normalization + Lookups)
- [x] Trained a **Wide & Deep** model (functional API)
- [x] Tracked metrics: **AUC / Precision / Recall**
- [x] Exported a **SavedModel** for serving/reuse

---

## 🧠 Dataset details (synthetic “telco churn”)

**Target:** `churn` (0/1)

**Example features:**
- Categorical (string): `contract_type`, `payment_method`, `internet_service`, `gender`
- Categorical (int): `senior_citizen`, `partner`, `dependents`, `paperless_billing`
- Numeric: `tenure_months`, `monthly_charges`, `total_charges`

The dataset is generated so churn is more likely for patterns like:
- month-to-month contracts
- electronic check payments
- shorter tenure
- higher monthly charges
- senior citizens

---

## 🚀 How to run

### 1) Setup environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run TFDV validation
```bash
mkdir -p artifacts/tfdv
python src/data_validation.py --data_dir data --out_dir artifacts/tfdv
```

Check:
- `artifacts/tfdv/anomalies_test.txt`
- `artifacts/tfdv/anomalies_test_anomalous.txt`
- `artifacts/tfdv/schema.pbtxt`

### 3) Train the Keras model
```bash
mkdir -p artifacts/model
python src/train_model.py --data_dir data --out_dir artifacts/model --epochs 8 --batch_size 64
```

Check:
- `artifacts/model/eval_metrics.json`
- `artifacts/model/saved_model/`
- `artifacts/model/training_log.csv`

---

## 🧪 Optional: run tests
```bash
pytest -q
```

---

## 📝 Notes
- This project is **TensorFlow-first** and does not depend on external downloads.
- The anomalous dataset is intentionally broken (unexpected categories, missing values, extreme numeric values) to showcase TFDV’s reports.

---

## Author
Manoj
