# 📊 Churn Prediction with TensorFlow Data Validation & Wide + Deep Learning

This project demonstrates a complete end-to-end machine learning workflow for tabular data using **TensorFlow Data Validation (TFDV)** and a **Wide & Deep neural network** in Keras.

The goal is to simulate a production-style ML pipeline that focuses not only on model accuracy, but also on **data quality, validation, and reproducibility**.

---

## 🔍 Project Overview

This lab walks through a realistic tabular ML lifecycle:

1. Validate data quality using TensorFlow Data Validation (TFDV)
2. Build a scalable input pipeline for structured data
3. Train a hybrid Wide & Deep neural network
4. Export reproducible artifacts for deployment and analysis

---

## 📦 Repository Structure

### 📁 Data
- `data/telco_churn_synthetic.csv` — complete synthetic dataset  
- `data/train.csv`, `data/val.csv`, `data/test.csv` — dataset splits  
- `data/test_anomalous.csv` — intentionally corrupted dataset for anomaly detection  

### 💻 Source Code
- `src/data_validation.py` — performs schema inference, statistics generation, and anomaly detection using TFDV  
- `src/train_model.py` — builds and trains a Wide & Deep Keras model with preprocessing layers  
- `tests/test_smoke.py` — lightweight sanity checks (optional)

### 📤 Generated Outputs
- `artifacts/tfdv/` — statistics, inferred schema, and anomaly reports  
- `artifacts/model/` — trained model, evaluation metrics, and logs  

---

## ✅ Key Highlights

### 🧪 Data Validation with TFDV
- Generated a synthetic churn dataset with mixed feature types  
- Computed dataset statistics for train/test splits  
- Automatically inferred schema from training data  
- Validated clean datasets against schema  
- Detected anomalies using a deliberately corrupted dataset  
- Compared train vs test distributions for drift analysis  

---

### 🤖 Modeling with TensorFlow
- Implemented an efficient `tf.data` input pipeline  
- Used built-in preprocessing layers for normalization and categorical encoding  
- Designed a Wide & Deep neural network using Keras Functional API  
- Evaluated performance using AUC, Precision, and Recall  
- Exported a deployable TensorFlow SavedModel  

---

## 🧠 Dataset Description

This project uses a **synthetic telco churn dataset** designed to resemble real-world customer behavior.

**Target variable:** `churn` (binary classification)

**Feature types:**
- String categorical: contract type, payment method, internet service, gender  
- Binary categorical: partner, dependents, senior citizen, paperless billing  
- Numerical: tenure months, monthly charges, total charges  

The data generation logic introduces realistic churn patterns such as:
- Higher churn for short-term contracts  
- Increased churn with higher monthly charges  
- Lower retention for customers with shorter tenure  

---

## 🚀 How to Run

### 1️⃣ Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
