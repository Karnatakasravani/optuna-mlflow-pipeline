<<<<<<< HEAD
# 🏗️ Optuna + MLflow Hyperparameter Optimization Pipeline

This project implements an **end-to-end machine-learning experimentation workflow** using:

✔ **Optuna** — hyperparameter optimization  
✔ **XGBoost Regressor** — model  
✔ **MLflow** — experiment tracking  
✔ **SQLite** — persistent study storage  
✔ **Docker** — reproducible execution  

The model is trained on the **California Housing dataset** and evaluated using **Root Mean Squared Error (RMSE)**.

---

## 📂 Project Structure

```
.
├── src/
│   ├── data_loader.py
│   ├── objective.py
│   ├── optimize.py
│   ├── evaluate.py
│   └── __init__.py
├── notebooks/
│   └── analysis.ipynb
├── outputs/
│   ├── optuna_study.db
│   ├── optimization_history.png
│   ├── param_importance.png
│   ├── results.json
│   └── mlruns/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📦 Installation (Local)

```bash
pip install -r requirements.txt
```

(Optional but recommended: use a virtual environment)

---

## ▶️ Running the Pipeline (Local)

### 🔹 Step 1 — Run Hyperparameter Optimization

```bash
python -m src.optimize
```

This will:

✅ Load the dataset  
✅ Run Optuna study  
✅ Train XGBoost models  
✅ Log runs to MLflow  
✅ Save plots + database to `outputs/`

Generated files include:

```
outputs/optuna_study.db
outputs/optimization_history.png
outputs/param_importance.png
outputs/results.json
outputs/mlruns/
```

---

### 🔹 Step 2 — Evaluate the Best Model

```bash
python -m src.evaluate
```

This computes:

✔ Best Trial Parameters  
✔ Validation RMSE  
✔ Test RMSE  

and stores them in:

```
outputs/results.json
```

---

### 🔹 Step 3 — View Experiments in MLflow UI

```bash
mlflow ui
```

Then open:

👉 http://127.0.0.1:5000

You will see all experiment runs.

---

## 🐳 Running with Docker (Required for Task)

### Build image

```bash
docker build -t optuna-mlflow-pipeline .
```

### Run container

Windows PowerShell:

```bash
docker run -v ${PWD}/outputs:/app/outputs optuna-mlflow-pipeline
```

Mac / Linux:

```bash
docker run -v $(pwd)/outputs:/app/outputs optuna-mlflow-pipeline
```

Outputs appear on your host machine.

---

## 📊 Results Summary

### Baseline Model (Default XGBoost)

**RMSE ≈ 0.46**

### Tuned Model (Optuna Best Trial)

**RMSE ≈ 0.48**

> Note: Your exact values may differ slightly due to randomness.

---

## 📒 Notebook Analysis

Notebook:

```
notebooks/analysis.ipynb
```

Contains:

✔ Data preprocessing  
✔ Baseline vs Tuned performance  
✔ Optuna visualizations  
✔ Summary discussion  

---

## 📌 Key Insights

- Optuna automates hyperparameter tuning efficiently  
- MLflow enables complete experiment tracking  
- Docker ensures consistent execution across systems  
- RMSE provides interpretable performance comparison  

---

## 🛠️ Tech Stack

| Tool | Purpose |
|-----|--------|
| Python | Core language |
| XGBoost | Regression model |
| Optuna | Hyperparameter tuning |
| MLflow | Experiment tracking |
| SQLite | Study storage |
| Docker | Reproducibility |

---

## ✅ Requirements

All dependencies are listed in:

```
requirements.txt
```

Install using:

```bash
pip install -r requirements.txt
```

---

## ✍️ Author

Sravani Karnataka

---

## ✔️ Status

Project **successfully implemented and tested** 🎉
"# optuna-mlflow-pipeline" 
=======
# 🏗️ Optuna + MLflow Hyperparameter Optimization Pipeline

This project implements an **end-to-end machine-learning experimentation workflow** using:

✔ **Optuna** — hyperparameter optimization  
✔ **XGBoost Regressor** — model  
✔ **MLflow** — experiment tracking  
✔ **SQLite** — persistent study storage  
✔ **Docker** — reproducible execution  

The model is trained on the **California Housing dataset** and evaluated using **Root Mean Squared Error (RMSE)**.

---

## 📂 Project Structure

```
.
├── src/
│   ├── data_loader.py
│   ├── objective.py
│   ├── optimize.py
│   ├── evaluate.py
│   └── __init__.py
├── notebooks/
│   └── analysis.ipynb
├── outputs/
│   ├── optuna_study.db
│   ├── optimization_history.png
│   ├── param_importance.png
│   ├── results.json
│   └── mlruns/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📦 Installation (Local)

```bash
pip install -r requirements.txt
```

(Optional but recommended: use a virtual environment)

---

## ▶️ Running the Pipeline (Local)

### 🔹 Step 1 — Run Hyperparameter Optimization

```bash
python -m src.optimize
```

This will:

✅ Load the dataset  
✅ Run Optuna study  
✅ Train XGBoost models  
✅ Log runs to MLflow  
✅ Save plots + database to `outputs/`

Generated files include:

```
outputs/optuna_study.db
outputs/optimization_history.png
outputs/param_importance.png
outputs/results.json
outputs/mlruns/
```

---

### 🔹 Step 2 — Evaluate the Best Model

```bash
python -m src.evaluate
```

This computes:

✔ Best Trial Parameters  
✔ Validation RMSE  
✔ Test RMSE  

and stores them in:

```
outputs/results.json
```

---

### 🔹 Step 3 — View Experiments in MLflow UI

```bash
mlflow ui
```

Then open:

👉 http://127.0.0.1:5000

You will see all experiment runs.

---

## 🐳 Running with Docker (Required for Task)

### Build image

```bash
docker build -t optuna-mlflow-pipeline .
```

### Run container

Windows PowerShell:

```bash
docker run -v ${PWD}/outputs:/app/outputs optuna-mlflow-pipeline
```

Mac / Linux:

```bash
docker run -v $(pwd)/outputs:/app/outputs optuna-mlflow-pipeline
```

Outputs appear on your host machine.

---

## 📊 Results Summary

### Baseline Model (Default XGBoost)

**RMSE ≈ 0.46**

### Tuned Model (Optuna Best Trial)

**RMSE ≈ 0.48**

> Note: Your exact values may differ slightly due to randomness.

---

## 📒 Notebook Analysis

Notebook:

```
notebooks/analysis.ipynb
```

Contains:

✔ Data preprocessing  
✔ Baseline vs Tuned performance  
✔ Optuna visualizations  
✔ Summary discussion  

---

## 📌 Key Insights

- Optuna automates hyperparameter tuning efficiently  
- MLflow enables complete experiment tracking  
- Docker ensures consistent execution across systems  
- RMSE provides interpretable performance comparison  

---

## 🛠️ Tech Stack

| Tool | Purpose |
|-----|--------|
| Python | Core language |
| XGBoost | Regression model |
| Optuna | Hyperparameter tuning |
| MLflow | Experiment tracking |
| SQLite | Study storage |
| Docker | Reproducibility |

---

## ✅ Requirements

All dependencies are listed in:

```
requirements.txt
```

Install using:

```bash
pip install -r requirements.txt
```

---

## ✍️ Author

Sravani Karnataka

---

## ✔️ Status

Project **successfully implemented and tested** 🎉
"# optuna-mlflow-pipeline" 
>>>>>>> 3b90c6a49849faadace8b7b4a7abf7a3fe068746
