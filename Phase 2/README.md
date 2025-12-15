## Phase 2 Overview

This phase introduces the **first working model** for the app.  
The goal is to establish a **baseline regression model** that can predict house prices from structured tabular data. This sets the benchmark for later improvements.

---

## Objectives

- Select and prepare the dataset (house prices).
- Perform basic preprocessing:
- Handle missing values.
- Encode categorical features.
- Scale numerical features if needed.
- Train a **baseline regression model** (Linear Regression).
- Save the trained model using `joblib` for later integration into the Streamlit app.
- Document model performance with simple metrics (MAE, RMSE, R²).

---

## Tech Stack

- Python
- Scikit-learn
- Pandas / NumPy
- Joblib

---

## Phase 2 Deliverables

- Dataset loaded and cleaned.
- Preprocessing pipeline implemented.
- Baseline regression model trained.
- Model saved for reuse.
- Performance metrics recorded.

---

## Next Steps

Phase 3 will focus on **Streamlit app development**, integrating the baseline model into a user interface and enabling predictions through a web form.
