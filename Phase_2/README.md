# ML Web App v1 — House Prices Prediction

**Phase 2: Baseline Model (California Housing Dataset)**

---

## Phase 2 Overview

This phase introduces the **first working model** for the app.  
We use the **California Housing dataset** (built into Scikit-learn) to quickly build and evaluate a baseline regression model. Later phases will expand to more complex datasets (e.g., Kaggle House Prices) for richer features and portfolio appeal.

---

## Objectives

- Load and prepare the **California Housing dataset**.
- Perform basic preprocessing (scaling features).
- Train a **baseline regression model** (Linear Regression).
- Save the trained model and scaler using `joblib`.
- Evaluate performance with **MAE, RMSE, and R²**.

---

## Tech Stack

- Python
- Scikit-learn
- Pandas / NumPy
- Joblib

---

## Project Structure (Phase 2)

```
Phase 2/
│── 01_load_data.py        # Load dataset and create DataFrame
│── 02_preprocess_data.py  # Scale features
│── 03_build_model.py      # Initialize Linear Regression model
│── 04_train_evaluate.py   # Train model and evaluate performance
│── 05_save_model.py       # Save model and scaler
│── models/                # Saved artifacts (linear_regression.pkl, scaler.pkl)
```

---

## How to Run

1. Navigate into the Phase 2 directory:
   ```powershell
   cd "Phase 2"
   ```
2. Run each script step by step:
   ```powershell
   python 01_load_data.py
   python 02_preprocess_data.py
   python 03_build_model.py
   python 04_train_evaluate.py
   python 05_save_model.py
   ```
3. Check the `models/` folder for saved files:
   - `linear_regression.pkl`
   - `scaler.pkl`

---

## Example Output

When running `04_train_evaluate.py`, you should see metrics like:

```
MAE: 0.53
RMSE: 0.72
R²: 0.61
```

Values may vary slightly depending on environment and random state.

---

## Next Steps

- **Phase 3**: Build the Streamlit app and integrate the baseline model.
- **Future Expansion**: Switch to **Kaggle House Prices dataset** for real-world complexity and portfolio appeal.
