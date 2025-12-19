# ML Web App v2 — House Prices Prediction

**Phase 5: Kaggle Dataset Integration & Model Comparison**

---

## Phase 5 Overview

Phase 5 expands the app by integrating the **Kaggle House Prices dataset** and introducing **multiple models** for comparison. Users can now explore richer features, compare predictions across algorithms, and visualize performance metrics.

---

## Objectives

- Import and preprocess the **Kaggle House Prices dataset** (`train.csv`, `test.csv`).
- Train and save multiple models:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Compare models using metrics (MAE, RMSE, R²).
- Enhance Streamlit app with:
  - Model selection dropdown.
  - Performance comparison charts.
  - Prediction download options.

---

## Project Structure (Phase 5)

```
Phase_5/
│── app.py                 # Streamlit app with model comparison
│── train_models.py        # Training script to build & save models
│── models/                # Saved models (linear_regression.pkl, rf.pkl, gb.pkl, scaler.pkl)
│── data/
│   ├── train.csv          # Kaggle training dataset
│   └── test.csv           # Kaggle test dataset
```

---

## Streamlit App (app.py)

### Key Features:

- **Model Selection**: Dropdown to choose between Linear Regression, Random Forest, Gradient Boosting.
- **Prediction Tab**: Sidebar inputs → prediction based on selected model.
- **Visualize Tab**:
  - Feature importance (for tree-based models).
  - Residual plots.
  - Model performance comparison chart.
- **About Tab**: Updated roadmap and dataset details.

---

## How to Run

1. Navigate to Phase 5 directory:
   ```powershell
   cd "Phase_5"
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Place Kaggle dataset files (`train.csv`, `test.csv`) in the `data/` folder.
4. Run the app:
   ```powershell
   streamlit run app.py
   ```
5. Open the local URL (usually `http://localhost:8501`) in your browser.

---

## Example Outputs

- **Predict Tab** → Predicted house value based on chosen model.
- **Visualize Tab** → Feature importance chart, residual plots, performance comparison.
- **About Tab** → Documentation of dataset, models, and roadmap.

---

## Next Steps

- **Future Phase**: Add **hyperparameter tuning** for better performance.
- Integrate **cross-validation** and ensemble methods.
