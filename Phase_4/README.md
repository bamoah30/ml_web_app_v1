# ML Web App v1 — House Prices Prediction

**Phase 4: Enhanced Features & User Experience**

---

## Phase 4 Overview

Phase 4 upgrades the baseline Streamlit app into a **multi-tab experience** with advanced visualizations and improved layout.  
Instead of a single-page app, users can now navigate between **Prediction**, **Visualization**, and **About** sections. This makes the app more professional, interactive, and portfolio-ready.

---

## Objectives

- Add **tabbed navigation** in Streamlit (`Predict`, `Visualize`, `About`).
- Improve UI with **sidebar inputs** and clear layout.
- Integrate **visualizations**:
  - Residual plots (errors between predicted vs actual).
  - Feature importance chart.
  - Prediction distribution.
- Provide **downloadable results** (CSV of predictions).
- Document project roadmap in the **About tab**.

---

## Tech Stack

- Python
- Streamlit (with tabs)
- Scikit-learn
- Pandas / NumPy
- Joblib
- Matplotlib / Seaborn

---

## Project Structure (Phase 4)

```
Phase 4/
│── app.py                 # Streamlit app with tabs & charts
│── models/                # Saved model + scaler
│── data/
│   ├── california_housing.csv     # Dataset for visualizations
│   └── sample_predictions.csv     # Optional predictions file

```

---

## How to Run

1. Navigate to Phase 4 directory:
   ```powershell
   cd "Phase_4"
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```powershell
   streamlit run app.py
   ```
4. Open the local URL (usually `http://localhost:8501`) in your browser.

---

## Example Outputs

- **Predict Tab** → Predicted house value + CSV download.
- **Visualize Tab** → Feature distribution + prediction vs median income charts.
- **About Tab** → Project roadmap and documentation.

---

## Next Steps

- **Phase 5**: Integrate Kaggle House Prices dataset for richer features.
- Add **model comparison** (Linear Regression vs Random Forest vs Gradient Boosting).
- Deploy app online (Streamlit Cloud, Heroku, or Azure).
