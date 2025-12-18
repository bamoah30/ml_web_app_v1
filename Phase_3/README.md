# ML Web App v1 — House Prices Prediction

**Phase 3: Streamlit App with Sidebar & Charts**

---

## Phase 3 Overview

This phase transforms the baseline regression model into a **user-friendly web application** using Streamlit.  
Users can input housing features via a **sidebar interface**, see predictions instantly, and explore **visualizations** such as feature distributions and prediction comparisons.

---

## Objectives

- Build a **Streamlit app** for interactive predictions.
- Load the saved model (`linear_regression.pkl`) and scaler (`scaler.pkl`) from Phase 2.
- Provide **sidebar inputs** for housing features.
- Display predictions in the main panel.
- Add **charts** for better insights (feature distributions, prediction vs. input comparisons).

---

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas / NumPy
- Joblib
- Matplotlib / Seaborn

---

## Project Structure (Phase 3)

```
Phase 3/
│── app.py                 # Streamlit app
│── models/
│   ├── linear_regression.pkl   # Trained model
│   └──scaler.pkl              # Preprocessing scaler
```

---

## How to Run

1. Navigate to Phase 3 directory:
   ```powershell
   cd "Phase_3"
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

- **Sidebar inputs** for housing features.
- **Main panel** shows predicted house value.
- **Charts** display:
  - Feature distribution (bar chart).
  - Prediction vs. Median Income (scatter plot).

---

## Next Steps

- **Phase 4**: Add tabs for navigation (Predict, Visualize, About).
- Include residual plots and feature importance charts.
- Prepare for Kaggle dataset integration.
