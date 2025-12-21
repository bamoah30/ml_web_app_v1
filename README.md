# ML Web App — House Prices Prediction

**Journey to Fullstack AI + Robotics Engineering (Week 11 Milestone)**

---

## Project Overview

This project documents the evolution of a machine learning workflow into a full **Streamlit web application**. Starting from a baseline regression model, each phase adds new functionality, datasets, and professional‑grade features. By Phase 5, the app integrates Kaggle’s House Prices dataset and supports multiple models with performance comparison.

This project is part of my **Week 11 milestone** in my journey to becoming a **Fullstack AI + Robotics Engineer** — demonstrating how disciplined engineering practices can be applied to AI experimentation.

---

## Objectives

- Build reproducible ML workflows across multiple phases.
- Train and evaluate regression models on real datasets.
- Develop an interactive Streamlit app with user inputs, tabs, and visualizations.
- Compare multiple models (Linear Regression, Random Forest, Gradient Boosting).
- Document the project for professional presentation and portfolio use.

---

## Project Structure

```
ML_Web_App/
│── Phase_2/                 # Baseline Linear Regression (California Housing)
│── Phase_3/                 # Streamlit app with sidebar inputs
│── Phase_4/                 # Tabs, residual plots, download options
│── Phase_5/                 # Kaggle dataset + model comparison
│   ├── app.py               # Streamlit app with dropdown model selection
│   ├── train_models.py      # Training script for 5 selected features
│   ├── models/              # Saved models (.pkl files)
│   ├── data/                # Kaggle dataset (train.csv, test.csv)
│   └── README.md            # Individual Phase project documentation
│── README.md                # General project documentation
└── requirements.txt         # Dependencies
```

---

## Tech Stack

- **Python**
- **Streamlit** (web app framework)
- **Scikit‑learn** (ML models & preprocessing)
- **Pandas / NumPy** (data handling)
- **Matplotlib / Seaborn** (visualizations)
- **Joblib** (model persistence)

---

## How to Run

1. Clone the repo and navigate to the desired phase.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. For Phase 5, download Kaggle’s House Prices dataset (`train.csv`, `test.csv`) and place in `Phase_5/data/`.
4. Train models:
   ```bash
   python train_models.py
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```
6. Open the local URL (usually `http://localhost:8501`) in your browser.

---

## Features by Phase

- **Phase 2** → Baseline Linear Regression on California Housing.
- **Phase 3** → Streamlit app with sidebar inputs.
- **Phase 4** → Tabs, residual plots, download options.
- **Phase 5** → Kaggle dataset integration, multiple models, performance comparison.

---

## Next Steps

- **Future Phase**: Add hyperparameter tuning and cross‑validation.

---

## Milestone Context

This project marks **Week 11** of my journey to becoming a **Fullstack AI + Robotics Engineer**. Each phase reflects disciplined learning, reproducibility, and the integration of engineering principles into AI workflows.
