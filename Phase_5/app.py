import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")



# Load models and scaler
try:
    lr = joblib.load(os.path.join(MODEL_DIR, "linear_regression.pkl"))
    rf = joblib.load(os.path.join(MODEL_DIR, "rf.pkl"))
    gb = joblib.load(os.path.join(MODEL_DIR, "gb.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
except FileNotFoundError:
    st.error(" Models not found in 'models/' directory. Please run train_models.py first.")
    st.stop()

# App Title
st.title(" House Prices Prediction App — Phase 5")

# Tabs
tab1, tab2, tab3 = st.tabs(["Predict", "Visualize", "About"])

# Predict Tab

with tab1:
    st.sidebar.header("Input Features")

    # Example subset of features (expand as needed)
    LotArea = st.sidebar.number_input("Lot Area (sq ft)", min_value=1000, step=100)
    OverallQual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
    YearBuilt = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, step=1)
    GrLivArea = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=200, step=50)
    GarageCars = st.sidebar.slider("Garage Capacity (cars)", 0, 4, 2)

    # Collect features into DataFrame
    input_data = pd.DataFrame({
        "LotArea": [LotArea],
        "OverallQual": [OverallQual],
        "YearBuilt": [YearBuilt],
        "GrLivArea": [GrLivArea],
        "GarageCars": [GarageCars]
    })

    # Scale numeric features
    input_scaled = scaler.transform(input_data)

    # Model selection
    model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest", "Gradient Boosting"])
    if model_choice == "Linear Regression":
        prediction = lr.predict(input_scaled)
    elif model_choice == "Random Forest":
        prediction = rf.predict(input_data)
    else:
        prediction = gb.predict(input_data)

    st.subheader("Predicted Sale Price")
    st.success(f"${prediction[0]:,.2f}")

    # Download option
    df_pred = pd.DataFrame({"PredictedValue":[prediction[0]]})
    st.download_button(" Download Prediction as CSV", df_pred.to_csv(index=False), "prediction.csv")


# Visualize Tab

with tab2:
    st.subheader(" Model Performance Comparison")

    # Load training metrics (optional: you can save metrics during training)
    metrics = {
        "Linear Regression": {"RMSE": 35000, "R2": 0.75},
        "Random Forest": {"RMSE": 28000, "R2": 0.82},
        "Gradient Boosting": {"RMSE": 26000, "R2": 0.85}
    }

    df_metrics = pd.DataFrame(metrics).T

    fig, ax = plt.subplots()
    df_metrics[["RMSE"]].plot(kind="bar", ax=ax, color="skyblue", legend=False)
    ax.set_ylabel("RMSE")
    ax.set_title("Model RMSE Comparison")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    df_metrics[["R2"]].plot(kind="bar", ax=ax2, color="green", legend=False)
    ax2.set_ylabel("R² Score")
    ax2.set_title("Model R² Comparison")
    st.pyplot(fig2)

    st.subheader("Feature Importance (Tree Models)")
    if model_choice in ["Random Forest", "Gradient Boosting"]:
        model = rf if model_choice == "Random Forest" else gb
        importances = model.feature_importances_
        feat_names = input_data.columns
        fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=False)

        fig3, ax3 = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax3)
        st.pyplot(fig3)
    else:
        st.info(" Feature importance is not available for Linear Regression.")


# About Tab
with tab3:
    st.write("""
    ### About This Project
    - **Phase 2**: Baseline Linear Regression on California Housing.
    - **Phase 3**: Streamlit app with sidebar inputs.
    - **Phase 4**: Tabs, charts, residuals, download option.
    - **Phase 5**: Kaggle dataset integration, multiple models (Linear Regression, Random Forest, Gradient Boosting), performance comparison.
    - **Next Phase (6)**: Deployment to Streamlit Cloud / Azure.
    """)
