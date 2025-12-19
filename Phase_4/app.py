import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load model and scaler
try:
    model = joblib.load("models/linear_regression.pkl")
    scaler = joblib.load("models/scaler.pkl")
except FileNotFoundError:
    st.error(" Model or scaler not found in 'models/' directory. Please run Phase 2 first.")
    st.stop()

# App Title
st.title(" House Price Prediction App — Phase 4")

# Tabs
tab1, tab2, tab3 = st.tabs(["Predict", "Visualize", "About"])

# Predict Tab
with tab1:
    st.sidebar.header("Input Features")
    MedInc = st.sidebar.number_input("Median Income (10k USD)", min_value=0.0, step=0.1)
    HouseAge = st.sidebar.number_input("House Age (years)", min_value=0.0, step=1.0)
    AveRooms = st.sidebar.number_input("Average Rooms", min_value=0.0, step=0.1)
    AveBedrms = st.sidebar.number_input("Average Bedrooms", min_value=0.0, step=0.1)
    Population = st.sidebar.number_input("Population", min_value=0.0, step=1.0)
    AveOccup = st.sidebar.number_input("Average Occupancy", min_value=0.0, step=0.1)
    Latitude = st.sidebar.number_input("Latitude", step=0.01)
    Longitude = st.sidebar.number_input("Longitude", step=0.01)

    # Prediction
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    st.subheader("Predicted Median House Value")
    st.success(f"${prediction[0]*100000:.2f}")

    # Download option
    df_pred = pd.DataFrame({"PredictedValue":[prediction[0]*100000]})
    st.download_button(" Download Prediction as CSV", df_pred.to_csv(index=False), "prediction.csv")

# Visualize Tab
with tab2:
    st.subheader(" Feature Distribution")
    df = pd.DataFrame(features, columns=["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"])
    fig, ax = plt.subplots()
    sns.barplot(data=df, orient="h", ax=ax)
    st.pyplot(fig)

    st.subheader(" Prediction vs Median Income")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["MedInc"], prediction, color="blue", label="Prediction")
    ax2.set_xlabel("Median Income")
    ax2.set_ylabel("Predicted House Value")
    ax2.legend()
    st.pyplot(fig2)

    # Residuals (if sample_predictions.csv exists)
    try:
        df_pred_sample = pd.read_csv("data/sample_predictions.csv")
        residuals = df_pred_sample["Actual"] - df_pred_sample["Predicted"]

        st.subheader(" Residual Plot (Actual - Predicted)")
        fig3, ax3 = plt.subplots()
        sns.histplot('residuals', kde=True, ax=ax3, color="red")
        ax3.set_xlabel("Residuals")
        st.pyplot(fig3)
    except FileNotFoundError:
        st.info(" No sample_predictions.csv found in 'data/'. Run data_setup.py to generate it.")

# About Tab
with tab3:
    st.write("""
    ### About This Project
    - **Phase 2**: Trained baseline Linear Regression model on California Housing dataset.
    - **Phase 3**: Built Streamlit app with sidebar inputs and charts.
    - **Phase 4**: Enhanced app with tabs, visualizations, residual plots, and download options.
    - **Next Phase (5)**: Integrate Kaggle House Prices dataset and compare multiple models.
    """)
