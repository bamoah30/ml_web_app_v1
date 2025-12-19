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
st.title(" House Price Prediction App")
st.write("Use the sidebar to input housing features and predict median house value.")


# Sidebar Inputs

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


# Visualization Section
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


# Footer

st.markdown("---")
st.caption("Phase 3 — Streamlit App with Sidebar Inputs & Charts")
