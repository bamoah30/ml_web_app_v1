import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)  # type: ignore
df["MedHouseVal"] = housing.target  # type: ignore

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "models/linear_regression.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and scaler saved successfully!")
