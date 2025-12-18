import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing

# ============================
# Step 1: Export California Housing dataset
# ============================
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names) # type: ignore
df["MedHouseVal"] = housing.target  # type: ignore

# Save dataset to CSV
df.to_csv("data/california_housing.csv", index=False)
print(" California Housing dataset saved to data/california_housing.csv")

# ============================
# Step 2: Generate sample predictions using Phase 2 model
# ============================
try:
    # Load model and scaler from models directory
    model = joblib.load("models/linear_regression.pkl")
    scaler = joblib.load("models/scaler.pkl")

    # Prepare features and target
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)

    # Save predictions to CSV
    df_pred = pd.DataFrame({"Actual": y, "Predicted": predictions})
    df_pred.to_csv("data/sample_predictions.csv", index=False)
    print(" Predictions saved to data/sample_predictions.csv")

except FileNotFoundError:
    print(" Model or scaler not found in 'models/' directory. Run Phase 2 first.")
