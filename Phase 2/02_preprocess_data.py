from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset again (or import X, y from step 1 if modularized)
housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Shape of X:", X.shape)
print("Shape of X_scaled:", X_scaled.shape)
