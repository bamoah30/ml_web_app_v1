from typing import cast
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.utils import Bunch

# Fetch dataset
result = fetch_california_housing(as_frame=True)

# Ensure correct typing
housing = cast(Bunch, result)

# Build DataFrame safely
X = housing.data
y = housing.target

df = X.copy()
df["MedHouseVal"] = y

# Split features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Shape of X:", X.shape)
print("Shape of X_scaled:", X_scaled.shape)
