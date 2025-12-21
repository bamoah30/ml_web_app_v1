import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Step 1: Load Kaggle dataset
df = pd.read_csv("data/train.csv")

# Target variable
y = df["SalePrice"]

# Restrict to chosen features
features = ["LotArea", "OverallQual", "YearBuilt", "GrLivArea", "GarageCars"]
X = df[features]


# Step 2: Handle missing values
X = X.fillna(X.median())


# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 4: Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 5: Train models
print("Training models...")

# Linear Regression (scaled)
lr = LinearRegression().fit(X_train_scaled, y_train)

# Random Forest (unscaled)
rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)

# Gradient Boosting (unscaled)
gb = GradientBoostingRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)


# Step 6: Evaluate models
def evaluate(model, X_t, y_t, name, scaled=False):
    if scaled:
        preds = model.predict(scaler.transform(X_t))
    else:
        preds = model.predict(X_t)
    rmse = np.sqrt(mean_squared_error(y_t, preds))
    r2 = r2_score(y_t, preds)
    print(f"{name} → RMSE: {rmse:.2f}, R²: {r2:.3f}")

evaluate(lr, X_test, y_test, "Linear Regression", scaled=True)
evaluate(rf, X_test, y_test, "Random Forest")
evaluate(gb, X_test, y_test, "Gradient Boosting")


# Step 7: Save models
joblib.dump(lr, "models/linear_regression.pkl")
joblib.dump(rf, "models/rf.pkl")
joblib.dump(gb, "models/gb.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print(" Models trained and saved in 'models/' directory")
