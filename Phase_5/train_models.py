import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Step 1: Load Kaggle dataset
df = pd.read_csv("data/train.csv")

# Target variable
y = df["SalePrice"]

# Drop ID and target from features
X = df.drop(["Id", "SalePrice"], axis=1)


# Step 2: Basic preprocessing

# Fill missing numeric values with median
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

# Fill missing categorical values with mode
cat_cols = X.select_dtypes(include=["object"]).columns
for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)


# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 4: Train modelso
print("Training models...")

# Linear Regression
lr = LinearRegression().fit(X_train_scaled, y_train)

# Random Forest (no scaling needed)
rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)

# Gradient Boosting (no scaling needed)
gb = GradientBoostingRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)

# Step 5: Evaluate models
def evaluate(model, X_t, y_t, name):
    preds = model.predict(X_t)
    rmse = np.sqrt(mean_squared_error(y_t, preds))
    r2 = r2_score(y_t, preds)
    print(f"{name}  RMSE: {rmse:.2f}, R²: {r2:.3f}")

evaluate(lr, X_test_scaled, y_test, "Linear Regression")
evaluate(rf, X_test, y_test, "Random Forest")
evaluate(gb, X_test, y_test, "Gradient Boosting")


# Step 6: Save models
joblib.dump(lr, "models/linear_regression.pkl")
joblib.dump(rf, "models/rf.pkl")
joblib.dump(gb, "models/gb.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print(" Models trained and saved in 'models/' directory")
