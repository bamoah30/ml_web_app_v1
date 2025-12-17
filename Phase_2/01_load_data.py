from typing import Any, cast
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset as a Pandas DataFrame (cast to Any to satisfy type checkers)
housing = cast(Any, fetch_california_housing(as_frame=True))
df = housing.frame if hasattr(housing, "frame") else pd.DataFrame(housing.data, columns=housing.feature_names)

# Features (X) and target (y)
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print(df.head())