
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("insurance.csv")


# bmi_cat
df["bmi_cat"] = pd.cut(df["bmi"],
                       bins=[0, 18.5, 25, 30, np.inf],
                       labels=["Underweight","Normal","Overweight","Obese"])

# features & target
X = df.drop("charges", axis=1)
y = df["charges"]

# Identify categorical & numerical columns

cat_cols = ["sex", "smoker", "region","bmi_cat"]
num_cols = ["age", "bmi", "children"]


# Create preprocessor
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cats", categorical_transformer, cat_cols),
        ("nums", numerical_transformer, num_cols)
    ])




# GradientBoostingRegressor Model
gb_pipeline = Pipeline(
    [
        ('preprocessor',preprocessor),
        ('model',GradientBoostingRegressor(n_estimators=100 , random_state=42))

     ]

  )



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


gb_pipeline.fit(X_train, y_train)


# Evaluation
y_pred = gb_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")



# Save model 
with open("insurance_gb_pipeline.pkl", "wb") as f:
    pickle.dump(gb_pipeline, f)

print("GradientBoostingRegressor pipeline saved as insurance_gb_pipeline.pkl")

