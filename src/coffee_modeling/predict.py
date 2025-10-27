"""Module for prediction"""

import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.makedirs("output", exist_ok=True)


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
model_pipeline = mlflow.sklearn.load_model("models:/coffee_model_pipeline/latest")

csv_path = "data/coffee_sales_full.csv"
raw_df = pd.read_csv(csv_path, encoding="latin-1")

daily_sales_features = model_pipeline.named_steps["preprocessing"].transform(csv_path)

X = daily_sales_features.drop("revenue", axis=1)
y = daily_sales_features["revenue"]

split_point = int(len(X) * 0.8)
X_test = X[split_point:]
y_test = y[split_point:]

print("Making predictions on the test set...")
predictions = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\nModel Evaluation on Test Set:")
print(f"  Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"  Root Mean Squared Error (RMSE): ${rmse:,.2f}")

print("\nSaving predictions to output/predictions.csv...")
pd.DataFrame(
    {"date": y_test.index, "actual_revenue": y_test, "predicted_revenue": predictions}
).to_csv("output/predictions.csv", index=False)

print("Inference complete.")
