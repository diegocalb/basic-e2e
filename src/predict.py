"""Module for prediction"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_processing import (
    aggregate_daily_sales,
    create_features,
    load_and_process_data,
)

os.makedirs("output", exist_ok=True)

raw_df = load_and_process_data()
daily_sales = aggregate_daily_sales(raw_df)
daily_sales_features = create_features(daily_sales)

X = daily_sales_features.drop("revenue", axis=1)
y = daily_sales_features["revenue"]

split_point = int(len(X) * 0.8)
X_test = X[split_point:]
y_test = y[split_point:]

print("Loading model from models/time_series_model.pkl...")
model = joblib.load("models/time_series_model.pkl")

print("Making predictions on the test set...")
predictions = model.predict(X_test)

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
