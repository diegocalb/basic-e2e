"""Module for training"""

import os

import joblib
from sklearn.ensemble import RandomForestRegressor

from src.data_processing import (
    aggregate_daily_sales,
    create_features,
    load_and_process_data,
)


def main():
    """Main function to run the data preparation and model training pipeline."""
    # --- Data Preparation ---
    print("Preparing data...")
    raw_df = load_and_process_data()
    daily_sales = aggregate_daily_sales(raw_df)
    daily_sales_features = create_features(daily_sales)

    # Split data
    X = daily_sales_features.drop("revenue", axis=1)
    y = daily_sales_features["revenue"]

    # Splitting data chronologically for time series
    split_point = int(len(X) * 0.8)  # Using 80% for training
    X_train, _ = X[:split_point], X[split_point:]
    y_train, _ = y[:split_point], y[split_point:]

    # --- Model Training ---
    print("Training model...")
    # Train a RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/time_series_model.pkl")

    print("Time series model trained and saved to models/time_series_model.pkl")

if __name__ == "__main__":
    main()
