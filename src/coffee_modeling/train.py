"""Module for training"""

import os

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

from coffee_modeling.data_processing import (
    aggregate_daily_sales,
    create_features,
    load_and_process_data,
)


def main():
    """
    Main function to run the data preparation and model training pipeline.

    This function performs the following steps:
    1. Loads and processes the raw sales data.
    2. Aggregates the data to daily sales.
    3. Creates time series features.
    4. Splits the data into training and testing sets (using an 80/20 split).
    5. Trains a RandomForestRegressor model on the training data.
    6. Saves the trained model to the 'models/' directory.
    """
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio_access_key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio_secret_key"

    with mlflow.start_run():
        print("Preparing data...")
        raw_df = load_and_process_data()
        daily_sales = aggregate_daily_sales(raw_df)
        daily_sales_features = create_features(daily_sales)

        X = daily_sales_features.drop("revenue", axis=1)
        y = daily_sales_features["revenue"]

        split_point = int(len(X) * 0.8)
        X_train, _ = X[:split_point], X[split_point:]
        y_train, _ = y[:split_point], y[split_point:]

        n_estimators = 100
        random_state = 42
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        print("Training model...")
        model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, "random_forest_model")

        print("Time series model trained and logged to MLflow.")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
