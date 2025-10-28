"""Module for training"""

import os

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from coffee_modeling.data_processing import preprocessing_pipeline


def main(csv_path="data/coffee_sales_full.csv", n_estimators=100, random_state=42):
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
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    daily_sales_features = preprocessing_pipeline.fit_transform(csv_path)

    X = daily_sales_features.drop("revenue", axis=1)
    y = daily_sales_features["revenue"]

    split_point = int(len(X) * 0.8)
    X_train, _ = X[:split_point], X[split_point:]
    y_train, _ = y[:split_point], y[split_point:]

    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )

    full_pipeline = Pipeline(
        [("preprocessing", preprocessing_pipeline), ("model", model)]
    )

    with mlflow.start_run():
        print(f"Logging to MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        print("Preparing data...")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        print("Training model...")
        full_pipeline.fit(X_train, y_train)

        # Loguear odo el pipeline
        mlflow.sklearn.log_model(full_pipeline, artifact_path="coffee_model_pipeline")

        print("Time series model trained and logged to MLflow.")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
