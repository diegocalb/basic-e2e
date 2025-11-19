"""Module for training"""

# pylint: disable=E0401

import os

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from coffee_modeling.data_processing import (
    preprocessing_pipeline,
    split_data,
)


def main(
    postgres_conn_id: str = "postgres_data_conn",
    postgres_table_name: str = "coffee_sales",
    n_estimators: int = 100,
    random_state: int = 42,
    holdout_days: int = 30,
):
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

    print(
        f"Loading from PostgreSQL table: {postgres_table_name} using connection: {postgres_conn_id}"
    )
    daily_sales_features = preprocessing_pipeline.fit_transform(
        None,  # X is None for the first step (load_data_from_postgres)
        load__conn_id=postgres_conn_id,
        load__table_name=postgres_table_name,
    )

    X_train, y_train, _, _ = split_data(daily_sales_features, holdout_days=holdout_days)

    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )

    full_pipeline = Pipeline(
        [("preprocessing", preprocessing_pipeline), ("model", model)]
    )

    with mlflow.start_run() as run:
        print(f"Logging to MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        print("Preparing data...")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        print("Training model...")
        full_pipeline.fit(X_train, y_train)

        mlflow.sklearn.log_model(full_pipeline, artifact_path="coffee_model_pipeline")

        print("Time series model trained and logged to MLflow.")
        print(f"MLflow Run ID: {run.info.run_id}")


if __name__ == "__main__":
    _postgres_conn_id = os.getenv("POSTGRES_CONN_ID", "postgres_data_conn")
    _postgres_table_name = os.getenv("POSTGRES_TABLE_NAME", "coffee_sales")
    _holdout_days = int(os.getenv("HOLDOUT_DAYS", "30"))

    print("Running train.py locally...")
    main(
        postgres_conn_id=_postgres_conn_id,
        postgres_table_name=_postgres_table_name,
        holdout_days=_holdout_days,
    )
