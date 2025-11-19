"""
Módulo para evaluar un modelo en un conjunto de datos de prueba histórico.

Este script carga el último pipeline de modelo desde el registro de modelos de MLflow,
realiza predicciones sobre el conjunto de datos de prueba y calcula métricas de rendimiento.
"""

# pylint: disable=E0401

import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sklearn.metrics import mean_absolute_error, mean_squared_error

from coffee_modeling.data_processing import split_data


def load_model(model_uri: str):
    """
    Carga el pipeline del modelo de scikit-learn desde MLflow.

    Args:
        model_uri (str): La URI del modelo en el registro de MLflow.

    Returns:
        sklearn.pipeline.Pipeline: El pipeline del modelo cargado.
    """
    print(f"Cargando modelo desde: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def make_predictions(
    model_pipeline,
    postgres_conn_id: str,
    postgres_table_name: str,
    holdout_days: int,
):
    """
    Prepara los datos y realiza predicciones utilizando el pipeline del modelo.

    Args:
        model_pipeline (sklearn.pipeline.Pipeline): El pipeline del modelo entrenado.
        postgres_conn_id (str): Airflow connection ID for PostgreSQL.
        postgres_table_name (str): Name of the table to load data from.
        holdout_days (int): Number of days to reserve for the test set.

    Returns:
        tuple: Una tupla conteniendo (y_test, predictions).
    """
    print("Preparando datos y generando predicciones...")
    daily_sales_features = model_pipeline.named_steps["preprocessing"].transform(
        None, load__conn_id=postgres_conn_id, load__table_name=postgres_table_name
    )

    _, _, X_test, y_test = split_data(daily_sales_features, holdout_days=holdout_days)

    predictions = model_pipeline.predict(X_test)
    return y_test, predictions


def evaluate_and_save(
    y_test: pd.Series,
    predictions: np.ndarray,
    postgres_conn_id: str,
    output_table: str,
):
    """
    Evalúa las predicciones, imprime las métricas y guarda los resultados.
    """
    print("\nEvaluando y guardando resultados...")
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\nEvaluación del Modelo en el Conjunto de Prueba:")
    print(f"  Error Absoluto Medio (MAE): ${mae:,.2f}")
    print(f"  Raíz del Error Cuadrático Medio (RMSE): ${rmse:,.2f}")

    results_df = pd.DataFrame(
        {
            "date": y_test.index,
            "actual_revenue": y_test,
            "predicted_revenue": predictions,
        }
    )

    pg_hook = PostgresHook(postgres_conn_id=postgres_conn_id)
    engine = pg_hook.get_sqlalchemy_engine()
    results_df.to_sql(output_table, engine, if_exists="replace", index=False)
    print(f"Resultados de la evaluación guardados en la tabla '{output_table}'.")


def main(
    postgres_conn_id: str,
    postgres_table_name: str,
    holdout_days: int,
    output_table: str,
):
    """Función principal para ejecutar el script de evaluación."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model_uri = "models:/coffee_model_pipeline/latest"

    model = load_model(model_uri)
    y_test, predictions = make_predictions(
        model, postgres_conn_id, postgres_table_name, holdout_days
    )
    evaluate_and_save(y_test, predictions, postgres_conn_id, output_table)

    print("\nEvaluación completada.")


if __name__ == "__main__":
    _postgres_conn_id = os.getenv("POSTGRES_CONN_ID", "postgres_data_conn")
    _postgres_table_name = os.getenv("POSTGRES_TABLE_NAME", "coffee_sales")
    _holdout_days = int(os.getenv("HOLDOUT_DAYS", "30"))
    _output_table = os.getenv("EVALUATION_OUTPUT_TABLE", "evaluation_results")

    main(
        postgres_conn_id=_postgres_conn_id,
        postgres_table_name=_postgres_table_name,
        holdout_days=_holdout_days,
        output_table=_output_table,
    )
