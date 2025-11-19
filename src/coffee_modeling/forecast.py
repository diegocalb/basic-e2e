"""
Módulo para generar pronósticos de ingresos para días futuros.

Este script carga el último modelo en 'Production', toma los datos históricos más
recientes y genera pronósticos de forma autorregresiva para un número
especificado de días futuros.
"""

# pylint: disable=E0401

import os
from datetime import timedelta

import mlflow
import pandas as pd


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


def create_future_features(last_data: pd.DataFrame, future_date: pd.Timestamp):
    """
    Crea un DataFrame de características para una única fecha futura.

    Args:
        last_data (pd.DataFrame): DataFrame que contiene los datos más recientes,
                                  incluyendo la columna 'revenue'.
        future_date (pd.Timestamp): La fecha para la que se generará el pronóstico.

    Returns:
        pd.DataFrame: Un DataFrame de una sola fila con las características para la predicción.
    """
    features = pd.DataFrame(index=[future_date])
    features["dayofweek"] = future_date.dayofweek
    features["month"] = future_date.month
    features["year"] = future_date.year
    features["dayofyear"] = future_date.dayofyear

    # Crear características de desfase (lag features) usando los datos más recientes
    for i in range(1, 8):
        features[f"lag_{i}"] = last_data["revenue"].shift(i)

    return features


def generate_forecast(
    model,
    postgres_conn_id: str,
    postgres_table_name: str,
    forecast_days: int = 7,
):
    """
    Genera pronósticos de ingresos para los próximos N días.

    Args:
        model: El pipeline de modelo entrenado.
        postgres_conn_id (str): Airflow connection ID for PostgreSQL.
        postgres_table_name (str): Name of the table to load data from.
        forecast_days (int): Número de días a pronosticar en el futuro.

    Returns:
        pd.DataFrame: Un DataFrame con las fechas y los ingresos pronosticados.
    """
    print("Cargando y preprocesando datos históricos completos...")
    historical_data = model.named_steps["preprocessing"].transform(
        None, load__conn_id=postgres_conn_id, load__table_name=postgres_table_name
    )

    last_date = historical_data.index.max()
    print(f"Último día en los datos: {last_date.date()}")

    forecasts = []
    current_data = historical_data.copy()

    for day in range(1, forecast_days + 1):
        future_date = last_date + timedelta(days=day)
        features_for_pred = create_future_features(current_data, future_date)
        prediction = model.named_steps["model"].predict(features_for_pred)[0]
        new_row = pd.DataFrame({"revenue": [prediction]}, index=[future_date])
        current_data = pd.concat([current_data, new_row])
        forecasts.append({"date": future_date, "predicted_revenue": prediction})

    return pd.DataFrame(forecasts)


if __name__ == "__main__":
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    _postgres_conn_id = os.getenv("POSTGRES_CONN_ID", "postgres_data_conn")
    _postgres_table_name = os.getenv("POSTGRES_TABLE_NAME", "coffee_sales")
    _forecast_days = int(os.getenv("FORECAST_DAYS", "7"))
    _output_table = os.getenv("FORECAST_OUTPUT_TABLE", "sales_forecast")

    MODEL_URI = "models:/coffee_model_pipeline/Production"

    model_pipeline = load_model(MODEL_URI)
    future_predictions = generate_forecast(
        model_pipeline,
        _postgres_conn_id,
        _postgres_table_name,
        _forecast_days,
    )

    print(f"\n--- Pronóstico de Ingresos para los Próximos {_forecast_days} Días ---")
    print(future_predictions)

    # Guardar en la base de datos de datos
    pg_hook = PostgresHook(postgres_conn_id=_postgres_conn_id)
    engine = pg_hook.get_sqlalchemy_engine()
    future_predictions.to_sql(_output_table, engine, if_exists="replace", index=False)
    print(f"\nPronóstico guardado en la tabla '{_output_table}' de la base de datos.")
