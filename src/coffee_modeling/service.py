"""
Módulo de servicio de BentoML para el pronóstico de ventas de café.

Este script define un servicio BentoML que:
1. Carga el modelo de forecasting desde el registro de modelos de MLflow (etapa 'Production').
2. Expone un endpoint de API `/forecast` que acepta el número de días a pronosticar.
3. Genera el pronóstico de forma autorregresiva.
4. Guarda los resultados del pronóstico en una tabla de PostgreSQL.
"""

import os
from datetime import timedelta

import bentoml
import pandas as pd
from bentoml.io import JSON
from pydantic import BaseModel
from sqlalchemy import create_engine

# --- Configuración del Servicio ---

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
POSTGRES_USER = os.getenv("POSTGRESDATA_USER", "airflow")
POSTGRES_PASSWORD = os.getenv("POSTGRESDATA_PASSWORD", "airflow")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres_data")
POSTGRES_DB = os.getenv("POSTGRESDATA_DB", "airflow")

DATABASE_URI = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
    f"{POSTGRES_HOST}:5432/{POSTGRES_DB}"
)

MODEL_NAME = "coffee_model_pipeline"
MODEL_TAG = "production"

bento_model = bentoml.mlflow.import_model(
    MODEL_NAME,
    model_uri=f"models:/{MODEL_NAME}/Production",
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
)


class ForecastParams(BaseModel):
    """Esquema de entrada para la API de pronóstico."""

    forecast_days: int = 7


model_runner = bento_model.to_runner()

svc = bentoml.Service("coffee_forecasting_service", runners=[model_runner])


def create_future_features(last_data: pd.DataFrame, future_date: pd.Timestamp):
    """Crea un DataFrame de características para una única fecha futura."""
    features = pd.DataFrame(index=[future_date])
    features["dayofweek"] = future_date.dayofweek
    features["month"] = future_date.month
    features["year"] = future_date.year
    features["dayofyear"] = future_date.dayofyear

    for i in range(1, 8):
        features[f"lag_{i}"] = last_data["revenue"].shift(i - 1).iloc[-1]

    return features


@svc.api(  # pylint: disable=no-member
    input=JSON(pydantic_model=ForecastParams),
    output=JSON(),
)
async def forecast(params: ForecastParams) -> dict:
    """
    Endpoint de API para generar y guardar pronósticos de ventas.
    """
    print(f"Solicitud recibida para pronosticar {params.forecast_days} días.")

    # 1. Cargar datos históricos usando el pipeline de preprocesamiento del modelo
    # El pipeline está diseñado para tomar `None` y cargar desde la DB
    historical_data = await model_runner.preprocessing.async_run(
        None,
        load__conn_id="postgres_data_conn",  # Se mantiene por compatibilidad
        load__table_name="coffee_sales",
    )

    last_date = historical_data.index.max()
    print(f"Último día en los datos históricos: {last_date.date()}")

    # 2. Generar pronósticos de forma autorregresiva
    forecasts = []
    current_data = historical_data.copy()

    for day in range(1, params.forecast_days + 1):
        future_date = last_date + timedelta(days=day)
        features_for_pred = create_future_features(current_data, future_date)

        # Usar el paso 'model' del pipeline para predecir
        prediction = await model_runner.model.async_run(features_for_pred)
        prediction_value = prediction[0]

        new_row = pd.DataFrame({"revenue": [prediction_value]}, index=[future_date])
        current_data = pd.concat([current_data, new_row])
        forecasts.append(
            {"date": future_date.isoformat(), "predicted_revenue": prediction_value}
        )

    # 3. Guardar pronósticos en PostgreSQL
    forecast_df = pd.DataFrame(forecasts)
    engine = create_engine(DATABASE_URI)
    forecast_df.to_sql("bento_forecasts", engine, if_exists="replace", index=False)
    print("Pronóstico guardado en la tabla 'bento_forecasts'.")

    return {"status": "success", "forecast": forecasts}
