"""
Módulo para generar pronósticos de ingresos para días futuros.

Este script carga el último modelo de producción, toma los datos históricos más
recientes y genera pronósticos de forma autorregresiva para un número
especificado de días futuros.
"""

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
        # Usamos .iloc[-i] para obtener el i-ésimo último valor de 'revenue'
        features[f"lag_{i}"] = last_data["revenue"].iloc[-i]

    return features


def generate_forecast(model, data_path: str, forecast_days: int = 7):
    """
    Genera pronósticos de ingresos para los próximos N días.

    Args:
        model: El pipeline de modelo entrenado.
        data_path (str): Ruta al archivo CSV con todos los datos históricos.
        forecast_days (int): Número de días a pronosticar en el futuro.

    Returns:
        pd.DataFrame: Un DataFrame con las fechas y los ingresos pronosticados.
    """
    print("Cargando y preprocesando datos históricos completos...")
    # Usamos el paso de preprocesamiento del pipeline para obtener todos los datos históricos
    # con las características correctas.
    historical_data = model.named_steps["preprocessing"].transform(data_path)

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
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    MODEL_URI = "models:/coffee_model_pipeline/latest"
    CSV_PATH = "data/coffee_sales_full.csv"
    FORECAST_DAYS = 7

    model_pipeline = load_model(MODEL_URI)
    future_predictions = generate_forecast(model_pipeline, CSV_PATH, FORECAST_DAYS)

    print("\n--- Pronóstico de Ingresos para los Próximos 7 Días ---")
    print(future_predictions)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "forecast.csv")
    future_predictions.to_csv(output_path, index=False)
    print(f"\nPronóstico guardado en: {output_path}")
