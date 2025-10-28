"""
Módulo para evaluar un modelo en un conjunto de datos de prueba histórico.

Este script carga el último pipeline de modelo desde el registro de modelos de MLflow,
realiza predicciones sobre el conjunto de datos de prueba y calcula métricas de rendimiento.
"""

import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


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


def make_predictions(model_pipeline, data_path: str):
    """
    Prepara los datos y realiza predicciones utilizando el pipeline del modelo.

    Args:
        model_pipeline (sklearn.pipeline.Pipeline): El pipeline del modelo entrenado.
        data_path (str): La ruta al archivo CSV con los datos brutos.

    Returns:
        tuple: Una tupla conteniendo (y_test, predictions).
    """
    print("Preparando datos y generando predicciones...")
    daily_sales_features = model_pipeline.named_steps["preprocessing"].transform(
        data_path
    )

    X = daily_sales_features.drop("revenue", axis=1)
    y = daily_sales_features["revenue"]

    split_point = int(len(X) * 0.8)
    X_test = X[split_point:]
    y_test = y[split_point:]

    predictions = model_pipeline.predict(X_test)
    return y_test, predictions


def evaluate_and_save(y_test: pd.Series, predictions: np.ndarray):
    """
    Evalúa las predicciones, imprime las métricas y guarda los resultados.
    """
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\nEvaluación del Modelo en el Conjunto de Prueba:")
    print(f"  Error Absoluto Medio (MAE): ${mae:,.2f}")
    print(f"  Raíz del Error Cuadrático Medio (RMSE): ${rmse:,.2f}")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predictions.csv")
    print(f"\nGuardando predicciones en {output_path}...")
    pd.DataFrame(
        {
            "date": y_test.index,
            "actual_revenue": y_test,
            "predicted_revenue": predictions,
        }
    ).to_csv(output_path, index=False)


def main():
    """Función principal para ejecutar el script de evaluación."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model_uri = "models:/coffee_model_pipeline/latest"
    csv_path = "data/coffee_sales_full.csv"

    model = load_model(model_uri)
    y_test, predictions = make_predictions(model, csv_path)
    evaluate_and_save(y_test, predictions)

    print("\nEvaluación completada.")


if __name__ == "__main__":
    main()
