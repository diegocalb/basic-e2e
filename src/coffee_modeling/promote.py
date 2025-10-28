"""
Módulo para la promoción de modelos de Staging a Production.

Este script compara el rendimiento del último modelo en 'Staging' con el modelo
actual en 'Production'. Si el modelo de 'Staging' muestra una mejora en el RMSE
superior a un umbral definido, se promueve a 'Production'.
"""

import os

import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error


def get_test_data(model_pipeline, data_path: str):
    """
    Prepara y devuelve el conjunto de datos de prueba.
    """
    print("Preparando conjunto de datos de prueba...")
    features = model_pipeline.named_steps["preprocessing"].transform(data_path)
    X = features.drop("revenue", axis=1)
    y = features["revenue"]
    split_point = int(len(X) * 0.8)
    X_test = X[split_point:]
    y_test = y[split_point:]
    return X_test, y_test


def evaluate_model_rmse(model, X_test, y_test):
    """
    Evalúa un modelo y devuelve su RMSE.
    """
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse


def main(model_name: str, rmse_improvement_threshold: float):
    """
    Función principal para ejecutar la lógica de promoción del modelo.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = mlflow.tracking.MlflowClient()

    try:
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not staging_versions:
            print("No hay modelos en 'Staging'. Saliendo.")
            return
        staging_version = staging_versions[0]
        print(f"Modelo en Staging encontrado: Versión {staging_version.version}")
    except mlflow.exceptions.RestException:
        print(f"El modelo '{model_name}' no existe o no tiene versiones en 'Staging'.")
        return

    # Cargar el modelo de Staging
    staging_model = mlflow.sklearn.load_model(staging_version.source)
    X_test, y_test = get_test_data(staging_model, "data/coffee_sales_full.csv")
    staging_rmse = evaluate_model_rmse(staging_model, X_test, y_test)
    print(
        f"RMSE del modelo en Staging (v{staging_version.version}): ${staging_rmse:,.2f}"
    )

    # Buscar y evaluar el modelo de Production
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_versions:
        print(
            "No hay modelo en 'Production'. Promoviendo el modelo de 'Staging' automáticamente."
        )
        client.transition_model_version_stage(
            name=model_name,
            version=staging_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Modelo v{staging_version.version} promovido a Production.")
        return

    prod_version = prod_versions[0]
    print(f"Modelo en Production encontrado: Versión {prod_version.version}")
    prod_model = mlflow.sklearn.load_model(prod_version.source)
    prod_rmse = evaluate_model_rmse(prod_model, X_test, y_test)
    print(f"RMSE del modelo en Production (v{prod_version.version}): ${prod_rmse:,.2f}")

    # Calculamos la mejora como (RMSE_antiguo - RMSE_nuevo) / RMSE_antiguo
    # Si el RMSE es menor, la mejora es positiva.
    improvement = (prod_rmse - staging_rmse) / prod_rmse
    print(f"\nMejora del rendimiento: {improvement:.2%}")

    if improvement > rmse_improvement_threshold:
        print(f"La mejora supera el umbral de {rmse_improvement_threshold:.2%}.")
        print(f"Promoviendo modelo v{staging_version.version} a 'Production'...")

        # Promover el nuevo modelo a Production
        client.transition_model_version_stage(
            name=model_name,
            version=staging_version.version,
            stage="Production",
            archive_existing_versions=True,  # Esto mueve el antiguo 'Production' a 'Archived'
        )
        print("¡Promoción completada!")
    else:
        print(
            f"La mejora no supera el umbral. Modelo v{staging_version.version} no será promovido."
        )


if __name__ == "__main__":
    MODEL_NAME = "coffee_model_pipeline"
    # Umbral de mejora: el nuevo modelo debe tener un RMSE un 5% menor
    # (es decir, el RMSE del nuevo modelo debe ser 95% o menos del RMSE del modelo actual)
    RMSE_IMPROVEMENT_THRESHOLD = 0.05

    main(MODEL_NAME, RMSE_IMPROVEMENT_THRESHOLD)
