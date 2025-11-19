"""
Módulo para la promoción de modelos de Staging a Production.
"""

# pylint: disable=E0401
import os
import time
from dataclasses import dataclass

import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error

from coffee_modeling.data_processing import split_data


@dataclass
class DataConfig:
    """Config para la conexión a Postgres"""

    postgres_conn_id: str
    postgres_table_name: str
    holdout_days: int


@dataclass
class PromotionConfig:
    """Config para la promoción"""

    model_name: str
    rmse_improvement_threshold: float
    model_obsolescence_days: int


def load_latest_version(client, model_name, stage):
    """Devuelve la última versión de un modelo en un stage dado."""
    versions = client.get_latest_versions(model_name, stages=[stage])
    return versions[0] if versions else None


def load_model_by_version(version):
    """Carga un modelo dado un objeto ModelVersion."""
    return mlflow.sklearn.load_model(version.source)


def prepare_holdout(model, data_config: DataConfig):
    """Genera el holdout test set usando el pipeline del modelo."""
    features = model.named_steps["preprocessing"].transform(
        None,
        load__conn_id=data_config.postgres_conn_id,
        load__table_name=data_config.postgres_table_name,
    )
    _, _, X_test, y_test = split_data(features, holdout_days=data_config.holdout_days)
    return X_test, y_test


def evaluate_rmse(model, X_test, y_test):
    """Calcula el RMSE de un modelo."""
    preds = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, preds))


def compute_model_age_days(version):
    """Devuelve edad del modelo en días."""
    creation_s = version.creation_timestamp / 1000
    return (time.time() - creation_s) / 86400


def evaluate_staging(client, promotion_config, data_config):
    """Evaluar staging"""
    staging_version = load_latest_version(
        client, promotion_config.model_name, "Staging"
    )
    if staging_version is None:
        print("No hay modelos en Staging.")
        return None, None, None, None

    print(f"Modelo en Staging encontrado: v{staging_version.version}")
    staging_model = load_model_by_version(staging_version)
    X_test, y_test = prepare_holdout(staging_model, data_config)
    staging_rmse = evaluate_rmse(staging_model, X_test, y_test)
    print(f"RMSE Staging: ${staging_rmse:,.2f}")
    return staging_version, X_test, y_test, staging_rmse


def evaluate_production(client, promotion_config, X_test, y_test):
    """Evaluar modelo producción"""
    prod_version = load_latest_version(
        client, promotion_config.model_name, "Production"
    )
    if prod_version is None:
        return None, None

    print(f"Modelo en Production encontrado: v{prod_version.version}")
    prod_model = load_model_by_version(prod_version)
    prod_rmse = evaluate_rmse(prod_model, X_test, y_test)
    print(f"RMSE Production: ${prod_rmse:,.2f}")

    return prod_version, prod_rmse


def promote_version(client, model_name, version, reason):
    """Promueve un modelo a Production."""
    print(f"🚀 Promoviendo v{version.version} a Production ({reason})...")
    client.transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print("Promoción completada.")


def main(data_config: DataConfig, promotion_config: PromotionConfig):
    """Función Main"""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = mlflow.MlflowClient()

    staging_version, X_test, y_test, staging_rmse = evaluate_staging(
        client, promotion_config, data_config
    )
    if staging_version is None:
        return

    prod_version, prod_rmse = evaluate_production(
        client, promotion_config, X_test, y_test
    )

    if prod_version is None:
        promote_version(
            client, promotion_config.model_name, staging_version, "Primer modelo"
        )
        return

    improvement = (prod_rmse - staging_rmse) / prod_rmse
    model_age_days = compute_model_age_days(prod_version)
    print(f"Mejora: {improvement:.2%}")
    print(f"Edad del modelo en Prod: {model_age_days:.1f} días")

    should_promote_due_perf = improvement > promotion_config.rmse_improvement_threshold
    is_obsolete = model_age_days > promotion_config.model_obsolescence_days

    if should_promote_due_perf or is_obsolete:
        reason = (
            f"Mejora {improvement:.2%}"
            if should_promote_due_perf
            else f"Obsolescencia > {promotion_config.model_obsolescence_days} días"
        )
        promote_version(client, promotion_config.model_name, staging_version, reason)
    else:
        print("No se cumple mejora ni obsolescencia.")


if __name__ == "__main__":
    data_cfg = DataConfig(
        postgres_conn_id=os.getenv("POSTGRES_CONN_ID", "postgres_data_conn"),
        postgres_table_name=os.getenv("POSTGRES_TABLE_NAME", "coffee_sales"),
        holdout_days=int(os.getenv("HOLDOUT_DAYS", "30")),
    )

    promo_cfg = PromotionConfig(
        model_name="coffee_model_pipeline",
        rmse_improvement_threshold=float(
            os.getenv("RMSE_IMPROVEMENT_THRESHOLD", "-0.05")
        ),
        model_obsolescence_days=int(os.getenv("MODEL_OBSOLESCENCE_DAYS", "60")),
    )

    main(data_cfg, promo_cfg)
