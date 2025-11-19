"""Module for data processing functions"""

# pylint: disable=E0401

from datetime import timedelta

import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def load_data_from_postgres(conn_id: str, table_name: str):
    """Loads and preprocesses the raw coffee sales data from a CSV file.
    Loads and preprocesses the raw coffee sales data from a PostgreSQL table.

    This function reads a CSV file, converts the 'transaction_date' column to
    datetime objects, and calculates a 'revenue' column by multiplying
    'transaction_qty' and 'unit_price'.

    Args:
        conn_id (str): The Airflow connection ID for PostgreSQL.
        table_name (str): The name of the table to load data from.

    Returns:
        pd.DataFrame: A DataFrame with processed data, including a 'revenue' column
                      and 'transaction_date' as datetime objects.
    """
    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    sql = f"SELECT * FROM {table_name}"
    df = pg_hook.get_pandas_df(sql)

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["revenue"] = df["transaction_qty"] * df["unit_price"]
    return df


def aggregate_daily(df: pd.DataFrame):
    """Aggregates transaction data to get total daily revenue.

    This function groups the data by date, sums the 'revenue' for each day,
    and resamples the data to a daily frequency to fill any missing days with zeros.

    Args:
        df (pd.DataFrame): DataFrame with transaction-level data, must contain
                           'transaction_date' and 'revenue' columns.

    Returns:
        pd.DataFrame: A DataFrame with daily aggregated revenue, indexed by date.
    """
    daily = df.groupby(df["transaction_date"].dt.date)["revenue"].sum().reset_index()
    daily.columns = ["date", "revenue"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date").resample("D").sum()
    return daily


def create_features_pipeline(df: pd.DataFrame):
    """Creates time series features from a datetime-indexed DataFrame.

    This function adds date-based features (day of week, month, year, day of year)
    and lag features (revenue from the previous 7 days). It then drops rows with NaN
    values that are created by the lag feature generation, making the data ready
    for model training.

    Args:
        df (pd.DataFrame): DataFrame with a datetime index and a 'revenue' column.

    Returns:
        pd.DataFrame: DataFrame with new date-based and lag features.
    """
    df = df.copy()
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    for i in range(1, 8):
        df[f"lag_{i}"] = df["revenue"].shift(i)
    df = df.dropna()
    return df


def split_data(features_df: pd.DataFrame, holdout_days: int = 30):
    """
    Divide el DataFrame de características en conjuntos de entrenamiento y prueba
    basados en una fecha de corte para el holdout.

    Args:
        features_df (pd.DataFrame): El DataFrame completo con características.
        holdout_days (int): Número de días a reservar para el conjunto de prueba
                            al final del dataset.

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    if not isinstance(features_df.index, pd.DatetimeIndex):
        raise ValueError(
            "features_df must have a DatetimeIndex for time-based splitting."
        )

    latest_date = features_df.index.max()
    split_date = latest_date - timedelta(days=holdout_days)

    X = features_df.drop("revenue", axis=1)
    y = features_df["revenue"]

    X_train = X[X.index <= split_date]
    y_train = y[y.index <= split_date]
    X_test = X[X.index > split_date]
    y_test = y[y.index > split_date]

    print(
        f"Split data: Training up to {split_date.date()}, Testing after {split_date.date()}"
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    return X_train, y_train, X_test, y_test


# --- Pipelines de preprocesamiento ---
load_transformer = FunctionTransformer(load_data_from_postgres)
aggregate_transformer = FunctionTransformer(aggregate_daily)
features_transformer = FunctionTransformer(create_features_pipeline)

# --- Pipeline completo ---
pipeline = Pipeline(
    [
        ("load", load_transformer),
        ("aggregate", aggregate_transformer),
        ("features", features_transformer),
    ]
)

preprocessing_pipeline = Pipeline(
    [
        ("load", load_transformer),
        ("aggregate", aggregate_transformer),
        ("features", features_transformer),
        ("imputer", SimpleImputer(strategy="mean")),
    ]
)
