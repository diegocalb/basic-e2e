"""Module for data processing functions"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def load_data(path="data/coffee_sales_full.csv"):
    """Loads and preprocesses the raw coffee sales data from a CSV file.

    This function reads a CSV file, converts the 'transaction_date' column to
    datetime objects, and calculates a 'revenue' column by multiplying
    'transaction_qty' and 'unit_price'.

    Args:
        path (str, optional): The file path to the CSV data.
            Defaults to "data/coffee_sales_full.csv".

    Returns:
        pd.DataFrame: A DataFrame with processed data, including a 'revenue' column
                      and 'transaction_date' as datetime objects.
    """
    df = pd.read_csv(path, encoding="latin-1")
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


def split_data(features_df: pd.DataFrame, test_size: float = 0.2):
    """
    Divide el DataFrame de características en conjuntos de entrenamiento y prueba.

    Args:
        features_df (pd.DataFrame): El DataFrame completo con características.
        test_size (float): La proporción del dataset a reservar para el conjunto de prueba.

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    X = features_df.drop("revenue", axis=1)
    y = features_df["revenue"]
    split_point = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    return X_train, y_train, X_test, y_test


# --- Pipelines de preprocesamiento ---
load_transformer = FunctionTransformer(load_data)
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

# --- Ejecutar ---
df_features = pipeline.fit_transform("data/coffee_sales_full.csv")

# --- Manejo de NaNs adicional (si quieres asegurar) ---
imputer = SimpleImputer(strategy="mean")
df_features_filled = pd.DataFrame(
    imputer.fit_transform(df_features),
    columns=df_features.columns,
    index=df_features.index,
)

preprocessing_pipeline = Pipeline(
    [
        ("load", load_transformer),
        ("aggregate", aggregate_transformer),
        ("features", features_transformer),
        ("imputer", SimpleImputer(strategy="mean")),
    ]
)
