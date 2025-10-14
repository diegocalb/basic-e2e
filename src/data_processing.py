"""Module for data processing functions"""

import pandas as pd


def load_and_process_data(path="data/coffee_sales_full.csv"):
    """
    Loads and preprocesses the raw coffee sales data from a CSV file.

    This function reads a CSV file, converts the transaction date column to
    datetime objects, and calculates a 'revenue' column.

    Args:
        path (str, optional): The file path to the CSV data.
            Defaults to "data/coffee_sales_full.csv".

    Returns:
        pd.DataFrame: A DataFrame with processed data.
    """
    df = pd.read_csv(path, encoding="latin-1")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["revenue"] = df["transaction_qty"] * df["unit_price"]
    return df


def aggregate_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction data to get total daily revenue.

    Args:
        df (pd.DataFrame): DataFrame with transaction-level data.

    Returns:
        pd.DataFrame: A DataFrame with daily aggregated revenue, indexed by date.
    """
    daily_sales = (
        df.groupby(df["transaction_date"].dt.date)["revenue"].sum().reset_index()
    )
    daily_sales.columns = ["date", "revenue"]
    daily_sales["date"] = pd.to_datetime(daily_sales["date"])
    daily_sales = daily_sales.set_index("date").resample("D").sum()
    return daily_sales


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time series features from a datetime index.

    Args:
        df (pd.DataFrame): DataFrame with a datetime index and a 'revenue' column.

    Returns:
        pd.DataFrame: DataFrame with new date-based and lag features.
    """
    features_df = df.copy()
    features_df["dayofweek"] = features_df.index.dayofweek
    features_df["month"] = features_df.index.month
    features_df["year"] = features_df.index.year
    features_df["dayofyear"] = features_df.index.dayofyear
    # Add lag features
    for i in range(1, 8):
        features_df[f"lag_{i}"] = features_df["revenue"].shift(i)

    # Drop rows with NaN values created by lag features
    features_df = features_df.dropna()
    return features_df
