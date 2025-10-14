import pandas as pd


def load_and_process_data(path="data/coffee_sales_full.csv"):
    """Loads and preprocesses the raw coffee sales data."""
    df = pd.read_csv(path, encoding="latin-1")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["revenue"] = df["transaction_qty"] * df["unit_price"]
    return df


def aggregate_daily_sales(df):
    """Aggregates transaction data to get total daily revenue."""
    daily_sales = (
        df.groupby(df["transaction_date"].dt.date)["revenue"].sum().reset_index()
    )
    daily_sales.columns = ["date", "revenue"]
    daily_sales["date"] = pd.to_datetime(daily_sales["date"])
    daily_sales = daily_sales.set_index("date").resample("D").sum()
    return daily_sales


def create_features(df):
    """Creates time series features from a datetime index."""
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
