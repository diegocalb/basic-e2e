import pandas as pd
import pytest
from src.data_processing import create_features


@pytest.fixture
def sample_daily_sales():
    """Creates a sample DataFrame that mimics the daily_sales data."""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=20, freq="D"))
    data = {"revenue": range(20)}
    df = pd.DataFrame(data, index=dates)
    return df


def test_create_features(sample_daily_sales):
    """
    Tests the create_features function to ensure it adds the correct columns
    and removes rows with NaN values.
    """
    features_df = create_features(sample_daily_sales)

    expected_cols = [
        "dayofweek",
        "month",
        "year",
        "dayofyear",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "lag_5",
        "lag_6",
        "lag_7",
    ]
    for col in expected_cols:
        assert (
            col in features_df.columns
        ), f"Column '{col}' is missing from the DataFrame"

    # Check that rows with NaNs (created by lag features) are dropped
    assert (
        not features_df.isnull().values.any()
    ), "DataFrame should not contain any NaN values"
