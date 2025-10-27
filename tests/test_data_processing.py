import pandas as pd
import pytest
from coffee_modeling.data_processing import (
    aggregate_daily,
    create_features_pipeline,
    load_data,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


@pytest.fixture
def sample_daily_sales():
    """
    Creates a sample DataFrame mimicking daily_sales data.
    """
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=20, freq="D"))
    data = {
        "transaction_date": dates,
        "transaction_qty": range(20),
        "unit_price": range(20, 40),
    }
    df = pd.DataFrame(data)
    return df


def test_preprocessing_pipeline(sample_daily_sales):
    """
    Tests the preprocessing pipeline end-to-end:
    load -> aggregate -> features -> imputer
    """
    load_transformer = FunctionTransformer(load_data)
    aggregate_transformer = FunctionTransformer(aggregate_daily)
    features_transformer = FunctionTransformer(create_features_pipeline)
    preprocessing_pipeline = Pipeline(
        [
            ("load", load_transformer),
            ("aggregate", aggregate_transformer),
            ("features", features_transformer),
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )

    df_processed = (
        preprocessing_pipeline.fit_transform("data/coffee_sales_full.csv")
        if False
        else preprocessing_pipeline.fit_transform(sample_daily_sales)
    )

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
        "revenue",
    ]

    for col in expected_cols:
        assert (
            col in df_processed.columns
        ), f"Column '{col}' is missing from the DataFrame"

    assert not pd.isnull(
        df_processed.values
    ).any(), "DataFrame should not contain any NaN values"
