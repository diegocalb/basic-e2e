"""Tests for data processing functions."""

import pandas as pd
import pytest
from coffee_modeling.data_processing import aggregate_daily


# --- Fixtures ---
@pytest.fixture
def raw_sales_df():
    """
    Creates a sample raw DataFrame with multiple transactions per day,
    mimicking the output of the load_data function.
    """
    data = {
        "transaction_date": [
            "2023-01-01",
            "2023-01-01",  # Dos transacciones el mismo día
            "2023-01-03",  # Un día sin transacciones en medio
        ],
        "revenue": [10.5, 20.0, 50.0],
    }
    df = pd.DataFrame(data)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


# --- Tests ---
def test_aggregate_daily(raw_sales_df):
    """
    Unit test for the aggregate_daily function.

    It checks for:
    1. Correct summation of revenue for a given day.
    2. Correct handling of days with no sales (resampling to 0).
    3. The output DataFrame has the correct structure.
    """
    # Ejecutar la función a probar
    daily_df = aggregate_daily(raw_sales_df)

    # 1. Verificar que el índice sea de tipo DatetimeIndex
    assert isinstance(daily_df.index, pd.DatetimeIndex)

    # 2. Verificar la agregación correcta para un día con múltiples transacciones
    assert daily_df.loc["2023-01-01"]["revenue"] == 30.5

    # 3. Verificar que el día sin ventas se rellenó con 0
    assert daily_df.loc["2023-01-02"]["revenue"] == 0.0

    # 4. Verificar el valor para un día con una sola transacción
    assert daily_df.loc["2023-01-03"]["revenue"] == 50.0

    # 5. Verificar que el DataFrame resultante tiene el tamaño esperado
    assert len(daily_df) == 3
