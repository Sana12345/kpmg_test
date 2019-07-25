import numpy as np
import pandas as pd

from mockerArima import generate_arima_mock_data

np.random.seed(123)


def test_generate_arima_mock_data_defaults():
    """Test generate_arima_mock_data with default settings"""
    df = generate_arima_mock_data()
    assert df.index.values[0] == np.datetime64('1989-08-10 09:00:00', 'm')
    assert df.index.values[-1] == np.datetime64('1989-08-11 10:00:00', 'm')
    assert isinstance(df, pd.DataFrame)


def test_generate_arima_mock_data():
    """Test generate_arima_mock_data with some settings"""
    df = generate_arima_mock_data(
        start_end=('1999-12-31 23:59:00', '2000-01-01 00:10:01'),
        freq='s',
        intercept=10,
        sigma=50,
        trend=1,
        trend_amplitude=20,
        seasonal_periodicity=10,
        seasonal_periodicity_2=50,
        seasonal_amplitude=300,
        seasonal_amplitude_2=1000,
        ar_params=np.array([0.5, 0.1, 0.2]),
        ma_params=np.array([0.2, 0.1, 0.2]),
        d=1
    )
    assert df.index[0] == np.datetime64('1999-12-31 23:59:00', 's')
    assert df.index[-1] == np.datetime64('2000-01-01 00:10:00', 's')
    assert isinstance(df, pd.DataFrame)
