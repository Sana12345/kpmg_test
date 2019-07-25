import numpy as np
import pandas as pd
import statsmodels.api as sm

from tst.generators.mocker import _genrate_datetime


def generate_arima_mock_data(start_end=None, freq='m', intercept=0, sigma=100,
                     trend=2, trend_amplitude=0.01, seasonal_periodicity=12,
                     seasonal_periodicity_2=4, seasonal_amplitude=100,
                     seasonal_amplitude_2=100, ar_params=np.array([0.9]),
                     ma_params=np.array([0.2]), d=0
                     ):
    """Generate time series with trend, seasonal and arima component
    Parameters
    ----------
    start_end : tuple, optional
        start and end of timestamp range, default is:
        ('1989-08-10 09:00:00', '1989-08-11 10:01:00')
    freq : str, optional
        frequency of series using numpy notation:
        * 'Y': year
        * 'M': month
        * 'D': day
        * 'h': hour
        * 'm': minute
        * 's': seconds
        * 'ms': miliseconds
        * 'us': microseconds
        * 'ns': nanoseconds
    intercept : int or float, optional intercept of ARMA component,
        note that this changes the unconditional mean in
        intercept / (1 - sum(ar_params))
    sigma : int or float, optional
        standard deviation of normal distribution from which values are drawn
    trend : int, optional:
        * 0 : no trend
        * 1 : linear trend
        * 2 : parabolic trend
    trend_amplitude : int or float, optional:
        multiplicative factor for trend component
    seasonal_periodicity : int or float, optional
        period of first seasonal component
    seasonal_periodicity_2 : int or float, optional
        periodicity of second seasonal component
    seasonal_amplitude : int or float, optional
        multiplicative factor for first seasonal component
    seasonal_amplitude_2 : int or float, optional
        multiplicative factor for second seasonal component
    ar_params : 1d array, optional
        coefficient for autoregressive lag polynomial, including zero lag
    ma_params : 1d array, optional
        coefficient for moving-average lag polynomial, including zero lag
    Returns
    -------
    mock_ts : pd.DataFrame
        frame with datetime index and random walk values
    Return value:
    """
    ind = _genrate_datetime(start_end, freq)
    N = ind.size

    #seasonal component
    x = np.arange(N)
    seasonal_component = (seasonal_amplitude * np.sin(2 * np.pi
                                                      * seasonal_periodicity
                                                      * x / N
                                                      )
                          + seasonal_amplitude_2
                          * np.sin(2 * np.pi * seasonal_periodicity_2 * x / N))
    #ARMA component
    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]
    arma_component = sm.tsa.arma_generate_sample(ar, ma, N, sigma)
    arma_component[0] = intercept / (1 - sum(ar_params))
    for n in range(1, N): #adding desired intercept to component
        arma_component[n] = arma_component[0] + arma_component[n]
    #depending on value for d, take ARMA component from I(0) to I(d)
    for _ in range(0, d):
        arma_component = np.cumsum(arma_component)
    if trend == 1: #add a linear trend
        trend_component = trend_amplitude*np.linspace(0, N, N, )
        data_concat = np.c_[trend_component, seasonal_component,
                            arma_component, trend_component
                            + seasonal_component + arma_component
                            ]
        mock_ts = pd.DataFrame(data=data_concat, index=ind,
                               columns=['trend_component', 'seasonal_component'
                                        , 'arima_component', 'summed'
                                        ]
                               )
        mock_ts.index.freq = mock_ts.index.inferred_freq
    elif trend == 2: #add a parabolic trend
        trend_component = np.linspace(0, N, N,)
        trend_component = trend_amplitude * (np.square(trend_component
                                                       - (N / 2)))
        data_concat = np.c_[trend_component, seasonal_component,
                            arma_component, trend_component
                            + seasonal_component + arma_component
                            ]
        mock_ts = pd.DataFrame(data=data_concat, index=ind,
                               columns=['trend_component',
                                        'seasonal_component',
                                        'arima_component', 'summed'
                                        ]
                               )
        mock_ts.index.freq = mock_ts.index.inferred_freq
    elif trend == 0: #no trend
        data_concat = np.c_[seasonal_component, arma_component,
                            seasonal_component + arma_component
                            ]
        mock_ts = pd.DataFrame(data=data_concat, index=ind,
                               columns=['seasonal_component',
                                        'arima_component', 'summed']
                               )
        mock_ts.index.freq = mock_ts.index.inferred_freq
    return mock_ts
