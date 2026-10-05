import time

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    """
    ARIMA(p,d,q) ajustado só no treino. A previsão no teste é ONE-STEP-AHEAD
    (mesma tarefa dos demais modelos): os parâmetros ficam congelados e o filtro
    de estado (Kalman) é reaplicado à série completa, usando y_{t-1}, y_{t-2}, ...
    observados para prever y_t.
    """

    def __init__(self, p=24, d=0, q=2, seasonal_order=None):
        self.order = (int(p), int(d), int(q))
        self.seasonal_order = tuple(seasonal_order) if seasonal_order else (0, 0, 0, 0)

    def fit_predict(self, series, train_end, test_start):
        s = time.perf_counter()
        fitted = ARIMA(series[:train_end], order=self.order, seasonal_order=self.seasonal_order).fit()
        self.train_time = time.perf_counter() - s
        self.n_params = int(len(fitted.params))
        s = time.perf_counter()
        full = fitted.apply(series)                      # mesmos parâmetros, série inteira
        pred = full.predict(start=test_start, end=len(series) - 1)
        self.inference_time = time.perf_counter() - s
        return np.asarray(pred, np.float32)

    @property
    def parameters(self):
        return self.n_params
