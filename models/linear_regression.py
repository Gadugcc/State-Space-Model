from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LinearRegression


def make_windows(
    series,
    context_length: int,
    horizon: int = 1,
):
    """
    Transforma uma série temporal em um problema supervisionado.

    Exemplo:

        context_length = 3

        X:
        [y(t-3), y(t-2), y(t-1)]

        y:
        y(t)

    Parameters
    ----------
    series:
        Série temporal 1D.

    context_length:
        Quantidade de observações anteriores utilizadas
        para prever o próximo valor.

    horizon:
        Horizonte da previsão.

        horizon=1:
            prevê y(t)

        horizon=2:
            prevê y(t+1)

        etc.
    """

    if context_length <= 0:
        raise ValueError(
            "context_length deve ser maior que zero."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon deve ser maior que zero."
        )

    series = np.asarray(
        series,
        dtype=np.float32,
    )

    X = []
    y = []

    max_t = (
        len(series)
        - horizon
        + 1
    )

    for t in range(
        context_length,
        max_t,
    ):

        X.append(
            series[
                t - context_length:t
            ]
        )

        y.append(
            series[
                t + horizon - 1
            ]
        )

    return (
        np.asarray(
            X,
            dtype=np.float32,
        ),
        np.asarray(
            y,
            dtype=np.float32,
        ),
    )


class LagLinearRegression:
    """
    Regressão Linear usando os lags da série como features.

    O modelo estima:

        y_t =
            beta_0
            + beta_1 y_(t-1)
            + beta_2 y_(t-2)
            + ...
            + beta_L y_(t-L)
            + epsilon_t

    Parameters
    ----------
    fit_intercept:
        Se True, estima o intercepto beta_0.

    positive:
        Se True, restringe os coeficientes para >= 0.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        positive: bool = False,
    ):

        self.fit_intercept = fit_intercept
        self.positive = positive

        self.model = LinearRegression(
            fit_intercept=fit_intercept,
            positive=positive,
        )

        self.train_time = None
        self.inference_time = None

    def fit(
        self,
        X,
        y,
    ):
        """
        Treina a regressão.
        """

        start_time = time.perf_counter()

        self.model.fit(
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
        )

        self.train_time = (
            time.perf_counter()
            - start_time
        )

        return self

    def predict(
        self,
        X,
    ):
        """
        Gera previsões.
        """

        start_time = time.perf_counter()

        predictions = self.model.predict(
            np.asarray(X, dtype=np.float64)
        )

        self.inference_time = (
            time.perf_counter()
            - start_time
        )

        return predictions

    @property
    def parameters(self):
        """
        Número de parâmetros estimados.

        Inclui:
            - coeficientes dos lags
            - intercepto, caso utilizado
        """

        n_coefficients = (
            self.model.coef_.size
        )

        intercept = (
            1
            if self.fit_intercept
            else 0
        )

        return int(
            n_coefficients
            + intercept
        )

    @property
    def coefficients(self):
        """
        Retorna os coeficientes estimados
        para cada lag.
        """

        return self.model.coef_

    @property
    def intercept(self):
        """
        Retorna o intercepto estimado.
        """

        return float(
            np.asarray(
                self.model.intercept_
            ).reshape(-1)[0]
        )