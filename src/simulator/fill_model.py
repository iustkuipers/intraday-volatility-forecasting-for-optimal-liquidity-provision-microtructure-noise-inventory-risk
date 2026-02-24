import numpy as np


def logistic(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def fill_probability(delta: float,
                     volatility: float,
                     a: float = 5.0,
                     b: float = 50.0,
                     c: float = 20.0) -> float:
    """
    Compute fill probability using logistic model:

    P(fill) = sigmoid(a - b * delta + c * volatility)

    Parameters
    ----------
    delta : float
        Spread size at time t
    volatility : float
        Forecasted volatility at time t
    a, b, c : floats
        Model coefficients

    Returns
    -------
    float
        Probability in (0,1)
    """

    x = a - b * delta + c * volatility
    return logistic(x)