# Synthia (https://github.com/dmey/synthia).
# Copyright (c) 2020 D. Meyer and T. Nagler. Licensed under the MIT License.

import numpy as np
from scipy.interpolate import PchipInterpolator

from .parameterizer import Parameterizer

class QuantileParameterizer(Parameterizer):
    def __init__(self, n_quantiles: int) -> None:
        self.n_quantiles = n_quantiles
    
    def fit(self, data: np.ndarray) -> None:
        self.quantiles = create_quantiles(data, self.n_quantiles)

    def generate(self, n_samples: int) -> np.ndarray:
        samples = interpolate(self.quantiles, n_samples)
        return samples

def create_quantiles(arr: np.ndarray, n_quantiles: int) -> np.ndarray:
    """ Create a quantile vector of user-defined size.

    Example:
    >>> arr = np.random.normal(size=100)
    >>> create_quantiles(arr, 3)
    array([-2.6197451 , -0.12695629,  1.85227818])
    """
    assert arr.ndim == 1, f"Input array must be 1D, given {arr.ndim}"

    x = np.linspace(0, 1, n_quantiles)
    q = np.quantile(arr, x)
    return q

def interpolate(arr: np.ndarray, grid_size: int) -> np.ndarray:
    """ Interpolate vector using scipy.interpolate.PchipInterpolator.

    Example:
    >>> arr = np.linspace(-1, 1, 10)**2   # y = x^2
    >>> interpolate(arr, 5)
    array([1.        , 0.25061728, 0.01234568, 0.25061728, 1.        ])
    """
    assert arr.ndim == 1, f"Input array must be 1D, given {arr.ndim}"
    f = PchipInterpolator(np.linspace(0, 1, len(arr)), arr)
    x = np.linspace(0, 1, grid_size) 
    return f(x)
