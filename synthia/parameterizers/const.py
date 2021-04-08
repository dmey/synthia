# Synthia (https://github.com/dmey/synthia).
# Copyright (c) 2020 D. Meyer and T. Nagler. Licensed under the MIT License.

import numpy as np

from .parameterizer import Parameterizer

class ConstParameterizer(Parameterizer):
    """Preserves the size of the original data. No downsampling is performed
    at fitting and no interpolation is perfomed at generation.
    """
    def __init__(self, val) -> None:
        self.val = val
    
    def fit(self, data: np.ndarray) -> None:
        pass

    def generate(self, n_samples: int) -> np.ndarray:
        """Returns original samples without interpolation. 
        """
        samples = np.full(n_samples, self.val)
        return samples
