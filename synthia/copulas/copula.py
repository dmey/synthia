# Synthia (https://github.com/dmey/synthia).
# Copyright (c) 2020 D. Meyer and T. Nagler. Licensed under the MIT License.

from abc import ABCMeta, abstractmethod
import numpy as np

class Copula(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, rank_standardized: np.ndarray) -> None:
        pass

    @abstractmethod
    def generate(self, n_samples: int, **kws) -> np.ndarray:
        pass
