from abc import ABCMeta, abstractmethod
import numpy as np

class Copula(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, rank_standardized: np.ndarray) -> None:
        pass

    @abstractmethod
    def generate(self, n_samples: int) -> np.ndarray:
        pass
