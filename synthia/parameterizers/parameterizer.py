from abc import ABCMeta, abstractmethod
import numpy as np

class Parameterizer(metaclass=ABCMeta):
    """ Parameterize 1D array data.
    """
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def generate(self, n_samples: int) -> np.ndarray:
        pass
