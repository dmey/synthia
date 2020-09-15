import numpy as np

from .parameterizer import Parameterizer

class ConstParameterizer(Parameterizer):
    def __init__(self, val) -> None:
        self.val = val
    
    def fit(self, data: np.ndarray) -> None:
        pass

    def generate(self, n_samples: int) -> np.ndarray:
        samples = np.full(n_samples, self.val)
        return samples
