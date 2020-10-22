# Synthia (https://github.com/dmey/synthia).
# Copyright (c) 2020 D. Meyer and T. Nagler. Licensed under the MIT License.

from typing import Iterable, Optional, List
import numpy as np

from .parameterizer import Parameterizer
from ..distribution import Distribution

class DistributionParameterizer(Parameterizer):
    """tbd

    """
    def __init__(self, dist_names: Optional[Iterable[str]]=None, verbose=False):
        """Args:
            dist_names (sequence of str, optional): Names of scipy distributions to use.
            verbose (bool, optional): If True, print progress messages.
        """
        self.dist_names = dist_names
        self.verbose = verbose
    
    @staticmethod
    def get_dist_names() -> List[str]:
        """Returns a list of names of all supported scipy distributions.

        Returns:
            List of distribution names
        """
        return Distribution.get_dist_names()

    def fit(self, data: np.ndarray) -> None:
        """Fits data on multiple distributions and keeps the best fit
        according to the Kolmogorov-Smirnov test.

        Args:
            y (ndarray): 1D array
        """
        self.dist = Distribution.fit(data, dist_names=self.dist_names, verbose=self.verbose)

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate n random samples from the best-fitting distribution.

        Args:
            n (int): Number of samples to generate.

        Returns:
            1D array of shape (n,)
        """
        samples = self.dist.random(n_samples)
        return samples
