# Synthia (https://github.com/dmey/synthia).
# Copyright (c) 2020 D. Meyer and T. Nagler. Licensed under the MIT License.

import numpy as np
try:
    import pyvinecopulib as pv
except ImportError:
    pv = None

from .copula import Copula


class IndependenceCopula(Copula):
    """The independence copula.
    """

    def fit(self, rank_standardized: np.ndarray) -> None:
        """Fit an independence copula (only stores how many features there are).

        Args:
            rank_standardized (ndarray): 2D array of shape (samples, feature)
                with values in range [0,1]

        Returns:
            None

        """
        self.n_features = rank_standardized.shape[1]

    def generate(self, n_samples: int, qrng=False, num_threads=1) -> np.ndarray:
        """Generate n_samples of a vector of independent uniform random variables.

        Args:
            n_samples (int): Number of samples to generate.
            qrng (bool, optional): If True, quasirandom numbers are generated using pyvinecopulib.

        Returns:
            2D array of shape (n_samples, feature) with independent entries.
        """
        # Uniform entries.
        if qrng:
            assert pv, "pyvinecopulib not installed but required for qrng=True"
            u = pv.simulate_uniform(n_samples, self.n_features, qrng=True)
        else:
            # ~3x faster than pv for generating pseudo-random numbers.
            u = np.random.uniform(size=(n_samples, self.n_features))
        return u
