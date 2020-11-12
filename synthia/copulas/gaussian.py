# Synthia (https://github.com/dmey/synthia).
# Copyright (c) 2020 D. Meyer and T. Nagler. Licensed under the MIT License.

from typing import Optional
import numpy as np
import scipy.stats
import scipy.linalg
try:
    import pyvinecopulib as pv
except ImportError:
    pv = None

from .copula import Copula


class GaussianCopula(Copula):
    """A Gaussian copula.
    """

    def fit(self, rank_standardized: np.ndarray) -> None:
        """Fit a Gaussian copula to data.

        Args:
            rank_standardized (ndarray): 2D array of shape (feature, feature)
                with values in range [0,1]

        Returns:
            None

        """
        self.n_features = rank_standardized.shape[1]
        # We need to use the standardised rank to compute
        # the correlation matrix as the ranks are done on the
        # maginals. The Gaussian copula is parameterized by a correlation 
        # matrix This matrix can be estimated by computing the correlations 
        # between the standardized ranks after they have been transformed by 
        # the normal ppf. (This is not the same as the correlation 
        # between ranks.) The reason is that the Gaussian copula is derived 
        # from a multivariate Gaussian distribution, where all margins are 
        # Gaussian.
        ppf = scipy.stats.norm.ppf(rank_standardized)
        self.corr = compute_norm_corr(ppf)

    def generate(self, n_samples: int, qrng: bool=False, seed: Optional[int]=None) -> np.ndarray:
        """Generate n_samples gaussian copula entries.

        Args:
            n_samples (int): Number of samples to generate.
            qrng (bool, optional): If True, quasirandom numbers are generated using pyvinecopulib.

        Returns:
            2D array of shape (n_samples, feature) with gaussian copula entries.
        """
        # Uniform entries.
        if qrng:
            assert pv, "pyvinecopulib not installed but required for qrng=True"
            seeds = [] if seed is None else [seed]
            u = pv.simulate_uniform(n_samples, self.n_features, qrng=True, seeds=seeds)
        else:
            # ~3x faster than pv for generating pseudo-random numbers.
            r = np.random.RandomState(seed)
            u = r.uniform(size=(n_samples, self.n_features))
        # Independent standard normal.
        z = scipy.stats.norm.ppf(u)  
        # Convert to real as sqrtm may return complex numbers.
        a = np.real(scipy.linalg.sqrtm(self.corr)) 
        # Multivariate ~ N(0, self.corr).
        x = z @ a
        # Gaussian copula entries.
        return np.real(scipy.stats.norm.cdf(x))


def compute_norm_corr(norm_ppf: np.ndarray) -> np.ndarray:
    """ Compute the correlation coefficient matrix of the features.
    
    Args:
        norm_ppf: 2D array of shape (sample, feature)

    Returns:
        2D array of shape (feature,feature) with values in range [-1,1].

    Todo:
        * Compute corrcoef iteratively to allow handling of big data.
    """
    assert norm_ppf.ndim == 2, f'Input array must be 2D, given: {norm_ppf.ndim}'
    n_features = norm_ppf.shape[1]
    # For optimization, does corrcoef always require the full copula matrix or can it
    # be computed iteratively to avoid loading all data into memory?
    corr = np.corrcoef(norm_ppf.T)
    assert corr.shape == (n_features, n_features)
    return corr
