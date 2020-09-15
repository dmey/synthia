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
                with values in range [-1,1]

        Returns:
            None

        """
        self.n_features = rank_standardized.shape[1]
        # We need to use the standardised rank to compute
        # the correlation matrix as the ranks are done on the
        # maginals. Removing this and computing the ppf directly
        # would only work if the marginals are normally distributed.
        # Furthermore, we avoid taking the correlation directly on the data
        # as this can lead to problems with the final results TODO: why? 
        # Also, using the rank loses information from the data and may change the correlation
        # in some cases:
        # arr = xr.DataArray([(10,0.3), (5,0.3), (150000,10.001)])
        # ranks = data_generator.compute_rank_standardized(arr).values
        # array([[0.5  , 0.375],
        #        [0.25 , 0.375],
        #        [0.75 , 0.75 ]])
        # compute_norm_corr(arr)
        # array([[1., 1.],
        #        [1., 1.]])
        # compute_norm_corr(scipy.stats.norm.ppf(ranks))
        # array([[1.       , 0.8660254],
        #        [0.8660254, 1.       ]])
        # This is because the rank ignores distances between values and only keeps
        # the ordering.
        # TODO: why?
        ppf = scipy.stats.norm.ppf(rank_standardized)
        self.corr = compute_norm_corr(ppf)

    def generate(self, n_samples: int, qrng=False, num_threads=1) -> np.ndarray:
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
            u = pv.simulate_uniform(n_samples, self.n_features, qrng=True)
        else:
            # ~3x faster than pv for generating pseudo-random numbers.
            u = np.random.uniform(size=(n_samples, self.n_features))
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
