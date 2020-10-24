# Synthia (https://github.com/dmey/synthia).
# Copyright (c) 2020 D. Meyer and T. Nagler. Licensed under the MIT License.

from typing import List, Optional, Union, Dict
import numpy as np
import xarray as xr

from ..parameterizers.parameterizer import Parameterizer
from ..copulas.copula import Copula
from ..util import to_feature_array, from_feature_array, per_feature

class CopulaDataGenerator:
    """Estimates the characteristics of a set of multi-feature samples and 
    generates synthetic samples with the same or modified characteristics,
    based on copulas.

    The input can be a numpy array or xarray DataArray of shape (sample, feature),
    or an xarray Dataset where all variables have shapes like (sample[, ...]).
    For Datasets, all extra dimensions except the first are treated as features.

    The output is in the same form as the input.

    Algorithm:

    1. Fitting phase
      a) (Gaussian copula)
         The multivariate correlation between features is estimated and stored as
         correlation matrix with shape (feature,feature).
         Matrix values are between -1 and 1 inclusive.
         
         (Vine copula)
         The pyvinecopulib package is used to fit a vine copula that captures
         the multivariate correlation between features. See that package for further details.
      
      b) (Optional)
         Some or all features of the input data are parameterized.
         If a feature is not parameterized, then the original data is used during generation.
         Parameterization may impact the quality of the synthetic samples.
         It can be useful for storing/re-distributing a data generator for later use
         without requiring the original data.
        
      c) Per-feature summary statistics (min, max, median) of the input data are computed.
         These statistics are only used if the synthetic samples should be generated
         with modified characteristics (uniformization, stretching).
    
    Note that all three steps in the fitting phase are independent from each other.

    2. Generation phase
      a) Generate new samples from the fitted copula model (Gaussian or Vine).

      b) Transform copula samples to the feature scale by the quantile transform.

      c) (Optional) Apply modifications (uniformization, stretching) if asked for.

    Example:
        >>> import xarray as xr
        >>> from scipy.stats import multivariate_normal
        >>> import synthia as syn
        >>> # Generate dataset ~ N(0, .5) with 1000 random samples and 2 features.
        >>> mvnorm = multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]])
        >>> arr = xr.DataArray(mvnorm.rvs(1000))
        >>> # Initialize the generator
        >>> generator = syn.CopulaDataGenerator(verbose=False)
        >>> # Fit the generator to the data using a Gaussian copula model.
        >>> generator.fit(arr, copula=syn.GaussianCopula(), parameterize_by=None)
        >>> # Generate twice as many samples from the Gaussian copula model.
        >>> synth = generator.generate(n_samples=2000)
    """

    def __init__(self, verbose=False) -> None:
        """
        Args:
            verbose (bool, optional): If True, prints progress messages during computation.
        """
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def fit(self, data: Union[np.ndarray, xr.DataArray, xr.Dataset],
            copula: Copula,
            parameterize_by: Optional[Union[Parameterizer, Dict[int, Parameterizer], Dict[str, Parameterizer]]]=None):
        """Fit the marginal distributions and copula model for all features.

        Args:
            data (ndarray or DataArray or Dataset): The input data, either a
                2D array of shape (sample, feature) or a dataset where all
                variables have the shape (sample[, ...]).

            copula: The underlying copula to use, for example a GaussianCopula object.

            parameterize_by (Parameterizer or mapping, optional): The
                following forms are valid:
                
                - Parameterizer
                - per-feature mapping {feature idx: Parameterizer} -- ndarray/DataArray only
                - per-variable mapping {var name: Parameterizer} -- Dataset only

        Returns:
            None
        """
        
        data, self.data_info = to_feature_array(data)
        
        self.dtype = data.dtype
        self.n_features = data.shape[1]

        self._log('computing rank data')
        rank_standardized = compute_rank_standardized(data)
        
        self._log('fitting copula')
        copula.fit(rank_standardized)

        self._log('parameterizing data')
        self.parameterizers = per_feature(parameterize_by, self.data_info)
    
        self.feature_data = dict()
        for i in range(self.n_features):
            if self.parameterizers[i] is None:
                self.feature_data[i] = data[:, i]
            else:
                self.parameterizers[i].fit(data[:, i])
        
        self.copula = copula

        self._log('determining range and median of data')
        self.feature_min = data.min(axis=0)
        self.feature_max = data.max(axis=0)
        self.feature_med = data.median(axis=0)
        assert self.feature_min.shape == (self.n_features,)

    def generate(self, n_samples: int,
                 uniformization_ratio: Union[float, Dict[int, float], Dict[str, float]] = 0,
                 stretch_factor: Union[float, Dict[int, float], Dict[str, float]] = 1, qrng=False, num_threads=1) \
                 -> Union[np.ndarray, xr.DataArray, xr.Dataset]:
        """Generate synthetic data from the model.

        Args:
            n_samples (int): Number of samples to generate.

            uniformization_ratio (float or mapping, optional): The
                following forms are valid:
                
                - ratio
                - per-feature mapping {feature idx: ratio} -- ndarray/DataArray only
                - per-variable mapping {var name: ratio} -- Dataset only

            stretch_factor (float or mapping, optional): The
                following forms are valid:
                
                - stretch factor
                - per-feature mapping {feature idx: stretch factor} -- ndarray/DataArray only
                - per-variable mapping {var name: stretch factor} -- Dataset only

        Returns:
            Synthetic samples in the form of the input data
        """
        assert n_samples > 0

        unif_ratio_per_feature = per_feature(uniformization_ratio, self.data_info)
        stretch_factor_per_feature = per_feature(stretch_factor, self.data_info)

        self._log(f'generating {n_samples} samples using copula')
        u = self.copula.generate(n_samples, qrng=qrng, num_threads=num_threads)

        self._log(f'applying quantiles')
        samples = np.empty((n_samples, self.n_features), dtype=self.dtype)
        for i in range(self.n_features):
            
            if self.parameterizers[i] is None:
                feature_samples = self.feature_data[i]
            else:
                feature_samples = self.parameterizers[i].generate(n_samples)

            samples[:,i] = np.quantile(feature_samples, q=u[:, i], interpolation='linear')

            if unif_ratio_per_feature[i] != 0:
                feature_min = self.feature_min[i].compute().item()
                feature_max = self.feature_max[i].compute().item()
                uniform = feature_min + u[:, i] * (feature_max - feature_min)
                samples[:,i] = (1 - unif_ratio_per_feature[i]) * samples[:,i] + unif_ratio_per_feature[i] * uniform
            
            if stretch_factor_per_feature[i] != 1:
                feature_med = self.feature_med[i].compute().item()
                samples[:,i] = stretch_factor_per_feature[i] * (samples[:,i] - feature_med) + feature_med

        samples = from_feature_array(samples, self.data_info)

        return samples

def compute_rank_standardized(data: xr.DataArray) -> xr.DataArray:
    """Compute per-feature percentage ranks of the data. Data is a 
    2D xarray of shape (sample, feature).

    Example:
       >>> # 3 samples, 2 features
       >>> data = xr.DataArray([(10,0.3), (5,0.2), (1500,0.1)])
       >>> compute_rank_standardized(data).values
       array([[0.5 , 0.75],
              [0.25, 0.5 ],
              [0.75, 0.25]])
    """

    assert data.ndim == 2, f'Input array must be 2D, given: {data.ndim}'
    # rank() requires all data in-memory, hence compute()
    rank = data.compute().rank(data.dims[0])
    rank_standardized = rank / (rank.max(data.dims[0]) + 1)
    assert rank_standardized.shape == data.shape
    return rank_standardized
