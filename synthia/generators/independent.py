from typing import List, Optional, Union, Dict
import numpy as np
import xarray as xr
try:
    import pyvinecopulib as pv
except ImportError:
    pv = None

from ..parameterizers.parameterizer import Parameterizer
from ..util import to_feature_array, from_feature_array, per_feature

class IndependentDataGenerator:
    # Thomas: Probably this is useless, I'll only look at the documentation 
    # when you convince me otherwise ;)
    def fit(self, data: Union[np.ndarray, xr.DataArray, xr.Dataset],
            parameterize_by: Optional[Union[Parameterizer, Dict[int, Parameterizer], Dict[str, Parameterizer]]]=None):
        """Fit marginal distributions for all features.

        Args:
            data (ndarray or DataArray or Dataset): The input data, either a
                1D array, a 2D array of shape (sample, feature)
                or a dataset where all variables have the shape (sample[, ...]).

            parameterize_by (Parameterizer or mapping, optional): The
                following forms are valid:
                
                - Parameterizer
                - per-feature mapping {feature idx: Parameterizer} -- ndarray/DataArray only
                - per-variable mapping {var name: Parameterizer} -- Dataset only

        Returns:
            None
        """

        data, self.data_info = to_feature_array(data, allow_1d=True)
        
        self.dtype = data.dtype
        self.n_features = data.shape[1]

        self.parameterizers = per_feature(parameterize_by, self.data_info)
    
        self.feature_data = dict()
        for i in range(self.n_features):
            if self.parameterizers[i] is None:
                self.feature_data[i] = data[:, i]
            else:
                self.parameterizers[i].fit(data[:, i])
        
        self.feature_min = data.min(axis=0)
        self.feature_max = data.max(axis=0)
        self.feature_med = data.median(axis=0)
        assert self.feature_min.shape == (self.n_features,)

    def generate(self, n_samples: int,
                 qrng=False,
                 uniformization_ratio: Union[float, Dict[int, float], Dict[str, float]] = 0,
                 stretch_factor: Union[float, Dict[int, float], Dict[str, float]] = 1) \
                 -> Union[np.ndarray, xr.DataArray, xr.Dataset]:
        """Generate synthetic data from the model.

        Args:
            n_samples (int): Number of samples to generate.

            qrng (bool, optional): If True, quasirandom numbers are generated using pyvinecopulib.

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

        samples = np.empty((n_samples, self.n_features), dtype=self.dtype)
        for i in range(self.n_features):
            
            if self.parameterizers[i] is None:
                feature_samples = self.feature_data[i]
            else:
                feature_samples = self.parameterizers[i].generate(n_samples)

            if qrng:
                assert pv, "pyvinecopulib not installed but required for qrng=True"
                u = pv.simulate_uniform(n_samples, 1, qrng=True)[:,0]
            else:
                # ~3x faster than pv for generating pseudo-random numbers.
                u = np.random.uniform(size=n_samples)

            samples[:,i] = np.quantile(feature_samples, q=u, interpolation='linear')

            if unif_ratio_per_feature[i] != 0:
                feature_min = self.feature_min[i].compute().item()
                feature_max = self.feature_max[i].compute().item()
                uniform = feature_min + u * (feature_max - feature_min)
                samples[:,i] = (1 - unif_ratio_per_feature[i]) * samples[:,i] + unif_ratio_per_feature[i] * uniform
            
            if stretch_factor_per_feature[i] != 1:
                feature_med = self.feature_med[i].compute().item()
                samples[:,i] = stretch_factor_per_feature[i] * (samples[:,i] - feature_med) + feature_med

        samples = from_feature_array(samples, self.data_info)

        return samples
