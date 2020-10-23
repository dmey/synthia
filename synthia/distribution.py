# Synthia (https://github.com/dmey/synthia).
# Copyright (c) 2020 D. Meyer and T. Nagler. Licensed under the MIT License.

from typing import List, Iterable, Optional, Union, Tuple
import warnings
import sys

import numpy as np
import xarray as xr
import scipy
import scipy.stats

from .util import measure

class Distribution(object):
    def __init__(self, dist_name, param):
        """
        Args:
            dist_name (str): Name of scipy distribution
            param (tuple of floats): Fitting parameters, see scipy.stats.rv_continuous.fit
        """
        self.dist_name = dist_name
        self.param = param

    @staticmethod
    def get_dist_names() -> List[str]:
        """Returns a list of names of all supported scipy distributions.

        Returns:
            List of distribution names
        """
        return [name for name in dir(scipy.stats)
                if isinstance(getattr(scipy.stats, name), scipy.stats.rv_continuous)]

    @classmethod
    def fit(cls, y: Union[np.ndarray, xr.DataArray],
            dist_names: Optional[Iterable[str]]=None,
            use_dask=False, verbose=False) -> 'Distribution':
        """Fits data on multiple distributions and returns the best fit
        according to the Kolmogorov-Smirnov test.

        Args:
            y (ndarray or DataArray): 1D array
            dist_names (sequence of str, optional): Names of scipy distributions to use.
            use_dask (bool, optional): If True, uses Dask to fit distributions in parallel.
            verbose (bool, optional): If True, print progress messages.
        
        Returns:
            Distribution
        """
        assert y.ndim == 1

        if dist_names is None:
            dist_names = cls.get_dist_names()

        if use_dask:
            from dask import delayed
        else:
            delayed = lambda f: f

        @delayed
        @measure('Fitting {dist_name}', enabled=verbose)
        def compute_fit_and_error(y, dist_name: str) -> Union[Exception, Tuple[tuple, float]]:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                dist = getattr(scipy.stats, dist_name)

                try:
                    param = dist.fit(y)
                except Exception as e:
                    return e

                # Apply the Kolmogorov-Smirnov test
                _, pvalue = scipy.stats.kstest(y, dist_name, args=param, N=y.size)
                return param, pvalue

        params_with_error = {}
        y = delayed(y)
        for dist_name in dist_names:
            params_with_error[dist_name] = compute_fit_and_error(y, dist_name)

        @delayed
        def create_from_best_dist(params_with_error):
            params_with_error_ = {}
            for dist_name, v in params_with_error.items():
                if isinstance(v, Exception):
                    print(f"Distribution {dist_name} raised an error: {v}", file=sys.stderr)
                else:
                    params_with_error_[dist_name] = v
            dist_name, (param, p) = max(params_with_error_.items(), key=lambda item: item[1][1])
            return Distribution(dist_name, param)

        dist = create_from_best_dist(params_with_error)
        if use_dask:
            dist = dist.compute()
        return dist

    def random(self, n: int=1) -> np.ndarray:
        """Generate n random samples from the distribution.

        Args:
            n (int): Number of samples to generate.

        Returns:
            1D array of shape (n,)
        """
        param = self.param
        dist = getattr(scipy.stats, self.dist_name)
        return dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=n)
