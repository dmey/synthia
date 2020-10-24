# Synthia (https://github.com/dmey/synthia).
# Copyright (c) 2020 D. Meyer and T. Nagler. Licensed under the MIT License.

"""A set of invertible transforms.
"""

from abc import ABCMeta, abstractmethod
import xarray as xr
import numpy as np

from scipy.special import boxcox, inv_boxcox
from synthia import util

# Avoid inf/nan in log/arctanh transformations when value is zero.
NUMERICAL_OFFSET = 1e-11

class Transformer(metaclass=ABCMeta):
    @abstractmethod
    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        pass

    @abstractmethod
    def revert(self, ds: xr.Dataset) -> xr.Dataset:
        pass


class BoxCoxTransformer(Transformer):
    """ Assumption: the first dimension is samples and all others are features.
    I.e. transformations will be done per feature.
    """
    def __init__(self, var_names, lmbda, boundary_location='left'):
        self.var_names = var_names
        
        if not isinstance(lmbda, list):
            lmbda = [lmbda for _ in var_names]
        else:
            assert len(var_names) == len(lmbda), 'lmbda must be the same size as var_names'
        self.lmbdas = lmbda
        
        if not isinstance(boundary_location, list):
            boundary_location = [boundary_location for _ in var_names]
        else:
            assert len(var_names) == len(boundary_location), 'boundary_location must be the same size as var_names'
        assert all(loc in ['left', 'right'] for loc in boundary_location), 'boundary_location must be left or right'
        self.boundary_locations = boundary_location
        
        self.shifting_factors = []

    def apply(self, ds):
        assert not self.shifting_factors, 'This function cannot be called twice.'

        ds = ds.astype('float64')

        for name, lmbda, boundary_location in \
                zip(self.var_names, self.lmbdas, self.boundary_locations):
            if boundary_location == 'right':
                ds = ds.assign({name: -ds[name]})

            sample_dim = ds[name].dims[0]
            stacked, stack_info = util.to_stacked_array(ds[[name]])
            mins = stacked.min(sample_dim) # feature

            shifting_factor_per_feature = abs(mins) + NUMERICAL_OFFSET # feature
            shifting_factor_per_feature.load()[mins >= NUMERICAL_OFFSET] = 0.
            self.shifting_factors.append(shifting_factor_per_feature)

            transformed = boxcox(stacked + shifting_factor_per_feature, lmbda) 
            unstacked = util.to_unstacked_dataset(transformed.values, stack_info)
            ds = ds.assign({name: unstacked[name]})
        return ds

    def revert(self, ds):
        for name, lmbda, shifting_factor_per_feature, boundary_location in \
                zip(self.var_names, self.lmbdas, self.shifting_factors, self.boundary_locations):
            stacked, stack_info = util.to_stacked_array(ds[[name]])
            
            # This does not guarantee that the generated samples are above zero.
            reverted = inv_boxcox(stacked, lmbda) - shifting_factor_per_feature
            
            unstacked = util.to_unstacked_dataset(reverted.values, stack_info)
            ds = ds.assign({name: unstacked[name]})
            if boundary_location == 'right':
                ds = ds.assign({name: -ds[name]})
        return ds


class ArcTanhTransformer(Transformer):
    """ Assumption: the first dimension is samples and all others are features.
    I.e. transformations will be done per feature.
    """
    def __init__(self, var_names):
        self.var_names = var_names
        self.mins = []
        self.maxs = []

    def apply(self, ds):
        assert not self.mins, 'This function cannot be called twice.'

        ds = ds.astype('float64')

        for name in self.var_names:
            sample_dim = ds[name].dims[0]
            stacked, stack_info = util.to_stacked_array(ds[[name]])
            
            mins = stacked.min(sample_dim) - NUMERICAL_OFFSET # feature
            maxs = stacked.max(sample_dim) + NUMERICAL_OFFSET # feature
            self.mins.append(mins)
            self.maxs.append(maxs)
            ds_normalized = (stacked - mins) / (maxs - mins)

            transformed = np.arctanh(2*ds_normalized - 1)
            unstacked = util.to_unstacked_dataset(transformed.values, stack_info)
            ds = ds.assign({name: unstacked[name]})
        return ds

    def revert(self, ds):
        for name, mins, maxs in zip(self.var_names, self.mins, self.maxs):
            stacked, stack_info = util.to_stacked_array(ds[[name]])

            reverted = .5 * (np.tanh(stacked) + 1.)
            ds_unnormalized = reverted * (maxs - mins) + mins
            
            unstacked = util.to_unstacked_dataset(ds_unnormalized.values, stack_info)
            ds = ds.assign({name: unstacked[name]})
        return ds

class CombinedTransformer(Transformer):
    def __init__(self, transformers):
        self.transformers = transformers

    def apply(self, ds):
        for t in self.transformers:
            ds = t.apply(ds)
        return ds

    def revert(self, ds):
        for t in self.transformers[::-1]:
            ds = t.revert(ds)
        return ds
