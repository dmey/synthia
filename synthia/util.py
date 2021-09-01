# Synthia (https://github.com/dmey/synthia).
# Copyright (c) 2020 D. Meyer and T. Nagler. Licensed under the MIT License.

from typing import Tuple, NamedTuple, List, Dict, Optional, Union, Any
import time
import operator
import copy
import inspect
from functools import reduce
import xarray as xr
import numpy as np


def measure(msg: Optional[str]=None, enabled=True):
    """
    Examples:
        @measure("Fitting {distribution}")
        def fit(data, distribution: str):
            ...

        @measure()
        def fit(data):
            ...
    """
    def inner(fn):
        if not enabled:
            return fn
        def wrapped_fn(*args, **kw):
            t0 = time.time()
            res = fn(*args, **kw)
            t1 = time.time()

            if msg is None:
                msg_ = fn.__name__
            else:
                spec = inspect.getfullargspec(fn)
                args_dict = dict(zip(spec.args, args))
                args_dict.update(kw)
                msg_ = msg.format(**args_dict)
            
            print(f'{msg_} ({t1-t0:.2f}s)')
            return res
        return wrapped_fn
    return inner

class StackInfoVar(NamedTuple):
    name: str
    dims: Tuple[str, ...]
    shape: Tuple[int, ...]
    dtype: np.dtype

StackInfo = List[StackInfoVar]

# Specialized stacking/unstacking functions (as opposed to
# using xarray's to_stacked_array/to_unstacked_dataset).
# This allows to have control over the exact stacking behaviour
# which in turn allows to store compact stacking metadata and use it
# to unstack arbitrary arrays not directly related to the input dataset object.
def to_stacked_array(ds: xr.Dataset, var_names=None, new_dim='stacked', name=None) -> Tuple[xr.DataArray, StackInfo]:
    # Sample dimension must be the first dimension in all variables.
    if not var_names:
        var_names = sorted(ds.data_vars)
    stack_info = []
    var_stacked = []
    for var_name in var_names:
        v = ds.data_vars[var_name]
        if len(v.dims) > 1:
            stacked = v.stack({new_dim: v.dims[1:]})
            stacked = stacked.drop(list(stacked.coords.keys()))
        else:
            stacked = v.expand_dims(new_dim, axis=-1)
        stack_info.append(StackInfoVar(var_name, v.dims, v.shape[1:], v.dtype))
        var_stacked.append(stacked)
    arr = xr.concat(var_stacked, new_dim)
    if name:
        arr = arr.rename(name)
    return arr, stack_info

def to_unstacked_dataset(arr: np.ndarray, stack_info: StackInfo) -> xr.Dataset:
    if type(arr) == xr.DataArray:
        arr = arr.values
    elif type(arr) == np.ndarray:
        pass
    else:
        raise RuntimeError('Passed array must be of type DataArray or ndarray')

    unstacked = {}
    curr_i = 0
    for var in stack_info:
        feature_len = 1
        unstacked_shape = [arr.shape[0],]
        for dim_len in var.shape:
            feature_len *= dim_len
            unstacked_shape.append(dim_len)
        var_slice = arr[:, curr_i:curr_i+feature_len]
        var_unstacked = var_slice.reshape(unstacked_shape)
        var_unstacked = var_unstacked.astype(var.dtype, copy=False)
        unstacked[var.name] = xr.DataArray(var_unstacked, dims=var.dims)
        curr_i += feature_len
    ds = xr.Dataset(unstacked)
    return ds

def to_feature_array(data: Union[np.ndarray, xr.DataArray, xr.Dataset],
                     types: Optional[Union[str, Dict[str,str], Dict[int,str]]]=None) -> Tuple[xr.DataArray, dict]:
    data_info = {}
    if isinstance(data, xr.Dataset):
        data, stack_info = to_stacked_array(data)
        data_info['stack_info'] = stack_info
    else:
        if isinstance(data, xr.DataArray):
            data_info['da_info'] = dict(
                dims=data.dims,
                name=data.name,
                attrs=data.attrs
            )
        data = xr.DataArray(data)
    assert data.ndim == 2, f'Input array must be 2D, given: {data.ndim}D'
    data_info['n_features'] = data.shape[1]

    types_ = per_feature(types, data_info)
    if not any(t == 'cat' for t in types_):
        return data, data_info
    
    # Transform categorical features into one-hot vectors.
    categories = {}
    features = []
    for i in range(data.shape[1]):
        if types_[i] == 'cat':
            one_hot, categories[i] = categories_to_one_hot(data[:,i].values)
            features.append(one_hot)
        else:
            features.append(data[:,i:i+1])
    data = xr.DataArray(np.concatenate(features, axis=1), dims=data.dims)
    data_info['categories'] = categories

    return data, data_info

def categories_to_one_hot(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert data.ndim == 1
    # determine all categories occurring in the data
    categories = np.unique(data)
    assert categories.ndim == 1
    n_categories = len(categories)
    # re-index categories, starting from 0
    indices_dtype = None
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if n_categories < np.iinfo(dtype).max:
            indices_dtype = dtype
    assert indices_dtype
    indices = np.empty(data.shape, indices_dtype)
    for index, cat_val in enumerate(categories):
        indices[data == cat_val] = index
    # create one-hot vectors
    one_hot = np.eye(n_categories)[indices]    
    assert one_hot.shape == (len(data), n_categories)
    return one_hot, categories

def one_hot_to_categories(data: np.ndarray, categories: np.ndarray) -> np.ndarray:
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2
    assert categories.ndim == 1
    assert data.shape[1] == len(categories)
    indices = np.argmax(data, axis=1)
    out = categories[indices]
    assert out.shape == (data.shape[0],)
    return out

def from_feature_array(data: np.ndarray, data_info: dict) -> Union[np.ndarray, xr.DataArray, xr.Dataset]:
    categories = data_info.get('categories')
    if categories:
        # Transform one-hot encoded categories into original values.
        features = []
        i_onehot = 0
        i_normal = 0
        while i_onehot < data.shape[1]:
            if i_normal in categories:
                n_categories = len(categories[i_normal])
                feature = one_hot_to_categories(data[:, i_onehot:i_onehot+n_categories], categories[i_normal])
                features.append(feature.reshape(-1, 1))
                i_onehot += n_categories
            else:
                features.append(data[:, i_onehot:i_onehot+1])
                i_onehot += 1
            i_normal += 1
        data = np.concatenate(features, axis=1)

    stack_info = data_info.get('stack_info')
    if stack_info:
        return to_unstacked_dataset(data, stack_info)
    da_info = data_info.get('da_info')
    if da_info:
        return xr.DataArray(data, **da_info)
    return data

def prod(iterable) -> int:
    return reduce(operator.mul, iterable, 1)

def per_feature(val: Optional[Union[Any, Dict[str,Any], Dict[int,Any]]], data_info: dict, default=None) -> List[Any]:
    n_features = data_info.get('n_features')
    stack_info = data_info.get('stack_info')

    if n_features is None:
        n_features_ = sum(prod(info.shape) for info in stack_info)
    else:
        n_features_ = n_features

    if val is None:
        out = [default] * n_features_
    elif isinstance(val, dict):
        if stack_info:
            feature_values = []
            current = 0
            for info in stack_info:
                feature_count = prod(info.shape)
                p = val.get(info.name, default)
                feature_values.extend(copy.copy(p) for _ in range(feature_count))
                current += feature_count
        else:
            feature_values = [copy.copy(val.get(i, default)) for i in range(n_features_)] # type: ignore
        out = feature_values
    else:
        out = [copy.copy(val) for _ in range(n_features_)]

    # Repeat values for categorical features since they are one-hot encoded
    categories = data_info.get('categories')
    if categories:
        out_ = []
        for i in range(n_features_):
            if i in categories:
                out_ += [out[i]] * len(categories[i])
            else:
                out_ += [out[i]]
        out = out_

    return out

def load_dataset(name='SAF-Synthetic') -> xr.Dataset:
    """ Return a dataset of 25 000 synthetic temperature profiles
    from the SAF dataset (http://dx.doi.org/10.13140/2.1.4476.8963).
    These were fitted with 6 fPCA componenets in Synthia version 0.2.0.
    """
    from urllib.request import urlopen
    
    if name != 'SAF-Synthetic':
        raise RuntimeError('Only SAF-Synthetic is currently supported')

    url = 'https://raw.githubusercontent.com/dmey/synthia/data/generator_saf_temperature_fpca_6.nc'
    ds = xr.load_dataset(urlopen(url).read())
    return ds
