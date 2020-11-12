# Synthia (https://github.com/dmey/synthia).
# Copyright (c) 2020 D. Meyer and T. Nagler. Licensed under the MIT License.

import numpy as np
import xarray as xr

from synthia.util import (
    to_stacked_array, to_unstacked_dataset,
    categories_to_one_hot, one_hot_to_categories,
    to_feature_array, from_feature_array,
    per_feature
)

def test_stacking_roundtrip():
    a = np.arange(12).reshape(2,6)
    b = np.arange(12,24).reshape(2,2,3)
    c = np.arange(2)
    ds = xr.Dataset({
        'a': (('sample', 'foo'), a),
        'b': (('sample', 'ff', 'bb'), b),
        'c': (('sample'), c)
        })
    arr, info = to_stacked_array(ds)
    ds_u = to_unstacked_dataset(arr.values, info)
    assert ds.identical(ds_u)

def test_one_hot_roundtrip():
    test_cases = [
        np.array([1, 4, 2, 1, 5]),
        np.array(['foo', 'bar', 'foo'])
    ]
    for data in test_cases:
        one_hot, categories = categories_to_one_hot(data)
        data_ = one_hot_to_categories(one_hot, categories)
        np.testing.assert_equal(data_, data)

def test_feature_array_roundtrip():
    a = np.array([[0, 1, 2], [3, 4, 5]])
    b = np.array([6, 7])
    c = np.array([8, 9])
    ds = xr.Dataset({
        'a': (('sample', 'foo'), a),
        'b': (('sample'), b),
        'c': (('sample'), c)
        })
    types = { 'a': 'cat' }

    arr, info = to_feature_array(ds, types)
    ds_u = from_feature_array(arr.values, info)
    assert isinstance(ds_u, xr.Dataset)
    assert ds.identical(ds_u)

def test_to_feature_array():
    a = np.array([[0, 1, 2], [3, 4, 5]])
    b = np.array([6, 7])
    c = np.array([8, 9])
    ds = xr.Dataset({
        'a': (('sample', 'foo'), a),
        'b': (('sample'), b),
        'c': (('sample'), c)
        })

    types = { 'c': 'cat' }
    arr, _ = to_feature_array(ds, types)
    np.testing.assert_equal(np.array([
        [0, 1, 2, 6, 1, 0],
        [3, 4, 5, 7, 0, 1],
    ]), arr)

    types = { 'a': 'cat' }
    arr, _ = to_feature_array(ds, types)
    np.testing.assert_equal(np.array([
        [1, 0, 1, 0, 1, 0, 6, 8],
        [0, 1, 0, 1, 0, 1, 7, 9],
    ]), arr)

def test_per_feature():
    a = np.array([[0, 1, 2], [3, 4, 5]])
    b = np.array([6, 7])
    ds = xr.Dataset({
        'a': (('sample', 'foo'), a),
        'b': (('sample'), b)
        })

    _, data_info = to_feature_array(ds)
    vals = per_feature(True, data_info)
    assert [True, True, True, True] == vals

    vals = per_feature({'a': 1, 'b': 2}, data_info)
    assert [1, 1, 1, 2] == vals

    vals = per_feature({'a': True}, data_info, default=False)
    assert [True, True, True, False] == vals

    types = { 'b': 'cat' }
    _, data_info = to_feature_array(ds, types)
    vals = per_feature({'a': 'foo', 'b': 'bar'}, data_info)
    assert ['foo', 'foo', 'foo', 'bar', 'bar'] == vals
