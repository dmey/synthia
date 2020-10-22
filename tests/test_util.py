# Synthia (https://github.com/dmey/synthia).
# Copyright (c) 2020 D. Meyer and T. Nagler. Licensed under the MIT License.

import numpy as np
import xarray as xr

from synthia.util import to_stacked_array, to_unstacked_dataset

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
