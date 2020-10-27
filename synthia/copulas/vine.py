# Synthia (https://github.com/dmey/synthia).
# Copyright (c) 2020 D. Meyer and T. Nagler. Licensed under the MIT License.

from typing import Optional
from multiprocessing import cpu_count
import tempfile
import os
import json

import numpy as np
try:
    import pyvinecopulib as pv
except ImportError:
    pv = None

from .copula import Copula

class VineCopula(Copula):
    """A Vine copula.
    """

    # Note: The 'controls' type is written in str notation to avoid early import.
    def __init__(self, controls: Optional['pv.FitControlsVinecop']=None) -> None:
        """
        Args:
            controls (pyvinecopulib.FitControlsVinecop, optional): Controls 
                for fitting vine copula models.
        """
        assert pv, "pyvinecopulib not installed but required for VineCopula()"
        if controls is None:
            controls = pv.FitControlsVinecop(num_threads=cpu_count())
        self.controls = controls

    def fit(self, rank_standardized: np.ndarray) -> None:
        """Fit a Vine copula to data.

        Args:
            rank_standardized (ndarray): 2D array of shape (feature, feature)
                with values in range [-1,1]

        Returns:
            None
        """
        self.model = pv.Vinecop(rank_standardized, controls=self.controls)

    def generate(self, n_samples: int, qrng=False, num_threads=1) -> np.ndarray:
        """Generate n_samples Vine copula entries.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            2D array of shape (n_samples, feature) with Vine copula entries.
        """
        u_sim = self.model.simulate(n_samples, qrng=qrng, num_threads=num_threads)
        return u_sim

    # Pickling support
    def __getstate__(self):
        state = {}
        try:
            handle, tmp = tempfile.mkstemp()
            os.close(handle)
            self.model.to_json(tmp)
            # re-format json to remove whitespace (50% reduction in size)
            with open(tmp) as f:
                data = json.load(f)
            state['model_json'] = json.dumps(data, separators=(',', ':'))
        finally:
            os.remove(tmp)
        return state
    
    def __setstate__(self, state):
        try:
            handle, tmp = tempfile.mkstemp()
            os.close(handle)
            with open(tmp, 'w') as f:
                f.write(state['model_json'])
            self.model = pv.Vinecop(tmp)
        finally:
            os.remove(tmp)
